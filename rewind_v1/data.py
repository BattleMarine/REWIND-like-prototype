"""Evidence-gated clip conversion. Frame numbers alone never provide timestamps."""

from dataclasses import dataclass, field
import math
import time

from PIL import Image
import torch

from rewind_archive.contracts import digest
from rewind_data.audit import read_json
from rewind_m1.local_rgb import _hash, _regular_read, decode_rgb_payload
from rewind_m1.rgb_frontend_artifacts import resize_images, training_size
from rewind_m2.data import encode_camera
from rewind_m3.cascade import UpperCondition
from rewind_prepare.converter import sha
from rewind_prepare.extracted import IMAGE_SIZE, _local, _pose_targets, _raw_pose
from rewind_prepare.spec import validate_spec
from rewind_prepare.temporal_selection import SCHEMA as SELECTION_SCHEMA, check_pin
from .frontend import ARCHITECTURE, frontend_fingerprint, predicted_features
from .teacher import WINDOW_FRAMES


EVIDENCE_SCHEMA = "archvis_v1_reviewed_temporal_evidence_v1"
REQUIRED_EVIDENCE = ("timestamp_and_stereo_sync", "camera_transform_and_coordinates",
                     "image_pose_alignment", "original_motion_independence")


@dataclass(frozen=True)
class Clip:
    features: torch.Tensor
    camera: torch.Tensor
    target_mm: torch.Tensor
    target_valid: torch.Tensor
    time_valid: torch.Tensor
    seconds: torch.Tensor
    source: str
    timings: dict = field(default_factory=dict)
    split: str = "train"
    input_sha256: str | None = None

    def select_joints(self, indices):
        features = self.features.unflatten(-1, (-1, 14))[..., indices, :].flatten(-2)
        return Clip(features, self.camera, self.target_mm[..., indices, :], self.target_valid[..., indices],
                    self.time_valid, self.seconds, self.source, self.timings, self.split, self.input_sha256)


def _hand_indices(names, spec):
    fingers, wrists = [], []
    for side in ("left", "right"):
        group = spec["hands"][side][1:]
        fingers.extend(names.index(name) for name in group)
        wrists.extend([names.index(spec["wrists"][side])] * len(group))
    return fingers, wrists


def hand_clip(clip, spec):
    """Use wrist-relative targets. Wrist validity affects local targets only."""
    fingers, wrists = _hand_indices(spec["joint_names"], spec)
    selected = clip.select_joints(fingers)
    mask = selected.target_valid & clip.target_valid[..., wrists]
    delta = selected.target_mm - clip.target_mm[..., wrists, :]
    return Clip(selected.features, selected.camera, torch.where(mask[..., None], delta, 0.), mask,
                selected.time_valid, selected.seconds, selected.source, selected.timings,
                selected.split, selected.input_sha256)


def gt_upper_train(clip, spec):
    if clip.split != "train":
        raise ValueError("rewind_v1_gt_upper_requires_train_split")
    upper = [spec["joint_names"].index(name) for name in spec["upper_joint_names"]]
    return UpperCondition(clip.target_mm[..., upper, :], clip.target_valid[..., upper], "gt_upper_train", None)


def attach_predicted_hands(predicted_body_mm, predicted_local_hand_mm, spec):
    """Attach each local finger to its predicted body wrist in device axes."""
    _, wrists = _hand_indices(spec["joint_names"], spec)
    body_names = spec["body_joint_names"]
    indices = [body_names.index(spec["joint_names"][index]) for index in wrists]
    return predicted_local_hand_mm + predicted_body_mm[..., indices, :]


def _checked_evidence(plan, evidence, expected_sha256):
    if evidence is None:
        raise ValueError("rewind_v1_training_blocked: " + ", ".join(REQUIRED_EVIDENCE))
    check_pin(evidence, EVIDENCE_SCHEMA, expected_sha256)
    if evidence["selection_sha256"] != plan["content_sha256"]:
        raise ValueError("rewind_v1_evidence_selection")
    if any(not _hash(evidence["review_artifact_sha256"].get(key)) for key in REQUIRED_EVIDENCE):
        raise ValueError("rewind_v1_review_artifact_pins")


def _field(field, dimensions):
    valid = torch.tensor(field["valid_mask"], dtype=torch.bool)
    points = torch.tensor([point if ok else [0.] * dimensions
                           for point, ok in zip(field["coordinates"], valid.tolist())], dtype=torch.float32)
    return points, valid


def _gt_features(fields):
    views = []
    for field, size in zip(fields, (IMAGE_SIZE, IMAGE_SIZE)):
        xy, valid = _field(field, 2)
        clean = torch.where(valid[:, None], 2 * xy / (xy.new_tensor(size) - 1) - 1, 0.)
        views.append(torch.cat((clean, valid[:, None].float(), clean.new_zeros(len(valid), 4)), dim=-1))
    return torch.cat(views, dim=-1).flatten()


def frame_image(raw, size):
    width, height = IMAGE_SIZE
    pair = []
    for side in ("left", "right"):
        pixels = decode_rgb_payload(raw[side], IMAGE_SIZE, width * height, Image)
        image = torch.frombuffer(pixels, dtype=torch.uint8).view(height, width, 3)
        pair.append(image.permute(2, 0, 1).float().div_(255.))
    return resize_images(torch.stack(pair)[None], size)[0]


def reviewed_clip(evidence, sequence, frames, scale_mm):
    records = evidence["frames"].get(sequence, {})
    maximum_dt = evidence["max_frame_delta_seconds"]
    if type(maximum_dt) not in (int, float) or not math.isfinite(maximum_dt) or maximum_dt <= 0:
        raise ValueError("rewind_v1_max_frame_delta")
    review = [records.get(str(index)) for index in frames]
    if any(row is None for row in review):
        raise ValueError("rewind_v1_reviewed_frame_evidence_missing")
    seconds = [row["time_seconds"] for row in review]
    if (any(type(t) not in (int, float) or not math.isfinite(t) for t in seconds)
            or any(not 0 < b - a <= maximum_dt for a, b in zip(seconds, seconds[1:]))):
        raise ValueError("rewind_v1_reviewed_time_gap")
    cameras = torch.stack([encode_camera(row["camera_pose"], scale_mm) for row in review])
    return cameras, seconds


def load_clip(cache, plan, sequence, start, *, split, selection_sha256, evidence, evidence_sha256,
              spec_path, scale_mm, source, frontend=None, frontend_size=None, frontend_sha256=None,
              check_budget=lambda: None):
    """Read one reviewed clip wholly inside the train or fixed validation selection."""
    check_pin(plan, SELECTION_SCHEMA, selection_sha256)
    _checked_evidence(plan, evidence, evidence_sha256)
    return _load_clip(cache, plan, sequence, start, split=split, selection_sha256=selection_sha256,
                      context_sha256=evidence_sha256, time_context=lambda frames: reviewed_clip(
                          evidence, sequence, frames, scale_mm),
                      spec_path=spec_path, scale_mm=scale_mm, source=source, frontend=frontend,
                      frontend_size=frontend_size, frontend_sha256=frontend_sha256, check_budget=check_budget)


def _load_clip(cache, plan, sequence, start, *, split, selection_sha256, context_sha256, time_context,
               spec_path, scale_mm, source, frontend, frontend_size, frontend_sha256, check_budget,
               context_label="evidence"):
    if split not in ("train", "validation") or cache.manifest["selection_sha256"] != selection_sha256:
        raise ValueError("rewind_v1_clip_not_selected")
    if split == "train" and [sequence, start] not in plan["train_clip_starts"]:
        raise ValueError("rewind_v1_clip_not_selected")
    frames = range(start, start + WINDOW_FRAMES)
    allowed = {tuple(row["key"]) for row in plan["records"] if row["split"] == split}
    if not all((sequence, index) in allowed for index in frames):
        raise ValueError("rewind_v1_clip_not_selected")
    spec_raw = _regular_read(_local(spec_path), 256 * 1024)
    if sha(spec_raw) != plan["spec_sha256"]:
        raise ValueError("rewind_v1_clip_spec_pin")
    spec = validate_spec(read_json(spec_raw))
    if source not in ("gt_2d_diagnostic", "predicted_2d") or (
        source == "gt_2d_diagnostic" and any(value is not None for value in (frontend, frontend_size, frontend_sha256))
    ) or (source == "predicted_2d" and (
        frontend is None or frontend.training or frontend_size is None or not _hash(frontend_sha256)
        or frontend_fingerprint(frontend) != frontend_sha256
    )):
        raise ValueError("rewind_v1_clip_observation_source")
    if source == "predicted_2d" and (
        frontend.joints != len(spec["joint_names"]) or
        any(size < 32 or size % 16 for size in training_size(frontend_size))
    ):
        raise ValueError("rewind_v1_clip_frontend_shape")
    input_sha256 = digest({"selection": selection_sha256, "cache": cache.manifest["content_sha256"],
                           context_label: context_sha256, "source": source, "scale_mm": scale_mm,
                           "frontend": ARCHITECTURE if frontend else None, "frontend_size": frontend_size,
                           "frontend_sha256": frontend_sha256, "preprocessing": "rgb_float_bilinear_align_corners_v1"})
    cameras, seconds = time_context(frames)
    features, targets, masks, images = [], [], [], []
    timing = {"local_cache_read_wall_seconds": 0., "cpu_preprocess_wall_seconds": 0.}
    for index in frames:
        check_budget()
        started = time.perf_counter()
        raw = cache.read_pair(sequence, index, split=split)
        timing["local_cache_read_wall_seconds"] += time.perf_counter() - started
        started = time.perf_counter()
        fields = _pose_targets(_raw_pose(raw["pose"], index), spec)
        target, valid = _field(fields[2], 3)
        targets.append(target)
        masks.append(valid)
        if source == "gt_2d_diagnostic":
            features.append(_gt_features(fields[:2]))
        else:
            images.append(frame_image(raw, frontend_size))
        timing["cpu_preprocess_wall_seconds"] += time.perf_counter() - started
    if source == "predicted_2d":
        started = time.perf_counter()
        device = next(frontend.parameters()).device
        # Small, fixed groups keep native decode and frontend allocation separate.
        for start in range(0, len(images), 2):
            check_budget()
            features.append(predicted_features(frontend, torch.stack(images[start:start + 2]).to(device)).cpu())
        features = torch.cat(features)
        timing["frontend_transfer_and_inference_wall_seconds"] = time.perf_counter() - started
    else:
        features = torch.stack(features)
    return Clip(features[None], cameras[None], torch.stack(targets)[None], torch.stack(masks)[None],
                torch.ones((1, WINDOW_FRAMES), dtype=torch.bool), torch.tensor([seconds], dtype=torch.float64),
                source, timing, split, input_sha256)
