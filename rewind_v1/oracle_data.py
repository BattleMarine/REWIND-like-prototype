"""One pinned train clip with explicit GT2D presence/confidence oracle features."""

import io
import json
import os
from pathlib import Path
import shutil
import time

import torch

from rewind_archive.contracts import canonical
from rewind_m1.local_rgb import _regular_read
from rewind_m2.data import encode_camera
from rewind_prepare.converter import sha, validate_pose_geometry
from rewind_prepare.extracted import _local, _pose_targets, _raw_pose
from rewind_prepare.spec import validate_spec
from rewind_prepare.temporal_cache import MANIFEST, RawCache, _atomic, _checked_plan
from rewind_prepare.temporal_selection import check_pin, seal
from scripts.v11_recovery import _measure
from scripts.visualize_archvis_v1 import SPEC_SHA
from .data import Clip, _field
from .diagnostic import HISTORICAL, validate_bundle
from .training import _source


ROOT = Path(__file__).resolve().parents[1]
RUN_ID = "V1.4-ORACLE-OVERFIT-001"
SCHEMA = "archvis_v14_oracle_clip_v1"
SEQUENCE = "ArchVisInterior_ArchVis_RT/Day/rp_adanna_rigged_003_ue4/SKM_MenReadingGlasses_Shape_01/003/Standing_Up__1_"
START, FRAMES = 0, 50
MIB = 1024**2
PREPARE_BYTES, PREPARE_SECONDS = 256*MIB, 300
ORACLE = {"source": "gt_2d_oracle_v14", "joint_major": True,
          "view_order": ["left", "right"],
          "per_view": ["x_normalized", "y_normalized", "valid", "confidence", "measurement_present", "reserved_zero", "reserved_zero"],
          "confidence": "1 for valid annotated coordinates, 0 for missing; oracle presence, not visibility",
          "coordinates": "2*xy/1023-1, missing only zeroed, no clamping of offscreen annotations",
          "hand_upper": "gt_upper_oracle", "body_has_gt3d_condition": False,
          "time_basis": "frame_index/25 assumed, measured timestamps/stereo synchronization unverified",
          "recorded_timestamps_verified": False, "stereo_sync_measured": False,
          "original_motion_independence_verified": False,
          "evaluation_scope": "same_train_clip_memorization_only"}


def source_identity():
    return {**_source(), "scripts/visualize_archvis_v1.py": sha((ROOT / "scripts/visualize_archvis_v1.py").read_bytes())}


def oracle_features(xy, valid):
    if (xy.ndim != 4 or xy.shape[1] != 2 or xy.shape[-1] != 2 or valid.shape != xy.shape[:-1]
            or valid.dtype != torch.bool or not torch.isfinite(xy[valid]).all()):
        raise ValueError("v14_oracle_2d_shape_or_mask")
    clean = torch.where(valid[..., None], 2*xy/1023-1, 0.)
    presence = valid.to(xy.dtype)[..., None]
    zeros = torch.zeros_like(presence)
    views = torch.cat((clean, presence, presence, presence, zeros, zeros), dim=-1)
    return views.transpose(1, 2).flatten(1)[None]


def prepare(args):
    started = time.monotonic()
    output = Path(args.output).absolute()
    if output.exists() or output.is_symlink() or shutil.disk_usage(output.parent).free < 512*MIB:
        raise ValueError("v14_preparation_fresh_output_and_space")
    if any(output.is_relative_to(_local(path, directory=True)) for path in
           (args.cache, args.inputs, args.diagnostic_bundle)):
        raise ValueError("v14_prepared_output_inside_sources")
    plan = json.loads(_regular_read(_local(args.inputs / "selection.json"), 16*MIB))
    _checked_plan(plan, HISTORICAL["selection_sha256"])
    _, _, camera_rows = validate_bundle(args.diagnostic_bundle, args.decision_sha256, plan)
    if [SEQUENCE, START] not in plan["train_clip_starts"]:
        raise ValueError("v14_fixed_train_clip")
    spec_raw = _regular_read(_local(args.inputs / "spec.json"), MIB)
    calibration_raw = _regular_read(_local(args.inputs / "calibration.json"), MIB)
    if sha(spec_raw) != plan["spec_sha256"] or sha(calibration_raw) != plan["calibration_sha256"]:
        raise ValueError("v14_geometry_metadata_pin")
    spec, calibration = validate_spec(json.loads(spec_raw)), json.loads(calibration_raw)
    cache = RawCache(args.cache, HISTORICAL["cache_sha256"])
    if (cache.manifest["selection_sha256"] != plan["content_sha256"]
            or cache.manifest["source_kind"] != "native_rgb_png_and_pose_json"
            or cache.manifest["preprocessing"] != "none_raw_source_bytes_v1"):
        raise ValueError("v14_raw_cache_binding")
    keys = [(SEQUENCE, frame) for frame in range(START, START+FRAMES)]
    selected = {tuple(row["key"]): row for row in plan["records"]}
    for key in keys:
        _, row = cache.index[key]
        if row["split"] != "train" or selected[key]["split"] != "train" or row["files"] != selected[key]["files"]:
            raise ValueError("v14_clip_file_pins")
    names = {cache.index[key][0] for key in keys}
    if len(names) != 1:
        raise ValueError("v14_single_pinned_part_required")
    part = next(p for p in cache.manifest["parts"] if p["name"] in names)
    if [tuple(row["key"]) for row in part["records"]] != keys or not 0 < part["bytes"] <= PREPARE_BYTES:
        raise ValueError("v14_part_scope_or_size")
    metadata = [args.inputs / name for name in ("selection.json", "spec.json", "calibration.json")]
    metadata += [args.diagnostic_bundle / name for name in
                 ("diagnostic_decision.json", "overlay_review.json", "audit/audit.json", "audit/camera_candidates.json")]
    member_names = [item["path"].encode("utf-8") for row in part["records"] for item in row["files"].values()]
    # The pinned producer writes ZIP_STORED members without comments/extras or ZIP64 for this 106MB part.
    zip_metadata = FRAMES*(128+sum(46+len(name) for name in member_names))+sum(30+len(name) for name in member_names)
    reserved_reads = (2*part["bytes"] + 3*(args.cache / MANIFEST).stat().st_size
                      + sum(path.stat().st_size for path in metadata) + zip_metadata)
    if reserved_reads > PREPARE_BYTES:
        raise ValueError("v14_preparation_read_budget")
    output.mkdir()
    local_cache = output / "cache"
    local_cache.mkdir()
    with _local(args.cache / part["name"]).open("rb") as source, (local_cache / part["name"]).open("xb") as target:
        if _measure(source, part["bytes"], started+PREPARE_SECONDS, target) != (part["bytes"], part["sha256"]):
            raise ValueError("v14_copied_part_hash")
        target.flush()
        os.fsync(target.fileno())
    manifest_raw = _regular_read(_local(args.cache / MANIFEST), 16*MIB)
    if json.loads(manifest_raw) != cache.manifest:
        raise ValueError("v14_cache_manifest_changed")
    _atomic(local_cache / MANIFEST, manifest_raw)
    local = RawCache(local_cache, HISTORICAL["cache_sha256"])
    pinned_cameras = {tuple(row["key"]): row for row in camera_rows}
    tensors, samples = audit_clip(local, keys, spec, calibration, pinned_cameras, started)
    data = io.BytesIO()
    torch.save(tensors, data)
    raw = data.getvalue()
    if len(raw) > 4*MIB:
        raise ValueError("v14_clip_tensor_size")
    _atomic(output / "oracle_clip.pt", raw)
    _atomic(output / "spec.json", spec_raw)
    valid = tensors["xy_valid"]
    offscreen = valid & ((tensors["xy"] < 0).any(-1) | (tensors["xy"] > 1023).any(-1))
    receipt = seal({"schema": SCHEMA, "run_id": RUN_ID, "data_kind": "dataset_sample", "selection_sha256": plan["content_sha256"],
                    "cache_sha256": HISTORICAL["cache_sha256"], "decision_sha256": args.decision_sha256,
                    "spec_sha256": sha(spec_raw), "calibration_sha256": sha(calibration_raw),
                    "spec": spec, "keys": keys, "split": "train", "oracle": ORACLE,
                    "part": {k: part[k] for k in ("name", "bytes", "sha256")}, "copied_bytes": part["bytes"],
                    "read_bytes_reserved": reserved_reads,
                    "read_reservation": {"raw_part_copy_bytes": part["bytes"],
                                         "local_member_payload_upper_bytes": part["bytes"],
                                         "local_zip_metadata_upper_bytes": zip_metadata,
                                         "scope": "Logical file reads; Drive wire traffic and filesystem read-ahead are not measured."},
                    "mask_counts": {"valid_2d_joint_views": int(valid.sum()), "offscreen_valid_2d_joint_views": int(offscreen.sum()),
                                    "missing_2d_joint_views": int((~valid).sum()), "valid_3d_joints": int(tensors["target_valid"].sum())},
                    "cached_subset_only": True, "samples": samples,
                    "tensor": {"name": "oracle_clip.pt", "bytes": len(raw), "sha256": sha(raw)},
                    "source_code_sha256": source_identity(), "elapsed_seconds": time.monotonic()-started})
    _atomic(output / "oracle_input.json", canonical(receipt)+b"\n")
    return receipt


def audit_clip(cache, keys, spec, calibration, pinned_cameras, started):
    xy, xy_valid, target, target_valid, cameras, samples = [], [], [], [], [], []
    from collections import defaultdict

    for key in keys:
        if time.monotonic()-started > PREPARE_SECONDS:
            raise TimeoutError("v14_clip_preparation_time")
        raw = cache.read_pair(*key, split="train")
        doc = _raw_pose(raw["pose"], key[1])
        residuals = defaultdict(list)
        pose = validate_pose_geometry(doc, spec, calibration, residuals)
        if not residuals["projection_px"]:
            raise ValueError("v14_pose_projection_missing")
        camera = encode_camera(pose, 100.)
        prior = encode_camera(pinned_cameras[key]["camera_pose"], 100.)
        if not torch.allclose(camera, prior, atol=1e-5, rtol=0):
            raise ValueError("v14_camera_audit_pin")
        fields = _pose_targets(doc, spec)
        views, masks = zip(*(_field(field, 2) for field in fields[:2]))
        xyz, mask = _field(fields[2], 3)
        xy.append(torch.stack(views))
        xy_valid.append(torch.stack(masks))
        target.append(xyz)
        target_valid.append(mask)
        cameras.append(camera)
        samples.append({"key": key, "split": "train", "files": cache.index[key][1]["files"],
                        "geometry_max": {name: max(values) for name, values in residuals.items() if values}})
    xy, xy_valid = torch.stack(xy), torch.stack(xy_valid)
    return {"xy": xy, "xy_valid": xy_valid, "features": oracle_features(xy, xy_valid),
            "camera": torch.stack(cameras)[None], "target_mm": torch.stack(target)[None],
            "target_valid": torch.stack(target_valid)[None],
            "time_valid": torch.ones((1, len(keys)), dtype=torch.bool),
            "seconds": torch.tensor([[key[1]/25 for key in keys]], dtype=torch.float64)}, samples


def load_prepared(root, expected):
    root = _local(root, directory=True)
    receipt = json.loads(_regular_read(root / "oracle_input.json", 4*MIB))
    check_pin(receipt, SCHEMA, expected)
    spec_raw = _regular_read(root / "spec.json", MIB)
    if (receipt["run_id"] != RUN_ID or receipt["oracle"] != ORACLE
            or receipt["selection_sha256"] != HISTORICAL["selection_sha256"]
            or receipt["cache_sha256"] != HISTORICAL["cache_sha256"] or receipt["split"] != "train"
            or receipt["keys"] != [[SEQUENCE, f] for f in range(START, START+FRAMES)]
            or receipt["source_code_sha256"] != source_identity()):
        raise ValueError("v14_prepared_identity")
    if sha(spec_raw) != SPEC_SHA or receipt["spec_sha256"] != SPEC_SHA or json.loads(spec_raw) != receipt["spec"]:
        raise ValueError("v14_prepared_joint_spec_pin")
    validate_spec(receipt["spec"])
    pin = receipt["tensor"]
    raw = _regular_read(root / "oracle_clip.pt", 4*MIB)
    if pin != {"name": "oracle_clip.pt", "bytes": len(raw), "sha256": sha(raw)}:
        raise ValueError("v14_prepared_tensor_pin")
    data = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=True)
    joints = len(receipt["spec"]["joint_names"])
    if (data["xy"].shape != (FRAMES, 2, joints, 2) or data["target_mm"].shape != (1, FRAMES, joints, 3)
            or data["target_valid"].shape != (1, FRAMES, joints)
            or data["target_valid"].dtype != torch.bool or data["time_valid"].shape != (1, FRAMES)
            or data["time_valid"].dtype != torch.bool or not data["time_valid"].all()
            or data["camera"].shape != (1, FRAMES, 2, 9)
            or any(data[key].dtype != torch.float32 for key in ("xy", "features", "camera", "target_mm"))
            or not torch.equal(data["seconds"], torch.arange(START, START+FRAMES, dtype=torch.float64)[None]/25)
            or not torch.equal(data["features"], oracle_features(data["xy"], data["xy_valid"]))
            or not all(torch.isfinite(value).all() for value in data.values())):
        raise ValueError("v14_prepared_tensor_contract")
    clip = Clip(data["features"], data["camera"], data["target_mm"], data["target_valid"], data["time_valid"],
                data["seconds"], ORACLE["source"], split="train", input_sha256=expected)
    return clip, receipt
