"""Fixed frame/clip data sources for the v1 runner."""

from pathlib import Path
import time

import torch

from rewind_archive.contracts import digest
from rewind_data.audit import read_json
from rewind_m1.archvis_v0 import VALIDATION
from rewind_m1.local_rgb import _regular_read
from rewind_m1.rgb_frontend_artifacts import scale_xy
from rewind_prepare.converter import sha
from rewind_prepare.extracted import IMAGE_SIZE, MAX_MANIFEST_BYTES, ROLES, _local, _pose_targets, _raw_pose
from rewind_prepare.spec import validate_spec
from rewind_prepare.temporal_cache import MANIFEST, RawCache, _check_part, _checked_plan, restore
from rewind_prepare.temporal_selection import contiguous_clips
from .data import Clip, REQUIRED_EVIDENCE, _checked_evidence, _field, frame_image, load_clip, reviewed_clip
from .frontend import ARCHITECTURE, predicted_features
from .teacher import WINDOW_FRAMES


def _keys(clips, width):
    return {(sequence, index) for sequence, start in clips for index in range(start, start + width)}


class RunData:
    def coverage(self):
        used = _keys(self.clips["validation"], self.width)
        val = set(self.frames["validation"])
        return {"frontend_validation_frames": len(val), "validation_keys_sha256": digest(sorted(val)),
                "teacher_validation_frames": len(used), "teacher_validation_clips": len(self.clips["validation"]),
                "teacher_validation_keys_sha256": digest(sorted(used)),
                "teacher_validation_excluded_keys": sorted(val - used),
                "train_unique_frames": len(self.frames["train"])}

    def _validate_membership(self):
        train, val = (set(self.frames[split]) for split in ("train", "validation"))
        if train & val or {key[0] for key in train} & {key[0] for key in val}:
            raise ValueError("rewind_v1_run_split_overlap")
        if (_keys(self.clips["train"], self.width) != train or
                len(train) != self.width * len(self.clips["train"]) or
                not self.clips["validation"] or not _keys(self.clips["validation"], self.width) <= val):
            raise ValueError("rewind_v1_run_clip_coverage")


class _CacheData(RunData):
    width = WINDOW_FRAMES

    def __init__(self, args, cfg, budget):
        self.budget = budget
        self.spec_path = _local(args.spec)
        self.plan = read_json(_regular_read(_local(args.selection), MAX_MANIFEST_BYTES))
        _checked_plan(self.plan, args.selection_sha256)
        if self.plan["validation_keys_sha256"] != VALIDATION:
            raise ValueError("rewind_v1_run_frozen_validation")
        spec_raw = _regular_read(self.spec_path, 256 * 1024)
        if sha(spec_raw) != self.plan["spec_sha256"]:
            raise ValueError("rewind_v1_run_spec_pin")
        self.spec = validate_spec(read_json(spec_raw))
        self.frames = {split: sorted(tuple(row["key"]) for row in self.plan["records"] if row["split"] == split)
                       for split in ("train", "validation")}
        self.clips = {"train": [tuple(row) for row in self.plan["train_clip_starts"]],
                      "validation": sorted(clip for _, clip in contiguous_clips(
                          self.frames["validation"], seed=self.plan["seed"], width=self.width))}
        self._validate_membership()
        self.scale = cfg["teacher"]["target_scale_mm"]
        context_identity = self._time_identity(args)
        local_ready = args.cache_local is not None and (args.cache_local / MANIFEST).exists()
        self.cache = RawCache(args.cache_local if local_ready else args.cache, args.cache_sha256)
        manifest = self.cache.manifest
        expected_binding = {key: self.plan[key] for key in ("spec_sha256", "calibration_sha256", "dataset_version",
                                                           "inventory_sha256", "validation_keys_sha256")}
        if (manifest["selection_sha256"] != args.selection_sha256 or manifest["binding"] != expected_binding
                or manifest["source_kind"] != "native_rgb_png_and_pose_json"
                or manifest["preprocessing"] != "none_raw_source_bytes_v1"
                or set(self.cache.index) != set(self.frames["train"] + self.frames["validation"])):
            raise ValueError("rewind_v1_run_cache_binding")
        for row in self.plan["records"]:
            _, cached = self.cache.index[tuple(row["key"])]
            if row["split"] != cached["split"] or any(
                any(cached["files"][role].get(key) != value for key, value in row["files"][role].items())
                for role in ROLES
            ):
                raise ValueError("rewind_v1_run_cache_pair_pin")
        self._check_cache(manifest)
        if args.cache_local and not local_ready:
            self.restore_metrics = restore(args.cache, expected_sha256=args.cache_sha256, output=args.cache_local,
                                           max_disk_bytes=cfg["limits"]["cache_restore_bytes"],
                                           max_seconds=budget.remaining())
            self.cache = RawCache(args.cache_local, args.cache_sha256)
        else:
            start = time.monotonic()
            for part in manifest["parts"]:
                budget.check()
                _check_part(self.cache.root / part["name"], part, budget.check)
            self.restore_metrics = {"mode": "use_existing_cache", "copied_bytes": 0,
                                    "verification_wall_seconds": time.monotonic()-start}
        self.identity = {"data_kind": "dataset_sample", "selection_sha256": args.selection_sha256,
                         "cache_sha256": args.cache_sha256, "spec_sha256": sha(spec_raw),
                         "cache_producer_sha256": manifest["producer_code_sha256"], **context_identity}

    def _check_cache(self, manifest):
        pass

    def frame(self, key, split, size):
        start = time.monotonic()
        raw = self.cache.read_pair(*key, split=split)
        read = time.monotonic() - start
        start = time.monotonic()
        fields = _pose_targets(_raw_pose(raw["pose"], key[1]), self.spec)
        xy, valid = zip(*(_field(field, 2) for field in fields[:2]))
        inputs = frame_image(raw, size), scale_xy(torch.stack(xy), IMAGE_SIZE, size), torch.stack(valid)
        return inputs, {"local_cache_read_wall_seconds": read, "cpu_preprocess_wall_seconds": time.monotonic()-start}


class ActualData(_CacheData):
    def _time_identity(self, args):
        self.evidence = read_json(_regular_read(_local(args.evidence), MAX_MANIFEST_BYTES))
        _checked_evidence(self.plan, self.evidence, args.evidence_sha256)
        review = _local(args.review_dir, directory=True)
        for key in REQUIRED_EVIDENCE:
            if sha(_regular_read(_local(review / (key + ".json")), 4 * 1024**2)) != self.evidence["review_artifact_sha256"][key]:
                raise ValueError("rewind_v1_run_review_pin:" + key)
        for split in self.clips:
            for sequence, start in self.clips[split]:
                self.budget.check()
                reviewed_clip(self.evidence, sequence, range(start, start + self.width), self.scale)
        return {"evidence_sha256": args.evidence_sha256, "review_pins": self.evidence["review_artifact_sha256"]}

    def clip(self, key, split, frontend, size, pin):
        return load_clip(self.cache, self.plan, *key, split=split, selection_sha256=self.identity["selection_sha256"],
                         evidence=self.evidence, evidence_sha256=self.identity["evidence_sha256"],
                         spec_path=self.spec_path, scale_mm=self.scale, source="predicted_2d",
                         frontend=frontend, frontend_size=size, frontend_sha256=pin, check_budget=self.budget.check)


class FixtureData(RunData):
    width = 4

    def __init__(self):
        self.spec = {"joint_names": ["left_wrist", "right_wrist", "left_tip", "right_tip"],
                     "body_joint_names": ["left_wrist", "right_wrist"], "upper_joint_names": ["left_wrist", "right_wrist"],
                     "hands": {"left": ["left_wrist", "left_tip"], "right": ["right_wrist", "right_tip"]},
                     "wrists": {"left": "left_wrist", "right": "right_wrist"},
                     "edges": [["left_wrist", "right_wrist"], ["left_wrist", "left_tip"], ["right_wrist", "right_tip"]]}
        self.clips = {"train": [("fixture_train_a", 0), ("fixture_train_b", 0)], "validation": [("fixture_val", 0)]}
        self.frames = {split: sorted(_keys(clips, self.width)) for split, clips in self.clips.items()}
        self.frames["validation"].append(("fixture_val", 8))
        self._validate_membership()
        self.identity = {"data_kind": "synthetic_fixture", "fixture_version": "v1_runner_fixture_v1",
                         "fixture_sha256": digest({"spec": self.spec, "frames": self.frames})}

    def frame(self, key, split, size):
        if key not in self.frames[split]:
            raise ValueError("rewind_v1_fixture_split")
        width, height = size
        seed = int(digest(key)[:15], 16)
        image = torch.rand((2, 3, height, width), generator=torch.Generator().manual_seed(seed))
        xy = torch.tensor([[[.3, .4], [.7, .4], [.2, .5], [.8, .5]]] * 2) * torch.tensor([width-1, height-1])
        return (image, xy, torch.ones((2, 4), dtype=torch.bool)), {}

    def clip(self, key, split, frontend, size, pin):
        if key not in self.clips[split]:
            raise ValueError("rewind_v1_fixture_clip")
        sequence, start = key
        inputs = torch.stack([self.frame((sequence, i), split, size)[0][0] for i in range(start, start+self.width)])
        features = predicted_features(frontend, inputs.to(next(frontend.parameters()).device)).cpu()
        target = torch.arange(self.width * 4 * 3, dtype=torch.float32).view(1, self.width, 4, 3)
        valid = torch.ones((1, self.width, 4), dtype=torch.bool)
        fingerprint = digest({"data": self.identity, "frontend": ARCHITECTURE, "size": size, "weights": pin})
        return Clip(features[None], torch.zeros((1, self.width, 2, 9)), target, valid, valid[..., 0],
                    torch.arange(self.width, dtype=torch.float64)[None] / 30, "predicted_2d", split=split,
                    input_sha256=fingerprint)
