"""Full-coverage train clips and bounded disk-backed predicted-2D feature maps."""

from collections.abc import Mapping
import io
import json
from pathlib import Path
import tempfile
import time

import torch

from rewind_archive.contracts import canonical, digest
from rewind_m1.local_rgb import _regular_read
from rewind_prepare.converter import sha
from rewind_prepare.extracted import MIB, _local
from rewind_prepare.temporal_cache import _atomic, _check_part, _hash_file
from rewind_prepare.temporal_selection import check_pin, seal
from .data import Clip
from .frontend import ARCHITECTURE, predicted_features
from .full_cache import FrameCache
from .full_contract import MANIFEST_SHA, RUN_ID, diagnostic_decision
from .run_data import RunData


FEATURE_LIMIT = 256*MIB


def pad_clip(features, camera, target, valid, *, sequence, start, split, input_sha256, width=50):
    """Append only zero placeholders; real slots and per-joint validity remain distinct."""
    real, joints = target.shape[:2]
    if (not 1 <= real <= width or features.shape != (real, 14*joints)
            or camera.shape != (real, 2, 9) or target.shape != (real, joints, 3)
            or valid.shape != (real, joints) or valid.dtype != torch.bool
            or any(v.dtype != torch.float32 or not torch.isfinite(v).all() for v in (features, camera, target))):
        raise ValueError("v12_training_pad_contract")
    def padded(value):
        result = value.new_zeros((width, *value.shape[1:]))
        result[:real] = value
        return result[None]
    time_valid = torch.arange(width)[None] < real
    return Clip(padded(features), padded(camera), padded(target), padded(valid), time_valid,
                torch.arange(start, start+width, dtype=torch.float64)[None]/25,
                "predicted_2d", split=split, input_sha256=input_sha256)


class FullData(RunData):
    width = 50

    def __init__(self, manifest, plan, spec_raw, cache_root, config, config_sha, budget):
        self.manifest, self.plan, self.spec, self.budget = manifest, plan, json.loads(spec_raw), budget
        self.frames = {split: sorted((r["sequence_id"], r["frame_index"]) for r in manifest["records"] if r["split"] == split)
                       for split in ("train", "validation")}
        self.clip_rows = {(r["sequence"], r["start_frame"]): r for r in plan["train_clips"]}
        self.clips = {"train": sorted(self.clip_rows), "validation": [tuple(k) for k in plan["teacher_validation_clips"]]}
        self.cache = FrameCache(cache_root, manifest, spec_raw, budget)
        decision = diagnostic_decision(config_sha, plan["content_sha256"])
        self.identity = {"data_kind": "dataset_sample", "run_kind": "nominal_25fps_diagnostic", "run_id": RUN_ID,
                         "manifest_sha256": MANIFEST_SHA, "plan_sha256": plan["content_sha256"],
                         "frame_cache_sha256": self.cache.identity, "spec_sha256": sha(spec_raw),
                         "decision_sha256": decision["content_sha256"], "diagnostic_decision": decision}
        self.restore_metrics = {"mode": "verified_preprocessed_cache", "copied_bytes": 0,
                                "verified_bytes": getattr(budget, "cache_verified_bytes", 0)}

    def coverage(self):
        return {**super().coverage(), "train_clips": len(self.clips["train"]),
                "train_real_frames": len(self.frames["train"]), "train_padding_slots": self.plan["summary"]["train"]["padding_slots"],
                "train_repeated_real_frames": 0, "train_clip_kinds": self.plan["summary"]["train"],
                "validation_reference_only": True}

    def frame(self, key, split, size):
        if list(size) != [128, 128]:
            raise ValueError("v12_training_cached_size")
        started = time.monotonic()
        image, target = self.cache.frame(key, split)
        return (image, target["xy"], target["xy_valid"]), {"prepared_frame_read_wall_seconds": time.monotonic()-started}

    def real_frames(self, key, split):
        if key not in self.clips[split]:
            raise ValueError("v12_training_clip_membership")
        return self.clip_rows[key]["real_frames"] if split == "train" else self.width

    def clip(self, key, split, frontend, size, input_sha):
        if list(size) != [128, 128] or frontend.training or any(p.requires_grad for p in frontend.parameters()):
            raise ValueError("v12_training_feature_frontend")
        sequence, start = key
        count, features, metadata, batch = self.real_frames(key, split), [], [], []
        started = time.monotonic()
        for frame in range(start, start+count):
            self.budget.check()
            image, target = self.cache.frame((sequence, frame), split)
            metadata.append(target)
            batch.append(image)
            if len(batch) == 2 or frame == start+count-1:
                with torch.no_grad():
                    features.append(predicted_features(frontend, torch.stack(batch).to(next(frontend.parameters()).device)).cpu())
                batch.clear()
        clip = pad_clip(torch.cat(features), torch.stack([r["camera"] for r in metadata]),
                        torch.stack([r["target_mm"] for r in metadata]), torch.stack([r["target_valid"] for r in metadata]),
                        sequence=sequence, start=start, split=split, input_sha256=input_sha)
        clip.timings["feature_build_wall_seconds"] = time.monotonic()-started
        return clip


class FeatureMap(Mapping):
    """One clip is deserialized at a time; no tensor dictionary grows with the dataset."""

    def __init__(self, store, split):
        self.store, self.split = store, split
        self.keys = tuple(store.data.clips[split])

    def __iter__(self):
        return iter(self.keys)

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, key):
        if key not in self.keys:
            raise KeyError(key)
        return self.store.load(key, self.split)


class FeatureStore:
    def __init__(self, root, data, frontend_pin):
        self.root, self.data = Path(root).absolute(), data
        self.binding = {"data": data.identity, "frontend_weight_sha256": frontend_pin,
                        "frontend": ARCHITECTURE, "size": [128, 128], "padding": "post_zero_time_invalid_v1"}
        self.identity = digest(self.binding)
        self.names = {(split, key): f"{split}-{i:05d}" for split, keys in data.clips.items() for i, key in enumerate(keys)}

    def _folder(self, key, split):
        return self.root / self.names[split, key]

    def _receipt(self, key, split):
        folder = _local(self._folder(key, split), directory=True)
        receipt = json.loads(_regular_read(folder / "receipt.json", MIB))
        check_pin(receipt, "archvis_v12_predicted_feature_v1", receipt.get("content_sha256"))
        if (receipt["input_sha256"] != self.identity or receipt["key"] != list(key) or receipt["split"] != split
                or receipt["real_frames"] != self.data.real_frames(key, split)
                or {p.name for p in folder.iterdir()} != {"clip.pt", "receipt.json"}):
            raise ValueError("v12_training_feature_binding")
        _check_part(folder / "clip.pt", {"name": "clip.pt", **receipt["tensor"]}, self.data.budget.check)
        return receipt

    def build(self, frontend, size):
        self.root.mkdir(parents=True, exist_ok=True)
        _local(self.root, directory=True)
        used, made = 0, 0
        for split, keys in self.data.clips.items():
            for key in keys:
                self.data.budget.check()
                folder = self._folder(key, split)
                if folder.exists():
                    receipt = self._receipt(key, split)
                else:
                    clip = self.data.clip(key, split, frontend, size, self.identity)
                    tensor = {name: getattr(clip, name) for name in ("features", "camera", "target_mm", "target_valid", "time_valid", "seconds")}
                    with tempfile.TemporaryDirectory(dir=self.root, prefix=".feature-") as directory:
                        staged = Path(directory) / "ready"
                        staged.mkdir()
                        torch.save(tensor, staged / "clip.pt")
                        size_bytes = (staged / "clip.pt").stat().st_size
                        if size_bytes > 4*MIB or used+size_bytes+MIB > FEATURE_LIMIT:
                            raise ValueError("v12_training_feature_disk_limit")
                        receipt = seal({"schema": "archvis_v12_predicted_feature_v1", "input_sha256": self.identity,
                                        "key": key, "split": split, "real_frames": self.data.real_frames(key, split),
                                        "tensor": {"bytes": size_bytes, "sha256": _hash_file(staged / "clip.pt")}})
                        _atomic(staged / "receipt.json", canonical(receipt)+b"\n")
                        staged.rename(folder)
                    made += 1
                used += receipt["tensor"]["bytes"] + 4096
                if used+4*MIB > FEATURE_LIMIT:
                    raise ValueError("v12_training_feature_disk_limit")
        return {"train": FeatureMap(self, "train"), "validation": FeatureMap(self, "validation")}, {
            "feature_disk_bytes": used, "feature_new_clips": made, "feature_identity": self.identity}

    def load(self, key, split):
        receipt = self._receipt(key, split)
        raw = _regular_read(self._folder(key, split) / "clip.pt", 4*MIB)
        if sha(raw) != receipt["tensor"]["sha256"]:
            raise ValueError("v12_training_feature_tensor_pin")
        tensor = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=True)
        clip = Clip(**tensor, source="predicted_2d", split=split, input_sha256=self.identity)
        real = self.data.real_frames(key, split)
        if (clip.time_valid.shape != (1, self.data.width) or clip.time_valid.dtype != torch.bool
                or not torch.equal(clip.time_valid, torch.arange(self.data.width)[None] < real)
                or clip.target_valid[..., real:, :].any()
                or not all(torch.isfinite(value).all() for value in tensor.values())):
            raise ValueError("v12_training_feature_mask")
        return clip
