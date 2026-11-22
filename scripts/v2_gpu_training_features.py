"""Bounded disk feature materialization, with verified same-seed Drive shards."""

import json
from pathlib import Path
import shutil
import tempfile
import time
import uuid
import zipfile

import torch

from rewind_archive.contracts import canonical
from rewind_prepare.temporal_cache import _atomic, _hash_file
from rewind_prepare.temporal_selection import seal
from rewind_v1.full_data import FeatureMap, FeatureStore
from rewind_v1.v2_retention import copy_verified
from scripts.archvis_pilot_bootstrap import extract_pinned_zip
from scripts.v2_gpu_training_storage import MIB, publish_json, read_sealed, transfer_in


GROUP = 24
GROUP_CAP = 32*MIB
SCHEMA = "rewind_v2_gpu_features_group_v1"


class PersistentFeatures(FeatureStore):
    def __init__(self, root, data, frontend_pin, drive, cfg, ledger, owner):
        super().__init__(root, data, frontend_pin)
        self.drive = drive / self.identity
        self.cfg, self.ledger, self.owner = cfg, ledger, owner
        self.root.mkdir(parents=True, exist_ok=True)
        self.drive.mkdir(parents=True, exist_ok=True)

    def _compute(self, frontend, size, key, split, used):
        clip = self.data.clip(key, split, frontend, size, self.identity)
        tensor = {name: getattr(clip, name) for name in ("features", "camera", "target_mm", "target_valid", "time_valid", "seconds")}
        with tempfile.TemporaryDirectory(dir=self.root, prefix=".feature-") as directory:
            staged = Path(directory)
            path = staged / "clip.pt"
            torch.save(tensor, path)
            size_bytes = path.stat().st_size
            if size_bytes > 4*MIB or used+size_bytes+4096 > self.cfg["resources"]["feature_bytes"]:
                raise ValueError("v2_gpu_feature_clip_cap")
            doc = seal({"schema": "archvis_v12_predicted_feature_v1", "input_sha256": self.identity,
                        "key": list(key), "split": split, "real_frames": self.data.real_frames(key, split),
                        "tensor": {"bytes": size_bytes, "sha256": _hash_file(path)}})
            folder = self._folder(key, split)
            temporary = staged / "ready"
            temporary.mkdir()
            path.rename(temporary / path.name)
            _atomic(temporary / "receipt.json", canonical(doc))
            temporary.rename(folder)
        return doc

    def _publish(self, tag, entries, marker, names):
        self.owner()
        with tempfile.TemporaryDirectory(dir=self.root, prefix=".feature-pack-") as directory:
            name = f"{tag}-{uuid.uuid4().hex}.zip"
            path = Path(directory) / name
            with zipfile.ZipFile(path, "x") as archive:
                for split, key in entries:
                    self.ledger.check()
                    folder = self._folder(key, split)
                    for file in ("clip.pt", "receipt.json"):
                        archive.write(folder / file, f"{folder.name}/{file}")
            pin = {"name": name, "bytes": path.stat().st_size, "sha256": _hash_file(path)}
            if pin["bytes"] > GROUP_CAP:
                raise ValueError("v2_gpu_feature_group_cap")
            copy_verified(path, self.drive / name, pin, self.ledger,
                          self.ledger.clock()+min(self.ledger.remaining(), self.cfg["copy_seconds"]))
            publish_json(marker, SCHEMA, {"identity": self.identity, "folders": names, "archive": pin}, self.owner)

    def _restore(self, doc, entries, names):
        pin = doc["archive"]
        if Path(pin["name"]).name != pin["name"] or not 0 < pin["bytes"] <= GROUP_CAP:
            raise ValueError("v2_gpu_feature_group_pin")
        if all(self._folder(key, split).is_dir() for split, key in entries):
            return
        with tempfile.TemporaryDirectory(dir=self.root, prefix=".feature-restore-") as directory:
            work = Path(directory)
            local = work / "input.zip"
            self.ledger.reserve("read", pin["bytes"] + 1)
            transfer_in(self.drive / pin["name"], local, pin, self.ledger)
            stage = work / "ready"
            extract_pinned_zip(local, pin["sha256"], stage, max_members=2*GROUP,
                               max_zip_bytes=GROUP_CAP, max_output_bytes=GROUP_CAP)
            if sorted(p.name for p in stage.iterdir()) != sorted(names):
                raise ValueError("v2_gpu_feature_group_scope")
            for split, key in entries:
                folder = self._folder(key, split)
                if not folder.exists():
                    shutil.move(stage / folder.name, folder)

    def build(self, frontend, size):
        started = time.monotonic()
        entries = [(split, key) for split, keys in self.data.clips.items() for key in keys]
        used, made = 0, 0
        for start in range(0, len(entries), GROUP):
            self.ledger.check()
            group = entries[start:start+GROUP]
            names = [self._folder(key, split).name for split, key in group]
            tag = f"group-{start//GROUP:05d}"
            marker = self.drive / f"{tag}.json"
            durable = marker.is_file()
            if durable:
                doc = read_sealed(marker, SCHEMA)
                if doc["identity"] != self.identity or doc["folders"] != names:
                    raise ValueError("v2_gpu_feature_group_identity")
                self._restore(doc, group, names)
            for split, key in group:
                self.ledger.check()
                if self._folder(key, split).exists():
                    receipt = self._receipt(key, split)
                else:
                    receipt = self._compute(frontend, size, key, split, used)
                    made += 1
                used += receipt["tensor"]["bytes"] + 4096
                if used > self.cfg["resources"]["feature_bytes"]:
                    raise ValueError("v2_gpu_feature_total_cap")
            if not durable:
                self._publish(tag, group, marker, names)
        return {s: FeatureMap(self, s) for s in self.data.clips}, {
            "feature_identity": self.identity, "feature_disk_bytes": used,
            "feature_new_clips": made, "feature_wall_seconds": time.monotonic()-started}
