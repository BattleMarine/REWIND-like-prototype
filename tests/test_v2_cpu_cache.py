"""Exercise the CPU entry with PNG/pose payloads and the real v2 retention format."""

import io
import json
from pathlib import Path
import shutil
import tempfile
import time
import unittest
from unittest.mock import patch
import zipfile

import numpy as np
from PIL import Image
import torch

from rewind_archive.contracts import canonical, digest
from rewind_prepare.temporal_selection import seal
from rewind_v1 import v2_contract as contract
from scripts import build_v2_cpu_cache as builder, run_v2_cpu_cache as runner


class CpuCacheTest(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.reference = self.root / "reference"
        (self.reference / "split").mkdir(parents=True)
        for name in ("configs", "scripts", "rgb", "pose", "runs"):
            (self.root / name).mkdir()
        original = contract.ROOT
        for name in ("configs/rewind_v2.json", "scripts/build_v2.py"):
            shutil.copyfile(original / name, self.root / name)
        root = patch.object(contract, "ROOT", self.root)
        root.start()
        self.addCleanup(root.stop)
        memory = patch.object(runner, "available_ram", return_value=8 * 1024**3)
        memory.start()
        self.addCleanup(memory.stop)
        self.spec = {
            "joint_names": ["one"],
            "coordinates": {"mm_per_raw_unit": 10., "raw_consistency_tolerance": 1e-5,
                            "image_sizes": [[1024, 1024], [1024, 1024]]},
            "pinned_sources": {"split_sha256": {}},
        }
        self.owners = {
            f"{scene}/Day/Avatar/Actor/001/{split}": split
            for scene in contract.SCENARIOS for split in ("train", "validation", "test")
        }
        for split in ("train", "validation", "test"):
            raw = ("\n".join(s for s, p in self.owners.items() if p == split) + "\n").encode()
            (self.reference / "split" / f"{split}.txt").write_bytes(raw)
            self.spec["pinned_sources"]["split_sha256"][split] = contract.sha(raw)
        spec_raw = canonical(self.spec)
        (self.reference / "spec.json").write_bytes(spec_raw)
        _, splits = contract.official_splits(self.reference / "split", self.spec, contract.SplitReader())

        stream = io.BytesIO()
        Image.new("RGB", (1024, 1024), (15, 25, 35)).save(stream, format="PNG")
        image = stream.getvalue()
        camera = {"trans": [1, 2, 3], "rot": [0, 0, 0], "open3d_trans": [1, -2, 3],
                  "open3d_rot": [0, 0, 0], "device_pts3d": [0, 0, 0]}
        first = min(s for s, p in self.owners.items() if p != "test")
        records = []
        for sequence, split in sorted(self.owners.items()):
            if split == "test":
                continue
            for frame in range(128 if sequence == first else 1):
                pose = canonical({"frame": frame, "camera_left": camera, "camera_right": camera,
                                  "joints": {"one": {"device_pts3d": [1, 2, 3], "camera_left_pts2d": [1, 2],
                                                     "camera_right_pts2d": [3, 4]}}})
                files = {}
                for role in ("left", "right", "pose"):
                    is_pose = role == "pose"
                    path = f"{sequence}/json/frame_{frame}.json" if is_pose else (
                        f"{sequence}/fisheye_final_image/camera_{role}/final_{frame}.png")
                    content = pose if is_pose else image
                    file = self.root / ("pose" if is_pose else "rgb") / path
                    file.parent.mkdir(parents=True, exist_ok=True)
                    file.write_bytes(content)
                    files[role] = {"path": path, "size_bytes": len(content), "sha256": contract.sha(content)}
                records.append({"sequence_id": sequence, "frame_index": frame, "split": split, "files": files})
        self.records = records
        inventory = seal({
            "schema": "rewind_v2_frozen_inventory_v1", "excluded": [], "failed": False, "capped": False,
            "missing": [], "duplicates": [], "sequences": [s for s, p in sorted(self.owners.items()) if p != "test"],
            "usable_keys": [[r["sequence_id"], r["frame_index"]] for r in records],
        })
        manifest = seal({
            "schema": "rewind_v2_audited_manifest_v1", "scenarios": list(contract.SCENARIOS),
            "spec_sha256": digest(self.spec), "official_splits": splits, "coordinates": "device_mm",
            "inventory": inventory, "records": records,
        })
        raw = canonical(manifest)
        self.archive = self.root / "manifest.zip"
        with zipfile.ZipFile(self.archive, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            archive.writestr(runner.MANIFEST_NAME, raw)
        self.pins = {
            "spec_sha256": contract.sha(spec_raw), "frames": len(records), "manifest_bytes": len(raw),
            "manifest_file_sha256": contract.sha(raw), "manifest_content_sha256": manifest["content_sha256"],
            "archive_sha256": contract.sha(self.archive.read_bytes()), "archive_bytes": self.archive.stat().st_size,
            "source_identity": digest(contract.source()), "revision": "0" * 40,
            "config_sha256": contract.sha((self.root / "configs/rewind_v2.json").read_bytes()),
            "first_128_source_bytes": sum(p["size_bytes"] for r in records[:128] for p in r["files"].values()),
        }
        self.pins_path = self.root / "pins.json"
        self.pins_path.write_bytes(canonical(self.pins))
        self.retained = self.root / "retained"

    def invocation(self, label, *, cache_name="cache", previous=None):
        work = self.root / "runs" / label
        work.mkdir()
        args = ["--pins", str(self.pins_path), "--work", str(work), "--cache", str(work.parent / cache_name),
                "--retained", str(self.retained), "--rgb-root", str(self.root / "rgb"),
                "--pose-root", str(self.root / "pose"), "--manifest-archive", str(self.archive),
                "--worker-seconds", "28000"]
        if previous:
            args += ["--retained-catalog-sha256", previous]
        with patch("builtins.print"):
            runner.main(args)
        return json.loads((work / "result.json").read_bytes())

    def test_prepares_128_then_restores_real_prefix_with_original_payloads_missing(self):
        first = self.invocation("first")
        self.assertEqual(first["local_completed_frames"], 128)
        self.assertEqual(first["prepare_read_bytes"], self.pins["first_128_source_bytes"])
        self.assertEqual(first["newly_prepared_frames"], 128)
        self.assertEqual(first["retention"]["verified_shards"], 1)
        previous = first["retention"]["catalog_file_sha256"]
        saved = self.retained / first["retained_package"]["name"]
        self.assertEqual(contract.sha(saved.read_bytes()), first["retained_package"]["sha256"])
        with zipfile.ZipFile(saved) as archive:
            images = np.load(io.BytesIO(archive.read("images.npy")), allow_pickle=False)
            targets = torch.load(io.BytesIO(archive.read("targets.pt")), map_location="cpu", weights_only=True)
        self.assertEqual(images.shape, (128, 2, 3, 128, 128))
        self.assertEqual(images.dtype, np.float32)
        self.assertEqual(targets["rotation64"].shape, (128, 2, 3, 3))
        self.assertEqual(targets["translation64"].dtype, torch.float64)
        self.assertEqual(targets["target_mm"][0, 0].tolist(), [10., 20., 30.])
        with self.assertRaisesRegex(ValueError, "catalog_pin"):
            self.invocation("wrong-pin", previous="f" * 64)
        row = self.records[0]["files"]["left"]
        (self.root / "rgb" / row["path"]).unlink()
        second = self.invocation("second", cache_name="new-vm", previous=previous)
        self.assertEqual(second["restore"]["restored_frames"], 128)
        self.assertEqual(second["target_start_index"], 128)
        self.assertEqual(second["retention"]["verified_shards"], 2)
        self.assertEqual(second["newly_prepared_frames"], 5)
        self.assertEqual((self.root / "runs/cache/shard-00000/images.npy").read_bytes(),
                         (self.root / "runs/new-vm/shard-00000/images.npy").read_bytes())
        self.assertEqual(len(second["imported_usage"]), 1)

    def test_same_vm_reuses_shard_after_copy_failure_and_imports_usage(self):
        first = self.invocation("first")
        previous = first["retention"]["catalog_file_sha256"]
        with patch.object(runner.retention, "user_copy_cache", side_effect=OSError("copy failure")):
            with self.assertRaisesRegex(OSError, "copy failure"):
                self.invocation("copy-failed", previous=previous)
        report = json.loads((self.root / "runs/copy-failed/result.json").read_bytes())
        self.assertEqual(report["local_completed_frames"], 133)
        self.assertEqual(report["status"], "failed")
        self.assertEqual(len(report["imported_usage"]), 1)
        self.assertEqual(contract.sha((self.retained / "catalog.json").read_bytes()), previous)
        retried = self.invocation("copy-retry", previous=previous)
        self.assertEqual(retried["prepare_read_bytes"], 0)
        self.assertEqual(retried["newly_prepared_frames"], 0)
        self.assertEqual(retried["retention"]["verified_shards"], 2)
        self.assertEqual(retried["imported_usage"], report["imported_usage"])

    def test_source_hash_failure_preserves_no_retained_catalog(self):
        row = self.records[0]["files"]["left"]
        path = self.root / "rgb" / row["path"]
        raw = path.read_bytes()
        path.write_bytes(b"x" + raw[1:])
        with self.assertRaisesRegex(ValueError, "v2_source_sha"):
            self.invocation("bad-source")
        self.assertFalse((self.retained / "catalog.json").exists())
        report = json.loads((self.root / "runs/bad-source/result.json").read_bytes())
        self.assertEqual(report["usage"]["status"], "failed")

    def test_notebook_default_off_compiles_and_needs_no_mount(self):
        doc = builder.notebook(self.pins)
        scope = {}
        with patch("builtins.print"):
            for cell in doc["cells"]:
                if cell["cell_type"] == "code":
                    exec(compile("".join(cell["source"]), "<notebook>", "exec"), scope)
        self.assertFalse(scope["RUN_ONE_SHARD"])
        invocation = self.root / "diag"
        invocation.mkdir()
        (invocation / "worker.log").write_text("synthetic preflight failure", encoding="utf-8")
        with patch("builtins.print"):
            scope["save_diagnostic"](invocation, self.retained, time.monotonic(), ValueError("preflight"))
        remote = json.loads((self.retained / "diag.json").read_bytes())
        self.assertEqual(remote["status"], "failed")
        self.assertIsNone(remote["worker"])
        self.assertEqual(remote["worker_log_tail"], "synthetic preflight failure")
