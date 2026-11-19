"""Exercise the pinned one-shard diagnostic locally, with a separate CUDA boundary."""

import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import numpy as np
import torch

from rewind_archive.contracts import canonical, digest
from rewind_prepare.temporal_selection import seal
from rewind_v1 import v2_cache as cache, v2_contract as contract
from rewind_v1 import v2_retention as retention
from rewind_v1.v2_budget import Ledger
from scripts import build_v2_gpu_cache_probe as builder, run_v2_gpu_cache_probe as probe
from scripts import build_v2_gpu_cache_restore as restore_builder, run_v2_gpu_cache_restore as restore


class ProbeTest(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        original = contract.ROOT
        for name in ("configs", "scripts", "reference/split", "drive"):
            (self.root / name).mkdir(parents=True)
        for name in ("configs/rewind_v2.json", "scripts/build_v2.py"):
            shutil.copyfile(original / name, self.root / name)
        for mock in (patch.object(contract, "ROOT", self.root),
                     patch.object(probe, "available_ram", return_value=8*1024**3)):
            mock.start()
            self.addCleanup(mock.stop)

    def prepare(self, *, prior_build=False):
        owners = {f"{scene}/Day/Avatar/Actor/001/{split}": split
                  for scene in contract.SCENARIOS for split in ("train", "validation", "test")}
        sources = self.root / "reference"
        file_pins = {}
        for split in ("train", "validation", "test"):
            raw = ("\n".join(s for s, p in owners.items() if p == split) + "\n").encode()
            (sources / "split" / f"{split}.txt").write_bytes(raw)
            file_pins[split] = contract.sha(raw)
        spec = {"joint_names": ["sample"], "pinned_sources": {"split_sha256": file_pins}}
        spec_raw = canonical(spec)
        (sources / "spec.json").write_bytes(spec_raw)
        _, split_sources = contract.official_splits(sources / "split", spec, contract.SplitReader())
        first = min(s for s, p in owners.items() if p != "test")
        records = []
        for sequence, split in sorted(owners.items()):
            if split == "test":
                continue
            for index in range(cache.CHUNK if sequence == first else 1):
                files = {}
                for role in ("left", "right", "pose"):
                    member = (f"json/frame_{index}.json" if role == "pose" else
                              f"fisheye_final_image/camera_{role}/final_{index}.png")
                    files[role] = {"path": f"{sequence}/{member}", "size_bytes": 1, "sha256": "a"*64}
                records.append({"sequence_id": sequence, "frame_index": index, "split": split, "files": files})
        inventory = seal({
            "schema": "rewind_v2_frozen_inventory_v1", "excluded": [], "failed": False, "capped": False,
            "missing": [], "duplicates": [], "sequences": [s for s, p in sorted(owners.items()) if p != "test"],
            "usable_keys": [[r["sequence_id"], r["frame_index"]] for r in records],
        })
        doc = seal({
            "schema": "rewind_v2_audited_manifest_v1", "scenarios": list(contract.SCENARIOS),
            "spec_sha256": digest(spec), "official_splits": split_sources, "coordinates": "device_mm",
            "inventory": inventory, "records": records,
        })
        raw = canonical(doc)
        manifest_archive = sources / "manifest.zip"
        with zipfile.ZipFile(manifest_archive, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            archive.writestr(probe.MANIFEST_NAME, raw)
        cfg_raw = (self.root / "configs/rewind_v2.json").read_bytes()
        binding = cache.binding(doc, spec, json.loads(cfg_raw))
        binding["torch"] = "2.14.1+cpu" if prior_build else "2.14.1+cpu-fixture"
        if prior_build:
            binding["source"]["rewind_v1/v2_cache.py"] = "f"*64
        cfg = json.loads(cfg_raw)
        origin = {"manifest": doc["content_sha256"], "source": digest(binding["source"]), "config": digest(cfg)}
        ledger = Ledger(self.root / "producer-usage", origin, cfg)
        with ledger.invocation("fixture_cache"):
            pass
        self.fixture = (doc, spec, cfg, ledger)
        self.pins = {
            "source_identity": digest(contract.source()), "revision": "0"*40,
            "spec_sha256": contract.sha(spec_raw), "config_sha256": contract.sha(cfg_raw),
            "embedded_manifest": "reference/manifest.zip", "archive_sha256": contract.sha(manifest_archive.read_bytes()),
            "manifest_bytes": len(raw), "manifest_file_sha256": contract.sha(raw),
            "manifest_content_sha256": doc["content_sha256"], "frames": len(records),
            "producer_binding_sha256": digest(binding),
            "producer_environment": {"torch": binding["torch"]},
        }
        n = cache.CHUNK
        export = self.root / "export"
        export.mkdir()
        np.save(export / "images.npy", np.full((n, 2, 3, 128, 128), 0.5, dtype=np.float32))
        torch.save({
            "xy": torch.ones(n, 2, 1, 2), "xy_valid": torch.ones(n, 2, 1, dtype=torch.bool),
            "target_mm": torch.ones(n, 1, 3)*3, "target_valid": torch.ones(n, 1, dtype=torch.bool),
            "camera": torch.ones(n, 2, 9), "rotation64": torch.ones(n, 2, 3, 3, dtype=torch.float64),
            "translation64": torch.ones(n, 2, 3, dtype=torch.float64),
        }, export / "targets.pt")
        parts = {}
        for name in ("images.npy", "targets.pt"):
            p = export / name
            parts[name] = {"bytes": p.stat().st_size, "sha256": contract.sha(p.read_bytes())}
        receipt = seal({
            "schema": "rewind_v2_shard_v1", "number": 0, "binding": binding,
            "records_sha256": digest(records[:n]), "frames": n, "files": parts,
        })
        staged = export / "shard.zip"
        with zipfile.ZipFile(staged, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for name in parts:
                archive.write(export / name, name)
            archive.writestr("receipt.json", canonical(receipt))
        archive_sha = contract.sha(staged.read_bytes())
        name = f"shard-00000-{archive_sha}.zip"
        self.shard = staged.rename(self.root / "drive" / name)
        package = {"number": 0, "name": name, "bytes": self.shard.stat().st_size,
                   "sha256": archive_sha, "receipt_sha256": receipt["content_sha256"]}
        catalog = seal({"schema": "rewind_v2_retained_catalog_v1", "binding": binding,
                        "packages": [package], "completed_frames": n, "status": "remote_verified",
                        "usage": ledger.snapshot()})
        catalog_raw = canonical(catalog)
        (self.root / "drive" / "catalog.json").write_bytes(catalog_raw)
        self.pins.update(package=package, catalog_file_sha256=contract.sha(catalog_raw),
                         catalog_bytes=len(catalog_raw), catalog_content_sha256=catalog["content_sha256"],
                         records_sha256=receipt["records_sha256"], shard_files=parts)
        self.pin_file = self.root / "pins.json"
        self.pin_file.write_bytes(canonical(self.pins))

    def restore_args(self, label):
        work = self.root / label
        work.mkdir()
        args = ["--pins", str(self.pin_file), "--work", str(work),
                "--input-dir", str(self.root / "drive"), "--seconds", "120"]
        return work, args

    def test_explicit_restore_uses_standard_batch_and_strict_loader_still_requires_full_cache(self):
        self.prepare(prior_build=True)
        self.pins["consumer_source_identity"] = digest(contract.source())
        self.pin_file.write_bytes(canonical(self.pins))
        work, args = self.restore_args("consumer")
        doc, spec, cfg, ledger = self.fixture

        def compare(batch):
            self.assertEqual(set(batch), {"image", "xy", "xy_valid"})
            self.assertEqual(tuple(batch["image"].shape), (2, 2, 3, 128, 128))
            self.assertTrue(torch.all(batch["image"] == 0.5))
            self.assertTrue(torch.all(batch["xy_valid"]))
            return {"all_exact": True}

        with (patch.object(torch, "__version__", "2.14.1+cu130"),
              patch.object(torch.version, "cuda", "13.0"),
              patch.object(torch.cuda, "is_available", return_value=True),
              patch.object(restore, "available_ram", return_value=8*1024**3),
              patch.object(restore, "cuda_round_trip", side_effect=compare),
              patch.object(restore, "frame_batch", wraps=restore.frame_batch) as batch):
            with self.assertRaisesRegex(ValueError, "v2_restore_binding_prefix"):
                retention.restore_cache(self.root / "drive", self.pins["catalog_file_sha256"],
                                        self.root / "strict", doc, spec, cfg, ledger)
            restore.main(args)
            batch.assert_called_once()
        report = json.loads((work / "result.json").read_bytes())
        self.assertEqual(report["status"], "passed")
        restored = json.loads((work / "restored/catalog.json").read_bytes())
        self.assertEqual(restored["binding"]["torch"], "2.14.1+cpu")
        with patch.object(cache, "binding", return_value=restored["binding"]):
            with self.assertRaisesRegex(ValueError, "v2_complete_cache_required"):
                cache.Cache(work / "restored", doc, spec, cfg, ledger)
        producer = restored["binding"]
        current = {**producer, "torch": "2.14.1+cu130", "source": contract.source()}
        policy = cache.ReadOnlyFirstShard(digest(producer), digest(current["source"]), self.pins["package"]["receipt_sha256"])
        for changes in ({"numpy": "other"}, {"torch": "2.14.1+cu129"}, {"manifest_sha256": "a"*64}):
            with self.assertRaisesRegex(ValueError, "v2_read_only_binding_pin"):
                policy.verify(producer, {**current, **changes})
        with self.assertRaisesRegex(ValueError, "v2_read_only_output_overlap"):
            retention.restore_cache(self.root / "drive", self.pins["catalog_file_sha256"],
                                    self.root / "drive/overlap", doc, spec, cfg, ledger, read_only=policy)

    def test_restore_notebook_sources_and_entry(self):
        overlay = dict.fromkeys(restore_builder.MODULES, "# fixture")
        doc = restore_builder.make_notebook({"source_name": "fixed.zip"}, overlay)
        scope = {}
        with patch("builtins.print"):
            for cell in doc["cells"]:
                if cell["cell_type"] == "code":
                    exec(compile("".join(cell["source"]), "<notebook>", "exec"), scope)
        source = "".join(doc["cells"][-1]["source"])
        self.assertIn('"scripts.run_v2_gpu_cache_restore"', source)
        self.assertIn('"v2_gpu_restore_041837Z"', source)
        self.assertFalse(scope["RUN_GPU_CACHE_RESTORE"])
        self.assertEqual(scope["OVERLAY"], overlay)

    def run_cli(self, work):
        work.mkdir()
        probe.main(["--pins", str(self.pin_file), "--work", str(work),
                    "--input-dir", str(self.root / "drive"), "--seconds", "120"])

    def test_runtime_mismatch_still_loads_original_batch_and_cuda_required(self):
        self.prepare()
        work = self.root / "no-cuda"
        with patch.object(torch.cuda, "is_available", return_value=False):
            with self.assertRaisesRegex(RuntimeError, "cuda_required"):
                self.run_cli(work)
        failed = json.loads((work / "result.json").read_bytes())
        self.assertEqual(failed["drive_input_bytes_observed"], self.pins["catalog_bytes"])
        self.assertEqual(failed["strict_restore_binding"]["status"], "runtime_mismatch")
        self.assertNotIn("cuda", failed)

        def compare(batch):
            self.assertEqual(batch["image"].shape, (2, 2, 3, 128, 128))
            self.assertTrue(torch.all(batch["image"] == 0.5))
            self.assertEqual(batch["target_mm"][0, 0].tolist(), [3.0]*3)
            self.assertEqual(batch["rotation64"].dtype, torch.float64)
            self.assertEqual(batch["target_valid"].dtype, torch.bool)
            return {"all_exact": True}

        work = self.root / "mock-cuda"
        with (patch.object(torch.cuda, "is_available", return_value=True),
              patch.object(probe, "cuda_round_trip", side_effect=compare)):
            self.run_cli(work)
        result = json.loads((work / "result.json").read_bytes())
        self.assertEqual(result["status"], "passed")
        self.assertFalse(result["strict_restore_binding"]["strict_binding_match"])
        self.assertEqual(result["batch"]["valid_counts"]["xy_valid"], [[1, 1], [1, 1]])
        self.assertEqual(result["drive_input_bytes_observed"], self.pins["catalog_bytes"] + self.shard.stat().st_size)
        self.assertFalse((work / "shard/catalog.json").exists())

    def test_wrong_archive_hash_never_reaches_tensor_or_cuda(self):
        self.prepare()
        with self.shard.open("r+b") as stream:
            stream.write(b"X")
        work = self.root / "bad-shard"
        with (patch.object(torch.cuda, "is_available", return_value=True),
              patch.object(probe, "cuda_round_trip") as cuda):
            with self.assertRaisesRegex(ValueError, "archive_sha"):
                self.run_cli(work)
            cuda.assert_not_called()
        report = json.loads((work / "result.json").read_bytes())
        self.assertEqual(report["stage"], "package")
        self.assertNotIn("batch", report)

    def test_binding_mismatch_classification(self):
        original = {"source": "original", "torch": "2.14.1+cpu", "numpy": "fixed", "pillow": "fixed"}
        self.assertEqual(probe.assess_binding(original, original)["status"], "match")
        self.assertEqual(probe.assess_binding(original, {**original, "torch": "2.14.1+cu130"})["status"],
                         "runtime_mismatch")
        self.assertEqual(probe.assess_binding(original, {**original, "source": "different"})["status"],
                         "non_runtime_mismatch")

    def test_notebook_default_off_and_missing_upload_stops_before_mount(self):
        notebook = builder.notebook({"source_name": "fixed.zip"}, "# Probe")
        scope = {}
        with patch("builtins.print"):
            for cell in notebook["cells"]:
                if cell["cell_type"] == "code":
                    exec(compile("".join(cell["source"]), "<notebook>", "exec"), scope)
            self.assertFalse(scope["RUN_GPU_CACHE_PROBE"])

            def temporary_colab(value):
                string = str(value)
                if string.startswith("/content"):
                    return self.root / string.removeprefix("/content").lstrip("/")
                return Path(value)

            scope["Path"] = temporary_colab
            with self.assertRaisesRegex(FileNotFoundError, "Upload: fixed.zip to /content"):
                scope["run_gpu_cache_probe"]()
        reports = list((self.root / "rewind_v2_gpu_cache_probe").glob("*/diagnostic.json"))
        self.assertEqual(len(reports), 1)
        self.assertIn("FileNotFoundError", reports[0].read_text(encoding="utf-8"))
