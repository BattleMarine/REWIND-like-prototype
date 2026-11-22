"""Portable archive integrity, native checkpoint roster and on-demand cached frames."""

import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
from unittest import TestCase
import zipfile

import torch

from rewind_v1 import v2_cache, v2_contract, v2_retention
from rewind_v1.v2_budget import Ledger
from scripts import v2_gpu128_a_bundle_format as fmt
from scripts.v2_gpu128_a_bundle_export import Deadline, model_roster, write_bundle
from scripts.v2_gpu128_a_bundle_local import BundleCache


def pin(raw):
    return {"bytes": len(raw), "sha256": fmt.sha(raw)}


def seal(body):
    return fmt.canonical({**body, "content_sha256": fmt.sha(fmt.canonical(body))})


class BundleTests(TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        scratch = tempfile.TemporaryDirectory()
        self.addCleanup(scratch.cleanup)
        self.root = Path(scratch.name).resolve()

    def test_stream_nonseekable_zip64_readback_and_failed_candidate_cleanup(self):
        source = self.root/"source.bin"
        source.write_bytes(b"known real bytes")
        item = pin(source.read_bytes())
        files = {"data/a.bin": (source, item)}
        output = self.root/"bundle.zip"
        report = write_bundle(output, files, {"scope": "local fixture"}, Deadline(30))
        self.assertEqual(report["sha256"], fmt.sha(output.read_bytes()))
        self.assertEqual(report["zip_readback_bytes"], output.stat().st_size)
        with fmt.BundleView(output) as view:
            self.assertEqual(view.index()["members"], {"data/a.bin": item})
            self.assertEqual(view.read("data/a.bin", 40, item), source.read_bytes())
            self.assertTrue(view.members["data/a.bin"].flag_bits & 8)  # Non-seekable data descriptor.
            target = self.root/"protected.bin"
            target.write_bytes(b"preserve")
            with self.assertRaises(FileExistsError):
                view.copy("data/a.bin", target, item)
            self.assertEqual(target.read_bytes(), b"preserve")
        with self.assertRaises(FileExistsError):
            write_bundle(output, files, {}, Deadline(30))
        source.write_bytes(b"known beta bytes")
        with self.assertRaisesRegex(ValueError, "sha256_or_size"):
            write_bundle(self.root/"broken.zip", files, {}, Deadline(30))
        self.assertFalse((self.root/"broken.zip").exists())
        self.assertEqual(list(self.root.glob(".broken.zip.*")), [])

    def test_native_model_roster_and_full_feature_group_scope(self):
        model = self.root/"model.zip"
        parts = {"last.pt": b"last", **{f"best-{k}-e003.pt": k.encode() for k in ("frontend", "body", "hand")}}
        with zipfile.ZipFile(model, "w") as source:
            for name, raw in parts.items():
                source.writestr(name, raw)
            source.writestr("recovery_receipt.json", seal({"schema": "rewind_v2_result_recovery_v1",
                                                         "files": {n: pin(r) for n, r in parts.items()}}))
        compare = {"binding": {"model": {key: fmt.sha(key.encode()) for key in ("frontend", "body", "hand")}}}
        self.assertEqual(set(model_roster(model, {"last": pin(parts["last.pt"])}, compare)), {"frontend", "body", "hand"})
        paths = self.root.joinpath(*fmt.FEATURES.split("/"))
        paths.mkdir(parents=True)
        clips = [f"{split}-{i:05d}" for split, count in (("train",3097),("validation",421)) for i in range(count)]
        for start in range(0, len(clips), 24):
            tag = f"group-{start//24:05d}"
            (paths/(tag+".json")).write_bytes(seal({"schema": "rewind_v2_gpu_features_group_v1",
                "identity": fmt.FEATURE_IDENTITY, "folders": clips[start:start+24],
                "archive": {"name":f"{tag}-{'0'*32}.zip", **pin(b'real feature')}}))
        with fmt.BundleView(self.root) as view:
            groups = fmt.feature_groups(view)
            self.assertEqual(len(groups), 147)
            self.assertEqual(sum(len(value[3]) for value in groups.values()),3518)
            (paths/"group-00146.json").write_bytes(b"{}")
            with self.assertRaises(ValueError):
                fmt.feature_groups(view)

    def test_nested_shard_receipt_then_original_cached_frame_read(self):
        cfg = v2_contract.config()
        ledger = Ledger(self.root/"usage", {"fixture": 1}, cfg)
        stage = self.root/"input"
        stage.mkdir()
        shard = stage/"shard-00000"
        shard.mkdir()
        import numpy as np
        np.save(shard/"images.npy", np.full((1,2,3,128,128), .5, dtype=np.float32))
        torch.save({"xy":torch.zeros(1,2,1,2), "xy_valid":torch.ones(1,2,1,dtype=torch.bool),
                    "target_mm":torch.zeros(1,1,3), "target_valid":torch.ones(1,1,dtype=torch.bool),
                    "camera":torch.zeros(1,2,9), "rotation64":torch.zeros(1,2,3,3,dtype=torch.float64),
                    "translation64":torch.zeros(1,2,3,dtype=torch.float64)}, shard/"targets.pt")
        rows = [{"sequence_id":"clip", "frame_index":0, "split":"train"}]
        binding = {"fixture":"native"}
        raw = seal({"schema":"rewind_v2_shard_v1", "number":0, "binding":binding, "frames":1,
                    "records_sha256":fmt.sha(fmt.canonical(rows)), "files":{n:pin((shard/n).read_bytes())
                    for n in ("images.npy", "targets.pt")}})
        (shard/"receipt.json").write_bytes(raw)
        native_zip = self.root/"native.zip"
        archive = self.root/"outer.zip"
        with ledger.invocation("test"):
            info = v2_retention.pack_shard(shard,native_zip,ledger)
            name = f"shard-00000-{info['sha256']}.zip"
            package = {**info, "name":name, "number":0, "receipt_sha256":json.loads(raw)["content_sha256"]}
            write_bundle(archive, {f"{fmt.LEGACY}/{name}": (native_zip, info)}, {"fixture":1}, Deadline(30))
            journal = SimpleNamespace(journal_id="fixture", doc={"records":rows}, binding=binding)
            entries = [{"index":0, "package":package, "content_sha256":"final"}]
            with fmt.BundleView(archive) as view:
                cached = BundleCache(self.root/"local", journal, {"binding":binding}, entries,
                                     {"joint_names":["j"]}, cfg, ledger, view)
                try:
                    self.assertEqual(cached.metrics["bundle_read_bytes"],0)
                    image, target = cached.frame(("clip",0),"train")
                    self.assertTrue(torch.equal(image,torch.full((2,3,128,128),.5)))
                    self.assertTrue(target["xy_valid"].all())
                    self.assertEqual(cached.metrics["bundle_read_bytes"],info["bytes"])
                    cached.frame(("clip",0),"train")
                    self.assertEqual(cached.metrics["bundle_read_bytes"],info["bytes"])
                finally:
                    cached.close()
