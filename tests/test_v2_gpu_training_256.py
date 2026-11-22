"""Local 256 input, exact range verification and interrupted training regressions."""

from contextlib import nullcontext
from copy import deepcopy
import io
import json
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import patch
import zipfile

import numpy as np
from PIL import Image
import torch

from rewind_archive.contracts import canonical, digest
from rewind_m1.rgb_frontend_artifacts import scale_xy
from rewind_prepare.temporal_selection import seal
from rewind_v1 import v2_contract as contract
from rewind_v1.frontend import PaperHeatmapFrontend, predicted_features
from rewind_v1.run_data import FixtureData
from rewind_v1.run_stages import evaluate_frontend
from rewind_v1.runner import LAST, _read_checkpoint
from rewind_v1.v2_budget import ResourceLimit
from scripts import run_v2_gpu_training as base
from scripts import run_v2_gpu_training_256 as large
from scripts.run_v2_cache256 import RECEIPT_SCHEMA, entry
from scripts.v2_gpu_training_cache256 import FINAL_IO_RESERVE, FRAME_BYTES, RangeCache256, ReadBudget, sha
from scripts.v2_gpu_training_comparison import draw_clip
from scripts.v2_gpu_training_pipeline import Training
from scripts.v2_gpu_training_storage import Checkpoints, DriveLedger, MIB


class GpuTraining256Tests(TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        temp_root = contract.ROOT/"outputs/test_tmp"
        temp_root.mkdir(parents=True, exist_ok=True)
        tmp = tempfile.TemporaryDirectory(dir=temp_root)
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.cfg = large.training_config(42)
        self.spec = FixtureData().spec
        self.owner = lambda: None
        self.identity = {"seed": 42, "source": "fixture256", "input_size": [256, 256]}
        self.ledger = DriveLedger(self.root/"usage", self.identity, deepcopy(self.cfg), self.owner)

    def targets(self, size):
        return {"xy": torch.full((size,2,4,2), 60.), "xy_valid": torch.ones(size,2,4,dtype=torch.bool),
                "target_mm": torch.arange(12, dtype=torch.float32).reshape(1,4,3).expand(size,-1,-1),
                "target_valid": torch.ones(size,4,dtype=torch.bool), "camera": torch.zeros(size,2,9),
                "rotation64": torch.eye(3,dtype=torch.float64).repeat(size,2,1,1),
                "translation64": torch.tensor([[0.,-80.,0.],[0.,80.,0.]], dtype=torch.float64).repeat(size,1,1)}

    def zipped(self, store, number, rows, prior, binding):
        pixels = np.stack([np.full((2,3,256,256), (frame["frame_index"]+1)/10, dtype=np.float32) for frame in rows])
        npy, pt = io.BytesIO(), io.BytesIO()
        np.save(npy, pixels, allow_pickle=False)
        torch.save(self.targets(len(rows)), pt)
        files = {"images.npy": npy.getvalue(), "targets.pt": pt.getvalue()}
        receipt = seal({"schema": RECEIPT_SCHEMA, "index": number, "binding": binding, "size": 256,
                        "records_sha256": digest(rows), "image_shape": list(pixels.shape), "image_dtype": "float32",
                        "files": {name: {"bytes": len(raw), "sha256": sha(raw)} for name, raw in files.items()}})
        archive = io.BytesIO()
        with zipfile.ZipFile(archive, "w") as bundle:
            for name, raw in (*files.items(), ("receipt.json", canonical(receipt))):
                with bundle.open(name, "w", force_zip64=True) as member:
                    member.write(raw)
        raw = archive.getvalue()
        checksum = sha(raw)
        pin = {"sha256": checksum, "bytes": len(raw), "name": f"shard-{number:05d}-{checksum}.zip",
               "receipt_sha256": receipt["content_sha256"]}
        path = store/pin["name"]
        path.write_bytes(raw)
        return entry(number, rows, prior, pin, binding), path, pixels

    def test_first_zip_verified_then_index_recovers_and_checks_remote_frame_bytes(self):
        store = self.root/"store"
        store.mkdir()
        docs = [{"sequence_id": "sequence", "frame_index": frame, "split": "train"} for frame in range(3)]
        binding = "fixture-binding"
        first, archive, images = self.zipped(store, 0, docs[:2], None, binding)
        tail, _, tail_images = self.zipped(store, 1, docs[2:], first["content_sha256"], binding)
        document = {"records": docs}
        markers = self.root/"indices"
        self.cfg["resources"]["cache_bytes"] = 5*MIB

        def loader(name, indices=markers):
            cache = RangeCache256(self.root/name, indices, store, [first, tail], document,
                                  self.spec, binding, self.cfg, self.ledger, self.owner)
            self.addCleanup(cache.close)
            return cache

        with self.ledger.session(time.monotonic()), patch("scripts.v2_gpu_training_cache256.CHUNK", 2):
            initial = loader("local-A")
            with patch("scripts.v2_gpu_training_cache256.publish_json", side_effect=KeyboardInterrupt), self.assertRaises(KeyboardInterrupt):
                initial.frame(("sequence", 0), "train")
            self.assertFalse((markers/"shard-00000.json").exists())
            initial.close()
            reused = loader("local-A")
            self.assertTrue(torch.equal(reused.frame(("sequence",1), "train")[0], torch.from_numpy(images[1])))
            self.assertEqual(reused.metrics["drive_zip_bytes"], 0)
            self.assertTrue(torch.equal(reused.frame(("sequence",2), "train")[0], torch.from_numpy(tail_images[0])))
            self.assertLessEqual(reused.disk.used, 5*MIB)
            restored = loader("new-vm")
            self.assertTrue(torch.equal(restored.frame(("sequence",1), "train")[0], torch.from_numpy(images[1])))
            self.assertEqual(restored.metrics["reused_index_shards"], 1)
            self.assertEqual(restored.metrics["drive_zip_bytes"], 0)
            self.assertLess(restored.metrics["drive_range_bytes"], first["package"]["bytes"])
            offset = restored.indices[0]["image_offset"]
            restored.close()
            with archive.open("r+b") as stream:
                stream.seek(offset)
                byte = stream.read(1)
                stream.seek(offset)
                stream.write(bytes([byte[0] ^ 1]))
            with self.assertRaisesRegex(ValueError, "range_sha256"):
                loader("corrupt-range").frame(("sequence",0), "train")
            unseen = self.root/"unseen"
            with self.assertRaisesRegex(ValueError, "transfer_source_sha256"):
                loader("corrupt-zip", unseen).frame(("sequence",0), "train")
            self.assertFalse((unseen/"shard-00000.json").exists())

    def test_normalized_features_and_native_eval_keep_same_units_and_real_grid(self):
        class KnownCoordinates(torch.nn.Module):
            def forward(self, images):
                batch, side = len(images), images.shape[-1]
                xy = torch.tensor([0.25, 0.75]).repeat(batch,2,1,1)*(side-1)
                confidence = torch.ones(batch,2,1)
                return None, xy, confidence

        all_metrics = []
        all_features = []
        for side in (128,256):
            model = KnownCoordinates().eval()
            images = torch.zeros(1,2,3,side,side)
            size = [side, side]
            all_features.append(predicted_features(model, images))
            actual = PaperHeatmapFrontend(1).eval()
            with torch.inference_mode():
                self.assertEqual(actual(images)[0].shape[-2:], (side//2, side//2))
            native = torch.tensor([0.25*1023+4, .75*1023-3]).repeat(2,1,1)
            xy = scale_xy(native,[1024,1024],size)
            data = SimpleNamespace(identity={"data_kind":"dataset_sample"}, frames={"validation":[("file",0)]},
                                   frame=lambda *_: ((images[0],xy,torch.ones(2,1,dtype=torch.bool)),{}))
            stage = SimpleNamespace(model=model, device="cpu", measure=lambda _:nullcontext(), runtime={})
            metrics = evaluate_frontend(stage, data, {"size":size,"batch_size":2}, SimpleNamespace(check=lambda:None), {})
            self.assertEqual(metrics["unit"], "native_px")
            all_metrics.append(metrics["score"])
        self.assertTrue(torch.equal(*all_features))
        for score in all_metrics:
            self.assertAlmostEqual(score, 5.0, places=3)

    def test_png_and_gif_draw_256_pixel_overlays(self):
        pixels = np.zeros((2,2,3,256,256), dtype=np.float32)
        predicted = np.zeros((2,2,1,2), dtype=np.float32)+180
        target = np.zeros_like(predicted)+249
        points = np.zeros((2,1,3), dtype=np.float32)
        path = self.root/"clip.gif"
        draw_clip(path, pixels, predicted, target, np.ones((2,2,1), dtype=bool), points, points,
                  np.ones((2,1),dtype=bool), ["root"], {"edges":[]})
        with Image.open(path.with_suffix(".png")) as png, Image.open(path) as gif:
            self.assertEqual(png.size, (768,276))
            self.assertEqual(gif.size, png.size)
            self.assertEqual(png.getpixel((249+256,249+20)), (84,229,150))

    def test_input_budget_pauses_and_exact_256_model_resumes(self):
        for seed in (42, 71, 103):
            self.assertEqual(base.training_config(seed)["training"], large.training_config(seed)["training"])
        spec = self.spec
        records = [{"sequence_id": f"{scenario}/{split}", "frame_index": i, "split": split, "files": {}}
                   for scenario in contract.SCENARIOS for split in ("train", "validation") for i in range(2)]
        doc = {"content_sha256": "fixture", "records": records}
        plan = contract.plan(doc, spec, self.cfg, self.root)
        runtime = json.loads((contract.ROOT/"configs/rewind_v12_full_training.json").read_bytes())
        runtime["epochs"] = dict.fromkeys(("frontend", "body", "hand"), 1)
        runtime["teacher"].update(hidden=8, heads=2, graph_layers=1, temporal_layers=1, diffusion_steps=20, sampling_steps=2)
        runtime["limits"]["checkpoint_every"] = 1
        vals = {k: v[0] for k, v in self.targets(1).items()}
        limited = False

        def frames(key, split):
            if limited:
                budget.reserve(FRAME_BYTES)
            return torch.full((2,3,256,256), .2+key[1]*.1), vals

        image_cache = SimpleNamespace(metrics={}, identity="same256", prepare_frontend=lambda:None, frame=frames)
        data = large.FullData256(doc, spec, self.cfg, plan, image_cache, self.ledger, self.identity["source"])

        def train(path, drive):
            drive.mkdir(exist_ok=True)
            model = Training(self.cfg, data, self.root/(path+"-features"), torch.device("cpu"), self.root/path,
                             drive, self.ledger, self.owner, self.identity)
            model.checkpoints = Checkpoints(drive/"checkpoints", self.root, self.identity, model, self.cfg, self.ledger, self.owner)
            return model

        with patch("scripts.v2_gpu_training_pipeline.runtime_config", side_effect=lambda *_: json.loads(json.dumps(runtime))):
            with self.ledger.session(time.monotonic()):
                whole = train("whole", self.root/"whole-drive")
                whole.root.mkdir()
                self.assertEqual(whole.run()["status"], "completed")
            with self.ledger.session(time.monotonic()):
                original = self.ledger.cfg["resources"]["read_bytes"]
                self.ledger.cfg["resources"]["read_bytes"] = FINAL_IO_RESERVE+2*FRAME_BYTES
                budget = ReadBudget(self.ledger)
                limited = True
                model = train("first", self.root/"stopped-drive")
                model.root.mkdir()
                paused = model.run()
                self.assertEqual(paused["status"], "paused")
                self.assertEqual(paused["checkpoint_cursor"]["updates"], 1)
                self.assertIn("input_read_reserve", paused["failure"]["reason"])
                tip = model.checkpoints.save()
                self.assertEqual(tip["cursor"]["updates"], 1)
            with self.ledger.session(time.monotonic()):
                limited = False
                self.ledger.cfg["resources"]["read_bytes"] = original
                restored = train("new-vm", self.root/"stopped-drive")
                resume = restored.checkpoints.restore(tip)
                restored.first_required = False
                self.assertEqual(restored.run(resume)["status"], "completed")
                self.assertEqual(restored.frontend_size, [256,256])
                self.assertEqual(restored.progress["updates"], whole.progress["updates"])
                states = [_read_checkpoint(p.root/LAST,p.last["sha256"],256*MIB)["fit"] for p in (whole,restored)]
                for key, value in states[0]["model"].items():
                    self.assertTrue(torch.equal(value, states[1]["model"][key]))
                self.assertTrue(torch.equal(states[0]["rng"]["torch_cpu"], states[1]["rng"]["torch_cpu"]))
