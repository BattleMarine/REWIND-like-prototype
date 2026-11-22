"""Warmstart split, actual CPU updates and immutable checkpoint dependencies."""

import json
from copy import deepcopy
from pathlib import Path
import tempfile
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import patch

import torch

from rewind_archive.contracts import canonical, digest
from rewind_prepare.temporal_cache import _hash_file
from rewind_v1 import v2_contract as contract
from rewind_v1.run_data import FixtureData
from rewind_v1.runner import Budget, Pipeline
from rewind_v1.v2_budget import Ledger
from scripts.run_v2_gpu_training import training_config
from scripts.v2_gpu128_a_bundle_predict import load_warmstart, use_warmstart
from scripts.v2_gpu128_a_bundle_tune import ROLES, select, warmstart
from scripts.v2_gpu128_a_bundle_train_full import LocalTraining, run_config
from scripts.v2_gpu_training_pipeline import CommonEvaluation, FullData


class BundleWarmstartTests(TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.cfg = training_config(42)
        document = {"content_sha256": "fixture", "records": [
            {"sequence_id": f"{scenario}/{split}", "frame_index": index, "split": split, "files": {}}
            for scenario in contract.SCENARIOS for split in ("train", "validation") for index in range(3)]}
        spec = FixtureData().spec
        plan = contract.plan(document, spec, self.cfg, self.root)
        values = {"xy": torch.ones(2, 4, 2)*28, "xy_valid": torch.ones(2, 4, dtype=torch.bool),
                  "target_mm": torch.arange(12, dtype=torch.float32).reshape(4, 3),
                  "target_valid": torch.ones(4, dtype=torch.bool), "camera": torch.zeros(2, 9),
                  "rotation64": torch.eye(3, dtype=torch.float64).repeat(2, 1, 1),
                  "translation64": torch.tensor([[0., -80., 0.], [0., 80., 0.]], dtype=torch.float64)}
        self.requests = []

        def frame(key, split):
            self.assertIn(f"/{split}", key[0])
            self.requests.append((key, split))
            return torch.full((2, 3, 128, 128), .2+key[1]*.1), values

        image_cache = SimpleNamespace(identity="fixture-cache", metrics={}, frame=frame)
        runtime = json.loads((contract.ROOT / "configs/rewind_v12_full_training.json").read_bytes())
        runtime["epochs"] = dict.fromkeys(ROLES, 1)
        runtime["teacher"].update(hidden=8, heads=2, graph_layers=1, temporal_layers=1, diffusion_steps=20, sampling_steps=2)
        runtime["limits"].update(max_seconds=120, output_bytes=100*1024**2, checkpoint_bytes=16*1024**2)
        self.budget = Budget(runtime["limits"])
        data = FullData(document, spec, self.cfg, plan, image_cache, self.budget, "fixture-source")
        original = self.root / "original"
        original.mkdir()
        self.pipeline = Pipeline(runtime, digest(runtime), data, torch.device("cpu"), original, self.budget)
        self.pipeline.identity = {"source": "fixture", "seed": 42}

    def test_deterministic_validation_selection_is_independent_from_train_seed(self):
        first = select(self.pipeline, "frontend", 6, 3, 42)
        second = select(self.pipeline, "frontend", 6, 3, 103)
        self.assertEqual(first["validation"], second["validation"])
        self.assertNotEqual(first["train"], second["train"])
        self.assertEqual({key[0].partition("/")[0] for key in first["validation"]}, set(contract.SCENARIOS))

    def test_full_entry_uses_configured_epochs_and_resumes_local_cursor(self):
        cfg = run_config((2, 1, 1), 71, 2)
        runtime = deepcopy(self.pipeline.cfg)
        runtime["seed"] = 71
        runtime["epochs"] = cfg["training"]["epochs"]
        self.pipeline.data.cache.prepare_frontend = lambda: None
        ledger = Ledger(self.root/"full-ledger", {"fixture":"full"}, self.cfg)
        self.pipeline.data.budget = self.pipeline.data.ledger = ledger
        session = SimpleNamespace(pipeline=self.pipeline, ledger=ledger)
        output = self.root/"full"
        output.mkdir()
        (output/"run").mkdir()
        identity = {"source":"fixture", "seed": 71}
        with patch("scripts.v2_gpu_training_pipeline.runtime_config", side_effect=lambda *_: deepcopy(runtime)):
            with ledger.invocation("start"):
                first = LocalTraining(cfg, session, output, identity, "fresh", lambda: None)
                self.assertEqual(first.run(stop_after_updates=1)["status"], "paused")
                pin = first.last["sha256"]
            with ledger.invocation("resume"):
                resumed = LocalTraining(cfg, session, output, identity, "fresh", lambda: None)
                report = resumed.run(resume_sha256=pin)
            self.assertEqual(report["status"], "completed")
            self.assertEqual(report["cursor"]["updates"],16)
            self.assertEqual([x["stage"] for x in report["cursor"]["history"]], ["frontend", "frontend", "body", "hand"])

    def test_all_three_warmstart_roles_update_and_keep_original_dependencies(self):
        pipe = self.pipeline
        frontend = pipe._make_frontend()
        pipe._save_epoch_model("frontend", frontend, {"score": 1.})
        pipe._load_best("frontend", frontend)
        key = pipe.data.clips["train"][0]
        clip = pipe.data.clip(key, "train", frontend.model, pipe.frontend_size, "c"*64)
        cascade = CommonEvaluation(pipe.data, {"train": {key: clip}, "validation": {}}, pipe.cfg, pipe.device, 0)
        body = cascade.make("body")
        pipe._save_epoch_model("body", body, {"score": 1.})
        hand = cascade.make("hand", pipe.progress["best"]["body"]["sha256"])
        pipe._save_epoch_model("hand", hand, {"score": 1.})
        old_files = {p.name: _hash_file(p) for p in pipe.root.iterdir()}
        bundle = SimpleNamespace(pipeline=pipe, ledger=self.budget, diagnostic={"checkpoint": {"content_sha256": "6"*64}})
        for index, role in enumerate(ROLES):
            with self.subTest(role=role):
                self.requests.clear()
                output = self.root / role
                output.mkdir()
                report = warmstart(bundle, output, role=role, train_items=3, validation_items=3, updates=1,
                                   learning_rate=0.0001, seed=17)
                self.assertGreater(report["changed_parameter_tensors"], 0)
                self.assertEqual(report["optimizer_steps_with_supervision"], 1)
                self.assertEqual(set(report["binding"]["upstream_checkpoints"]), set(ROLES[:index]))
                self.assertLessEqual(len(set(self.requests)), 18)
                self.assertIn("score", report["validation_before"])
                self.assertIn("score", report["validation_after"])
                checkpoint = output / "warmstart.pt"
                self.assertEqual(_hash_file(checkpoint), report["checkpoint"]["sha256"])
                saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
                self.assertEqual(saved["fit"]["scheduler"]["last_epoch"], 1)
                self.assertFalse(saved["binding"]["original_optimizer_loaded"])
                (output/"warmstart.json").write_bytes(canonical(report))
                selected_role, selected_sha, weights = load_warmstart(bundle, output)
                self.assertEqual((selected_role, selected_sha), (role, report["checkpoint"]["sha256"]))
                selected = pipe._make_frontend() if role == "frontend" else cascade.make(
                    role, pipe.progress["best"]["body"]["sha256"] if role == "hand" else None)
                use_warmstart(selected, selected_sha, weights)
                self.assertTrue(all(torch.equal(t, selected.model.state_dict()[n]) for n,t in weights.items()))
                bad = {**report, "binding": {**report["binding"], "parent_receipt_sha256": "0"*64}}
                (output/"warmstart.json").write_bytes(canonical(bad))
                with self.assertRaisesRegex(ValueError, "warmstart_parent"):
                    load_warmstart(bundle, output)
                if role == "hand":
                    self.assertEqual(report["validation_after"]["upper_source"], "predicted_frozen_body")
                    self.assertEqual(report["validation_after"]["body_checkpoint_sha256"],
                                     pipe.progress["best"]["body"]["sha256"])
        self.assertEqual({p.name: _hash_file(p) for p in pipe.root.iterdir()}, old_files)
