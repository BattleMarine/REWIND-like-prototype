"""Local regression for full v2 GPU orchestration, with no Colab or real Drive writes."""

import hashlib
import json
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import patch

import torch

from rewind_prepare.temporal_cache import _hash_file
from rewind_v1.data import Clip
from rewind_v1.run_data import FixtureData
from rewind_v1.runner import LAST
from rewind_v1.v2_contract import ROOT
from rewind_v1 import v2_contract as contract
from scripts.run_v2_gpu_training import training_config
from scripts.v2_gpu_training_features import PersistentFeatures
from scripts.v2_gpu_training_pipeline import CommonEvaluation, FullData, Training, merge_metrics
from scripts.v2_gpu_training_storage import Checkpoints, DriveLedger, verify_first


class GpuTrainingTests(TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        tests = ROOT / "outputs/test_tmp"
        tests.mkdir(parents=True, exist_ok=True)
        temporary = tempfile.TemporaryDirectory(dir=tests)
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.cfg = training_config(42)
        self.owner = lambda: None
        self.identity = {"seed": 42, "experiment": "fixture"}
        self.ledger = DriveLedger(self.root / "usage", self.identity, self.cfg, self.owner)

    def test_verified_A_gate_can_be_recreated_after_checkpoint_publication(self):
        old = self.root / "old"
        old.mkdir()
        stage = torch.nn.Linear(2, 2)
        optimizer = torch.optim.Adam(stage.parameters())
        loss = stage(torch.tensor([[1., 2.]])).square().sum()
        loss.backward()
        optimizer.step()
        checkpoint = old / LAST
        progress = {"stage": "frontend", "epoch": 0, "batch": 1, "updates": 1, "best": {}, "milestones": {},
                    "partial_loss_sums": {"supervised_joint_views": 2, "first_update_gradient_l2": 1.5}}
        binding = {"experiment": self.identity}
        torch.save({"binding": binding, "progress": progress, "fit": {"optimizer": optimizer.state_dict()}}, checkpoint)
        pin = {"name": LAST, "sha256": _hash_file(checkpoint), "bytes": checkpoint.stat().st_size}
        pipe = SimpleNamespace(root=old, last=pin, durable_progress=progress, environment={}, binding=binding)
        root = self.root / "checkpoints"
        with self.ledger.session(time.monotonic()):
            save = Checkpoints(root, self.root, self.identity, pipe, self.cfg, self.ledger, self.owner)
            with patch.object(save, "publish_first", side_effect=KeyboardInterrupt), self.assertRaises(KeyboardInterrupt):
                save.save(first=True)
            self.assertFalse((root / "first-update.json").exists())
            restore = Checkpoints(root, self.root, self.identity, SimpleNamespace(root=self.root / "new", binding=binding),
                                  self.cfg, self.ledger, self.owner)
            tip = restore.history()[-1]
            self.assertEqual(restore.restore(tip), pin["sha256"])
            self.assertEqual(len(verify_first(root, self.root, self.identity, self.ledger, self.cfg)), 64)
            restore.publish_first(tip)
            archive = root / tip["archive"]["name"]
            raw = archive.read_bytes()
            archive.write_bytes(raw[:-1] + bytes([raw[-1] ^ 1]))
            with self.assertRaisesRegex(ValueError, "source_sha256"):
                verify_first(root, self.root, self.identity, self.ledger, self.cfg)

    def test_actual_cascade_eval_call_uses_shared_noise_for_all_seeds(self):
        data = FixtureData()
        target = torch.arange(48, dtype=torch.float32).reshape(1, 4, 4, 3)
        valid = torch.ones((1, 4, 4), dtype=torch.bool)
        clip = Clip(torch.ones(1, 4, 56), torch.zeros(1, 4, 2, 9), target, valid, valid[..., 0],
                    torch.arange(4, dtype=torch.float64)[None]/25, "predicted_2d", split="validation", input_sha256="fixture")
        clips = {"train": {("train", 0): clip}, "validation": {("validation", 0): clip}}
        evaluations = []
        for seed in (42, 71, 103):
            cascade = CommonEvaluation(data, clips, {"seed": seed, "limits": {"feature_bytes": 10240}}, torch.device("cpu"), 0)
            noise_hashes = []

            def predict(selected, noise, **condition):
                noise_hashes.append(hashlib.sha256(noise.numpy().tobytes()).hexdigest())
                return selected.target_mm + noise

            model = SimpleNamespace(model=torch.nn.Identity(), predict=predict, runtime={}, checkpoint_sha256="body-fixture")
            budget = SimpleNamespace(check=lambda: None)
            body = cascade.evaluate("body", model, budget, {}, diagnostics=True)
            hand = cascade.evaluate("hand", model, budget, {}, model, diagnostics=True)
            self.assertEqual(merge_metrics("hand", [hand, hand])["score"], hand["score"])
            evaluations.append((body, hand, noise_hashes))
        self.assertEqual(evaluations[0], evaluations[1])
        self.assertEqual(evaluations[1], evaluations[2])
        self.assertEqual(evaluations[0][2][0], evaluations[0][2][1])
        self.assertNotEqual(evaluations[0][2][0], evaluations[0][2][2])

    def test_frozen_feature_group_resumes_in_empty_local_workspace(self):
        data = FixtureData()
        data.real_frames = lambda _key, _split: data.width
        data.budget = self.ledger
        frontend = torch.nn.Linear(2, 2)

        def features(key, split, _model, _size, pin):
            valid = torch.ones(1, 4, 4, dtype=torch.bool)
            return Clip(torch.ones(1, 4, 56), torch.zeros(1, 4, 2, 9), torch.zeros(1, 4, 4, 3), valid, valid[..., 0],
                        torch.arange(4, dtype=torch.float64)[None]/25, "predicted_2d", split=split, input_sha256=pin)

        data.clip = features
        with self.ledger.session(time.monotonic()):
            first = PersistentFeatures(self.root / "first", data, "weights", self.root / "drive", self.cfg, self.ledger, self.owner)
            clips, info = first.build(frontend, [128, 128])
            self.assertEqual(info["feature_new_clips"], 3)
            data.clip = lambda *args: self.fail("durable clips should be restored rather than recomputed")
            new = PersistentFeatures(self.root / "new", data, "weights", self.root / "drive", self.cfg, self.ledger, self.owner)
            recovered, stats = new.build(frontend, [128, 128])
            k = data.clips["validation"][0]
            self.assertEqual(stats["feature_new_clips"], 0)
            self.assertTrue(torch.equal(recovered["validation"][k].features, clips["validation"][k].features))

    def test_training_adapter_updates_and_resumes_through_all_roles(self):
        records = [{"sequence_id": f"{scenario}/{split}", "frame_index": i, "split": split, "files": {}}
                   for scenario in contract.SCENARIOS for split in ("train", "validation") for i in range(2)]
        document = {"content_sha256": "fixture", "records": records}
        spec = FixtureData().spec
        plan = contract.plan(document, spec, self.cfg, self.root)
        values = {"xy": torch.ones(2, 4, 2)*28, "xy_valid": torch.ones(2, 4, dtype=torch.bool),
                  "target_mm": torch.arange(12, dtype=torch.float32).reshape(4, 3),
                  "target_valid": torch.ones(4, dtype=torch.bool), "camera": torch.zeros(2, 9),
                  "rotation64": torch.eye(3, dtype=torch.float64).repeat(2, 1, 1),
                  "translation64": torch.tensor([[0., -80., 0.], [0., 80., 0.]], dtype=torch.float64)}
        image_cache = SimpleNamespace(identity="fixture-cache", metrics={}, prepare_frontend=lambda: None,
                                      frame=lambda k, s: (torch.full((2, 3, 128, 128), .2+k[1]*.1), values))
        data = FullData(document, spec, self.cfg, plan, image_cache, self.ledger, "fixture-source")
        drive = self.root / "slot-A"
        drive.mkdir()
        runtime = json.loads((ROOT / "configs/rewind_v12_full_training.json").read_bytes())
        runtime["epochs"] = dict.fromkeys(("frontend", "body", "hand"), 1)
        runtime["teacher"].update(hidden=8, heads=2, graph_layers=1, temporal_layers=1, diffusion_steps=20, sampling_steps=2)
        runtime["limits"]["checkpoint_every"] = 1
        identity = {**self.identity, "source": "fixture-source"}

        def trainer(name):
            pipe = Training(self.cfg, data, self.root / (name + "-features"), torch.device("cpu"), self.root / name,
                            drive, self.ledger, self.owner, identity)
            pipe.checkpoints = Checkpoints(drive / "checkpoints", self.root, identity, pipe, self.cfg, self.ledger, self.owner)
            return pipe

        with patch("scripts.v2_gpu_training_pipeline.runtime_config", return_value=runtime):
            with self.ledger.session(time.monotonic()):
                first = trainer("first")
                first.root.mkdir()
                self.assertEqual(first.run(stop_after_updates=4)["status"], "paused")
                self.assertEqual(first.progress["stage"], "body")
                tip = first.checkpoints.save()
            with self.ledger.session(time.monotonic()):
                resumed = trainer("resumed")
                pin = resumed.checkpoints.restore(tip)
                resumed.first_required = False
                final = resumed.run(pin)
                self.assertEqual(final["status"], "completed")
                self.assertEqual(final["cursor"]["updates"], 9)
                for row, role in zip(final["cursor"]["history"], ("frontend", "body", "hand"), strict=True):
                    self.assertEqual(row["stage"], role)
                    self.assertEqual(set(row["validation"]["per_scenario_reference"]), set(contract.SCENARIOS))
                self.assertEqual(final["cursor"]["history"][-1]["validation"]["real_frames"], 6)
                self.assertEqual(final["cursor"]["history"][-1]["validation"]["upper_source"], "predicted_frozen_body")
