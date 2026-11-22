"""Local promotion and verification against the fixed trainer functions."""

from contextlib import ExitStack
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import patch

import torch

from rewind_archive.contracts import canonical
from rewind_v1.run_stages import FrameStage
from rewind_v1.runner import _fit_state
from rewind_v1.v2_budget import Ledger
from scripts import recover_v2_gpu128_a_6200 as recovery


class RecoveryTests(TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name).resolve()
        self.seed = self.root / recovery.RUN
        self.candidate = self.seed / recovery.CANDIDATE
        self.candidate.mkdir(parents=True)
        self.checkpoints = self.seed/"checkpoints"
        self.checkpoints.mkdir()
        self.cfg = recovery.training_config(42)
        self.identity = recovery.identity_for("A", {"source_sha256": recovery.SOURCE_SHA, "manifest_content_sha256": recovery.MANIFEST_SHA})
        self.environment = {"device": "cuda:0", "device_name": "fixture", "device_memory_bytes": 1}
        runtime = {"seed": 42, "frontend": {"size": [128,128], "learning_rate": 0.0001, "scheduler_step": 1,
                   "scheduler_gamma": 1.0}, "epochs": {"frontend": 3, "body": 3, "hand": 3},
                   "limits": {"checkpoint_bytes": 32*recovery.MIB}}
        self.binding = {"experiment": self.identity, "v2_config": self.cfg, "source_code_sha256": recovery.SOURCE_SHA,
                        "format": "rewind_v2_gpu_training_v1", "config": runtime, "environment": {"device": "cuda:0"}}
        stage = FrameStage(runtime["frontend"], 2, torch.device("cpu"))
        for parameter in stage.model.parameters():
            parameter.grad = torch.ones_like(parameter)
        stage.optimizer.step()
        stage.scheduler.step()
        fit = _fit_state(stage)
        fit["rng"]["torch_cuda"] = [torch.ones(8, dtype=torch.uint8)]
        original = self.root/"source-step1"
        original.mkdir()
        cursor = {**recovery.CURSOR, "batch": 1, "updates": 1}
        original_pin, progress = self.write_checkpoint(original, cursor, fit)
        initial = SimpleNamespace(last=original_pin, root=original, durable_progress=progress, environment=self.environment)
        ledger = Ledger(self.root/"original-usage", self.identity, recovery.recovery_config(self.cfg))
        with ledger.invocation("fixture"):
            manager = recovery.Checkpoints(self.checkpoints, self.root, self.identity, initial, ledger.cfg, ledger, lambda: None)
            first = manager.save(first=True)
        gate = self.pin(self.checkpoints/"first-update.json")
        gate_body = recovery.json.loads((self.checkpoints/"first-update.json").read_bytes())
        fit["scheduler"]["last_epoch"] = 6200
        original_value = next(iter(fit["optimizer"]["state"].values()))
        original_value["step"] = torch.tensor(6200.)
        candidate_pin, candidate_progress = self.write_checkpoint(self.candidate, recovery.CURSOR, fit)
        (self.candidate/"run.json").write_bytes(canonical({**self.binding, "actual_environment": self.environment,
            "last_checkpoint": candidate_pin, "checkpoint_cursor": candidate_progress,
            "cursor": {**candidate_progress, "batch": 6214, "updates": 6214}}))
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        for key, value in {"LAST_PIN": candidate_pin, "REPORT_PIN": self.pin(self.candidate/"run.json"), "GATE_PIN": gate,
                           "FIRST_GATE": gate_body["content_sha256"], "FIRST_STEP": first["content_sha256"]}.items():
            self.stack.enter_context(patch.object(recovery, key, value))

    def pin(self, path):
        raw = path.read_bytes()
        return {"name": path.name, "bytes": len(raw), "sha256": recovery.sha(raw)}

    def write_checkpoint(self, root, cursor, fit):
        progress = {**cursor, "best": {}, "history": [], "milestones": {},
                    "partial_loss_sums": {"first_update_gradient_l2": 1., "supervised_joint_views": 2}}
        path = root/"last.pt"
        torch.save({"binding": self.binding, "progress": progress, "fit": fit}, path)
        return self.pin(path), progress

    def execute(self, publish=False):
        scratch = Path(tempfile.mkdtemp(dir=self.root, prefix="local-"))
        result = {}
        recovery.recover(self.root, scratch, publish, 2, result)
        return result

    def test_dry_publish_normal_restore_and_idempotence(self):
        before = {p.name: self.pin(p) for p in self.checkpoints.iterdir()}
        self.assertEqual(self.execute()["status"], "verified_only")
        self.assertEqual({p.name: self.pin(p) for p in self.checkpoints.iterdir()}, before)
        result = self.execute(True)
        self.assertEqual(result["status"], "published_verified")
        self.assertEqual(result["receipt"]["last"], recovery.LAST_PIN)
        self.assertEqual(result["receipt"]["previous"], recovery.FIRST_STEP)
        self.assertEqual(result["receipt"]["sequence"], 2)
        after = {p.name: self.pin(p) for p in self.checkpoints.iterdir()}
        repeated = self.execute(True)
        self.assertEqual(repeated["status"], "already_published_verified")
        self.assertEqual(repeated["normal_resume_tip"], result["normal_resume_tip"])
        self.assertEqual({p.name: self.pin(p) for p in self.checkpoints.iterdir()}, after)

    def test_bad_candidate_lock_and_unreceipted_zip_fail_closed(self):
        with recovery.writer(self.seed):
            with self.assertRaisesRegex(ValueError, "locked"):
                self.execute(True)
        orphan = self.checkpoints/"result-000002-orphan.zip"
        orphan.write_bytes(b"untouched")
        with self.assertRaisesRegex(ValueError, "unreceipted"):
            self.execute(True)
        self.assertEqual(orphan.read_bytes(), b"untouched")
        report = self.candidate/"run.json"
        raw = report.read_bytes()
        report.write_bytes(raw[:-1]+b"!")
        with self.assertRaisesRegex(ValueError, "recovery_pin: run.json"):
            self.execute(True)
        self.assertFalse((self.checkpoints/"step-000002.json").exists())
