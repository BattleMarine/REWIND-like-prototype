from contextlib import redirect_stdout
import copy
import io
import json
from pathlib import Path
from types import SimpleNamespace
import shutil
import subprocess
import sys
import tempfile
import time
from unittest import TestCase
from unittest.mock import Mock, patch

import torch

from rewind_archive.contracts import canonical, digest
from rewind_m1.training import rng_state
from rewind_prepare.converter import sha, validate_pose_geometry
from rewind_prepare.temporal_selection import seal
from rewind_v1 import oracle_cli, oracle_data, oracle_overfit
from rewind_v1.data import gt_upper_train
from rewind_v1.runner import Budget
from scripts import v14_recovery
from scripts.build_archvis_v14_oracle import notebook_document
import test_prepare


class OracleOverfitTests(TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(dir=oracle_data.ROOT)
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.prepared = self.root / "prepared"
        self.prepared.mkdir()
        spec_raw = (oracle_data.ROOT / "configs/real_archvis62_v1.json").read_bytes().replace(b"\r\n", b"\n")
        self.assertEqual(sha(spec_raw), oracle_data.SPEC_SHA)
        self.spec = json.loads(spec_raw)
        (self.prepared / "spec.json").write_bytes(spec_raw)
        joints = len(self.spec["joint_names"])
        xy = torch.full((50, 2, joints, 2), 300.)
        xy[0, 0, 0] = torch.tensor([1300., -20.])
        xy_valid = torch.ones((50, 2, joints), dtype=torch.bool)
        xy_valid[1, 1, 1] = False
        xy[1, 1, 1] = 0.
        target = torch.arange(50*joints*3, dtype=torch.float32).view(1, 50, joints, 3)/20
        data = {"xy": xy, "xy_valid": xy_valid, "features": oracle_data.oracle_features(xy, xy_valid),
                "target_mm": target, "target_valid": torch.ones((1, 50, joints), dtype=torch.bool),
                "camera": torch.zeros((1, 50, 2, 9)), "time_valid": torch.ones((1, 50), dtype=torch.bool),
                "seconds": torch.arange(50, dtype=torch.float64)[None]/25}
        buffer = io.BytesIO()
        torch.save(data, buffer)
        raw = buffer.getvalue()
        (self.prepared / "oracle_clip.pt").write_bytes(raw)
        self.input = seal({"schema": oracle_data.SCHEMA, "run_id": oracle_data.RUN_ID, "data_kind": "synthetic_fixture",
            "oracle": oracle_data.ORACLE, "selection_sha256": oracle_data.HISTORICAL["selection_sha256"],
            "cache_sha256": oracle_data.HISTORICAL["cache_sha256"], "spec": self.spec,
            "spec_sha256": sha(spec_raw), "split": "train", "keys": [[oracle_data.SEQUENCE, f] for f in range(50)],
            "tensor": {"name": "oracle_clip.pt", "bytes": len(raw), "sha256": sha(raw)},
            "source_code_sha256": oracle_data.source_identity()})
        (self.prepared / "oracle_input.json").write_bytes(canonical(self.input)+b"\n")
        self.config = json.loads((oracle_data.ROOT / "configs/rewind_v14_oracle_overfit.json").read_bytes())
        self.config.update(updates_per_model=3, milestones=[0, 2, 3], checkpoint_every=1, log_every=1)
        self.config["teacher"].update(hidden=8, heads=2, graph_layers=1, temporal_layers=1,
                                     diffusion_steps=20, attention_past_frames=4, learning_rate=.001)
        self.config["limits"].update(max_seconds=120, output_bytes=128*1024**2, checkpoint_bytes=16*1024**2)

    def experiment(self, name, prepared=None):
        path = self.root / name
        path.mkdir(exist_ok=True)
        prepared = prepared or self.prepared
        clip, receipt = oracle_data.load_prepared(prepared, self.input["content_sha256"])
        return oracle_overfit.OracleExperiment(self.config, clip, receipt, "cpu", path,
            Budget(self.config["limits"]), prepared / "oracle_clip.pt")

    def test_oracle_features_keep_offscreen_and_distinguish_missing_presence(self):
        clip, _ = oracle_data.load_prepared(self.prepared, self.input["content_sha256"])
        features = clip.features.reshape(50, 62, 14)
        self.assertGreater(float(features[0, 0, 0]), 1.)
        self.assertLess(float(features[0, 0, 1]), -1.)
        self.assertEqual(features[0, 0, 2:5].tolist(), [1., 1., 1.])
        self.assertEqual(features[1, 1, 7:].tolist(), [0.]*7)
        self.assertTrue(clip.target_valid.all())
        path = self.prepared / "oracle_clip.pt"
        path.write_bytes(path.read_bytes()+b"damage")
        with self.assertRaisesRegex(ValueError, "tensor_pin"):
            oracle_data.load_prepared(self.prepared, self.input["content_sha256"])

    def test_body_rejects_upper_and_evaluation_preserves_training_rng(self):
        experiment = self.experiment("rng")
        with self.assertRaisesRegex(ValueError, "body_rejects_gt_upper"):
            experiment.stages["body"].condition(gt_upper_train(experiment.clip, self.spec))
        with self.assertRaisesRegex(ValueError, "hand_requires_gt_upper"):
            experiment.stages["hand"].condition(None)
        before = rng_state()
        experiment.evaluate()
        after = rng_state()
        self.assertTrue(torch.equal(before["torch_cpu"], after["torch_cpu"]))
        self.assertEqual(before["python"], after["python"])
        result = experiment.evaluations["0"]
        self.assertIn("k1_target_noised_oracle_reconstruction", result["metrics"]["body"])
        baseline = result["constant_pose_gt_clip_mean_baseline"]["body"]
        self.assertEqual(baseline["predicted_first_last_displacement_mm"], 0.)
        self.assertGreater(baseline["target_first_last_displacement_mm"], 0.)

    def test_missing_gt_wrist_masks_oracle_global_hand_but_not_present_finger_target(self):
        experiment = self.experiment("missing_wrist")
        wrist = self.spec["joint_names"].index(self.spec["wrists"]["left"])
        experiment.clip.target_valid[0, 0, wrist] = False
        from rewind_v1.data import hand_clip
        experiment.hand_clip = hand_clip(experiment.clip, self.spec)
        experiment.upper = gt_upper_train(experiment.clip, self.spec)
        experiment.evaluate()
        metrics = experiment.evaluations["0"]["metrics"]
        self.assertEqual(metrics["hand_global_gt_wrist_oracle"]["valid_joints"], 1980)
        self.assertEqual(metrics["hand"]["ddim10_pure_noise"]["valid_joints"], 1980)
        self.assertEqual(metrics["hand_global_predicted_wrist_gt_upper_oracle"]["valid_joints"], 2000)

    def test_pause_resume_matches_weights_optimizer_and_rng_without_raw_cache_reads(self):
        full = self.experiment("full")
        with patch.object(oracle_data.RawCache, "read_pair", side_effect=AssertionError("no training I/O")):
            final = full.run()
            paused = self.experiment("paused").run(stop_after=1)
            resumed = self.experiment("paused")
            result = resumed.run(paused["last_checkpoint"]["sha256"])
        self.assertEqual((final["status"], result["status"]), ("completed", "completed"))
        self.assertEqual(result["optimizer_updates"], 6)
        for role in ("body", "hand"):
            for key, value in full.stages[role].model.state_dict().items():
                self.assertTrue(torch.equal(value, resumed.stages[role].model.state_dict()[key]), f"{role}/{key}")
            left, right = full.stages[role].optimizer.state_dict(), resumed.stages[role].optimizer.state_dict()
            for key, state in left["state"].items():
                for name, value in state.items():
                    self.assertTrue(torch.equal(value, right["state"][key][name]))
        self.assertEqual(final["evaluations"], result["evaluations"])
        state = torch.load(self.root / "paused/last.pt", weights_only=True, map_location="cpu")
        self.assertTrue(torch.equal(state["fits"]["body"]["rng"]["torch_cpu"], state["fits"]["hand"]["rng"]["torch_cpu"]))

    def test_recovery_cold_restore_and_binding_mismatch(self):
        report = self.experiment("original").run(stop_after=1)
        archive = self.root / "recovery.zip"
        receipt = oracle_cli.recover(self.root / "original", archive)
        with patch.object(oracle_cli, "validate_config"):
            restored = oracle_cli.restore(archive, receipt["sha256"], self.root / "restored")
        resumed = self.experiment("restored/training", Path(restored["prepared"]))
        self.assertEqual(resumed.run(restored["last_sha256"])["status"], "completed")
        state = torch.load(self.root / "original/last.pt", weights_only=True, map_location="cpu")
        state["binding"]["input_sha256"] = "f"*64
        torch.save(state, self.root / "original/last.pt")
        pin = sha((self.root / "original/last.pt").read_bytes())
        with self.assertRaisesRegex(ValueError, "resume_binding"):
            self.experiment("original").run(pin)

    def test_resume_identity_survives_immediate_failure_before_new_save(self):
        paused = self.experiment("interrupt_resume").run(stop_after=1)
        expected_sha = paused["last_checkpoint"]["sha256"]
        experiment = self.experiment("interrupt_resume")
        experiment.resume(expected_sha)
        self.assertEqual(experiment.last["sha256"], expected_sha)
        with patch.object(experiment.budget, "check", side_effect=RuntimeError("immediate failure")):
            with self.assertRaisesRegex(RuntimeError, "immediate failure"):
                experiment.run(expected_sha)
        report = json.loads((experiment.root / "run.json").read_bytes())
        self.assertEqual(report["last_checkpoint"]["sha256"], expected_sha)
        self.assertEqual(report["durable_paired_updates"], 1)

    def test_nonfinite_loss_rejects_before_optimizer_and_scheduler_change(self):
        experiment = self.experiment("nonfinite")
        stage = experiment.stages["body"]
        stage.update(experiment.body_clip, None, False)
        parameters = {key: value.clone() for key, value in stage.model.state_dict().items()}
        optimizer = copy.deepcopy(stage.optimizer.state_dict())
        scheduler = stage.scheduler.state_dict().copy()
        loss = torch.tensor(float("inf"), requires_grad=True)
        with patch.object(oracle_overfit, "motion_losses", return_value=(loss, {"position": loss, "frame": loss})):
            with self.assertRaisesRegex(ValueError, "nonfinite_loss"):
                stage.update(experiment.body_clip, None, False)
        for key, value in parameters.items():
            self.assertTrue(torch.equal(value, stage.model.state_dict()[key]))
        for key, item in optimizer["state"].items():
            for name, value in item.items():
                self.assertTrue(torch.equal(value, stage.optimizer.state_dict()["state"][key][name]))
        self.assertEqual(stage.scheduler.state_dict(), scheduler)
        self.assertEqual(stage.step, 1)

    def test_pose_audit_rechecks_geometry_and_camera_against_pinned_rows(self):
        fixture = test_prepare.PreparationTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        pose_path = fixture.raw / fixture.seq / "json/frame_0.json"
        raw = pose_path.read_bytes()
        from collections import defaultdict
        camera = validate_pose_geometry(json.loads(raw), fixture.spec, fixture.calib, defaultdict(list))
        key = (fixture.seq, 0)
        cache = SimpleNamespace(index={key: ("part.zip", {"files": {"pose": {"sha256": sha(raw)}}})},
                                read_pair=lambda *args, **kwargs: {"pose": raw})
        tensors, samples = oracle_data.audit_clip(cache, [key], fixture.spec, fixture.calib,
                                                 {key: {"camera_pose": camera}}, time.monotonic())
        self.assertEqual(tensors["features"].shape, (1, 1, 62*14))
        self.assertEqual(samples[0]["files"]["pose"]["sha256"], sha(raw))
        camera = copy.deepcopy(camera)
        camera["left"]["translation"][0] += 10
        with self.assertRaisesRegex(ValueError, "camera_audit_pin"):
            oracle_data.audit_clip(cache, [key], fixture.spec, fixture.calib, {key: {"camera_pose": camera}}, time.monotonic())

    def test_cuda_guard_and_notebook_disabled_or_failure_finally(self):
        with patch.object(oracle_cli, "require_cuda_training", side_effect=RuntimeError("CUDA required")), \
             patch.object(oracle_cli, "_regular_read") as read:
            with self.assertRaisesRegex(RuntimeError, "CUDA required"):
                oracle_cli.main(["train", "--config", "missing", "--config-sha256", "x", "--prepared", "missing",
                                 "--input-sha256", "x", "--source-sha256", digest(oracle_data.source_identity()), "--output", "missing"])
            read.assert_not_called()
        cells = {cell["id"]: "".join(cell["source"]) for cell in notebook_document({})["cells"]}
        with patch("subprocess.run") as child:
            exec(cells["oracle-overfit"], {"WORK": self.root})
            exec(cells["recover"], {})
            child.assert_not_called()
        saving = Mock()
        context = {"WORK": self.root, "prepare_drive": Mock(return_value="attempt"), "sys": sys,
                   "tracked_child": Mock(side_effect=RuntimeError("fixture stop")), "save_backup": saving}
        with self.assertRaisesRegex(RuntimeError, "fixture stop"):
            exec(cells["oracle-overfit"].replace("RUN_ORACLE_OVERFIT = False", "RUN_ORACLE_OVERFIT = True"), context)
        saving.assert_called_once_with("attempt", download=False)

    def test_notebook_interrupt_finally_in_child_process(self):
        # A deliberate KeyboardInterrupt retains a Ctrl-C exit status on this Windows Python.
        script = '''
from pathlib import Path
from unittest.mock import Mock
import sys
from scripts.build_archvis_v14_oracle import notebook_document
cells = {c["id"]: "".join(c["source"]) for c in notebook_document({})["cells"]}
def saved(*args, **kwargs):
    print("RECOVERY_FINALLY_REACHED", flush=True)
context = {"WORK": Path.cwd(), "prepare_drive": Mock(return_value="attempt"), "sys": sys,
           "tracked_child": Mock(side_effect=KeyboardInterrupt("fixture interrupt")), "save_backup": saved}
exec(cells["oracle-overfit"].replace("RUN_ORACLE_OVERFIT = False", "RUN_ORACLE_OVERFIT = True"), context)
'''
        child = subprocess.run([sys.executable, "-c", script], cwd=oracle_data.ROOT,
                               timeout=30, capture_output=True, text=True)
        self.assertNotEqual(child.returncode, 0)
        self.assertIn("RECOVERY_FINALLY_REACHED", child.stdout)
        self.assertIn("KeyboardInterrupt: fixture interrupt", child.stderr)

    def test_drive_copy_corruption_preserves_local_and_new_attempt_retry(self):
        self.experiment("backup_run").run(stop_after=1)
        work = self.root / "work"
        work.mkdir()
        drive = self.root / "drive"
        drive.mkdir()
        with patch.object(v14_recovery.subprocess, "run") as child:
            package = oracle_cli.recover(self.root / "backup_run", self.root / "training.zip")
            child.side_effect = None
            child.return_value = SimpleNamespace(stdout=json.dumps(package))
            original = v14_recovery.package(work, self.root / "backup_run")
        # The real package command publishes to a unique local path. Move the fixture archive there.
        local = Path(original["local_zip"])
        shutil.copyfile(self.root / "training.zip", local)
        attempt = Path(v14_recovery.prepare(drive)["attempt"])
        measure = v14_recovery._measure

        def corrupt(stream, size, deadline, destination=None):
            result = measure(stream, size, deadline, destination)
            if destination is not None:
                destination.seek(0)
                destination.write(b"X")
            return result

        with patch.object(v14_recovery, "_measure", side_effect=corrupt):
            with self.assertRaisesRegex(ValueError, "drive_archive_hash"):
                v14_recovery.publish(work, local, attempt)
        self.assertTrue(local.is_file())
        self.assertFalse((attempt / "receipt.json").exists())
        retry = Path(v14_recovery.prepare(drive)["attempt"])
        self.assertEqual(v14_recovery.publish(work, local, retry)["state"], "verified")
        with self.assertRaises(FileExistsError):
            v14_recovery.publish(work, local, retry)
