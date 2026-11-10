import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from unittest import TestCase
from unittest.mock import Mock, patch
import zipfile

import torch

from rewind_v1.expansion import RUN_ID
from rewind_v1.runner import FORMAT
from scripts import v11_recovery as recovery
from scripts.build_archvis_v1_diagnostic import _expansion_cells


class V11RecoveryTests(TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1])
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.work = self.root / "work"
        self.drive = self.root / "drive"
        self.run = self.work / "training"
        self.run.mkdir(parents=True)
        self.drive.mkdir()
        (self.work / "notebook_training_result.json").write_text('{"status":"stopped"}', encoding="utf-8")

    def test_real_training_recovery_publish_and_attempt_collision(self):
        checkpoint = self.run / "last.pt"
        torch.save({"binding": {"format": FORMAT, "input": {"run_id": RUN_ID}},
                    "progress": {"best": {}, "milestones": {}}}, checkpoint)
        (self.run / "run.json").write_text('{"status":"paused"}', encoding="utf-8")
        local = recovery.package(self.work, self.run)
        archive = Path(local["local_zip"])
        with zipfile.ZipFile(archive) as bundle:
            self.assertIsNone(bundle.testzip())
            self.assertEqual(set(bundle.namelist()), {"run.json", "last.pt", "recovery_receipt.json"})
        attempt = recovery.prepare_attempt(self.drive)
        result = recovery.publish(self.work, archive, attempt)
        receipt = json.loads(Path(result["receipt"]).read_bytes())
        self.assertEqual(receipt["state"], "verified")
        self.assertEqual(receipt["last_sha256"], hashlib.sha256(checkpoint.read_bytes()).hexdigest())
        self.assertEqual((attempt / "training.zip").read_bytes(), archive.read_bytes())
        second_archive = Path(recovery.package(self.work, self.run)["local_zip"])
        with self.assertRaises(FileExistsError):
            recovery.publish(self.work, second_archive, attempt)
        self.assertEqual(json.loads(Path(result["receipt"]).read_bytes()), receipt)
        self.assertEqual((attempt / "training.zip").read_bytes(), archive.read_bytes())

    def test_drive_readback_mismatch_preserves_local_and_allows_new_attempt(self):
        local = recovery.package(self.work, self.run)
        archive = Path(local["local_zip"])
        original = archive.read_bytes()
        attempt = recovery.prepare_attempt(self.drive)
        measure = recovery._measure

        def corrupt_copy(stream, size, deadline, destination=None):
            result = measure(stream, size, deadline, destination)
            if destination is not None:
                destination.seek(0)
                destination.write(b"X")
            return result

        with patch.object(recovery, "_measure", side_effect=corrupt_copy):
            with self.assertRaisesRegex(ValueError, "drive_archive_hash"):
                recovery.publish(self.work, archive, attempt)
        self.assertEqual(archive.read_bytes(), original)
        self.assertFalse((attempt / "receipt.json").exists())
        self.assertTrue((attempt / "status.zip.partial").is_file())
        failure = json.loads((archive.parent / (attempt.name + ".json")).read_bytes())
        self.assertEqual(failure["state"], "failed")
        retry = recovery.prepare_attempt(self.drive)
        self.assertNotEqual(retry, attempt)
        self.assertEqual(recovery.publish(self.work, archive, retry)["state"], "verified")

    def test_notebook_starts_disabled_and_saves_when_training_raises(self):
        helpers, _, launch, recover = _expansion_cells()
        exec(launch, {})
        exec(recover, {})
        context = {"Path": Path}
        exec(helpers, context)
        child = Mock()
        context.update(DRIVE_ROOT=self.drive, recovery_child=child)
        with patch.object(Path, "is_mount", return_value=False):
            with self.assertRaises(FileNotFoundError):
                context["prepare_recovery"](False)
        child.assert_not_called()

        training = Mock(side_effect=[None, RuntimeError("training failed")])
        saving = Mock(side_effect=OSError("copy failed"))
        context.update(RUN=self.root / "fresh", verified=self.root, SOURCE=self.root, INPUTS=self.root,
                       DIAGNOSTIC_BUNDLE=self.root, WORK=self.work, sys=sys, json=json, tracked_child=training,
                       prepare_recovery=Mock(return_value="attempt"), save_recovery=saving,
                       PINS={"expansion_decision": {"content_sha256": "decision"}, "config_path": "config",
                             "config_sha256": "", "source_code_sha256": "", "selection_sha256": "",
                             "cache_sha256": "", "decision_sha256": "", "training_hard_seconds": 7200})
        with self.assertRaisesRegex(RuntimeError, "training failed"):
            exec(launch.replace("RUN_EPOCH_EXPANSION = False", "RUN_EPOCH_EXPANSION = True"), context)
        saving.assert_called_once_with("attempt", browser_download=False)

    def test_notebook_copy_timeout_writes_local_failure_receipt(self):
        helpers, _, _, _ = _expansion_cells()
        local = recovery.package(self.work, self.run)
        archive = Path(local["local_zip"])
        attempt = recovery.prepare_attempt(self.drive)
        from contextlib import suppress

        context = {"Path": Path, "WORK": self.work, "json": json, "suppress": suppress}
        exec(helpers, context)
        context["recovery_child"] = Mock(side_effect=subprocess.TimeoutExpired("publish", recovery.COPY_SECONDS))
        with self.assertRaises(subprocess.TimeoutExpired):
            context["save_recovery"](str(attempt), str(archive))
        self.assertTrue(archive.is_file())
        failure = json.loads((archive.parent / (attempt.name + "-notebook-error.json")).read_bytes())
        self.assertEqual((failure["state"], failure["type"]), ("failed", "TimeoutExpired"))
