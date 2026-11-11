import ast
from contextlib import redirect_stdout, suppress
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from unittest import TestCase
from unittest.mock import Mock, patch
import zipfile

from scripts import v12_recovery as recovery
from scripts.build_archvis_v12_prepare import BACKUP_TIMEOUTS, TIMEOUTS, notebook_document


class V12RecoveryTests(TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1])
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.work, self.base, self.drive = (self.root / name for name in ("work", "recovery", "drive"))
        for path in (self.work, self.base, self.drive):
            path.mkdir()
        (self.work / "notebook_status.json").write_text('{"phase":"audit","status":"stopped"}', encoding="utf-8")
        (self.work / "audit_status.json").write_text('{"pairs_verified":3}', encoding="utf-8")
        self.context = {"phase": "audit", "original_work": str(self.work.resolve()), "source_revision": "a"*40,
                        "source_zip_sha256": "b"*64, "input_zip_sha256": "c"*64,
                        "snapshot_sha256": "d"*64, "coverage_sha256": "e"*64}
        self.context_path = self.work / "context.json"
        self.context_path.write_text(json.dumps(self.context), encoding="utf-8")
        self.pins = {"source": {"revision": "a"*40, "sha256": "b"*64}, "inputs": {"sha256": "c"*64},
                     "snapshot_sha256": "d"*64, "coverage_sha256": "e"*64}

    def local_zip(self):
        return Path(recovery.package(self.base, self.work, self.context_path)["local_zip"])

    def helpers(self):
        notebook = notebook_document(self.pins)
        cells = {cell["id"]: "".join(cell["source"]) for cell in notebook["cells"]}
        definitions = ast.Module(body=[node for node in ast.parse(cells["bootstrap"]).body
                                       if isinstance(node, ast.FunctionDef)], type_ignores=[])
        namespace = {"Path": Path, "json": json, "subprocess": subprocess, "time": time, "sys": sys,
                     "SOURCE": self.root, "PINS": self.pins, "suppress": suppress,
                     "RECOVERY_BASE": self.base, "DRIVE_ROOT": self.drive,
                     "TIMEOUTS": TIMEOUTS, "BACKUP_TIMEOUTS": BACKUP_TIMEOUTS,
                     "MOUNT_DRIVE": False, "BROWSER_DOWNLOAD": False}
        exec(compile(definitions, "notebook_helpers", "exec"), namespace)
        return namespace, cells

    def test_core_recover_copy_metadata_and_duplicate_attempt(self):
        archive = self.local_zip()
        before = archive.read_bytes()
        with zipfile.ZipFile(archive) as bundle:
            self.assertIsNone(bundle.testzip())
            self.assertEqual(set(bundle.namelist()), {"audit_status.json", "notebook_status.json", "recovery_receipt.json"})
        attempt = recovery.prepare_attempt(self.drive, "audit")
        result = recovery.publish(self.base, archive, attempt, self.context_path)
        receipt_path = Path(result["receipt"])
        receipt_before = receipt_path.read_bytes()
        receipt = json.loads(receipt_before)
        self.assertEqual(receipt["context"], self.context)
        self.assertEqual(receipt["state"], "verified")
        second = self.local_zip()
        with self.assertRaises(FileExistsError):
            recovery.publish(self.base, second, attempt, self.context_path)
        self.assertEqual(receipt_path.read_bytes(), receipt_before)
        self.assertEqual((attempt / "reports.zip").read_bytes(), before)

    def test_corrupt_readback_keeps_local_zip_and_retry_uses_new_attempt(self):
        archive = self.local_zip()
        before = archive.read_bytes()
        attempt = recovery.prepare_attempt(self.drive, "audit")
        measure = recovery._measure

        def corrupt(stream, size, deadline, destination=None):
            result = measure(stream, size, deadline, destination)
            if destination is not None:
                destination.seek(0)
                destination.write(b"X")
            return result

        with patch.object(recovery, "_measure", side_effect=corrupt):
            with self.assertRaisesRegex(ValueError, "drive_archive_hash"):
                recovery.publish(self.base, archive, attempt, self.context_path)
        self.assertEqual(archive.read_bytes(), before)
        self.assertFalse((attempt / "receipt.json").exists())
        status = json.loads((archive.parent / (attempt.name+".json")).read_bytes())
        self.assertEqual(status["state"], "failed")
        retry = recovery.prepare_attempt(self.drive, "audit")
        self.assertEqual(recovery.publish(self.base, archive, retry, self.context_path)["state"], "verified")

    def test_source_corruption_is_not_published(self):
        archive = self.local_zip()
        data = archive.read_bytes()
        archive.write_bytes(b"X"+data[1:])
        attempt = recovery.prepare_attempt(self.drive, "audit")
        with self.assertRaisesRegex(ValueError, "local_archive_hash"):
            recovery.publish(self.base, archive, attempt, self.context_path)
        self.assertTrue(archive.is_file())
        self.assertFalse((attempt / "receipt.json").exists())

    def test_phase_finally_runs_for_success_failure_timeout_and_interrupt(self):
        cases = [None, subprocess.CalledProcessError(1, "audit"), subprocess.TimeoutExpired("audit", 21630),
                 KeyboardInterrupt("stop")]
        for error in cases:
            with self.subTest(error=type(error).__name__):
                namespace, _ = self.helpers()
                namespace["prepare_backup"] = Mock(return_value="attempt-test")
                namespace["save_backup"] = Mock()
                with patch("subprocess.run", side_effect=error):
                    if error is None:
                        namespace["tracked_phase"]("audit", self.work, [])
                    else:
                        with self.assertRaises(type(error)):
                            namespace["tracked_phase"]("audit", self.work, [])
                namespace["save_backup"].assert_called_once()
                state = json.loads((self.work / "notebook_status.json").read_bytes())
                self.assertEqual(state["phase"], "audit")
                self.assertEqual(state["status"], "complete" if error is None else
                                 "failed" if isinstance(error, subprocess.CalledProcessError) else "stopped")

    def test_backup_timeout_receipt_and_original_phase_error(self):
        archive = self.local_zip()
        namespace, _ = self.helpers()
        namespace["recovery_child"] = Mock(side_effect=subprocess.TimeoutExpired("publish", 630))
        with redirect_stdout(io.StringIO()):
            with self.assertRaises(subprocess.TimeoutExpired):
                namespace["save_backup"](self.work, self.context_path, "attempt-timeout", str(archive))
        self.assertTrue(archive.is_file())
        self.assertEqual(json.loads((self.work / "attempt-timeout-backup-error.json").read_bytes())["type"], "TimeoutExpired")
        namespace["prepare_backup"] = Mock(return_value="attempt-test")
        namespace["save_backup"] = Mock(side_effect=OSError("Drive failed"))
        original = KeyboardInterrupt("original stop")
        with patch("subprocess.run", side_effect=original), redirect_stdout(io.StringIO()):
            with self.assertRaises(KeyboardInterrupt) as caught:
                namespace["tracked_phase"]("preflight", self.work, [])
        self.assertIs(caught.exception, original)

    def test_no_mount_and_default_cells_do_not_execute_phase(self):
        namespace, cells = self.helpers()
        namespace["recovery_child"] = Mock()
        with patch.object(Path, "is_mount", return_value=False):
            with self.assertRaises(FileNotFoundError):
                namespace["prepare_backup"]("audit", False)
        namespace["recovery_child"].assert_not_called()
        with patch("subprocess.run") as child, redirect_stdout(io.StringIO()):
            exec(cells["metadata-preflight"], {})
            exec(cells["restore-audit"], {})
            exec(cells["payload-audit"], {})
            exec(cells["recover-reports"], {})
            child.assert_not_called()
        self.assertEqual(TIMEOUTS, {"preflight": 7230, "audit": 21630, "recover": 600})
        self.assertEqual(BACKUP_TIMEOUTS, {"prepare": 60, "package": 630, "publish": 630})
