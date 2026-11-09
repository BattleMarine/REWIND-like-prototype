"""Exercise the launch cell emitted by the v0 Colab notebook builder."""

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import zipfile

from rewind_archive.contracts import canonical
from scripts import archvis_pilot as pilot, build_archvis_v0 as builder
from scripts.archvis_pilot_restore import (
    _restore_read, _restore_require, _restore_seal, _restore_sha,
)


class ArchVisV0NotebookTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1])
        self.addCleanup(directory.cleanup)
        self.work = Path(directory.name)
        restore = self.work / "restore"
        restore.mkdir()
        original = canonical({"cells": [{"id": "bootstrap",
            "source": ['WORK = Path("/content/archvis_1000_pilot_restored")\n']}]})
        (restore / "archvis_1000_pilot_restore_001.ipynb").write_bytes(original)
        prepared = restore / "archvis_1000_prepared_001.zip"
        with zipfile.ZipFile(prepared, "w") as archive:
            archive.writestr("manifest/manifest.json", canonical({
                "content_sha256": builder.MANIFEST,
                "selection": {"validation_keys_sha256": builder.VALIDATION},
                "records": [],
            }))
        output = self.work / "built"
        with patch.multiple(builder, RESTORE_NOTEBOOK=_restore_sha(original),
                            PREPARED_ZIP=_restore_sha(prepared.read_bytes()), WALL_SECONDS=1):
            builder.build(restore, output)
        plan_file = output / "v0_plan.json"
        plan = json.loads(plan_file.read_bytes())
        (self.work / plan_file.name).write_bytes(plan_file.read_bytes())
        with zipfile.ZipFile(output / "archvis_v0_source_overlay.zip") as archive:
            archive.extractall(self.work / "source")
        notebook = json.loads((output / "archvis_1000_v0.ipynb").read_bytes())
        cell = next("".join(c["source"]) for c in notebook["cells"] if c["id"] == "train")
        self.code = cell.replace('APPROVED_V0_PLAN_SHA256 = ""',
                                 f'APPROVED_V0_PLAN_SHA256 = "{plan["content_sha256"]}"')
        self.namespace = dict(Path=Path, sys=sys, WORK=self.work, V0_PLAN=plan, pilot=pilot,
                              canonical=canonical, read_json=json.loads, _restore_require=_restore_require,
                              _restore_read=_restore_read, _restore_seal=_restore_seal, _restore_sha=_restore_sha,
                              restored={"mount_root": str(self.work)}, RESTORE_FILE_PINS={},
                              PINS={"source": {"revision": builder.BASE_REVISION}},
                              confirm_original_roots=lambda *a, **kw: {"roots": {"rgb": "rgb", "pose": "pose"}})
        resources = dict(cuda_available=True, gpu_total_bytes=16 * 1024**3, host_available_bytes=16 * 1024**3,
                         disk_free_bytes=4 * 1024**3)
        resource_mock = patch.object(pilot, "runtime_resources", return_value=resources)
        resource_mock.start()
        self.addCleanup(resource_mock.stop)

    def run_cpu_child(self, script):
        run = subprocess.run

        def local(command, *, cwd, timeout):
            self.assertEqual(self.receipt()["status"], "started")
            self.assertEqual(command[:3], [sys.executable, "-m", "rewind_m1.archvis_v0"])
            self.assertEqual(timeout, 1)
            return run([sys.executable, "-c", script], cwd=cwd, timeout=timeout)

        with patch.object(subprocess, "run", side_effect=local):
            exec(self.code, self.namespace)

    def receipt(self):
        return json.loads((self.work / "v0_launch.json").read_bytes())

    def assert_failure(self, kind):
        self.assertEqual(self.receipt()["status"], "stopped")
        self.assertEqual(self.receipt()["failure_type"], kind)
        failure = json.loads((self.work / "v0_failure.json").read_bytes())
        self.assertEqual(failure["type"], kind)
        self.assertFalse(list(self.work.glob(".v0-launch-*")))

    def test_success_records_exit_and_keeps_one_launch_gate(self):
        self.run_cpu_child("pass")
        self.assertEqual(self.receipt()["status"], "exited")
        self.assertEqual(self.receipt()["returncode"], 0)
        self.assertFalse((self.work / "v0_failure.json").exists())
        with patch.object(subprocess, "run") as child, self.assertRaisesRegex(ValueError, "v0_one_launch"):
            exec(self.code, self.namespace)
        child.assert_not_called()

    def test_child_failure_keeps_exit_code_and_exception(self):
        with self.assertRaises(subprocess.CalledProcessError) as failure:
            self.run_cpu_child("raise SystemExit(7)")
        self.assertEqual(failure.exception.returncode, 7)
        self.assertEqual(self.receipt()["returncode"], 7)
        self.assert_failure("CalledProcessError")

    def test_timeout_keeps_original_exception(self):
        with self.assertRaises(subprocess.TimeoutExpired):
            self.run_cpu_child("import time; time.sleep(5)")
        self.assert_failure("TimeoutExpired")

    def test_keyboard_interrupt_keeps_original_exception(self):
        with (
            patch.object(subprocess, "run", side_effect=KeyboardInterrupt("stop")),
            self.assertRaisesRegex(KeyboardInterrupt, "stop"),
        ):
            exec(self.code, self.namespace)
        self.assert_failure("KeyboardInterrupt")

    def test_failed_diagnostics_do_not_mask_child_error(self):
        original_write = pilot._write

        def write(path, value):
            if path.name == "v0_failure.json":
                raise OSError("failure report unavailable")
            return original_write(path, value)

        child_error = subprocess.CalledProcessError(7, ["test"])
        with (
            patch.object(subprocess, "run", side_effect=child_error),
            patch.object(os, "replace", side_effect=OSError("receipt replace unavailable")),
            patch.object(pilot, "_write", side_effect=write),
            self.assertRaises(subprocess.CalledProcessError) as failure,
        ):
            exec(self.code, self.namespace)
        self.assertIs(failure.exception, child_error)
        self.assertEqual(self.receipt()["status"], "started")
        self.assertFalse(list(self.work.glob(".v0-launch-*")))
