"""독립 fixture/worker와 notebook의 로컬 모의 입출력 검사."""
import ast
import hashlib
import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
import types
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

try:
    import torch
    from PIL import Image
except ImportError:
    torch = None

if torch is not None:
    import rewind_m1.diagnostic_smoke as worker
    from rewind_m1.local_fixture import create_fixture
    from rewind_m1.local_rgb import LocalStereoRGB
    from rewind_m1.training import source_identity


@unittest.skipUnless(torch is not None,"torch와 Pillow가 필요합니다")
class SmokeTests(unittest.TestCase):
    def setUp(self):
        root = Path(__file__).resolve().parents[1]/"outputs"/"test_tmp"
        root.mkdir(parents=True,exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=root)
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.project = Path(__file__).resolve().parents[1]

    def test_worker_complete_without_test_fixtures(self):
        revision = source_identity()["revision"]
        source = self.root/"source"; source.mkdir()
        for package in ("rewind_m1","rewind_data","rewind_archive"):
            shutil.copytree(self.project/package,source/package,ignore=shutil.ignore_patterns("__pycache__"))
        (source/"SOURCE_REVISION.txt").write_text(revision,encoding="utf-8")
        result = subprocess.run([sys.executable,"-B","-m","rewind_m1.diagnostic_smoke","--output",str(self.root/"run"),
            "--expected-revision",revision],cwd=source,capture_output=True,text=True,timeout=180,
            env={**os.environ,"OMP_NUM_THREADS":"1","MKL_NUM_THREADS":"1","CUDA_VISIBLE_DEVICES":""})
        self.assertEqual(result.returncode,0,result.stderr)
        summary = json.loads((self.root/"run/summary.json").read_text(encoding="utf-8"))
        self.assertEqual(summary["status"],"passed")
        self.assertTrue(all(summary["checks"].values()))
        fixture = self.root/"run/fixture"
        loader = LocalStereoRGB(fixture/"manifest.json",fixture)
        self.assertEqual(len(loader),4)
        self.assertEqual(summary["metrics"]["2d"]["paired_count"],8)
        self.assertEqual(summary["metrics"]["3d"]["valid_joints"],4)
        pose = json.loads((fixture/"features.json").read_text())
        self.assertEqual(len({r["motion_id"] for r in pose["records"]}),2)
        self.assertEqual(len({r["sequence_id"] for r in pose["records"]}),2)
        original = (self.root/"run/summary.json").read_bytes()
        with self.assertRaises(FileExistsError): worker.run_smoke(self.root/"run",revision)
        self.assertEqual((self.root/"run/summary.json").read_bytes(),original)
        for module in ("local_fixture.py","diagnostic_smoke.py"):
            self.assertNotIn("unittest",(self.project/"rewind_m1"/module).read_text(encoding="utf-8"))

    def test_source_failure_and_fixture_existing(self):
        with patch.object(worker,"ROOT",self.root):
            with self.assertRaises(FileNotFoundError): worker.run_smoke(self.root/"bad_source","a"*40)
        summary = json.loads((self.root/"bad_source/summary.json").read_text())
        self.assertEqual(summary["status"],"failed")
        self.assertFalse((self.root/"bad_source/diagnostic").exists())
        receipt = create_fixture(self.root/"fixture")
        self.assertEqual(receipt["data_kind"],"synthetic_fixture")
        with self.assertRaises(FileExistsError): create_fixture(self.root/"fixture")

    def test_notebook_mock_timeout_and_export(self):
        notebook = json.loads((self.project/"notebooks/local_m1_diagnostic.ipynb").read_text(encoding="utf-8"))
        code = ["".join(c["source"]) for c in notebook["cells"] if c["cell_type"] == "code"]
        self.assertEqual(len(code),4)
        for cell in code: ast.parse(cell)
        revision = "a"*40
        source = io.BytesIO()
        with zipfile.ZipFile(source,"w") as archive:
            archive.writestr("rewind_m1/diagnostic_smoke.py","# mock source\n")
            archive.writestr("SOURCE_REVISION.txt",revision)
        raw = source.getvalue()
        downloads = []
        files = types.SimpleNamespace(upload=lambda:{"source.zip":raw},download=downloads.append)
        colab = types.ModuleType("google.colab"); colab.files=files
        google = types.ModuleType("google"); google.colab=colab
        first = code[0].replace('EXPECTED_REVISION = ""',f'EXPECTED_REVISION = "{revision}"').replace(
            'EXPECTED_SOURCE_SHA256 = ""',f'EXPECTED_SOURCE_SHA256 = "{hashlib.sha256(raw).hexdigest()}"').replace(
            'dir="/content"',f'dir={str(self.root)!r}')
        scope = {}
        with patch.dict(sys.modules,{"google":google,"google.colab":colab}):
            exec(compile(first,"source-cell","exec"),scope)
        self.assertTrue(scope["SOURCE_OK"])
        exec(compile(code[1],"environment-cell","exec"),scope)
        class TimeoutProcess:
            def __init__(self,*args,**kwargs):
                self.stdout=io.BytesIO(b"worker start\n"); self.stderr=io.BytesIO(b""); self.returncode=None
            def wait(self,timeout):
                if self.returncode is None: raise subprocess.TimeoutExpired("mock",timeout)
                return self.returncode
            def poll(self): return self.returncode
            def kill(self): self.returncode=-9
        with patch("subprocess.Popen",TimeoutProcess):
            exec(compile(code[2],"worker-cell","exec"),scope)
        self.assertEqual(scope["supervision"]["status"],"timeout")
        exec(compile(code[3],"export-cell","exec"),scope)
        self.assertEqual(len(downloads),1)
        with zipfile.ZipFile(downloads[0]) as bundle:
            report=json.loads(bundle.read("supervision.json"))
            self.assertEqual(report["status"],"timeout")
            self.assertIn("export_report.json",bundle.namelist())
            self.assertFalse(any(name.startswith("source/") for name in bundle.namelist()))
        before = Path(downloads[0]).read_bytes()
        before_report = (scope["M1_RUN_ROOT"]/"export_report.json").read_bytes()
        exec(compile(code[3],"repeat-export-cell","exec"),scope)
        self.assertEqual(Path(downloads[-1]).read_bytes(),before)
        self.assertEqual((scope["M1_RUN_ROOT"]/"export_report.json").read_bytes(),before_report)

        class BrokenPipe(io.BytesIO):
            def read(self,*args): raise OSError("injected log read failure")
        class LogFailureProcess:
            def __init__(self,*args,**kwargs):
                self.stdout=BrokenPipe(); self.stderr=io.BytesIO(); self.returncode=0
                (scope["M1_RUN_ROOT"]/"check").mkdir()
                (scope["M1_RUN_ROOT"]/"check/summary.json").write_text('{"status":"passed"}',encoding="utf-8")
            def wait(self,timeout): return 0
            def poll(self): return 0
            def kill(self): pass
        scope["M1_RUN_ROOT"] = self.root/"log_failure"; scope["M1_RUN_ROOT"].mkdir()
        scope.pop("M1_STARTED")
        with patch("subprocess.Popen",LogFailureProcess):
            exec(compile(code[2],"log-failure-cell","exec"),scope)
        self.assertEqual(scope["supervision"]["status"],"failed")
        self.assertIn("OSError",scope["supervision"]["pump_errors"])

        class CleanupFailureProcess(TimeoutProcess):
            def kill(self): raise OSError("injected cleanup failure")
        scope["M1_RUN_ROOT"] = self.root/"cleanup_failure"; scope["M1_RUN_ROOT"].mkdir()
        scope.pop("M1_STARTED")
        with patch("subprocess.Popen",CleanupFailureProcess):
            exec(compile(code[2],"cleanup-failure-cell","exec"),scope)
        self.assertEqual(scope["supervision"]["status"],"failed")
        self.assertIn("OSError",scope["supervision"]["cleanup_errors"])
        self.assertTrue((scope["M1_RUN_ROOT"]/"supervision.json").exists())

        scope["M1_RUN_ROOT"] = self.root/"partial_export"; scope["M1_RUN_ROOT"].mkdir()
        (scope["M1_RUN_ROOT"]/"large.bin").write_bytes(b"x"*5000)
        exec(compile(code[3].replace("80*1024**2","4096"),"partial-export-cell","exec"),scope)
        self.assertEqual(scope["export_report"]["status"],"partial_export")
        self.assertTrue(scope["export_report"]["excluded"])
        self.assertTrue((scope["M1_RUN_ROOT"]/"large.bin").exists())
        scope["M1_RUN_ROOT"] = self.root/"interrupted_export"; scope["M1_RUN_ROOT"].mkdir()
        (scope["M1_RUN_ROOT"]/"source_upload.json").write_text('{}',encoding="utf-8")
        with patch.object(zipfile.ZipFile,"write",side_effect=KeyboardInterrupt):
            with self.assertRaises(KeyboardInterrupt):
                exec(compile(code[3],"interrupted-export-cell","exec"),scope)
        self.assertFalse((scope["M1_RUN_ROOT"]/"local_m1_diagnostic_result.zip").exists())
        self.assertFalse(list(scope["M1_RUN_ROOT"].glob("*.zip.tmp")))
        self.assertTrue((scope["M1_RUN_ROOT"]/"source_upload.json").exists())
