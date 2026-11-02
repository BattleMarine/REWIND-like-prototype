"""실제 pose67 notebook 셀을 작은 ZIP/가짜 Colab으로 검증한다. 원격 호출 없음."""
import ast
from contextlib import redirect_stdout
import hashlib
import io
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch
from zipfile import ZipFile, ZIP_STORED

import test_discovery_notebook_session as session_tests
import test_pose67 as pose_tests

ROOT = Path(__file__).resolve().parents[1]
NB = json.loads((ROOT/'notebooks/pose_missing67.ipynb').read_bytes())
PREP, PREFLIGHT, RUN, RECOVERY = (''.join(c['source']) for c in NB['cells'][2:])


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def archive(rows):
    stream = io.BytesIO()
    with ZipFile(stream, 'w', ZIP_STORED) as z:
        for name, data in rows.items():
            z.writestr(name, data)
    return stream.getvalue()


class Pose67NotebookTests(unittest.TestCase):
    def setUp(self):
        parent = ROOT/'outputs/test_tmp'
        parent.mkdir(parents=True, exist_ok=True)
        tmp = tempfile.TemporaryDirectory(dir=parent)
        self.addCleanup(tmp.cleanup)
        self.base = Path(tmp.name)
        self.revision = 'a'*40
        self.source = archive({'SOURCE_REVISION.txt': self.revision+'\n', 'package/marker.py': 'value=1\n'})
        self.targets = ['fixture_'+str(i) for i in range(67)]
        self.target_sha = sha(json.dumps(self.targets, ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode())
        prefix = b'fixture_prefix'
        self.expected = {'manifest_sha256': 'b'*64, 'prefix_sha256': sha(prefix), 'prefix_bytes': len(prefix)}
        self.cache = archive({'prefix.bin': prefix, 'manifest.json': json.dumps({
            'content_sha256': self.expected['manifest_sha256'], 'prefix_sha256': sha(prefix), 'prefix_bytes': len(prefix)})})
        self.input = archive({'targets.json': json.dumps(self.targets), 'config.json': json.dumps({
            'job': {'exact_targets': self.targets, 'includes': self.targets, 'cache_expectation': self.expected},
            'targets_content_sha256': self.target_sha, 'cache_zip_sha256': sha(self.cache), 'cache_zip_bytes': len(self.cache)})})
        old_path, old_bytecode = list(sys.path), sys.dont_write_bytecode
        self.addCleanup(lambda: sys.path.__setitem__(slice(None), old_path))
        self.addCleanup(setattr, sys, 'dont_write_bytecode', old_bytecode)

    def context(self):
        return dict(SOURCE_ZIP_SHA256=sha(self.source), INPUT_ZIP_SHA256=sha(self.input),
            CACHE_ZIP_SHA256=sha(self.cache), TARGETS_SHA256=self.target_sha,
            SOURCE_REVISION=self.revision, NOTEBOOK_REVISION='c'*40, JOB_ID='fixture67',
            SOURCE_ROOT=str(self.base/'source'), INPUT_ROOT=str(self.base/'inputs'),
            CACHE_INPUT=str(self.base/'warm.zip'), WORKSPACE=str(self.base/'work'),
            RECOVERY_ZIP=str(self.base/'fixture_preparation_result.zip'))

    def execute(self, code, context, uploads=()):
        colab = ModuleType('google.colab')
        uploads = iter(uploads)
        self.downloads, self.uploads = [], 0
        def upload():
            self.uploads += 1
            return next(uploads)
        colab.files = SimpleNamespace(upload=upload, download=self.downloads.append)
        psutil = ModuleType('psutil')
        psutil.virtual_memory = lambda: SimpleNamespace(available=4*1024**3)
        with patch.dict(sys.modules, {'google.colab': colab, 'psutil': psutil}), \
             patch.object(shutil, 'disk_usage', return_value=SimpleNamespace(free=4*1024**3)), \
             patch.object(subprocess, 'check_call', side_effect=AssertionError('no installation')):
            output = io.StringIO()
            with redirect_stdout(output):
                exec(compile(code, 'actual_pose67_cell', 'exec'), context)
            self.stdout = output.getvalue()

    def normal(self, context):
        self.execute(PREP, context, [{'s.zip': self.source}, {'i.zip': self.input}, {'c.zip': self.cache}])

    def test_three_uploads_and_exact_reuse_without_install_or_work(self):
        context = self.context()
        self.normal(context)
        self.assertEqual(self.uploads, 3)
        cache = Path(context['CACHE_INPUT'])
        before = cache.stat().st_mtime_ns
        self.assertEqual(cache.read_bytes(), self.cache)
        self.normal(context)
        self.assertEqual(cache.stat().st_mtime_ns, before)
        self.assertFalse(Path(context['WORKSPACE']).exists())

    def test_missing_or_changed_warm_cache_never_creates_roots(self):
        for supplied in ({}, {'bad.zip': b'corrupt'}):
            context = self.context()
            with self.assertRaisesRegex(AssertionError, 'cache ZIP'):
                self.execute(PREP, context, [{'s': self.source}, {'i': self.input}, supplied])
            for key in ('SOURCE_ROOT', 'INPUT_ROOT', 'CACHE_INPUT', 'WORKSPACE'):
                self.assertFalse(Path(context[key]).exists())
        cache = Path(context['CACHE_INPUT'])
        cache.write_bytes(b'changed')
        with self.assertRaises(AssertionError):
            self.normal(context)
        self.assertEqual(cache.read_bytes(), b'changed')
        self.assertFalse(Path(context['SOURCE_ROOT']).exists())

    def test_stale_import_blocks_before_input_or_auth(self):
        context = self.context()
        stale = ModuleType('rewind_archive')
        stale.__file__ = str(self.base/'previous/rewind_archive/__init__.py')
        with patch.dict(sys.modules, {'rewind_archive': stale}):
            with self.assertRaisesRegex(AssertionError, '새 CPU 런타임'):
                self.execute(PREFLIGHT, context)
        self.assertFalse(Path(context['WORKSPACE']).exists())

    def test_preparation_failure_recovery_needs_no_source_inputs_or_auth(self):
        context = self.context()
        self.execute(RECOVERY, context)
        result = Path(context['RECOVERY_ZIP'])
        self.assertEqual(self.downloads, [str(result)])
        self.assertIn(sha(result.read_bytes()), self.stdout)
        with ZipFile(result) as z:
            self.assertIsNone(z.testzip())
            self.assertFalse(json.loads(z.read('recovery.json'))['train_ready'])
        self.execute(RECOVERY, context)
        self.assertFalse(Path(context['SOURCE_ROOT']).exists())

    def test_completed_workflow_recovery_without_new_remote_requests(self):
        from rewind_archive.workflow import make_preflight, run_workflow_job
        helper = pose_tests.Pose67Tests()
        helper.setUp()
        self.addCleanup(helper.doCleanups)
        helper.fixture()
        context = self.context()
        context['WORKSPACE'] = str(helper.base/'work')
        context['JOB_ID'] = helper.config['job_id']
        source = Path(context['SOURCE_ROOT'])
        source.mkdir()
        (source/'SOURCE_REVISION.txt').write_text(self.revision)
        root, record = run_workflow_job(helper.config, make_preflight(helper.config, {}), {},
                                       Path(context['WORKSPACE'])/'outputs', lambda: helper.session)
        calls_before = len(helper.session.calls)
        self.execute(RECOVERY, context)
        self.assertEqual(len(helper.session.calls), calls_before)
        self.assertEqual(len(self.downloads), 2)
        with ZipFile(self.downloads[0]) as z:
            self.assertIsNone(z.testzip())
            review = json.loads(z.read(root.name+'/pilot/pose67_review.json'))
            self.assertTrue(review['selected_json_verification']['selected_targets_passed'])
        self.assertTrue(Path(self.downloads[1]).is_file())

    def test_all_cells_parse_and_auth_factory_matches_reviewed_rgb(self):
        for cell in NB['cells']:
            if cell['cell_type'] == 'code':
                ast.parse(''.join(cell['source']))
        rgb = json.loads((ROOT/'notebooks/archvis_rgb_discovery.ipynb').read_bytes())
        def factory(code):
            return next(n for n in ast.parse(code).body if isinstance(n, ast.FunctionDef) and n.name == 'session_factory')
        self.assertEqual(ast.dump(factory(RUN)), ast.dump(factory(''.join(rgb['cells'][3]['source']))))
        self.assertNotIn('authenticate_user', RECOVERY)
        self.assertNotIn('run_workflow_job', RECOVERY)


@unittest.skipUnless(session_tests.available(), '보존 transport_deps 환경 필요')
class Pose67NotebookSessionTests(session_tests.NotebookSessionTests):
    def factory(self, env):
        node = next(n for n in ast.parse(RUN).body if isinstance(n, ast.FunctionDef) and n.name == 'session_factory')
        exec(compile(ast.Module(body=[node], type_ignores=[]), 'actual_pose67_factory', 'exec'), env)
        return env['session_factory']
