"""실제 pair notebook의 준비/회수 셀과 인증 factory를 원격 없이 실행한다."""
import ast
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

import test_discovery_notebook_upload as upload_tests
import test_discovery_notebook_session as session_tests

ROOT = Path(__file__).resolve().parents[1]
NB = json.loads((ROOT/'notebooks/rgb102_pairs.ipynb').read_bytes())
PREP, PREFLIGHT, RUN, RECOVERY = (''.join(c['source']) for c in NB['cells'][2:])


class PairNotebookTests(unittest.TestCase):
    def setUp(self):
        self.helper = upload_tests.NotebookUploadTests()
        self.helper.setUp()
        self.addCleanup(self.helper.doCleanups)
        self.context = self.helper.context()
        config = {'fixture': 'fixed_pair_config'}
        self.helper.input = upload_tests.archive({'config.json': json.dumps(config)})
        self.context.update(INPUT_ZIP_SHA256=upload_tests.sha(self.helper.input),
            CONFIG_SHA256=upload_tests.sha(json.dumps(config, ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode()),
            RESULT_ZIP=str(self.helper.base/'result.zip'))

    def execute(self, code, uploads=()):
        psutil = ModuleType('psutil')
        psutil.virtual_memory = lambda: SimpleNamespace(available=4*1024**3)
        with patch.dict(sys.modules, {'psutil': psutil}), \
             patch('shutil.disk_usage', return_value=SimpleNamespace(free=4*1024**3)):
            self.helper.execute(code, self.context, uploads)

    def test_two_zip_normal_exact_reuse_and_input_missing_no_mutation(self):
        with self.assertRaisesRegex(AssertionError, 'input ZIP'):
            self.execute(PREP, [{'s': self.helper.source}, {}])
        self.assertFalse(Path(self.context['SOURCE_ROOT']).exists())
        self.execute(PREP, [{'s': self.helper.source}, {'i': self.helper.input}])
        stamp = Path(self.context['SOURCE_ROOT'])/'SOURCE_REVISION.txt'
        before = stamp.stat().st_mtime_ns
        self.execute(PREP, [{'s': self.helper.source}, {'i': self.helper.input}])
        self.assertEqual(stamp.stat().st_mtime_ns, before)
        self.assertFalse(Path(self.context['WORKSPACE']).exists())

    def test_stale_modules_before_auth_and_preparation_recovery(self):
        stale = ModuleType('rewind_discovery')
        stale.__file__ = str(self.helper.base/'previous/rewind_discovery/__init__.py')
        with patch.dict(sys.modules, {'rewind_discovery': stale}):
            with self.assertRaisesRegex(AssertionError, '새 CPU 런타임'):
                self.execute(PREFLIGHT)
        self.execute(RECOVERY)
        self.assertEqual(self.helper.downloads, [self.context['RECOVERY_ZIP']])
        self.assertTrue(Path(self.context['RECOVERY_ZIP']).is_file())
        self.execute(RECOVERY)

    def test_ast_and_reviewed_auth_factory_same(self):
        for cell in NB['cells']:
            if cell['cell_type'] == 'code':
                ast.parse(''.join(cell['source']))
        rgb = json.loads((ROOT/'notebooks/archvis_rgb_discovery.ipynb').read_bytes())
        def factory(code):
            return next(n for n in ast.parse(code).body if isinstance(n, ast.FunctionDef) and n.name == 'session_factory')
        self.assertEqual(ast.dump(factory(RUN)), ast.dump(factory(''.join(rgb['cells'][3]['source']))))
        self.assertNotIn('authenticate_user', RECOVERY)
        self.assertNotIn('session_factory', RECOVERY)


@unittest.skipUnless(session_tests.available(), '보존 transport_deps 필요')
class PairNotebookSessionTests(session_tests.NotebookSessionTests):
    def factory(self, env):
        node = next(n for n in ast.parse(RUN).body if isinstance(n, ast.FunctionDef) and n.name == 'session_factory')
        exec(compile(ast.Module(body=[node], type_ignores=[]), 'actual_pair_factory', 'exec'), env)
        return env['session_factory']
