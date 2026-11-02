"""실제 notebook 준비/회수와 인증 factory를 원격 없이 검사한다."""
import ast
import json
from pathlib import Path
import sys
from types import ModuleType,SimpleNamespace
import unittest
from unittest.mock import patch
import test_discovery_notebook_upload as upload
import test_discovery_notebook_session as auth

ROOT=Path(__file__).resolve().parents[1]
NB=json.loads((ROOT/'notebooks/archvis_rgb_preindex.ipynb').read_bytes())
PREP,PREFLIGHT,RUN,RECOVERY=(''.join(c['source']) for c in NB['cells'][2:])


class PreindexNotebookTests(unittest.TestCase):
    def setUp(self):
        self.helper=upload.NotebookUploadTests();self.helper.setUp();self.addCleanup(self.helper.doCleanups)
        self.context=self.helper.context()

    def execute(self,code,uploads=()):
        psutil=ModuleType('psutil');psutil.virtual_memory=lambda:SimpleNamespace(available=4*1024**3)
        with patch.dict(sys.modules,{'psutil':psutil}),patch('shutil.disk_usage',return_value=SimpleNamespace(free=4*1024**3)):
            self.helper.execute(code,self.context,uploads)

    def test_two_zip_missing_normal_and_exact_reuse(self):
        with self.assertRaisesRegex(AssertionError,'input ZIP'):
            self.execute(PREP,[{'s':self.helper.source},{}])
        self.assertFalse(Path(self.context['SOURCE_ROOT']).exists())
        self.execute(PREP,[{'s':self.helper.source},{'i':self.helper.input}])
        stamp=Path(self.context['SOURCE_ROOT'])/'SOURCE_REVISION.txt';before=stamp.stat().st_mtime_ns
        self.execute(PREP,[{'s':self.helper.source},{'i':self.helper.input}])
        self.assertEqual(stamp.stat().st_mtime_ns,before)
        self.assertFalse(Path(self.context['WORKSPACE']).exists())

    def test_stale_import_blocks_and_preparation_recovery_without_source(self):
        stale=ModuleType('rewind_discovery');stale.__file__=str(self.helper.base/'old/rewind_discovery/__init__.py')
        with patch.dict(sys.modules,{'rewind_discovery':stale}),self.assertRaisesRegex(AssertionError,'새 CPU 런타임'):
            self.execute(PREFLIGHT)
        self.execute(RECOVERY)
        self.assertEqual(len(self.helper.downloads),1)
        self.assertTrue(Path(self.helper.downloads[0]).is_file())
        self.execute(RECOVERY)

    def test_ast_and_actual_auth_factory_same_as_reviewed_rgb(self):
        for c in NB['cells']:
            if c['cell_type']=='code':ast.parse(''.join(c['source']))
        rgb=json.loads((ROOT/'notebooks/archvis_rgb_discovery.ipynb').read_bytes())
        def factory(s):return next(n for n in ast.parse(s).body if isinstance(n,ast.FunctionDef) and n.name=='session_factory')
        self.assertEqual(ast.dump(factory(RUN)),ast.dump(factory(''.join(rgb['cells'][3]['source']))))
        self.assertNotIn('authenticate_user',RECOVERY)
        self.assertIn('disk_limit=768*1024**2',RECOVERY)


@unittest.skipUnless(auth.available(),'보존 transport_deps 필요')
class PreindexNotebookSessionTests(auth.NotebookSessionTests):
    def factory(self,env):
        node=next(n for n in ast.parse(RUN).body if isinstance(n,ast.FunctionDef) and n.name=='session_factory')
        exec(compile(ast.Module(body=[node],type_ignores=[]),'actual_preindex_factory','exec'),env)
        return env['session_factory']


class PreindexInputTests(unittest.TestCase):
    @unittest.skipUnless((ROOT/'outputs/preindex_20261103/developer/inputs/config.json').exists(),'실제 고정 입력 필요')
    def test_actual_fixed_input_preflight_and_pin_rejection(self):
        from rewind_discovery.preindex import preflight
        inputs=ROOT/'outputs/preindex_20261103/developer/inputs'
        config,args=preflight(inputs,'1a94db146a28aeddac51fc7cf59e82661358ff15ed3158aeda422de5d0ea7d44')
        self.assertEqual(config['baseline']['keys']['stereo_pose_observed']['train'],102)
        self.assertEqual(args['resume']['anchor']['generation'],9)
        with self.assertRaisesRegex(Exception,'preindex_config_pin_or_profile'):
            preflight(inputs,'0'*64)
