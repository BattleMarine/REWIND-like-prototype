"""Drive 준비 notebook 셀 모의 실행. 실제 mount/auth/Drive/Colab 요청 없음."""
import ast,json,sys
from pathlib import Path
from types import ModuleType,SimpleNamespace
import unittest
from unittest.mock import patch
import test_pose67_notebook as fixtures
from rewind_archive.profiles import target269_names
from rewind_archive.contracts import digest

ROOT=Path(__file__).resolve().parents[1]
NB=json.loads((ROOT/'notebooks/pose_missing269_drive.ipynb').read_bytes())
PREP,STAGE,PREFLIGHT,RUN,RECOVERY=(''.join(c['source']) for c in NB['cells'][2:])


class DriveNotebookTests(unittest.TestCase):
    def setUp(self):
        self.h=fixtures.Pose67NotebookTests();self.h.setUp();self.addCleanup(self.h.doCleanups)
        h=self.h;h.targets=target269_names();h.target_sha=digest(h.targets)
        h.input=fixtures.archive({'targets.json':json.dumps(h.targets),'config.json':json.dumps({
            'job':{'exact_targets':h.targets,'includes':h.targets,'cache_expectation':h.expected},
            'targets_content_sha256':h.target_sha,'cache_zip_sha256':fixtures.sha(h.cache),'cache_zip_bytes':246522402})})
        self.context=h.context();self.context.update(CACHE_DRIVE_PATH='',CACHE_STAGE_REPORT=str(h.base/'stage.json'))

    def test_two_small_uploads_only_and_cache_path_required_before_auth(self):
        h=self.h
        h.execute(PREP,self.context,[{'s':h.source},{'i':h.input}])
        self.assertEqual(h.uploads,2)
        self.assertFalse(Path(self.context['CACHE_INPUT']).exists())
        with self.assertRaisesRegex(AssertionError,'정확한 Drive 경로'):
            h.execute(STAGE,self.context)
        self.assertFalse(Path(self.context['WORKSPACE']).exists())

    def test_mount_precedes_explicit_stage_and_report_recovery_needs_no_auth(self):
        import rewind_archive.cache_staging as stage
        h=self.h;events=[]
        colab=ModuleType('google.colab')
        colab.auth=SimpleNamespace(authenticate_user=lambda:events.append('auth'))
        colab.drive=SimpleNamespace(mount=lambda p:events.append(('mount',p)))
        drive_path=str(h.base/'drive/explicit-cache.zip')
        self.context.update(SOURCE_ROOT=str(ROOT),CACHE_DRIVE_PATH=drive_path)
        def copy(source,target,**kwargs):
            events.append(('stage',source))
            Path(self.context['CACHE_STAGE_REPORT']).write_text(json.dumps({'identity':{
                'expected_sha256':self.context['CACHE_ZIP_SHA256'],'expected_bytes':246522402,
                'target':str(Path(self.context['CACHE_INPUT']).resolve())},
                'attempts':[{'status':'failed_or_interrupted','partial_preserved':True}]}),encoding='utf8')
            return {'report':{'status':'mocked'}}
        with patch.dict(sys.modules,{'google.colab':colab}),patch('os.path.ismount',return_value=True),patch.object(stage,'stage_pose269_cache',copy):
            exec(compile(STAGE,'actual_drive_stage_cell','exec'),self.context)
        self.assertEqual(events,['auth',('mount','/content/drive'),('stage',drive_path)])
        with self.assertRaises(AssertionError):h.execute(PREFLIGHT,self.context)
        h.execute(RECOVERY,self.context)
        from zipfile import ZipFile
        with ZipFile(h.downloads[0]) as z:
            self.assertIn('cache_stage_report.json',z.namelist())
            self.assertIsNone(z.testzip())
        first=h.downloads[0]
        info=json.loads(Path(self.context['CACHE_STAGE_REPORT']).read_bytes())
        info['attempts'].append({'status':'failed_or_interrupted','partial_bytes':123})
        Path(self.context['CACHE_STAGE_REPORT']).write_text(json.dumps(info),encoding='utf8')
        h.execute(RECOVERY,self.context)
        self.assertNotEqual(h.downloads[0],first)
        self.assertTrue(Path(first).exists())
        with ZipFile(h.downloads[0]) as z:
            self.assertEqual(json.loads(z.read('cache_stage_report.json')),info)

    def test_ast_job_guard_and_auth_factory_preserved(self):
        old=json.loads((ROOT/'notebooks/pose_missing269.ipynb').read_bytes())
        for c in NB['cells']:
            if c['cell_type']=='code':ast.parse(''.join(c['source']))
        def factory(code):return next(n for n in ast.parse(code).body if isinstance(n,ast.FunctionDef) and n.name=='session_factory')
        self.assertEqual(ast.dump(factory(RUN)),ast.dump(factory(''.join(old['cells'][4]['source']))))
        self.assertNotIn('files.upload',STAGE)
        self.assertNotIn('cache_uploaded',PREP)
        self.assertNotIn('authenticate_user',RECOVERY)
        self.assertNotIn('auth.authenticate_user()',RUN)
        config={};exec(''.join(NB['cells'][1]['source']),config)
        previous={};exec(''.join(old['cells'][1]['source']),previous)
        self.assertEqual(config['WORKSPACE'],previous['WORKSPACE'])
        self.assertEqual(config['JOB_ID'],previous['JOB_ID'])
        self.assertEqual(config['CACHE_DRIVE_PATH'],'')
