"""로컬 CPU smoke와 중단/회수 경로. CUDA 실훈련을 실행하지 않는다."""
import copy
import contextlib
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import types
from unittest.mock import patch
import zipfile

import torch

from rewind_m1.data import synthetic_document
from rewind_m1.gpu_pilot import Budget, PilotBudgetStop, run_pilot, verify_m1_export
from rewind_m1.gpu_pilot_recovery import NAME, recover_bundle
from rewind_m1.models import make_model
from rewind_m1.shards import ShardDataset, build_t1_catalog
from rewind_m1.shard_cli import export_checkpoint
from rewind_m1.shard_training import FORMAT
from rewind_m1.training import rng_state, save_checkpoint, source_identity

ROOT=Path(__file__).resolve().parents[1]


class PilotTests(unittest.TestCase):
    def setUp(self):
        parent=ROOT/'outputs'/'test_tmp'; parent.mkdir(parents=True,exist_ok=True)
        self.temp=tempfile.TemporaryDirectory(dir=parent); self.addCleanup(self.temp.cleanup)
        self.base=Path(self.temp.name); self.revision=source_identity()['revision']

    def test_cpu_smoke_complete_and_recovery_does_not_train(self):
        root=self.base/'cpu'
        report=run_pilot(root,profile='cpu_smoke',job_id='cpu_smoke',expected_revision=self.revision)
        self.assertEqual(report['status'],'passed',report.get('failure'))
        self.assertEqual(report['limits']['attempts'],48); self.assertEqual(report['limits']['seconds'],120.)
        self.assertFalse(report['gpu_executed']); self.assertFalse(report['gpu_training_attempted'])
        self.assertEqual(report['resources']['successful_updates'],16)
        self.assertTrue(report['results']['fp32']['rgb']['full_state_equal'])
        self.assertTrue(report['results']['fp32']['m1']['full_state_equal'])
        before=(root/'report.json').read_bytes()
        with patch('rewind_m1.rgb_training.RGBTrainer.train_step',side_effect=AssertionError('recovery must not train')):
            bundle=recover_bundle(root); self.assertEqual(bundle,recover_bundle(root))
        with zipfile.ZipFile(bundle['path']) as archive:
            self.assertIn('report.json',archive.namelist())
            self.assertIn('fp32/rgb/resumed.pt',archive.namelist())
            self.assertIn('fp32/features/joined_0.json',archive.namelist())
        with self.assertRaises(FileExistsError):
            run_pilot(root,profile='cpu_smoke',job_id='cpu_smoke',expected_revision=self.revision)
        self.assertEqual(before,(root/'report.json').read_bytes())

    def test_budget_stop_preserves_last_durable_and_can_recover(self):
        original=Budget.check
        def limited(budget):
            original(budget)
            if budget.attempts>=1: raise PilotBudgetStop('injected_global_budget_stop')
        root=self.base/'bounded'
        with patch.object(Budget,'check',limited):
            report=run_pilot(root,profile='cpu_smoke',job_id='bounded',expected_revision=self.revision)
        self.assertEqual(report['status'],'bounded_stop')
        self.assertEqual(report['resources']['attempts'],1)
        self.assertTrue(report['last_durable'])
        self.assertTrue(all(r['step']==0 for r in report['last_durable'].values()))
        self.assertTrue(Path(recover_bundle(root)['path']).is_file())

    def test_gpu_unavailable_never_falls_back_or_marks_training(self):
        fake_root=self.base/'packaged'; fake_root.mkdir(); (fake_root/'SOURCE_REVISION.txt').write_text(self.revision)
        with patch('rewind_m1.gpu_pilot.ROOT',fake_root),patch('torch.cuda.is_available',return_value=False):
            report=run_pilot(self.base/'no_gpu',profile='gpu',job_id='no_gpu',expected_revision=self.revision,source_zip_sha256='a'*64)
        self.assertEqual(report['status'],'failed')
        self.assertEqual(report['failure']['reason'],'pilot_cuda_required_no_cpu_fallback')
        self.assertEqual(report['device'],'cuda')
        self.assertFalse(report['gpu_available']); self.assertFalse(report['gpu_training_attempted']); self.assertFalse(report['gpu_executed'])
        self.assertFalse((self.base/'no_gpu/fixture').exists())

    def test_amp_metadata_export_uses_cpu_model_not_amp_trainer(self):
        doc=synthetic_document(); path=self.base/'features.json'; path.write_text(json.dumps(doc),encoding='utf8')
        catalog=build_t1_catalog(['features.json'],self.base)
        config={'model':'single_frame','window_size':1,'hidden':8,'dropout':.1,'batch_size':2,'learning_rate':.005,
                'seed':37,'target_scale_mm':100.,'amp':True,'scheduler_step':2,'scheduler_gamma':.8}
        model=make_model(config,28,2)
        state={'format_version':FORMAT,'training_scope':'synthetic_fixture_only','full_training_ready':False,
               'config':config,'catalog':catalog,'model':model.state_dict(),'source':source_identity(),'step':4}
        checkpoint=self.base/'mock_amp.pt'; save_checkpoint(state,checkpoint)
        identity=hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        rng=rng_state(); export_checkpoint(state,identity,self.base/'weights.pt')
        data=ShardDataset(catalog,self.base).load(0)[1]['train']
        # 실제AMP/GPU state가 아닌 명시 mock metadata의 export 경계 검사다.
        self.assertTrue(verify_m1_export(state,identity,self.base/'weights.pt',catalog,data.features,data.time_valid))
        current=rng_state(); self.assertTrue(torch.equal(rng['torch_cpu'],current['torch_cpu'])); self.assertEqual(rng['python'],current['python'])

    def test_failure_reporting_does_not_mask_original_failure(self):
        from rewind_m1.gpu_pilot import save_report as original
        def writer(report,path):
            if isinstance(report,dict) and report.get('status') in ('failed','bounded_stop'): raise OSError('injected_report_write')
            return original(report,path)
        with patch('rewind_m1.gpu_pilot.save_report',writer):
            report=run_pilot(self.base/'bad_source',profile='cpu_smoke',job_id='bad_source',expected_revision='0'*40)
        self.assertEqual(report['failure']['reason'],'pilot_source_revision_mismatch')
        self.assertIn('OSError',report['report_write_errors'])

    def test_source_metadata_only_recovery_and_no_incomplete_zip_reuse(self):
        root=self.base/'source_failure'; root.mkdir()
        (root/'source_upload.json').write_text(json.dumps({'status':'failed'}))
        source=root/'source'; source.mkdir(); (source/'never_include.py').write_text('ignored')
        result=recover_bundle(root)
        with zipfile.ZipFile(result['path']) as archive: self.assertEqual(archive.namelist(),['source_upload.json'])
        incomplete=self.base/'incomplete'; incomplete.mkdir(); (incomplete/'source_upload.json').write_text('{}')
        with zipfile.ZipFile(incomplete/NAME,'x') as archive: archive.writestr('partial.json','{}')
        with self.assertRaisesRegex(ValueError,'pilot_recovery_zip_without_completion_receipt'): recover_bundle(incomplete)

    def test_recovery_module_imports_no_training_or_torch(self):
        code="import sys; import rewind_m1.gpu_pilot_recovery; assert 'torch' not in sys.modules and 'rewind_m1.gpu_pilot' not in sys.modules; print('stdlib_only')"
        result=subprocess.run([sys.executable,'-X','utf8','-c',code],cwd=ROOT,capture_output=True,text=True,check=True)
        self.assertEqual(result.stdout.strip(),'stdlib_only')

    def test_notebook_four_code_cells_compile_and_have_no_outputs(self):
        notebook=json.loads((ROOT/'notebooks/synthetic_gpu_pilot.ipynb').read_bytes())
        cells=[c for c in notebook['cells'] if c['cell_type']=='code']; self.assertEqual(len(cells),4)
        for i,cell in enumerate(cells):
            text=''.join(cell['source']); compile(text,f'cell_{i}','exec')
            self.assertIsNone(cell['execution_count']); self.assertEqual(cell['outputs'],[])
        recovery=''.join(cells[-1]['source'])
        self.assertNotIn('gpu_pilot import',recovery); self.assertNotIn('Popen',recovery)
        self.assertIn('process.kill()', ''.join(cells[2]['source']))
        self.assertIn('EXPECTED_REVISION = ""',''.join(cells[0]['source']))

    def test_notebook_environment_cuda_probe_failure_persists_false(self):
        notebook=json.loads((ROOT/'notebooks/synthetic_gpu_pilot.ipynb').read_bytes())
        code=''.join(notebook['cells'][2]['source'])
        fake=types.ModuleType('torch'); fake.__spec__=torch.__spec__; fake.__version__=str(torch.__version__); fake.version=torch.version
        def fail(*args): raise RuntimeError('injected_cuda_probe')
        fake.cuda=types.SimpleNamespace(is_available=lambda:True,get_device_properties=fail)
        namespace={'SOURCE_OK':True,'PROJECT_ROOT':self.base,'PILOT_SESSION_ROOT':self.base,'sys':sys,
                   're':__import__('re'),'json':json}
        with patch.dict(sys.modules,{'torch':fake}),contextlib.redirect_stdout(io.StringIO()): exec(code,namespace)
        report=json.loads((self.base/'environment.json').read_bytes())
        self.assertFalse(namespace['ENV_READY']); self.assertFalse(report['ready'])
        self.assertEqual(report['error']['reason'],'injected_cuda_probe')

    def test_notebook_explicit_recovery_root_wins_and_never_loads_model(self):
        notebook=json.loads((ROOT/'notebooks/synthetic_gpu_pilot.ipynb').read_bytes())
        root=self.base/'explicit_recovery'; root.mkdir(); (root/'source_upload.json').write_text('{"status":"failed"}')
        code=''.join(notebook['cells'][4]['source']).replace('RECOVERY_ROOT = ""','RECOVERY_ROOT = '+repr(str(root)),1)
        calls=[]; google=types.ModuleType('google'); colab=types.ModuleType('google.colab')
        colab.files=types.SimpleNamespace(download=calls.append)
        with patch.dict(sys.modules,{'google':google,'google.colab':colab}),contextlib.redirect_stdout(io.StringIO()):
            exec(code,{'PILOT_SESSION_ROOT':self.base/'wrong_root'})
        self.assertEqual(calls,[str(root/'source_failure_result.zip')])

    def test_notebook_rejects_sample_zip_and_keeps_upload_failure_receipt(self):
        notebook=json.loads((ROOT/'notebooks/synthetic_gpu_pilot.ipynb').read_bytes())
        buffer=io.BytesIO()
        with zipfile.ZipFile(buffer,'w') as archive: archive.writestr('sample.png',b'not_source')
        raw=buffer.getvalue()
        code=''.join(notebook['cells'][1]['source'])
        code=code.replace('EXPECTED_REVISION = ""','EXPECTED_REVISION = '+repr('c'*40),1)
        code=code.replace('EXPECTED_SOURCE_SHA256 = ""','EXPECTED_SOURCE_SHA256 = '+repr(hashlib.sha256(raw).hexdigest()),1)
        code=code.replace('JOB_ID = ""','JOB_ID = "sample_rejection"',1)
        code=code.replace('Path("/content")','Path('+repr(str(self.base))+')',1)
        google=types.ModuleType('google'); colab=types.ModuleType('google.colab')
        colab.files=types.SimpleNamespace(upload=lambda:{'sample.zip':raw})
        namespace={}
        with patch.dict(sys.modules,{'google':google,'google.colab':colab}),contextlib.redirect_stdout(io.StringIO()): exec(code,namespace)
        report=json.loads((self.base/'rewind_gpu_pilot_sample_rejection/source_upload.json').read_bytes())
        self.assertFalse(namespace['SOURCE_OK']); self.assertEqual(report['status'],'failed')
        self.assertEqual(report['error_type'],'ValueError')

    def test_notebook_interrupt_kills_child_and_preserves_supervision(self):
        notebook=json.loads((ROOT/'notebooks/synthetic_gpu_pilot.ipynb').read_bytes())
        code=''.join(notebook['cells'][3]['source'])
        root=self.base/'supervisor'; root.mkdir()
        class Child:
            pid=987654
            returncode=None
            stdout=io.BytesIO(); stderr=io.BytesIO()
            def poll(self): return self.returncode
            def kill(self): self.returncode=-9
            def wait(self,timeout=None): return self.returncode
        child=Child(); subprocess_fake=types.ModuleType('subprocess'); subprocess_fake.PIPE=object()
        subprocess_fake.Popen=lambda *a,**k:child
        psutil_fake=types.SimpleNamespace(Process=lambda pid:types.SimpleNamespace(memory_info=lambda:types.SimpleNamespace(rss=1024)),
                                          NoSuchProcess=ProcessLookupError)
        namespace={'PILOT_SESSION_ROOT':root,'ENV_READY':True,'JOB_ID':'fake_supervisor','PROJECT_ROOT':self.base,
                   'EXPECTED_REVISION':self.revision,'EXPECTED_SOURCE_SHA256':'a'*64,'sys':sys,'os':__import__('os'),
                   'json':json,'psutil':psutil_fake}
        with patch.dict(sys.modules,{'subprocess':subprocess_fake}),patch('time.sleep',side_effect=KeyboardInterrupt),contextlib.redirect_stdout(io.StringIO()):
            exec(code,namespace)
        report=json.loads((root/'supervision.json').read_bytes())
        self.assertEqual(report['status'],'interrupted'); self.assertFalse(report['child_alive'])
        self.assertEqual(child.returncode,-9)


if __name__=='__main__': unittest.main()
