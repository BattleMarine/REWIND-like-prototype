"""Local fixture checks; no Colab, remote Drive or actual GPU training."""
import ast
from copy import deepcopy
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import tempfile
import sys
from types import SimpleNamespace,ModuleType
from unittest import TestCase
from unittest.mock import patch
import zipfile

import torch

from rewind_archive.contracts import digest
from rewind_m1.training import rng_state
from rewind_prepare.converter import sha
from rewind_v1 import train_clip_bundle as bundle,train_clip_runs as runs
from rewind_v1.train_clip_analysis import Analysis,TraceModel
from rewind_v1.run_data import FixtureData
from rewind_v1.data import Clip
from rewind_v1.teacher import V1Body,V1Hand
from rewind_v1.camera_training import model_fingerprint
from rewind_v1.train_clip_cli import package
from rewind_v1.train_clip_cli import main as cli_main
from rewind_v1.cross_recovery import prepare_attempt,publish
from scripts.build_train_clip_diagnostics import notebook_document

ROOT=Path(__file__).resolve().parents[1]


class TrainClipTests(TestCase):
    def setUp(self):
        torch.set_num_threads(1);torch.manual_seed(71)
        (ROOT/'outputs/test_tmp').mkdir(parents=True,exist_ok=True)
        tmp=tempfile.TemporaryDirectory(dir=ROOT/'outputs/test_tmp');self.addCleanup(tmp.cleanup);self.root=Path(tmp.name)
        self.cfg=bundle.config('v153');self.cfg['teacher'].update(hidden=8,heads=2,graph_layers=1,temporal_layers=1,diffusion_steps=20)
        self.cfg.update(updates_per_model=4,milestones=[0,2,4],checkpoint_every=1,log_every=1)
        self.spec=FixtureData().spec;self.models={}
        for r,ns in (('body',self.spec['body_joint_names']),('hand',['left_tip','right_tip'])):
            edges=[(ns.index(a),ns.index(b)) for a,b in self.spec['edges'] if a in ns and b in ns]
            self.models[r]=(V1Body(self.cfg['teacher'],len(ns),edges) if r=='body' else V1Hand(self.cfg['teacher'],len(ns),edges,2)).eval()
        target=torch.arange(48,dtype=torch.float32).reshape(1,4,4,3)
        valid=torch.ones(1,4,4,dtype=torch.bool)
        self.clip=Clip(torch.randn(1,4,56),torch.zeros(1,4,2,9),target,valid,torch.ones(1,4,dtype=torch.bool),torch.arange(4,dtype=torch.float64)[None]/25,'predicted_2d',input_sha256='c'*64)
        self.inputs={'clip':self.clip,'spec':self.spec,'metadata':{'sequence':'fixture_train','frames':[0,1,2,3]},
            'feature_identity':'c'*64,'anchor':{'fixture':True},'bundle':{'content_sha256':'a'*64,'parent_source_sha256':'b'*64}}

    def trainer(self,name):return runs.PredictedClipTraining(self.inputs,deepcopy(self.cfg),'cpu',self.root/name,fixture_models=deepcopy(self.models))

    def assert_nested(self,a,b):
        if torch.is_tensor(a):self.assertTrue(torch.equal(a,b))
        elif isinstance(a,dict):
            self.assertEqual(a.keys(),b.keys())
            for k in a:self.assert_nested(a[k],b[k])
        elif isinstance(a,(tuple,list)):
            self.assertEqual(len(a),len(b))
            for x,y in zip(a,b):self.assert_nested(x,y)
        else:self.assertEqual(a,b)

    def test_sampler_trace_exact_and_direct_x0_units_mask(self):
        a=Analysis(self.cfg,self.clip,self.spec,self.inputs['metadata'],'cpu')
        for role,c in a.clips.items():
            m=a.condition(self.models,role);observer=TraceModel(m,c,a.edges[role],100)
            x=a.schedule.sample(m,c.features,c.camera,c.time_valid,a.noise[role],10)
            y=a.schedule.sample(observer,c.features,c.camera,c.time_valid,a.noise[role],10)
            self.assertTrue(torch.equal(x,y));self.assertEqual(len(observer.rows),10)
        before={r:sha(n.numpy().tobytes()) for r,n in a.noise.items()}
        a.sample(self.models,1);a.sample(self.models,10)
        self.assertEqual(before,a.noise_pins)
        class Oracle(torch.nn.Module):
            def __init__(self,clean):super().__init__();self.clean=clean;self.eval()
            def forward(self,*args):return self.clean,self.clean
        perfect={r:Oracle(c.target_mm/100) for r,c in a.clips.items()}
        with patch.object(a,'condition',side_effect=lambda models,r:models[r]):
            p,d=a.reconstruct(perfect,20)
        for r in p:self.assertTrue(torch.allclose(p[r],a.clips[r].target_mm,atol=1e-5))
        self.assertGreater(d['alpha_bar'],0);self.assertIn('not pure-noise',d['kind'])
        # Missing wrist removes local target only, not the native global target.
        valid=self.clip.target_valid.clone();valid[0,0,0]=False
        masked=Analysis(self.cfg,replace(self.clip,target_valid=valid),self.spec,self.inputs['metadata'],'cpu')
        m=masked.metrics({r:c.target_mm for r,c in masked.clips.items()})
        self.assertEqual(m['hand']['valid_joints'],7)
        self.assertEqual(m['hand_global_predicted_wrist']['native_global_mask']['valid_joints'],8)

    def test_resume_matches_uninterrupted_all_training_state(self):
        whole=self.trainer('whole');whole.run()
        partial=self.trainer('resumed');partial.run(stop_after=2)
        resumed=self.trainer('resumed');resumed.run(resume=partial.last['sha256'])
        self.assertEqual(resumed.step,4)
        for r in whole.stages:
            for attr in ('model','optimizer','scheduler','scaler'):
                self.assert_nested(getattr(whole.stages[r],attr).state_dict(),getattr(resumed.stages[r],attr).state_dict())
        a=torch.load(whole.root/'last.pt',weights_only=True);b=torch.load(resumed.root/'last.pt',weights_only=True)
        self.assert_nested(a['rng'],b['rng']);self.assertEqual(resumed.initial,partial.initial)
        self.assertGreater(b['save_budget_allowance_seconds'],0)
        self.assertLess(b['elapsed'],b['save_budget_allowance_seconds'])
        self.assertEqual(len(resumed.snapshots),3)

    def test_pair_half_update_and_posthistory_interrupt_rollback(self):
        trainer=self.trainer('half');before=deepcopy({r:s.model.state_dict() for r,s in trainer.stages.items()});rng=deepcopy(rng_state())
        with patch.object(trainer.stages['hand'],'update',side_effect=KeyboardInterrupt):
            with self.assertRaises(KeyboardInterrupt):trainer.update_pair()
        self.assertEqual(trainer.step,0)
        for r in before:self.assert_nested(before[r],trainer.stages[r].model.state_dict());self.assertEqual(trainer.stages[r].step,0)
        self.assert_nested(rng,rng_state())
        class FailingList(list):
            def append(self,value):super().append(value);raise KeyboardInterrupt
        trainer.history=FailingList()
        with self.assertRaises(KeyboardInterrupt):trainer.update_pair()
        self.assertEqual(trainer.step,0);self.assertEqual(trainer.history,[])
        for r in before:self.assert_nested(before[r],trainer.stages[r].model.state_dict())

    def test_resume_rejects_changed_environment_artifacts_and_budget(self):
        t=self.trainer('binding');t.run(stop_after=1);pin=t.last['sha256']
        changed=self.trainer('binding');changed.binding['environment']['precision']='fp16'
        with self.assertRaisesRegex(ValueError,'resume_binding'):changed.resume(pin)
        state=torch.load(t.root/'last.pt',weights_only=True);state['elapsed']=self.cfg['limits']['cumulative_seconds']+1
        torch.save(state,t.root/'bad.pt')
        # Exact pin alone cannot authorize changed budgets/cursors.
        raw=(t.root/'last.pt').read_bytes();(t.root/'last.pt').write_bytes((t.root/'bad.pt').read_bytes())
        with self.assertRaisesRegex(ValueError,'resume_binding'):self.trainer('binding').resume(sha((t.root/'last.pt').read_bytes()))
        (t.root/'last.pt').write_bytes(raw)
        artifact=t.root/next(iter(t.snapshots.values()))['name'];artifact.write_bytes(b'broken')
        with self.assertRaises(ValueError):self.trainer('binding').resume(pin)

    def test_diagnostic_invariance_and_visual_artifacts(self):
        cfg=deepcopy(self.cfg);cfg.update(milestones=[0,2],sampling_steps=[1,10],gt_noised_timesteps=[1,20])
        models={k:deepcopy(self.models) for k in cfg['milestones']}
        report=runs.diagnostic(self.inputs,cfg,'cpu',self.root/'diagnostic',fixture_models=models)
        self.assertTrue(report['input_exact_unchanged'])
        self.assertTrue(all(v['exact_unchanged'] for v in report['model_invariance'].values()))
        self.assertEqual(len(report['evaluations']['2']),4)
        for group in report['evaluations'].values():
            for e in group.values():
                self.assertIn('hand_wrist_relative',e['shared_axis_bounds'])
                for p in e['artifacts']:self.assertEqual(sha((self.root/'diagnostic'/p['name']).read_bytes()),p['sha256'])

    def test_result_package_and_local_retention_once(self):
        root=self.root/'run-a';(root/'experiment').mkdir(parents=True);(root/'inputs').mkdir()
        (root/'operation.json').write_text('{"state":"failed"}')
        (root/'inputs/parent-model.pt').write_bytes(b'excluded input')
        (root/'experiment/last.pt').write_bytes(b'new optimizer state')
        archive=package(root,self.root/'result.zip','v153')
        with zipfile.ZipFile(archive['archive']) as z:
            self.assertEqual(set(z.namelist()),{'operation.json','experiment/last.pt','recovery_receipt.json'})
        drive=self.root/'fake_drive';drive.mkdir();attempt=prepare_attempt(drive,'run-a','inference',namespace='V153-PREDICTED-CLIP-001')
        receipt=publish(archive,attempt,run_id='V153-PREDICTED-CLIP-001');self.assertEqual(receipt['state'],'verified')
        with self.assertRaises(FileExistsError):publish(archive,attempt,run_id='V153-PREDICTED-CLIP-001')
        self.assertTrue(Path(archive['archive']).is_file())

    def test_notebook_default_off_and_resume_existing_fresh_run(self):
        for experiment in ('v152','v153'):
            doc=notebook_document({'source':{'name':'source.zip','sha256':'a'*64,'revision':'0'*40},'source_code_sha256':'b'*64,'bundle':{'manifest_sha256':'c'*64,'name':'input.zip','sha256':'d'*64}},experiment)
            cells=[''.join(c['source']) for c in doc['cells'] if c['cell_type']=='code']
            for c in cells:compile(c,'notebook','exec')
            tree=ast.parse(cells[0]);functions={n.name:n for n in tree.body if isinstance(n,ast.FunctionDef)}
            scope={'Path':Path,'WORK':self.root,'EXPERIMENT':experiment,'FRESH_RUN':self.root/('run-'+experiment)}
            existing=scope['FRESH_RUN'];(existing/'experiment').mkdir(parents=True,exist_ok=True)
            exec(compile(ast.Module(body=[functions['select_resume']],type_ignores=[]),'resume','exec'),scope)
            with self.assertRaises(ValueError):scope['select_resume'](existing,'e'*64)
            (existing/'experiment/last.pt').write_bytes(b'durable')
            if experiment=='v153':scope['select_resume'](existing,'e'*64);self.assertEqual(scope['RUN'],existing)
            else:
                with self.assertRaises(ValueError):scope['select_resume'](existing,'e'*64)
            self.assertIn('RUN_EXPERIMENT=False',cells[1]);self.assertIn('SHOW_SAVED_RESULTS=False',cells[2])

    def test_notebook_failure_finally_preserves_and_publishes_once(self):
        doc=notebook_document({},'v153');tree=ast.parse(''.join(doc['cells'][1]['source']))
        function=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='run_experiment')
        folder=self.root/'run-finally';folder.mkdir();drive_path=self.root/'drive/MyDrive';drive_path.mkdir(parents=True)
        google=ModuleType('google');colab=ModuleType('google.colab');colab.drive=SimpleNamespace(mount=lambda p:None)
        calls=[];records=[]
        def execute(args,timeout):
            calls.append(args[0])
            if args[0]=='backup-prepare':return {'attempt':'fake-attempt'}
            if args[0]=='prepare':return {'status':'verified'}
            if args[0]=='run':raise RuntimeError('injected_compute_failure')
            if args[0]=='package':return {'archive':str(self.root/'retained.zip'),'sha256':'a'*64,'bytes':20,'package_wall_seconds':.1}
            if args[0]=='publish':return {'state':'verified'}
            raise AssertionError(args)
        scope={'Path':Path,'DRIVE':drive_path,'RUN':folder,'RECOVERY':self.root,'BOOTSTRAP_WALL':.1,
            'PINS':{'source_code_sha256':'a'*64,'bundle':{'manifest_sha256':'b'*64,'name':'input.zip','sha256':'c'*64}},
            'POLICY':self.cfg,'DEVICE':'cuda','execute':execute,'record':lambda s:records.append(deepcopy(s)),
            'json':json,'time':__import__('time'),'uuid':__import__('uuid')}
        exec(compile(ast.Module(body=[function],type_ignores=[]),'finally','exec'),scope)
        with patch.dict(sys.modules,{'google':google,'google.colab':colab}),patch.object(Path,'is_mount',return_value=True):
            with self.assertRaisesRegex(RuntimeError,'injected_compute_failure'):scope['run_experiment']()
        self.assertEqual(calls,['backup-prepare','prepare','run','package','publish'])
        self.assertEqual(records[-1]['state'],'failed');self.assertTrue((folder/'retention_receipt.json').is_file())
        # A copy failure keeps the same local result and does not retry publication.
        old=scope['execute']
        def failed_copy(args,timeout):
            if args[0]=='publish':calls.append('publish');raise OSError('copy_failure')
            return old(args,timeout)
        scope['execute']=failed_copy;calls.clear()
        with patch.dict(sys.modules,{'google':google,'google.colab':colab}),patch.object(Path,'is_mount',return_value=True):
            with self.assertRaisesRegex(RuntimeError,'injected_compute_failure'):scope['run_experiment']()
        self.assertEqual(calls.count('publish'),1);self.assertIn('retention_failure',records[-1])

    def test_actual_pinned_bundle_and_u0_fingerprints(self):
        archive=ROOT/'outputs/train_clip_diagnostics/input_bundle.zip'
        if not archive.is_file():self.skipTest('local parent bundle not committed')
        inputs=bundle.extract(archive,'9b898777a61613da6f14537ec55a00973f4b77c2c3519e59643bb82308d187c2','a4d6c655f703250e5c2f3187de4f32e9098e78904d06f87793c4e379ac247353',self.root/'actual')
        models=bundle.load_models(inputs,0,'cpu')
        self.assertEqual({r:model_fingerprint(m) for r,m in models.items()},inputs['run']['initial'])
        self.assertEqual(inputs['clip'].source,'predicted_2d');self.assertEqual(inputs['clip'].split,'train')
        p=self.root/'actual/features/train-00.pt';raw=bytearray(p.read_bytes());raw[-1]^=1;p.write_bytes(raw)
        with self.assertRaisesRegex(ValueError,'file_sha'):bundle.checked(self.root/'actual',inputs['bundle']['content_sha256'])

    def test_cli_source_fixed_cuda_and_prepare_receipt(self):
        from contextlib import redirect_stdout,redirect_stderr
        import io
        capture=io.StringIO()
        with redirect_stdout(capture):cli_main(['--experiment','v152','source'])
        source=digest(bundle.source_identity());self.assertEqual(json.loads(capture.getvalue())['source_sha256'],source)
        run=self.root/'run-cli';run.mkdir()
        common=['--run',str(run),'--source-sha256',source,'--manifest-sha256','a'*64]
        with redirect_stderr(io.StringIO()),self.assertRaises(SystemExit):cli_main(['--experiment','v153','run']+common+['--device','cpu'])
        with patch('rewind_v1.train_clip_cli._linux_available_ram',return_value=8*1024**3), \
                patch('rewind_v1.train_clip_cli.shutil.disk_usage',return_value=SimpleNamespace(free=20*1024**3)), \
                patch.object(bundle,'extract',return_value={'bundle':{'content_sha256':'a'*64}}),redirect_stdout(io.StringIO()):
            archive=self.root/'fake.zip';archive.write_bytes(b'fixture')
            cli_main(['--experiment','v153','prepare']+common+['--bundle',str(archive),'--bundle-sha256','b'*64])
        self.assertEqual(json.loads((run/'input_status.json').read_bytes())['status'],'verified')
        # New optimizer state must be empty; teacher.step zero and parent weights exact.
        trainer=self.trainer('fresh')
        for role,stage in trainer.stages.items():
            self.assertEqual(stage.step,0);self.assertEqual(stage.optimizer.state_dict()['state'],{})
            self.assertEqual(model_fingerprint(stage.model),model_fingerprint(self.models[role]))
