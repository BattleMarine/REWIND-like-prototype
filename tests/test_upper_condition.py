"""Frozen inference and attribution checks; no training or external Drive operations."""

import ast
from copy import deepcopy
import json
from pathlib import Path
import shutil
import tempfile
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import patch

import torch
from PIL import Image

from rewind_archive.contracts import canonical
from rewind_prepare.converter import sha
from rewind_prepare.temporal_selection import seal
from rewind_v1 import upper_condition as upper,upper_condition_bundle as bundle,cross_inference as infer
from rewind_v1 import camera_training as parent
from rewind_v1.data import gt_upper_train,Clip
from rewind_v1.run_data import FixtureData
from rewind_v1.teacher import V1Body,V1Hand
from rewind_v1.upper_condition_cli import package
from rewind_v1.cross_recovery import prepare_attempt,publish
from scripts.build_v151_upper_condition import notebook_document


ROOT=Path(__file__).resolve().parents[1]
QA=ROOT/'outputs/v151_upper_condition_20261115'


class UpperConditionTests(TestCase):
    def setUp(self):
        (ROOT/'outputs/test_tmp').mkdir(parents=True,exist_ok=True)
        temp=tempfile.TemporaryDirectory(dir=ROOT/'outputs/test_tmp');self.addCleanup(temp.cleanup);self.root=Path(temp.name)
        self.spec=FixtureData().spec;self.cfg=deepcopy(parent.POLICY)
        self.cfg['teacher'].update(hidden=8,heads=2,graph_layers=1,temporal_layers=1,diffusion_steps=20,sampling_steps=2)
        torch.manual_seed(71);self.models={}
        for role in ('body','hand'):
            names=infer.view._group_names(self.spec,role);edges=[(names.index(a),names.index(b)) for a,b in self.spec['edges'] if a in names and b in names]
            m=V1Body(self.cfg['teacher'],len(names),edges) if role=='body' else V1Hand(self.cfg['teacher'],len(names),edges,len(self.spec['upper_joint_names']))
            self.models[role]=m.eval().requires_grad_(False)
        self.models['spec']=self.spec
        self.features=torch.randn(50,56);self.keys=[('fixture_validation',i) for i in range(50)]
        r=torch.eye(3,dtype=torch.float64).expand(50,2,3,3).tolist();t=[[[-1000.-i,60.,20.],[-1000.-i,-60.,20.]] for i in range(50)]
        self.camera,_=parent.CameraOrigin('first_midpoint').apply(r,t,'fixture_validation',0,list(range(50)))
        self.target=torch.arange(600,dtype=torch.float32).reshape(50,4,3)/10
        self.valid=torch.ones(50,4,dtype=torch.bool)

    def inputs(self):
        original=infer.sample_features(self.models,self.features,self.camera,self.keys,self.cfg,torch.device('cpu'))
        return {'spec':self.spec,'bundle':{'content_sha256':'a'*64},'run':{'environment':{'device':'fixture_cpu'}},'clips':[{
            'keys':self.keys,'camera':self.camera,'metadata':{'sequence':'fixture_validation'},
            'data':{'features':self.features,'target':self.target,'target_valid':self.valid},
            'parent':{'content_sha256':'b'*64,'variants':{'baseline':{'prediction':{k:v.tolist() for k,v in original.items()}}}}}]}

    def test_real_bundle_relative_path_chain_and_frozen_model_load(self):
        receipt=QA/'bundle_receipt.json'
        if not receipt.is_file():self.skipTest('Developer compact bundle is local-only evidence')
        pin=json.loads(receipt.read_bytes());source=QA/pin['name'];relative=self.root.relative_to(ROOT)/'bundle'
        checked=bundle.extract_bundle(source,pin['sha256'],pin['manifest_sha256'],relative)
        self.assertEqual(len(checked['clips']),6)
        with patch.object(torch.optim,'Adam',side_effect=AssertionError('no optimizer')), \
                patch.object(parent.TeacherRun,'__init__',side_effect=AssertionError('no training runner')):
            models,cfg=upper.load_models(relative,checked,torch.device('cpu'))
        self.assertEqual(cfg['seed'],71)
        self.assertTrue(all(not m.training and not any(p.requires_grad for p in m.parameters()) for k,m in models.items() if k!='spec'))
        self.assertFalse((relative/'train/last.pt').exists())
        # A corrupted selected payload is rejected even though JSON metadata is untouched.
        p=relative/'eval_data/eval-00.pt';raw=bytearray(p.read_bytes());raw[-1]^=1;p.write_bytes(raw)
        with self.assertRaisesRegex(ValueError,'member_sha'):bundle.checked_bundle(relative,pin['manifest_sha256'])

    def test_common_mask_factorial_and_gt_upper_gate_remains_train_only(self):
        target=self.target[:2].clone();valid=self.valid[:2].clone();valid[0,0]=False
        fingers,wrists=infer._hand_indices(self.spec['joint_names'],self.spec)
        local=target[:,fingers]-target[:,wrists];body=target[:,:2]+100
        result,_,mask,_=upper.factorial_metrics(body,{'A':local,'B':local},target,valid,self.spec)
        self.assertEqual(int(mask.sum()),3)
        for c in result.values():
            self.assertEqual(c['local_common_mask']['count'],3);self.assertEqual(c['local_common_mask']['mean_mm'],0)
            self.assertAlmostEqual(c['attachments']['gt_wrist_diagnostic']['common_mask_error']['mean_mm'],0.,places=6)
            self.assertAlmostEqual(c['attachments']['predicted_wrist']['common_mask_error']['mean_mm'],3**.5*100,places=5)
            self.assertEqual(c['attachments']['predicted_wrist']['parent_native_mask_error']['count'],4)
        target[0,0]=float('nan');values,mask=upper.upper_values(body,target,valid,self.spec,'gt_upper_diagnostic')
        self.assertFalse(mask[0,0]);self.assertTrue(torch.equal(values[0,0],torch.zeros(3)))
        clip=Clip(torch.zeros(1,2,56),torch.zeros(1,2,2,9),torch.nan_to_num(target)[None],valid[None],torch.ones(1,2,dtype=torch.bool),
            torch.zeros(1,2),'predicted_2d',split='validation')
        with self.assertRaisesRegex(ValueError,'requires_train'):gt_upper_train(clip,self.spec)

    def test_A_sampler_matches_existing_and_GT_future_does_not_change_prefix(self):
        expected=infer.sample_features(self.models,self.features,self.camera,self.keys,self.cfg,torch.device('cpu'))
        body=upper.sample_group(self.models['body'],'body',self.features,self.camera,self.keys,self.cfg,self.spec,torch.device('cpu'))
        a,valid=upper.upper_values(body,self.target,self.valid,self.spec,'predicted_upper')
        local=upper.sample_hand(self.models['hand'],a,valid,self.features,self.camera,self.keys,self.cfg,self.spec,torch.device('cpu'))
        self.assertTrue(torch.equal(body,expected['body']));self.assertTrue(torch.equal(local,expected['hand_local']))
        gt,mask=upper.upper_values(body,self.target,self.valid,self.spec,'gt_upper_diagnostic')
        b=upper.sample_hand(self.models['hand'],gt,mask,self.features,self.camera,self.keys,self.cfg,self.spec,torch.device('cpu'))
        future=gt.clone();future[30:]+=999
        changed=upper.sample_hand(self.models['hand'],future,mask,self.features,self.camera,self.keys,self.cfg,self.spec,torch.device('cpu'))
        torch.testing.assert_close(b[:25],changed[:25],atol=.01,rtol=1e-5)

    def test_positive_call_counts_render_recovery_and_no_optimizer(self):
        inputs=self.inputs();calls=[];actual=upper.sample_group
        def track(model,role,features,camera,keys,*args):calls.append((role,len(keys)));return actual(model,role,features,camera,keys,*args)
        run=self.root/'run';run.mkdir()
        with patch.object(upper,'load_models',return_value=(self.models,self.cfg)),patch.object(upper,'sample_group',side_effect=track), \
                patch.object(torch.optim,'Adam',side_effect=AssertionError('no optimizer')):
            result=upper.run(self.root,inputs,run/'inference',torch.device('cpu'))
        self.assertEqual(calls.count(('body',50)),1);self.assertEqual(calls.count(('hand',50)),3)
        self.assertEqual(calls.count(('body',25)),1);self.assertEqual(calls.count(('hand',25)),2)
        self.assertEqual(result['model_fingerprints_before'],result['model_fingerprints_after'])
        clip=json.loads((run/'inference/clip-00/predictions.json').read_bytes())
        self.assertEqual(clip['input_fingerprints_before'],clip['input_fingerprints_after'])
        self.assertEqual(clip['cross_run_parent_comparison']['status'],'within_tolerance')
        self.assertEqual(clip['same_run_A_repeat']['status'],'stable_within_fixed_tolerance')
        self.assertEqual(set(clip['artifacts']['bounds_per_row_mm']),{'body','local_hand','global_pred_wrist'})
        with Image.open(run/'inference/clip-00/comparison.gif') as image:
            self.assertEqual(image.n_frames,50);self.assertEqual(image.size,(738,810))
        self.assertEqual(len(list((run/'inference/clip-00').glob('frame-*.png'))),3)
        # Retention excludes a deliberately present input model.
        (run/'inputs').mkdir();(run/'inputs/model.pt').write_bytes(b'never publish')
        archived=package(run,self.root/'results.zip');drive=self.root/'drive';drive.mkdir()
        attempt=prepare_attempt(drive,'run-fixture','inference',namespace=bundle.RUN_ID)
        saved=publish(archived,attempt,run_id=bundle.RUN_ID);self.assertEqual(saved['state'],'verified')
        import zipfile
        with zipfile.ZipFile(archived['archive']) as z:self.assertFalse(any(n.startswith('inputs/') for n in z.namelist()))
        qa_path=__import__('os').environ.get('V151_QA_IMAGE')
        if qa_path:shutil.copyfile(run/'inference/clip-00/frame-000000025.png',qa_path)

    def test_mutated_model_is_rejected_and_failure_report_retained(self):
        inputs=self.inputs()
        def mutate(*args,**kwargs):
            next(self.models['body'].parameters()).add_(1.)
            return {'files':[],'bounds_per_row_mm':{}}
        with patch.object(upper,'load_models',return_value=(self.models,self.cfg)),patch.object(upper,'render',side_effect=mutate):
            with self.assertRaisesRegex(ValueError,'model_mutated'):upper.run(self.root,inputs,self.root/'failed',torch.device('cpu'))
        self.assertTrue((self.root/'failed/failure.json').is_file())

    def test_notebook_default_off_and_operational_failure_retains_once(self):
        doc=notebook_document({})
        for c in doc['cells']:
            if c['cell_type']!='code':continue
            src=''.join(c['source']);compile(src,c['id'],'exec')
            if c['id']!='bootstrap':exec(src,{'run_diagnostic':lambda:self.fail('default run'),'display_saved':lambda:self.fail('default show')})
        code=''.join(doc['cells'][1]['source']);node=next(n for n in ast.parse(code).body if isinstance(n,ast.FunctionDef) and n.name=='run_diagnostic')
        for fail in (None,'prepare','infer','package','publish'):
            calls=[];root=self.root/('operation-'+str(fail));root.mkdir();drive=self.root/'drive';drive.mkdir(exist_ok=True)
            def execute(args,timeout):
                calls.append(args[0])
                if args[0]==fail:raise RuntimeError('fixture '+fail)
                if args[0]=='backup-prepare':return {'attempt':'attempt'}
                if args[0]=='package':return {'archive':'local.zip','sha256':'f'*64,'bytes':10,'package_wall_seconds':.01}
                return {'status':'complete'}
            env={'execute':execute,'DRIVE':drive,'RUN':root,'RECOVERY':self.root,'BOOTSTRAP_WALL':.01,'DEVICE':'cpu','Path':Path,
                'json':json,'time':__import__('time'),'uuid':__import__('uuid'),'record':lambda *a:None,
                'PINS':{'source_code_sha256':'f'*64,'bundle':{'manifest_sha256':'e'*64,'sha256':'d'*64,'name':'input.zip'}}}
            exec(compile(ast.Module(body=[node],type_ignores=[]),'operation-fixture','exec'),env)
            with patch.dict('sys.modules',{'google':SimpleNamespace(),'google.colab':SimpleNamespace(drive=SimpleNamespace(mount=lambda *a:None))}),patch.object(Path,'is_mount',return_value=True):
                if fail:
                    with self.assertRaises(RuntimeError):env['run_diagnostic']()
                else:env['run_diagnostic']()
            self.assertEqual(calls.count('package'),1);self.assertEqual(calls.count('publish'),0 if fail=='package' else 1)
