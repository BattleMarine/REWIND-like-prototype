"""Local fixtures and immutable reference checks; no external data or GPU execution."""

import copy
import ast
import io
import json
from pathlib import Path
import shutil
import tempfile
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch

from rewind_archive.contracts import canonical
from rewind_prepare.converter import sha
from rewind_prepare.temporal_selection import seal
from rewind_v1 import camera_control as control,cross_data as data,cross_inference as inference
from rewind_v1 import cross_recovery as recovery
from rewind_v1.camera_control_cli import EXTRA_REPORTS,checked_stage_inputs,run as run_stage
from rewind_archive.contracts import digest
from scripts.build_v12_camera_control import notebook_document
import test_cross_scenario as fixtures
import test_prepare


ROOT=Path(__file__).resolve().parents[1]
REFERENCE=ROOT/'outputs/v12_cross_scenario_20261115/user_test/completed_001/results.zip'


class CameraControlTests(TestCase):
    def setUp(self):
        (ROOT/'outputs/test_tmp').mkdir(parents=True,exist_ok=True)
        fixtures.CrossScenarioTests.setUp(self)

    def models(self):return fixtures.CrossScenarioTests.models(self)

    def archvis(self):
        seqs=[f'ArchVisInterior_ArchVis_RT/Day/person/glasses/0{i}1/Motion__1_' for i in (1,2)]
        raw=('\n'.join(self.sequences+seqs)+'\n').encode()
        (self.inputs/'split/validation.txt').write_bytes(raw)
        self.spec['pinned_sources']['split_sha256']['validation']=sha(raw)
        for seq in seqs:
            for role in ('left','right','pose'):
                folder=self.pose/seq/'json' if role=='pose' else self.rgb/seq/'fisheye_final_image'/('camera_'+role)
                folder.mkdir(parents=True)
                for f in range(50):(folder/(f'frame_{f}.json' if role=='pose' else f'final_{f}.png')).write_bytes(b'x')
        return seqs

    def test_explicit_context_archvis_only_partial_zero_and_old_scope(self):
        seqs=self.archvis();ctx=control.context()
        plan=data.select(self.inputs,self.rgb,self.pose,self.root/'archvis',context=ctx)
        self.assertEqual([c['sequence'] for c in plan['clips']],seqs)
        self.assertEqual(plan['run_id'],control.RUN_ID)
        data.checked_selection(self.root/'archvis/selection.json',plan['content_sha256'],context=ctx)
        with self.assertRaisesRegex(ValueError,'source_or_run'):
            data.checked_selection(self.root/'archvis/selection.json',plan['content_sha256'])
        (self.pose/seqs[0]/'json/frame_25.json').unlink()
        partial=data.select(self.inputs,self.rgb,self.pose,self.root/'partial',context=ctx)
        self.assertEqual(partial['status'],'partial_availability')
        (self.pose/seqs[1]/'json/frame_25.json').unlink()
        with self.assertRaisesRegex(ValueError,'zero_available'):
            data.select(self.inputs,self.rgb,self.pose,self.root/'empty',context=ctx)
        # A self-sealed test sequence still fails official membership at preparation.
        changed=copy.deepcopy(plan);c=changed['clips'][0];old=c['sequence'];c['sequence']=self.test
        c['environment']=self.test.split('/')[0]
        changed=seal({k:v for k,v in changed.items() if k!='content_sha256'})
        (self.root/'test.json').write_bytes(canonical(changed))
        with self.assertRaises(ValueError):data.prepare(self.inputs,self.root/'test.json',changed['content_sha256'],self.root/'bad',context=ctx)

    def test_reference_pin_and_chain_tamper_reorder_key_hash_model(self):
        if not REFERENCE.is_file():self.skipTest('fixed result ZIP is local-only evidence')
        baseline=self.root/'baseline';ref=control.load_reference(REFERENCE,baseline)
        self.assertEqual(len(ref['clips']),4)
        for operation in ('tamper','order','key','hash','model','source'):
            root=self.root/operation;shutil.copytree(baseline,root)
            if operation=='tamper':
                p=root/'inference/report.json';p.write_bytes(p.read_bytes()+b' ')
            else:
                name='prepared/prepared.json' if operation in ('order','hash') else 'inference/report.json'
                if operation=='key':name='inference/clip-00/predictions.json'
                p=root/name;d=json.loads(p.read_bytes())
                if operation=='order':d['clips'].reverse()
                elif operation=='key':d['keys'].reverse()
                elif operation=='hash':d['clips'][0]['frames'][0]['files']['pose']['sha256']='0'*64
                elif operation=='model':d['model_zip_sha256']='0'*64
                else:d['source_sha256']='0'*64
                d=seal({k:v for k,v in d.items() if k!='content_sha256'});p.write_bytes(canonical(d))
                # Even an updated receipt cannot hide a broken dependency chain.
                receipt=json.loads((root/'recovery_receipt.json').read_bytes())
                receipt['files'][name]={'bytes':p.stat().st_size,'sha256':sha(p.read_bytes())}
                (root/'recovery_receipt.json').write_bytes(canonical(receipt))
            with self.subTest(operation=operation),self.assertRaises(ValueError):control.checked_reference(root)
        wrong=self.root/'wrong.zip';raw=bytearray(REFERENCE.read_bytes());raw[20]^=1;wrong.write_bytes(raw)
        with self.assertRaisesRegex(ValueError,'sha256'):control.load_reference(wrong,self.root/'wrong-extracted')

    def test_camera_units_encoded_axes_statistics_and_frame_alignment(self):
        fixture=test_prepare.PreparationTests();fixture.setUp();self.addCleanup(fixture.doCleanups)
        native=json.loads(fixture.add_frame(fixture.seq,7).read_bytes())
        row=control.camera_record(native,self.spec,self.calib,fixture.seq,7,{'sha256':'f'*64,'size_bytes':123})
        self.assertEqual(row['key'],[fixture.seq,7]);self.assertEqual(len(row['encoded_camera_18']),18)
        self.assertEqual(row['world_to_camera_translation_mm'],[[-60.,0.,0.],[60.,0.,0.]])
        self.assertEqual(row['translation_norm_mm'],[60.,60.]);self.assertEqual(row['baseline_mm'],120.)
        self.assertAlmostEqual(row['encoded_camera_18'][6],-.6,places=6)
        stat=control.statistics([[0,10],[10,20],[20,30]])
        self.assertEqual(stat['mean'],[10.,20.]);self.assertEqual(stat['p05'],[1.,11.]);self.assertEqual(stat['p95'],[19.,29.])
        np.testing.assert_allclose(stat['std'],[np.sqrt(200/3)]*2)
        with self.assertRaises(ValueError):control.statistics([[float('nan')]])
        with self.assertRaisesRegex(ValueError,'frame_alignment'):
            control.observation({'frames':[row]}, {'keys':[[fixture.seq,8]]},self.spec)

    def test_reference_receipt_pin_is_carried_between_child_stages(self):
        run=self.root/'staged';run.mkdir();(run/'reference').mkdir()
        (run/'reference/recovery_receipt.json').write_bytes(b'changed receipt')
        status={'run_id':control.RUN_ID,'source_sha256':digest(control.source_identity()),'completed':['reference','select','model'],
                'state':'stage_complete','costs':{},'reference_receipt_sha256':sha(b'original receipt')}
        data.write_report(run/'control_status.json',status)
        with patch('rewind_v1.camera_control_cli.preflight',return_value={}),self.assertRaisesRegex(ValueError,'receipt_changed'):
            run_stage(SimpleNamespace(run=run,stage='prepare'))
        self.assertEqual(json.loads((run/'control_status.json').read_bytes())['state'],'failed')

    def test_notebook_default_off_compile_and_independent_stage_deadlines(self):
        nb=notebook_document({});code=''
        for i,c in enumerate(nb['cells']):
            if c['cell_type']!='code':continue
            src=''.join(c['source']);compile(src,f'cell-{i}','exec');code+=src
            if c['id']!='bootstrap':exec(src,{'run_control':lambda *a:self.fail('default run'),'display_saved':lambda:self.fail('default display')})
        for stage in ('reference','select','model','prepare','infer'):self.assertIn("('"+stage+"',",code)
        self.assertIn('process.kill();process.wait(timeout=30)',code)
        self.assertIn('finally:',code);self.assertIn('display_saved()',code)

    def test_notebook_intermediate_retention_and_failure_no_retry(self):
        nb=notebook_document({});src=''.join(nb['cells'][1]['source'])
        node=next(n for n in ast.parse(src).body if isinstance(n,ast.FunctionDef) and n.name=='run_control')
        for failure,expected in ((None,3),('model',2),('publish',1)):
            calls=[];records=[]
            def execute(args,timeout):
                calls.append(args)
                if args[0]=='run':
                    stage=args[args.index('--stage')+1]
                    if stage==failure:raise RuntimeError('fixture stage failure')
                    return {'status':'complete'}
                if args[0]=='backup-prepare':return {'attempt':'attempt-'+str(len(calls))}
                if args[0]=='package':return {'archive':'fixture.zip','bytes':10,'sha256':'f'*64}
                if args[0]=='publish':
                    if failure=='publish':raise RuntimeError('fixture retention failure')
                    return {'state':'verified'}
                return {}
            fake_drive=SimpleNamespace(parent=SimpleNamespace(is_mount=lambda:True),is_dir=lambda:True)
            env={'execute':execute,'RUN':self.root/'run-fixture','WORK':self.root,'SOURCE':self.root,'INPUTS':self.inputs,
                 'REFERENCE':REFERENCE,'RGB_ROOT':self.rgb,'POSE_ROOT':self.pose,'MODEL_SOURCE':self.root/'m',
                 'MODEL_LOCAL':self.root/'local','DRIVE_ROOT':fake_drive,'RECOVERY':self.root,
                 'PINS':{'source_code_sha256':'f'*64},'LIMITS':control.LIMITS,'uuid':__import__('uuid'),
                 'record':lambda status:records.append(copy.deepcopy(status))}
            exec(compile(ast.Module(body=[node],type_ignores=[]),'fixture-notebook','exec'),env)
            with patch.dict('sys.modules',{'google':SimpleNamespace(),'google.colab':SimpleNamespace(drive=SimpleNamespace(mount=lambda *a:None))}):
                if failure:
                    with self.assertRaises(RuntimeError):env['run_control']('cpu')
                else:env['run_control']('cpu')
            self.assertEqual(sum(c[0]=='publish' for c in calls),expected)
            if failure=='publish':self.assertFalse(any('--stage' in c and 'model' in c for c in calls))

    def test_positive_camera_recovery_inference_comparison_package_and_hash_rejection(self):
        seqs=self.archvis();ctx=control.context()
        fixture=test_prepare.PreparationTests();fixture.setUp();self.addCleanup(fixture.doCleanups)
        png=io.BytesIO();Image.new('RGB',(1024,1024),(40,80,120)).save(png,format='PNG')
        for seq in self.sequences+seqs:
            for frame in range(50):
                raw=fixture.add_frame(seq,frame).read_bytes()
                (self.pose/seq/'json'/f'frame_{frame}.json').write_bytes(raw)
                for side in ('left','right'):(self.rgb/seq/'fisheye_final_image'/('camera_'+side)/f'final_{frame}.png').write_bytes(png.getvalue())
        city=self.root/'city';city.mkdir()
        city_plan=data.select(self.inputs,self.rgb,self.pose,city/'selection')
        city_prepared=data.prepare(self.inputs,city/'selection/selection.json',city_plan['content_sha256'],city/'prepared')
        # Synthetic reference predictions used only in this fixture; production references remain pinned.
        reference={'selection':city_plan,'prepared':city_prepared,'report':{'content_sha256':'d'*64},'clips':[]}
        for a,b in zip(city_plan['clips'],city_prepared['clips']):
            tensor=torch.load(city/'prepared'/f"clip-{b['number']:02d}/clip.pt",weights_only=True)
            idx=[self.spec['joint_names'].index(n) for n in self.spec['body_joint_names']]
            prediction={'keys':[[b['sequence'],r['frame']] for r in b['frames']],
                'prediction':{'body':(tensor['target'][:,idx]+10).tolist()},'gt_target_device_mm':tensor['target'].tolist(),
                'gt_target_valid':tensor['target_valid'].tolist(),'aggregate':{'body':{'mean':17.32},
                    'body_shape_sums':{k:1. for k in ('predicted_mean_bone_mm','target_mean_bone_mm','predicted_mean_spread_mm','target_mean_spread_mm')}}}
            prediction=seal(prediction);reference['clips'].append({'selection':a,'prepared':b,'prediction':prediction})
        run=self.root/'run';run.mkdir()
        plan=data.select(self.inputs,self.rgb,self.pose,run/'selection',context=ctx)
        budget=data.Budget(3600)
        prepared=data.prepare(self.inputs,run/'selection/selection.json',plan['content_sha256'],run/'prepared',context=ctx,budget=budget)
        before=budget.read_bytes
        cameras=control.recover_cameras(self.inputs,run/'prepared',reference,self.pose,run/'cameras',budget)
        pins={'selection_sha256':plan['content_sha256'],'prepared_sha256':prepared['content_sha256'],'cameras_sha256':cameras['content_sha256']}
        checked_stage_inputs(run,pins,with_cameras=True)
        altered=copy.deepcopy(cameras);altered['prepared_sha256']='0'*64
        altered=seal({k:v for k,v in altered.items() if k!='content_sha256'})
        (run/'cameras/cameras.json').write_bytes(canonical(altered))
        with self.assertRaises(ValueError):checked_stage_inputs(run,pins,with_cameras=True)
        with self.assertRaisesRegex(ValueError,'camera_prepared_binding'):
            checked_stage_inputs(run,{**pins,'cameras_sha256':altered['content_sha256']},with_cameras=True)
        (run/'cameras/cameras.json').write_bytes(canonical(cameras))
        self.assertEqual(len(cameras['clips']),6)
        self.assertEqual(cameras['total_preparation_source_bytes'],budget.read_bytes)
        self.assertEqual(cameras['city_pose_source_bytes'],budget.read_bytes-before)
        self.assertEqual(sum(len(c['frames']) for c in cameras['clips']),300)
        models,cfg=self.models()
        class Frontend(torch.nn.Module):
            def forward(self,images):
                n=len(images);return torch.zeros(n,2,62,1,1),torch.full((n,2,62,2),64.),torch.full((n,2,62),.8)
        models['frontend']=Frontend().eval();models.pop('spec')
        with patch.object(inference,'assets',return_value=data.assets(self.inputs)), \
                patch.object(inference.view,'_models',return_value=(models,{'fixture':True},{'config':cfg},{'fixture':True})), \
                patch.object(torch.optim,'Adam',side_effect=AssertionError('no training')):
            report=inference.infer(self.inputs,run/'selection/selection.json',plan['content_sha256'],run/'prepared',
                prepared['content_sha256'],self.root/'fixture-model.zip',run/'inference',torch.device('cpu'),context=ctx)
        self.assertEqual(report['status'],'complete')
        comparison=control.compare(self.inputs,cameras,run/'inference',reference,run/'comparison',data.Budget(3600),inference_report=report)
        self.assertEqual(len(comparison['by_clip']),6)
        self.assertAlmostEqual(comparison['by_clip'][2]['centered_body_error_mean_mm'],0.,places=10)
        self.assertEqual(comparison['by_clip'][2]['centered_body_valid_joints'],50*22)
        self.assertTrue((run/'comparison/comparison.md').is_file())
        archive=recovery.package(run,self.root/'results.zip',run_id=control.RUN_ID,extra_reports=EXTRA_REPORTS)
        drive=self.root/'drive';drive.mkdir();attempt=recovery.prepare_attempt(drive,'run-fixture','inference',namespace=control.RUN_ID)
        self.assertEqual(recovery.publish(archive,attempt,run_id=control.RUN_ID)['state'],'verified')
        self.assertIn(control.RUN_ID,str(attempt))
        files=[p.relative_to(run).as_posix() for p in recovery.result_files(run,extra_reports=EXTRA_REPORTS)]
        self.assertIn('cameras/cameras.json',files);self.assertFalse(any('/raw/' in n or n.endswith('.pt') for n in files))
        # Fixed city pose corruption must fail before producing a replacement camera report.
        row=reference['clips'][0]['selection']['frames'][0]['files']['pose']
        path=self.pose/row['path'];path.write_bytes(b' '*row['size_bytes'])
        with self.assertRaisesRegex(ValueError,'pose_hash_mismatch'):
            control.recover_cameras(self.inputs,run/'prepared',reference,self.pose,self.root/'bad-cameras',data.Budget(3600))
