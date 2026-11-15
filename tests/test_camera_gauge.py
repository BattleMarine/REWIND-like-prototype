"""Fixed local gauge fixtures, never production training or remote evaluation."""

import ast
import copy
import io
import json
from pathlib import Path
import shutil
import tempfile
from unittest import TestCase
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch

from rewind_archive.contracts import canonical
from rewind_m2.data import encode_camera
from rewind_prepare.converter import sha
from rewind_prepare.temporal_selection import seal
from rewind_v1 import camera_gauge as gauge,camera_control as control,cross_data as data,cross_inference as inference
from rewind_v1.camera_gauge_cli import package
from rewind_v1.cross_recovery import prepare_attempt,publish
from scripts.build_v12_camera_gauge import notebook_document
import test_camera_control as helpers
import test_cross_scenario as cross_helpers
import test_prepare


ROOT=Path(__file__).resolve().parents[1]
CITY=ROOT/'outputs/v12_cross_scenario_20261115/user_test/completed_001/results.zip'
CONTROL=ROOT/'outputs/v12_camera_control_20261115/user_test/completed_001/results.zip'


class GaugeTests(TestCase):
    def setUp(self):
        (ROOT/'outputs/test_tmp').mkdir(parents=True,exist_ok=True)
        cross_helpers.CrossScenarioTests.setUp(self)

    def models(self):return cross_helpers.CrossScenarioTests.models(self)

    def test_actual_reference_exact_keys_and_tamper_rejected(self):
        if not CITY.is_file() or not CONTROL.is_file():self.skipTest('immutable reference files unavailable locally')
        root=self.root/'reference';refs=gauge.load_references(CITY,CONTROL,root)
        self.assertEqual(len(refs['clips']),6);self.assertEqual(sum(len(c['camera']['frames']) for c in refs['clips']),300)
        p=root/'control/cameras/cameras.json';d=json.loads(p.read_bytes());d['clips'].reverse()
        d=seal({k:v for k,v in d.items() if k!='content_sha256'});p.write_bytes(canonical(d))
        receipt=json.loads((root/'control/recovery_receipt.json').read_bytes())
        receipt['files']['cameras/cameras.json']={'bytes':p.stat().st_size,'sha256':sha(p.read_bytes())}
        (root/'control/recovery_receipt.json').write_bytes(canonical(receipt))
        with self.assertRaises(ValueError):gauge.checked_references(root)
        with self.assertRaisesRegex(ValueError,'zip_size'):gauge.load_references(CONTROL,CITY,self.root/'swapped')

    def camera_rows(self):
        rows=[]
        for frame in range(5):
            angle=.2+frame*.1;c,s=np.cos(angle),np.sin(angle)
            r=np.array([[c,-s,0],[s,c,0],[0,0,1.]])
            centers=np.array([[262000+frame*80,1500,-2300],[262120+frame*80,1500,-2300.]])
            t=-np.einsum('ij,vj->vi',r,centers)
            poses={side:{'rotation_matrix':r.tolist(),'translation':t[i].tolist()} for i,side in enumerate(('left','right'))}
            rows.append({'key':['seq',frame],'rotation_matrix':[r.tolist(),r.tolist()],
                'world_to_camera_translation_mm':t.tolist(),'encoded_camera_18':encode_camera(poses,100.).flatten().tolist()})
        return rows

    def test_common_gauge_invariants_units_first_frame_only_and_float32(self):
        rows=self.camera_rows();original=copy.deepcopy(rows)
        for name in gauge.VARIANTS:
            delta=gauge.camera_delta(rows,name);encoded,audit=gauge.transform_gauge(rows,delta)
            self.assertEqual(encoded.shape,(5,2,9));self.assertEqual(rows,original)
            self.assertLess(audit['max_residuals']['projection_camera_space_max_abs_mm'],1e-6)
            self.assertLess(audit['max_residuals']['relative_stereo_translation_max_abs_mm'],1e-6)
            self.assertLess(audit['max_residuals']['float32_translation_roundtrip_max_abs_mm'],.1)
            if name=='first_midpoint':
                self.assertLess(float(np.linalg.norm(np.asarray(audit['transformed_translation_mm'])[0],axis=1).max()),61)
            if name=='world_x_plus_100m':np.testing.assert_array_equal(delta,[100000,0,0])
        altered=copy.deepcopy(rows);altered[1]['world_to_camera_translation_mm']=[[9e9]*3]*2
        np.testing.assert_array_equal(gauge.camera_delta(rows,'first_midpoint'),gauge.camera_delta(altered,'first_midpoint'))
        with self.assertRaises(ValueError):gauge.camera_delta(rows,'gt_best')
        with self.assertRaises(ValueError):gauge.transform_gauge(rows,[float('nan'),0,0])

    def test_repeat_threshold_and_cross_run_mismatch_are_distinct(self):
        base={'body':torch.ones(2,3,3)};repeat={'body':base['body']+.001}
        tiny={'body':base['body']+.002};large={'body':base['body']+100.}
        self.assertEqual(gauge.paired_metrics(tiny,base,repeat)['body']['coordinates_above_repeat_aware_tolerance'],0)
        self.assertEqual(gauge.paired_metrics(large,base,repeat)['body']['coordinates_above_repeat_aware_tolerance'],18)
        self.assertEqual(gauge.repeat_stability(base,repeat)['status'],'stable_within_fixed_tolerance')
        self.assertEqual(gauge.repeat_stability(base,large)['status'],'inconclusive_repeat_unstable')
        ref={'prediction':{'body':large['body'].tolist()},'predicted_2d_native':torch.zeros(2,2,3,2).tolist()}
        result=gauge.reference_comparison(base,torch.zeros(2,2,3,2),ref)
        self.assertEqual(result['status'],'reference_mismatch')

    def test_default_off_notebook_compile_and_shared_retention(self):
        nb=notebook_document({})
        for i,c in enumerate(nb['cells']):
            if c['cell_type']!='code':continue
            src=''.join(c['source']);compile(src,f'cell-{i}','exec')
            if c['id']!='bootstrap':exec(src,{'run_gauge':lambda *a:self.fail('default run'),'display_saved':lambda:self.fail('default display')})
        src=''.join(nb['cells'][1]['source'])
        self.assertIn("'--control-reference',str(CONTROL_REFERENCE)",src)
        self.assertIn('rewind_v1.camera_gauge_cli',src);self.assertIn('range(6)',src)
        self.assertIn('process.kill();process.wait(timeout=30)',src)
        self.assertIn("if stage in ('select','prepare')",src)

    def make_references(self):
        seqs=helpers.CameraControlTests.archvis(self)
        fixture=test_prepare.PreparationTests();fixture.setUp();self.addCleanup(fixture.doCleanups)
        png=io.BytesIO();Image.new('RGB',(1024,1024),(40,80,120)).save(png,format='PNG')
        for seq in self.sequences+seqs:
            for frame in range(50):
                raw=fixture.add_frame(seq,frame).read_bytes();(self.pose/seq/'json'/f'frame_{frame}.json').write_bytes(raw)
                for side in ('left','right'):(self.rgb/seq/'fisheye_final_image'/('camera_'+side)/f'final_{frame}.png').write_bytes(png.getvalue())
        old=self.root/'old';old.mkdir();plan=data.select(self.inputs,self.rgb,self.pose,old/'selection',context=gauge.context())
        prepared=data.prepare(self.inputs,old/'selection/selection.json',plan['content_sha256'],old/'prepared',context=gauge.context())
        refs={'clips':[],'binding':self.binding,'model_weights':{'fixture':True},'implementation':{'fixture':True},
            'reference_environments':{'fixture':{'device':'cpu','torch':str(torch.__version__)}},
            'reference_report_sha256':{'city':'a'*64,'control':'b'*64}}
        idx=[self.spec['joint_names'].index(n) for n in self.spec['body_joint_names']]
        fingers,wrists=inference._hand_indices(self.spec['joint_names'],self.spec)
        for i,(a,b) in enumerate(zip(plan['clips'],prepared['clips'])):
            tensor=torch.load(old/'prepared'/f'clip-{i:02d}/clip.pt',weights_only=True)
            pred={'whole':tensor['target'],'body':tensor['target'][:,idx],'hand_global':tensor['target'][:,fingers],
                  'hand_local':tensor['target'][:,fingers]-tensor['target'][:,wrists]}
            prediction=seal({'keys':[[a['sequence'],r['frame']] for r in a['frames']],
                'prediction':{k:v.tolist() for k,v in pred.items()},'gt_target_device_mm':tensor['target'].tolist(),
                'gt_target_valid':tensor['target_valid'].tolist(),'predicted_2d_native':torch.zeros(50,2,62,2).tolist()})
            cameras=[]
            for row in b['frames']:
                native=json.loads((self.pose/a['sequence']/'json'/f"frame_{row['frame']}.json").read_bytes())
                cameras.append(control.camera_record(native,self.spec,self.calib,a['sequence'],row['frame'],row['files']['pose']))
            refs['clips'].append({'selection':a,'prepared':b,'prediction':prediction,
                'camera':{'frames':cameras},'origin':'control' if i<2 else 'city','origin_number':i if i<2 else i-2})
        return refs

    def test_positive_six_clips_all_variants_render_and_complete_recovery(self):
        refs=self.make_references();run=self.root/'run';run.mkdir()
        with patch.object(data,'_listing',side_effect=AssertionError('no new directory scan')):
            plan=gauge.fixed_plan(self.inputs,refs,self.rgb,self.pose,run/'selection')
        gauge.checked_plan(run/'selection/selection.json',plan['content_sha256'],refs)
        prepared=data.prepare(self.inputs,run/'selection/selection.json',plan['content_sha256'],run/'prepared',context=gauge.context())
        models,cfg=self.models()
        class Frontend(torch.nn.Module):
            def forward(self,images):
                n=len(images);return torch.zeros(n,2,62,1,1),torch.full((n,2,62,2),64.),torch.full((n,2,62),.8)
        models['frontend']=Frontend().eval();models.pop('spec')
        calls=[];actual=inference.sample_features
        def tracked(models,features,camera,keys,config,device):
            calls.append((len(keys),sha(features.contiguous().numpy().tobytes())));return actual(models,features,camera,keys,config,device)
        with patch.object(inference.view,'_models',return_value=(models,refs['model_weights'],{'config':cfg},refs['implementation'])), \
                patch.object(inference,'sample_features',side_effect=tracked),patch.object(torch.optim,'Adam',side_effect=AssertionError('no training')):
            result=gauge.evaluate(self.inputs,plan,run/'prepared',prepared['content_sha256'],refs,self.root/'fixture.zip',run/'inference',torch.device('cpu'))
        self.assertEqual(len(result['clips']),6);self.assertTrue(result['repeat_stable_all'])
        self.assertFalse(result['reference_match_all'])  # fixture reference differs; paired evidence still reported.
        for i in range(6):
            self.assertEqual(len({s for n,s in calls[i*5:(i+1)*5] if n==50}),1)
            root=run/'inference'/f'clip-{i:02d}'
            for variant in gauge.VARIANTS:self.assertTrue((root/(variant+'.json')).is_file())
            with Image.open(root/'comparison.gif') as image:
                self.assertEqual(image.n_frames,50);self.assertEqual(image.size,(952,810))
            self.assertEqual(len(list(root.glob('frame-*.png'))),3)
        archive=package(run,self.root/'results.zip')
        import zipfile
        with zipfile.ZipFile(archive['archive']) as z:
            names=set(z.namelist())
            self.assertEqual(sum(n.endswith('comparison.gif') for n in names),6)
            self.assertEqual(sum(n.endswith(tuple(v+'.json' for v in gauge.VARIANTS)) for n in names),24)
            self.assertTrue(all(f'prepared/clip-{i:02d}/manifest.json' in names for i in range(6)))
            self.assertFalse(any('/raw/' in n or n.endswith('.pt') for n in names))
            self.assertEqual(len(z.namelist()),len(names))
        drive=self.root/'drive';drive.mkdir();attempt=prepare_attempt(drive,'run-fixture','inference',namespace=gauge.RUN_ID)
        self.assertEqual(publish(archive,attempt,run_id=gauge.RUN_ID)['state'],'verified')
        # Optional expected SHA must reject a wrong observed payload even with unchanged metadata.
        bad=copy.deepcopy(plan);bad['clips'][0]['frames'][0]['files']['left']['sha256']='0'*64
        bad=seal({k:v for k,v in bad.items() if k!='content_sha256'});p=self.root/'bad.json';p.write_bytes(canonical(bad))
        with self.assertRaisesRegex(ValueError,'fixed_source_hash_mismatch'):
            data.prepare(self.inputs,p,bad['content_sha256'],self.root/'bad-prepared',context=gauge.context())
        changed=copy.deepcopy(plan);changed['clips'].reverse();changed=seal({k:v for k,v in changed.items() if k!='content_sha256'})
        p.write_bytes(canonical(changed))
        with self.assertRaisesRegex(ValueError,'plan_order_keys'):gauge.checked_plan(p,changed['content_sha256'],refs)
