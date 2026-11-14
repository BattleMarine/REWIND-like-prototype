"""Bounded local fixtures, never Drive data or cross-scenario performance evidence."""

import copy
import io
import json
from pathlib import Path
import tempfile
from unittest import TestCase
from unittest.mock import patch

from PIL import Image
import torch

from rewind_archive.contracts import canonical
from rewind_prepare.converter import sha
from rewind_prepare.temporal_selection import seal
from rewind_v1 import cross_data as data, cross_inference as inference
from rewind_v1.cross_recovery import package,prepare_attempt,publish
from rewind_v1.teacher import V1Body,V1Hand
from scripts.build_v12_cross_scenario import notebook_document
import test_prepare


ROOT=Path(__file__).resolve().parents[1]


class CrossScenarioTests(TestCase):
    def setUp(self):
        temporary=tempfile.TemporaryDirectory(dir=ROOT/'outputs/test_tmp');self.addCleanup(temporary.cleanup)
        self.root=Path(temporary.name);self.rgb=self.root/'rgb';self.pose=self.root/'pose'
        self.rgb.mkdir();self.pose.mkdir();self.inputs=self.root/'inputs';self.inputs.mkdir();(self.inputs/'split').mkdir()
        self.spec=json.loads((ROOT/'configs/real_archvis62_v1.json').read_bytes())
        self.sequences=[f'{env}/Day/person/glasses/{i:03d}/Motion__1_' for env in data.ENVIRONMENTS for i in (1,2)]
        self.train='ArchVisInterior_ArchVis_RT/Day/person/glasses/001/Motion__1_'
        self.test=data.ENVIRONMENTS[0]+'/Day/person/glasses/099/Official_Test'
        for name,rows in (('train',[self.train]),('validation',self.sequences),('test',[self.test])):
            raw=('\n'.join(rows)+'\n').encode();(self.inputs/'split'/f'{name}.txt').write_bytes(raw)
            self.spec['pinned_sources']['split_sha256'][name]=sha(raw)
        self.binding={'spec_sha256':'f'*64,'calibration_sha256':'e'*64,'split_files':self.spec['pinned_sources']['split_sha256']}
        self.calib={'size':[1024,1024],'image_center':[512.,512.],'polynomialW2C':[500.,100.],'affine':[1.,0.,0.]}
        self.asset_patch=patch.object(data,'assets',return_value=({'keys':{'train':[(self.train,0)]}},self.spec,self.calib,self.binding))
        self.asset_patch.start();self.addCleanup(self.asset_patch.stop)
        for seq in self.sequences:
            for role in ('left','right','pose'):
                folder=self.pose/seq/'json' if role=='pose' else self.rgb/seq/'fisheye_final_image'/('camera_'+role)
                folder.mkdir(parents=True)
                for frame in range(50):
                    (folder/(f'frame_{frame}.json' if role=='pose' else f'final_{frame}.png')).write_bytes(b'x')

    def selected(self,name='selection'):
        return data.select(self.inputs,self.rgb,self.pose,self.root/name)

    def test_validation_only_sealed_selection_quota_gap_and_overlap(self):
        plan=self.selected()
        self.assertEqual(plan['status'],'complete');self.assertEqual(len(plan['clips']),4)
        self.assertEqual(plan['payload_bytes_read'],0)
        self.assertTrue(all(c['split']=='validation' for c in plan['clips']))
        self.assertTrue(plan['clips'][0]['overlap_proxy']['train_person_matches'])
        self.assertTrue(plan['clips'][0]['overlap_proxy']['train_motion_label_matches'])
        # Losing one frame leaves no50-frame run; one clip for this environment is explicitly partial.
        (self.pose/self.sequences[0]/'json/frame_25.json').unlink()
        partial=self.selected('partial');self.assertEqual(partial['status'],'partial_availability')
        self.assertEqual(partial['selected_counts'][data.ENVIRONMENTS[0]],1)
        # No validation candidate left is a failure, never a train fallback.
        (self.pose/self.sequences[1]/'json/frame_25.json').unlink()
        with self.assertRaisesRegex(ValueError,'zero_available'):self.selected('empty')

    def test_self_sealed_test_membership_bad_roles_and_model_pins_rejected(self):
        plan=self.selected();path=self.root/'modified.json'
        for operation in ('test','role','model','environment'):
            changed=copy.deepcopy(plan)
            if operation=='test':
                old=changed['clips'][0]['sequence'];changed['clips'][0]['sequence']=self.test
                for row in changed['clips'][0]['frames']:
                    for item in row['files'].values():item['path']=item['path'].replace(old,self.test)
            elif operation=='role':del changed['clips'][0]['frames'][0]['files']['pose']
            elif operation=='model':changed['model_zip_sha256']='0'*64
            else:changed['clips'][0]['environment']='Unapproved'
            changed=seal({k:v for k,v in changed.items() if k!='content_sha256'});path.write_bytes(canonical(changed))
            with self.subTest(operation=operation),self.assertRaises(ValueError):
                data.prepare(self.inputs,path,changed['content_sha256'],self.root/('bad-'+operation))

    def test_payload_geometry_failure_does_not_replace_selected_sequence(self):
        plan=self.selected()
        with self.assertRaises(Exception):
            data.prepare(self.inputs,self.root/'selection/selection.json',plan['content_sha256'],self.root/'prepared')
        failure=json.loads((self.root/'prepared/failure.json').read_bytes())
        self.assertEqual(failure['completed_clips'],[])
        self.assertFalse((self.root/'prepared/prepared.json').exists())
        self.assertEqual(json.loads((self.root/'selection/selection.json').read_bytes())['content_sha256'],plan['content_sha256'])

    def models(self):
        cfg=json.loads((ROOT/'configs/rewind_v12_full_training.json').read_bytes())
        cfg['teacher'].update(hidden=8,heads=2,graph_layers=1,temporal_layers=1,diffusion_steps=20)
        torch.manual_seed(71);models={}
        for role in ('body','hand'):
            names=inference.view._group_names(self.spec,role)
            edges=[(names.index(a),names.index(b)) for a,b in self.spec['edges'] if a in names and b in names]
            model=V1Body(cfg['teacher'],len(names),edges) if role=='body' else V1Hand(cfg['teacher'],len(names),edges,len(self.spec['upper_joint_names']))
            models[role]=model.eval().requires_grad_(False)
        models['spec']=self.spec
        return models,cfg

    def test_causality_sequence_reset_and_no_optimizer(self):
        models,cfg=self.models();features=torch.randn(50,62*14);camera=torch.randn(50,2,9)
        keys=[(self.sequences[0],i) for i in range(50)]
        with patch.object(torch.optim,'Adam',side_effect=AssertionError('no training')):
            whole=inference.sample_features(models,features,camera,keys,cfg,torch.device('cpu'))
            prefix=inference.sample_features(models,features[:25],camera[:25],keys[:25],cfg,torch.device('cpu'))
            changed=features.clone();changed[25:]=1000
            future=inference.sample_features(models,changed,camera,keys,cfg,torch.device('cpu'))
            inference.sample_features(models,features,camera,[(self.sequences[1],i) for i in range(50)],cfg,torch.device('cpu'))
            reset=inference.sample_features(models,features,camera,keys,cfg,torch.device('cpu'))
        for name in whole:
            torch.testing.assert_close(whole[name][:25],prefix[name],atol=.01,rtol=1e-5)
            torch.testing.assert_close(whole[name][:25],future[name][:25],atol=.01,rtol=1e-5)
            self.assertTrue(torch.equal(whole[name],reset[name]))

    def test_missing_wrist_local_mask_keeps_global_metric_and_joint_weighting(self):
        gt=torch.zeros(50,62,3);mask=torch.ones(50,62,dtype=torch.bool)
        mask[0,self.spec['joint_names'].index('hand_l')]=False
        tensors={'target':gt,'target_valid':mask,'left_xy':torch.full((50,62,2),512.),'right_xy':torch.full((50,62,2),512.),
                 'left_xy_valid':torch.ones(50,62,dtype=torch.bool),'right_xy_valid':torch.ones(50,62,dtype=torch.bool)}
        prediction={'body':torch.ones(50,22,3),'hand_local':torch.ones(50,40,3),'hand_global':torch.ones(50,40,3)}
        metrics,_,_=inference.metric_sums(prediction,torch.zeros(50,62*14),tensors,self.spec)
        total=inference.aggregate(metrics)
        self.assertEqual(total['hand_local']['count'],1980);self.assertEqual(total['hand_global']['count'],2000)
        self.assertAlmostEqual(total['body']['mean'],3**.5,places=6)

    def test_recovery_excludes_raw_tensors_model_and_hashes_reports(self):
        run=self.root/'run';(run/'prepared/clip-00/raw/pose').mkdir(parents=True)
        (run/'prepared/clip-00/raw/pose/frame.json').write_bytes(b'private raw')
        (run/'prepared/clip-00/clip.pt').write_bytes(b'tensor')
        (run/'model.zip').write_bytes(b'model')
        (run/'prepared/clip-00/manifest.json').write_bytes(b'{"source":"fixture"}')
        (run/'notebook_status.json').write_bytes(b'{"status":"failed"}')
        pin=package(run,self.root/'result.zip')
        import zipfile
        with zipfile.ZipFile(pin['archive']) as z:
            self.assertEqual(set(z.namelist()),{'prepared/clip-00/manifest.json','notebook_status.json','recovery_receipt.json'})
            receipt=json.loads(z.read('recovery_receipt.json'))
            for name,item in receipt['files'].items():self.assertEqual(item['sha256'],sha(z.read(name)))

    def test_positive_prepare_infer_gif_and_local_copy_readback(self):
        fixture=test_prepare.PreparationTests();fixture.setUp();self.addCleanup(fixture.doCleanups)
        png=io.BytesIO();Image.new('RGB',(1024,1024),(40,80,120)).save(png,format='PNG')
        for seq in (self.sequences[0],self.sequences[2]):
            for frame in range(50):
                native=fixture.add_frame(seq,frame).read_bytes()
                (self.pose/seq/'json'/f'frame_{frame}.json').write_bytes(native)
                for side in ('left','right'):
                    (self.rgb/seq/'fisheye_final_image'/('camera_'+side)/f'final_{frame}.png').write_bytes(png.getvalue())
        for seq in (self.sequences[1],self.sequences[3]):(self.pose/seq/'json/frame_25.json').unlink()
        run=self.root/'run';run.mkdir()
        plan=data.select(self.inputs,self.rgb,self.pose,run/'selection')
        prepared=data.prepare(self.inputs,run/'selection/selection.json',plan['content_sha256'],run/'prepared')
        self.assertEqual(len(prepared['clips']),2)
        self.assertGreater(prepared['source_bytes_read'],0)
        models,cfg=self.models()
        class Frontend(torch.nn.Module):
            def forward(self,images):
                n=len(images);xy=torch.full((n,2,62,2),64.);confidence=torch.full((n,2,62),.8)
                return torch.zeros(n,2,62,1,1),xy,confidence
        models['frontend']=Frontend().eval();models.pop('spec')
        source_assets=data.assets(self.inputs)
        with patch.object(inference,'assets',return_value=source_assets), \
             patch.object(inference.view,'_models',return_value=(models,{'fixture':True},{'config':cfg},{'fixture':True})), \
             patch.object(torch.optim,'Adam',side_effect=AssertionError('no optimizer')):
            report=inference.infer(self.inputs,run/'selection/selection.json',plan['content_sha256'],run/'prepared',
                                   prepared['content_sha256'],self.root/'fixture-model.zip',run/'inference',torch.device('cpu'))
        self.assertEqual(report['status'],'partial_availability')
        for number in range(2):
            folder=run/'inference'/f'clip-{number:02d}'
            clip=json.loads((folder/'predictions.json').read_bytes())
            self.assertEqual(len(clip['frame_metrics']),50)
            self.assertEqual(clip['projection_scope'],'single shared transform over all50frame GT+pred points')
            for artifact in clip['artifacts']:
                with Image.open(folder/artifact['name']) as image:
                    self.assertEqual(image.size,(676,1144))
                    if artifact['name']=='clip.gif':self.assertEqual(image.n_frames,50)
            self.assertEqual(len(list(folder.glob('frame-*.png'))),3)
        archive=package(run,self.root/'results.zip');fake_drive=self.root/'fake-drive';fake_drive.mkdir()
        attempt=prepare_attempt(fake_drive,'run-fixture','inference')
        receipt=publish(archive,attempt)
        self.assertEqual(receipt['state'],'verified')
        self.assertEqual(sha((attempt/'results.zip').read_bytes()),archive['sha256'])
        with self.assertRaises(FileExistsError):publish(archive,attempt)

    def test_notebook_default_noop_and_all_cells_compile(self):
        notebook=notebook_document({})
        for i,cell in enumerate(notebook['cells']):
            if cell['cell_type']!='code':continue
            code=''.join(cell['source']);compile(code,f'cell-{i}','exec')
            if i!=1:
                def forbidden(*args,**kwargs):raise AssertionError('default execution')
                exec(code,{'phase':forbidden})
