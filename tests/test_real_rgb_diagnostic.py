"""실제 RGB 진단의 resize/cache/예측 전용 입력 계약. 학습 update는 수행하지 않는다."""
import copy
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]
PREPARED=ROOT/'outputs/real_stereo_sample_20261102/root/prepared'
try:
    import torch
    from PIL import Image
except (ImportError,OSError):AVAILABLE=False
else:AVAILABLE=(PREPARED/'pose/candidate.json').is_file()
if AVAILABLE:
    import rewind_m1.real_rgb_diagnostic as rgb
    from rewind_archive.contracts import digest
    from rewind_m1.frontend import weights_hash,read_prediction_cache
    from rewind_m1.data import DataContractError,load_document
    from rewind_m1.data import content_hash
    from rewind_m1.local_diagnostic import _preserve_state
    from rewind_m1.real_gt2d_diagnostic import load_diagnostic,DiagnosticTrainer,CONFIG,equal_state
    from rewind_m1.training import read_checkpoint,save_checkpoint


@unittest.skipUnless(AVAILABLE,'검토된 실제 Serena sample과 torch/Pillow 필요')
class RealRGBDiagnosticTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pins={'candidate_sha256':hashlib.sha256((PREPARED/'pose/candidate.json').read_bytes()).hexdigest(),
            'rgb_manifest_sha256':hashlib.sha256((PREPARED/'rgb/manifest.json').read_bytes()).hexdigest()}
        root=ROOT/'outputs/real_rgb_diagnostic_20261102/test_tmp';root.mkdir(parents=True,exist_ok=True)
        cls.tmp=tempfile.TemporaryDirectory(dir=root);cls.addClassCleanup(cls.tmp.cleanup)
        cls.weights=Path(cls.tmp.name)/'weights.pt'
        with _preserve_state():
            cls.data=load_diagnostic(PREPARED,diagnostic_only=True,**cls.pins)
            cls.images,cls.gt,cls.valid,cls.manifest=rgb.load_images(PREPARED,cls.data)
            cls.front=rgb.FrontendTrainer(cls.images,cls.gt,cls.valid,cls.manifest)
            cls.fingerprint=weights_hash(cls.front.model)
            cls.cache=rgb.build_cache(cls.front.model,cls.images,cls.data,cls.manifest,cls.weights)

    def predictions(self,cache=None):
        return rgb.cache_predictions(cache or self.cache,self.data,self.manifest,self.weights,self.fingerprint)

    def test_endpoint_resize_native_manifest_and_supervision_masks(self):
        self.assertEqual(tuple(self.images.shape),(2,2,3,64,64));self.assertEqual(rgb.RESIZE['gt_to_network_scale'],[63/1023]*2)
        self.assertEqual(self.manifest['source_views'][0]['views']['left']['image_size'],[1024,1024])
        self.assertEqual(int(self.valid.sum()),248)
        with _preserve_state():
            front=rgb.FrontendTrainer(self.images,self.gt,self.valid,self.manifest)
            self.assertEqual(front.fit()['inside_joint_views'],206)
        scaled=torch.tensor([0.,1023.])*torch.tensor(63/1023)
        self.assertTrue(torch.allclose(scaled,torch.tensor([0.,63.])))
        self.assertEqual(int(self.data.tensors.target_valid.sum()),124)

    def test_cache_reload_and_native_coordinate_transform(self):
        with _preserve_state():
            before=self.predictions();after=self.predictions(json.loads(json.dumps(self.cache)))
        self.assertEqual(before,after)
        self.assertEqual(self.cache['network_cache']['image_sizes'],[[64,64],[64,64]])
        self.assertEqual(self.cache['native_cache']['image_sizes'],[[1024,1024],[1024,1024]])
        for small,native in zip(self.cache['network_cache']['records'],self.cache['native_cache']['records']):
            for side in ('left','right'):
                for a,b in zip(small['prediction'][side]['coordinates'],native['prediction'][side]['coordinates']):self.assertEqual(b,[a[0]*1023/63,a[1]*1023/63])
        with self.assertRaises(DataContractError):read_prediction_cache(self.cache,None,'0'*64)

    def test_hash_source_resize_weights_and_key_tampering_rejected(self):
        for key in ('weights_state_sha256','weights_file_sha256'):
            bad=copy.deepcopy(self.cache);bad[key]='0'*64;bad['content_sha256']=digest({k:v for k,v in bad.items() if k!='content_sha256'})
            with self.assertRaisesRegex(ValueError,'cache_provenance'):self.predictions(bad)
        bad=copy.deepcopy(self.cache);bad['input_manifest']['source_views'][0]['views']['left']['sha256']='0'*64
        bad['content_sha256']=digest({k:v for k,v in bad.items() if k!='content_sha256'})
        with self.assertRaisesRegex(ValueError,'cache_provenance'):self.predictions(bad)
        bad=copy.deepcopy(self.cache);bad['native_cache']['records'][0]['frame_index']=999
        with self.assertRaisesRegex(ValueError,'schema_hash'):self.predictions(bad)
        bad=copy.deepcopy(self.cache)
        for key in ('network_cache','native_cache'):
            bad[key]['records'].pop();bad[key]['content_sha256']=content_hash({k:v for k,v in bad[key].items() if k!='content_sha256'})
        bad['content_sha256']=digest({k:v for k,v in bad.items() if k!='content_sha256'})
        with self.assertRaisesRegex(ValueError,'cache_keys'):self.predictions(bad)
        bad=copy.deepcopy(self.cache);native=bad['native_cache']
        native['records'][0]['prediction']['left']['coordinates'][0][0]+=1
        native['content_sha256']=content_hash({k:v for k,v in native.items() if k!='content_sha256'})
        bad['content_sha256']=digest({k:v for k,v in bad.items() if k!='content_sha256'})
        with self.assertRaisesRegex(ValueError,'cache_transform'):self.predictions(bad)
        weights=torch.load(self.weights,map_location='cpu',weights_only=True)
        for key,value in (('joint_names',list(reversed(weights['joint_names']))),('data_kind','synthetic_fixture'),
                          ('manifest_hash','0'*64),('frontend_version','wrong'),('frontend_checkpoint_sha256','0'*64)):
            wrong=copy.deepcopy(weights);wrong[key]=value;path=Path(self.tmp.name)/'wrong.pt';torch.save(wrong,path)
            bad=copy.deepcopy(self.cache);bad['weights_file_sha256']=rgb.sha(rgb.read(path));bad['content_sha256']=digest({k:v for k,v in bad.items() if k!='content_sha256'})
            with self.assertRaisesRegex(ValueError,'weights_contract'):rgb.cache_predictions(bad,self.data,self.manifest,path,self.fingerprint)

    def test_missing_prediction_has_no_gt_fallback_and_invalid_mask_does_not_mask_target(self):
        with _preserve_state():predictions=self.predictions()
        missing=copy.deepcopy(predictions);del missing[next(iter(missing))]
        with self.assertRaisesRegex(ValueError,'predictions_required'):rgb.predicted_data(missing,self.data,self.cache)
        invalid=copy.deepcopy(predictions);first=invalid[self.data.tensors.anchors[0]]['left']
        first['valid_mask'][0]=False;first['coordinates'][0]=None;first['confidence'][0]=None
        packed=rgb.predicted_data(invalid,self.data,self.cache)
        self.assertTrue(torch.equal(packed.tensors.features[0,0,:7],torch.zeros(7)))
        self.assertEqual(int(packed.tensors.target_valid.sum()),124)

    def test_gt_feature_change_has_no_effect_on_cached_prediction_input(self):
        with _preserve_state():predictions=self.predictions()
        first=rgb.predicted_data(predictions,self.data,self.cache)
        changed=copy.deepcopy(self.data);changed.tensors.features+=99999
        second=rgb.predicted_data(predictions,changed,self.cache)
        self.assertTrue(torch.equal(first.tensors.features,second.tensors.features))
        self.assertTrue(torch.equal(first.tensors.target_valid,self.data.tensors.target_valid))
        self.assertEqual(first.contract['input_source'],'predicted_2d')

    def test_native_2d_error_separates_outside_without_zeroing(self):
        with _preserve_state():metrics=rgb.errors_2d(self.predictions(),self.data,self.gt,self.valid)
        self.assertEqual((metrics['inside_count'],metrics['all_valid_count'],metrics['outside_count']),(206,248,42))
        self.assertGreater(metrics['outside_mean_px'],0.)
        self.assertAlmostEqual(metrics['all_valid_mean_px'],(metrics['inside_mean_px']*206+metrics['outside_mean_px']*42)/248,places=3)
        inside=self.valid&(self.gt[...,0]>=0)&(self.gt[...,0]<=1023)&(self.gt[...,1]>=0)&(self.gt[...,1]<=1023)
        with _preserve_state():empty_outside=rgb.errors_2d(self.predictions(),self.data,self.gt,inside)
        self.assertEqual(empty_outside['outside_count'],0);self.assertIsNone(empty_outside['outside_mean_px'])

    def test_frontend_and_predicted_checkpoint_roles_and_state_reload(self):
        with _preserve_state():
            front=rgb.FrontendTrainer(self.images,self.gt,self.valid,self.manifest);state=rgb.checkpoint(front,'frontend')
            restored=rgb.FrontendTrainer(self.images,self.gt,self.valid,self.manifest);rgb.resume(restored,state,'frontend')
            self.assertTrue(equal_state(state,rgb.checkpoint(restored,'frontend')))
            pred=rgb.predicted_data(self.predictions(),self.data,self.cache);trainer=rgb.PredictedTrainer(pred);checkpoint=trainer.checkpoint()
            self.assertEqual(checkpoint['data_identity']['input_source'],'predicted_2d')
            other=rgb.PredictedTrainer(pred);other.resume(checkpoint);self.assertTrue(equal_state(checkpoint,other.checkpoint()))
            with self.assertRaisesRegex(ValueError,'checkpoint_role'):rgb.resume(front,checkpoint,'frontend')
            with self.assertRaisesRegex(ValueError,'checkpoint_role'):DiagnosticTrainer(self.data,CONFIG).resume(checkpoint)
            path=Path(self.tmp.name)/'predicted.pt';save_checkpoint(checkpoint,path)
            with self.assertRaisesRegex(ValueError,'checkpoint'):read_checkpoint(path)

    def test_explicit_diagnostic_approval_and_general_gate_unchanged(self):
        with self.assertRaisesRegex(ValueError,'diagnostic_only'):rgb.run(PREPARED,Path(self.tmp.name)/'not-started',diagnostic_only=False,**self.pins)
        candidate=json.loads((PREPARED/'pose/candidate.json').read_bytes())
        with self.assertRaises(DataContractError):load_document(candidate,1)
        self.assertFalse(candidate['training_eligibility']['train_ready'])


if __name__=='__main__':unittest.main()
