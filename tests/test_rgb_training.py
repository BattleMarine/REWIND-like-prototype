"""합성2D trainer의 resize·재개·native cache/join. 실제 학습/원격 요청 없음."""
import contextlib
import copy
import io
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch

from test_rgb_shards import fixture
from test_shard_training import equal
from rewind_archive.contracts import digest
from rewind_m1.data import load_document
from rewind_m1.frontend import heatmap_loss
from rewind_m1.local_rgb import LocalStereoRGB
from rewind_m1.prediction_join import join_predictions
from rewind_m1.rgb_frontend_artifacts import (build_cache, export_weights, load_weights, preprocessing_version,
    scale_xy)
from rewind_m1.rgb_shards import build_catalog, open_rgb
from rewind_m1.rgb_training import RGBTrainer
from rewind_m1.rgb_training_cli import main, read_checkpoint
from rewind_m1.training import rng_state, save_checkpoint

ROOT=Path(__file__).resolve().parents[1]


class RGBTrainingTests(unittest.TestCase):
    def setUp(self):
        parent=ROOT/'outputs'/'test_tmp'; parent.mkdir(parents=True,exist_ok=True)
        self.temp=tempfile.TemporaryDirectory(dir=parent); self.addCleanup(self.temp.cleanup)
        self.base=Path(self.temp.name); self.specs=fixture(self.base)
        self.catalog=build_catalog(self.specs,self.base)
        self.config={'seed':43,'batch_size':3,'learning_rate':.005,'scheduler_step':2,'scheduler_gamma':.8,
            'amp':False,'training_image_size':[4,4],'frontend_version':'small_cnn_rgb_shards_test_v1',
            'max_native_tensor_bytes':32*1024**2,'max_heatmap_elements':32*1024**2}
        self.write_inputs()

    def write_inputs(self):
        (self.base/'catalog.json').write_text(json.dumps(self.catalog),encoding='utf8')
        (self.base/'config.json').write_text(json.dumps(self.config),encoding='utf8')

    def trainer(self): return RGBTrainer(self.config,self.catalog,self.base)

    def cli_train(self, output, *args):
        with contextlib.redirect_stdout(io.StringIO()):
            return main(['train','--config',str(self.base/'config.json'),'--catalog',str(self.base/'catalog.json'),
                         '--output',str(output),*args])

    def test_resume_equal_mid_chunk_chunk_and_epoch_boundaries_tail_counts(self):
        for batch_size,total,cuts in ((1,9,(1,2,4)),(3,5,(1,2,4))):
            self.config['batch_size']=batch_size
            trainer=self.trainer(); paths={}
            for _ in range(total):
                trainer.train_step()
                if trainer.step in cuts:
                    path=self.base/f'b{batch_size}_{trainer.step}.pt'; save_checkpoint(trainer.checkpoint(),path)
                    paths[trainer.step]=path
            expected=copy.deepcopy(trainer.checkpoint())
            self.assertEqual(expected['samples_seen'],total if batch_size==1 else total*2)
            for cut,path in paths.items():
                restored=self.trainer(); restored.resume(read_checkpoint(path)[0])
                while restored.step<total: restored.train_step()
                self.assertTrue(equal(expected,restored.checkpoint()),(batch_size,cut))

    def test_one_native_sample_resize_outside_and_missing_mask(self):
        trainer=self.trainer()
        from rewind_m1.rgb_supervision import LocalRGBSupervision
        original=LocalRGBSupervision.load_batch
        def single(supervision,indices):
            self.assertEqual(len(indices),1)
            return original(supervision,indices)
        with patch.object(LocalRGBSupervision,'load_batch',single):
            batch=trainer._batch(0,'train',[0,1])
        self.assertEqual(tuple(batch['images'].shape),(2,2,3,4,4))
        self.assertFalse(batch['valid'][1,0,1])
        outside=trainer._batch(2,'validation',[1])
        self.assertGreater(float(outside['gt_xy'][0,1,1,0]),3.)
        self.assertTrue(outside['valid'][0,1,1])
        restored=scale_xy(outside['gt_xy'],[4,4],[8,6])
        self.assertTrue(torch.allclose(restored,outside['native_gt_xy'],atol=1e-6))

    def test_validation_weighted_sums_native_all_inside_outside_and_rng(self):
        # validation을3frame으로 만들어 마지막 작은 batch와 관절 누락을 함께 확인한다.
        ppath=self.base/'validation_pose.json'; rpath=self.base/'validation_rgb.json'
        pose=json.loads(ppath.read_bytes()); rgb=json.loads(rpath.read_bytes())
        p=copy.deepcopy(pose['records'][1]); p['frame_index']=2; p['joints']['tip']['camera_left_pts2d']=None
        r=copy.deepcopy(rgb['records'][1]); r['frame_index']=2
        for side,view in r['views'].items():
            target=f"{r['sequence_id']}/fisheye_final_image/camera_{side}/final_2.png"
            shutil.copyfile(self.base/'inputs'/view['path'],self.base/'inputs'/target)
            view['path']=view['member']=target
        pose['records'].append(p); rgb['records'].append(r)
        rgb['content_sha256']=digest({k:v for k,v in rgb.items() if k!='content_sha256'})
        ppath.write_text(json.dumps(pose),encoding='utf8'); rpath.write_text(json.dumps(rgb),encoding='utf8')
        self.catalog=build_catalog(self.specs,self.base); self.config['batch_size']=2
        trainer=self.trainer(); trainer.train_step(); before=copy.deepcopy(trainer.checkpoint())
        result=trainer.evaluate()
        self.assertTrue(equal(before,trainer.checkpoint()))
        self.assertEqual((result['records'],result['valid_joints'],result['inside_joints'],result['outside_joints']),(3,11,9,2))
        total,denominator,distance=0.,0,0.
        trainer.model.eval()
        with torch.no_grad():
            for i in range(3):
                batch=trainer._batch(2,'validation',[i]); logits,xy,_=trainer.model(batch['images'])
                loss,inside=heatmap_loss(logits.float(),batch['gt_xy'],batch['valid'],[4,4])
                count=int(inside.sum())*4; total+=float(loss)*count; denominator+=count
                error=torch.linalg.vector_norm(scale_xy(xy,[4,4],[8,6])-batch['native_gt_xy'],dim=-1)
                distance+=float(error[batch['valid']].sum())
        self.assertEqual(result['heatmap_elements'],denominator)
        self.assertAlmostEqual(result['heatmap_mse'],total/denominator,places=8)
        self.assertAlmostEqual(result['native_distance_sum_pixels'],distance,places=5)
        trainer.model.eval(); rng=rng_state(); trainer.evaluate()
        self.assertFalse(trainer.model.training); self.assertTrue(equal(rng,rng_state()))

    def test_allocation_checks_before_png_and_real_release_rejected(self):
        for config,error in (({**self.config,'max_native_tensor_bytes':1},'rgb_native_sample_tensor_limit'),
                             ({**self.config,'max_heatmap_elements':1},'rgb_heatmap_allocation_limit')):
            with patch.object(LocalStereoRGB,'_decode_into',side_effect=AssertionError('must not allocate/decode')):
                with self.assertRaisesRegex(ValueError,error): RGBTrainer(config,self.catalog,self.base)
        from rewind_m1.data import VERIFICATIONS
        from test_rgb_shards import signed
        real=copy.deepcopy(self.catalog)
        real['contract'].update(data_kind='dataset_sample',verified=dict.fromkeys(VERIFICATIONS,True),
                                verification_evidence=dict.fromkeys(VERIFICATIONS,'declared'))
        with self.assertRaisesRegex(ValueError,'rgb_dataset_sample_release_not_implemented'):
            RGBTrainer(self.config,signed(real),self.base)

    def test_fp32_loss_mock_amp_skip_and_failure_cursor(self):
        trainer=self.trainer(); original=heatmap_loss
        def fp32(logits,*args,**kwargs):
            self.assertEqual(logits.dtype,torch.float32)
            return original(logits,*args,**kwargs)
        trainer.scaler.get_scale=unittest.mock.Mock(side_effect=[8.,4.])
        trainer.scaler.step=unittest.mock.Mock(); trainer.scaler.update=unittest.mock.Mock()
        cursor=trainer.sampler.state_dict(); scheduler=copy.deepcopy(trainer.scheduler.state_dict())
        with patch('rewind_m1.rgb_training.heatmap_loss',fp32): result=trainer.train_step()
        self.assertFalse(result['updated']); self.assertEqual(trainer.sampler.state_dict(),cursor)
        self.assertEqual(trainer.scheduler.state_dict(),scheduler)
        self.assertEqual((trainer.step,trainer.samples_seen,trainer.attempts,trainer.skipped_updates),(0,0,1,1))
        trainer=self.trainer(); cursor=trainer.sampler.state_dict()
        with patch.object(trainer.optimizer,'step',side_effect=OSError('injected_optimizer')):
            with self.assertRaises(OSError): trainer.train_step()
        self.assertEqual(trainer.sampler.state_dict(),cursor)
        with self.assertRaisesRegex(ValueError,'shard_failed_state_cannot_checkpoint'): trainer.checkpoint()

    def test_zero_inside_explicit_failure_and_durable_initial_checkpoint(self):
        for spec in self.specs[:2]:
            path=self.base/spec['pose']; pose=json.loads(path.read_bytes())
            for r in pose['records']:
                for joint in r['joints'].values():
                    for side in ('left','right'): joint['camera_'+side+'_pts2d']=[99.,99.]
            path.write_text(json.dumps(pose),encoding='utf8')
        self.catalog=build_catalog(self.specs,self.base); self.write_inputs()
        output=self.base/'empty_supervision'
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit): self.cli_train(output,'--steps','1')
        state,_=read_checkpoint(output/'last.pt')
        self.assertEqual((state['step'],state['samples_seen']),(0,0))
        self.assertEqual(json.loads((output/'report.json').read_bytes())['status'],'failed')

    def test_resume_rejection_is_atomic_for_source_config_cursor_and_optimizer(self):
        trainer=self.trainer(); trainer.train_step(); expected=copy.deepcopy(trainer.checkpoint())
        for kind in ('source','config','cursor','optimizer','rng'):
            state=copy.deepcopy(expected)
            if kind=='source': state['source']['code_sha256']={}
            if kind=='config': state['config']['training_image_size']=[6,6]
            if kind=='cursor': state['cursor']['catalog_sha256']='a'*64
            if kind=='optimizer': next(iter(state['optimizer']['state'].values()))['exp_avg']=torch.zeros(1)
            if kind=='rng': state['rng']['torch_cpu']=torch.tensor([1],dtype=torch.uint8)
            with self.assertRaises((ValueError,RuntimeError)): trainer.resume(state)
            self.assertTrue(equal(expected,trainer.checkpoint()),kind)

    def test_png_mutation_after_preflight_does_not_advance(self):
        trainer=self.trainer(); shard,_=trainer.sampler.peek(3)
        row=self.catalog['shards'][shard]; rgb=json.loads((self.base/row['rgb']['path']).read_bytes())
        view=rgb['records'][0]['views']['left']; path=self.base/row['rgb']['root']/view['path']
        raw=bytearray(path.read_bytes()); raw[-1]^=1; path.write_bytes(raw)
        with self.assertRaises(ValueError): trainer.train_step()
        self.assertEqual(trainer.step,0); self.assertEqual(trainer.samples_seen,0)

    def test_cli_train_resume_evaluate_export_cache_join_and_weights_rng(self):
        output=self.base/'run'; self.cli_train(output,'--steps','1')
        self.cli_train(output,'--resume',str(output/'last.pt'),'--epochs','2','--log-limit','2',
                       '--export-weights',str(output/'weights.pt'))
        state,identity=read_checkpoint(output/'last.pt')
        self.assertEqual((state['step'],state['samples_seen']),(4,8))
        self.assertEqual(len(json.loads((output/'report.json').read_bytes())['log']),2)
        args=dict(expected_catalog_sha256=self.catalog['content_sha256'],expected_checkpoint_sha256=identity,
                  expected_source={k:state['source'][k] for k in ('revision','code_sha256')},
                  expected_contract=self.catalog['contract'],expected_preprocessing_version=preprocessing_version([4,4]))
        rng=rng_state(); model,artifact=load_weights(output/'weights.pt',**args)
        self.assertTrue(equal(rng,rng_state()))
        model.train(); before=copy.deepcopy(model.state_dict())
        for index,spec in enumerate(self.specs):
            loader=LocalStereoRGB(self.base/spec['rgb_manifest'],self.base/spec['rgb_root'])
            rng=rng_state(); cache=build_cache(loader,model,artifact)
            self.assertTrue(equal(rng,rng_state())); self.assertTrue(model.training)
            self.assertTrue(equal(before,model.state_dict()))
            self.assertEqual(cache['image_sizes'],[[8,6],[8,6]])
            self.assertEqual(cache['rgb_shard_provenance']['checkpoint_sha256'],identity)
            self.assertEqual(cache['rgb_shard_provenance']['weights_artifact_sha256'],artifact['artifact_file_sha256'])
            pose=json.loads((self.base/spec['pose']).read_bytes()); original=load_document(pose,1)[1]
            joined=join_predictions(pose,cache,loader.manifest_sha256); converted=load_document(joined,1)[1]
            for split in original:
                self.assertTrue(torch.equal(original[split].target_mm,converted[split].target_mm))
                self.assertTrue(torch.equal(original[split].target_valid,converted[split].target_valid))
        with contextlib.redirect_stdout(io.StringIO()):
            main(['evaluate','--checkpoint',str(output/'last.pt'),'--catalog',str(self.base/'catalog.json'),
                  '--output',str(output/'evaluation.json')])
            main(['cache','--checkpoint',str(output/'last.pt'),'--weights',str(output/'weights.pt'),
                  '--catalog',str(self.base/'catalog.json'),'--shard','0','--output',str(output/'cache.json')])
        with self.assertRaisesRegex(ValueError,'rgb_weights_expected_identity'):
            load_weights(output/'weights.pt',**{**args,'expected_checkpoint_sha256':'b'*64})
        # export는 학습 원본 파일 없이 checkpoint만 사용한다.
        shutil.rmtree(self.base/'inputs')
        for p in self.base.glob('*_pose.json'): p.unlink()
        with contextlib.redirect_stdout(io.StringIO()):
            main(['export','--checkpoint',str(output/'last.pt'),'--output',str(output/'offline.pt')])

    def test_cli_failure_preserves_last_and_resume_identity_protection(self):
        output=self.base/'interrupted'; original=RGBTrainer.train_step
        def broken(trainer):
            if trainer.step==1:
                with torch.no_grad(): next(trainer.model.parameters()).add_(123.)
                raise OSError('injected_after_last')
            return original(trainer)
        with patch.object(RGBTrainer,'train_step',broken), contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit): self.cli_train(output,'--steps','3')
        state,_=read_checkpoint(output/'last.pt'); self.assertEqual(state['step'],1)
        self.assertEqual(json.loads((output/'report.json').read_bytes())['durable_checkpoint']['step'],1)
        self.cli_train(output,'--resume',str(output/'last.pt'),'--steps','3')
        preserved=(output/'last.pt').read_bytes(); report=(output/'report.json').read_bytes()
        other=self.base/'other'; self.cli_train(other,'--steps','1')
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit): self.cli_train(output,'--resume',str(other/'last.pt'),'--steps','4')
        self.assertEqual((output/'last.pt').read_bytes(),preserved); self.assertEqual((output/'report.json').read_bytes(),report)

    def test_inference_module_does_not_import_trainer_or_cli(self):
        code="import sys; import rewind_m1.rgb_frontend_artifacts; assert not any(n in sys.modules for n in ['rewind_m1.training','rewind_m1.shard_cli','rewind_m1.rgb_training','rewind_m1.rgb_training_cli']); print('independent')"
        result=subprocess.run([sys.executable,'-X','utf8','-c',code],cwd=ROOT,capture_output=True,text=True,check=True)
        self.assertEqual(result.stdout.strip(),'independent')

    def test_rgb_only_cache_loader_pins_raw_file_bounds_and_needs_no_pose(self):
        from test_rgb_shards import signed
        for p in self.base.glob('*_pose.json'): p.unlink()
        loader=open_rgb(self.catalog,self.base,0)
        self.assertEqual(len(loader),2)
        for key,value in (('max_batch_tensor_bytes',1),('max_records',1),('max_rgb_manifest_bytes',1)):
            changed=copy.deepcopy(self.catalog); changed['limits'][key]=value
            with self.assertRaises(ValueError): open_rgb(signed(changed),self.base,0)
        changed=copy.deepcopy(self.catalog); changed['shards'][0]['sequences'][0].update(first_frame=10,last_frame=11)
        with self.assertRaisesRegex(ValueError,'rgb_shard_rgb_chunk_bounds'):
            open_rgb(signed(changed),self.base,0)
        path=self.base/self.specs[0]['rgb_manifest']; path.write_bytes(path.read_bytes()+b' ')
        with self.assertRaisesRegex(ValueError,'rgb_shard_rgb_file_hash'): open_rgb(self.catalog,self.base,0)

    def test_export_preserves_original_model_rng_and_serialized_limit(self):
        trainer=self.trainer(); trainer.train_step()
        state=copy.deepcopy(trainer.model.state_dict()); rng=rng_state()
        path=self.base/'direct_weights.pt'
        export_weights(trainer.model,self.config,self.catalog['contract'],self.catalog['content_sha256'],
                       trainer.source,trainer.step,'c'*64,path)
        self.assertTrue(trainer.model.training); self.assertTrue(equal(state,trainer.model.state_dict()))
        self.assertTrue(equal(rng,rng_state()))
        with patch('rewind_m1.rgb_frontend_artifacts.MAX_BYTES',1):
            with self.assertRaisesRegex(ValueError,'rgb_weights_byte_limit'):
                export_weights(trainer.model,self.config,self.catalog['contract'],self.catalog['content_sha256'],
                               trainer.source,trainer.step,'c'*64,self.base/'oversize.pt')
        self.assertFalse((self.base/'oversize.pt').exists()); self.assertTrue(equal(rng,rng_state()))


if __name__=='__main__': unittest.main()
