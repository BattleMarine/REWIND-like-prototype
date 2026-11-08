"""합성 RGB shard의 작은2D CNN 학습. Resize loss는 FP32, 실제 release 미지원."""
import copy
import math
import os
import platform

import torch
from PIL import __version__ as pillow_version

from .data import InputContract
from .frontend import StereoHeatmapNet, heatmap_loss
from .rgb_frontend_artifacts import preprocessing_version, resize_images, scale_xy, training_size
from .rgb_shards import RGBEpochSampler, RGBShardDataset, validate_catalog
from .shards import _integer, _require
from .shard_training import ShardTrainer
from .training import require_torch_api, restore_rng, rng_state, seed_run, source_identity

FORMAT = 'm1_rgb_shard_checkpoint_v1'


def validate_config(value):
    required={'seed','batch_size','learning_rate','scheduler_step','scheduler_gamma','amp','training_image_size',
              'frontend_version','max_native_tensor_bytes','max_heatmap_elements'}
    _require(isinstance(value,dict) and set(value)==required, 'rgb_trainer_config_fields')
    _require(_integer(value['seed'],0,2**32-1) and _integer(value['batch_size'],1,4) and
             _integer(value['scheduler_step'],1,10**9) and type(value['amp']) is bool, 'rgb_trainer_config_integer')
    for key,maximum in (('learning_rate',.1),('scheduler_gamma',1.)):
        _require(type(value[key]) in (int,float) and math.isfinite(value[key]) and 0 < value[key] <= maximum,
                 'rgb_trainer_config_'+key)
    training_size(value['training_image_size'])
    _require(isinstance(value['frontend_version'],str) and 0 < len(value['frontend_version'].strip()) <= 128,
             'rgb_trainer_frontend_version')
    _require(_integer(value['max_native_tensor_bytes'],1,128*1024**2) and
             _integer(value['max_heatmap_elements'],1,32*1024**2), 'rgb_trainer_allocation_limit')
    return copy.deepcopy(value)


class RGBTrainer(ShardTrainer):
    CHECKPOINT_FORMAT = FORMAT

    def _new_sampler(self):
        return RGBEpochSampler(self.catalog,'train',self.config['seed'])

    def _open_data(self, catalog, base):
        catalog = validate_catalog(catalog)
        contract = InputContract.from_dict(catalog['contract'])
        _require(contract.data_kind=='synthetic_fixture','rgb_dataset_sample_release_not_implemented')
        return catalog,contract,RGBShardDataset(catalog,base)

    def _make_model(self, joints):
        return StereoHeatmapNet(joints).to(self.device)

    def _loss(self, logits, gt_xy, valid):
        return heatmap_loss(logits.float(), gt_xy, valid, self.config['training_image_size'])

    def _pose_metadata(self, metadata):
        return metadata['pose_record']

    def _update_observation(self):
        return {}

    def __init__(self, config, catalog, base, device='cpu', check_budget=None):
        require_torch_api(); self.config=validate_config(config)
        self.check_budget=check_budget or (lambda:None)
        self.catalog,self.contract,self.dataset=self._open_data(catalog,base)
        width,height=self.contract.image_sizes[0]; joints=len(self.contract.joint_names)
        _require(24*width*height+18*joints <= min(config['max_native_tensor_bytes'],self.catalog['limits']['max_batch_tensor_bytes']),
                 'rgb_native_sample_tensor_limit')
        tw,th=config['training_image_size']
        _require(config['batch_size'] <= self.catalog['limits']['max_records'] and
                 config['batch_size']*2*joints*((tw+1)//2)*((th+1)//2) <= config['max_heatmap_elements'],
                 'rgb_heatmap_allocation_limit')
        report=self.dataset.verify_all(check_budget=self.check_budget)
        self.counts=report['metadata_records']
        _require({'train','validation'} <= set(self.counts), 'rgb_requires_train_validation')
        self.device=torch.device(device)
        _require(self.device.type in ('cpu','cuda') and (self.device.type!='cuda' or torch.cuda.is_available()), 'rgb_device_unavailable')
        _require(not config['amp'] or self.device.type=='cuda','rgb_amp_requires_cuda')
        if self.device.type=='cuda':
            _require(os.environ.get('CUBLAS_WORKSPACE_CONFIG',':4096:8') in (':4096:8',':16:8'),'rgb_cuda_workspace')
        seed_run(config['seed'])
        self.model=self._make_model(joints)
        self.optimizer=torch.optim.Adam(self.model.parameters(),lr=config['learning_rate'])
        self.scheduler=torch.optim.lr_scheduler.StepLR(self.optimizer,config['scheduler_step'],config['scheduler_gamma'])
        self.scaler=torch.amp.GradScaler('cuda',enabled=config['amp'])
        self.sampler=self._new_sampler()
        self.step=self.samples_seen=self.attempts=self.skipped_updates=0; self.poisoned=False
        self.source=source_identity()
        self.environment={'python':platform.python_version(),'torch':str(torch.__version__),'pillow':pillow_version,
            'device':str(self.device),'cuda_runtime':torch.version.cuda,'threads':torch.get_num_threads(),'amp':config['amp'],
            'training_image_size':list(config['training_image_size']),'align_corners':True,
            'preprocessing_version':preprocessing_version(config['training_image_size']),
            'sampler_random_implementation':'python_random_v1','validation_precision':'float32'}
        if self.device.type=='cuda':
            props=torch.cuda.get_device_properties(self.device)
            self.environment.update(gpu_name=props.name,gpu_total_memory_bytes=props.total_memory,
                gpu_capability=list(torch.cuda.get_device_capability(self.device)),
                cublas_workspace_config=os.environ['CUBLAS_WORKSPACE_CONFIG'])

    def _batch(self, shard, split, indices):
        supervision=self.dataset.load(shard,split)
        images,targets,masks,native_targets,keys=[],[],[],[],[]
        for i in indices:
            self.check_budget()
            native=supervision.load_batch([i])
            images.append(resize_images(native['images'],self.config['training_image_size']))
            targets.append(scale_xy(native['gt_xy'],list(self.contract.image_sizes[0]),self.config['training_image_size']))
            masks.append(native['annotation_valid']); native_targets.append(native['gt_xy'])
            pose=self._pose_metadata(native['metadata'][0]); keys.append([pose['sequence_id'],pose['frame_index']])
            del native
        return {'images':torch.cat(images),'gt_xy':torch.cat(targets),'valid':torch.cat(masks),
                'native_gt_xy':torch.cat(native_targets),'keys':keys}

    def train_step(self):
        _require(not self.poisoned,'shard_trainer_failed_reload_checkpoint_required')
        shard,indices=self.sampler.peek(self.config['batch_size']); self.attempts+=1
        try:
            batch=self._batch(shard,'train',indices)
            self.model.train(); self.optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=self.device.type,dtype=torch.float16,enabled=self.config['amp']):
                logits=self.model(batch['images'].to(self.device))[0]
            # softmax/target/reduction은 AMP context 밖에서 FP32로 계산한다.
            loss,inside=self._loss(logits,batch['gt_xy'].to(self.device),batch['valid'].to(self.device))
            _require(torch.isfinite(loss).item(),'rgb_nonfinite_loss')
            self.scaler.scale(loss).backward(); self.scaler.unscale_(self.optimizer)
            finite=all(p.grad is None or torch.isfinite(p.grad).all().item() for p in self.model.parameters())
            _require(finite or self.config['amp'],'rgb_nonfinite_gradient')
            old=self.scaler.get_scale(); self.scaler.step(self.optimizer); self.scaler.update()
            if self.scaler.get_scale() < old:
                self.skipped_updates+=1
                return {'updated':False,'step':self.step,'amp_skipped':True,'batch_samples':len(indices)}
            _require(all(torch.isfinite(p).all().item() for p in self.model.parameters()) and
                     all(not isinstance(v,torch.Tensor) or torch.isfinite(v).all().item()
                         for s in self.optimizer.state.values() for v in s.values()),'rgb_nonfinite_updated_state')
            self.scheduler.step(); self.sampler.commit(self.config['batch_size'])
            self.step+=1; self.samples_seen+=len(indices)
            return {'updated':True,'step':self.step,'loss':float(loss.detach()),'inside_joints':int(inside.sum()),
                    'batch_samples':len(indices),'keys':batch['keys'],'coverage':self.coverage(),
                    **self._update_observation()}
        except BaseException:
            self.poisoned=True
            raise

    def evaluate(self, split='validation', final_test=False):
        _require(not self.poisoned,'shard_trainer_failed_reload_checkpoint_required')
        _require(split in self.counts,'rgb_evaluation_split_missing')
        _require(split!='test' or final_test is True,'rgb_test_requires_final_test')
        mode,rng=self.model.training,rng_state()
        totals={'records':0,'valid_joints':0,'inside_joints':0,'outside_joints':0,'zero_inside_batches':0,
                'heatmap_squared_sum':0.,'heatmap_elements':0,'native_distance_sum_pixels':0.,
                'inside_distance_sum_pixels':0.,'outside_distance_sum_pixels':0.}
        self.dataset.clear()
        try:
            self.model.eval()
            with torch.no_grad():
                for shard,row in enumerate(self.catalog['shards']):
                    if not any(g['split']==split for g in row['sequences']): continue
                    count=sum(g['records'] for g in row['sequences'] if g['split']==split)
                    for start in range(0,count,self.config['batch_size']):
                        self.check_budget()
                        batch=self._batch(shard,split,list(range(start,min(start+self.config['batch_size'],count))))
                        logits,xy,_=self.model(batch['images'].to(self.device))
                        _require(torch.isfinite(logits).all().item() and torch.isfinite(xy).all().item(),
                                 'rgb_nonfinite_evaluation')
                        target=batch['gt_xy'].to(self.device); valid=batch['valid'].to(self.device)
                        width,height=self.config['training_image_size']
                        inside=valid & (target[...,0]>=0) & (target[...,0]<=width-1) & (target[...,1]>=0) & (target[...,1]<=height-1)
                        n=int(inside.sum()); totals['inside_joints']+=n
                        if n:
                            loss,_=heatmap_loss(logits.float(),target,valid,self.config['training_image_size'])
                            elements=n*logits.shape[-1]*logits.shape[-2]
                            totals['heatmap_squared_sum']+=float(loss)*elements; totals['heatmap_elements']+=elements
                        else: totals['zero_inside_batches']+=1
                        native=scale_xy(xy.float(),self.config['training_image_size'],list(self.contract.image_sizes[0]))
                        error=torch.linalg.vector_norm(native-batch['native_gt_xy'].to(self.device),dim=-1)
                        _require(torch.isfinite(error[valid]).all().item(),'rgb_nonfinite_evaluation')
                        outside=valid & ~inside
                        totals['valid_joints']+=int(valid.sum()); totals['outside_joints']+=int(outside.sum())
                        totals['native_distance_sum_pixels']+=float(error[valid].sum())
                        totals['inside_distance_sum_pixels']+=float(error[inside].sum())
                        totals['outside_distance_sum_pixels']+=float(error[outside].sum())
                        totals['records']+=len(batch['keys'])
                        del batch,logits,xy,target,valid,inside,native,error,outside
                    self.dataset.clear()
        finally:
            self.dataset.clear(); self.model.train(mode); restore_rng(rng)
        return {**totals,'split':split,'heatmap_mse':totals['heatmap_squared_sum']/totals['heatmap_elements'] if totals['heatmap_elements'] else None,
                'native_mean_error_pixels':totals['native_distance_sum_pixels']/totals['valid_joints'] if totals['valid_joints'] else None,
                'evaluation_kind':'synthetic_code_check','full_training_ready':False}
