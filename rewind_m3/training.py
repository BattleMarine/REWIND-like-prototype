"""GT upper 학습과 고정 body prediction 평가를 분리한 hand teacher."""
import copy
import hashlib
import platform
from pathlib import Path

import torch

from rewind_m1.training import Trainer as RegressionTrainer,require_torch_api,seed_run
from rewind_m2.training import validate_config as validate_m2,source_identity as m2_source,position_losses
from rewind_m2.diffusion import DiffusionSchedule
from rewind_m2.streaming import anchor_noise
from .data import prepare,synthetic_document
from .models import HandDenoiser,BoundHand
from .cascade import FrozenBody,upper_condition,cached_body_windows,reconstruct
from .metrics import cascade_metrics

ROOT = Path(__file__).resolve().parents[1]


def source_identity():
    source = m2_source()
    source['code_sha256'].update({p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest()
                                for p in sorted((ROOT/'rewind_m3').glob('*.py'))})
    return source


def validate_config(config):
    if not isinstance(config,dict) or config.get('conditioning') not in ('upper','none'):
        raise ValueError('conditioning=upper/none 명시 필요')
    validate_m2({k:v for k,v in config.items() if k!='conditioning'})
    return copy.deepcopy(config)


def read_checkpoint(path):
    doc = torch.load(path,map_location='cpu',weights_only=True)
    if not isinstance(doc,dict) or doc.get('format_version')!='m3_checkpoint_v1':
        raise ValueError('M3 checkpoint 형식 필요')
    return doc


class Trainer(RegressionTrainer):
    def __init__(self,config,body_checkpoint,document=None,device='cpu'):
        require_torch_api()
        self.config = cfg = validate_config(config)
        if device!='cpu' or cfg['amp']:
            raise ValueError('M3 합성 구현 묶음은 CPU float32만 지원; GPU/AMP 미검증')
        self.device = torch.device('cpu')
        self.document = copy.deepcopy(document if document is not None else synthetic_document())
        self.contract,self.roles,self.fingers,self.splits,self.cameras,self.manifest_hash,edges,self.wrist_per_finger = prepare(self.document,cfg['window_size'],cfg['target_scale_mm'])
        if not {'train','validation'}.issubset(self.splits) or any(not d.target_valid[:,-1].any(1).all() for d in self.splits.values()):
            raise ValueError('train/validation 및 유효 local GT가 있는 anchor 필요')
        self.frozen = FrozenBody(Path(body_checkpoint),self.document,device)
        self.body_cache = self.frozen.predict()
        seed_run(cfg['seed'])
        self.model = HandDenoiser(cfg,len(self.fingers),edges,len(self.roles['upper_joint_names']),cfg['conditioning']=='upper')
        self.diffusion = DiffusionSchedule(cfg['diffusion_steps'],cfg['beta_start'],cfg['beta_end'])
        self.optimizer = torch.optim.Adam(self.model.parameters(),lr=cfg['learning_rate'])
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer,cfg['scheduler_step'],cfg['scheduler_gamma'])
        self.scaler = torch.amp.GradScaler('cuda',enabled=False)
        self.step = 0
        self.source = source_identity()
        self.environment = {'python':platform.python_version(),'torch':str(torch.__version__),'device':'cpu',
                            'cuda_runtime':torch.version.cuda,'threads':torch.get_num_threads(),'amp':False}
        self.runtime_observation = {}
        self.hand_contract = {'coordinate_frame':'wrist_relative_parallel_device','distance_unit':'mm',
            'joint_names':self.fingers,'upper_joint_names':self.roles['upper_joint_names'],
            'wrist_owner':'body','wrists':self.roles['wrists'],'wrist_local_value':[0.,0.,0.]}

    def condition(self,data,source):
        condition = upper_condition(data,self.roles,source,self.frozen.identity,self.body_cache if source=='predicted_frozen_body' else None)
        return condition.require(source,self.frozen.identity)

    def batch_loss(self,indices,k,noise,condition=None):
        data,cfg = self.splits['train'],self.config
        source = 'gt_upper_train' if cfg['conditioning']=='upper' else 'none'
        condition = self.condition(data,source) if condition is None else condition.require(source,self.frozen.identity)
        time = data.time_valid[indices]
        target,valid = data.target_mm[indices],data.target_valid[indices]&time[...,None]
        clean = torch.where(valid[...,None],target/cfg['target_scale_mm'],0.)
        noisy = self.diffusion.q_sample(clean,noise,k)
        upper = condition.values_mm[indices]/cfg['target_scale_mm'] if condition.values_mm is not None else None
        upper_valid = condition.valid[indices] if condition.valid is not None else None
        prediction,frame = self.model(noisy,data.features[indices],self.cameras['train'][indices],k,time,upper=upper,upper_valid=upper_valid)
        return position_losses(prediction.float(),frame.float(),target,valid,time,cfg['target_scale_mm'],cfg['frame_loss_weight'])

    def train_step(self):
        cfg,data = self.config,self.splits['train']
        indices = torch.randint(len(data.anchors),(cfg['batch_size'],))
        k = torch.randint(1,cfg['diffusion_steps']+1,(len(indices),))
        noise = torch.randn(len(indices),cfg['window_size'],len(self.fingers),3)
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        loss,_,_ = self.batch_loss(indices,k,noise)
        self.scaler.scale(loss).backward()
        self.scaler.unscale_(self.optimizer)
        if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in self.model.parameters()):
            raise ValueError('비유한 hand gradient; update 중단')
        self.scaler.step(self.optimizer);self.scaler.update();self.scheduler.step()
        self.step += 1
        self.frozen.assert_unchanged()
        return float(loss.detach())

    @torch.no_grad()
    def predict(self,split='validation',condition_source=None):
        source = condition_source or ('predicted_frozen_body' if self.config['conditioning']=='upper' else 'none')
        expected = ('predicted_frozen_body','gt_upper_diagnostic') if self.config['conditioning']=='upper' else ('none',)
        if source not in expected:
            raise ValueError('평가 condition source/모델 conditioning 불일치')
        if split not in self.splits:
            raise ValueError('split 없음')
        self.frozen.assert_unchanged()
        data,cfg = self.splits[split],self.config
        condition = self.condition(data,source)
        noise = anchor_noise(data,cfg['seed']+20000)
        previous = self.model.training
        self.model.eval()
        local = []
        try:
            for start in range(0,len(data.anchors),cfg['batch_size']):
                end = start+cfg['batch_size']
                upper = condition.values_mm[start:end]/cfg['target_scale_mm'] if condition.values_mm is not None else None
                valid = condition.valid[start:end] if condition.valid is not None else None
                bound = BoundHand(self.model,upper,valid).eval()
                result = self.diffusion.sample(bound,data.features[start:end],self.cameras[split][start:end],data.time_valid[start:end],noise[start:end],cfg['sampling_steps'])
                local.append(result[:,-1]*cfg['target_scale_mm'])
            local = torch.cat(local)
            body = cached_body_windows(data,self.body_cache,self.frozen.identity,len(self.roles['body_joint_names']))[:,-1]
            whole,global_fingers = reconstruct(body,local,self.roles,self.fingers)
            self.frozen.assert_unchanged()
            return {'body_mm':body,'local_mm':local,'global_fingers_mm':global_fingers,'whole_body_mm':whole,
                    'condition_source':source,'attachment_source':'predicted_frozen_body_wrist'}
        finally:
            self.model.train(previous)

    def evaluate(self,split='validation',final_test=False,condition_source=None):
        if split=='test' and not final_test:
            raise ValueError('Test는 명시적 최종 평가만 허용')
        prediction = self.predict(split,condition_source)
        data = self.splits[split]
        metrics = cascade_metrics(prediction['body_mm'],prediction['local_mm'],prediction['global_fingers_mm'],data,self.roles,self.fingers)
        mask = data.target_valid[:,-1]
        residual = prediction['local_mm'][mask]-data.target_mm[:,-1][mask]
        return {'split':split,'normalized_mse':float((residual/self.config['target_scale_mm']).square().mean()),
                'metrics':metrics,'data_kind':self.contract.data_kind,'input_source':self.contract.input_source,
                'evaluation_kind':'synthetic_code_check' if self.contract.data_kind=='synthetic_fixture' else 'gt_input_diagnostic' if self.contract.input_source=='gt_2d' else 'predicted_input_baseline',
                'condition_source':prediction['condition_source'],'attachment_source':prediction['attachment_source'],
                'training_condition_source':'gt_upper_train' if self.config['conditioning']=='upper' else 'none',
                'frozen_body':self.frozen.identity,'frozen_body_unchanged':True,
                'sampling_steps':self.config['sampling_steps'],'noise_policy':'frame_keyed_sha256_seed_v1','noise_seed':self.config['seed']+20000}

    def checkpoint(self):
        self.frozen.assert_unchanged()
        doc = super().checkpoint()
        doc.update(format_version='m3_checkpoint_v1',hand_contract=self.hand_contract,m3_contract=self.roles,
            frozen_body=self.frozen.identity,training_condition_source='gt_upper_train' if self.config['conditioning']=='upper' else 'none',
            diffusion={'alpha_bar':self.diffusion.alpha_bar,'schedule':self.config['schedule'],'prediction':'x0','ddim_eta':0,
                       'sample_timesteps':self.diffusion.timesteps(self.config['sampling_steps'])})
        return doc

    def validate_checkpoint_identity(self,checkpoint):
        if checkpoint.get('format_version')!='m3_checkpoint_v1' or checkpoint.get('frozen_body')!=self.frozen.identity or checkpoint.get('hand_contract')!=self.hand_contract or checkpoint.get('m3_contract')!=self.roles:
            raise ValueError('M3 hand/body checkpoint identity/계약 불일치')
        diffusion = checkpoint.get('diffusion',{})
        if diffusion.get('schedule')!=self.config['schedule'] or diffusion.get('prediction')!='x0' or diffusion.get('ddim_eta')!=0 or diffusion.get('sample_timesteps')!=self.diffusion.timesteps(self.config['sampling_steps']) or not torch.equal(diffusion.get('alpha_bar',torch.empty(0)),self.diffusion.alpha_bar):
            raise ValueError('hand diffusion schedule 불일치')
        if checkpoint.get('config')!=self.config or checkpoint.get('manifest_hash')!=self.manifest_hash or checkpoint.get('source',{}).get('code_sha256')!=self.source['code_sha256']:
            raise ValueError('M3 checkpoint config/data/code 불일치')
        expected_source = 'gt_upper_train' if self.config['conditioning']=='upper' else 'none'
        if checkpoint.get('training_condition_source')!=expected_source:
            raise ValueError('hand checkpoint 학습 condition source 불일치')

    def resume(self,checkpoint):
        self.validate_checkpoint_identity(checkpoint)
        super().resume(checkpoint)
