"""합성 body teacher 학습, masked x0/보조 loss와 완전한 재개 상태."""
import copy
import hashlib
import math
import os
import platform
from pathlib import Path

import torch

from rewind_m1.metrics import masked_mse,metric_sums
from rewind_m1.training import Trainer as RegressionTrainer,require_torch_api,seed_run,source_identity as m1_source
from .data import load_document,synthetic_document
from .diffusion import DiffusionSchedule
from .models import BodyDenoiser
from .streaming import anchor_noise

ROOT = Path(__file__).resolve().parents[1]


def source_identity():
    source = m1_source()
    source['code_sha256'].update({p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest()
                                for p in sorted((ROOT/'rewind_m2').glob('*.py'))})
    return source


def validate_config(config):
    required = {'hidden','heads','graph_layers','temporal_layers','window_size','attention_past_frames','dropout',
        'diffusion_steps','beta_start','beta_end','schedule','sampling_steps','ddim_eta','batch_size','learning_rate',
        'seed','target_scale_mm','frame_loss_weight','amp','scheduler_step','scheduler_gamma'}
    if not isinstance(config,dict) or set(config)!=required:
        raise ValueError('M2 config key 불일치')
    for key in ('hidden','heads','graph_layers','temporal_layers','window_size','diffusion_steps','sampling_steps','batch_size','scheduler_step'):
        if type(config[key]) is not int or config[key]<1:
            raise ValueError(key+': 양의 정수 필요')
    for key in ('attention_past_frames','seed'):
        if type(config[key]) is not int or config[key]<0:
            raise ValueError(key+': 비음수 정수 필요')
    if config['hidden']%config['heads'] or (config['hidden']//config['heads'])%2:
        raise ValueError('RoPE head dim은 짝수, hidden은 heads 배수 필요')
    for key in ('dropout','learning_rate','target_scale_mm','frame_loss_weight','scheduler_gamma','ddim_eta'):
        if type(config[key]) not in (int,float) or not math.isfinite(config[key]):
            raise ValueError(key+': 유한한 수치 필요')
    if not 0<=config['dropout']<1 or config['learning_rate']<=0 or config['target_scale_mm']<=0 or config['frame_loss_weight']<0 or not 0<config['scheduler_gamma']<=1:
        raise ValueError('M2 config 범위 오류')
    if type(config['amp']) is not bool or config['schedule']!='cosine_beta_endpoints_v1' or config['ddim_eta']!=0:
        raise ValueError('명시적 cosine_beta_endpoints_v1 / DDIM eta=0만 지원')
    DiffusionSchedule(config['diffusion_steps'],config['beta_start'],config['beta_end']).timesteps(config['sampling_steps'])
    return copy.deepcopy(config)


def position_losses(prediction, frame_prediction, target_mm, valid, time_valid, scale_mm, frame_weight):
    # loss mask로 indexing하기 전에 q_sample의 target 입력도 별도로 정리해야 한다.
    mask = valid & time_valid[...,None]
    simple = masked_mse(prediction*scale_mm,target_mm,mask,scale_mm)
    frame = masked_mse(frame_prediction*scale_mm,target_mm,mask,scale_mm)
    return simple+frame_weight*frame, simple, frame


def read_checkpoint(path):
    doc = torch.load(path,map_location='cpu',weights_only=True)
    if not isinstance(doc,dict) or doc.get('format_version')!='m2_checkpoint_v1':
        raise ValueError('M2 checkpoint 형식 필요')
    return doc


class Trainer(RegressionTrainer):
    # Atomic 저장·RNG·resume config/source/environment guard는 M1 계약을 재사용한다.
    def __init__(self, config, document=None, device='cpu'):
        require_torch_api()
        self.config = cfg = validate_config(config)
        self.document = copy.deepcopy(document if document is not None else synthetic_document())
        self.contract,self.splits,self.cameras,self.manifest_hash,edges = load_document(self.document,cfg['window_size'],cfg['target_scale_mm'])
        if not {'train','validation'}.issubset(self.splits):
            raise ValueError('분리된 train/validation 필요')
        if any(not split.target_valid[:,-1].any(dim=1).all() for split in self.splits.values()):
            raise ValueError('유효 GT가 전부 누락된 anchor 거부')
        self.device = torch.device(device)
        if self.device.type not in ('cpu','cuda') or (self.device.type=='cuda' and not torch.cuda.is_available()):
            raise ValueError('사용 가능 CPU/CUDA device 필요')
        if cfg['amp'] and self.device.type!='cuda':
            raise ValueError('AMP는 CUDA에서만 지원')
        if self.device.type=='cuda' and os.environ.get('CUBLAS_WORKSPACE_CONFIG',':4096:8') not in (':4096:8',':16:8'):
            raise ValueError('CUDA deterministic workspace 설정 불일치')
        seed_run(cfg['seed'])
        self.model = BodyDenoiser(cfg,len(self.contract.joint_names),edges).to(self.device)
        self.diffusion = DiffusionSchedule(cfg['diffusion_steps'],cfg['beta_start'],cfg['beta_end'])
        self.optimizer = torch.optim.Adam(self.model.parameters(),lr=cfg['learning_rate'])
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer,cfg['scheduler_step'],cfg['scheduler_gamma'])
        self.scaler = torch.amp.GradScaler('cuda',enabled=cfg['amp'])
        self.step = 0
        self.source = source_identity()
        self.environment = {'python':platform.python_version(),'torch':str(torch.__version__),'device':str(self.device),
                            'cuda_runtime':torch.version.cuda,'threads':torch.get_num_threads(),'amp':cfg['amp']}
        self.runtime_observation = {}
        if self.device.type=='cuda':
            props = torch.cuda.get_device_properties(self.device)
            self.environment.update(gpu_name=props.name,gpu_total_memory_bytes=props.total_memory,
                                    gpu_capability=list(torch.cuda.get_device_capability(self.device)),
                                    cublas_workspace_config=os.environ['CUBLAS_WORKSPACE_CONFIG'])
            free,total = torch.cuda.mem_get_info(self.device)
            self.runtime_observation = {'gpu_free_memory_bytes_at_init':free,'gpu_memory_total_bytes_at_init':total}

    def batch_loss(self, data, camera, indices, k, noise):
        cfg = self.config
        x,time = data.features[indices].to(self.device),data.time_valid[indices].to(self.device)
        target = data.target_mm[indices].to(self.device)
        valid = data.target_valid[indices].to(self.device)&time[...,None]
        # Masked NaN을 먼저 제거한다. 누락 GT 좌표로 유효 joint를 오염시키지 않는다.
        clean = torch.where(valid[...,None],target/cfg['target_scale_mm'],0.)
        noisy = self.diffusion.q_sample(clean,noise,k)
        noisy = torch.where(time[...,None,None],noisy,0.)
        prediction,frame = self.model(noisy,x,camera[indices].to(self.device),k,time)
        return position_losses(prediction.float(),frame.float(),target,valid,time,cfg['target_scale_mm'],cfg['frame_loss_weight'])

    def train_step(self):
        data,cfg = self.splits['train'],self.config
        indices = torch.randint(len(data.anchors),(cfg['batch_size'],))
        k = torch.randint(1,cfg['diffusion_steps']+1,(len(indices),),device=self.device)
        noise = torch.randn((len(indices),cfg['window_size'],len(self.contract.joint_names),3),device=self.device)
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=self.device.type,dtype=torch.float16,enabled=cfg['amp']):
            loss,_,_ = self.batch_loss(data,self.cameras['train'],indices,k,noise)
        self.scaler.scale(loss).backward()
        self.scaler.unscale_(self.optimizer)
        if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in self.model.parameters()):
            raise ValueError('비유한 gradient; update 중단')
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.scheduler.step()
        self.step += 1
        return float(loss.detach())

    @torch.no_grad()
    def evaluate(self, split='validation', final_test=False):
        if split=='test' and not final_test:
            raise ValueError('Test는 명시적 최종 평가에만 사용')
        if split not in self.splits:
            raise ValueError('split 없음')
        data,cfg = self.splits[split],self.config
        previous = self.model.training
        self.model.eval()
        try:
            # Frame 키별 전용 CPU generator: 호출순서/미래/학습 RNG에 무관하다.
            noise = anchor_noise(data,cfg['seed']+10000).to(self.device)
            total = {'distance_sum_mm':0.,'squared_sum_mm2':0.,'valid_joints':0}
            for start in range(0,len(data.anchors),cfg['batch_size']):
                end = start+cfg['batch_size']
                prediction = self.diffusion.sample(self.model,data.features[start:end].to(self.device),self.cameras[split][start:end].to(self.device),
                    data.time_valid[start:end].to(self.device),noise[start:end],cfg['sampling_steps'])
                values = metric_sums(prediction[:,-1]*cfg['target_scale_mm'],data.target_mm[start:end,-1].to(self.device),data.target_valid[start:end,-1].to(self.device))
                for key in total:
                    total[key] += values[key]
            count = total['valid_joints']
            return {'split':split,'mpjpe_mm':total['distance_sum_mm']/count,
                    'normalized_mse':total['squared_sum_mm2']/(3*count*cfg['target_scale_mm']**2),
                    'valid_joints':count,'sampling_steps':cfg['sampling_steps'],'ddim_eta':0,
                    'noise_policy':'frame_keyed_sha256_seed_v1','noise_seed':cfg['seed']+10000,
                    'evaluation_kind':'synthetic_code_check' if self.contract.data_kind=='synthetic_fixture' else
                        ('gt_input_diagnostic' if self.contract.input_source=='gt_2d' else 'predicted_input_baseline'),
                    'data_kind':self.contract.data_kind,'input_source':self.contract.input_source}
        finally:
            self.model.train(previous)

    def checkpoint(self):
        doc = super().checkpoint()
        doc.update(format_version='m2_checkpoint_v1',m2_contract=self.document['m2_contract'],
            diffusion={'prediction':'x0','schedule':self.config['schedule'],'alpha_bar':self.diffusion.alpha_bar,
                       'ddim_eta':0,'sample_timesteps':self.diffusion.timesteps(self.config['sampling_steps'])},
            inference_policy='anchor_window_recompute_no_kv_cache')
        return doc

    def resume(self, checkpoint):
        if checkpoint.get('format_version')!='m2_checkpoint_v1' or checkpoint.get('m2_contract')!=self.document['m2_contract']:
            raise ValueError('M2 checkpoint/계약 불일치')
        diffusion = checkpoint.get('diffusion',{})
        if diffusion.get('schedule')!=self.config['schedule'] or not torch.equal(diffusion.get('alpha_bar',torch.empty(0)),self.diffusion.alpha_bar):
            raise ValueError('diffusion schedule 불일치')
        super().resume(checkpoint)
