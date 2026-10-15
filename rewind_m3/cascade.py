"""실제 checkpoint의 고정 body 예측과 출처가 명시된 upper condition 연결."""
from dataclasses import dataclass
import hashlib

import torch

from rewind_m1.data import content_hash
from rewind_m1.frontend import weights_hash
from rewind_m2.training import Trainer as BodyTrainer,read_checkpoint as read_body
from rewind_m2.streaming import anchor_noise
from .data import body_document


def prediction_hash(values):
    return content_hash([[seq,index,tensor.detach().cpu().tolist()] for (seq,index),tensor in sorted(values.items())])


@dataclass
class BodyCache:
    values_mm: dict
    frozen_identity: dict
    payload_sha256: str
    source: str = 'predicted_frozen_body'

    def validate(self, identity):
        if self.source!='predicted_frozen_body' or self.frozen_identity!=identity or self.payload_sha256!=prediction_hash(self.values_mm):
            raise ValueError('body prediction cache 출처/identity/hash 불일치')


@dataclass(frozen=True)
class UpperCondition:
    values_mm: torch.Tensor | None
    valid: torch.Tensor | None
    source: str
    body_checkpoint_sha256: str | None

    def require(self, expected_source, body_identity):
        if self.source!=expected_source:
            raise ValueError('GT/예측/none condition 혼용 거부')
        if expected_source=='none':
            if self.values_mm is not None or self.valid is not None:
                raise ValueError('none condition은 pose tensor가 없어야 함')
        elif expected_source=='predicted_frozen_body':
            if self.body_checkpoint_sha256!=body_identity['checkpoint_sha256']:
                raise ValueError('예측 condition body checkpoint 불일치')
        elif self.body_checkpoint_sha256 is not None:
            raise ValueError('GT condition에 예측 checkpoint 출처 부여 금지')
        return self


class FrozenBody:
    def __init__(self, checkpoint_path, document, device='cpu'):
        checkpoint = read_body(checkpoint_path)
        self.trainer = BodyTrainer(checkpoint['config'],body_document(document),device)
        self.trainer.resume(checkpoint)
        self.model = self.trainer.model.eval().requires_grad_(False)
        for parameter in self.model.parameters():
            parameter.grad = None
        self.identity = {'checkpoint_sha256':hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
            'weights_sha256':weights_hash(self.model),'config':checkpoint['config'],'manifest_hash':checkpoint['manifest_hash'],
            'input_contract':checkpoint['input_contract'],'m2_contract':checkpoint['m2_contract'],
            'source_code_sha256':checkpoint['source']['code_sha256'],'checkpoint_step':checkpoint['step']}
        self.snapshot = {k:v.detach().clone() for k,v in self.model.state_dict().items()}

    def assert_unchanged(self):
        if self.model.training or any(p.requires_grad or p.grad is not None for p in self.model.parameters()) or any(not torch.equal(v,self.model.state_dict()[k]) for k,v in self.snapshot.items()):
            raise ValueError('frozen body weight/gradient/mode 변경 감지')

    @torch.no_grad()
    def predict(self):
        self.assert_unchanged()
        t = self.trainer
        values = {}
        for split,data in t.splits.items():
            noise = anchor_noise(data,t.config['seed']+10000).to(t.device)
            for start in range(0,len(data.anchors),t.config['batch_size']):
                end = start+t.config['batch_size']
                prediction = t.diffusion.sample(self.model,data.features[start:end].to(t.device),t.cameras[split][start:end].to(t.device),
                    data.time_valid[start:end].to(t.device),noise[start:end],t.config['sampling_steps'])[:,-1]*t.config['target_scale_mm']
                for key,value in zip(data.anchors[start:end],prediction):
                    if key in values:
                        raise ValueError('body prediction 중복 frame key')
                    values[key] = value.detach().cpu().clone()
        self.assert_unchanged()
        return BodyCache(values,self.identity,prediction_hash(values))


def cached_body_windows(data,cache,identity,body_joints):
    cache.validate(identity)
    n,t = data.time_valid.shape
    result = torch.zeros(n,t,body_joints,3)
    for row,(sequence,index) in enumerate(data.anchors):
        for slot in range(t):
            if data.time_valid[row,slot]:
                key = (sequence,index-t+1+slot)
                if key not in cache.values_mm:
                    raise ValueError('body prediction frame 누락; GT fallback 금지')
                value = cache.values_mm[key]
                if value.shape!=(body_joints,3) or not torch.isfinite(value).all():
                    raise ValueError('body prediction shape/finite 오류')
                result[row,slot] = value
    return result


def upper_condition(data,roles,source,body_identity,cache=None):
    if source=='none':
        return UpperCondition(None,None,'none',None)
    upper_indices = [roles['body_joint_names'].index(n) for n in roles['upper_joint_names']]
    if source in ('gt_upper_train','gt_upper_diagnostic'):
        valid = data.body_target_valid[:,:,upper_indices]&data.time_valid[...,None]
        values = torch.where(valid[...,None],data.body_target_mm[:,:,upper_indices],0.)
        return UpperCondition(values,valid,source,None)
    if source!='predicted_frozen_body' or cache is None:
        raise ValueError('지원하지 않는 upper source 또는 prediction cache 누락')
    values = cached_body_windows(data,cache,body_identity,len(roles['body_joint_names']))[:,:,upper_indices]
    valid = data.time_valid[...,None].expand(*values.shape[:-1]).clone()
    return UpperCondition(values,valid,source,body_identity['checkpoint_sha256'])


def reconstruct(body_mm,local_mm,roles,fingers):
    body_names = roles['body_joint_names']
    if body_mm.shape[:-2]!=local_mm.shape[:-2] or body_mm.shape[-2:]!=(len(body_names),3) or local_mm.shape[-2:]!=(len(fingers),3):
        raise ValueError('body/local reconstruction shape 오류')
    owners = [body_names.index(roles['wrists'][side]) for side in ('left','right') for _ in roles['hands'][side][1:]]
    global_fingers = body_mm[...,owners,:]+local_mm
    # 손목은 body에 단 한번 존재한다. 각 hand의 implicit local wrist는 정확히 0이다.
    return torch.cat((body_mm,global_fingers),dim=-2),global_fingers
