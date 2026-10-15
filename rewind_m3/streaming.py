"""Body anchor 예측을 hand history에 연결하는 bounded two-stage streaming."""
from collections import deque

import torch
from rewind_m2.streaming import AnchorSampler,keyed_frame_noise
from .models import BoundHand
from .cascade import reconstruct


class CascadeStream:
    def __init__(self,trainer):
        self.trainer = trainer
        body = trainer.frozen.trainer
        self.body = AnchorSampler(body.model,body.diffusion,body.config['window_size'],body.config['sampling_steps'])
        self.history = deque(maxlen=trainer.config['window_size'])
        self.sequence_id,self.frame_index = None,None

    def reset(self):
        self.body.reset();self.history.clear()
        self.sequence_id,self.frame_index = None,None

    @torch.no_grad()
    def push(self,sequence_id,frame_index,body_features,hand_features,body_camera,hand_camera):
        t = self.trainer
        if not isinstance(sequence_id,str) or not sequence_id or type(frame_index) is not int or frame_index<0:
            raise ValueError('sequence/frame 식별 오류')
        if body_features.shape!=(len(t.roles['body_joint_names'])*14,) or hand_features.shape!=(len(t.fingers)*14,) or body_camera.shape!=(2,9) or hand_camera.shape!=(2,9):
            raise ValueError('cascade streaming shape 오류')
        if t.model.training or any(x.dtype!=torch.float32 or x.device.type!='cpu' or not torch.isfinite(x).all() for x in (body_features,hand_features,body_camera,hand_camera)):
            raise ValueError('eval/CPU/finite streaming 입력 필요')
        if sequence_id==self.sequence_id and frame_index<=self.frame_index:
            raise ValueError('중복/역순 frame 거부')
        t.frozen.assert_unchanged()
        old_body = (list(self.body.history),self.body.sequence_id,self.body.frame_index)
        try:
            bc = t.frozen.trainer.config
            body_noise = keyed_frame_noise(sequence_id,frame_index,len(t.roles['body_joint_names']),bc['seed']+10000)
            body_mm = self.body.push(sequence_id,frame_index,body_features,body_camera,body_noise)*bc['target_scale_mm']
            rows = list(self.history) if sequence_id==self.sequence_id and frame_index==self.frame_index+1 else []
            hand_noise = keyed_frame_noise(sequence_id,frame_index,len(t.fingers),t.config['seed']+20000)
            rows = (rows+[(hand_features.clone(),hand_camera.clone(),body_mm.clone(),hand_noise)])[-t.config['window_size']:]
            w = t.config['window_size'];size = len(rows)
            x = torch.zeros(1,w,len(t.fingers)*14)
            camera = torch.zeros(1,w,2,9)
            body_window = torch.zeros(1,w,len(t.roles['body_joint_names']),3)
            noise = torch.zeros(1,w,len(t.fingers),3)
            for slot,row in enumerate(rows,w-size):
                x[0,slot],camera[0,slot],body_window[0,slot],noise[0,slot] = row
            time = torch.tensor([[False]*(w-size)+[True]*size])
            if t.config['conditioning']=='upper':
                indices = [t.roles['body_joint_names'].index(n) for n in t.roles['upper_joint_names']]
                upper = body_window[:,:,indices]/t.config['target_scale_mm']
                valid = time[...,None].expand(1,w,len(indices))
            else:
                upper,valid = None,None
            local = t.diffusion.sample(BoundHand(t.model,upper,valid).eval(),x,camera,time,noise,t.config['sampling_steps'])[0,-1]*t.config['target_scale_mm']
            whole,global_fingers = reconstruct(body_mm,local,t.roles,t.fingers)
            t.frozen.assert_unchanged()
        except Exception:
            self.body.history.clear();self.body.history.extend(old_body[0])
            self.body.sequence_id,self.body.frame_index = old_body[1:]
            raise
        self.history.clear();self.history.extend(rows)
        self.sequence_id,self.frame_index = sequence_id,frame_index
        return {'body_mm':body_mm,'local_mm':local,'global_fingers_mm':global_fingers,'whole_body_mm':whole,
                'condition_source':'predicted_frozen_body' if t.config['conditioning']=='upper' else 'none',
                'attachment_source':'predicted_frozen_body_wrist'}
