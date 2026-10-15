"""동일 anchor window를 재계산하는 bounded streaming. DDIM/KV cache 없음."""
from collections import deque
import hashlib
import json

import torch


def keyed_frame_noise(sequence_id, frame_index, joints, seed):
    """미래/호출순서와 무관한 원본 frame별 초기 noise. 학습 RNG는 소비하지 않는다."""
    if not isinstance(sequence_id,str) or not sequence_id or type(frame_index) is not int or frame_index<0 or type(joints) is not int or joints<1 or type(seed) is not int or seed<0:
        raise ValueError('frame noise 식별자/seed 오류')
    key = json.dumps([seed,sequence_id,frame_index],separators=(',',':')).encode('utf-8')
    value = int.from_bytes(hashlib.sha256(key).digest()[:8],'big')%(2**63)
    return torch.randn(joints,3,generator=torch.Generator().manual_seed(value))


def anchor_noise(data, seed):
    """중첩 anchor windows에서도 같은 raw frame에는 같은 noise를 배치한다."""
    n,t,j,_ = data.target_mm.shape
    noise = torch.zeros(n,t,j,3)
    for row,(sequence,index) in enumerate(data.anchors):
        for slot in range(t):
            if data.time_valid[row,slot]:
                noise[row,slot] = keyed_frame_noise(sequence,index-t+1+slot,j,seed)
    return noise


class AnchorSampler:
    def __init__(self, model, schedule, window_size, sampling_steps):
        if type(window_size) is not int or window_size<1:
            raise ValueError('window_size 양의 정수 필요')
        schedule.timesteps(sampling_steps)
        self.model,self.schedule = model,schedule
        self.window_size,self.sampling_steps = window_size,sampling_steps
        self.history = deque(maxlen=window_size)
        self.sequence_id,self.frame_index = None,None

    def reset(self):
        self.history.clear()
        self.sequence_id,self.frame_index = None,None

    @torch.no_grad()
    def push(self, sequence_id, frame_index, features, camera, noise):
        if not isinstance(sequence_id,str) or not sequence_id or type(frame_index) is not int or frame_index<0:
            raise ValueError('sequence/frame 식별 오류')
        if features.shape!=(self.model.joints*14,) or camera.shape!=(2,9) or noise.shape!=(self.model.joints,3):
            raise ValueError('streaming frame shape 오류')
        tensors = (features,camera,noise)
        parameter = next(self.model.parameters())
        if any(x.device!=parameter.device or x.dtype!=parameter.dtype or not torch.isfinite(x).all() for x in tensors):
            raise ValueError('streaming finite/device/dtype 오류')
        if self.model.training:
            raise ValueError('streaming에는 eval model 필요')
        same_sequence = sequence_id==self.sequence_id
        if same_sequence and frame_index<=self.frame_index:
            raise ValueError('중복/역순 frame 거부')
        # 거부된 push는 기존 history를 바꾸지 않는다.
        rows = list(self.history) if same_sequence and frame_index==self.frame_index+1 else []
        rows = (rows+[tuple(x.detach().clone() for x in tensors)])[-self.window_size:]
        size = len(rows)
        packed = [torch.zeros((1,self.window_size,*x.shape),device=x.device,dtype=x.dtype) for x in tensors]
        for slot,row in enumerate(rows,start=self.window_size-size):
            for out,value in zip(packed,row):
                out[0,slot] = value
        valid = torch.zeros((1,self.window_size),device=features.device,dtype=torch.bool)
        valid[:,-size:] = True
        result = self.schedule.sample(self.model,packed[0],packed[1],valid,packed[2],self.sampling_steps)[0,-1].clone()
        self.history.clear();self.history.extend(rows)
        self.sequence_id,self.frame_index = sequence_id,frame_index
        return result
