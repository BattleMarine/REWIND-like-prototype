"""M2 block을 유지하며 명시적 upper pose MLP feature를 concatenate한다."""
import torch
from torch import nn
from rewind_m2.models import BodyDenoiser,mlp


class HandDenoiser(BodyDenoiser):
    def __init__(self, config, joints, edges, upper_joints, conditioned):
        super().__init__(config,joints,edges)
        self.upper_joints,self.conditioned = upper_joints,conditioned
        if conditioned:
            # Known-mask를 함께 보존하는 것은 프로젝트 구현 선택이다.
            self.upper_encoder = mlp(upper_joints*4,self.hidden)
            self.input = nn.Linear(3+14+self.hidden*4,self.hidden)

    def forward(self,noisy,features,camera,k,time_valid,positions=None,upper=None,upper_valid=None):
        if not self.conditioned:
            if upper is not None or upper_valid is not None:
                raise ValueError('none 모델에 upper condition 제공 금지')
            return super().forward(noisy,features,camera,k,time_valid,positions)
        if noisy.ndim!=4:
            raise ValueError('noisy B,T,J,3 필요')
        b,t,j,c = noisy.shape
        if j!=self.joints or c!=3 or features.shape!=(b,t,j*14) or camera.shape!=(b,t,2,9) or time_valid.shape!=(b,t) or time_valid.dtype!=torch.bool or k.shape!=(b,) or k.dtype!=torch.long:
            raise ValueError('hand denoiser shape/mask/k 오류')
        if upper is None or upper_valid is None or upper.shape!=(b,t,self.upper_joints,3) or upper_valid.shape!=upper.shape[:-1] or upper_valid.dtype!=torch.bool:
            raise ValueError('명시적 upper coordinates/valid 필요')
        if (k<1).any() or (k>self.diffusion_steps).any():
            raise ValueError('k 1..K 필요')
        if positions is None:
            positions = torch.arange(t,device=noisy.device)
        if positions.shape!=(t,) or not torch.isfinite(positions).all():
            raise ValueError('RoPE positions 오류')
        noisy = torch.where(time_valid[...,None,None],noisy,0.)
        features = torch.where(time_valid[...,None],features,0.)
        camera = torch.where(time_valid[...,None,None],camera,0.)
        upper_valid = upper_valid & time_valid[...,None]
        upper = torch.where(upper_valid[...,None],upper,0.)
        if not all(torch.isfinite(x).all() for x in (noisy,features,camera,upper)):
            raise ValueError('유효 hand/upper token 비유한 값')
        phase = k.float()[:,None]*self.diffusion_frequency[None]
        time = self.time_encoder(torch.cat((phase.sin(),phase.cos()),dim=-1))
        cam = self.camera_encoder(camera).flatten(-2)
        upper_feature = self.upper_encoder(torch.cat((upper.flatten(2),upper_valid.to(upper.dtype)),dim=-1))
        condition = torch.cat((cam,time[:,None].expand(-1,t,-1),upper_feature),dim=-1)
        x = self.input(torch.cat((noisy,features.reshape(b,t,j,14),condition[:,:,None].expand(-1,-1,j,-1)),dim=-1))+self.joint_embedding
        for block in self.spatial:
            x = block(x,self.adjacency)
        x = torch.where(time_valid[...,None,None],x,0.)
        frame = torch.where(time_valid[...,None,None],self.frame_head(x),0.)
        for block in self.temporal:
            x = block(x,time_valid,positions)
        return torch.where(time_valid[...,None,None],self.final_head(x),0.),frame


class BoundHand(nn.Module):
    """동일 upper 조건을 모든 DDIM step에 명시적으로 전달하는 일시적인 호출 객체."""
    def __init__(self,model,upper=None,upper_valid=None):
        super().__init__()
        self.model,self.upper,self.upper_valid = model,upper,upper_valid

    def forward(self,*args,**kwargs):
        return self.model(*args,**kwargs,upper=self.upper,upper_valid=self.upper_valid)
