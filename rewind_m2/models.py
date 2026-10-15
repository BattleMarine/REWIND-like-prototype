"""Graph convolution + spatial attention, per-joint RoPE causal temporal x0 denoiser."""
import math
import torch
from torch import nn


def mlp(inputs, hidden):
    return nn.Sequential(nn.Linear(inputs,hidden),nn.SiLU(),nn.Linear(hidden,hidden))


class GraphBlock(nn.Module):
    def __init__(self, hidden, heads, dropout):
        super().__init__()
        self.norm_g = nn.LayerNorm(hidden)
        self.graph = nn.Linear(hidden,hidden)
        self.norm_s = nn.LayerNorm(hidden)
        self.attention = nn.MultiheadAttention(hidden,heads,dropout=dropout,batch_first=True)
        self.norm_f = nn.LayerNorm(hidden)
        self.ff = nn.Sequential(nn.Linear(hidden,hidden*2),nn.SiLU(),nn.Dropout(dropout),nn.Linear(hidden*2,hidden))
        self.drop = nn.Dropout(dropout)

    def forward(self, x, adjacency):
        x = x+self.drop(self.graph(torch.einsum('ij,btjd->btid',adjacency,self.norm_g(x))))
        b,t,j,d = x.shape
        h = self.norm_s(x).reshape(b*t,j,d)
        x = x+self.drop(self.attention(h,h,h,need_weights=False)[0].reshape(b,t,j,d))
        return x+self.drop(self.ff(self.norm_f(x)))


class TemporalBlock(nn.Module):
    def __init__(self, hidden, heads, past_frames, dropout):
        super().__init__()
        self.heads,self.head_dim,self.past_frames = heads,hidden//heads,past_frames
        self.norm = nn.LayerNorm(hidden)
        self.qkv = nn.Linear(hidden,3*hidden)
        self.out = nn.Linear(hidden,hidden)
        self.norm_f = nn.LayerNorm(hidden)
        self.ff = nn.Sequential(nn.Linear(hidden,hidden*2),nn.SiLU(),nn.Dropout(dropout),nn.Linear(hidden*2,hidden))
        self.drop = nn.Dropout(dropout)
        self.register_buffer('frequency',10000.**(-torch.arange(0,self.head_dim,2).float()/self.head_dim))

    def rotary(self, x, positions):
        angle = positions.to(x.dtype)[:,None]*self.frequency.to(x.dtype)[None,:]
        c,s = angle.cos()[None,None],angle.sin()[None,None]
        even,odd = x[...,0::2],x[...,1::2]
        return torch.stack((even*c-odd*s,even*s+odd*c),dim=-1).flatten(-2)

    def forward(self, x, valid, positions):
        b,t,j,d = x.shape
        h = self.norm(x).permute(0,2,1,3).reshape(b*j,t,d)
        q,k,v = self.qkv(h).reshape(b*j,t,3,self.heads,self.head_dim).permute(2,0,3,1,4).unbind(0)
        q,k = self.rotary(q,positions),self.rotary(k,positions)
        distance = torch.arange(t,device=x.device)[:,None]-torch.arange(t,device=x.device)[None,:]
        local = (distance>=0)&(distance<=self.past_frames)
        key_valid = valid[:,None,:].expand(b,j,t).reshape(b*j,t)
        allowed = local[None,None]&key_valid[:,None,None,:]
        # Padding query의 all-masked softmax를 회피하고 결과는 즉시 0으로 만든다.
        fallback = (~key_valid)[:,None,:,None]&torch.eye(t,device=x.device,dtype=torch.bool)[None,None]
        allowed = allowed|fallback
        scores = (q@k.transpose(-2,-1))/math.sqrt(self.head_dim)
        weights = scores.masked_fill(~allowed,float('-inf')).softmax(-1)
        weights = self.drop(weights)
        out = (weights@v).transpose(1,2).reshape(b,j,t,d).permute(0,2,1,3)
        x = x+self.drop(self.out(out))
        x = x+self.drop(self.ff(self.norm_f(x)))
        return torch.where(valid[...,None,None],x,0.)


class BodyDenoiser(nn.Module):
    def __init__(self, config, joints, edges):
        super().__init__()
        hidden,heads = config['hidden'],config['heads']
        self.joints,self.hidden = joints,hidden
        self.diffusion_steps = config['diffusion_steps']
        self.camera_encoder = mlp(9,hidden)
        self.time_encoder = mlp(hidden,hidden)
        self.input = nn.Linear(3+14+hidden*3,hidden)
        self.joint_embedding = nn.Parameter(torch.randn(joints,hidden)*.02)
        adjacency = torch.eye(joints)
        for i,j in edges:
            adjacency[i,j] = adjacency[j,i] = 1
        adjacency = adjacency/adjacency.sum(-1,keepdim=True)
        self.register_buffer('adjacency',adjacency)
        self.register_buffer('diffusion_frequency',10000.**(-torch.arange(hidden//2).float()/(hidden//2)))
        self.spatial = nn.ModuleList(GraphBlock(hidden,heads,config['dropout']) for _ in range(config['graph_layers']))
        self.temporal = nn.ModuleList(TemporalBlock(hidden,heads,config['attention_past_frames'],config['dropout']) for _ in range(config['temporal_layers']))
        self.frame_head = nn.Linear(hidden,3)
        self.final_head = nn.Linear(hidden,3)

    def forward(self, noisy, features, camera, k, time_valid, positions=None):
        if noisy.ndim != 4:
            raise ValueError('noisy B,T,J,3 필요')
        b,t,j,c = noisy.shape
        if j != self.joints or c != 3 or features.shape != (b,t,j*14) or camera.shape != (b,t,2,9) or time_valid.shape != (b,t) or time_valid.dtype != torch.bool or k.shape!=(b,) or k.dtype!=torch.long:
            raise ValueError('denoiser shape/mask/k 계약 오류')
        if (k<1).any() or (k>self.diffusion_steps).any():
            raise ValueError('denoiser k는 1..K 필요')
        if positions is None:
            positions = torch.arange(t,device=noisy.device)
        if positions.shape != (t,) or not torch.isfinite(positions).all():
            raise ValueError('RoPE positions shape/finite 오류')
        noisy = torch.where(time_valid[...,None,None],noisy,0.)
        features = torch.where(time_valid[...,None],features,0.)
        camera = torch.where(time_valid[...,None,None],camera,0.)
        if not all(torch.isfinite(x).all() for x in (noisy,features,camera)):
            raise ValueError('관측 token 비유한 입력')
        phase = k.float()[:,None]*self.diffusion_frequency[None]
        time = self.time_encoder(torch.cat((phase.sin(),phase.cos()),dim=-1))
        cam = self.camera_encoder(camera).flatten(-2)
        conditioning = torch.cat((cam,time[:,None].expand(-1,t,-1)),dim=-1)
        x = self.input(torch.cat((noisy,features.reshape(b,t,j,14),conditioning[:,:,None].expand(-1,-1,j,-1)),dim=-1))+self.joint_embedding
        for block in self.spatial:
            x = block(x,self.adjacency)
        x = torch.where(time_valid[...,None,None],x,0.)
        frame = torch.where(time_valid[...,None,None],self.frame_head(x),0.)
        for block in self.temporal:
            x = block(x,time_valid,positions)
        final = torch.where(time_valid[...,None,None],self.final_head(x),0.)
        return final,frame
