"""명시적 endpoint cosine beta 가정, x0 학습과 deterministic DDIM."""
import math
import torch


class DiffusionSchedule:
    def __init__(self, steps, beta_start, beta_end):
        if type(steps) is not int or steps < 2:
            raise ValueError('diffusion_steps >=2 필요')
        if any(type(v) not in (int,float) or not math.isfinite(v) for v in (beta_start,beta_end)) or not 0 < beta_start <= beta_end < 1:
            raise ValueError('0<beta_start<=beta_end<1 필요')
        self.steps = steps
        # 논문은 endpoint와 cosine만 명시. 아래 beta interpolation은 프로젝트 가정이다.
        u = torch.linspace(0,1,steps,dtype=torch.float64)
        self.beta = beta_start+(beta_end-beta_start)*(1-torch.cos(math.pi*u))/2
        self.alpha_bar = torch.cat((torch.ones(1,dtype=torch.float64),torch.cumprod(1-self.beta,dim=0)))

    def alpha(self, k, like):
        if not isinstance(k,torch.Tensor) or k.dtype != torch.long or k.ndim != 1 or len(k) != len(like) or (k<0).any() or (k>self.steps).any():
            raise ValueError('batch별 diffusion k int64 범위 불일치')
        return self.alpha_bar.to(device=like.device,dtype=like.dtype)[k.to(like.device)].reshape(-1,*([1]*(like.ndim-1)))

    def q_sample(self, clean, noise, k):
        if clean.shape != noise.shape or not torch.isfinite(clean).all() or not torch.isfinite(noise).all():
            raise ValueError('q_sample shape/finite 오류')
        a = self.alpha(k,clean)
        return a.sqrt()*clean+(1-a).sqrt()*noise

    def timesteps(self, sampling_steps):
        if type(sampling_steps) is not int or not 1 <= sampling_steps <= self.steps:
            raise ValueError('sampling_steps 범위 오류')
        if sampling_steps==1:
            return [self.steps]
        return torch.linspace(self.steps,1,sampling_steps,dtype=torch.float64).round().long().tolist()

    @torch.no_grad()
    def sample(self, model, features, camera, time_valid, noise, sampling_steps, positions=None):
        if model.training:
            raise ValueError('DDIM에는 eval model 필요')
        if noise.ndim!=4 or noise.shape[-1]!=3 or time_valid.shape!=noise.shape[:2] or time_valid.dtype!=torch.bool:
            raise ValueError('DDIM noise B,T,J,3 / bool time mask 필요')
        ks = self.timesteps(sampling_steps)
        x = torch.where(time_valid[...,None,None],noise,0.)
        for current,following in zip(ks,ks[1:]+[0]):
            k = torch.full((len(x),),current,device=x.device,dtype=torch.long)
            clean,_ = model(x,features,camera,k,time_valid) if positions is None else model(x,features,camera,k,time_valid,positions=positions)
            a = self.alpha(k,x)
            next_a = self.alpha(torch.full_like(k,following),x)
            epsilon = (x-a.sqrt()*clean)/(1-a).sqrt()
            # eta=0; 마지막 alpha_bar[0]=1이므로 정확히 x0 head를 반환한다.
            x = next_a.sqrt()*clean+(1-next_a).sqrt()*epsilon
            x = torch.where(time_valid[...,None,None],x,0.)
        if not torch.isfinite(x).all():
            raise ValueError('DDIM 비유한 출력')
        return x
