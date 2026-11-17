"""Shared fixed-noise x0 diagnostics; sampling delegates to the original DDIM."""
from dataclasses import replace
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
import torch

from rewind_archive.contracts import canonical
from rewind_m2.diffusion import DiffusionSchedule
from rewind_m2.streaming import keyed_frame_noise
from rewind_m3.models import BoundHand
from rewind_prepare.converter import sha
from rewind_prepare.temporal_cache import _atomic
from scripts.visualize_archvis_v1 import _draw_joints, _shared_3d_groups, DARK
from .data import hand_clip, gt_upper_train, attach_predicted_hands
from .oracle_overfit import _metrics as _base_metrics, _constant_pose


def _metrics(predicted,clip,edges):
    result=_base_metrics(predicted,clip,edges)
    valid=clip.target_valid&clip.time_valid[...,None];distance=(predicted-clip.target_mm).norm(dim=-1).double()
    result['error_sum_mm']=float(distance[valid].sum())
    result['per_frame']=[{'count':int(mask.sum()),'sum_mm':float(row[mask].sum())} for row,mask in zip(distance[0],valid[0])]
    return result


class TraceModel(torch.nn.Module):
    """Observe x0/frame heads without changing any sampler arithmetic."""
    def __init__(self, model, clip, edges, scale, check=lambda: None):
        super().__init__();self.model=model;self.clip=clip;self.edges=edges
        self.scale=scale;self.check=check;self.rows=[]
        self.eval()

    def forward(self, x, features, camera, k, time_valid, **kwargs):
        self.check()
        clean,frame=self.model(x,features,camera,k,time_valid,**kwargs)
        self.rows.append({'k':int(k[0]),'x0':_metrics(clean*self.scale,self.clip,self.edges),
                          'frame_head':_metrics(frame*self.scale,self.clip,self.edges)})
        return clean,frame


class Analysis:
    def __init__(self, cfg, clip, spec, metadata, device, check=lambda: None):
        self.cfg,self.spec,self.metadata,self.device,self.check=cfg,spec,metadata,torch.device(device),check
        self.clip=replace(clip,**{n:getattr(clip,n).to(device) for n in ('features','camera','target_mm','target_valid','time_valid','seconds')})
        if clip.source!='predicted_2d' or clip.split!='train':raise ValueError('predicted_first_train_required')
        self.names={'body':spec['body_joint_names'],'hand':[n for s in ('left','right') for n in spec['hands'][s][1:]]}
        self.clips={'body':self.clip.select_joints([spec['joint_names'].index(n) for n in self.names['body']]),'hand':hand_clip(self.clip,spec)}
        self.edges={r:[(ns.index(a),ns.index(b)) for a,b in spec['edges'] if a in ns and b in ns] for r,ns in self.names.items()}
        self.upper=gt_upper_train(self.clip,spec);t=cfg['teacher']
        self.schedule=DiffusionSchedule(t['diffusion_steps'],t['beta_start'],t['beta_end']);self.scale=t['target_scale_mm']
        self.noise={r:torch.stack([keyed_frame_noise(metadata['sequence'],f,len(ns),cfg['seed']+cfg['noise_offsets'][r]) for f in metadata['frames']])[None].to(device) for r,ns in self.names.items()}
        self.noise_pins={r:sha(x.cpu().numpy().tobytes()) for r,x in self.noise.items()}

    def condition(self, models, role):
        model=models[role].eval()
        if role=='hand':model=BoundHand(model,self.upper.values_mm/self.scale,self.upper.valid).eval()
        return model

    @torch.no_grad()
    def sample(self, models, steps):
        predictions={};traces={}
        for role,clip in self.clips.items():
            observer=TraceModel(self.condition(models,role),clip,self.edges[role],self.scale,self.check)
            predictions[role]=self.schedule.sample(observer,clip.features,clip.camera,clip.time_valid,self.noise[role],steps)*self.scale
            traces[role]=observer.rows
        return predictions,traces

    @torch.no_grad()
    def reconstruct(self,models,k):
        predictions={};frames={}
        for role,clip in self.clips.items():
            self.check();mask=clip.target_valid&clip.time_valid[...,None]
            clean=torch.where(mask[...,None],clip.target_mm/self.scale,0.)
            timestep=torch.full((len(clean),),k,dtype=torch.long,device=self.device)
            noisy=self.schedule.q_sample(clean,self.noise[role],timestep)
            x0,frame=self.condition(models,role)(noisy,clip.features,clip.camera,timestep,clip.time_valid)
            predictions[role]=x0*self.scale;frames[role]=_metrics(frame*self.scale,clip,self.edges[role])
        alpha=float(self.schedule.alpha_bar[k])
        return predictions,{'k':k,'alpha_bar':alpha,'snr':alpha/(1-alpha),'frame_heads':frames,
                            'kind':'GT-noised direct x0 restoration; not pure-noise generation'}

    def metrics(self,predictions):
        result={r:_metrics(predictions[r],clip,self.edges[r]) for r,clip in self.clips.items()}
        fingers=[self.spec['joint_names'].index(n) for n in self.names['hand']]
        global_clip=self.clip.select_joints(fingers)
        # Common local mask includes wrist validity; native global mask is also recorded.
        common=replace(global_clip,target_valid=self.clips['hand'].target_valid)
        for name,body in (('predicted_wrist',predictions['body']),('gt_wrist_diagnostic',self.clips['body'].target_mm)):
            global_hand=attach_predicted_hands(body,predictions['hand'],self.spec)
            result['hand_global_'+name]={'common_local_mask':_metrics(global_hand,common,self.edges['hand'])}
            if name=='predicted_wrist':result['hand_global_'+name]['native_global_mask']=_metrics(global_hand,global_clip,self.edges['hand'])
        return result

    def baselines(self):
        return {r:_metrics(_constant_pose(c),c,self.edges[r]) for r,c in self.clips.items()}

    def combined(self,predictions):
        hands=attach_predicted_hands(predictions['body'],predictions['hand'],self.spec)
        points=dict(zip(self.names['body']+self.names['hand'],torch.cat((predictions['body'],hands),-2).unbind(-2)))
        return torch.stack([points[n] for n in self.spec['joint_names']],-2)

    def artifacts(self,root,label,predictions):
        root=Path(root);combined=self.combined(predictions).detach().cpu()
        payload={'units':'device mm; no alignment','hand_condition':'GT upper train diagnostic','noise_sha256':self.noise_pins,
            'body_mm':predictions['body'].detach().cpu().tolist(),'hand_local_mm':predictions['hand'].detach().cpu().tolist(),
            'combined_mm':combined.tolist(),'target_mm':self.clip.target_mm.cpu().tolist(),
            'target_valid':self.clip.target_valid.cpu().tolist(),'time_valid':self.clip.time_valid.cpu().tolist(),
            'hand_local_valid':self.clips['hand'].target_valid.cpu().tolist()}
        json_path=root/f'predictions-{label}.json';_atomic(json_path,canonical(payload))
        target=self.clip.target_mm.cpu();valid=(self.clip.target_valid&self.clip.time_valid[...,None]).cpu()
        frames=sorted({0,combined.shape[1]//2,combined.shape[1]-1});size=320
        image=Image.new('RGB',(size*4+60,90+len(frames)*(size+25)),DARK);draw=ImageDraw.Draw(image);font=ImageFont.load_default(size=14)
        draw.text((12,8),f'{label}: predicted F9 / GT upper / same train clip',fill='white',font=font)
        draw.text((12,30),'Whole body: common axes across all rows. Hand local: separate common axes; both wrist origins overlap at zero.',fill='white',font=font)
        hand_names=self.names['hand']+[self.spec['wrists'][s] for s in ('left','right')]
        hand_spec={'joint_names':hand_names,'body_joint_names':hand_names[-2:],'edges':[(a,b) for a,b in self.spec['edges'] if a in hand_names and b in hand_names and not (a in hand_names[-2:] and b in hand_names[-2:])]}
        local=predictions['hand'].detach().cpu();local_gt=self.clips['hand'].target_mm.cpu();local_valid=self.clips['hand'].target_valid.cpu()
        whole_groups=[];local_groups=[]
        for f in frames:
            whole_groups.extend((combined[0,f].tolist(),[p if ok else None for p,ok in zip(target[0,f].tolist(),valid[0,f].tolist())]))
            local_groups.extend((local[0,f].tolist()+[[0.,0.,0.]]*2,[p if ok else None for p,ok in zip(local_gt[0,f].tolist(),local_valid[0,f].tolist())]+[[0.,0.,0.]]*2))
        whole_projected=_shared_3d_groups(whole_groups,size);local_projected=_shared_3d_groups(local_groups,size)
        bounds={}
        for name,groups in (('whole_device',whole_groups),('hand_wrist_relative',local_groups)):
            present=torch.tensor([p for g in groups for p in g if p is not None])
            bounds[name]={'min_mm':present.amin(0).tolist(),'max_mm':present.amax(0).tolist(),'shared_frames':[self.metadata['frames'][f] for f in frames]}
        for row,f in enumerate(frames):
            groups=whole_projected[row*2:row*2+2]+local_projected[row*2:row*2+2]
            for col,(name,group) in enumerate(zip(('body prediction','body GT','local hands prediction','local hands GT'),groups)):
                x,y=12+col*(size+12),64+row*(size+25)
                draw.text((x,y),f'frame {self.metadata["frames"][f]} / {name}',fill='white',font=font)
                tile=Image.new('RGB',(size,size),'#202a38');_draw_joints(tile,group,self.spec if col<2 else hand_spec);image.paste(tile,(x,y+22))
        png=root/f'comparison-{label}.png';image.save(png)
        return {'artifacts':[{'name':p.name,'bytes':p.stat().st_size,'sha256':sha(p.read_bytes())} for p in (json_path,png)],'shared_axis_bounds':bounds}

    def criteria(self,metrics):
        result={};criteria=self.cfg['reference_criteria'];baselines=self.baselines()
        for role in self.clips:
            m=metrics[role];limit=criteria['body_mpjpe_mm' if role=='body' else 'hand_local_mpjpe_mm']
            ratios={s:m['predicted_'+s+'_mm']/m['target_'+s+'_mm'] if m['target_'+s+'_mm'] else None for s in ('bone','spread')}
            result[role]={'shape_ratios':ratios,'spatial_reference_pass':m['mpjpe_mm']<=limit and all(v is not None and criteria['shape_ratio_min']<=v<=criteria['shape_ratio_max'] for v in ratios.values()),
                'better_than_gt_clip_mean':m['mpjpe_mm']<baselines[role]['mpjpe_mm'],
                'first_middle_last_below_limit':all(v is not None and v<=limit for v in m['frame_mpjpe_mm'].values())}
        return result
