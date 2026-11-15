"""Paired camera-representation training; reuses the unchanged v1 teacher update."""

from copy import deepcopy
from dataclasses import replace
import json
import hashlib
import math
from pathlib import Path
import platform
import time

import torch

from rewind_archive.contracts import digest
from rewind_m1.training import seed_run,rng_state,restore_rng
from rewind_m2.data import encode_camera
from rewind_prepare.converter import sha
from rewind_prepare.temporal_cache import _atomic
from rewind_prepare.temporal_cache import _check_part
from .data import hand_clip,gt_upper_train
from .runner import Budget,_fit_state,_restore_state,_publish,_read_checkpoint
from .training import TeacherRun,_source
from .cross_data import write_report


ROOT=Path(__file__).resolve().parents[1]
CONFIG=ROOT/'configs/rewind_v15_camera_training.json'
POLICY=json.loads(CONFIG.read_bytes());RUN_ID=POLICY['run_id'];ARMS=tuple(POLICY['arms'])


def source_identity():
    from .camera_gauge import source_identity as inherited_source
    names=('configs/rewind_v15_camera_training.json','scripts/build_v15_camera_training.py')
    return {**inherited_source(),**{n:sha((ROOT/n).read_bytes()) for n in names}}


def model_fingerprint(model):
    hasher=hashlib.sha256()
    for name,value in sorted(model.state_dict().items()):
        x=value.detach().cpu().contiguous()
        hasher.update(repr((name,str(x.dtype),tuple(x.shape))).encode());hasher.update(x.numpy().tobytes())
    return hasher.hexdigest()


def pilot_config(cfg=POLICY):
    result=deepcopy(cfg);result['updates_per_teacher']=cfg['pilot_pairs']
    result['milestones']=[0,cfg['pilot_pairs']];result['checkpoint_every']=1
    return result


def rng_fingerprint():
    r=rng_state()
    return digest({'python':r['python'],'cpu':sha(r['torch_cpu'].numpy().tobytes()),
                   'cuda':[sha(x.cpu().numpy().tobytes()) for x in r['torch_cuda']]})


class TrainingBudget(Budget):
    replacing_bytes=0
    def storage(self,root,reserve):
        import shutil
        used=sum(p.stat().st_size for p in root.iterdir() if p.is_file())
        if used-self.replacing_bytes+reserve>self.limits['output_bytes'] or shutil.disk_usage(root).free<reserve:
            raise TimeoutError('v15_output_limit')
    def check(self):
        super().check()
        try:
            import resource
            if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024>POLICY['limits']['observed_rss_bytes']:
                raise MemoryError('v15_observed_rss_4GiB')
        except ImportError:pass


class CameraOrigin:
    """Causal clip-local full-float64 anchor with explicit streaming continuation."""
    def __init__(self,mode,state=None):
        if mode not in ARMS:raise ValueError('v15_camera_mode')
        self.mode=mode;self.state=deepcopy(state)
        if state is not None and state['mode']!=mode:raise ValueError('v15_anchor_mode')

    def apply(self,rotations,translations,sequence,start,frames):
        r=torch.as_tensor(rotations,dtype=torch.float64);t=torch.as_tensor(translations,dtype=torch.float64)
        if (r.shape!=(len(frames),2,3,3) or t.shape!=(len(frames),2,3) or not frames
                or not torch.isfinite(r).all() or not torch.isfinite(t).all()
                or list(frames)!=list(range(frames[0],frames[0]+len(frames)))):raise ValueError('v15_camera_chunk_contract')
        if not torch.allclose(r.transpose(-1,-2)@r,torch.eye(3,dtype=torch.float64).expand_as(r),atol=1e-6,rtol=0):
            raise ValueError('v15_camera_rotation')
        key=[sequence,start]
        if self.state is None:
            if frames[0]!=start:raise ValueError('v15_anchor_requires_first_frame')
            midpoint=-(r[0].transpose(-1,-2)@t[0,...,None])[...,0].mean(0)
            self.state={'mode':self.mode,'clip':key,'first_frame':start,'next_frame':start,'anchor_world_mm':midpoint.tolist()}
        if self.state['clip']!=key or self.state['next_frame']!=frames[0]:raise ValueError('v15_anchor_reset_or_position_required')
        anchor=t.new_tensor(self.state['anchor_world_mm']) if self.mode=='first_midpoint' else t.new_zeros(3)
        centered=t+torch.einsum('tvij,j->tvi',r,anchor)
        result=[]
        for rr,tt in zip(r,centered,strict=True):
            result.append(encode_camera({s:{'rotation_matrix':rr[i].tolist(),'translation':tt[i].tolist()} for i,s in enumerate(('left','right'))},100.))
        self.state['next_frame']=frames[-1]+1
        encoded=torch.stack(result)
        roundtrip=(encoded[...,6:].double()*100-centered).abs().max()
        if float(roundtrip)>.1:raise ValueError('v15_camera_encoding_error')
        return encoded,{'state':deepcopy(self.state),'float32_translation_roundtrip_max_abs_mm':float(roundtrip)}

    def reset(self):self.state=None


def camera_clip(clip,metadata,arm):
    origin=CameraOrigin(arm)
    camera,audit=origin.apply(metadata['rotation'],metadata['translation_mm'],metadata['sequence'],metadata['start'],metadata['frames'])
    return replace(clip,camera=camera[None]),audit


def shifted_metadata(metadata,delta):
    result=deepcopy(metadata);r=torch.tensor(metadata['rotation'],dtype=torch.float64)
    t=torch.tensor(metadata['translation_mm'],dtype=torch.float64)
    result['translation_mm']=(t-torch.einsum('tvij,j->tvi',r,torch.tensor(delta,dtype=torch.float64))).tolist()
    return result


class PairedTraining:
    def __init__(self,cfg,spec,clips,metadata,input_identity,device,output,*,fixture=False,pilot=False):
        if not fixture and cfg!=(pilot_config() if pilot else POLICY):raise ValueError('v15_fixed_config')
        self.cfg,self.spec,self.clips,self.metadata=cfg,spec,clips,metadata
        self.root=Path(output);self.root.mkdir(exist_ok=True);self.device=torch.device(device)
        if not fixture and self.device.type!='cuda':raise ValueError('v15_training_requires_supported_cuda')
        if not clips or any(c.split!='train' or c.source!='predicted_2d' for c in clips.values()):raise ValueError('v15_train_only')
        self.keys=sorted(clips);self.position={'role':'body','pair':0,'total_pairs':0};self.elapsed=0.;self.history=[]
        self.budget=TrainingBudget({'max_seconds':cfg['limits']['train_seconds'],'output_bytes':cfg['limits']['output_bytes'],
                            'checkpoint_bytes':cfg['limits']['checkpoint_bytes']},shutdown_reserve=60)
        self.binding={'format':'v15_paired_camera_training_v1','run_id':RUN_ID,'config':cfg,'source':source_identity(),
            'input':input_identity,'keys':self.keys,'spec':spec,'environment':{'device':str(device),'torch':str(torch.__version__),
                'python':platform.python_version(),'cuda':torch.version.cuda,'gpu':torch.cuda.get_device_name(device) if self.device.type=='cuda' else 'CPU',
                'vram_bytes':torch.cuda.get_device_properties(device).total_memory if self.device.type=='cuda' else None,'precision':'fp32'}}
        self.stages={a:{} for a in ARMS};self.anchors={};self.initial={};self.last=None;self.snapshots={}
        for i,role in enumerate(('body','hand')):
            names=spec['body_joint_names'] if role=='body' else [n for s in ('left','right') for n in spec['hands'][s][1:]]
            edges=[(names.index(a),names.index(b)) for a,b in spec['edges'] if a in names and b in names]
            for arm in ARMS:
                seed_run(cfg['seed']+1+i)
                self.stages[arm][role]=TeacherRun(cfg['teacher'],joints=len(names),edges=edges,source='predicted_2d',
                    input_sha256=input_identity['feature_identity'],device=device,
                    **({'upper_joints':len(spec['upper_joint_names']),'body_checkpoint_sha256':'0'*64} if role=='hand' else {}))
            fingerprints=[model_fingerprint(self.stages[a][role].model) for a in ARMS]
            if len(set(fingerprints))!=1:raise ValueError('v15_initial_states_differ')
            self.initial[role]=fingerprints[0]
        for arm in ARMS:
            for key in self.keys:
                _,audit=camera_clip(clips[key],metadata[key],arm);self.anchors[arm,str(key)]=audit['state']
        self.initial_rng=rng_fingerprint();self.transaction_copy_seconds=0.

    def snapshot(self,role,step):
        paired_pins={}
        for arm in ARMS:
            pin=self.publish(self.root/f'{arm}-{role}-u{step:04d}.pt',{'binding':self.binding,'arm':arm,'role':role,
                'step':step,'model':self.stages[arm][role].model.state_dict(),'initial':self.initial[role]},128*1024**2)
            paired_pins[f'{arm}/{role}/{step}']=pin
        self.snapshots.update(paired_pins)

    def save(self):
        self.last=self.publish(self.root/'last.pt',{'binding':self.binding,'position':self.position,'elapsed':self.elapsed,
            'anchors':{str(k):v for k,v in self.anchors.items()},'fits':{a:{r:_fit_state(s) for r,s in roles.items()} for a,roles in self.stages.items()},
            'rng':rng_state(),'initial':self.initial,'snapshots':self.snapshots,'history':self.history})
        self.report('checkpointed');return self.last

    def publish(self,path,state,cap=None):
        old=self.budget.limits['checkpoint_bytes'];self.budget.replacing_bytes=path.stat().st_size if path.exists() else 0
        if cap is not None:self.budget.limits['checkpoint_bytes']=min(cap,old)
        try:return _publish(path,state,self.budget)
        finally:self.budget.limits['checkpoint_bytes']=old;self.budget.replacing_bytes=0

    def resume(self,pin):
        state=_read_checkpoint(self.root/'last.pt',pin,self.cfg['limits']['checkpoint_bytes'])
        if state['binding']!=self.binding or state['anchors']!={str(k):v for k,v in self.anchors.items()}:
            raise ValueError('v15_resume_binding_or_anchor')
        p=state['position']
        if (p['role'] not in ('body','hand','complete') or not 0<=p['pair']<=self.cfg['updates_per_teacher']
                or p['total_pairs']!=p['pair']+(self.cfg['updates_per_teacher'] if p['role'] in ('hand','complete') else 0)):
            raise ValueError('v15_resume_position')
        if (type(state['elapsed']) not in (int,float) or not math.isfinite(state['elapsed'])
                or not 0<=state['elapsed']<=self.cfg['limits']['train_cumulative_seconds']):raise ValueError('v15_resume_cumulative_time')
        for arm,roles in self.stages.items():
            for role,stage in roles.items():
                n=(self.cfg['updates_per_teacher'] if p['role']=='complete' or role=='body' and p['role']=='hand'
                   else p['pair'] if role==p['role'] else 0)
                _restore_state(stage,state['fits'][arm][role],n)
        restore_rng(state['rng']);self.position=p;self.elapsed=state['elapsed'];self.history=state['history']
        self.snapshots=state['snapshots'];self.initial=state['initial'];self.last={'sha256':pin,'name':'last.pt','bytes':(self.root/'last.pt').stat().st_size}
        for pin in self.snapshots.values():_check_part(self.root/pin['name'],pin)
        if self.position['role'] in ('hand','complete'):
            for arm in ARMS:self.stages[arm]['hand'].spec['body_checkpoint_sha256']=self.snapshots[f'{arm}/body/{self.cfg["updates_per_teacher"]}']['sha256']

    def update_pair(self):
        role=self.position['role'];step=self.position['pair']
        if role not in ('body','hand'):raise ValueError('v15_completed_training')
        # Both arms see the same epoch permutation and random diffusion draws.
        epoch,slot=divmod(step,len(self.keys));generator=torch.Generator().manual_seed(self.cfg['seed']+epoch)
        key=self.keys[int(torch.randperm(len(self.keys),generator=generator)[slot])]
        copying=time.monotonic();before_rng=rng_state();before={a:deepcopy(_fit_state(self.stages[a][role])) for a in ARMS};metrics={}
        before_position=deepcopy(self.position);before_history=len(self.history)
        self.transaction_copy_seconds+=time.monotonic()-copying
        body_idx=[self.spec['joint_names'].index(n) for n in self.spec['body_joint_names']]
        try:
            for arm in ARMS:
                restore_rng(before_rng)
                clip,_=camera_clip(self.clips[key],self.metadata[key],arm)
                stage=self.stages[arm][role]
                if role=='body':metrics[arm]=stage.update(clip.select_joints(body_idx))
                else:metrics[arm]=stage.update(hand_clip(clip,self.spec),upper=gt_upper_train(clip,self.spec))
            self.position['pair']+=1;self.position['total_pairs']+=1
            if step%20==0:self.history.append({'role':role,'pair':step+1,'clip':list(key),'metrics':metrics})
        except BaseException:
            self.position=before_position;del self.history[before_history:]
            for arm in ARMS:_restore_state(self.stages[arm][role],before[arm],step)
            restore_rng(before_rng);raise
        return metrics

    def report(self,status,failure=None):
        doc={'schema':'v15_camera_training_report_v1',**self.binding,'status':status,'position':deepcopy(self.position),
            'optimizer_updates':2*self.position['total_pairs'],'elapsed_training_seconds':self.elapsed,'last':self.last,
            'initial':self.initial,'snapshots':self.snapshots,'history':self.history,'failure':failure,
            'initial_rng':self.initial_rng,'transaction_copy_wall_seconds_this_invocation':self.transaction_copy_seconds,
            'gpu_peak_allocated_bytes':torch.cuda.max_memory_allocated(self.device) if self.device.type=='cuda' else None,
            'scope':'fixed1000train camera representation controlled diagnostic; six regression clips are not independent generalization'}
        write_report(self.root/'run.json',doc);return doc

    def run(self,*,stop_after_pairs=None):
        start=time.monotonic();failure=None;status='paused'
        try:
            while self.position['role']!='complete':
                if self.elapsed+time.monotonic()-start>=self.cfg['limits']['train_cumulative_seconds']-60:
                    status='cumulative_budget_exhausted';break
                self.budget.check();role=self.position['role'];step=self.position['pair']
                if step in self.cfg['milestones'] and f'absolute/{role}/{step}' not in self.snapshots:self.snapshot(role,step)
                if step==self.cfg['updates_per_teacher']:
                    if role=='body':
                        self.position.update(role='hand',pair=0)
                        for arm in ARMS:self.stages[arm]['hand'].spec['body_checkpoint_sha256']=self.snapshots[f'{arm}/body/{step}']['sha256']
                    else:self.position['role']='complete'
                    continue
                if stop_after_pairs is not None and self.position['total_pairs']>=stop_after_pairs:break
                self.update_pair()
                if self.position['pair']%self.cfg['checkpoint_every']==0:
                    self.elapsed=min(self.cfg['limits']['train_cumulative_seconds'],self.elapsed+time.monotonic()-start);start=time.monotonic();self.save()
            if self.position['role']=='complete':status='complete'
        except (TimeoutError,KeyboardInterrupt) as error:failure={'type':type(error).__name__,'reason':str(error)}
        except BaseException as error:
            status='failed';failure={'type':type(error).__name__,'reason':str(error)[:2000]}
        finally:
            self.elapsed=min(self.cfg['limits']['train_cumulative_seconds'],self.elapsed+time.monotonic()-start);self.save()
        return self.report(status,failure)
