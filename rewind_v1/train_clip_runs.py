"""Bounded v152 inference and fresh-u0 paired v153 training."""
from contextlib import suppress
from copy import deepcopy
from pathlib import Path
import platform
import time

import torch

from rewind_archive.contracts import canonical, digest
from rewind_prepare.converter import sha
from rewind_m1.training import seed_run, rng_state, restore_rng
from rewind_prepare.temporal_cache import _atomic, _locked, _check_part
from . import train_clip_bundle as bundle
from .train_clip_analysis import Analysis
from .oracle_overfit import OracleStage
from .camera_training import TrainingBudget, model_fingerprint
from .run_stages import preserve_rng
from .runner import _publish, _read_checkpoint, _fit_state, _restore_state


def sha_tensor(value):return sha(value.detach().cpu().contiguous().numpy().tobytes())


def environment(device):
    d=torch.device(device)
    return {'python':platform.python_version(),'torch':str(torch.__version__),'cuda':torch.version.cuda,
        'device':str(d),'device_name':torch.cuda.get_device_name(d) if d.type=='cuda' else 'CPU',
        'vram_bytes':torch.cuda.get_device_properties(d).total_memory if d.type=='cuda' else None,
        'precision':'fp32','threads':torch.get_num_threads()}


def binding(cfg,inputs,device):
    return {'format':'v15_train_clip_derivative_v1','run_id':cfg['run_id'],'config':cfg,'source':bundle.source_identity(),
        'input_bundle_sha256':inputs['bundle']['content_sha256'],'feature_identity':inputs['feature_identity'],
        'parent_source_sha256':inputs['bundle']['parent_source_sha256'],'camera_anchor':inputs['anchor'],
        'environment':environment(device),'scope':'same first train50 clip; predicted F9; GT upper; no generalization claim'}


def diagnostic(inputs,cfg,device,output,*,fixture_models=None):
    if fixture_models is None and (cfg!=bundle.config('v152') or torch.device(device).type!='cuda'):raise ValueError('v152_fixed_config_cuda')
    root=Path(output);root.mkdir(parents=True,exist_ok=True)
    budget=TrainingBudget(dict(cfg['limits']),shutdown_reserve=30);analysis=Analysis(cfg,inputs['clip'],inputs['spec'],inputs['metadata'],device,budget.check)
    def input_fingerprint():
        return digest({n:sha_tensor(getattr(inputs['clip'],n)) for n in ('features','camera','target_mm','target_valid','time_valid','seconds')})
    before_input=input_fingerprint()
    report={**binding(cfg,inputs,device),'status':'running','noise_sha256':analysis.noise_pins,'evaluations':{},'constant_gt_clip_mean_diagnostic':analysis.baselines(),
        'input_tensor_fingerprint_before':before_input,'model_invariance':{}}
    def save(status,error=None):
        report.update(status=status,failure=error,elapsed_seconds=time.monotonic()-budget.start)
        _atomic(root/'run.json',canonical(report))
    with _locked(root):
        if any(p.name!='.lock' for p in root.iterdir()):raise ValueError('v152_new_output_required')
        try:
            for step in cfg['milestones']:
                budget.check();models=bundle.load_models(inputs,step,device) if fixture_models is None else fixture_models[step]
                before={r:model_fingerprint(m) for r,m in models.items()}
                records={};report['evaluations'][str(step)]=records
                for count in (cfg['sampling_steps'] if step==max(cfg['milestones']) else [10]):
                    predictions,traces=analysis.sample(models,count);label=f'u{step:04d}-ddim{count}'
                    records[label]={'kind':'pure-noise deterministic DDIM generation','sampling_steps':count,
                        'metrics':analysis.metrics(predictions),'x0_head_trace':traces,**analysis.artifacts(root,label,predictions)}
                    budget.storage(root,0);save('running')
                if step==max(cfg['milestones']):
                    for k in cfg['gt_noised_timesteps']:
                        predictions,details=analysis.reconstruct(models,k);label=f'u{step:04d}-gt-noised-k{k}'
                        records[label]={**details,'metrics':analysis.metrics(predictions),**analysis.artifacts(root,label,predictions)}
                        budget.storage(root,0);save('running')
                after={r:model_fingerprint(m) for r,m in models.items()}
                report['model_invariance'][str(step)]={'before':before,'after':after,'exact_unchanged':before==after}
                if before!=after:raise ValueError('v152_model_mutated')
                del models
            report['input_tensor_fingerprint_after']=input_fingerprint()
            report['input_exact_unchanged']=report['input_tensor_fingerprint_after']==before_input
            if not report['input_exact_unchanged']:raise ValueError('v152_input_mutated')
            save('complete');return report
        except BaseException as e:
            save('stopped' if isinstance(e,KeyboardInterrupt) else 'failed',{'type':type(e).__name__,'reason':str(e)[:512]});raise


class PredictedClipTraining:
    def __init__(self,inputs,cfg,device,output,*,fixture_models=None):
        if fixture_models is None and (cfg!=bundle.config('v153') or torch.device(device).type!='cuda'):raise ValueError('v153_fixed_config_cuda')
        self.cfg,self.inputs,self.device=cfg,inputs,torch.device(device)
        self.root=Path(output);self.root.mkdir(parents=True,exist_ok=True)
        self.budget=TrainingBudget(dict(cfg['limits']),shutdown_reserve=60)
        self.binding=binding(cfg,inputs,device);self.step=0;self.durable_step=None;self.last=None
        self.elapsed_prior=0.;self.budget_allowance_prior=0.;self.history=[];self.evaluations={};self.snapshots={}
        self.analysis=Analysis(cfg,inputs['clip'],inputs['spec'],inputs['metadata'],device,self.check)
        models=bundle.load_models(inputs,0,'cpu',frozen=False) if fixture_models is None else fixture_models
        self.stages={}
        for role,names in self.analysis.names.items():
            stage=OracleStage(cfg['teacher'],names,self.analysis.edges[role],len(inputs['spec']['upper_joint_names']) if role=='hand' else 0,self.device)
            stage.model.load_state_dict(models[role].state_dict(),strict=True);self.stages[role]=stage
        self.initial={r:model_fingerprint(s.model) for r,s in self.stages.items()}
        self.binding['initial_original_u0']=self.initial
        # Initial weights come from parent u0; all optimization/RNG state is new.
        seed_run(cfg['seed']);self.initial_rng=rng_state()
        self.binding['initialization']='parent original u0 weights only; new optimizer/scheduler/scaler/RNG/cursor'

    def elapsed(self):return self.elapsed_prior+time.monotonic()-self.budget.start

    def check(self):
        self.budget.check()
        if self.elapsed()+self.budget_allowance_prior>=self.cfg['limits']['cumulative_seconds']-60:raise TimeoutError('v153_cumulative_time_limit')

    def publish(self,path,value,cap=None):
        old=self.budget.limits['checkpoint_bytes'];self.budget.replacing_bytes=path.stat().st_size if path.exists() else 0
        if cap is not None:self.budget.limits['checkpoint_bytes']=min(cap,old)
        try:return _publish(path,value,self.budget)
        finally:self.budget.limits['checkpoint_bytes']=old;self.budget.replacing_bytes=0

    def report(self,status,failure=None):
        result={**self.binding,'status':status,'paired_updates':self.step,'optimizer_updates':2*self.step,
            'durable_paired_updates':self.durable_step,'last_checkpoint':self.last,'snapshots':self.snapshots,
            'evaluations':self.evaluations,'history':self.history,'failure':failure,'noise_sha256':self.analysis.noise_pins,
            'elapsed_this_invocation_seconds':time.monotonic()-self.budget.start,'cumulative_elapsed_seconds':self.elapsed(),
            'cumulative_budget_allowance_seconds':self.budget_allowance_prior,
            'cumulative_elapsed_scope':'measured active wall through checkpoint state capture plus current invocation; previous last-write interval excluded and covered by separate allowance',
            'resume_scope':'explicit existing local run + last SHA; same runtime supported; new VM operational restore not validated',
            'interpretation':'Provisional memorization criteria; failure does not establish a bug or authorize more training'}
        if self.device.type=='cuda':result['peak_cuda_allocated_bytes']=torch.cuda.max_memory_allocated(self.device)
        raw=canonical(result);self.budget.replacing_bytes=(self.root/'run.json').stat().st_size if (self.root/'run.json').exists() else 0
        try:self.budget.storage(self.root,len(raw));_atomic(self.root/'run.json',raw)
        finally:self.budget.replacing_bytes=0
        return result

    def save(self):
        # Reserve one minute for this save in persisted cumulative accounting.
        pin=self.publish(self.root/'last.pt',{'binding':self.binding,'step':self.step,'elapsed':self.elapsed(),
            'save_budget_allowance_seconds':self.budget_allowance_prior+60,
            'fits':{r:_fit_state(s) for r,s in self.stages.items()},'rng':rng_state(),
            'history':self.history,'evaluations':self.evaluations,'snapshots':self.snapshots})
        self.last=pin;self.durable_step=self.step
        return pin

    def resume(self,expected):
        state=_read_checkpoint(self.root/'last.pt',expected,self.cfg['limits']['checkpoint_bytes'])
        if (state['binding']!=self.binding or type(state['step']) is not int or not 0<=state['step']<=self.cfg['updates_per_model']
                or not 0<=state['elapsed']<=self.cfg['limits']['cumulative_seconds']
                or not 0<=state['save_budget_allowance_seconds']<=self.cfg['limits']['cumulative_seconds']):raise ValueError('v153_resume_binding_or_budget')
        for p in state['snapshots'].values():_check_part(self.root/p['name'],p)
        for e in state['evaluations'].values():
            for p in e['artifacts']:_check_part(self.root/p['name'],p)
        for r,s in self.stages.items():_restore_state(s,state['fits'][r],state['step'])
        restore_rng(state['rng']);self.step=self.durable_step=state['step'];self.elapsed_prior=state['elapsed']
        self.budget_allowance_prior=state['save_budget_allowance_seconds']
        self.history,self.evaluations,self.snapshots=state['history'],state['evaluations'],state['snapshots']
        self.last={'name':'last.pt','bytes':(self.root/'last.pt').stat().st_size,'sha256':expected}

    def evaluate(self):
        self.check()
        with preserve_rng():
            models={r:s.model for r,s in self.stages.items()};prediction,traces=self.analysis.sample(models,10)
            metrics=self.analysis.metrics(prediction);label=f'{self.step:05d}'
            artifacts=self.analysis.artifacts(self.root,label,prediction)
            pin=self.publish(self.root/f'models-{label}.pt',{'binding':self.binding,'step':self.step,
                'models':{r:s.model.state_dict() for r,s in self.stages.items()}},256*1024**2)
            record={'metrics':metrics,'x0_head_trace':traces,'reference_criteria':self.analysis.criteria(metrics),
                'constant_gt_clip_mean_diagnostic':self.analysis.baselines(),**artifacts}
            self.snapshots[str(self.step)]=pin;self.evaluations[str(self.step)]=record

    def update_pair(self):
        # An exception anywhere in a pair must not publish a half-updated model.
        states=deepcopy({r:_fit_state(s) for r,s in self.stages.items()});before_rng=rng_state();before=self.step;length=len(self.history)
        try:
            log=self.step==0 or (self.step+1)%self.cfg['log_every']==0;started=time.monotonic()
            metrics={r:s.update(self.analysis.clips[r],self.analysis.upper if r=='hand' else None,log) for r,s in self.stages.items()}
            if self.device.type=='cuda':torch.cuda.synchronize(self.device)
            self.step+=1
            if log:self.history.append({'paired_update':self.step,'metrics':metrics,'wall_seconds':time.monotonic()-started})
        except BaseException:
            self.step=before;del self.history[length:]
            for r,s in self.stages.items():_restore_state(s,states[r],before)
            restore_rng(before_rng);raise

    def run(self,resume=None,stop_after=None):
        with _locked(self.root):
            if resume:self.resume(resume)
            elif any(p.name!='.lock' for p in self.root.iterdir()):raise ValueError('v153_fresh_or_explicit_resume')
            target=min(self.cfg['updates_per_model'],self.step+stop_after) if stop_after else self.cfg['updates_per_model']
            try:
                if self.last is None:self.save()
                while True:
                    self.check()
                    if self.step in self.cfg['milestones'] and str(self.step) not in self.evaluations:
                        self.evaluate();self.save()
                    if self.step>=target:
                        self.save();return self.report('complete' if self.step==self.cfg['updates_per_model'] else 'paused')
                    self.update_pair()
                    if self.step%self.cfg['checkpoint_every']==0:self.save();self.report('running')
            except BaseException as e:
                failure={'type':type(e).__name__,'reason':str(e)[:512]}
                with suppress(Exception):self.save()
                status='stopped' if isinstance(e,KeyboardInterrupt) else 'paused' if isinstance(e,TimeoutError) else 'failed'
                with suppress(Exception):self.report(status,failure)
                if isinstance(e,TimeoutError):return self.report(status,failure)
                raise
