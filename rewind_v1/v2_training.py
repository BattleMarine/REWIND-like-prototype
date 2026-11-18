"""V2 disk data and unchanged staged learning, gated by explicit decisions."""
from collections.abc import Mapping
from copy import deepcopy
import json
from pathlib import Path
import time

import torch

from rewind_archive.contracts import canonical,digest
from rewind_prepare.temporal_cache import _atomic,_hash_file
from . import v2_contract as contract
from .v2_cache import Cache
from .full_data import pad_clip,FeatureStore
from .camera_training import CameraOrigin
from .frontend import frontend_fingerprint,predicted_features
from .run_stages import CascadeStages,preserve_rng,evaluate_frontend
from .runner import Pipeline,Budget


class ScenarioMap(Mapping):
    def __init__(self,base,scenario):self.base=base;self.keys=[k for k in base if k[0].startswith(scenario+'/')]
    def __iter__(self):return iter(self.keys)
    def __len__(self):return len(self.keys)
    def __getitem__(self,key):
        if key not in self.keys:raise KeyError(key)
        return self.base[key]


class Data:
    width=50
    def __init__(self,doc,spec,cfg,plan,cache,ledger):
        self.spec,self.doc,self.cfg,self.plan,self.ledger=spec,doc,cfg,plan,ledger;self.cache=Cache(cache,doc,spec,cfg,ledger);self.budget=ledger
        self.frames={s:sorted((r['sequence_id'],r['frame_index']) for r in doc['records'] if r['split']==s) for s in ('train','validation')}
        self.clip_rows={(r['sequence'],r['start_frame']):r for r in plan['clips']}
        self.clips={s:sorted(k for k,r in self.clip_rows.items() if r['split']==s) for s in self.frames}
        self.identity={'data_kind':'dataset_sample','run_kind':'nominal_25fps_diagnostic','manifest':doc['content_sha256'],'cache':self.cache.identity,'camera':cfg['camera'],'source':contract.source(),
            'padding':'post_zero_invalid','clock':'nominal25fps diagnostic','spec':digest(spec)}
        self.restore_metrics={}

    def coverage(self):return {'counts':self.plan['counts'],'summary':self.plan['summary'],'validation_reference_only':True}
    def real_frames(self,key,split):return self.clip_rows[key]['real_frames']
    def frame(self,key,split,size):
        if list(size)!=[128,128]:raise ValueError('v2_frame_size')
        image,target=self.cache.frame(key,split);return (image,target['xy'],target['xy_valid']),{}

    def clip(self,key,split,frontend,size,input_sha):
        if frontend.training or any(p.requires_grad for p in frontend.parameters()):raise ValueError('v2_frozen_frontend_required')
        sequence,start=key;features=[];metadata=[];batch=[];count=self.real_frames(key,split)
        for frame in range(start,start+count):
            self.ledger.check();image,target=self.cache.frame((sequence,frame),split);metadata.append(target);batch.append(image)
            if len(batch)==2 or frame==start+count-1:
                with torch.no_grad():features.append(predicted_features(frontend,torch.stack(batch).to(next(frontend.parameters()).device)).cpu())
                batch.clear()
        origin=CameraOrigin('first_midpoint')
        camera,audit=origin.apply(torch.stack([r['rotation64'] for r in metadata]),torch.stack([r['translation64'] for r in metadata]),sequence,start,list(range(start,start+count)))
        clip=pad_clip(torch.cat(features),camera,torch.stack([r['target_mm'] for r in metadata]),torch.stack([r['target_valid'] for r in metadata]),
            sequence=sequence,start=start,split=split,input_sha256=input_sha)
        clip.timings['camera_anchor_world_mm']=audit['state']['anchor_world_mm']
        return clip


class TrainingBudget(Budget):
    def __init__(self,limits,ledger):super().__init__(limits);self.ledger=ledger
    def check(self):self.ledger.check()
    def remaining(self):return self.ledger.remaining()


def runtime_config(cfg,plan):
    contract.training_ready(cfg)
    base=json.loads((contract.ROOT/'configs/rewind_v12_full_training.json').read_bytes());base['epochs']=cfg['training']['epochs'];base['seed']=cfg['training']['seed']
    train_frames=sum(v['train'] for v in plan['counts'].values());train_clips=sum(r['split']=='train' for r in plan['clips'])
    updates=base['epochs']['frontend']*((train_frames+1)//2)+(base['epochs']['body']+base['epochs']['hand'])*train_clips
    base['limits'].update(max_seconds=cfg['seconds_per_invocation'],max_updates=updates,output_bytes=cfg['resources']['run_bytes'],
        feature_bytes=cfg['resources']['feature_bytes'],checkpoint_bytes=cfg['resources']['checkpoint_bytes'])
    return base


class V2Pipeline(Pipeline):
    def __init__(self,cfg,doc,spec,plan,cache,features,device,output,ledger,backup=None):
        runtime=runtime_config(cfg,plan);self.v2cfg=cfg;self.ledger=ledger;self.backup=backup;self.backup_last=ledger.clock()
        data=Data(doc,spec,cfg,plan,cache,ledger);self.feature_root=Path(features)
        super().__init__(runtime,digest(cfg),data,device,output,TrainingBudget(runtime['limits'],ledger))
        self.binding.update(format='rewind_v2_pipeline_v1',v2_config=cfg,source_code_sha256=contract.source())
        self.epoch_expansion=True;self.progress['milestones']={}

    def _features(self):
        with preserve_rng():
            frontend=self._make_frontend();self._load_best('frontend',frontend)
            store=FeatureStore(self.feature_root,self.data,frontend_fingerprint(frontend.model))
            clips,stats=store.build(frontend.model,self.frontend_size)
            self.feature_store_identity=stats['feature_identity'];self.cascade=CascadeStages(self.data,clips,self.cfg,self.device,0)

    def _save_last(self,stage):
        super()._save_last(stage);self.ledger.persist()
        if self.backup is not None and self.ledger.clock()-self.backup_last>=3600:
            from .v2_budget import ResourceLimit
            self.ledger.check()
            if self.ledger.remaining()<self.v2cfg['package_seconds']+self.v2cfg['copy_seconds']:raise TimeoutError('v2_periodic_backup_terminal_reserve')
            for kind in ('read','write'):
                if self.ledger.current[kind+'_bytes_reserved']+4*self.v2cfg['resources']['archive_bytes']>self.v2cfg['resources'][kind+'_bytes']:raise ResourceLimit('v2_periodic_backup_reserve_final_'+kind)
            self.backup();self.backup_last=self.ledger.clock()

    def _save_epoch_model(self,role,stage,metrics):
        # Keep the original pooled score for selection, add independently labeled scenario metrics.
        scenarios={}
        frames,clips=self.data.frames['validation'],self.data.clips['validation']
        feature_validation=self.cascade.clips['validation'] if self.cascade is not None else None
        try:
            with preserve_rng():
                for scenario in contract.SCENARIOS:
                    self.data.frames['validation']=[k for k in frames if k[0].startswith(scenario+'/')]
                    self.data.clips['validation']=[k for k in clips if k[0].startswith(scenario+'/')]
                    if role=='frontend':m=evaluate_frontend(stage,self.data,self.cfg['frontend'],self.budget,self.timing)
                    else:
                        self.cascade.clips['validation']=ScenarioMap(feature_validation,scenario)
                        m=self.cascade.evaluate(role,stage,self.budget,self.timing,self.frozen_body)
                    scenarios[scenario]=m
        finally:
            self.data.frames['validation'],self.data.clips['validation']=frames,clips
            if feature_validation is not None:self.cascade.clips['validation']=feature_validation
        metrics['per_scenario_reference']=scenarios;super()._save_epoch_model(role,stage,metrics)

    def run(self,*args,**kwargs):
        try:return super().run(*args,**kwargs)
        except TimeoutError as e:
            if self.durable_progress is not None:self.progress=deepcopy(self.durable_progress)
            return self._report('paused',{'type':type(e).__name__,'reason':str(e)})
