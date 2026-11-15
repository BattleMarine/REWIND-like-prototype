"""Pinned selected-part restoration and shared fixed-F9 features for v1.5."""

import io
import json
import os
from pathlib import Path
import time

import torch

from rewind_archive.contracts import digest
from rewind_m1.local_rgb import _regular_read
from rewind_prepare.converter import sha
from rewind_prepare.temporal_cache import RawCache,MANIFEST,_atomic,_check_part
from rewind_prepare.temporal_selection import seal,check_pin
from scripts.v11_recovery import _measure
from scripts import visualize_archvis_v1 as view
from . import camera_training as training,camera_gauge as gauge,cross_data as cross
from .audit import load_inputs
from .data import Clip,_load_clip
from .diagnostic import HISTORICAL,validate_bundle,CACHE_PARTS,CACHE_PART_BYTES
from .oracle_data import _check_clip_file_pins
from .frontend import frontend_fingerprint,predicted_features


def _save_tensor(path,value,limit):
    stream=io.BytesIO();torch.save(value,stream);raw=stream.getvalue()
    if len(raw)>limit:raise ValueError('v15_feature_tensor_limit')
    _atomic(path,raw);return {'name':Path(path).name,'bytes':len(raw),'sha256':sha(raw)}


def restore_train_parts(inputs,evidence,cache_source,local,output):
    plan,assets=load_inputs(inputs,HISTORICAL['selection_sha256'])
    _,_,cameras=validate_bundle(Path(evidence),training.POLICY['decision_sha256'],plan)
    cache=RawCache(cache_source,HISTORICAL['cache_sha256'])
    if (len(cache.manifest['parts'])!=CACHE_PARTS or sum(p['bytes'] for p in cache.manifest['parts'])!=CACHE_PART_BYTES
            or cache.manifest['selection_sha256']!=plan['content_sha256']):raise ValueError('v15_historical_cache_binding')
    keys=[(seq,f) for seq,start in plan['train_clip_starts'] for f in range(start,start+50)]
    if len(keys)!=1000 or len(set(keys))!=1000 or len(plan['train_clip_starts'])!=20:raise ValueError('v15_fixed1000_train')
    _check_clip_file_pins(plan,cache,keys)
    names={cache.index[k][0] for k in keys};parts=[p for p in cache.manifest['parts'] if p['name'] in names]
    limits=training.POLICY['limits'];total=sum(p['bytes'] for p in parts)
    if total>limits['raw_restore_bytes']:raise ValueError('v15_selected_parts_copy_limit')
    local=Path(local);local.mkdir(exist_ok=True);deadline=time.monotonic()+limits['raw_restore_seconds'];copied=reused=0
    for part in parts:
        target=local/part['name']
        if target.exists():_check_part(target,part,lambda: _time(deadline));reused+=part['bytes'];continue
        temporary=target.with_suffix('.partial')
        try:
            with (Path(cache_source)/part['name']).open('rb') as src,temporary.open('xb') as dst:
                if _measure(src,part['bytes'],deadline,dst)!=(part['bytes'],part['sha256']):raise ValueError('v15_raw_copy_hash')
                dst.flush();os.fsync(dst.fileno())
            temporary.rename(target);copied+=part['bytes']
        finally:temporary.unlink(missing_ok=True)
    raw=_regular_read(Path(cache_source)/MANIFEST,16*cross.MIB)
    if json.loads(raw)!=cache.manifest:raise ValueError('v15_raw_manifest_changed')
    if (local/MANIFEST).exists() and (local/MANIFEST).read_bytes()!=raw:raise ValueError('v15_local_cache_identity')
    _atomic(local/MANIFEST,raw)
    receipt=seal({'schema':'v15_selected_train_parts_v1','run_id':training.RUN_ID,'source_sha256':digest(training.source_identity()),
        'selection_sha256':plan['content_sha256'],'cache_sha256':HISTORICAL['cache_sha256'],'parts':parts,
        'copied_bytes':copied,'verified_local_bytes':reused,'selected_part_bytes':total,'keys':keys,
        'scope':'Partial selected train parts with original full manifest retained; not a full 65-part cache restore'})
    cross.write_report(output,receipt);return receipt


def _time(deadline):
    if time.monotonic()>deadline:raise TimeoutError('v15_prepare_timeout')


def build_train_features(inputs,evidence,cache_local,model,output,device):
    output=Path(output);output.mkdir()
    plan,assets=load_inputs(inputs,HISTORICAL['selection_sha256'])
    decision,_,cameras=validate_bundle(Path(evidence),training.POLICY['decision_sha256'],plan)
    spec=json.loads(assets['spec.json']);bykey={tuple(r['key']):r['camera_pose'] for r in cameras}
    frontend,pins,_,_=view._models(model,spec,False,training_sha256=cross.MODEL_SHA,run_id=view.FULL_RUN_ID)
    frontend=frontend['frontend'].to(device);fingerprint=frontend_fingerprint(frontend)
    feature_identity=digest({'selection':plan['content_sha256'],'raw_cache':HISTORICAL['cache_sha256'],
        'frontend_pin':pins['frontend'],'frontend_fingerprint':fingerprint,'image_size':[128,128],'source':'predicted_2d'})
    cache=RawCache(cache_local,HISTORICAL['cache_sha256']);rows=[];deadline=time.monotonic()+training.POLICY['limits']['feature_seconds'];read_bytes=0
    observed=cross.Budget(training.POLICY['limits']['feature_seconds'])
    for number,(seq,start) in enumerate(plan['train_clip_starts']):
        keys=[(seq,f) for f in range(start,start+50)];_check_clip_file_pins(plan,cache,keys)
        read_bytes+=sum(cache.index[k][1]['files'][role]['size_bytes'] for k in keys for role in ('left','right','pose'))
        if read_bytes>training.POLICY['limits']['feature_source_bytes']:raise ValueError('v15_feature_raw_read_limit')
        def clock(frames):
            from rewind_m2.data import encode_camera
            return torch.stack([encode_camera(bykey[seq,f],100.) for f in frames]),[f/25 for f in frames]
        clip=_load_clip(cache,plan,seq,start,split='train',selection_sha256=plan['content_sha256'],context_sha256=decision['content_sha256'],
            time_context=clock,spec_path=Path(inputs)/'spec.json',scale_mm=100.,source='predicted_2d',frontend=frontend,frontend_size=[128,128],
            frontend_sha256=fingerprint,check_budget=observed.check,context_label='diagnostic_decision')
        meta={'sequence':seq,'start':start,'frames':[k[1] for k in keys],
            'rotation':[[bykey[k][s]['rotation_matrix'] for s in ('left','right')] for k in keys],
            'translation_mm':[[bykey[k][s]['translation'] for s in ('left','right')] for k in keys]}
        absolute,_=training.camera_clip(clip,meta,'absolute')
        if not torch.equal(absolute.camera,clip.camera):raise ValueError('v15_train_original_camera_mismatch')
        tensors={n:getattr(clip,n) for n in ('features','camera','target_mm','target_valid','time_valid','seconds')}
        pin=_save_tensor(output/f'train-{number:02d}.pt',tensors,4*cross.MIB)
        rows.append({'key':[seq,start],'metadata':meta,'tensor':pin})
    receipt=seal({'schema':'v15_features_v1','run_id':training.RUN_ID,'source_sha256':digest(training.source_identity()),
        'feature_identity':feature_identity,'selection_sha256':plan['content_sha256'],'cache_sha256':HISTORICAL['cache_sha256'],
        'frontend':pins['frontend'],'frontend_pretraining_frames':20888,'teacher_train_frames':1000,'teacher_train_clips':20,
        'spec':spec,'clips':rows,'raw_member_bytes_read':read_bytes,'feature_bytes':sum(r['tensor']['bytes'] for r in rows),
        'time_basis':'frame/25 nominal; timestamps/stereo/motion independence unverified'})
    if receipt['feature_bytes']>training.POLICY['limits']['feature_bytes']:raise ValueError('v15_total_feature_limit')
    cross.write_report(output/'features.json',receipt);return receipt


def load_features(root,pin):
    root=Path(root);doc=json.loads(_regular_read(root/'features.json',16*cross.MIB));check_pin(doc,'v15_features_v1',pin)
    if doc['run_id']!=training.RUN_ID or doc['source_sha256']!=digest(training.source_identity()):raise ValueError('v15_feature_source')
    clips={};metadata={}
    for row in doc['clips']:
        key=tuple(row['key']);p=row['tensor'];raw=_regular_read(root/p['name'],4*cross.MIB,p['bytes'])
        if sha(raw)!=p['sha256']:raise ValueError('v15_feature_hash')
        arrays=torch.load(io.BytesIO(raw),map_location='cpu',weights_only=True)
        if not all(torch.isfinite(v).all() for v in arrays.values()):raise ValueError('v15_nonfinite_features')
        clips[key]=Clip(**arrays,source='predicted_2d',split='train',input_sha256=doc['feature_identity']);metadata[key]=row['metadata']
    if len(clips)!=20 or sum(c.features.shape[1] for c in clips.values())!=1000:raise ValueError('v15_feature_fixed_scope')
    return doc,clips,metadata
