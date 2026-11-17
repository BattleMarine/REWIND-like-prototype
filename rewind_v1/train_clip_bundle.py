"""Minimal pinned first-train-clip and original midpoint snapshots, shared by v152/v153."""

import hashlib
import io
import json
from pathlib import Path
import tempfile
import time
import zipfile

import torch

from rewind_archive.contracts import canonical,digest
from rewind_m1.local_rgb import _regular_read
from rewind_prepare.converter import sha
from rewind_prepare.temporal_selection import seal,check_pin
from scripts.archvis_pilot_bootstrap import extract_pinned_zip
from . import upper_condition_bundle as base,camera_training as parent
from .upper_condition import IMPLEMENTATIONS
from .data import Clip
from .oracle_data import SEQUENCE
from .teacher import V1Body,V1Hand
from .diagnostic import HISTORICAL


ROOT=base.ROOT;STEPS=(0,200,1000);MAX_BUNDLE=512*1024**2
CONFIGS={'v152':'configs/v152_sampling_diagnostic.json','v153':'configs/v153_predicted_clip.json'}


def config(experiment):return json.loads((ROOT/CONFIGS[experiment]).read_bytes())


def source_identity():
    names=(*CONFIGS.values(),'scripts/build_train_clip_diagnostics.py')
    return {**base.source_identity(),**{n:sha((ROOT/n).read_bytes()) for n in names}}


def selected():
    return ('recovery_receipt.json','setup.json','train/run.json','features/features.json','features/train-00.pt','evaluation/report.json',
        *(f'train/first_midpoint-{role}-u{step:04d}.pt' for step in STEPS for role in ('body','hand')))


def build_bundle(archive,output):
    output=Path(output);archive=Path(archive);started=time.monotonic();deadline=started+600;reads=[0]
    if output.exists() or archive.is_symlink() or archive.stat().st_size!=base.POLICY['parent_zip_bytes']:raise ValueError('clip_bundle_parent_size_or_output')
    with archive.open('rb') as stream:
        if base._measure(stream,2*1024**3,deadline,reads)['sha256']!=base.POLICY['parent_zip_sha256']:raise ValueError('clip_bundle_parent_sha')
        stream.seek(0)
        with zipfile.ZipFile(stream) as z:
            names=z.namelist()
            if len(names)>128 or len(names)!=len(set(names)) or sum(i.file_size for i in z.infolist())>2*1024**3:raise ValueError('clip_bundle_archive_cap')
            receipt=json.loads(z.read('recovery_receipt.json'))
            if receipt['source_sha256']!=base.POLICY['parent_source_sha256'] or receipt['run_id']!=base.POLICY['parent_run_id'] or set(names)!=set(receipt['files'])|{'recovery_receipt.json'}:raise ValueError('clip_bundle_receipt')
            for name,pin in receipt['files'].items():
                with z.open(name) as member:
                    if base._measure(member,pin['bytes'],deadline,reads)!=pin:raise ValueError('clip_bundle_parent_member')
                if name.endswith('.json'):
                    if pin['bytes']>32*1024**2:raise ValueError('clip_bundle_json_cap')
                    raw=z.read(name);reads[0]+=len(raw);d=json.loads(raw)
                    if 'content_sha256' in d:check_pin(d,d['schema'],d['content_sha256'])
            if sum(z.getinfo(n).file_size for n in selected())>MAX_BUNDLE-1024**2:raise ValueError('clip_bundle_size')
            with tempfile.TemporaryDirectory(dir=output.parent) as tmp:
                root=Path(tmp);pins={}
                for name in selected():
                    p=root/name;p.parent.mkdir(parents=True,exist_ok=True)
                    with z.open(name) as src,p.open('xb') as dst:pins[name]=base._measure(src,MAX_BUNDLE,deadline,reads,dst)
                doc=seal({'schema':'v15_first_train_bundle_v1','parent_zip_sha256':base.POLICY['parent_zip_sha256'],
                    'parent_source_sha256':base.POLICY['parent_source_sha256'],'files':pins,'parent_logical_read_bytes':reads[0],
                    'scope':'exact first train clip predicted F9 features plus u0/u200/u1000 midpoint weights; no optimizer/raw/eval tensors'})
                (root/'bundle.json').write_bytes(canonical(doc));checked(root,doc['content_sha256'])
                with zipfile.ZipFile(output,'x') as target:
                    for name in (*selected(),'bundle.json'):target.write(root/name,name)
    if output.stat().st_size>MAX_BUNDLE:raise ValueError('clip_bundle_zip_cap')
    with output.open('rb') as stream:pin=hashlib.file_digest(stream,'sha256').hexdigest()
    return {'name':output.name,'bytes':output.stat().st_size,'sha256':pin,'manifest_sha256':doc['content_sha256'],'build_wall_seconds':time.monotonic()-started,'parent_logical_read_bytes':reads[0]}


def extract(archive,pin,manifest_pin,root):
    extract_pinned_zip(archive,pin,root,max_zip_bytes=MAX_BUNDLE,max_output_bytes=MAX_BUNDLE,max_members=64)
    return checked(root,manifest_pin)


def checked(root,pin):
    root=base.cross._local(root,directory=True);doc=base._doc(root,'bundle.json');check_pin(doc,'v15_first_train_bundle_v1',pin)
    if doc['parent_zip_sha256']!=base.POLICY['parent_zip_sha256'] or doc['parent_source_sha256']!=base.POLICY['parent_source_sha256'] or set(doc['files'])!=set(selected()):raise ValueError('clip_bundle_scope')
    if {p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file()}!=set(selected())|{'bundle.json'}:raise ValueError('clip_bundle_members')
    for name,p in doc['files'].items():
        raw=_regular_read(root/name,128*1024**2,p['bytes'])
        if sha(raw)!=p['sha256']:raise ValueError('clip_bundle_file_sha')
    receipt=base._doc(root,'recovery_receipt.json');run=base._doc(root,'train/run.json');setup=base._doc(root,'setup.json')
    features=base._doc(root,'features/features.json','v15_features_v1');evaluation=base._doc(root,'evaluation/report.json','v15_camera_eval_report_v1')
    if (receipt['source_sha256']!=base.POLICY['parent_source_sha256'] or digest(run['source'])!=base.POLICY['parent_source_sha256']
            or run['status']!='complete' or run['config']!=parent.POLICY or run['position']!={'role':'complete','pair':1000,'total_pairs':2000}
            or setup['features_sha256']!=features['content_sha256'] or run['input']['features_sha256']!=features['content_sha256']
            or features['selection_sha256']!=HISTORICAL['selection_sha256'] or features['cache_sha256']!=HISTORICAL['cache_sha256']
            or features['frontend']!=run['input']['frontend'] or features['source_sha256']!=base.POLICY['parent_source_sha256']
            or canonical(features['spec'])!=canonical(run['spec']) or len(features['clips'])!=20
            or evaluation['content_sha256']!=base.POLICY['parent_report_sha256']):raise ValueError('clip_bundle_parent_chain')
    for name,p in doc['files'].items():
        if name!='recovery_receipt.json' and receipt['files'].get(name)!=p:raise ValueError('clip_bundle_original_pin')
    row=features['clips'][0];key=[SEQUENCE,0];meta=row['metadata']
    if (row['key']!=key or run['keys'][0]!=key or evaluation['train_fit']['key']!=key
            or meta['sequence']!=SEQUENCE or meta['start']!=0 or meta['frames']!=list(range(50))
            or row['tensor']!={'name':'train-00.pt',**doc['files']['features/train-00.pt']}):raise ValueError('clip_bundle_first_train_membership')
    tensors=torch.load(io.BytesIO(_regular_read(root/'features/train-00.pt',4*1024**2)),map_location='cpu',weights_only=True)
    if (tensors['features'].shape!=(1,50,868) or tensors['target_mm'].shape!=(1,50,62,3) or tensors['camera'].shape!=(1,50,2,9)
            or tensors['target_valid'].dtype!=torch.bool or not tensors['time_valid'].all()
            or not all(torch.isfinite(v).all() for v in tensors.values())):raise ValueError('clip_bundle_train_tensor')
    clip=Clip(**tensors,source='predicted_2d',split='train',input_sha256=features['feature_identity'])
    original,_=parent.camera_clip(clip,meta,'absolute')
    if not torch.equal(original.camera,clip.camera):raise ValueError('clip_bundle_original_camera')
    clip,anchor=parent.camera_clip(clip,meta,'first_midpoint')
    for step in STEPS:
        for role in ('body','hand'):
            name=f'train/first_midpoint-{role}-u{step:04d}.pt'
            if run['snapshots'][f'first_midpoint/{role}/{step}']!={'name':Path(name).name,**doc['files'][name]}:raise ValueError('clip_bundle_snapshot_pin')
    return {'bundle':doc,'run':run,'spec':features['spec'],'clip':clip,'metadata':meta,'anchor':anchor,'root':root,'feature_identity':features['feature_identity']}


def load_models(inputs,step,device,*,frozen=True):
    if step not in STEPS:raise ValueError('clip_snapshot_step')
    for name in (*IMPLEMENTATIONS,'rewind_v1/oracle_overfit.py'):
        raw=(ROOT/name).read_bytes();lf=raw.replace(b'\r\n',b'\n')
        if inputs['run']['source'][name] not in (sha(raw),sha(lf),sha(lf.replace(b'\n',b'\r\n'))):raise ValueError('clip_original_implementation:'+name)
    models={};spec=inputs['spec'];cfg=inputs['run']['config']['teacher']
    for role in ('body','hand'):
        path=inputs['root']/f'train/first_midpoint-{role}-u{step:04d}.pt';raw=_regular_read(path,128*1024**2)
        state=torch.load(io.BytesIO(raw),map_location='cpu',weights_only=True)
        if (state['step']!=step or state['role']!=role or state['arm']!='first_midpoint'
                or any(canonical(v)!=canonical(inputs['run'].get(k)) for k,v in state['binding'].items())):raise ValueError('clip_snapshot_binding')
        names=spec['body_joint_names'] if role=='body' else [n for s in ('left','right') for n in spec['hands'][s][1:]]
        edges=[(names.index(a),names.index(b)) for a,b in spec['edges'] if a in names and b in names]
        model=V1Body(cfg,len(names),edges) if role=='body' else V1Hand(cfg,len(names),edges,len(spec['upper_joint_names']))
        model.load_state_dict(state['model'],strict=True)
        if not all(torch.isfinite(v).all() for v in model.state_dict().values()):raise ValueError('clip_snapshot_finite')
        if step==0 and parent.model_fingerprint(model)!=inputs['run']['initial'][role]:raise ValueError('clip_original_initial_fingerprint')
        models[role]=model.to(device).eval().requires_grad_(not frozen)
    return models
