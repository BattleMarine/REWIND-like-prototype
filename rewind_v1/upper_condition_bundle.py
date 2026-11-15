"""Verify a pinned parent once and retain only frozen diagnostic inputs, never optimizer state."""

import hashlib
import io
import json
from pathlib import Path,PurePosixPath
import tempfile
import time
import zipfile

import torch

from rewind_archive.contracts import canonical,digest
from rewind_m1.local_rgb import _regular_read
from rewind_prepare.converter import sha
from rewind_prepare.temporal_selection import seal,check_pin
from scripts.archvis_pilot_bootstrap import extract_pinned_zip
from . import camera_training as parent,cross_data as cross


ROOT=Path(__file__).resolve().parents[1]
CONFIG=ROOT/'configs/v151_upper_condition_001.json'
POLICY=json.loads(CONFIG.read_bytes());LIMITS=POLICY['limits'];RUN_ID=POLICY['run_id']


def source_identity():
    return {**parent.source_identity(),**{n:sha((ROOT/n).read_bytes()) for n in ('configs/v151_upper_condition_001.json','scripts/build_v151_upper_condition.py')}}


def selected_files():
    result=['recovery_receipt.json','setup.json','train/run.json','eval_data/eval_features.json','evaluation/report.json']
    result += [f'train/first_midpoint-{role}-u1000.pt' for role in ('body','hand')]
    for i in range(6):result += [f'eval_data/eval-{i:02d}.pt',f'evaluation/clip-{i:02d}/report.json',f'evaluation/clip-{i:02d}/first_midpoint.json']
    return tuple(result)


def _doc(root,name,schema=None):
    d=json.loads(_regular_read(root/name,32*cross.MIB))
    if schema:check_pin(d,schema,d.get('content_sha256'))
    return d


def _measure(stream,maximum,deadline,budget,output=None):
    h=hashlib.sha256();total=0
    while block:=stream.read(min(cross.MIB,maximum-total+1)):
        if time.monotonic()>deadline:raise TimeoutError('v151_parent_verify_timeout')
        total+=len(block);budget[0]+=len(block)
        if total>maximum or budget[0]>LIMITS['parent_read_bytes']:raise ValueError('v151_parent_read_limit')
        h.update(block)
        if output is not None:output.write(block)
    return {'bytes':total,'sha256':h.hexdigest()}


def build_bundle(archive,output):
    archive=Path(archive);output=Path(output)
    if output.exists() or output.is_symlink() or archive.is_symlink():raise ValueError('v151_new_bundle_output')
    if archive.stat().st_size!=POLICY['parent_zip_bytes']:raise ValueError('v151_parent_zip_size')
    deadline=time.monotonic()+LIMITS['parent_seconds'];budget=[0]
    with archive.open('rb') as stream:
        if _measure(stream,LIMITS['parent_zip_bytes'],deadline,budget)['sha256']!=POLICY['parent_zip_sha256']:raise ValueError('v151_parent_zip_sha')
        stream.seek(0)
        with zipfile.ZipFile(stream) as z:
            infos=z.infolist();names=[i.filename for i in infos]
            if (len(infos)>LIMITS['parent_members'] or len(set(names))!=len(names)
                    or sum(i.file_size for i in infos)>LIMITS['parent_zip_bytes']
                    or any(PurePosixPath(n).is_absolute() or '..' in PurePosixPath(n).parts or '\\' in n for n in names)):
                raise ValueError('v151_parent_archive_members')
            def read_json(name):
                if z.getinfo(name).file_size>32*cross.MIB:raise ValueError('v151_parent_json_cap')
                raw=z.read(name);budget[0]+=len(raw)
                if time.monotonic()>deadline:raise TimeoutError('v151_parent_verify_timeout')
                if budget[0]>LIMITS['parent_read_bytes']:raise ValueError('v151_parent_read_limit')
                return json.loads(raw)
            receipt=read_json('recovery_receipt.json')
            if (receipt['schema']!='v15_training_recovery_v1' or receipt['run_id']!=POLICY['parent_run_id']
                    or receipt['source_sha256']!=POLICY['parent_source_sha256'] or len(receipt['files'])!=90
                    or set(names)!=set(receipt['files'])|{'recovery_receipt.json'}):raise ValueError('v151_parent_receipt')
            for name,pin in receipt['files'].items():
                if z.getinfo(name).file_size!=pin['bytes']:raise ValueError('v151_parent_member_size')
                with z.open(name) as member:
                    if _measure(member,pin['bytes'],deadline,budget)!=pin:raise ValueError('v151_parent_member_sha')
                if name.endswith('.json'):
                    if pin['bytes']>32*cross.MIB:raise ValueError('v151_parent_json_cap')
                    d=read_json(name)
                    if 'content_sha256' in d:check_pin(d,d['schema'],d['content_sha256'])
            selected=selected_files();pins={}
            if sum(z.getinfo(n).file_size for n in selected)>LIMITS['bundle_bytes']-cross.MIB:raise ValueError('v151_bundle_size')
            with tempfile.TemporaryDirectory(dir=output.parent) as temp:
                root=Path(temp)
                for name in selected:
                    p=root/name;p.parent.mkdir(parents=True,exist_ok=True)
                    with z.open(name) as src,p.open('xb') as dst:pins[name]=_measure(src,LIMITS['bundle_bytes'],deadline,budget,dst)
                manifest=seal({'schema':'v151_upper_bundle_v1','run_id':RUN_ID,'parent_zip_bytes':POLICY['parent_zip_bytes'],
                    'parent_zip_sha256':POLICY['parent_zip_sha256'],'parent_source_sha256':POLICY['parent_source_sha256'],
                    'parent_report_sha256':POLICY['parent_report_sha256'],'files':pins,
                    'parent_files_verified':len(receipt['files']),'parent_logical_bytes_read':budget[0],
                    'scope':'Original frozen model and eval members only; parent archive fully hashed and all member pins verified; no optimizer extraction'})
                (root/'bundle.json').write_bytes(canonical(manifest));checked_bundle(root,manifest['content_sha256'])
                with zipfile.ZipFile(output,'x',compression=zipfile.ZIP_STORED) as target:
                    for name in (*selected,'bundle.json'):target.write(root/name,name)
    if output.stat().st_size>LIMITS['bundle_bytes']:raise ValueError('v151_bundle_zip_cap')
    with output.open('rb') as stream:pin=hashlib.file_digest(stream,'sha256').hexdigest()
    return {'name':output.name,'bytes':output.stat().st_size,'sha256':pin,'manifest_sha256':manifest['content_sha256'],
            'parent_zip_sha256':POLICY['parent_zip_sha256'],'parent_logical_bytes_read':budget[0],
            'bundle_build_wall_seconds':time.monotonic()-(deadline-LIMITS['parent_seconds'])}


def extract_bundle(archive,pin,manifest_pin,output):
    extract_pinned_zip(archive,pin,output,max_zip_bytes=LIMITS['bundle_bytes'],max_output_bytes=LIMITS['bundle_bytes'],max_members=LIMITS['bundle_members'])
    return checked_bundle(output,manifest_pin)


def checked_bundle(root,expected):
    root=cross._local(root,directory=True);bundle=_doc(root,'bundle.json');check_pin(bundle,'v151_upper_bundle_v1',expected)
    if (bundle['run_id']!=RUN_ID or bundle['parent_zip_sha256']!=POLICY['parent_zip_sha256']
            or bundle['parent_zip_bytes']!=POLICY['parent_zip_bytes'] or bundle['parent_source_sha256']!=POLICY['parent_source_sha256']
            or bundle['parent_report_sha256']!=POLICY['parent_report_sha256'] or bundle['parent_files_verified']!=90
            or set(bundle['files'])!=set(selected_files())):raise ValueError('v151_bundle_scope_parent')
    actual={p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file()}
    if actual!=set(selected_files())|{'bundle.json'}:raise ValueError('v151_bundle_extra_members')
    for name,pin in bundle['files'].items():
        p=cross._local(root/name)
        if not p.is_relative_to(root):raise ValueError('v151_bundle_member_path')
        raw=_regular_read(p,LIMITS['bundle_bytes'],pin['bytes'])
        if sha(raw)!=pin['sha256']:raise ValueError('v151_bundle_member_sha')
    receipt=_doc(root,'recovery_receipt.json');setup=_doc(root,'setup.json');run=_doc(root,'train/run.json')
    features=_doc(root,'eval_data/eval_features.json','v15_eval_features_v1');report=_doc(root,'evaluation/report.json','v15_camera_eval_report_v1')
    if (receipt['source_sha256']!=POLICY['parent_source_sha256'] or setup['source_sha256']!=POLICY['parent_source_sha256']
            or digest(run['source'])!=POLICY['parent_source_sha256'] or report['content_sha256']!=POLICY['parent_report_sha256']
            or report['source_sha256']!=POLICY['parent_source_sha256'] or features['source_sha256']!=POLICY['parent_source_sha256']
            or run['run_id']!=POLICY['parent_run_id'] or run['status']!='complete' or run['optimizer_updates']!=4000
            or run['position']!={'role':'complete','pair':1000,'total_pairs':2000} or run['config']!=parent.POLICY
            or setup['eval_features_sha256']!=features['content_sha256'] or report['feature_receipt_sha256']!=features['content_sha256']
            or report['checkpoint_sha256']!=run['last']['sha256'] or len(features['clips'])!=6 or len(report['clips'])!=6
            or features['frontend']!=run['input']['frontend']):raise ValueError('v151_parent_chain')
    for name,pin in bundle['files'].items():
        if name!='recovery_receipt.json' and receipt['files'].get(name)!=pin:raise ValueError('v151_parent_selected_pin')
    for role in ('body','hand'):
        name=f'train/first_midpoint-{role}-u1000.pt';pin=bundle['files'][name]
        if pin['sha256']!=POLICY['model_sha256'][role] or run['snapshots'][f'first_midpoint/{role}/1000']!={'name':Path(name).name,**pin}:
            raise ValueError('v151_final_weight_pin')
    clips=[]
    for i,(row,summary) in enumerate(zip(features['clips'],report['clips'],strict=True)):
        clip=_doc(root,f'evaluation/clip-{i:02d}/report.json','v15_camera_eval_clip_v1')
        baseline=_doc(root,f'evaluation/clip-{i:02d}/first_midpoint.json','v15_camera_eval_arm_v1')
        meta=row['metadata'];keys=[[meta['sequence'],f] for f in meta['frames']]
        if (meta['frames']!=list(range(meta['start'],meta['start']+50)) or row['entry']['split']!='validation'
                or clip['number']!=i or clip['sequence']!=meta['sequence'] or row['entry']['sequence']!=meta['sequence']
                or [r['frame'] for r in row['entry']['frames']]!=meta['frames'] or clip['content_sha256']!=summary['content_sha256']
                or clip['arms']['first_midpoint']['content_sha256']!=baseline['content_sha256'] or baseline['keys']!=keys
                or baseline['arm']!='first_midpoint' or baseline['source_sha256']!=POLICY['parent_source_sha256']):raise ValueError('v151_clip_order_keys')
        pin=row['tensor'];name=f'eval_data/eval-{i:02d}.pt'
        if pin!={'name':Path(name).name,**bundle['files'][name]}:raise ValueError('v151_eval_tensor_pin')
        tensors=torch.load(io.BytesIO(_regular_read(root/name,4*cross.MIB)),map_location='cpu',weights_only=True)
        if (tensors['features'].shape!=(50,868) or tensors['camera'].shape!=(50,2,9) or tensors['target'].shape!=(50,62,3)
                or tensors['target_valid'].shape!=(50,62) or tensors['target_valid'].dtype!=torch.bool
                or not all(torch.isfinite(v).all() for v in tensors.values())
                or sha(tensors['features'].numpy().tobytes())!=baseline['features_sha256']
                or not torch.equal(tensors['target'],torch.tensor(baseline['gt_target_device_mm']))
                or not torch.equal(tensors['target_valid'],torch.tensor(baseline['gt_target_valid']))):raise ValueError('v151_frozen_feature_gt')
        absolute,_=parent.CameraOrigin('absolute').apply(meta['rotation'],meta['translation_mm'],meta['sequence'],meta['start'],meta['frames'])
        if not torch.equal(absolute,tensors['camera']):raise ValueError('v151_raw_camera_mismatch')
        camera,audit=parent.CameraOrigin('first_midpoint').apply(meta['rotation'],meta['translation_mm'],meta['sequence'],meta['start'],meta['frames'])
        anchor=audit['state'];old=baseline['anchor']['state']
        if (anchor['clip']!=old['clip'] or max(abs(a-b) for a,b in zip(anchor['anchor_world_mm'],old['anchor_world_mm']))>1e-6):
            raise ValueError('v151_parent_anchor')
        clips.append({'metadata':meta,'data':tensors,'camera':camera,'parent':baseline,'keys':keys})
    if len({c['metadata']['sequence'] for c in clips})!=6:raise ValueError('v151_six_distinct_clips')
    return {'bundle':bundle,'run':run,'spec':features['spec'],'clips':clips,'frontend':features['frontend']}
