"""Standalone v1.2 share loader/API and synthetic inference smoke (no accuracy claim)."""

import argparse
import hashlib
import io
import json
from pathlib import Path
import time

import torch

from rewind_archive.contracts import canonical
from rewind_prepare.converter import sha
from rewind_prepare.temporal_selection import check_pin
from rewind_v1.frontend import PaperHeatmapFrontend
from rewind_v1.teacher import V1Body,V1Hand
from rewind_v1.cross_inference import predict as pipeline_predict
from scripts import visualize_archvis_v1 as view


PARENT_SHA='48661e75e0969ca140ab9cf0e543d73d2bb3b9ec108d30652cf7fb4f4b76b506'
BINDING=('format','input','config','config_sha256','source_code_sha256','environment','coverage')
EXPECTED={
    'frontend':('best-frontend-e009.pt','19363625849d7fc72232cf3af7e438e0c3c594eb52f17c7c15d5a44296153f38'),
    'body':('best-body-e010.pt','94a072551fd3da40621da5a589fbc3d6f919d3e7b52d29ea1c7f6099ef681305'),
    'hand':('best-hand-e009.pt','d6c610ead412cc0495d4adf99ed3872e8c434ebfabecb2e9f551f5c1eb766f8f')}
IMPLEMENTATIONS=('rewind_v1/frontend.py','rewind_v1/data.py','rewind_m1/frontend.py','rewind_m1/rgb_frontend_artifacts.py',
    'rewind_m1/local_rgb.py','rewind_v1/teacher.py','rewind_v1/attention.py','rewind_m2/models.py','rewind_m2/data.py',
    'rewind_m2/diffusion.py','rewind_m2/streaming.py','rewind_m3/models.py','rewind_prepare/converter.py','rewind_prepare/coordinates.py')


def verify_manifest(package_root):
    root=Path(package_root).resolve();manifest=json.loads((root/'SHA256_MANIFEST.json').read_bytes())
    if manifest['schema']!='v12_share_manifest_v1' or manifest['parent_training_zip_sha256']!=PARENT_SHA:
        raise ValueError('share_manifest_identity')
    for name,pin in manifest['files'].items():
        path=root/name
        if '..' in Path(name).parts or not path.resolve().is_relative_to(root) or path.is_symlink():raise ValueError('share_manifest_path')
        if path.stat().st_size!=pin['bytes']:raise ValueError('share_manifest_size:'+name)
        with path.open('rb') as stream:actual=hashlib.file_digest(stream,'sha256').hexdigest()
        if actual!=pin['sha256']:raise ValueError('share_manifest_sha:'+name)
    return manifest


def load_models(package_root,device='cpu'):
    """Verify packaged bytes and original binding, strict-load F9/B10/H9; no optimizer."""
    root=Path(package_root).resolve()
    if Path(view.ROOT).resolve()!=(root/'source').resolve():raise ValueError('share_import_from_packaged_source_required')
    verify_manifest(root);device=torch.device(device)
    if device.type not in ('cpu','cuda') or device.type=='cuda' and not torch.cuda.is_available():
        raise ValueError('share_explicit_supported_device_required')
    report=json.loads((root/'metadata/run.json').read_bytes());spec_raw=(root/'metadata/spec.json').read_bytes();spec=json.loads(spec_raw)
    receipt=json.loads((root/'metadata/original_checkpoint_receipt.json').read_bytes())
    pins=json.loads((root/'metadata/checkpoint_pins.json').read_bytes())
    check_pin(receipt,'archvis_v12_training_recovery_v1',receipt['content_sha256'])
    if (pins['parent_training_zip_sha256']!=PARENT_SHA or pins['original_run_sha256']!=receipt['files']['run.json']['sha256']
            or sha(spec_raw)!=view.SPEC_SHA or report['input']['spec_sha256']!=view.SPEC_SHA
            or report['status']!='completed' or report['cursor']['updates']!=115720 or report['input']['run_id']!=view.FULL_RUN_ID):
        raise ValueError('share_original_metadata_binding')
    view.validate_full_config(report['config'])
    for name in IMPLEMENTATIONS:
        raw=(root/'source'/name).read_bytes();lf=raw.replace(b'\r\n',b'\n')
        if report['source_code_sha256'][name] not in (sha(raw),sha(lf),sha(lf.replace(b'\n',b'\r\n'))):
            raise ValueError('share_original_implementation:'+name)
    models={};dependencies={}
    for role,(name,expected) in EXPECTED.items():
        pin=pins['models'][role];raw=(root/'models'/name).read_bytes()
        if pin['sha256']!=expected or sha(raw)!=expected or pin['bytes']!=len(raw) or pin['name']!=name:
            raise ValueError('share_original_weight_pin:'+role)
        original=receipt['files'][name]
        if original['sha256']!=expected or original['bytes']!=len(raw):raise ValueError('share_receipt_weight:'+role)
        best=report['cursor']['best'][role]
        state=torch.load(io.BytesIO(raw),map_location='cpu',weights_only=True)
        if (state['stage']!=role or state['epoch']!=best['epoch'] or state['score']!=best['score']
                or state['dependencies']!=dependencies or pin['dependencies']!=dependencies
                or any(canonical(state['binding'][k])!=canonical(report[k]) for k in BINDING)):
            raise ValueError('share_original_checkpoint_binding:'+role)
        if role=='frontend':model=PaperHeatmapFrontend(len(spec['joint_names']))
        else:
            names=view._group_names(spec,role);edges=[(names.index(a),names.index(b)) for a,b in spec['edges'] if a in names and b in names]
            cfg=report['config']['teacher']
            model=V1Body(cfg,len(names),edges) if role=='body' else V1Hand(cfg,len(names),edges,len(spec['upper_joint_names']))
        model.load_state_dict(state['model'],strict=True)
        if not all(torch.isfinite(x).all() for x in model.state_dict().values()):raise ValueError('share_nonfinite_weights')
        models[role]=model.to(device).eval().requires_grad_(False);dependencies[role]=expected
    models.update(spec=spec,config=report['config'],device=device)
    return models


@torch.inference_mode()
def predict(models,images,camera,keys):
    """images float32[T,2,3,128,128] RGB 0..1; camera float32[T,2,9]; keys same-sequence contiguous.

    Camera is original absolute world→camera R columns0/1 then t_mm/100, left/right.
    Returns CPU tensors whole/body/hand_local/hand_global in device-coordinate mm. No GT input.
    """
    if (not isinstance(images,torch.Tensor) or images.dtype!=torch.float32 or images.ndim!=5
            or images.shape[1:]!=(2,3,128,128) or not 1<=len(images)<=50 or not torch.isfinite(images).all()
            or (images<0).any() or (images>1).any()):raise ValueError('share_RGB_input_contract')
    if (not isinstance(camera,torch.Tensor) or camera.dtype!=torch.float32 or camera.shape!=(len(images),2,9)
            or not torch.isfinite(camera).all()):raise ValueError('share_camera_input_contract')
    a,b=camera[...,:3],camera[...,3:6]
    if not (torch.allclose(a.norm(dim=-1),torch.ones_like(a[...,0]),atol=1e-5,rtol=0)
            and torch.allclose(b.norm(dim=-1),torch.ones_like(b[...,0]),atol=1e-5,rtol=0)
            and torch.allclose((a*b).sum(-1),torch.zeros_like(a[...,0]),atol=1e-5,rtol=0)):
        raise ValueError('share_camera_rotation6d')
    if not isinstance(keys,(list,tuple)) or len(keys)!=len(images):raise ValueError('share_frame_keys')
    for i,key in enumerate(keys):
        if (not isinstance(key,(tuple,list)) or len(key)!=2 or not isinstance(key[0],str) or not key[0]
                or type(key[1]) is not int or key[1]<0 or key[0]!=keys[0][0] or key[1]!=keys[0][1]+i):
            raise ValueError('share_frame_keys')
    if any(models[r].training or any(p.requires_grad for p in models[r].parameters()) for r in EXPECTED):
        raise ValueError('share_frozen_eval_required')
    predictions,_=pipeline_predict(models,images,camera,keys,models['config'],models['device'])
    return predictions


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--package-root',type=Path,default=Path(__file__).resolve().parents[2])
    parser.add_argument('--device',choices=('cpu','cuda'),default='cpu');parser.add_argument('--frames',type=int,default=2)
    args=parser.parse_args();torch.set_num_threads(1)
    if not 1<=args.frames<=50:raise ValueError('share_smoke_frames_1_to_50')
    start=time.monotonic();models=load_models(args.package_root,args.device)
    images=torch.linspace(0,1,128)[None,None,None,None,:].expand(args.frames,2,3,128,128).contiguous()
    camera=torch.tensor([[[1.,0,0,0,1,0,-.6,0,0],[1.,0,0,0,1,0,.6,0,0]]]).expand(args.frames,2,9).contiguous()
    result=predict(models,images,camera,[('synthetic_share_smoke',i) for i in range(args.frames)])
    print(json.dumps({'status':'passed','scope':'synthetic inference smoke only, not pose accuracy or real RGB evaluation',
        'shapes':{k:list(v.shape) for k,v in result.items()},'all_finite':all(bool(torch.isfinite(v).all()) for v in result.values()),
        'device':args.device,'precision':'fp32','ddim_steps':10,'torch':str(torch.__version__),'elapsed_wall_seconds':time.monotonic()-start}))


if __name__=='__main__':main()
