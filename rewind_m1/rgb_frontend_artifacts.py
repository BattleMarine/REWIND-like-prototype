"""명시 resize의 2D 추론 weights와 native 좌표 cache. 학습/CLI/UI import 없음."""
import copy
import hashlib
import io
from pathlib import Path
import random

import torch
from torch.nn import functional as F

from .data import InputContract, content_hash
from .frontend import StereoHeatmapNet, prediction_cache, weights_hash
from .local_rgb import LocalStereoRGB
from .prediction_join import _serialized
from .shards import _hash, _integer, _read, _require

FORMAT = 'm1_rgb_resized_frontend_weights_v1'
MAX_BYTES = 64*1024**2


def training_size(size):
    _require(isinstance(size,list) and len(size)==2 and all(_integer(v,4,256) for v in size), 'rgb_training_size')
    return list(size)


def preprocessing_version(size):
    width,height = training_size(size)
    return f'rgb_bilinear_ac1_{width}x{height}_native_v1'


def resize_images(images, size):
    width,height = training_size(size)
    _require(images.ndim==5 and images.shape[0]==1 and images.shape[1:3]==(2,3) and
             images.dtype==torch.float32 and min(images.shape[-2:])>=4, 'rgb_resize_one_native_sample')
    return F.interpolate(images.flatten(0,1),size=(height,width),mode='bilinear',align_corners=True).reshape(1,2,3,height,width)


def scale_xy(xy, source_size, destination_size):
    _require(len(source_size)==len(destination_size)==2 and
             all(type(v) is int and v>=2 for v in (*source_size,*destination_size)), 'rgb_coordinate_sizes')
    factor = xy.new_tensor([(destination_size[i]-1)/(source_size[i]-1) for i in (0,1)])
    return xy*factor  # 범위 밖 GT를 clip하지 않는다.


def _validate(meta):
    _require(isinstance(meta,dict) and set(meta)=={'input_contract','training_image_size','preprocessing_version',
             'frontend_version','catalog_sha256','checkpoint_sha256','source','training_step'}, 'rgb_weights_metadata')
    contract = InputContract.from_dict(meta['input_contract'])
    _require(contract.input_source=='gt_2d' and contract.data_kind=='synthetic_fixture' and
             1 <= len(contract.joint_names) <= 256 and contract.image_sizes[0]==contract.image_sizes[1], 'rgb_weights_contract')
    _require(meta['preprocessing_version']==preprocessing_version(meta['training_image_size']), 'rgb_weights_preprocessing')
    _require(isinstance(meta['frontend_version'],str) and 0 < len(meta['frontend_version'].strip()) <= 128 and
             _hash(meta['catalog_sha256']) and _hash(meta['checkpoint_sha256']) and
             _integer(meta['training_step'],0,2**63-1), 'rgb_weights_identity')
    source=meta['source']
    _require(isinstance(source,dict) and set(source)=={'revision','code_sha256'} and
             isinstance(source['revision'],str) and 0 < len(source['revision']) <= 128 and
             isinstance(source['code_sha256'],dict) and 0 < len(source['code_sha256']) <= 1024 and
             all(isinstance(k,str) and _hash(v) for k,v in source['code_sha256'].items()), 'rgb_weights_source')
    return contract


def export_weights(model, config, contract, catalog_sha256, source, step, checkpoint_sha256, output):
    meta={'input_contract':copy.deepcopy(contract),'training_image_size':training_size(config['training_image_size']),
          'preprocessing_version':preprocessing_version(config['training_image_size']),
          'frontend_version':config['frontend_version'],'catalog_sha256':catalog_sha256,
          'checkpoint_sha256':checkpoint_sha256,'source':{k:copy.deepcopy(source[k]) for k in ('revision','code_sha256')},
          'training_step':step}
    parsed=_validate(meta)
    state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    with torch.random.fork_rng(devices=[]): reference=StereoHeatmapNet(len(parsed.joint_names))
    reference.load_state_dict(state,strict=True)
    _require(all(t.dtype==torch.float32 and torch.isfinite(t).all().item() for t in state.values()), 'rgb_weights_nonfinite_dtype')
    fingerprint=weights_hash(reference)
    artifact={'format_version':FORMAT,'weights_only':True,'metadata':meta,'metadata_sha256':content_hash(meta),
              'frontend_checkpoint_sha256':fingerprint,'model':state}
    buffer=io.BytesIO(); torch.save(artifact,buffer)
    _require(buffer.getbuffer().nbytes <= MAX_BYTES, 'rgb_weights_byte_limit')
    path=Path(output); path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('xb') as stream: stream.write(buffer.getbuffer())
    return fingerprint


def load_weights(path, *, expected_catalog_sha256, expected_checkpoint_sha256, expected_source,
                 expected_contract, expected_preprocessing_version, device='cpu'):
    raw=_read(path,MAX_BYTES)
    artifact=torch.load(io.BytesIO(raw),map_location='cpu',weights_only=True)
    _require(isinstance(artifact,dict) and set(artifact)=={'format_version','weights_only','metadata','metadata_sha256',
             'frontend_checkpoint_sha256','model'} and artifact['format_version']==FORMAT and artifact['weights_only'] is True,
             'rgb_weights_format')
    meta=artifact['metadata']; contract=_validate(meta)
    _require(content_hash(meta)==artifact['metadata_sha256'], 'rgb_weights_metadata_hash')
    _require(meta['catalog_sha256']==expected_catalog_sha256 and meta['checkpoint_sha256']==expected_checkpoint_sha256 and
             meta['source']==expected_source and meta['input_contract']==expected_contract and
             meta['preprocessing_version']==expected_preprocessing_version, 'rgb_weights_expected_identity')
    with torch.random.fork_rng(devices=[]): model=StereoHeatmapNet(len(contract.joint_names))
    _require(isinstance(artifact['model'],dict) and
             all(isinstance(t,torch.Tensor) and t.dtype==torch.float32 and torch.isfinite(t).all().item()
                 for t in artifact['model'].values()), 'rgb_weights_nonfinite_dtype')
    model.load_state_dict(artifact['model'],strict=True)
    _require(weights_hash(model)==artifact['frontend_checkpoint_sha256'], 'rgb_weights_fingerprint')
    model.to(device).eval()
    artifact['artifact_file_sha256']=hashlib.sha256(raw).hexdigest()
    return model, copy.deepcopy(artifact)


def build_cache(loader, model, artifact, *, check_budget=None):
    """한 native sample씩 resize 추론하고 기존 native cache schema로 반환한다."""
    _require(isinstance(loader,LocalStereoRGB) and 1 <= len(loader) <= 256, 'rgb_cache_loader_record_limit')
    _require(artifact.get('format_version')==FORMAT and artifact.get('weights_only') is True, 'rgb_weights_format')
    meta=artifact['metadata']; contract=_validate(meta)
    _require(content_hash(meta)==artifact['metadata_sha256'] and weights_hash(model)==artifact['frontend_checkpoint_sha256'],
             'rgb_cache_weights_identity')
    _require(_hash(artifact.get('artifact_file_sha256')),'rgb_cache_loaded_artifact_sha_required')
    check=check_budget or (lambda:None)
    previous_mode=model.training; python_rng=random.getstate()
    device=next(model.parameters()).device
    cuda_devices=list(range(torch.cuda.device_count())) if device.type=='cuda' else []
    records, splits = [], []
    body={'schema_version':'predicted_2d_cache_v1','data_kind':contract.data_kind,'input_source':'predicted_2d',
          'manifest_hash':loader.manifest_sha256,'image_sizes':[list(s) for s in contract.image_sizes],
          'joint_names':list(contract.joint_names),'frontend_version':meta['frontend_version'],
          'frontend_checkpoint_sha256':artifact['frontend_checkpoint_sha256'],
          'preprocessing_version':meta['preprocessing_version'],'records':records,'record_splits':splits,
          'rgb_shard_provenance':{'checkpoint_sha256':meta['checkpoint_sha256'],'catalog_sha256':meta['catalog_sha256'],
              'weights_artifact_sha256':artifact['artifact_file_sha256'],'weights_metadata_sha256':artifact['metadata_sha256'],
              'weights_format':FORMAT,'source_revision':meta['source']['revision']}}
    output_bytes=len(_serialized({**body,'content_sha256':'0'*64}))
    try:
        with torch.random.fork_rng(devices=cuda_devices):
            model.eval()
            with torch.no_grad():
                for i in range(len(loader)):
                    check()
                    info=loader.metadata(i); native=list(contract.image_sizes[0])
                    _require(info['data_kind']==contract.data_kind and
                             [info['record']['views'][s]['image_size'] for s in ('left','right')]==[native,native],
                             'rgb_cache_native_contract')
                    sample=loader[i]
                    resized=resize_images(sample['images'].unsqueeze(0),meta['training_image_size'])
                    del sample
                    key=(info['record']['sequence_id'],info['record']['frame_index'])
                    partial=prediction_cache(model,resized.to(device),[key],list(contract.joint_names),
                        meta['frontend_version'],meta['preprocessing_version'],loader.manifest_sha256,contract.data_kind)
                    record=partial['records'][0]
                    for side in ('left','right'):
                        xy=torch.tensor(record['prediction'][side]['coordinates'],dtype=torch.float32)
                        record['prediction'][side]['coordinates']=scale_xy(xy,meta['training_image_size'],native).tolist()
                    split=info['record']['split']
                    output_bytes+=len(_serialized(record))+len(json_bytes(split))+(2 if records else 0)
                    _require(output_bytes+1 <= 4*1024**2,'rgb_cache_byte_limit')
                    records.append(record); splits.append(split)
                    del resized,partial
    finally:
        model.train(previous_mode); random.setstate(python_rng)
    cache={**body,'content_sha256':content_hash(body)}
    _serialized(cache)
    _require(weights_hash(model)==artifact['frontend_checkpoint_sha256'], 'rgb_cache_weights_changed')
    return cache


def json_bytes(value):
    import json
    return json.dumps(value,separators=(',',':'),ensure_ascii=True).encode('utf8')
