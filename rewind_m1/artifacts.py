"""UI가 trainer/CLI를 import하지 않고 사용하는 M1 regression 추론 가중치."""
import copy
import io
import math
from pathlib import Path

import torch

from .data import InputContract, content_hash
from .models import make_model
from .shards import _hash, _integer, _read, _require

FORMAT = 'm1_regression_weights_v1'
MAX_BYTES = 64*1024**2


def _metadata(meta):
    _require(isinstance(meta, dict) and set(meta) == {'model_config', 'input_contract',
             'catalog_sha256', 'source', 'training_step', 'checkpoint_sha256'}, 'weights_metadata_fields')
    config = meta['model_config']
    _require(isinstance(config, dict) and set(config) == {'model', 'window_size', 'hidden', 'dropout', 'target_scale_mm'},
             'weights_model_config')
    _require(config['model'] in ('single_frame', 'temporal') and
             _integer(config['window_size'], 1, 256) and _integer(config['hidden'], 1, 1024), 'weights_model_config')
    _require(type(config['dropout']) in (int, float) and math.isfinite(config['dropout']) and
             0 <= config['dropout'] < 1 and type(config['target_scale_mm']) in (int,float) and
             math.isfinite(config['target_scale_mm']) and config['target_scale_mm'] > 0, 'weights_model_config')
    contract = InputContract.from_dict(meta['input_contract'])
    _require(len(contract.joint_names) <= 256, 'weights_joint_limit')
    _require(_hash(meta['catalog_sha256']) and _hash(meta['checkpoint_sha256']) and
             _integer(meta['training_step'], 0, 2**63-1), 'weights_identity')
    source = meta['source']
    _require(isinstance(source, dict) and set(source) == {'revision', 'code_sha256'} and
             isinstance(source['revision'], str) and 0 < len(source['revision']) <= 128 and
             isinstance(source['code_sha256'], dict) and 0 < len(source['code_sha256']) <= 1024 and
             all(isinstance(k, str) and 0 < len(k) <= 1024 and _hash(v) for k,v in source['code_sha256'].items()),
             'weights_source_identity')
    return contract


def _state_hash(state):
    import hashlib
    fingerprint = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        _require(isinstance(name, str) and isinstance(tensor, torch.Tensor) and
                 tensor.dtype == torch.float32 and torch.isfinite(tensor).all().item(), 'weights_tensor')
        value = tensor.detach().cpu().contiguous()
        fingerprint.update(name.encode())
        fingerprint.update(str(tuple(value.shape)).encode())
        fingerprint.update(bytes(value.view(torch.uint8).flatten().tolist()))
    return fingerprint.hexdigest()


def export_weights(model, config, input_contract, catalog_sha256, source, training_step, path, *, checkpoint_sha256):
    """신뢰하는 로컬 모델용. 재개 파일은 별도로 보존해야 한다."""
    meta = {'model_config': {k:config[k] for k in ('model','window_size','hidden','dropout','target_scale_mm')},
            'input_contract': copy.deepcopy(input_contract), 'catalog_sha256': catalog_sha256,
            'source': {k:copy.deepcopy(source[k]) for k in ('revision','code_sha256')}, 'training_step': training_step,
            'checkpoint_sha256': checkpoint_sha256}
    contract = _metadata(meta)
    state = {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    with torch.random.fork_rng(devices=[]):
        reference = make_model(meta['model_config'], 14*len(contract.joint_names), len(contract.joint_names))
    reference.load_state_dict(state, strict=True)
    artifact = {'format_version': FORMAT, 'weights_only': True, 'metadata': meta,
                'metadata_sha256': content_hash(meta), 'state_sha256': _state_hash(state), 'model': state}
    buffer = io.BytesIO()
    torch.save(artifact, buffer)
    _require(buffer.getbuffer().nbytes <= MAX_BYTES, 'weights_serialized_byte_limit')
    path = Path(path)
    _require(not path.exists(), 'weights_output_exists')
    path.parent.mkdir(parents=True, exist_ok=True)
    # 단일 writer용이며 기존 가중치를 덮어쓰지 않는다.
    with path.open('xb') as stream:
        stream.write(buffer.getbuffer())
    return artifact['state_sha256']


def load_weights(path, *, expected_contract, expected_catalog_sha256, expected_source,
                 expected_checkpoint_sha256, expected_model_config, device='cpu'):
    """소비자가 기대한 계약·데이터·코드 identity를 명시해야 한다."""
    artifact = torch.load(io.BytesIO(_read(path, MAX_BYTES)), map_location='cpu', weights_only=True)
    _require(isinstance(artifact, dict) and set(artifact) == {'format_version', 'weights_only', 'metadata',
             'metadata_sha256', 'state_sha256', 'model'} and artifact['format_version'] == FORMAT and
             artifact['weights_only'] is True, 'weights_format')
    meta = artifact['metadata']
    contract = _metadata(meta)
    _require(content_hash(meta) == artifact['metadata_sha256'], 'weights_metadata_hash')
    _require(meta['input_contract'] == expected_contract and meta['catalog_sha256'] == expected_catalog_sha256 and
             meta['source'] == expected_source and meta['checkpoint_sha256'] == expected_checkpoint_sha256 and
             meta['model_config'] == expected_model_config, 'weights_expected_identity')
    _require(isinstance(artifact['model'], dict) and _state_hash(artifact['model']) == artifact['state_sha256'],
             'weights_state_hash')
    with torch.random.fork_rng(devices=[]):
        model = make_model(meta['model_config'], len(contract.joint_names)*14, len(contract.joint_names))
    try:
        model.load_state_dict(artifact['model'], strict=True)
    except RuntimeError as exc:
        raise ValueError('weights_model_shape') from exc
    model.to(device).eval()
    return model, copy.deepcopy(meta)
