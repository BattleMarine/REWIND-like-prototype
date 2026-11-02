"""훈련 승인 권한을 발급하지 않는 오프라인 RGB2D 추론 artifact."""
import copy
import io

import torch

from rewind_archive.contracts import digest
from rewind_m1.frontend import StereoHeatmapNet,weights_hash
from rewind_m1.rgb_frontend_artifacts import preprocessing_version,resize_images,scale_xy
from .contracts import code_identity,read,seal
from rewind_bundle_data.integrity import local,require,sha,MIB
from rewind_m1.local_rgb import _regular_read

FORMAT = 'bounded_released_rgb2d_weights_v1'
CHECKPOINT = 'bounded_released_rgb2d_checkpoint_v1'
MAX_BYTES = 64*MIB


def checkpoint(path):
    raw = _regular_read(local(path),MAX_BYTES)
    value = torch.load(io.BytesIO(raw),map_location='cpu',weights_only=True)
    require(value.get('format_version') == CHECKPOINT and value.get('training_scope') in
            ('reviewed_real_rgb2d_t1_only','development_fixture_rgb2d_t1_only') and value.get('full_training_ready') is False,
            'release_checkpoint_format')
    return value,sha(raw)


def export_checkpoint(state,checkpoint_sha256,output):
    require(state.get('format_version') == CHECKPOINT and state.get('full_training_ready') is False, 'release_export_checkpoint')
    catalog = state['catalog'];binding = catalog['binding']
    require(catalog['content_sha256'] == seal(catalog)['content_sha256'] and binding['content_sha256'] == seal(binding)['content_sha256'] and
            binding['mode'] in ('reviewed_real','development_fixture') and
            state['training_scope'] == binding['mode']+'_rgb2d_t1_only' and binding['config_sha256'] == digest(state['config']) and
            state['source']['code_sha256'] == binding['code_sha256'] == code_identity(), 'release_export_identity')
    with torch.random.fork_rng(devices=[]), torch.device('cpu'):model = StereoHeatmapNet(62).float()
    model.load_state_dict(state['model'],strict=True)
    require(all(t.dtype == torch.float32 and torch.isfinite(t).all().item() for t in model.state_dict().values()), 'release_weights_nonfinite')
    meta = {'release_binding':copy.deepcopy(binding),'checkpoint_sha256':checkpoint_sha256,'training_step':state['step'],
            'training_image_size':state['config']['training_image_size'],
            'preprocessing_version':preprocessing_version(state['config']['training_image_size']),
            'frontend_version':state['config']['frontend_version'],'source':copy.deepcopy(state['source']),
            'permission':'inference_only_not_training_authorization','input_kind':'native_rgb','output_kind':'predicted_2d_native_pixels'}
    artifact = {'format_version':FORMAT,'metadata':meta,'metadata_sha256':digest(meta),
                'model_sha256':weights_hash(model),'model':{k:t.detach().cpu().clone() for k,t in model.state_dict().items()}}
    buffer = io.BytesIO();torch.save(artifact,buffer)
    require(buffer.getbuffer().nbytes <= MAX_BYTES, 'release_weights_bytes')
    from pathlib import Path
    path = Path(output)
    with path.open('xb') as stream:stream.write(buffer.getbuffer())
    return artifact['model_sha256']


class ReleasedInference:
    def __init__(self,path,*,expected_file_sha256):
        raw = _regular_read(local(path),MAX_BYTES)
        require(sha(raw) == expected_file_sha256, 'release_inference_file_pin')
        artifact = torch.load(io.BytesIO(raw),map_location='cpu',weights_only=True)
        require(artifact.get('format_version') == FORMAT and artifact['metadata_sha256'] == digest(artifact['metadata']) and
                artifact['metadata'].get('permission') == 'inference_only_not_training_authorization', 'release_inference_metadata')
        self.metadata = copy.deepcopy(artifact['metadata'])
        require(self.metadata['preprocessing_version'] == preprocessing_version(self.metadata['training_image_size']), 'release_inference_preprocessing')
        with torch.random.fork_rng(devices=[]), torch.device('cpu'):self.model = StereoHeatmapNet(62).float()
        self.model.load_state_dict(artifact['model'],strict=True)
        require(all(t.dtype == torch.float32 and torch.isfinite(t).all().item() for t in self.model.state_dict().values()) and
                weights_hash(self.model) == artifact['model_sha256'], 'release_inference_weights')
        self.model.eval()

    def predict(self,images):
        require(images.shape == (1,2,3,1024,1024) and images.dtype == torch.float32 and images.device.type == 'cpu', 'release_inference_native_sample')
        with torch.no_grad():
            _,xy,confidence = self.model(resize_images(images,self.metadata['training_image_size']))
        return {'coordinates':scale_xy(xy,self.metadata['training_image_size'],[1024,1024]),'confidence':confidence,
                'input_source':'predicted_2d','training_authorization':False,'metadata':copy.deepcopy(self.metadata)}
