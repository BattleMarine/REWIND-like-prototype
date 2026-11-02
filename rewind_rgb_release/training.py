"""검토된 RGB2D release 또는 명시 개발 fixture의 공통 학습 코어 adapter."""
import copy
from types import SimpleNamespace

import torch

from rewind_archive.contracts import digest
from rewind_m1.rgb_frontend_artifacts import resize_images, scale_xy
from rewind_m1.rgb_training import RGBTrainer
from rewind_m1.shards import EpochSampler
from .contracts import admit,check_authority,code_identity,seal

FORMAT = 'bounded_released_rgb2d_checkpoint_v1'


class BundleSampler(EpochSampler):
    POLICY = 'reviewed_rgb2d_bundle_epoch_python_random_v1'

    def _policy(self):
        return self.POLICY

    def __init__(self,catalog,seed):
        self.catalog,self.seed,self.split,self.shuffle = catalog,seed,'train',True
        self.sizes = {0:sum(r['records'] for r in catalog['shards'][0]['sequences'] if r['split'] == 'train')}
        self.epoch = self.shard_position = self.offset = 0


class BundleDataset:
    def __init__(self,readers):
        self.readers = readers
        self.active = None

    def clear(self):
        for r in self.readers.values():
            r._cached_path,r._cached = None,None
        self.active = None

    def verify_all(self,check_budget):
        check_budget()
        return {'metadata_records':{s:len(r) for s,r in self.readers.items()}}

    def load(self,shard,split):
        if shard != 0 or split not in self.readers:
            raise ValueError('released_dataset_split_or_shard')
        if self.active != split:
            self.clear();self.active = split
        return self.readers[split]


class ReleasedRGBTrainer(RGBTrainer):
    CHECKPOINT_FORMAT = FORMAT

    @property
    def TRAINING_SCOPE(self):
        return self.binding['mode']+'_rgb2d_t1_only'

    def _open_data(self,request,base):
        if self.config['batch_size'] > 2:
            raise ValueError('release_batch_size_limit')
        if 'request' in request:
            if set(request) != {'request','content_sha256'} or seal(request)['content_sha256'] != request['content_sha256']:
                raise ValueError('release_request_envelope')
            request = request['request']
        self._request = copy.deepcopy(request)
        readers,self.binding = admit(request,self.config,self.check_budget)
        contract = copy.deepcopy(readers['train']._contract)
        groups = {}
        for seq,frame,split in self.binding['data']['key_coverage']:
            if split in ('train','validation'):
                row = groups.setdefault(seq,{'sequence_id':seq,'split':split,'records':0})
                row['records'] += 1
        catalog = seal({'schema_version':'released_rgb2d_bundle_catalog_v1','binding':self.binding,
            'contract':contract,'limits':{'max_records':32,'max_batch_tensor_bytes':64*1024**2},
            'shards':[{'sequences':list(groups.values())}]})
        return catalog,SimpleNamespace(**contract),BundleDataset(readers)

    def __init__(self,config,request,base=None,device='cpu',check_budget=None):
        super().__init__(config,request,base,device,check_budget)
        self.source['code_sha256'] = code_identity()

    def _new_sampler(self):
        return BundleSampler(self.catalog,self.config['seed'])

    def _make_model(self,joints):
        from rewind_m1.frontend import StereoHeatmapNet
        with torch.device('cpu'):
            return StereoHeatmapNet(joints).to(device=self.device,dtype=torch.float32)

    def train_step(self):
        self.check_budget();check_authority(self._request,self.binding);self.check_budget()
        return super().train_step()

    def resume(self,state):
        self.check_budget();check_authority(self._request,self.binding)
        return super().resume(state)

    def _batch(self,shard,split,indices):
        reader = self.dataset.load(shard,split)
        images,targets,masks,native_targets,keys = [],[],[],[],[]
        for index in indices:
            self.check_budget()
            native = reader.load_batch([index])
            images.append(resize_images(native['images'],self.config['training_image_size']))
            targets.append(scale_xy(native['gt_xy'],[1024,1024],self.config['training_image_size']))
            masks.append(native['annotation_valid']);native_targets.append(native['gt_xy'])
            row = native['metadata'][0];keys.append([row['sequence_id'],row['frame_index']])
            del native
        return {'images':torch.cat(images),'gt_xy':torch.cat(targets),'valid':torch.cat(masks),
                'native_gt_xy':torch.cat(native_targets),'keys':keys}

    def evaluate(self,split='validation',final_test=False):
        if split not in ('train','validation') or final_test:
            raise ValueError('release_train_validation_only_no_test')
        self.check_budget();check_authority(self._request,self.binding)
        result = super().evaluate(split,False)
        return {**result,'evaluation_kind':self.TRAINING_SCOPE,'release_binding_sha256':self.binding['content_sha256']}
