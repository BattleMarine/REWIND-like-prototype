"""V2 atomic shards preserve original float64 camera transforms."""
from collections import OrderedDict
from dataclasses import dataclass
import io
import json
from pathlib import Path
import tempfile
import os

import numpy as np
import torch

from rewind_archive.contracts import canonical,digest
from rewind_m1.local_rgb import _regular_read
from rewind_prepare.coordinates import camera_transform
from rewind_prepare.extracted import _payload_path,_raw_pose
from rewind_prepare.temporal_cache import _atomic,_hash_file,_locked,_check_part
from rewind_prepare.temporal_selection import seal,check_pin
from .full_cache import _frame
from . import v2_contract as contract

CHUNK=128


@dataclass(frozen=True)
class ReadOnlyFirstShard:
    """Exact producer and consumer pins for the observed CPU-to-CUDA build pair."""

    producer_binding_sha256: str
    consumer_source_sha256: str
    receipt_sha256: str

    def verify(self, producer, consumer):
        source_changes = {"rewind_v1/v2_cache.py", "rewind_v1/v2_retention.py", "rewind_v1/v2_training.py"}
        old_source, new_source = producer["source"], consumer["source"]
        changes = {name for name in old_source.keys() | new_source.keys()
                   if old_source.get(name) != new_source.get(name)}
        if (digest(producer) != self.producer_binding_sha256
                or digest(new_source) != self.consumer_source_sha256
                or old_source.keys() != new_source.keys() or not changes <= source_changes
                or (producer["torch"], consumer["torch"]) != ("2.14.1+cpu", "2.14.1+cu130")
                or {k: v for k, v in producer.items() if k not in {"source", "torch"}}
                != {k: v for k, v in consumer.items() if k not in {"source", "torch"}}):
            raise ValueError("v2_read_only_binding_pin")


def binding(doc,spec,cfg):
    return {'schema':'rewind_v2_cache_binding_v1','manifest_sha256':doc['content_sha256'],'spec_sha256':digest(spec),
        'source':contract.source(),'camera':cfg['camera'],'preprocessing':'float32_stereo128_bilinear_align_corners_v1',
        'torch':str(torch.__version__),'numpy':str(np.__version__),'pillow':__import__('PIL').__version__,'chunk':CHUNK}


def receipt(root,number,bind,rows,ledger):
    folder=Path(root)/f'shard-{number:05d}'
    if folder.is_symlink():raise ValueError('v2_shard_symlink')
    doc=json.loads(_regular_read(folder/'receipt.json',1024**2));check_pin(doc,'rewind_v2_shard_v1',doc.get('content_sha256'))
    if (doc['binding']!=bind or doc['records_sha256']!=digest(rows) or doc['number']!=number
            or {p.name for p in folder.iterdir()}!={'images.npy','targets.pt','receipt.json'}
            or set(doc['files'])!={'images.npy','targets.pt'}):raise ValueError('v2_shard_binding')
    for name,pin in doc['files'].items():
        ledger.check();_check_part(folder/name,{'name':name,**pin},ledger.check)
        ledger.local_io('read',pin['bytes'])
    return doc


def catalog(root,bind,shards,total):
    doc=seal({'schema':'rewind_v2_local_catalog_v1','binding':bind,'shards':list(shards),'completed_frames':sum(p['frames'] for p in shards),
        'total_frames':total,'status':'local_complete' if sum(p['frames'] for p in shards)==total else 'local_partial'})
    _atomic(Path(root)/'catalog.json',canonical(doc));return doc


def prepare(doc,spec,cfg,rgb,pose,output,ledger,*,frame_builder=None,max_shards=None):
    root=Path(output);root.mkdir(parents=True,exist_ok=True);bind=binding(doc,spec,cfg);records=doc['records'];pins=[]
    if root.is_symlink():raise ValueError('v2_cache_symlink')
    with _locked(root):
        old=root/'catalog.json';prior=None
        allowed={'.lock','catalog.json'}|{f'shard-{n:05d}' for n in range((len(records)+CHUNK-1)//CHUNK)}
        if any(p.name not in allowed and not p.name.startswith(('.pending-','.quarantine-','.pack-','.restore-')) for p in root.iterdir()):raise ValueError('v2_unexpected_cache_members')
        if old.exists():
            prior=json.loads(old.read_bytes());check_pin(prior,'rewind_v2_local_catalog_v1',prior.get('content_sha256'))
            if prior['binding']!=bind:raise ValueError('v2_cache_catalog_binding')
            expected=[f'shard-{i:05d}' for i in range(len(prior['shards']))]
            if any(not (root/n).is_dir() for n in expected):raise ValueError('v2_missing_durable_shard_restore_required')
        else:
            if any(p.name!='.lock' for p in root.iterdir()):raise ValueError('v2_cache_catalog_missing_explicit_quarantine_required')
            catalog(root,bind,[],len(records));prior=json.loads(old.read_bytes())
        # Reconcile a fully published shard if interruption preceded its catalog commit.
        for number in range((len(records)+CHUNK-1)//CHUNK):
            ledger.check();rows=records[number*CHUNK:(number+1)*CHUNK];folder=root/f'shard-{number:05d}'
            if folder.exists():
                r=receipt(root,number,bind,rows,ledger)
                if prior is not None and number<len(prior['shards']) and prior['shards'][number]!=r:raise ValueError('v2_catalog_receipt_mismatch')
            else:
                if max_shards is not None and number>=max_shards:break
                estimate=len(rows)*410000+1024**2
                used=sum(p.stat().st_size for p in root.rglob('*') if p.is_file())
                if used+estimate>cfg['resources']['cache_bytes']:raise ValueError('v2_cache_disk_cap')
                ledger.reserve('write',estimate)
                ledger.reserve('read',sum(p['size_bytes'] for row in rows for p in row['files'].values()))
                with tempfile.TemporaryDirectory(dir=root,prefix='.pending-') as tmp:
                    staged=Path(tmp)/folder.name;staged.mkdir()
                    images=np.lib.format.open_memmap(staged/'images.npy',mode='w+',dtype=np.float32,shape=(len(rows),2,3,128,128));metadata=[]
                    try:
                        for i,row in enumerate(rows):
                            ledger.check();raw={}
                            for role,pin in row['files'].items():
                                raw[role]=_regular_read(_payload_path(Path(pose if role=='pose' else rgb),pin['path']),8*1024**2,pin['size_bytes'])
                                ledger.observed('read',len(raw[role]),persist=False)
                                if contract.sha(raw[role])!=pin['sha256']:raise ValueError('v2_source_sha')
                            if frame_builder is None:
                                image,target=_frame(raw,row['frame_index'],spec);raw_doc=_raw_pose(raw['pose'],row['frame_index']);coord=spec['coordinates']
                                cameras=[camera_transform(raw_doc['camera_'+s],coord['mm_per_raw_unit'],coord['raw_consistency_tolerance'])[0] for s in ('left','right')]
                                target['rotation64']=torch.tensor([c['rotation_matrix'] for c in cameras],dtype=torch.float64)
                                target['translation64']=torch.tensor([c['translation'] for c in cameras],dtype=torch.float64)
                            else:image,target=frame_builder(raw,row,spec)
                            if target['rotation64'].dtype!=torch.float64 or target['translation64'].dtype!=torch.float64:raise ValueError('v2_float64_camera_required')
                            images[i]=image.numpy();metadata.append(target)
                        images.flush()
                    finally:del images
                    with (staged/'images.npy').open('r+b') as f:os.fsync(f.fileno())
                    tensors={n:torch.stack([r[n] for r in metadata]) for n in metadata[0]}
                    if not all(torch.isfinite(v).all() for v in tensors.values()):raise ValueError('v2_cache_nonfinite')
                    with (staged/'targets.pt').open('xb') as f:torch.save(tensors,f);f.flush();os.fsync(f.fileno())
                    files={n:{'bytes':(staged/n).stat().st_size,'sha256':_hash_file(staged/n,check_time=ledger.check)} for n in ('images.npy','targets.pt')}
                    r=seal({'schema':'rewind_v2_shard_v1','number':number,'binding':bind,'records_sha256':digest(rows),'frames':len(rows),'files':files})
                    _atomic(staged/'receipt.json',canonical(r));ledger.check();staged.rename(folder)
                    ledger.observed('write',sum(p['bytes'] for p in files.values()))
            pins.append(r)
            if prior is None or len(pins)>len(prior['shards']):result=catalog(root,bind,pins,len(records));prior=result
            else:result=prior
        if not pins:result=catalog(root,bind,[],len(records))
        return result


def quarantine(root,name,expected_receipt_file_sha):
    """Explicit local recovery only: rename inside verified cache; never delete."""
    import uuid
    root=Path(root).resolve();path=root/name
    if Path(name).name!=name or path.is_symlink() or not path.is_dir() or path.resolve().parent!=root or not name.startswith(('shard-','.pending-')):raise ValueError('v2_quarantine_path')
    if name.startswith('shard-') and _hash_file(path/'receipt.json')!=expected_receipt_file_sha:raise ValueError('v2_quarantine_receipt_pin')
    with _locked(root):
        target=root/('.quarantine-'+uuid.uuid4().hex);path.rename(target)
    return str(target)


class Cache:
    def __init__(self,root,doc,spec,cfg,ledger):
        self._open(root,doc,spec,cfg,ledger)

    @classmethod
    def open_read_only(cls,root,doc,spec,cfg,ledger,*,policy,local_catalog_sha256):
        instance=cls.__new__(cls)
        instance._open(root,doc,spec,cfg,ledger,policy=policy,local_catalog_sha256=local_catalog_sha256)
        return instance

    def _open(self,root,doc,spec,cfg,ledger,*,policy=None,local_catalog_sha256=None):
        self.root,self.doc,self.spec,self.ledger=Path(root),doc,spec,ledger;self.bind=binding(doc,spec,cfg);self.loaded=OrderedDict()
        cat=json.loads(_regular_read(self.root/'catalog.json',32*1024**2))
        if policy is None:
            check_pin(cat,'rewind_v2_local_catalog_v1',cat.get('content_sha256'))
            if cat['binding']!=self.bind or cat['status']!='local_complete' or cat['completed_frames']!=len(doc['records']) or len(cat['shards'])!=(len(doc['records'])+CHUNK-1)//CHUNK:raise ValueError('v2_complete_cache_required')
        else:
            check_pin(cat,'rewind_v2_local_catalog_v1',local_catalog_sha256)
            policy.verify(cat['binding'],self.bind)
            if (cat['status']!='local_partial' or cat['completed_frames']!=CHUNK or len(cat['shards'])!=1
                    or cat['shards'][0]['content_sha256']!=policy.receipt_sha256
                    or cat['total_frames']!=len(doc['records']) or len(doc['records'])<=CHUNK):
                raise ValueError('v2_read_only_first_shard')
            self.bind=cat['binding']
        for i,pin in enumerate(cat['shards']):
            if receipt(root,i,self.bind,doc['records'][i*CHUNK:(i+1)*CHUNK],ledger)!=pin:raise ValueError('v2_cache_pin')
        self.identity=cat['content_sha256'];self.index={(r['sequence_id'],r['frame_index']):i for i,r in enumerate(doc['records'][:cat['completed_frames']])}

    def frame(self,key,split):
        i=self.index[tuple(key)];r=self.doc['records'][i]
        if r['split']!=split:raise ValueError('v2_frame_split')
        number,slot=divmod(i,CHUNK)
        if number not in self.loaded:
            folder=self.root/f'shard-{number:05d}';images=np.load(folder/'images.npy',mmap_mode='r',allow_pickle=False)
            targets=torch.load(io.BytesIO(_regular_read(folder/'targets.pt',4*1024**2)),weights_only=True,map_location='cpu')
            self.ledger.local_io('read',(folder/'targets.pt').stat().st_size)
            count=len(self.doc['records'][number*CHUNK:(number+1)*CHUNK]);j=len(self.spec['joint_names'])
            shapes={'xy':(count,2,j,2),'xy_valid':(count,2,j),'target_mm':(count,j,3),'target_valid':(count,j),'camera':(count,2,9),'rotation64':(count,2,3,3),'translation64':(count,2,3)}
            if images.shape!=(count,2,3,128,128) or images.dtype!=np.float32 or set(targets)!=set(shapes):raise ValueError('v2_cache_shape')
            for n,shape in shapes.items():
                dtype=torch.float64 if n.endswith('64') else torch.bool if n.endswith('valid') else torch.float32
                if targets[n].shape!=shape or targets[n].dtype!=dtype or not torch.isfinite(targets[n]).all():raise ValueError('v2_target_shape_dtype')
            self.loaded[number]=(images,targets)
            if len(self.loaded)>2:self.loaded.popitem(last=False)[1][0]._mmap.close()
        self.loaded.move_to_end(number);image,targets=self.loaded[number]
        x=torch.from_numpy(np.array(image[slot],copy=True))
        self.ledger.local_io('read',x.numel()*x.element_size())
        if not torch.isfinite(x).all() or x.min()<0 or x.max()>1:raise ValueError('v2_image_range')
        return x,{n:t[slot].clone() for n,t in targets.items()}

    def close(self):
        for x,_ in self.loaded.values():x._mmap.close()
        self.loaded.clear()
