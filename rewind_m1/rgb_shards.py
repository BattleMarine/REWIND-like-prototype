"""독립 frame용 RGB+GT chunk 카탈로그와 한 shard metadata 공급. Resize/학습은 별도."""
import copy
import hashlib
import json
from pathlib import Path

from .data import InputContract, content_hash
from .local_rgb import LocalStereoRGB, RGBLimits, _no_link, _path as canonical_path
from .rgb_supervision import LocalRGBSupervision
from .shards import EpochSampler, _hash, _integer, _path, _read, _require, _sequences

SCHEMA = 'm1_rgb_gt_shards_v1'
DEFAULT_LIMITS = {'max_catalog_bytes':4*1024**2, 'max_shards':4096,
    'max_pose_bytes':4*1024**2, 'max_rgb_manifest_bytes':4*1024**2,
    'max_records':256, 'max_payload_bytes':256*1024**2, 'max_batch_tensor_bytes':128*1024**2}


def _limits(value):
    _require(isinstance(value,dict) and set(value) == set(DEFAULT_LIMITS), 'rgb_shard_limits_fields')
    _require(all(_integer(value[k],1,ceiling) for k,ceiling in DEFAULT_LIMITS.items()), 'rgb_shard_limits_range')
    return copy.deepcopy(value)


def _size(doc, maximum):
    total = 0
    for piece in json.JSONEncoder(sort_keys=True,separators=(',',':'),allow_nan=False).iterencode(doc):
        total += len(piece.encode('utf8'))
        _require(total+1 <= maximum, 'rgb_shard_catalog_byte_limit')
    return total


def _base(value):
    base = Path(value).absolute()
    for part in (base,*base.parents): _no_link(part)
    _require(base.is_dir(), 'rgb_shard_base_directory')
    return base.resolve()


def _resolve(base, value):
    canonical_path(value)
    return _path(base,value)


def _contract(doc):
    contract = InputContract.from_dict(doc)
    _require(contract.input_source == 'gt_2d' and len(contract.joint_names) <= 256,
             'rgb_shard_gt_contract')
    _require(contract.image_sizes[0] == contract.image_sizes[1] and
             all(v >= 4 for v in contract.image_sizes[0]) and
             contract.image_sizes[0][0]*contract.image_sizes[0][1] <= RGBLimits().max_pixels_per_view,
             'rgb_shard_native_image_sizes')
    return contract


def validate_catalog(doc):
    _require(isinstance(doc,dict) and set(doc) == {'schema_version','dataset_identity','contract','limits','shards','content_sha256'}
             and doc['schema_version'] == SCHEMA, 'rgb_shard_catalog_fields')
    limits = _limits(doc['limits'])
    rows = doc['shards']
    _require(isinstance(rows,list) and 1 <= len(rows) <= limits['max_shards'], 'rgb_shard_count_limit')
    identity = doc['dataset_identity']
    _require(isinstance(identity,dict) and set(identity) == {'id','dataset_version'} and
             all(isinstance(v,str) and 0 < len(v.encode('utf8')) <= 128 for v in identity.values()),
             'rgb_shard_dataset_identity')
    _contract(doc['contract'])
    ids, paths, sequences, motions, ranges = set(), set(), {}, {}, {}
    for row in rows:
        _require(isinstance(row,dict) and set(row) == {'id','pose','rgb','records','payload_bytes','sequences'},
                 'rgb_shard_descriptor_fields')
        identifier = row['id']
        _require(isinstance(identifier,str) and 0 < len(identifier.encode('utf8')) <= 128 and identifier not in ids,
                 'rgb_shard_id')
        ids.add(identifier)
        _require(_integer(row['records'],1,limits['max_records']) and
                 _integer(row['payload_bytes'],1,limits['max_payload_bytes']), 'rgb_shard_record_payload_limit')
        for kind, maximum in (('pose',limits['max_pose_bytes']),('rgb',limits['max_rgb_manifest_bytes'])):
            item = row[kind]
            _require(isinstance(item,dict) and set(item) == ({'path','size_bytes','sha256'} if kind=='pose' else
                     {'path','size_bytes','sha256','content_sha256','root'}), 'rgb_shard_file_fields')
            path = canonical_path(item['path'])
            _require(path.casefold() not in paths, 'rgb_shard_duplicate_metadata_path')
            paths.add(path.casefold())
            _require(_integer(item['size_bytes'],1,maximum) and _hash(item['sha256']), 'rgb_shard_file_identity')
            if kind == 'rgb':
                canonical_path(item['root'])
                _require(_hash(item['content_sha256']), 'rgb_shard_rgb_content_hash')
        groups = row['sequences']
        _require(isinstance(groups,list) and 1 <= len(groups) <= row['records'], 'rgb_shard_sequence_rows')
        local, count = set(), 0
        for group in groups:
            _require(isinstance(group,dict) and set(group) == {'sequence_id','motion_id','split','records','first_frame','last_frame'},
                     'rgb_shard_sequence_fields')
            seq, motion, split = group['sequence_id'], group['motion_id'], group['split']
            canonical_path(seq)
            _require(isinstance(motion,str) and 0 < len(motion.encode('utf8')) <= 1024 and
                     split in ('train','validation','test') and seq not in local, 'rgb_shard_sequence_identity')
            local.add(seq)
            _require(sequences.get(seq,(motion,split)) == (motion,split), 'rgb_shard_sequence_motion_split_changed')
            _require(motions.get(motion,split) == split, 'rgb_shard_motion_split_overlap')
            sequences[seq], motions[motion] = (motion,split), split
            first, last, number = group['first_frame'], group['last_frame'], group['records']
            _require(_integer(first,0,10**12-1) and _integer(last,first,10**12-1) and
                     _integer(number,1,row['records']) and number <= last-first+1, 'rgb_shard_chunk_bounds')
            ranges.setdefault(seq,[]).append((first,last))
            count += number
        _require(count == row['records'], 'rgb_shard_record_count')
    for chunks in ranges.values():
        ordered = sorted(chunks)
        _require(all(a[1] < b[0] for a,b in zip(ordered,ordered[1:])), 'rgb_shard_chunk_overlap')
    _size(doc,limits['max_catalog_bytes'])
    _require(_hash(doc['content_sha256']) and doc['content_sha256'] ==
             content_hash({k:v for k,v in doc.items() if k!='content_sha256'}), 'rgb_shard_catalog_hash')
    return copy.deepcopy(doc)


def read_catalog(path):
    raw = _read(path,DEFAULT_LIMITS['max_catalog_bytes'])
    doc = validate_catalog(json.loads(raw))
    _require(len(raw) <= doc['limits']['max_catalog_bytes'], 'rgb_shard_catalog_byte_limit')
    return doc


def _payload(raw, item, reason):
    _require(len(raw) == item['size_bytes'] and hashlib.sha256(raw).hexdigest() == item['sha256'],reason)
    return json.loads(raw)


def _payload_bytes(rows, maximum):
    total = 0
    for record in rows:
        _require(isinstance(record,dict) and isinstance(record.get('views'),dict) and
                 set(record['views']) == {'left','right'}, 'rgb_shard_rgb_views')
        for view in record['views'].values():
            _require(isinstance(view,dict) and _integer(view.get('size_bytes'),1,RGBLimits().max_file_bytes),
                     'rgb_shard_png_file_limit')
            total += view['size_bytes']
            _require(total <= maximum, 'rgb_shard_payload_byte_limit')
    return total


def _documents(base, row, limits, contract, identity):
    pose = _payload(_read(_resolve(base,row['pose']['path']),limits['max_pose_bytes']),row['pose'],'rgb_shard_pose_hash')
    rgb = _payload(_read(_resolve(base,row['rgb']['path']),limits['max_rgb_manifest_bytes']),row['rgb'],'rgb_shard_rgb_file_hash')
    _require(isinstance(pose,dict) and pose.get('schema_version') == 'm1_features_v1' and
             pose.get('contract') == contract, 'rgb_shard_pose_contract')
    _require(isinstance(pose.get('manifest'),dict) and all(pose['manifest'].get(k) == v for k,v in identity.items()),
             'rgb_shard_pose_dataset_identity')
    records = pose.get('records')
    _require(isinstance(records,list) and len(records) == row['records'], 'rgb_shard_pose_record_count')
    _require(_sequences(records) == row['sequences'], 'rgb_shard_sequence_declaration_mismatch')
    _require(isinstance(rgb,dict) and rgb.get('dataset_version') == identity['dataset_version'] and
             rgb.get('data_kind') == contract['data_kind'] and rgb.get('content_sha256') == row['rgb']['content_sha256'],
             'rgb_shard_rgb_dataset_identity')
    rgb_rows = rgb.get('records')
    _require(isinstance(rgb_rows,list) and len(rgb_rows) == row['records'], 'rgb_shard_rgb_record_count')
    total = _payload_bytes(rgb_rows,limits['max_payload_bytes'])
    _require(total == row['payload_bytes'], 'rgb_shard_payload_declaration_mismatch')
    return pose


class RGBShardDataset:
    """한 shard/선택 split의 기존 supervision만 cache. 반환값의 외부 참조는 호출자 책임."""
    def __init__(self, catalog, base):
        self.catalog = validate_catalog(catalog)
        self.base = _base(base)
        self._cached = self._key = None

    def clear(self):
        self._cached = self._key = None

    def load(self, index, split):
        _require(_integer(index,0,len(self.catalog['shards'])-1) and split in ('train','validation','test'),
                 'rgb_shard_index_split')
        row = self.catalog['shards'][index]
        _require(any(g['split']==split for g in row['sequences']), 'rgb_shard_split_missing')
        if self._key != (index,split):
            self.clear()
            limits = self.catalog['limits']
            pose = _documents(self.base,row,limits,self.catalog['contract'],self.catalog['dataset_identity'])
            loader = LocalStereoRGB(_resolve(self.base,row['rgb']['path']),_resolve(self.base,row['rgb']['root']),
                limits=RGBLimits(max_manifest_bytes=limits['max_rgb_manifest_bytes'],max_records=limits['max_records'],
                                 max_batch_tensor_bytes=limits['max_batch_tensor_bytes']))
            supervision = LocalRGBSupervision(loader,pose,expected_manifest_sha256=row['rgb']['content_sha256'],
                joint_names=self.catalog['contract']['joint_names'],split=split)
            self._cached, self._key = supervision, (index,split)
        return self._cached

    def verify_all(self, full_png=False, check_budget=None):
        _require(type(full_png) is bool, 'rgb_shard_full_png_flag')
        check = check_budget or (lambda:None)
        counts, verified = {}, 0
        try:
            for index,row in enumerate(self.catalog['shards']):
                for split in ('train','validation','test'):
                    if not any(g['split']==split for g in row['sequences']): continue
                    check()
                    supervision = self.load(index,split)
                    counts[split] = counts.get(split,0)+len(supervision)
                    if full_png:
                        for i in range(len(supervision)):
                            check()
                            batch = supervision.load_batch([i])
                            verified += 1
                            del batch
                    del supervision
                    self.clear()
        finally:
            self.clear()
        return {'schema_version':'rgb_gt_shard_readiness_v1','catalog_sha256':self.catalog['content_sha256'],
                'metadata_verified':True,'metadata_records':counts,'full_png_verified':full_png,'png_samples_verified':verified,
                'temporal_ready':False,'train_ready':False,'full_training_ready':False,
                'scope':'local_gt_metadata_and_requested_png_only',
                'blockers':['rgb_shard_trainer_not_connected','dataset_sample_release_not_implemented','colab_gpu_pilot_not_measured']}


def build_catalog(specs, base, limits=None):
    """기존 파일을 변경하지 않고 RGB/pose 선언을 고정한다. PNG decode는 명시 full 검사에서만."""
    limits = _limits(limits or DEFAULT_LIMITS)
    _require(isinstance(specs,list) and 1 <= len(specs) <= limits['max_shards'], 'rgb_shard_count_limit')
    base = _base(base)
    contract, identity, rows, catalog_bytes = None, None, [], 0
    for spec in specs:
        _require(isinstance(spec,dict) and set(spec)=={'id','pose','rgb_manifest','rgb_root'}, 'rgb_shard_build_spec')
        raw = _read(_resolve(base,spec['pose']),limits['max_pose_bytes'])
        pose = json.loads(raw)
        _require(isinstance(pose,dict) and isinstance(pose.get('records'),list) and
                 1 <= len(pose['records']) <= limits['max_records'], 'rgb_shard_pose_record_limit')
        if contract is None:
            contract = pose.get('contract'); _contract(contract)
            identity = {k:pose.get('manifest',{}).get(k) for k in ('id','dataset_version')}
            catalog_bytes = _size({'schema_version':SCHEMA,'dataset_identity':identity,'contract':contract,
                'limits':limits,'shards':[],'content_sha256':'0'*64},limits['max_catalog_bytes'])
        _require(pose.get('contract') == contract, 'rgb_shard_pose_contract')
        rgb_raw = _read(_resolve(base,spec['rgb_manifest']),limits['max_rgb_manifest_bytes'])
        rgb = json.loads(rgb_raw)
        _require(isinstance(rgb,dict) and isinstance(rgb.get('records'),list) and
                 len(rgb['records']) == len(pose['records']), 'rgb_shard_rgb_record_count')
        payload = _payload_bytes(rgb['records'],limits['max_payload_bytes'])
        row = {'id':spec['id'],'pose':{'path':spec['pose'],'size_bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()},
            'rgb':{'path':spec['rgb_manifest'],'size_bytes':len(rgb_raw),'sha256':hashlib.sha256(rgb_raw).hexdigest(),
                   'content_sha256':rgb.get('content_sha256'),'root':spec['rgb_root']},
            'records':len(pose['records']),'payload_bytes':payload,'sequences':_sequences(pose['records'])}
        # 각 descriptor만 한 번 계수하고 append 전에 중단한다. 전체 누적 재직렬화는 하지 않는다.
        catalog_bytes += _size(row,limits['max_catalog_bytes']) + (1 if rows else 0)
        _require(catalog_bytes+1 <= limits['max_catalog_bytes'], 'rgb_shard_catalog_byte_limit')
        rows.append(row)
    body = {'schema_version':SCHEMA,'dataset_identity':identity,'contract':contract,'limits':limits,'shards':rows}
    _size(body,limits['max_catalog_bytes'])
    catalog = validate_catalog({**body,'content_sha256':content_hash(body)})
    RGBShardDataset(catalog,base).verify_all()
    return catalog


class RGBEpochSampler(EpochSampler):
    """2D독립frame chunk용 카탈로그를 검증하고 기존 순회/cursor 연산만 재사용한다."""
    POLICY = 'm1_rgb_shard_epoch_python_random_v1'

    def __init__(self, catalog, split, seed, shuffle=True):
        self.catalog = validate_catalog(catalog)
        _require(split in ('train','validation','test') and _integer(seed,0,2**32-1) and type(shuffle) is bool,
                 'rgb_shard_sampler_arguments')
        self.split, self.seed, self.shuffle = split, seed, shuffle
        self.sizes = {i:sum(g['records'] for g in row['sequences'] if g['split']==split)
                      for i,row in enumerate(self.catalog['shards'])}
        self.sizes = {i:n for i,n in self.sizes.items() if n}
        _require(bool(self.sizes), 'rgb_shard_sampler_empty_split')
        self.epoch = self.shard_position = self.offset = 0
