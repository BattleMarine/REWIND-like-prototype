"""완결 sequence 단위 M1 특징 shard. 한 shard만 tensor화하며 원격 준비는 하지 않는다."""
import copy
import hashlib
import json
import random
import re
from pathlib import Path, PurePosixPath

from .data import DataContractError, InputContract, content_hash, load_document

SCHEMA = 'm1_feature_shards_v1'
T1_SCHEMA = 'm1_t1_feature_shards_v1'
T1_POLICY = 'm1_t1_feature_epoch_python_random_v1'
DEFAULT_LIMITS = {'max_catalog_bytes': 4*1024**2, 'max_shards': 4096,
                  'max_shard_bytes': 32*1024**2, 'max_records': 4096,
                  'max_tensor_elements': 8*1024**2}


def _require(condition, reason):
    if not condition:
        raise DataContractError(reason)


def _integer(value, low, high):
    return type(value) is int and low <= value <= high


def _hash(value):
    return isinstance(value, str) and re.fullmatch('[0-9a-f]{64}', value) is not None


def _read(path, limit):
    path = Path(path)
    _require(not path.is_symlink() and path.is_file(), 'shard_regular_file_required')
    _require(path.stat().st_size <= limit, 'shard_byte_limit')
    with path.open('rb') as stream:
        raw = stream.read(limit+1)
    _require(len(raw) <= limit, 'shard_byte_limit')
    return raw


def _limits(value):
    _require(isinstance(value, dict) and set(value) == set(DEFAULT_LIMITS), 'shard_limits_fields')
    for key, ceiling in DEFAULT_LIMITS.items():
        _require(_integer(value[key], 1, ceiling), 'shard_limit_range_'+key)
    return copy.deepcopy(value)


def _json_size(value, maximum):
    total=0
    for piece in json.JSONEncoder(sort_keys=True,separators=(',',':'),allow_nan=False).iterencode(value):
        total+=len(piece.encode('utf8'))
        _require(total+1 <= maximum,'shard_catalog_byte_limit')
    return total


def validate_model_catalog(config, catalog):
    if catalog['schema_version'] == T1_SCHEMA:
        _require(config['model']=='single_frame' and config['window_size']==1,
                 't1_requires_single_frame_window_one')


def _relative(value):
    _require(isinstance(value, str) and 0 < len(value.encode('utf8')) <= 1024,
             'shard_relative_path')
    path = PurePosixPath(value)
    _require(not path.is_absolute() and ':' not in value and '\\' not in value
             and path.as_posix() == value and all(p not in ('.', '..') for p in path.parts),
             'shard_relative_path')
    return value


def _path(base, value):
    relative = _relative(value)
    base = Path(base).resolve()
    path = base / relative
    _require(path.resolve().is_relative_to(base), 'shard_path_escape')
    # 부모 symlink도 원본 경로 별칭으로 사용하지 않는다.
    _require(not any(p.is_symlink() for p in [path, *path.parents] if p != base and p.is_relative_to(base)),
             'shard_symlink')
    return path


def _sequences(records):
    result, seen, previous = [], set(), None
    for record in records:
        seq = record.get('sequence_id')
        _require(isinstance(seq, str) and 0 < len(seq.encode('utf8')) <= 1024, 'shard_sequence_id')
        motion, split = record.get('motion_id'), record.get('split')
        _require(isinstance(motion, str) and 0 < len(motion.encode('utf8')) <= 1024
                 and split in ('train', 'validation', 'test'), 'shard_motion_split')
        frame = record.get('frame_index')
        _require(_integer(frame, 0, 10**12), 'shard_frame_index')
        if seq != previous:
            _require(seq not in seen, 'shard_sequence_not_contiguous')
            seen.add(seq)
            result.append({'sequence_id': seq, 'motion_id': motion, 'split': split,
                           'records': 0, 'first_frame': frame, 'last_frame': frame})
        row = result[-1]
        _require(row['motion_id'] == motion and row['split'] == split, 'shard_sequence_identity_changed')
        _require(row['records'] == 0 or frame > row['last_frame'], 'shard_frame_order')
        row['records'] += 1
        row['last_frame'] = frame
        previous = seq
    return result


def validate_catalog(doc):
    _require(isinstance(doc, dict) and set(doc) == {'schema_version', 'dataset_identity', 'contract', 'window_size',
             'limits', 'shards', 'content_sha256'}, 'shard_catalog_fields')
    _require(doc['schema_version'] in (SCHEMA,T1_SCHEMA) and doc['content_sha256'] ==
             content_hash({k:v for k,v in doc.items() if k != 'content_sha256'}), 'shard_catalog_hash')
    limits = _limits(doc['limits'])
    identity = doc['dataset_identity']
    _require(isinstance(identity, dict) and set(identity) == {'id', 'dataset_version'} and
             all(isinstance(v, str) and 0 < len(v.encode('utf8')) <= 1024 for v in identity.values()),
             'shard_dataset_identity')
    _require(_integer(doc['window_size'], 1, 256), 'shard_window_limit')
    t1=doc['schema_version']==T1_SCHEMA
    _require(not t1 or doc['window_size']==1,'t1_requires_window_one')
    contract = InputContract.from_dict(doc['contract'])
    shards = doc['shards']
    _require(isinstance(shards, list) and 1 <= len(shards) <= limits['max_shards'], 'shard_count_limit')
    sequences, motions, paths = set(), {}, set()
    owners, ranges = {}, {}
    for shard in shards:
        _require(isinstance(shard, dict) and set(shard) == {'path', 'sha256', 'size_bytes', 'records', 'sequences'},
                 'shard_descriptor_fields')
        path = _relative(shard['path'])
        _require(path.casefold() not in paths, 'shard_duplicate_path')
        paths.add(path.casefold())
        _require(_hash(shard['sha256']), 'shard_descriptor_hash')
        _require(_integer(shard['size_bytes'], 1, limits['max_shard_bytes']), 'shard_byte_limit')
        _require(_integer(shard['records'], 1, limits['max_records']), 'shard_record_limit')
        # float features14J + target3J + target maskJ + time mask의 요소 수를 먼저 제한한다.
        elements = shard['records'] * doc['window_size'] * (18*len(contract.joint_names)+1)
        _require(elements <= limits['max_tensor_elements'], 'shard_tensor_element_limit')
        rows = shard['sequences']
        _require(isinstance(rows, list) and 1 <= len(rows) <= shard['records'], 'shard_sequence_rows')
        count, local = 0, set()
        for row in rows:
            _require(isinstance(row, dict) and set(row) == {'sequence_id', 'motion_id', 'split', 'records', 'first_frame', 'last_frame'},
                     'shard_sequence_fields')
            seq, motion, split = row['sequence_id'], row['motion_id'], row['split']
            _require(all(isinstance(x, str) and 0 < len(x.encode('utf8')) <= 1024 for x in (seq, motion)),
                     'shard_sequence_motion_id')
            if t1:
                _require(seq not in local,'t1_duplicate_sequence_group')
                _require(owners.get(seq,(motion,split))==(motion,split),'t1_sequence_motion_split_changed')
                owners[seq]=(motion,split); local.add(seq)
            else:
                _require(seq not in sequences, 'shard_sequence_crosses_shards')
            sequences.add(seq)
            _require(split in ('train', 'validation', 'test') and motions.get(motion, split) == split,
                     'shard_motion_split_overlap')
            motions[motion] = split
            _require(_integer(row['records'], 1, shard['records']) and
                     _integer(row['first_frame'], 0, 10**12) and
                     _integer(row['last_frame'], row['first_frame'], 10**12) and
                     row['records'] <= row['last_frame']-row['first_frame']+1, 'shard_sequence_bounds')
            if t1: ranges.setdefault(seq,[]).append((row['first_frame'],row['last_frame']))
            count += row['records']
        _require(count == shard['records'], 'shard_record_count')
    if t1:
        for chunks in ranges.values():
            ordered=sorted(chunks)
            _require(all(a[1] < b[0] for a,b in zip(ordered,ordered[1:])), 't1_chunk_overlap')
    return copy.deepcopy(doc)


def read_catalog(path):
    raw = _read(path, DEFAULT_LIMITS['max_catalog_bytes'])
    doc = validate_catalog(json.loads(raw))
    _require(len(raw) <= doc['limits']['max_catalog_bytes'], 'shard_catalog_byte_limit')
    return doc


def _document(raw, catalog, descriptor):
    _require(len(raw) == descriptor['size_bytes'] and hashlib.sha256(raw).hexdigest() == descriptor['sha256'],
             'shard_payload_identity')
    doc = json.loads(raw)
    _require(isinstance(doc, dict) and doc.get('contract') == catalog['contract'], 'shard_contract_mismatch')
    _require(isinstance(doc.get('manifest'), dict) and
             all(doc['manifest'].get(k) == v for k,v in catalog['dataset_identity'].items()),
             'shard_dataset_identity_mismatch')
    records = doc.get('records')
    _require(isinstance(records, list) and len(records) == descriptor['records'], 'shard_record_count')
    _require(_sequences(records) == descriptor['sequences'], 'shard_sequence_declaration_mismatch')
    return doc


class ShardDataset:
    """로컬 prepared 특징만 지원. 교체 전에 이전 tensor 참조를 버린다."""
    def __init__(self, catalog, base):
        self.catalog = validate_catalog(catalog)
        self.base = Path(base).resolve()
        self._cached = None
        self._index = None

    def clear(self):
        self._cached = self._index = None

    def load(self, index):
        _require(_integer(index, 0, len(self.catalog['shards'])-1), 'shard_index')
        if self._index != index:
            self.clear()
            row = self.catalog['shards'][index]
            raw = _read(_path(self.base, row['path']), self.catalog['limits']['max_shard_bytes'])
            doc = _document(raw, self.catalog, row)
            contract, splits, identity = load_document(doc, self.catalog['window_size'])
            _require(all(s.target_valid[:, -1].any(dim=1).all().item() for s in splits.values()),
                     'shard_anchor_without_target')
            self._cached = (contract, splits, identity)
            self._index = index
        return self._cached

    def verify_all(self, check_budget=None):
        counts = {}
        try:
            for index in range(len(self.catalog['shards'])):
                if check_budget is not None:
                    check_budget()
                loaded = self.load(index)
                for split, data in loaded[1].items():
                    counts[split] = counts.get(split, 0) + len(data.anchors)
                del loaded
        finally:
            self.clear()
        return counts


def build_catalog(paths, base, window_size=1, limits=None):
    """이미 준비한 특징 파일의 전체 읽기 검증. 원본 분할/정렬/승격은 하지 않는다."""
    return _build_catalog(paths,base,window_size,limits,SCHEMA)


def build_t1_catalog(paths, base, limits=None):
    """원본 sequence 이름을 유지한 독립 T1 특징 chunk만 명시적으로 묶는다."""
    return _build_catalog(paths,base,1,limits,T1_SCHEMA)


def _build_catalog(paths, base, window_size, limits, schema):
    limits = _limits(limits or DEFAULT_LIMITS)
    _require(_integer(window_size, 1, 256), 'shard_window_limit')
    _require(isinstance(paths, list) and 1 <= len(paths) <= limits['max_shards'], 'shard_count_limit')
    rows, contract, identity, catalog_bytes = [], None, None, 0
    for name in paths:
        raw = _read(_path(base, name), limits['max_shard_bytes'])
        doc = json.loads(raw)
        _require(isinstance(doc, dict) and isinstance(doc.get('records'), list), 'shard_document')
        _require(1 <= len(doc['records']) <= limits['max_records'], 'shard_record_limit')
        if contract is None:
            contract = doc.get('contract')
            identity = {k:doc.get('manifest', {}).get(k) for k in ('id', 'dataset_version')}
            catalog_bytes=_json_size({'schema_version':schema,'dataset_identity':identity,'contract':contract,
                'window_size':window_size,'limits':limits,'shards':[],'content_sha256':'0'*64},limits['max_catalog_bytes'])
        _require(doc.get('contract') == contract, 'shard_contract_mismatch')
        _require(isinstance(doc.get('manifest'), dict) and
                 all(doc['manifest'].get(k) == v for k,v in identity.items()), 'shard_dataset_identity_mismatch')
        row={'path': name, 'sha256': hashlib.sha256(raw).hexdigest(), 'size_bytes': len(raw),
             'records': len(doc['records']), 'sequences': _sequences(doc['records'])}
        catalog_bytes+=_json_size(row,limits['max_catalog_bytes'])+(1 if rows else 0)
        _require(catalog_bytes+1 <= limits['max_catalog_bytes'],'shard_catalog_byte_limit')
        rows.append(row)
    body = {'schema_version': schema, 'dataset_identity': identity, 'contract': contract, 'window_size': window_size,
            'limits': limits, 'shards': rows}
    catalog = validate_catalog({**body, 'content_sha256': content_hash(body)})
    _require(len(json.dumps(catalog).encode()) <= limits['max_catalog_bytes'], 'shard_catalog_byte_limit')
    ShardDataset(catalog, base).verify_all()
    return catalog


class EpochSampler:
    """고정 split의 shard/anchor 순회. peek 후 성공한 update만 commit한다."""
    POLICY = 'm1_shard_epoch_python_random_v1'

    def _policy(self):
        return T1_POLICY if self.catalog['schema_version']==T1_SCHEMA else self.POLICY

    def __init__(self, catalog, split, seed, shuffle=True):
        self.catalog = validate_catalog(catalog)
        _require(split in ('train', 'validation', 'test') and _integer(seed, 0, 2**32-1)
                 and type(shuffle) is bool, 'shard_sampler_arguments')
        self.split, self.seed, self.shuffle = split, seed, shuffle
        self.sizes = {i: sum(r['records'] for r in row['sequences'] if r['split'] == split)
                      for i, row in enumerate(self.catalog['shards'])}
        self.sizes = {i:n for i,n in self.sizes.items() if n}
        _require(bool(self.sizes), 'shard_sampler_empty_split')
        self.epoch, self.shard_position, self.offset = 0, 0, 0

    def _order(self):
        order = list(self.sizes)
        if self.shuffle:
            random.Random(f'{self.seed}:{self.epoch}:shards').shuffle(order)
        return order

    def peek(self, batch_size):
        _require(_integer(batch_size, 1, self.catalog['limits']['max_records']), 'shard_batch_size')
        shard = self._order()[self.shard_position]
        indices = list(range(self.sizes[shard]))
        if self.shuffle:
            random.Random(f'{self.seed}:{self.epoch}:{shard}:anchors').shuffle(indices)
        return shard, indices[self.offset:self.offset+batch_size]

    def commit(self, batch_size):
        shard, indices = self.peek(batch_size)
        self.offset += len(indices)
        if self.offset == self.sizes[shard]:
            self.offset = 0
            self.shard_position += 1
            if self.shard_position == len(self.sizes):
                self.shard_position = 0
                self.epoch += 1

    def state_dict(self):
        return {'policy': self._policy(), 'catalog_sha256': self.catalog['content_sha256'],
                'split': self.split, 'seed': self.seed, 'shuffle': self.shuffle,
                'epoch': self.epoch, 'shard_position': self.shard_position, 'offset': self.offset}

    def load_state_dict(self, state):
        expected = self.state_dict()
        _require(isinstance(state, dict) and set(state) == set(expected), 'shard_cursor_fields')
        _require(all(state[k] == expected[k] for k in ('policy', 'catalog_sha256', 'split', 'seed', 'shuffle')),
                 'shard_cursor_identity')
        _require(type(state['shuffle']) is bool and _integer(state['seed'], 0, 2**32-1), 'shard_cursor_identity')
        _require(_integer(state['epoch'], 0, 2**63-1) and
                 _integer(state['shard_position'], 0, len(self.sizes)-1), 'shard_cursor_position')
        old = self.epoch
        self.epoch = state['epoch']
        shard = self._order()[state['shard_position']]
        self.epoch = old
        _require(_integer(state['offset'], 0, self.sizes[shard]-1), 'shard_cursor_position')
        self.epoch, self.shard_position, self.offset = state['epoch'], state['shard_position'], state['offset']
