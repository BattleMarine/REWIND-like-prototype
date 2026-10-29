"""단일 SQLite 트랜잭션으로 catalog와 gzip restart 상태를 함께 전진시킨다."""
import bisect
import copy
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import sqlite3
import tarfile
import tempfile
import time
from dataclasses import dataclass, asdict
import zlib

from rewind_archive.contracts import canonical, digest, safe_path, PreparationError
from rewind_archive.extraction import pax_fields

SCHEMA = 'persistent_local_archive_index_v1'
MIB = 1024**2
CHUNK = 65536


class IndexFailure(ValueError):
    def __init__(self, reason):
        self.reason = reason
        super().__init__(reason)


@dataclass(frozen=True)
class IndexLimits:
    max_read_bytes: int = 256*MIB
    max_decoded_bytes: int = 128*MIB
    max_file_bytes: int = 8*MIB
    max_members: int = 20000
    max_points: int = 4096
    max_state_bytes: int = 16*MIB
    max_index_bytes: int = 32*MIB
    max_disk_bytes: int = 128*MIB
    max_rss_bytes: int = 512*MIB
    max_seconds: float = 120.

    def __post_init__(self):
        for key, value in asdict(self).items():
            if key == 'max_seconds':
                if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
                    raise IndexFailure('invalid_limits')
            elif type(value) is not int or value < 1:
                raise IndexFailure('invalid_limits')
        if self.max_members > 100000 or self.max_points > 100000:
            raise IndexFailure('unsafe_catalog_limits')


class Budget:
    def __init__(self, limits):
        self.limits = limits
        self.start = time.monotonic()
        self.read_bytes = self.decoded_bytes = 0
        self.trace = []
        try:
            import psutil
        except ImportError:
            raise IndexFailure('install_requirements_index') from None
        self.process = psutil.Process()
        self.peak_rss = 0
        self.last_source_failure = None

    def check(self):
        if time.monotonic()-self.start > self.limits.max_seconds:
            raise IndexFailure('elapsed_limit')
        rss = self.process.memory_info().rss
        self.peak_rss = max(self.peak_rss, rss)
        if rss > self.limits.max_rss_bytes:
            raise IndexFailure('rss_limit')

    def stats(self):
        return {'source_read_bytes': self.read_bytes, 'returned_decoded_bytes': self.decoded_bytes,
                'seconds': time.monotonic()-self.start, 'observed_peak_rss': self.peak_rss,
                'source_read_trace': self.trace, 'network_requested': False}


def _stat(path):
    path = Path(path)
    if path.is_symlink() or any(p.is_symlink() or getattr(p, 'is_junction', lambda: False)() for p in path.parents):
        raise IndexFailure('source_link')
    s = path.stat()
    if not path.is_file():
        raise IndexFailure('source_not_regular')
    return {'size': s.st_size, 'mtime_ns': s.st_mtime_ns, 'dev': s.st_dev, 'ino': s.st_ino}


def make_source(paths, archive_id, kind, dataset_version='local_unverified', expected_sha256=None):
    """완전히 존재하는 로컬 조각의 순서를 고정한다. 전체 SHA를 읽는 함수가 아니다."""
    if not isinstance(paths, (list, tuple)) or not 1 <= len(paths) <= 256:
        raise IndexFailure('source_parts')
    if kind not in ('rgb', 'pose', 'mixed') or not re.fullmatch(r'[A-Za-z0-9_.-]{1,128}', archive_id):
        raise IndexFailure('source_identity')
    if not isinstance(dataset_version, str) or not re.fullmatch(r'[A-Za-z0-9_.-]{1,128}', dataset_version):
        raise IndexFailure('dataset_identity')
    hashes = [None]*len(paths) if expected_sha256 is None else expected_sha256
    if not isinstance(hashes,(list,tuple)):
        raise IndexFailure('source_hashes')
    if len(hashes) != len(paths):
        raise IndexFailure('source_hashes')
    parts = []
    for i, (path, h) in enumerate(zip(paths, hashes)):
        p = Path(path).absolute()
        if h is not None and (not isinstance(h, str) or not re.fullmatch('[a-f0-9]{64}', h)):
            raise IndexFailure('source_hashes')
        meta = _stat(p)
        if meta['size'] < 1:
            raise IndexFailure('empty_source_part')
        parts.append({'index': i, 'path': str(p), 'stat': meta, 'expected_sha256': h})
    spec = {'schema': 'ordered_local_gzip_source_v1', 'archive_id': archive_id, 'kind': kind,
            'dataset_version': dataset_version, 'parts': parts,
            'total_bytes': sum(p['stat']['size'] for p in parts)}
    _validate_source(spec)
    return spec


def _validate_source(spec):
    if not isinstance(spec, dict) or set(spec) != {'schema', 'archive_id', 'kind', 'dataset_version', 'parts', 'total_bytes'}:
        raise IndexFailure('source_fields')
    if spec['schema'] != 'ordered_local_gzip_source_v1' or spec['kind'] not in ('rgb', 'pose', 'mixed'):
        raise IndexFailure('source_schema')
    if any(not isinstance(spec[k],str) or not re.fullmatch(r'[A-Za-z0-9_.-]{1,128}',spec[k]) for k in ('archive_id','dataset_version')):
        raise IndexFailure('source_identity')
    if not isinstance(spec['parts'], list) or not 1 <= len(spec['parts']) <= 256:
        raise IndexFailure('source_parts')
    names = set(); file_ids=set()
    for i, p in enumerate(spec['parts']):
        if not isinstance(p,dict) or set(p) != {'index', 'path', 'stat', 'expected_sha256'} or type(p['index']) is not int or p['index'] != i:
            raise IndexFailure('source_order_or_gap')
        if not isinstance(p['path'],str) or not isinstance(p['stat'],dict) or set(p['stat']) != {'size','mtime_ns','dev','ino'} or any(type(v) is not int or v<0 for v in p['stat'].values()):
            raise IndexFailure('source_stat_types')
        if p['expected_sha256'] is not None and (not isinstance(p['expected_sha256'],str) or not re.fullmatch('[a-f0-9]{64}',p['expected_sha256'])):
            raise IndexFailure('source_hashes')
        if '\0' in p['path'] or len(p['path'])>4096:raise IndexFailure('source_path')
        path = Path(p['path']);key=str(path.resolve()).casefold();fid=(p['stat']['dev'],p['stat']['ino'])
        if not path.is_absolute() or key in names or fid[1] and fid in file_ids or p['stat'].get('size', 0) < 1:
            raise IndexFailure('source_duplicate_or_size')
        names.add(key);file_ids.add(fid)
        if _stat(path) != p['stat']:
            raise IndexFailure('source_changed')
    if type(spec['total_bytes']) is not int or spec['total_bytes'] != sum(p['stat']['size'] for p in spec['parts']):
        raise IndexFailure('source_total_length')


class SplitSource(io.RawIOBase):
    def __init__(self, spec, budget):
        self.spec, self.budget = spec, budget
        _validate_source(spec)
        self.size = spec['total_bytes']
        self.ends = []; n = 0
        for p in spec['parts']:
            n += p['stat']['size']; self.ends.append(n)
        self.pos = 0
        self.handles = {}
        self.mode = 'rb'

    def readable(self): return True
    def seekable(self): return True
    def tell(self): return self.pos

    def seek(self, offset, whence=0):
        target = offset if whence == 0 else self.pos+offset if whence == 1 else self.size+offset if whence == 2 else -1
        if target < 0:
            raise IndexFailure('negative_source_seek')
        self.pos = target
        return target

    def read(self, size=-1):
        try:
            return self._read(size)
        except IndexFailure as exc:
            self.budget.last_source_failure = exc.reason
            raise

    def _read(self, size):
        if not 0 <= size <= MIB:
            raise IndexFailure('source_read_buffer_limit')
        self.budget.check()
        size = min(size, max(0, self.size-self.pos))
        if self.budget.read_bytes+size > getattr(self.budget,'active_read_limit',self.budget.limits.max_read_bytes):
            raise IndexFailure('read_bytes_limit')
        chunks = []
        while size:
            i = bisect.bisect_right(self.ends, self.pos)
            p = self.spec['parts'][i]
            if _stat(p['path']) != p['stat']:
                raise IndexFailure('source_changed')
            if i not in self.handles:
                self.handles[i] = open(p['path'], 'rb')
            f = self.handles[i]
            before = os.fstat(f.fileno())
            if (before.st_size, before.st_mtime_ns, before.st_dev, before.st_ino) != tuple(p['stat'][k] for k in ('size', 'mtime_ns', 'dev', 'ino')):
                raise IndexFailure('source_changed')
            start = 0 if i == 0 else self.ends[i-1]
            amount = min(size, self.ends[i]-self.pos, CHUNK)
            f.seek(self.pos-start); data = f.read(amount)
            if len(data) != amount or _stat(p['path']) != p['stat']:
                raise IndexFailure('source_changed_or_short')
            self.budget.trace.append({'part': i, 'start': self.pos, 'bytes': amount})
            if len(self.budget.trace) > 100000:
                raise IndexFailure('read_trace_limit')
            self.budget.read_bytes += amount
            self.pos += amount; size -= amount; chunks.append(data)
            self.budget.check()
        return b''.join(chunks)

    def close(self):
        for f in self.handles.values(): f.close()
        super().close()


def _gzip(source, spacing, auto_build=True):
    try:
        import indexed_gzip
    except ImportError:
        raise IndexFailure('install_requirements_index') from None
    return indexed_gzip.IndexedGzipFile(fileobj=source, spacing=spacing, buffer_size=4*spacing,
        readbuf_size=CHUNK, readall_buf_size=CHUNK, drop_handles=False,auto_build=auto_build)


def _read(gz, size, budget, exact=True):
    result = []
    while size:
        budget.check()
        if budget.decoded_bytes >= budget.limits.max_decoded_bytes:
            raise IndexFailure('decoded_bytes_limit')
        take = min(size, CHUNK, budget.limits.max_decoded_bytes-budget.decoded_bytes)
        data = gz.read(take)
        budget.decoded_bytes += len(data)
        if not data:
            if exact: raise IndexFailure('incomplete_tar')
            break
        result.append(data); size -= len(data)
    return b''.join(result)


def _pack_index(gz, limits):
    points = list(gz.seek_points())
    if len(points) > limits.max_points:
        raise IndexFailure('index_points_limit')
    # dictionary32KiB + per-point metadata를 export 할당 전에 보수적으로 제한한다.
    if len(points)*(32768+128)+4096 > limits.max_index_bytes:
        raise IndexFailure('index_bytes_limit')
    stream = io.BytesIO(); gz.export_index(fileobj=stream)
    data = stream.getvalue()
    if len(data) > limits.max_index_bytes:
        raise IndexFailure('index_bytes_limit')
    return data, points
