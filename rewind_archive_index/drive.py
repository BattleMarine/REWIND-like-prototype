"""런타임 주입 session과 bounded RAM Range cache. 인증/네트워크 자동 시작 없음."""
from collections import OrderedDict
from dataclasses import dataclass
import bisect
import io
import re
import time

from rewind_archive.contracts import Limits, Budget as TransferBudget, PreparationError, digest, validate_manifest
from rewind_archive.sources import DriveSource
from rewind_archive.request_policy import validate_no_hidden_retries
from rewind_archive.diagnostics import safe_request_failure
from .source import IndexFailure, MIB, CHUNK

DRIVE_SCHEMA='ordered_drive_gzip_source_v1'


def make_drive_source(manifest):
    manifest=validate_manifest(manifest)
    if manifest['source_scope']!='user_drive' or len(manifest['parts'])!=manifest['total_parts']:
        raise IndexFailure('complete_drive_metadata_required')
    parts=[]
    for p in manifest['parts']:
        if p['source']['type']!='drive':raise IndexFailure('drive_parts_required')
        parts.append({'index':p['index'],'file_id':p['source']['file_id'],'size':p['size_bytes'],
            'version':p['identity']['version'],'modified_time':p['identity']['modified_time']})
    spec={'schema':DRIVE_SCHEMA,'archive_id':manifest['archive_id'],'kind':manifest['kind'],
          'dataset_version':manifest['dataset_version'],'parts':parts,'total_bytes':sum(p['size'] for p in parts)}
    validate_drive_spec(spec)
    return spec


def validate_drive_spec(spec):
    if not isinstance(spec,dict) or set(spec)!={'schema','archive_id','kind','dataset_version','parts','total_bytes'} or spec['schema']!=DRIVE_SCHEMA:
        raise IndexFailure('drive_spec_fields')
    if spec['kind'] not in ('rgb','pose','mixed') or any(not isinstance(spec[k],str) or not re.fullmatch(r'[A-Za-z0-9_.-]{1,128}',spec[k]) for k in ('archive_id','dataset_version')):
        raise IndexFailure('drive_spec_identity')
    if not isinstance(spec['parts'],list) or not 1<=len(spec['parts'])<=256:raise IndexFailure('drive_part_count')
    seen=set()
    for i,p in enumerate(spec['parts']):
        if not isinstance(p,dict) or set(p)!={'index','file_id','size','version','modified_time'} or type(p['index']) is not int or p['index']!=i:
            raise IndexFailure('drive_part_order')
        if not isinstance(p['file_id'],str) or not re.fullmatch(r'[A-Za-z0-9_-]{1,200}',p['file_id']) or p['file_id'] in seen:
            raise IndexFailure('drive_duplicate_or_id')
        if type(p['size']) is not int or p['size']<1:raise IndexFailure('drive_part_size')
        if any(not isinstance(p[k],str) or not 1<=len(p[k])<=128 or any(ord(c)<32 for c in p[k]) for k in ('version','modified_time')):
            raise IndexFailure('drive_part_version')
        seen.add(p['file_id'])
    if type(spec['total_bytes']) is not int or spec['total_bytes']!=sum(p['size'] for p in spec['parts']):raise IndexFailure('drive_total_size')


@dataclass(frozen=True)
class DriveLimits:
    max_body_bytes:int=16*MIB
    max_requests:int=128
    max_seconds:float=60.
    max_cache_bytes:int=4*MIB
    range_bytes:int=256*1024

    def __post_init__(self):
        # 기존 core의 Limits 검사도 함께 재사용한다.
        self.transport_limits()
        if type(self.max_cache_bytes) is not int or not self.range_bytes<=self.max_cache_bytes<=64*MIB:
            raise IndexFailure('drive_cache_limit')

    def transport_limits(self,seconds=None):
        return Limits(max_transfer_bytes=self.max_body_bytes,max_ranges=self.max_requests,
            max_elapsed_seconds=self.max_seconds if seconds is None else seconds,
            request_timeout_seconds=min(5.,self.max_seconds if seconds is None else seconds),media_range_bytes=self.range_bytes)


class DriveRuntime:
    def __init__(self,session,limits=None):
        self.session=session;self.limits=limits or DriveLimits()
        self.cache=OrderedDict();self.cache_bytes=0;self.operation=None;self.failed=False

    def _start(self,spec,budget):
        validate_drive_spec(spec)
        if self.operation is budget:
            if self.fingerprint!=digest(spec):raise IndexFailure('drive_runtime_source_conflict')
            if self.failed:raise IndexFailure('drive_operation_invalid')
            return
        validate_no_hidden_retries(self.session)
        self.operation=budget;self.fingerprint=digest(spec);self.failed=False;self.finished=False
        self.hits=self.misses=self.cache_read_bytes=0
        self.last_failure=None
        self.request_start=getattr(getattr(self,'transfer',None),'ranges',0)
        self.body_start=getattr(getattr(self,'transfer',None),'transfer_bytes',0)
        self.media_start=getattr(getattr(self,'transfer',None),'media_requests',0)
        self.metadata_start=getattr(getattr(self,'transfer',None),'metadata_requests',0)
        if self.limits.max_seconds is None:
            if budget.limits.max_seconds is not None:raise IndexFailure('drive_time_policy_mismatch')
            if not hasattr(self,'transfer'):self.transfer=self.limits.new_transfer_budget()
        else:
            remaining=min(self.limits.max_seconds,budget.limits.max_seconds-(time.monotonic()-budget.start))
            if remaining<=0:raise IndexFailure('elapsed_limit')
            # 같은 runtime의 scan/extract는 하나의 phase 전송 예산을 공유한다.
            if not hasattr(self,'transfer'):self.transfer=TransferBudget(self.limits.transport_limits(remaining))
        self.sources=[]
        for p in spec['parts']:
            self.sources.append(DriveSource({'size_bytes':p['size'],'source':{'file_id':p['file_id']},
                'identity':{'version':p['version'],'modified_time':p['modified_time']}},self.session))
        budget.remote_stats=self.stats
        try:
            for source in self.sources:source.check_metadata(self.transfer)
        except PreparationError as exc:self._fail(exc)

    def _fail(self,exc):
        self.cache.clear();self.cache_bytes=0;self.failed=True
        self.last_failure=safe_request_failure(exc.request_failure)
        if self.operation is not None:self.operation.last_source_failure=exc.reason
        raise IndexFailure(exc.reason) from None

    def source(self,spec,budget):
        self._start(spec,budget)
        return DriveReader(spec,budget,self)

    def block(self,part,start,size):
        if self.failed:raise IndexFailure('drive_operation_invalid')
        key=(self.fingerprint,part,start,size)
        if key in self.cache:
            self.hits+=1;self.cache.move_to_end(key);data=self.cache[key];self.cache_read_bytes+=len(data);return data
        self.misses+=1
        # 각 uncached Range의 pre/media/post 외에 operation의 마지막 metadata를 예약한다.
        if self.transfer.ranges+3+len(self.sources)>self.transfer.limits.max_ranges:raise IndexFailure('ranges_limit')
        if self.transfer.transfer_bytes+size+(2+len(self.sources))*DriveSource.METADATA_RESPONSE_BYTES>self.transfer.limits.max_transfer_bytes:
            raise IndexFailure('transfer_bytes_limit')
        if self.limits.max_seconds is not None and self.transfer.check_time()<=min(10.,self.limits.max_seconds/4):raise IndexFailure('elapsed_limit')
        try:data=self.sources[part].read_range(start,start+size-1,self.transfer,lambda _:None)
        except PreparationError as exc:self._fail(exc)
        while self.cache_bytes+len(data)>self.limits.max_cache_bytes:
            _,old=self.cache.popitem(last=False);self.cache_bytes-=len(old)
        self.cache[key]=data;self.cache_bytes+=len(data)
        return data

    def finish(self,budget):
        if self.operation is not budget or self.failed:raise IndexFailure('drive_operation_invalid')
        try:
            for source in self.sources:source.check_metadata(self.transfer,'metadata_post')
        except PreparationError as exc:self._fail(exc)
        self.finished=True
        self.last_io_end=time.monotonic()

    def stats(self):
        t=self.transfer
        return {'network_requested':t.ranges>self.request_start,'remote_response_body_bytes':t.transfer_bytes-self.body_start,
            'remote_http_requests':t.ranges-self.request_start,'remote_media_requests':t.media_requests-self.media_start,
            'remote_metadata_requests':t.metadata_requests-self.metadata_start,'remote_cache_hits':self.hits,'remote_cache_misses':self.misses,
            'remote_cache_read_bytes':self.cache_read_bytes,'remote_cache_resident_bytes':self.cache_bytes,
            'remote_final_identity_checked':self.finished and not self.failed,
            'phase_http_requests':t.ranges,'phase_response_body_bytes':t.transfer_bytes,
            'phase_seconds':(self.last_io_end if self.finished else time.monotonic())-t.started,
            'remote_request_failure':self.last_failure}


class DriveReader(io.RawIOBase):
    def __init__(self,spec,budget,runtime):
        self.spec=spec;self.budget=budget;self.runtime=runtime;self.pos=0;self.mode='rb'
        self.size=spec['total_bytes'];self.ends=[];total=0
        for p in spec['parts']:total+=p['size'];self.ends.append(total)

    def readable(self):return True
    def seekable(self):return True
    def tell(self):return self.pos
    def seek(self,offset,whence=0):
        target=offset if whence==0 else self.pos+offset if whence==1 else self.size+offset if whence==2 else -1
        if target<0:raise IndexFailure('negative_source_seek')
        self.pos=target;return target

    def read(self,size=-1):
        try:return self._read(size)
        except IndexFailure as exc:self.budget.last_source_failure=exc.reason;raise

    def _read(self,size):
        self.budget.check()
        if not 0<=size<=MIB:raise IndexFailure('source_read_buffer_limit')
        size=min(size,max(0,self.size-self.pos))
        if self.budget.read_bytes+size>getattr(self.budget,'active_read_limit',self.budget.limits.max_read_bytes):raise IndexFailure('read_bytes_limit')
        output=[]
        while size:
            part=bisect.bisect_right(self.ends,self.pos);base=0 if part==0 else self.ends[part-1]
            relative=self.pos-base;start=relative//self.runtime.limits.range_bytes*self.runtime.limits.range_bytes
            length=min(self.runtime.limits.range_bytes,self.spec['parts'][part]['size']-start)
            block=self.runtime.block(part,start,length);inside=relative-start;amount=min(size,len(block)-inside)
            if amount<=0:raise IndexFailure('drive_cache_range')
            output.append(block[inside:inside+amount]);self.budget.read_bytes+=amount
            self.budget.trace.append({'part':part,'start':self.pos,'bytes':amount})
            if len(self.budget.trace)>100000:raise IndexFailure('read_trace_limit')
            self.pos+=amount;size-=amount;self.budget.check()
        return b''.join(output)
