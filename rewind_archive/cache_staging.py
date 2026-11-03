"""명시 readonly 경로의 cache를 bounded local 파일로 준비한다. 원본 archive 요청 없음."""
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
import time

from .contracts import PreparationError,canonical

CHUNK_BYTES=1024**2
CACHE_BYTES=246522402
TARGET8_CACHE_BYTES=250718584
TARGET8_CACHE_SHA256='e8516bf6214bac6253551ec778dad6a57a3c6d6657af4200cab12994713f996f'
CACHE_SHA256='f902b25aadb1932c4c734795592e122061f8d30410bdeed5bb3866528ebff437'
DISK_BYTES=256*1024**2
REPORT_BYTES=65536
VALIDATOR_METADATA_BYTES=2*1024**2
SECONDS=300.


def checked(path):
    value=Path(path).absolute()
    for part in (value,*value.parents):
        if part.is_symlink() or getattr(part,'is_junction',lambda:False)():
            raise PreparationError('cache_stage_link')
    return value.resolve(strict=False)


def stage_file(source,target,*,expected_bytes,expected_sha256,readonly_root,local_root,forbidden_roots,
               consumed_job,report_path,validator,clock=time.monotonic,seconds=SECONDS,max_bytes=CACHE_BYTES):
    """새 partial만 xb로 복사한다. 이미 있는 exact target은 검증 후 읽기 재사용한다."""
    if type(max_bytes) is not int or max_bytes not in (CACHE_BYTES,TARGET8_CACHE_BYTES):
        raise PreparationError('cache_stage_size_scope')
    if type(expected_bytes) is not int or not 0<expected_bytes<=max_bytes or not re.fullmatch('[a-f0-9]{64}',expected_sha256):
        raise PreparationError('cache_stage_expected_pin')
    if type(seconds) not in (int,float) or not 0<seconds<=SECONDS:
        raise PreparationError('cache_stage_time_config')
    source,target,readonly_root,local_root,job,report_path=map(checked,(source,target,readonly_root,local_root,consumed_job,report_path))
    partial=checked(target.with_name(target.name+'.partial'))
    lock=checked(target.with_name(target.name+'.stage.lock'))
    forbidden=[checked(p) for p in forbidden_roots]
    if job.exists():raise PreparationError('cache_stage_consumed_job')
    if source==target or source in (partial,lock,report_path) or target==report_path or partial==report_path or lock==report_path:
        raise PreparationError('cache_stage_path_collision')
    if source.exists() and target.exists() and os.path.samefile(source,target):
        raise PreparationError('cache_stage_same_file')
    if not source.is_relative_to(readonly_root) or not target.is_relative_to(local_root) or not report_path.is_relative_to(local_root):
        raise PreparationError('cache_stage_explicit_roots')
    if any(p.is_relative_to(readonly_root) for p in (target,partial,lock,report_path)):
        raise PreparationError('cache_stage_readonly_destination')
    if any(p.is_relative_to(job) for p in (target,partial,lock,report_path)):
        raise PreparationError('cache_stage_inside_job')
    if any(p.is_relative_to(root) for p in (source,target,partial,lock,report_path) for root in forbidden):
        raise PreparationError('cache_stage_source_or_input_tree')
    started=clock();source_read=local_read=written=0;owned_lock=False
    identity={'source':str(source),'target':str(target),'expected_bytes':expected_bytes,'expected_sha256':expected_sha256}
    history=[]
    if report_path.exists():
        if not report_path.is_file() or report_path.stat().st_size>REPORT_BYTES:raise PreparationError('cache_stage_report_limit')
        previous=json.loads(report_path.read_bytes())
        if previous.get('identity')!=identity:raise PreparationError('cache_stage_report_pin')
        history=previous.get('attempts',[])
        if not isinstance(history,list) or len(history)>=8:raise PreparationError('cache_stage_attempt_limit')
    attempt={'status':'started','remote_archive_requested':False}
    report={'schema':'bounded_cache_stage_v1','identity':identity,'attempts':history+[attempt],
        'limits':{'chunk_bytes':CHUNK_BYTES,'source_read_bytes':expected_bytes,'local_verify_bytes':expected_bytes,
            'seconds':seconds,'staging_disk_bytes':DISK_BYTES},
        'scope':'separate cache preparation; logical file reads, not FUSE wire traffic; validator ZIP metadata counted separately; mount/auth and archive prepare excluded; cooperative boundaries'}

    def guard():
        if clock()-started>seconds:raise PreparationError('cache_stage_time_limit')
        size=sum(p.stat().st_size for p in (target,partial) if p.exists())
        if size+2*REPORT_BYTES>DISK_BYTES:raise PreparationError('cache_stage_disk_limit')
    def save():
        raw=canonical(report)+b'\n'
        if len(raw)>REPORT_BYTES:raise PreparationError('cache_stage_report_limit')
        report_path.parent.mkdir(parents=True,exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=report_path.parent,prefix='.cache-stage-',delete=False) as stream:
            temp=Path(stream.name);stream.write(raw)
        try:os.replace(temp,report_path)
        finally:temp.unlink(missing_ok=True)
    def verify_local(path):
        nonlocal local_read
        if not path.is_file() or path.stat().st_size!=expected_bytes:raise PreparationError('cache_stage_local_size')
        h=hashlib.sha256()
        with path.open('rb') as stream:
            while data:=stream.read(CHUNK_BYTES):
                local_read+=len(data);guard()
                if local_read>expected_bytes:raise PreparationError('cache_stage_local_read_limit')
                h.update(data)
        if h.hexdigest()!=expected_sha256:raise PreparationError('cache_stage_local_sha256')
        guard()
        try:validated=validator(path)
        finally:
            if hasattr(validator,'metadata_read_bytes'):attempt['validator_metadata_read_bytes']=validator.metadata_read_bytes
        guard()
        return validated
    save()
    try:
        guard()
        lock.parent.mkdir(parents=True,exist_ok=True)
        with lock.open('xb') as stream:stream.write(b'bounded-cache-stage\n')
        owned_lock=True
        if partial.exists():raise PreparationError('cache_stage_partial_exists')
        if target.exists():
            validated=verify_local(target)
            attempt.update(status='verified_existing_staging',source_unchanged='not_read_on_reuse')
        else:
            if not source.is_file() or source.stat().st_size!=expected_bytes:raise PreparationError('cache_stage_source_size')
            before=source.stat()
            h=hashlib.sha256()
            with source.open('rb') as src,partial.open('xb') as dst:
                remaining=expected_bytes
                while remaining:
                    guard();data=src.read(min(CHUNK_BYTES,remaining))
                    source_read+=len(data)
                    guard()
                    if not data:raise PreparationError('cache_stage_source_truncated')
                    if dst.write(data)!=len(data):raise PreparationError('cache_stage_short_write')
                    written+=len(data);remaining-=len(data);h.update(data)
                dst.flush();os.fsync(dst.fileno())
            after=source.stat()
            if (before.st_size,before.st_mtime_ns,before.st_ino)!=(after.st_size,after.st_mtime_ns,after.st_ino):
                raise PreparationError('cache_stage_source_stat_changed')
            if h.hexdigest()!=expected_sha256:raise PreparationError('cache_stage_source_sha256')
            validated=verify_local(partial)
            guard()
            if os.name=='nt':os.rename(partial,target)  # Windows no-clobber rename
            else:os.link(partial,target);partial.unlink()  # local filesystem의 atomic no-clobber 게시
            attempt.update(status='verified_copied_staging',source_unchanged='readonly_stat_and_full_byte_pin_verified')
        if isinstance(validated,dict) and 'path' in validated:validated={**validated,'path':str(target)}
        attempt['validation']=validated
    except BaseException as exc:
        attempt.update(status='failed_or_interrupted',error_type=type(exc).__name__,
            reason=exc.reason if isinstance(exc,PreparationError) else 'cache_stage_io_failed')
        raise
    finally:
        if owned_lock:
            try:lock.unlink()
            except OSError:attempt['lock_cleanup_failed']=True
        attempt.update(source_read_bytes=source_read,local_verify_bytes=local_read,written_bytes=written,
            elapsed_seconds=clock()-started,partial_preserved=partial.exists(),
            partial_bytes=partial.stat().st_size if partial.exists() else 0)
        save()
    return {'path':str(target),'cache_input':validated,'report':report}


def stage_pose269_cache(source,target,*,source_root,input_root,workspace,report_path,readonly_root,local_root):
    from .prefix_cache import input_spec
    from .cache_policy import TAIL352_POLICY
    from .profiles import TARGET269_CACHE_EXPECTATION
    class Validator:
        metadata_read_bytes=0
        def read(self,stream,size):
            raw=stream.read(size);self.metadata_read_bytes+=len(raw)
            if self.metadata_read_bytes>VALIDATOR_METADATA_BYTES:raise PreparationError('cache_stage_validator_read_limit')
            return raw
        def __call__(self,path):
            spec=input_spec(path,policy=TAIL352_POLICY,read_callback=self.read)
            if spec['manifest_sha256']!=TARGET269_CACHE_EXPECTATION['manifest_sha256']:
                raise PreparationError('cache_stage_manifest_pin')
            return spec
    return stage_file(source,target,expected_bytes=CACHE_BYTES,expected_sha256=CACHE_SHA256,
        readonly_root=readonly_root,local_root=local_root,forbidden_roots=(source_root,input_root),
        consumed_job=workspace,report_path=report_path,validator=Validator())


def stage_pose8_cache(source,target,*,source_root,input_root,workspace,report_path,readonly_root,local_root):
    from .prefix_cache import input_spec
    from .cache_policy import TAIL352_POLICY
    from .profiles import TARGET8_CACHE_EXPECTATION
    class Validator:
        metadata_read_bytes=0
        def read(self,stream,size):
            raw=stream.read(size);self.metadata_read_bytes+=len(raw)
            if self.metadata_read_bytes>VALIDATOR_METADATA_BYTES:raise PreparationError('cache_stage_validator_read_limit')
            return raw
        def __call__(self,path):
            spec=input_spec(path,policy=TAIL352_POLICY,read_callback=self.read)
            if spec['manifest_sha256']!=TARGET8_CACHE_EXPECTATION['manifest_sha256']:
                raise PreparationError('cache_stage_manifest_pin')
            return spec
    return stage_file(source,target,expected_bytes=TARGET8_CACHE_BYTES,expected_sha256=TARGET8_CACHE_SHA256,max_bytes=TARGET8_CACHE_BYTES,
        readonly_root=readonly_root,local_root=local_root,forbidden_roots=(source_root,input_root),
        consumed_job=workspace,report_path=report_path,validator=Validator())
