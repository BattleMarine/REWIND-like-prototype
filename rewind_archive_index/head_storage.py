"""단일 writer용 mount HEAD 게시. Hard link/fsync/분산 atomicity를 요구하지 않는다."""
import hashlib
import json
import os
from pathlib import Path
import uuid

from .extension_records import _read,RECORD_LIMIT
from .source import IndexFailure,canonical,digest


def _bytes(value):
    if not isinstance(value,dict) or value.get('schema')!='archive_prepare_head_v1' or value.get('content_sha256')!=digest({k:v for k,v in value.items() if k!='content_sha256'}):
        raise IndexFailure('prepare_head_identity_or_hash')
    raw=canonical(value)+b'\n'
    if len(raw)>RECORD_LIMIT:raise IndexFailure('prepare_record_size')
    return raw


def _existing(path):
    raw=_read(path,path.parent,RECORD_LIMIT)
    try:value=json.loads(raw)
    except (ValueError,RecursionError):raise IndexFailure('prepare_head_invalid_json') from None
    _bytes(value)
    return raw,value


def publish_head(path,value,expected,*,state,progress):
    """초기 exclusive 생성, 갱신 전 검증된 이전본 보존. 실패한 파일은 지우지 않는다."""
    path=Path(path).absolute();raw=_bytes(value)
    state.clear();state.update(verified=False,mount_atomicity_claimed=False,durability_proven=False)
    if any(p.is_symlink() or getattr(p,'is_junction',lambda:False)() for p in (path,*path.parents)):
        raise IndexFailure('prepare_path_link')
    def operation(name):
        state['operation']=name
        progress(name)
    def compare():
        if expected is None:
            if path.exists() or path.is_symlink():raise IndexFailure('prepare_head_cas_conflict')
            return None
        old,document=_existing(path)
        if document!=expected:raise IndexFailure('prepare_head_cas_conflict')
        return old
    operation('head_check_pre');old=compare()
    if old is None:
        state['method']='exclusive_create'
        operation('head_create_pre')
        with path.open('xb') as stream:stream.write(raw)
    else:
        backup=path.with_name(path.stem+'-previous-'+hashlib.sha256(old).hexdigest()+'.json')
        state['previous_path']=str(backup)
        operation('head_backup_pre')
        try:
            with backup.open('xb') as stream:stream.write(old)
        except FileExistsError:
            pass  # 같은 내용의 닫힌 backup만 재사용한다. 불완전 backup은 거부한다.
        if _read(backup,path.parent,RECORD_LIMIT)!=old:raise IndexFailure('prepare_head_backup_conflict')
        staging=path.with_name('.head-'+uuid.uuid4().hex+'.json')
        state.update(method='closed_replace',staging_path=str(staging))
        operation('head_stage_pre')
        with staging.open('xb') as stream:stream.write(raw)
        if _read(staging,path.parent,RECORD_LIMIT)!=raw:raise IndexFailure('prepare_head_staging_hash')
        operation('head_cas_pre');compare()
        operation('head_replace_pre')
        # FUSE rename의 atomicity/durability를 가정하지 않는다. 이전 HEAD는 backup에 남는다.
        os.replace(staging,path)
    operation('head_verify_pre')
    observed,document=_existing(path)
    if observed!=raw or document!=value:raise IndexFailure('prepare_head_publish_hash')
    state.update(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest(),verified=True,
                 mount_atomicity_claimed=False,durability_proven=False)
    operation('head_publish_post')
    return state
