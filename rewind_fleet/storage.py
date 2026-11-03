"""불변 bounded pack/catalog/commit과 readback marker. Live SQLite는 local cache만 사용."""
import contextlib
import hashlib
import json
import os
from pathlib import Path,PurePosixPath
import re
import time
import uuid
from rewind_archive.contracts import canonical
from .window import check

def sha(raw):return hashlib.sha256(raw).hexdigest()
def path(value):
    p=Path(value).absolute()
    check(not any(x.is_symlink() or getattr(x,'is_junction',lambda:False)() for x in (p,*p.parents)),'fleet_link')
    return p

def status(root,*,io_limit=None):
    """원본 네트워크·mount 시도 없이 작은 HEAD/lease/pending 상태만 회수한다."""
    report={'archive_network_requested':False,'errors':[],'head':None,'pending':None,'lease':None,'artifact_logical_io':0};store=None
    try:
        p=path(root);report['store_exists']=p.is_dir()
        if not p.is_dir():return report
        store=Store(p);store.read_limit=io_limit
        try:
            key=store.head();report['head']=key
            if key:
                ledger=store.committed(key)
                for field in ('pending','settlement','usage','plan_sha256','statuses','complete'):report[field]=ledger.get(field)
        except Exception as exc:report['errors'].append({'stage':'head','type':type(exc).__name__,'reason':getattr(exc,'reason','unreadable')})
        if (p/'WRITER.lease').exists():
            try:report['lease']=json.loads(store.read('WRITER.lease',4096))
            except Exception as exc:report['errors'].append({'stage':'lease','type':type(exc).__name__,'reason':getattr(exc,'reason','unreadable')})
    except Exception as exc:report['errors'].append({'stage':'store','type':type(exc).__name__,'reason':getattr(exc,'reason','unreadable')})
    if store is not None:report['artifact_logical_io']=store.io_bytes
    return report

class Store:
    def __init__(self,root,guard=lambda:None):
        self.root=path(root);self.root.mkdir(parents=True,exist_ok=True);self.guard=guard;self.io_bytes=0;self.written_bytes=0;self.write_check=lambda n:None;self.read_limit=None
    def read(self,name,maximum):
        check(isinstance(name,str) and bool(name) and '\\' not in name and ':' not in name and
            not PurePosixPath(name).is_absolute() and '..' not in PurePosixPath(name).parts and
            name==PurePosixPath(name).as_posix() and name!='.','fleet_relative_object_name')
        self.guard();p=path(self.root/name)
        check(p.is_relative_to(self.root) and p.is_file() and p.stat().st_size<=maximum,'fleet_object_path_or_size')
        if self.read_limit is not None:check(self.io_bytes+p.stat().st_size<=self.read_limit,'fleet_store_read_io_limit')
        with p.open('rb') as f:raw=f.read(maximum+1)
        self.io_bytes+=len(raw);check(len(raw)<=maximum,'fleet_object_size');self.guard();return raw
    def put(self,raw):
        check(len(raw)<=4*1024**2,'fleet_object_capacity');key=sha(raw);name='objects/'+key
        p=path(self.root/name);p.parent.mkdir(exist_ok=True)
        if not p.exists():
            self.guard();self.write_check(len(raw))
            stage=path(p.with_name('.stage-'+uuid.uuid4().hex+'.partial'))
            with stage.open('xb') as f:f.write(raw)
            self.io_bytes+=len(raw);self.written_bytes+=len(raw)
            check(self.read(stage.relative_to(self.root).as_posix(),len(raw))==raw,'fleet_stage_readback')
            check(not p.exists(),'fleet_object_publish_conflict');os.rename(stage,p)
        check(self.read(name,len(raw))==raw,'fleet_object_readback');return key
    def blob(self,key,maximum=4*1024**2):
        check(isinstance(key,str) and re.fullmatch('[a-f0-9]{64}',key),'fleet_object_key')
        raw=self.read('objects/'+key,maximum);check(sha(raw)==key,'fleet_object_hash');return raw
    def document(self,key):return json.loads(self.blob(key))
    def commit(self,doc):
        raw=canonical(doc)+b'\n';key=self.put(raw);marker=path(self.root/('commit-'+key+'.json'))
        body=canonical({'schema':'fleet_commit_marker_v1','commit':key})+b'\n'
        if not marker.exists():
            self.write_check(len(body))
            stage=path(marker.with_name('.marker-'+uuid.uuid4().hex+'.partial'))
            with stage.open('xb') as f:f.write(body)
            self.io_bytes+=len(body);self.written_bytes+=len(body)
            check(self.read(stage.name,1024)==body and not marker.exists(),'fleet_marker_stage')
            os.rename(stage,marker)
        check(self.read(marker.name,1024)==body,'fleet_commit_marker_readback');return key
    def committed(self,key):
        check(isinstance(key,str) and re.fullmatch('[a-f0-9]{64}',key),'fleet_object_key')
        marker=json.loads(self.read('commit-'+key+'.json',1024));check(marker=={'schema':'fleet_commit_marker_v1','commit':key},'fleet_marker_identity')
        return self.document(key)
    def repair_object(self,key,raw,*,owner_confirmed_stopped=False):
        check(owner_confirmed_stopped is True and sha(raw)==key and len(raw)<=4*1024**2,'fleet_repair_confirmation_or_hash')
        p=path(self.root/'objects'/key);old=self.read('objects/'+key,4*1024**2)
        check(old!=raw,'fleet_repair_normal_object_forbidden')
        quarantine=path(self.root/('quarantine-'+key+'-'+uuid.uuid4().hex))
        os.rename(p,quarantine) # 명시 복구에서만 이동하며 손상 원본은 삭제하지 않는다.
        return self.put(raw)
    def head(self):
        p=path(self.root/'HEAD.json')
        if not p.exists():return None
        raw=self.read('HEAD.json',4096);doc=json.loads(raw)
        check(set(doc)=={'schema','commit'} and doc['schema']=='fleet_head_v1','fleet_head_schema')
        self.committed(doc['commit']);return doc['commit']
    def publish(self,key,expected):
        check(self.head()==expected,'fleet_head_conflict');self.committed(key)
        raw=canonical({'schema':'fleet_head_v1','commit':key})+b'\n'
        if expected:
            backup=path(self.root/('head-previous-'+expected+'.json'))
            if not backup.exists():
                old=canonical({'schema':'fleet_head_v1','commit':expected})+b'\n';self.write_check(len(old))
                with backup.open('xb') as f:f.write(old)
                self.written_bytes+=len(old);self.io_bytes+=len(old)
        temp=self.root/('.head-'+uuid.uuid4().hex)
        self.write_check(len(raw))
        with temp.open('xb') as f:f.write(raw)
        self.written_bytes+=len(raw);self.io_bytes+=len(raw)
        check(self.head()==expected,'fleet_head_conflict');os.replace(temp,self.root/'HEAD.json')
        check(self.head()==key,'fleet_head_readback')
    @contextlib.contextmanager
    def lease(self,owner):
        check(isinstance(owner,str) and re.fullmatch('[A-Za-z0-9_-]{1,64}',owner),'fleet_owner')
        p=path(self.root/'WRITER.lease');token=uuid.uuid4().hex
        # Advisory single-writer convention. Mount exclusive-create는 분산 lock 보장이 아니다.
        with p.open('xb') as f:f.write(canonical({'owner':owner,'token':token,'created':time.time()}))
        try:yield token
        finally:
            check(json.loads(p.read_bytes())['token']==token,'fleet_lease_changed');p.unlink()
