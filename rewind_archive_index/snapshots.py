"""닫힌 SQLite snapshot + exact receipt. Mount sync의 원자성/전역 최신판정은 가정하지 않는다."""
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import shutil
import sqlite3
import tempfile
import time
import uuid
import math
import struct
from zipfile import ZipFile,ZIP_STORED

from .core import ArchiveIndex
from .source import IndexFailure,IndexLimits,canonical,digest,MIB,CHUNK

FORMAT='persistent_archive_snapshot_v1'
FIELDS={'schema','snapshot_id','filename','bytes','source_sha256','generation','db_sha256','lineage','parent_snapshot_id'}


@dataclass(frozen=True)
class SnapshotLimits:
    max_db_bytes:int=64*MIB
    max_bundle_bytes:int=72*MIB
    max_io_bytes:int=512*MIB
    max_disk_bytes:int=256*MIB
    max_seconds:float=60.

    def __post_init__(self):
        for key,value in vars(self).items():
            if key=='max_seconds':
                if type(value) not in (int,float) or not math.isfinite(value) or value<=0:raise IndexFailure('snapshot_limits')
            elif type(value) is not int or value<1:raise IndexFailure('snapshot_limits')


class IO:
    def __init__(self,limits):self.limits=limits;self.start=time.monotonic();self.bytes=0
    def check(self,n=0):
        self.bytes+=n
        if self.bytes>self.limits.max_io_bytes:raise IndexFailure('snapshot_io_limit')
        if time.monotonic()-self.start>self.limits.max_seconds:raise IndexFailure('snapshot_elapsed_limit')
    def copy(self,src,dst):
        while data:=src.read(CHUNK):self.check(len(data));dst.write(data)
    def hash(self,path):
        h=hashlib.sha256()
        with path.open('rb') as f:
            while data:=f.read(CHUNK):self.check(len(data));h.update(data)
        return h.hexdigest()


def _path(path):
    p=Path(path).absolute()
    if p.is_symlink() or any(x.is_symlink() or getattr(x,'is_junction',lambda:False)() for x in p.parents):raise IndexFailure('snapshot_link')
    return p


def _local_db(path):
    p=_path(path)
    if str(p).replace('\\','/').startswith('/content/drive/'):
        raise IndexFailure('live_db_must_be_local')
    return p


def _publish_file(temp,target):
    if os.name=='nt':os.rename(temp,target)
    else:os.link(temp,target);temp.unlink()


def _receipt(value):
    if not isinstance(value,dict) or set(value)!=FIELDS or value['schema']!=FORMAT:raise IndexFailure('snapshot_receipt_schema')
    import re
    for key in ('snapshot_id','source_sha256','db_sha256'):
        if not isinstance(value[key],str) or not re.fullmatch('[a-f0-9]{64}',value[key]):raise IndexFailure('snapshot_receipt_hash')
    if value['parent_snapshot_id'] is not None and (not isinstance(value['parent_snapshot_id'],str) or not re.fullmatch('[a-f0-9]{64}',value['parent_snapshot_id'])):raise IndexFailure('snapshot_parent')
    if not isinstance(value['lineage'],str) or not re.fullmatch('[a-f0-9]{32}',value['lineage']):raise IndexFailure('snapshot_lineage')
    if type(value['generation']) is not int or value['generation']<0 or type(value['bytes']) is not int or value['bytes']<1:raise IndexFailure('snapshot_generation')
    if not isinstance(value['filename'],str) or value['filename']!=f"g{value['generation']}-{value['snapshot_id']}.zip":raise IndexFailure('snapshot_filename')
    return value


def read_receipt(path):
    p=_path(path)
    with p.open('rb') as f:raw=f.read(65537)
    if len(raw)>65536:raise IndexFailure('snapshot_receipt_limit')
    return _receipt(json.loads(raw))


def _anchor(index):
    db=index._connect()
    try:
        if not db.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='snapshot_anchor'").fetchone():return None
        row=db.execute('SELECT receipt FROM snapshot_anchor WHERE id=1').fetchone()
        return _receipt(json.loads(row[0])) if row else None
    finally:db.close()


def _save_anchor(index,receipt,expected_generation,prior=None):
    db=index._connect()
    try:
        db.execute('BEGIN IMMEDIATE');state,_=index._load(db)
        if state['generation']!=expected_generation:raise IndexFailure('snapshot_generation_changed')
        db.execute('CREATE TABLE IF NOT EXISTS snapshot_anchor (id INTEGER PRIMARY KEY CHECK(id=1), receipt BLOB NOT NULL)')
        row=db.execute('SELECT receipt FROM snapshot_anchor WHERE id=1').fetchone()
        current=_receipt(json.loads(row[0])) if row else None
        if current!=prior:raise IndexFailure('snapshot_anchor_changed')
        db.execute('INSERT OR REPLACE INTO snapshot_anchor VALUES(1,?)',(canonical(receipt),));db.commit()
    except BaseException:db.rollback();raise
    finally:db.close()


def _check_anchor(receipt,known):
    if known is None:return
    _receipt(known)
    if receipt['source_sha256']!=known['source_sha256'] or receipt['lineage']!=known['lineage']:raise IndexFailure('snapshot_branch_conflict')
    if receipt['generation']<known['generation']:raise IndexFailure('snapshot_stale')
    if receipt['generation']==known['generation'] and receipt['snapshot_id']!=known['snapshot_id']:raise IndexFailure('snapshot_generation_conflict')
    if receipt['generation']>known['generation'] and receipt['parent_snapshot_id']!=known['snapshot_id']:raise IndexFailure('snapshot_parent_conflict')


def export_snapshot(index,directory,known_anchor=None,limits=None):
    """local/mounted directory에 불변 파일을 게시한다. 원본 archive를 읽지 않는다."""
    limits=limits or SnapshotLimits();io=IO(limits);directory=_path(directory);directory.mkdir(parents=True,exist_ok=True)
    _local_db(index.path)
    prior=_anchor(index)
    if known_anchor is not None and prior!=_receipt(known_anchor):raise IndexFailure('snapshot_anchor_conflict')
    if index.path.stat().st_size>limits.max_db_bytes:raise IndexFailure('snapshot_db_limit')
    if 3*index.path.stat().st_size+65536>limits.max_disk_bytes:raise IndexFailure('snapshot_disk_limit')
    # staging SQLite는 항상 live index 옆 로컬 filesystem에 둔다. Mount에는 닫힌 ZIP만 쓴다.
    with tempfile.TemporaryDirectory(prefix='.snapshot-',dir=index.path.parent) as tempdir:
        temp=Path(tempdir);dbpath=temp/'index.sqlite'
        source=index._connect();dest=sqlite3.connect(dbpath)
        try:
            source.backup(dest,pages=64,progress=lambda status,left,total:io.check(64*4096))
        finally:dest.close();source.close()
        if dbpath.stat().st_size>limits.max_db_bytes:raise IndexFailure('snapshot_db_limit')
        backup=ArchiveIndex(dbpath,index.limits);state,blob=backup._snapshot()
        if prior:
            if prior['source_sha256']!=state['source_sha256']:raise IndexFailure('snapshot_source_conflict')
            if state['generation']<=prior['generation']:raise IndexFailure('snapshot_not_advanced')
        manifest={'schema':FORMAT,'source_sha256':state['source_sha256'],'generation':state['generation'],
            'db_sha256':io.hash(dbpath),'lineage':prior['lineage'] if prior else uuid.uuid4().hex,
            'parent_snapshot_id':prior['snapshot_id'] if prior else None,'checkpoint_schema':state['schema'],
            'catalog_sha256':state['catalog_sha256'],'index_sha256':hashlib.sha256(blob).hexdigest(),
            'engine_version':state['engine_version']}
        bundle=temp/'snapshot.zip'
        with ZipFile(bundle,'x',ZIP_STORED) as z:
            with dbpath.open('rb') as src,z.open('index.sqlite','w') as dst:io.copy(src,dst)
            z.writestr('manifest.json',canonical(manifest))
        if bundle.stat().st_size>limits.max_bundle_bytes:raise IndexFailure('snapshot_bundle_limit')
        snapshot_id=io.hash(bundle);filename=f"g{state['generation']}-{snapshot_id}.zip"
        receipt={k:manifest[k] for k in ('schema','source_sha256','generation','db_sha256','lineage','parent_snapshot_id')}
        receipt.update(snapshot_id=snapshot_id,filename=filename,bytes=bundle.stat().st_size)
        target=directory/filename;receipt_path=directory/(filename+'.receipt.json')
        if target.exists() or receipt_path.exists():raise IndexFailure('snapshot_path_reuse')
        # Mount의rename이원자적이라는가정없음: reader는receipt+전체hash일치이후만사용한다.
        with target.open('xb') as f,bundle.open('rb') as src:io.copy(src,f)
        if io.hash(target)!=snapshot_id:raise IndexFailure('snapshot_publish_hash')
        with receipt_path.open('xb') as f:f.write(canonical(receipt)+b'\n')
        _save_anchor(index,receipt,state['generation'],prior)
    return {'receipt':receipt,'receipt_path':str(receipt_path),'io_bytes':io.bytes,'seconds':time.monotonic()-io.start,'archive_network_requested':False}


def import_snapshot(receipt_path,target,*,expected_source_sha256,expected_snapshot_id,min_generation,known_anchor=None,limits=None,index_limits=None):
    """정확히 지정한receipt와신뢰anchor만검증한다. latest 자동선택/병합은없다."""
    limits=limits or SnapshotLimits();io=IO(limits);receipt_path=_path(receipt_path);receipt=read_receipt(receipt_path)
    if receipt['source_sha256']!=expected_source_sha256 or receipt['snapshot_id']!=expected_snapshot_id:raise IndexFailure('snapshot_expected_identity')
    if type(min_generation) is not int or min_generation<0 or receipt['generation']<min_generation:raise IndexFailure('snapshot_stale')
    _check_anchor(receipt,known_anchor)
    target=_local_db(target)
    if target.exists():raise IndexFailure('snapshot_target_exists')
    bundle=_path(receipt_path.parent/receipt['filename'])
    if bundle.stat().st_size!=receipt['bytes'] or bundle.stat().st_size>limits.max_bundle_bytes:raise IndexFailure('snapshot_bundle_size')
    if io.hash(bundle)!=receipt['snapshot_id']:raise IndexFailure('snapshot_bundle_hash')
    # ZipFile central directory를 할당하기 전에 정확한2member/작은directory형식만허용한다.
    with bundle.open('rb') as f:
        f.seek(-22,2);tail=f.read(22)
    if len(tail)!=22:raise IndexFailure('snapshot_zip_directory')
    magic,disk,cdisk,n1,n2,length,offset,comment=struct.unpack('<4s4H2LH',tail)
    if magic!=b'PK\x05\x06' or disk or cdisk or n1!=2 or n2!=2 or length>4096 or comment or offset+length!=bundle.stat().st_size-22:
        raise IndexFailure('snapshot_zip_directory')
    target.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.restore-',dir=target.parent) as tempdir:
        dbpath=Path(tempdir)/'index.sqlite'
        with ZipFile(bundle) as z:
            entries=z.infolist()
            if len(entries)!=2 or {p.filename for p in entries}!={'index.sqlite','manifest.json'}:raise IndexFailure('snapshot_zip_members')
            if any(p.compress_type!=ZIP_STORED or p.flag_bits&1 or p.extra or p.comment for p in entries):raise IndexFailure('snapshot_zip_encoding')
            if z.getinfo('index.sqlite').file_size>limits.max_db_bytes or z.getinfo('manifest.json').file_size>65536:raise IndexFailure('snapshot_zip_size')
            if 3*z.getinfo('index.sqlite').file_size+bundle.stat().st_size>limits.max_disk_bytes:raise IndexFailure('snapshot_disk_limit')
            manifest=json.loads(z.read('manifest.json'))
            for key in ('schema','source_sha256','generation','db_sha256','lineage','parent_snapshot_id'):
                if manifest.get(key)!=receipt[key]:raise IndexFailure('snapshot_manifest_receipt_mix')
            with z.open('index.sqlite') as src,dbpath.open('xb') as dst:io.copy(src,dst)
        if io.hash(dbpath)!=receipt['db_sha256']:raise IndexFailure('snapshot_db_hash')
        restored=ArchiveIndex(dbpath,index_limits);state,blob=restored._snapshot()
        if state['source_sha256']!=receipt['source_sha256'] or state['generation']!=receipt['generation'] or state['schema']!=manifest.get('checkpoint_schema'):
            raise IndexFailure('snapshot_checkpoint_identity')
        if state['catalog_sha256']!=manifest.get('catalog_sha256') or hashlib.sha256(blob).hexdigest()!=manifest.get('index_sha256') or state['engine_version']!=manifest.get('engine_version'):
            raise IndexFailure('snapshot_checkpoint_hash_or_version')
        if blob and state['engine_version']!=ArchiveIndex._engine_version():raise IndexFailure('index_engine_version_changed')
        prior=_anchor(restored);_save_anchor(restored,receipt,state['generation'],prior)
        _publish_file(dbpath,target)
    return {'index_path':str(target),'receipt':receipt,'io_bytes':io.bytes,'seconds':time.monotonic()-io.start,'archive_network_requested':False}
