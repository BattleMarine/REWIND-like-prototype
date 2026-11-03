"""닫힌 DB를 새 local 복사본으로 compact한다. 공식 resume head로 승격하지 않는다."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import sqlite3
import time

from rewind_archive.contracts import canonical,digest
from .core import ArchiveIndex
from .source import IndexLimits,IndexFailure,MIB,SCHEMA
from .snapshots import _path

INPUT_LIMIT=64*MIB
DISK_LIMIT=256*MIB
SECONDS=60.


def compact_copy(source,output,*,expected_sha256):
    source,output=_path(source),_path(output)
    if output.exists() or source.is_relative_to(output) or not re.fullmatch('[a-f0-9]{64}',expected_sha256):
        raise IndexFailure('compact_new_output_and_explicit_pin_required')
    output.mkdir(parents=True,exist_ok=False)
    started=time.monotonic();reason=None
    report={'schema':'bounded_local_db_compaction_v1','status':'started','official_resume_head':False,
        'source_sha256_expected':expected_sha256,'remote_requests':0,'new_source_coverage':False,
        'limits':{'input_bytes':INPUT_LIMIT,'source_plus_output_bytes':DISK_LIMIT,'seconds':SECONDS},
        'sqlite_version':sqlite3.sqlite_version,'partial_preserved':False,
        'guard_scope':'cooperative Python/SQLite progress checks; source closed, sidecars forbidden; output-directory disk inventory',
        'sqlite_temp_store':'MEMORY','sqlite_cache_kib':8192}
    def source_size():
        try:return source.stat().st_size
        except OSError:return None
    def closed_source():
        for suffix in ('-wal','-shm','-journal'):
            if source.with_name(source.name+suffix).exists():raise IndexFailure('compact_active_sidecar')
    def guard():
        nonlocal reason
        if time.monotonic()-started>SECONDS:reason='compact_time_limit'
        closed_source()
        files=list(output.rglob('*'));size=source_size()
        if size is None:raise IndexFailure('compact_source_missing')
        if len(files)>32 or size+sum(p.stat().st_size for p in files if p.is_file())>DISK_LIMIT:
            reason='compact_disk_limit'
        if reason:raise IndexFailure(reason)
    def sha_file(path):
        h=hashlib.sha256()
        with path.open('rb') as stream:
            while data:=stream.read(1024**2):guard();h.update(data)
        return h.hexdigest()
    def progress():
        try:guard();return 0
        except IndexFailure:return 1
    def connect(path):
        db=sqlite3.connect(path.as_uri()+'?mode=ro',uri=True,timeout=0)
        db.execute('PRAGMA temp_store=MEMORY')
        db.execute('PRAGMA cache_size=-8192')
        db.set_progress_handler(progress,1024)
        return db
    def rows(db):
        row=db.execute('SELECT state,state_sha,gzidx,gzidx_sha FROM checkpoint WHERE id=1').fetchone()
        if row is None or len(row[0])>16*MIB or len(row[2])>32*MIB:
            raise IndexFailure('compact_checkpoint_limits')
        if hashlib.sha256(row[0]).hexdigest()!=row[1] or hashlib.sha256(row[2]).hexdigest()!=row[3]:
            raise IndexFailure('compact_checkpoint_hash')
        state=json.loads(row[0])
        if state.get('schema')!=SCHEMA or digest(state['source'])!=state['source_sha256']:
            raise IndexFailure('compact_checkpoint_identity')
        ArchiveIndex(source,IndexLimits())._validate_state(state)
        present=db.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='snapshot_anchor'").fetchone()
        anchor=db.execute('SELECT receipt FROM snapshot_anchor WHERE id=1').fetchone() if present else None
        return row,anchor,state
    partial=output/'compact.partial.sqlite';locker=None
    try:
        if not source.is_file() or source.stat().st_size>INPUT_LIMIT:raise IndexFailure('compact_source_limit')
        report['source_bytes']=source.stat().st_size
        closed_source()
        if sha_file(source)!=expected_sha256:raise IndexFailure('compact_source_pin')
        guard()
        # 별도 read transaction의 SHARED lock을 전체 검증 동안 유지한다.
        # VACUUM은 이 transaction과 다른 readonly connection에서 실행한다.
        locker=connect(source);locker.execute('BEGIN');rows(locker)
        report['source_lock']='shared read transaction; no WAL/journal source accepted'
        if sha_file(source)!=expected_sha256:raise IndexFailure('compact_source_changed')
        db=connect(source)
        try:
            before,anchor,state=rows(db)
            report.update(generation=state['generation'],members=len(state['members']),source_fingerprint=state['source_sha256'],
                freelist_before=db.execute('PRAGMA freelist_count').fetchone()[0])
            db.execute('VACUUM INTO ?',(str(partial),))
        finally:db.close()
        guard()
        copy=connect(partial)
        try:
            if copy.execute('PRAGMA integrity_check').fetchall()!=[('ok',)]:raise IndexFailure('compact_sqlite_integrity')
            after,after_anchor,after_state=rows(copy)
            if after!=before or after_anchor!=anchor or after_state!=state:raise IndexFailure('compact_semantics_changed')
            report['freelist_after']=copy.execute('PRAGMA freelist_count').fetchone()[0]
        finally:copy.close()
        if sha_file(source)!=expected_sha256:raise IndexFailure('compact_source_changed')
        report.update(source_sha256_after=expected_sha256,source_unchanged=True,
            checkpoint_state_blob_anchor_bytes_identical=True,sqlite_integrity='passed',output_sha256=sha_file(partial),
            output_bytes=partial.stat().st_size)
        final=output/'compact.sqlite'
        if final.exists():raise IndexFailure('compact_output_exists')
        partial.rename(final)
        report.update(status='verified_local_copy_not_resume_head',output_file='compact.sqlite')
    except BaseException as exc:
        report.update(status='failed_or_interrupted',error_type=type(exc).__name__,
            reason=reason or (exc.reason if isinstance(exc,IndexFailure) else 'compact_operation_failed'),partial_preserved=partial.exists())
        raise
    finally:
        if locker is not None:locker.close()
        report['elapsed_seconds']=time.monotonic()-started
        size=source_size()
        report['observed_source_plus_output_bytes']=(size+sum(p.stat().st_size for p in output.rglob('*') if p.is_file())) if size is not None else None
        (output/'report.json').write_bytes(canonical(report)+b'\n')
    return report


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-db',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--expected-sha256',required=True)
    args=parser.parse_args(argv)
    print(json.dumps(compact_copy(args.input_db,args.output,expected_sha256=args.expected_sha256),ensure_ascii=False,indent=2))


if __name__=='__main__':main()
