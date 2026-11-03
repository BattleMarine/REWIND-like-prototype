"""한정 tar scan과 bounded seek pack. 전체 index/catalog RAM 조립 없음."""
import hashlib
import json
import sqlite3
import struct
import tarfile
import zlib
import sys
from pathlib import Path
from rewind_archive.contracts import canonical,safe_path
from rewind_archive.extraction import pax_fields
from rewind_archive_index.source import IndexFailure,_validate_source
from .window import check,open_engine,export,resume
from .storage import path

def initial(source):
    return {'source_sha256':hashlib.sha256(canonical(source)).hexdigest(),'cursor':0,'boundary':0,'ordinal':0,
        'crc32':0,'tar_complete':False,'eof':False,'trailer_crc_match':False,'complete':False,'status':'partial'}

def cache(store,head,workspace,key,*,cold_rebuild=False):
    workspace=path(workspace)
    check(not str(workspace).replace('\\','/').startswith('/content/drive/') and not workspace.is_relative_to(store.root),'fleet_live_cache_local_only')
    workspace.mkdir(parents=True,exist_ok=True);db=sqlite3.connect(workspace/(key+'.sqlite'))
    db.execute('PRAGMA synchronous=FULL')
    db.execute('CREATE TABLE IF NOT EXISTS rows(name TEXT PRIMARY KEY,ordinal INTEGER UNIQUE,row BLOB,commit_key TEXT)')
    db.execute('CREATE TABLE IF NOT EXISTS meta(head TEXT)')
    db.execute('CREATE TABLE IF NOT EXISTS history(n INTEGER PRIMARY KEY,k TEXT)')
    db.execute('CREATE TABLE IF NOT EXISTS restoration(target TEXT,previous TEXT,k TEXT,n INTEGER,phase TEXT,last_n INTEGER)')
    current=db.execute('SELECT head FROM meta').fetchone()
    if (current[0] if current else None)!=head:
        try:
            previous=current[0] if current else None
            if cold_rebuild:
                db.execute('DELETE FROM rows');db.execute('DELETE FROM history');db.execute('DELETE FROM restoration');previous=None
            pending=db.execute('SELECT target,previous,k,n,phase,last_n FROM restoration').fetchone()
            if pending:
                check(pending[0]==head and pending[1]==previous,'fleet_restore_target_changed');_,_,k,n,phase,last_n=pending
            else:
                db.execute('DELETE FROM history');k=head;n=0;phase='walk';last_n=None
                db.execute('INSERT INTO restoration VALUES(?,?,?,?,?,?)',(head,previous,k,n,phase,last_n));db.commit()
            started_io=store.io_bytes
            while k and k!=previous:
                if phase!='walk':break
                store.guard();doc=store.committed(k);check(doc['schema']=='fleet_archive_commit_v1','fleet_archive_schema')
                db.execute('INSERT INTO history VALUES(?,?)',(n,k));k=doc['parent'];n+=1
                db.execute('UPDATE restoration SET k=?,n=?',(k,n));db.commit()
                if store.io_bytes-started_io>=4*1024**2:db.close();return None
            if phase=='walk':
                check(k==previous,'fleet_cache_nonancestor_requires_rebuild');phase='apply';last_n=n
                db.execute('UPDATE restoration SET phase=?,last_n=?',(phase,last_n));db.commit()
            for order,k in db.execute('SELECT n,k FROM history WHERE n<? ORDER BY n DESC',(last_n,)):
                doc=store.committed(k)
                for row in store.document(doc['catalog'])['rows']:
                    db.execute('INSERT INTO rows VALUES(?,?,?,?)',(row['name'].casefold(),row['ordinal'],canonical(row),k))
                db.execute('UPDATE restoration SET last_n=?',(order,));db.commit()
                if store.io_bytes-started_io>=4*1024**2:db.close();return None
            db.execute('DELETE FROM meta');db.execute('INSERT INTO meta VALUES(?)',(head,));db.execute('DELETE FROM restoration');db.execute('DELETE FROM history');db.commit()
        except BaseException:db.rollback();db.close();raise
    return db

def scan(state,pack,source,budget,spacing,db,max_members=64):
    check(hashlib.sha256(canonical(source.spec)).hexdigest()==state['source_sha256'],'fleet_source_identity')
    state=dict(state);rows=[];before=state['cursor'];gz=None;reason=None;primary=None;handled=False
    def engine_call(call):
        # C 확장에 감싸진 이번 호출의 source 오류만 복원한다.
        budget.last_source_failure=None
        try:return call()
        except Exception as exc:
            if budget.last_source_failure:raise IndexFailure(budget.last_source_failure) from exc
            raise
    try:
        gz=engine_call(lambda:open_engine(source,spacing,pack,state['cursor']));crc=state['crc32']
        def read(n):
            nonlocal crc
            budget.check();check(budget.decoded_bytes+n<=budget.limits.max_decoded_bytes,'decoded_bytes_limit')
            raw=engine_call(lambda:gz.read(n));budget.decoded_bytes+=len(raw);crc=zlib.crc32(raw,crc);budget.check();return raw
        def exact(n):
            raw=read(n)
            check(len(raw)==n,'fleet_incomplete_source');return raw
        def checkpoint():state.update(cursor=gz.tell(),crc32=crc)
        def tail():
            while True:
                raw=read(65536)
                check(not raw.strip(b'\0'),'fleet_nonzero_tar_tail')
                checkpoint()
                if not raw:break
            source.seek(source.size-8);raw=source.read(8);check(len(raw)==8,'fleet_incomplete_source')
            expected_crc,isize=struct.unpack('<II',raw)
            check(expected_crc==crc and isize==state['cursor']%(2**32),'fleet_trailer_crc_or_size')
            state.update(eof=True,trailer_crc_match=True,status='catalog_complete_crc_pending')
        if state['tar_complete']:
            if not state['eof']:tail()
        else:
            while len(rows)<max_members:
                chain=state['boundary'];pending=None;pax={}
                while True:
                    header=exact(512)
                    if header==bytes(512):
                        check(pending is None and not pax and exact(512)==bytes(512),'fleet_tar_end')
                        state.update(tar_complete=True,tar_end=chain+1024);checkpoint();tail();break
                    try:member=tarfile.TarInfo.frombuf(header,'utf-8','strict')
                    except (tarfile.HeaderError,ValueError,UnicodeError):raise IndexFailure('fleet_tar_header') from None
                    if member.type in (tarfile.GNUTYPE_LONGNAME,tarfile.XHDTYPE):
                        check(0<=member.size<=65536,'fleet_tar_metadata_limit');raw=exact(member.size)
                        check(not exact((-member.size)%512).strip(b'\0'),'fleet_tar_padding')
                        if member.type==tarfile.GNUTYPE_LONGNAME:
                            check(pending is None and raw.endswith(b'\0') and b'\0' not in raw[:-1],'fleet_tar_longname');pending=raw[:-1].decode('utf8')
                        else:check(not pax,'fleet_duplicate_pax');pax=pax_fields(raw)
                        continue
                    check(member.type in (tarfile.REGTYPE,tarfile.AREGTYPE,tarfile.DIRTYPE),'fleet_tar_type')
                    name=safe_path(pax.get('path',pending or member.name));size=int(pax.get('size',member.size))
                    check(0<=size<=budget.limits.max_file_bytes and (not member.isdir() or size==0),'fleet_tar_size')
                    check(not db.execute('SELECT 1 FROM rows WHERE name=?',(name.casefold(),)).fetchone() and not any(r['name'].casefold()==name.casefold() for r in rows),'fleet_duplicate_name')
                    at=gz.tell();h=hashlib.sha256();left=size
                    while left:
                        raw=exact(min(left,65536));h.update(raw);left-=len(raw)
                    check(not exact((-size)%512).strip(b'\0'),'fleet_tar_padding')
                    row={'ordinal':state['ordinal'],'chain_start':chain,'name':name,'offset':at,'size':size,'end':gz.tell(),
                         'sha256':h.hexdigest(),'type':'directory' if member.isdir() else 'file'}
                    rows.append(row);state.update(ordinal=state['ordinal']+1,boundary=gz.tell());checkpoint();break
                if state['tar_complete']:break
    except BaseException as exc:
        primary=exc
        reason=exc.reason if isinstance(exc,IndexFailure) else ('fleet_incomplete_source' if isinstance(exc,EOFError) else None)
        if reason not in ('read_bytes_limit','decoded_bytes_limit','elapsed_limit','transfer_bytes_limit','ranges_limit','fleet_incomplete_source'):raise
        handled=True
        state['status']='incomplete_source' if reason=='fleet_incomplete_source' else 'partial'
    finally:
        try:
            newpack=(pack if primary is not None and (not handled or state['cursor']==before) else export(gz)) if gz else None
            if gz:gz.close();gz=None
            if newpack and (primary is None or handled) and not (handled and state['cursor']==before):
                from .window import decode
                h,points=decode(newpack,allow_empty=True)
                if not points or max(h[1],points[-1][0][1])<state['cursor']:
                    budget.active_read_limit=budget.limits.max_read_bytes
                    if not points and 0<h[1]<=spacing:
                        gz=engine_call(lambda:open_engine(source,spacing,None,0));engine_call(gz.build_full_index)
                    else:gz=engine_call(lambda:open_engine(source,spacing,newpack,state['cursor']))
                    newpack=export(gz)
        except BaseException as secondary:
            secondary.fleet_checkpoint_failure=True
            if primary is not None:
                secondary.fleet_scan_primary=primary
                if not handled:
                    primary.fleet_checkpoint_error=secondary
                    raise primary
                raise secondary from primary
            raise
        finally:
            active=sys.exc_info()[1];cleanup=[]
            for resource in (gz,source):
                if resource is not None:
                    try:resource.close()
                    except BaseException as exc:cleanup.append(exc)
            if cleanup:
                original=active or primary
                if original is not None:
                    original.fleet_cleanup_errors=cleanup
                    if active is None:raise original
                else:
                    cleanup[0].fleet_cleanup_errors=cleanup[1:]
                    raise cleanup[0]
    if handled and state['cursor']==before:
        return state,rows,pack,{'reason':reason,'decoded_advanced':0,'stats':budget.stats()}
    check(newpack is not None,'fleet_window_missing')
    from .window import decode
    h,points=decode(newpack);coverage=max(h[1],points[-1][0][1])
    check(not rows or rows[-1]['end']<=coverage,'fleet_window_coverage_pending')
    selected,start=resume(newpack,state['cursor'])
    check(start>0 or state['cursor']<4*spacing,'fleet_restart_gap_unbounded')
    state['complete']=False
    return state,rows,newpack,{'reason':reason,'decoded_advanced':state['cursor']-before,'stats':budget.stats()}

def lookup(store,head,workspace,key,name):
    db=cache(store,head,workspace,key)
    check(db is not None,'fleet_catalog_restore_progress_retry_with_budget')
    try:
        found=db.execute('SELECT row,commit_key FROM rows WHERE name=?',(safe_path(name).casefold(),)).fetchone()
        check(found is not None,'fleet_member_not_indexed');return json.loads(found[0]),store.committed(found[1])
    finally:db.close()

def extract(store,head,workspace,key,name,source,budget,spacing):
    row,commit=lookup(store,head,workspace,key,name)
    check(commit['state']['source_sha256']==hashlib.sha256(canonical(source.spec)).hexdigest(),'fleet_source_identity')
    check(commit['start_cursor']<=row['offset'] and row['offset']+row['size']<=commit['state']['cursor'],'fleet_requested_window_coverage')
    from .window import decode
    import indexed_gzip,io
    check(indexed_gzip.__version__=='1.10.3','fleet_engine_version')
    pack=store.blob(commit['pack']);header,points=decode(pack)
    coverage=max(header[1],points[-1][0][1])
    check(row['offset']+row['size']<=coverage,'fleet_requested_window_not_indexed')
    gz=indexed_gzip.IndexedGzipFile(fileobj=source,spacing=spacing,buffer_size=4*spacing,readbuf_size=65536,auto_build=False)
    gz.import_index(fileobj=io.BytesIO(pack));gz.seek(row['offset']);h=hashlib.sha256();data=bytearray()
    try:
        while len(data)<row['size']:
            budget.check();check(budget.decoded_bytes<budget.limits.max_decoded_bytes,'decoded_bytes_limit')
            raw=gz.read(min(65536,row['size']-len(data),budget.limits.max_decoded_bytes-budget.decoded_bytes))
            budget.decoded_bytes+=len(raw);check(bool(raw),'fleet_incomplete_source');data.extend(raw);h.update(raw)
        check(h.hexdigest()==row['sha256'],'fleet_extraction_hash');return bytes(data)
    finally:gz.close();source.close()
