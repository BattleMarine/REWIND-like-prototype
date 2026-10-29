"""SQLite atomic checkpoint와 연속 tar catalog. 원격 source는 지원하지 않는다."""
import copy
import hashlib
import io
import json
import os
from pathlib import Path
import re
import sqlite3
import tarfile
import tempfile
import zlib
from .source import (SCHEMA,MIB,CHUNK,IndexFailure,IndexLimits,Budget,SplitSource,
    make_source,_validate_source,_gzip,_read,_pack_index,canonical,digest,safe_path,pax_fields)

class ArchiveIndex:
    def __init__(self, path, limits=None, runtime=None):
        self.path = Path(path).absolute()
        self.limits = limits or IndexLimits()
        self.runtime=runtime
        if self.path.is_symlink() or any(p.is_symlink() or getattr(p, 'is_junction', lambda: False)() for p in self.path.parents):
            raise IndexFailure('index_path_link')

    def _source(self,spec,budget):
        if spec['schema']=='ordered_drive_gzip_source_v1':
            if self.runtime is None:raise IndexFailure('drive_runtime_required')
            return self.runtime.source(spec,budget)
        return SplitSource(spec,budget)

    def _finish_source(self,spec,budget):
        _validate_source(spec)
        if spec['schema']=='ordered_drive_gzip_source_v1':self.runtime.finish(budget)

    def _result(self,budget,**values):
        stats=budget.stats()
        return {**self.status(),**values,'stats':stats,'network_requested':stats['network_requested']}

    def _connect(self):
        if not self.path.is_file() or self.path.stat().st_size > self.limits.max_disk_bytes:
            raise IndexFailure('index_missing_or_disk_limit')
        db = sqlite3.connect(str(self.path), timeout=0)
        db.setlimit(sqlite3.SQLITE_LIMIT_LENGTH, self.limits.max_state_bytes+self.limits.max_index_bytes+4096)
        db.execute('PRAGMA synchronous=FULL')
        return db

    def create(self, source, spacing=MIB):
        if type(spacing) is not int or not CHUNK <= spacing <= 16*MIB:
            raise IndexFailure('spacing_limit')
        _validate_source(source)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        state = {'schema': SCHEMA, 'source': copy.deepcopy(source), 'source_sha256': digest(source),
            'generation': 0, 'spacing': spacing, 'members': [], 'boundary': 0,
            'catalog_sha256':digest([]), 'engine_version':None,
            'index_coverage': 0, 'catalog_complete': False, 'tar_end': None,
            'gzip_crc_valid': False, 'complete': False}
        if 3*(len(canonical(state))+16384)>self.limits.max_disk_bytes:
            raise IndexFailure('checkpoint_disk_limit')
        if len(canonical(state))>self.limits.max_state_bytes:raise IndexFailure('checkpoint_size_limit')
        fd, name = tempfile.mkstemp(prefix='.index-', suffix='.sqlite', dir=self.path.parent); os.close(fd)
        temp = Path(name)
        try:
            db=sqlite3.connect(name)
            try:
                db.execute('PRAGMA synchronous=FULL')
                db.execute('CREATE TABLE checkpoint (id INTEGER PRIMARY KEY CHECK(id=1), state BLOB NOT NULL, state_sha TEXT NOT NULL, gzidx BLOB NOT NULL, gzidx_sha TEXT NOT NULL)')
                raw = canonical(state)
                db.execute('INSERT INTO checkpoint VALUES(1,?,?,?,?)', (raw, hashlib.sha256(raw).hexdigest(), b'', hashlib.sha256(b'').hexdigest()))
                db.commit()
            finally:db.close()
            # Windows rename은 기존 destination을 거부한다. POSIX에서는 no-clobber link를 쓴다.
            if os.name=='nt':os.rename(temp,self.path)
            else:os.link(temp,self.path)
        finally:
            temp.unlink(missing_ok=True)
        return self.status()

    def _load(self, db):
        row = db.execute('SELECT state,state_sha,gzidx,gzidx_sha FROM checkpoint WHERE id=1').fetchone()
        if row is None: raise IndexFailure('checkpoint_missing')
        raw, sh, blob, bh = row
        if len(raw) > self.limits.max_state_bytes or len(blob) > self.limits.max_index_bytes:
            raise IndexFailure('checkpoint_size_limit')
        if hashlib.sha256(raw).hexdigest() != sh or hashlib.sha256(blob).hexdigest() != bh:
            raise IndexFailure('checkpoint_checksum')
        state = json.loads(raw)
        if state.get('schema') != SCHEMA or digest(state['source']) != state['source_sha256']:
            raise IndexFailure('checkpoint_schema_or_source_checksum')
        _validate_source(state['source'])
        self._validate_state(state)
        return state, blob

    def _validate_state(self, state):
        cursor = 0; seen = set()
        if state.get('catalog_sha256') != digest(state['members']):
            raise IndexFailure('catalog_checksum')
        if type(state['generation']) is not int or state['generation'] < 0 or len(state['members']) > self.limits.max_members:
            raise IndexFailure('checkpoint_generation_or_members')
        for ordinal, row in enumerate(state['members']):
            name = safe_path(row['name'])
            if row['ordinal'] != ordinal or row['chain_start'] != cursor or row['end'] <= cursor or name.casefold() in seen:
                raise IndexFailure('catalog_order_gap_or_duplicate')
            if row['offset'] < cursor+512 or row['end'] != row['offset']+row['size']+(-row['size'])%512:
                raise IndexFailure('catalog_boundary')
            if row['type'] not in ('file','directory') or type(row['size']) is not int or not 0 <= row['size'] <= self.limits.max_file_bytes:
                raise IndexFailure('catalog_type_or_size')
            if not re.fullmatch('[0-9a-f]{64}', row['sha256']): raise IndexFailure('catalog_hash')
            seen.add(name.casefold()); cursor = row['end']
        if cursor != state['boundary']:
            raise IndexFailure('catalog_noncontiguous')
        if state['complete'] != (state['catalog_complete'] and state['gzip_crc_valid']):
            raise IndexFailure('completion_contract')
        if state['catalog_complete'] and state['tar_end'] != cursor+1024:
            raise IndexFailure('tar_end_boundary')

    def _commit(self, expected, state, blob):
        state = copy.deepcopy(state)
        state['catalog_sha256'] = digest(state['members'])
        self._validate_state(state)
        raw = canonical(state)
        if len(raw) > self.limits.max_state_bytes or len(blob) > self.limits.max_index_bytes:
            raise IndexFailure('checkpoint_size_limit')
        # 이전DB + rollback journal + 새페이지가 동시에 존재하는 보수적 예약.
        estimate = 3*(self.path.stat().st_size+len(raw)+len(blob)+16384)
        if estimate > self.limits.max_disk_bytes: raise IndexFailure('checkpoint_disk_limit')
        db = self._connect()
        try:
            db.execute('BEGIN IMMEDIATE')
            old, _ = self._load(db)
            if old['generation'] != expected: raise IndexFailure('stale_generation')
            if state['source_sha256'] != old['source_sha256'] or state['spacing'] != old['spacing']:
                raise IndexFailure('source_or_policy_conflict')
            if state['members'][:len(old['members'])] != old['members'] or state['boundary'] < old['boundary'] or state['index_coverage'] < old['index_coverage']:
                raise IndexFailure('progress_regression_or_conflict')
            if old['catalog_complete'] and not state['catalog_complete'] or old['complete'] and not state['complete']:
                raise IndexFailure('completion_regression')
            state = copy.deepcopy(state); state['generation'] = expected+1; raw = canonical(state)
            db.execute('UPDATE checkpoint SET state=?,state_sha=?,gzidx=?,gzidx_sha=? WHERE id=1',
                (raw, hashlib.sha256(raw).hexdigest(), blob, hashlib.sha256(blob).hexdigest()))
            db.commit()
        except BaseException:
            db.rollback(); raise
        finally: db.close()
        return state

    def _snapshot(self):
        db = self._connect()
        try: return self._load(db)
        finally: db.close()

    def status(self):
        s, blob = self._snapshot()
        return {'schema': SCHEMA, 'archive_id': s['source']['archive_id'], 'kind': s['source']['kind'],
            'generation': s['generation'], 'members': len(s['members']), 'boundary': s['boundary'],
            'index_coverage': s['index_coverage'], 'index_bytes': len(blob), 'source_total_bytes': s['source']['total_bytes'],
            'catalog_complete': s['catalog_complete'], 'gzip_crc_valid': s['gzip_crc_valid'],
            'complete': s['complete'], 'train_ready': False, 'network_requested': False}

    def scan(self, max_new_members=128, expected_generation=None, verify_end=True):
        if type(max_new_members) is not int or not 1 <= max_new_members <= self.limits.max_members:
            raise IndexFailure('scan_member_budget')
        state, blob = self._snapshot(); previous = state['generation']
        if expected_generation is not None and expected_generation != previous: raise IndexFailure('stale_generation')
        budget = Budget(self.limits)
        if state['source']['schema']=='ordered_drive_gzip_source_v1':verify_end=False
        # source budget 종료 뒤에도 마지막 완전 boundary의 index를 확정할 여유를 예약한다.
        budget.active_read_limit=self.limits.max_read_bytes-min(4*state['spacing'],self.limits.max_read_bytes//2)
        if state['catalog_complete']:
            return self.verify() if verify_end and not state['complete'] else self._result(budget)
        source = self._source(state['source'], budget); gz = _gzip(source, state['spacing'])
        reason = 'member_batch_limit'; pending_name = None; pending_pax = {}; new = 0; chain_metadata = False
        try:
            self._restore(gz,blob,state)
            gz.seek(state['boundary'])
            names = {r['name'].casefold() for r in state['members']}
            while new < max_new_members:
                chain_start = state['boundary']; header = _read(gz,512,budget)
                if header == bytes(512):
                    if chain_metadata or pending_name or pending_pax or _read(gz,512,budget) != bytes(512):
                        raise IndexFailure('invalid_tar_end')
                    state['catalog_complete'] = True; state['tar_end'] = state['boundary']+1024
                    reason = 'catalog_complete_crc_pending'; break
                try: member = tarfile.TarInfo.frombuf(header,'utf-8','strict')
                except (tarfile.HeaderError, UnicodeError, ValueError): raise IndexFailure('invalid_tar_header') from None
                if member.size < 0: raise IndexFailure('negative_tar_size')
                if member.type in (tarfile.GNUTYPE_LONGNAME,tarfile.XHDTYPE,tarfile.XGLTYPE):
                    chain_metadata = True
                    if member.size > CHUNK: raise IndexFailure('tar_metadata_limit')
                    data = _read(gz,member.size,budget); padding = _read(gz,(-member.size)%512,budget)
                    if padding.strip(b'\0'): raise IndexFailure('nonzero_tar_padding')
                    if member.type == tarfile.GNUTYPE_LONGNAME:
                        if pending_name is not None or not data.endswith(b'\0') or b'\0' in data[:-1]: raise IndexFailure('invalid_longname')
                        pending_name = safe_path(data[:-1].decode('utf-8'))
                    else:
                        values = pax_fields(data)
                        if member.type == tarfile.XGLTYPE:
                            if set(values)&{'path','size'}: raise IndexFailure('unsupported_global_pax')
                        elif pending_pax: raise IndexFailure('duplicate_pax')
                        else: pending_pax = values
                    continue
                name = safe_path(pending_pax.get('path',pending_name or member.name)); size = member.size
                if 'size' in pending_pax:
                    if not re.fullmatch(r'0|[1-9][0-9]{0,15}',pending_pax['size']): raise IndexFailure('pax_size')
                    size = int(pending_pax['size'])
                pending_name = None; pending_pax = {}; chain_metadata = False
                if name.casefold() in names: raise IndexFailure('duplicate_tar_member')
                if member.type not in (tarfile.DIRTYPE,tarfile.REGTYPE,tarfile.AREGTYPE): raise IndexFailure('links_or_special_member')
                if member.type == tarfile.DIRTYPE and size: raise IndexFailure('directory_payload')
                if size > self.limits.max_file_bytes: raise IndexFailure('file_bytes_limit')
                offset = gz.tell(); h = hashlib.sha256(); left = size
                while left:
                    data = _read(gz,min(CHUNK,left),budget); h.update(data); left -= len(data)
                if _read(gz,(-size)%512,budget).strip(b'\0'): raise IndexFailure('nonzero_tar_padding')
                if len(state['members']) >= self.limits.max_members: raise IndexFailure('catalog_members_limit')
                state['members'].append({'ordinal':len(state['members']),'chain_start':chain_start,'name':name,
                    'offset':offset,'size':size,'end':gz.tell(),'sha256':h.hexdigest(),
                    'type':'directory' if member.type == tarfile.DIRTYPE else 'file'})
                names.add(name.casefold()); state['boundary'] = gz.tell(); new += 1
        except Exception as exc:
            stop=exc.reason if isinstance(exc,IndexFailure) else budget.last_source_failure
            if stop not in ('read_bytes_limit','decoded_bytes_limit','elapsed_limit','incomplete_tar','catalog_members_limit','transfer_bytes_limit','ranges_limit'):
                raise
            reason = stop
        finally:
            try: new_blob, points = _pack_index(gz,self.limits)
            finally: gz.close(); source.close()
        if state['source']['schema']=='ordered_drive_gzip_source_v1' and self.runtime.failed:
            return self._result(budget,stop_reason=reason,checkpoint_deferred=True,new_members=0)
        if state['source']['schema']=='ordered_drive_gzip_source_v1' and new==0 and reason in ('ranges_limit','transfer_bytes_limit','elapsed_limit','read_bytes_limit'):
            return self._result(budget,stop_reason=reason,checkpoint_deferred=True,new_members=0)
        budget.active_read_limit=self.limits.max_read_bytes
        budget.last_source_failure=None
        # 새 handle의 seek로 lazy buffer 안에 머물렀던 범위도 restart index로 확정한다.
        checkpoint_source=self._source(state['source'],budget);checkpoint_gz=_gzip(checkpoint_source,state['spacing'])
        try:
            self._restore(checkpoint_gz,new_blob,{**state,'engine_version':self._engine_version()})
            checkpoint_gz.seek(state['boundary'])
            new_blob,points=_pack_index(checkpoint_gz,self.limits)
        except Exception as exc:
            stop=exc.reason if isinstance(exc,IndexFailure) else budget.last_source_failure
            if stop not in ('read_bytes_limit','decoded_bytes_limit','elapsed_limit','transfer_bytes_limit','ranges_limit'):raise
            return self._result(budget,stop_reason=stop,checkpoint_deferred=True,new_members=0)
        finally:checkpoint_gz.close();checkpoint_source.close()
        self._finish_source(state['source'],budget)
        import indexed_gzip
        state['engine_version'] = indexed_gzip.__version__
        coverage=max([0]+[u for u,c in points if c < state['source']['total_bytes']])
        if coverage<state['index_coverage']:raise IndexFailure('index_progress_regression')
        state['index_coverage'] = coverage
        state = self._commit(previous,state,new_blob)
        result = self._result(budget,stop_reason=reason,new_members=new)
        if state['catalog_complete'] and verify_end:
            try: result['verification'] = self._verify(state,new_blob,budget)
            except IndexFailure as exc:
                if exc.reason not in ('read_bytes_limit','decoded_bytes_limit','elapsed_limit'): raise
                result['verification'] = {'complete':False,'stop_reason':exc.reason,'crc_pending':True}
            result.update(self.status()); result['stats'] = budget.stats()
        return result

    @staticmethod
    def _engine_version():
        try:import indexed_gzip
        except ImportError:raise IndexFailure('install_requirements_index') from None
        return indexed_gzip.__version__

    def _restore(self,gz,blob,state):
        if blob:
            if state['engine_version'] != self._engine_version():raise IndexFailure('index_engine_version_changed')
            gz.import_index(fileobj=io.BytesIO(blob))

    def _verify(self, state, blob, budget):
        if state['complete']: return self.status()
        if not state['catalog_complete']: raise IndexFailure('catalog_not_complete')
        source = SplitSource(state['source'],budget); decoder = zlib.decompressobj(31); pos = 0
        try:
            while data := source.read(CHUNK):
                if decoder.eof: raise IndexFailure('trailing_or_concatenated_gzip')
                pending = data
                while pending:
                    budget.check()
                    if budget.decoded_bytes >= self.limits.max_decoded_bytes: raise IndexFailure('decoded_bytes_limit')
                    try: output = decoder.decompress(pending,min(CHUNK,self.limits.max_decoded_bytes-budget.decoded_bytes))
                    except zlib.error: raise IndexFailure('gzip_crc_or_decode') from None
                    pending = decoder.unconsumed_tail; budget.decoded_bytes += len(output)
                    after = max(0,state['tar_end']-pos)
                    if output[after:].strip(b'\0'): raise IndexFailure('nonzero_tar_tail')
                    pos += len(output)
                    if decoder.unused_data: raise IndexFailure('trailing_or_concatenated_gzip')
            if not decoder.eof or pos < state['tar_end']: raise IndexFailure('gzip_incomplete')
        finally: source.close()
        _validate_source(state['source'])
        state = copy.deepcopy(state); state['gzip_crc_valid'] = state['complete'] = True
        self._commit(state['generation'],state,blob)
        return {**self.status(),'stats':budget.stats()}

    def verify(self):
        state, blob = self._snapshot()
        if state['source']['schema']=='ordered_drive_gzip_source_v1':raise IndexFailure('drive_full_crc_deferred')
        return self._verify(state,blob,Budget(self.limits))

    def lookup(self, name):
        name = safe_path(name); state, _ = self._snapshot()
        for row in state['members']:
            if row['name'] == name and row['type'] == 'file': return copy.deepcopy(row)
        raise IndexFailure('member_not_indexed')

    def extract(self, names, output):
        if not isinstance(names,list) or not 1 <= len(names) <= 32 or len(set(names)) != len(names):
            raise IndexFailure('explicit_unique_names_required')
        state, blob = self._snapshot(); wanted = {safe_path(n) for n in names}
        rows = [r for r in state['members'] if r['name'] in wanted and r['type'] == 'file']
        if len(rows) != len(wanted): raise IndexFailure('member_not_indexed')
        if self.path.stat().st_size+sum(r['size'] for r in rows)+self.limits.max_state_bytes>self.limits.max_disk_bytes:
            raise IndexFailure('output_disk_limit')
        output = Path(output).absolute()
        if output.exists(): raise IndexFailure('new_output_required')
        if any(p.is_symlink() or getattr(p,'is_junction',lambda:False)() for p in output.parents): raise IndexFailure('output_link')
        output.mkdir(parents=True)
        budget = Budget(self.limits); source = self._source(state['source'],budget); gz = _gzip(source,state['spacing'],auto_build=False)
        files = []
        try:
            self._restore(gz,blob,state)
            for row in rows:  # 물리 archive 순서, 요청 순서나 문자열 frame 정렬이 아니다.
                target = output/'files'/row['name']; target.parent.mkdir(parents=True,exist_ok=True)
                gz.seek(row['offset']); h = hashlib.sha256(); left = row['size']
                with target.open('xb') as f:
                    while left:
                        data = _read(gz,min(CHUNK,left),budget); f.write(data); h.update(data); left -= len(data)
                if h.hexdigest() != row['sha256']: raise IndexFailure('member_hash_changed')
                files.append({**row,'relative_path':str(target.relative_to(output).as_posix())})
        finally: gz.close(); source.close()
        self._finish_source(state['source'],budget)
        report = {'schema':'persistent_index_extraction_v1','source_sha256':state['source_sha256'],
            'generation':state['generation'],'files':files,'stats':budget.stats(),'train_ready':False,
            'whole_archive_integrity_verified':state['complete'],'existing_archive_verify_compatible':False}
        report['network_requested']=report['stats']['network_requested']
        report['content_sha256'] = digest(report)
        if len(canonical(report))>self.limits.max_state_bytes:raise IndexFailure('extraction_report_limit')
        (output/'extraction.json').write_bytes(canonical(report)+b'\n')
        return report

    def audit_source_sha(self):
        state, _ = self._snapshot()
        if state['source']['schema']=='ordered_drive_gzip_source_v1':raise IndexFailure('drive_full_sha_deferred')
        budget = Budget(self.limits); source = SplitSource(state['source'],budget)
        results = []
        try:
            for i,p in enumerate(state['source']['parts']):
                h = hashlib.sha256(); left = p['stat']['size']
                while left:
                    data = source.read(min(CHUNK,left)); h.update(data); left -= len(data)
                value = h.hexdigest()
                if p['expected_sha256'] and value != p['expected_sha256']: raise IndexFailure('source_sha_changed')
                results.append({'index':i,'sha256':value,'expected_matched':p['expected_sha256'] == value if p['expected_sha256'] else None})
        finally: source.close()
        return {'parts':results,'stats':budget.stats(),'scope':'explicit full-source SHA audit; not automatic on resume'}
