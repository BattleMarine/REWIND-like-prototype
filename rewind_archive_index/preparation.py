"""실제 archive prepare 요청에서 사용하는 opt-in 자동 인덱싱. 단일 writer용."""
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import re
import time
import uuid

from rewind_archive.contracts import Limits, PreparationError, digest, canonical, safe_path, target_summary
from rewind_archive.extraction import code_hashes
from .core import ArchiveIndex
from .drive import DriveRuntime, DriveLimits, make_drive_source
from .source import IndexFailure, IndexLimits, make_source, MIB
from .snapshots import export_snapshot, import_snapshot, _anchor, _receipt
from .extension_records import atomic_record, exception_kind

OPTIONS = {'workspace', 'snapshot_directory', 'input_snapshot_directory', 'anchor', 'batch_members', 'max_batches', 'spacing','request_id','provenance'}
BOUNDED = {'read_bytes_limit','decoded_bytes_limit','elapsed_limit','ranges_limit','transfer_bytes_limit',
           'prepare_read_limit','prepare_decoded_limit','prepare_elapsed_limit','prepare_disk_limit','rss_limit',
           'catalog_members_limit','checkpoint_size_limit','checkpoint_disk_limit','prepare_members_limit'}


def _path(value, *, local=False):
    path=Path(value).absolute()
    if any(p.is_symlink() or getattr(p,'is_junction',lambda:False)() for p in (path,*path.parents)):
        raise IndexFailure('prepare_path_link')
    if local and str(path).replace('\\','/').startswith('/content/drive/'):
        raise IndexFailure('live_db_must_be_local')
    return path


def _read(path, maximum=2*MIB):
    path=_path(path)
    if not path.is_file() or path.stat().st_size>maximum:raise IndexFailure('prepare_record_size')
    with path.open('rb') as stream:raw=stream.read(maximum+1)
    if len(raw)>maximum:raise IndexFailure('prepare_record_size')
    return json.loads(raw)


def _source(manifest, base):
    if all(p['source']['type']=='drive' for p in manifest['parts']):return make_drive_source(manifest)
    if len(manifest['parts'])!=manifest['total_parts'] or any(p['source']['type']!='local' for p in manifest['parts']):
        raise IndexFailure('prepare_complete_local_or_drive_required')
    paths=[]
    for part in manifest['parts']:
        path=Path(base)/part['source']['path']
        if path.stat().st_size!=part['size_bytes'] or path.stat().st_mtime_ns!=part['identity']['mtime_ns']:
            raise IndexFailure('prepare_local_identity_changed')
        paths.append(path)
    return make_source(paths,manifest['archive_id'],manifest['kind'],manifest['dataset_version'])


def _options(options):
    if not isinstance(options,dict) or set(options)-OPTIONS or not {'workspace','snapshot_directory'}<=set(options):
        raise IndexFailure('prepare_index_options')
    result={'anchor':None,'batch_members':16,'max_batches':8,'spacing':MIB,'request_id':uuid.uuid4().hex,'provenance':None,**options}
    if not isinstance(result['request_id'],str) or not re.fullmatch('[A-Za-z0-9_-]{1,64}',result['request_id']):raise IndexFailure('prepare_request_id')
    if result['provenance'] is not None:
        from .extension_records import validate_provenance
        result['provenance']=validate_provenance(result['provenance'])
    for key,maximum in (('batch_members',16),('max_batches',8)):
        if type(result[key]) is not int or not 1<=result[key]<=maximum:raise IndexFailure('prepare_batch_limit')
    if type(result['spacing']) is not int or not 65536<=result['spacing']<=16*MIB:raise IndexFailure('spacing_limit')
    if result['anchor'] is not None:_receipt(result['anchor'])
    result['workspace']=_path(result['workspace'],local=True)
    result['snapshot_directory']=_path(result['snapshot_directory'])
    result['input_snapshot_directory']=_path(result.get('input_snapshot_directory',result['snapshot_directory']))
    if result['workspace']==result['snapshot_directory'] or result['workspace'].is_relative_to(result['snapshot_directory']) or result['snapshot_directory'].is_relative_to(result['workspace']):
        raise IndexFailure('prepare_separate_storage_required')
    return result


def _head(path, source_sha, anchor):
    if not path.exists():return None
    value=_read(path)
    body={k:v for k,v in value.items() if k!='content_sha256'}
    if value.get('schema')!='archive_prepare_head_v1' or digest(body)!=value.get('content_sha256') or value.get('source_sha256')!=source_sha:
        raise IndexFailure('prepare_head_identity_or_hash')
    rows=value.get('receipts')
    if not isinstance(rows,list) or not 1<=len(rows)<=64:raise IndexFailure('prepare_head_history_limit')
    for row in rows:_receipt(row)
    for previous,current in zip(rows,rows[1:]):
        if current['parent_snapshot_id']!=previous['snapshot_id'] or current['lineage']!=previous['lineage'] or current['generation']<=previous['generation']:
            raise IndexFailure('prepare_head_lineage')
    if any(row['source_sha256']!=source_sha for row in rows):raise IndexFailure('prepare_head_source')
    if anchor is not None and anchor not in rows:raise IndexFailure('prepare_anchor_conflict')
    return value


class Resources:
    """같은 요청의 scan/extract에 남은 예산을 배분한다. Runtime은 교체하지 않는다."""
    def __init__(self, limits):
        self.started=time.monotonic();self.limits=limits;self.paths=set()
        self.read=0;self.decoded=0;self.calls=[];self.members=0
        self.peak_rss=0
        self.index_limits=IndexLimits(max_read_bytes=32*MIB,
            max_decoded_bytes=min(limits.max_decompressed_bytes,32*MIB),max_seconds=min(limits.max_elapsed_seconds,300.),
            max_file_bytes=min(limits.max_file_bytes,8*MIB),max_disk_bytes=min(limits.max_disk_bytes,128*MIB),
            max_state_bytes=min(16*MIB,limits.max_disk_bytes//8),max_index_bytes=min(32*MIB,limits.max_disk_bytes//8))

    def check(self,reserve=0):
        import psutil
        self.peak_rss=max(self.peak_rss,psutil.Process().memory_info().rss)
        if self.peak_rss>self.index_limits.max_rss_bytes:raise IndexFailure('rss_limit')
        if time.monotonic()-self.started>=self.index_limits.max_seconds:raise IndexFailure('prepare_elapsed_limit')
        roots=[root for root in self.paths if not any(root!=other and root.is_relative_to(other) for other in self.paths)]
        total=0;count=0
        for root in roots:
            if not root.exists():continue
            for path in root.rglob('*'):
                count+=1
                if count>10000:raise IndexFailure('prepare_inventory_limit')
                if path.is_symlink() or getattr(path,'is_junction',lambda:False)():raise IndexFailure('prepare_path_link')
                if time.monotonic()-self.started>=self.index_limits.max_seconds:raise IndexFailure('prepare_elapsed_limit')
                if path.is_file():total+=path.stat().st_size
                if total+2*MIB+reserve>self.index_limits.max_disk_bytes:raise IndexFailure('prepare_disk_limit')
        if total+2*MIB+reserve>self.index_limits.max_disk_bytes:raise IndexFailure('prepare_disk_limit')

    def call(self,index,name,*args,**kwargs):
        self.check();read=self.index_limits.max_read_bytes-self.read;decoded=self.index_limits.max_decoded_bytes-self.decoded
        if read<=0:raise IndexFailure('prepare_read_limit')
        if decoded<=0:raise IndexFailure('prepare_decoded_limit')
        index.limits=replace(self.index_limits,max_read_bytes=read,max_decoded_bytes=decoded,
            max_seconds=max(.001,self.index_limits.max_seconds-(time.monotonic()-self.started)))
        previous=getattr(index,'last_budget',None)
        result=None
        try:
            result=getattr(index,name)(*args,**kwargs)
            if name=='scan':self.members+=result.get('new_members',0)
            return result
        finally:
            current=getattr(index,'last_budget',None)
            stats=current.stats() if current is not None and current is not previous else result.get('stats') if result else None
            if stats is not None:
                self.read+=stats['source_read_bytes'];self.decoded+=stats['returned_decoded_bytes']
                self.calls.append({'operation':name,'stats':stats})

    def report(self):
        return {'source_read_bytes':self.read,'returned_decoded_bytes':self.decoded,
                'elapsed_seconds':time.monotonic()-self.started,'calls':list(self.calls),'scanned_members':self.members,
                'member_limit':min(128,self.limits.max_members),'observed_peak_rss':self.peak_rss,'index_limits':asdict(self.index_limits)}


class PreparationIndex(ArchiveIndex):
    def _source(self,spec,budget):
        self.last_budget=budget
        return super()._source(spec,budget)

    def _commit(self,expected,state,blob):
        old,_=self._snapshot()
        for row in state['members'][len(old['members']):]:
            if len(row['name'].encode('utf-8'))>self.preparation_path_limit:raise IndexFailure('prepare_path_limit')
        return super()._commit(expected,state,blob)


class PreparationContext:
    """여러 준비 요청/복원 검증까지 하나의 누적 예산·DriveRuntime을 공유한다."""
    def __init__(self,limits,authorized_session=None):
        if not isinstance(limits,Limits):raise IndexFailure('limits_required')
        self.limits=limits;self.resources=Resources(limits);self.session=authorized_session
        self.runtime=None;self.source_sha256=None

    def bind(self,spec):
        fingerprint=digest(spec)
        if self.source_sha256 is not None and self.source_sha256!=fingerprint:raise IndexFailure('prepare_context_source_conflict')
        self.source_sha256=fingerprint
        if spec['schema']=='ordered_drive_gzip_source_v1' and self.runtime is None:
            if self.session is None:raise IndexFailure('drive_session_required')
            self.runtime=DriveRuntime(self.session,DriveLimits(max_body_bytes=min(8*MIB,self.limits.max_transfer_bytes),
                max_requests=min(120,self.limits.max_ranges),max_seconds=min(300.,self.limits.max_elapsed_seconds),range_bytes=self.limits.media_range_bytes))

def backend_hashes():
    return {p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(Path(__file__).parent.glob('*.py'))}


def prepare_indexed(manifest,output,targets,limits,options,*,base='.',authorized_session=None,context=None):
    """명시 target 준비. 정확한 source별 HEAD만 읽으며 디렉터리 latest 탐색은 하지 않는다."""
    opts=_options(options);spec=_source(manifest,base);fingerprint=digest(spec)
    if limits.max_metadata_bytes<64*1024:raise IndexFailure('prepare_metadata_limit_unsupported')
    if limits.io_chunk_bytes!=65536:raise IndexFailure('prepare_io_chunk_unsupported')
    if opts['anchor'] is not None and opts['anchor']['source_sha256']!=fingerprint:raise IndexFailure('prepare_anchor_source')
    output=_path(output);workspace=opts['workspace']/fingerprint
    snapshot_dir=opts['snapshot_directory'];head_path=snapshot_dir/(fingerprint+'-head.json')
    if output.exists():
        from rewind_archive.matching import verify_extraction
        old=verify_extraction(output,manifest)
        if old.get('exact_targets')!=targets or not old.get('targets_complete') or old.get('persistent_index',{}).get('source_sha256')!=fingerprint:
            raise IndexFailure('prepare_existing_output_mismatch')
        return {**old,'reuse':{'verified':True,'network_requested':False,'source_identity_scope':'pinned_manifest_not_live_remote'},
                'request_stats':{'source_read_bytes':0,'returned_decoded_bytes':0,'network_requested':False}}
    context=context or PreparationContext(limits,authorized_session)
    if not isinstance(context,PreparationContext) or context.limits!=limits:raise IndexFailure('prepare_context_limits_conflict')
    if authorized_session is not None and context.session is not authorized_session:raise IndexFailure('prepare_context_session_conflict')
    resources=context.resources
    resources.paths.update({opts['workspace'],snapshot_dir,output})
    resources.check()
    workspace.mkdir(parents=True,exist_ok=True);snapshot_dir.mkdir(parents=True,exist_ok=True)
    if output==workspace or output.is_relative_to(workspace) or workspace.is_relative_to(output) or output.is_relative_to(snapshot_dir) or snapshot_dir.is_relative_to(output):
        raise IndexFailure('prepare_separate_output_required')
    # mount의 강한 분산 lock을 보장하지 않는다. 단일 writer의 겹침·중단 기록만 거부한다.
    lock=snapshot_dir/(fingerprint+'-writer.lock')
    try:
        with lock.open('x',encoding='utf-8') as stream:stream.write('single_writer\n')
    except FileExistsError:raise IndexFailure('prepare_writer_active_or_interrupted') from None
    record=workspace/('request-'+opts['request_id']+'.json')
    if record.exists():
        lock.unlink(missing_ok=True)
        raise IndexFailure('prepare_request_already_consumed')
    report={'schema':'archive_index_preparation_v1','status':'started','stage':'initial','source_sha256':fingerprint,
            'request_id':opts['request_id'],
            'archive_manifest_sha256':digest(manifest),'targets':targets,'scans':[],'train_ready':False,
            'record_path':str(record),'persistent_head_path':str(head_path),'network_requested':False,
            'provenance':opts['provenance'],'backend_sha256':backend_hashes()}
    index=None;runtime=None;files=[];reason='prepare_started';head=None;primary=None
    def save(stage):
        report['stage']=stage;report['resources']=resources.report();atomic_record(record,report,initial=not record.exists())
    def publish():
        nonlocal head
        resources.check()
        before=_head(head_path,fingerprint,opts['anchor'])
        if before!=head:raise IndexFailure('prepare_head_cas_conflict')
        prior=head['receipts'][-1] if head else opts['anchor']
        unchanged=prior is not None and index.status()['generation']==prior['generation']
        if unchanged and head is not None:return
        history=head['receipts'] if head else ([prior] if prior else [])
        if len(history)>=64:raise IndexFailure('prepare_head_history_limit')
        if not unchanged:
            resources.check(3*index.path.stat().st_size+65536)
            saved=export_snapshot(index,snapshot_dir,known_anchor=prior)
            report['last_published_snapshot']=saved;save('snapshot_published')
            history=history+[saved['receipt']]
        if _head(head_path,fingerprint,opts['anchor'])!=before:raise IndexFailure('prepare_head_cas_conflict')
        body={'schema':'archive_prepare_head_v1','source_sha256':fingerprint,'receipts':history}
        new={**body,'content_sha256':digest(body)}
        atomic_record(head_path,new,initial=not head_path.exists());head=new
        report['head']=head;save('head_saved')
    try:
        save('restore')
        head=_head(head_path,fingerprint,opts['anchor']);receipt=head['receipts'][-1] if head else opts['anchor']
        db=workspace/'index.sqlite'
        if not db.exists():
            if receipt is None:ArchiveIndex(db,resources.index_limits).create(spec,opts['spacing'])
            else:
                folder=opts['input_snapshot_directory'] if receipt==opts['anchor'] else snapshot_dir
                resources.check(3*receipt['bytes']+65536)
                report['restore']=import_snapshot(folder/(receipt['filename']+'.receipt.json'),db,
                    expected_source_sha256=fingerprint,expected_snapshot_id=receipt['snapshot_id'],min_generation=receipt['generation'],
                    known_anchor=receipt,index_limits=resources.index_limits)
        context.bind(spec);runtime=context.runtime
        index=PreparationIndex(db,resources.index_limits,runtime=runtime);state,_=index._snapshot()
        index.preparation_path_limit=limits.max_path_bytes
        if state['source_sha256']!=fingerprint or state['spacing']!=opts['spacing']:raise IndexFailure('prepare_local_source_or_spacing_conflict')
        if _anchor(index)!=receipt:raise IndexFailure('prepare_local_anchor_conflict')
        if receipt and state['generation']<receipt['generation']:raise IndexFailure('prepare_local_generation_regression')
        report['before']=index.status();publish();save('targets_lookup')
        for batch in range(opts['max_batches']):
            state,_=index._snapshot();available={r['name'] for r in state['members'] if r['type']=='file'}
            if set(targets)<=available:break
            if state['catalog_complete']:reason='indexed_targets_not_found';break
            remaining_members=min(limits.max_members,128)-resources.members
            if remaining_members<=0:raise IndexFailure('prepare_members_limit')
            save('scan_pre')
            result=resources.call(index,'scan',min(opts['batch_members'],remaining_members),expected_generation=state['generation'],verify_end=False)
            report['scans'].append(result);publish();save('scan_post')
            if result.get('checkpoint_deferred') or result['boundary']<=state['boundary']:
                reason=result.get('stop_reason') if result.get('checkpoint_deferred') else 'index_no_progress';break
        else:reason='index_batch_limit'
        state,_=index._snapshot();available={r['name'] for r in state['members'] if r['type']=='file'}
        if set(targets)<=available:
            selected=[r for r in state['members'] if r['name'] in targets and r['type']=='file']
            if sum(r['size'] for r in selected)+2*MIB>limits.max_disk_bytes:raise IndexFailure('prepare_payload_disk_limit')
            save('extract_pre');extracted=resources.call(index,'extract',targets,output)
            files=[{'member':r['name'],'relative_path':r['relative_path'],'size_bytes':r['size'],'sha256':r['sha256'],
                    'tar_complete':True,'kind':manifest['kind']} for r in extracted['files']]
            report['extraction']=extracted;reason='targets_complete';report['status']='passed'
        else:report['status']='partial'
    except BaseException as exc:
        primary=exc
        reason=getattr(exc,'reason','index_preparation_failed')
        if not isinstance(reason,str) or not re.fullmatch('[a-z0-9_]{1,100}',reason):reason='index_preparation_failed'
        report['status']='partial' if reason in BOUNDED else 'failed'
        report['failure_stage']=report['stage']
        if not isinstance(exc,Exception):
            report.update(status='interrupted',interruption={'type':exception_kind(exc),'stage':report['stage']})
            raise
    finally:
        cleanup_error=None
        try:
            report['stop_reason']=reason
            if index is not None:report['after']=index.status()
            if runtime is not None and hasattr(runtime,'transfer'):
                report['transport']=runtime.stats();report['network_requested']=runtime.transfer.ranges>0
            save('finished' if report['status']!='interrupted' else report['stage'])
        except BaseException as exc:
            cleanup_error=exc
            report['cleanup_exception']={'type':exception_kind(exc),'stage':'cleanup'}
            try:atomic_record(record,report)
            except BaseException:pass
        try:lock.unlink(missing_ok=True)
        except BaseException as exc:
            cleanup_error=cleanup_error or exc
            report['cleanup_exception']={'type':exception_kind(exc),'stage':'lock_release'}
            try:atomic_record(record,report)
            except BaseException:pass
        if primary is None and cleanup_error is not None:raise cleanup_error
    usable=report['status'] in {'passed','partial'}
    legacy_reason=reason if reason=='targets_complete' or not usable else 'members_limit'
    legacy={'schema_version':'archive_extraction_v1','extractor_version':'persistent_index_opt_in_v1','code_sha256':code_hashes(),
        'persistent_backend_sha256':backend_hashes(),
        'archive_manifest':manifest,'archive_manifest_sha256':digest(manifest),'selection':targets,'limits':asdict(limits),
        'stop_reason':legacy_reason,'failure_class':'bounded_stop' if usable else 'invalid_or_changed_source','files':files,
        'inspection_candidates_usable':usable,'train_ready':False,'whole_archive_integrity_verified':False,
        'gzip_eof_crc_valid':False,'tar_end_blocks_seen':False,'operation':'extract','profile':'persistent_index_opt_in_v1',
        'report_disk_reserve_bytes':2*MIB,'stats':{**resources.report(),'payload_disk_bytes':sum(r['size_bytes'] for r in files),
        'network_requested':report['network_requested']},'persistent_index':report,**target_summary(targets,files,usable)}
    output.mkdir(parents=True,exist_ok=True)
    raw=canonical({**legacy,'content_sha256':digest(legacy)})+b'\n'
    if len(raw)>2*MIB:raise IndexFailure('prepare_report_limit')
    with (output/'extraction_manifest.json').open('xb') as stream:stream.write(raw)
    return legacy
