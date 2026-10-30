"""B coverage 밖을 확장하고, 저장된 C 인덱스로 새 JSON을 cold 검증하는 한 번의 pilot."""
from dataclasses import replace
import hashlib
import io
import json
from pathlib import Path
import re
import time

from .core import ArchiveIndex
from .source import IndexFailure,canonical,digest,MIB
from .source import _gzip
from .drive import DriveRuntime,make_drive_source
from .pilot import PILOT_INDEX_LIMITS,PILOT_DRIVE_LIMITS,RUNTIME_ID
from .snapshots import import_snapshot,export_snapshot
from rewind_archive.contracts import safe_path


class PhaseResources:
    """호출마다 reset하지 않는 C 전체 read/decoded/time/disk ledger."""
    def __init__(self,workspace,output):
        self.workspace=workspace;self.output=output;self.started=time.monotonic()
        self.read=0;self.decoded=0;self.calls=[];self.seen=set();self.peak=0

    def check(self):
        import psutil
        rss=psutil.Process().memory_info().rss;self.peak=max(self.peak,rss)
        if rss>PILOT_INDEX_LIMITS.max_rss_bytes:raise IndexFailure('rss_limit')
        size=sum(p.stat().st_size for root in (self.workspace,self.output) if root.exists() for p in root.rglob('*') if p.is_file())
        if size>PILOT_INDEX_LIMITS.max_disk_bytes:raise IndexFailure('phase_disk_limit')
        if time.monotonic()-self.started>=300:raise IndexFailure('phase_elapsed_limit')

    def call(self,index,name,*args,**kwargs):
        self.check();remaining_read=32*MIB-self.read;remaining_decoded=32*MIB-self.decoded
        if remaining_read<=0:raise IndexFailure('phase_read_limit')
        if remaining_decoded<=0:raise IndexFailure('phase_decoded_limit')
        index.limits=replace(PILOT_INDEX_LIMITS,max_read_bytes=remaining_read,max_decoded_bytes=remaining_decoded,
            max_seconds=max(.001,300-(time.monotonic()-self.started)))
        before=getattr(index.runtime,'operation',None)
        try:return getattr(index,name)(*args,**kwargs)
        finally:
            operation=getattr(index.runtime,'operation',None)
            if operation is not None and operation is not before and operation not in self.seen:
                self.seen.add(operation);stats=operation.stats();self.read+=stats['source_read_bytes'];self.decoded+=stats['returned_decoded_bytes']
                self.calls.append({'operation':name,'stats':stats})

    def report(self):
        return {'read_bytes':self.read,'decoded_bytes':self.decoded,'seconds':time.monotonic()-self.started,
            'observed_peak_rss':self.peak,'read_limit':32*MIB,'decoded_limit':32*MIB,'seconds_limit':300,
            'disk_limit':PILOT_INDEX_LIMITS.max_disk_bytes,'calls':self.calls}


def evidence(index):
    state,blob=index._snapshot()
    return {**index.status(),'source_sha256':state['source_sha256'],'catalog_sha256':state['catalog_sha256'],
        'blob_sha256':hashlib.sha256(blob).hexdigest(),'blob_bytes':len(blob),'engine_version':state['engine_version']}


def stored_points(state,blob):
    """원본 read를 금지한 metadata-only handle로 저장 blob의 point만 검사한다."""
    class MetadataOnly(io.RawIOBase):
        mode='rb'
        def __init__(self):self.pos=0
        def readable(self):return True
        def seekable(self):return True
        def tell(self):return self.pos
        def seek(self,offset,whence=0):
            self.pos=offset if whence==0 else self.pos+offset if whence==1 else state['source']['total_bytes']+offset
            return self.pos
        def read(self,size=-1):
            if size==0:return b''
            raise IndexFailure('extension_point_inspection_read_forbidden')
    source=MetadataOnly();gz=_gzip(source,state['spacing'],auto_build=False)
    try:
        gz.import_index(fileobj=io.BytesIO(blob))
        return [[u,c] for u,c in gz.seek_points() if c<state['source']['total_bytes']]
    finally:gz.close();source.close()


def _prefix(old,current):
    if current[:len(old)]!=old:raise IndexFailure('extension_old_catalog_changed')
    cursor=0
    for ordinal,row in enumerate(current):
        if row['ordinal']!=ordinal or row['chain_start']!=cursor or row['end']<=cursor:raise IndexFailure('extension_catalog_order')
        cursor=row['end']


def _anchor(anchor,spec):
    if not isinstance(anchor,dict) or set(anchor)!={'schema','runtime_id','receipt','selected'} or anchor['schema']!='persistent_drive_pilot_anchor_v1':raise IndexFailure('extension_b_anchor_required')
    if anchor['runtime_id']==RUNTIME_ID:raise IndexFailure('extension_fresh_runtime_required')
    if anchor['receipt']['source_sha256']!=digest(spec):raise IndexFailure('extension_source_mismatch')
    if not isinstance(anchor['selected'],list) or not 1<=len(anchor['selected'])<=2:raise IndexFailure('extension_old_selection')
    for row in anchor['selected']:
        if set(row)!={'name','sha256'} or not re.fullmatch('[a-f0-9]{64}',row['sha256']):raise IndexFailure('extension_old_selection')
        safe_path(row['name'])


def _json_files(report,output,expected):
    actual={row['name']:row for row in report['files']}
    for row in expected:
        entry=actual.get(row['name'])
        if entry is None or entry['sha256']!=row['sha256']:raise IndexFailure('extension_member_hash')
        data=(output/entry['relative_path']).read_bytes()
        if hashlib.sha256(data).hexdigest()!=row['sha256'] or not isinstance(json.loads(data),dict):raise IndexFailure('extension_json_content')


def run_extension(*,run_id,workspace,input_snapshot_directory,output_snapshot_directory,manifest,anchor,session_factory):
    if not isinstance(run_id,str) or not re.fullmatch('[A-Za-z0-9_-]{1,64}',run_id):raise IndexFailure('extension_run_id')
    workspace=Path(workspace).absolute();workspace.mkdir(parents=True,exist_ok=True)
    if str(workspace).replace('\\','/').startswith('/content/drive/'):raise IndexFailure('live_db_must_be_local')
    incoming=Path(input_snapshot_directory).resolve();outgoing=Path(output_snapshot_directory).resolve()
    if incoming==outgoing:raise IndexFailure('extension_separate_snapshot_directories')
    marker=workspace/(run_id+'-C.json')
    if marker.exists():raise IndexFailure('pilot_run_already_consumed')
    if outgoing.exists():raise IndexFailure('extension_new_output_directory_required')
    spec=make_drive_source(manifest);_anchor(anchor,spec)
    with marker.open('x',encoding='utf-8') as f:json.dump({'status':'started','phase':'C','run_id':run_id},f)
    report={'schema':'persistent_drive_extension_report_v1','run_id':run_id,'phase':'C','status':'started',
        'max_batches':8,'batch_members':16,'index_limits':vars(PILOT_INDEX_LIMITS),'drive_limits':vars(PILOT_DRIVE_LIMITS),
        'full_archive_crc_requested':False,'b_receipt':anchor['receipt'],'scans':[]}
    runtime=None;session=None;ledger=None;index=None
    bounded={'read_bytes_limit','decoded_bytes_limit','elapsed_limit','ranges_limit','transfer_bytes_limit',
        'phase_read_limit','phase_decoded_limit','phase_elapsed_limit','phase_disk_limit','rss_limit','catalog_members_limit'}
    try:
        if ArchiveIndex._engine_version()!='1.10.3':raise IndexFailure('pilot_engine_1_10_3_required')
        receipt=anchor['receipt'];db=workspace/(run_id+'-C.sqlite')
        report['restore_b']=import_snapshot(incoming/(receipt['filename']+'.receipt.json'),db,
            expected_source_sha256=digest(spec),expected_snapshot_id=receipt['snapshot_id'],min_generation=receipt['generation'],known_anchor=receipt)
        session=session_factory();runtime=DriveRuntime(session,PILOT_DRIVE_LIMITS)
        index=ArchiveIndex(db,PILOT_INDEX_LIMITS,runtime=runtime)
        old,old_blob=index._snapshot();old_rows=old['members'];old_coverage=old['index_coverage']
        old_points=stored_points(old,old_blob);old_max_compressed=max(c for u,c in old_points)
        report['before']=evidence(index);report['old_coverage']=old_coverage
        report.update(old_restart_points=old_points,point_inspection_source_reads=0)
        ledger=PhaseResources(workspace,outgoing);candidate=None;restart=None;expected_generation=old['generation'];previous_rows=old_rows
        for batch in range(8):
            result=ledger.call(index,'scan',16,verify_end=False,expected_generation=expected_generation)
            state,blob=index._snapshot();_prefix(previous_rows,state['members'])
            if state['generation']!=result['generation'] or len(state['members'])-len(previous_rows)!=result.get('new_members',0):raise IndexFailure('extension_concurrent_checkpoint')
            expected_generation=state['generation'];previous_rows=state['members']
            report['scans'].append({'batch':batch+1,'result':result,'checkpoint':evidence(index),'old_catalog_exact_prefix':True})
            if result.get('checkpoint_deferred') or result.get('new_members',0)==0:break
            if state['boundary']>old_coverage and state['index_coverage']>old_coverage:
                points=stored_points(state,blob);added=[p for p in points if p[0]>old_coverage and p not in old_points]
                for row in state['members'][len(old_rows):]:
                    valid=[p for p in added if p[0]<=row['offset']]
                    if row['type']=='file' and row['name'].endswith('.json') and valid and row['end']<=state['index_coverage']:
                        candidate=row;restart=max(valid);break
            if candidate is not None:break
            if state['catalog_complete']:break
        report['after_scan']=evidence(index)
        if candidate is None:
            report.update(status='partial',stop_reason='extension_coverage_or_new_json_not_reached')
        else:
            new={'name':candidate['name'],'sha256':candidate['sha256']}
            report['new_member']=candidate;report['new_restart_point']=restart
            report['after_restart_points']=stored_points(state,blob)
            selected=anchor['selected']+[new];report['selected']=selected
            preview=workspace/(run_id+'-C-preview')
            report['pre_snapshot_extraction']=ledger.call(index,'extract',[r['name'] for r in selected],preview)
            _json_files(report['pre_snapshot_extraction'],preview,selected);ledger.check()
            report['snapshot']=export_snapshot(index,outgoing,known_anchor=receipt)
            c=report['snapshot']['receipt']
            if c['parent_snapshot_id']!=receipt['snapshot_id'] or c['lineage']!=receipt['lineage']:raise IndexFailure('extension_snapshot_parent')
            saved={'schema':'persistent_drive_extension_anchor_v1','runtime_id':RUNTIME_ID,'receipt':c,'selected':selected}
            anchor_path=workspace/(run_id+'-C-anchor.json');anchor_path.write_bytes(canonical(saved)+b'\n')
            report.update(anchor_path=str(anchor_path),anchor_sha256=hashlib.sha256(anchor_path.read_bytes()).hexdigest())
            restored=workspace/(run_id+'-C-restored.sqlite')
            ledger.check()
            report['restore_c']=import_snapshot(report['snapshot']['receipt_path'],restored,
                expected_source_sha256=digest(spec),expected_snapshot_id=c['snapshot_id'],min_generation=c['generation'],known_anchor=receipt)
            check=ArchiveIndex(restored,PILOT_INDEX_LIMITS,runtime=runtime);restored_state,_=check._snapshot()
            _prefix(old_rows,restored_state['members'])
            if restored_state!=index._snapshot()[0]:raise IndexFailure('extension_restored_state_changed')
            # 실제 새 handle의 새 JSON seek를 먼저 확인한다. 다른 파일의 read-ahead로 대체하지 않는다.
            report['cold_cache_clear']={'resident_bytes_before':runtime.cache_bytes,'phase_requests_before':runtime.transfer.ranges,
                'phase_body_before':runtime.transfer.transfer_bytes,'phase_clock_started':runtime.transfer.started}
            runtime.cache.clear();runtime.cache_bytes=0
            base=workspace/(run_id+'-C-selected');new_output=base/'new';old_output=base/'old'
            new_report=ledger.call(check,'extract',[new['name']],new_output);_json_files(new_report,new_output,[new])
            read_min=min(t['start'] for t in new_report['stats']['source_read_trace'])
            report['cold_new_extraction']=new_report
            if read_min<=old_max_compressed or abs(read_min-restart[1])>1:raise IndexFailure('extension_new_restart_not_used')
            old_report=ledger.call(check,'extract',[r['name'] for r in anchor['selected']],old_output)
            _json_files(old_report,old_output,anchor['selected']);ledger.check()
            report['verification_extractions']={'new':new_report,'old':old_report}
            report['extraction']={'schema':'extension_verified_files_v1','files':[
                {**r,'relative_path':'new/'+r['relative_path']} for r in new_report['files']]+[
                {**r,'relative_path':'old/'+r['relative_path']} for r in old_report['files']]}
            report.update(status='passed',after=evidence(check),old_catalog_exact_prefix=True,
                persisted_index_validation={'auto_build':False,'cold_new_member_first':True,'shared_runtime':True,
                    'snapshot_id':c['snapshot_id'],'new_read_min':read_min,'expected_new_compressed_point':restart[1],
                    'old_max_compressed_point':old_max_compressed,'bit_alignment_tolerance':1})
    except Exception as exc:
        reason=exc.reason if isinstance(exc,IndexFailure) else 'extension_operation_failed'
        report.update(status='partial' if reason in bounded and 'snapshot' not in report else 'failed',error=reason)
    finally:
        if index is not None:
            try:report['latest_saved_checkpoint']=evidence(index)
            except Exception:report['latest_saved_checkpoint_unavailable']=True
        if ledger is not None:report['phase_resources']=ledger.report()
        if runtime is not None and hasattr(runtime,'transfer'):report['transport']=runtime.stats()
        if session is not None:
            try:session.close()
            except Exception:report['session_close_failed']=True
        marker.write_bytes(canonical(report)+b'\n')
    return report
