"""명시 1-job 누적 budget, batch별 immutable snapshot/receipt와 실패 회수."""
from dataclasses import asdict,dataclass,replace
import hashlib
import json
from pathlib import Path
import re
import time

from rewind_archive.contracts import canonical,digest
from rewind_archive.drive_metadata import build_drive_manifest,MetadataLimits,validate_selection
from rewind_archive.matching import identify
from rewind_archive_index.core import ArchiveIndex
from rewind_archive_index.drive import DriveRuntime,DriveLimits,make_drive_source
from rewind_archive_index.extension_records import atomic_record
from rewind_archive_index.snapshots import export_snapshot,import_snapshot,SnapshotLimits,_path
from rewind_archive_index.source import IndexFailure,IndexLimits,MIB
from rewind_archive.request_policy import validate_no_hidden_retries
from .timing import Timing

PREINDEX_PROFILE='archvis_preindex512_v1'


@dataclass(frozen=True)
class DiscoveryLimits:
    body_bytes:int=256*MIB
    requests:int=2048
    read_bytes:int=512*MIB
    decoded_bytes:int=256*MIB
    seconds:int=600
    rss_bytes:int=1024*MIB
    disk_bytes:int=256*MIB
    batch_members:int=32
    batches:int=16
    reserve_seconds:int=30

    def __post_init__(self):
        maxima={'body_bytes':256*MIB,'requests':2048,'read_bytes':512*MIB,'decoded_bytes':256*MIB,
                 'seconds':600,'rss_bytes':1024*MIB,'disk_bytes':256*MIB,'batch_members':32,'batches':16,'reserve_seconds':30}
        if any(type(v) is not int or not 1<=v<=maxima[k] for k,v in asdict(self).items()) or self.reserve_seconds>=self.seconds:
            raise IndexFailure('discovery_limits')


@dataclass(frozen=True)
class PreindexLimits(DiscoveryLimits):
    body_bytes:int=512*MIB
    requests:int=4096
    read_bytes:int=1024*MIB
    decoded_bytes:int=512*MIB
    seconds:int=1200
    disk_bytes:int=768*MIB
    batches:int=32

    def __post_init__(self):
        maxima={'body_bytes':512*MIB,'requests':4096,'read_bytes':1024*MIB,'decoded_bytes':512*MIB,
                 'seconds':1200,'rss_bytes':1024*MIB,'disk_bytes':768*MIB,'batch_members':32,'batches':32,'reserve_seconds':30}
        if any(type(v) is not int or not 1<=v<=maxima[k] for k,v in asdict(self).items()) or self.reserve_seconds>=self.seconds:
            raise IndexFailure('preindex_limits')


def check(value,reason):
    if not value:raise IndexFailure(reason)


def preindex_storage_plan(limits,index_limits,db_bytes,other_bytes,last_snapshot_bytes):
    """동시 peak를 계산한다. DB 실크기를 core checkpoint peak 한도로 쓰지 않는다."""
    snap=SnapshotLimits()
    checkpoint=3*(db_bytes+index_limits.max_state_bytes+index_limits.max_index_bytes+16384)
    snapshot=2*snap.max_db_bytes+2*snap.max_bundle_bytes
    recovery=max(last_snapshot_bytes,snap.max_bundle_bytes)
    available=limits.disk_bytes-other_bytes-recovery-4*MIB
    return {'current_db_bytes':db_bytes,'other_persistent_bytes':other_bytes,
        'checkpoint_peak_bound_bytes':checkpoint,'snapshot_peak_bound_bytes':snapshot,
        'snapshot_db_bound_bytes':snap.max_db_bytes,'snapshot_zip_bound_bytes':snap.max_bundle_bytes,
        'recovery_reserve_bytes':recovery,'report_reserve_bytes':4*MIB,
        'checkpoint_budget_bytes':max(0,available),
        'required_total_bytes':other_bytes+max(checkpoint,snapshot)+recovery+4*MIB,
        'scope':'separate checkpoint and snapshot peaks; reserve latest recovery and persistent files within total disk; no deletion'}


def load(path,maximum):
    path=_path(path)
    check(path.is_file() and path.stat().st_size<=maximum,'discovery_input_limit')
    with path.open('rb') as stream:raw=stream.read(maximum+1)
    check(len(raw)<=maximum,'discovery_input_limit')
    return json.loads(raw),hashlib.sha256(raw).hexdigest()


def input_keys(split_paths,split_pins,pose_path,pose_sha):
    owners={}
    for split in ('train','validation','test'):
        path=_path(split_paths[split]);check(path.stat().st_size<=4*MIB,'discovery_split_limit')
        raw=path.read_bytes();check(hashlib.sha256(raw).hexdigest()==split_pins[split],'discovery_split_pin')
        for seq in raw.decode('utf-8-sig').splitlines():
            check(seq not in owners,'discovery_split_duplicate');owners[seq]=split
    pose,pin=load(pose_path,6*MIB)
    check(pin==pose_sha and pose.get('content_sha256')==digest({k:v for k,v in pose.items() if k!='content_sha256'}) and
          pose.get('seek_index') is False and pose.get('full_source_listing_claimed') is False,'discovery_pose_observation_pin')
    check(len(pose['entries'])<=16000 and len(pose['sequences'])<=256,'discovery_pose_catalog_limit')
    keys=set()
    for row in pose['entries']:
        if row['kind']=='pose' and row['payload_complete'] is True:
            check(type(row['frame']) is int and row['frame']>=0,'discovery_pose_key')
            keys.add((pose['sequences'][row['sequence_ref']],row['frame']))
    return owners,keys,{'pose_observed_complete_keys':len(keys),'pose_catalog_truncated':pose['truncated'],
                       'pose_seek_index':False,'pose_catalog_sha256':pin}


def counts(state,owners,pose_keys):
    keys={'left':set(),'right':set()};other=0
    for row in state['members']:
        if row['type']!='file':continue
        try:seq,frame,side=identify(row['name'],'rgb')
        except ValueError:other+=1;continue
        keys[side].add((seq,frame))
    stereo=keys['left']&keys['right']
    groups={**keys,'stereo':stereo,'left_pose_observed':keys['left']&pose_keys,
            'right_pose_observed':keys['right']&pose_keys,'stereo_pose_observed':stereo&pose_keys}
    return {'keys':{name:{split:sum(owners.get(s,'unknown')==split for s,f in values)
                         for split in ('train','validation','test','unknown')} for name,values in groups.items()},
            'candidate_examples':[list(k) for k in sorted(stereo&pose_keys)[:32]],'other_complete_files':other,
            'actual_extracted_triplets':0,'png_decode_verified':False,'train_ready':False}


def run(*,job_id,selection,expected_metadata,split_paths,split_pins,pose_catalog,pose_sha256,
        workspace,snapshot_directory,session_factory,limits=None,resume=None,profile=None,runtime_cache_bytes=4*MIB):
    limits=limits or DiscoveryLimits();validate_selection(selection)
    preindex=profile==PREINDEX_PROFILE
    check((preindex and type(limits) is PreindexLimits and resume is not None) or
          (profile is None and type(limits) is DiscoveryLimits),'discovery_explicit_profile_limits')
    check(type(runtime_cache_bytes) is int and runtime_cache_bytes in ((4*MIB,16*MIB) if preindex else (4*MIB,)),
          'discovery_runtime_cache_policy')
    if preindex:check(resume['anchor']['bytes']+4*MIB<=limits.disk_bytes,'discovery_recovery_base_budget')
    check(selection['kind']=='rgb','discovery_rgb_only')
    check(isinstance(job_id,str) and re.fullmatch('[A-Za-z0-9_-]{1,64}',job_id),'discovery_job_id')
    workspace=_path(workspace);snapshots=_path(snapshot_directory)
    check(not str(workspace).replace('\\','/').startswith('/content/drive/'),'discovery_live_local_only')
    check(not workspace.is_relative_to(snapshots) and not snapshots.is_relative_to(workspace),'discovery_separate_storage')
    # 새 job dir 자체가 재실행 소비 marker다. 새 phase에는 명시 새 경로/anchor가 필요하다.
    workspace.mkdir(parents=True,exist_ok=False);snapshots.mkdir(parents=True,exist_ok=False)
    report={'schema':'bounded_rgb_discovery_v1','job_id':job_id,'status':'started','train_ready':False,
            'limits':asdict(limits),'selection_sha256':digest(selection),'events':[],'last_receipt':None,
            'archive_complete_claimed':False,'actual_extracted_triplets':0}
    path=workspace/'report.json';atomic_record(path,report,initial=True)
    started=time.monotonic();runtime=None;session=None;read_bytes=decoded=0;meta_body=meta_requests=0
    timing=Timing() if preindex else None
    if preindex:report.update(profile=profile,scan_attempts=[],cumulative_cache={'hits':0,'misses':0,'read_bytes':0},
        durable_discovered=None,working_discovered=None,runtime_cache_bytes=runtime_cache_bytes,runtime_range_bytes=MIB)
    def measured(name,function,*args,**kwargs):
        return timing.call(name,function,*args,**kwargs) if timing is not None else function(*args,**kwargs)
    last=resume['anchor'] if resume else None
    report['last_receipt']=last
    def save():
        report.update(elapsed_seconds=time.monotonic()-started,read_bytes=read_bytes,decoded_bytes=decoded,
            body_bytes=meta_body+(runtime.transfer.transfer_bytes if runtime and hasattr(runtime,'transfer') else 0),
            requests=meta_requests+(runtime.transfer.ranges if runtime and hasattr(runtime,'transfer') else 0))
        if timing is not None:report['timing']=timing.snapshot()
        atomic_record(path,report)
    def guard(reserve=False):
        import psutil
        rss=psutil.Process().memory_info().rss
        report['observed_peak_rss']=max(report.get('observed_peak_rss',0),rss)
        check(rss<=limits.rss_bytes,'discovery_rss_limit')
        check(time.monotonic()-started < limits.seconds-(limits.reserve_seconds if reserve else 0),'discovery_time_reserve' if reserve else 'discovery_time_limit')
        total=0
        for root in (workspace,snapshots):
            for n,p in enumerate(root.rglob('*')):
                check(n<4096,'discovery_inventory_limit');_path(p)
                if p.is_file():total+=p.stat().st_size
        check(total+MIB<=limits.disk_bytes,'discovery_disk_limit')
        return total
    try:
        owners,pose_keys,pose_report=measured('input_validation',input_keys,split_paths,split_pins,pose_catalog,pose_sha256)
        report.update(pose_observation=pose_report,split_sha256=split_pins)
        check(len(expected_metadata)==len(selection['parts']),'discovery_expected_metadata')
        guard(True);session=measured('session_factory',session_factory)
        if timing is not None:
            validate_no_hidden_retries(session)
            check(not hasattr(session,'_rewind_request_timing'),'discovery_timing_hook_conflict')
            session._rewind_request_timing=timing
        metadata_limits=MetadataLimits(max_files=len(selection['parts']),max_requests=len(selection['parts']),
            max_transfer_bytes=min(128*1024,limits.body_bytes),max_elapsed_seconds=min(30,limits.seconds-limits.reserve_seconds))
        metadata=measured('initial_metadata',build_drive_manifest,selection,session,metadata_limits)
        meta_body=metadata['report']['stats']['transfer_bytes'];meta_requests=metadata['report']['stats']['requests']
        atomic_record(workspace/'metadata.json',metadata)
        if metadata['report']['stop_reason']=='interrupted':raise KeyboardInterrupt()
        check(metadata['manifest'] is not None,'discovery_metadata_failed')
        manifest=metadata['manifest']
        for part,expected in zip(manifest['parts'],expected_metadata):
            check(set(expected)=={'index','name','file_id','size_bytes','modified_time'} and
                all(part[k]==expected[k] for k in ('index','name','size_bytes')) and
                part['source']['file_id']==expected['file_id'] and part['identity']['modified_time']==expected['modified_time'],
                'discovery_source_metadata_changed')
        spec=make_drive_source(manifest);source_sha=digest(spec)
        atomic_record(workspace/'manifest.json',manifest);report['source_sha256']=source_sha
        check(meta_body<limits.body_bytes and meta_requests<limits.requests,'discovery_metadata_budget_exhausted')
        runtime=DriveRuntime(session,DriveLimits(max_body_bytes=limits.body_bytes-meta_body,
            max_requests=limits.requests-meta_requests,max_seconds=max(.001,limits.seconds-(time.monotonic()-started)-limits.reserve_seconds),
            range_bytes=MIB,max_cache_bytes=runtime_cache_bytes))
        index_limits=IndexLimits(max_read_bytes=limits.read_bytes,max_decoded_bytes=limits.decoded_bytes,
            max_rss_bytes=limits.rss_bytes,max_disk_bytes=limits.disk_bytes,max_seconds=limits.seconds)
        db=workspace/'index.sqlite'
        if resume:
            check(set(resume)=={'receipt_path','receipt_file_sha256','anchor'},'discovery_resume_fields')
            receipt,receipt_sha=load(resume['receipt_path'],16384)
            check(receipt_sha==resume['receipt_file_sha256'] and receipt==last,'discovery_trusted_receipt_pin')
            report['last_receipt_path']=str(_path(resume['receipt_path']))
            used=guard(True)
            report['restore']=measured('snapshot_restore',import_snapshot,resume['receipt_path'],db,expected_source_sha256=source_sha,
                expected_snapshot_id=last['snapshot_id'],min_generation=last['generation'],known_anchor=last,index_limits=index_limits,
                limits=SnapshotLimits(max_disk_bytes=max(1,limits.disk_bytes-used-MIB),
                    max_seconds=max(.001,limits.seconds-(time.monotonic()-started)-limits.reserve_seconds)))
            guard(True)
        else:ArchiveIndex(db,index_limits).create(spec)
        index=ArchiveIndex(db,index_limits,runtime);report['before']=index.status();save()
        if preindex:
            state,_=index._snapshot()
            report['before_discovered']=counts(state,owners,pose_keys)
            report['discovered']=report['before_discovered']
            report['working_discovered']=report['before_discovered']
            report['durable_discovered']=report['before_discovered']
            report['durable_members']=len(state['members'])
        for batch in range(limits.batches):
            used=guard(True)
            if preindex:
                db_bytes=db.stat().st_size
                plan=preindex_storage_plan(limits,index_limits,db_bytes,used-db_bytes,last['bytes'])
                report['storage_plan']=plan
                check(plan['required_total_bytes']<=limits.disk_bytes,'discovery_scan_snapshot_recovery_reserve')
                # 이 값은 live DB cap이 아니라 기존 core의 journal/new page 포함
                # checkpoint peak 예산이다. 다른 영속 파일·회수 공간을 먼저 뺀다.
                index_limits=replace(index_limits,max_disk_bytes=plan['checkpoint_budget_bytes'])
            remaining_read=limits.read_bytes-read_bytes;remaining_decode=limits.decoded_bytes-decoded
            check(remaining_read>0 and remaining_decode>0,'discovery_io_exhausted')
            state,_=index._snapshot();previous=len(state['members']);generation=state['generation']
            check(previous<index_limits.max_members,'discovery_catalog_limit')
            index.limits=replace(index_limits,max_read_bytes=remaining_read,max_decoded_bytes=remaining_decode,
                max_seconds=max(.001,limits.seconds-(time.monotonic()-started)-limits.reserve_seconds))
            report['active_stage']='scan';save();prior_operation=runtime.operation
            attempt=None
            if preindex:
                attempt={'batch':batch+1,'status':'started'};report['scan_attempts'].append(attempt)
                before_timing=timing.snapshot()
            result=None
            try:result=measured('scan',index.scan,min(limits.batch_members,index_limits.max_members-previous),expected_generation=generation,verify_end=False)
            finally:
                if runtime.operation is not None and runtime.operation is not prior_operation:
                    stats=runtime.operation.stats();read_bytes+=stats['source_read_bytes'];decoded+=stats['returned_decoded_bytes']
                    if preindex:
                        attempt['stats']={k:v for k,v in stats.items() if k!='source_read_trace'}
                        report['observed_peak_rss']=max(report.get('observed_peak_rss',0),stats['observed_peak_rss'])
                        for dest,source in (('hits','remote_cache_hits'),('misses','remote_cache_misses'),('read_bytes','remote_cache_read_bytes')):
                            report['cumulative_cache'][dest]+=stats.get(source,0)
                if preindex:
                    after_timing=timing.snapshot()
                    call_delta={}
                    for name in ('http_request_to_headers','http_body_read1'):
                        a=before_timing['calls'].get(name,{});b=after_timing['calls'].get(name,{})
                        call_delta[name]={k:b.get(k,0)-a.get(k,0) for k in ('calls','failed_calls','wall_seconds','process_cpu_seconds')}
                    attempt.update(status='returned' if result is not None else 'failed_or_interrupted',
                        timing_delta={**{k:after_timing[k]-before_timing[k] for k in ('wall_seconds','process_cpu_seconds')},
                            'http_calls':call_delta})
                    save()
            check(read_bytes<=limits.read_bytes and decoded<=limits.decoded_bytes,'discovery_cumulative_io_limit')
            current,_=index._snapshot()
            check(current['members'][:previous]==state['members'] and current['source_sha256']==source_sha,'discovery_saved_prefix_changed')
            event={'batch':batch+1,'new_members':len(current['members'])-previous,'generation':current['generation'],
                   'boundary':current['boundary'],'index_coverage':current['index_coverage'],'stop_reason':result.get('stop_reason'),
                   'checkpoint_deferred':bool(result.get('checkpoint_deferred'))}
            old_counts=report.get('discovered')
            report['events'].append(event);report['discovered']=counts(current,owners,pose_keys)
            if preindex:
                report['working_discovered']=report['discovered']
                event['new_keys_by_split']={k:{s:n-old_counts['keys'][k][s] for s,n in values.items()}
                    for k,values in report['discovered']['keys'].items()}
                report['validation_rgb_stereo_found']=report['discovered']['keys']['stereo']['validation']>0
            if current['generation']>generation:
                report['active_stage']='snapshot';save();used=guard()
                if preindex:check(db.stat().st_size<=SnapshotLimits().max_db_bytes,'snapshot_db_limit')
                check(used+3*db.stat().st_size+MIB<=limits.disk_bytes,'discovery_snapshot_disk_reserve')
                remaining=limits.seconds-(time.monotonic()-started)
                check(remaining>1,'discovery_snapshot_time_reserve')
                snap=measured('snapshot_export',export_snapshot,index,snapshots,known_anchor=last,
                    limits=SnapshotLimits(max_disk_bytes=limits.disk_bytes,max_seconds=remaining))
                last=snap['receipt'];report['last_receipt']=last;event['receipt']=last
                if preindex:
                    event['snapshot']=snap
                    event['snapshot_timing']={k:timing.calls['snapshot_export']['last_'+k]
                        for k in ('wall_seconds','process_cpu_seconds')}
                    report['durable_members']=len(current['members'])
                    report['durable_discovered']=report['working_discovered']
                report['last_receipt_path']=snap['receipt_path'];save()
                check(guard()+last['bytes']+MIB<=limits.disk_bytes,'discovery_recovery_disk_reserve')
            if result.get('checkpoint_deferred') or not event['new_members'] or current['catalog_complete']:
                report['status']='bounded_stop';break
        else:report['status']='completed_requested_batches'
        report['after']=index.status();report['active_stage']='finished';save();guard()
    except BaseException as exc:
        reason=getattr(exc,'reason',None)
        bounded=preindex and reason in {'discovery_snapshot_disk_reserve','discovery_recovery_disk_reserve',
            'discovery_disk_limit','discovery_time_reserve','discovery_time_limit','discovery_io_exhausted',
            'snapshot_disk_limit','snapshot_db_limit','snapshot_bundle_limit','discovery_snapshot_time_reserve',
            'discovery_catalog_limit','index_points_limit','index_bytes_limit','checkpoint_size_limit',
            'discovery_scan_snapshot_recovery_reserve','checkpoint_disk_limit'}
        report.update(status='bounded_stop' if bounded else 'interrupted' if isinstance(exc,KeyboardInterrupt) else 'failed',
            failure={'type':type(exc).__name__,'reason':reason if isinstance(reason,str) and re.fullmatch('[a-zA-Z0-9_:-]{1,128}',reason) else 'discovery_operation_failed'})
        save()
        if not isinstance(exc,Exception):raise
    finally:
        if session is not None:
            if timing is not None and getattr(session,'_rewind_request_timing',None) is timing:
                del session._rewind_request_timing
            try:session.close()
            except Exception:report['session_close_failed']=True;save()
        if preindex:save()
    return report
