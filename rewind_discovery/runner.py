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


def check(value,reason):
    if not value:raise IndexFailure(reason)


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
        workspace,snapshot_directory,session_factory,limits=None,resume=None):
    limits=limits or DiscoveryLimits();validate_selection(selection)
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
    last=resume['anchor'] if resume else None
    report['last_receipt']=last
    def save():
        report.update(elapsed_seconds=time.monotonic()-started,read_bytes=read_bytes,decoded_bytes=decoded,
            body_bytes=meta_body+(runtime.transfer.transfer_bytes if runtime and hasattr(runtime,'transfer') else 0),
            requests=meta_requests+(runtime.transfer.ranges if runtime and hasattr(runtime,'transfer') else 0))
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
        owners,pose_keys,pose_report=input_keys(split_paths,split_pins,pose_catalog,pose_sha256)
        report.update(pose_observation=pose_report,split_sha256=split_pins)
        check(len(expected_metadata)==len(selection['parts']),'discovery_expected_metadata')
        guard(True);session=session_factory()
        metadata_limits=MetadataLimits(max_files=len(selection['parts']),max_requests=len(selection['parts']),
            max_transfer_bytes=min(128*1024,limits.body_bytes),max_elapsed_seconds=min(30,limits.seconds-limits.reserve_seconds))
        metadata=build_drive_manifest(selection,session,metadata_limits)
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
            range_bytes=MIB,max_cache_bytes=4*MIB))
        index_limits=IndexLimits(max_read_bytes=limits.read_bytes,max_decoded_bytes=limits.decoded_bytes,
            max_rss_bytes=limits.rss_bytes,max_disk_bytes=limits.disk_bytes,max_seconds=limits.seconds)
        db=workspace/'index.sqlite'
        if resume:
            check(set(resume)=={'receipt_path','receipt_file_sha256','anchor'},'discovery_resume_fields')
            receipt,receipt_sha=load(resume['receipt_path'],16384)
            check(receipt_sha==resume['receipt_file_sha256'] and receipt==last,'discovery_trusted_receipt_pin')
            report['last_receipt_path']=str(_path(resume['receipt_path']))
            used=guard(True)
            report['restore']=import_snapshot(resume['receipt_path'],db,expected_source_sha256=source_sha,
                expected_snapshot_id=last['snapshot_id'],min_generation=last['generation'],known_anchor=last,index_limits=index_limits,
                limits=SnapshotLimits(max_disk_bytes=max(1,limits.disk_bytes-used-MIB),
                    max_seconds=max(.001,limits.seconds-(time.monotonic()-started)-limits.reserve_seconds)))
            guard(True)
        else:ArchiveIndex(db,index_limits).create(spec)
        index=ArchiveIndex(db,index_limits,runtime);report['before']=index.status();save()
        for batch in range(limits.batches):
            guard(True)
            remaining_read=limits.read_bytes-read_bytes;remaining_decode=limits.decoded_bytes-decoded
            check(remaining_read>0 and remaining_decode>0,'discovery_io_exhausted')
            state,_=index._snapshot();previous=len(state['members']);generation=state['generation']
            check(previous<index_limits.max_members,'discovery_catalog_limit')
            index.limits=replace(index_limits,max_read_bytes=remaining_read,max_decoded_bytes=remaining_decode,
                max_seconds=max(.001,limits.seconds-(time.monotonic()-started)-limits.reserve_seconds))
            report['active_stage']='scan';save();prior_operation=runtime.operation
            result=None
            try:result=index.scan(min(limits.batch_members,index_limits.max_members-previous),expected_generation=generation,verify_end=False)
            finally:
                if runtime.operation is not None and runtime.operation is not prior_operation:
                    stats=runtime.operation.stats();read_bytes+=stats['source_read_bytes'];decoded+=stats['returned_decoded_bytes']
            check(read_bytes<=limits.read_bytes and decoded<=limits.decoded_bytes,'discovery_cumulative_io_limit')
            current,_=index._snapshot()
            check(current['members'][:previous]==state['members'] and current['source_sha256']==source_sha,'discovery_saved_prefix_changed')
            event={'batch':batch+1,'new_members':len(current['members'])-previous,'generation':current['generation'],
                   'boundary':current['boundary'],'index_coverage':current['index_coverage'],'stop_reason':result.get('stop_reason'),
                   'checkpoint_deferred':bool(result.get('checkpoint_deferred'))}
            report['events'].append(event);report['discovered']=counts(current,owners,pose_keys)
            if current['generation']>generation:
                report['active_stage']='snapshot';save();used=guard()
                check(used+3*db.stat().st_size+MIB<=limits.disk_bytes,'discovery_snapshot_disk_reserve')
                remaining=limits.seconds-(time.monotonic()-started)
                check(remaining>1,'discovery_snapshot_time_reserve')
                snap=export_snapshot(index,snapshots,known_anchor=last,
                    limits=SnapshotLimits(max_disk_bytes=limits.disk_bytes,max_seconds=remaining))
                last=snap['receipt'];report['last_receipt']=last;event['receipt']=last
                report['last_receipt_path']=snap['receipt_path'];save()
                check(guard()+last['bytes']+MIB<=limits.disk_bytes,'discovery_recovery_disk_reserve')
            if result.get('checkpoint_deferred') or not event['new_members'] or current['catalog_complete']:
                report['status']='bounded_stop';break
        else:report['status']='completed_requested_batches'
        report['after']=index.status();report['active_stage']='finished';save();guard()
    except BaseException as exc:
        reason=getattr(exc,'reason',None)
        report.update(status='interrupted' if isinstance(exc,KeyboardInterrupt) else 'failed',
            failure={'type':type(exc).__name__,'reason':reason if isinstance(reason,str) and re.fullmatch('[a-zA-Z0-9_:-]{1,128}',reason) else 'discovery_operation_failed'})
        save()
        if not isinstance(exc,Exception):raise
    finally:
        if session is not None:
            try:session.close()
            except Exception:report['session_close_failed']=True;save()
    return report
