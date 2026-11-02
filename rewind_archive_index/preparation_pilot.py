"""B 저장본에서 시작하는 자동 prepare 통합 시험과 원본 읽기 없는 결과 회수."""
from dataclasses import asdict
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
import zipfile

from rewind_archive.contracts import Limits,canonical,digest,safe_path
from rewind_archive.extraction import prepare
from rewind_archive.matching import verify_extraction
from .core import ArchiveIndex
from .drive import make_drive_source
from .preparation import PreparationContext,_path
from .extension_records import (atomic_record,exception_kind,validate_provenance,validate_run,
    _read,_check_bundle,FILE_LIMIT,RECORD_LIMIT,BUNDLE_LIMIT,FILE_COUNT_LIMIT)
from .snapshots import _receipt
from .source import IndexFailure,MIB

INTEGRATION_LIMITS=Limits(max_transfer_bytes=8*MIB,max_ranges=120,max_elapsed_seconds=300.,
    max_decompressed_bytes=32*MIB,max_file_bytes=8*MIB,max_files=3,max_members=128,max_disk_bytes=32*MIB)


def run_preparation_integration(*,run_id,workspace,input_snapshot_directory,output_snapshot_directory,
                                manifest,anchor,targets,provenance,session_factory):
    validate_run(run_id);pins=validate_provenance(provenance)
    if len(run_id)>54:raise IndexFailure('prepare_pilot_run_id_length')
    spec=make_drive_source(manifest);receipt=_receipt(anchor['receipt'])
    if receipt['source_sha256']!=digest(spec):raise IndexFailure('prepare_anchor_source')
    if not isinstance(targets,list) or len(targets)!=3:raise IndexFailure('prepare_pilot_three_targets')
    for row in targets:
        if not isinstance(row,dict) or set(row)!={'name','sha256'} or not isinstance(row['sha256'],str) or not re.fullmatch('[a-f0-9]{64}',row['sha256']):
            raise IndexFailure('prepare_pilot_targets')
        if safe_path(row['name'])!=row['name']:raise IndexFailure('prepare_pilot_targets')
    expected={row['name']:row['sha256'] for row in targets}
    if len(expected)!=3 or any(expected.get(row['name'])!=row['sha256'] for row in anchor['selected']):raise IndexFailure('prepare_pilot_old_targets')
    work=_path(workspace,local=True);work.mkdir(parents=True,exist_ok=True)
    incoming=_path(input_snapshot_directory);outgoing=_path(output_snapshot_directory)
    if incoming==outgoing or outgoing.exists():raise IndexFailure('prepare_pilot_new_snapshot_directory_required')
    marker=work/(run_id+'-auto.json')
    report={'schema':'archive_prepare_integration_v1','run_id':run_id,'phase':'C','status':'started',
        'execution_metadata':pins,'source_sha256':digest(spec),'limits':asdict(INTEGRATION_LIMITS),'targets':targets,
        'transport_recording':'not_recorded','transport':{'status':'unknown'},'stages':[],'train_ready':False}
    atomic_record(marker,report,initial=True)
    session=None;context=None;primary=None
    def mark(name,edge):
        value={'name':name,'edge':edge,'at_utc':datetime.now(timezone.utc).isoformat()}
        report['stage']=value;report['stages'].append(value);atomic_record(marker,report)
        print({'stage':name,'edge':edge,'status':report['status']})
    def checked(path):
        result=verify_extraction(path,manifest)
        if not result['targets_complete'] or {r['member']:r['sha256'] for r in result['files']}!=expected:
            raise IndexFailure('prepare_pilot_target_hash')
        context.resources.check()
        return result
    try:
        mark('initial','pre')
        if ArchiveIndex._engine_version()!='1.10.3':raise IndexFailure('pilot_engine_1_10_3_required')
        mark('initial','post');mark('session_factory','pre')
        session=session_factory();context=PreparationContext(INTEGRATION_LIMITS,session)
        mark('session_factory','post')
        options={'workspace':str(work/'live-first'),'input_snapshot_directory':str(incoming),'snapshot_directory':str(outgoing),
                 'anchor':receipt,'batch_members':16,'max_batches':8,'spacing':MIB,'request_id':run_id+'-first','provenance':pins}
        names=[row['name'] for row in targets]
        mark('prepare','pre')
        first=prepare(manifest,work/'first',names,INTEGRATION_LIMITS,exact_targets=names,
                      persistent_index=options,persistent_context=context)
        report['first_status']=first['persistent_index']['status'];report['first_record']=first['persistent_index']['record_path']
        if not first['targets_complete']:
            report['status']='partial' if first['inspection_candidates_usable'] else 'failed'
        else:
            checked(work/'first');initial=first['persistent_index'];before=initial['before'];after=initial['after']
            if before['generation']!=receipt['generation'] or not initial['scans'] or after['generation']<=before['generation'] or after['boundary']<=before['boundary'] or after['index_coverage']<=before['index_coverage']:
                raise IndexFailure('prepare_pilot_extension_contract')
            report['first_verified']=True;mark('prepare','post')
            mark('restore_prepare','pre')
            # 같은 Runtime의 cache만 비우고 전송·시각·read ledger는 유지한다.
            context.runtime.cache.clear();context.runtime.cache_bytes=0
            second=prepare(manifest,work/'restored',names,INTEGRATION_LIMITS,exact_targets=names,
                persistent_index={**options,'workspace':str(work/'live-restored'),'request_id':run_id+'-restored'},persistent_context=context)
            report['restored_status']=second['persistent_index']['status'];report['restored_record']=second['persistent_index']['record_path']
            checked(work/'restored')
            if second['persistent_index']['scans'] or 'restore' not in second['persistent_index']:raise IndexFailure('prepare_pilot_restore_contract')
            if second['persistent_index']['restore']['receipt']!=initial['head']['receipts'][-1] or second['persistent_index']['before']['generation']!=after['generation']:
                raise IndexFailure('prepare_pilot_restore_receipt')
            report['restored_verified']=True;mark('restore_prepare','post');mark('reuse','pre')
            previous=(context.runtime.transfer.ranges,context.runtime.transfer.transfer_bytes,context.resources.read,context.resources.decoded)
            reused=prepare(manifest,work/'first',names,INTEGRATION_LIMITS,exact_targets=names,
                persistent_index=options,persistent_context=context)
            if not reused.get('reuse',{}).get('verified') or previous!=(context.runtime.transfer.ranges,context.runtime.transfer.transfer_bytes,context.resources.read,context.resources.decoded):
                raise IndexFailure('prepare_pilot_reuse_contract')
            context.resources.check()
            report['reuse']=reused['request_stats'];report['verification_status']='passed';mark('reuse','post')
    except BaseException as exc:
        primary=exc;reason=getattr(exc,'reason','prepare_integration_failed')
        if not isinstance(reason,str) or not re.fullmatch('[a-z0-9_]{1,100}',reason):reason='prepare_integration_failed'
        report.update(status='failed' if isinstance(exc,Exception) else 'interrupted',error=reason,
                      failure_stage=report.get('stage'),exception_type=exception_kind(exc))
        if not isinstance(exc,Exception):raise
    finally:
        cleanup=None
        def collect():
            if context is not None:
                report['resources']=context.resources.report()
                if context.runtime is not None and hasattr(context.runtime,'transfer'):
                    report['transport']=context.runtime.stats();report['transport_recording']='recorded'
        for operation in (lambda:mark('cleanup','pre'),collect,session.close if session is not None else lambda:None):
            try:operation()
            except BaseException as exc:cleanup=cleanup or exc
        if cleanup is not None:
            report['cleanup_exception']=exception_kind(cleanup)
            if primary is None:report['status']='failed' if isinstance(cleanup,Exception) else 'interrupted'
        elif report.get('verification_status')=='passed':report['status']='passed'
        try:mark('cleanup','post')
        except BaseException as exc:
            cleanup=cleanup or exc;report['cleanup_exception']=exception_kind(exc)
            if primary is None:report['status']='failed' if isinstance(exc,Exception) else 'interrupted'
        try:atomic_record(marker,report)
        except BaseException as exc:cleanup=cleanup or exc
        if primary is None and cleanup is not None and not isinstance(cleanup,Exception):raise cleanup
    return report


def recover_preparation_integration(*,workspace,run_id,provenance):
    """고정 run/pin의 디스크 JSON과 검증된 선택 파일만 회수한다."""
    validate_run(run_id);pins=validate_provenance(provenance);work=_path(workspace,local=True)
    bundle=work/(run_id+'-auto-report.zip')
    if bundle.exists():
        _check_bundle(bundle,work,run_id,pins,
            allowed_names={'report.json','first-extraction.json','restored-extraction.json','first-request.json','restored-request.json'},
            selected_prefixes=('selected/files/',))
        return {'path':str(bundle),'reused':True}
    marker=work/(run_id+'-auto.json');contents={}
    try:
        report=json.loads(_read(marker,work,RECORD_LIMIT))
        if report.get('run_id')!=run_id or report.get('phase')!='C' or report.get('execution_metadata')!=pins:raise IndexFailure('recovery_provenance_mismatch')
        if report.get('schema')!='archive_prepare_integration_v1' or report.get('status') not in {'started','partial','failed','interrupted','passed'}:raise ValueError()
        stages=report.get('stages')
        if not isinstance(stages,list) or len(stages)>32:raise ValueError()
        for value in stages:
            if not isinstance(value,dict) or value.get('name') not in {'initial','session_factory','prepare','restore_prepare','reuse','cleanup'} or value.get('edge') not in {'pre','post'}:
                raise ValueError()
        if report['status']=='passed' and not (report.get('first_verified') is True and report.get('restored_verified') is True and report.get('verification_status')=='passed'):
            raise ValueError()
        # 생성기가 쓰는 필드만 선택한다. 원시 예외/credential 확장 필드는 복사하지 않는다.
        allowed={'schema','run_id','phase','status','execution_metadata','source_sha256','limits','targets','transport_recording',
                 'transport','stages','train_ready','stage','first_status','restored_status','first_verified','restored_verified',
                 'reuse','verification_status','failure_stage','exception_type','error','cleanup_exception','resources'}
        report={key:value for key,value in report.items() if key in allowed}
    except (OSError,ValueError,AttributeError,RecursionError) as exc:
        if isinstance(exc,IndexFailure) and exc.reason=='recovery_provenance_mismatch':raise
        report={'schema':'archive_prepare_integration_v1','run_id':run_id,'phase':'C','status':'offline_diagnostic',
                'execution_metadata':pins,'reason':'marker_missing_or_invalid','train_ready':False}
    for label in (() if report['status']=='offline_diagnostic' else ('first','restored')):
        extraction=work/label/'extraction_manifest.json'
        if extraction.is_file():
            data=_read(extraction,work,RECORD_LIMIT);doc=json.loads(data)
            if doc.get('content_sha256')!=digest({k:v for k,v in doc.items() if k!='content_sha256'}):raise IndexFailure('recovery_preparation_report_hash')
            record=doc.get('persistent_index',{})
            if record.get('provenance')!=pins or record.get('source_sha256')!=report.get('source_sha256') or record.get('request_id')!=run_id+'-'+label:
                raise IndexFailure('recovery_provenance_mismatch')
            contents[label+'-extraction.json']=data
        elif report.get('source_sha256'):
            source=report['source_sha256']
            if not isinstance(source,str) or not re.fullmatch('[a-f0-9]{64}',source):raise IndexFailure('recovery_source')
            request=work/('live-first' if label=='first' else 'live-restored')/source/('request-'+run_id+'-'+label+'.json')
            if request.is_file():
                data=_read(request,work,RECORD_LIMIT);value=json.loads(data)
                if value.get('schema')!='archive_index_preparation_v1' or value.get('request_id')!=run_id+'-'+label or value.get('provenance')!=pins or value.get('source_sha256')!=source:
                    raise IndexFailure('recovery_provenance_mismatch')
                contents[label+'-request.json']=data
    if report.get('restored_verified') or report.get('first_verified'):
        label='restored' if report.get('restored_verified') else 'first'
        verified=verify_extraction(work/label)
        if len(verified['files'])!=3:raise IndexFailure('recovery_selected_count')
        for row in verified['files']:
            relative=safe_path(row['relative_path']);data=_read(work/label/relative,work,FILE_LIMIT)
            if hashlib.sha256(data).hexdigest()!=row['sha256']:raise IndexFailure('recovery_selected_hash')
            contents['selected/'+relative]=data
    contents['report.json']=canonical(report)+b'\n'
    info={'schema':'extension_recovery_bundle_v1','run_id':run_id,'phase':'C','execution_metadata':pins,
          'files':{name:{'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()} for name,data in contents.items()}}
    contents['recovery.json']=canonical(info)+b'\n'
    if len(contents)>FILE_COUNT_LIMIT or sum(map(len,contents.values()))>BUNDLE_LIMIT:raise IndexFailure('recovery_bundle_limit')
    fd,name=tempfile.mkstemp(prefix='.prepare-report-',dir=work);os.close(fd)
    try:
        with zipfile.ZipFile(name,'w',zipfile.ZIP_DEFLATED) as archive:
            for relative,data in contents.items():archive.writestr(relative,data)
        if os.name=='nt':os.rename(name,bundle)
        else:os.link(name,bundle)
    finally:Path(name).unlink(missing_ok=True)
    return {'path':str(bundle),'reused':False,'status':report['status']}
