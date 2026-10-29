"""명시적 CPU A/B pilot. 실패/완료 marker를 소비하며 자동 반복하지 않는다."""
import hashlib
import json
from pathlib import Path
import re
import uuid

from .core import ArchiveIndex
from .source import IndexFailure,IndexLimits,canonical,digest,MIB
from .drive import DriveRuntime,DriveLimits,make_drive_source
from .snapshots import export_snapshot,import_snapshot
from rewind_archive.contracts import safe_path

RUNTIME_ID=uuid.uuid4().hex
PILOT_INDEX_LIMITS=IndexLimits(max_read_bytes=32*MIB,max_decoded_bytes=32*MIB,max_seconds=300.)
PILOT_DRIVE_LIMITS=DriveLimits(max_body_bytes=8*MIB,max_requests=120,max_seconds=300.)


def run_phase(*,phase,run_id,workspace,snapshot_directory,manifest,session_factory,anchor=None):
    if phase not in ('A','B') or not isinstance(run_id,str) or not re.fullmatch('[A-Za-z0-9_-]{1,64}',run_id):raise IndexFailure('pilot_phase_or_id')
    workspace=Path(workspace).absolute();workspace.mkdir(parents=True,exist_ok=True)
    if str(workspace).replace('\\','/').startswith('/content/drive/'):raise IndexFailure('live_db_must_be_local')
    marker=workspace/(run_id+'-'+phase+'.json')
    if marker.exists():raise IndexFailure('pilot_run_already_consumed')
    spec=make_drive_source(manifest)
    if phase=='B':
        if not isinstance(anchor,dict) or set(anchor)!={'schema','runtime_id','receipt','selected'} or anchor['schema']!='persistent_drive_pilot_anchor_v1':raise IndexFailure('pilot_anchor_required')
        if anchor['runtime_id']==RUNTIME_ID:raise IndexFailure('pilot_fresh_runtime_required')
        if anchor['receipt']['source_sha256']!=digest(spec):raise IndexFailure('pilot_source_mismatch')
        if not isinstance(anchor['selected'],list) or not 1<=len(anchor['selected'])<=2:raise IndexFailure('pilot_selected_anchor')
        for row in anchor['selected']:
            if not isinstance(row,dict) or set(row)!={'name','sha256'} or not isinstance(row['sha256'],str) or not re.fullmatch('[a-f0-9]{64}',row['sha256']):raise IndexFailure('pilot_selected_anchor')
            safe_path(row['name'])
    # 인증과 실제 scan은 이 marker 뒤에만 수행하며 실패해도 지우지 않는다.
    with marker.open('x',encoding='utf-8') as f:json.dump({'status':'started','run_id':run_id,'phase':phase},f)
    report={'schema':'persistent_drive_pilot_report_v1','run_id':run_id,'phase':phase,'status':'started',
        'index_limits':vars(PILOT_INDEX_LIMITS),'drive_limits':vars(PILOT_DRIVE_LIMITS),'full_archive_crc_requested':False}
    session=None;runtime=None
    try:
        if ArchiveIndex._engine_version()!='1.10.3':raise IndexFailure('pilot_engine_1_10_3_required')
        db=workspace/(run_id+'-'+phase+'.sqlite')
        if phase=='B':
            receipt=anchor['receipt'];receipt_path=Path(snapshot_directory)/(receipt['filename']+'.receipt.json')
            report['restore']=import_snapshot(receipt_path,db,expected_source_sha256=digest(spec),
                expected_snapshot_id=receipt['snapshot_id'],min_generation=receipt['generation'],known_anchor=receipt,
                index_limits=PILOT_INDEX_LIMITS)
        session=session_factory();runtime=DriveRuntime(session,PILOT_DRIVE_LIMITS)
        index=ArchiveIndex(db,PILOT_INDEX_LIMITS,runtime=runtime)
        if phase=='A':index.create(spec)
        report['before']=index.status();report['scan']=index.scan(16,verify_end=False)
        if report['scan'].get('checkpoint_deferred') or report['scan']['members']<=report['before']['members']:raise IndexFailure('pilot_no_saved_progress')
        if report['scan'].get('new_members')!=16:raise IndexFailure('pilot_fixed_batch_incomplete')
        state,_=index._snapshot()
        if phase=='A':
            rows=[row for row in state['members'] if row['type']=='file' and row['name'].endswith('.json')][:2]
            if not rows:raise IndexFailure('pilot_no_json_in_fixed_batch')
            selected=[{'name':row['name'],'sha256':row['sha256']} for row in rows]
        else:selected=anchor['selected']
        report['extraction']=index.extract([r['name'] for r in selected],workspace/(run_id+'-'+phase+'-selected'))
        actual={r['name']:r['sha256'] for r in report['extraction']['files']}
        if any(actual.get(r['name'])!=r['sha256'] for r in selected):raise IndexFailure('pilot_saved_member_changed')
        for row in report['extraction']['files']:
            payload=(workspace/(run_id+'-'+phase+'-selected')/row['relative_path']).read_bytes()
            if hashlib.sha256(payload).hexdigest()!=row['sha256']:raise IndexFailure('pilot_json_hash_changed')
            if not isinstance(json.loads(payload),dict):raise IndexFailure('pilot_json_object_required')
        report['snapshot']=export_snapshot(index,snapshot_directory,known_anchor=anchor['receipt'] if phase=='B' else None)
        saved={'schema':'persistent_drive_pilot_anchor_v1','runtime_id':RUNTIME_ID,
            'receipt':report['snapshot']['receipt'],'selected':selected}
        anchor_path=workspace/(run_id+'-'+phase+'-anchor.json');anchor_path.write_bytes(canonical(saved)+b'\n')
        report.update(status='passed',anchor_path=str(anchor_path),anchor_sha256=hashlib.sha256(anchor_path.read_bytes()).hexdigest(),
            selected=selected,after=index.status())
    except Exception as exc:
        report.update(status='failed',error=exc.reason if isinstance(exc,IndexFailure) else 'pilot_operation_failed')
    finally:
        if runtime is not None and hasattr(runtime,'transfer'):report['transport']=runtime.stats()
        if session is not None:
            try:session.close()
            except Exception:report['session_close_failed']=True
        marker.write_bytes(canonical(report)+b'\n')
    return report
