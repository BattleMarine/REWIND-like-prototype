"""원본별 persistent prepare와 영속 전역 예산. 원본·출력 자동 삭제 없음."""
from contextlib import contextmanager
import copy
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import re
import time

from rewind_archive.contracts import Limits, canonical, digest, safe_path, validate_manifest
from rewind_archive.matching import identify, verify_extraction
from rewind_archive_index.extension_records import atomic_record
from rewind_archive_index.preparation import PreparationContext, _path, _source, prepare_indexed
from rewind_archive_index.source import MIB

SCHEMA = 'bounded_preparation_queue_v1'
MAXIMUM = {'read_bytes':32*MIB, 'decoded_bytes':32*MIB, 'members':128,
           'http_requests':120, 'body_bytes':8*MIB, 'elapsed_seconds':300.}
MAX_DISK = 128*MIB


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def integer(value, low, high):
    return type(value) is int and low <= value <= high


def hashed(doc):
    return {**doc, 'content_sha256':digest({k:v for k,v in doc.items() if k != 'content_sha256'})}


def read(path):
    path = _path(path)
    require(path.is_file() and path.stat().st_size <= MIB, 'queue_record_size')
    with path.open('rb') as stream:
        raw = stream.read(MIB+1)
    require(len(raw) <= MIB, 'queue_record_size')
    return json.loads(raw)


def relative(value):
    require(isinstance(value, str) and safe_path(value) == value, 'queue_relative_path')
    return value


def _metrics(value, upper, *, positive=False):
    require(isinstance(value, dict) and set(value) == set(MAXIMUM), 'queue_budget_fields')
    for key, maximum in upper.items():
        v = value[key]
        if key == 'elapsed_seconds':
            require(type(v) in (int,float) and (v > 0 if positive else v >= 0) and v <= maximum, 'queue_budget_range')
        else:
            require(integer(v, 1 if positive else 0, maximum), 'queue_budget_range')


def validate_plan(plan):
    require(isinstance(plan, dict) and set(plan) == {'schema_version','dataset_version','budgets','max_disk_bytes',
            'backend_limits','jobs','content_sha256'} and plan['schema_version'] == SCHEMA and
            hashed(plan)['content_sha256'] == plan['content_sha256'], 'queue_plan_schema_hash')
    _metrics(plan['budgets'], MAXIMUM, positive=True)
    require(integer(plan['max_disk_bytes'], 4*MIB, MAX_DISK), 'queue_disk_limit')
    limits = Limits.from_dict(plan['backend_limits'])
    require(limits.io_chunk_bytes == 65536 and limits.max_metadata_bytes >= 65536,
            'queue_backend_unsupported_io_metadata')
    require(limits.max_elapsed_seconds <= 300 and limits.max_decompressed_bytes <= 32*MIB and
            limits.max_transfer_bytes <= 8*MIB and limits.max_ranges <= 120 and limits.max_disk_bytes <= MAX_DISK,
            'queue_backend_limit_widening')
    jobs = plan['jobs']
    require(isinstance(jobs, list) and 1 <= len(jobs) <= 16, 'queue_job_count')
    ids, keys = set(), {}
    for job in jobs:
        require(isinstance(job,dict) and set(job) == {'id','manifest','source_base','source_sha256','targets','index_options'},
                'queue_job_fields')
        require(isinstance(job['id'],str) and re.fullmatch('[A-Za-z0-9_-]{1,48}',job['id']) and job['id'] not in ids,
                'queue_job_id')
        ids.add(job['id'])
        manifest = validate_manifest(job['manifest'])
        require(manifest['dataset_version'] == plan['dataset_version'], 'queue_dataset_version_mismatch')
        kinds = {p['source']['type'] for p in manifest['parts']}
        require(kinds in ({'local'}, {'drive'}) and len(manifest['parts']) == manifest['total_parts'],
                'queue_complete_local_or_drive_source_required')
        relative(job['source_base'])
        require(isinstance(job['source_sha256'],str) and re.fullmatch('[0-9a-f]{64}',job['source_sha256']),
                'queue_source_identity')
        options = job['index_options']
        require(isinstance(options,dict) and set(options) == {'batch_members','max_batches','spacing'} and
                integer(options['batch_members'],1,16) and integer(options['max_batches'],1,8) and
                integer(options['spacing'],65536,16*MIB), 'queue_index_options')
        targets = job['targets']
        require(isinstance(targets,list) and 1 <= len(targets) <= min(32,limits.max_files), 'queue_explicit_targets')
        for target in targets:
            require(relative(target) == target, 'queue_target_path')
            seq, frame, view = identify(target, manifest['kind'])
            require(view not in keys.setdefault((seq,frame),set()), 'queue_duplicate_frame_view')
            keys[(seq,frame)].add(view)
    require(all(views == {'left','right','pose'} for views in keys.values()), 'queue_incomplete_stereo_pose_keys')
    require(len(canonical(plan)) <= MIB, 'queue_plan_size')
    return copy.deepcopy(plan)


def make_plan(jobs, *, budgets=None, max_disk_bytes=MAX_DISK, backend_limits=None):
    """jobs의 원본 SHA는 호출자가 _source와 같은 고정 source spec에서 계산한다."""
    require(bool(jobs), 'queue_job_count')
    backend = backend_limits or replace(Limits(), max_ranges=120)
    return validate_plan(hashed({'schema_version':SCHEMA, 'dataset_version':jobs[0]['manifest']['dataset_version'],
        'budgets':budgets or dict(MAXIMUM), 'max_disk_bytes':max_disk_bytes,
        'backend_limits':asdict(backend), 'jobs':jobs}))


@contextmanager
def writer(root):
    """OS 단일-host 잠금. 파일을 남기며 프로세스 종료 시 OS가 잠금을 회수한다."""
    path = _path(root/'writer.lock')
    with path.open('a+b') as stream:
        if path.stat().st_size == 0:
            stream.write(b'0'); stream.flush()
        stream.seek(0)
        import os
        if os.name == 'nt':
            import msvcrt
            try: msvcrt.locking(stream.fileno(),msvcrt.LK_NBLCK,1)
            except OSError as exc: raise ValueError('queue_writer_active') from exc
            try: yield
            finally:
                stream.seek(0); msvcrt.locking(stream.fileno(),msvcrt.LK_UNLCK,1)
        else:
            import fcntl
            try: fcntl.flock(stream.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
            except OSError as exc: raise ValueError('queue_writer_active') from exc
            try: yield
            finally: fcntl.flock(stream.fileno(),fcntl.LOCK_UN)


def _save(root, state):
    state['content_sha256'] = hashed(state)['content_sha256']
    atomic_record(root/'state.json', state)


def initialize_queue(plan, root):
    """새 장부를 명시적으로 생성. 잃어버린 state를 run/inspect가 자동 초기화하지 않는다."""
    plan = validate_plan(plan); root = _path(root,local=True)
    root.mkdir(parents=True,exist_ok=False)
    with writer(root):
        atomic_record(root/'plan.json',plan,initial=True)
        return _state(root,plan,create=True)


def _state(root, plan, *, create=False):
    path = root/'state.json'
    if not path.exists():
        require(create, 'queue_state_missing_no_automatic_reset')
        state = {'schema_version':'preparation_queue_state_v1','plan_sha256':plan['content_sha256'],
            'cursor':0,'charged':{k:0 for k in MAXIMUM},'observed':{k:0 for k in MAXIMUM},
            'inflight':None,'next_attempt':0,'status':'pending','train_ready':False,
            'jobs':[{'id':job['id'],'status':'pending','attempts':[],'output':None,'output_manifest_sha256':None,
                     'consumed':None} for job in plan['jobs']],
            'events':[],'matching':None}
        _save(root,state)
        return state
    state = read(path)
    require(read(root/'plan.json') == plan, 'queue_pinned_plan_changed')
    require(state.get('schema_version') == 'preparation_queue_state_v1' and state.get('plan_sha256') == plan['content_sha256']
            and hashed(state)['content_sha256'] == state.get('content_sha256') and state.get('train_ready') is False,
            'queue_state_identity_hash')
    _metrics(state['charged'],plan['budgets']); _metrics(state['observed'],state['charged'])
    require(integer(state['cursor'],0,len(plan['jobs'])) and integer(state['next_attempt'],0,64)
            and [r['id'] for r in state['jobs']] == [r['id'] for r in plan['jobs']], 'queue_state_cursor')
    for i,row in enumerate(state['jobs']):
        require((row['status'] == 'prepared') == (i < state['cursor']), 'queue_state_job_order')
        require(len(row['attempts']) <= 64, 'queue_state_attempt_limit')
        if row['output'] is not None: relative(row['output'])
    return state


def _disk(root, limit, check=lambda:None):
    total = 0
    for i,path in enumerate(root.rglob('*')):
        check()
        require(i < 10000, 'queue_inventory_limit')
        _path(path)
        if path.is_file(): total += path.stat().st_size
        require(total+2*MIB <= limit, 'queue_disk_exhausted_preserved')
    return total


def _reserve(root, plan, state, kind, metrics):
    require(state['inflight'] is None, 'queue_unknown_cost_blocked')
    require(state['next_attempt'] < 64, 'queue_attempt_limit')
    _metrics(metrics, {k:plan['budgets'][k]-state['charged'][k] for k in MAXIMUM})
    require(metrics['elapsed_seconds'] > 0, 'queue_elapsed_exhausted')
    token = state['next_attempt']; state['next_attempt'] += 1
    state['inflight'] = {'id':token,'kind':kind,'reservation':metrics}
    for key,value in metrics.items(): state['charged'][key] += value
    state['status'] = 'running'
    _save(root,state)
    return token


def _settle(root, plan, state, actual, event):
    reserved = state['inflight']['reservation']
    _metrics(actual, reserved)
    for key,value in actual.items():
        state['observed'][key] += value
        # 미해결 예약이 있으면 새 phase를 시작할 수 없다. 부동소수점 상쇄 오차도 피한다.
        state['charged'][key] = state['observed'][key]
    state['events'] = (state['events']+[event])[-64:]
    state['inflight'] = None
    _save(root,state)


def _remaining(plan, state):
    return {k:plan['budgets'][k]-state['charged'][k] for k in MAXIMUM}


def _time_phase(root, plan, state, kind, operation):
    previous_status = state['status']
    remaining = _remaining(plan,state)
    reservation = {k:0 for k in MAXIMUM}; reservation['elapsed_seconds'] = remaining['elapsed_seconds']
    _reserve(root,plan,state,kind,reservation)
    started = time.monotonic()
    def check():
        require(time.monotonic()-started < reservation['elapsed_seconds'], 'queue_elapsed_exceeded_unknown_reserved')
    try:
        check()
        result = operation(check)
        check()
        elapsed = time.monotonic()-started
        require(elapsed <= reservation['elapsed_seconds'], 'queue_elapsed_exceeded_unknown_reserved')
        actual = dict(reservation); actual['elapsed_seconds'] = elapsed
        _settle(root,plan,state,actual,{'kind':kind,'elapsed_seconds':elapsed})
        state['status'] = previous_status
        _save(root,state)
        return result
    except BaseException as exc:
        # 검사 중 예외/중단도 시간을 보수적으로 예약한 채 남긴다.
        state['status'] = 'unknown_cost_blocked'
        state['failure'] = {'phase':kind,'type':type(exc).__name__,'reason':str(exc)[:256]}
        _save(root,state)
        raise


def _match(root, plan, state, check):
    """이미지/pose를 파일명 키와 고정 추출 hash로 대조. 시간/내용 alignment 검증은 별도다."""
    expected, actual = {}, {}
    for job in plan['jobs']:
        for target in job['targets']:
            seq, frame, view = identify(target,job['manifest']['kind'])
            expected.setdefault((seq,frame),set()).add(view)
    for job,row in zip(plan['jobs'],state['jobs']):
        check()
        report = verify_extraction(root/row['output'],job['manifest'])
        require(digest(report) == row['output_manifest_sha256'], 'queue_matching_report_changed')
        for entry in report['files']:
            check()
            seq, frame, view = identify(entry['member'],entry['kind'])
            views = actual.setdefault((seq,frame),{})
            require(view not in views, 'queue_actual_duplicate_view')
            views[view] = {'job_id':job['id'],'member':entry['member'],'sha256':entry['sha256']}
    require(set(actual) == set(expected) and all(set(actual[k]) == expected[k] == {'left','right','pose'} for k in expected),
            'queue_actual_matching_failed')
    result = hashed({'schema_version':'preparation_queue_matching_v1','plan_sha256':plan['content_sha256'],
        'matched_triplet_count':len(actual),'train_ready':False,'time_alignment':'unverified',
        'rgb_pose_content_alignment':'unverified','whole_archive_coverage':'not_measured',
        'frames':[{'sequence_id':seq,'frame_index':frame,'views':views} for (seq,frame),views in sorted(actual.items())]})
    check()
    atomic_record(root/'matching.json',result)
    check()
    return result


def run_queue(plan, root, *, base, max_jobs=16, retry=False, authorized_session=None):
    plan = validate_plan(plan); root = _path(root,local=True); base = _path(base)
    require(integer(max_jobs,1,16) and type(retry) is bool, 'queue_run_arguments')
    # live queue/index는 Drive mount에 두지 않는다. snapshot 복사/원격 내구성 연결은 후속 검증.
    sources = []
    for job in plan['jobs']:
        source_base = _path(base/job['source_base'])
        spec = _source(job['manifest'],source_base)
        require(digest(spec) == job['source_sha256'], 'queue_live_source_identity_changed')
        for part in spec['parts']:
            if 'path' in part:
                source_path = _path(part['path'])
                require(not source_path.is_relative_to(root) and not root.is_relative_to(source_path.parent),
                        'queue_source_output_overlap')
        if spec['schema'] == 'ordered_drive_gzip_source_v1':
            require(authorized_session is not None, 'queue_drive_session_required')
        sources.append(source_base)
    require(root.is_dir(), 'queue_initialize_required')
    with writer(root):
        state = _state(root,plan)
        if state['inflight'] is not None:
            state['status'] = 'unknown_cost_blocked'; _save(root,state)
            return state
        _time_phase(root,plan,state,'disk_inventory',lambda check:_disk(root,plan['max_disk_bytes'],check))
        # 이전 완료본도 새 세션에서 실제 hash를 대조한다. 잘못된 완료는 전진시키지 않는다.
        if state['cursor']:
            def verify_completed(check):
                for i in range(state['cursor']):
                    check()
                    result = verify_extraction(root/state['jobs'][i]['output'],plan['jobs'][i]['manifest'])
                    require(result['targets_complete'] and result['exact_targets'] == plan['jobs'][i]['targets'],
                            'queue_completed_output_mismatch')
                    require(digest(result) == state['jobs'][i]['output_manifest_sha256'] and
                            {f['member'] for f in result['files']} == set(plan['jobs'][i]['targets']),
                            'queue_completed_payload_scope_changed')
            _time_phase(root,plan,state,'verify_prepared',verify_completed)
        completed = 0
        while state['cursor'] < len(plan['jobs']) and completed < max_jobs:
            i = state['cursor']; job = plan['jobs'][i]; row = state['jobs'][i]
            if row['attempts'] and not retry:
                state['status'] = 'explicit_retry_required'; _save(root,state); return state
            remaining = _remaining(plan,state)
            remote = job['manifest']['parts'][0]['source']['type'] == 'drive'
            if not all(remaining[k] > 0 for k in ('read_bytes','decoded_bytes','members','elapsed_seconds')) or (
                    remote and not (remaining['http_requests'] > 0 and remaining['body_bytes'] > 0)):
                state['status']='budget_exhausted'; _save(root,state); return state
            limits = Limits.from_dict(plan['backend_limits'])
            seconds = min(limits.max_elapsed_seconds,remaining['elapsed_seconds'])
            limits = replace(limits,max_elapsed_seconds=seconds,request_timeout_seconds=min(limits.request_timeout_seconds,seconds),
                max_decompressed_bytes=min(limits.max_decompressed_bytes,remaining['decoded_bytes']),
                max_members=min(limits.max_members,remaining['members']),
                max_disk_bytes=min(limits.max_disk_bytes,plan['max_disk_bytes']),
                max_ranges=min(limits.max_ranges,remaining['http_requests']) if remote else limits.max_ranges,
                max_transfer_bytes=min(limits.max_transfer_bytes,remaining['body_bytes']) if remote else limits.max_transfer_bytes)
            context = PreparationContext(limits,authorized_session)
            reservation = {'read_bytes':min(32*MIB,remaining['read_bytes']),
                'decoded_bytes':min(32*MIB,remaining['decoded_bytes'],limits.max_decompressed_bytes),
                'members':min(128,remaining['members'],limits.max_members),
                'http_requests':min(120,limits.max_ranges) if remote else 0,
                'body_bytes':min(8*MIB,limits.max_transfer_bytes) if remote else 0,'elapsed_seconds':seconds}
            context.resources.index_limits = replace(context.resources.index_limits,
                max_read_bytes=reservation['read_bytes'],max_decoded_bytes=reservation['decoded_bytes'],max_seconds=seconds)
            context.resources.paths.add(root)
            token = _reserve(root,plan,state,'prepare_'+job['id'],reservation)
            output_name = f'attempts/{token:03d}_{job["id"]}'
            output = root/output_name
            options = {**job['index_options'],'workspace':str(root/'indexes'),
                       'snapshot_directory':str(root/'snapshots'), 'request_id':f'queue_{token:03d}_{job["id"]}'}
            started = time.monotonic(); result = None; failure = None; primary = None
            try:
                result = prepare_indexed(job['manifest'],output,job['targets'],limits,options,
                    base=sources[i],authorized_session=authorized_session,context=context)
                if result['targets_complete']:
                    checked = verify_extraction(output,job['manifest'])
                    require(checked['exact_targets'] == job['targets'] and
                            {f['member'] for f in checked['files']} == set(job['targets']), 'queue_targets_changed')
            except BaseException as exc:
                primary = exc
                failure = {'type':type(exc).__name__,'reason':str(exc)[:256]}
            observed = {'read_bytes':context.resources.read,'decoded_bytes':context.resources.decoded,
                'members':context.resources.members,'http_requests':0,'body_bytes':0,'elapsed_seconds':time.monotonic()-started}
            if remote and context.runtime is not None and hasattr(context.runtime,'transfer'):
                observed.update(http_requests=context.runtime.transfer.ranges,body_bytes=context.runtime.transfer.transfer_bytes)
            event = {'kind':'prepare','id':token,'job_id':job['id'],'output':output_name,
                     'status':'failed' if failure else 'prepared' if result['targets_complete'] else 'partial',
                     'observed':observed,'failure':failure}
            row['attempts'].append(event)
            backend = result.get('persistent_index',{}) if result is not None else {}
            if failure or backend.get('status') in ('failed','interrupted') or backend.get('cleanup_exception'):
                # 예외 경로의 미계측 여부를 증명할 수 없으므로 보수적으로 전체 예약을 유지한다.
                row['status']='failed'; state['status']='unknown_cost_blocked'; _save(root,state)
                if primary is not None and not isinstance(primary,Exception): raise primary
                return state
            # 초과/미계측이면 예약을 반환하지 않는다. 원래 결과 파일은 항상 보존한다.
            try:
                _metrics(observed,reservation)
            except ValueError:
                state['status']='unknown_cost_blocked'; _save(root,state); return state
            row['status'] = event['status']
            if row['status'] == 'prepared':
                row['output'] = output_name; row['output_manifest_sha256'] = digest(checked)
                state['cursor'] += 1; completed += 1
            state['status'] = 'partial' if row['status'] != 'prepared' else 'pending'
            _settle(root,plan,state,observed,event)
            _time_phase(root,plan,state,'disk_inventory',lambda check:_disk(root,plan['max_disk_bytes'],check))
            if failure or row['status'] != 'prepared': return state
        if state['cursor'] == len(plan['jobs']):
            matched = _time_phase(root,plan,state,'match_triplets',lambda check:_match(root,plan,state,check))
            state['matching'] = {'path':'matching.json','content_sha256':matched['content_sha256'],
                                 'matched_triplet_count':matched['matched_triplet_count']}
            state['status'] = 'prepared_not_training_release'
        else: state['status'] = 'bounded_job_stop'
        _save(root,state)
        return state


def mark_consumed(plan, root, job_id, consumer_artifact_sha256):
    """사용자가 관리하는 소비자 artifact ID만 기록. 파일 삭제나 학습 완료 판정은 하지 않는다."""
    plan = validate_plan(plan); root = _path(root,local=True)
    require(isinstance(consumer_artifact_sha256,str) and re.fullmatch('[0-9a-f]{64}',consumer_artifact_sha256),
            'queue_consumer_identity')
    with writer(root):
        state = _state(root,plan)
        require(state['inflight'] is None, 'queue_unknown_cost_blocked')
        row = next((r for r in state['jobs'] if r['id'] == job_id),None)
        require(row is not None and row['status'] == 'prepared', 'queue_job_not_prepared')
        require(row['consumed'] in (None,consumer_artifact_sha256), 'queue_consumer_conflict')
        row['consumed'] = consumer_artifact_sha256
        _save(root,state)
        return state
