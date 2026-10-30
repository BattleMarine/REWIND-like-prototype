"""C 실행의 로컬 단계 기록과 인증 없는 제한적 보고서 회수."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import tempfile
import time
import zipfile

from .source import IndexFailure, canonical, MIB

RECORD_LIMIT = 2 * MIB
FILE_LIMIT = 8 * MIB
BUNDLE_LIMIT = 20 * MIB
FILE_COUNT_LIMIT = 8
STAGES = {'initial', 'restore_b', 'b_summary', 'session_factory', 'runtime_init',
          'index_init', 'scan_batch', 'preview', 'snapshot', 'restore_c',
          'cold_verify', 'cleanup'}
PIN_KEYS = {'source_revision', 'source_zip_sha256', 'manifest_sha256', 'anchor_sha256', 'environment'}


def validate_provenance(value):
    """호출자가 바이트 대조한 pin의 필수 형식과 환경 필드를 검사한다."""
    if not isinstance(value, dict) or set(value) != PIN_KEYS:
        raise IndexFailure('extension_provenance_required')
    for key in PIN_KEYS - {'environment'}:
        width = 40 if key == 'source_revision' else 64
        if not isinstance(value[key], str) or not re.fullmatch('[a-f0-9]{%d}' % width, value[key]):
            raise IndexFailure('extension_provenance_pin')
    env = value['environment']
    if not isinstance(env, dict) or set(env) != {'python', 'indexed_gzip', 'psutil'}:
        raise IndexFailure('extension_provenance_environment')
    if any(not isinstance(v, str) or not re.fullmatch('[A-Za-z0-9.+_-]{1,64}', v) for v in env.values()):
        raise IndexFailure('extension_provenance_environment')
    return json.loads(canonical(value))


def validate_run(run_id, phase='C'):
    if phase != 'C' or not isinstance(run_id, str) or not re.fullmatch('[A-Za-z0-9_-]{1,64}', run_id):
        raise IndexFailure('extension_run_id')


def atomic_record(path, report, *, initial=False):
    """완성된 작은 JSON만 게시한다. 최초 게시는 기존 소비 marker를 덮지 않는다."""
    path = Path(path)
    raw = canonical(report) + b'\n'
    if len(raw) > RECORD_LIMIT:
        raise IndexFailure('extension_record_size')
    if path.is_symlink():
        raise IndexFailure('extension_record_symlink')
    fd, name = tempfile.mkstemp(prefix='.c-record-', dir=path.parent)
    try:
        with os.fdopen(fd, 'wb') as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        if initial:
            try:
                if os.name == 'nt':os.rename(name, path)
                else:os.link(name, path)
            except FileExistsError:
                raise IndexFailure('pilot_run_already_consumed') from None
        else:
            # Windows의 짧은 공유 잠금만 최대 30ms 기다린다. 원본 I/O 재시도와 무관하다.
            for attempt in range(4):
                try:
                    os.replace(name, path)
                    break
                except PermissionError:
                    if os.name != 'nt' or attempt == 3:raise
                    time.sleep(.01)
    finally:
        Path(name).unlink(missing_ok=True)


def stage(report, marker, name, edge, *, batch=None):
    if name not in STAGES or edge not in {'pre', 'post'} or (batch is not None and not 1 <= batch <= 8):
        raise IndexFailure('extension_stage')
    event = {'name': name, 'edge': edge, 'at_utc': datetime.now(timezone.utc).isoformat()}
    if batch is not None:
        event['batch'] = batch
    report['stage'] = event
    report.setdefault('stages', []).append(event)
    if len(report['stages']) > 64:
        raise IndexFailure('extension_stage_count')
    atomic_record(marker, report)


def exception_kind(exc):
    # 사용자 정의 예외의 class 이름도 외부 문자열이므로 그대로 기록하지 않는다.
    for cls in (KeyboardInterrupt, SystemExit, GeneratorExit):
        if isinstance(exc, cls):
            return cls.__name__
    return 'Exception' if isinstance(exc, Exception) else 'BaseException'


def _read(path, root, limit):
    """모든 경로 요소의 symlink와 root 탈출을 거부하고 크기 제한 안에서 읽는다."""
    path = Path(path)
    root = Path(root).absolute()
    try:
        relative = path.absolute().relative_to(root)
    except ValueError:
        raise IndexFailure('recovery_path') from None
    current = root
    for part in relative.parts:
        if part in {'.', '..'} or current.is_symlink() or getattr(current, 'is_junction', lambda:False)():
            raise IndexFailure('recovery_path')
        current = current / part
        if current.is_symlink() or getattr(current, 'is_junction', lambda:False)():
            raise IndexFailure('recovery_symlink')
    if not current.is_file() or current.stat().st_size > limit:
        raise IndexFailure('recovery_file_size_or_missing')
    with current.open('rb') as stream:
        raw = stream.read(limit + 1)
    if len(raw) > limit:
        raise IndexFailure('recovery_file_size_or_missing')
    return raw


def _member_path(value):
    if not isinstance(value, str) or not value or '\\' in value or ':' in value or value.startswith('/') or any(ord(c)<32 for c in value):
        raise IndexFailure('recovery_path')
    if any(part in {'', '.', '..'} for part in value.split('/')):
        raise IndexFailure('recovery_path')
    return value


def _numeric(value):
    """회수 통계는 수치와 참/거짓으로 제한한다."""
    return {k: v for k, v in value.items() if isinstance(k, str) and re.fullmatch('[a-z_]{1,64}', k)
            and (type(v) in (int, float, bool) or v is None)} if isinstance(value, dict) else {}


def _stats(value):
    result = _numeric(value)
    if isinstance(value, dict) and 'source_read_trace' in value:
        rows = value['source_read_trace']
        if not isinstance(rows, list) or len(rows) > 4096:
            raise IndexFailure('recovery_trace_limit')
        result['source_read_trace'] = [_numeric(row) for row in rows]
    return result


def _summary(report):
    # 원시 message/URL/credential 및 임의의 확장 필드를 회수 ZIP에 복사하지 않는다.
    result = {k: report[k] for k in ('schema', 'run_id', 'phase', 'status', 'execution_metadata')}
    events = report.get('stages', [])
    if not isinstance(events, list) or len(events) > 64:
        raise IndexFailure('recovery_stages')
    result['stages'] = []
    for event in events:
        if not isinstance(event, dict) or event.get('name') not in STAGES or event.get('edge') not in {'pre', 'post'}:
            raise IndexFailure('recovery_stages')
        at = event.get('at_utc')
        if not isinstance(at, str) or not re.fullmatch('[0-9T:.+Z-]{1,40}', at):
            raise IndexFailure('recovery_stages')
        clean = {'name': event['name'], 'edge': event['edge'], 'at_utc': at}
        if 'batch' in event:
            if type(event['batch']) is not int or not 1 <= event['batch'] <= 8:
                raise IndexFailure('recovery_stages')
            clean['batch'] = event['batch']
        result['stages'].append(clean)
    for key in ('error', 'stop_reason'):
        value = report.get(key)
        if isinstance(value, str) and re.fullmatch('[a-z0-9_]{1,100}', value):
            result[key] = value
    for key in ('interruption', 'cleanup_exception'):
        value = report.get(key)
        if isinstance(value, dict) and value.get('type') in {'KeyboardInterrupt', 'SystemExit', 'GeneratorExit', 'Exception', 'BaseException'}:
            result[key] = {'type': value['type'], 'stage': value.get('stage') if value.get('stage') in STAGES else 'unknown'}
    for key in ('before', 'after', 'after_scan', 'restored_b_checkpoint', 'latest_saved_checkpoint', 'transport', 'phase_resources', 'persisted_index_validation', 'cold_cache_clear'):
        if key in report:
            result[key] = _stats(report[key])
            for name, val in report[key].items() if isinstance(report[key], dict) else ():
                if name.endswith('_sha256') and isinstance(val, str) and re.fullmatch('[a-f0-9]{64}', val):
                    result[key][name] = val
    result['transport_recording'] = 'recorded' if report.get('transport_recording') == 'recorded' else 'not_recorded'
    if result['transport_recording'] == 'not_recorded':result['transport'] = {'status':'unknown'}
    for key in ('old_coverage', 'old_catalog_exact_prefix', 'full_archive_crc_requested'):
        if type(report.get(key)) in (int, bool):result[key] = report[key]
    if report.get('verification_status') == 'passed':result['verification_status'] = 'passed'
    for key in ('restore_b', 'restore_c'):
        if isinstance(report.get(key), dict):
            result[key] = _numeric(report[key])
            if isinstance(report[key].get('receipt'), dict):
                from .snapshots import _receipt
                result[key]['receipt'] = _receipt(report[key]['receipt'])
            if isinstance(report[key].get('stats'), dict):result[key]['stats'] = _numeric(report[key]['stats'])
    result['scans'] = [{'batch': r.get('batch'), 'result': _numeric(r.get('result')), 'stats': _stats(r.get('result', {}).get('stats')), 'checkpoint': _numeric(r.get('checkpoint'))}
                       for r in report.get('scans', [])[:8] if isinstance(r, dict)]
    for key in ('cold_new_extraction', 'pre_snapshot_extraction'):
        if isinstance(report.get(key), dict):
            result[key] = {'stats': _stats(report[key].get('stats'))}
    if isinstance(report.get('verification_extractions'), dict):
        result['verification_extractions'] = {k: {'stats': _stats(v.get('stats'))}
                                              for k, v in report['verification_extractions'].items() if k in {'old', 'new'} and isinstance(v, dict)}
    if isinstance(report.get('phase_resources'), dict):
        result['phase_resources']['calls'] = [{'operation': r.get('operation') if r.get('operation') in {'scan', 'extract'} else 'unknown', 'stats': _stats(r.get('stats'))}
                                               for r in report['phase_resources'].get('calls', [])[:16] if isinstance(r, dict)]
    from .snapshots import _receipt
    if 'b_receipt' in report:
        result['b_receipt'] = _receipt(report['b_receipt'])
    if 'snapshot' in report:
        result['snapshot_receipt'] = _receipt(report['snapshot']['receipt'])
    if 'new_member' in report:
        row = report['new_member']
        result['new_member'] = {**_numeric(row), 'name': _member_path(row['name']), 'sha256': row['sha256']}
    for key in ('old_restart_points', 'after_restart_points', 'new_restart_point'):
        if key in report:
            value = report[key]
            rows = [value] if key == 'new_restart_point' else value
            if not isinstance(rows, list) or len(rows) > 4096 or any(not isinstance(r, (list, tuple)) or len(r) != 2 or any(type(x) is not int or x < 0 for x in r) for r in rows):
                raise IndexFailure('recovery_points')
            result[key] = value
    return result


def recover_report(*, workspace, run_id, provenance, phase='C'):
    """디스크만 읽어 ZIP을 생성/재사용한다. 원본 DB·marker와 인증에는 접근하지 않는다."""
    validate_run(run_id, phase)
    pins = validate_provenance(provenance)
    root = Path(workspace).absolute()
    if not root.is_dir() or root.is_symlink() or root.resolve() != root:
        raise IndexFailure('recovery_workspace')
    marker = root / (run_id + '-C.json')
    bundle = root / (run_id + '-C-report.zip')
    if bundle.exists() or bundle.is_symlink():
        _check_bundle(bundle, root, run_id, pins)
        return {'path': str(bundle), 'reused': True}
    diagnostic = None
    try:
        raw = _read(marker, root, RECORD_LIMIT)
        report = json.loads(raw)
    except (OSError, ValueError, IndexFailure, RecursionError):
        report = None
        diagnostic = 'marker_missing_invalid_or_oversize'
    if not isinstance(report, dict) or not report.get('execution_metadata'):
        diagnostic = diagnostic or 'provenance_not_recorded'
    else:
        try:
            recorded_pins = validate_provenance(report['execution_metadata'])
        except IndexFailure:
            recorded_pins = None
            diagnostic = 'provenance_invalid'
        if report.get('run_id') != run_id or report.get('phase') != phase or (recorded_pins is not None and recorded_pins != pins):
            raise IndexFailure('recovery_provenance_mismatch')
        if report.get('schema') != 'persistent_drive_extension_report_v1' or report.get('status') not in {'started', 'partial', 'failed', 'interrupted', 'passed'}:
            diagnostic = 'marker_schema_or_status_invalid'
    summary = None
    if not diagnostic:
        try:
            summary = _summary(report)
        except (TypeError, AttributeError, KeyError, ValueError, IndexFailure, RecursionError):
            diagnostic = 'marker_structure_invalid'
    contents = {}
    if diagnostic:
        summary = {'run_id': run_id, 'phase': phase, 'status': 'offline_diagnostic', 'reason': diagnostic,
                   'execution_metadata': pins, 'execution_provenance_verified': False}
    else:
        summary['execution_provenance_verified'] = True
        summary['marker_sha256'] = hashlib.sha256(raw).hexdigest()
        files = report.get('extraction', {}).get('files', [])
        if not isinstance(files, list) or len(files) > 3:
            raise IndexFailure('recovery_selected_count')
        for row in files:
            relative = _member_path(row['relative_path'])
            name = _member_path(row['name'])
            if relative not in {'new/files/' + name, 'old/files/' + name} or not name.endswith('.json'):
                raise IndexFailure('recovery_selected_path')
            data = _read(root / (run_id + '-C-selected') / relative, root, min(FILE_LIMIT, BUNDLE_LIMIT - RECORD_LIMIT - sum(map(len, contents.values()))))
            if hashlib.sha256(data).hexdigest() != row.get('sha256') or not isinstance(json.loads(data), dict):
                raise IndexFailure('recovery_selected_hash_or_json')
            contents['selected/' + relative] = data
        anchor = root / (run_id + '-C-anchor.json')
        if report.get('anchor_sha256'):
            data = _read(anchor, root, RECORD_LIMIT)
            value = json.loads(data)
            if hashlib.sha256(data).hexdigest() != report['anchor_sha256'] or value.get('schema') != 'persistent_drive_extension_anchor_v1':
                raise IndexFailure('recovery_anchor')
            # 허용된 생성 anchor만 포함한다. 원시 JSON의 추가 필드를 복사하지 않는다.
            if set(value) != {'schema', 'runtime_id', 'receipt', 'selected'} or value.get('selected') != report.get('selected') or value.get('receipt') != report.get('snapshot', {}).get('receipt'):
                raise IndexFailure('recovery_anchor')
            from .snapshots import _receipt
            _receipt(value['receipt'])
            if not isinstance(value['runtime_id'], str) or not re.fullmatch('[a-f0-9]{32}', value['runtime_id']):
                raise IndexFailure('recovery_anchor')
            if not isinstance(value['selected'], list) or not 1 <= len(value['selected']) <= 3:
                raise IndexFailure('recovery_anchor')
            for row in value['selected']:
                if set(row) != {'name', 'sha256'} or not re.fullmatch('[a-f0-9]{64}', row['sha256']):
                    raise IndexFailure('recovery_anchor')
                _member_path(row['name'])
            contents['anchor.json'] = data
    contents['report.json'] = canonical(summary) + b'\n'
    inventory = {'schema': 'extension_recovery_bundle_v1', 'run_id': run_id, 'phase': phase,
                 'execution_metadata': pins, 'status': summary['status'],
                 'files': {name: {'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()} for name, data in contents.items()}}
    contents['recovery.json'] = canonical(inventory) + b'\n'
    if len(contents) > FILE_COUNT_LIMIT or sum(map(len, contents.values())) > BUNDLE_LIMIT:
        raise IndexFailure('recovery_bundle_limit')
    fd, temporary = tempfile.mkstemp(prefix='.c-recovery-', dir=root)
    os.close(fd)
    try:
        with zipfile.ZipFile(temporary, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
            for name, data in contents.items():
                archive.writestr(name, data)
        if Path(temporary).stat().st_size > BUNDLE_LIMIT:
            raise IndexFailure('recovery_bundle_limit')
        if os.name == 'nt':os.rename(temporary, bundle)
        else:os.link(temporary, bundle)
    finally:
        Path(temporary).unlink(missing_ok=True)
    return {'path': str(bundle), 'reused': False, 'status': summary['status']}


def _check_bundle(path, root, run_id, pins):
    data = _read(path, root, BUNDLE_LIMIT)
    import io
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        entries = archive.infolist()
        if len(entries) > FILE_COUNT_LIMIT or sum(e.file_size for e in entries) > BUNDLE_LIMIT:
            raise IndexFailure('recovery_bundle_limit')
        names = [e.filename for e in entries]
        if len(names) != len(set(names)) or not {'recovery.json', 'report.json'} <= set(names):
            raise IndexFailure('recovery_bundle_members')
        for entry in entries:
            _member_path(entry.filename)
            if entry.file_size > FILE_LIMIT or stat.S_ISLNK(entry.external_attr >> 16):
                raise IndexFailure('recovery_bundle_members')
        info = json.loads(archive.read('recovery.json'))
        if info.get('schema') != 'extension_recovery_bundle_v1' or info.get('run_id') != run_id or info.get('phase') != 'C' or info.get('execution_metadata') != pins:
            raise IndexFailure('recovery_provenance_mismatch')
        if set(names) != set(info.get('files', {})) | {'recovery.json'}:
            raise IndexFailure('recovery_bundle_members')
        for name, expected in info['files'].items():
            if name not in {'report.json', 'anchor.json'} and not name.startswith(('selected/new/', 'selected/old/')):
                raise IndexFailure('recovery_bundle_members')
            raw = archive.read(name)
            if len(raw) != expected['bytes'] or hashlib.sha256(raw).hexdigest() != expected['sha256']:
                raise IndexFailure('recovery_bundle_hash')
