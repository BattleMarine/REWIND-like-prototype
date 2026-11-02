"""저장 인덱스에서 명시 stereo만 추출·내용 감사한다. scan/학습 release 없음."""
from dataclasses import asdict, dataclass, replace
import hashlib
import json
from pathlib import Path
import re
import time
from zipfile import ZipFile, ZIP_STORED

from rewind_archive.contracts import canonical, digest, validate_manifest, PreparationError
from rewind_archive.drive_metadata import build_drive_manifest, MetadataLimits
from rewind_archive.matching import identify, contiguous_runs
from rewind_archive.request_policy import validate_no_hidden_retries
from rewind_archive_index.core import ArchiveIndex
from rewind_archive_index.drive import DriveRuntime, DriveLimits, make_drive_source
from rewind_archive_index.extension_records import atomic_record
from rewind_archive_index.snapshots import import_snapshot, SnapshotLimits, _path
from rewind_archive_index.source import IndexFailure, IndexLimits, MIB
from rewind_prepare.converter import validate_candidate


@dataclass(frozen=True)
class PairLimits:
    body_bytes: int = 512*MIB
    requests: int = 3072
    seconds: int = 1200
    read_bytes: int = 1024*MIB
    decoded_bytes: int = 256*MIB
    rss_bytes: int = 1024*MIB
    work_bytes: int = 512*MIB
    zip_bytes: int = 384*MIB
    batch_files: int = 32

    def __post_init__(self):
        caps = dict(body_bytes=512*MIB, requests=3072, seconds=1200, read_bytes=1024*MIB,
                    decoded_bytes=256*MIB, rss_bytes=1024*MIB, work_bytes=512*MIB,
                    zip_bytes=384*MIB, batch_files=32)
        if any(type(v) is not int or not 1 <= v <= caps[k] for k, v in asdict(self).items()):
            raise IndexFailure('pair_limits')


def check(value, reason):
    if not value:
        raise IndexFailure(reason)


def sha_file(path):
    h = hashlib.sha256()
    with _path(path).open('rb') as stream:
        while data := stream.read(65536):
            h.update(data)
    return h.hexdigest()


def read(path, maximum):
    path = _path(path)
    check(path.is_file() and path.stat().st_size <= maximum, 'pair_input_size')
    return json.loads(path.read_bytes())


def preflight(input_root):
    root = _path(input_root)
    pins = read(root/'input_manifest.json', MIB)
    check(pins.get('schema') == 'pinned_pair_inputs_v1' and len(pins['files']) <= 128, 'pair_input_manifest')
    actual = {p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file()}
    check(actual == set(pins['files']) | {'input_manifest.json'}, 'pair_input_file_set')
    for name, row in pins['files'].items():
        path = _path(root/name)
        check(path.is_relative_to(root) and path.is_file() and path.stat().st_size == row['bytes'] <= 72*MIB
              and sha_file(path) == row['sha256'], 'pair_input_pin')
    config = read(root/'config.json', MIB)
    check(config.get('schema') == 'indexed_stereo_pairs_v1' and config.get('train_ready') is False, 'pair_config')
    check(isinstance(config['job_id'], str) and re.fullmatch('[A-Za-z0-9_-]{1,64}', config['job_id']), 'pair_job_id')
    limits = PairLimits(**config['limits'])
    manifest = validate_manifest(config['rgb_manifest'])
    check(manifest['kind'] == 'rgb' and digest(make_drive_source(manifest)) == config['source_sha256'], 'pair_rgb_source')
    candidate = read(root/'pose/candidate.json', 64*MIB)
    validate_candidate(candidate)
    check(candidate['content_sha256'] == config['candidate_sha256'] and candidate['training_eligibility']['train_ready'] is False,
          'pair_pose_candidate')
    pose_rows = {(r['sequence_id'], r['frame_index']): r for r in candidate['records']}
    keys, names = set(), set()
    check(isinstance(config['pairs'], list) and 1 <= len(config['pairs']) <= 102, 'pair_count')
    for pair in config['pairs']:
        key = (pair['sequence'], pair['frame'])
        check(key not in keys and key in pose_rows and set(pair['rgb']) == {'left', 'right'}, 'pair_unique_pose_key')
        keys.add(key)
        for side, row in pair['rgb'].items():
            check(identify(row['name'], 'rgb') == (*key, side) and row['name'] not in names
                  and row['type'] == 'file' and type(row['size']) is int and 0 < row['size'] <= 8*MIB,
                  'pair_unique_rgb_key_view')
            names.add(row['name'])
    check(keys == set(pose_rows), 'pair_exact_pose_coverage')
    check(sum(v['size'] for p in config['pairs'] for v in p['rgb'].values())+96*MIB <= limits.work_bytes, 'pair_payload_disk_plan')
    return config, digest(config)


class ExtractOnlyIndex(ArchiveIndex):
    def _source(self, spec, budget):
        self.last_budget = budget
        return super()._source(spec, budget)

    def scan(self, *args, **kwargs):
        raise IndexFailure('pair_scan_forbidden')

    def _commit(self, *args, **kwargs):
        raise IndexFailure('pair_index_mutation_forbidden')


def run(input_root, workspace, session_factory):
    """한 번의 job. 실패 chunk도 보존하며 누적 예산을 재설정하지 않는다."""
    root, work = _path(input_root), _path(workspace)
    config, config_sha = preflight(root)
    limits = PairLimits(**config['limits'])
    check(not work.is_relative_to(root) and not root.is_relative_to(work), 'pair_separate_workspace')
    check(not str(work).replace('\\', '/').startswith('/content/drive/'), 'pair_live_local_only')
    work.mkdir(parents=True, exist_ok=False)
    report = {'schema': 'indexed_stereo_pair_run_v1', 'job_id': config['job_id'], 'config_sha256': config_sha,
        'status': 'started', 'train_ready': False, 'limits': asdict(limits), 'chunks': [], 'files': [],
        'pairs': [], 'target_pairs': len(config['pairs']), 'source_sha256': config['source_sha256'],
        'archive_complete_claimed': False, 'scan_performed': False, 'observed_peak_rss': 0}
    started = time.monotonic()
    runtime = session = index = None
    read_bytes = decoded = metadata_body = metadata_requests = 0

    def save():
        report.update(elapsed_seconds=time.monotonic()-started, source_read_bytes=read_bytes,
            returned_decoded_bytes=decoded, body_bytes=metadata_body+(runtime.transfer.transfer_bytes if runtime and hasattr(runtime, 'transfer') else 0),
            requests=metadata_requests+(runtime.transfer.ranges if runtime and hasattr(runtime, 'transfer') else 0))
        atomic_record(work/'report.json', report, initial=not (work/'report.json').exists())

    def guard(reserve=0):
        import psutil
        rss = psutil.Process().memory_info().rss
        report['observed_peak_rss'] = max(report['observed_peak_rss'], rss)
        check(rss <= limits.rss_bytes, 'pair_rss_limit')
        check(time.monotonic()-started < limits.seconds-reserve, 'pair_time_limit')
        total = 0
        for n, p in enumerate(work.rglob('*')):
            check(n < 2048, 'pair_inventory_limit')
            _path(p)
            if p.is_file():
                total += p.stat().st_size
        check(total+MIB <= limits.work_bytes, 'pair_work_disk_limit')

    save()
    try:
        report['stage'] = 'restore'
        guard(30)
        receipt = config['snapshot_receipt']
        receipt_path = root/'snapshot'/(receipt['filename']+'.receipt.json')
        check(sha_file(receipt_path) == config['receipt_file_sha256'] and read(receipt_path, 16384) == receipt, 'pair_receipt_pin')
        index_limits = IndexLimits(max_read_bytes=limits.read_bytes, max_decoded_bytes=limits.decoded_bytes,
            max_seconds=limits.seconds, max_rss_bytes=limits.rss_bytes, max_disk_bytes=limits.work_bytes)
        report['restore'] = import_snapshot(receipt_path, work/'index.sqlite', expected_source_sha256=config['source_sha256'],
            expected_snapshot_id=receipt['snapshot_id'], min_generation=receipt['generation'], known_anchor=receipt,
            limits=SnapshotLimits(max_disk_bytes=limits.work_bytes), index_limits=index_limits)
        db_sha = sha_file(work/'index.sqlite')
        # import_snapshot은 local DB의 신뢰 anchor를 기록한다. 원본 ZIP/DB SHA와
        # 복원 직후 local DB SHA를 구분하고 추출 전후에는 후자를 그대로 보존한다.
        report['restored_db_before_sha256'] = db_sha
        report['snapshot_db_sha256'] = receipt['db_sha256']
        index = ExtractOnlyIndex(work/'index.sqlite', index_limits)
        state, _ = index._snapshot()
        check(state['source_sha256'] == config['source_sha256'] and state['generation'] == receipt['generation'], 'pair_index_identity')
        wanted = {r['name']: r for p in config['pairs'] for r in p['rgb'].values()}
        ordered = [r for r in state['members'] if r['name'] in wanted]
        check(len(ordered) == len(wanted) and all(r == wanted[r['name']] for r in ordered), 'pair_index_exact_rows')
        report['stage'] = 'initial_metadata'
        save()
        session = session_factory()
        validate_no_hidden_retries(session)
        expected = config['rgb_manifest']
        selection = {k: expected[k] for k in ('archive_id', 'dataset_version', 'kind', 'total_parts')}
        selection.update(schema_version='drive_part_selection_v1', parts=[{
            'index': p['index'], 'name': p['name'], 'file_id': p['source']['file_id']} for p in expected['parts']])
        metadata = build_drive_manifest(selection, session, MetadataLimits(max_files=len(expected['parts']),
            max_requests=len(expected['parts']), max_transfer_bytes=min(128*1024, limits.body_bytes),
            max_elapsed_seconds=min(30, limits.seconds-(time.monotonic()-started))))
        atomic_record(work/'metadata.json', metadata)
        metadata_body = metadata['report']['stats']['transfer_bytes']
        metadata_requests = metadata['report']['stats']['requests']
        report['initial_metadata_stop_reason'] = metadata['report']['stop_reason']
        check(metadata['manifest'] is not None, 'pair_initial_metadata_failed')
        check(metadata['manifest'] == expected, 'pair_source_identity_changed')
        guard(30)
        check(metadata_body < limits.body_bytes and metadata_requests < limits.requests, 'pair_metadata_budget')
        runtime = DriveRuntime(session, DriveLimits(max_body_bytes=limits.body_bytes-metadata_body,
            max_requests=limits.requests-metadata_requests, max_seconds=limits.seconds-(time.monotonic()-started)-30,
            range_bytes=MIB, max_cache_bytes=4*MIB))
        index.runtime = runtime
        candidate = read(root/'pose/candidate.json', 64*MIB)
        poses = {(r['sequence_id'], r['frame_index']): r for r in candidate['records']}
        representatives = set()
        for start in range(0, len(ordered), limits.batch_files):
            report['stage'] = 'extract'
            guard(30)
            check(read_bytes < limits.read_bytes and decoded < limits.decoded_bytes, 'pair_shared_read_decode_limit')
            index.limits = replace(index_limits, max_read_bytes=limits.read_bytes-read_bytes,
                max_decoded_bytes=limits.decoded_bytes-decoded, max_seconds=limits.seconds-(time.monotonic()-started)-30)
            chunk = work/'chunks'/f'{start//limits.batch_files:03d}'
            item = {'path': chunk.relative_to(work).as_posix(), 'status': 'started', 'expected_files': len(ordered[start:start+limits.batch_files])}
            report['chunks'].append(item)
            save()
            before = getattr(index, 'last_budget', None)
            try:
                extracted = index.extract([r['name'] for r in ordered[start:start+limits.batch_files]], chunk)
                item['status'] = 'extracted_verified'
                for row in extracted['files']:
                    path = chunk/row['relative_path']
                    report['stage'] = 'png_audit'
                    guard(20)
                    audited = audit_png(path, row, poses, work, representatives)
                    report['files'].append({**row, 'path': path.relative_to(work).as_posix(), **audited})
                    if len(report['files']) % 8 == 0:
                        save()
                item['status'] = 'content_audited'
            finally:
                budget = getattr(index, 'last_budget', None)
                if budget is not None and budget is not before:
                    stats = budget.stats()
                    item['stats'] = stats
                    report['observed_peak_rss'] = max(report['observed_peak_rss'], stats['observed_peak_rss'])
                    read_bytes += stats['source_read_bytes']
                    decoded += stats['returned_decoded_bytes']
                save()
        report['status'] = 'completed'
    except BaseException as exc:
        # 예외 문자열/credential은 회수 보고서에 넣지 않는다.
        reason = exc.reason if isinstance(exc, (IndexFailure, PreparationError)) else type(exc).__name__
        report.update(status='partial_or_failed', stop_reason=reason, error_type=type(exc).__name__)
    finally:
        cleanup_errors = []
        if session is not None:
            try:
                session.close()
            except BaseException as exc:
                cleanup_errors.append({'stage': 'session_close', 'error_type': type(exc).__name__})
        if index is not None:
            try:
                report['index_after_sha256'] = sha_file(work/'index.sqlite')
                report['index_unchanged'] = report['index_after_sha256'] == report.get('restored_db_before_sha256')
            except BaseException as exc:
                cleanup_errors.append({'stage': 'index_after_sha256', 'error_type': type(exc).__name__})
                report['index_unchanged'] = False
            if not report['index_unchanged']:
                report.update(status='partial_or_failed', stop_reason='pair_index_changed')
        if cleanup_errors:
            report['cleanup_errors'] = cleanup_errors
            if report['status'] == 'completed':
                report.update(status='partial_or_failed', stop_reason='pair_cleanup_failed')
        files = {(f['sequence'], f['frame'], f['side']): f for f in report['files']}
        for pair in config['pairs']:
            seq, frame = pair['sequence'], pair['frame']
            sides = [side for side in ('left', 'right') if (seq, frame, side) in files]
            report['pairs'].append({'sequence': seq, 'frame': frame, 'audited_views': sides,
                'complete': sides == ['left', 'right'], 'pose_present': True,
                'split': next(r['split'] for r in candidate['records'] if (r['sequence_id'], r['frame_index']) == (seq, frame))
                    if 'candidate' in locals() else 'not_audited'})
        report['complete_pairs'] = sum(p['complete'] for p in report['pairs'])
        report['all_targets_content_audited'] = report['status'] == 'completed' and report['complete_pairs'] == len(config['pairs'])
        report.update(human_visual_alignment_confirmed=False, timestamp_semantics='unverified',
            original_motion_identity='unverified', validation_pairs=0, whole_archive_integrity_verified=False)
        save()
    return report


def audit_png(path, row, poses, work, representatives):
    """원본 PNG를 한 장씩 decode. GT annotation을 visibility로 취급하지 않는다."""
    from PIL import Image, ImageDraw
    seq, frame, side = identify(row['name'], 'rgb')
    check(path.stat().st_size == row['size'] and sha_file(path) == row['sha256'], 'pair_png_payload_hash')
    pose = poses[(seq, frame)]
    field = pose['fields']['camera_'+side+'_pts2d']
    check(len(field['coordinates']) == len(field['valid_mask']) == 62, 'pair_joint62_contract')
    with Image.open(path) as image:
        check(image.format == 'PNG' and image.size == (1024, 1024) and image.mode == 'RGB', 'pair_png_native_format_size')
        image.load()
        key = (seq, side)
        overlay = None
        if key not in representatives:
            canvas = image.copy()
            draw = ImageDraw.Draw(canvas)
            for number, (point, valid) in enumerate(zip(field['coordinates'], field['valid_mask'])):
                if valid and 0 <= point[0] < 1024 and 0 <= point[1] < 1024:
                    x, y = point
                    draw.ellipse((x-4, y-4, x+4, y+4), outline='#00ff88' if number < 22 else '#ffcc33', width=2)
            directory = work/'overlays'
            directory.mkdir(exist_ok=True)
            target = directory/f'{len(representatives):02d}_{side}.png'
            canvas.save(target)
            overlay = target.relative_to(work).as_posix()
            representatives.add(key)
    outside = sum(bool(valid and not (0 <= p[0] <= 1023 and 0 <= p[1] <= 1023)) for p, valid in zip(field['coordinates'], field['valid_mask']))
    return {'sequence': seq, 'frame': frame, 'side': side, 'png_decode': 'passed', 'native_size': [1024, 1024],
        'joint_count': 62, 'raw_joint_count': len(pose['source']['raw_joint_names']), 'distance_unit': 'mm',
        'annotation_valid_count': sum(field['valid_mask']), 'outside_image_count': outside,
        'visibility_semantics': 'unknown', 'pose_sha256': pose['source']['sha256'], 'representative_overlay': overlay}


def recover(input_root, workspace, output, *, expected_config_sha256):
    """완료·실패chunk와 raw pose를 회수한다. 원격/추출/인증 호출 없음."""
    root, work, output = map(_path, (input_root, workspace, output))
    config, pin = preflight(root)
    check(pin == expected_config_sha256, 'pair_recovery_config_pin')
    check(not output.exists() and not output.is_relative_to(root) and not output.is_relative_to(work), 'pair_recovery_new_output')
    limits = PairLimits(**config['limits'])
    selected = []
    report = {'status': 'report_missing_execution_state_unknown'}
    if (work/'report.json').is_file():
        report = read(work/'report.json', 8*MIB)
        check(report['config_sha256'] == pin and report['job_id'] == config['job_id'], 'pair_recovery_job_pin')
    if work.exists():
        for n, p in enumerate(work.rglob('*')):
            check(n < 2048, 'pair_recovery_inventory')
            _path(p)
            if p.is_file() and p.relative_to(work).parts[0] != 'index.sqlite':
                selected.append((p, 'run/'+p.relative_to(work).as_posix()))
    # 원본 index snapshot은 이미 별도 보존되어 있다. 작은 pose/metadata를 결과에 함께 넣는다.
    for p in root.rglob('*'):
        if p.is_file() and p.relative_to(root).parts[0] != 'snapshot':
            selected.append((p, 'inputs/'+p.relative_to(root).as_posix()))
    total = sum(p.stat().st_size for p, _ in selected)
    check(total+MIB <= limits.zip_bytes, 'pair_recovery_zip_limit')
    disk = sum(p.stat().st_size for p in work.rglob('*') if p.is_file()) if work.exists() else 0
    check(disk <= limits.work_bytes and disk+total+MIB <= limits.work_bytes+limits.zip_bytes, 'pair_recovery_disk_limit')
    started = time.monotonic()
    with ZipFile(output, 'x', ZIP_STORED) as z:
        for p, name in selected:
            check(time.monotonic()-started < 120, 'pair_recovery_time_limit')
            z.write(p, name)
        z.writestr('recovery.json', canonical({'schema': 'indexed_pair_recovery_v1', 'job_id': config['job_id'],
            'config_sha256': pin, 'status': report['status'], 'archive_network_requested': False, 'train_ready': False}))
    return {'path': str(output), 'bytes': output.stat().st_size, 'sha256': sha_file(output), 'network_requested': False}
