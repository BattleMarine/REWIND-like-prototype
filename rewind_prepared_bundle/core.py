"""원본 producer manifest를 보존하며 RGB와 job별 pose candidate를 연결한다."""
import hashlib
from pathlib import Path
import time

from rewind_archive.contracts import canonical, digest, safe_path
from rewind_archive.matching import identify, verify_extraction
from rewind_archive_index.preparation import _path
from rewind_data.audit import read_json
from rewind_m1.rgb_manifest import build_manifest, _selection
from rewind_prepare.converter import Budget, convert, motion_entries, official_splits, validate_candidate
from rewind_prepare.spec import validate_spec
from rewind_prepare_queue.core import _state, hashed, validate_plan

MIB = 1024**2
LIMITS = {'triplets':32, 'rgb_jobs':4, 'payload_bytes':64*MIB,
          'rgb_payload_bytes':32*MIB, 'pose_payload_bytes':32*MIB,
          'metadata_bytes':8*MIB, 'output_bytes':128*MIB, 'output_files':512,
          'elapsed_seconds':120, 'rss_bytes':1024*MIB}
MARKER = 'COMPLETE.json'


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def local(path):
    # UNC 및 명시 Drive mount를 로컬 bundle 입력으로 사용하지 않는다.
    require(not str(path).replace('\\', '/').startswith(('//', '/content/drive/')), 'bundle_local_path_required')
    return _path(path).resolve(strict=False)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


class Guard:
    def __init__(self):
        import psutil
        self.process = psutil.Process()
        self.started = time.monotonic()
        self.peak_rss = 0
        self.pins = {}
        self.metadata_bytes = 0
        self.output_bytes = 0
        self.output_files = 0

    def check(self):
        self.peak_rss = max(self.peak_rss, self.process.memory_info().rss)
        require(time.monotonic()-self.started <= LIMITS['elapsed_seconds'], 'bundle_elapsed_limit')
        require(self.peak_rss <= LIMITS['rss_bytes'], 'bundle_rss_limit')

    def read(self, path, maximum=MIB, *, metadata=False):
        self.check()
        path = local(path)
        require(path.is_file() and path.stat().st_size <= maximum, 'bundle_input_size_or_missing')
        with path.open('rb') as stream:
            raw = stream.read(maximum+1)
        require(len(raw) <= maximum and path.stat().st_size == len(raw), 'bundle_input_changed')
        pin = (len(raw), sha(raw))
        require(path not in self.pins or self.pins[path] == pin, 'bundle_input_changed')
        if metadata and path not in self.pins:
            self.metadata_bytes += len(raw)
            require(self.metadata_bytes <= LIMITS['metadata_bytes'], 'bundle_metadata_limit')
        self.pins[path] = pin
        self.check()
        return raw

    def write(self, path, raw):
        self.check()
        require(self.output_files < LIMITS['output_files'] and
                self.output_bytes+len(raw) <= LIMITS['output_bytes']-8192, 'bundle_output_limit')
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('xb') as stream:
            stream.write(raw)
        self.output_files += 1
        self.output_bytes += len(raw)
        self.check()

    def document(self, path, doc):
        raw = canonical(doc)+b'\n'
        require(len(raw) <= 64*MIB, 'bundle_document_limit')
        self.write(path, raw)

    def unchanged(self):
        for path, (size, expected) in list(self.pins.items()):
            require(local(path).is_file() and path.stat().st_size == size, 'bundle_input_changed')
            require(sha(self.read(path, size)) == expected, 'bundle_input_changed')


def inventory(root, guard):
    rows, total = [], 0
    paths = []
    for count, path in enumerate(root.rglob('*')):
        guard.check()
        require(count < 4096, 'bundle_inventory_limit')
        path = local(path)
        require(path.is_relative_to(root), 'bundle_inventory_escape')
        paths.append(path)
    for path in sorted(paths, key=lambda p:p.relative_to(root).as_posix()):
        guard.check()
        if path.is_dir():
            continue
        require(path.is_file(), 'bundle_regular_file_required')
        if path == root/MARKER:
            continue
        require(len(rows) < LIMITS['output_files'] and path.stat().st_size <= 64*MIB, 'bundle_inventory_limit')
        size = path.stat().st_size
        total += size
        require(total <= LIMITS['output_bytes']-8192, 'bundle_output_limit')
        h = hashlib.sha256()
        with path.open('rb') as stream:
            while chunk := stream.read(65536):
                guard.check()
                h.update(chunk)
        require(path.stat().st_size == size, 'bundle_inventory_changed')
        rows.append({'path':path.relative_to(root).as_posix(), 'size_bytes':size, 'sha256':h.hexdigest()})
    return rows, total


def verify_bundle(root):
    """완료 marker와 모든 로컬 파일을 대조한다. 입력 queue나 원본은 필요 없다."""
    root = local(root)
    guard = Guard()
    marker = read_json(guard.read(root/MARKER, MIB))
    require(marker.get('schema_version') == 'prepared_bundle_completion_v1' and
            marker.get('content_sha256') == hashed(marker)['content_sha256'] and
            marker.get('status') == 'prepared_not_training_release' and marker.get('train_ready') is False,
            'bundle_completion_invalid')
    rows, total = inventory(root, guard)
    require(rows == marker['files'] and total == marker['total_bytes'], 'bundle_inventory_mismatch')
    report = read_json(guard.read(root/'report.json', MIB))
    require(report.get('train_ready') is False and report.get('status') == marker['status'] and
            report.get('input_sha256') == marker.get('input_sha256'), 'bundle_report_completion_mismatch')
    names = {'plan','state','matching','selection','spec','calibration','motion_map','train','validation','test'}
    require(isinstance(marker.get('input_sha256'),dict) and set(marker['input_sha256']) == names, 'bundle_input_pins')
    for name, expected in marker['input_sha256'].items():
        path = root/'metadata/split'/(name+'.txt') if name in ('train','validation','test') else root/'metadata'/(name+'.json')
        require(sha(guard.read(path,4*MIB)) == expected, 'bundle_input_pin_mismatch')
    return marker


def build_bundle(plan_path, queue_root, selection_path, spec_path, calibration_path,
                 split_dir, motion_map_path, output, *, manifest_id):
    """queue 전체 target을 명시 선택한 새 로컬 bundle만 만든다. 기존 출력 재개 없음."""
    queue_root, output = local(queue_root), local(output)
    metadata_paths = [local(p) for p in (plan_path, selection_path, spec_path, calibration_path, motion_map_path)]
    split_dir = local(split_dir)
    protected = [queue_root, split_dir, *metadata_paths]
    require(all(not output.is_relative_to(p) and not p.is_relative_to(output) for p in protected), 'bundle_output_input_overlap')
    require(output.parent.is_dir(), 'bundle_output_parent_required')
    output.mkdir(exist_ok=False)
    report = {'schema_version':'prepared_bundle_report_v1', 'status':'failed', 'train_ready':False,
              'training_release':'not_implemented', 'training_run':False, 'queue_modified':False,
              'source_identity_scope':'pinned_plan_and_extraction_not_live_archive',
              'limits':dict(LIMITS), 'blockers':[], 'pose_candidates':[]}
    guard = None
    try:
        guard = Guard()
        raw_metadata = {}
        def document(path, maximum=MIB):
            raw = guard.read(path, maximum, metadata=True)
            raw_metadata[local(path)] = raw
            return read_json(raw)

        plan = validate_plan(document(plan_path))
        require(document(queue_root/'plan.json') == plan, 'bundle_pinned_plan_changed')
        state = document(queue_root/'state.json')
        require(_state(queue_root, plan) == state and state['status'] == 'prepared_not_training_release' and
                state['inflight'] is None and state['cursor'] == len(plan['jobs']), 'bundle_queue_not_completed')
        matching = document(queue_root/'matching.json')
        require(matching.get('schema_version') == 'preparation_queue_matching_v1' and
                matching.get('content_sha256') == hashed(matching)['content_sha256'] and
                matching.get('plan_sha256') == plan['content_sha256'] and matching.get('train_ready') is False and
                state['matching'] == {'path':'matching.json', 'content_sha256':matching['content_sha256'],
                                     'matched_triplet_count':matching.get('matched_triplet_count')}, 'bundle_matching_identity')
        selection = document(selection_path)
        selected, selected_rows = _selection(selection)
        require(len(selected) <= LIMITS['triplets'], 'bundle_triplet_limit')
        # sequence 순서까지 canonical하게 만든다. 사용자가 제공한 frame 순서는 기존 validator가 검사한다.
        selection = sorted(selection, key=lambda row:row['sequence_id'])
        expected_rows = {}
        for job in plan['jobs']:
            for member in job['targets']:
                seq, frame, view = identify(member, job['manifest']['kind'])
                expected_rows.setdefault((seq,frame), {})[view] = (job['id'],member)
        require(selected == set(expected_rows), 'bundle_selection_must_equal_complete_queue')
        require(len(matching['frames']) == len(selected) == matching['matched_triplet_count'], 'bundle_matching_count')
        rgb_jobs = [j for j in plan['jobs'] if j['manifest']['kind'] == 'rgb']
        require(1 <= len(rgb_jobs) <= LIMITS['rgb_jobs'], 'bundle_rgb_job_limit')
        require(len({j['manifest']['source_scope'] for j in plan['jobs']}) == 1, 'bundle_mixed_source_scope')

        spec = validate_spec(document(spec_path, 262144))
        calibration = document(calibration_path)
        require(sha(raw_metadata[local(calibration_path)]) == spec['pinned_sources']['scaramuzza_sha256'] and
                calibration.get('size') == [1024,1024], 'bundle_calibration_pin')
        for split in ('train','validation','test'):
            raw = guard.read(split_dir/(split+'.txt'), 4*MIB, metadata=True)
            raw_metadata[local(split_dir/(split+'.txt'))] = raw
        owners, _ = official_splits(split_dir, spec, Budget(spec['limits']))
        motion = document(motion_map_path)
        sequences = {seq for seq,_ in selected}
        motions = motion_entries(motion, sequences, owners)
        require(all(row['split'] == owners[row['sequence_id']] for row in selected_rows), 'bundle_selection_split_mismatch')

        prepared, actual, payload = [], {}, {'rgb':0, 'pose':0}
        for job,row in zip(plan['jobs'], state['jobs']):
            guard.check()
            source = local(queue_root/row['output'])
            require(source.is_relative_to(queue_root/'attempts'), 'bundle_extraction_escape')
            path = source/'extraction_manifest.json'
            doc = document(path)
            require(digest(doc) == row['output_manifest_sha256'] and doc.get('targets_complete') is True and
                    doc.get('exact_targets') == job['targets'] and
                    doc.get('persistent_index',{}).get('source_sha256') == job['source_sha256'], 'bundle_extraction_pin')
            require({f['member'] for f in doc['files']} == set(job['targets']) and
                    len(doc['files']) == len(job['targets']), 'bundle_extraction_target_scope')
            for entry in doc['files']:
                kind = job['manifest']['kind']
                require(type(entry['size_bytes']) is int and 1 <= entry['size_bytes'] <= (4*MIB if kind == 'pose' else 8*MIB), 'bundle_payload_file_limit')
                payload[kind] += entry['size_bytes']
                require(payload[kind] <= LIMITS[kind+'_payload_bytes'] and sum(payload.values()) <= LIMITS['payload_bytes'], 'bundle_payload_limit')
                require(entry['relative_path'] == 'files/'+safe_path(entry['member']), 'bundle_payload_path')
                file = local(source/entry['relative_path'])
                require(file.is_file() and file.stat().st_size == entry['size_bytes'], 'bundle_payload_missing_size')
                seq, frame, view = identify(entry['member'], kind)
                views = actual.setdefault((seq,frame), {})
                require(view not in views, 'bundle_duplicate_view')
                views[view] = {'job_id':job['id'], 'member':entry['member'], 'sha256':entry['sha256']}
            prepared.append((job, source, doc))
        recomputed = [{'sequence_id':seq, 'frame_index':frame, 'views':views} for (seq,frame),views in sorted(actual.items())]
        require(recomputed == matching['frames'] and set(actual) == selected and
                all(set(v) == {'left','right','pose'} for v in actual.values()), 'bundle_matching_recomputed_mismatch')
        for key, views in actual.items():
            require(all((v['job_id'],v['member']) == expected_rows[key][side] for side,v in views.items()), 'bundle_plan_view_mismatch')
        for job, source, doc in prepared:
            if job['manifest']['kind'] == 'pose':
                job_sequences = {identify(f['member'],'pose')[0] for f in doc['files']}
                subset = {'schema_version':'explicit_motion_map_v1', 'sequences':{s:motions[s] for s in sorted(job_sequences)}}
                fixed_bytes = sum(len(raw_metadata[local(p)]) for p in
                                  [calibration_path, *(split_dir/(s+'.txt') for s in ('train','validation','test'))])
                # convert가 raw JSON을 앞뒤로 두 번 읽는 비용까지 기존 spec 상한 안에 둔다.
                estimated_input = 2*sum(f['size_bytes'] for f in doc['files'])+len(canonical(subset))+1+fixed_bytes+len(raw_metadata[local(source/'extraction_manifest.json')])
                require(len(doc['files']) <= spec['limits']['max_files'] and
                        estimated_input <= spec['limits']['max_total_input_bytes'], 'bundle_pose_converter_input_limit')
                input_sizes = [f['size_bytes'] for f in doc['files']]+[len(canonical(subset))+1,
                    len(raw_metadata[local(source/'extraction_manifest.json')]),len(raw_metadata[local(calibration_path)]),
                    *(len(raw_metadata[local(split_dir/(s+'.txt'))]) for s in ('train','validation','test'))]
                require(max(input_sizes) <= spec['limits']['max_file_bytes'], 'bundle_pose_converter_file_limit')

        input_hashes = {name:sha(raw_metadata[local(path)]) for name,path in
                        [('plan',plan_path), ('state',queue_root/'state.json'), ('matching',queue_root/'matching.json'),
                         ('selection',selection_path), ('spec',spec_path), ('calibration',calibration_path), ('motion_map',motion_map_path),
                         *((s,split_dir/(s+'.txt')) for s in ('train','validation','test'))]}
        report.update(plan_sha256=plan['content_sha256'], state_sha256=state['content_sha256'],
                      matching_sha256=matching['content_sha256'], matched_triplet_count=len(selected),
                      payload_bytes=payload, source_scope=plan['jobs'][0]['manifest']['source_scope'],
                      evidence_scope=('synthetic_integration_only' if plan['jobs'][0]['manifest']['source_scope'] == 'synthetic_fixture'
                                      else 'local_extraction_diagnostic_only'), input_sha256=input_hashes)
        for name, path in [('plan',plan_path), ('state',queue_root/'state.json'), ('matching',queue_root/'matching.json'),
                           ('selection',selection_path), ('spec',spec_path), ('calibration',calibration_path), ('motion_map',motion_map_path)]:
            guard.write(output/'metadata'/(name+'.json'), raw_metadata[local(path)])
        for split in ('train','validation','test'):
            guard.write(output/'metadata/split'/(split+'.txt'), raw_metadata[local(split_dir/(split+'.txt'))])

        # 완전 extraction을 그대로 복사한다. 원본 report 재발급/부분 합성은 하지 않는다.
        source_reports = []
        for job, source, doc in prepared:
            require(verify_extraction(source,job['manifest']) == doc, 'bundle_extraction_changed')
            destination = output/'inputs'/job['id']
            guard.write(destination/'extraction_manifest.json', raw_metadata[local(source/'extraction_manifest.json')])
            for entry in doc['files']:
                raw = guard.read(source/entry['relative_path'], 8*MIB)
                require(len(raw) == entry['size_bytes'] and sha(raw) == entry['sha256'], 'bundle_payload_changed')
                guard.write(destination/entry['relative_path'],raw)
            verify_extraction(destination,job['manifest'])
            source_reports.append({'job_id':job['id'], 'source_sha256':job['source_sha256'],
                                   'extraction_manifest_sha256':digest(doc), 'producer_content_sha256':doc['content_sha256'],
                                   'path':f"inputs/{job['id']}/extraction_manifest.json"})

        # RGB builder 최악 metadata 8MiB를 호출 전에 예약한다. 기존32MiB payload 상한은 위에서 검사했다.
        require(guard.output_bytes+8*MIB <= LIMITS['output_bytes']-8192, 'bundle_rgb_output_reserve')
        rgb, rgb_report = build_manifest(output/'inputs', [output/'inputs'/j['id'] for j in rgb_jobs],
                                         selection, [1024,1024], output/'rgb', manifest_id=manifest_id)
        files, guard.output_bytes = inventory(output,guard)
        guard.output_files = len(files)
        blockers = set(['candidate_schema_not_training_release', 'rgb_pose_alignment_not_checked',
                        'independent_validation_not_verified', 'candidate_to_learning_shard_not_implemented',
                        'timestamps_and_whole_archive_coverage_unverified'])
        record_map = {}
        for job, source, doc in prepared:
            if job['manifest']['kind'] != 'pose':
                continue
            job_sequences = {identify(f['member'],'pose')[0] for f in doc['files']}
            subset = {'schema_version':'explicit_motion_map_v1', 'sequences':{s:motions[s] for s in sorted(job_sequences)}}
            folder = output/'pose'/job['id']
            guard.document(folder/'motion_map.json', subset)
            candidate,audit = convert(output/'inputs'/job['id']/'files', output/'inputs'/job['id']/'extraction_manifest.json',
                                      output/'metadata/split',folder/'motion_map.json',output/'metadata/spec.json',output/'metadata/calibration.json')
            validate_candidate(candidate)
            expected_keys = {identify(f['member'],'pose')[:2] for f in doc['files']}
            require([(r['sequence_id'],r['frame_index']) for r in candidate['records']] == sorted(expected_keys), 'bundle_candidate_keys')
            for index,r in enumerate(candidate['records']):
                key = (r['sequence_id'],r['frame_index'])
                require(key not in record_map and r['split'] == owners[key[0]] and r['motion_id'] == motions[key[0]]['motion_id'], 'bundle_candidate_identity')
                record_map[key] = {'candidate':f"pose/{job['id']}/candidate.json", 'record_index':index, 'job_id':job['id']}
            candidate_raw = canonical(candidate)+b'\n'
            require(len(candidate_raw) <= spec['limits']['max_output_bytes'], 'bundle_pose_converter_output_limit')
            guard.write(folder/'candidate.json',candidate_raw)
            guard.document(folder/'audit.json',audit)
            blockers.update(audit['blockers'])
            report['pose_candidates'].append({'job_id':job['id'], 'path':f"pose/{job['id']}/candidate.json",
                'content_sha256':candidate['content_sha256'], 'records':len(candidate['records']),
                'motion_map_parent_sha256':sha(raw_metadata[local(motion_map_path)]), 'motion_map_sequence_subset':sorted(job_sequences),
                'blockers':audit['blockers']})
        require(set(record_map) == selected, 'bundle_candidate_coverage')
        provenance = {'schema_version':'prepared_bundle_provenance_v1', 'train_ready':False,
            'plan_sha256':plan['content_sha256'], 'matching_sha256':matching['content_sha256'], 'sources':source_reports,
            'rgb_root':'inputs', 'rgb_manifest':'rgb/manifest.json', 'rgb_content_sha256':rgb['content_sha256'],
            'records':[{'sequence_id':r['sequence_id'], 'frame_index':r['frame_index'], 'split':r['split'],
                        'rgb_record_index':i, 'pose':record_map[(r['sequence_id'],r['frame_index'])],
                        'views':actual[(r['sequence_id'],r['frame_index'])]} for i,r in enumerate(rgb['records'])],
            'adapter_code_sha256':{p.name:sha(p.read_bytes()) for p in sorted(Path(__file__).parent.glob('*.py'))}}
        guard.document(output/'provenance.json',hashed(provenance))
        guard.unchanged()
        report.update(status='prepared_not_training_release', blockers=sorted(blockers),
                      candidate_blocker_scope='each_pose_job_separately_not_bundle_split_coverage',
                      rgb_manifest_sha256=rgb['content_sha256'], rgb_verification_scope=rgb_report['verification_scope'],
                      elapsed_seconds=time.monotonic()-guard.started, observed_peak_rss_bytes=guard.peak_rss)
        guard.document(output/'report.json',report)
        rows,total = inventory(output,guard)
        marker = hashed({'schema_version':'prepared_bundle_completion_v1', 'status':'prepared_not_training_release',
                         'train_ready':False, 'input_sha256':input_hashes, 'files':rows, 'total_bytes':total})
        guard.document(output/MARKER,marker)  # 마지막 파일. 실패 디렉터리에는 완료 marker가 없다.
        return marker
    except BaseException as exc:
        # 기존 producer 출력은 건드리지 않는다. adapter가 만든 부분 파일은 검토를 위해 보존한다.
        (output/MARKER).unlink(missing_ok=True)
        report.update(status='failed', error_type=type(exc).__name__, reason=str(exc)[:256])
        if guard:
            report.update(elapsed_seconds=time.monotonic()-guard.started, observed_peak_rss_bytes=guard.peak_rss)
        with (output/'report.json').open('wb') as stream:
            stream.write(canonical(report)+b'\n')
        raise
