"""고정 g9/235MiB 캐시를 읽어 RGB102 추출 입력을 만든다. 원격·학습 없음."""
import argparse
from collections import defaultdict
import gzip
import hashlib
import io
import json
from pathlib import Path
import shutil
import sqlite3
import sys
import tarfile
import time
from zipfile import ZipFile, ZIP_DEFLATED

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from rewind_archive.contracts import canonical, digest, safe_path
from rewind_archive.matching import identify
from rewind_prepare.converter import convert

RGB_SHA = 'd3e35128ee38278716515f3b111d306c81e5f3cb1df57ae70f31a1366a79cee1'
CACHE_SHA = 'f902b25aadb1932c4c734795592e122061f8d30410bdeed5bb3866528ebff437'
POSE_SHA = '7ebe1cea2b88e7945a380068e0ac7d3fc26e34794437900f896eae0482865211'
SNAPSHOT = 'ec213878a03784b22fa7e6d53f26f16a61409aa8854228e177fd97f4e2101f8b'
RECEIPT_SHA = '3d4c6c11953318661b5cf905155b7e0384cb87430fe9f97192dff5df4f7351e9'
SOURCE_SHA = '73b7007fc0586e72ddc0109290c79af93b7e6954c2f65a3e708bd6e8917b2c37'
MIB = 1024**2


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        while data := stream.read(65536):
            h.update(data)
    return h.hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical(value)+b'\n')


class LimitedDecoded:
    def __init__(self, stream):
        self.stream, self.total, self.started = stream, 0, time.monotonic()

    def read(self, n):
        if not 0 <= n <= 65536 or time.monotonic()-self.started > 120 or self.total+n > 2400*MIB:
            raise ValueError('local_pose_decode_budget')
        data = self.stream.read(n)
        self.total += len(data)
        return data


def prepare(args):
    out = args.output.absolute()
    if out.exists():
        raise ValueError('new_output_required')
    for path, pin, maximum in ((args.rgb_result, RGB_SHA, 32*MIB), (args.cache, CACHE_SHA, 256*MIB),
                               (args.pose_result, POSE_SHA, 8*MIB)):
        if path.is_symlink() or not path.is_file() or path.stat().st_size > maximum or file_sha(path) != pin:
            raise ValueError('fixed_input_pin')
    out.mkdir(parents=True)
    inputs = out/'inputs'
    inputs.mkdir()
    with ZipFile(args.rgb_result) as z:
        snapshot_name = 'snapshot/g9-'+SNAPSHOT+'.zip'
        raw = z.read(snapshot_name)
        receipt_raw = z.read(snapshot_name+'.receipt.json')
        if hashlib.sha256(raw).hexdigest() != SNAPSHOT or hashlib.sha256(receipt_raw).hexdigest() != RECEIPT_SHA:
            raise ValueError('g9_pin')
        receipt = json.loads(receipt_raw)
        snapshot = inputs/'snapshot'
        snapshot.mkdir()
        (snapshot/receipt['filename']).write_bytes(raw)
        (snapshot/(receipt['filename']+'.receipt.json')).write_bytes(receipt_raw)
        with ZipFile(io.BytesIO(raw)) as bundle:
            db_raw = bundle.read('index.sqlite')
        if hashlib.sha256(db_raw).hexdigest() != receipt['db_sha256']:
            raise ValueError('g9_db_hash')
        db = sqlite3.connect(':memory:')
        try:
            db.deserialize(db_raw)
            state_raw, state_sha = db.execute('SELECT state,state_sha FROM checkpoint').fetchone()
        finally:
            db.close()
        if hashlib.sha256(state_raw).hexdigest() != state_sha:
            raise ValueError('g9_state_hash')
        state = json.loads(state_raw)
        if state['source_sha256'] != SOURCE_SHA or state['generation'] != 9:
            raise ValueError('g9_identity')
        rgb_manifest = json.loads(z.read('manifest.json'))
    views = defaultdict(dict)
    for row in state['members']:
        if row['type'] != 'file':
            continue
        seq, frame, side = identify(row['name'], 'rgb')
        if side in views[(seq, frame)]:
            raise ValueError('duplicate_rgb_key_side')
        views[(seq, frame)][side] = row
    pairs = [{'sequence': seq, 'frame': frame, 'rgb': sides} for (seq, frame), sides in sorted(views.items()) if set(sides) == {'left', 'right'}]
    if len(pairs) != 102 or sum(v['size'] for p in pairs for v in p['rgb'].values()) != 247611434:
        raise ValueError('fixed102_required')
    wanted = {p['sequence']+'/json/frame_'+str(p['frame'])+'.json' for p in pairs}
    with ZipFile(args.cache) as z:
        manifest = json.loads(z.read('manifest.json'))
        if manifest['content_sha256'] != digest({k: v for k, v in manifest.items() if k != 'content_sha256'}):
            raise ValueError('cache_manifest_hash')
        h = hashlib.sha256()
        with z.open('prefix.bin') as stream:
            while data := stream.read(65536):
                h.update(data)
        if h.hexdigest() != manifest['prefix_sha256'] or manifest['prefix_bytes'] != 246415360:
            raise ValueError('cache_prefix_hash')
        entries, total = [], 0
        with z.open('prefix.bin') as compressed, gzip.GzipFile(fileobj=compressed) as gz:
            decoded = LimitedDecoded(gz)
            with tarfile.open(fileobj=decoded, mode='r|') as tar:
                for member in tar:
                    if member.name not in wanted:
                        continue
                    name = safe_path(member.name)
                    if not member.isfile() or not 0 < member.size <= 4*MIB or total+member.size > 16*MIB:
                        raise ValueError('pose_payload_budget')
                    payload = tar.extractfile(member).read(member.size+1)
                    if len(payload) != member.size or not isinstance(json.loads(payload), dict):
                        raise ValueError('pose_payload_invalid')
                    target = inputs/'pose/files'/name
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with target.open('xb') as stream:
                        stream.write(payload)
                    entries.append({'member': name, 'relative_path': 'files/'+name, 'kind': 'pose',
                        'size_bytes': len(payload), 'sha256': hashlib.sha256(payload).hexdigest(), 'tar_complete': True})
                    total += len(payload)
                    if len(entries) == len(wanted):
                        break
        if {e['member'] for e in entries} != wanted:
            raise ValueError('local_pose_targets_missing')
    # 새67개와 기존2개를 독립 원본 보고서 SHA에 대조한다. 나머지33개는 이번 로컬 추출이다.
    with ZipFile(args.pose_result) as z:
        em = json.loads(z.read('archive_job_pose_missing67_001/pilot/extraction_manifest.json'))
    old = json.loads((args.references.parent/'inputs/pose/extraction_manifest.json').read_bytes())
    observed = {e['member']: e for e in entries}
    for entry in [*em['files'], *old['files']]:
        current = observed[entry['member']]
        if current['sha256'] != entry['sha256'] or current['size_bytes'] != entry['size_bytes']:
            raise ValueError('existing69_pose_changed')
    pose_report = {'schema_version': 'archive_extraction_v1', 'archive_manifest': manifest['archive_manifest'],
        'inspection_candidates_usable': True, 'failure_class': 'none', 'train_ready': False,
        'whole_archive_integrity_verified': False, 'files': entries,
        'producer': 'bounded_local_prefix_tar_selection', 'cache_zip_sha256': CACHE_SHA,
        'cache_manifest_sha256': manifest['content_sha256'], 'prefix_sha256': manifest['prefix_sha256'],
        'gzip_eof_crc_verified': False, 'returned_decoded_bytes': decoded.total, 'remote_requests': 0}
    pose_report['content_sha256'] = digest(pose_report)
    write(inputs/'pose/extraction_manifest.json', pose_report)
    metadata = inputs/'metadata'
    metadata.mkdir()
    shutil.copytree(args.references/'split', metadata/'split')
    shutil.copyfile(args.references/'scaramuzza.json', metadata/'scaramuzza.json')
    sequences = sorted({p['sequence'] for p in pairs})
    write(metadata/'motion_map.json', {'schema_version': 'explicit_motion_map_v1', 'sequences': {
        seq: {'motion_id': 'declared:'+seq, 'identity_status': 'declared_unverified',
              'evidence': 'archive sequence별 선언. 원본 motion 독립성은 미검증.'} for seq in sequences}})
    candidate, audit = convert(inputs/'pose/files', inputs/'pose/extraction_manifest.json', metadata/'split',
        metadata/'motion_map.json', ROOT/'configs/real_archvis62_v1.json', metadata/'scaramuzza.json')
    write(inputs/'pose/candidate.json', candidate)
    write(inputs/'pose/audit.json', audit)
    config = {'schema': 'indexed_stereo_pairs_v1', 'job_id': 'rgb102_pairs_001', 'pairs': pairs,
        'snapshot_receipt': receipt, 'receipt_file_sha256': RECEIPT_SHA, 'rgb_manifest': rgb_manifest,
        'source_sha256': SOURCE_SHA, 'pose_extraction_sha256': pose_report['content_sha256'],
        'candidate_sha256': candidate['content_sha256'], 'train_ready': False,
        'limits': {'body_bytes': 512*MIB, 'requests': 3072, 'seconds': 1200, 'read_bytes': 1024*MIB,
                   'decoded_bytes': 256*MIB, 'rss_bytes': 1024*MIB, 'work_bytes': 512*MIB,
                   'zip_bytes': 384*MIB, 'batch_files': 32}}
    write(inputs/'config.json', config)
    # 모든 입력을 한 번 고정한다. g9 DB를 수정하거나 cache 원본을 복사하지 않는다.
    pins = {p.relative_to(inputs).as_posix(): {'bytes': p.stat().st_size, 'sha256': file_sha(p)}
            for p in sorted(inputs.rglob('*')) if p.is_file()}
    write(inputs/'input_manifest.json', {'schema': 'pinned_pair_inputs_v1', 'files': pins})
    bundle = out/'rgb102_inputs.zip'
    with ZipFile(bundle, 'x', ZIP_DEFLATED) as z:
        for p in sorted(inputs.rglob('*')):
            if p.is_file():
                z.write(p, p.relative_to(inputs).as_posix())
    summary = {'status': 'prepared', 'pairs': 102, 'rgb_payload_bytes': 247611434, 'pose_payload_bytes': total,
        'pose_previous69_hashes_matched': True, 'new_local_pose_files': 33, 'pose_returned_decoded_bytes': decoded.total,
        'cache_original_sha256': CACHE_SHA, 'input_zip_sha256': file_sha(bundle), 'input_zip_bytes': bundle.stat().st_size,
        'config_sha256': digest(config), 'remote_requests': 0, 'train_ready': False, 'pose_audit': audit}
    write(out/'preparation_report.json', summary)
    print(json.dumps({k: v for k, v in summary.items() if k != 'pose_audit'}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ('rgb-result', 'cache', 'pose-result', 'references', 'output'):
        parser.add_argument('--'+flag, type=Path, required=True)
    prepare(parser.parse_args())
