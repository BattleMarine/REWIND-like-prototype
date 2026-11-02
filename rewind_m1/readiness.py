"""오프라인 M1 특징 shard 사전 검사. full dataset 학습 승인·원격 준비를 수행하지 않는다."""
import argparse
import hashlib
import json
from pathlib import Path

from rewind_prepare.converter import validate_candidate
from .shards import DEFAULT_LIMITS, ShardDataset, _read, build_catalog, read_catalog


def inspect_catalog(path, full_read=False):
    catalog = read_catalog(path)
    declared = {}
    for shard in catalog['shards']:
        for seq in shard['sequences']:
            declared[seq['split']] = declared.get(seq['split'], 0) + seq['records']
    report = {'schema_version': 'm1_training_readiness_v1', 'status': 'declarations_valid',
              'catalog_sha256': catalog['content_sha256'], 'declaration_records': declared,
              'full_payload_read': False, 'verified_records': None, 'full_training_ready': False,
              'supported_path': 'local_m1_features_complete_sequence_shards',
              'model_support': {'m1_regression': 'loader_and_sampler_only', 'm2_body': 'not_connected',
                                'm3_whole_body': 'not_connected', 'rgb_2d_frontend': 'not_connected'},
              'blockers': ['shard_trainer_not_connected', 'persistent_preparation_supply_not_connected',
                           'colab_gpu_pilot_not_measured']}
    if not {'train', 'validation'} <= set(declared):
        report['blockers'].append('required_train_validation_splits_missing')
    if catalog['contract']['data_kind'] == 'dataset_sample':
        report['blockers'] += ['release_evidence_not_independently_verified', 'original_motion_identity_not_independently_verified']
    else:
        report['blockers'].append('synthetic_fixture_is_not_dataset_readiness')
    if full_read:
        report['verified_records'] = ShardDataset(catalog, Path(path).parent).verify_all()
        report['full_payload_read'] = True
        report['status'] = 'local_payloads_valid'
    return report


def inspect_candidate(path):
    raw = _read(path, DEFAULT_LIMITS['max_shard_bytes'])
    doc = validate_candidate(json.loads(raw))
    eligibility = doc['training_eligibility']
    return {'schema_version': 'm1_candidate_readiness_v1', 'status': 'candidate_blocked',
            'candidate_sha256': hashlib.sha256(raw).hexdigest(), 'full_training_ready': False,
            'existing_training_eligibility': eligibility,
            'scope': 'candidate_schema_and_declared_blockers_only_not_raw_or_geometry_revalidation',
            'records': len(doc['records']), 'blockers': list(eligibility['blockers'])}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    build = sub.add_parser('build')
    build.add_argument('--base', type=Path, required=True)
    build.add_argument('--shard', action='append', required=True)
    build.add_argument('--window-size', type=int, default=1)
    build.add_argument('--limits', type=Path)
    build.add_argument('--output', type=Path, required=True, help='base 바로 아래 catalog 경로')
    inspect = sub.add_parser('inspect')
    inspect.add_argument('--catalog', type=Path, required=True)
    inspect.add_argument('--full-read', action='store_true')
    candidate = sub.add_parser('candidate')
    candidate.add_argument('--candidate', type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == 'build':
            if args.output.resolve().parent != args.base.resolve():
                raise ValueError('catalog_output_must_be_in_base')
            limits = json.loads(_read(args.limits, 4096)) if args.limits else None
            catalog = build_catalog(args.shard, args.base, args.window_size, limits)
            with args.output.open('x', encoding='utf8') as stream:
                stream.write(json.dumps(catalog, ensure_ascii=False, indent=2)+'\n')
            result = inspect_catalog(args.output, full_read=False)
            result['build_payloads_fully_read'] = True
        elif args.command == 'inspect':
            result = inspect_catalog(args.catalog, args.full_read)
        else:
            result = inspect_candidate(args.candidate)
        print(json.dumps(result, ensure_ascii=False, indent=2))
    except (ValueError, OSError, KeyError, TypeError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
