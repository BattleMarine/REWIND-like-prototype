"""UI 없는 합성 M1 shard train/evaluate/export. 실제 데이터 release와 공급은 후속."""
import argparse
from collections import deque
import hashlib
import json
from pathlib import Path
import time

import torch

from .__main__ import save_report
from .artifacts import export_weights
from .data import InputContract
from .models import make_model
from .shards import _integer, _read, _require, read_catalog, validate_catalog
from .shard_training import FORMAT, ShardTrainer
from .training import save_checkpoint, source_identity, validate_config

MAX_CHECKPOINT_BYTES = 256*1024**2


def read_checkpoint(path):
    import io
    raw = _read(path, MAX_CHECKPOINT_BYTES)
    state = torch.load(io.BytesIO(raw), map_location='cpu', weights_only=True)
    _require(isinstance(state, dict) and state.get('format_version') == FORMAT, 'shard_checkpoint_format')
    return state, hashlib.sha256(raw).hexdigest()


def export_checkpoint(checkpoint, checkpoint_sha256, output):
    """고정 checkpoint만 읽는다. 원본 shard 접근이나 optimizer 재개는 하지 않는다."""
    _require(checkpoint.get('format_version') == FORMAT and
             checkpoint.get('training_scope') == 'synthetic_fixture_only' and
             checkpoint.get('full_training_ready') is False, 'shard_checkpoint_format')
    config = validate_config(checkpoint['config'])
    catalog = validate_catalog(checkpoint['catalog'])
    contract = InputContract.from_dict(catalog['contract'])
    _require(config['window_size'] == catalog['window_size'] and config['hidden'] <= 1024
             and len(contract.joint_names) <= 256 and contract.data_kind == 'synthetic_fixture', 'shard_export_model_contract')
    _require(checkpoint['source']['code_sha256'] == source_identity()['code_sha256'], 'shard_export_code_mismatch')
    with torch.random.fork_rng(devices=[]):
        model = make_model(config, len(contract.joint_names)*14, len(contract.joint_names))
    model.load_state_dict(checkpoint['model'], strict=True)
    return export_weights(model, config, catalog['contract'], catalog['content_sha256'],
                          checkpoint['source'], checkpoint['step'], output, checkpoint_sha256=checkpoint_sha256)


def train(args, *, trainer_class=ShardTrainer, catalog_reader=read_catalog,
          checkpoint_reader=read_checkpoint, exporter=export_checkpoint, run_schema='m1_shard_run_v1'):
    output = args.output
    config = json.loads(_read(args.config, 65536))
    catalog = catalog_reader(args.catalog)
    for key in ('max_attempts', 'checkpoint_every', 'evaluate_every', 'log_limit'):
        _require(_integer(getattr(args, key), 1, 10**9 if key == 'max_attempts' else 10000), 'shard_cli_'+key)
    _require(type(args.max_seconds) in (int,float) and 0 < args.max_seconds <= 86400, 'shard_cli_max_seconds')
    _require(_integer(args.steps if args.steps is not None else args.epochs, 1, 10**9), 'shard_cli_total_target')
    output.mkdir(parents=True, exist_ok=True)
    last, report_path = output/'last.pt', output/'report.json'
    _require(args.resume is not None or not last.exists(), 'shard_output_exists_use_resume')
    _require(args.resume is not None or not report_path.exists(), 'shard_output_report_exists')
    if args.resume is not None and (last.exists() or report_path.exists()):
        _require(last.exists() and hashlib.sha256(_read(args.resume, MAX_CHECKPOINT_BYTES)).digest() ==
                 hashlib.sha256(_read(last, MAX_CHECKPOINT_BYTES)).digest(), 'shard_resume_output_identity_mismatch')
    if args.export_weights is not None:
        _require(not args.export_weights.exists() and args.export_weights.resolve() not in (last.resolve(), report_path.resolve()),
                 'shard_weights_output_conflict')
    trainer = None
    log = deque(maxlen=args.log_limit)
    started = time.monotonic()
    initial_attempts = 0
    durable = None
    can_write_report = not report_path.exists()
    report = {'schema_version':run_schema, 'status':'started', 'full_training_ready':False,
              'training_scope':'synthetic_fixture_only', 'limits':{'max_attempts':args.max_attempts,
              'max_seconds':args.max_seconds, 'log_limit':args.log_limit},
              'target':{'steps':args.steps, 'epochs':args.epochs}, 'log':[]}

    def record():
        report.update(elapsed_seconds=time.monotonic()-started, log=list(log), durable_checkpoint=durable)
        save_report(report, report_path)

    def check_budget():
        if time.monotonic()-started >= args.max_seconds:
            raise TimeoutError('shard_cooperative_time_budget')

    def persist():
        nonlocal durable
        check_budget()
        state = trainer.checkpoint()
        save_checkpoint(state, last)
        durable = {'path':str(last), 'sha256':hashlib.sha256(last.read_bytes()).hexdigest(),
                   'step':trainer.step, 'samples_seen':trainer.samples_seen, 'coverage':trainer.coverage()}
        report.update(step=trainer.step, samples_seen=trainer.samples_seen, coverage=trainer.coverage(),
                      attempts=trainer.attempts, skipped_updates=trainer.skipped_updates,
                      source=trainer.source, environment=trainer.environment,
                      catalog_sha256=catalog['content_sha256'])
        record()

    try:
        trainer = trainer_class(config, catalog, args.catalog.parent, args.device, check_budget=check_budget)
        if args.resume:
            checkpoint, checkpoint_sha256 = checkpoint_reader(args.resume)
            durable = {'path':str(args.resume), 'sha256':checkpoint_sha256, 'step':checkpoint['step'],
                       'samples_seen':checkpoint['samples_seen'], 'cursor':checkpoint['cursor']}
            trainer.resume(checkpoint)
            report['resumed_checkpoint_sha256'] = checkpoint_sha256
            _require(args.steps is None or args.steps >= trainer.step, 'shard_target_before_resume_step')
            _require(args.epochs is None or args.epochs >= trainer.sampler.epoch, 'shard_target_before_resume_epoch')
        can_write_report = True
        initial_attempts = trainer.attempts
        # 실패/중단에 대비해 update0 또는 재개 지점부터 일관된 파일을 확보한다.
        persist()
        while (trainer.step < args.steps if args.steps is not None else trainer.sampler.epoch < args.epochs):
            if time.monotonic()-started >= args.max_seconds or trainer.attempts-initial_attempts >= args.max_attempts:
                report['status'] = 'bounded_stop'
                break
            event = trainer.train_step()
            # sample ID 전체를 장기 로그로 누적하지 않는다. 배치 앞8개만 진단용으로 보존한다.
            log.append({**event, 'keys':event.get('keys', [])[:8]})
            if event['updated'] and trainer.step % args.evaluate_every == 0:
                report['validation'] = trainer.evaluate()
            if trainer.step % args.checkpoint_every == 0 or not event['updated']:
                persist()
        else:
            report['status'] = 'completed_requested_target'
        if report['status'] != 'bounded_stop':
            report['validation'] = trainer.evaluate()
        persist()
        if args.export_weights:
            check_budget()
            checkpoint, identity = checkpoint_reader(last)
            report['weights'] = {'path':str(args.export_weights),
                'state_sha256':exporter(checkpoint, identity, args.export_weights),
                'checkpoint_sha256':identity}
            if can_write_report:
                record()
        return report
    except BaseException as exc:
        # 성공 update 후에도 저장 전에 중단되면 이전 durable 파일을 유지한다.
        report.update(status='bounded_stop' if isinstance(exc, TimeoutError) else
            'interrupted' if isinstance(exc, (KeyboardInterrupt, SystemExit)) else 'failed',
            failure={'type':type(exc).__name__, 'reason':str(exc)[:512]},
            in_memory_state_not_saved=True, step_at_failure=trainer.step if trainer else None)
        try:
            if can_write_report:
                record()
        except OSError:
            pass
        if isinstance(exc, TimeoutError):
            return report
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    run = sub.add_parser('train')
    run.add_argument('--config', type=Path, required=True)
    run.add_argument('--catalog', type=Path, required=True)
    run.add_argument('--output', type=Path, required=True)
    target = run.add_mutually_exclusive_group(required=True)
    target.add_argument('--steps', type=int)
    target.add_argument('--epochs', type=int)
    run.add_argument('--resume', type=Path)
    run.add_argument('--device', default='cpu')
    run.add_argument('--max-attempts', type=int, default=1000, help='이번 실행의 시도 상한; AMP skip 포함')
    run.add_argument('--max-seconds', type=float, default=300., help='준비/평가 포함 cooperative 상한')
    run.add_argument('--checkpoint-every', type=int, default=1)
    run.add_argument('--evaluate-every', type=int, default=10)
    run.add_argument('--log-limit', type=int, default=64)
    run.add_argument('--export-weights', type=Path)
    evaluate = sub.add_parser('evaluate')
    evaluate.add_argument('--checkpoint', type=Path, required=True)
    evaluate.add_argument('--catalog', type=Path, required=True)
    evaluate.add_argument('--split', choices=('train','validation','test'), default='validation')
    evaluate.add_argument('--final-test', action='store_true')
    evaluate.add_argument('--device', default='cpu')
    evaluate.add_argument('--output', type=Path, required=True)
    export = sub.add_parser('export')
    export.add_argument('--checkpoint', type=Path, required=True)
    export.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == 'train':
            result = train(args)
        elif args.command == 'evaluate':
            _require(not args.output.exists(), 'shard_evaluation_output_exists')
            checkpoint, identity = read_checkpoint(args.checkpoint)
            trainer = ShardTrainer(checkpoint['config'], read_catalog(args.catalog), args.catalog.parent, args.device)
            trainer.resume(checkpoint)
            result = {'checkpoint_sha256':identity, 'metrics':trainer.evaluate(args.split, args.final_test),
                      'catalog_sha256':trainer.catalog['content_sha256'], 'source':trainer.source,
                      'environment':trainer.environment}
            save_report(result, args.output)
        else:
            checkpoint, identity = read_checkpoint(args.checkpoint)
            result = {'checkpoint_sha256':identity, 'state_sha256':export_checkpoint(checkpoint, identity, args.output)}
        print(json.dumps(result, ensure_ascii=False, indent=2))
    except (ValueError, OSError, KeyError, TypeError, RuntimeError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
