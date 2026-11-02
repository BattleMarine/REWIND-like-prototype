"""검토 제안/차단 감사와 실제 release 또는 전용 fixture의 bounded RGB2D 실행."""
import argparse
import json
from pathlib import Path

from rewind_archive.contracts import canonical,digest
from rewind_bundle_data.integrity import local,require,MIB
from .contracts import audit_candidate,draft,read,seal


def output_directory(path,protected):
    path = Path(path).absolute();parent = local(path.parent);path = parent/path.name
    require(not path.exists() and not path.is_symlink() and
            all(not path.is_relative_to(p) and not p.is_relative_to(path) for p in protected), 'release_new_separate_output')
    path.mkdir()
    return path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command',required=True)
    audit = sub.add_parser('audit-candidate');audit.add_argument('--candidate',required=True);audit.add_argument('--output',required=True)
    proposal = sub.add_parser('propose')
    for k in ('bundle','completion-sha256','config','review','output'):proposal.add_argument('--'+k,required=True)
    for name in ('train','development-fixture-train','validate','development-fixture-validate'):
        run = sub.add_parser(name)
        for k in ('bundle','completion-sha256','config','output'):run.add_argument('--'+k,required=True)
        if not name.startswith('development-fixture'):
            for k in ('release','review','policy','policy-sha256'):run.add_argument('--'+k,required=True)
        run.add_argument('--steps',type=int,required=name.endswith('train'))
        if name.endswith('validate'):run.add_argument('--checkpoint',required=True)
        run.add_argument('--resume',type=Path);run.add_argument('--device',default='cpu')
        run.add_argument('--max-attempts',type=int,default=32);run.add_argument('--max-seconds',type=float,default=300.)
        run.add_argument('--checkpoint-every',type=int,default=1);run.add_argument('--evaluate-every',type=int,default=4)
        run.add_argument('--log-limit',type=int,default=32);run.add_argument('--export-weights',type=Path)
    export = sub.add_parser('export');export.add_argument('--checkpoint',required=True);export.add_argument('--output',required=True)
    args = parser.parse_args(argv)
    if args.command == 'export':
        from .weights import checkpoint,export_checkpoint
        protected = local(args.checkpoint)
        target = Path(args.output).absolute()
        require(local(target.parent)/target.name != protected, 'release_export_input_overlap')
        state,pin = checkpoint(protected)
        return export_checkpoint(state,pin,target)
    if args.command == 'audit-candidate':
        output = output_directory(args.output,[local(args.candidate)])
        result = audit_candidate(args.candidate)
        (output/'report.json').write_bytes(canonical(result)+b'\n')
        return result
    if args.command == 'propose':
        output = output_directory(args.output,[local(p) for p in (args.bundle,args.review,args.config)])
        try:
            result = draft(args.bundle,args.completion_sha256,read(args.config,65536)[0],args.review)
            (output/'release.json').write_bytes(canonical(result)+b'\n')
            (output/'report.json').write_bytes(canonical({'status':'proposal_only_no_authority','release_issued':False})+b'\n')
            return result
        except Exception as exc:
            (output/'report.json').write_bytes(canonical({'status':'blocked','release_issued':False,'reason':str(exc)[:256]})+b'\n')
            raise
    from rewind_m1.shard_cli import train
    from .training import ReleasedRGBTrainer
    from .weights import checkpoint,export_checkpoint
    mode = 'development_fixture' if args.command.startswith('development-fixture') else 'reviewed_real'
    request = {'mode':mode,'bundle':str(local(args.bundle)),'completion_sha256':args.completion_sha256}
    if mode == 'reviewed_real':
        request.update(release=str(local(args.release)),review=str(local(args.review)),policy=str(local(args.policy)),
                       expected_policy_sha256=args.policy_sha256) # 명시 운영자 인자. 파일에서 추론하지 않는다.
    protected = [local(args.bundle),local(args.config)]
    protected += [local(p) for p in (args.release,args.review,args.policy)] if mode == 'reviewed_real' else []
    if args.resume:protected.append(local(args.resume))
    args.output = output_directory(args.output,protected)
    require((args.steps is None or 1 <= args.steps <= 64) and 1 <= args.max_attempts <= 128 and 0 < args.max_seconds <= 600 and
            (mode != 'development_fixture' or args.device == 'cpu'), 'release_run_bounds')
    if args.command.endswith('validate'):
        import time
        started = time.monotonic()
        def check():
            if time.monotonic()-started >= args.max_seconds:raise TimeoutError('release_validation_budget')
        try:
            state,identity = checkpoint(args.checkpoint)
            trainer = ReleasedRGBTrainer(read(args.config,65536)[0],request,device=args.device,check_budget=check)
            trainer.resume(state)
            result = {'status':'passed','checkpoint_sha256':identity,'metrics':trainer.evaluate(),
                      'training_scope':trainer.TRAINING_SCOPE,'training_run':False,'full_training_ready':False}
            (args.output/'report.json').write_bytes(canonical(result)+b'\n')
            return result
        except Exception as exc:
            (args.output/'report.json').write_bytes(canonical({'status':'failed','reason':str(exc)[:256]})+b'\n')
            raise
    if args.export_weights:
        target = Path(args.export_weights).absolute()
        require(target.is_relative_to(args.output) and not target.exists(), 'release_export_new_output_required')
    # 공통 durable CLI에 실제 request를 제공한다. 임시 synthetic catalog는 만들지 않는다.
    envelope = seal({'request':request})
    args.catalog = Path(args.bundle)/'COMPLETE.json';args.config = Path(args.config);args.epochs = None
    return train(args,trainer_class=ReleasedRGBTrainer,catalog_reader=lambda _:envelope,
                 checkpoint_reader=checkpoint,exporter=export_checkpoint,run_schema='bounded_released_rgb2d_run_v1',
                 training_scope=mode+'_rgb2d_t1_only')


if __name__ == '__main__':
    main()
