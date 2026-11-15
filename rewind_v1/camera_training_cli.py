"""Fixed source v1.5 prepare/pilot/train/evaluate commands with explicit resumption."""

import argparse
from contextlib import redirect_stdout
import gc
import json
from pathlib import Path
import shutil
import sys
import time

import torch

from rewind_archive.contracts import canonical,digest
from . import camera_training as train,camera_training_data as data,camera_training_eval as evaluation,camera_training_recovery as recovery
from . import camera_gauge as gauge,cross_data as cross,cross_recovery
from .cross_cli import _linux_available_ram,model_copy


def preflight(root):
    if shutil.disk_usage(root).free<train.POLICY['limits']['initial_free_disk_bytes']:raise ValueError('v15_initial_disk_16GiB')
    available=_linux_available_ram()
    if available<train.POLICY['limits']['initial_available_ram_bytes']:raise MemoryError('v15_initial_MemAvailable_4GiB')
    return {'available_ram_bytes':available,'ram_scope':'OS MemAvailable, not cgroup guarantee'}


def experiment(args,pilot=False):
    setup=json.loads((args.run/'setup.json').read_bytes());doc,clips,metadata=data.load_features(args.run/'features',setup['features_sha256'])
    cfg=train.pilot_config() if pilot else train.POLICY
    with redirect_stdout(sys.stderr):
        result=train.PairedTraining(cfg,doc['spec'],clips,metadata,{'feature_identity':doc['feature_identity'],
            'features_sha256':doc['content_sha256'],'frontend':doc['frontend'],'frontend_pretraining_frames':20888,
            'teacher_training_frames':1000},args.device,args.run/('pilot' if pilot else 'train'),pilot=pilot)
    return result


def run(args):
    invocation_start=time.monotonic();preflight(args.run);setup_path=args.run/'setup.json'
    if setup_path.exists():
        setup=json.loads(setup_path.read_bytes())
        if setup['source_sha256']!=digest(train.source_identity()):raise ValueError('v15_setup_source')
    else:setup={'run_id':train.RUN_ID,'source_sha256':digest(train.source_identity())}
    stage=args.stage
    if stage=='model':result=model_copy(args.model_source,args.model_local,args.run/'model_status.json')
    elif stage=='references':
        if (args.run/'reference').exists():raise ValueError('v15_no_reference_overwrite')
        gauge.load_references(args.reference,args.control_reference,args.run/'reference')
        setup['reference_receipts']={n:train.sha((args.run/'reference'/n/'recovery_receipt.json').read_bytes()) for n in ('city','control')}
        result={'status':'verified'}
    elif stage=='raw':result=data.restore_train_parts(args.inputs,args.evidence,args.cache,args.work/'raw_train_cache',args.run/'raw_restore.json')
    elif stage=='features':
        result=data.build_train_features(args.inputs,args.evidence,args.work/'raw_train_cache',args.model_local,args.run/'features',args.device)
        setup['features_sha256']=result['content_sha256']
    elif stage=='eval-prepare':
        if any(train.sha((args.run/'reference'/n/'recovery_receipt.json').read_bytes())!=p for n,p in setup['reference_receipts'].items()):raise ValueError('v15_reference_changed')
        refs=gauge.checked_references(args.run/'reference')
        previous=setup.get('eval_features_sha256')
        if previous and (args.run/'eval_data/eval_features.json').exists():
            if json.loads((args.run/'eval_data/eval_features.json').read_bytes())['content_sha256']!=previous:
                raise ValueError('v15_restored_eval_receipt_pin')
        result=evaluation.prepare_eval(args.inputs,refs,args.model_local,args.rgb_root,args.pose_root,args.run/'eval_data',args.device)
        if previous and previous!=result['content_sha256']:raise ValueError('v15_rebuilt_feature_receipt_changed')
        setup['eval_features_sha256']=result['content_sha256']
    elif stage=='pilot':
        runner=experiment(args,True);result=runner.run()
        if result['status']!='complete':raise RuntimeError('v15_pilot_not_complete')
        setup.update(pilot_initial=result['initial'],pilot_initial_rng=result['initial_rng'],pilot_complete=True)
        del runner;gc.collect()
        if torch.cuda.is_available():torch.cuda.empty_cache()
    elif stage=='train':
        runner=experiment(args)
        if args.resume_sha256:
            runner.resume(args.resume_sha256)
        else:
            if (args.run/'train/last.pt').exists():raise ValueError('v15_explicit_resume_pin_required')
            if not setup.get('pilot_complete') or runner.initial!=setup['pilot_initial'] or runner.initial_rng!=setup['pilot_initial_rng']:
                raise ValueError('v15_pilot_reset_initial_or_rng_mismatch')
        # Charge restore/identity work plus a conservative final checkpoint allowance.
        runner.elapsed=min(runner.cfg['limits']['train_cumulative_seconds'],runner.elapsed+time.monotonic()-invocation_start+60)
        result=runner.run()
    else:
        if not args.resume_sha256:raise ValueError('v15_final_checkpoint_pin_required')
        runner=experiment(args);runner.resume(args.resume_sha256)
        result=evaluation.evaluate(runner,args.run/'eval_data',args.run/'evaluation',expected_pin=setup['eval_features_sha256'])
    cross.write_report(setup_path,setup)
    return {k:v for k,v in result.items() if k in ('status','content_sha256','last','position','copied_bytes','raw_member_bytes_read')}


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='command',required=True)
    sub.add_parser('source')
    run_parser=sub.add_parser('run')
    for n in ('inputs','evidence','cache','work','run','model-source','model-local','reference','control-reference','rgb-root','pose-root'):
        run_parser.add_argument('--'+n,type=Path,required=True)
    run_parser.add_argument('--stage',choices=('model','references','raw','features','eval-prepare','pilot','train','evaluate'),required=True)
    run_parser.add_argument('--source-sha256',required=True);run_parser.add_argument('--device',choices=('cpu','cuda'),default='cuda')
    run_parser.add_argument('--resume-sha256')
    pack=sub.add_parser('package');pack.add_argument('--run',type=Path,required=True);pack.add_argument('--output',type=Path,required=True)
    backup=sub.add_parser('backup-prepare');backup.add_argument('--drive-root',type=Path,required=True);backup.add_argument('--run-id',required=True)
    publish=sub.add_parser('publish')
    for n in ('archive','attempt'):publish.add_argument('--'+n,type=Path,required=True)
    publish.add_argument('--sha256',required=True);publish.add_argument('--bytes',type=int,required=True)
    restore=sub.add_parser('restore')
    restore.add_argument('--archive',type=Path,required=True);restore.add_argument('--sha256',required=True);restore.add_argument('--output',type=Path,required=True)
    args=parser.parse_args(argv);torch.set_num_threads(1)
    if args.command=='source':result={'source_sha256':digest(train.source_identity())}
    elif args.command=='run':
        if args.source_sha256!=digest(train.source_identity()):raise ValueError('v15_source_pin')
        result=run(args)
    elif args.command=='package':result=recovery.package(args.run,args.output)
    elif args.command=='restore':result=recovery.restore(args.archive,args.sha256,args.output)
    elif args.command=='backup-prepare':result={'attempt':str(cross_recovery.prepare_attempt(args.drive_root,args.run_id,'inference',namespace=train.RUN_ID))}
    else:
        if not 0<args.bytes<=train.POLICY['limits']['output_bytes']:raise ValueError('v15_archive_size')
        result=cross_recovery.publish({'archive':str(args.archive),'sha256':args.sha256,'bytes':args.bytes},args.attempt,run_id=train.RUN_ID)
    print(canonical(result).decode())


if __name__=='__main__':main()
