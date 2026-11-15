"""Bounded single-action ArchVis control with pinned city camera comparison, no training."""

import argparse
import json
from pathlib import Path
import time

import torch

from rewind_archive.contracts import canonical,digest
from . import camera_control as control,cross_data as cross,cross_inference as inference,cross_recovery as recovery
from .cross_cli import preflight,model_copy


EXTRA_REPORTS=('cameras/cameras.json','comparison/report.json','comparison/comparison.md',
               'comparison/camera_distributions.png','control_status.json')


def checked_stage_inputs(run,status,*,with_cameras=False):
    ctx=control.context();run=Path(run)
    plan=cross.checked_selection(run/'selection/selection.json',status['selection_sha256'],context=ctx)
    if not with_cameras:return plan,None,None
    prepared=cross.read_sealed(run/'prepared/prepared.json','v12_cross_prepared_v1',status['prepared_sha256'],context=ctx)
    cameras=cross.read_sealed(run/'cameras/cameras.json','v12_camera_records_v1',status['cameras_sha256'],context=ctx)
    if (prepared['selection_sha256']!=status['selection_sha256'] or cameras['prepared_sha256']!=status['prepared_sha256']
            or cameras['model_zip_sha256']!=cross.MODEL_SHA or cameras['binding']!=prepared['binding']):
        raise ValueError('control_stage_camera_prepared_binding')
    return plan,prepared,cameras


def run(args):
    """One independently timed child stage; notebook composes these into one action."""
    ctx=control.context();run=Path(args.run);stage=args.stage
    state=run/'control_status.json'
    if stage=='reference':
        if state.exists() or any((run/n).exists() for n in ('reference','selection','prepared','inference')):
            raise FileExistsError('control_fresh_attempt_required')
        status={'run_id':control.RUN_ID,'source_sha256':digest(control.source_identity()),'state':'started','costs':{},'completed':[]}
    else:
        status=json.loads(state.read_bytes())
        order=['reference','select','model','prepare','infer']
        if (status['run_id']!=control.RUN_ID or status['source_sha256']!=digest(control.source_identity())
                or status['completed']!=order[:order.index(stage)] or status['state']=='failed'):
            raise ValueError('control_stage_order_or_binding')
    status['phase']=stage;cross.write_report(state,status);start=time.monotonic()
    try:
        preflight(run)
        if stage=='reference':
            control.load_reference(args.reference,run/'reference')
            status['reference_receipt_sha256']=control.sha((run/'reference/recovery_receipt.json').read_bytes())
            result={'state':'verified','reference_zip_sha256':control.POLICY['reference_sha256']}
        elif stage=='select':
            result=cross.select(args.inputs,args.rgb_root,args.pose_root,run/'selection',context=ctx)
            status.update(selection_sha256=result['content_sha256'],selection_status=result['status'])
        elif stage=='model':result=model_copy(args.model_source,args.model_local,run/'model_status.json')
        else:
            if control.sha((run/'reference/recovery_receipt.json').read_bytes())!=status['reference_receipt_sha256']:
                raise ValueError('control_reference_receipt_changed_between_stages')
            reference=control.checked_reference(run/'reference')
            plan,prepared,cameras=checked_stage_inputs(run,status,with_cameras=stage=='infer')
            if stage=='prepare':
                budget=cross.Budget(control.LIMITS['prepare_seconds'])
                prepared=cross.prepare(args.inputs,run/'selection/selection.json',plan['content_sha256'],run/'prepared',context=ctx,budget=budget)
                cameras=control.recover_cameras(args.inputs,run/'prepared',reference,args.pose_root,run/'cameras',budget)
                budget.check()
                status['costs'].update(total_preparation_source_bytes=budget.read_bytes,
                    city_pose_source_bytes=cameras['city_pose_source_bytes'],archvis_pose_local_reread_bytes=cameras['archvis_pose_local_reread_bytes'])
                result={'prepared_sha256':prepared['content_sha256'],'cameras_sha256':cameras['content_sha256']}
                status.update(result)
            else:
                device=torch.device(args.device)
                if device.type=='cuda':
                    if not torch.cuda.is_available():raise RuntimeError('control_requested_cuda_unavailable_no_fallback')
                    torch.cuda.reset_peak_memory_stats(device)
                budget=cross.Budget(control.LIMITS['infer_seconds'])
                report=inference.infer(args.inputs,run/'selection/selection.json',plan['content_sha256'],run/'prepared',
                    prepared['content_sha256'],args.model_local,run/'inference',device,context=ctx,budget=budget)
                if (report['model_weights']!=reference['report']['model_weights']
                        or report['inference_implementation_sha256']!=reference['report']['inference_implementation_sha256']):
                    raise ValueError('control_model_or_implementation_reference_mismatch')
                result=control.compare(args.inputs,cameras,run/'inference',reference,run/'comparison',budget,inference_report=report)
                budget.check();status['comparison_sha256']=result['content_sha256']
        status['completed'].append(stage);status['state']='complete' if stage=='infer' else 'stage_complete'
        recovery.result_files(run,extra_reports=EXTRA_REPORTS)
        return {k:v for k,v in result.items() if k in ('state','status','content_sha256','prepared_sha256','cameras_sha256','reference_zip_sha256')}
    except BaseException as error:
        status.update(state='failed',type=type(error).__name__,reason=str(error)[:2000]);raise
    finally:
        status['costs'][stage+'_seconds']=time.monotonic()-start
        cross.write_report(state,status)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);commands=parser.add_subparsers(dest='command',required=True)
    commands.add_parser('source')
    check=commands.add_parser('preflight');check.add_argument('--work',type=Path,required=True)
    execute=commands.add_parser('run')
    for name in ('inputs','run','rgb-root','pose-root','reference','model-source','model-local'):
        execute.add_argument('--'+name,type=Path,required=True)
    execute.add_argument('--stage',choices=('reference','select','model','prepare','infer'),required=True)
    execute.add_argument('--source-sha256',required=True);execute.add_argument('--device',choices=('cpu','cuda'),default='cpu')
    pack=commands.add_parser('package')
    pack.add_argument('--run',type=Path,required=True);pack.add_argument('--output',type=Path,required=True)
    backup=commands.add_parser('backup-prepare')
    backup.add_argument('--drive-root',type=Path,required=True);backup.add_argument('--run-id',required=True)
    publish=commands.add_parser('publish')
    publish.add_argument('--archive',type=Path,required=True);publish.add_argument('--bytes',type=int,required=True)
    publish.add_argument('--sha256',required=True);publish.add_argument('--attempt',type=Path,required=True)
    args=parser.parse_args(argv);torch.set_num_threads(1)
    if args.command=='source':result={'source_sha256':digest(control.source_identity())}
    elif args.command=='preflight':result=preflight(args.work)
    elif args.command=='run':
        if args.source_sha256!=digest(control.source_identity()):raise ValueError('control_source_pin')
        result=run(args)
    elif args.command=='package':result=recovery.package(args.run,args.output,run_id=control.RUN_ID,extra_reports=EXTRA_REPORTS)
    elif args.command=='backup-prepare':
        result={'attempt':str(recovery.prepare_attempt(args.drive_root,args.run_id,'inference',namespace=control.RUN_ID))}
    else:
        if not 0<args.bytes<=control.LIMITS['result_bytes']:raise ValueError('control_archive_limit')
        result=recovery.publish({'archive':str(args.archive),'bytes':args.bytes,'sha256':args.sha256},args.attempt,run_id=control.RUN_ID)
    print(canonical(result).decode())


if __name__=='__main__':main()
