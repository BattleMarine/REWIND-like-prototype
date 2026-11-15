"""Independently bounded stages and exact whitelist for the six-clip gauge diagnostic."""

import argparse
import json
from pathlib import Path
import time

import torch

from rewind_archive.contracts import canonical,digest
from . import camera_gauge as gauge,cross_data as cross,cross_recovery as recovery
from .cross_cli import preflight,model_copy


def extra_reports():
    files=['gauge_status.json','inference/comparison.md']
    for number in range(6):
        files += [f'prepared/clip-{number:02d}/manifest.json'] if number>=4 else []
        files += [f'inference/clip-{number:02d}/{name}.json' for name in gauge.VARIANTS]
        files += [f'inference/clip-{number:02d}/'+name for name in ('report.json','comparison.gif')]
    return tuple(files)


def package(run,output):
    # Frame ids need not start at0. Add only names verified against the sealed clip artifact list.
    extra=list(extra_reports());root=Path(run)
    for number in range(6):
        p=root/f'inference/clip-{number:02d}/report.json'
        if not p.is_file():continue
        doc=cross.read_sealed(p,'v12_camera_gauge_clip_v1',json.loads(p.read_bytes())['content_sha256'],context=gauge.context())
        keys=doc['keys'];allowed={'comparison.gif'}|{f'frame-{keys[t][1]:09d}.png' for t in (0,25,49)}
        for artifact in doc['artifacts']:
            if artifact['name'] not in allowed:raise ValueError('gauge_package_artifact_name')
            path=p.parent/artifact['name'];raw=cross._regular_read(path,gauge.LIMITS['result_bytes'],artifact['bytes'])
            if gauge.sha(raw)!=artifact['sha256']:raise ValueError('gauge_package_artifact_hash')
            name=path.relative_to(root).as_posix()
            # first4 PNGs are already included by legacy recovery; remaining2 are explicit.
            if name not in extra and not(number<4 and artifact['name'].startswith('frame-')):extra.append(name)
    return recovery.package(run,output,run_id=gauge.RUN_ID,extra_reports=tuple(extra))


def reference_receipt_pins(run):
    return {name:gauge.sha((Path(run)/'reference'/name/'recovery_receipt.json').read_bytes()) for name in ('city','control')}


def checked_stage(run,status):
    if reference_receipt_pins(run)!=status['reference_receipt_sha256']:raise ValueError('gauge_reference_changed_between_stages')
    references=gauge.checked_references(Path(run)/'reference')
    plan=gauge.checked_plan(Path(run)/'selection/selection.json',status['selection_sha256'],references)
    return references,plan


def run_stage(args):
    root=Path(args.run);stage=args.stage;path=root/'gauge_status.json'
    order=('reference','select','model','prepare','infer')
    if stage=='reference':
        if path.exists() or any((root/n).exists() for n in ('reference','selection','prepared','inference')):
            raise FileExistsError('gauge_fresh_run_required')
        status={'run_id':gauge.RUN_ID,'source_sha256':digest(gauge.source_identity()),'completed':[],'state':'started','costs':{}}
    else:
        status=json.loads(cross._regular_read(path,cross.MIB))
        if (status['run_id']!=gauge.RUN_ID or status['source_sha256']!=digest(gauge.source_identity())
                or status['completed']!=list(order[:order.index(stage)]) or status['state']=='failed'):
            raise ValueError('gauge_stage_order_or_source')
    start=time.monotonic();status['phase']=stage;cross.write_report(path,status)
    try:
        preflight(root)
        if stage=='reference':
            gauge.load_references(args.reference,args.control_reference,root/'reference')
            status['reference_receipt_sha256']=reference_receipt_pins(root)
            result={'status':'verified','reference_zip_bytes':sum(p['bytes'] for p in gauge.POLICY['references'].values())}
        elif stage=='select':
            if reference_receipt_pins(root)!=status['reference_receipt_sha256']:raise ValueError('gauge_reference_changed_between_stages')
            refs=gauge.checked_references(root/'reference')
            plan=gauge.fixed_plan(args.inputs,refs,args.rgb_root,args.pose_root,root/'selection')
            status['selection_sha256']=plan['content_sha256'];result={'status':'complete','selection_sha256':plan['content_sha256']}
        elif stage=='model':result=model_copy(args.model_source,args.model_local,root/'model_status.json')
        else:
            refs,plan=checked_stage(root,status)
            if stage=='prepare':
                budget=cross.Budget(gauge.LIMITS['prepare_seconds'])
                prepared=cross.prepare(args.inputs,root/'selection/selection.json',plan['content_sha256'],root/'prepared',context=gauge.context(),budget=budget)
                gauge.verify_prepared(root/'prepared',prepared['content_sha256'],plan,refs);budget.check()
                status['prepared_sha256']=prepared['content_sha256'];status['costs']['source_bytes_read']=budget.read_bytes
                result={'status':'complete','prepared_sha256':prepared['content_sha256'],'source_bytes_read':budget.read_bytes}
            else:
                device=torch.device(args.device)
                if device.type=='cuda':
                    if not torch.cuda.is_available():raise RuntimeError('gauge_requested_cuda_unavailable')
                    torch.cuda.reset_peak_memory_stats(device)
                result=gauge.evaluate(args.inputs,plan,root/'prepared',status['prepared_sha256'],refs,args.model_local,root/'inference',device)
                status.update(report_sha256=result['content_sha256'],reference_match_all=result['reference_match_all'],repeat_stable_all=result['repeat_stable_all'])
                result={k:result[k] for k in ('status','content_sha256','reference_match_all','repeat_stable_all')}
        status['completed'].append(stage);status['state']='complete' if stage=='infer' else 'stage_complete';return result
    except BaseException as error:
        status.update(state='failed',type=type(error).__name__,reason=str(error)[:2000]);raise
    finally:
        status['costs'][stage+'_seconds']=time.monotonic()-start;cross.write_report(path,status)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);commands=parser.add_subparsers(dest='command',required=True)
    commands.add_parser('source')
    check=commands.add_parser('preflight');check.add_argument('--work',type=Path,required=True)
    run=commands.add_parser('run')
    for n in ('inputs','run','rgb-root','pose-root','reference','control-reference','model-source','model-local'):
        run.add_argument('--'+n,type=Path,required=True)
    run.add_argument('--stage',choices=('reference','select','model','prepare','infer'),required=True)
    run.add_argument('--source-sha256',required=True);run.add_argument('--device',choices=('cpu','cuda'),default='cpu')
    pack=commands.add_parser('package');pack.add_argument('--run',type=Path,required=True);pack.add_argument('--output',type=Path,required=True)
    backup=commands.add_parser('backup-prepare');backup.add_argument('--drive-root',type=Path,required=True);backup.add_argument('--run-id',required=True)
    publish=commands.add_parser('publish')
    for n in ('archive','attempt'):publish.add_argument('--'+n,type=Path,required=True)
    publish.add_argument('--sha256',required=True);publish.add_argument('--bytes',type=int,required=True)
    args=parser.parse_args(argv);torch.set_num_threads(1)
    if args.command=='source':result={'source_sha256':digest(gauge.source_identity())}
    elif args.command=='preflight':result=preflight(args.work)
    elif args.command=='run':
        if args.source_sha256!=digest(gauge.source_identity()):raise ValueError('gauge_source_pin')
        result=run_stage(args)
    elif args.command=='package':result=package(args.run,args.output)
    elif args.command=='backup-prepare':result={'attempt':str(recovery.prepare_attempt(args.drive_root,args.run_id,'inference',namespace=gauge.RUN_ID))}
    else:
        if not 0<args.bytes<=gauge.LIMITS['result_bytes']:raise ValueError('gauge_archive_limit')
        result=recovery.publish({'archive':str(args.archive),'bytes':args.bytes,'sha256':args.sha256},args.attempt,run_id=gauge.RUN_ID)
    print(canonical(result).decode())


if __name__=='__main__':main()
