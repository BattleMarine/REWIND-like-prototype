"""Explicit default-off experiment entry points; no remote data fetching."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import time
import zipfile

import torch

from rewind_archive.contracts import canonical,digest
from . import train_clip_bundle as bundle,train_clip_runs as runs,cross_recovery as recovery,cross_data as cross
from .cross_cli import _linux_available_ram


def package(root,output,experiment):
    cfg=bundle.config(experiment);started=time.monotonic();deadline=started+cfg['limits']['package_seconds']
    root=cross._local(root,directory=True);output=Path(output).absolute()
    if output.exists() or output.is_symlink() or output.is_relative_to(root):raise ValueError('clip_package_new_path')
    files=[p for p in root.iterdir() if p.is_file() and p.name in ('operation.json','input_status.json')]
    folder=root/'experiment'
    if folder.is_dir():files += [p for p in folder.iterdir() if p.is_file() and (p.name in ('run.json','last.pt') or
        p.name.startswith(('models-','predictions-','comparison-')) and p.suffix in ('.pt','.png','.json'))]
    pins={}
    for p in files:
        cross._local(p)
        if time.monotonic()>deadline:raise TimeoutError('clip_package_timeout')
        with p.open('rb') as f:h=hashlib.file_digest(f,'sha256').hexdigest()
        pins[p.relative_to(root).as_posix()]={'bytes':p.stat().st_size,'sha256':h}
    raw=canonical({'schema':'v15_clip_recovery_v1','run_id':cfg['run_id'],'source_sha256':digest(bundle.source_identity()),'files':pins})
    if sum(p['bytes'] for p in pins.values())+len(raw)+16384>cfg['limits']['output_bytes']:raise ValueError('clip_package_cap')
    with tempfile.NamedTemporaryFile(dir=output.parent,delete=False) as f:tmp=Path(f.name)
    try:
        with zipfile.ZipFile(tmp,'w') as z:
            for p in files:
                if time.monotonic()>deadline:raise TimeoutError('clip_package_timeout')
                z.write(p,p.relative_to(root).as_posix())
            z.writestr('recovery_receipt.json',raw)
        if tmp.stat().st_size>cfg['limits']['output_bytes']:raise ValueError('clip_package_zip_cap')
        tmp.rename(output)
    finally:tmp.unlink(missing_ok=True)
    with output.open('rb') as f:h=hashlib.file_digest(f,'sha256').hexdigest()
    return {'archive':str(output),'bytes':output.stat().st_size,'sha256':h,'package_wall_seconds':time.monotonic()-started}


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--experiment',choices=('v152','v153'),required=True)
    sub=parser.add_subparsers(dest='command',required=True);sub.add_parser('source')
    b=sub.add_parser('bundle');b.add_argument('--parent',type=Path,required=True);b.add_argument('--output',type=Path,required=True)
    for name in ('prepare','run'):
        p=sub.add_parser(name);p.add_argument('--run',type=Path,required=True);p.add_argument('--source-sha256',required=True);p.add_argument('--manifest-sha256',required=True)
        if name=='prepare':p.add_argument('--bundle',type=Path,required=True);p.add_argument('--bundle-sha256',required=True)
        else:p.add_argument('--device',choices=('cuda',),default='cuda');p.add_argument('--resume-sha256')
    p=sub.add_parser('package');p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p=sub.add_parser('backup-prepare');p.add_argument('--drive-root',type=Path,required=True);p.add_argument('--run-id',required=True)
    p=sub.add_parser('publish')
    for n in ('archive','attempt'):p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--bytes',type=int,required=True);p.add_argument('--sha256',required=True);p.add_argument('--package-wall-seconds',type=float,required=True)
    args=parser.parse_args(argv);cfg=bundle.config(args.experiment);started=time.monotonic();torch.set_num_threads(1)
    if args.command=='source':result={'source_sha256':digest(bundle.source_identity())}
    elif args.command=='bundle':result=bundle.build_bundle(args.parent,args.output)
    elif args.command in ('prepare','run'):
        if args.source_sha256!=digest(bundle.source_identity()):raise ValueError('clip_source_pin')
        free=shutil.disk_usage(args.run).free;ram=_linux_available_ram()
        if free<cfg['limits']['initial_free_disk_bytes'] or ram<cfg['limits']['initial_available_ram_bytes']:raise MemoryError('clip_disk8GiB_ram4GiB_preflight')
        if args.command=='prepare':
            inputs=bundle.extract(args.bundle,args.bundle_sha256,args.manifest_sha256,args.run/'inputs')
            result={'status':'verified','manifest_sha256':inputs['bundle']['content_sha256'],'bundle_sha256':args.bundle_sha256,
                'bundle_bytes':args.bundle.stat().st_size,'input_wall_seconds':time.monotonic()-started,'free_disk_bytes':free,'available_ram_bytes':ram}
            if result['input_wall_seconds']>cfg['limits']['input_seconds']:raise TimeoutError('clip_input_timeout')
            cross.write_report(args.run/'input_status.json',result)
        else:
            if json.loads((args.run/'input_status.json').read_bytes())['manifest_sha256']!=args.manifest_sha256:raise ValueError('clip_input_stage_pin')
            if not torch.cuda.is_available():raise RuntimeError('clip_requested_cuda_unavailable')
            if args.experiment=='v152' and args.resume_sha256:raise ValueError('v152_no_training_resume')
            inputs=bundle.checked(args.run/'inputs',args.manifest_sha256);torch.cuda.reset_peak_memory_stats()
            if args.experiment=='v152':report=runs.diagnostic(inputs,cfg,args.device,args.run/'experiment')
            else:report=runs.PredictedClipTraining(inputs,cfg,args.device,args.run/'experiment').run(resume=args.resume_sha256)
            result={'status':report['status'],'compute_phase_wall_seconds':time.monotonic()-started,'last_checkpoint':report.get('last_checkpoint')}
    elif args.command=='package':result=package(args.run,args.output,args.experiment)
    elif args.command=='backup-prepare':result={'attempt':str(recovery.prepare_attempt(args.drive_root,args.run_id,'inference',namespace=cfg['run_id']))}
    else:
        if not 0<args.bytes<=cfg['limits']['output_bytes'] or not 0<=args.package_wall_seconds<=cfg['limits']['package_seconds']+30:raise ValueError('clip_publish_cap')
        result=recovery.publish({'archive':str(args.archive),'bytes':args.bytes,'sha256':args.sha256,'package_wall_seconds':args.package_wall_seconds},args.attempt,run_id=cfg['run_id'])
    print(canonical(result).decode())


if __name__=='__main__':main()
