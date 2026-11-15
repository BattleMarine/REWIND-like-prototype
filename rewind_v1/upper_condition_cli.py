"""Bounded bundle verification, frozen inference and report-only retention."""

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
from rewind_m1.training import seed_run
from . import upper_condition_bundle as bundle,upper_condition as upper,cross_data as cross,cross_recovery as recovery
from .cross_cli import _linux_available_ram


def preflight(root):
    free=shutil.disk_usage(root).free;available=_linux_available_ram()
    if free<bundle.LIMITS['initial_free_disk_bytes']:raise ValueError('v151_disk_2GiB')
    if available<bundle.LIMITS['initial_available_ram_bytes']:raise MemoryError('v151_MemAvailable_4GiB')
    return {'free_disk_bytes':free,'available_ram_bytes':available,'ram_scope':'Linux MemAvailable, not cgroup headroom'}


def package(root,output):
    started=time.monotonic();deadline=started+bundle.LIMITS['package_seconds'];root=cross._local(root,directory=True);output=Path(output).absolute()
    if output.exists() or output.is_relative_to(root):raise ValueError('v151_new_package_output')
    files=[p for p in root.iterdir() if p.name in ('input_status.json','operation.json') and p.is_file()]
    inference=root/'inference'
    if inference.is_dir():
        files += [p for p in inference.iterdir() if p.name in ('report.json','failure.json','comparison.md') and p.is_file()]
        for i in range(6):
            clip=inference/f'clip-{i:02d}'
            if clip.is_dir():files += [p for p in clip.iterdir() if p.is_file() and (p.name in ('predictions.json','comparison.gif') or p.name in ('frame-000000000.png','frame-000000025.png','frame-000000049.png'))]
    pins={}
    for p in files:
        cross._local(p)
        if time.monotonic()>deadline:raise TimeoutError('v151_package_timeout')
        with p.open('rb') as stream:h=hashlib.file_digest(stream,'sha256').hexdigest()
        pins[p.relative_to(root).as_posix()]={'bytes':p.stat().st_size,'sha256':h}
    receipt={'schema':'v151_result_recovery_v1','run_id':bundle.RUN_ID,'source_sha256':digest(bundle.source_identity()),'files':pins}
    raw=canonical(receipt)
    if sum(p['bytes'] for p in pins.values())+len(raw)+16384>bundle.LIMITS['result_bytes']:raise ValueError('v151_package_sum_limit')
    with tempfile.NamedTemporaryFile(dir=output.parent,delete=False) as stream:temporary=Path(stream.name)
    try:
        with zipfile.ZipFile(temporary,'w') as z:
            for p in files:
                if time.monotonic()>deadline:raise TimeoutError('v151_package_timeout')
                z.write(p,p.relative_to(root).as_posix())
            z.writestr('recovery_receipt.json',raw)
        if temporary.stat().st_size>bundle.LIMITS['result_bytes']:raise ValueError('v151_package_zip_limit')
        temporary.rename(output)
    finally:temporary.unlink(missing_ok=True)
    with output.open('rb') as stream:pin=hashlib.file_digest(stream,'sha256').hexdigest()
    return {'archive':str(output),'bytes':output.stat().st_size,'sha256':pin,'package_wall_seconds':time.monotonic()-started}


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='command',required=True)
    sub.add_parser('source')
    make=sub.add_parser('bundle');make.add_argument('--parent',type=Path,required=True);make.add_argument('--output',type=Path,required=True)
    for n in ('prepare','infer'):
        p=sub.add_parser(n);p.add_argument('--run',type=Path,required=True);p.add_argument('--source-sha256',required=True)
        p.add_argument('--manifest-sha256',required=True)
        if n=='prepare':p.add_argument('--bundle',type=Path,required=True);p.add_argument('--bundle-sha256',required=True)
        else:p.add_argument('--device',choices=('cpu','cuda'),default='cuda')
    pack=sub.add_parser('package');pack.add_argument('--run',type=Path,required=True);pack.add_argument('--output',type=Path,required=True)
    backup=sub.add_parser('backup-prepare');backup.add_argument('--drive-root',type=Path,required=True);backup.add_argument('--run-id',required=True)
    publish=sub.add_parser('publish')
    for n in ('archive','attempt'):publish.add_argument('--'+n,type=Path,required=True)
    publish.add_argument('--bytes',type=int,required=True);publish.add_argument('--sha256',required=True)
    publish.add_argument('--package-wall-seconds',type=float,required=True)
    args=parser.parse_args(argv);torch.set_num_threads(1);start=time.monotonic()
    if args.command=='source':result={'source_sha256':digest(bundle.source_identity())}
    elif args.command=='bundle':result=bundle.build_bundle(args.parent,args.output)
    elif args.command in ('prepare','infer'):
        if args.source_sha256!=digest(bundle.source_identity()):raise ValueError('v151_source_pin')
        preflight(args.run)
        if args.command=='prepare':
            inputs=bundle.extract_bundle(args.bundle,args.bundle_sha256,args.manifest_sha256,args.run/'inputs')
            result={'status':'verified','manifest_sha256':inputs['bundle']['content_sha256'],'bundle_sha256':args.bundle_sha256,
                'bundle_bytes':args.bundle.stat().st_size,'input_wall_seconds':time.monotonic()-start}
            cross.write_report(args.run/'input_status.json',result)
        else:
            old=json.loads((args.run/'input_status.json').read_bytes())
            if old['manifest_sha256']!=args.manifest_sha256:raise ValueError('v151_input_stage_pin')
            inputs=bundle.checked_bundle(args.run/'inputs',args.manifest_sha256);device=torch.device(args.device)
            if device.type=='cuda':
                if not torch.cuda.is_available():raise RuntimeError('v151_requested_cuda_unavailable')
                torch.cuda.reset_peak_memory_stats(device)
            seed_run(bundle.POLICY['seed'])
            report=upper.run(args.run/'inputs',inputs,args.run/'inference',device)
            result={'status':report['status'],'content_sha256':report['content_sha256'],'inference_phase_wall_seconds':time.monotonic()-start}
    elif args.command=='package':result=package(args.run,args.output)
    elif args.command=='backup-prepare':result={'attempt':str(recovery.prepare_attempt(args.drive_root,args.run_id,'inference',namespace=bundle.RUN_ID))}
    else:
        if not 0<args.bytes<=bundle.LIMITS['result_bytes']:raise ValueError('v151_publish_size')
        if not 0<=args.package_wall_seconds<=bundle.LIMITS['package_seconds']+30:raise ValueError('v151_package_time_metadata')
        result=recovery.publish({'archive':str(args.archive),'bytes':args.bytes,'sha256':args.sha256,
            'package_wall_seconds':args.package_wall_seconds},args.attempt,run_id=bundle.RUN_ID)
    print(canonical(result).decode())


if __name__=='__main__':main()
