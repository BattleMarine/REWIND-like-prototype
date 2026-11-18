"""Single bounded v2 operation. No retry loop; cache copies require explicit user invocation."""
import argparse
from contextlib import suppress
import json
from pathlib import Path
import shutil
import tempfile
import uuid

import torch

from rewind_archive.contracts import canonical,digest
from rewind_m1.local_rgb import _regular_read
from rewind_prepare.temporal_cache import _atomic,_hash_file
from . import v2_contract as contract,v2_cache as cache,v2_retention as retention
from .v2_budget import Ledger
from .v2_training import V2Pipeline
from .v2_space import requirements,available_ram


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=('source','plan','prepare','copy-cache','restore-cache','train','restore-run','quarantine'))
    p.add_argument('--config',type=Path,default=contract.ROOT/'configs/rewind_v2.json');p.add_argument('--config-sha256')
    for name in ('manifest','spec','splits','work','rgb-root','pose-root','cache','retained','run','features','result-destination','archive'):
        p.add_argument('--'+name,type=Path)
    for name in ('manifest-sha256','spec-sha256','source-sha256','catalog-sha256','resume-sha256','archive-sha256'):p.add_argument('--'+name)
    p.add_argument('--archive-bytes',type=int);p.add_argument('--device',default='cuda');p.add_argument('--max-shards',type=int)
    p.add_argument('--shard-name');p.add_argument('--receipt-file-sha256')
    args=p.parse_args(argv);torch.set_num_threads(1)
    if args.action=='source':print(canonical({'source_sha256':digest(contract.source())}).decode());return
    if any(getattr(args,n) is None for n in ('manifest','spec','splits','work','manifest_sha256','spec_sha256','source_sha256','config_sha256')):p.error('Pinned manifest/spec/splits/config/source and work are required')
    raw=_regular_read(args.config,1024**2);cfg=json.loads(raw)
    if contract.sha(raw)!=args.config_sha256 or cfg!=contract.config() or args.source_sha256!=digest(contract.source()):raise ValueError('v2_config_source_pin')
    doc,spec,_=contract.load(args.manifest,args.manifest_sha256,args.spec,args.spec_sha256,args.splits)
    args.work.mkdir(parents=True,exist_ok=True);plan=contract.plan(doc,spec,cfg,args.work)
    ledger=Ledger(args.work/'usage',{'manifest':doc['content_sha256'],'config':digest(cfg),'source':digest(contract.source())},cfg)
    def required(*names):
        if any(getattr(args,n) is None for n in names):raise ValueError('v2_required_paths:'+','.join(names))
    with ledger.invocation(args.action):
        if args.action=='plan':result=plan
        elif args.action=='quarantine':
            required('cache','shard_name')
            result={'status':'quarantined','path':cache.quarantine(args.cache,args.shard_name,args.receipt_file_sha256)}
        else:
            paths_needed={'prepare':('cache','rgb_root','pose_root'),'copy-cache':('cache','retained'),
                'restore-cache':('cache','retained','catalog_sha256'),'train':('cache','run','features','result_destination'),
                'restore-run':('cache','run','features','archive','archive_sha256','archive_bytes')}
            required(*paths_needed[args.action])
            if args.action in ('train','restore-run'):contract.training_ready(cfg)
            space=requirements(args.action,plan,cfg,vars(args),ledger.check)
            _atomic(args.work/('space-'+args.action+'.json'),canonical(space))
            if not space['fit']:raise ValueError('v2_action_additional_disk_space')
            if available_ram()<cfg['resources']['rss_bytes']:raise MemoryError('v2_available_ram_preflight')
            if args.action=='prepare':
                required('cache','rgb_root','pose_root');result=cache.prepare(doc,spec,cfg,args.rgb_root,args.pose_root,args.cache,ledger,max_shards=args.max_shards)
            elif args.action=='copy-cache':
                required('cache','retained');result=retention.user_copy_cache(args.cache,args.retained,doc,spec,cfg,ledger)
            elif args.action=='restore-cache':
                required('cache','retained','catalog_sha256');result=retention.restore_cache(args.retained,args.catalog_sha256,args.cache,doc,spec,cfg,ledger)
            else:
                contract.training_ready(cfg);required('run','cache','features')
                if args.device!='cuda' or not torch.cuda.is_available():raise ValueError('v2_supported_cuda_required')
                if args.action=='train':
                    required('result_destination');args.result_destination.mkdir(parents=True,exist_ok=True)
                    probe=args.result_destination/('preflight-'+uuid.uuid4().hex+'.json');raw=canonical({'source':args.source_sha256,'run':str(args.run)})
                    with probe.open('xb') as f:f.write(raw);f.flush();__import__('os').fsync(f.fileno())
                    if _regular_read(probe,1024**2)!=raw:raise ValueError('v2_result_destination_preflight')
                    args.run.mkdir(exist_ok=True)
                package_folder=args.work/'results';package_folder.mkdir(exist_ok=True)
                def backup():
                    local=package_folder/('result-'+uuid.uuid4().hex+'.zip')
                    pin=retention.package_run(args.run,local,cfg,ledger,reserve_copy=args.result_destination is not None)
                    if args.result_destination is None:return {'local':str(local),'status':'local_only'}
                    args.result_destination.mkdir(parents=True,exist_ok=True);target=args.result_destination/local.name
                    retention.copy_verified(local,target,pin,ledger,ledger.clock()+min(cfg['copy_seconds'],ledger.remaining(terminal=True)),pre_reserved=True)
                    return {'local':str(local),'remote':str(target),'status':'remote_verified','sha256':pin['sha256']}
                pipeline=V2Pipeline(cfg,doc,spec,plan,args.cache,args.features,args.device,args.run,ledger,backup)
                try:
                    if args.action=='restore-run':
                        required('archive','archive_sha256','archive_bytes')
                        result={'last_sha256':retention.restore_run(args.archive,{'sha256':args.archive_sha256,'bytes':args.archive_bytes},args.run,cfg,ledger,pipeline.binding)}
                    else:
                        try:result=pipeline.run(resume_sha256=args.resume_sha256)
                        except BaseException as e:
                            _atomic(args.run/'launch_error.json',canonical({'status':'stopped' if isinstance(e,KeyboardInterrupt) else 'failed','type':type(e).__name__,'reason':str(e)[:512]}));raise
                        finally:
                            saved=backup();_atomic(args.work/'retention.json',canonical(saved))
                finally:pipeline.data.cache.close()
        _atomic(args.work/('status-'+args.action+'.json'),canonical(result))
    print(canonical(result).decode())


if __name__=='__main__':main()
