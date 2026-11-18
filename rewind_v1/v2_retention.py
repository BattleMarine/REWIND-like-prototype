"""Bounded explicit shard copy and immutable model recovery; no automatic cache upload."""
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time
import zipfile
import shutil

from rewind_archive.contracts import canonical,digest
from rewind_m1.local_rgb import _regular_read
from rewind_prepare.temporal_cache import _atomic,_hash_file,_locked
from rewind_prepare.temporal_selection import seal,check_pin
from scripts.archvis_pilot_bootstrap import extract_pinned_zip
from .v2_cache import receipt,catalog,CHUNK


def copy_verified(source,target,pin,ledger,deadline,*,pre_reserved=False):
    if target.exists() or target.is_symlink():
        if target.is_symlink() or target.stat().st_size!=pin['bytes']:raise ValueError('v2_copy_existing_size')
        ledger.reserve('read',pin['bytes'])
        if _hash_file(target,check_time=lambda:check_deadline(ledger,deadline))!=pin['sha256']:raise ValueError('v2_copy_existing_sha')
        ledger.observed('read',pin['bytes']);return False
    if shutil.disk_usage(target.parent).free<pin['bytes']:raise ValueError('v2_copy_target_free_space')
    if not pre_reserved:ledger.reserve('read',pin['bytes']*2);ledger.reserve('write',pin['bytes'])
    tmp=target.with_name(target.name+'.partial')
    if tmp.exists():raise ValueError('v2_partial_copy_requires_explicit_recovery')
    try:
        h=hashlib.sha256();size=0
        with Path(source).open('rb') as src,tmp.open('xb') as dst:
            while block:=src.read(1024**2):
                check_deadline(ledger,deadline);size+=len(block)
                if size>pin['bytes']:raise ValueError('v2_copy_growth')
                h.update(block);dst.write(block)
            dst.flush();os.fsync(dst.fileno())
        if size!=pin['bytes'] or h.hexdigest()!=pin['sha256']:raise ValueError('v2_copy_source_pin')
        if _hash_file(tmp,check_time=lambda:check_deadline(ledger,deadline))!=pin['sha256']:raise ValueError('v2_copy_readback_pin')
        ledger.observed('read',2*size);ledger.observed('write',size);tmp.rename(target);return True
    finally:tmp.unlink(missing_ok=True)


def check_deadline(ledger,deadline):
    ledger.remaining(terminal=True)
    if ledger.clock()>deadline:raise TimeoutError('v2_retention_deadline')


def user_copy_cache(cache,retained,doc,spec,cfg,ledger):
    """Only invoked by the user's explicit copy switch; never a preparation callback."""
    root=Path(cache);retained=Path(retained);retained.mkdir(parents=True,exist_ok=True)
    local=json.loads(_regular_read(root/'catalog.json',32*1024**2));check_pin(local,'rewind_v2_local_catalog_v1',local.get('content_sha256'))
    from .v2_cache import binding
    if local['binding']!=binding(doc,spec,cfg):raise ValueError('v2_copy_catalog_binding')
    packages=[];copied=0
    with _locked(retained):
        remote_path=retained/'catalog.json';remote=None
        if remote_path.exists():
            remote=json.loads(_regular_read(remote_path,32*1024**2));check_pin(remote,'rewind_v2_retained_catalog_v1',remote.get('content_sha256'))
            if remote['binding']!=local['binding'] or len(remote['packages'])>len(local['shards']):raise ValueError('v2_retained_binding_or_downgrade')
            packages=list(remote['packages'])
            for i,p in enumerate(packages):
                if p['number']!=i or p['receipt_sha256']!=local['shards'][i]['content_sha256'] or p['name']!=f'shard-{i:05d}-{p["sha256"]}.zip':raise ValueError('v2_retained_prefix_pin')
                blob=retained/p['name']
                if blob.is_symlink() or not blob.is_file() or blob.stat().st_size!=p['bytes']:raise ValueError('v2_retained_missing_blob')
            # Reuse the previous completed readback receipt, not a claim of current full rehash.
            ledger.import_usage(remote['usage'])
        for i in range(len(packages),len(local['shards'])):
            r=local['shards'][i]
            ledger.check()
            if receipt(root,i,local['binding'],doc['records'][i*CHUNK:(i+1)*CHUNK],ledger)!=r:raise ValueError('v2_copy_local_receipt')
            with tempfile.TemporaryDirectory(dir=root,prefix='.pack-') as tmp:
                path=Path(tmp)/'shard.zip'
                with zipfile.ZipFile(path,'x') as z:
                    for n in ('images.npy','targets.pt','receipt.json'):
                        info=zipfile.ZipInfo(n,date_time=(1980,1,1,0,0,0));info.external_attr=0o100600<<16
                        with (root/f'shard-{i:05d}'/n).open('rb') as src,z.open(info,'w',force_zip64=True) as out:
                            while block:=src.read(1024**2):ledger.check();out.write(block)
                size=path.stat().st_size
                ledger.local_io('read',size);ledger.local_io('write',size)
                if size>64*1024**2:raise ValueError('v2_shard_zip_cap')
                pin={'bytes':size,'sha256':_hash_file(path,check_time=ledger.check)}
                name=f'shard-{i:05d}-{pin["sha256"]}.zip'
                made=copy_verified(path,retained/name,pin,ledger,ledger.clock()+min(cfg['copy_seconds'],ledger.remaining()))
                copied+=int(made);packages.append({'number':i,'name':name,**pin,'receipt_sha256':r['content_sha256']})
            # Every completed shard leaves a durable verified remote prefix.
            ledger.persist()
            remote=seal({'schema':'rewind_v2_retained_catalog_v1','binding':local['binding'],'packages':list(packages),
                'local_catalog_sha256':local['content_sha256'],'usage':ledger.snapshot(),
                'completed_frames':sum(x['frames'] for x in local['shards'][:len(packages)]),'status':'remote_verified'})
            _atomic(retained/'catalog.json',canonical(remote))
        if remote is None:
            ledger.persist();remote=seal({'schema':'rewind_v2_retained_catalog_v1','binding':local['binding'],'packages':[],
                'local_catalog_sha256':local['content_sha256'],'usage':ledger.snapshot(),'completed_frames':0,'status':'remote_verified'})
            _atomic(remote_path,canonical(remote))
        return {'copied_shards':copied,'verified_shards':len(packages),'catalog_sha256':remote['content_sha256'],'catalog_file_sha256':_hash_file(retained/'catalog.json')}


def restore_cache(retained,expected_catalog_sha,output,doc,spec,cfg,ledger):
    from .v2_cache import binding
    root=Path(output);root.mkdir(parents=True,exist_ok=True);retained=Path(retained)
    raw=_regular_read(retained/'catalog.json',32*1024**2)
    if contract_sha(raw)!=expected_catalog_sha:raise ValueError('v2_retained_catalog_file_pin')
    remote=json.loads(raw);check_pin(remote,'rewind_v2_retained_catalog_v1',remote.get('content_sha256'))
    bind=binding(doc,spec,cfg)
    if remote['binding']!=bind or [p['number'] for p in remote['packages']]!=list(range(len(remote['packages']))):raise ValueError('v2_restore_binding_prefix')
    ledger.import_usage(remote['usage'])
    pins=[];prior=None
    with _locked(root):
        old=root/'catalog.json'
        if old.exists():
            prior=json.loads(old.read_bytes());check_pin(prior,'rewind_v2_local_catalog_v1',prior.get('content_sha256'))
            if prior['binding']!=bind or len(prior['shards'])>len(remote['packages']):raise ValueError('v2_restore_catalog_downgrade')
        for p in remote['packages']:
            ledger.check();number=p['number'];folder=root/f'shard-{number:05d}'
            if p['name']!=f'shard-{number:05d}-{p["sha256"]}.zip' or not 0<p['bytes']<=64*1024**2:raise ValueError('v2_restore_package')
            if not folder.exists():
                with tempfile.TemporaryDirectory(dir=root,prefix='.restore-') as tmp:
                    local=Path(tmp)/'shard.zip';copy_verified(retained/p['name'],local,p,ledger,ledger.clock()+min(cfg['copy_seconds'],ledger.remaining()))
                    staged=Path(tmp)/f'shard-{number:05d}'
                    extract_pinned_zip(local,p['sha256'],staged,max_zip_bytes=64*1024**2,max_output_bytes=64*1024**2,max_members=3)
                    r=receipt(Path(tmp),number,bind,doc['records'][number*CHUNK:(number+1)*CHUNK],ledger)
                    if r['content_sha256']!=p['receipt_sha256']:raise ValueError('v2_restore_receipt_pin')
                    staged.rename(folder)
            r=receipt(root,number,bind,doc['records'][number*CHUNK:(number+1)*CHUNK],ledger)
            if r['content_sha256']!=p['receipt_sha256']:raise ValueError('v2_restore_existing_corruption')
            pins.append(r)
            if prior is not None and number<len(prior['shards']) and prior['shards'][number]!=r:raise ValueError('v2_restore_existing_pin_mismatch')
            if prior is None or len(pins)>len(prior['shards']):prior=catalog(root,bind,pins,len(doc['records']))
        if not remote['packages'] and prior is None:catalog(root,bind,[],len(doc['records']))
    return {'restored_frames':sum(r['frames'] for r in pins),'status':'local_complete' if sum(r['frames'] for r in pins)==len(doc['records']) else 'local_partial'}


def contract_sha(raw):return hashlib.sha256(raw).hexdigest()


def package_run(run,output,cfg,ledger,*,reserve_copy=False):
    from .runner import _read_checkpoint
    root=Path(run);output=Path(output)
    if output.exists() or output.is_relative_to(root):raise ValueError('v2_result_new_output')
    selected={'run.json','launch_error.json'};last=root/'last.pt'
    if last.is_file():
        state=_read_checkpoint(last,_hash_file(last),cfg['resources']['checkpoint_bytes']);selected.add('last.pt')
        pins=list(state['progress']['best'].values())+[p for group in state['progress'].get('milestones',{}).values() for p in group.values()]
        for p in pins:
            if Path(p['name']).name!=p['name']:raise ValueError('v2_result_model_name')
            from rewind_prepare.temporal_cache import _check_part
            _check_part(root/p['name'],p,lambda:ledger.remaining(terminal=True));selected.add(p['name'])
    files={n:{'bytes':(root/n).stat().st_size,'sha256':_hash_file(root/n)} for n in selected if (root/n).is_file()}
    upper=sum(p['bytes'] for p in files.values())+16*1024**2
    if upper>cfg['resources']['archive_bytes']:raise ValueError('v2_result_zip_cap')
    if shutil.disk_usage(output.parent).free<upper:raise ValueError('v2_package_target_free_space')
    ledger.reserve('write',upper)
    if reserve_copy:ledger.reserve('read',2*upper);ledger.reserve('write',upper)
    ledger.persist();report=seal({'schema':'rewind_v2_result_recovery_v1','files':files,'usage':ledger.snapshot(),
        'publication_accounting':'package and optional copy reservations captured before archive; final observed copy wall/bytes remain in local ledger'})
    deadline=ledger.clock()+min(cfg['package_seconds'],ledger.remaining(terminal=True))
    with tempfile.NamedTemporaryFile(dir=output.parent,delete=False) as f:tmp=Path(f.name)
    try:
        with zipfile.ZipFile(tmp,'w') as z:
            for name in files:check_deadline(ledger,deadline);z.write(root/name,name)
            z.writestr('recovery_receipt.json',canonical(report))
        if tmp.stat().st_size>cfg['resources']['archive_bytes']:raise ValueError('v2_result_actual_zip_cap')
        tmp.rename(output)
    finally:tmp.unlink(missing_ok=True)
    ledger.observed('write',output.stat().st_size)
    return {'bytes':output.stat().st_size,'sha256':_hash_file(output),'name':output.name}


def restore_run(archive,pin,output,cfg,ledger,expected_binding):
    from .runner import _read_checkpoint
    output=Path(output)
    if output.exists():raise ValueError('v2_restore_fresh_run')
    with tempfile.TemporaryDirectory(dir=output.parent) as tmp:
        local=Path(tmp)/'input.zip';copy_verified(archive,local,pin,ledger,ledger.clock()+min(cfg['copy_seconds'],ledger.remaining()))
        ledger.reserve('write',cfg['resources']['archive_bytes'])
        staged=Path(tmp)/'run';extract_pinned_zip(local,pin['sha256'],staged,max_zip_bytes=cfg['resources']['archive_bytes'],max_output_bytes=cfg['resources']['archive_bytes'],max_members=256)
        ledger.local_io('read',pin['bytes']*2)
        doc=json.loads(_regular_read(staged/'recovery_receipt.json',32*1024**2));check_pin(doc,'rewind_v2_result_recovery_v1',doc.get('content_sha256'))
        if {p.name for p in staged.iterdir()}!=set(doc['files'])|{'recovery_receipt.json'}:raise ValueError('v2_restore_files')
        for n,p in doc['files'].items():
            if Path(n).name!=n:raise ValueError('v2_restore_name')
            from rewind_prepare.temporal_cache import _check_part
            _check_part(staged/n,{'name':n,**p},ledger.check)
        state=_read_checkpoint(staged/'last.pt',doc['files']['last.pt']['sha256'],cfg['resources']['checkpoint_bytes'])
        if state['binding']!=expected_binding:raise ValueError('v2_restore_training_binding_environment')
        dependencies=list(state['progress']['best'].values())+[p for g in state['progress'].get('milestones',{}).values() for p in g.values()]
        if any(p['name'] not in doc['files'] or doc['files'][p['name']]!={k:p[k] for k in ('bytes','sha256')} for p in dependencies):raise ValueError('v2_restore_dependencies')
        ledger.import_usage(doc['usage']);(staged/'recovery_receipt.json').unlink();staged.rename(output)
    ledger.observed('write',sum(p['bytes'] for p in doc['files'].values()))
    return doc['files']['last.pt']['sha256']
