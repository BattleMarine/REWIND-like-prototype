"""V1.5 model/derived-feature retention; excludes original data and raw caches."""

import hashlib
import json
from pathlib import Path
import tempfile
import time
import zipfile

from rewind_archive.contracts import canonical,digest
from rewind_m1.local_rgb import _regular_read
from scripts.archvis_pilot_bootstrap import extract_pinned_zip
from . import camera_training as train,cross_recovery as base


def files(root):
    root=Path(root);result=[]
    for p in root.rglob('*'):
        if not p.is_file():continue
        name=p.relative_to(root).as_posix();parts=Path(name).parts
        allowed=(len(parts)==1 and p.name in ('setup.json','operation.json','model_status.json','raw_restore.json')
            or parts[0]=='features' and len(parts)==2 and (p.name=='features.json' or p.name.startswith('train-') and p.suffix=='.pt')
            or parts[0]=='train' and len(parts)==2 and (p.name in ('run.json','last.pt') or p.suffix=='.pt' and '-u' in p.name)
            or parts[0]=='pilot' and len(parts)==2 and p.name=='run.json'
            or parts[0]=='eval_data' and len(parts)==2 and (p.name=='eval_features.json' or p.name.startswith('eval-') and p.suffix=='.pt')
            or parts[0]=='evaluation' and (len(parts)==2 and p.name=='report.json' or len(parts)==3 and
                (p.name in ('report.json','absolute.json','first_midpoint.json','comparison.gif') or p.name.startswith('frame-') and p.suffix=='.png')))
        if allowed:
            base._local(p);result.append(p)
    if sum(p.stat().st_size for p in result)>train.POLICY['limits']['output_bytes']:raise ValueError('v15_recovery_file_sum')
    return sorted(result)


def package(root,output):
    root=Path(root);output=Path(output)
    if output.exists() or output.is_relative_to(root):raise ValueError('v15_new_recovery_output')
    pins={};deadline=time.monotonic()+600;items=files(root)
    for path in items:
        if time.monotonic()>deadline:raise TimeoutError('v15_package_timeout')
        with path.open('rb') as stream:pin=hashlib.file_digest(stream,'sha256').hexdigest()
        pins[path.relative_to(root).as_posix()]={'bytes':path.stat().st_size,'sha256':pin}
    receipt={'schema':'v15_training_recovery_v1','run_id':train.RUN_ID,'source_sha256':digest(train.source_identity()),'files':pins}
    with tempfile.NamedTemporaryFile(dir=output.parent,delete=False) as stream:temporary=Path(stream.name)
    try:
        with zipfile.ZipFile(temporary,'w') as z:
            for path in items:
                if time.monotonic()>deadline:raise TimeoutError('v15_package_timeout')
                z.write(path,path.relative_to(root).as_posix())
            z.writestr('recovery_receipt.json',canonical(receipt))
        if temporary.stat().st_size>train.POLICY['limits']['output_bytes']:raise ValueError('v15_recovery_zip_limit')
        temporary.rename(output)
    finally:temporary.unlink(missing_ok=True)
    with output.open('rb') as stream:pin=hashlib.file_digest(stream,'sha256').hexdigest()
    return {'archive':str(output),'bytes':output.stat().st_size,'sha256':pin}


def restore(archive,pin,output):
    extract_pinned_zip(archive,pin,output,max_zip_bytes=train.POLICY['limits']['output_bytes'],
        max_output_bytes=train.POLICY['limits']['output_bytes'],max_members=1000)
    root=Path(output);receipt=json.loads((root/'recovery_receipt.json').read_bytes())
    if receipt['run_id']!=train.RUN_ID or receipt['source_sha256']!=digest(train.source_identity()):raise ValueError('v15_restore_source')
    names={p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file()}
    if names!=set(receipt['files'])|{'recovery_receipt.json'}:raise ValueError('v15_restore_members')
    for name,p in receipt['files'].items():
        raw=_regular_read(root/name,train.POLICY['limits']['checkpoint_bytes'],p['bytes'])
        if train.sha(raw)!=p['sha256']:raise ValueError('v15_restore_file_hash')
    return {'status':'verified','restored_files':len(receipt['files'])}
