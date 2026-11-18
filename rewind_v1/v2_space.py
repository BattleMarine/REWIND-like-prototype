"""Action-specific incremental free space, grouped by actual output filesystem."""
import ctypes
import json
import os
from pathlib import Path
import shutil

from rewind_m1.local_rgb import _regular_read
from rewind_prepare.temporal_selection import check_pin

MIB=1024**2


def available_ram():
    if os.name!='nt':
        from .cross_cli import _linux_available_ram
        return _linux_available_ram()
    class Memory(ctypes.Structure):
        _fields_=[('length',ctypes.c_uint32),('load',ctypes.c_uint32)]+[(n,ctypes.c_uint64) for n in ('total','available','total_page','available_page','total_virtual','available_virtual','extended')]
    status=Memory();status.length=ctypes.sizeof(status)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):raise OSError('v2_windows_available_memory')
    return status.available


def existing_parent(path):
    p=Path(path).absolute()
    while not p.exists():
        if p.parent==p:raise ValueError('v2_output_filesystem_missing')
        p=p.parent
    if p.is_file():p=p.parent
    return p


def payload_bytes(path,check):
    p=Path(path)
    if not p.exists():return 0
    if p.is_symlink():raise ValueError('v2_output_symlink')
    total=0
    for item in p.rglob('*'):
        check()
        if item.is_symlink():raise ValueError('v2_output_symlink')
        if item.is_file():total+=item.stat().st_size
    return total


def read_catalog(path,schema):
    d=json.loads(_regular_read(path,32*MIB));check_pin(d,schema,d.get('content_sha256'));return d


def requirements(action,plan,cfg,paths,check=lambda:None):
    """Return incremental bytes only; installed cache is not reserved a second time."""
    r=cfg['resources'];needs=[]
    def add(name,size):
        path=paths.get(name)
        if path is None:raise ValueError('v2_required_output:'+name)
        needs.append({'path':str(path),'additional_bytes':max(0,int(size)),'purpose':name})
    add('work',32*MIB)
    if action=='prepare':
        if plan['estimated_cache_bytes']>r['cache_bytes']:raise ValueError('v2_cache_capacity_decision_required')
        add('cache',max(0,plan['estimated_cache_bytes']-payload_bytes(paths['cache'],check))+64*MIB)
    elif action=='copy-cache':
        local=read_catalog(Path(paths['cache'])/'catalog.json','rewind_v2_local_catalog_v1');done=0
        remote_path=Path(paths['retained'])/'catalog.json'
        if remote_path.exists():
            remote=read_catalog(remote_path,'rewind_v2_retained_catalog_v1')
            if remote['binding']!=local['binding'] or len(remote['packages'])>len(local['shards']):raise ValueError('v2_retained_binding_or_downgrade')
            done=len(remote['packages'])
        pending=local['shards'][done:]
        add('cache',64*MIB if pending else 0)
        missing=sum(sum(p['bytes'] for p in row['files'].values())+MIB for row in pending)
        add('retained',min(missing,r['write_bytes'],r['read_bytes']//2)+(2*MIB if pending else 0))
    elif action=='restore-cache':
        remote=read_catalog(Path(paths['retained'])/'catalog.json','rewind_v2_retained_catalog_v1')
        missing=sum(p['bytes'] for p in remote['packages'] if not (Path(paths['cache'])/f'shard-{p["number"]:05d}').exists())
        if missing+payload_bytes(paths['cache'],check)>r['cache_bytes']:raise ValueError('v2_cache_capacity_decision_required')
        add('cache',missing+64*MIB)
    elif action=='train':
        if plan['estimated_cache_bytes']>r['cache_bytes'] or plan['estimated_feature_bytes']>r['feature_bytes']:raise ValueError('v2_training_capacity_decision_required')
        add('features',max(0,plan['estimated_feature_bytes']-payload_bytes(paths['features'],check))+4*MIB)
        add('run',max(0,r['run_bytes']-payload_bytes(paths['run'],check))+r['checkpoint_bytes'])
        add('work',2*r['archive_bytes'])
        add('result_destination',r['archive_bytes'])
    elif action=='restore-run':add('run',2*r['archive_bytes'])
    else:raise ValueError('v2_space_unknown_action')
    groups={}
    for need in needs:
        check();parent=existing_parent(need['path']);volume=parent.stat().st_dev
        g=groups.setdefault(volume,{'volume':str(parent),'additional_bytes':0,'free_bytes':shutil.disk_usage(parent).free,'purposes':[]})
        g['additional_bytes']+=need['additional_bytes'];g['purposes'].append(need['purpose'])
    return {'action':action,'volumes':list(groups.values()),'fit':all(g['additional_bytes']<=g['free_bytes'] for g in groups.values()),'scope':'incremental output bytes; installed payload already consumes measured free space'}
