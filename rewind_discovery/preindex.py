"""고정 입력을 인증 전에 검증하고 기존 discovery.run 인자를 만든다."""
from dataclasses import asdict
from pathlib import Path
import hashlib

from rewind_archive.contracts import digest,safe_path
from rewind_archive_index.snapshots import _path
from .runner import PREINDEX_PROFILE,PreindexLimits,load,check,input_keys,MIB


def sha_file(path):
    h=hashlib.sha256()
    with _path(path).open('rb') as stream:
        while raw:=stream.read(65536):h.update(raw)
    return h.hexdigest()


def preflight(input_root,expected_config_sha256):
    root=_path(input_root)
    pins,_=load(root/'input_pins.json',65536)
    actual={p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file()}
    check(len(pins)<=16 and actual==set(pins)|{'input_pins.json'},'preindex_input_inventory')
    for name,pin in pins.items():
        check(safe_path(name)==name,'preindex_input_path')
        path=_path(root/name)
        check(path.is_relative_to(root) and path.stat().st_size==pin['bytes']<=72*MIB and
              sha_file(path)==pin['sha256'],'preindex_input_pin')
    config,_=load(root/'config.json',MIB)
    check(digest(config)==expected_config_sha256 and config['schema_version']=='archvis_preindex_inputs_v1' and
          config['profile']==PREINDEX_PROFILE and config['limits']==asdict(PreindexLimits()),'preindex_config_pin_or_profile')
    cache=config.get('runtime_cache_bytes',4*MIB)
    check(type(cache) is int and cache in (4*MIB,16*MIB),'preindex_runtime_cache_policy')
    resume=config['resume'];receipt_path=root/safe_path(resume['receipt_relative_path'])
    receipt,raw_sha=load(receipt_path,16384)
    check(receipt==resume['anchor'] and raw_sha==resume['receipt_file_sha256'] and
          receipt['source_sha256']==config['source_sha256'],'preindex_receipt_pin')
    bundle=receipt_path.parent/receipt['filename']
    check(bundle.stat().st_size==receipt['bytes'] and sha_file(bundle)==receipt['snapshot_id'],'preindex_snapshot_pin')
    split_paths={s:root/(s+'.txt') for s in ('train','validation','test')}
    input_keys(split_paths,config['split_pins'],root/'pose_catalog.json',config['pose_catalog_sha256'])
    return config,dict(job_id=config['job_id'],selection=config['selection'],expected_metadata=config['expected_metadata'],
        split_paths=split_paths,split_pins=config['split_pins'],pose_catalog=root/'pose_catalog.json',
        pose_sha256=config['pose_catalog_sha256'],limits=PreindexLimits(**config['limits']),profile=PREINDEX_PROFILE,
        runtime_cache_bytes=cache,
        resume={'receipt_path':receipt_path,'receipt_file_sha256':resume['receipt_file_sha256'],'anchor':receipt})
