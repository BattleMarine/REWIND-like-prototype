"""Independent v2 manifest contract; preserves all frozen v1 contracts."""
import json
from collections import Counter
from pathlib import Path,PurePosixPath
import shutil

from rewind_archive.contracts import digest
from rewind_m1.local_rgb import _regular_read
from rewind_prepare.converter import official_splits,sha
from rewind_prepare.full_archvis import clip_coverage
from rewind_prepare.temporal_selection import seal,check_pin
from rewind_archive.matching import identify

ROOT=Path(__file__).resolve().parents[1]
SCENARIOS=('ArchVisInterior_ArchVis_RT','CityPark_Showcase','DowntownWest_Demo_Environment')


def config():return json.loads((ROOT/'configs/rewind_v2.json').read_bytes())


def source():
    from .training import _source
    return {**_source(),'configs/rewind_v2.json':sha((ROOT/'configs/rewind_v2.json').read_bytes()),
        'scripts/build_v2.py':sha((ROOT/'scripts/build_v2.py').read_bytes())}


def relative(name):
    if not isinstance(name,str) or '\\' in name or ':' in name or PurePosixPath(name).is_absolute() or any(p in ('','.','..') for p in name.split('/')):raise ValueError('v2_relative_path')
    return name


class SplitReader:
    def read(self,path):return _regular_read(path,32*1024**2)


def load(manifest_path,manifest_sha,spec_path,spec_sha,splits):
    raw=_regular_read(manifest_path,128*1024**2);spec_raw=_regular_read(spec_path,1024**2)
    if sha(raw)!=manifest_sha or sha(spec_raw)!=spec_sha:raise ValueError('v2_input_file_pin')
    doc=json.loads(raw);spec=json.loads(spec_raw)
    owners,split_sources=official_splits(Path(splits),spec,SplitReader())
    return validate(doc,spec,owners,split_sources),spec,spec_raw


def validate(doc,spec,owners,split_sources):
    check_pin(doc,'rewind_v2_audited_manifest_v1',doc.get('content_sha256'))
    if (doc['scenarios']!=list(SCENARIOS) or doc['spec_sha256']!=digest(spec) or doc['official_splits']!=split_sources
            or doc['coordinates']!='device_mm' or not doc['records']):raise ValueError('v2_manifest_contract')
    records=doc['records'];keys=[];sequence_owners={};counts=Counter()
    for r in records:
        seq=relative(r['sequence_id']);scenario=seq.split('/')[0];split=r['split'];frame=r['frame_index']
        if (scenario not in SCENARIOS or split not in ('train','validation') or owners.get(seq)!=split
                or type(frame) is not int or frame<0 or set(r['files'])!={'left','right','pose'}):raise ValueError('v2_record_membership')
        if sequence_owners.setdefault(seq,split)!=split:raise ValueError('v2_split_overlap')
        for role,p in r['files'].items():
            relative(p['path'])
            if identify(p['path'],'pose' if role=='pose' else 'rgb')!=(seq,frame,role):raise ValueError('v2_path_key_role')
            if type(p['size_bytes']) is not int or not 0<p['size_bytes']<=8*1024**2 or not isinstance(p['sha256'],str) or len(p['sha256'])!=64 or any(c not in '0123456789abcdef' for c in p['sha256']):raise ValueError('v2_file_pin')
        keys.append((seq,frame));counts[scenario,split]+=1
    if keys!=sorted(set(keys)):raise ValueError('v2_duplicate_or_unsorted')
    if any(not counts[s,p] for s in SCENARIOS for p in ('train','validation')):raise ValueError('v2_missing_scenario_split')
    inventory=doc['inventory']
    check_pin(inventory,'rewind_v2_frozen_inventory_v1',inventory.get('content_sha256'))
    if inventory['failed'] or inventory['capped'] or inventory['missing'] or inventory['duplicates']:raise ValueError('v2_inventory_incomplete')
    expected=[tuple(k) for k in inventory['usable_keys']]
    if expected!=keys or inventory['excluded'] or set(inventory['sequences'])!={s for s,p in owners.items() if p in ('train','validation') and s.split('/')[0] in SCENARIOS}:raise ValueError('v2_full_inventory_coverage_required')
    return doc


def plan(doc,spec,cfg,workspace):
    keys={p:[(r['sequence_id'],r['frame_index']) for r in doc['records'] if r['split']==p] for p in ('train','validation')}
    summary,clips=clip_coverage(keys);counts={s:{p:sum(k[0].startswith(s+'/') for k in keys[p]) for p in keys} for s in SCENARIOS}
    n=len(doc['records']);c=len(clips);j=len(spec['joint_names'])
    images=n*393216;metadata=n*(j*(4*4+3*4+2+1)+2*9*4+2*12*8)+((n+127)//128)*65536
    features=c*(50*((14+3)*j*4+j+18*4+8+1)+8192)
    cache=images+metadata;resources=cfg['resources']
    working=cache+features+resources['run_bytes']+3*resources['archive_bytes']+resources['checkpoint_bytes']+128*1024**2
    free=shutil.disk_usage(workspace).free
    return seal({'schema':'rewind_v2_plan_v1','manifest_sha256':doc['content_sha256'],'spec_sha256':digest(spec),
        'counts':counts,'summary':summary,'clips':clips,'source_bytes':sum(p['size_bytes'] for r in doc['records'] for p in r['files'].values()),
        'image_bytes':images,'estimated_cache_bytes':cache,'estimated_feature_bytes':features,'estimated_working_bytes':working,'free_disk_bytes':free,
        'resource_fit':cache<=resources['cache_bytes'] and features<=resources['feature_bytes'] and working<=free,
        'minimum_source_read_invocations':(sum(p['size_bytes'] for r in doc['records'] for p in r['files'].values())+resources['read_bytes']-1)//resources['read_bytes'],
        'estimates_only':True,'train_updates_per_epoch_at_batch2':(len(keys['train'])+1)//2+2*sum(r['split']=='train' for r in clips),
        'clock':'nominal frame/25 diagnostic; recorded timestamps/stereo synchronization/motion independence not newly verified'})


def training_ready(cfg):
    t=cfg['training']
    if any(v is None for v in t.values()):raise ValueError('v2_training_decisions_pending')
    # This infrastructure supports the existing fresh / GT-upper train policy only.
    if t['initialization']!='fresh' or t['upper_policy']!='gt_train_predicted_validation' or t['evaluation_policy']!='per_scenario_reference':raise ValueError('v2_training_policy_not_implemented')
    if set(t['epochs'])!={'frontend','body','hand'} or any(type(n) is not int or not 1<=n<=100 for n in t['epochs'].values()) or type(t['seed']) is not int:raise ValueError('v2_training_config')
