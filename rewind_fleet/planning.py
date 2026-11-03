"""관측 inventory와 출처 있는 추정치의 cost×2. Unknown을0으로 채우지 않는다."""
import copy
import math
import re
from rewind_archive.contracts import canonical
from .storage import sha
from .window import check,MAX_PACK

MIB=1024**2
METRICS=('source_bytes','requests','seconds','artifact_io_bytes','storage_bytes','ram_bytes','disk_bytes')
AUTO_PROFILE='bounded_auto_v1'
AUTO_ENGINE={'buffer_size':MIB,'spacing':MIB,'readbuf_size':65536,'max_members':64,'max_pack_bytes':4*MIB,
             'ram_buffer_allowance_bytes':4*MIB,'strategy':'bounded_chunk_new_engine_per_segment'}
def seal(doc):return {**doc,'content_sha256':sha(canonical(doc))}
def validate(doc,schema):
    check(doc.get('schema')==schema and doc.get('content_sha256')==sha(canonical({k:v for k,v in doc.items() if k!='content_sha256'})),'fleet_document_hash')
    return doc

def inventory(observed):
    archives=[]
    for row in observed['archives']:
        category,name=row['category'],row['name'];parts=[]
        for f in row['files']:
            match=re.fullmatch(re.escape(name)+r'\.tar\.gz(\d+)',f['name'])
            check(match is not None,'fleet_part_name')
            parts.append({'index':int(match[1]),'name':f['name'],'file_id':f['id'],'size':f['size_bytes'],
                'modified_time':f['modified_time'],'version':None})
        parts.sort(key=lambda p:p['index']);indices=[p['index'] for p in parts]
        check(len({p['file_id'] for p in parts})==len(parts),'fleet_duplicate_part')
        kind='pose' if 'pose' in category else 'depth' if 'depth' in category else 'rgb'
        archives.append({'key':category+'/'+name,'category':category,'name':name,'kind':kind,
            'split_scope':'test' if '_test_' in category else 'train_val','parts':parts,'official_part_count':None,
            'classification':'observed_contiguous_official_unknown' if indices==list(range(len(parts))) else 'missing_or_reordered_parts',
            'runtime_versions_verified':False,'training_allowed':False})
    archives.sort(key=lambda a:(a['split_scope']=='test',{'rgb':0,'pose':1,'depth':2}[a['kind']],a['key']))
    return seal({'schema':'fleet_inventory_v1','archives':archives,'official_registry_complete':False,
        'known_compressed_bytes':sum(p['size'] for a in archives for p in a['parts']),
        'observed_files':sum(len(a['parts']) for a in archives),'source_inventory_sha256':sha(canonical(observed))})

def source(a):
    check(a['parts'] and [p['index'] for p in a['parts']]==list(range(len(a['parts']))),'fleet_part_order')
    check(all(isinstance(p['version'],str) and p['version'] for p in a['parts']),'fleet_runtime_version_required')
    return {'schema':'ordered_drive_gzip_source_v1','archive_id':sha(a['key'].encode())[:32],
        'kind':a['kind'] if a['kind'] in ('rgb','pose') else 'mixed','dataset_version':'fleet_observed_20261103',
        'parts':[{k:p[k] for k in ('index','file_id','size','version','modified_time')} for p in a['parts']],
        'total_bytes':sum(p['size'] for p in a['parts'])}

def official(inv,registry,expected_sha256):
    validate(inv,'fleet_inventory_v1');check(sha(canonical(registry))==expected_sha256,'fleet_official_registry_pin')
    by_key={r['key']:r for r in registry['archives']};check(len(by_key)==len(registry['archives']),'fleet_official_duplicate')
    doc=copy.deepcopy(inv);doc.pop('content_sha256')
    for a in doc['archives']:
        row=by_key.get(a['key'])
        if row:
            check(row['official_part_count']==len(a['parts']) and a['classification']=='observed_contiguous_official_unknown','fleet_official_count_mismatch')
            a['official_part_count']=row['official_part_count'];a['official_count_source']={k:row[k] for k in ('script_url','script_sha256')}
    doc['official_registry_sha256']=expected_sha256;doc['official_registry_complete']=all(a['official_part_count']==len(a['parts']) for a in doc['archives'])
    return seal(doc)

def pin(inv,session,selected,*,max_requests=64,max_seconds=300):
    """명시 archive의 metadata만 읽고 size/time을 대조한 뒤 version을 고정한다."""
    from rewind_archive.drive_metadata import validate_file,FIELDS
    from rewind_archive.sources import session_opener,read_metadata_body,close_response
    from rewind_archive.contracts import Budget,Limits
    import json,sys
    import time
    validate(inv,'fleet_inventory_v1');doc=copy.deepcopy(inv);doc.pop('content_sha256');check(set(selected)<={a['key'] for a in doc['archives']},'fleet_unknown_archive')
    check(type(max_requests) is int and 1<=max_requests<=512 and type(max_seconds) in (int,float) and 0<max_seconds<=1800,'fleet_pin_budget')
    check(sum(len(a['parts']) for a in doc['archives'] if a['key'] in selected)<=max_requests,'fleet_pin_request_plan')
    started=time.monotonic();requests=body=0
    budget=Budget(Limits(max_ranges=max_requests,max_transfer_bytes=2*MIB,max_elapsed_seconds=max_seconds,request_timeout_seconds=min(5,max_seconds)))
    opener=session_opener(session,'metadata_initial')
    for a in doc['archives']:
        if a['key'] not in selected:continue
        check([p['index'] for p in a['parts']]==list(range(len(a['parts']))),'fleet_part_order')
        for p in a['parts']:
            budget.request('metadata',max_requests)
            response=opener('https://www.googleapis.com/drive/v3/files/'+p['file_id']+'?fields='+FIELDS,{'Accept-Encoding':'identity'},budget.timeout())
            try:
                raw=read_metadata_body(response,budget,16384,lambda _:None,operation='metadata_initial')
                r=validate_file(json.loads(raw),p)
                check(p['size']==int(r['size']) and p['modified_time']==r['modifiedTime'],'fleet_metadata_source_changed')
                p['version']=r['version']
            finally:close_response(response,'metadata_initial','fleet_metadata_pin_failed',preserve_active=sys.exc_info()[0] is not None)
        a['runtime_versions_verified']=True
    doc['metadata_pin_stats']={'requests':budget.ranges,'response_body_bytes':budget.transfer_bytes,'seconds':time.monotonic()-started,'archive_media_body_bytes':0,
        'request_limit':max_requests,'seconds_limit':max_seconds}
    return seal(doc)

def plan(inv,estimates,selected,*,verification='catalog_only',scope='full',storage_backend='mounted_remote',execution_profile='legacy'):
    validate(inv,'fleet_inventory_v1')
    check(verification in ('catalog_only','full_bounded') and scope in ('full','calibration') and storage_backend in ('local','mounted_remote'),'fleet_plan_policy')
    check(execution_profile in ('legacy',AUTO_PROFILE) and (execution_profile=='legacy' or scope=='full'),'fleet_execution_profile')
    check(selected and len(selected)==len(set(selected)),'fleet_plan_selection')
    indexed={a['key']:a for a in inv['archives']};check(set(selected)<=set(indexed),'fleet_unknown_archive')
    rows=[];all_missing=[]
    for key in selected:
        a=indexed[key];e=estimates.get(key,{})
        missing=[]
        if a['classification']!='observed_contiguous_official_unknown':missing.append('part_contiguity')
        if not a['runtime_versions_verified'] or any(not p.get('version') for p in a['parts']):missing.append('runtime_version')
        if scope=='full' and a['official_part_count']!=len(a['parts']):missing.append('official_part_count')
        required=('decoded_bytes','members','source_bytes_per_second','http_requests_per_second',
            'artifact_bytes_per_second','python_base_rss_bytes','replay_fraction','catalog_bytes_per_member','seconds_per_segment','failure_residue_bytes','basis')
        for field in required:
            if field not in e or e[field] is None:missing.append(field)
        base=None
        if not any(field in required for field in missing):
            check(isinstance(e['basis'],str) and bool(e['basis'].strip()),'fleet_estimate_basis')
            for k in required[:-1]:check(type(e[k]) in (int,float) and math.isfinite(e[k]) and (e[k]>=0 if k=='replay_fraction' else e[k]>0),'fleet_estimate_value')
            check(e['replay_fraction']<=10,'fleet_replay_estimate')
            decoded=e['decoded_bytes'];members=e['members'];c=sum(p['size'] for p in a['parts'])
            if scope=='calibration':
                check('probe_compressed_bytes' in e and 0<e['probe_compressed_bytes']<=32*MIB,'fleet_calibration_probe');c=min(c,e['probe_compressed_bytes'])
            effective_source_window=(4 if scope=='calibration' else 28)*MIB
            scan_segments=math.ceil(max(decoded/(32*MIB),members/64,c*(1+e['replay_fraction'])/effective_source_window))+1
            restore_segments=math.ceil(members*e['catalog_bytes_per_member']/(4*MIB))+math.ceil(scan_segments*8192/(4*MIB))
            segments=scan_segments+restore_segments+(1 if verification=='full_bounded' else 0)
            passes=2 if verification=='full_bounded' else 1
            media=math.ceil(c*(1+e['replay_fraction'])/MIB)*passes
            requests=3*media+2*len(a['parts'])*segments*passes+len(a['parts'])
            body=c*(1+e['replay_fraction'])*passes+requests*4096
            dictionaries=math.ceil(decoded/MIB)*(32768+18)+segments*32768
            catalog=members*e['catalog_bytes_per_member'];artifacts=dictionaries+catalog+segments*8192
            storage=2*artifacts+e['failure_residue_bytes']+2*MAX_PACK
            io=4*artifacts+2*catalog+segments*(8192+256*1024)
            ram=e['python_base_rss_bytes']+3*MAX_PACK+4*MIB+4*MIB+8*MIB+8*MIB
            disk=2*catalog+4*MAX_PACK+e['failure_residue_bytes']+(storage if storage_backend=='local' else 0)
            seconds=body/e['source_bytes_per_second']+requests/e['http_requests_per_second']+io/e['artifact_bytes_per_second']+segments*(5+e['seconds_per_segment'])
            base=dict(source_bytes=body,requests=requests,seconds=seconds,artifact_io_bytes=io,storage_bytes=storage,ram_bytes=ram,disk_bytes=disk)
            base={k:math.ceil(v) for k,v in base.items()}
        rows.append({'key':key,'source_identity':sha(canonical(a)),'base':base,'cap':{k:2*v for k,v in base.items()} if base else None,'missing':missing,'estimates':copy.deepcopy(e)})
        all_missing.extend(key+':'+f for f in missing)
    total=None
    if all(r['base'] is not None for r in rows):
        total={m:(max(r['base'][m] for r in rows) if m=='ram_bytes' else sum(r['base'][m] for r in rows)) for m in METRICS}
    doc={'schema':'fleet_cost_plan_v1','inventory_sha256':inv['content_sha256'],'selected':selected,'scope':scope,
        'verification_policy':verification,'storage_backend':storage_backend,'policy':'explicit_estimate_times_two',
        'archive_plans':rows,'base':total,'cap':{k:2*v for k,v in total.items()} if total else None,
        'complete_estimate':not all_missing,'missing':all_missing,'training_allowed':False,
        'notes':['gzip ISIZE is modulo2^32, never decoded-size estimate','artifact IO is logical read/write, not FUSE wire bytes',
            'catalog_only has no automatic full CRC second pass','retention: no automatic deletion; storage includes old+new and failure residue']}
    if execution_profile==AUTO_PROFILE:
        doc.update(execution_profile=AUTO_PROFILE,engine=copy.deepcopy(AUTO_ENGINE))
        doc['notes'].append('1MiB output chunk fits existing 4MiB buffer RAM allowance; unchanged spacing/cost formula, estimates still require evidence')
    return seal(doc)

def calibration_plan(inv,key,*,storage_backend='mounted_remote'):
    # 첫 probe를 위한 명시 계획 가정이다. 전체 archive의 해제 크기/처리량 실측으로 사용하지 않는다.
    estimate={'decoded_bytes':32*MIB,'members':64,'probe_compressed_bytes':4*MIB,
        'source_bytes_per_second':MIB,'http_requests_per_second':1,'artifact_bytes_per_second':10*MIB,
        'python_base_rss_bytes':512*MIB,'replay_fraction':1,'catalog_bytes_per_member':1024,
        'seconds_per_segment':5,'failure_residue_bytes':8*MIB,
        'basis':'bounded_probe_allocations_v1: 4MiB source goal/32MiB decoded allowance; throughput values are planning priors, not full-run measurements'}
    doc=plan(inv,{key:estimate},[key],scope='calibration',storage_backend=storage_backend)
    doc.pop('content_sha256');doc['probe_policy']='bounded_probe_allocations_v1';doc['automatic_segments']=1
    return seal(doc)
