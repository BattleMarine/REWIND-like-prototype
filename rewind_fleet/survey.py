"""고정 survey request→1회 runtime pin→영속 plan과 bounded 상태 회수."""
import copy,hashlib,heapq,json,os,re,sys,time,uuid,zipfile
from pathlib import Path
from rewind_archive.contracts import canonical,safe_path
from .planning import validate,seal,survey_plan,pin,SURVEY_POLICY
from .storage import path,sha
from .window import check
from .control import coverage
from .runner import failure_chain

PIN_BASE={'body_bytes':1024**2,'seconds':None}
RECOVERY_BASE={'output_bytes':8*1024**2,'io_bytes':16*1024**2,'seconds':None}
PREPARE_BASE={'output_bytes':2*1024**2,'io_bytes':4*1024**2,'seconds':None}

def doubled(values):return {k:2*v if v is not None else None for k,v in values.items()}

def request(inv):
    validate(inv,'fleet_inventory_v1')
    ordered=sorted(inv['archives'],key=lambda a:(a.get('split_scope')=='test',a.get('name',a['key']),{'rgb':0,'pose':1,'depth':2}[a['kind']]))
    proposal=survey_plan(inv,[a['key'] for a in ordered])
    pin_base={'requests':sum(len(a['parts']) for a in ordered),**PIN_BASE}
    return seal({'schema':'fleet_survey_request_v1','policy':SURVEY_POLICY,'observed_inventory_sha256':inv['content_sha256'],
        'selected':proposal['selected'],'segments_per_archive':2,'approved_base':proposal['base'],'approved_cap':proposal['cap'],
        'pin_base':pin_base,'pin_cap':doubled(pin_base),
        'recovery_base':RECOVERY_BASE,'recovery_cap':doubled(RECOVERY_BASE),
        'prepare_base':PREPARE_BASE,'prepare_cap':doubled(PREPARE_BASE),'time_policy':'observe_only',
        'allocation_not_full_forecast':True,'whole_archive_estimate_complete':False,'automatic_full_transition':False})

def load_json(p,maximum=4*1024**2):
    p=path(p);check(p.is_file() and p.stat().st_size<=maximum,'fleet_prepare_input_size')
    return json.loads(p.read_bytes())

def publish_json(p,doc):
    p=path(p);raw=canonical(doc)+b'\n';check(not p.exists(),'fleet_prepare_new_file_required')
    temp=path(p.with_name('.'+p.name+'-'+uuid.uuid4().hex))
    with temp.open('xb') as f:f.write(raw)
    check(temp.read_bytes()==raw and not p.exists(),'fleet_prepare_readback');os.rename(temp,p)
    check(p.read_bytes()==raw,'fleet_prepare_readback')

def validate_pinned(observed,runtime):
    validate(runtime,'fleet_inventory_v1');copy_inv=copy.deepcopy(runtime);copy_inv.pop('content_sha256');copy_inv.pop('metadata_pin_stats',None)
    for a in copy_inv['archives']:
        a['runtime_versions_verified']=False
        for p in a['parts']:p['version']=None
    check(seal(copy_inv)==observed,'fleet_runtime_pin_source_changed')

def prepared(observed,policy,root,*,session_factory=None,source_sha256=None):
    """완료 pin은 재사용. 실패/불완전 준비는 자동 repin/예산 reset하지 않는다."""
    validate(observed,'fleet_inventory_v1');validate(policy,'fleet_survey_request_v1')
    check(policy==request(observed),'fleet_survey_request_policy')
    check(source_sha256 is None or isinstance(source_sha256,str) and re.fullmatch('[a-f0-9]{64}',source_sha256),'fleet_prepared_source_pin')
    root=path(root);root.mkdir(parents=True,exist_ok=True)
    started=time.monotonic();written=io_bytes=0;limits=policy['prepare_cap']
    def emit(p,doc):
        nonlocal written,io_bytes
        amount=len(canonical(doc))+1;written+=amount;io_bytes+=3*amount
        check(written<=limits['output_bytes'] and io_bytes<=limits['io_bytes'],'fleet_prepared_persist_budget')
        publish_json(p,doc)
    marker=root/'PREPARED.json'
    if marker.exists():
        check(sum((root/n).stat().st_size for n in ('PREPARED.json','runtime_inventory.json','plan.json'))<=limits['io_bytes'],'fleet_prepared_reuse_io_limit')
        manifest=validate(load_json(marker),'fleet_prepared_survey_v1')
        check(manifest['request_sha256']==policy['content_sha256'],'fleet_prepared_request_conflict')
        check(manifest.get('source_zip_sha256')==source_sha256,'fleet_prepared_source_conflict')
        runtime=load_json(root/'runtime_inventory.json');plan=load_json(root/'plan.json')
        check(runtime['content_sha256']==manifest['inventory_sha256'] and plan['content_sha256']==manifest['plan_sha256'],'fleet_prepared_pin_conflict')
    else:
        check(not any(root.iterdir()),'fleet_incomplete_preparation_preserve_no_automatic_repin')
        emit(root/'PREPARING.json',{'schema':'fleet_pin_reservation_v1','request_sha256':policy['content_sha256'],
            'reserved':policy['pin_cap'],'unknown_consumption_is_not_automatically_refunded':True})
        session=None
        try:
            check(session_factory is not None,'fleet_prepare_session_required');session=session_factory()
            runtime=pin(observed,session,policy['selected'],max_requests=policy['pin_cap']['requests'],max_seconds=policy['pin_cap']['seconds'],survey_allocation=True)
            validate_pinned(observed,runtime);plan=survey_plan(runtime,policy['selected'])
            check(plan['complete_estimate'] and plan['base']==policy['approved_base'] and plan['cap']==policy['approved_cap'],'fleet_survey_approved_numeric_policy')
            emit(root/'runtime_inventory.json',runtime);emit(root/'plan.json',plan)
            manifest=seal({'schema':'fleet_prepared_survey_v1','request_sha256':policy['content_sha256'],
                'inventory_sha256':runtime['content_sha256'],'plan_sha256':plan['content_sha256'],
                'pin_stats':runtime['metadata_pin_stats'],'complete':True,'whole_archive_estimate_complete':False,
                'preparation_limits':limits,'source_zip_sha256':source_sha256})
            emit(marker,manifest)
        except BaseException as exc:
            if not (root/'PREPARE_ERROR.json').exists():
                try:emit(root/'PREPARE_ERROR.json',{'schema':'fleet_prepare_error_v1','chain':failure_chain(exc),'pin_reservation_preserved':True})
                except Exception:pass
            raise
        finally:
            if session:
                active=sys.exc_info()[1]
                try:session.close()
                except BaseException as exc:
                    if active is not None:active.fleet_cleanup_errors=[exc]
                    else:raise
    validate_pinned(observed,runtime)
    check(plan==survey_plan(runtime,policy['selected']) and plan['cap']==policy['approved_cap'] and plan['complete_estimate'],'fleet_prepared_plan_recalculation')
    return manifest

def stage_zip(archive,expected_sha256,target):
    """작은 고정 source/input ZIP만 검증·해제. 원본 cache와 raw archive에는 사용하지 않는다."""
    archive=path(archive);target=path(target)
    check(archive.stat().st_size<=16*1024**2 and sha(archive.read_bytes())==expected_sha256,'fleet_bundle_sha_or_size')
    with zipfile.ZipFile(archive) as z:
        infos=z.infolist();check(len(infos)<=4096 and sum(i.file_size for i in infos)<=64*1024**2,'fleet_bundle_expansion')
        wanted={}
        for info in infos:
            name=safe_path(info.filename.rstrip('/'));check(not info.is_dir() or info.file_size==0,'fleet_bundle_directory')
            check((info.external_attr>>16)&0o170000!=0o120000,'fleet_bundle_symlink')
            if not info.is_dir():
                check(name.casefold() not in wanted,'fleet_bundle_duplicate');wanted[name.casefold()]=(name,info)
        if target.exists():
            actual={p.relative_to(target).as_posix().casefold():p for p in target.rglob('*') if p.is_file()}
            check(set(actual)==set(wanted),'fleet_bundle_reuse_exact_files')
            for key,(name,info) in wanted.items():check(sha(actual[key].read_bytes())==sha(z.read(info)),'fleet_bundle_reuse_hash')
            return
        target.mkdir(parents=True)
        for name,info in wanted.values():
            output=path(target/name);check(output.is_relative_to(target),'fleet_bundle_path');output.parent.mkdir(parents=True,exist_ok=True)
            raw=z.read(info)
            with output.open('xb') as f:f.write(raw)

def recover(output,*,store=None,prepared_root=None,workspace=None,policy=None,extra_files=None):
    """원본/pack store 전체를 복제하지 않는 metadata-only bounded ZIP."""
    started=time.monotonic();cap=doubled(RECOVERY_BASE);io_bytes=0;files={};skipped=[]
    def add(name,p):
        nonlocal io_bytes
        if p is None:return
        p=path(p)
        if not p.is_file():return
        size=p.stat().st_size
        if size>1024**2 or io_bytes+size>cap['io_bytes']//2 or sum(map(len,files.values()))+size>cap['output_bytes']//2-65536:
            skipped.append(name);return
        raw=p.read_bytes();io_bytes+=len(raw);check(io_bytes<=cap['io_bytes']//2,'fleet_recovery_io')
        files[name]=raw
    for name,p in (extra_files or {}).items():add(safe_path(name),p)
    if store:
        status=coverage(store,io_limit=8*1024**2);io_bytes+=status['artifact_logical_io']
        files['coverage.json']=canonical(status)+b'\n'
        for n in ('HEAD.json','RUN.json','WRITER.lease','STOP.json'):add('store/'+n,Path(store)/n)
    if prepared_root:
        for n in ('PREPARING.json','PREPARED.json','PREPARE_ERROR.json','runtime_inventory.json','plan.json'):add('prepared/'+n,Path(prepared_root)/n)
    if workspace and Path(workspace).is_dir():
        # 고정 report와 제한·redact된 worker log tail만 포함한다.
        for p in heapq.nlargest(8,Path(workspace).glob('run-*.json'),key=lambda p:p.stat().st_mtime_ns):add('reports/'+p.name,p)
        logs=[]
        for p in heapq.nlargest(8,Path(workspace).glob('worker-*.log'),key=lambda p:p.stat().st_mtime_ns):
            logs.append({'name':p.name,'size':p.stat().st_size,'raw_log_in_runtime':str(p)})
            if io_bytes+131072>cap['io_bytes']//2 or sum(map(len,files.values()))+131072>cap['output_bytes']//2-65536:
                skipped.append(p.name);continue
            with p.open('rb') as f:f.seek(max(0,p.stat().st_size-131072));raw=f.read(131072)
            io_bytes+=len(raw);check(io_bytes<=cap['io_bytes']//2,'fleet_recovery_io')
            text=raw.decode('utf8',errors='replace')
            text=re.sub(r'https?://\S+','[URL]',text)
            text=re.sub(r'(?i)^.*(?:authorization|bearer |sk-[a-z0-9_-]+|access_token|refresh_token|client_secret|api_key).*$', '[REDACTED]',text,flags=re.MULTILINE)
            files['logs/'+p.name+'.redacted_tail.txt']=text.encode('utf8')
        files['worker_logs.json']=canonical(logs)+b'\n'
    if policy:files['survey_request.json']=canonical(policy)+b'\n'
    files['RECOVERY.json']=canonical({'schema':'fleet_survey_recovery_v1','source_or_pack_data_included':False,
        'store_restore_requires_original_persistent_store':True,'skipped_oversized_or_budgeted':skipped,'limits':cap,
        'metadata_snapshot_atomic':False,'active_worker_may_advance':True})+b'\n'
    check(sum(map(len,files.values()))<=cap['output_bytes']//2,'fleet_recovery_size')
    output=path(output);check(not output.exists(),'fleet_recovery_new_file');output.parent.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(output,'x',compression=zipfile.ZIP_DEFLATED) as z:
        for name,raw in files.items():z.writestr(name,raw)
    io_bytes+=output.stat().st_size
    check(output.stat().st_size<=cap['output_bytes'] and io_bytes+output.stat().st_size<=cap['io_bytes'],'fleet_recovery_limit')
    digest=sha(output.read_bytes());io_bytes+=output.stat().st_size
    return {'path':str(output),'bytes':output.stat().st_size,'sha256':digest,'source_or_pack_data_included':False,
            'logical_io_bytes':io_bytes,'elapsed_seconds':time.monotonic()-started,'time_policy':'observe_only'}
