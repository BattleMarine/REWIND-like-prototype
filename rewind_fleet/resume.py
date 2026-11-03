"""정확히 고정된 survey 실패1건의 명시적 source 승계·전액예약 보존 재개."""
import argparse,copy,json,re
from pathlib import Path
from rewind_archive.contracts import canonical
from .planning import validate,seal,survey_plan
from .storage import Store,path,sha
from .survey import load_json,recover
from .runner import recover_pending,run,failure_chain
from .window import check

LIMITS={'output_bytes':4*1024**2,'io_bytes':8*1024**2,'seconds':None}

def resume_request(case,new_source_sha256):
    validate(case,'fleet_survey_resume_case_v1')
    check(isinstance(new_source_sha256,str) and re.fullmatch('[a-f0-9]{64}',new_source_sha256) and
          new_source_sha256!=case['old_source_zip_sha256'],'fleet_resume_new_source_pin')
    return seal({'schema':'fleet_survey_resume_request_v1','case':case,'new_source_zip_sha256':new_source_sha256,
        'limits':dict(LIMITS),'same_plan_and_budget':True,'unknown_reservation_charged_in_full':True,
        'time_policy':'observe_only','automatic_retry':False,'single_launch':True})

def _checked_request(request,new_source_sha256):
    validate(request,'fleet_survey_resume_request_v1')
    check(request==resume_request(request['case'],new_source_sha256),'fleet_resume_request_or_source_conflict')
    check(request['case']['owner_stopped_evidence']['worker_exit_code']==0 and
          request['case']['owner_stopped_evidence']['lease_absent'] is True,'fleet_resume_owner_stop_evidence')
    return request['case']

def _paths(request):
    key=request['content_sha256']
    return {name:'RESUME-'+name.upper()+'-'+key+'.json' for name in ('intent','complete','launch')}

def _unchanged(doc):
    return {k:v for k,v in doc.items() if k not in ('parent','pending','recovery')}

def _storage_guard(ctx,plan):
    if getattr(ctx,'storage_guard_installed',False):return
    store=ctx.store;existing=sum(p.stat().st_size for p in store.root.rglob('*') if p.is_file());prior=store.write_check
    def bounded_write(n):
        prior(n);check(existing+store.written_bytes+n<=plan['cap']['storage_bytes'],'fleet_resume_storage_limit')
    store.write_check=bounded_write;ctx.storage_guard_installed=True

class _Context:
    def __init__(self,root):
        check(path(root).is_dir(),'fleet_resume_existing_store_required')
        self.store=Store(root);self.read_bytes=0
        self.store.read_limit=LIMITS['io_bytes']
        self.store.guard=lambda:check(self.read_bytes+self.store.io_bytes<=LIMITS['io_bytes'],'fleet_resume_io_limit')
        self.store.write_check=lambda n:check(self.store.written_bytes+n<=LIMITS['output_bytes'] and
            self.read_bytes+self.store.io_bytes+n<=LIMITS['io_bytes'],'fleet_resume_write_limit')
    def read(self,p):
        p=path(p);size=p.stat().st_size
        check(size<=4*1024**2 and self.read_bytes+self.store.io_bytes+size<=LIMITS['io_bytes'],'fleet_resume_io_limit')
        raw=p.read_bytes();self.read_bytes+=len(raw);self.store.read_limit=LIMITS['io_bytes']-self.read_bytes
        return raw
    def write(self,name,doc):
        raw=canonical(doc)+b'\n';self.store.write_check(len(raw));p=path(self.store.root/name)
        check(not p.exists(),'fleet_resume_marker_exists')
        with p.open('xb') as f:f.write(raw)
        self.store.io_bytes+=len(raw);self.store.written_bytes+=len(raw)
        check(self.store.read(name,4*1024**2)==raw,'fleet_resume_marker_readback')

def _preflight(root,prepared_root,request,new_source_sha256,ctx):
    case=_checked_request(request,new_source_sha256);store=ctx.store;names=_paths(request)
    check(not (store.root/'WRITER.lease').exists(),'fleet_resume_writer_active')
    raw=ctx.read(Path(prepared_root)/'PREPARED.json');manifest=validate(json.loads(raw),'fleet_prepared_survey_v1')
    check(sha(raw)==case['prepared_file_sha256'] and manifest['content_sha256']==case['prepared_content_sha256'],'fleet_resume_prepared_conflict')
    check(manifest['source_zip_sha256']==case['old_source_zip_sha256'] and manifest['request_sha256']==case['original_request_sha256'],'fleet_resume_old_source_conflict')
    inv=validate(json.loads(ctx.read(Path(prepared_root)/'runtime_inventory.json')),'fleet_inventory_v1')
    plan=validate(json.loads(ctx.read(Path(prepared_root)/'plan.json')),'fleet_cost_plan_v1')
    check(inv['content_sha256']==case['inventory_sha256']==manifest['inventory_sha256'] and
        plan['content_sha256']==case['plan_sha256']==manifest['plan_sha256'],'fleet_resume_original_plan_conflict')
    check(plan==survey_plan(inv,plan['selected'],storage_backend=plan['storage_backend']),'fleet_resume_plan_recalculation')
    head=store.head();original=store.committed(case['head'])
    check(original['schema']=='fleet_ledger_v1' and original['plan_sha256']==case['plan_sha256'] and original['pending']==case['pending'],
        'fleet_resume_original_pending_conflict')
    check(original['usage']==case['usage'] and original['survey_segments']==case['survey_segments'] and
          original['progress']==case['progress'] and original['survey_quota']==2 and original['selected_archives']==plan['selected'],
          'fleet_resume_original_progress_conflict')
    check(not (store.root/names['launch']).exists(),'fleet_resume_already_launched_review_required')
    receipt=None
    if (store.root/names['complete']).exists():
        check((store.root/names['intent']).exists(),'fleet_resume_receipt_without_intent')
        intent=validate(json.loads(store.read(names['intent'],4*1024**2)),'fleet_resume_intent_v1')
        receipt=validate(json.loads(store.read(names['complete'],4*1024**2)),'fleet_resume_complete_v1')
        check(intent['request_sha256']==request['content_sha256'] and receipt['request_sha256']==request['content_sha256'] and
              receipt['intent_sha256']==intent['content_sha256'],'fleet_resume_receipt_conflict')
        check(intent['old_head']==receipt['old_head']==case['head'] and intent['plan_sha256']==receipt['plan_sha256']==case['plan_sha256'] and
              intent['old_source_zip_sha256']==case['old_source_zip_sha256'] and intent['new_source_zip_sha256']==receipt['new_source_zip_sha256']==new_source_sha256 and
              intent['prepared_content_sha256']==case['prepared_content_sha256'] and intent['pending_token']==case['pending']['token'],
              'fleet_resume_receipt_pin_conflict')
        check(head==receipt['recovered_head'],'fleet_resume_recovered_head_changed_review_required')
        recovered=store.committed(head)
        check(recovered=={**original,'parent':case['head'],'pending':None,'recovery':'unknown_consumption_charged_full_reservation'},
              'fleet_resume_recovered_ledger_conflict')
        phase='ready_once'
    else:
        check(not (store.root/names['intent']).exists(),'fleet_resume_incomplete_migration_review_required')
        check(head==case['head'],'fleet_resume_old_head_changed_review_required');phase='original_pending'
    return {'phase':phase,'head':head,'original_head':case['head'],'plan_sha256':case['plan_sha256'],
        'inventory_sha256':case['inventory_sha256'],'pending':case['pending'] if phase=='original_pending' else None,
        'usage':original['usage'],'survey_segments':original['survey_segments'],'progress':original['progress'],
        'old_source_zip_sha256':case['old_source_zip_sha256'],'new_source_zip_sha256':new_source_sha256,
        'prepared_unchanged':True,'archive_network_requested':False,'read_only':True},inv,plan,original,receipt

def preflight(root,prepared_root,request,*,new_source_sha256):
    ctx=_Context(root)
    report,*_= _preflight(root,prepared_root,request,new_source_sha256,ctx)
    report['logical_io_bytes']=ctx.read_bytes+ctx.store.io_bytes
    return report

def apply_once(root,prepared_root,request,*,new_source_sha256,_context=None):
    """입력에서 고정한 owner 종료 증거와 현재 lease 부재로 정확1건만 복구한다."""
    ctx=_context or _Context(root);report,inv,plan,original,receipt=_preflight(root,prepared_root,request,new_source_sha256,ctx)
    _storage_guard(ctx,plan)
    if receipt is not None:return receipt
    case=request['case'];store=ctx.store;names=_paths(request)
    # intent가 보인 뒤 crash/실패하면 다음 호출은 자동 추측·복구하지 않는다.
    intent=seal({'schema':'fleet_resume_intent_v1','request_sha256':request['content_sha256'],
        'old_head':case['head'],'plan_sha256':case['plan_sha256'],'prepared_content_sha256':case['prepared_content_sha256'],
        'old_source_zip_sha256':case['old_source_zip_sha256'],'new_source_zip_sha256':new_source_sha256,
        'pending_token':case['pending']['token'],'unchanged_ledger_sha256':sha(canonical(_unchanged(original)))})
    ctx.write(names['intent'],intent)
    # 기존 recover_pending 계약을 그대로 사용: 미확인 소비는 전액 유지한다.
    recovered_head=recover_pending(store,case['head'],case['pending']['token'],owner_confirmed_stopped=True)
    recovered=store.committed(recovered_head)
    check(recovered=={**original,'parent':case['head'],'pending':None,'recovery':'unknown_consumption_charged_full_reservation'},
        'fleet_resume_recovery_invariants_failed')
    check(store.head()==recovered_head,'fleet_resume_recovery_head_changed')
    receipt=seal({'schema':'fleet_resume_complete_v1','request_sha256':request['content_sha256'],
        'intent_sha256':intent['content_sha256'],'recovered_head':recovered_head,'old_head':case['head'],
        'plan_sha256':case['plan_sha256'],'new_source_zip_sha256':new_source_sha256,
        'unchanged_ledger_sha256':sha(canonical(_unchanged(recovered))),'unknown_reservation_charged_in_full':True})
    ctx.write(names['complete'],receipt)
    return receipt

def resume_once(root,prepared_root,request,*,new_source_sha256,workspace,owner,session_factory=None,local_sources=None):
    ctx=_Context(root);receipt=apply_once(root,prepared_root,request,new_source_sha256=new_source_sha256,_context=ctx)
    # 새 호출에서도 HEAD/receipt를 다시 대조하고 launch marker는 딱 한 번만 쓴다.
    report,inv,plan,*_=_preflight(root,prepared_root,request,new_source_sha256,ctx)
    ctx.write(_paths(request)['launch'],seal({'schema':'fleet_resume_launch_v1','request_sha256':request['content_sha256'],
        'recovered_head':receipt['recovered_head'],'new_source_zip_sha256':new_source_sha256}))
    check(ctx.store.head()==receipt['recovered_head'],'fleet_resume_launch_head_changed')
    return run(inv,plan,root,workspace,owner=owner,single_writer=True,max_segments=1000000,session_seconds=None,
        session_factory=session_factory,local_sources=local_sources,resume_stopped=False)

def recover_resume(output,*,root,prepared_root,workspace,request_file,diagnostic_file=None):
    request=load_json(request_file);validate(request,'fleet_survey_resume_request_v1')
    extras={'resume/request.json':request_file}
    if diagnostic_file:extras['resume/diagnostic.json']=diagnostic_file
    for name in _paths(request).values():extras['resume/'+name]=Path(root)/name
    return recover(output,store=root,prepared_root=prepared_root,workspace=workspace,extra_files=extras)

def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    for flag in ('store','prepared','request','request-sha256','new-source-sha256'):parser.add_argument('--'+flag,required=True)
    parser.add_argument('--apply-once',action='store_true');parser.add_argument('--workspace');parser.add_argument('--owner',default='fleet_survey_resume_001');parser.add_argument('--output')
    args=parser.parse_args(argv);phase='request_validation'
    try:
        request=load_json(args.request)
        check(request.get('content_sha256')==args.request_sha256,'fleet_resume_external_request_pin')
        from .__main__ import authorized_factory,save
        if args.output:check(not path(args.output).exists(),'fleet_resume_new_output_required')
        phase='apply_once' if args.apply_once else 'read_only_preflight'
        if args.apply_once:
            check(args.workspace is not None and args.output is not None,'fleet_resume_output_required')
            result=resume_once(args.store,args.prepared,request,new_source_sha256=args.new_source_sha256,workspace=args.workspace,owner=args.owner,
                session_factory=lambda:authorized_factory()())
        else:result=preflight(args.store,args.prepared,request,new_source_sha256=args.new_source_sha256)
        if args.output:save(args.output,result)
    except BaseException as exc:
        result={'schema':'fleet_resume_error_v1','phase':phase,'chain':failure_chain(exc),
            'request_sha256':args.request_sha256,'automatic_retry':False,'remote_state_must_be_reviewed':True}
        if args.output:
            diagnostic=path(args.output+'.error.json');diagnostic.parent.mkdir(parents=True,exist_ok=True)
            raw=canonical(result)+b'\n';check(len(raw)<=8192 and not diagnostic.exists(),'fleet_resume_diagnostic_limit')
            with diagnostic.open('xb') as f:f.write(raw)
        print(json.dumps(result,ensure_ascii=False));raise SystemExit(1)
    print(json.dumps(result,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
