"""고정 plan의 persistent reservation→한정 segment→정산. 실제 실행은 명시 CLI에서만."""
import copy
import json
import time
import zlib
import re
import os
from dataclasses import asdict
from pathlib import Path
import psutil
from rewind_archive.contracts import canonical
from rewind_archive_index.source import IndexLimits,Budget,SplitSource,_validate_source,IndexFailure
from rewind_archive_index.drive import DriveRuntime,DriveLimits
from .planning import validate,source as drive_source,plan as make_plan,seal,calibration_plan,AUTO_PROFILE,survey_plan
from .storage import Store,sha,path
from .window import check
from .scanner import initial,cache,scan
from .control import stop_request
from .timing import ObservedIndexLimits,ObservedIndexBudget,ObservedDriveLimits

MIB=1024**2
COUNTERS=('source_bytes','requests','seconds','artifact_io_bytes')
STEP={'source_bytes':16*MIB,'requests':1000,'seconds':120,'artifact_io_bytes':16*MIB}
TAIL={'seconds':5,'artifact_io_bytes':256*1024}

def failure_chain(exc):
    """메시지·경로·응답 본문 없이 제한된 type/reason chain만 보관한다."""
    chain=[];seen=set()
    while exc is not None and id(exc) not in seen and len(chain)<8:
        seen.add(id(exc));reason=getattr(exc,'reason',None)
        chain.append({'type':type(exc).__name__,'reason':reason if isinstance(reason,str) and re.fullmatch(r'[A-Za-z0-9_]{1,128}',reason) else None})
        exc=exc.__cause__ or (None if exc.__suppress_context__ else exc.__context__)
    return chain

def observed_stats(budget,runtime):
    result={}
    for name,callback in [('budget',budget.stats if budget else None),('runtime',runtime.stats if runtime else None)]:
        if callback:
            try:
                stats=callback();trace=stats.pop('source_read_trace',None)
                if trace is not None:stats['source_read_trace_summary']={'reads':len(trace),'first':trace[:2],'last':trace[-2:]}
                result[name]=stats
            except BaseException as exc:result[name]={'observation_error':failure_chain(exc)}
    return result

def reconcile(store,ledger,head):
    held=ledger.get('settlement')
    if not held:return ledger
    p=store.root/('settlement-'+held['token']+'.json')
    if not p.exists():return ledger
    try:
        raw=store.read(p.name,4096);receipt=validate(json.loads(raw),'fleet_settlement_v1')
        check(receipt['head']==head and receipt['token']==held['token'],'fleet_settlement_identity')
        result=copy.deepcopy(ledger)
        for k in COUNTERS:
            actual=receipt['charged'][k]
            if held['reserved'][k] is None:
                check(k=='seconds' and ledger.get('time_policy')=='observe_only' and type(actual) in (int,float) and actual>=0,'fleet_settlement_actual')
                result['observed_seconds']=result.get('observed_seconds',0)+actual
                continue
            check(type(actual) in (int,float) and 0<=actual<=held['reserved'][k],'fleet_settlement_actual')
            refund=held['reserved'][k]-actual;result['usage'][k]-=refund;result['archive_usage'][held['archive']][k]-=refund
        result['settlement']=None;return result
    except (ValueError,IndexFailure,KeyError):return ledger # Partial receipt never refunds an unknown reservation.

def full_verify(reader,budget,tar_end):
    budget.active_read_limit=budget.limits.max_read_bytes
    decoder=zlib.decompressobj(31);decoded=0
    try:
        while raw:=reader.read(65536):
            check(not decoder.eof,'fleet_trailing_gzip_stream')
            while raw:
                budget.check();remaining=budget.limits.max_decoded_bytes-budget.decoded_bytes
                try:data=decoder.decompress(raw,max(1,min(65536,remaining)))
                except zlib.error:raise IndexFailure('fleet_full_crc_failed') from None
                check(len(data)<=remaining,'decoded_bytes_limit')
                budget.decoded_bytes+=len(data);raw=decoder.unconsumed_tail
                check(not data[max(0,tar_end-decoded):].strip(b'\0'),'fleet_nonzero_tar_tail');decoded+=len(data)
                check(not decoder.unused_data,'fleet_trailing_gzip_stream')
        check(decoder.eof and decoded>=tar_end,'fleet_full_crc_incomplete')
        return decoded
    finally:reader.close()

def release_stale_lease(root,token,*,owner_confirmed_stopped=False):
    check(owner_confirmed_stopped is True,'fleet_owner_stop_confirmation_required');p=path(root)/'WRITER.lease'
    check(json.loads(p.read_bytes())['token']==token,'fleet_stale_token');p.unlink()

def recover_pending(store,expected_head,token,*,owner_confirmed_stopped=False):
    check(owner_confirmed_stopped is True and store.head()==expected_head,'fleet_pending_recovery_confirmation')
    with store.lease('explicit_recovery'):
        doc=store.committed(expected_head);check(doc['pending'] and doc['pending']['token']==token,'fleet_pending_token')
        doc=copy.deepcopy(doc);doc.update(parent=expected_head,pending=None,recovery='unknown_consumption_charged_full_reservation')
        key=store.commit(doc);store.publish(key,expected_head);return key

def run(inv,plan,store_root,workspace,*,owner,single_writer=False,max_segments=8,session_seconds=900,session_factory=None,local_sources=None,resume_stopped=False):
    validate(inv,'fleet_inventory_v1');validate(plan,'fleet_cost_plan_v1')
    recomputed=(survey_plan(inv,plan['selected'],storage_backend=plan['storage_backend']) if plan['scope']=='survey' else
        calibration_plan(inv,plan['selected'][0],storage_backend=plan['storage_backend']) if plan['scope']=='calibration' else
        make_plan(inv,{r['key']:r['estimates'] for r in plan['archive_plans']},plan['selected'],
        verification=plan['verification_policy'],scope=plan['scope'],storage_backend=plan['storage_backend'],execution_profile=plan.get('execution_profile','legacy')))
    check(recomputed==plan,'fleet_plan_recalculation')
    automatic=plan.get('execution_profile')==AUTO_PROFILE
    survey=plan['scope']=='survey'
    check(single_writer is True and plan['complete_estimate'] and plan['inventory_sha256']==inv['content_sha256'],'fleet_incomplete_plan_or_writer')
    if plan['scope']=='calibration':check(max_segments==1,'fleet_calibration_one_segment')
    if survey:check(session_seconds is None and plan.get('time_policy')=='observe_only','fleet_survey_observe_only_session')
    check(type(max_segments) is int and 1<=max_segments<=(16 if plan['scope']=='calibration' else 1000000) and
        (survey or type(session_seconds) in (int,float) and 0<session_seconds<=(1800 if plan['scope']=='calibration' else 43200)),'fleet_session_limits')
    workspace=path(workspace);workspace.mkdir(parents=True,exist_ok=True)
    check(not str(workspace).replace('\\','/').startswith('/content/drive/'),'fleet_live_cache_local_only')
    indexed={a['key']:a for a in inv['archives']};archive_plans={a['key']:a for a in plan['archive_plans']}
    step={**STEP,'source_bytes':16*MIB if plan['scope']=='calibration' else 64*MIB}
    started=time.monotonic();deadline=None if survey else started+session_seconds;store=Store(store_root);current_reservation=None;segment_started=None;io_start=0
    charged_counters=tuple(k for k in COUNTERS if not (survey and k=='seconds'))
    storage_start=sum(p.stat().st_size for p in store.root.rglob('*') if p.is_file())
    def guard():
        if deadline is not None:check(time.monotonic()<deadline,'fleet_session_time')
        check(psutil.Process().memory_info().rss<=plan['cap']['ram_bytes'],'fleet_ram_limit')
        check(sum(p.stat().st_size for p in workspace.rglob('*') if p.is_file())<=plan['cap']['disk_bytes'],'fleet_local_disk_limit')
        # IO byte ledger is conservative: every written byte is also included in io_bytes.
        check(storage_start+store.written_bytes<=plan['cap']['storage_bytes'],'fleet_storage_reservation')
        if current_reservation:
            if current_reservation['seconds'] is not None:check(time.monotonic()-segment_started<current_reservation['seconds'],'fleet_segment_time')
            check(store.io_bytes-io_start<current_reservation['artifact_io_bytes'],'fleet_segment_artifact_io')
    store.guard=guard;events=[];stop='segment_count_limit'
    store.write_check=lambda n:check(storage_start+store.written_bytes+n<=plan['cap']['storage_bytes'],'fleet_storage_reservation')
    with store.lease(owner):
        head=store.head()
        if head:
            ledger=store.committed(head);check(ledger['schema']=='fleet_ledger_v1' and ledger['plan_sha256']==plan['content_sha256'],'fleet_plan_conflict')
            check(not ledger['pending'],'fleet_pending_reservation_requires_recovery')
        else:
            ledger={'schema':'fleet_ledger_v1','plan_sha256':plan['content_sha256'],'parent':None,'archives':{},'statuses':{},
                'usage':{k:None if survey and k=='seconds' else 0 for k in COUNTERS},'archive_usage':{},'pending':None,'complete':False,'train_ready':False}
            head=store.commit(ledger);store.publish(head,None)
        if automatic:
            ledger=copy.deepcopy(ledger);ledger.setdefault('selected_archives',plan['selected'])
            ledger.setdefault('plan_cap',plan['cap'])
            ledger.setdefault('progress',{'segments':0,'files':0,'directories':0,'decoded_advanced':0})
            if survey:
                ledger.setdefault('survey_segments',{});ledger.setdefault('survey_quota',plan['segments_per_archive']);ledger.setdefault('scope','survey')
                ledger.setdefault('time_policy','observe_only');ledger.setdefault('observed_seconds',0)
            request=stop_request(store,plan['content_sha256'])
            if resume_stopped and request:ledger['stop_acknowledged']=request['token']
        for _ in range(max_segments):
            if automatic:
                request=stop_request(store,plan['content_sha256'])
                if request and request['token']!=ledger.get('stop_acknowledged'):stop='stop_requested';break
                if deadline is not None and time.monotonic()+15>=deadline:stop='session_time_boundary';break
            guard();prior_ledger=ledger;ledger=reconcile(store,ledger,head);selected=None
            for name in plan['selected']:
                if survey and ledger['survey_segments'].get(name,0)>=plan['segments_per_archive']:continue
                state=ledger['statuses'].get(name)
                if not state or not state['eof'] or (plan['verification_policy']=='full_bounded' and not state['complete']):selected=name;break
            if selected is None:
                if ledger!=prior_ledger:
                    ledger['parent']=head;newhead=store.commit(ledger);store.publish(newhead,head);head=newhead
                stop='survey_allocations_finished' if survey else 'all_selected_catalogs_finished';break
            a=indexed[selected];ap=archive_plans[selected];spec=(local_sources or {}).get(selected)
            spec=spec if spec is not None else drive_source(a)
            # Local fixture injection never becomes a remote identity fallback.
            if local_sources is None:check(sha(canonical(a))==ap['source_identity'],'fleet_inventory_source_changed')
            old_ref=ledger['archives'].get(selected);old=store.committed(old_ref) if old_ref else None
            state=old['state'] if old else initial(spec)
            check(state['source_sha256']==sha(canonical(spec)) and (not old or old['source_sha256']==state['source_sha256']),'fleet_source_identity')
            verify=state['eof'] and plan['verification_policy']=='full_bounded'
            if verify and (spec['total_bytes']>32*MIB or ap['estimates']['decoded_bytes']>32*MIB):
                stop='full_crc_requires_separate_large_verifier';break
            archive_used=ledger['archive_usage'].get(selected,{k:0 for k in COUNTERS})
            reserve={k:min(step[k],plan['cap'][k]-ledger['usage'][k],ap['cap'][k]-archive_used[k]) for k in charged_counters}
            if survey:reserve['seconds']=None
            elif automatic:reserve['seconds']=min(reserve['seconds'],max(0,deadline-time.monotonic()-5))
            if reserve['source_bytes']<65536 or reserve['requests']<2*len(spec['parts'])+6 or (reserve['seconds'] is not None and reserve['seconds']<15) or reserve['artifact_io_bytes']<MIB:
                stop='global_or_archive_budget_exhausted';break
            import uuid
            token=uuid.uuid4().hex;next_ledger=copy.deepcopy(ledger);next_ledger['parent']=head
            next_ledger['pending']={'archive':selected,'token':token,'reserved':reserve,'owner':owner}
            next_ledger['archive_usage'].setdefault(selected,{k:0 for k in COUNTERS})
            if survey:next_ledger['archive_usage'][selected]['seconds']=None
            for k in charged_counters:next_ledger['usage'][k]+=reserve[k];next_ledger['archive_usage'][selected][k]+=reserve[k]
            reserved_head=store.commit(next_ledger);store.publish(reserved_head,head);head=reserved_head;ledger=next_ledger
            current_reservation=reserve;segment_started=time.monotonic();io_start=store.io_bytes
            session=runtime=None;db=None;budget=None;phase='budget_setup';cleanup_errors=[]
            try:
                limits=(ObservedIndexLimits if survey else IndexLimits)(max_read_bytes=min(32*MIB,max(65536,int(reserve['source_bytes'])//2)),max_decoded_bytes=32*MIB,
                    max_seconds=None if survey else reserve['seconds']-10,max_rss_bytes=plan['cap']['ram_bytes'])
                budget=(ObservedIndexBudget if survey else Budget)(limits)
                budget.active_read_limit=limits.max_read_bytes-min(4*MIB,limits.max_read_bytes//2)
                phase='catalog_restore';cache_key=sha(canonical(spec));db=cache(store,old_ref,workspace,cache_key)
                restoring=db is None
                if restoring:
                    rows=[];newpack=None;event={'restoration':'bounded_local_cache_progress','stats':budget.stats()}
                else:
                    pack=store.blob(old['pack']) if old else None
                    phase='source_open'
                    if spec['schema']=='ordered_drive_gzip_source_v1':
                        check(session_factory is not None,'fleet_session_factory_required');session=session_factory()
                        runtime=DriveRuntime(session,(ObservedDriveLimits if survey else DriveLimits)(max_body_bytes=int(reserve['source_bytes']),max_requests=int(reserve['requests']),
                            max_seconds=None if survey else reserve['seconds']-10,range_bytes=MIB,max_cache_bytes=4*MIB));reader=runtime.source(spec,budget)
                    else:reader=SplitSource(spec,budget)
                    phase='full_verify' if verify else 'scan'
                    if verify:
                        decoded=full_verify(reader,budget,state['tar_end']);state={**state,'complete':True,'status':'complete','verified_decoded_bytes':decoded};rows=[];newpack=pack;event={'verification':'full_single_gzip','stats':budget.stats()}
                    else:state,rows,newpack,event=scan(state,pack,reader,budget,1024**2,db,max_members=64,
                        **({'buffer_size':plan['engine']['buffer_size'],
                            'should_stop':lambda:time.monotonic()-segment_started>=(plan['segment_yield_seconds'] if survey else max(1,reserve['seconds']-15))} if automatic else {}))
                phase='source_finish';_validate_source(spec)
                if runtime:runtime.finish(budget)
                else:_validate_source(spec)
                stats=event.get('stats',{});trace=stats.pop('source_read_trace',[])
                stats['source_read_trace_summary']={'reads':len(trace),'first':trace[:2],'last':trace[-2:]}
                if automatic and not restoring:
                    state={**state,'files':(old['state'].get('files',0) if old else 0)+sum(r['type']=='file' for r in rows),
                        'directories':(old['state'].get('directories',0) if old else 0)+sum(r['type']=='directory' for r in rows),
                        'source_read_max_end':max([r['start']+r['bytes'] for r in trace]+[old['state'].get('source_read_max_end',0) if old else 0])}
                    if newpack:
                        from .window import decode
                        _,points=decode(newpack)
                        state['restart_point']=list([p[0] for p in points if p[0][1]<=state['cursor']][-1])
                phase='archive_commit'
                if restoring or (not rows and state['cursor']==(old['state']['cursor'] if old else 0) and not verify and not state['eof']):archive_head=old_ref
                else:
                    doc={'schema':'fleet_archive_commit_v1','parent':old_ref,'state':state,'start_cursor':old['state']['cursor'] if old else 0,
                        'pack':store.put(newpack),'catalog':store.put(canonical({'rows':rows})),'source_sha256':sha(canonical(spec)),
                        'event':event}
                    archive_head=store.commit(doc)
                actual_source=runtime.transfer.transfer_bytes if runtime else budget.read_bytes
                actual_requests=runtime.transfer.ranges if runtime else 0
                phase='ledger_publish'
                settled=copy.deepcopy(ledger);settled.update(parent=head,pending=None)
                if archive_head is not None:settled['archives'][selected]=archive_head
                settled['settlement']={'token':token,'archive':selected,'reserved':reserve}
                settled['statuses'][selected]={'eof':state['eof'],'complete':state['complete']}
                if automatic:
                    settled['last_archive']=selected
                    settled['last_stop_reason']='incomplete_source' if state['status']=='incomplete_source' else (
                        'no_committed_progress' if not rows and not state['eof'] and not verify and not restoring else 'resume_same_plan_within_remaining_budget')
                    progress=settled['progress'];progress['segments']+=1
                    progress['files']+=sum(r['type']=='file' for r in rows);progress['directories']+=sum(r['type']=='directory' for r in rows)
                    progress['decoded_advanced']+=state['cursor']-(old['state']['cursor'] if old else 0)
                    if survey and not restoring:settled['survey_segments'][selected]=settled['survey_segments'].get(selected,0)+1
                settled['complete']=all(settled['statuses'].get(n,{}).get('complete') for n in plan['selected']) and inv['official_registry_complete']
                newhead=store.commit(settled);store.publish(newhead,head);head=newhead;ledger=settled
                phase='settlement_receipt'
                charged={'source_bytes':actual_source,'requests':actual_requests,'seconds':time.monotonic()-segment_started+(0 if survey else TAIL['seconds']),
                    'artifact_io_bytes':store.io_bytes-io_start+TAIL['artifact_io_bytes']}
                check(all(charged[k]<=reserve[k] for k in charged_counters),'fleet_settlement_reservation_exceeded')
                receipt=seal({'schema':'fleet_settlement_v1','head':head,'token':token,'charged':charged})
                receipt_raw=canonical(receipt)+b'\n';receipt_name='settlement-'+token+'.json';store.write_check(len(receipt_raw))
                with (store.root/receipt_name).open('xb') as f:f.write(receipt_raw)
                store.io_bytes+=len(receipt_raw);store.written_bytes+=len(receipt_raw)
                check(store.read(receipt_name,4096)==receipt_raw,'fleet_settlement_readback')
                events.append({'archive':selected,'state':state,'event':event,'reservation':reserve,'actual_source_bytes':actual_source,
                    'actual_requests':actual_requests,'artifact_logical_io':store.io_bytes-io_start})
                if automatic and len(events)>64:events.pop(0)
                if state['status']=='incomplete_source':stop='incomplete_source';break
                if not rows and not state['eof'] and not verify and not restoring:stop='no_committed_progress';break
            except BaseException as exc:
                stop=exc.reason if isinstance(exc,IndexFailure) else type(exc).__name__
                events.append({'archive':selected,'status':'failed_reservation_kept','reason':stop,'phase':phase,
                    'exception_chain':failure_chain(exc),'source_reason':getattr(budget,'last_source_failure',None),
                    'checkpoint_error':failure_chain(getattr(exc,'fleet_checkpoint_error',None)),
                    'scan_primary':failure_chain(getattr(exc,'fleet_scan_primary',None)),
                    'checkpoint_failed':bool(getattr(exc,'fleet_checkpoint_failure',False) or getattr(exc,'fleet_checkpoint_error',None)),
                    'cleanup_errors':[failure_chain(e) for e in getattr(exc,'fleet_cleanup_errors',[])],
                    'reservation':dict(reserve),'observations':observed_stats(budget,runtime),
                    'limits':asdict(budget.limits) if budget else None,
                    'active_read_limit':getattr(budget,'active_read_limit',None),
                    'segment_seconds':time.monotonic()-segment_started,'artifact_logical_io':store.io_bytes-io_start,
                    'pending_reservation_kept':ledger['pending'] is not None,'unknown_consumption_refunded':False})
                if automatic and len(events)>64:events.pop(0)
                break
            finally:
                for resource in (db,session):
                    if resource is not None:
                        try:resource.close()
                        except BaseException as exc:cleanup_errors.append(failure_chain(exc))
                if cleanup_errors:
                    events[-1].setdefault('cleanup_errors',[]).extend(cleanup_errors)
                    if events[-1].get('status')!='failed_reservation_kept':stop='cleanup_failed'
                current_reservation=None
            if cleanup_errors:break
    effective=reconcile(store,ledger,head)
    dataset_complete=ledger['complete'] and set(plan['selected'])=={a['key'] for a in inv['archives']}
    report={'schema':'fleet_run_report_v1','head':head,'stop_reason':stop,'events':events,'usage':ledger['usage'],'reconciled_usage':effective['usage'],
        'pending':ledger['pending'],'complete':ledger['complete'],'selected_archives_complete':ledger['complete'],'dataset_complete':dataset_complete,
        'plan_sha256':plan['content_sha256'],'train_ready':False,
        'seconds':time.monotonic()-started,'artifact_logical_io':store.io_bytes,'durability_new_runtime_verified':False,
        'mount_atomicity_or_distributed_lock_claimed':False}
    if automatic:report.update(execution_profile=AUTO_PROFILE,progress=ledger['progress'],recent_events_limit=64,
        stop_latency='cooperative: at most one bounded segment; hard kill retains unknown reservation')
    if survey:report.update(scope='survey',survey_segments=ledger['survey_segments'],survey_complete=stop=='survey_allocations_finished',
        whole_archive_estimate_complete=False,time_policy='observe_only',observed_seconds=effective.get('observed_seconds',0))
    if automatic:
        # 마지막 정상 정산의 256KiB IO/5초 tail 안에서 작은 head-bound 요약만 남긴다.
        summary={'schema':'fleet_run_summary_v1','head':head,'plan_sha256':plan['content_sha256'],'stop_reason':stop,
            'progress':ledger['progress'],'last_archive':ledger.get('last_archive'),'pending':ledger['pending'] is not None,
            'last_failure':{k:events[-1].get(k) for k in ('phase','reason','exception_chain')} if events and events[-1].get('status') else None}
        raw=canonical(summary)+b'\n';report['summary_persisted']=False
        try:
            check(len(raw)<=8192 and (deadline is None or time.monotonic()<deadline),'fleet_summary_budget')
            # 0 segment 재조회는 새 reservation 없이 저장하지 않는다.
            check(segment_started is not None and (survey or time.monotonic()-segment_started<reserve['seconds']),'fleet_summary_budget')
            check(store.io_bytes-io_start+3*len(raw)<=reserve['artifact_io_bytes'],'fleet_summary_budget')
            store.write_check(len(raw));temp=store.root/('.run-'+token)
            with temp.open('xb') as f:f.write(raw)
            store.io_bytes+=len(raw);store.written_bytes+=len(raw)
            os.replace(temp,store.root/'RUN.json')
            check(store.read('RUN.json',8192)==raw,'fleet_summary_readback');report['summary_persisted']=True
        except Exception as exc:report['summary_error']=failure_chain(exc)
        report.update(artifact_logical_io=store.io_bytes,seconds=time.monotonic()-started)
    import uuid
    report_file=workspace/('report-'+owner+'-'+uuid.uuid4().hex+'.json');report['local_report']=str(report_file)
    with report_file.open('xb') as f:f.write(canonical(report)+b'\n')
    return report
