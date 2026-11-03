"""원본 요청 없는 협력 stop과 bounded coverage 조회."""
import json,os,uuid
from .storage import Store,path,status
from .window import check
from rewind_archive.contracts import canonical

def stop_request(store,plan_sha256):
    p=store.root/'STOP.json'
    if not p.exists():return None
    doc=json.loads(store.read('STOP.json',4096))
    check(set(doc)=={'schema','plan_sha256','token'} and doc['schema']=='fleet_stop_v1' and
          doc['plan_sha256']==plan_sha256 and isinstance(doc['token'],str) and len(doc['token'])==32,'fleet_stop_identity')
    return doc

def request_stop(root,expected_plan_sha256):
    check(path(root).is_dir(),'fleet_store_missing');store=Store(root);head=store.head()
    check(head is not None,'fleet_stop_requires_existing_run');ledger=store.committed(head)
    check(ledger['plan_sha256']==expected_plan_sha256,'fleet_stop_plan_pin')
    doc={'schema':'fleet_stop_v1','plan_sha256':expected_plan_sha256,'token':uuid.uuid4().hex}
    temp=path(store.root/('.stop-'+doc['token']))
    with temp.open('xb') as f:f.write(canonical(doc)+b'\n')
    os.replace(temp,store.root/'STOP.json')
    check(stop_request(store,expected_plan_sha256)==doc,'fleet_stop_readback');return doc

def coverage(root):
    """최신 archive commit만 조회. 전체 catalog/pack/원본은 읽지 않는다."""
    report=status(root);report.update(archives=[],catalog_complete=False,full_integrity_complete=False,
        pair_count=None,pair_count_reason='requires_sequence_frame_rgb_pose_join',current_archive=None)
    if not report.get('head'):return report
    try:
        store=Store(root)
        def guard():check(store.io_bytes<=16*1024**2,'fleet_coverage_io_limit')
        store.guard=guard;ledger=store.committed(report['head'])
        report['current_archive']=ledger.get('last_archive')
        report['progress']=ledger.get('progress')
        report['usage_reserved_or_charged']=ledger['usage']
        from .runner import reconcile
        report['usage_reconciled']=reconcile(store,ledger,report['head'])['usage']
        if ledger.get('plan_cap'):report['remaining']={k:ledger['plan_cap'][k]-report['usage_reconciled'][k] for k in ledger['usage']}
        for key,ref in ledger['archives'].items():
            check(len(report['archives'])<128,'fleet_coverage_archive_limit')
            doc=store.committed(ref);s=doc['state']
            report['archives'].append({'archive':key,'head':ref,'members':s['ordinal'],'files':s.get('files'),
                'directories':s.get('directories'),'decoded_cursor':s['cursor'],'restart_point':s.get('restart_point'),
                'source_read_max_end':s.get('source_read_max_end'),'catalog_eof':s['eof'],
                'rolling_trailer_match':s['trailer_crc_match'],'full_integrity_complete':s['complete'],
                'completion_level':'full_integrity_verified' if s['complete'] else 'catalog_eof_crc_pending' if s['eof'] else 'partial'})
        selected=ledger.get('selected_archives')
        report['catalog_complete']=bool(selected) and all(ledger['statuses'].get(k,{}).get('eof') for k in selected)
        report['full_integrity_complete']=ledger['complete']
        stop=stop_request(store,ledger['plan_sha256'])
        report['stop_request']=stop
        active_stop=stop and stop['token']!=ledger.get('stop_acknowledged')
        report['next_action']=('explicit_pending_recovery_required' if ledger['pending'] else 'writer_active' if report.get('lease') else
            'resume_to_acknowledge_stop' if active_stop else 'catalog_finished_integrity_separate' if report['catalog_complete'] else
            ledger.get('last_stop_reason','resume_same_plan_within_remaining_budget'))
        if (store.root/'RUN.json').exists():
            summary=json.loads(store.read('RUN.json',8192))
            if summary.get('schema')=='fleet_run_summary_v1' and summary.get('head')==report['head'] and summary.get('plan_sha256')==ledger['plan_sha256']:
                report['last_run']=summary
                if not ledger['pending'] and not report.get('lease') and not active_stop:report['next_action']=summary['stop_reason']
    except Exception as exc:report['errors'].append({'stage':'coverage','type':type(exc).__name__,'reason':getattr(exc,'reason','unreadable')})
    return report
