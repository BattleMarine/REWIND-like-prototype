"""인증/원본 read/scan 없이 마지막 정상 snapshot과 보고서를 회수한다."""
import hashlib
from pathlib import Path
from zipfile import ZipFile,ZIP_STORED

from rewind_archive.contracts import canonical
from rewind_archive_index.snapshots import _path,_receipt
from .runner import load,check,MIB


def recover(workspace,snapshot_directory,output,*,job_id,selection_sha256,disk_limit=256*MIB,input_snapshot_directory=None):
    workspace=_path(workspace);snapshots=_path(snapshot_directory);output=_path(output)
    check(not output.exists() and not output.is_relative_to(workspace) and not output.is_relative_to(snapshots),'discovery_recovery_new_separate_output')
    report_present=(workspace/'report.json').is_file()
    if report_present:
        report,_=load(workspace/'report.json',MIB)
        check(report.get('job_id')==job_id and report.get('selection_sha256')==selection_sha256,'discovery_recovery_job_pin')
        selected=[(workspace/'report.json','report.json')]
    else:
        # 누락된 보고서를 성공/미실행으로 추정하지 않고 회수 상태만 기록한다.
        report={'status':'report_missing_execution_state_unknown','last_receipt':None}
        selected=[]
    for name in ('manifest.json','metadata.json'):
        if (workspace/name).is_file():selected.append((workspace/name,name))
    check(all(p.stat().st_size<=MIB for p,_ in selected),'discovery_recovery_report_limit')
    last=report.get('last_receipt')
    if last:
        last=_receipt(last);directory=snapshots
        if not (directory/last['filename']).exists() and input_snapshot_directory is not None:
            directory=_path(input_snapshot_directory)
        bundle=directory/last['filename'];receipt_path=directory/(last['filename']+'.receipt.json')
        receipt,_=load(receipt_path,16384);check(receipt==last,'discovery_recovery_receipt')
        check(bundle.is_file() and bundle.stat().st_size==last['bytes']<=72*MIB,'discovery_recovery_bundle_size')
        h=hashlib.sha256()
        with _path(bundle).open('rb') as stream:
            while data:=stream.read(65536):h.update(data)
        check(h.hexdigest()==last['snapshot_id'],'discovery_recovery_bundle_hash')
        selected += [(bundle,'snapshot/'+bundle.name),(receipt_path,'snapshot/'+receipt_path.name)]
    total=0
    for root in (workspace,snapshots):
        if root.exists():
            for n,p in enumerate(root.rglob('*')):
                check(n<4096,'discovery_recovery_inventory');_path(p)
                if p.is_file():total+=p.stat().st_size
    check(total+sum(p.stat().st_size for p,_ in selected)+MIB<=disk_limit,'discovery_recovery_disk_limit')
    with ZipFile(output,'x',ZIP_STORED) as z:
        for path,name in selected:z.write(_path(path),name)
        z.writestr('recovery.json',canonical({'job_id':job_id,'selection_sha256':selection_sha256,
            'last_receipt':last,'report_present':report_present,'execution_status':report['status'],
            'archive_network_requested':False,'train_ready':False}))
    return {'zip':str(output),'bytes':output.stat().st_size,'last_receipt':last,'network_requested':False,'report_present':report_present}
