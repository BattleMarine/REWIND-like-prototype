"""고정 pose371 감사 산출물에서 신규269 RGB 추출 입력만 만든다. 원격/학습 없음."""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time
from zipfile import ZipFile,ZIP_DEFLATED

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from rewind_archive.contracts import canonical,digest
from rewind_prepare.converter import validate_candidate
from rewind_discovery.pairs import RGB269_PROFILE,RGB269Limits,RGB269_RECEIPT_SHA,RGB269_SNAPSHOT_SHA,preflight,sha_file
MIB=1024**2


def prepare(audit_root,output):
    audit_root,output=Path(audit_root).resolve(),Path(output).absolute();started=time.monotonic()
    if output.exists() or not output.is_relative_to(ROOT/'outputs') or output.is_relative_to(audit_root):raise ValueError('new_separate_output_required')
    summary=json.loads((audit_root/'summary.json').read_bytes())
    if summary['status']!='pose_audit_passed_rgb_plan_only' or summary['new_pose']!=269 or summary['total_pose']!=371 or summary['train_ready'] is not False:raise ValueError('fixed_pose_audit_required')
    candidate=validate_candidate(json.loads((audit_root/'pose269/candidate.json').read_bytes()))
    pairs=json.loads((audit_root/'rgb269_pairs.json').read_bytes())
    output.mkdir(parents=True);inputs=output/'inputs';inputs.mkdir()
    def guard():
        if time.monotonic()-started>600 or sum(p.stat().st_size for p in output.rglob('*') if p.is_file())>256*MIB:raise ValueError('rgb269_input_prepare_budget')
    def write(p,value):
        guard();p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(canonical(value)+b'\n');guard()
    shutil.copytree(audit_root/'metadata',inputs/'metadata')
    shutil.copytree(audit_root/'pose269',inputs/'pose')
    # 두 sequence의 독립 converter 감사와 원본 ZIP별 provenance를 유지한다.
    for i in (0,1):
        group=inputs/'pose/batches'/str(i);group.mkdir(parents=True)
        for name in ('audit.json','extraction_manifest.json','motion_map.json'):
            shutil.copyfile(audit_root/'groups'/str(i)/name,group/name)
    motion_sequences={}
    for i in (0,1):
        motion_sequences.update(json.loads((inputs/'pose/batches'/str(i)/'motion_map.json').read_bytes())['sequences'])
    write(inputs/'metadata/motion_map.json',{'schema_version':'explicit_motion_map_v1','sequences':motion_sequences})
    for row in candidate['records']:
        p=inputs/'pose/files'/row['source']['member']
        if p.stat().st_size!=row['source']['size_bytes'] or sha_file(p)!=row['source']['sha256']:raise ValueError('pose_candidate_raw_pin')
    result=ROOT/'outputs/test_archvis_preindex_g16_002_20261103/archvis_rgb_preindex_g16_002_result.zip'
    if sha_file(result)!='320343f3c65a780d1258877da56a539f777ea2acba115b210de65c78dfb9ce13':raise ValueError('fixed_g25_result_pin')
    with ZipFile(result) as z:
        name='snapshot/g25-'+RGB269_SNAPSHOT_SHA+'.zip'
        raw=z.read(name);receipt_raw=z.read(name+'.receipt.json');receipt=json.loads(receipt_raw)
        if hashlib.sha256(raw).hexdigest()!=RGB269_SNAPSHOT_SHA or hashlib.sha256(receipt_raw).hexdigest()!=RGB269_RECEIPT_SHA:raise ValueError('fixed_g25_snapshot_pin')
        snapshot=inputs/'snapshot';snapshot.mkdir()
        (snapshot/receipt['filename']).write_bytes(raw);(snapshot/(receipt['filename']+'.receipt.json')).write_bytes(receipt_raw)
        rgb_manifest=json.loads(z.read('manifest.json'))
    config={'schema':RGB269_PROFILE,'job_id':'rgb269_pairs_001','pairs':pairs,'snapshot_receipt':receipt,
      'receipt_file_sha256':RGB269_RECEIPT_SHA,'rgb_manifest':rgb_manifest,'source_sha256':summary['g25_source_sha256'],
      'pose_extraction_sha256':json.loads((inputs/'pose/extraction_manifest.json').read_bytes())['content_sha256'],
      'candidate_sha256':candidate['content_sha256'],'train_ready':False,'limits':asdict(RGB269Limits())}
    write(inputs/'config.json',config)
    pins={p.relative_to(inputs).as_posix():{'bytes':p.stat().st_size,'sha256':sha_file(p)} for p in sorted(inputs.rglob('*')) if p.is_file()}
    write(inputs/'input_manifest.json',{'schema':'pinned_pair_inputs_v1','files':pins})
    checked,pin=preflight(inputs)
    bundle=output/'rgb269_inputs.zip'
    with ZipFile(bundle,'x',ZIP_DEFLATED) as z:
        for p in sorted(inputs.rglob('*')):
            guard()
            if p.is_file():z.write(p,p.relative_to(inputs).as_posix())
    report={'schema':'rgb269_inputs_prepared_v1','status':'prepared','config_sha256':pin,'input_zip_sha256':sha_file(bundle),
      'input_zip_bytes':bundle.stat().st_size,'input_unpacked_bytes':sum(p.stat().st_size for p in inputs.rglob('*') if p.is_file()),
      'input_files':len(pins)+1,'pairs':len(checked['pairs']),'rgb_payload_bytes':summary['rgb_payload_bytes'],
      'limits':config['limits'],'remote_requests':0,'train_ready':False,'elapsed_seconds':time.monotonic()-started}
    write(output/'preparation_report.json',report);return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audit-root',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();print(json.dumps(prepare(args.audit_root,args.output),indent=2))
