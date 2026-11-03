"""원본 고정 pose269+102 감사와 g25 RGB 비용 조사. 원격/학습 없음."""
from collections import defaultdict
import copy,hashlib,json,shutil,sqlite3,struct,sys,time
import argparse
from pathlib import Path
from zipfile import ZipFile
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from rewind_archive.contracts import canonical,digest,safe_path
from rewind_archive.matching import identify
from rewind_archive.profiles import target269_names
from rewind_prepare.converter import convert,validate_candidate
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output',type=Path,required=True)
args=parser.parse_args()
MIB=1024**2;out=args.output.absolute();started=time.monotonic()
if not out.is_relative_to(ROOT/'outputs'):raise ValueError('output_must_be_under_project_outputs')
if out.exists():raise ValueError('new_output_required')
out.mkdir()
def guard():
    assert time.monotonic()-started<600
    assert sum(p.stat().st_size for p in out.rglob('*') if p.is_file())<256*MIB
def write(p,d):
    guard();p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(canonical(d)+b'\n');guard()
def sha_file(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        while raw:=f.read(MIB):h.update(raw);guard()
    return h.hexdigest()
old=ROOT/'outputs/rgb102_diagnostic_20261103/developer/run/input_audit'
metadata=out/'metadata';shutil.copytree(old/'metadata',metadata)
sources=[('outputs/test_pose_missing269_001_20261103/archive_job_pose_missing269_001.zip','61bd105189ca2c0165dbb6dbac3f7c002e4937a00888daf4f8a3229f9ed411e2',257),
 ('outputs/pose12_recovery_20261103/root/recovered_pose4.zip','7d2db3c4abe4ec8ac6495c9d19560f37e393ea0e0273b3271e64ea2956a3781a',4),
 ('outputs/test_pose_missing8_001_20261103/archive_job_pose_missing8_001.zip','fc31bd09f0c46e70a0ccabff98d71a25db9e7a78491e3ae34b48c3a8b8361e13',8)]
rows={};provenance={};archive=None
for name,pin,count in sources:
    p=ROOT/name;assert not p.is_symlink() and sha_file(p)==pin
    with ZipFile(p) as z:
        if count==4:
            audit=json.loads(z.read('audit.json'));entries=audit['files'];prefix=''
            names={i.filename:i for i in z.infolist()}
            def payload(row):return z.read(row['relative_path'])
        else:
            emname=next(n for n in z.namelist() if n.endswith('/extraction_manifest.json'))
            em=json.loads(z.read(emname));assert em['content_sha256']==digest({k:v for k,v in em.items() if k!='content_sha256'})
            entries=em['files'];prefix=emname.rsplit('/',1)[0]+'/'
            if archive is None:archive=em['archive_manifest']
            else:assert em['archive_manifest']==archive
            def payload(row):return z.read(prefix+row['relative_path'])
        assert len(entries)==count
        for row in entries:
            guard();member=safe_path(row['member']);raw=payload(row)
            assert member not in rows and len(raw)==row['size_bytes'] and hashlib.sha256(raw).hexdigest()==row['sha256'] and isinstance(json.loads(raw),dict)
            target=out/'pose269/files'/member;target.parent.mkdir(parents=True,exist_ok=True)
            with target.open('xb') as f:f.write(raw)
            rows[member]={'member':member,'relative_path':'files/'+member,'size_bytes':len(raw),'sha256':row['sha256'],'kind':'pose','tar_complete':True}
            provenance[member]={'result_zip':name,'result_zip_sha256':pin,'source_member':prefix+row['relative_path']}
assert set(rows)==set(target269_names()) and len(rows)==269
def em_for(entries):
    em={'schema_version':'archive_extraction_v1','archive_manifest':archive,'inspection_candidates_usable':True,'failure_class':'none',
      'train_ready':False,'whole_archive_integrity_verified':False,'files':entries,'producer':'pinned_result_union_without_remote_requests'}
    em['content_sha256']=digest(em);return em
write(out/'pose269/extraction_manifest.json',em_for(list(rows.values())))
write(out/'pose269/provenance.json',provenance)
# 원래256files/32MiB spec을 바꾸지 않고 sequence별147/122를 각각 변환한다.
groups=defaultdict(list)
for name,row in rows.items():groups[identify(name,'pose')[0]].append(row)
candidates=[];audits=[]
for i,(seq,entries) in enumerate(sorted(groups.items())):
    group=out/'groups'/str(i)
    for row in entries:
        p=group/row['relative_path'];p.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(out/'pose269'/row['relative_path'],p)
    write(group/'extraction_manifest.json',em_for(entries))
    write(group/'motion_map.json',{'schema_version':'explicit_motion_map_v1','sequences':{seq:{'motion_id':'declared:'+seq,'identity_status':'declared_unverified','evidence':'archive sequence별 선언. 원본 motion 독립성은 미검증.'}}})
    candidate,audit=convert(group/'files',group/'extraction_manifest.json',metadata/'split',group/'motion_map.json',ROOT/'configs/real_archvis62_v1.json',metadata/'scaramuzza.json')
    write(group/'candidate.json',candidate);write(group/'audit.json',audit);candidates.append(candidate);audits.append(audit)
# 전체269 후보는 검증된 서로 다른 전체 sequence 결과의 union이며 일반 release가 아니다.
combined=copy.deepcopy(candidates[0]);combined['records']=sorted([r for c in candidates for r in c['records']],key=lambda r:(r['sequence_id'],r['frame_index']))
combined['manifest']['raw_files']=[r for c in candidates for r in c['manifest']['raw_files']]
combined['manifest']['source_hashes']={'batch_candidates':[c['content_sha256'] for c in candidates],'spec':candidates[0]['manifest']['source_hashes']['spec'],
 'calibration':candidates[0]['manifest']['source_hashes']['calibration'],'official_split_files':candidates[0]['manifest']['source_hashes']['official_split_files']}
combined['training_eligibility']['blockers']=sorted({b for c in candidates for b in c['training_eligibility']['blockers']})
combined['content_sha256']=digest({k:v for k,v in combined.items() if k!='content_sha256'});validate_candidate(combined)
write(out/'pose269/candidate.json',combined)
# 기존102를 원 raw 그대로 재감사하고 기존records/contract/차단사유를 비교한다.
old_zip=ROOT/'outputs/test_rgb102_pairs_001_20261103/rgb102_pairs_001_result.zip'
assert sha_file(old_zip)=='ec11b7eb5c4a69f76acb2e5f5045d8c7c560096a6dd33b2249e5d3a6a3e1dcae'
with ZipFile(old_zip) as z:
    old_config=json.loads(z.read('inputs/config.json'))
    assert digest(old_config)=='414279e6af3049dd88400e425162f898d940b4e00f714872f6225291e0f2b42c'
    old_pins=json.loads(z.read('inputs/input_manifest.json'))['files']
for name in ('pose/candidate.json','pose/extraction_manifest.json'):
    assert sha_file(old/name)==old_pins[name]['sha256'] and (old/name).stat().st_size==old_pins[name]['bytes']
old_report=json.loads((old/'pose/extraction_manifest.json').read_bytes());old_candidate=validate_candidate(json.loads((old/'pose/candidate.json').read_bytes()))
assert old_candidate['content_sha256']==old_config['candidate_sha256']
candidate102,audit102=convert(old/'pose/files',old/'pose/extraction_manifest.json',metadata/'split',metadata/'motion_map.json',ROOT/'configs/real_archvis62_v1.json',metadata/'scaramuzza.json')
assert candidate102['records']==old_candidate['records'] and candidate102['contract']==old_candidate['contract'] and candidate102['training_eligibility']==old_candidate['training_eligibility']
write(out/'old102_audit.json',audit102)
allrecords=candidate102['records']+combined['records'];keys={(r['sequence_id'],r['frame_index']) for r in allrecords};assert len(keys)==371 and all(r['split']=='train' for r in allrecords)
# Canonical g25 DB 복사본만 읽는다. 원본 SHA 고정, snapshot 내용 별도 준비 단계에서 pin 대조.
dbpath=ROOT/'outputs/test_archvis_preindex_g16_002_20261103/root_review/index.sqlite'
dbsha=sha_file(dbpath);assert dbsha=='6272b03c276d8edbdbbb6377ae3f19fb28095d4beaf9f7d12f9984b031019fbc'
db=sqlite3.connect('file:'+str(dbpath)+'?mode=ro',uri=True)
try:state_raw,state_sha,blob,blob_sha=db.execute('SELECT state,state_sha,gzidx,gzidx_sha FROM checkpoint').fetchone()
finally:db.close()
assert hashlib.sha256(state_raw).hexdigest()==state_sha and hashlib.sha256(blob).hexdigest()==blob_sha
state=json.loads(state_raw);assert state['generation']==25 and len(state['members'])==782
views=defaultdict(dict)
for row in state['members']:
    if row['type']=='file':
        seq,frame,side=identify(row['name'],'rgb');assert side not in views[(seq,frame)];views[(seq,frame)][side]=row
stereo={k for k,v in views.items() if set(v)=={'left','right'}};assert stereo==keys
newkeys={(r['sequence_id'],r['frame_index']) for r in combined['records']};oldkeys=keys-newkeys
assert len(newkeys)==269 and len(oldkeys)==102
ordered=[r for r in state['members'] if r['type']=='file' and identify(r['name'],'rgb')[:2] in newkeys]
assert len(ordered)==538
assert blob[:7]==b'GZIDX\x01\x00'
compressed,uncompressed,spacing,window,npoints=struct.unpack_from('<QQIII',blob,7);offset=35;points=[];flags=[]
for i in range(npoints):
    c,u,bits,flag=struct.unpack_from('<QQBB',blob,offset);offset+=18;points.append((c,u,bits));flags.append(flag)
assert len(blob)==offset+sum(flags)*window
payload=sum(r['size'] for r in ordered);seek=[];chunks=[]
for start in range(0,len(ordered),32):
    members=ordered[start:start+32];minimum=members[0]['offset'];end=max(r['offset']+r['size'] for r in members)
    preceding=max((p for p in points if p[1]<=minimum),key=lambda p:p[1])
    following=next((p for p in points if p[1]>=end),None)
    # Restart 이후 재해제/Range 정렬을 포함한 계획 추정. 실제 engine read trace나 엄밀한 상계가 아니다.
    last=following[0] if following else state['coverage']['compressed_bytes'] if isinstance(state.get('coverage'),dict) else compressed
    chunks.append({'index':len(chunks),'files':len(members),'payload_bytes':sum(r['size'] for r in members),
      'decoded_start':minimum,'decoded_end':end,'restart_compressed':preceding[0],'restart_decoded':preceding[1],
      'estimated_compressed_end':last,'estimated_range_bytes':max(0,((last+MIB-1)//MIB)*MIB-(preceding[0]//MIB)*MIB)})
    seek.extend(r['offset'] for r in members)
numeric={}
for a in audits+[audit102]:
    for k,v in a['numeric_validation'].items():
        x=numeric.setdefault(k,{'count':0,'max':0.,'sum':0.});x['count']+=v['count'];x['max']=max(x['max'],v['max']);x['sum']+=v['mean']*v['count']
for v in numeric.values():v['mean']=v.pop('sum')/v['count']
fields={}
for a in audits+[audit102]:
    for k,v in a['field_counts'].items():fields[k]=fields.get(k,0)+v
summary={'status':'pose_audit_passed_rgb_plan_only','new_pose':269,'total_pose':371,'validation':0,'old102_records_contract_blockers_unchanged':True,
 'old102_source_zip_sha256':'ec11b7eb5c4a69f76acb2e5f5045d8c7c560096a6dd33b2249e5d3a6a3e1dcae',
 'numeric_validation':numeric,'field_counts':fields,'selected_joint_count':62,'distance_unit':'mm','unit_status':audit102['unit_status'],
 'train_ready':False,'rgb_new_extracted':0,'remote_requests':0,'pose_source_pins':sources,'rgb_pairs':269,'rgb_files':538,
 'rgb_payload_bytes':payload,'rgb_max_file_bytes':max(r['size'] for r in ordered),'rgb_offsets':[min(seek),max(r['offset']+r['size'] for r in ordered)],
 'rgb_batches32':chunks,'estimated_compressed_range_sum_bytes':sum(c['estimated_range_bytes'] for c in chunks),
 'g25_db_sha256':dbsha,'g25_source_sha256':state['source_sha256'],'g25_index_blob_bytes':len(blob),'g25_index_points':npoints,'dictionary_bytes':sum(flags)*window,
 'elapsed_seconds':time.monotonic()-started,'pose_audit_disk_bytes':sum(p.stat().st_size for p in out.rglob('*') if p.is_file()),'limits':{'disk_bytes':256*MIB,'seconds':600}}
write(out/'summary.json',summary);write(out/'rgb269_pairs.json',[{'sequence':seq,'frame':frame,'rgb':views[(seq,frame)]} for seq,frame in sorted(newkeys)])
print(json.dumps({k:v for k,v in summary.items() if k not in ('pose_source_pins','rgb_batches32','field_counts')},indent=2))
