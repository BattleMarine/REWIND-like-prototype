"""Build a pinned, default-off single-action control notebook after orchestration review."""

import argparse
import json
from pathlib import Path
import subprocess
import sys
import tempfile

from rewind_archive.contracts import canonical
from rewind_prepare.converter import sha
from rewind_prepare.full_archvis import INPUT_ZIP,frozen_snapshot
from rewind_v1.camera_control import POLICY,LIMITS,load_reference
from scripts.archvis_pilot_bootstrap import extract_pinned_zip
from scripts.build_archvis_pilot_handoff import _cell


MIB=1024**2
NOTEBOOK='v12_camera_control_001.ipynb'


def notebook_document(pins):
    bootstrap=Path(__file__).with_name('archvis_pilot_bootstrap.py').read_text(encoding='utf-8')
    bootstrap+=f'\nPINS={pins!r}\nPOLICY={POLICY!r}\nLIMITS={LIMITS!r}\n'
    bootstrap+='''
import json,subprocess,sys,tempfile,uuid
os.environ['OMP_NUM_THREADS']='1'
os.environ['MKL_NUM_THREADS']='1'
UPLOADS=Path('/content')
WORK=Path('/content/v12_camera_control_001');WORK.mkdir(exist_ok=True)
VERIFIED=Path(tempfile.mkdtemp(prefix='verified-',dir=WORK))
for key,cap,expanded,count in [('source',64,192,3000),('inputs',16,32,64)]:
    pin=PINS[key]
    extract_pinned_zip(UPLOADS/pin['name'],pin['sha256'],VERIFIED/key,max_zip_bytes=cap*1024**2,
        max_output_bytes=expanded*1024**2,max_members=count,revision=pin.get('revision'))
SOURCE,INPUTS=VERIFIED/'source',VERIFIED/'inputs'
REFERENCE=UPLOADS/PINS['reference']['name']
RUN=WORK/('run-'+uuid.uuid4().hex);RUN.mkdir()
RECOVERY=WORK/'recovery';RECOVERY.mkdir(exist_ok=True)
DRIVE_ROOT=Path('/content/drive/MyDrive')
MODEL_SOURCE=DRIVE_ROOT/POLICY['model_drive_relative']
MODEL_LOCAL=WORK/('model-'+POLICY['model_sha256']+'.zip')
RGB_ROOT=DRIVE_ROOT/POLICY['rgb_drive_relative']
POSE_ROOT=DRIVE_ROOT/POLICY['pose_drive_relative']

def execute(arguments,timeout):
    with subprocess.Popen([sys.executable,'-m','rewind_v1.camera_control_cli']+arguments,
            cwd=SOURCE,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True) as process:
        try:stdout,stderr=process.communicate(timeout=timeout)
        except BaseException:
            process.kill();process.wait(timeout=30)
            print('terminated child',process.pid,'exit',process.returncode);raise
    if process.returncode:raise RuntimeError(stderr[-3000:])
    return json.loads(stdout)

if execute(['source'],60)['source_sha256']!=PINS['source_code_sha256']:
    raise ValueError('control_bootstrap_source_pin')
print('fresh run',RUN)

def record(status):
    temporary=RUN/'.notebook_status.json.partial'
    temporary.write_text(json.dumps(status,ensure_ascii=False),encoding='utf-8')
    temporary.replace(RUN/'notebook_status.json')

def run_control(device):
    from google.colab import drive
    drive.mount('/content/drive')
    if not DRIVE_ROOT.parent.is_mount() or not DRIVE_ROOT.is_dir():raise ValueError('control_drive_mount_required')
    execute(['preflight','--work',str(WORK)],60)
    attempt=execute(['backup-prepare','--drive-root',str(DRIVE_ROOT),'--run-id',RUN.name],60)['attempt']
    status={'state':'started','run':str(RUN),'source_sha256':PINS['source_code_sha256'],'retained':[]}
    record(status);operation_error=backup_error=None
    def retain():
        nonlocal attempt,backup_error
        target=attempt;attempt=None  # A failed publication is never automatically retried.
        local=RECOVERY/(RUN.name+'-'+uuid.uuid4().hex+'.zip')
        try:
            archive=execute(['package','--run',str(RUN),'--output',str(local)],630)
            saved=execute(['publish','--archive',archive['archive'],'--sha256',archive['sha256'],
                '--bytes',str(archive['bytes']),'--attempt',target],630)
            status['retained'].append(saved);record(status);print('verified backup',saved)
        except BaseException as error:
            backup_error=error;status['retention_failure']={'type':type(error).__name__,'reason':str(error)[:2000],'local_zip':str(local)}
            record(status);print('backup failed; keep local results',local);raise
    try:
        arguments=['run','--inputs',str(INPUTS),'--run',str(RUN),'--rgb-root',str(RGB_ROOT),
            '--pose-root',str(POSE_ROOT),'--reference',str(REFERENCE),'--model-source',str(MODEL_SOURCE),
            '--model-local',str(MODEL_LOCAL),'--source-sha256',PINS['source_code_sha256'],'--device',device]
        # Separate child deadlines, one user action. No automatic retry or fallback.
        for stage,seconds in [('reference',600),('select',LIMITS['metadata_seconds']),('model',LIMITS['model_copy_seconds']),
                ('prepare',LIMITS['prepare_seconds']),('infer',LIMITS['infer_seconds'])]:
            status['stage']=stage;record(status)
            result=execute(arguments+['--stage',stage],seconds+30);print(stage,result)
            if stage=='select':status['selection_status']=result['status']
            if stage in ('select','prepare'):
                retain()
                attempt=execute(['backup-prepare','--drive-root',str(DRIVE_ROOT),'--run-id',RUN.name],60)['attempt']
        status['state']='complete'
    except BaseException as error:
        operation_error=error;status.update(state='failed',type=type(error).__name__,reason=str(error)[:2000])
    finally:
        record(status)
        if attempt is not None:
            try:retain()
            except BaseException:pass  # Already recorded; preserve operation error first.
    if operation_error is not None:raise operation_error
    if backup_error is not None:raise backup_error

def display_saved():
    from IPython.display import display,Image as InlineImage,Markdown
    total=0
    comparison=RUN/'comparison/report.json'
    if comparison.is_file():
        if comparison.stat().st_size>32*1024**2:raise ValueError('control_display_report_limit')
        doc=json.loads(comparison.read_bytes())
        table=RUN/'comparison/comparison.md'
        if table.stat().st_size>1024**2:raise ValueError('control_display_table_limit')
        display(Markdown(table.read_text(encoding='utf-8')))
        for item in doc['artifacts']:
            if item['name']!='camera_distributions.png':raise ValueError('control_display_artifact_name')
            content=(RUN/'comparison'/item['name']).read_bytes();total+=len(content)
            if len(content)!=item['bytes'] or hashlib.sha256(content).hexdigest()!=item['sha256']:raise ValueError('control_display_hash')
            display(InlineImage(data=content,format='png'))
    for number in range(2):
        folder=RUN/'inference'/f'clip-{number:02d}';info=folder/'predictions.json'
        if not info.is_file():continue
        if info.stat().st_size>8*1024**2:raise ValueError('control_display_prediction_limit')
        doc=json.loads(info.read_bytes());print(doc['sequence'],'validation / nominal25FPS')
        item=next(x for x in doc['artifacts'] if x['name']=='clip.gif')
        path=folder/'clip.gif'
        if path.is_symlink() or path.stat().st_size!=item['bytes']:raise ValueError('control_display_path_size')
        total+=item['bytes']
        if total>LIMITS['result_bytes']:raise ValueError('control_display_limit')
        content=path.read_bytes()
        if hashlib.sha256(content).hexdigest()!=item['sha256']:raise ValueError('control_display_hash')
        display(InlineImage(data=content,format='gif'))
'''
    action='''
RUN_CONTROL=False
DEVICE='cpu'  # Explicit cpu/cuda; supported CUDA GPU, no model-name restriction.
if RUN_CONTROL:
    run_control(DEVICE)
    display_saved()
'''
    display='''
SHOW_SAVED_RESULTS=False
if SHOW_SAVED_RESULTS:display_saved()
'''
    return {'cells':[
        _cell('markdown','guide','# V12-CAMERA-CONTROL-001\n고정 v1.2 모델로 ArchVis validation 최대2×50frame을 동일경로 추론하고 고정 도시200pose만 회수해 조건을 비교합니다. 새 학습·도시 재추론·정규화 변경은 없습니다. 기존 CUDA torch/Pillow/NumPy 환경과 requirements-v12.txt를 사용합니다. source/input/reference ZIP을 /content에 준비합니다.\n'),
        _cell('code','bootstrap',bootstrap),
        _cell('markdown','run-guide','RUN_CONTROL 한 번으로 선정seal→모델확인→준비/pose검사→추론/비교→Drive자동보관→저장그림표시를 실행합니다. 모두 기본 꺼짐입니다. metadata30분/100k/32MiB,prepare60분/원본합2GiB,infer+render60분,모델600초/1GiB,결과파일합/ZIP각256MiB,정상시선정후/준비후/최종최대3회 각package600초/copyreadback600초 별도입니다. 중간실패는해당종료본1회회수하며게시실패자동재시도없음. reference검증600초도별도입니다. free disk8GiB/MemAvailable4GiB/RSS4GiB 관측guard.0클립 실패,1클립 partial이며 대체·재시도 없음. finally는 브라우저 단절시 VM 생존과 VM종료를 구분하며 VM소실에는 보장되지 않습니다.\n'),
        _cell('code','run',action),
        _cell('markdown','view-guide','기존 결과만 다시 표시하려면 다음 스위치를 사용합니다. 재추론하지 않습니다.\n'),
        _cell('code','view',display)],'metadata':{'kernelspec':{'display_name':'Python3','language':'python','name':'python3'}},'nbformat':4,'nbformat_minor':5}


def build(args):
    if args.output.exists() or len({args.source_zip.name,args.input_zip.name,args.reference_zip.name})!=3:
        raise ValueError('control_builder_new_output_names')
    with tempfile.TemporaryDirectory(dir=args.output.parent) as directory:
        root=Path(directory);source=root/'source';inputs=root/'inputs'
        source_pin=extract_pinned_zip(args.source_zip,args.source_sha256,source,max_zip_bytes=64*MIB,max_output_bytes=192*MIB,max_members=3000,revision=args.revision)
        input_pin=extract_pinned_zip(args.input_zip,INPUT_ZIP,inputs,max_zip_bytes=16*MIB,max_output_bytes=32*MIB,max_members=64)
        if json.loads((source/'configs/v12_camera_control_001.json').read_bytes())!=POLICY:raise ValueError('control_builder_policy_mismatch')
        frozen_snapshot(inputs);load_reference(args.reference_zip,root/'reference')
        child=subprocess.run([sys.executable,'-m','rewind_v1.camera_control_cli','source'],cwd=source,timeout=60,capture_output=True,text=True,check=True)
        pins={'source':{**source_pin,'name':args.source_zip.name,'revision':args.revision},'inputs':{**input_pin,'name':args.input_zip.name},
            'reference':{'name':args.reference_zip.name,'bytes':POLICY['reference_bytes'],'sha256':POLICY['reference_sha256']},
            'source_code_sha256':json.loads(child.stdout)['source_sha256'],'model_sha256':POLICY['model_sha256'],'model_bytes':POLICY['model_bytes']}
    raw=(json.dumps(notebook_document(pins),ensure_ascii=False,indent=1)+'\n').encode();args.output.mkdir()
    (args.output/NOTEBOOK).write_bytes(raw)
    receipt={'schema':'v12_camera_control_handoff_v1','pins':pins,'policy':POLICY,'notebook':{'name':NOTEBOOK,'bytes':len(raw),'sha256':sha(raw)}}
    (args.output/'handoff_receipt.json').write_bytes(canonical(receipt)+b'\n');return receipt


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('source-zip','input-zip','reference-zip','output'):parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--source-sha256',required=True);parser.add_argument('--revision',required=True)
    print(canonical(build(parser.parse_args())).decode())


if __name__=='__main__':main()
