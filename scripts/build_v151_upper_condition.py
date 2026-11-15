"""Pinned source + compact diagnostic bundle only; default-off Colab inference."""

import argparse
import json
from pathlib import Path
import subprocess
import sys
import tempfile

from rewind_archive.contracts import canonical
from rewind_prepare.converter import sha
from rewind_v1 import upper_condition_bundle as bundle
from scripts.archvis_pilot_bootstrap import extract_pinned_zip
from scripts.build_archvis_pilot_handoff import _cell


NOTEBOOK='v151_upper_condition.ipynb';MIB=1024**2


def notebook_document(pins):
    code=Path(__file__).with_name('archvis_pilot_bootstrap.py').read_text(encoding='utf-8')
    code+=f'\nPINS={pins!r}\nPOLICY={bundle.POLICY!r}\n'
    code+='''
import json,subprocess,sys,tempfile,uuid,time,signal
os.environ['OMP_NUM_THREADS']='1';os.environ['MKL_NUM_THREADS']='1'
started=time.monotonic()
WORK=Path('/content/v151_upper_condition');WORK.mkdir(exist_ok=True)
SOURCE=Path(tempfile.mkdtemp(prefix='source-parent-',dir=WORK))/'source'
pin=PINS['source']
extract_pinned_zip(Path('/content')/pin['name'],pin['sha256'],SOURCE,max_zip_bytes=64*1024**2,
    max_output_bytes=192*1024**2,max_members=3000,revision=pin['revision'])
RUN=WORK/('run-'+uuid.uuid4().hex);RUN.mkdir()
RECOVERY=WORK/'recovery';RECOVERY.mkdir(exist_ok=True)
DRIVE=Path('/content/drive/MyDrive')
DEVICE='cuda'  # Explicit cpu/cuda; any supported CUDA GPU, no automatic fallback.

def execute(args,timeout):
    with subprocess.Popen([sys.executable,'-m','rewind_v1.upper_condition_cli']+args,cwd=SOURCE,
            stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True) as child:
        try:stdout,stderr=child.communicate(timeout=timeout)
        except BaseException:
            if child.poll() is None:child.send_signal(signal.SIGINT)
            try:child.communicate(timeout=30)
            except subprocess.TimeoutExpired:child.kill();child.wait(timeout=30)
            raise
    if child.returncode:raise RuntimeError(stderr[-3000:])
    return json.loads(stdout)

if execute(['source'],60)['source_sha256']!=PINS['source_code_sha256']:raise ValueError('v151_source_identity')
BOOTSTRAP_WALL=time.monotonic()-started
print('fresh run',RUN,'bootstrap wall seconds',BOOTSTRAP_WALL)

def record(status):
    tmp=RUN/'.operation.partial';tmp.write_text(json.dumps(status,ensure_ascii=False),encoding='utf-8');tmp.replace(RUN/'operation.json')

def run_diagnostic():
    from google.colab import drive
    drive.mount('/content/drive')
    if not DRIVE.parent.is_mount() or not DRIVE.is_dir():raise ValueError('v151_drive_mount_required')
    attempt=execute(['backup-prepare','--drive-root',str(DRIVE),'--run-id',RUN.name],60)['attempt']
    status={'state':'started','bootstrap_wall_seconds':BOOTSTRAP_WALL,'phase_wall_seconds':{},'source_sha256':PINS['source_code_sha256']}
    operation_error=retention_error=None;record(status)
    try:
        common=['--run',str(RUN),'--source-sha256',PINS['source_code_sha256'],'--manifest-sha256',PINS['bundle']['manifest_sha256']]
        for phase,seconds,args in [('prepare',600,['--bundle',str(Path('/content')/PINS['bundle']['name']),'--bundle-sha256',PINS['bundle']['sha256']]),
                ('infer',3600,['--device',DEVICE])]:
            status['phase']=phase;record(status);start=time.monotonic()
            try:result=execute([phase]+common+args,seconds+30)
            finally:status['phase_wall_seconds'][phase]=time.monotonic()-start;record(status)
            print(phase,result)
        status['state']='complete'
    except BaseException as error:
        operation_error=error;status.update(state='failed',type=type(error).__name__,reason=str(error)[:2000])
    finally:
        record(status);local=RECOVERY/(RUN.name+'-'+uuid.uuid4().hex+'.zip')
        try:
            archive=execute(['package','--run',str(RUN),'--output',str(local)],630)
            saved=execute(['publish','--archive',archive['archive'],'--sha256',archive['sha256'],'--bytes',str(archive['bytes']),
                '--package-wall-seconds',str(archive['package_wall_seconds']),'--attempt',attempt],630)
            (RUN/'retention_receipt.json').write_text(json.dumps(saved,ensure_ascii=False),encoding='utf-8')
            print('verified backup; measured package/copy seconds',saved)
        except BaseException as error:
            retention_error=error;status['retention_failure']={'type':type(error).__name__,'reason':str(error)[:2000],'local_zip':str(local)}
            record(status);print('retention failed; keep local results',local)
    if operation_error:raise operation_error
    if retention_error:raise retention_error

def display_saved():
    from IPython.display import display,Image as InlineImage,Markdown
    table=RUN/'inference/comparison.md'
    if table.exists():
        if table.stat().st_size>1024**2:raise ValueError('v151_display_table_limit')
        display(Markdown(table.read_text(encoding='utf-8')))
    total=0
    for number in range(6):
        folder=RUN/'inference'/f'clip-{number:02d}';p=folder/'predictions.json'
        if not p.is_file():continue
        if p.stat().st_size>16*1024**2:raise ValueError('v151_display_json_limit')
        doc=json.loads(p.read_bytes());print(doc['sequence'],'GT-conditioned diagnostic')
        item=next(a for a in doc['artifacts']['files'] if a['name']=='comparison.gif');path=folder/item['name']
        if path.is_symlink() or path.stat().st_size!=item['bytes']:raise ValueError('v151_display_size_path')
        total+=item['bytes']
        if total>POLICY['limits']['result_bytes']:raise ValueError('v151_display_total_limit')
        raw=path.read_bytes()
        if hashlib.sha256(raw).hexdigest()!=item['sha256']:raise ValueError('v151_display_hash')
        print('Separate row axes, actual bounds mm',doc['artifacts']['bounds_per_row_mm'])
        display(InlineImage(data=raw,format='gif'))
'''
    action='''RUN_DIAGNOSTIC=False
if RUN_DIAGNOSTIC:
    run_diagnostic()
    display_saved()
'''
    show='''SHOW_SAVED_RESULTS=False
if SHOW_SAVED_RESULTS:display_saved()
'''
    return {'cells':[_cell('markdown','guide','# v1.5.1 GT 상체 조건 진단\n재학습 없는 frozen 평가입니다. source ZIP과작은진단bundle만 /content에준비합니다. 원본1.55GB·rawRGB·rawcache·F9재추론·optimizer복원은필요없습니다. Body/특징/camera를고정하고 A예측상체/B GT상체의hand local을비교합니다. GT조건은배포성능이아닙니다. 모든RUN기본false. CUDA기본·CPU명시선택·모델명제약없음.\n'),
        _cell('code','bootstrap',code),_cell('markdown','contract','입력검증600초,전체추론/반복/prefix/렌더3600초,결과합/ZIP128MiB,package600초/copyreadback600초별도. disk2GiB/MemAvailable4GiB/RSS4GiB관측guard. 각phase childtimeout+30초,중단SIGINT후30초내미종료시kill/wait. 종료finally보고서자동Drive보관·게시실패no-retry,VM소실에는보장없음. 시간은phase실측wall이며allowance를실측에더하지않습니다. package/copy실측은별도retention receipt에남습니다.\n'),
        _cell('code','run',action),_cell('code','show',show)],'metadata':{'kernelspec':{'display_name':'Python3','language':'python','name':'python3'}},'nbformat':4,'nbformat_minor':5}


def build(args):
    if args.output.exists() or args.source_zip.name==args.bundle_zip.name:raise ValueError('v151_builder_new_output_names')
    with tempfile.TemporaryDirectory(dir=args.output.parent) as temp:
        root=Path(temp);src=extract_pinned_zip(args.source_zip,args.source_sha256,root/'source',max_zip_bytes=64*MIB,
            max_output_bytes=192*MIB,max_members=3000,revision=args.revision)
        inputs=bundle.extract_bundle(args.bundle_zip,args.bundle_sha256,args.bundle_manifest_sha256,root/'inputs')
        if json.loads((root/'source/configs/v151_upper_condition_001.json').read_bytes())!=bundle.POLICY:raise ValueError('v151_builder_policy')
        child=subprocess.run([sys.executable,'-m','rewind_v1.upper_condition_cli','source'],cwd=root/'source',timeout=60,capture_output=True,text=True,check=True)
        pins={'source':{**src,'name':args.source_zip.name,'revision':args.revision},'source_code_sha256':json.loads(child.stdout)['source_sha256'],
            'bundle':{'name':args.bundle_zip.name,'bytes':args.bundle_zip.stat().st_size,'sha256':args.bundle_sha256,'manifest_sha256':args.bundle_manifest_sha256},
            'parent_zip_sha256':inputs['bundle']['parent_zip_sha256']}
    raw=(json.dumps(notebook_document(pins),ensure_ascii=False,indent=1)+'\n').encode();args.output.mkdir();(args.output/NOTEBOOK).write_bytes(raw)
    result={'schema':'v151_upper_handoff_v1','pins':pins,'policy':bundle.POLICY,'notebook':{'name':NOTEBOOK,'bytes':len(raw),'sha256':sha(raw)}}
    (args.output/'handoff_receipt.json').write_bytes(canonical(result));return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for n in ('source-zip','bundle-zip','output'):parser.add_argument('--'+n,type=Path,required=True)
    for n in ('source-sha256','bundle-sha256','bundle-manifest-sha256','revision'):parser.add_argument('--'+n,required=True)
    print(canonical(build(parser.parse_args())).decode())


if __name__=='__main__':main()
