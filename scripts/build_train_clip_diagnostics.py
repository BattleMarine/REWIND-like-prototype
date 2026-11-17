"""Build independently pinned v152/v153 Colab handoffs, with one shared input bundle."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import tempfile

from rewind_archive.contracts import canonical
from rewind_prepare.converter import sha
from rewind_v1 import train_clip_bundle as bundle
from scripts.archvis_pilot_bootstrap import extract_pinned_zip
from scripts.build_archvis_pilot_handoff import _cell


def notebook_document(pins,experiment):
    cfg=bundle.config(experiment)
    code=Path(__file__).with_name('archvis_pilot_bootstrap.py').read_text(encoding='utf-8')
    code+=f'\nPINS={pins!r}\nPOLICY={cfg!r}\nEXPERIMENT={experiment!r}\n'
    code+='''
import json,subprocess,sys,tempfile,uuid,time,signal
os.environ['OMP_NUM_THREADS']='1';os.environ['MKL_NUM_THREADS']='1'
started=time.monotonic()
WORK=Path('/content')/(EXPERIMENT+'_train_clip');WORK.mkdir(exist_ok=True)
SOURCE=Path(tempfile.mkdtemp(prefix='source-',dir=WORK))/'source'
pin=PINS['source']
extract_pinned_zip(Path('/content')/pin['name'],pin['sha256'],SOURCE,max_zip_bytes=64*1024**2,
    max_output_bytes=192*1024**2,max_members=3000,revision=pin['revision'])
FRESH_RUN=WORK/('run-'+uuid.uuid4().hex);FRESH_RUN.mkdir()
RUN=FRESH_RUN
RECOVERY=WORK/'recovery';RECOVERY.mkdir(exist_ok=True)
DRIVE=Path('/content/drive/MyDrive');DEVICE='cuda'

def execute(args,timeout):
    with subprocess.Popen([sys.executable,'-m','rewind_v1.train_clip_cli','--experiment',EXPERIMENT]+args,
            cwd=SOURCE,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True) as child:
        try:stdout,stderr=child.communicate(timeout=timeout)
        except BaseException:
            if child.poll() is None:child.send_signal(signal.SIGINT)
            try:child.communicate(timeout=60)
            except subprocess.TimeoutExpired:child.kill();child.wait(timeout=30)
            raise
    if child.returncode:raise RuntimeError(stderr[-3000:])
    return json.loads(stdout)

if execute(['source'],60)['source_sha256']!=PINS['source_code_sha256']:raise ValueError('clip_source_identity')
BOOTSTRAP_WALL=time.monotonic()-started
print('fresh local run',RUN,'bootstrap seconds',BOOTSTRAP_WALL)

def select_resume(path,sha256):
    global RUN
    candidate=Path(path).resolve()
    if (EXPERIMENT!='v153' or not candidate.is_relative_to(WORK.resolve())
            or not candidate.name.startswith('run-') or not (candidate/'experiment/last.pt').is_file()
            or len(sha256)!=64):raise ValueError('explicit_existing_run_and_last_sha_required')
    RUN=candidate

def record(status):
    p=RUN/'.operation.partial';p.write_text(json.dumps(status,ensure_ascii=False),encoding='utf-8');p.replace(RUN/'operation.json')

def run_experiment(resume_sha=None):
    from google.colab import drive
    drive.mount('/content/drive')
    if not DRIVE.parent.is_mount() or not DRIVE.is_dir():raise ValueError('drive_mount_required')
    attempt=execute(['backup-prepare','--drive-root',str(DRIVE),'--run-id',RUN.name],60)['attempt']
    status={'state':'started','bootstrap_wall_seconds':BOOTSTRAP_WALL,'phase_wall_seconds':{},'source_sha256':PINS['source_code_sha256'],
        'resume':bool(resume_sha),'run':str(RUN)}
    operation_error=retention_error=None;record(status)
    common=['--run',str(RUN),'--source-sha256',PINS['source_code_sha256'],'--manifest-sha256',PINS['bundle']['manifest_sha256']]
    phases=[] if resume_sha else [('prepare',600,['--bundle',str(Path('/content')/PINS['bundle']['name']),'--bundle-sha256',PINS['bundle']['sha256']])]
    phases += [('run',POLICY['limits']['max_seconds'],['--device',DEVICE]+(['--resume-sha256',resume_sha] if resume_sha else []))]
    try:
        for phase,seconds,args in phases:
            status['phase']=phase;record(status);start=time.monotonic()
            try:result=execute([phase]+common+args,seconds+30)
            finally:status['phase_wall_seconds'][phase]=time.monotonic()-start;record(status)
            print(phase,result)
            if phase=='run':status['compute_status']=result['status']
        status['state']='finished'
    except BaseException as error:
        operation_error=error;status.update(state='failed',type=type(error).__name__,reason=str(error)[:2000])
    finally:
        record(status);local=RECOVERY/(RUN.name+'-'+uuid.uuid4().hex+'.zip')
        try:
            archive=execute(['package','--run',str(RUN),'--output',str(local)],630)
            saved=execute(['publish','--archive',archive['archive'],'--sha256',archive['sha256'],'--bytes',str(archive['bytes']),
                '--package-wall-seconds',str(archive['package_wall_seconds']),'--attempt',attempt],630)
            (RUN/'retention_receipt.json').write_text(json.dumps(saved,ensure_ascii=False),encoding='utf-8')
            print('verified backup',saved)
        except BaseException as error:
            retention_error=error;status['retention_failure']={'type':type(error).__name__,'reason':str(error)[:2000],'local_zip':str(local)}
            record(status);print('retention failed; keep local outputs',local)
    if operation_error:raise operation_error
    if retention_error:raise retention_error

def display_saved():
    from IPython.display import display,Image as InlineImage
    p=RUN/'experiment/run.json'
    if not p.is_file():return
    if p.stat().st_size>16*1024**2:raise ValueError('display_report_cap')
    report=json.loads(p.read_bytes());print('status',report['status'],'same train / GT upper diagnostic')
    records=report['evaluations'].values()
    if EXPERIMENT=='v152':records=[r for group in records for r in group.values()]
    total=0
    for entry in records:
        for item in entry.get('artifacts',[]):
            if not item['name'].endswith('.png'):continue
            image_path=RUN/'experiment'/item['name']
            if Path(item['name']).name!=item['name'] or image_path.is_symlink() or image_path.stat().st_size!=item['bytes']:raise ValueError('display_image_path_size')
            total+=item['bytes']
            if total>POLICY['limits']['output_bytes']:raise ValueError('display_total_cap')
            raw=image_path.read_bytes()
            if hashlib.sha256(raw).hexdigest()!=item['sha256']:raise ValueError('display_image_sha')
            print('shared axes bounds mm',entry['shared_axis_bounds']);display(InlineImage(data=raw,format='png'))
'''
    action='''RUN_EXPERIMENT=False
RESUME_EXISTING_RUN=False
EXISTING_RUN_PATH=''  # Same runtime, explicit earlier run; never the new empty FRESH_RUN.
LAST_CHECKPOINT_SHA256=''
if RUN_EXPERIMENT:
    if RESUME_EXISTING_RUN:select_resume(EXISTING_RUN_PATH,LAST_CHECKPOINT_SHA256)
    run_experiment(LAST_CHECKPOINT_SHA256 if RESUME_EXISTING_RUN else None)
    display_saved()
'''
    return {'cells':[_cell('markdown','guide',f'# {cfg["run_id"]}\n고정 첫 train50 / 예측 F9 / 중점 카메라 / GT 상체 진단입니다. 실제 GPU 실행 전이며 모든 RUN 기본 False. source ZIP과 공통 약368MB bundle만 /content에 준비합니다. rawRGB·frontend 재학습·6클립 회귀 반복은 없습니다.\n'),
        _cell('code','bootstrap',code),_cell('markdown','limits',f'입력600초·64member/512MiB. compute {cfg["limits"]["max_seconds"]}초(중단 저장60초 포함), package600초/copy readback600초 별도. disk8GiB/MemAvailable4GiB/RSS4GiB. SIGINT후60초 저장대기·미종료kill/wait30초. 종료finally 새Drive경로 보관·hash검증, 실패시로컬보존·자동재시도없음. VM소실 보장은없습니다. v153만 기존RUN+lastSHA의 동일런타임 재개를지원하며 새VM복원은미검증입니다. allowance와실측wall은구분합니다.\n'),
        _cell('code','run',action),_cell('code','show','SHOW_SAVED_RESULTS=False\nif SHOW_SAVED_RESULTS:display_saved()\n')],
        'metadata':{'kernelspec':{'display_name':'Python3','language':'python','name':'python3'}},'nbformat':4,'nbformat_minor':5}


def build(args):
    if args.output.exists() or args.source_zip.name==args.bundle_zip.name:raise ValueError('clip_builder_new_paths')
    with tempfile.TemporaryDirectory(dir=args.output.parent) as temp:
        root=Path(temp);src=extract_pinned_zip(args.source_zip,args.source_sha256,root/'source',max_zip_bytes=64*1024**2,max_output_bytes=192*1024**2,max_members=3000,revision=args.revision)
        inputs=bundle.extract(args.bundle_zip,args.bundle_sha256,args.bundle_manifest_sha256,root/'inputs')
        if json.loads((root/'source'/bundle.CONFIGS[args.experiment]).read_bytes())!=bundle.config(args.experiment):raise ValueError('clip_builder_config')
        child=subprocess.run([sys.executable,'-m','rewind_v1.train_clip_cli','--experiment',args.experiment,'source'],cwd=root/'source',timeout=60,capture_output=True,text=True,check=True)
        pins={'source':{**src,'name':args.source_zip.name,'revision':args.revision},'source_code_sha256':json.loads(child.stdout)['source_sha256'],
            'bundle':{'name':args.bundle_zip.name,'bytes':args.bundle_zip.stat().st_size,'sha256':args.bundle_sha256,'manifest_sha256':args.bundle_manifest_sha256},'parent_zip_sha256':inputs['bundle']['parent_zip_sha256']}
    raw=(json.dumps(notebook_document(pins,args.experiment),ensure_ascii=False,indent=1)+'\n').encode();args.output.mkdir()
    name=args.experiment+'_train_clip.ipynb';(args.output/name).write_bytes(raw)
    receipt={'schema':'v15_clip_handoff_v1','experiment':args.experiment,'pins':pins,'config':bundle.config(args.experiment),'notebook':{'name':name,'bytes':len(raw),'sha256':sha(raw)}}
    (args.output/'handoff_receipt.json').write_bytes(canonical(receipt));return receipt


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--experiment',choices=('v152','v153'),required=True)
    for n in ('source-zip','bundle-zip','output'):p.add_argument('--'+n,type=Path,required=True)
    for n in ('source-sha256','bundle-sha256','bundle-manifest-sha256','revision'):p.add_argument('--'+n,required=True)
    print(canonical(build(p.parse_args())).decode())


if __name__=='__main__':main()
