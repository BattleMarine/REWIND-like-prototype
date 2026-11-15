"""Frozen-source v1.5 notebook with explicit stages and automatic bounded result retention."""

import argparse
import json
from pathlib import Path
import subprocess
import sys
import tempfile

from rewind_archive.contracts import canonical
from rewind_prepare.converter import sha
from rewind_prepare.full_archvis import INPUT_ZIP
from rewind_v1 import camera_training as train,camera_gauge as gauge
from rewind_v1.audit import load_inputs
from rewind_v1.diagnostic import HISTORICAL,validate_bundle
from scripts.archvis_pilot_bootstrap import extract_pinned_zip
from scripts.build_archvis_pilot_handoff import _cell


NOTEBOOK='v15_camera_training.ipynb';MIB=1024**2


def notebook_document(pins):
    bootstrap=Path(__file__).with_name('archvis_pilot_bootstrap.py').read_text(encoding='utf-8')
    bootstrap+=f'\nPINS={pins!r}\nPOLICY={train.POLICY!r}\nGAUGE={gauge.POLICY!r}\n'
    bootstrap+='''
import json,subprocess,sys,uuid,signal,tempfile
os.environ['OMP_NUM_THREADS']='1';os.environ['MKL_NUM_THREADS']='1'
WORK=Path('/content/v15_camera_training');WORK.mkdir(exist_ok=True)
VERIFIED=Path(tempfile.mkdtemp(prefix='verified-',dir=WORK))
for key,cap,expanded,count in [('source',64,192,3000),('inputs',16,32,64),('evidence',16,32,16)]:
    pin=PINS[key]
    extract_pinned_zip(Path('/content')/pin['name'],pin['sha256'],VERIFIED/key,max_zip_bytes=cap*1024**2,
        max_output_bytes=expanded*1024**2,max_members=count,revision=pin.get('revision'))
SOURCE,INPUTS,EVIDENCE=VERIFIED/'source',VERIFIED/'inputs',VERIFIED/'evidence'
RUN=WORK/('run-'+uuid.uuid4().hex);RUN.mkdir()
RECOVERY=WORK/'recovery';RECOVERY.mkdir(exist_ok=True)
DRIVE=Path('/content/drive/MyDrive')
MODEL_LOCAL=WORK/'v12_fixed_model.zip'

def execute(args,timeout):
    with subprocess.Popen([sys.executable,'-m','rewind_v1.camera_training_cli']+args,cwd=SOURCE,
            stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True) as child:
        try:stdout,stderr=child.communicate(timeout=timeout)
        except BaseException:
            child.send_signal(signal.SIGINT)
            try:child.communicate(timeout=60)
            except subprocess.TimeoutExpired:child.kill();child.wait(timeout=30)
            raise
    if child.returncode:raise RuntimeError(stderr[-3000:])
    return json.loads(stdout)

if execute(['source'],60)['source_sha256']!=PINS['source_code_sha256']:raise ValueError('v15_source_pin')

def stage(name,seconds,resume=''):
    from google.colab import drive
    drive.mount('/content/drive')
    if not DRIVE.parent.is_mount() or not DRIVE.is_dir():raise ValueError('v15_mount_required')
    attempt=execute(['backup-prepare','--drive-root',str(DRIVE),'--run-id',RUN.name],60)['attempt']
    operation_error=retention_error=None;result=None
    status={'stage':name,'state':'started','run':str(RUN)}
    (RUN/'operation.json').write_text(json.dumps(status))
    try:
        args=['run','--stage',name,'--work',str(WORK),'--run',str(RUN),'--inputs',str(INPUTS),'--evidence',str(EVIDENCE),
            '--cache',str(DRIVE/POLICY['raw_drive_relative']),'--model-source',str(DRIVE/GAUGE['model_drive_relative']),
            '--model-local',str(MODEL_LOCAL),'--reference',str(Path('/content')/PINS['reference']['name']),
            '--control-reference',str(Path('/content')/PINS['control_reference']['name']),
            '--rgb-root',str(DRIVE/GAUGE['rgb_drive_relative']),'--pose-root',str(DRIVE/GAUGE['pose_drive_relative']),
            '--device',DEVICE,'--source-sha256',PINS['source_code_sha256']]
        if resume:args+=['--resume-sha256',resume]
        result=execute(args,seconds+60);status.update(state='exited',result=result)
    except BaseException as error:
        operation_error=error;status.update(state='failed',type=type(error).__name__,reason=str(error)[:2000])
    finally:
        (RUN/'operation.json').write_text(json.dumps(status));local=RECOVERY/(RUN.name+'-'+uuid.uuid4().hex+'.zip')
        try:
            archive=execute(['package','--run',str(RUN),'--output',str(local)],630)
            saved=execute(['publish','--archive',archive['archive'],'--sha256',archive['sha256'],'--bytes',str(archive['bytes']),'--attempt',attempt],630)
            print('verified backup',saved)
        except BaseException as error:
            retention_error=error;print('retention failed; keep local',local,type(error).__name__,str(error))
    if operation_error:raise operation_error
    if retention_error:raise retention_error
    print(result);return result
'''
    prepare="""RUN_PREPARE=False
DEVICE='cuda'  # Any supported CUDA GPU; exact environment required on resume.
if RUN_PREPARE:
    for name,seconds in [('model',600),('references',600),('raw',3600),('features',3600),('eval-prepare',3600)]:stage(name,seconds)
"""
    fit="""RUN_TRAIN=False
RESUME_CHECKPOINT_SHA256=''
if RUN_TRAIN:
    if not RESUME_CHECKPOINT_SHA256:stage('pilot',600)
    result=stage('train',POLICY['limits']['train_seconds'],RESUME_CHECKPOINT_SHA256)
    print('Training status',result['status'],'last pin',result['last'])
"""
    evaluate="""RUN_EVALUATE=False
FINAL_CHECKPOINT_SHA256=''
if RUN_EVALUATE:
    if not FINAL_CHECKPOINT_SHA256:raise ValueError('v15_fixed_final_pin_required')
    stage('evaluate',3600,FINAL_CHECKPOINT_SHA256)
    from IPython.display import display,Image as InlineImage
    for number in range(6):
        folder=RUN/'evaluation'/f'clip-{number:02d}';doc=json.loads((folder/'report.json').read_bytes())
        item=next(a for a in doc['artifacts'] if a['name']=='comparison.gif');raw=(folder/'comparison.gif').read_bytes()
        if len(raw)!=item['bytes'] or hashlib.sha256(raw).hexdigest()!=item['sha256']:raise ValueError('v15_display_hash')
        display(InlineImage(data=raw,format='gif'))
"""
    restore="""RUN_RESTORE=False
RESTORE_ZIP=''
RESTORE_ZIP_SHA256=''
if RUN_RESTORE:
    if any(RUN.iterdir()):raise ValueError('v15_restore_requires_fresh_run')
    RUN.rmdir()
    print(execute(['restore','--archive',RESTORE_ZIP,'--sha256',RESTORE_ZIP_SHA256,'--output',str(RUN)],630))
    print('Restored derived features and checkpoints. Set exact last SHA for RUN_TRAIN; no automatic retry.')
"""
    eval_restore="""RUN_RESTORE_EVAL_RAW=False
if RUN_RESTORE_EVAL_RAW:
    # A new VM restores derived features, then reobtains only pinned RGB/pose for GIFs.
    for name,seconds in [('model',600),('references',600),('eval-prepare',3600)]:stage(name,seconds)
"""
    cells=[_cell('markdown','guide','# V1.5 카메라 표현 통제 학습\nF9는20888train사전학습,두군B/H는동일1000train·각1000update새초기화입니다. 제한예산진단이며독립일반화·full수렴증명아닙니다. 모든RUN기본false. 전용계약 docs/V15_CAMERA_TRAINING.md를따릅니다. 준비/실장치pilot/본학습/최종평가를분리하며실패자동반복없음. 각단계finally자동Drive보관(package600/copy600),VM소실에는보장없음.\n'),
        _cell('code','bootstrap',bootstrap),_cell('code','restore',restore),_cell('code','prepare',prepare),_cell('code','train',fit),
        _cell('code','restore-eval-raw',eval_restore),_cell('code','evaluate',evaluate)]
    return {'cells':cells,'metadata':{'kernelspec':{'display_name':'Python3','language':'python','name':'python3'}},'nbformat':4,'nbformat_minor':5}


def build(args):
    if args.output.exists():raise ValueError('v15_new_builder_output')
    with tempfile.TemporaryDirectory(dir=args.output.parent) as tmp:
        root=Path(tmp);pins={}
        for key,path,expected,cap,expanded,count,rev in (
                ('source',args.source_zip,args.source_sha256,64,192,3000,args.revision),('inputs',args.input_zip,INPUT_ZIP,16,32,64,None),
                ('evidence',args.evidence_zip,train.POLICY['evidence_zip_sha256'],16,32,16,None)):
            p=extract_pinned_zip(path,expected,root/key,max_zip_bytes=cap*MIB,max_output_bytes=expanded*MIB,max_members=count,revision=rev)
            pins[key]={**p,'name':path.name,**({'revision':rev} if rev else {})}
        plan,_=load_inputs(root/'inputs',HISTORICAL['selection_sha256']);validate_bundle(root/'evidence',train.POLICY['decision_sha256'],plan)
        gauge.load_references(args.reference_zip,args.control_reference_zip,root/'reference')
        if len({p['name'] for p in pins.values()}|{args.reference_zip.name,args.control_reference_zip.name})!=5:raise ValueError('v15_unique_input_names')
        for key,path,name in (('reference',args.reference_zip,'city'),('control_reference',args.control_reference_zip,'control')):
            pins[key]={'name':path.name,**gauge.POLICY['references'][name]}
        if json.loads((root/'source/configs/rewind_v15_camera_training.json').read_bytes())!=train.POLICY:raise ValueError('v15_builder_config')
        child=subprocess.run([sys.executable,'-m','rewind_v1.camera_training_cli','source'],cwd=root/'source',timeout=60,capture_output=True,text=True,check=True)
        pins['source_code_sha256']=json.loads(child.stdout)['source_sha256']
    raw=(json.dumps(notebook_document(pins),ensure_ascii=False,indent=1)+'\n').encode();args.output.mkdir();(args.output/NOTEBOOK).write_bytes(raw)
    receipt={'schema':'v15_camera_training_handoff_v1','pins':pins,'policy':train.POLICY,'notebook':{'name':NOTEBOOK,'bytes':len(raw),'sha256':sha(raw)}}
    (args.output/'handoff_receipt.json').write_bytes(canonical(receipt));return receipt


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for n in ('source-zip','input-zip','evidence-zip','reference-zip','control-reference-zip','output'):parser.add_argument('--'+n,type=Path,required=True)
    parser.add_argument('--source-sha256',required=True);parser.add_argument('--revision',required=True)
    print(canonical(build(parser.parse_args())).decode())


if __name__=='__main__':main()
