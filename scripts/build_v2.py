"""Pinned default-off v2 notebook; one explicitly selected bounded operation."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import tempfile

from rewind_archive.contracts import canonical
from rewind_prepare.converter import sha
from rewind_v1 import v2_contract as contract
from scripts.archvis_pilot_bootstrap import extract_pinned_zip
from scripts.build_archvis_pilot_handoff import _cell


def notebook(pins):
    bootstrap=Path(__file__).with_name('archvis_pilot_bootstrap.py').read_text(encoding='utf-8')
    bootstrap+=f'\nPINS={pins!r}\nPOLICY={contract.config()!r}\n'
    bootstrap+='''
import subprocess,sys,signal,tempfile,time,json
WORK=Path('/content/rewind_v2');WORK.mkdir(exist_ok=True)
SOURCE=Path(tempfile.mkdtemp(prefix='source-',dir=WORK))/'source'
extract_pinned_zip(Path('/content')/PINS['source_name'],PINS['source_sha256'],SOURCE,max_zip_bytes=64*1024**2,max_output_bytes=192*1024**2,max_members=3000,revision=PINS['revision'])
def execute(arguments):
    started=time.monotonic()
    with subprocess.Popen([sys.executable,'-m','rewind_v1.v2_cli']+arguments,cwd=SOURCE) as child:
        try:code=child.wait(timeout=POLICY['external_timeout_seconds'])
        except BaseException:
            if child.poll() is None:child.send_signal(signal.SIGINT)
            try:child.wait(timeout=60)
            except subprocess.TimeoutExpired:child.kill();child.wait(timeout=30)
            raise
    print('invocation measured wall',time.monotonic()-started)
    if code:raise RuntimeError('v2 worker failed; keep local shards/checkpoints/status/ledger')
'''
    run='''RUN_OPERATION=False
ACTION='plan'  # plan/prepare/copy-cache/restore-cache/train/restore-run; exactly one call, no auto retry
USER_EXPLICIT_CACHE_COPY=False
CONFIG_SHA256=''
MANIFEST_PATH=''
MANIFEST_SHA256=''
SPEC_PATH=''
SPEC_SHA256=''
SPLITS_DIRECTORY=''
RGB_ROOT=''
POSE_ROOT=''
CACHE=WORK/'cache'
RUN=WORK/'training'
FEATURES=WORK/'features'
RETAINED_DIRECTORY=''
RETAINED_CATALOG_SHA256=''
LAST_SHA256=''
EXTRA_ARGS=[]  # restore-run archive/path/SHA/bytes or exact result destination
if RUN_OPERATION:
    if ACTION=='copy-cache' and not USER_EXPLICIT_CACHE_COPY:raise ValueError('explicit user cache copy required')
    if ACTION in ('copy-cache','restore-cache','train'):
        from google.colab import drive
        drive.mount('/content/drive')
    args=[ACTION,'--work',str(WORK),'--config-sha256',CONFIG_SHA256,'--source-sha256',PINS['source_identity'],
        '--manifest',MANIFEST_PATH,'--manifest-sha256',MANIFEST_SHA256,'--spec',SPEC_PATH,'--spec-sha256',SPEC_SHA256,
        '--splits',SPLITS_DIRECTORY,'--cache',str(CACHE),'--run',str(RUN),'--features',str(FEATURES)]
    if ACTION=='prepare':args += ['--rgb-root',RGB_ROOT,'--pose-root',POSE_ROOT]
    if ACTION in ('copy-cache','restore-cache'):args += ['--retained',RETAINED_DIRECTORY]
    if ACTION=='restore-cache':args += ['--catalog-sha256',RETAINED_CATALOG_SHA256]
    if ACTION=='train' and LAST_SHA256:args += ['--resume-sha256',LAST_SHA256]
    execute(args+EXTRA_ARGS)
'''
    return {'nbformat':4,'nbformat_minor':5,'metadata':{'kernelspec':{'name':'python3','display_name':'Python3','language':'python'}},
        'cells':[_cell('markdown','guide','# v2 입력·재개 인프라\n모든 실행은 기본OFF. 1회 전체28,800초 안에 종료22분을 예약하므로 계산/중간보존은최대7시간38분. 외부28,920초와SIGINT60초/kill30초. 학습설정미확정시차단합니다. 원격/새VM/GPU수동시험미완료. 캐시Drive보존은사용자가명시실행하며자동주기복사는하지않습니다. 전체누적예산미확정·자동재호출없음.\n'),
        _cell('code','bootstrap',bootstrap),_cell('code','run',run)]}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('source','output'):p.add_argument('--'+n,type=Path,required=True)
    for n in ('source-sha256','revision'):p.add_argument('--'+n,required=True)
    args=p.parse_args()
    if args.output.exists():raise ValueError('v2_new_handoff_output')
    with tempfile.TemporaryDirectory(dir=args.output.parent) as tmp:
        root=Path(tmp)/'source';extract_pinned_zip(args.source,args.source_sha256,root,max_zip_bytes=64*1024**2,max_output_bytes=192*1024**2,max_members=3000,revision=args.revision)
        if json.loads((root/'configs/rewind_v2.json').read_bytes())!=contract.config():raise ValueError('v2_builder_policy_pin')
        child=subprocess.run([sys.executable,'-m','rewind_v1.v2_cli','source'],cwd=root,capture_output=True,text=True,check=True,timeout=60)
        pins={'source_name':args.source.name,'source_sha256':args.source_sha256,'revision':args.revision,'source_identity':json.loads(child.stdout)['source_sha256']}
    raw=(json.dumps(notebook(pins),ensure_ascii=False,indent=1)+'\n').encode();args.output.mkdir();(args.output/'v2.ipynb').write_bytes(raw)
    (args.output/'receipt.json').write_bytes(canonical({'pins':pins,'notebook_sha256':sha(raw),'bytes':len(raw),'status':'infrastructure_only_training_decisions_pending'}))


if __name__=='__main__':main()
