"""Build a frozen v1.2 training notebook; all execution and user-cache copy switches default off."""

import argparse
import json
from pathlib import Path
import subprocess
import sys
import tempfile

from rewind_archive.contracts import canonical
from rewind_prepare.converter import sha
from rewind_prepare.full_archvis import INPUT_ZIP
from rewind_v1.full_contract import AUDIT_ZIP_SHA, load_audit
from scripts.archvis_pilot_bootstrap import extract_pinned_zip
from scripts.build_archvis_pilot_handoff import _cell


MIB = 1024**2
CONFIG = "configs/rewind_v12_full_training.json"
NOTEBOOK = "archvis_v12_full_training.ipynb"


def notebook_document(pins):
    bootstrap = Path(__file__).with_name("archvis_pilot_bootstrap.py").read_text(encoding="utf-8")
    bootstrap += f"\nPINS = {pins!r}\n"
    bootstrap += '''
from contextlib import suppress
import json
import shutil
import subprocess
import sys
import tempfile
import time
os.environ["OMP_NUM_THREADS"]="1"
os.environ["MKL_NUM_THREADS"]="1"

UPLOADS = Path("/content")
WORK = Path("/content/archvis_v12_training")
WORK.mkdir(exist_ok=True)
verified = Path(tempfile.mkdtemp(prefix="verified-", dir=WORK))
for key, cap, expanded, count in (("source",64,192,3000),("inputs",16,32,64)):
    pin = PINS[key]
    extract_pinned_zip(UPLOADS/pin["name"],pin["sha256"],verified/key,
                       max_zip_bytes=cap*1024**2,max_output_bytes=expanded*1024**2,
                       max_members=count,revision=pin.get("revision"))
SOURCE, INPUTS = verified/"source", verified/"inputs"
AUDIT = UPLOADS/PINS["audit"]["name"]
with AUDIT.open("rb") as stream:
    if hashlib.file_digest(stream,"sha256").hexdigest()!=PINS["audit"]["sha256"]:
        raise ValueError("v12_audit_zip_pin")
CONFIG = SOURCE/PINS["config_path"]
if hashlib.sha256(CONFIG.read_bytes()).hexdigest()!=PINS["config_sha256"]:
    raise ValueError("v12_config_pin")
source_check=subprocess.run([sys.executable,"-m","rewind_v1.full_cli","source"],cwd=SOURCE,
                            timeout=60,capture_output=True,text=True,check=True)
if json.loads(source_check.stdout)["source_sha256"]!=PINS["source_code_sha256"]:
    raise ValueError("v12_source_code_pin")
CACHE, FEATURES, RUN = WORK/"frames", WORK/"features", WORK/"training"
DRIVE_ROOT = Path("/content/drive/MyDrive")

def mount_drive():
    from google.colab import drive
    drive.mount("/content/drive")
    if not DRIVE_ROOT.parent.is_mount() or not DRIVE_ROOT.is_dir():
        raise ValueError("v12_drive_mount_required")

def common(action):
    return [sys.executable,"-m","rewind_v1.full_cli",action,"--config",str(CONFIG),
            "--config-sha256",PINS["config_sha256"],"--source-sha256",PINS["source_code_sha256"],
            "--inputs",str(INPUTS),"--audit-archive",str(AUDIT)]

def child(action,extra,timeout):
    status={"phase":action,"status":"started","timeout_seconds":timeout}
    path=WORK/("notebook_"+action+"_result.json")
    path.write_text(json.dumps(status,ensure_ascii=False),encoding="utf-8")
    try:
        result=subprocess.run(common(action)+extra,cwd=SOURCE,timeout=timeout,check=True)
    except BaseException as error:
        status.update(status="stopped",type=type(error).__name__,reason=str(error)[:512])
        path.write_text(json.dumps(status,ensure_ascii=False),encoding="utf-8")
        raise
    status.update(status="exited",returncode=result.returncode)
    path.write_text(json.dumps(status,ensure_ascii=False),encoding="utf-8")
    return result
'''
    prepare = '''
RUN_PREPARE = False
RGB_ROOT = ""
POSE_ROOT = ""
MAX_NEW_CHUNKS = 1  # First chunk check. Root-fixed handoff specifies later bounded calls.
if RUN_PREPARE:
    mount_drive()
    if not RGB_ROOT or not POSE_ROOT:
        raise ValueError("v12_rgb_pose_roots_required")
    extra=["--rgb-root",RGB_ROOT,"--pose-root",POSE_ROOT,"--output",str(CACHE),
           "--work",str(WORK),"--drive-root",str(DRIVE_ROOT)]
    if MAX_NEW_CHUNKS is not None:
        extra += ["--max-chunks",str(MAX_NEW_CHUNKS)]
    child("prepare",extra,8460)
'''
    user_cache_copy = '''
# USER ACTION ONLY. The developer agent does not execute this cache upload cell.
USER_COPY_COMPLETED_CACHE = False
if USER_COPY_COMPLETED_CACHE:
    mount_drive()
    sys.path.insert(0,str(SOURCE))
    from rewind_archive.contracts import canonical,digest
    from rewind_v1.full_contract import load_audit,bind_inputs
    from rewind_v1.full_cache import cache_binding,CHUNK
    from rewind_v1.full_cache_retention import pack_chunk,catalog
    manifest,_,snapshot=load_audit(AUDIT)
    _,spec_raw=bind_inputs(INPUTS,manifest,snapshot)
    binding=cache_binding(manifest,spec_raw)
    destination=DRIVE_ROOT/"REWIND-like prototype/reports/cache/v12_prepared"/("input-"+digest(binding))
    destination.mkdir(parents=True,exist_ok=True)
    started=time.monotonic();written=readback=0;packages=[]
    def check_copy():
        if time.monotonic()-started>=7080:
            raise TimeoutError("v12_user_cache_copy_time_limit")
    def measured(path):
        global readback
        checksum=hashlib.sha256()
        with path.open("rb") as stream:
            while block:=stream.read(1024**2):
                check_copy();readback+=len(block)
                if readback>16*1024**3:raise ValueError("v12_user_cache_readback_limit")
                checksum.update(block)
        return checksum.hexdigest()
    def copy_new(local,remote,expected):
        global written
        check_copy()
        if remote.exists():
            if remote.stat().st_size!=local.stat().st_size or measured(remote)!=expected:
                raise ValueError("v12_user_cache_existing_copy_mismatch")
            return
        if written+local.stat().st_size>8*1024**3:
            raise TimeoutError("v12_user_cache_write_limit")
        partial=remote.with_name(remote.name+".partial-"+os.urandom(8).hex())
        with local.open("rb") as source,partial.open("xb") as target:
            while block:=source.read(1024**2):
                check_copy();target.write(block);written+=len(block)
            target.flush();os.fsync(target.fileno())
        if partial.stat().st_size!=local.stat().st_size or measured(partial)!=expected:
            raise ValueError("v12_user_cache_copy_hash")
        if remote.exists():raise FileExistsError("v12_user_cache_no_overwrite")
        partial.rename(remote)
    for number in range((len(manifest["records"])+CHUNK-1)//CHUNK):
        check_copy()
        if not (CACHE/f"chunk-{number:05d}"/"receipt.json").is_file():break
        # Only one temporary <=64MiB package exists, alongside the original cache.
        with tempfile.TemporaryDirectory(dir=WORK,prefix="user-cache-pack-") as directory:
            package=pack_chunk(CACHE,manifest,spec_raw,number,Path(directory)/"chunk.zip")
            name=f"chunk-{number:05d}-{package['sha256']}.zip"
            copy_new(Path(package["output"]),destination/name,package["sha256"])
            packages.append(package)
            doc=catalog(binding,packages)
            raw=canonical(doc)+b"\\n";pin=hashlib.sha256(raw).hexdigest()
            name=f"catalog-{len(packages):05d}-{pin}.json"
            local=Path(directory)/name;local.write_bytes(raw)
            copy_new(local,destination/name,pin)
        print("verified retained prefix",doc["completed_frames"],str(destination/name),pin)
    print("User cache copy bytes",written,"Drive readback bytes",readback)
'''
    cache_restore = '''
RUN_RESTORE_USER_CACHE = False
RETAINED_DIRECTORY = ""
RETAINED_CATALOG = ""
RETAINED_CATALOG_SHA256 = ""
if RUN_RESTORE_USER_CACHE:
    mount_drive()
    child("cache-restore",["--retained",RETAINED_DIRECTORY,"--catalog",RETAINED_CATALOG,
          "--catalog-sha256",RETAINED_CATALOG_SHA256,"--output",str(CACHE)],7330)
'''
    train = '''
RUN_TRAIN = False
RESUME_SHA256 = ""
RESTORE_TRAINING_ARCHIVE = ""
RESTORE_TRAINING_ARCHIVE_SHA256 = ""
STOP_AFTER_UPDATES = 20  # Initial GPU pilot; later fixed calls use the approved value.
if RUN_TRAIN:
    mount_drive()
    if RESTORE_TRAINING_ARCHIVE:
        if RESUME_SHA256:raise ValueError("v12_choose_archive_or_existing_checkpoint")
        restored=subprocess.run([sys.executable,"-m","rewind_v1.full_cli","restore",
            "--archive",RESTORE_TRAINING_ARCHIVE,"--archive-sha256",RESTORE_TRAINING_ARCHIVE_SHA256,
            "--output",str(RUN)],cwd=SOURCE,timeout=630,capture_output=True,text=True,check=True)
        RESUME_SHA256=json.loads(restored.stdout)["last_sha256"]
    extra=["--cache",str(CACHE),"--features",str(FEATURES),"--output",str(RUN),
           "--work",str(WORK),"--drive-root",str(DRIVE_ROOT)]
    if RESUME_SHA256:extra += ["--resume-sha256",RESUME_SHA256]
    if STOP_AFTER_UPDATES is not None:extra += ["--stop-after-updates",str(STOP_AFTER_UPDATES)]
    try:
        child("train",extra,8460)
    except BaseException:
        # A killed child may leave a remote lease. Do not steal it or auto-retry Drive writes.
        recovery=WORK/"recovery";recovery.mkdir(exist_ok=True)
        local=Path(tempfile.mkdtemp(prefix="fallback-",dir=recovery))/"training.zip"
        with suppress(BaseException):
            subprocess.run([sys.executable,"-m","rewind_v1.full_cli","recover","--run",str(RUN),
                            "--output",str(local)],cwd=SOURCE,timeout=630,check=True)
            print("Preserved local fallback",str(local))
        raise
'''
    return {"cells":[
        _cell("markdown", "guide", "# v1.2 ArchVis 전체 train 진단 학습\n고정 감사 결과를 사용합니다. 실제 시간 동기·motion 독립성은 미확인이고 validation은 참고값입니다. 모든 RUN 스위치는 기본 꺼짐입니다. 한 run에는 한 active writer만 사용하고 새 런타임 재개 전에 이전 호출 종료를 확인합니다.\n"),
        _cell("code", "bootstrap", bootstrap),
        _cell("markdown", "prepare-guide", "## 청크 준비\n첫 청크 확인 뒤 root의 고정 예산으로 준비를 이어갑니다. 호출당2시간/원본읽기8GiB이며 전체는 최소7호출입니다. 원본 감사는 반복하지 않습니다.\n"),
        _cell("code", "prepare", prepare),
        _cell("markdown", "cache-copy-guide", "## 사용자가 완료 캐시를 Drive에 보존\n이 셀은 사용자 수행용입니다. 에이전트는 실행하지 않습니다. 준비 호출 뒤 완료 prefix를 보존하고 출력 catalog의 정확한 경로·SHA를 기록합니다. Drive10–12GiB 공간은 별도 확인해야 합니다. 기존 파일은 덮어쓰지 않으며 청크 복사와 readback 검증 후 catalog를 게시합니다.\n"),
        _cell("code", "user-cache-copy", user_cache_copy),
        _cell("markdown", "cache-restore-guide", "## 새 런타임 캐시 복원\n원본 RGB/pose를 재읽지 않고 고정 catalog의 완료 청크만 복원합니다. 부분 catalog는 부분 준비 상태로 남습니다. 복원 호출도2시간/보존 ZIP 읽기8GiB 이내입니다.\n"),
        _cell("code", "cache-restore", cache_restore),
        _cell("markdown", "training-guide", "## GPU pilot·학습 재개\n지원 CUDA GPU를 사용하며10epoch/115,720update 고정 설정을 유지합니다. 첫20update pilot 후 root의 고정 인계를 따릅니다. 로컬25update 저장과 원격60분/호출종료 보관을 구분합니다. 입력 캐시는 모델 ZIP에 포함하지 않습니다. VM 종료에는 finally 실행을 보장하지 않으며 stale lease는 자동 탈취하지 않습니다.\n"),
        _cell("code", "training", train)],
        "metadata":{"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"}},
        "nbformat":4,"nbformat_minor":5}


def build(args):
    if args.output.exists() or len({args.source_zip.name,args.input_zip.name,args.audit_zip.name})!=3:
        raise ValueError("v12_training_builder_output_or_names")
    load_audit(args.audit_zip)
    with tempfile.TemporaryDirectory(dir=args.output.parent) as directory:
        work=Path(directory)
        source,inputs=work/"source",work/"inputs"
        source_pin=extract_pinned_zip(args.source_zip,args.source_sha256,source,max_zip_bytes=64*MIB,
                                     max_output_bytes=192*MIB,max_members=3000,revision=args.revision)
        input_pin=extract_pinned_zip(args.input_zip,INPUT_ZIP,inputs,max_zip_bytes=16*MIB,max_output_bytes=32*MIB,max_members=64)
        config_sha=sha((source/CONFIG).read_bytes())
        child=subprocess.run([sys.executable,"-m","rewind_v1.full_cli","source"],cwd=source,
                             timeout=60,check=True,capture_output=True,text=True)
        code_sha=json.loads(child.stdout)["source_sha256"]
        result=subprocess.run([sys.executable,"-m","rewind_v1.full_cli","plan","--config",str(source/CONFIG),
            "--config-sha256",config_sha,"--source-sha256",code_sha,"--inputs",str(inputs),
            "--audit-archive",str(args.audit_zip.absolute()),"--output",str(work/"plan")],cwd=source,
            timeout=60,check=True,capture_output=True,text=True)
        plan=json.loads(result.stdout)
        pins={"source":{**source_pin,"name":args.source_zip.name,"revision":args.revision},
              "inputs":{**input_pin,"name":args.input_zip.name},
              "audit":{"name":args.audit_zip.name,"sha256":AUDIT_ZIP_SHA},
              "source_code_sha256":code_sha,"config_path":CONFIG,"config_sha256":config_sha,**plan}
    raw=(json.dumps(notebook_document(pins),ensure_ascii=False,indent=1)+"\n").encode()
    args.output.mkdir()
    (args.output/NOTEBOOK).write_bytes(raw)
    receipt={"schema":"archvis_v12_training_handoff_v1","pins":pins,
             "notebook":{"name":NOTEBOOK,"bytes":len(raw),"sha256":sha(raw)}}
    (args.output/"handoff_receipt.json").write_bytes(canonical(receipt)+b"\n")
    return receipt


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ("source-zip","input-zip","audit-zip","output"):
        parser.add_argument("--"+name,type=Path,required=True)
    for name in ("source-sha256","revision"):
        parser.add_argument("--"+name,required=True)
    print(canonical(build(parser.parse_args())).decode())


if __name__=="__main__":main()
