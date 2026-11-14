"""Fixed-source, default-off Colab handoff for validation-only continuous cross-scenario evaluation."""

import argparse
import json
from pathlib import Path
import subprocess
import sys
import tempfile

from rewind_archive.contracts import canonical
from rewind_prepare.converter import sha
from rewind_prepare.full_archvis import INPUT_ZIP,frozen_snapshot
from rewind_v1.cross_data import POLICY,LIMITS
from scripts.archvis_pilot_bootstrap import extract_pinned_zip
from scripts.build_archvis_pilot_handoff import _cell


MIB=1024**2
NOTEBOOK="v12_cross_scenario_001.ipynb"


def notebook_document(pins):
    bootstrap=Path(__file__).with_name("archvis_pilot_bootstrap.py").read_text(encoding="utf-8")
    bootstrap+=f"\nPINS={pins!r}\nPOLICY={POLICY!r}\nLIMITS={LIMITS!r}\n"
    bootstrap+='''
import json
import subprocess
import sys
import tempfile
import uuid
os.environ["OMP_NUM_THREADS"]="1"
os.environ["MKL_NUM_THREADS"]="1"
UPLOADS=Path("/content")
WORK=Path("/content/v12_cross_scenario_001");WORK.mkdir(exist_ok=True)
VERIFIED=Path(tempfile.mkdtemp(prefix="verified-",dir=WORK))
for key,cap,expanded,count in (("source",64,192,3000),("inputs",16,32,64)):
    pin=PINS[key]
    extract_pinned_zip(UPLOADS/pin["name"],pin["sha256"],VERIFIED/key,max_zip_bytes=cap*1024**2,
                      max_output_bytes=expanded*1024**2,max_members=count,revision=pin.get("revision"))
SOURCE,INPUTS=VERIFIED/"source",VERIFIED/"inputs"
RUN=WORK/("run-"+uuid.uuid4().hex);RUN.mkdir()
RECOVERY=WORK/"recovery";RECOVERY.mkdir(exist_ok=True)
DRIVE_ROOT=Path("/content/drive/MyDrive")
MODEL_SOURCE=DRIVE_ROOT/POLICY["model_drive_relative"]
MODEL_LOCAL=WORK/("model-"+POLICY["model_sha256"]+".zip")
RGB_ROOT=DRIVE_ROOT/POLICY["rgb_drive_relative"]
POSE_ROOT=DRIVE_ROOT/POLICY["pose_drive_relative"]

def execute(arguments,timeout):
    command=[sys.executable,"-m","rewind_v1.cross_cli"]+arguments
    with subprocess.Popen(command,cwd=SOURCE,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True) as process:
        try:stdout,stderr=process.communicate(timeout=timeout)
        except BaseException:
            process.kill();process.wait(timeout=30)
            print("terminated child",process.pid,"exit",process.returncode)
            raise
    if process.returncode:raise RuntimeError(stderr[-2000:])
    return json.loads(stdout)

if execute(["source"],60)["source_sha256"]!=PINS["source_code_sha256"]:
    raise ValueError("cross_bootstrap_source_pin")
print("fresh run",str(RUN))

def record(doc):
    temporary=RUN/".notebook_status.json.partial"
    temporary.write_text(json.dumps(doc,ensure_ascii=False),encoding="utf-8")
    temporary.replace(RUN/"notebook_status.json")

def phase(name,arguments,timeout,copy_model=False):
    from google.colab import drive
    drive.mount("/content/drive")
    if not DRIVE_ROOT.parent.is_mount() or not DRIVE_ROOT.is_dir():raise ValueError("cross_drive_mount_required")
    execute(["preflight","--work",str(WORK)],60)
    attempt=execute(["backup-prepare","--drive-root",str(DRIVE_ROOT),"--run-id",RUN.name,"--phase",name],60)["attempt"]
    status={"phase":name,"status":"started","source_sha256":PINS["source_code_sha256"],"run":str(RUN)}
    record(status);operation_error=backup_error=None;result=None
    try:
        if copy_model:
            execute(["model-copy","--source",str(MODEL_SOURCE),"--output",str(MODEL_LOCAL),
                     "--status",str(RUN/"model_status.json")],630)
        result=execute(arguments,timeout)
        status.update(status="exited",result=result)
    except BaseException as error:
        operation_error=error;status.update(status="failed",type=type(error).__name__,reason=str(error)[:2000])
    finally:
        record(status)
        local=RECOVERY/(RUN.name+"-"+name+"-"+uuid.uuid4().hex+".zip")
        try:
            archive=execute(["package","--run",str(RUN),"--output",str(local)],630)
            saved=execute(["publish","--archive",archive["archive"],"--sha256",archive["sha256"],
                           "--bytes",str(archive["bytes"]),"--attempt",attempt],630)
            print("verified backup",json.dumps(saved,ensure_ascii=False))
        except BaseException as error:
            backup_error=error
            status["retention_failure"]={"type":type(error).__name__,"reason":str(error)[:2000],"local_zip":str(local)}
            record(status);print("backup failed; keep local results",str(local))
    if operation_error is not None:raise operation_error
    if backup_error is not None:raise backup_error
    print(json.dumps(result,ensure_ascii=False));return result

def base(command):
    return [command,"--inputs",str(INPUTS),"--source-sha256",PINS["source_code_sha256"]]
'''
    selection='''
RUN_SELECTION=False
if RUN_SELECTION:
    phase("selection",base("select")+["--rgb-root",str(RGB_ROOT),"--pose-root",str(POSE_ROOT),
        "--output",str(RUN/"selection")],LIMITS["metadata_seconds"]+30)
'''
    preparation='''
RUN_PREPARE=False
SELECTION_SHA256=""  # Record the completed preselection seal before this phase.
if RUN_PREPARE:
    if not SELECTION_SHA256:raise ValueError("cross_record_selection_sha_first")
    phase("prepared",base("prepare")+["--selection",str(RUN/"selection/selection.json"),
        "--selection-sha256",SELECTION_SHA256,"--output",str(RUN/"prepared")],LIMITS["prepare_seconds"]+30)
'''
    inference='''
RUN_INFERENCE=False
PREPARED_SHA256=""
DEVICE="cpu"  # Explicit cpu or cuda. No automatic fallback or A100 requirement.
if RUN_INFERENCE:
    if not SELECTION_SHA256 or not PREPARED_SHA256:raise ValueError("cross_record_input_pins_first")
    phase("inference",base("infer")+["--selection",str(RUN/"selection/selection.json"),
        "--selection-sha256",SELECTION_SHA256,"--prepared",str(RUN/"prepared"),
        "--prepared-sha256",PREPARED_SHA256,"--model",str(MODEL_LOCAL),"--device",DEVICE,
        "--output",str(RUN/"inference")],LIMITS["infer_seconds"]+30,copy_model=True)
'''
    display='''
SHOW_EXISTING_RESULTS=False
SHOW_KEYFRAMES=False
if SHOW_EXISTING_RESULTS:
    from IPython.display import display,Image as InlineImage
    shown_bytes=0
    for number in range(4):
        folder=RUN/"inference"/f"clip-{number:02d}"
        info=folder/"predictions.json"
        if not info.is_file():continue
        if info.stat().st_size>4*1024**2:raise ValueError("cross_display_json_limit")
        clip=json.loads(info.read_bytes())
        print(clip["environment"],clip["sequence"],"validation / nominal25FPS")
        for item in clip["artifacts"]:
            name=item["name"]
            if name!="clip.gif" and not (SHOW_KEYFRAMES and re.fullmatch(r"frame-\\d{9}\\.png",name)):continue
            path=folder/name
            if path.is_symlink() or not path.resolve().is_relative_to(RUN.resolve()):raise ValueError("cross_display_path")
            if path.stat().st_size!=item["bytes"]:raise ValueError("cross_display_size")
            shown_bytes+=item["bytes"]
            if shown_bytes>LIMITS["result_bytes"]:raise ValueError("cross_display_total_limit")
            content=path.read_bytes()
            if hashlib.sha256(content).hexdigest()!=item["sha256"]:raise ValueError("cross_display_hash")
            display(InlineImage(data=content,format="gif" if name=="clip.gif" else "png"))
'''
    return {"cells":[
        _cell("markdown","guide","# V12-CROSS-SCENARIO-001\n재학습 없는 고정 v1.2 평가입니다. v1.3 혼합학습·v1.5 실행이 아닙니다. requirements-v12.txt의 기존 CUDA torch/Pillow/NumPy 환경을 먼저 확인합니다. 공식 validation만 가용성으로 사전 선택하며 test/train fallback은 없습니다. 모든 실행 스위치는 기본 꺼짐입니다.\n"),
        _cell("code","bootstrap",bootstrap),
        _cell("markdown","selection-guide","## 1. 목록과 선정\n두 시나리오별 최대2개 서로 다른 sequence의 첫 연속50frame을 선택합니다. metadata30분/100kentries/32MiB. 1개면 partial,0개면 실패입니다. 출력 selection SHA와 skip 근거를 기록합니다.\n"),
        _cell("code","select",selection),
        _cell("markdown","prepare-guide","## 2. 파일 준비·검증\n고정 목록만 최대200pair·원본2GiB·60분. SHA는 관측 파일 identity이며 공식 전체 원본 무결성 주장이 아닙니다. Decode/62관절·camera/단위·projection·mask 오류는 다른 후보로 교체하지 않고 중단합니다. raw와 모델 ZIP은 결과 보관에서 제외됩니다.\n"),
        _cell("code","prepare",preparation),
        _cell("markdown","infer-guide","## 3. 연속클립 추론\nCPU/CUDA 명시·FP32·DDIM10·seed71 offsets100/200,60분. 모델 copy/hash600초/1GiB는 별도입니다. 명목25FPS·stereo/motion 독립성 미검증·이름 overlap proxy를 기록합니다. 각 phase 종료 package600초+copy/readback600초가 별도로 추가될 수 있으며 VM 종료에는 finally를 보장하지 않습니다. 결과파일 합/ZIP각256MiB,초기free disk8GiB/available RAM4GiB,RSS4GiB 관측guard입니다. 자동 재호출·OOM fallback·데이터 업로드는 없습니다.\n"),
        _cell("code","infer",inference),
        _cell("markdown","display-guide","## 4. 저장된 결과 보기\nSHOW_EXISTING_RESULTS만 켜면 최대4개 기존 GIF를 표시합니다. SHOW_KEYFRAMES는 각클립3PNG도 표시합니다. 새 추론·모델 로드·업로드는 실행하지 않습니다.\n"),
        _cell("code","display",display)],
        "metadata":{"kernelspec":{"display_name":"Python3","language":"python","name":"python3"}},"nbformat":4,"nbformat_minor":5}


def build(args):
    if args.output.exists() or args.source_zip.name==args.input_zip.name:raise ValueError("cross_builder_new_output")
    with tempfile.TemporaryDirectory(dir=args.output.parent) as directory:
        root=Path(directory);source=root/"source";inputs=root/"inputs"
        source_pin=extract_pinned_zip(args.source_zip,args.source_sha256,source,max_zip_bytes=64*MIB,max_output_bytes=192*MIB,max_members=3000,revision=args.revision)
        input_pin=extract_pinned_zip(args.input_zip,INPUT_ZIP,inputs,max_zip_bytes=16*MIB,max_output_bytes=32*MIB,max_members=64)
        if json.loads((source/"configs/v12_cross_scenario_001.json").read_bytes())!=POLICY:
            raise ValueError("cross_builder_policy_mismatch")
        frozen_snapshot(inputs)
        child=subprocess.run([sys.executable,"-m","rewind_v1.cross_cli","source"],cwd=source,timeout=60,capture_output=True,text=True,check=True)
        pins={"source":{**source_pin,"name":args.source_zip.name,"revision":args.revision},
              "inputs":{**input_pin,"name":args.input_zip.name},"source_code_sha256":json.loads(child.stdout)["source_sha256"],
              "model_sha256":POLICY["model_sha256"],"model_bytes":POLICY["model_bytes"]}
    raw=(json.dumps(notebook_document(pins),ensure_ascii=False,indent=1)+"\n").encode();args.output.mkdir()
    (args.output/NOTEBOOK).write_bytes(raw)
    result={"schema":"v12_cross_handoff_v1","pins":pins,"policy":POLICY,
            "notebook":{"name":NOTEBOOK,"bytes":len(raw),"sha256":sha(raw)}}
    (args.output/"handoff_receipt.json").write_bytes(canonical(result)+b"\n");return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ("source-zip","input-zip","output"):parser.add_argument("--"+name,type=Path,required=True)
    parser.add_argument("--source-sha256",required=True);parser.add_argument("--revision",required=True)
    print(canonical(build(parser.parse_args())).decode())


if __name__=="__main__":main()
