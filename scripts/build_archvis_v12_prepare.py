"""Generate the pinned CPU-only Colab notebook for ArchVis v1.2 preparation."""

import argparse
import json
from pathlib import Path
import subprocess
import sys
import tempfile

from rewind_archive.contracts import canonical
from rewind_prepare.converter import sha
from rewind_prepare.full_archvis import AUDIT_LIMITS, INPUT_ZIP, MIB, REPORT_LIMIT, STAT_LIMITS
from scripts.archvis_pilot_bootstrap import extract_pinned_zip
from scripts.build_archvis_pilot_handoff import _cell


NOTEBOOK = "archvis_v12_prepare.ipynb"
TIMEOUTS = {"preflight": int(STAT_LIMITS.max_seconds) + 30,
            "audit": int(AUDIT_LIMITS.max_seconds) + 30, "recover": 600}


def notebook_document(pins):
    bootstrap = Path(__file__).with_name("archvis_pilot_bootstrap.py").read_text(encoding="utf-8")
    bootstrap += f"\nPINS = {pins!r}\nTIMEOUTS = {TIMEOUTS!r}\n"
    bootstrap += '''
import json
import shutil
import subprocess
import sys
import tempfile
import time

UPLOADS = Path("/content")
BASE = UPLOADS / "archvis_v12_prepare"
BASE.mkdir(exist_ok=True)
if shutil.disk_usage(BASE).free < 1536 * 1024**2:
    raise ValueError("v12_workspace_free_space")
from PIL import Image

VERIFIED = Path(tempfile.mkdtemp(prefix="verified-", dir=BASE))
for key, zipped, unpacked, count in (("source", 64, 192, 3000), ("inputs", 16, 32, 64)):
    pin = PINS[key]
    extract_pinned_zip(UPLOADS / pin["name"], pin["sha256"], VERIFIED / key,
                       max_zip_bytes=zipped * 1024**2, max_output_bytes=unpacked * 1024**2,
                       max_members=count, revision=pin.get("revision"))
SOURCE = VERIFIED / "source"
INPUTS = VERIFIED / "inputs"

def tracked_phase(name, work, arguments):
    status_path = work / "notebook_status.json"
    state = {"phase": name, "status": "started", "timeout_seconds": TIMEOUTS[name],
             "source_revision": PINS["source"]["revision"], "source_zip_sha256": PINS["source"]["sha256"],
             "input_zip_sha256": PINS["inputs"]["sha256"], "snapshot_sha256": PINS["snapshot_sha256"]}
    def save():
        temporary = work / ".notebook_status.partial"
        temporary.write_text(json.dumps(state, ensure_ascii=False), encoding="utf-8")
        temporary.replace(status_path)
    save()
    started = time.monotonic()
    command = [sys.executable, "-m", "rewind_prepare.full_archvis", name, "--work", str(work), *arguments]
    try:
        subprocess.run(command, cwd=SOURCE, timeout=TIMEOUTS[name], check=True)
    except BaseException as exc:
        state.update(status="stopped" if isinstance(exc, (KeyboardInterrupt, subprocess.TimeoutExpired)) else "failed",
                     error=type(exc).__name__, reason=str(exc)[:512])
        raise
    else:
        state["status"] = "complete"
    finally:
        state["elapsed_seconds"] = round(time.monotonic() - started, 3)
        save()
'''
    preflight = '''
RUN_PREFLIGHT = False
MOUNT_DRIVE = False
RGB_ROOT = ""
POSE_ROOT = ""
WORK = None
if RUN_PREFLIGHT:
    if not RGB_ROOT or not POSE_ROOT:
        raise ValueError("v12_rgb_and_pose_roots_required")
    if MOUNT_DRIVE:
        from google.colab import drive
        drive.mount("/content/drive")
    WORK = Path(tempfile.mkdtemp(prefix="attempt-", dir=BASE))
    print(f"결과 경로: {WORK}")
    DATA_ARGS = ["--inputs", str(INPUTS), "--rgb-root", RGB_ROOT, "--pose-root", POSE_ROOT,
                 "--snapshot-sha256", PINS["snapshot_sha256"]]
    tracked_phase("preflight", WORK, DATA_ARGS)
else:
    print("고정 인계와 추출본 경로를 확인한 뒤 목록 검사를 실행합니다.")
'''
    audit = '''
RUN_PAYLOAD_AUDIT = False
SOURCE_PLAN_SHA256 = ""
if RUN_PAYLOAD_AUDIT:
    if WORK is None or not re.fullmatch(r"[0-9a-f]{64}", SOURCE_PLAN_SHA256):
        raise ValueError("v12_reviewed_source_plan_pin_required")
    tracked_phase("audit", WORK, DATA_ARGS + ["--source-plan-sha256", SOURCE_PLAN_SHA256])
else:
    print("목록 검사 보고와 실제 읽기 예산을 확인한 뒤 source_plan_sha256을 입력합니다.")
'''
    recover = '''
RUN_RECOVERY = False
RECOVER_ATTEMPT = ""
if RUN_RECOVERY:
    from google.colab import files
    recovery_work = WORK
    if RECOVER_ATTEMPT:
        if not re.fullmatch(r"attempt-[a-z0-9_]+", RECOVER_ATTEMPT):
            raise ValueError("v12_recovery_attempt_name")
        recovery_work = BASE / RECOVER_ATTEMPT
    if recovery_work is None:
        raise ValueError("v12_recovery_work_required")
    output = next((UPLOADS / f"archvis_v12_prepare-{index:02d}.zip" for index in range(1, 101)
                   if not (UPLOADS / f"archvis_v12_prepare-{index:02d}.zip").exists()), None)
    if output is None:
        raise ValueError("v12_recovery_slots_exhausted")
    command = [sys.executable, "-m", "rewind_prepare.full_archvis", "recover",
               "--work", str(recovery_work), "--output", str(output)]
    subprocess.run(command, cwd=SOURCE, timeout=TIMEOUTS["recover"], check=True)
    files.download(str(output))
'''
    cells = [
        _cell("markdown", "guide", """# ArchVis v1.2 전체 입력 준비
고정 소스와 입력 ZIP을 `/content`에 업로드합니다. CPU와 Pillow가 필요합니다.
목록 검사에서 RGB·pose 추출본 경로를 지정하고 보존 snapshot train 20,888페어와 validation 2,236페어를 대조합니다.
목록 검사는 stat만 실행합니다. 일치하면 보고의 파일 수와 읽기 예상량을 확인하고 `source_plan_sha256`을 실제 파일 감사 셀에 입력합니다.
실제 감사는 선택된 파일을 직접 읽고 PNG·pose·기하 수치·기존 해시를 검사합니다. 상한은 읽기 65 GiB, 목록 검사 15분, 감사 6시간입니다.
로컬 결과와 회수 ZIP에는 최대 512 MiB 예산 및 1 GiB 여유 공간을 둡니다. 준비에 raw cache나 GPU 학습은 포함되지 않습니다.
중단·실패 시 마지막 셀에서 회수 ZIP을 다운로드합니다. 강제 종료 시 가장 최근에 게시한 진행 상태가 남습니다.
실패한 작업에는 이어읽기 지점이 없으므로 결과를 먼저 보관합니다. 검토 이후 새 작업 경로를 사용합니다.
"""),
        _cell("code", "bootstrap", bootstrap),
        _cell("code", "metadata-preflight", preflight),
        _cell("code", "payload-audit", audit),
        _cell("code", "recover-reports", recover),
    ]
    for cell in cells[1:]:
        compile("".join(cell["source"]), cell["id"], "exec")
    return {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python",
                                                        "name": "python3"}},
            "nbformat": 4, "nbformat_minor": 5}


def build(args):
    if args.output.exists() or args.source_zip.name == args.input_zip.name:
        raise ValueError("v12_handoff_output_or_names")
    with tempfile.TemporaryDirectory(dir=args.output.parent) as temp:
        work = Path(temp).absolute()
        source, inputs = work / "source", work / "inputs"
        source_pin = extract_pinned_zip(args.source_zip, args.source_sha256, source,
                                        max_zip_bytes=64*MIB, max_output_bytes=192*MIB,
                                        max_members=3000, revision=args.revision)
        input_pin = extract_pinned_zip(args.input_zip, INPUT_ZIP, inputs, max_zip_bytes=16*MIB,
                                       max_output_bytes=32*MIB, max_members=64)
        command = [sys.executable, "-m", "rewind_prepare.full_archvis", "plan", "--inputs", str(inputs),
                   "--work", str(work / "plan")]
        process = subprocess.run(command, cwd=source, timeout=60, check=True, capture_output=True, text=True)
        result = json.loads(process.stdout)
        pins = {"source": {**source_pin, "name": args.source_zip.name, "revision": args.revision},
                "inputs": {**input_pin, "name": args.input_zip.name},
                "snapshot_sha256": result["snapshot_sha256"], "coverage_sha256": result["coverage_sha256"]}
    raw = (json.dumps(notebook_document(pins), ensure_ascii=False, indent=1) + "\n").encode()
    args.output.mkdir()
    (args.output / NOTEBOOK).write_bytes(raw)
    receipt = {"schema": "archvis_v12_preparation_handoff_v1", "pins": pins, "timeouts": TIMEOUTS,
               "read_cap_bytes": AUDIT_LIMITS.max_read_bytes, "report_cap_bytes": REPORT_LIMIT,
               "notebook": {"name": NOTEBOOK, "sha256": sha(raw), "bytes": len(raw)}}
    (args.output / "handoff_receipt.json").write_bytes(canonical(receipt) + b"\n")
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source-zip", "input-zip", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    for name in ("source-sha256", "revision"):
        parser.add_argument("--" + name, required=True)
    print(canonical(build(parser.parse_args())).decode())


if __name__ == "__main__":
    main()
