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
from scripts.v12_recovery import SECONDS


NOTEBOOK = "archvis_v12_prepare.ipynb"
TIMEOUTS = {"preflight": int(STAT_LIMITS.max_seconds) + 30,
            "audit": int(AUDIT_LIMITS.max_seconds) + 30, "recover": 600}
BACKUP_TIMEOUTS = {"prepare": 60, "package": SECONDS + 30, "publish": SECONDS + 30}


def notebook_document(pins):
    bootstrap = Path(__file__).with_name("archvis_pilot_bootstrap.py").read_text(encoding="utf-8")
    bootstrap += f"\nPINS = {pins!r}\nTIMEOUTS = {TIMEOUTS!r}\nBACKUP_TIMEOUTS = {BACKUP_TIMEOUTS!r}\n"
    bootstrap += '''
import json
import shutil
import subprocess
import sys
import tempfile
import time
import uuid
from contextlib import suppress

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
RECOVERY_BASE = BASE / "recovery"
RECOVERY_BASE.mkdir(exist_ok=True)
DRIVE_ROOT = Path("/content/drive/MyDrive")

def recovery_child(action, **paths):
    command = [sys.executable, "-m", "scripts.v12_recovery", action]
    for name, value in paths.items():
        command += ["--" + name.replace("_", "-"), str(value)]
    child = subprocess.run(command, cwd=SOURCE, timeout=BACKUP_TIMEOUTS[action], capture_output=True, text=True)
    if child.returncode:
        raise RuntimeError(f"{action}: {child.stderr.strip()[-1500:]}")
    return json.loads(child.stdout)

def backup_context(phase, work):
    if phase not in ("preflight", "audit"):
        raise ValueError("v12_backup_phase")
    context = {"phase": phase, "original_work": str(work.resolve()),
               "source_revision": PINS["source"]["revision"], "source_zip_sha256": PINS["source"]["sha256"],
               "input_zip_sha256": PINS["inputs"]["sha256"], "snapshot_sha256": PINS["snapshot_sha256"],
               "coverage_sha256": PINS["coverage_sha256"]}
    path = work / ("backup_context-" + phase + ".json")
    path.write_text(json.dumps(context), encoding="utf-8")
    return path

def prepare_backup(phase, mount):
    if mount:
        from google.colab import drive
        drive.mount("/content/drive")
    if not DRIVE_ROOT.parent.is_mount() or not DRIVE_ROOT.is_dir():
        raise FileNotFoundError(f"Drive를 먼저 마운트해야 합니다. {DRIVE_ROOT}")
    attempt = recovery_child("prepare", drive_root=DRIVE_ROOT, phase=phase)["attempt"]
    print(f"Drive 결과 폴더 {attempt}")
    return attempt

def save_backup(work, context_path, attempt, local_zip="", browser_download=False):
    try:
        if not local_zip:
            local_zip = recovery_child("package", base=RECOVERY_BASE, work=work, context=context_path)["local_zip"]
        print(f"로컬 회수 ZIP {local_zip}")
        result = recovery_child("publish", base=RECOVERY_BASE, local_zip=local_zip, attempt=attempt, context=context_path)
    except Exception as error:
        failure = work / (Path(attempt).name + "-backup-error.json")
        with suppress(OSError), failure.open("x", encoding="utf-8") as stream:
            json.dump({"state": "failed", "type": type(error).__name__, "reason": str(error)[:512],
                       "local_zip": str(local_zip), "context_path": str(context_path), "drive_attempt": str(attempt)},
                      stream, ensure_ascii=False)
        print(f"회수 또는 Drive 보관 실패. 로컬 ZIP과 실패 영수증을 확인하고 마지막 셀에서 재시도하세요. {local_zip} {failure}")
        raise
    print(json.dumps(result, ensure_ascii=False))
    if browser_download:
        from google.colab import files
        files.download(local_zip)

def tracked_phase(name, work, arguments):
    context_path = backup_context(name, work)
    attempt = prepare_backup(name, MOUNT_DRIVE)
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
    phase_error = None
    try:
        subprocess.run(command, cwd=SOURCE, timeout=TIMEOUTS[name], check=True)
    except BaseException as exc:
        phase_error = exc
        state.update(status="stopped" if isinstance(exc, (KeyboardInterrupt, subprocess.TimeoutExpired)) else "failed",
                     error=type(exc).__name__, reason=str(exc)[:512])
        raise
    else:
        state["status"] = "complete"
    finally:
        state["elapsed_seconds"] = round(time.monotonic() - started, 3)
        status_error = None
        try:
            save()
        except Exception as error:
            status_error = error
            print(f"단계 상태 저장 실패. {type(error).__name__} {error}")
        try:
            save_backup(work, context_path, attempt, browser_download=BROWSER_DOWNLOAD)
        except Exception as error:
            if phase_error is None:
                raise
            print(f"원래 단계 오류와 별도로 보관에도 실패했습니다. {type(error).__name__} {error}")
        if status_error is not None and phase_error is None:
            raise status_error
'''
    preflight = '''
RUN_PREFLIGHT = False
MOUNT_DRIVE = True
BROWSER_DOWNLOAD = False
RGB_ROOT = ""
POSE_ROOT = ""
WORK = None
if RUN_PREFLIGHT:
    if not RGB_ROOT or not POSE_ROOT:
        raise ValueError("v12_rgb_and_pose_roots_required")
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
SOURCE_PLAN_SHA256 = globals().get("RESTORED_SOURCE_PLAN_SHA256", "")
RESUME_AUDIT = bool(globals().get("RESTORED_SOURCE_PLAN_SHA256", ""))
if RUN_PAYLOAD_AUDIT:
    if WORK is None or not re.fullmatch(r"[0-9a-f]{64}", SOURCE_PLAN_SHA256):
        raise ValueError("v12_reviewed_source_plan_pin_required")
    arguments = DATA_ARGS + ["--source-plan-sha256", SOURCE_PLAN_SHA256]
    if RESUME_AUDIT:
        arguments += ["--resume"]
    tracked_phase("audit", WORK, arguments)
else:
    print("목록 검사 보고와 실제 읽기 예산을 확인한 뒤 source_plan_sha256을 입력합니다.")
'''
    restore = '''
RUN_RESTORE = False
RESUME_ARCHIVE = ""
RESUME_ARCHIVE_SHA256 = ""
RESUME_SOURCE_PLAN_SHA256 = ""
if RUN_RESTORE:
    if not RGB_ROOT or not POSE_ROOT or not all(re.fullmatch(r"[0-9a-f]{64}", pin) for pin in
                                               (RESUME_ARCHIVE_SHA256, RESUME_SOURCE_PLAN_SHA256)):
        raise ValueError("v12_restore_roots_and_pins_required")
    if MOUNT_DRIVE:
        from google.colab import drive
        drive.mount("/content/drive")
    restored_work = BASE / ("attempt-" + uuid.uuid4().hex)
    command = [sys.executable, "-m", "rewind_prepare.full_archvis", "restore-audit", "--work", str(restored_work),
               "--inputs", str(INPUTS), "--archive", RESUME_ARCHIVE, "--archive-sha256", RESUME_ARCHIVE_SHA256,
               "--snapshot-sha256", PINS["snapshot_sha256"], "--source-plan-sha256", RESUME_SOURCE_PLAN_SHA256]
    subprocess.run(command, cwd=SOURCE, timeout=TIMEOUTS["recover"]+30, check=True)
    WORK = restored_work
    RESTORED_SOURCE_PLAN_SHA256 = RESUME_SOURCE_PLAN_SHA256
    DATA_ARGS = ["--inputs", str(INPUTS), "--rgb-root", RGB_ROOT, "--pose-root", POSE_ROOT,
                 "--snapshot-sha256", PINS["snapshot_sha256"]]
    print(f"복원 경로 {WORK}. 다음 감사 셀에서 RESUME_AUDIT를 확인합니다.")
'''
    recover = '''
RUN_RECOVERY = False
RECOVER_ATTEMPT = ""
RECOVERY_LOCAL_ZIP = ""
MOUNT_RECOVERY_DRIVE = True
BROWSER_DOWNLOAD = False
if RUN_RECOVERY:
    recovery_work = WORK
    if RECOVER_ATTEMPT:
        if not re.fullmatch(r"attempt-[a-z0-9_]+", RECOVER_ATTEMPT):
            raise ValueError("v12_recovery_attempt_name")
        recovery_work = BASE / RECOVER_ATTEMPT
    if recovery_work is None:
        raise ValueError("v12_recovery_work_required")
    if RECOVERY_LOCAL_ZIP:
        receipt = json.loads((Path(RECOVERY_LOCAL_ZIP).parent / "local_receipt.json").read_text(encoding="utf-8"))
        phase = receipt["context"]["phase"]
    else:
        phase = json.loads((recovery_work / "notebook_status.json").read_text(encoding="utf-8"))["phase"]
    context_path = backup_context(phase, recovery_work)
    save_backup(recovery_work, context_path, prepare_backup(phase, MOUNT_RECOVERY_DRIVE),
                RECOVERY_LOCAL_ZIP, BROWSER_DOWNLOAD)
'''
    cells = [
        _cell("markdown", "guide", """# ArchVis v1.2 전체 입력 준비
고정 소스와 입력 ZIP을 `/content`에 업로드합니다. CPU와 Pillow가 필요합니다.
목록 검사에서 RGB·pose 추출본 경로를 지정하고 보존 snapshot train 20,888페어와 validation 2,236페어를 대조합니다.
목록 검사는 stat만 실행합니다. 일치하면 보고의 파일 수와 읽기 예상량을 확인하고 `source_plan_sha256`을 실제 파일 감사 셀에 입력합니다.
실제 감사는 선택된 파일을 직접 읽고 PNG·pose·기하 수치·기존 해시를 검사합니다. 상한은 읽기 65 GiB, 목록 검사 120분, 감사 6시간입니다.
로컬 결과와 회수 ZIP에는 최대 512 MiB 예산 및 1 GiB 여유 공간을 둡니다. 준비에 raw cache나 GPU 학습은 포함되지 않습니다.
각 검사 실행 전에 Drive 마운트와 새 결과 폴더의 쓰기를 확인합니다. 성공·실패·시간 초과·일반 중단 후 회수 ZIP을 자동 보관합니다.
결과는 Drive의 reports/v1_branch_experiments/v12_recovery 아래 단계별 새 attempt에 저장합니다. 복사본 크기와 SHA를 다시 읽어 확인한 뒤 receipt.json을 기록합니다.
보관 작업의 내부 상한은 각각600초이며 외부 호출 상한은630초입니다. 브라우저 다운로드 기본값은False입니다.
보관에 실패하면 로컬 ZIP을 유지합니다. 마지막 셀에 표시된 ZIP 경로를 넣어 새 attempt로 재시도할 수 있습니다.
VM 자체가 종료되면 finally를 실행할 수 없습니다. 이미 실행 중인 기존 노트북에는 이번 저장 기능이 소급 적용되지 않습니다.
완료 페어는 매번 journal과 head에 확정합니다. 같은 런타임에서는 감사 셀의 RESUME_AUDIT를 True로 지정해 이어갑니다.
새 런타임은 복원 셀에 회수 ZIP·ZIP SHA·기존 source_plan SHA를 입력합니다. journal이 없는 구버전 카운트 전용 ZIP은 재개할 수 없습니다.
재개는 완료 원본의 path·size·mtime_ns를 대조하고 미완료 페어부터 내용을 읽습니다. 완료 SHA는 과거 내용 검사 당시의 증거입니다.
같은 크기와 mtime을 보존한 내용 변경까지 메타데이터로 검출하지는 못합니다. 마지막 전체 메타데이터 대조도6시간 상한에 포함됩니다.
VM 강제 종료 전 로컬에만 확정된 기록은 유실될 수 있습니다. 재개는 사용자가 다음 호출을 실행하며 무한 자동 반복하지 않습니다.
"""),
        _cell("code", "bootstrap", bootstrap),
        _cell("code", "metadata-preflight", preflight),
        _cell("code", "restore-audit", restore),
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
        if not (source / "scripts/v12_recovery.py").is_file():
            raise ValueError("v12_backup_source_missing")
        if not (source / "rewind_prepare/full_audit_journal.py").is_file():
            raise ValueError("v12_audit_resume_source_missing")
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
               "backup_timeouts": BACKUP_TIMEOUTS,
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
