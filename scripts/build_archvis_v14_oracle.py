"""Build the fixed-source Colab handoff for one-clip GT2D/GT-upper overfit diagnostics."""

import argparse
import json
from pathlib import Path
import subprocess
import sys
import tempfile

from rewind_archive.contracts import canonical
from rewind_prepare.converter import sha
from rewind_v1.audit import load_inputs
from rewind_v1.diagnostic import HISTORICAL, validate_bundle
from rewind_v1.oracle_overfit import validate_config
from scripts.archvis_pilot_bootstrap import extract_pinned_zip
from scripts.build_archvis_pilot_handoff import _cell
from scripts.build_archvis_v1_diagnostic import notebook_document as diagnostic_notebook


MIB = 1024**2
CONFIG = "configs/rewind_v14_oracle_overfit.json"


def notebook_document(pins):
    original = diagnostic_notebook(pins)
    bootstrap = "".join(original["cells"][1]["source"])
    for before, after in (('Path("/content/archvis_v1_diagnostic_001")', 'Path("/content/archvis_v14_oracle")'),
                          ('"rewind_v1", "source"', '"rewind_v1.oracle_cli", "source"')):
        if bootstrap.count(before) != 1:
            raise ValueError("v14_bootstrap_contract")
        bootstrap = bootstrap.replace(before, after)
    bootstrap += '''
import shutil

def backup_child(action, timeout, **paths):
    command = [sys.executable, "-m", "scripts.v14_recovery", action]
    for key, value in paths.items():
        command += ["--"+key.replace("_", "-"), str(value)]
    child = subprocess.run(command, cwd=SOURCE, timeout=timeout, check=True, capture_output=True, text=True)
    return json.loads(child.stdout)

def save_backup(attempt, local_zip="", download=False):
    try:
        if not local_zip:
            local_zip = backup_child("package", 630, work=WORK, run=RUN)["local_zip"]
        print(f"로컬 회수 ZIP {local_zip}")
        result = backup_child("publish", 630, work=WORK, local_zip=local_zip, attempt=attempt)
        print(json.dumps(result, ensure_ascii=False))
    except Exception as error:
        failure = WORK / (Path(attempt).name+"-backup-error.json")
        with suppress(OSError), failure.open("x", encoding="utf-8") as stream:
            json.dump({"state":"failed", "type":type(error).__name__, "reason":str(error)[:512],
                       "local_zip":str(local_zip), "drive_attempt":str(attempt)}, stream, ensure_ascii=False)
        print(f"결과 보관 실패. 로컬 ZIP을 유지하고 마지막 셀에서 재시도합니다. {local_zip} {failure}")
        raise
    if download:
        from google.colab import files
        files.download(local_zip)

def prepare_drive():
    from google.colab import drive
    drive.mount("/content/drive")
    root = Path("/content/drive/MyDrive")
    if not root.parent.is_mount() or not root.is_dir():
        raise ValueError("v14_drive_mount_required")
    return backup_child("prepare", 60, drive_root=root)["attempt"]
'''
    launch = '''
RUN_ORACLE_OVERFIT = False
CACHE_SOURCE = "/content/drive/MyDrive/REWIND-like prototype/reports/v1_actual_001/raw_cache"
PREPARED = WORK / "prepared"
PREPARED_SHA256 = ""
RESUME_SHA256 = ""
RESUME_ARCHIVE = ""
RESUME_ARCHIVE_SHA256 = ""
BROWSER_DOWNLOAD = False

if RUN_ORACLE_OVERFIT:
    attempt = prepare_drive()
    try:
        tracked_child("training", [sys.executable, "-m", "rewind_v1.device"], timeout=60, phase="cuda_preflight")
        if shutil.disk_usage(WORK).free < 5*1024**3:
            raise ValueError("v14_local_free_space_5GiB")
        if RESUME_ARCHIVE:
            if not RESUME_ARCHIVE_SHA256 or RESUME_SHA256 or PREPARED_SHA256:
                raise ValueError("v14_restore_archive_and_sha_only")
            restored = tracked_child("training", [sys.executable, "-m", "rewind_v1.oracle_cli", "restore",
                "--archive", RESUME_ARCHIVE, "--archive-sha256", RESUME_ARCHIVE_SHA256,
                "--output", str(WORK / "restored")], timeout=630, phase="oracle_restore", capture_output=True)
            restored = json.loads(restored.stdout)
            PREPARED, RUN = Path(restored["prepared"]), Path(restored["run"])
            PREPARED_SHA256, RESUME_SHA256 = restored["input_sha256"], restored["last_sha256"]
        elif not PREPARED_SHA256:
            prepared = tracked_child("training", [sys.executable, "-m", "rewind_v1.oracle_cli", "prepare",
                "--inputs", str(INPUTS), "--diagnostic-bundle", str(DIAGNOSTIC_BUNDLE),
                "--decision-sha256", PINS["decision_sha256"], "--source-sha256", PINS["source_code_sha256"],
                "--cache", CACHE_SOURCE, "--output", str(PREPARED)], timeout=330, phase="one_clip_prepare", capture_output=True)
            PREPARED_SHA256 = json.loads(prepared.stdout)["content_sha256"]
        command = [sys.executable, "-m", "rewind_v1.oracle_cli", "train", "--prepared", str(PREPARED),
                   "--input-sha256", PREPARED_SHA256, "--config", str(SOURCE / PINS["config_path"]),
                   "--config-sha256", PINS["config_sha256"], "--source-sha256", PINS["source_code_sha256"],
                   "--output", str(RUN)]
        if RESUME_SHA256:
            command += ["--resume-sha256", RESUME_SHA256]
        tracked_child("training", command, timeout=7200, phase="gt2d_gtupper_oracle_overfit")
    finally:
        failed = sys.exc_info()[0] is not None
        try:
            save_backup(attempt, download=BROWSER_DOWNLOAD)
        except Exception as error:
            if not failed:
                raise
            print(f"원래 작업 오류와 보관 오류를 함께 확인합니다. {type(error).__name__} {error}")
'''
    recovery = '''
RUN_RECOVERY = False
RECOVERY_LOCAL_ZIP = ""
BROWSER_DOWNLOAD = False
if RUN_RECOVERY:
    save_backup(prepare_drive(), RECOVERY_LOCAL_ZIP, BROWSER_DOWNLOAD)
'''
    guide = """# v1.4 단일 train 클립 oracle 암기 진단
고정 소스/입력/진단 근거 ZIP을 /content에 올리고 CUDA GPU를 사용합니다. 실제 GPU·VRAM·시간을 기록합니다.
고정 train1,000 중 Adanna Standing_Up__1_ frame0–49만 사용합니다. 기존 raw 캐시의 한 part만 SHA검증 후 복사합니다.
GT2D의 유효 좌표·confidence·측정존재를 입력하며 화면 밖 좌표도 clamp하지 않습니다. Hand에만 GT상체를 의도적으로 입력합니다.
Body/hand 각5,000 optimizer update(총10,000), paired0/200/1000/5000에서 고정noise DDIM10과 별도의GT노이즈k1진단을 저장합니다.
같은클립의 MPJPE·뼈/퍼짐·gradient·평균자세 baseline·처음/중간/끝 예측을 확인합니다. 일반화·논문성능 평가가 아닙니다.
20/10mm 및 shape비율0.8–1.2는 프로젝트 잠정참고기준입니다. 예산 내 미수렴으로 구현버그를 확정하지 않습니다.
준비300초/읽기256MiB, 학습호출2시간(종료여유120초), run2GiB/회수1GiB, 로컬여유5GiB입니다.
호출종료 시 모델·보고서·작은oracle입력텐서를 Drive의 새v14_oracle attempt에 복사하고 크기/SHA를 재읽기합니다. raw RGB캐시는 업로드하지 않습니다.
Drive receipt의 verified/training 및 last_sha256을 확인합니다. 보관실패 시 로컬ZIP으로 수동재시도합니다.
VM 자체 종료 시 finally를 실행하지 못하며, 실행 중 주기적Drive보존을 보장하지 않습니다.
동일런타임재개는 PREPARED_SHA256/RESUME_SHA256, 새런타임은 회수ZIP경로/SHA만 입력합니다. source/config/input/environment 일치가 필요합니다.
"""
    cells = [_cell("markdown", "guide", guide), _cell("code", "bootstrap", bootstrap),
             _cell("code", "oracle-overfit", launch), _cell("code", "recover", recovery)]
    for cell in cells[1:]:
        compile("".join(cell["source"]), cell["id"], "exec")
    return {**original, "cells": cells}


def build(args):
    if args.output.exists() or len({path.name for path in (args.source_zip, args.input_zip, args.evidence_zip)}) != 3:
        raise ValueError("v14_new_handoff_output")
    with tempfile.TemporaryDirectory(dir=args.output.parent) as directory:
        root, pins = Path(directory), {}
        for key, path, pin, zipped, unpacked, members, revision in (
            ("source", args.source_zip, args.source_sha256, 64, 192, 3000, args.revision),
            ("inputs", args.input_zip, args.input_sha256, 16, 32, 64, None),
            ("diagnostic_bundle", args.evidence_zip, args.evidence_sha256, 16, 32, 16, None)):
            result = extract_pinned_zip(path, pin, root / key, max_zip_bytes=zipped*MIB,
                max_output_bytes=unpacked*MIB, max_members=members, revision=revision)
            pins[key] = {**result, "name": path.name}
            if revision:
                pins[key]["revision"] = revision
        if not (root / "source/scripts/v14_recovery.py").is_file():
            raise ValueError("v14_backup_source_missing")
        plan, _ = load_inputs(root / "inputs", HISTORICAL["selection_sha256"])
        validate_bundle(root / "diagnostic_bundle", args.decision_sha256, plan)
        raw = (root / "source" / CONFIG).read_bytes()
        validate_config(json.loads(raw))
        child = subprocess.run([sys.executable, "-m", "rewind_v1.oracle_cli", "source"], cwd=root / "source",
                               timeout=60, check=True, capture_output=True, text=True)
        pins.update(source_code_sha256=json.loads(child.stdout)["source_sha256"], config_path=CONFIG,
                    config_sha256=sha(raw), selection_sha256=HISTORICAL["selection_sha256"],
                    decision_sha256=args.decision_sha256)
    raw = (json.dumps(notebook_document(pins), ensure_ascii=False, indent=1)+"\n").encode()
    args.output.mkdir()
    name = "archvis_v14_oracle_overfit.ipynb"
    (args.output / name).write_bytes(raw)
    receipt = {"schema": "archvis_v14_oracle_handoff_v1", "pins": pins,
               "notebook": {"name": name, "bytes": len(raw), "sha256": sha(raw)}}
    (args.output / "handoff_receipt.json").write_bytes(canonical(receipt)+b"\n")
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source-zip", "input-zip", "evidence-zip", "output"):
        parser.add_argument("--"+name, type=Path, required=True)
    for name in ("source-sha256", "input-sha256", "evidence-sha256", "decision-sha256", "revision"):
        parser.add_argument("--"+name, required=True)
    print(canonical(build(parser.parse_args())).decode())


if __name__ == "__main__":
    main()
