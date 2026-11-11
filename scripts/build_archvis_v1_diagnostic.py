"""Build the frozen V1-DIAGNOSTIC-001 Colab handoff."""

import argparse
import json
from pathlib import Path
import subprocess
import sys
import tempfile

from rewind_archive.contracts import canonical
from rewind_m1.local_rgb import _hash
from rewind_prepare.converter import sha
from rewind_v1.audit import load_inputs
from rewind_v1.diagnostic import HISTORICAL, validate_bundle, validate_pilot
from rewind_v1.expansion import HARD_SECONDS, RUN_ID, expansion_decision, validate_expansion
from rewind_v1.runner import validate_config
from scripts.archvis_pilot_bootstrap import extract_pinned_zip
from scripts.build_archvis_pilot_handoff import _cell
from scripts.build_archvis_v1_handoff import CONFIG, MIB, notebook_document as audit_notebook
from scripts.v11_recovery import COPY_SECONDS, PACKAGE_SECONDS


EXPANSION_CONFIG = "configs/rewind_v1_epoch_expansion.json"


def _expansion_cells():
    helpers = f"\nV11_PACKAGE_SECONDS = {PACKAGE_SECONDS + 30}\nV11_COPY_SECONDS = {COPY_SECONDS + 30}\n"
    helpers += '''
DRIVE_ROOT = Path("/content/drive/MyDrive")

def recovery_child(action, timeout, **paths):
    command = [sys.executable, "-m", "scripts.v11_recovery", action]
    for name, value in paths.items():
        command += ["--" + name.replace("_", "-"), str(value)]
    child = subprocess.run(command, cwd=SOURCE, timeout=timeout, capture_output=True, text=True)
    if child.returncode:
        raise RuntimeError(f"{action}: {child.stderr.strip()[-1500:]}")
    return json.loads(child.stdout)

def prepare_recovery(mount):
    if mount:
        from google.colab import drive
        drive.mount("/content/drive")
    if not DRIVE_ROOT.parent.is_mount() or not DRIVE_ROOT.is_dir():
        raise FileNotFoundError(f"Drive를 먼저 마운트해야 합니다. {DRIVE_ROOT}")
    attempt = recovery_child("prepare", 60, drive_root=DRIVE_ROOT)["attempt"]
    print(f"Drive 결과 폴더 {attempt}")
    return attempt

def save_recovery(attempt, local_zip="", browser_download=False):
    if not local_zip:
        local_zip = recovery_child("package", V11_PACKAGE_SECONDS, work=WORK, run=RUN)["local_zip"]
    local_parent = Path(local_zip).resolve().parent
    if not local_parent.is_relative_to((WORK / "recovery").resolve()):
        raise ValueError("현재 런타임의 회수 폴더에서 로컬 ZIP을 선택해야 합니다.")
    print(f"로컬 회수 ZIP {local_zip}")
    try:
        result = recovery_child("publish", V11_COPY_SECONDS, work=WORK, local_zip=local_zip, attempt=attempt)
    except Exception as error:
        failure = local_parent / (Path(attempt).name + "-notebook-error.json")
        with suppress(OSError), failure.open("x", encoding="utf-8") as stream:
            json.dump({"state": "failed", "type": type(error).__name__, "reason": str(error)[:512],
                       "local_zip": str(local_zip), "drive_attempt": str(attempt)}, stream, ensure_ascii=False)
        print(f"Drive 복사 또는 검증 실패. 보존된 로컬 ZIP으로 마지막 셀에서 재시도하세요. {local_zip}")
        raise
    print(json.dumps(result, ensure_ascii=False))
    if browser_download:
        from google.colab import files
        files.download(local_zip)
'''
    launch = '''
RUN_EPOCH_EXPANSION = False
MOUNT_DRIVE = True
BROWSER_DOWNLOAD = False
CACHE_SOURCE = "/content/drive/MyDrive/REWIND-like prototype/reports/v1_actual_001/raw_cache"
RESUME_SHA256 = ""
RESUME_ARCHIVE = ""
RESUME_ARCHIVE_SHA256 = ""

if RUN_EPOCH_EXPANSION:
    if not (RESUME_SHA256 or RESUME_ARCHIVE) and RUN.exists():
        raise FileExistsError(f"기존 작업 경로가 있습니다. 재개하거나 새 런타임을 사용하세요. {RUN}")
    if (bool(RESUME_ARCHIVE) != bool(RESUME_ARCHIVE_SHA256)) or (RESUME_ARCHIVE and RESUME_SHA256):
        raise ValueError("회수 ZIP 경로·SHA 또는 기존 last SHA 중 한 방식만 사용합니다.")
    attempt = prepare_recovery(MOUNT_DRIVE)
    EXPANSION_DECISION = verified / "epoch_expansion_decision.json"
    EXPANSION_DECISION.write_text(json.dumps(PINS["expansion_decision"]), encoding="utf-8")
    if RESUME_ARCHIVE:
        restored = tracked_child("epoch_restore", [sys.executable, "-m", "rewind_v1", "restore-epoch-expansion",
                                 "--archive", RESUME_ARCHIVE, "--archive-sha256", RESUME_ARCHIVE_SHA256,
                                 "--output", str(RUN)], timeout=600, phase="restore", capture_output=True)
        RESUME_SHA256 = json.loads(restored.stdout)["last_sha256"]
    command = [sys.executable, "-m", "rewind_v1", "epoch-expansion", "--output", str(RUN),
               "--config", str(SOURCE / PINS["config_path"]), "--config-sha256", PINS["config_sha256"],
               "--source-sha256", PINS["source_code_sha256"],
               "--selection", str(INPUTS / "selection.json"), "--selection-sha256", PINS["selection_sha256"],
               "--spec", str(INPUTS / "spec.json"), "--cache", CACHE_SOURCE,
               "--cache-local", str(WORK / "cache"), "--cache-sha256", PINS["cache_sha256"],
               "--diagnostic-bundle", str(DIAGNOSTIC_BUNDLE), "--decision-sha256", PINS["decision_sha256"],
               "--expansion-decision", str(EXPANSION_DECISION), "--expansion-decision-sha256",
               PINS["expansion_decision"]["content_sha256"]]
    if RESUME_SHA256:
        command += ["--resume-sha256", RESUME_SHA256]
    try:
        tracked_child("training", [sys.executable, "-m", "rewind_v1.device"],
                      timeout=60, phase="cuda_preflight")
        tracked_child("training", command, timeout=PINS["training_hard_seconds"], phase="same_1000_pairs_epoch_expansion")
    finally:
        training_failed = sys.exc_info()[0] is not None
        try:
            save_recovery(attempt, browser_download=BROWSER_DOWNLOAD)
        except Exception as error:
            if not training_failed:
                raise
            print(f"학습 오류와 별도로 회수에도 실패했습니다. {type(error).__name__} {error}")
'''
    recover = '''
RUN_RECOVERY = False
MOUNT_RECOVERY_DRIVE = True
RECOVERY_LOCAL_ZIP = ""
BROWSER_DOWNLOAD = False

if RUN_RECOVERY:
    save_recovery(prepare_recovery(MOUNT_RECOVERY_DRIVE), RECOVERY_LOCAL_ZIP, BROWSER_DOWNLOAD)
'''
    guide = f"""# {RUN_ID} 재실행
고정 소스·입력·기존 진단 근거 ZIP을 /content에 올립니다. CUDA GPU에서 동일한 train 1,000페어를 처음부터 단계별 10epoch 학습합니다.
기존 Drive raw cache를 로컬에 재사용합니다. 명목 25FPS 가정과 참고용 validation 표시는 기존 진단과 같습니다.
단계별 epoch 1·5·10 모델과 재개용 last.pt를 별도로 보존합니다. 각 모델의 앞 단계 의존성은 checkpoint와 run.json에서 확인합니다.
학습 호출 상한은 2시간입니다. RUN을 켜면 먼저 Drive 결과 폴더의 쓰기를 확인합니다.
학습 호출이 끝나면 회수 ZIP을 최대 1 GiB까지 생성하고 새 Drive attempt 폴더에 복사합니다. 복사본의 크기와 SHA-256을 다시 읽어 검증한 뒤 receipt.json을 씁니다.
브라우저 다운로드는 선택 사항입니다. Drive 복사가 실패하면 같은 런타임의 마지막 셀에서 표시된 로컬 ZIP으로 재시도합니다.
강제 VM 종료 전에 중간 결과를 Drive에 저장하는 경로는 없습니다.
Drive receipt.json의 state가 verified, kind가 training이고 last_sha256이 있어야 모델 회수가 완료된 것입니다.
새 런타임 재개에는 회수 ZIP 경로와 SHA를, 같은 런타임에는 last_sha256을 사용합니다. 시작 전 로컬 여유 공간 12GiB를 확보합니다.
"""
    return helpers, guide, launch, recover


def notebook_document(pins):
    expansion = "expansion_decision" in pins
    work_name = "archvis_v11_rerun" if expansion else "archvis_v1_diagnostic_001"
    original = audit_notebook(pins)
    bootstrap = "".join(next(cell["source"] for cell in original["cells"] if cell["id"] == "bootstrap"))
    for before, after in (
        ('Path("/content/archvis_v1_fixed")', f'Path("/content/{work_name}")'),
        ('("approval", 16, 32, 16)', '("diagnostic_bundle", 16, 32, 16)'),
        ('APPROVAL = verified / "approval"', 'DIAGNOSTIC_BUNDLE = verified / "diagnostic_bundle"'),
    ):
        if bootstrap.count(before) != 1:
            raise ValueError("archvis_v1_diagnostic_bootstrap_contract")
        bootstrap = bootstrap.replace(before, after)
    launch = '''
RUN_DIAGNOSTIC = False
MOUNT_DRIVE = True
CACHE_SOURCE = "/content/drive/MyDrive/REWIND-like prototype/reports/v1_actual_001/raw_cache"
RESUME_SHA256 = ""

if RUN_DIAGNOSTIC:
    tracked_child("training", [sys.executable, "-m", "rewind_v1.device"],
                  timeout=60, phase="cuda_preflight")
    if MOUNT_DRIVE:
        from google.colab import drive
        drive.mount("/content/drive")
    command = [sys.executable, "-m", "rewind_v1", "diagnostic", "--output", str(RUN),
               "--config", str(SOURCE / PINS["config_path"]), "--config-sha256", PINS["config_sha256"],
               "--source-sha256", PINS["source_code_sha256"],
               "--selection", str(INPUTS / "selection.json"), "--selection-sha256", PINS["selection_sha256"],
               "--spec", str(INPUTS / "spec.json"), "--cache", CACHE_SOURCE,
               "--cache-local", str(WORK / "cache"), "--cache-sha256", PINS["cache_sha256"],
               "--diagnostic-bundle", str(DIAGNOSTIC_BUNDLE), "--decision-sha256", PINS["decision_sha256"]]
    if RESUME_SHA256:
        command += ["--resume-sha256", RESUME_SHA256]
    tracked_child("training", command, timeout=PINS["training_hard_seconds"], phase="nominal_25fps_diagnostic")
'''
    recover = '''
RUN_RECOVERY = False
if RUN_RECOVERY:
    from google.colab import files
    for module, mode_args, name in (
        ("rewind_v1.audit", ["--work", str(WORK), "--max-bytes", str(32*1024**2)], "v1_diagnostic_001_status"),
        ("rewind_v1", ["--run", str(RUN), "--max-bytes", str(1024**3)], "v1_diagnostic_001_training"),
    ):
        if module == "rewind_v1" and not ((RUN / "run.json").is_file() or (RUN / "launch_error.json").is_file()
                                          or (RUN / "last.pt").is_file()):
            continue
        target = next((UPLOADS / f"{name}-{index:02d}.zip" for index in range(1, 101)
                       if not (UPLOADS / f"{name}-{index:02d}.zip").exists()), None)
        if target is None:
            print(f"{name}: 새 회수 ZIP 경로가 필요합니다.")
            continue
        try:
            subprocess.run([sys.executable, "-m", module, "recover", *mode_args, "--output", str(target)],
                           cwd=SOURCE, timeout=600, check=True)
            files.download(str(target))
        except Exception as exc:
            print(f"{name}: {type(exc).__name__}: {exc}")
'''
    cells = [_cell("markdown", "guide", """# V1-DIAGNOSTIC-001
고정 소스 ZIP, 입력 ZIP, 진단 근거 ZIP을 /content에 올립니다. 학습은 CUDA GPU에서 시작합니다. 실제 GPU 이름과 VRAM을 기록합니다.
Drive의 보존 raw cache를 로컬 SSD로 복원하며 원본 추출 RGB·pose 재읽기나 감사 재실행은 없습니다.
시간은 frame index ÷ 25라는 명목 가정입니다. 시간과 stereo 동기는 측정 검증되지 않았습니다.
원본 motion 독립성은 미확인이고 공식 validation 수치는 참고 지표입니다.
frontend, body, hand를 각각 1 epoch 진단 학습합니다. 실패·중단 뒤 마지막 셀에서 결과를 회수합니다.
"""), _cell("code", "bootstrap", bootstrap), _cell("code", "diagnostic", launch), _cell("code", "recover", recover)]
    if expansion:
        helpers, guide, launch, recover = _expansion_cells()
        cells = [_cell("markdown", "guide", guide), _cell("code", "bootstrap", bootstrap + helpers),
                 _cell("code", "epoch-expansion", launch), _cell("code", "recover", recover)]
    for cell in cells[1:]:
        compile("".join(cell["source"]), cell["id"], "exec")
    return {**original, "cells": cells}


def build(args, *, expansion=False):
    output = Path(args.output)
    if output.exists() or not all(_hash(pin) for pin in (
        args.source_sha256, args.input_sha256, args.evidence_sha256, args.decision_sha256
    )):
        raise ValueError("archvis_v1_diagnostic_builder_args")
    with tempfile.TemporaryDirectory(dir=output.parent) as directory:
        root = Path(directory)
        pins = {}
        for key, path, pin, zip_cap, unpacked_cap, count, revision in (
            ("source", args.source_zip, args.source_sha256, 64, 192, 3000, args.revision),
            ("inputs", args.input_zip, args.input_sha256, 16, 32, 64, None),
            ("diagnostic_bundle", args.evidence_zip, args.evidence_sha256, 16, 32, 16, None),
        ):
            checked = extract_pinned_zip(path, pin, root / key, max_zip_bytes=zip_cap*MIB,
                                         max_output_bytes=unpacked_cap*MIB, max_members=count, revision=revision)
            pins[key] = {**checked, "name": path.name}
            if revision is not None:
                pins[key]["revision"] = revision
        plan, _ = load_inputs(root / "inputs", HISTORICAL["selection_sha256"])
        validate_bundle(root / "diagnostic_bundle", args.decision_sha256, plan)
        if expansion and not (root / "source/scripts/v11_recovery.py").is_file():
            raise ValueError("archvis_v11_recovery_source_missing")
        config_path = EXPANSION_CONFIG if expansion else CONFIG
        config_raw = (root / "source" / config_path).read_bytes()
        config = validate_config(json.loads(config_raw))
        (validate_expansion if expansion else validate_pilot)(config)
        child = subprocess.run([sys.executable, "-m", "rewind_v1", "source"], cwd=root / "source",
                               timeout=60, capture_output=True, text=True, check=True)
        source_identity = json.loads(child.stdout)["source_sha256"]
        if not _hash(source_identity):
            raise ValueError("archvis_v1_diagnostic_builder_source")
        pins.update(source_code_sha256=source_identity, config_path=config_path, config_sha256=sha(config_raw),
                    selection_sha256=HISTORICAL["selection_sha256"], cache_sha256=HISTORICAL["cache_sha256"],
                    decision_sha256=args.decision_sha256, prior_audit_source_sha256=HISTORICAL["audit_source_sha256"],
                    training_hard_seconds=HARD_SECONDS if expansion else config["limits"]["max_seconds"]+120)
        if expansion:
            pins["expansion_decision"] = expansion_decision(args.decision_sha256, pins["config_sha256"])
    raw = (json.dumps(notebook_document(pins), ensure_ascii=False, indent=1) + "\n").encode()
    output.mkdir()
    name = "archvis_v1_epoch_expansion.ipynb" if expansion else "archvis_v1_diagnostic_001.ipynb"
    (output / name).write_bytes(raw)
    receipt = {"schema": "archvis_v1_epoch_expansion_handoff_v1" if expansion else "archvis_v1_diagnostic_handoff_v1", "pins": pins,
               "notebook": {"name": name, "sha256": sha(raw), "bytes": len(raw)}}
    receipt_name = "epoch_expansion_handoff_receipt.json" if expansion else "diagnostic_handoff_receipt.json"
    if expansion:
        (output / "epoch_expansion_decision.json").write_bytes(canonical(pins["expansion_decision"]) + b"\n")
    (output / receipt_name).write_bytes(canonical(receipt) + b"\n")
    return receipt


def main(*, expansion=False):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source-zip", "input-zip", "evidence-zip", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    for name in ("source-sha256", "input-sha256", "evidence-sha256", "decision-sha256", "revision"):
        parser.add_argument("--" + name, required=True)
    print(canonical(build(parser.parse_args(), expansion=expansion)).decode("utf-8"))


if __name__ == "__main__":
    main()
