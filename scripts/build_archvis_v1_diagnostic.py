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
from rewind_v1.runner import validate_config
from scripts.archvis_pilot_bootstrap import extract_pinned_zip
from scripts.build_archvis_pilot_handoff import _cell
from scripts.build_archvis_v1_handoff import CONFIG, MIB, notebook_document as audit_notebook


def notebook_document(pins):
    original = audit_notebook(pins)
    bootstrap = "".join(next(cell["source"] for cell in original["cells"] if cell["id"] == "bootstrap"))
    for before, after in (
        ('Path("/content/archvis_v1_fixed")', 'Path("/content/archvis_v1_diagnostic_001")'),
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
                  timeout=60, phase="a100_preflight")
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
고정 소스 ZIP, 입력 ZIP, 진단 근거 ZIP을 /content에 올립니다. 학습은 A100에서만 시작합니다.
Drive의 보존 raw cache를 로컬 SSD로 복원하며 원본 추출 RGB·pose 재읽기나 감사 재실행은 없습니다.
시간은 frame index ÷ 25라는 명목 가정입니다. 시간과 stereo 동기는 측정 검증되지 않았습니다.
원본 motion 독립성은 미확인이고 공식 validation 수치는 참고 지표입니다.
frontend, body, hand를 각각 1 epoch 진단 학습합니다. 실패·중단 뒤 마지막 셀에서 결과를 회수합니다.
"""), _cell("code", "bootstrap", bootstrap), _cell("code", "diagnostic", launch), _cell("code", "recover", recover)]
    for cell in cells[1:]:
        compile("".join(cell["source"]), cell["id"], "exec")
    return {**original, "cells": cells}


def build(args):
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
        config_raw = (root / "source" / CONFIG).read_bytes()
        config = validate_config(json.loads(config_raw))
        validate_pilot(config)
        child = subprocess.run([sys.executable, "-m", "rewind_v1", "source"], cwd=root / "source",
                               timeout=60, capture_output=True, text=True, check=True)
        source_identity = json.loads(child.stdout)["source_sha256"]
        if not _hash(source_identity):
            raise ValueError("archvis_v1_diagnostic_builder_source")
        pins.update(source_code_sha256=source_identity, config_path=CONFIG, config_sha256=sha(config_raw),
                    selection_sha256=HISTORICAL["selection_sha256"], cache_sha256=HISTORICAL["cache_sha256"],
                    decision_sha256=args.decision_sha256, prior_audit_source_sha256=HISTORICAL["audit_source_sha256"],
                    training_hard_seconds=config["limits"]["max_seconds"]+120)
    raw = (json.dumps(notebook_document(pins), ensure_ascii=False, indent=1) + "\n").encode()
    output.mkdir()
    name = "archvis_v1_diagnostic_001.ipynb"
    (output / name).write_bytes(raw)
    receipt = {"schema": "archvis_v1_diagnostic_handoff_v1", "pins": pins,
               "notebook": {"name": name, "sha256": sha(raw), "bytes": len(raw)}}
    (output / "diagnostic_handoff_receipt.json").write_bytes(canonical(receipt) + b"\n")
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source-zip", "input-zip", "evidence-zip", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    for name in ("source-sha256", "input-sha256", "evidence-sha256", "decision-sha256", "revision"):
        parser.add_argument("--" + name, required=True)
    print(canonical(build(parser.parse_args())).decode("utf-8"))


if __name__ == "__main__":
    main()
