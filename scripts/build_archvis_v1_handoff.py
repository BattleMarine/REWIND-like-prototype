"""Build a fixed Colab audit handoff from separately pinned source and input ZIPs."""

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
from rewind_v1.runner import validate_config
from scripts.archvis_pilot_bootstrap import extract_pinned_zip
from scripts.build_archvis_pilot_handoff import _cell


MIB = 1024**2
CONFIG = "configs/rewind_v1_pipeline_pilot_candidate.json"
LIMITS = {
    "pack-read-bytes": 16*1024*MIB, "pack-disk-bytes": 12*1024*MIB, "pack-buffer-bytes": 512*MIB,
    "pack-seconds": 7200, "workers": 4, "audit-read-bytes": 32*1024*MIB, "audit-report-bytes": 128*MIB,
    "audit-seconds": 3600,
}


def notebook_document(pins):
    bootstrap = Path(__file__).with_name("archvis_pilot_bootstrap.py").read_text(encoding="utf-8")
    bootstrap += f"\nPINS = {pins!r}\nAUDIT_LIMITS = {LIMITS!r}\n"
    bootstrap += '''
from contextlib import suppress
import json
import subprocess
import sys
import tempfile

UPLOADS = Path("/content")
WORK = Path("/content/archvis_v1_fixed")
WORK.mkdir(exist_ok=True)
verified = Path(tempfile.mkdtemp(prefix="verified-", dir=WORK))
for key, zip_cap, unpacked_cap, count in (("source", 64, 192, 3000), ("inputs", 16, 32, 64), ("approval", 16, 32, 16)):
    if key in PINS:
        pin = PINS[key]
        extract_pinned_zip(UPLOADS / pin["name"], pin["sha256"], verified / key,
                           max_zip_bytes=zip_cap*1024**2, max_output_bytes=unpacked_cap*1024**2,
                           max_members=count, revision=pin.get("revision"))
SOURCE = verified / "source"
INPUTS = verified / "inputs"
APPROVAL = verified / "approval"
RUN = WORK / "training"
IDENTITY = subprocess.run([sys.executable, "-m", "rewind_v1", "source"], cwd=SOURCE,
                          timeout=60, capture_output=True, text=True, check=True)
if json.loads(IDENTITY.stdout)["source_sha256"] != PINS["source_code_sha256"]:
    raise ValueError("archvis_v1_source_code_pin")
if hashlib.sha256((SOURCE / PINS["config_path"]).read_bytes()).hexdigest() != PINS["config_sha256"]:
    raise ValueError("archvis_v1_training_config_pin")

def tracked_child(name, command, timeout, phase, capture_output=False):
    receipt = {"status": "started", "timeout_seconds": timeout, "phase": phase}
    path = WORK / ("notebook_" + name + "_result.json")
    def record():
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(dir=WORK, delete=False) as stream:
                temporary = Path(stream.name)
                stream.write(json.dumps(receipt, ensure_ascii=False).encode("utf-8"))
                stream.flush()
                os.fsync(stream.fileno())
            temporary.replace(path)
        finally:
            if temporary is not None:
                with suppress(OSError):
                    temporary.unlink(missing_ok=True)
    record()
    try:
        result = subprocess.run(command, cwd=SOURCE, timeout=timeout, check=True,
                                capture_output=capture_output, text=capture_output)
    except BaseException as exc:
        receipt.update(status="stopped", failure_type=type(exc).__name__, reason=str(exc)[:512])
        if isinstance(exc, subprocess.CalledProcessError) and exc.stderr:
            receipt["stderr_tail"] = exc.stderr[-2048:]
        with suppress(BaseException):
            record()
        raise
    receipt.update(status="exited", returncode=result.returncode)
    record()
    return result
'''
    prep = '''
RUN_PREPARE = False
RGB_ROOT = ""
POSE_ROOT = ""
RETAINED_CACHE = ""
RETAINED_CACHE_SHA256 = ""
MOUNT_DRIVE = False

if RUN_PREPARE:
    if MOUNT_DRIVE:
        from google.colab import drive
        drive.mount("/content/drive")
    command = [sys.executable, "-m", "rewind_v1.audit", "prepare", "--work", str(WORK),
               "--inputs", str(INPUTS), "--selection-sha256", PINS["selection_sha256"],
               "--source-sha256", PINS["source_code_sha256"]]
    for name, value in AUDIT_LIMITS.items():
        command += ["--" + name, str(value)]
    for name, value in (("rgb-root", RGB_ROOT), ("pose-root", POSE_ROOT),
                        ("retained-cache", RETAINED_CACHE), ("retained-cache-sha256", RETAINED_CACHE_SHA256)):
        if value:
            command += ["--" + name, value]
    tracked_child("prepare", command, timeout=11880, phase="raw_pack_and_audit")
else:
    print("고정 준비 인계를 확인한 뒤 감사 셀을 실행합니다.")
'''
    train = '''
RUN_TRAINING = False
RESUME_SHA256 = ""
if "approval" not in PINS:
    print("시간·영상/pose·카메라·원본 motion 검토의 고정 승인 묶음이 없어 학습을 차단합니다.")
elif RUN_TRAINING:
    approval = PINS["approval"]
    audit_work = APPROVAL if (APPROVAL / "audit/audit.json").is_file() else WORK
    gate = tracked_child("training", [sys.executable, "-m", "rewind_v1.audit", "check-training", "--work", str(audit_work),
                           "--audit-sha256", approval["audit_sha256"], "--selection-sha256", PINS["selection_sha256"],
                           "--evidence-path", str(APPROVAL / "evidence.json"),
                           "--evidence-sha256", approval["evidence_sha256"]], timeout=120,
                         phase="external_evidence_gate", capture_output=True)
    checked = json.loads(gate.stdout)
    command = [sys.executable, "-m", "rewind_v1", "actual", "--output", str(RUN),
               "--config", str(SOURCE / PINS["config_path"]), "--config-sha256", PINS["config_sha256"],
               "--selection", str(INPUTS / "selection.json"), "--selection-sha256", PINS["selection_sha256"],
               "--spec", str(INPUTS / "spec.json"), "--source-sha256", PINS["source_code_sha256"],
               "--cache", str(WORK / "cache"), "--cache-sha256", checked["cache_sha256"],
               "--evidence", str(APPROVAL / "evidence.json"), "--evidence-sha256", approval["evidence_sha256"],
               "--review-dir", str(APPROVAL / "review")]
    if RESUME_SHA256:
        command += ["--resume-sha256", RESUME_SHA256]
    if RETAINED_CACHE and not (WORK / "cache/cache_manifest.json").is_file():
        cache_arg = command.index("--cache") + 1
        command[cache_arg] = RETAINED_CACHE
        command += ["--cache-local", str(WORK / "cache")]
    tracked_child("training", command, timeout=PINS["training_hard_seconds"], phase="cuda_pilot")
'''
    recover = '''
RUN_RECOVERY = False
if RUN_RECOVERY:
    from google.colab import files
    for submodule, args, filename in (
        ("rewind_v1.audit", ["--work", str(WORK), "--max-bytes", str(160*1024**2)], "archvis_v1_audit_recovery.zip"),
        ("rewind_v1", ["--run", str(RUN), "--max-bytes", str(1024**3)], "archvis_v1_training_recovery.zip"),
    ):
        if submodule == "rewind_v1" and not RUN.is_dir():
            continue
        target = UPLOADS / filename
        for index in range(1, 101):
            if not target.exists():
                break
            original = Path(filename)
            target = UPLOADS / f"{original.stem}-{index:02d}{original.suffix}"
        try:
            subprocess.run([sys.executable, "-m", submodule, "recover", *args, "--output", str(target)],
                           cwd=SOURCE, timeout=600, check=True)
            files.download(str(target))
        except Exception as exc:
            print(f"{filename}: {type(exc).__name__}: {exc}")
'''
    cells = [_cell("markdown", "guide", """# ArchVis v1 고정 입력 검증
고정 소스와 입력 ZIP을 /content에 올립니다. 준비 단계는 CPU로도 실행할 수 있습니다.
Python의 PyTorch·Pillow·psutil이 필요합니다. 의존성이 없으면 인계 환경을 확인합니다.
고정 선택은 train 1,000페어와 공식 validation 2,236페어입니다. Raw cache가 있으면 고정 해시로 복원합니다.
시간 후보 필드, 카메라 검산, GT 오버레이, motion 비교는 검토용 근거입니다. 학습은 별도 고정 승인 묶음과 CUDA GPU가 필요합니다. 실제 GPU 이름과 VRAM을 기록합니다.
정상 차단·오류·중단 뒤에도 마지막 셀에서 보고를 회수합니다. Raw pack 파일은 결과 ZIP에 포함되지 않습니다.
기존 회수 ZIP은 보존하고 새 회수에는 번호를 붙입니다.
"""), _cell("code", "bootstrap", bootstrap), _cell("code", "prepare", prep),
        _cell("code", "conditional-training", train), _cell("code", "recover", recover)]
    for cell in cells[1:]:
        compile("".join(cell["source"]), cell["id"], "exec")
    return {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python",
            "name": "python3"}}, "nbformat": 4, "nbformat_minor": 5}


def build(args):
    output = Path(args.output)
    if output.exists() or not all(_hash(value) for value in (args.source_sha256, args.input_sha256, args.selection_sha256)):
        raise ValueError("archvis_v1_handoff_pins_or_output")
    with tempfile.TemporaryDirectory(dir=output.parent) as directory:
        work = Path(directory)
        source_pin = extract_pinned_zip(args.source_zip, args.source_sha256, work / "source", max_zip_bytes=64*MIB,
                                        max_output_bytes=192*MIB, max_members=3000, revision=args.revision)
        input_pin = extract_pinned_zip(args.input_zip, args.input_sha256, work / "inputs", max_zip_bytes=16*MIB,
                                       max_output_bytes=32*MIB, max_members=64)
        load_inputs(work / "inputs", args.selection_sha256)
        process = subprocess.run([sys.executable, "-m", "rewind_v1", "source"], cwd=work / "source",
                                 timeout=60, check=True, capture_output=True, text=True)
        identity = json.loads(process.stdout)["source_sha256"]
        if not _hash(identity):
            raise ValueError("archvis_v1_handoff_source_identity")
        config_raw = (work / "source" / CONFIG).read_bytes()
        cfg = validate_config(json.loads(config_raw))
        if cfg["epochs"] != {"frontend": 1, "body": 1, "hand": 1} or cfg["limits"]["max_updates"] != 540:
            raise ValueError("archvis_v1_handoff_pilot_contract")
        pins = {"source": {**source_pin, "name": args.source_zip.name, "revision": args.revision},
                "inputs": {**input_pin, "name": args.input_zip.name}, "selection_sha256": args.selection_sha256,
                "source_code_sha256": identity, "config_path": CONFIG, "config_sha256": sha(config_raw),
                "training_hard_seconds": cfg["limits"]["max_seconds"]+120}
        approval_args = (args.approval_zip, args.approval_sha256, args.audit_sha256, args.evidence_sha256)
        if any(approval_args):
            if not all(approval_args) or not all(_hash(pin) for pin in approval_args[1:]):
                raise ValueError("archvis_v1_handoff_approval_pins")
            approved = extract_pinned_zip(args.approval_zip, args.approval_sha256, work / "approval",
                                          max_zip_bytes=16*MIB, max_output_bytes=32*MIB, max_members=16)
            pins["approval"] = {**approved, "name": args.approval_zip.name, "audit_sha256": args.audit_sha256,
                                "evidence_sha256": args.evidence_sha256}
    raw = (json.dumps(notebook_document(pins), ensure_ascii=False, indent=1) + "\n").encode()
    output.mkdir()
    (output / "archvis_v1_fixed.ipynb").write_bytes(raw)
    receipt = {"schema": "archvis_v1_fixed_handoff_v1", "pins": pins, "audit_limits": LIMITS,
               "notebook": {"name": "archvis_v1_fixed.ipynb", "sha256": sha(raw), "bytes": len(raw)}}
    (output / "handoff_receipt.json").write_bytes(canonical(receipt) + b"\n")
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source-zip", "input-zip", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    for name in ("source-sha256", "input-sha256", "selection-sha256", "revision"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--approval-zip", type=Path)
    for name in ("approval-sha256", "audit-sha256", "evidence-sha256"):
        parser.add_argument("--" + name)
    print(canonical(build(parser.parse_args())).decode("utf-8"))


if __name__ == "__main__":
    main()
