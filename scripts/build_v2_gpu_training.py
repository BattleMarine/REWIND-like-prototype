"""Offline release of the final-cache v2 training notebook and one pinned source ZIP."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import zipfile

from rewind_archive.contracts import canonical, digest
from rewind_m1.local_rgb import _regular_read
from rewind_prepare.temporal_cache import _hash_file
from scripts.archvis_pilot_bootstrap import extract_pinned_zip
from scripts.build_archvis_pilot_handoff import _cell
from scripts.build_v2_bulk_cache import MODULES as BULK_MODULES
from scripts.build_v2_bulk_cache_budgeted import SOURCE_IDENTITY
from scripts.build_v2_gpu_cache_probe import REVISION, SOURCE_SHA, input_pins


ROOT = Path(__file__).resolve().parents[1]
SOURCE_NAME = "v2_gpu_training_128_v1.zip"
FINAL_EVIDENCE_SHA = "1e3aa2b19ffc9afe31b3f87c6991229c86c924a2de7b6db213a5ca67a49c3aed"
TRAIN_MODULES = ("rewind_v1/runner.py", "scripts/run_v2_gpu_training.py", "scripts/v2_gpu_training_storage.py",
                 "scripts/v2_gpu_training_pipeline.py", "scripts/v2_gpu_training_features.py", "scripts/v2_gpu_training_comparison.py")


SETTINGS = '''RUN_V2_TRAINING = False
SLOT = "A"  # A=42, B=71, C=103. Choose one per GPU runtime.
DRIVE_PROJECT = "/content/drive/MyDrive/REWIND-like prototype"
EXPECTED_RESUME_TIP = ""  # Optional: exact latest receipt, never selects an older checkpoint.
RECOVER_WRITER_OWNER = ""  # Only after confirming the previous worker/VM cannot still write.
'''


RUN = '''
import json
import signal
import subprocess
import sys
import tempfile
import time

def invoke_once():
    started = time.monotonic()
    if SLOT not in ("A", "B", "C"):
        raise ValueError("Choose slot A, B or C")
    work = Path("/content/rewind_v2_gpu_training_128_v1") / SLOT
    work.mkdir(parents=True, exist_ok=True)
    invocation = Path(tempfile.mkdtemp(prefix="invocation-", dir=work))
    print("Local invocation:", invocation, flush=True)
    mount = Path("/content/drive")
    project = Path(DRIVE_PROJECT)
    if project == mount or not project.is_relative_to(mount) or ".." in project.parts:
        raise ValueError("Choose the project under mounted Drive")
    upload = Path("/content") / PINS["source_name"]
    if not upload.is_file():
        raise FileNotFoundError(f"Upload {upload.name} to /content")
    from google.colab import drive
    drive.mount(str(mount), timeout_ms=120_000)
    project = project.resolve(strict=True)
    if not project.is_relative_to(mount.resolve(strict=True)):
        raise ValueError("Project resolves outside Drive")
    source = invocation / "source"
    extract_pinned_zip(upload, PINS["source_sha256"], source, max_zip_bytes=64*1024**2,
                       max_output_bytes=192*1024**2, max_members=3000, revision=PINS["revision"])
    pin_file = invocation / "pins.json"
    pin_file.write_text(json.dumps(PINS), encoding="utf-8")
    command = [sys.executable, "-u", "-m", "scripts.run_v2_gpu_training", "--slot", SLOT,
               "--started", str(started), "--project", str(project), "--invocation", str(invocation),
               "--workspace", str(work), "--pins", str(pin_file), "--resume-tip", EXPECTED_RESUME_TIP,
               "--recover-writer", RECOVER_WRITER_OWNER]
    log_path = invocation / "worker.log"
    deadline = started + 28800
    try:
        with log_path.open("xb") as log, subprocess.Popen(command, cwd=source, stdout=log, stderr=subprocess.STDOUT) as child:
            offset = 0
            try:
                while child.poll() is None:
                    with log_path.open("rb") as stream:
                        stream.seek(offset)
                        for raw in iter(stream.readline, b""):
                            if not raw.endswith(b"\\n"):
                                break
                            offset = stream.tell()
                            print(raw.decode("utf-8", errors="replace").rstrip(), flush=True)
                    remaining = deadline-90-time.monotonic()
                    if remaining <= 0:
                        raise TimeoutError("GPU training notebook worker deadline")
                    try:
                        child.wait(timeout=min(15, remaining))
                    except subprocess.TimeoutExpired:
                        pass
            except BaseException:
                if child.poll() is None:
                    child.send_signal(signal.SIGINT)
                    remaining = min(60, max(0, deadline-20-time.monotonic()))
                    try:
                        child.wait(timeout=remaining)
                    except subprocess.TimeoutExpired:
                        child.kill()
                        child.wait(timeout=min(15, max(0.1, deadline-time.monotonic())))
                raise
            if child.returncode:
                raise RuntimeError(f"GPU training failed with exit {child.returncode}")
    finally:
        if log_path.is_file():
            with log_path.open("rb") as stream:
                stream.seek(max(0, stream.seek(0, 2)-16*1024))
                print(stream.read().decode("utf-8", errors="replace"), flush=True)
        print("Invocation seconds:", round(time.monotonic()-started, 3), flush=True)
        print("Diagnostic:", invocation / "result.json", flush=True)

if RUN_V2_TRAINING:
    invoke_once()
else:
    print("실행 스위치가 꺼져 있습니다.")
'''


def source_map(path):
    command = [sys.executable, "-c", "import json; from rewind_v1.v2_contract import source; print(json.dumps(source()))"]
    return json.loads(subprocess.run(command, cwd=path, capture_output=True, text=True, check=True, timeout=60).stdout)


def write_source_zip(root, path):
    with zipfile.ZipFile(path, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=3) as archive:
        for file in sorted(root.rglob("*")):
            name = file.relative_to(root)
            if not file.is_file() or "__pycache__" in name.parts:
                continue
            entry = zipfile.ZipInfo(name.as_posix(), date_time=(1980, 1, 1, 0, 0, 0))
            entry.external_attr = 0o100600 << 16
            archive.writestr(entry, file.read_bytes(), compress_type=zipfile.ZIP_DEFLATED, compresslevel=3)


def notebook(pins):
    info = ("# v2 GPU 학습 · 128 전체 캐시\n\n"
            f"Colab GPU 런타임에 `{SOURCE_NAME}`을 `/content`로 올리세요. "
            "A(seed 42)를 먼저 실행하고, A의 `first-update.json` 및 체크포인트가 Drive에 생성된 뒤 B(71)·C(103)를 실행합니다. "
            "세 슬롯은 서로 다른 GPU 런타임을 사용합니다. 실행 스위치는 기본 OFF입니다. 한 번 실행하면 최대 8시간 동안 학습하거나 재개합니다.\n\n"
            "완료 또는 중단 시 로컬 `result.json`과 각 슬롯 Drive의 최신 `checkpoints/step-*.json`을 회수하세요. "
            "완료본에는 `comparison.json`과 비교 ZIP이 추가됩니다. GPU 소프트웨어 버전·충분한 VM 디스크와 Drive 공간을 확인하세요.\n")
    bootstrap = (ROOT / "scripts/archvis_pilot_bootstrap.py").read_text(encoding="utf-8")
    code = bootstrap + f"\nPINS = {pins!r}\n" + RUN
    compile(SETTINGS, "training-settings", "exec")
    compile(code, "training-notebook", "exec")
    return {"nbformat": 4, "nbformat_minor": 5,
            "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"}},
            "cells": [_cell("markdown", "guide", info), _cell("code", "settings", SETTINGS), _cell("code", "train", code)]}


def build(base_source, catalog, cpu_report, completed_report, output):
    from scripts.run_v2_gpu_training import RESOURCE_LIMITS
    from scripts.v2_gpu_training_storage import SEEDS, EVAL_SEED, BOUNDARY, SHARDS, FRAMES
    if output.exists() or not output.parent.is_dir():
        raise ValueError("v2_gpu_output_new_directory_required")
    pins, _ = input_pins(base_source, catalog, cpu_report)
    evidence = _regular_read(completed_report, 192*1024)
    if hashlib.sha256(evidence).hexdigest() != FINAL_EVIDENCE_SHA:
        raise ValueError("v2_gpu_final_report_file_pin")
    report = json.loads(evidence)
    worker = report["worker"]
    if (report["pins"]["consumer_source_identity"] != SOURCE_IDENTITY or worker["status"] != "passed"
            or (worker["boundary_sha256"], worker["completed_shards"], worker["completed_frames"]) != (BOUNDARY, SHARDS, FRAMES)):
        raise ValueError("v2_gpu_final_report_scope")
    with tempfile.TemporaryDirectory(dir=output.parent, prefix=".gpu-training-") as directory:
        temp = Path(directory)
        root, release = temp / "source", temp / "release"
        release.mkdir()
        extract_pinned_zip(base_source, SOURCE_SHA, root, max_zip_bytes=64*1024**2,
                           max_output_bytes=192*1024**2, max_members=3000, revision=REVISION)
        for name in BULK_MODULES:
            (root / name).write_bytes((ROOT / name).read_text(encoding="utf-8").encode("utf-8"))
        producer = source_map(root)
        if digest(producer) != SOURCE_IDENTITY:
            raise ValueError("v2_gpu_builder_bulk_producer_pin")
        for name in TRAIN_MODULES:
            (root / name).write_bytes((ROOT / name).read_text(encoding="utf-8").encode("utf-8"))
        trainer = source_map(root)
        zip_path = release / SOURCE_NAME
        write_source_zip(root, zip_path)
        pins.update(source_name=SOURCE_NAME, source_sha256=_hash_file(zip_path), base_source_sha256=SOURCE_SHA,
                    bulk_producer_source=producer, bulk_producer_source_identity=SOURCE_IDENTITY,
                    training_core_sha256=digest(trainer), training_modules={n: _hash_file(root / n) for n in TRAIN_MODULES},
                    final_evidence_sha256=FINAL_EVIDENCE_SHA, final_boundary_sha256=BOUNDARY, seeds=SEEDS,
                    eval_seed=EVAL_SEED, resources=RESOURCE_LIMITS)
        name = "v2_gpu_training_128_v1.ipynb"
        raw = canonical(notebook(pins))
        (release / name).write_bytes(raw)
        (release / "receipt.json").write_bytes(canonical({"pins": pins, "source_bytes": zip_path.stat().st_size,
            "notebook": name, "notebook_bytes": len(raw), "notebook_sha256": hashlib.sha256(raw).hexdigest(), "default_off": True}))
        release.rename(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ("base-source", "catalog", "cpu-report", "completed-report", "output"):
        parser.add_argument("--"+arg, type=Path, required=True)
    build(**vars(parser.parse_args()))
