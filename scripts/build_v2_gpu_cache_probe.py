"""Build a default-off GPU probe notebook while reusing the ed98124 CPU source ZIP."""

import argparse
import json
from pathlib import Path
from string import Template
import tempfile

from rewind_archive.contracts import canonical, digest
from rewind_m1.local_rgb import _regular_read
from rewind_prepare.converter import sha
from scripts.archvis_pilot_bootstrap import extract_pinned_zip
from scripts.build_archvis_pilot_handoff import _cell
from scripts.run_v2_gpu_cache_probe import MAX_SECONDS, load_catalog


SOURCE_SHA = "c73016fbdcc2daef64ae2ffeba418209f433a96e52c843c481b4b53f627ae493"
REVISION = "ed981245b57d46535abb096ba6a280c8c333ead3"
CATALOG_SHA = "fe606a869c80599b742ed192fd99960faa1cf2c6242208c3bc1173eb0888ac26"
CPU_REPORT_SHA = "1df7514648538c5478227912552bf4dba002de38a2cce784a36b24aef330dcdd"
DRIVER = "run_v2_gpu_cache_probe.py"


SETTINGS = '''RUN_GPU_CACHE_PROBE = False
DRIVE_PROJECT = "/content/drive/MyDrive/REWIND-like prototype"
'''


PROBE_DRIVER_BLOCK = '''        driver = DRIVER_TEXT.encode("utf-8")
        if hashlib.sha256(driver).hexdigest() != PINS["probe_source_sha256"]:
            raise ValueError("GPU probe source pin")
        driver_path = source / "scripts" / "run_v2_gpu_cache_probe.py"
        with driver_path.open("xb") as stream:
            stream.write(driver)
'''
DEFAULT_COMMAND_BLOCK = '''        if remaining <= 0:
            raise TimeoutError("No time left for the probe worker")
        command = [sys.executable, "-m", "$worker_module",
                   "--pins", str(pin_file), "--work", str(invocation),
                   "--input-dir", str(parent / "v2_cpu_cache_041837Z"), "--seconds", str(remaining)]
'''


RUN_TEMPLATE = Template('''
import json
import shutil
import signal
import subprocess
import sys
import tempfile
import time

def report_timeout(_signum, _frame):
    raise TimeoutError("Drive report publication timed out")

def save_report(invocation, retained, started, error):
    log_path = invocation / "worker.log"
    log = ""
    if log_path.is_file():
        with log_path.open("rb") as stream:
            stream.seek(max(0, log_path.stat().st_size - 32*1024))
            log = stream.read(32*1024).decode("utf-8", errors="replace")
    worker_path = invocation / "result.json"
    worker = None
    if worker_path.is_file() and worker_path.stat().st_size <= 128*1024:
        try:
            worker = json.loads(worker_path.read_bytes())
        except (ValueError, OSError):
            pass
    report = {
        "status": $success_rule
                  else "stopped" if isinstance(error, KeyboardInterrupt) else "failed",
        "elapsed_to_report_seconds": round(time.monotonic()-started, 3),
        "outer_limit_seconds": LIMIT,
        "error": {"type": type(error).__name__, "message": str(error)[:512]} if error else None,
        "pins": PINS, "worker": worker, "worker_log_tail": log,
    }
    raw = json.dumps(report, ensure_ascii=False).encode("utf-8")
    if len(raw) > 192*1024:
        raise ValueError("GPU diagnostic exceeds report limit")
    local = invocation / "diagnostic.json"
    local.write_bytes(raw)
    print("Local diagnostic:", local, flush=True)
    if log:
        print(log, flush=True)
    if retained is None:
        return
    remaining = min(60, int(LIMIT-(time.monotonic()-started)))
    if remaining <= 0:
        raise TimeoutError("No time left for Drive report")
    prev_handler = signal.signal(signal.SIGALRM, report_timeout)
    prev_alarm = signal.alarm(remaining)
    try:
        retained.mkdir(exist_ok=True)
        target = retained / (invocation.name + ".json")
        if target.exists() or target.is_symlink():
            raise FileExistsError(target)
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(dir=retained, prefix=".report-", delete=False) as stream:
                temporary = Path(stream.name)
                if stream.write(raw) != len(raw):
                    raise OSError("Short Drive diagnostic write")
                stream.flush()
                os.fsync(stream.fileno())
            temporary.rename(target)
            with target.open("rb") as stream:
                if stream.read(len(raw) + 1) != raw:
                    raise OSError("Drive diagnostic readback mismatch")
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
    finally:
        signal.signal(signal.SIGALRM, prev_handler)
        signal.alarm(prev_alarm)
    print("Drive diagnostic:", target, "SHA-256:", hashlib.sha256(raw).hexdigest(), flush=True)

def $entrypoint():
    started = time.monotonic()
    work = Path("/content/$work_folder")
    work.mkdir(exist_ok=True)
    invocation = Path(tempfile.mkdtemp(prefix="invocation-", dir=work))
    print("Local invocation folder:", invocation, flush=True)
    retained = None
    error = None
    try:
        upload = Path("/content") / PINS["source_name"]
        if not upload.is_file() or upload.is_symlink():
            raise FileNotFoundError(f"Colab Files → Upload: {PINS['source_name']} to /content. Expected {upload}")
        mount = Path("/content/drive")
        project = Path(DRIVE_PROJECT)
        if project == mount or not project.is_relative_to(mount) or ".." in project.parts:
            raise ValueError("Set DRIVE_PROJECT to the project within mounted Drive")
        from google.colab import drive
        drive.mount(str(mount), timeout_ms=120_000)
        project = project.resolve(strict=True)
        if not project.is_relative_to(mount.resolve(strict=True)):
            raise ValueError("Drive project resolved outside the mount")
        parent = (project / "reports" / "cache").resolve(strict=True)
        if not parent.is_relative_to(project):
            raise ValueError("Drive cache folder resolved outside the project")
        candidate = parent / "$report_folder"
        if candidate.is_symlink():
            raise ValueError("Drive report destination is a link")
        retained = candidate
        if shutil.disk_usage(invocation).free < 512*1024**2:
            raise OSError("At least 512 MiB of local free space is required")
        source = invocation / "source"
        extract_pinned_zip(upload, PINS["source_sha256"], source,
                           max_zip_bytes=64*1024**2, max_output_bytes=192*1024**2, max_members=3000,
                           revision=PINS["revision"])
$driver_block        pin_file = invocation / "pins.json"
        pin_file.write_text(json.dumps(PINS), encoding="utf-8")
        remaining = int(LIMIT-(time.monotonic()-started)) - 90
$command_block        with (invocation / "worker.log").open("xb") as log, subprocess.Popen(
                command, cwd=source, stdout=log, stderr=subprocess.STDOUT) as child:
            try:
                code = child.wait(timeout=remaining)
            except BaseException:
                if child.poll() is None:
                    child.send_signal(signal.SIGINT)
                try:
                    child.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    child.kill()
                    child.wait(timeout=10)
                raise
        if code:
            raise RuntimeError(f"GPU cache worker exited with code {code}")
    except BaseException as failure:
        error = failure
        raise
    finally:
        try:
            save_report(invocation, retained, started, error)
        except Exception as report_error:
            print("Drive report failed:", report_error, "Local folder:", invocation, flush=True)
            if error is None:
                raise
        finally:
            print("Notebook wall seconds:", round(time.monotonic()-started, 3), flush=True)

if $run_switch:
    $entrypoint()
else:
    print("실행 스위치가 꺼져 있습니다.")
''')

RUN = RUN_TEMPLATE.substitute(
    entrypoint="run_gpu_cache_probe", work_folder="rewind_v2_gpu_cache_probe", report_folder="v2_gpu_cache_041837Z",
    driver_block=PROBE_DRIVER_BLOCK,
    command_block=Template(DEFAULT_COMMAND_BLOCK).substitute(worker_module="scripts.run_v2_gpu_cache_probe"),
    run_switch="RUN_GPU_CACHE_PROBE",
    success_rule='"passed" if error is None and worker and worker["status"] == "passed"',
)


def notebook(pins, driver):
    bootstrap = Path(__file__).with_name("archvis_pilot_bootstrap.py").read_text(encoding="utf-8")
    guide = (
        "# v2 첫 CPU 캐시의 CUDA 읽기 시험\n\n"
        "Colab의 하드웨어 가속기를 GPU로 선택합니다. 이전 CPU 캐시 실행에 사용한 고정 source ZIP 하나를 "
        f"/content에 올립니다. 필요한 파일은 `{pins['source_name']}`입니다. "
        "설정 셀에서 Drive 프로젝트 경로를 확인하고 실행 스위치를 켭니다. "
        "Drive의 첫 캐시 샤드 한 개를 검증한 뒤 기존 샤드 로더의 train 두 프레임을 CUDA로 왕복 전송합니다.\n\n"
        "정식 복원 binding의 일치 여부와 CUDA 배치 비교 판정은 따로 기록합니다. "
        "종료 후 Drive 진단 JSON을 회수합니다. Drive 진단 게시에 실패하면 출력에 표시된 로컬 경로의 "
        "diagnostic.json과 worker.log를 회수합니다. 기본 실행 스위치는 꺼져 있습니다.\n"
    )
    return {
        "nbformat": 4, "nbformat_minor": 5,
        "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"}},
        "cells": [
            _cell("markdown", "guide", guide),
            _cell("code", "settings", SETTINGS),
            _cell("code", "probe", bootstrap + f"\nPINS = {pins!r}\nDRIVER_TEXT = {driver!r}\nLIMIT = {MAX_SECONDS}\n" + RUN),
        ],
    }


def input_pins(source, catalog, cpu_report):
    catalog_raw = _regular_read(catalog, 64 * 1024, 13700)
    cpu_raw = _regular_read(cpu_report, 64 * 1024)
    if sha(catalog_raw) != CATALOG_SHA or sha(cpu_raw) != CPU_REPORT_SHA:
        raise ValueError("v2_gpu_original_catalog_or_report_pin")
    cpu = json.loads(cpu_raw)
    pins = dict(cpu["pins"])
    if (pins["source_sha256"], pins["revision"], pins["source_name"]) != (SOURCE_SHA, REVISION, source.name):
        raise ValueError("v2_gpu_builder_source_pin")
    observed = json.loads(catalog_raw)
    worker = cpu["worker"]
    shard = worker["shard_receipt"]
    driver = Path(__file__).with_name(DRIVER).read_text(encoding="utf-8")
    pins.update(catalog_bytes=len(catalog_raw), catalog_file_sha256=CATALOG_SHA,
                catalog_content_sha256=observed["content_sha256"], cpu_report_file_sha256=CPU_REPORT_SHA,
                package=worker["retained_package"], records_sha256=shard["records_sha256"],
                shard_files=shard["files"], producer_environment=worker["runtime"],
                producer_binding_sha256=digest(shard["binding"]), probe_source_sha256=sha(driver.encode("utf-8")))
    load_catalog(catalog_raw, pins)
    return pins, driver


def build(source, catalog, cpu_report, output):
    if output.exists() or not output.parent.is_dir():
        raise ValueError("v2_gpu_new_output")
    pins, driver = input_pins(source, catalog, cpu_report)
    with tempfile.TemporaryDirectory(dir=output.parent) as tmp:
        root = Path(tmp) / "source"
        extract_pinned_zip(source, SOURCE_SHA, root, max_zip_bytes=64*1024**2, max_output_bytes=192*1024**2,
                           max_members=3000, revision=REVISION)
        raw = canonical(notebook(pins, driver))
        stage = Path(tmp) / "publish"
        stage.mkdir()
        (stage / "v2_gpu_cache_probe.ipynb").write_bytes(raw)
        (stage / "receipt.json").write_bytes(canonical({
            "pins": pins, "notebook_sha256": sha(raw), "bytes": len(raw), "default_off": True,
        }))
        stage.rename(output)


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "catalog", "cpu-report", "output"):
        cli.add_argument("--" + name, type=Path, required=True)
    build(**vars(cli.parse_args()))
