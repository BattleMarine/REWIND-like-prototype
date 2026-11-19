"""Build a default-off CPU notebook from frozen source and the 04:18 manifest receipt."""

import argparse
import json
from pathlib import Path
import subprocess
import sys
import tempfile

from rewind_archive.contracts import canonical
from rewind_m1.local_rgb import _regular_read
from rewind_prepare.converter import sha
from scripts.archvis_pilot_bootstrap import extract_pinned_zip
from scripts.build_archvis_pilot_handoff import _cell
from scripts.resume_v2_drive_metadata import sha_file
from scripts.run_v2_cpu_cache import ARCHIVE_CAP, MANIFEST_CAP, MANIFEST_NAME, MIB, SECONDS


SNAPSHOT_SHA = "d61aacaaecf38c427d6c62d31668edbb1be9d7bb2ce1fa5b1db0afd2a961dfc5"
SPEC_SHA = "32f1cfda25bc2b05745d2b534f0fd21eec0defbd379cd1d865c152e52c74b187"
EMBEDDED_ARCHIVE = "reference/v2_snapshot_manifest.zip"
SETTINGS = '''RUN_ONE_SHARD = False
# First run: leave empty. To advance after a successful run, enter the retained catalog FILE SHA-256.
RETAINED_CATALOG_FILE_SHA256 = ""
DRIVE_PROJECT = "/content/drive/MyDrive/REWIND-like prototype"
'''
WORKER = '''
import json
import shutil
import signal
import subprocess
import sys
import tempfile
import time

def file_tail(path, cap=48*1024):
    if not path.is_file():
        return ""
    with path.open("rb") as stream:
        stream.seek(max(0, path.stat().st_size - cap))
        return stream.read(cap).decode("utf-8", errors="replace")

def diagnostic_deadline(_signum, _frame):
    raise TimeoutError("Drive diagnostic deadline exceeded")

def save_diagnostic(invocation, retained, started, error):
    worker_report = invocation / "result.json"
    worker = None
    if worker_report.is_file() and worker_report.stat().st_size <= 512*1024:
        worker = json.loads(worker_report.read_text(encoding="utf-8"))
    log = file_tail(invocation / "worker.log")
    report = {
        "status": "remote_verified" if error is None and worker and worker["status"] == "remote_verified"
                  else "stopped" if isinstance(error, KeyboardInterrupt) else "failed",
        "error": {"type": type(error).__name__, "message": str(error)[:512]} if error else None,
        "pins": PINS, "prior_catalog_file_sha256": RETAINED_CATALOG_FILE_SHA256 or None,
        "elapsed_before_report_seconds": round(time.monotonic() - started, 3),
        "worker": worker, "worker_log_tail": log,
    }
    local = invocation / "diagnostic.json"
    raw = json.dumps(report, ensure_ascii=False, indent=2).encode("utf-8")
    local.write_bytes(raw)
    print("Local diagnostic:", local, flush=True)
    if log:
        print(log, flush=True)
    if retained is None:
        return
    seconds = min(60, int(LIMIT - (time.monotonic() - started)))
    if seconds <= 0:
        raise TimeoutError("No time left for the Drive diagnostic")
    has_alarm = hasattr(signal, "SIGALRM")
    if has_alarm:
        handler = signal.signal(signal.SIGALRM, diagnostic_deadline)
        alarm = signal.alarm(seconds)
    try:
        retained.mkdir(exist_ok=True)
        target = retained / (invocation.name + ".json")
        if target.exists() or target.is_symlink():
            raise FileExistsError(target)
        with tempfile.NamedTemporaryFile(dir=retained, prefix=".report-", delete=False) as stream:
            temporary = Path(stream.name)
            try:
                stream.write(raw)
                stream.flush()
                os.fsync(stream.fileno())
            except BaseException:
                stream.close()
                temporary.unlink(missing_ok=True)
                raise
        try:
            temporary.rename(target)
            with target.open("rb") as stream:
                if stream.read(len(raw) + 1) != raw:
                    raise OSError("Diagnostic report readback failed")
        finally:
            temporary.unlink(missing_ok=True)
    finally:
        if has_alarm:
            signal.alarm(alarm)
            signal.signal(signal.SIGALRM, handler)
    print("Drive diagnostic:", target, "SHA-256:", hashlib.sha256(raw).hexdigest(), flush=True)
    if worker and worker.get("retention"):
        print("Retained catalog file SHA-256:", worker["retention"]["catalog_file_sha256"], flush=True)

def run_one_shard():
    started = time.monotonic()
    work = Path("/content/rewind_v2_cpu_cache")
    work.mkdir(exist_ok=True)
    invocation = Path(tempfile.mkdtemp(prefix="invocation-", dir=work))
    print("Local invocation folder:", invocation, flush=True)
    retained = None
    error = None
    try:
        project = Path(DRIVE_PROJECT)
        mount = Path("/content/drive")
        if not project.is_relative_to(mount) or project == mount or ".." in project.parts:
            raise ValueError("Set DRIVE_PROJECT to the existing mounted project folder")
        from google.colab import drive
        drive.mount(str(mount), timeout_ms=120_000)
        project = project.resolve(strict=True)
        if not project.is_relative_to(mount.resolve(strict=True)):
            raise ValueError("Drive project resolved outside the mounted Drive")
        report_parent = (project / "reports" / "cache").resolve(strict=True)
        if not report_parent.is_relative_to(project):
            raise ValueError("Drive report folder resolved outside the project")
        candidate = report_parent / "v2_cpu_cache_041837Z"
        if candidate.is_symlink():
            raise ValueError("Drive retention folder must be a directory")
        retained = candidate
        if RETAINED_CATALOG_FILE_SHA256 and not re.fullmatch("[0-9a-f]{64}", RETAINED_CATALOG_FILE_SHA256):
            raise ValueError("Expected the retained catalog FILE SHA-256")
        if shutil.disk_usage(work).free < 768 * 1024**2:
            raise OSError("At least 768 MiB of local space is required before extraction")
        source = invocation / "source"
        extract_pinned_zip(Path("/content") / PINS["source_name"], PINS["source_sha256"], source,
                           max_zip_bytes=64*1024**2, max_output_bytes=192*1024**2,
                           max_members=3000, revision=PINS["revision"])
        pin_file = invocation / "pins.json"
        pin_file.write_text(json.dumps(PINS), encoding="utf-8")
        data = project / "datasets" / "UnrealEgo2"
        remaining = int(LIMIT - (time.monotonic() - started)) - 180
        cmd = [sys.executable, "-m", "scripts.run_v2_cpu_cache",
               "--pins", str(pin_file), "--manifest-archive", str(source / PINS["embedded_manifest"]),
               "--work", str(invocation), "--cache", str(work / "cache"),
               "--rgb-root", str(data / "UnrealEgoData2_train_val_rgb"),
               "--pose-root", str(data / "UnrealEgoData2_train_val_pose"),
               "--retained", str(retained), "--worker-seconds", str(remaining)]
        if RETAINED_CATALOG_FILE_SHA256:
            cmd += ["--retained-catalog-sha256", RETAINED_CATALOG_FILE_SHA256]
        if remaining <= 2520:
            raise TimeoutError("Insufficient time remaining for the cache worker")
        with (invocation / "worker.log").open("xb") as log, subprocess.Popen(
                cmd, cwd=source, stdout=log, stderr=subprocess.STDOUT) as child:
            try:
                code = child.wait(timeout=remaining)
            except BaseException:
                if child.poll() is None:
                    child.send_signal(signal.SIGINT)
                try:
                    child.wait(timeout=60)
                except subprocess.TimeoutExpired:
                    child.kill()
                    child.wait(timeout=30)
                raise
        if code:
            raise RuntimeError(f"Cache worker exited with code {code}")
    except BaseException as failure:
        error = failure
        raise
    finally:
        try:
            save_diagnostic(invocation, retained, started, error)
        except Exception as report_error:
            print("Diagnostic Drive backup failed:", report_error, "Local folder:", invocation, flush=True)
            if error is None:
                raise
        finally:
            print("Notebook invocation wall seconds:", round(time.monotonic() - started, 3), flush=True)

if RUN_ONE_SHARD:
    run_one_shard()
else:
    print("실행 스위치가 꺼져 있습니다.")
'''


def notebook(pins):
    bootstrap = Path(__file__).with_name("archvis_pilot_bootstrap.py").read_text(encoding="utf-8")
    guide = (
        "# v2 CPU 캐시 첫 샤드\n\n"
        "런타임은 CPU를 선택합니다. 고정 source ZIP 하나를 이름 그대로 /content에 올립니다. "
        "첫 호출은 Drive의 고정 RGB·pose 384개를 검증해 기존 v2 캐시의 처음 128프레임을 생성합니다. "
        "결과 샤드 ZIP, 카탈로그, 실행 보고서를 Drive에 보관합니다.\n\n"
        "성공 후 다음 한 샤드는 별도 실행입니다. 출력의 retained catalog **file SHA-256**을 "
        "설정에 입력하면 같은 캐시 형식의 기존 접두 구간을 사용합니다. 전체 캐시와 학습은 별도 작업입니다. "
        "기본 실행 스위치는 꺼져 있습니다. 설정 셀에서 실행을 켠 뒤 전체 실행하고, "
        "종료 후 출력의 Drive 진단 JSON과 카탈로그 파일 SHA-256을 보관합니다. "
        "Drive를 마운트할 수 없으면 표시된 로컬 진단 경로를 확인합니다.\n"
    )
    return {
        "nbformat": 4, "nbformat_minor": 5,
        "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"}},
        "cells": [
            _cell("markdown", "guide", guide),
            _cell("code", "settings", SETTINGS),
            _cell("code", "cache", bootstrap + f"\nPINS = {pins!r}\nLIMIT = {SECONDS}\n" + WORKER),
        ],
    }


def build(source, source_sha256, revision, manifest_receipt, output):
    if output.exists() or not output.parent.is_dir():
        raise ValueError("v2_pilot_new_output")
    data = json.loads(_regular_read(manifest_receipt, MIB))
    if (data["frames"] != 136011 or data["member"] != MANIFEST_NAME
            or data["provenance"]["snapshot_db_sha256"] != SNAPSHOT_SHA):
        raise ValueError("v2_pilot_snapshot_receipt")
    with tempfile.TemporaryDirectory(dir=output.parent) as temp:
        root = Path(temp) / "source"
        extract_pinned_zip(source, source_sha256, root, max_zip_bytes=64*MIB, max_output_bytes=192*MIB,
                           max_members=3000, revision=revision)
        for path in ("scripts/run_v2_cpu_cache.py", "reference/split/train.txt",
                     "reference/split/validation.txt", "reference/split/test.txt"):
            if not (root / path).is_file():
                raise ValueError("v2_pilot_source_missing:" + path)
        if sha_file(root / "reference/spec.json") != SPEC_SHA:
            raise ValueError("v2_pilot_builder_spec_pin")
        archive = root / EMBEDDED_ARCHIVE
        if archive.stat().st_size != data["archive_bytes"]:
            raise ValueError("v2_pilot_builder_archive_size")
        dest = Path(temp) / "manifest"
        extract_pinned_zip(archive, data["archive_sha256"], dest,
                           max_zip_bytes=ARCHIVE_CAP, max_output_bytes=MANIFEST_CAP, max_members=1)
        manifest = dest / MANIFEST_NAME
        if manifest.stat().st_size != data["manifest_bytes"] or sha_file(manifest) != data["manifest_file_sha256"]:
            raise ValueError("v2_pilot_builder_manifest_pin")
        child = subprocess.run([sys.executable, "-m", "rewind_v1.v2_cli", "source"], cwd=root,
                               capture_output=True, text=True, timeout=60, check=True)
        pins = {key: data[key] for key in ("archive_bytes", "archive_sha256", "manifest_bytes",
                                          "manifest_file_sha256", "manifest_content_sha256", "frames",
                                          "first_128_source_bytes")}
        pins.update(source_name=source.name, source_sha256=source_sha256, revision=revision,
                    source_identity=json.loads(child.stdout)["source_sha256"],
                    config_sha256=sha_file(root / "configs/rewind_v2.json"), spec_sha256=SPEC_SHA,
                    source_snapshot_db_sha256=SNAPSHOT_SHA, embedded_manifest=EMBEDDED_ARCHIVE)
        raw = canonical(notebook(pins))
        publish = Path(temp) / "handoff"
        publish.mkdir()
        (publish / "v2_cpu_cache.ipynb").write_bytes(raw)
        (publish / "receipt.json").write_bytes(canonical({
            "pins": pins, "notebook_sha256": sha(raw), "bytes": len(raw), "default_off": True,
        }))
        publish.rename(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "manifest-receipt", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    for name in ("source-sha256", "revision"):
        parser.add_argument("--" + name, required=True)
    build(**vars(parser.parse_args()))
