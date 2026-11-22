"""Build the default-off one-time A-128 recovery notebook and matching driver."""

import argparse
import hashlib
import json
from pathlib import Path
import tempfile

from rewind_archive.contracts import canonical
from scripts.build_archvis_pilot_handoff import _cell


ROOT = Path(__file__).resolve().parents[1]
DRIVER = "recover_v2_gpu128_a_6200.py"
NOTEBOOK = "v2_gpu128_a_recovery_6200.ipynb"
BASELINE_ZIP_SHA = "9a68d8e45bcf3abf6fab3004740faa6805d4ce3fda5118c3a7102011937849d8"
BASELINE_RECEIPT_SHA = "7a699521ec937b4481ec544f9d1e01fc7a7f7abb58d20442afab86c85e73d43d"
SETTINGS = '''RUN_RECOVERY = False
PUBLISH_VERIFIED_CANDIDATE = False
DRIVE_PROJECT = "/content/drive/MyDrive/REWIND-like prototype"
'''

RUN = '''
import signal
import subprocess
import sys
import tempfile
import time

def recover_once():
    started = time.monotonic()
    mount = Path("/content/drive")
    project = Path(DRIVE_PROJECT)
    if project == mount or not project.is_relative_to(mount) or ".." in project.parts:
        raise ValueError("Set the project directory below the Drive mount")
    upload = Path("/content") / PINS["source_name"]
    if not upload.is_file() or upload.is_symlink():
        raise FileNotFoundError(f"Upload the existing fixed 128 ZIP to {upload}")
    from google.colab import drive
    drive.mount(str(mount), timeout_ms=120_000)
    project = project.resolve(strict=True)
    if not project.is_relative_to(mount.resolve(strict=True)):
        raise ValueError("The project resolved outside the Drive mount")
    work = Path(tempfile.mkdtemp(prefix="v2-128-A-recovery-", dir="/content"))
    print("Local recovery evidence:", work / "recovery-result.json", flush=True)
    source = work / "frozen-128"
    extract_pinned_zip(upload, PINS["source_sha256"], source, max_zip_bytes=64*1024**2,
                       max_output_bytes=192*1024**2, max_members=3000, revision=PINS["revision"])
    content = DRIVER_TEXT.encode("utf-8")
    if hashlib.sha256(content).hexdigest() != PINS["driver_sha256"]:
        raise ValueError("Recovery driver SHA mismatch")
    driver = work / "recover_v2_gpu128_a_6200.py"
    with driver.open("xb") as stream:
        stream.write(content)
    command = [sys.executable, "-u", "-c",
               "import runpy, sys; runpy.run_path(sys.argv.pop(1), run_name='__main__')",
               str(driver), "--project", str(project), "--work", str(work)]
    if PUBLISH_VERIFIED_CANDIDATE:
        command.append("--publish")
    remaining = 880-(time.monotonic()-started)
    if remaining <= 0:
        raise TimeoutError("Recovery cell time limit")
    with subprocess.Popen(command, cwd=source) as worker:
        try:
            status = worker.wait(timeout=remaining)
        except BaseException:
            if worker.poll() is None:
                worker.send_signal(signal.SIGINT)
            try:
                worker.wait(timeout=10)
            except subprocess.TimeoutExpired:
                worker.kill()
                worker.wait(timeout=5)
            raise
    if status:
        raise RuntimeError(f"Recovery exited with status {status}")

if RUN_RECOVERY:
    recover_once()
else:
    print("복구 스위치가 꺼져 있습니다.")
'''


def build(source, source_receipt, output):
    raw = source_receipt.read_bytes()
    if hashlib.sha256(raw).hexdigest() != BASELINE_RECEIPT_SHA:
        raise ValueError("recovery_original_source_receipt")
    receipt = json.loads(raw)["pins"]
    if (source.name != receipt["source_name"] or hashlib.sha256(source.read_bytes()).hexdigest() != BASELINE_ZIP_SHA
            or receipt["source_sha256"] != BASELINE_ZIP_SHA or output.exists() or not output.parent.is_dir()):
        raise ValueError("recovery_original_zip_or_output")
    driver = (ROOT/"scripts"/DRIVER).read_text(encoding="utf-8")
    bootstrap = (ROOT/"scripts/archvis_pilot_bootstrap.py").read_text(encoding="utf-8")
    driver_sha = hashlib.sha256(driver.encode("utf-8")).hexdigest()
    pins = {"source_name": receipt["source_name"], "source_sha256": BASELINE_ZIP_SHA,
            "revision": receipt["revision"], "driver_sha256": driver_sha}
    body = bootstrap + f"\nPINS = {pins!r}\nDRIVER_TEXT = {driver!r}\n" + RUN
    compile(driver, DRIVER, "exec")
    compile(SETTINGS, "settings", "exec")
    compile(body, NOTEBOOK, "exec")
    guide = ("# v2 128 A · 6,200 업데이트 복구\n\n"
             "CPU 런타임의 `/content`에 기존 `v2_gpu_training_128_v1.zip`을 올립니다. "
             "A의 학습 worker가 종료된 상태에서 `RUN_RECOVERY`를 켭니다. 기본 모드는 검증만 수행합니다. "
             "게시가 승인된 경우에만 `PUBLISH_VERIFIED_CANDIDATE`를 켭니다. "
             "복구 셀이 출력한 `normal_resume_tip`은 원래 128 A 노트북의 재개 설정에 사용합니다.\n")
    notebook = canonical({"nbformat": 4, "nbformat_minor": 5,
                          "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"}},
                          "cells": [_cell("markdown", "guide", guide), _cell("code", "settings", SETTINGS), _cell("code", "recover", body)]})
    with tempfile.TemporaryDirectory(dir=output.parent, prefix=".gpu128-recovery-") as directory:
        staged = Path(directory)/"release"
        staged.mkdir()
        (staged/DRIVER).write_bytes(driver.encode("utf-8"))
        (staged/NOTEBOOK).write_bytes(notebook)
        (staged/"receipt.json").write_bytes(canonical({**pins, "source_receipt_sha256": BASELINE_RECEIPT_SHA,
            "driver": DRIVER, "notebook": NOTEBOOK, "notebook_sha256": hashlib.sha256(notebook).hexdigest(),
            "default_off": True, "default_publish": False}))
        staged.rename(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "source-receipt", "output"):
        parser.add_argument("--"+name, required=True, type=Path)
    build(**vars(parser.parse_args()))
