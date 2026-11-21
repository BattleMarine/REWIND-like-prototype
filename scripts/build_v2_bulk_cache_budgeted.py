"""Issue the frozen continuation notebook for the approved 90h / 60GiB budget."""

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
from scripts.build_v2_bulk_cache import MODULES
from scripts.build_v2_gpu_cache_probe import REVISION, RUN_TEMPLATE, SOURCE_SHA, input_pins
from scripts.build_v2_gpu_cache_restore import OVERLAY_BLOCK
from scripts.v2_bulk_quota import BASELINE_SHA, CLOSURE, DRIVE_CAP, SESSION_CAP, TARGET_SHARDS, TIME_CAP


SOURCE_IDENTITY = "2e31d7365b55f802dbd80c3ee5fbc65cff5d650be8e3a3ca74fe63a8ec4949ef"
RESUME_REPORT_SHA = "6e9e32258d11938392912fbe2d48dd635a28a4f300c370a58e3306c86aa60cb6"
PRE_RESERVATION_SECONDS = 240
COMMAND = '''        data = project / "datasets" / "UnrealEgo2"
        command = [sys.executable, "-m", "scripts.run_v2_bulk_cache_budgeted",
                   "--pins", str(pin_file), "--work", str(invocation), "--started", str(started),
                   "--legacy", str(parent / "v2_cpu_cache_041837Z"), "--bulk", str(parent / "v2_bulk_cache_041837Z"),
                   "--rgb", str(data / "UnrealEgoData2_train_val_rgb"),
                   "--pose", str(data / "UnrealEgoData2_train_val_pose"),
                   "--budget-tip", EXPECTED_BUDGET_TIP_SHA256, "--recover-writer", RECOVER_WRITER_OWNER,
                   "--prefetch-workers", str(PREFETCH_WORKERS)]
'''
WAIT = '''            deadline = started + PRE_RESERVATION_SECONDS
            granted = False
            try:
                while child.poll() is None:
                    grant = invocation / "quota-grant.json"
                    if not granted and grant.is_file():
                        data = json.loads(grant.read_bytes())
                        deadline = float(data["deadline"]) - CLOSURE
                        if not started < deadline < started + LIMIT:
                            raise ValueError("Invalid worker reservation deadline")
                        granted = True
                    left = deadline - time.monotonic()
                    if left <= 0:
                        raise TimeoutError("CPU cache reservation deadline reached")
                    try:
                        child.wait(timeout=min(1, left))
                    except subprocess.TimeoutExpired:
                        continue
                code = child.returncode
            except BaseException:
                if child.poll() is None:
                    child.send_signal(signal.SIGINT)
                try:
                    child.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    child.kill()
                    child.wait(timeout=10)
                raise
'''
REPORT_LIMIT = '''
def diagnostic_seconds(invocation, started):
    grant = invocation / "quota-grant.json"
    deadline = started + PRE_RESERVATION_SECONDS
    if grant.is_file():
        value = float(json.loads(grant.read_bytes())["deadline"])
        if started < value <= started + LIMIT:
            deadline = value
    return min(60, int(deadline - time.monotonic()))
'''


def notebook(pins, overlay):
    bootstrap = Path(__file__).with_name("archvis_pilot_bootstrap.py").read_text(encoding="utf-8")
    run = RUN_TEMPLATE.substitute(
        entrypoint="run_v2_bulk_budgeted", work_folder="rewind_v2_bulk_budgeted",
        report_folder="v2_bulk_cpu_budget_reports_041837Z", driver_block=OVERLAY_BLOCK,
        command_block=COMMAND, wait_block=WAIT, run_switch="RUN_APPROVED_CACHE",
        report_remaining="diagnostic_seconds(invocation, started)",
        success_rule='worker["status"] if error is None and worker and worker["status"] in ("passed", "paused")',
    )
    guide = ("# v2 확정 예산 CPU 캐시 재개\n\n"
             f"CPU 런타임을 고르고 `{pins['source_name']}`을 /content에 올립니다. "
             "기존 4샤드 뒤부터 1,063샤드까지 같은 Drive 경계에서 재개합니다. "
             "기본 OFF인 실행 스위치를 켜면 한 번 실행합니다. 종료 후 보고서와 budget tip SHA를 회수합니다.\n")
    settings = ('RUN_APPROVED_CACHE = False\nDRIVE_PROJECT = "/content/drive/MyDrive/REWIND-like prototype"\n'
                'EXPECTED_BUDGET_TIP_SHA256 = ""\nRECOVER_WRITER_OWNER = ""\nPREFETCH_WORKERS = 4\n')
    constants = (f"\nLIMIT = {SESSION_CAP}\nCLOSURE = {CLOSURE}\nPRE_RESERVATION_SECONDS = {PRE_RESERVATION_SECONDS}\n"
                 + REPORT_LIMIT)
    return {"nbformat": 4, "nbformat_minor": 5,
            "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"}},
            "cells": [_cell("markdown", "guide", guide), _cell("code", "settings", settings),
                      _cell("code", "budget", bootstrap + f"\nPINS = {pins!r}\nOVERLAY = {overlay!r}\n" + constants + run)]}


def build(source, catalog, cpu_report, resume_report, output):
    if output.exists() or not output.parent.is_dir():
        raise ValueError("v2_quota_new_output")
    pins, _ = input_pins(source, catalog, cpu_report)
    recovered = _regular_read(resume_report, 64 * 1024)
    if sha(recovered) != RESUME_REPORT_SHA:
        raise ValueError("v2_quota_resume_report_pin")
    evidence = json.loads(recovered)["worker"]
    if (evidence["boundary_sha256"] != BASELINE_SHA or evidence["completed_shards"] != 4
            or evidence["completed_frames"] != 512 or evidence["status"] != "passed"):
        raise ValueError("v2_quota_resume_report_boundary")
    pins.update(baseline_boundary_sha256=BASELINE_SHA, resume_report_sha256=RESUME_REPORT_SHA,
                approved_target_shards=TARGET_SHARDS, approved_time_seconds=TIME_CAP, approved_drive_bytes=DRIVE_CAP)
    repo = Path(__file__).resolve().parent.parent
    names = (*MODULES, "scripts/run_v2_bulk_cache_budgeted.py", "scripts/v2_bulk_quota.py", "scripts/v2_ordered_prefetch.py")
    overlay = {name: (repo / name).read_text(encoding="utf-8") for name in names}
    with tempfile.TemporaryDirectory(dir=output.parent) as temp:
        root = Path(temp) / "source"
        extract_pinned_zip(source, SOURCE_SHA, root, max_zip_bytes=64*1024**2, max_output_bytes=192*1024**2,
                           max_members=3000, revision=REVISION)
        for name, text in overlay.items():
            (root / name).write_bytes(text.encode("utf-8"))
        proc = subprocess.run([sys.executable, "-m", "rewind_v1.v2_cli", "source"], cwd=root, capture_output=True,
                              text=True, timeout=60, check=True)
        pins["consumer_source_identity"] = json.loads(proc.stdout)["source_sha256"]
        if pins["consumer_source_identity"] != SOURCE_IDENTITY:
            raise ValueError("v2_quota_existing_producer_identity")
        pins["consumer_overlay"] = {name: sha(text.encode("utf-8")) for name, text in overlay.items()}
        raw = canonical(notebook(pins, overlay))
        stage = Path(temp) / "publish"
        stage.mkdir()
        (stage / "v2_bulk_cache_budgeted_cpu.ipynb").write_bytes(raw)
        (stage / "receipt.json").write_bytes(canonical({
            "pins": pins, "notebook_sha256": sha(raw), "bytes": len(raw), "default_off": True,
        }))
        stage.rename(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "catalog", "cpu-report", "resume-report", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    build(**vars(parser.parse_args()))
