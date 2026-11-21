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
from scripts.v2_cpu_cache_presets import (ACCELERATED, NINETY_MINUTES, OVERNIGHT, PREFETCH_ACCELERATED, PRESETS,
                                         READ_8H, READ_90, SESSION_90, STANDARD, WRITE_8H, WRITE_90)


SOURCE_IDENTITY = "2e31d7365b55f802dbd80c3ee5fbc65cff5d650be8e3a3ca74fe63a8ec4949ef"
RESUME_REPORT_SHA = "6e9e32258d11938392912fbe2d48dd635a28a4f300c370a58e3306c86aa60cb6"
PRE_RESERVATION_SECONDS = 240
EVIDENCE_SHA = {
    4: "df49ae6ea5817cc7fbbcc21cdcfd8449722c31f1f6eece85dd46f41f89d2c291",
    8: "3cdf2a25af1e16f40ef3f38ae41c9630d5c625e4d5911181767bc0327ed31aef",
    "latest": "3a9b99e0f4adde44137faffcd9d3588464db31ae8df036d40b02c69ad242ab31",
}
OPERATING_BOUNDARY = "11b19dc1d82ecf58872b55c446da68e22eb84ccd5825dc800e91b9ea893dde00"
OVERNIGHT_BOUNDARY = "ce372854fa8dfb7521a9ed4cbe48426cab3832fe6d318a7a044963e77de36bd1"
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


def notebook(pins, overlay, *, preset=STANDARD):
    accelerated = preset in ACCELERATED
    command = COMMAND
    if accelerated:
        command += f'        command += ["--preset", "{preset}"]\n'
    bootstrap = Path(__file__).with_name("archvis_pilot_bootstrap.py").read_text(encoding="utf-8")
    run = RUN_TEMPLATE.substitute(
        entrypoint="run_v2_bulk_budgeted", work_folder="rewind_v2_bulk_budgeted",
        report_folder="v2_bulk_cpu_budget_reports_041837Z", driver_block=OVERLAY_BLOCK,
        command_block=command, wait_block=WAIT, run_switch="RUN_APPROVED_CACHE",
        report_remaining="diagnostic_seconds(invocation, started)",
        success_rule='worker["status"] if error is None and worker and worker["status"] in ("passed", "paused")',
    )
    guide = ("# v2 확정 예산 CPU 캐시 재개\n\n"
             f"CPU 런타임을 고르고 `{pins['source_name']}`을 /content에 올립니다. "
             "기존 4샤드 뒤부터 1,063샤드까지 같은 Drive 경계에서 재개합니다. "
             "기본 OFF인 실행 스위치를 켜면 한 번 실행합니다. 종료 후 보고서와 budget tip SHA를 회수합니다.\n")
    if accelerated:
        label, boundary = ("90분", 228) if preset == NINETY_MINUTES else ("8시간", 245)
        guide = (f"# v2 CPU 캐시 {label} 재개\n\n"
                 f"CPU 런타임을 선택하고 `{pins['source_name']}`을 /content에 올립니다. "
                 f"직전 실제 호출의 최신 budget tip SHA를 입력합니다. 샤드 {boundary} 경계는 검증할 최소 기준입니다. "
                 "실행은 기본 OFF이고 한 번만 호출합니다. 종료 후 새 tip과 진단을 회수합니다.\n")
    settings = ('RUN_APPROVED_CACHE = False\nDRIVE_PROJECT = "/content/drive/MyDrive/REWIND-like prototype"\n'
                'EXPECTED_BUDGET_TIP_SHA256 = ""\nRECOVER_WRITER_OWNER = ""\n'
                f'PREFETCH_WORKERS = {PREFETCH_ACCELERATED if accelerated else 4}\n')
    constants = (f"\nLIMIT = {SESSION_90 if preset == NINETY_MINUTES else SESSION_CAP}\nCLOSURE = {CLOSURE}\n"
                 f"PRE_RESERVATION_SECONDS = {PRE_RESERVATION_SECONDS}\n"
                 + REPORT_LIMIT)
    return {"nbformat": 4, "nbformat_minor": 5,
            "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"}},
            "cells": [_cell("markdown", "guide", guide), _cell("code", "settings", settings),
                      _cell("code", "budget", bootstrap + f"\nPINS = {pins!r}\nOVERLAY = {overlay!r}\n" + constants + run)]}


def performance_evidence(path, sample):
    raw = _regular_read(path, 64 * 1024)
    if sha(raw) != EVIDENCE_SHA[sample]:
        raise ValueError("v2_quota_performance_report_pin")
    doc = json.loads(raw)
    worker = doc["worker"]
    evidence, usage = worker["performance"], worker["usage"]
    workers = 4 if sample == 4 else 8
    completed = {4: 211, 8: 228, "latest": 245}[sample]
    if (doc["pins"]["consumer_source_identity"] != SOURCE_IDENTITY or worker["stop_reason"] != "read_budget"
            or worker["completed_shards"] != completed or worker["starting_shards"] != completed - 17
            or worker["added_shards"] != 17 or evidence["requested_workers"] != workers
            or evidence["newly_encoded_frames"] != 2176):
        raise ValueError("v2_quota_performance_report_scope")
    return {"sha256": EVIDENCE_SHA[sample], "completed_shards": worker["completed_shards"],
            "boundary_sha256": worker["boundary_sha256"], "added_shards": worker["added_shards"],
            "append_seconds": evidence["append_seconds"], "source_bytes": evidence["source_bytes"],
            "read_bytes_observed": usage["read_bytes_observed"], "write_bytes_observed": usage["write_bytes_observed"],
            "read_bytes_reserved": usage["read_bytes_reserved"], "write_bytes_reserved": usage["write_bytes_reserved"]}


def build(source, catalog, cpu_report, resume_report, output, *, preset=STANDARD,
          evidence_4=None, evidence_8=None, evidence_overnight=None):
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
    if preset in ACCELERATED:
        if evidence_4 is None or evidence_8 is None:
            raise ValueError("v2_quota_performance_reports_required")
        samples = {"4": performance_evidence(evidence_4, 4), "8": performance_evidence(evidence_8, 8)}
        if samples["8"]["boundary_sha256"] != OPERATING_BOUNDARY:
            raise ValueError("v2_quota_operating_boundary")
        if preset == OVERNIGHT:
            if evidence_overnight is None:
                raise ValueError("v2_quota_overnight_report_required")
            samples["latest"] = performance_evidence(evidence_overnight, "latest")
            if samples["latest"]["boundary_sha256"] != OVERNIGHT_BOUNDARY:
                raise ValueError("v2_quota_overnight_boundary")
            anchor, seconds, read, write = "latest", SESSION_CAP, READ_8H, WRITE_8H
        else:
            if evidence_overnight is not None:
                raise ValueError("v2_quota_overnight_report_preset")
            anchor, seconds, read, write = "8", SESSION_90, READ_90, WRITE_90
        pins.update(invocation_preset=preset, invocation_seconds=seconds, invocation_read_bytes=read,
                    invocation_write_bytes=write, performance_evidence=samples,
                    operating_baseline={k: samples[anchor][k] for k in ("completed_shards", "boundary_sha256")})
    elif preset != STANDARD or any(item is not None for item in (evidence_4, evidence_8, evidence_overnight)):
        raise ValueError("v2_quota_invocation_preset_arguments")
    repo = Path(__file__).resolve().parent.parent
    names = (*MODULES, "scripts/run_v2_bulk_cache_budgeted.py", "scripts/v2_bulk_quota.py",
             "scripts/v2_ordered_prefetch.py", "scripts/v2_cpu_cache_presets.py")
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
        raw = canonical(notebook(pins, overlay, preset=preset))
        stage = Path(temp) / "publish"
        stage.mkdir()
        name = f"v2_bulk_cache_{preset}_cpu.ipynb" if preset in ACCELERATED else "v2_bulk_cache_budgeted_cpu.ipynb"
        (stage / name).write_bytes(raw)
        (stage / "receipt.json").write_bytes(canonical({
            "pins": pins, "notebook_sha256": sha(raw), "bytes": len(raw), "default_off": True,
        }))
        stage.rename(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "catalog", "cpu-report", "resume-report", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--preset", choices=PRESETS, default=STANDARD)
    parser.add_argument("--evidence-4", type=Path)
    parser.add_argument("--evidence-8", type=Path)
    parser.add_argument("--evidence-overnight", type=Path)
    build(**vars(parser.parse_args()))
