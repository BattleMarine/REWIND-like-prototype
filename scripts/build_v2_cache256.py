"""Freeze independent 256px CPU pilot and manual long-continuation notebooks."""

import argparse
import json
from pathlib import Path
import subprocess
import sys
import tempfile

from rewind_archive.contracts import canonical
from rewind_prepare.converter import sha
from scripts.archvis_pilot_bootstrap import extract_pinned_zip
from scripts.build_archvis_pilot_handoff import _cell
from scripts.build_v2_bulk_cache_budgeted import PRE_RESERVATION_SECONDS, REPORT_LIMIT, SOURCE_IDENTITY, WAIT
from scripts.build_v2_gpu_cache_probe import REVISION, RUN_TEMPLATE, SOURCE_SHA, input_pins
from scripts.build_v2_gpu_cache_restore import OVERLAY_BLOCK
from scripts.build_v2_resolution_ab import OVERLAY
from scripts.run_v2_cache256 import CLOSURE, LONG_SECONDS, PILOT_SECONDS, REPORTS, STORE


MODULES = (*OVERLAY, "scripts/run_v2_bulk_cache.py", "scripts/run_v2_gpu_cache_probe.py",
           "scripts/v2_bulk_quota.py", "scripts/run_v2_cache256.py")
COMMAND = '''        data = project / "datasets" / "UnrealEgo2"
        command = [sys.executable, "-m", "scripts.run_v2_cache256", PINS["mode"],
                   "--pins", str(pin_file), "--work", str(invocation), "--started", str(started),
                   "--store", str(parent / STORE), "--rgb", str(data / "UnrealEgoData2_train_val_rgb"),
                   "--pose", str(data / "UnrealEgoData2_train_val_pose"),
                   "--budget-tip", EXPECTED_256_TIP_SHA256, "--recover-writer", RECOVER_256_WRITER_OWNER]
        if PINS["mode"] == "long":
            command += ["--pilot-receipt", VALIDATED_PILOT_RECEIPT_SHA256,
                        "--additional-seconds", str(CONFIRMED_ADDITIONAL_CPU_SECONDS),
                        "--additional-gib", str(CONFIRMED_ADDITIONAL_DRIVE_GIB)]
'''


def notebook(pins, overlay):
    pilot = pins["mode"] == "pilot"
    settings = ('RUN_CACHE256 = False\nDRIVE_PROJECT = "/content/drive/MyDrive/REWIND-like prototype"\n'
                'EXPECTED_256_TIP_SHA256 = ""\nRECOVER_256_WRITER_OWNER = ""\n')
    if not pilot:
        settings += ('VALIDATED_PILOT_RECEIPT_SHA256 = ""\nCONFIRMED_ADDITIONAL_CPU_SECONDS = 0\n'
                     'CONFIRMED_ADDITIONAL_DRIVE_GIB = 0\n')
    intro = ("별도 CPU Colab에서 최대 30분에 첫 3샤드를 생성·게시·다시 내려받아 내용까지 검증합니다."
             if pilot else
             "별도 CPU Colab에서 파일럿 검증 receipt, 최신 256 tip, 사용자 승인 추가 CPU 초·Drive GiB가 있어야 실행합니다. 최대 9시간 수동 단일 호출입니다.")
    boot = Path(__file__).with_name("archvis_pilot_bootstrap.py").read_text(encoding="utf-8")
    body = RUN_TEMPLATE.substitute(
        entrypoint="run_cache256", work_folder="rewind_v2_cache256", report_folder=REPORTS,
        driver_block=OVERLAY_BLOCK, command_block=COMMAND, wait_block=WAIT, run_switch="RUN_CACHE256",
        report_remaining="diagnostic_seconds(invocation, started)",
        success_rule='worker["status"] if error is None and worker and worker["status"] in ("passed", "paused")',
    )
    guide = ("# v2 256 전용 캐시\n\n"
             f"`{pins['source_name']}`을 /content에 올립니다. {intro} "
             "기본 OFF입니다. 기존 128 캐시 위치와 예산을 사용하지 않습니다.\n")
    constants = (f"\nLIMIT = {PILOT_SECONDS if pilot else LONG_SECONDS}\nCLOSURE = {CLOSURE}\n"
                 f"PRE_RESERVATION_SECONDS = {PRE_RESERVATION_SECONDS}\nSTORE = {STORE!r}\n")
    return {"nbformat": 4, "nbformat_minor": 5, "metadata": {
        "kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"}},
        "cells": [_cell("markdown", "guide", guide), _cell("code", "settings", settings), _cell(
            "code", "worker", boot + f"\nPINS = {pins!r}\nOVERLAY = {overlay!r}\n" + constants + REPORT_LIMIT + body)]}


def build(source, catalog, cpu_report, output):
    if output.exists() or not output.parent.is_dir():
        raise ValueError("cache256_new_output")
    pins, _ = input_pins(source, catalog, cpu_report)
    prior = pins["producer_environment"]
    pins["cpu_runtime"] = {k: prior[k] for k in ("torch", "numpy", "pillow")}
    repo = Path(__file__).resolve().parent.parent
    overlay = {name: (repo/name).read_text(encoding="utf-8") for name in MODULES}
    with tempfile.TemporaryDirectory(dir=output.parent) as tmp:
        tmp = Path(tmp)
        root = tmp/"source"
        extract_pinned_zip(source, SOURCE_SHA, root, max_zip_bytes=64*1024**2, max_output_bytes=192*1024**2,
                           max_members=3000, revision=REVISION)
        for name, text in overlay.items():
            (root/name).write_bytes(text.encode("utf-8"))
        process = subprocess.run([sys.executable, "-m", "rewind_v1.v2_cli", "source"], cwd=root, capture_output=True,
                                 text=True, timeout=60, check=True)
        pins["consumer_source_identity"] = json.loads(process.stdout)["source_sha256"]
        if pins["consumer_source_identity"] != SOURCE_IDENTITY:
            raise ValueError("cache256_existing_source_identity")
        subprocess.run([sys.executable, "-m", "scripts.run_v2_cache256", "--help"], cwd=root,
                       capture_output=True, timeout=60, check=True)
        pins["consumer_overlay"] = {name:sha(text.encode("utf-8")) for name,text in overlay.items()}
        stage = tmp/"publish"
        stage.mkdir()
        artifacts = {}
        for mode in ("pilot","long"):
            raw = canonical(notebook({**pins, "mode":mode}, overlay))
            name = f"v2_cache256_{mode}_cpu.ipynb"
            (stage/name).write_bytes(raw)
            artifacts[name] = {"sha256":sha(raw), "bytes":len(raw)}
        (stage/"receipt.json").write_bytes(canonical({"pins":pins,"notebooks":artifacts,"default_off":True}))
        stage.rename(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source","catalog","cpu-report","output"):
        parser.add_argument("--"+name,type=Path,required=True)
    build(**vars(parser.parse_args()))
