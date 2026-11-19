"""Freeze the read-only restoration notebook on the original ed98124 source ZIP."""

import argparse
import json
from pathlib import Path
from string import Template
import subprocess
import sys
import tempfile

from rewind_archive.contracts import canonical
from rewind_m1.local_rgb import _regular_read
from rewind_prepare.converter import sha
from scripts.archvis_pilot_bootstrap import extract_pinned_zip
from scripts.build_archvis_pilot_handoff import _cell
from scripts.build_v2_gpu_cache_probe import DEFAULT_COMMAND_BLOCK, REVISION, RUN_TEMPLATE, SOURCE_SHA, input_pins
from scripts.run_v2_gpu_cache_probe import MAX_SECONDS


MODULES = ("rewind_v1/v2_cache.py", "rewind_v1/v2_retention.py", "rewind_v1/v2_training.py",
           "scripts/run_v2_gpu_cache_probe.py", "scripts/run_v2_gpu_cache_restore.py")
GPU_REPORT_SHA = "8d2b5fefe7e88f6d1f5f5e943b9cf215ae48f247ceff7ae768aaed7a15038538"
OVERLAY_BLOCK = '''        for name, text in OVERLAY.items():
            raw = text.encode("utf-8")
            if hashlib.sha256(raw).hexdigest() != PINS["consumer_overlay"][name]:
                raise ValueError("Restoration source pin: " + name)
            (source / name).write_bytes(raw)
'''


def make_notebook(pins, overlay):
    bootstrap = Path(__file__).with_name("archvis_pilot_bootstrap.py").read_text(encoding="utf-8")
    run = RUN_TEMPLATE.substitute(
        entrypoint="run_gpu_cache_restore", work_folder="rewind_v2_gpu_cache_restore",
        report_folder="v2_gpu_restore_041837Z", driver_block=OVERLAY_BLOCK,
        command_block=Template(DEFAULT_COMMAND_BLOCK).substitute(worker_module="scripts.run_v2_gpu_cache_restore"),
        run_switch="RUN_GPU_CACHE_RESTORE",
        success_rule='"passed" if error is None and worker and worker["status"] == "passed"',
    )
    guide = (
        "# v2 첫 CPU 캐시 정식 복원 및 CUDA 배치 읽기\n\n"
        f"GPU 런타임을 고르고 `{pins['source_name']}` 하나를 /content에 업로드합니다. "
        "실행 스위치는 기본 OFF입니다. Drive 경로를 확인하고 켠 뒤 전체 실행합니다. "
        "첫 샤드를 정식 함수로 로컬 복원하고 표준 학습 프런트엔드 배치 2프레임을 CUDA 왕복 전송합니다. "
        "실제 학습과 전체 캐시 생성은 실행하지 않습니다. "
        "종료 시 Drive 진단 JSON 경로와 SHA가 표시됩니다. Drive 게시가 실패하면 로컬 진단 경로를 확인합니다.\n"
    )
    settings = 'RUN_GPU_CACHE_RESTORE = False\nDRIVE_PROJECT = "/content/drive/MyDrive/REWIND-like prototype"\n'
    return {
        "nbformat": 4, "nbformat_minor": 5,
        "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"}},
        "cells": [
            _cell("markdown", "guide", guide), _cell("code", "settings", settings),
            _cell("code", "restore", bootstrap + f"\nPINS = {pins!r}\nOVERLAY = {overlay!r}\nLIMIT = {MAX_SECONDS}\n" + run),
        ],
    }


def build(source, catalog, cpu_report, gpu_report, output):
    if output.exists() or not output.parent.is_dir():
        raise ValueError("v2_gpu_new_output")
    pins, _ = input_pins(source, catalog, cpu_report)
    raw_evidence = _regular_read(gpu_report, 192*1024)
    if sha(raw_evidence) != GPU_REPORT_SHA:
        raise ValueError("v2_gpu_restore_prior_test_pin")
    prior = json.loads(raw_evidence)["worker"]
    difference = {"torch": {"producer": "2.14.1+cpu", "runtime": "2.14.1+cu130"}}
    if (prior["strict_restore_binding"]["differences"] != difference or not prior["cuda"]["all_exact"]
            or prior["catalog_file_sha256"] != pins["catalog_file_sha256"]):
        raise ValueError("v2_gpu_restore_prior_test_scope")
    pins["prior_gpu_report_sha256"] = GPU_REPORT_SHA
    repo = Path(__file__).resolve().parent.parent
    overlay = {name: (repo / name).read_text(encoding="utf-8") for name in MODULES}
    with tempfile.TemporaryDirectory(dir=output.parent) as tmp:
        root = Path(tmp) / "source"
        extract_pinned_zip(source, SOURCE_SHA, root, max_zip_bytes=64*1024**2, max_output_bytes=192*1024**2,
                           max_members=3000, revision=REVISION)
        for name, text in overlay.items():
            (root / name).write_bytes(text.encode("utf-8"))
        runner = subprocess.run([sys.executable, "-m", "rewind_v1.v2_cli", "source"], cwd=root,
                                capture_output=True, text=True, timeout=60, check=True)
        pins["consumer_source_identity"] = json.loads(runner.stdout)["source_sha256"]
        pins["consumer_overlay"] = {name: sha(text.encode("utf-8")) for name, text in overlay.items()}
        raw = canonical(make_notebook(pins, overlay))
        stage = Path(tmp) / "publish"
        stage.mkdir()
        (stage / "v2_gpu_cache_restore.ipynb").write_bytes(raw)
        (stage / "receipt.json").write_bytes(canonical({
            "pins": pins, "notebook_sha256": sha(raw), "bytes": len(raw), "default_off": True,
        }))
        stage.rename(output)


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "catalog", "cpu-report", "gpu-report", "output"):
        cli.add_argument("--" + name, type=Path, required=True)
    build(**vars(cli.parse_args()))
