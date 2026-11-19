"""Build CPU append and CUDA boundary notebooks on the pinned original source ZIP."""

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
from scripts.build_v2_gpu_cache_probe import REVISION, RUN_TEMPLATE, SOURCE_SHA, input_pins
from scripts.build_v2_gpu_cache_restore import OVERLAY_BLOCK
from scripts.run_v2_bulk_cache import CPU_SECONDS, GPU_SECONDS


MODULES = ("rewind_v1/v2_cache.py", "rewind_v1/v2_retention.py", "rewind_v1/v2_training.py",
           "scripts/run_v2_gpu_cache_probe.py", "scripts/run_v2_bulk_cache.py")
COMMAND = '''        if remaining <= MINIMUM_SECONDS:
            raise TimeoutError("No time left for the cache worker")
        command = [sys.executable, "-m", "scripts.run_v2_bulk_cache", MODE,
                   "--pins", str(pin_file), "--work", str(invocation), "--seconds", str(remaining),
                   "--legacy", str(parent / "v2_cpu_cache_041837Z"), "--bulk", str(parent / "v2_bulk_cache_041837Z"),
                   "--boundary", EXPECTED_BOUNDARY_SHA256]
        if MODE == "cpu":
            data = project / "datasets" / "UnrealEgo2"
            command += ["--target", str(TARGET_TOTAL_SHARDS),
                        "--rgb", str(data / "UnrealEgoData2_train_val_rgb"),
                        "--pose", str(data / "UnrealEgoData2_train_val_pose")]
            if RECOVER_WRITER_OWNER:
                command += ["--recover-writer", RECOVER_WRITER_OWNER]
        else:
            command += ["--right", str(RIGHT_SHARD_INDEX)]
'''


def notebook(pins, overlay, mode):
    cpu = mode == "cpu"
    limit = CPU_SECONDS if cpu else GPU_SECONDS
    minimum = 2520 if cpu else 30
    bootstrap = Path(__file__).with_name("archvis_pilot_bootstrap.py").read_text(encoding="utf-8")
    setup = ('RUN_V2_BULK = False\nDRIVE_PROJECT = "/content/drive/MyDrive/REWIND-like prototype"\n'
             'EXPECTED_BOUNDARY_SHA256 = ""\n')
    if cpu:
        setup += "TARGET_TOTAL_SHARDS = 3  # includes the original first shard\n"
        setup += 'RECOVER_WRITER_OWNER = ""  # only after confirming the previous VM has stopped\n'
        description = ("현재 검증된 첫 샤드를 포함한 목표 누적 샤드 수를 지정합니다. "
                       "재개 시 이전 보고서의 boundary SHA를 입력합니다. 다음 샤드부터 예산 내에서 연속 보관합니다.")
    else:
        setup += "RIGHT_SHARD_INDEX = 1  # crossing indices 0 and 1\n"
        description = ("CPU 보고서의 완료 boundary SHA를 입력하고 읽을 오른쪽 샤드 번호를 지정합니다. "
                       "두 인접 샤드의 train 두 프레임을 정식 GPU 학습 배치 로더로 읽습니다.")
    run = RUN_TEMPLATE.substitute(
        entrypoint="run_v2_bulk", work_folder=f"rewind_v2_bulk_{mode}",
        report_folder=f"v2_bulk_{mode}_reports_041837Z", driver_block=OVERLAY_BLOCK,
        command_block=COMMAND, run_switch="RUN_V2_BULK",
        success_rule='worker["status"] if error is None and worker and worker["status"] in ("passed", "paused")',
    )
    guide = (f"# v2 캐시 {mode.upper()}\n\n`{pins['source_name']}`을 /content에 업로드하고 "
             f"{mode.upper()} 런타임을 선택합니다. {description} 기본 스위치는 꺼져 있습니다.\n")
    return {"nbformat": 4, "nbformat_minor": 5,
            "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"}},
            "cells": [_cell("markdown", "guide", guide), _cell("code", "settings", setup),
                      _cell("code", "run", bootstrap + f"\nPINS = {pins!r}\nOVERLAY = {overlay!r}\n"
                            + f"MODE = {mode!r}\nLIMIT = {limit}\nMINIMUM_SECONDS = {minimum}\n" + run)]}


def build(source, catalog, cpu_report, output):
    if output.exists() or not output.parent.is_dir():
        raise ValueError("v2_bulk_new_output")
    pins, _ = input_pins(source, catalog, cpu_report)
    repo = Path(__file__).resolve().parent.parent
    overlay = {name: (repo / name).read_text(encoding="utf-8") for name in MODULES}
    with tempfile.TemporaryDirectory(dir=output.parent) as temp:
        root = Path(temp) / "source"
        extract_pinned_zip(source, SOURCE_SHA, root, max_zip_bytes=64*1024**2, max_output_bytes=192*1024**2,
                           max_members=3000, revision=REVISION)
        for name, text in overlay.items():
            (root / name).write_bytes(text.encode("utf-8"))
        result = subprocess.run([sys.executable, "-m", "rewind_v1.v2_cli", "source"], cwd=root, capture_output=True,
                                text=True, timeout=60, check=True)
        pins["consumer_source_identity"] = json.loads(result.stdout)["source_sha256"]
        pins["consumer_overlay"] = {name: sha(text.encode("utf-8")) for name, text in overlay.items()}
        stage = Path(temp) / "publish"
        stage.mkdir()
        receipt = {"default_off": True, "pins": pins, "notebooks": {}}
        for mode in ("cpu", "gpu"):
            name = f"v2_bulk_cache_{mode}.ipynb"
            raw = canonical(notebook(pins, overlay, mode))
            (stage / name).write_bytes(raw)
            receipt["notebooks"][name] = {"bytes": len(raw), "sha256": sha(raw)}
        (stage / "receipt.json").write_bytes(canonical(receipt))
        stage.rename(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ("source", "catalog", "cpu-report", "output"):
        parser.add_argument("--" + arg, type=Path, required=True)
    build(**vars(parser.parse_args()))
