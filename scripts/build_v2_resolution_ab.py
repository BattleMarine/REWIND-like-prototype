"""Freeze the isolated v2 stereo 128/256 Colab experiment."""

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
from scripts.build_v2_bulk_cache_budgeted import SOURCE_IDENTITY
from scripts.build_v2_gpu_cache_probe import (DEFAULT_REPORT_REMAINING, DEFAULT_WAIT_BLOCK,
                                              REVISION, RUN_TEMPLATE, SOURCE_SHA, input_pins)
from scripts.build_v2_gpu_cache_restore import OVERLAY_BLOCK
from scripts.run_v2_cpu_cache import ARCHIVE_CAP, MANIFEST_CAP, MANIFEST_NAME, read_manifest
from scripts.run_v2_resolution_ab import SECONDS, SOURCE_CAP, selection, selection_pin


OVERLAY = ("rewind_v1/v2_cache.py", "rewind_v1/v2_retention.py", "rewind_v1/v2_training.py",
           "scripts/v2_ordered_prefetch.py", "scripts/run_v2_resolution_ab.py")
COMMAND = '''        data = project / "datasets" / "UnrealEgo2"
        command = [sys.executable, "-m", "scripts.run_v2_resolution_ab",
                   "--pins", str(pin_file), "--work", str(invocation), "--seconds", str(remaining),
                   "--rgb", str(data / "UnrealEgoData2_train_val_rgb"),
                   "--pose", str(data / "UnrealEgoData2_train_val_pose"),
                   "--drive-results", str(retained)]
'''


def notebook(pins, overlay):
    bootstrap = Path(__file__).with_name("archvis_pilot_bootstrap.py").read_text(encoding="utf-8")
    run = RUN_TEMPLATE.substitute(
        entrypoint="run_resolution_ab", work_folder="rewind_v2_resolution_ab",
        report_folder="v2_resolution_ab_041837Z", driver_block=OVERLAY_BLOCK,
        command_block=COMMAND, wait_block=DEFAULT_WAIT_BLOCK, run_switch="RUN_RESOLUTION_AB",
        report_remaining=DEFAULT_REPORT_REMAINING,
        success_rule='"passed" if error is None and worker and worker["status"] == "passed"',
    )
    guide = ("# v2 스테레오 캐시 128 / 256 비교\n\n"
             f"운영 캐시와 **별도의 CPU Colab**에서 `{pins['source_name']}`을 /content에 올립니다. "
             "Drive 프로젝트 경로 확인 후 기본 OFF 설정을 켜면, 고정 manifest의 공식 train 128페어씩을 비교합니다. "
             "같은 원본을 한 번만 로컬에 저장하고 각 해상도를 비교합니다. "
             "운영 Drive I/O와 동시에 실행하면 초기 읽기 시간은 서로 영향을 받을 수 있습니다. "
             "결과 ZIP은 reports/cache/v2_resolution_ab_041837Z에 저장합니다.\n")
    setup = 'RUN_RESOLUTION_AB = False\nDRIVE_PROJECT = "/content/drive/MyDrive/REWIND-like prototype"\n'
    return {"nbformat": 4, "nbformat_minor": 5, "metadata": {
                "kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"}},
            "cells": [_cell("markdown", "guide", guide), _cell("code", "settings", setup), _cell(
                "code", "run", bootstrap + f"\nPINS = {pins!r}\nOVERLAY = {overlay!r}\nLIMIT = {SECONDS}\n" + run)]}


def build(source, catalog, cpu_report, output):
    if output.exists() or not output.parent.is_dir():
        raise ValueError("resolution_ab_new_output")
    pins, _ = input_pins(source, catalog, cpu_report)
    repo = Path(__file__).resolve().parent.parent
    overlay = {name: (repo/name).read_text(encoding="utf-8") for name in OVERLAY}
    with tempfile.TemporaryDirectory(dir=output.parent) as temp:
        temp = Path(temp)
        root = temp/"source"
        extract_pinned_zip(source, SOURCE_SHA, root, max_zip_bytes=64*1024**2, max_output_bytes=192*1024**2,
                           max_members=3000, revision=REVISION)
        for name, text in overlay.items():
            (root/name).write_bytes(text.encode("utf-8"))
        process = subprocess.run([sys.executable, "-m", "rewind_v1.v2_cli", "source"], cwd=root, capture_output=True,
                                 text=True, timeout=60, check=True)
        pins["consumer_source_identity"] = json.loads(process.stdout)["source_sha256"]
        if pins["consumer_source_identity"] != SOURCE_IDENTITY:
            raise ValueError("resolution_ab_existing_source_identity")
        extract_pinned_zip(root/pins["embedded_manifest"], pins["archive_sha256"], temp/"manifest",
                           max_zip_bytes=ARCHIVE_CAP, max_output_bytes=MANIFEST_CAP, max_members=1)
        manifest, _ = read_manifest(temp/"manifest"/MANIFEST_NAME, pins, root/"reference")
        pins["sample"] = selection_pin(selection(manifest["records"]))
        if sum(item["source_bytes"] for item in pins["sample"].values()) > SOURCE_CAP:
            raise ValueError("resolution_ab_source_cap")
        pins["consumer_overlay"] = {name: sha(text.encode("utf-8")) for name, text in overlay.items()}
        raw = canonical(notebook(pins, overlay))
        stage = temp/"publish"
        stage.mkdir()
        (stage/"v2_resolution_ab_cpu.ipynb").write_bytes(raw)
        (stage/"receipt.json").write_bytes(canonical({
            "pins": pins, "notebook_sha256": sha(raw), "bytes": len(raw), "default_off": True,
        }))
        stage.rename(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "catalog", "cpu-report", "output"):
        parser.add_argument("--"+name, type=Path, required=True)
    build(**vars(parser.parse_args()))
