"""Freeze the self-contained, default-off v2.1.0 clip-error Colab notebook."""

import argparse
import json
from pathlib import Path
import subprocess

from scripts.v210_tail import (EXPECTED, FILES, GROUPS, INPUT_BYTES, INPUT_SHA, MIB, OUTPUT_CAP, SUMMARY_SHA,
                               WORKER_SECONDS, canonical, sha)


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = "v210_clip_errors_cpu_20261122.ipynb"


def cell(kind, name, content):
    result = {"cell_type": kind, "id": name, "metadata": {}, "source": content.splitlines(keepends=True)}
    if kind == "code":
        result.update(execution_count=None, outputs=[])
    return result


def build(output):
    worker = (ROOT / "scripts/v210_tail.py").read_text(encoding="utf-8")
    controller = (ROOT / "scripts/v210_tail_colab.py").read_text(encoding="utf-8")
    compile(worker, "embedded worker", "exec")
    pins = {"input_sha256": INPUT_SHA, "input_bytes": INPUT_BYTES, "source_summary_sha256": SUMMARY_SHA,
            "worker_sha256": sha(worker.encode()), "controller_sha256": sha(controller.encode()),
            "output_names": list(FILES), "counts": {"groups": len(GROUPS),
                                                    "clips": sum(count for count, _ in EXPECTED.values()),
                                                    "real_frames": sum(frames for _, frames in EXPECTED.values())},
            "limits": {"worker_seconds": WORKER_SECONDS, "total_seconds": 360,
                       "result_file_bytes": OUTPUT_CAP, "scratch_free_bytes": 64 * MIB}}
    settings = '''RUN_CLIP_ERRORS = False
DRIVE_PROJECT = "/content/drive/MyDrive/REWIND-like prototype"
'''
    command = controller + f'''
WORKER_SOURCE = {worker!r}
PINS = {pins!r}

if RUN_CLIP_ERRORS:
    run_tail_notebook("/content/v210_groups.zip", DRIVE_PROJECT, WORKER_SOURCE, PINS)
else:
    print("RUN_CLIP_ERRORS가 꺼져 있습니다.")
'''
    compile(settings, "notebook settings", "exec")
    compile(command, "notebook worker controller", "exec")
    readme = f'''# v2.1.0 클립별 오차와 꼬리 진단

CPU 런타임에서 회수본 `v210_groups.zip` ({INPUT_BYTES:,}바이트)만 `/content`에 올립니다.
SHA-256은 `{INPUT_SHA}`입니다.
프로젝트 경로를 확인하고 `RUN_CLIP_ERRORS = True`로 바꾼 다음 아래 셀을 순서대로 실행합니다.
Drive는 결과 보관에 사용합니다. 마지막에 표시되는 `result.json` 경로와 SHA-256을 회수합니다.
'''
    notebook = canonical({"nbformat": 4, "nbformat_minor": 5, "metadata": {
        "accelerator": "CPU", "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}},
        "cells": [cell("markdown", "intro", readme), cell("code", "settings", settings),
                  cell("code", "run", command)]})
    base = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    receipt = {"schema": "rewind_v210_clip_errors_colab_release_v1", "default_off": True, "base_revision": base,
               "notebook": {"name": NOTEBOOK, "bytes": len(notebook), "sha256": sha(notebook)}, "pins": pins}
    output.mkdir()
    (output / NOTEBOOK).write_bytes(notebook)
    (output / "receipt.json").write_bytes(canonical(receipt))
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.output)["notebook"]))


if __name__ == "__main__":
    main()
