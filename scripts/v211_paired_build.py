"""Freeze the independent v2.1.0 vs v2.1.1 three-seed CPU notebook."""

import argparse
import json
from pathlib import Path
import subprocess

from scripts import v211_paired as paired
from scripts.v210_tail import INPUT_SHA, SUMMARY_SHA, canonical, sha
from scripts.v210_tail_build import cell


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = "v211_paired_clips_cpu_20261122.ipynb"


def build(output):
    def script(name):
        return (ROOT / "scripts" / name).read_text(encoding="utf-8")

    modules = {name: script(name) for name in ("v210_tail.py", "v211_paired.py")}
    support = script("v210_tail_colab.py").split("\ndef run_tail_notebook", 1)[0]
    controller = script("v211_paired_colab.py")
    pins = {"modules": {name: sha(source.encode()) for name, source in modules.items()},
            "controller_sha256": sha((support + "\n" + controller).encode()),
            "baseline_archive_sha256": INPUT_SHA, "baseline_summary_sha256": SUMMARY_SHA,
            "seed_summary_sha256": paired.SEED_SUMMARY_SHA, "outputs": list(paired.FILES),
            "limits": {"worker_seconds": paired.WORKER_SECONDS, "total_seconds": 420,
                       "input_read_bytes": paired.INPUT_BUDGET, "result_file_bytes": paired.OUTPUT_CAP,
                       "scratch_free_bytes": 128 * paired.MIB}}
    for name, source in modules.items():
        compile(source, name, "exec")
    settings = '''RUN_PAIRED = False
INPUT_MODE = "drive"
DRIVE_PROJECT = "/content/drive/MyDrive/REWIND-like prototype"
'''
    command = support + "\n" + controller + f'''
WORKER_MODULES = {modules!r}
PINS = {pins!r}

if RUN_PAIRED:
    run_paired_notebook(DRIVE_PROJECT, INPUT_MODE, WORKER_MODULES, PINS)
else:
    print("RUN_PAIRED가 꺼져 있습니다.")
'''
    compile(settings, "paired settings", "exec")
    compile(command, NOTEBOOK, "exec")
    intro = """# v2.1.0 A와 v2.1.1 세 seed의 같은 클립 비교

Colab CPU 런타임을 사용합니다. drive 모드는 기존 Drive 예측/영수증을 읽습니다.
zip 모드는 v210_groups.zip과 seed-42/, seed-71/, seed-103/를 담은 v211_groups.zip을 /content에 올립니다.
실행은 기본 OFF입니다. 결과를 보고 승인받은 뒤 설정 셀과 실행 셀을 순서대로 실행합니다.
"""
    doc = canonical({"nbformat": 4, "nbformat_minor": 5, "metadata": {"accelerator": "CPU", "kernelspec": {
        "display_name": "Python 3", "language": "python", "name": "python3"}},
        "cells": [cell("markdown", "guide", intro), cell("code", "settings", settings), cell("code", "run", command)]})
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    receipt = {"schema": "rewind_v211_paired_colab_release_v1", "base_revision": revision,
               "default_off": True, "notebook": {"name": NOTEBOOK, "bytes": len(doc), "sha256": sha(doc)},
               "pins": pins}
    output.mkdir()
    (output / NOTEBOOK).write_bytes(doc)
    (output / "receipt.json").write_bytes(canonical(receipt))
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.output)["notebook"]))


if __name__ == "__main__":
    main()
