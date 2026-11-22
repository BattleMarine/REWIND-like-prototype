"""Build the small pinned v2 tuning CPU Colab notebook and deployable kit."""

import argparse
import json
from pathlib import Path
import subprocess
import tempfile
import zipfile

from scripts.v2_gpu128_a_bundle_format import (BundleView, INDEX, MIB, SOURCE, SOURCE_RECEIPT, SOURCE_RECEIPT_SHA,
                                               SOURCE_SHA, canonical, sha)
from scripts.v2_tuning_cpu_common import DESIGN, JOBS, ROOT_INDEX_SHA, require


ROOT = Path(__file__).resolve().parents[1]
KIT = "v2_tuning_cpu_20261122_kit.zip"
NOTEBOOK = "v2_tuning_cpu_20261122.ipynb"
CPU_MODULES = ("common", "inputs", "reference", "selection", "trials", "worker")
COMMON_TOOLS = ("v2_gpu128_a_bundle_format.py", "v2_gpu128_a_bundle_local.py")
SOURCES = (SOURCE, SOURCE_RECEIPT)


def cell(kind, name, source):
    result = {"cell_type": kind, "metadata": {}, "id": name, "source": source.splitlines(keepends=True)}
    if kind == "code":
        result.update(execution_count=None, outputs=[])
    return result


def payload(base):
    with BundleView(base) as view:
        raw = view.read(INDEX, 2 * MIB)
        require(sha(raw) == ROOT_INDEX_SHA, "A index SHA mismatch")
        index = view.index()
        members = index["members"]
        require(members[SOURCE]["sha256"] == SOURCE_SHA and members[SOURCE_RECEIPT]["sha256"] == SOURCE_RECEIPT_SHA,
                "A source pins disagree")
        files = {INDEX: raw}
        for key in SOURCES:
            files[key] = view.read(key, 32 * MIB, members[key])
        for name in COMMON_TOOLS:
            key = f"tools/{name}"
            files[f"scripts/{name}"] = view.read(key, 2 * MIB, members[key])
    for module in CPU_MODULES:
        key = f"scripts/v2_tuning_cpu_{module}.py"
        files[key] = (ROOT / key).read_text(encoding="utf-8").encode()
    for key, raw in files.items():
        if key.endswith(".py"):
            compile(raw, key, "exec")
    return files, json.loads(files[SOURCE_RECEIPT])["pins"]


def build(base, output):
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    files, upstream = payload(base)
    build_head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    design_sha = subprocess.check_output(["git", "rev-parse", DESIGN], cwd=ROOT, text=True).strip()
    controller = (ROOT / "scripts/v2_tuning_cpu_colab.py").read_text(encoding="utf-8")
    bootstrap = (ROOT / "scripts/archvis_pilot_bootstrap.py").read_text(encoding="utf-8")
    with tempfile.TemporaryDirectory(dir=output.parent, prefix=".v2-tuning-cpu-build-") as temp:
        staging = Path(temp) / "release"
        staging.mkdir()
        with zipfile.ZipFile(staging / KIT, "x") as archive:
            for name, raw in sorted(files.items()):
                archive.writestr(zipfile.ZipInfo(name), raw)
        kit_raw = (staging / KIT).read_bytes()
        pins = {"kit_sha256": sha(kit_raw), "source_name": SOURCE, "source_sha256": SOURCE_SHA,
                "source_revision": upstream["revision"]}
        settings = f'''RUN_CPU = False
JOB_NAMES = ["inputs", "reference", "trials"]
DRIVE_PROJECT = "/content/drive/MyDrive/REWIND-like prototype"
KIT_UPLOAD = "/content/{KIT}"
'''
        invoke = f'''
if RUN_CPU:
    run_cpu_notebook(JOB_NAMES, DRIVE_PROJECT, KIT_UPLOAD, {pins!r}, {JOBS!r})
else:
    print("RUN_CPU가 꺼져 있습니다.")
'''
        source = bootstrap + "\n" + controller + "\n" + invoke
        compile(settings, NOTEBOOK + " settings", "exec")
        compile(source, NOTEBOOK, "exec")
        intro = """# v2 파생 시험 CPU 검사

Colab의 CPU 런타임을 사용해 주세요. 동봉한 약 24 MiB kit ZIP을 `/content`에 올립니다.
부모 파일은 기존 Drive 프로젝트에서 읽고, 결과는 `reports/v2_tuning_cpu_20261122/run-*/`에 저장합니다.
세 검사는 서로 입력·출력 의존성이 없습니다. 한 런타임에서 한 번에 실행하거나 여러 CPU 런타임에서 검사를 나눌 수 있습니다.
마지막 인쇄에 표시되는 `invocation-result.json`의 경로와 SHA-256을 회수합니다.
"""
        notebook = canonical({"nbformat": 4, "nbformat_minor": 5,
                              "metadata": {"accelerator": "CPU", "kernelspec": {
                                  "display_name": "Python 3", "language": "python", "name": "python3"}},
                              "cells": [cell("markdown", "guide", intro), cell("code", "settings", settings),
                                        cell("code", "cpu-controller", source)]})
        (staging / NOTEBOOK).write_bytes(notebook)
        guide = (ROOT / "docs/V2_TUNING_CPU_HANDOFF.md").read_text(encoding="utf-8").encode()
        (staging / "GUIDE_KO.md").write_bytes(guide)
        receipt = {"schema": "rewind_v2_tuning_cpu_release_v1", "design_revision": design_sha,
                   "workspace_base_commit": build_head, "status": "default_off", "base_index_sha256": ROOT_INDEX_SHA,
                   "pins": pins, "jobs": JOBS,
                   "files": {key: {"bytes": len(raw), "sha256": sha(raw)} for key, raw in
                             ((NOTEBOOK, notebook), (KIT, kit_raw), ("GUIDE_KO.md", guide))},
                   "kit_members": {key: {"bytes": len(raw), "sha256": sha(raw)} for key, raw in files.items()}}
        (staging / "receipt.json").write_bytes(canonical(receipt))
        require(not output.exists(), "CPU release output must be new")
        staging.rename(output)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--a-bundle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    receipt = build(args.a_bundle, args.output)
    print(json.dumps(receipt["files"], ensure_ascii=False))


if __name__ == "__main__":
    main()
