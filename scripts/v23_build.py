"""Freeze the v2.3 Colab notebook, independent overlay and A input pins."""

import argparse
import json
from pathlib import Path
import subprocess
import tempfile
import zipfile

from scripts.v210_tail import INPUT_SHA as V210_GROUPS_SHA, SUMMARY_SHA as V210_SUMMARY_SHA, canonical, sha
from scripts.v210_tail_build import cell


ROOT = Path(__file__).resolve().parents[1]
V210_RECEIPT_SHA = "ac766ffff915349fbbc3528337dec3fb3ac0d86152611d8bd9011f92c5eee2dd"
NOTEBOOK = "v23_ddim_128_A_cuda_20261122.ipynb"
TOOLKIT = "v23_ddim_128_A_tools.zip"
MIB = 1024**2


def build(source, reference_dir, summary, output):
    source_raw = (reference_dir / "receipt.json").read_bytes()
    if sha(source_raw) != V210_RECEIPT_SHA:
        raise ValueError("v23_requires_final_v210_release_receipt")
    prior = json.loads(source_raw)
    pins = prior["pins"]
    if source.name != pins["base"]["source_name"] or sha(source.read_bytes()) != pins["base"]["source_sha256"]:
        raise ValueError("v23_parent_source_SHA")
    summary_raw = summary.read_bytes()
    if sha(summary_raw) != V210_SUMMARY_SHA:
        raise ValueError("v23_archived_v210_summary_SHA")
    historical = json.loads(summary_raw)
    if (historical["binding"]["code"] != pins["modules"]
            or historical["counts"] != {"clips": 421, "groups": 18, "real_frames": 16425}):
        raise ValueError("v23_archived_v210_summary_binding")
    raw_tools = (reference_dir / prior["tools"]).read_bytes()
    if sha(raw_tools) != prior["tools_sha256"]:
        raise ValueError("v23_v210_overlay_SHA")
    with zipfile.ZipFile(reference_dir / prior["tools"]) as archive:
        modules = {name: archive.read(name) for name in pins["modules"]}
    if {name: sha(raw) for name, raw in modules.items()} != pins["modules"]:
        raise ValueError("v23_v210_overlay_member_SHA")
    for name in ("v23_eval.py", "v23_run.py"):
        modules[name] = (ROOT / "scripts" / name).read_text(encoding="utf-8").encode()
    for name, raw in modules.items():
        compile(raw, name, "exec")
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    with tempfile.TemporaryDirectory(dir=output.parent, prefix=".v23-release-") as temp:
        release = Path(temp) / "release"
        release.mkdir()
        with zipfile.ZipFile(release / TOOLKIT, "x") as archive:
            for name, raw in sorted(modules.items()):
                archive.writestr(zipfile.ZipInfo(name), raw, compress_type=zipfile.ZIP_DEFLATED)
        tools = (release / TOOLKIT).read_bytes()
        fixed = {**pins, "modules": {name: sha(raw) for name, raw in modules.items()},
                 "reference_10": {"summary_file_sha256": V210_SUMMARY_SHA, "groups_zip_sha256": V210_GROUPS_SHA,
                                  "gpu": historical["environments"][0]["gpu"],
                                  "macro_mm": historical["metrics"]["macro_score_mm"]},
                 "toolkit": {"name": TOOLKIT, "bytes": len(tools), "sha256": sha(tools)},
                 "limits": {"seconds": 3600, "work_reserve_seconds": 150, "worker_read_bytes": 2048 * MIB,
                            "worker_logical_write_bytes": 2048 * MIB, "result_file_bytes": 2 * MIB,
                            "scratch_free_bytes": 4096 * MIB},
                 "planned_drive_payload_bytes": {"checkpoint": 435178624,
                                                 "validation_feature_groups": sum(g["archive"]["bytes"]
                                                                                  for g in pins["groups"].values())},
                 "output_folder": "v23_ddim128_a_v1"}
        settings = '''RUN_V23 = False
DRIVE_PROJECT = "/content/drive/MyDrive/REWIND-like prototype"
'''
        bootstrap = (ROOT / "scripts/archvis_pilot_bootstrap.py").read_text(encoding="utf-8")
        helpers = (ROOT / "scripts/v210_tail_colab.py").read_text(encoding="utf-8").split("\ndef run_tail_notebook", 1)[0]
        controller = (ROOT / "scripts/v23_colab.py").read_text(encoding="utf-8")
        code = bootstrap + "\n" + helpers + "\n" + controller + f'''
PINS = {fixed!r}
if RUN_V23:
    run_v23_notebook(DRIVE_PROJECT, PINS)
else:
    print("RUN_V23가 꺼져 있습니다.")
'''
        compile(settings, "v2.3 settings", "exec")
        compile(code, NOTEBOOK, "exec")
        intro = f"""# v2.3 A DDIM 정확도·지연시간 비교

CUDA GPU 하나를 선택합니다. 고정 A의 body와 hand에 10·5·2 step을 적용합니다.
전체 공식 validation 품질과 고정 3×50프레임 bundle에서 설정당 warm-up 1회와 동기화 시간 5회를 기록합니다.
`{source.name}`과 `{TOOLKIT}`을 /content에 올립니다. 설정은 기본 OFF입니다.
"""
        notebook = canonical({"nbformat": 4, "nbformat_minor": 5, "metadata": {"accelerator": "GPU",
                                  "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}},
                              "cells": [cell("markdown", "guide", intro), cell("code", "settings", settings),
                                        cell("code", "run", code)]})
        (release / NOTEBOOK).write_bytes(notebook)
        guide = (ROOT / "docs/V23_DDIM.md").read_bytes()
        (release / "GUIDE_KO.md").write_bytes(guide)
        revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
        receipt = {"schema": "rewind_v23_colab_release_v1", "default_off": True, "base_revision": revision,
                   "v210_release_receipt_sha256": V210_RECEIPT_SHA, "pins": fixed,
                   "files": {name: {"bytes": len(raw), "sha256": sha(raw)} for name, raw in
                             ((NOTEBOOK, notebook), (TOOLKIT, tools), ("GUIDE_KO.md", guide))}}
        (release / "receipt.json").write_bytes(canonical(receipt))
        release.rename(output)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "reference-dir", "summary", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    print(json.dumps(build(**vars(parser.parse_args()))["files"]))


if __name__ == "__main__":
    main()
