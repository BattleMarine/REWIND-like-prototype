"""Issue the v2.4 numerical and short paired pilot notebook with pinned A inputs."""

import argparse
import json
from pathlib import Path
import subprocess
import tempfile
import zipfile

from scripts import v2_gpu128_a_bundle_format as fmt
from scripts.v210_tail_build import cell


ROOT = Path(__file__).resolve().parents[1]
A_INDEX_SHA = "4a4e1c119e08f0026ef4af9a87c8edbb757934fbcee7ace1b61c2e424e080f40"
PARENT_HELPERS_RECEIPT_SHA = "60a66d4917286b51b695f7b445c50b391b6accd53b49118d2d245b6644925aaa"
NOTEBOOK = "v24_diffusion_pilot_128_A_20261122.ipynb"
TOOLKIT = "v24_diffusion_pilot_tools.zip"


def build(source, a_bundle, parent_release, output):
    def script(name):
        return (ROOT / "scripts" / name).read_text(encoding="utf-8")

    known = (parent_release / "receipt.json").read_bytes()
    if fmt.sha(known) != PARENT_HELPERS_RECEIPT_SHA:
        raise ValueError("v24_frozen_parent_helper_receipt")
    prior = json.loads(known)["pins"]
    if source.name != prior["base"]["source_name"] or fmt.sha(source.read_bytes()) != fmt.SOURCE_SHA:
        raise ValueError("v24_frozen_source_pin")
    prior_zip = parent_release / prior["toolkit"]["name"]
    if fmt.sha(prior_zip.read_bytes()) != prior["toolkit"]["sha256"]:
        raise ValueError("v24_frozen_helpers_zip_pin")
    with zipfile.ZipFile(prior_zip) as archive:
        modules = {name: archive.read(name) for name in prior["modules"]}
    if {name: fmt.sha(raw) for name, raw in modules.items()} != prior["modules"]:
        raise ValueError("v24_frozen_helpers_member_pin")
    modules.update({name: script(name).encode() for name in ("v24_body.py", "v24_run.py")})
    for name, data in modules.items():
        compile(data, name, "exec")
    with fmt.BundleView(a_bundle) as view:
        if fmt.sha(view.read(fmt.INDEX, 2 * fmt.MIB)) != A_INDEX_SHA:
            raise ValueError("v24_fixed_A_index")
        members = view.index()["members"]
        groups = {}
        for i, (marker, (_, _, archive_pin, _)) in enumerate(fmt.feature_groups(view, members).items()):
            raw = json.loads(view.read(marker, 64 * 1024, members[marker]))
            if marker != f"{fmt.FEATURES}/group-{i:05d}.json" or archive_pin["sha256"] != raw["archive"]["sha256"]:
                raise ValueError("v24_feature_group_pin")
            groups[str(i)] = {"marker": members[marker], "archive": raw["archive"]}
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    with tempfile.TemporaryDirectory(dir=output.parent, prefix=".v24-pilot-") as tmp:
        staging = Path(tmp) / "release"
        staging.mkdir()
        with zipfile.ZipFile(staging / TOOLKIT, "x") as archive:
            for name, data in sorted(modules.items()):
                archive.writestr(zipfile.ZipInfo(name), data, compress_type=zipfile.ZIP_DEFLATED)
        toolkit = (staging / TOOLKIT).read_bytes()
        pins = {"base": prior["base"], "evidence": prior["evidence"], "groups": groups,
                "modules": {name: fmt.sha(raw) for name, raw in modules.items()}, "A_index_sha256": A_INDEX_SHA,
                "toolkit": {"name": TOOLKIT, "bytes": len(toolkit), "sha256": fmt.sha(toolkit)},
                "limits": {"pilot_seconds": 2700, "numerical_seconds": 300, "pilot_worker_terminal_reserve_seconds": 300,
                           "pilot_read_bytes": 2 * fmt.GIB, "pilot_logical_write_bytes": 3 * fmt.GIB,
                           "report_bytes": 2 * fmt.MIB, "state_bytes": 192 * fmt.MIB, "pilot_scratch_free_bytes": 4 * fmt.GIB,
                           "numerical_scratch_free_bytes": 256 * fmt.MIB},
                "scope": {"seed": 42, "updates_per_arm": 24, "train_clips": 24,
                          "validation_clips": 6, "validation_frames": 300, "evaluation_updates": [0, 12, 24]},
                "drive_payload_ceiling": {"checkpoint": 435178624, "selected_feature_zips": sum(sorted(
                    (g["archive"]["bytes"] for g in groups.values()), reverse=True)[:30])},
                "output_folder": "v24_diffusion_pilot128_a_v1"}
        settings = '''RUN_V24 = False
MODE = "numerical"
DRIVE_PROJECT = "/content/drive/MyDrive/REWIND-like prototype"
'''
        bootstrap = script("archvis_pilot_bootstrap.py")
        support = script("v210_tail_colab.py").split("\ndef run_tail_notebook", 1)[0]
        code = bootstrap + "\n" + support + "\n" + script("v24_colab.py") + f'''
PINS = {pins!r}
if RUN_V24:
    run_v24_notebook(DRIVE_PROJECT, MODE, PINS)
else:
    print("RUN_V24가 꺼져 있습니다.")
'''
        compile(settings, NOTEBOOK + " settings", "exec")
        compile(code, NOTEBOOK, "exec")
        intro = f"""# v2.4 x0/v 수치 검사와 paired 파일럿

`{source.name}` 및 `{TOOLKIT}`을 /content에 올립니다.
기본 OFF입니다. numerical 모드는 CPU 수치 검사, pilot 모드는 CUDA 한 대의 arm당 24 update입니다.
두 모드는 별도 UUID를 사용합니다. 본 비교 학습은 이 발급 범위 밖입니다.
"""
        doc = fmt.canonical({"nbformat": 4, "nbformat_minor": 5, "metadata": {"kernelspec": {
            "display_name": "Python 3", "language": "python", "name": "python3"}},
            "cells": [cell("markdown", "guide", intro), cell("code", "settings", settings), cell("code", "run", code)]})
        (staging / NOTEBOOK).write_bytes(doc)
        guide = (ROOT / "docs/V24_DIFFUSION_PILOT.md").read_bytes()
        (staging / "GUIDE_KO.md").write_bytes(guide)
        revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
        receipt = {"schema": "rewind_v24_pilot_release_v1", "default_off": True, "base_revision": revision,
                   "parent_helpers_receipt_sha256": PARENT_HELPERS_RECEIPT_SHA, "pins": pins,
                   "files": {name: {"bytes": len(raw), "sha256": fmt.sha(raw)} for name, raw in
                             ((NOTEBOOK, doc), (TOOLKIT, toolkit), ("GUIDE_KO.md", guide))}}
        (staging / "receipt.json").write_bytes(fmt.canonical(receipt))
        staging.rename(output)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "a-bundle", "parent-release", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    print(json.dumps(build(**vars(parser.parse_args()))["files"]))


if __name__ == "__main__":
    main()
