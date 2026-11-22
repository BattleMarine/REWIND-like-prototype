"""Issue fixed A/B replacement notebooks from the unchanged 256 run contract."""

import argparse
import ast
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
import zipfile

from rewind_archive.contracts import canonical
from scripts.archvis_pilot_bootstrap import extract_pinned_zip
from scripts.build_v2_gpu_training import ROOT, RUN, write_source_zip
from scripts.build_archvis_pilot_handoff import _cell


ORIGINAL_RECEIPT_SHA = "e571cdf31c6ede59a77c2ac175f7346d80bc8a1339624b901baade26af1620aa"
ORIGINAL_SOURCE_SHA = "2ace5591e9911b12b18dffa966e8f2da88cbefc40d54b48b4295af1130796d7f"
REVISION = "v2_gpu_training_256_parallel_r2"
SOURCE = f"{REVISION}.zip"
WORKER = "scripts.run_v2_gpu_training_256_parallel"
CACHE_OVERLAY = "scripts/v2_gpu_training_cache256_parallel.py"
OVERLAYS = {
    CACHE_OVERLAY: "5624b31fc3b2e49fb2e1a1d12a912f5fa01ce67eb35dceeb846d9b11548d8d05",
    "scripts/run_v2_gpu_training_256_parallel.py": "52ffdc29dc8aa76d4f9492df2f6ce34ba7b7d6b74e2c494936aa75020fbd65c5",
}
SLOTS = {"A": 42, "B": 71}
MIB = 1024**2


def pin(raw):
    return {"bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}


def loader_settings(raw):
    assignments = {node.targets[0].id: ast.literal_eval(node.value)
                   for node in ast.parse(raw).body
                   if isinstance(node, ast.Assign) and len(node.targets) == 1
                   and isinstance(node.targets[0], ast.Name) and node.targets[0].id in {"WORKERS", "WINDOW"}}
    if assignments != {"WORKERS": 4, "WINDOW": 8}:
        raise ValueError("v2_gpu256_parallel_loader_settings_changed")
    return {"workers": assignments["WORKERS"], "lookahead_frames": assignments["WINDOW"]}


def build_notebook(slot, pins, delivery):
    settings = f'''RUN_V2_TRAINING = False
SLOT = "{slot}"  # Fixed replacement for {slot}-seed-{SLOTS[slot]}.
DRIVE_PROJECT = "/content/drive/MyDrive/REWIND-like prototype"
EXPECTED_RESUME_TIP = ""  # Required: latest step receipt content_sha256 for this slot.
RECOVER_WRITER_OWNER = ""  # Only after confirming the previous worker/VM is no longer active.
'''
    patch = {
        'if SLOT not in ("A", "B", "C"):\n        raise ValueError("Choose slot A, B or C")':
            'if SLOT != FIXED_SLOT:\n        raise ValueError("This notebook is fixed to its original slot")\n'
            '    if not re.fullmatch(r"[0-9a-f]{64}", EXPECTED_RESUME_TIP):\n'
            '        raise ValueError("Enter this slot\'s latest step receipt content_sha256 before resuming")',
        "import json\nimport signal": "import hashlib\nimport json\nimport re\nimport signal",
        "rewind_v2_gpu_training_128_v1": "rewind_v2_gpu_training_256_v1",
        'upload = Path("/content") / PINS["source_name"]': 'upload = Path("/content") / DELIVERY["name"]',
        'extract_pinned_zip(upload, PINS["source_sha256"], source,': 'extract_pinned_zip(upload, DELIVERY["sha256"], source,',
        '    pin_file = invocation / "pins.json"':
            '    for name, expected in DELIVERY["overlay_files"].items():\n'
            '        raw = (source / name).read_bytes()\n'
            '        current = {"bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}\n'
            '        if current != expected:\n'
            '            raise ValueError(f"Verified overlay installation failed: {name}")\n'
            '    print("Verified source archive and overlay", DELIVERY["sha256"], flush=True)\n'
            '    print("Original training identity source", PINS["source_sha256"], flush=True)\n'
            '    pin_file = invocation / "pins.json"',
        '"scripts.run_v2_gpu_training"': f'"{WORKER}"',
    }
    run = RUN
    for before, after in patch.items():
        if run.count(before) != 1:
            raise ValueError("v2_gpu256_parallel_notebook_template_drift")
        run = run.replace(before, after)
    bootstrap = (ROOT / "scripts/archvis_pilot_bootstrap.py").read_text(encoding="utf-8")
    code = bootstrap + f"\nFIXED_SLOT = {slot!r}\nPINS = {pins!r}\nDELIVERY = {delivery!r}\n" + run
    intro = f"""# 256 GPU 학습 교체본 · {slot} 슬롯 (seed {SLOTS[slot]})

기존 {slot} 실행의 체크포인트에서 재개하는 R2 교체본입니다. 두 슬롯의 기존 Drive 경로를 유지합니다.
현재 {slot} worker가 정상 종료·백업을 마친 뒤 더 이상 쓸 수 없는지 확인하세요.
같은 슬롯의 최신 step 영수증에서 content_sha256을 가져와 EXPECTED_RESUME_TIP에 넣습니다.
{SOURCE}를 /content에 올립니다. 실행은 기본 OFF입니다.
"""
    compile(settings, f"{slot} settings", "exec")
    compile(code, f"{slot} replacement notebook", "exec")
    return canonical({"nbformat": 4, "nbformat_minor": 5, "metadata": {"accelerator": "GPU",
                                  "kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"}},
                      "cells": [_cell("markdown", "guide", intro), _cell("code", "settings", settings),
                                _cell("code", "train", code)]})


def check_source(old, new, expected_modules):
    with zipfile.ZipFile(old) as baseline, zipfile.ZipFile(new) as release:
        old_names = set(baseline.namelist())
        names = release.namelist()
        if len(set(names)) != len(names) or set(names) != old_names | expected_modules.keys():
            raise ValueError("v2_gpu256_parallel_source_layout")
        for name in old_names:
            if baseline.read(name) != release.read(name):
                raise ValueError(f"v2_gpu256_parallel_baseline_changed: {name}")
        for name, expected in expected_modules.items():
            if pin(release.read(name)) != expected:
                raise ValueError(f"v2_gpu256_parallel_overlay_pin: {name}")


def build(original_source, original_receipt, output):
    if output.exists() or output.is_symlink() or not output.parent.is_dir():
        raise ValueError("v2_gpu256_parallel_new_output_directory")
    old_receipt = original_receipt.read_bytes()
    if pin(old_receipt)["sha256"] != ORIGINAL_RECEIPT_SHA:
        raise ValueError("v2_gpu256_parallel_original_receipt")
    old = json.loads(old_receipt)
    pins = old["pins"]
    if pins["source_sha256"] != ORIGINAL_SOURCE_SHA or pins["size"] != 256 or original_source.stat().st_size != old["source_bytes"]:
        raise ValueError("v2_gpu256_parallel_original_source_pin")
    overlays = {name: (ROOT / name).read_bytes() for name in OVERLAYS}
    modules = {name: pin(raw) for name, raw in overlays.items()}
    if {name: item["sha256"] for name, item in modules.items()} != OVERLAYS:
        raise ValueError("v2_gpu256_parallel_final_overlay_sha256")
    settings = loader_settings(overlays[CACHE_OVERLAY])
    with tempfile.TemporaryDirectory(dir=output.parent, prefix=".gpu256-parallel-") as directory:
        root, release = Path(directory) / "source", Path(directory) / "release"
        release.mkdir()
        extract_pinned_zip(original_source, ORIGINAL_SOURCE_SHA, root, max_zip_bytes=64*MIB,
                           max_output_bytes=192*MIB, max_members=3000, revision=pins["revision"])
        for name, raw in overlays.items():
            with (root / name).open("xb") as handle:
                handle.write(raw)
        archive = release / SOURCE
        write_source_zip(root, archive)
        check_source(original_source, archive, modules)
        transport = {"name": SOURCE, **pin(archive.read_bytes()), "overlay_files": modules}
        notebooks = {}
        for slot, seed in SLOTS.items():
            name = f"{REVISION}-{slot}.ipynb"
            raw = build_notebook(slot, pins, transport)
            (release / name).write_bytes(raw)
            notebooks[slot] = {"name": name, "slot": slot, "seed": seed, "default_off": True,
                               "latest_resume_receipt_required": True, **pin(raw)}
        guide = (ROOT / "docs/V2_GPU_TRAINING_256_PARALLEL.md").read_bytes()
        (release / "GUIDE_KO.md").write_bytes(guide)
        base_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
        receipt = {"schema": "rewind_v2_gpu256_parallel_release_v1", "release_revision": REVISION,
                   "workspace_base_commit": base_commit, "original_receipt_sha256": ORIGINAL_RECEIPT_SHA,
                   "runtime_pins": pins, "source": transport, "notebooks": notebooks,
                   "worker_module": WORKER, "loader": settings,
                   "guide": {"name": "GUIDE_KO.md", **pin(guide)}}
        (release/"receipt.json").write_bytes(canonical(receipt))
        release.rename(output)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("original-source", "original-receipt", "output"):
        parser.add_argument(f"--{name}", required=True, type=Path)
    doc = build(**vars(parser.parse_args()))
    print(json.dumps({"source": doc["source"], "notebooks": doc["notebooks"]}))


if __name__ == "__main__":
    main()
