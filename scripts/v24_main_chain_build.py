"""Issue a notebook wrapper around the two fixed v2.4 metrics R2 profiles."""

import argparse
import ast
import hashlib
import json
from pathlib import Path
import tempfile
import zipfile

from scripts.v24_main_chain import V24_CHAIN_OVERHEAD_SECONDS, v24_chain_digest


ROOT = Path(__file__).resolve().parents[1]
BASES = {
    "T4": ("v24_main_t4_metrics_r2", "v24_main128_A_paired_t4_metrics_r2.ipynb",
           "72859f1c88571f3908fa8885fe7d248f6acc73a8e507b9999f3d73e0dcf3587b"),
    "L4": ("v24_main_l4_metrics_r2", "v24_main128_A_paired_candidate_metrics_r2.ipynb",
           "98d9e4c44bff7764fe544eb83a7bf9770cb9c8803e4e2c458c112865648103f0"),
}
TOOLKIT_SHA = "71819e873749c2178b7c820a3abd040adce967b0d1b71e6e8445540dfba70e6f"
NOTEBOOK = "v24_main128_A_contiguous_metrics_r2_20261123.ipynb"
RETURN_LINE = '    return {"path": str(output / "result.json"), "status": result["status"], **pin}\n'


def file_pin(raw):
    return {"bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}


def read_profile(parent, name):
    dirname, notebook, receipt_sha = BASES[name]
    directory = parent / dirname
    raw = (directory / "receipt.json").read_bytes()
    if file_pin(raw)["sha256"] != receipt_sha:
        raise ValueError(f"Unexpected {name} metrics R2 release receipt")
    receipt = json.loads(raw)
    pins = receipt["pins"]
    notebook_raw = (directory / notebook).read_bytes()
    protocol_raw = (directory / "candidate-protocol.json").read_bytes()
    kit_path = directory / pins["toolkit"]["name"]
    kit_raw = kit_path.read_bytes()
    kit_pin = file_pin(kit_raw)
    if (file_pin(notebook_raw) != receipt["files"][notebook]
            or file_pin(protocol_raw) != receipt["files"]["candidate-protocol.json"]
            or file_pin(protocol_raw)["sha256"] != v24_chain_digest(pins["protocol"])
            or file_pin(protocol_raw)["sha256"] != pins["protocol_sha256"] or kit_pin["sha256"] != TOOLKIT_SHA
            or kit_pin != receipt["files"][kit_path.name]
            or kit_pin != {k: pins["toolkit"][k] for k in ("bytes", "sha256")}):
        raise ValueError(f"Metrics R2 file or protocol mismatch for {name}")
    with zipfile.ZipFile(kit_path) as archive:
        if len(archive.namelist()) != len(pins["modules"]) or {n: file_pin(archive.read(n))["sha256"] for n in archive.namelist()} != pins["modules"]:
            raise ValueError(f"Unexpected metrics R2 module bytes for {name}")
    cells = json.loads(notebook_raw)["cells"]
    code = "".join(next(c for c in cells if c["id"] == "run")["source"])
    original, binding = code.split("\nPINS = ", 1)
    if ast.literal_eval(binding.splitlines()[0]) != pins:
        raise ValueError(f"Embedded metrics R2 pins mismatch for {name}")
    evidence = {"directory": dirname, "receipt_sha256": receipt_sha, "notebook_sha256": file_pin(notebook_raw)["sha256"],
                "protocol_sha256": pins["protocol_sha256"], "toolkit": pins["toolkit"]}
    return original, pins, evidence


def cell(kind, name, source):
    value = {"cell_type": kind, "id": name, "metadata": {}, "source": source.splitlines(keepends=True)}
    if kind == "code":
        value.update(execution_count=None, outputs=[])
    return value


def build(parent, output):
    t4_code, t4, t4_evidence = read_profile(parent, "T4")
    l4_code, l4, l4_evidence = read_profile(parent, "L4")
    if t4_code != l4_code or t4["modules"] != l4["modules"] or t4["base"] != l4["base"] or t4["groups"] != l4["groups"]:
        raise ValueError("Metrics R2 profiles do not share a controller, worker or inputs")
    controller = (ROOT / "scripts/v24_main_colab.py").read_text(encoding="utf-8")
    if not controller.endswith(RETURN_LINE) or not t4_code.endswith(controller.removesuffix(RETURN_LINE)):
        raise ValueError("Only a direct published-receipt return can differ from the pinned metrics R2 controller")
    chain = (ROOT / "scripts/v24_main_chain.py").read_text(encoding="utf-8")
    settings = '''RUN_V24_CHAIN = False
GPU_PROFILE = ""  # "T4" or "L4"
DRIVE_PROJECT = "/content/drive/MyDrive/REWIND-like prototype"
ARM = ""  # "x0" or "v"
SEED = None
APPROVED_BOUNDARIES = ()
APPROVED_PROTOCOL_SHA256 = ""
RESUME_RESULT = ""
RESUME_RESULT_SHA256 = ""
APPROVED_EXTENSION = False
'''
    code = t4_code + RETURN_LINE + "\n" + chain + f'''
METRICS_R2_PROFILES = {{"T4": {t4!r}, "L4": {l4!r}}}
if RUN_V24_CHAIN:
    if GPU_PROFILE not in METRICS_R2_PROFILES:
        raise ValueError("Select the approved metrics R2 T4 or L4 profile")
    run_v24_chain(DRIVE_PROJECT, ARM, SEED, APPROVED_BOUNDARIES, RESUME_RESULT, RESUME_RESULT_SHA256,
                  APPROVED_PROTOCOL_SHA256, APPROVED_EXTENSION, METRICS_R2_PROFILES[GPU_PROFILE], run_segment=run_v24_main)
else:
    print("v2.4 연속 구간 실행 승인 대기")
'''
    compile(settings, "v24 chain settings", "exec")
    compile(code, "v24 chain controller", "exec")
    intro = "# v2.4 metrics R2 연속 구간\n\n별도 승인된 한 arm·seed의 연속 경계를 실행 셀 한 번으로 진행합니다. 기본 OFF입니다. GUIDE_KO.md를 확인합니다.\n"
    doc = {"nbformat": 4, "nbformat_minor": 5, "metadata": {"accelerator": "GPU", "kernelspec": {
        "display_name": "Python 3", "language": "python", "name": "python3"}},
        "cells": [cell("markdown", "guide", intro), cell("code", "settings", settings), cell("code", "chain", code)]}
    files = {NOTEBOOK: json.dumps(doc, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(),
             "GUIDE_KO.md": (ROOT / "docs/V24_DIFFUSION_MAIN_CHAIN.md").read_bytes()}
    receipt = {"schema": "rewind_v24_main_chain_release_v1", "status": "awaiting_execution_approval", "default_off": True,
               "base_profiles": {"T4": t4_evidence, "L4": l4_evidence}, "chain_overhead_seconds": V24_CHAIN_OVERHEAD_SECONDS,
               "files": {name: file_pin(raw) for name, raw in files.items()}}
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    with tempfile.TemporaryDirectory(dir=output.parent, prefix=".v24-chain-") as tmp:
        stage = Path(tmp) / "release"
        stage.mkdir()
        for name, raw in files.items():
            (stage / name).write_bytes(raw)
        (stage / "receipt.json").write_bytes(json.dumps(receipt, sort_keys=True, separators=(",", ":")).encode())
        stage.rename(output)
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    print(json.dumps(build(**vars(parser.parse_args())), ensure_ascii=False))
