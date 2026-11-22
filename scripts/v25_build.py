"""Issue one v2.5 notebook with the verified head, control and overlay embedded."""

import argparse
import base64
import io
import json
from pathlib import Path
import tempfile
import zipfile
import zlib

from rewind_archive.contracts import canonical, digest
from rewind_prepare.temporal_selection import check_pin
from scripts.v210_tail_build import cell
from scripts.v211_trial import FIT_SCHEMA, HEAD, SUMMARY_SCHEMA
from scripts.v23_build import SUPERVISOR_RESERVE
from scripts.v23_run import RESULT_CAP
from scripts.v25_run import (COUNTS, DESTINATION, FEATURE, FINAL, NOISE_OFFSETS, READ_CAP, RESULTS, SECONDS, STEP,
                             TERMINAL_RESERVE, WORKSPACE_FREE, WRITE_CAP, fingerprint)


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = "v25_ddim2_selected_head_128_A_cuda.ipynb"
SOURCE = "9a68d8e45bcf3abf6fab3004740faa6805d4ce3fda5118c3a7102011937849d8"
RELEASE = "0cdca3e4a4b8bccf6cf3b698b0278cae234e287eb7b648a83895f0f0d7f5c113"
CONTROL = "46fa90bc192c1212d72154037a03cba5f17609ebc9e49d1ba83db20ab46ce21e"
FIT = "7ce420ae1cf1426ee52c9746d42cb8cf3a28084940bcafe2727446c51be33ae9"
SUMMARY = "e88911c078a1d6d0fec3db843114f08dc84c338592dfe81bfa3d437f46e0b4f3"


def verify(path, expected):
    raw = path.read_bytes()
    if fingerprint(raw) != expected:
        raise ValueError(f"v25_build_pin_mismatch: {path.name}")
    return raw


def project_source(name):
    return (ROOT/"scripts"/name).read_text(encoding="utf-8")


def collect(source, v23_release, v23_quality, v211):
    prior = json.loads(verify(v23_release/"receipt.json", RELEASE))
    frozen = prior["pins"]
    verify(source, SOURCE)
    if source.name != frozen["base"]["source_name"] or SOURCE != frozen["base"]["source_sha256"]:
        raise ValueError("v25_frozen_source_identity")
    control_raw = verify(v23_quality, CONTROL)
    control = json.loads(control_raw)
    noise = control["initial_noise_by_clip"]
    if (control["schema"] != "rewind_v23_quality_v1" or control["counts"] != COUNTS
            or control["evaluation_noise_seed"] != 71 or control["initial_noise_offsets"] != list(NOISE_OFFSETS)
            or control["binding"]["overlay"] != frozen["modules"] or control["binding"]["source"] != SOURCE
            or STEP not in control["sampling_steps"] or control["binding"]["checkpoint_receipt"] != FINAL
            or len(noise) != COUNTS["clips"] or sum(row["real_frames"] for row in noise) != COUNTS["real_frames"]
            or fingerprint(canonical(noise)) != control["noise_manifest_sha256"]):
        raise ValueError("v25_fixed_v23_validation_evidence")
    fit = json.loads(verify(v211/"seed-71-fit-03097.json", FIT))
    summary = json.loads(verify(v211/"seed-71-summary.json", SUMMARY))
    check_pin(fit, FIT_SCHEMA, fit["content_sha256"])
    check_pin(summary, SUMMARY_SCHEMA, summary["content_sha256"])
    artifact = fit["artifact"]
    head = verify(v211/"seed-71"/artifact["name"], artifact["sha256"])
    bound = summary["binding"]
    basis = {key: val for key, val in bound.items() if key not in ("fit", "body_sha256", "sampling_steps")}
    if (len(head) != artifact["bytes"] or fit["step"] != 3097 or bound["fit"] != fit["content_sha256"]
            or fit["binding_sha256"] != digest(basis) or bound["body_sha256"] != artifact["sha256"]
            or bound["method"] != "v2.1.1" or bound["seed"] != 71 or bound["trainable"] != list(HEAD)
            or bound["reference"] != FINAL or bound["feature_identity"] != FEATURE or bound["source"] != SOURCE
            or bound["original_model"] != control["binding"]["weights"] or summary["counts"] != COUNTS
            or bound["comparison_common"]["parent"] != control["binding"]["parent"]):
        raise ValueError("v25_seed71_final_head_evidence")
    toolkit = verify(v23_release/frozen["toolkit"]["name"], frozen["toolkit"]["sha256"])
    with zipfile.ZipFile(io.BytesIO(toolkit)) as archive:
        modules = {name: archive.read(name) for name in frozen["modules"]}
    if {name: fingerprint(raw) for name, raw in modules.items()} != frozen["modules"]:
        raise ValueError("v25_parent_overlay_members")
    for name in ("v211_trial.py", "v25_run.py"):
        modules[name] = project_source(name).encode()
    for name, raw in modules.items():
        compile(raw, name, "exec")
    pins = {key: frozen[key] for key in ("base", "evidence", "groups", "A_index_sha256") if key in frozen}
    pins.update({"modules": {name: fingerprint(raw) for name, raw in modules.items()}, "output_folder": DESTINATION,
                 "head": {**artifact, "method": "v2.1.1", "seed": 71, "step": 3097, "fit_receipt_file_sha256": FIT,
                          "summary_file_sha256": SUMMARY, "fit_content_sha256": fit["content_sha256"],
                          "backbone_sha256": fit["backbone_sha256"], "binding_sha256": fit["binding_sha256"]},
                 "result_names": list(RESULTS), "baseline": {"bytes": len(control_raw), "sha256": CONTROL,
                    "noise_sha256": control["noise_manifest_sha256"], "macro_mm": control["quality"]["2"]["macro_score_mm"]},
                 "limits": {"seconds": SECONDS, "supervisor_reserve_seconds": SUPERVISOR_RESERVE,
                            "work_reserve_seconds": TERMINAL_RESERVE, "result_file_bytes": RESULT_CAP,
                            "scratch_free_bytes": WORKSPACE_FREE, "logical_read_bytes": READ_CAP, "logical_write_bytes": WRITE_CAP}})
    return pins, {**modules, "selected-head.pt": head, "v23-quality.json": control_raw}


def build(source, v23_release, v23_quality, v211, output):
    pins, members = collect(source, v23_release, v23_quality, v211)
    embedded = {name: base64.b64encode(zlib.compress(raw, 9)).decode("ascii") for name, raw in members.items()}
    settings = 'RUN_V25 = False\nDRIVE_PROJECT = "/content/drive/MyDrive/REWIND-like prototype"\n'
    bootstrap = project_source("archvis_pilot_bootstrap.py")
    publisher = project_source("v210_tail_colab.py").split("\ndef run_tail_notebook", 1)[0]
    code = bootstrap+"\n"+publisher+"\n"+project_source("v25_colab.py")+f"""
PINS = {pins!r}
EMBEDDED = {embedded!r}
if RUN_V25:
    run_v25_notebook(DRIVE_PROJECT, PINS, EMBEDDED)
else:
    print("RUN_V25가 꺼져 있습니다.")
"""
    compile(settings, "v2.5 settings", "exec")
    compile(code, NOTEBOOK, "exec")
    intro = f"""# v2.5 · v2.1.1 seed 71 head + DDIM 2 단계 공식 평가

승인 후 CUDA GPU 하나를 선택하고 기존 `{source.name}`만 `/content`에 올립니다.
`DRIVE_PROJECT`를 확인한 뒤 `RUN_V25 = True`로 바꿉니다. 기본값은 `False`입니다.
기존 A의 예측 2D·hand와 seed 71 출력층을 사용합니다. 이 노트북은 검증 421클립·16,425프레임의 품질을 한 번 평가합니다.
v2.3 A 2단계의 반올림 전 종합 점수는 {pins['baseline']['macro_mm']!r} mm입니다. 직접 비교에 사용한 환경은 L4, Torch 2.14.1+cu130, CUDA 13.0입니다. Noise·평가 환경이 일치하면 종합 오차가 낮은 쪽을 판정합니다.
`reports/{DESTINATION}/run-…/`의 `quality.json`, `verdict.json`, `worker-result.json`, `result.json`을 회수합니다.
Drive 게시 실패 시 같은 VM의 출력에 표시된 `/content/rewind-v25-…` 폴더에 결과와 로그가 남습니다. VM 종료 전에 회수합니다.
호출 전체 60분, 작업 종료 여유 150초, 논리 읽기 2 GiB·쓰기 4 GiB, 로컬 스크래치 4 GiB입니다.
"""
    notebook = canonical({"nbformat": 4, "nbformat_minor": 5, "metadata": {"accelerator": "GPU",
                          "kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"}},
                          "cells": [cell("markdown", "guide", intro), cell("code", "settings", settings), cell("code", "run", code)]})
    receipt = {"schema": "rewind_v25_colab_release_v1", "default_off": True, "v23_release_receipt_sha256": RELEASE,
               "pins": pins, "files": {NOTEBOOK: {"bytes": len(notebook), "sha256": fingerprint(notebook)}}}
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    with tempfile.TemporaryDirectory(dir=output.parent, prefix=".v25-release-") as tmp:
        release = Path(tmp)/"release"
        release.mkdir()
        (release/NOTEBOOK).write_bytes(notebook)
        (release/"receipt.json").write_bytes(canonical(receipt))
        release.rename(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "v23-release", "v23-quality", "v211", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    build(**vars(parser.parse_args()))
