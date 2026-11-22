"""Issue a pinned v2.1.1 overlay and default-off Colab notebook offline."""

import argparse
import json
from pathlib import Path
import tempfile
import zipfile

from rewind_archive.contracts import canonical, digest
from rewind_prepare.temporal_selection import check_pin
from scripts.build_archvis_pilot_handoff import _cell
from scripts.v210_build import BASE_INDEX_SHA, ROOT, RUN, SOURCE_SHA, sha
from scripts.v210_eval import A, COMPARE, DIAGNOSTIC, FEATURE, FINAL, RECEIPT
from scripts.v211_run import DESTINATION, LEGACY_INPUTS
from scripts.v2_tuning_cpu_selection import plan


MODULES = ("v210_eval.py", "v210_metrics.py", "v211_progress.py", "v211_run.py", "v211_trial.py")
NOTEBOOK = "v211_body_head_128_A_cuda_r2.ipynb"
TOOLS = "v211_body_head_tools_r2.zip"
CONFIG = '''RUN_V211 = False
DRIVE_PROJECT = "/content/drive/MyDrive/REWIND-like prototype"
RECOVER_WRITER_OWNER = ""
'''


def collect_pins(source, a_bundle):
    if sha(source.read_bytes()) != SOURCE_SHA:
        raise ValueError("v211_source_zip")
    with zipfile.ZipFile(a_bundle) as archive:
        raw = archive.read("BUNDLE_INDEX.json")
        if sha(raw) != BASE_INDEX_SHA:
            raise ValueError("v211_A_bundle_index")
        members = json.loads(raw)["members"]

        def original(name):
            raw = archive.read(name)
            if members[name] != {"bytes": len(raw), "sha256": sha(raw)}:
                raise ValueError(f"v211_A_archive_member: {name}")
            return json.loads(raw)

        base = original("source/receipt.json")["pins"]
        if base["source_sha256"] != SOURCE_SHA or source.name != base["source_name"]:
            raise ValueError("v211_frozen_source_pin")
        step = original(RECEIPT)
        check_pin(step, "rewind_v2_gpu_training_checkpoint_v1", FINAL)
        diagnostic, comparison = original(DIAGNOSTIC), original(COMPARE)
        if diagnostic["checkpoint"] != step or diagnostic["comparison"] != comparison or diagnostic["status"] != "completed":
            raise ValueError("v211_A_evidence_scope")
        groups = {}
        folder = f"{A}/features/{FEATURE}"
        for group in range(147):
            key = f"{folder}/group-{group:05d}.json"
            marker = original(key)
            check_pin(marker, "rewind_v2_gpu_features_group_v1", marker["content_sha256"])
            package = marker["archive"]
            if marker["identity"] != FEATURE or members[f"{folder}/{package['name']}"] != {k: package[k] for k in ("bytes", "sha256")}:
                raise ValueError("v211_feature_index")
            groups[str(group)] = {"marker": members[key], "archive": package}
        return {"base": base, "evidence": {name: members[name] for name in (DIAGNOSTIC, RECEIPT, COMPARE)},
                "groups": groups, "A_index_sha256": BASE_INDEX_SHA,
                "trial_plan": plan(step["identity"], comparison["binding"]["model"], comparison["macro_score_mm"])}


def build(source, a_bundle, output, previous_receipt):
    pins = collect_pins(source, a_bundle)
    guide = ("# v2.1.1 A body 최종 head · CUDA 시작 수정 R2\n\n"
             "seed 42·71·103을 독립적으로 순차 실행합니다. 각 3,097 updates, 1epoch, batch 1, Adam 5e-5, FP32. "
             "평가 seed 71, 예측 2D·예측 상체, 공식 validation 421클립·16,425프레임, DDIM 10. "
             "세 시나리오의 body/hand-global MPJPE를 동일 가중 평균합니다. 호출 상한은 8시간입니다. "
             "소요 시간과 승인된 누적 GPU 예산은 오케스트레이션 인계에서 확인하세요.\n\n"
             f"128 source ZIP `{source.name}`과 `{TOOLS}`를 Colab `/content`에 업로드합니다. "
             f"A 모델과 feature 캐시는 기존 Drive에서 읽고 결과는 `{DESTINATION}`에 저장합니다. "
             "CUDA GPU 하나를 선택합니다. 실행 스위치는 기본 OFF입니다.")
    issue(output, pins, previous_receipt, modules=MODULES, tag="v211", method="v2.1.1", notebook=NOTEBOOK, tools_name=TOOLS,
          config=CONFIG, guide=guide, destination=DESTINATION)


def issue(output, pins, previous_receipt, *, modules, tag, method, notebook, tools_name, config, guide, destination):
    if output.exists() or output.is_symlink() or not output.parent.is_dir():
        raise ValueError("v2_tuning_output_exists_or_parent_missing")
    overlays = {name: (ROOT/"scripts"/name).read_text(encoding="utf-8").encode() for name in modules}
    pins["modules"] = {name: sha(raw) for name, raw in overlays.items()}
    pins["cpu_selection_source_sha256"] = sha((ROOT/"scripts/v2_tuning_cpu_selection.py").read_text(encoding="utf-8").encode())
    previous = json.loads(previous_receipt.read_bytes())["pins"]
    old_modules = previous["modules"]
    if (digest(previous) != LEGACY_INPUTS[method] or {**pins, "modules": old_modules} != previous
            or set(pins["modules"]) != set(old_modules)):
        raise ValueError("v2_tuning_startup_release_input_drift")
    # The original v2.1.1 release predates the already-reviewed common runtime
    # in 590780b; these two shared modules must match that revision byte-for-byte.
    approved_updates = {"v211_run.py": pins["modules"]["v211_run.py"]}
    if method == "v2.1.1":
        approved_updates.update({
            "v211_progress.py": "afd23a1ab6e0c1a692a39dcfcd817f960a698a745c4b51484cbd9de5dc5e57d4",
            "v211_trial.py": "8b7ed884e419cd411eee1dd84d7b85247eb4b8d3d3293bcfa5a87cd26aa354ed",
        })
    if pins["modules"] != {**old_modules, **approved_updates}:
        raise ValueError("v2_tuning_startup_release_unexpected_module_drift")
    pins["resume_modules"] = old_modules
    with tempfile.TemporaryDirectory(dir=output.parent, prefix=f".{tag}-build-") as temporary:
        root = Path(temporary)/"release"
        root.mkdir()
        tools = root/tools_name
        with zipfile.ZipFile(tools, "x", compression=zipfile.ZIP_DEFLATED) as writer:
            for name, raw in overlays.items():
                entry = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
                entry.external_attr = 0o100600<<16
                writer.writestr(entry, raw, compress_type=zipfile.ZIP_DEFLATED)
        tool_hash = sha(tools.read_bytes())
        boot = (ROOT/"scripts/archvis_pilot_bootstrap.py").read_text(encoding="utf-8")
        run = (RUN.replace("v210_eval", f"{tag}_run").replace("v210", tag).replace("V210", tag.upper())
               .replace("v2.1.0", method).replace("max_members=2)", "max_members=len(PINS['modules']))"))
        run = run.replace("    if code:\n", '''    if code:
        try:
            result = json.loads((work/"result.json").read_text(encoding="utf-8"))
            print(json.dumps({key: result[key] for key in ("status", "phase", "failure", "cuda_peak") if key in result},
                             ensure_ascii=False, indent=2), flush=True)
        except (OSError, ValueError):
            pass
''', 1)
        code = boot+f"\nPINS = {pins!r}\nTOOL_NAME = {tools_name!r}\nTOOL_SHA = {tool_hash!r}\n"+run
        compile(config, "settings", "exec")
        compile(code, notebook, "exec")
        guide += (f"\n\n- A 체크포인트 receipt `{FINAL}`\n- source `{SOURCE_SHA}`\n"
                  f"- manifest `{pins['base']['manifest_content_sha256']}`\n"
                  f"- cache `{pins['base']['final_boundary_sha256']}`\n"
                  f"- trial plan `{pins['trial_plan']['content_sha256']}`")
        raw = canonical({"nbformat": 4, "nbformat_minor": 5,
                         "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"}},
                         "cells": [_cell("markdown", "guide", guide), _cell("code", "settings", config), _cell("code", "run", code)]})
        (root/notebook).write_bytes(raw)
        (root/"receipt.json").write_bytes(canonical({"pins": pins, "notebook": notebook, "notebook_sha256": sha(raw),
            "tools": tools_name, "tools_sha256": tool_hash, "default_off": True, "output_folder": destination,
            "release": "cuda_startup_r2", "resume_inputs_sha256": LEGACY_INPUTS[method]}))
        (root/"trial-plan.json").write_bytes(canonical(pins["trial_plan"]))
        root.rename(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "a-bundle", "output", "previous-receipt"):
        parser.add_argument("--"+name, type=Path, required=True)
    build(**vars(parser.parse_args()))
