"""Issue the default-off v2.2 extension from the completed original three pairs."""

import argparse
import json
from pathlib import Path
import zipfile

from rewind_archive.contracts import digest
from rewind_prepare.temporal_selection import check_pin
from scripts.v210_eval import FINAL, RECEIPT
from scripts.v211_build import MODULES, collect_pins, overlay_sources, pack, sha
from scripts.v211_trial import SEEDS as INITIAL
from scripts.v220_eval import COMPARISON
from scripts.v220_final_run import DESTINATION, io_budget
from scripts.v220_final_upper import SEEDS
from scripts.v220_hand import BRANCH, METHOD


NOTEBOOK = "v220_final_seeds_137_181_cuda_r2.ipynb"
TOOLS = "v220_final_seeds_tools_r2.zip"
INITIAL_RELEASE = "59945b7419a7272b748e72e6c8699c7f69ff9955361d6fcdac3972c9de41cd8e"
INITIAL_COMPARISON = "4b27038d48959026b7c99bdfed75207501c40b661ce8097002554573c2fbf2c9"
INITIAL_DIAGNOSTIC = "26bfd33568212adc5490437703d6b5fc143cf85b39278fba6a73de6aa02191b2"
CONFIG = '''RUN_V220_FINAL = False
DRIVE_PROJECT = "/content/drive/MyDrive/REWIND-like prototype"
RECOVER_WRITER_OWNER = ""
'''


def verified(path, expected):
    raw = path.read_bytes()
    if sha(raw) != expected:
        raise ValueError(f"v220_final_initial_evidence_sha256: {path.name}")
    return json.loads(raw), len(raw)


def build(source, a_bundle, initial_receipt, comparison, diagnostic, output):
    pins = collect_pins(source, a_bundle)
    release, _ = verified(initial_receipt, INITIAL_RELEASE)
    original, size = verified(comparison, INITIAL_COMPARISON)
    status, _ = verified(diagnostic, INITIAL_DIAGNOSTIC)
    check_pin(original, COMPARISON, original["content_sha256"])
    check_pin(status, "rewind_v220_invocation_v1", status["content_sha256"])
    old = release["pins"]
    if (status["status"] != "completed" or status["release_inputs_sha256"] != digest(old)
            or status["comparison_sha256"] != original["content_sha256"]
            or status["body_prediction_cache"]["index_sha256"] != original["predicted_body_cache_sha256"]
            or tuple(row["seed"] for row in original["pairs"]) != INITIAL
            or any(old[name] != value for name, value in pins.items())
            or pins["trial_plan"]["branches"][METHOD] != BRANCH
            or pins["trial_plan"]["conditional_additional_seeds"] != list(SEEDS)):
        raise ValueError("v220_final_completed_initial_comparison")
    pins["initial"] = {"comparison": {"bytes": size, "sha256": INITIAL_COMPARISON,
                                      "content_sha256": original["content_sha256"]},
                       "release_receipt_sha256": INITIAL_RELEASE, "diagnostic_sha256": INITIAL_DIAGNOSTIC,
                       "module_shas": old["modules"]}
    pins["final_seeds"] = list(SEEDS)
    with zipfile.ZipFile(a_bundle) as archive:
        raw = archive.read(RECEIPT)
    if {"bytes": len(raw), "sha256": sha(raw)} != pins["evidence"][RECEIPT]:
        raise ValueError("v220_final_parent_receipt_pin")
    parent = json.loads(raw)
    check_pin(parent, "rewind_v2_gpu_training_checkpoint_v1", FINAL)
    budget = pins["io_budget"] = io_budget(pins, parent["archive"]["bytes"])
    overlays = overlay_sources(pins, (*MODULES, "v220_upper.py", "v220_hand.py", "v220_eval.py",
                                     "v220_final_upper.py", "v220_final_run.py"))
    guide = ("# v2.2 최종 비교 R2 · 추가 matched seed 137·181\n\n"
             "GT:예측 상체 100:0과 50:50에서 seed 137·181만 각각 A hand e2로 시작합니다. "
             "각 3,097 updates·batch 1·Adam 5e-5, 공식 validation 421클립·16,425프레임, FP32. "
             "기존 3쌍과 검증된 body 상체 캐시를 사용하고 새 train 클립의 예측만 추가합니다.\n\n"
             "1회 8시간/논리 읽기 16 GiB/쓰기 8 GiB, 종료 여유 5분입니다. 기본 OFF이며 승인된 호출만 실행합니다. "
             f"`{source.name}`과 `{TOOLS}`를 /content에 올립니다. 출력은 `{DESTINATION}`입니다.")
    pack(output, pins, overlays, tag="v220_final", method=METHOD, notebook=NOTEBOOK, tools_name=TOOLS,
         config=CONFIG, guide=guide, destination=DESTINATION,
         release_fields={"release": "v220_final_seeds_r2", "resume_inputs_sha256": digest(pins), "io_budget": budget,
                         "initial_observed_usage": status["usage"]})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "a-bundle", "initial-receipt", "comparison", "diagnostic", "output"):
        parser.add_argument("--"+name, type=Path, required=True)
    build(**vars(parser.parse_args()))
