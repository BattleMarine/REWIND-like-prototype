"""Issue a default-off v2.1.4 notebook before the parent decision exists."""

import argparse
from pathlib import Path

from scripts.v211_build import MODULES, collect_pins, overlay_sources, pack
from scripts.v214_decision import DECISION_NAME, SCREENING_SHA, SHA_ENV
from scripts.v214_run import DESTINATION
from scripts.v214_wrist import BRANCH, METHOD


NOTEBOOK = "v214_wrist_weight_128_A_cuda.ipynb"
TOOLS = "v214_wrist_weight_tools.zip"
CONFIG = f'''import os

RUN_V214 = False
DRIVE_PROJECT = "/content/drive/MyDrive/REWIND-like prototype"
RECOVER_WRITER_OWNER = ""
V214_DECISION_SHA256 = ""
if RUN_V214 and not V214_DECISION_SHA256:
    raise ValueError("오케스트레이션에서 검증한 부모 판정 SHA-256이 필요합니다.")
os.environ["{SHA_ENV}"] = V214_DECISION_SHA256
'''


def build(source, a_bundle, output):
    pins = collect_pins(source, a_bundle)
    if pins["trial_plan"]["branches"][METHOD] != BRANCH:
        raise ValueError("v214_cpu_plan_branch")
    modules = (*MODULES, "v212_lora.py", "v2_gpu128_a_bundle_format.py", "v2_tuning_cpu_common.py", "v2_tuning_cpu_selection.py",
               "v214_decision.py", "v214_eval.py", "v214_run.py", "v214_wrist.py")
    overlays = overlay_sources(pins, modules)
    guide = ("# v2.1.4 A 손목 위치 손실 가중치 2\n\n"
             "최종 선정 결과를 재검증해 head 또는 LoRA를 고르고, A에서 독립 초기화합니다. "
             "seed 42·71·103, 각 3,097 updates, batch 1, Adam 5e-5, 공식 validation 421클립·16,425프레임. "
             "matched weight 1은 기존 예측을 읽습니다.\n\n"
             f"`{source.name}`, `{TOOLS}`, 오케스트레이션 승인 `{DECISION_NAME}`을 `/content`에 올립니다. "
             "검토된 결정의 SHA-256을 설정하기 전에는 기본 OFF 상태를 유지합니다. "
             f"CUDA 한 대, 호출당 8시간, 결과 `{DESTINATION}`.")
    pack(output, pins, overlays, tag="v214", method=METHOD, notebook=NOTEBOOK, tools_name=TOOLS,
         config=CONFIG, guide=guide, destination=DESTINATION,
         release_fields={"release": "wrist_weight_conditional_v1", "decision_required": True, "screening_sha256": SCREENING_SHA})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "a-bundle", "output"):
        parser.add_argument("--"+name, type=Path, required=True)
    build(**vars(parser.parse_args()))
