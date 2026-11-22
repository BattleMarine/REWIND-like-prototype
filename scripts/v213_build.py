"""Issue the fixed, default-off full-body v2.1.3 CUDA notebook offline."""

import argparse
from pathlib import Path

from rewind_archive.contracts import digest
from scripts.v211_build import MODULES, collect_pins, overlay_sources, pack
from scripts.v213_body import BRANCH, METHOD
from scripts.v213_run import DESTINATION


NOTEBOOK = "v213_body_full_128_A_cuda.ipynb"
TOOLS = "v213_body_full_tools.zip"
CONFIG = '''RUN_V213 = False
DRIVE_PROJECT = "/content/drive/MyDrive/REWIND-like prototype"
RECOVER_WRITER_OWNER = ""
'''


def build(source, a_bundle, output):
    pins = collect_pins(source, a_bundle)
    if pins["trial_plan"]["branches"][METHOD] != BRANCH:
        raise ValueError("v213_cpu_trial_plan_branch")
    overlays = overlay_sources(pins, (*MODULES, "v213_body.py", "v213_run.py"))
    guide = ("# v2.1.3 A body 전체 미세조정\n\n"
             "A에서 독립 시작합니다. Seed 42·71·103 각 3,097 updates, 1epoch, batch 1, Adam 5e-5, FP32. "
             "평가 seed 71, 예측 2D·예측 상체, 공식 validation 421클립·16,425프레임, DDIM 10. "
             "세 시나리오의 body/hand-global MPJPE를 동일 가중 평균합니다.\n\n"
             "호출당 8시간 및 read 8 GiB/write 2 GiB. Fit 체크포인트는 첫 update, 1550, 3097에서 저장합니다. "
             "I/O 여유가 부족하면 다음 저장 경계를 시작하기 전에 멈춥니다. 승인된 예산 안에서만 수동 재개합니다.\n\n"
             f"`{source.name}`과 `{TOOLS}`를 Colab `/content`에 올리고 CUDA GPU 하나를 선택합니다. "
             f"기존 Drive A 모델·feature를 사용하며 결과는 `{DESTINATION}`에 보관합니다. 실행 스위치는 기본 OFF입니다.")
    pack(output, pins, overlays, tag="v213", method=METHOD, notebook=NOTEBOOK, tools_name=TOOLS,
         config=CONFIG, guide=guide, destination=DESTINATION,
         release_fields={"release": "full_body_v1", "resume_inputs_sha256": digest(pins)})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "a-bundle", "output"):
        parser.add_argument("--"+name, type=Path, required=True)
    build(**vars(parser.parse_args()))
