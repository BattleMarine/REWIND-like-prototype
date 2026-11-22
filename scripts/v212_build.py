"""Issue the pinned, default-off v2.1.2 LoRA Colab notebook offline."""

import argparse
from pathlib import Path

from scripts.v211_build import MODULES, collect_pins, issue
from scripts.v212_lora import BRANCH, METHOD
from scripts.v212_run import DESTINATION


NOTEBOOK = "v212_body_lora_128_A_cuda.ipynb"
TOOLS = "v212_body_lora_tools.zip"
CONFIG = '''RUN_V212 = False
DRIVE_PROJECT = "/content/drive/MyDrive/REWIND-like prototype"
RECOVER_WRITER_OWNER = ""
'''


def build(source, a_bundle, output):
    pins = collect_pins(source, a_bundle)
    if pins["trial_plan"]["branches"][METHOD] != BRANCH:
        raise ValueError("v212_cpu_trial_plan_branch")
    modules = (*MODULES, "v212_lora.py", "v212_run.py")
    guide = ("# v2.1.2 A body 시간축 QKV LoRA\n\n"
             "세 temporal fused QKV의 rank 4·alpha 4·dropout 0 어댑터만 학습합니다. 초기 ΔW는 0입니다. "
             "seed 42·71·103 독립 순차 실행, 각 3,097 updates, 1epoch, batch 1, Adam 5e-5, FP32. "
             "평가 seed 71, 예측 2D·예측 상체, 공식 validation 421클립·16,425프레임, DDIM 10. "
             "세 시나리오의 body/hand-global MPJPE를 동일 가중 평균합니다. 호출 상한은 8시간입니다. "
             "승인된 GPU 예산과 실행 순서는 오케스트레이션 인계를 따르세요.\n\n"
             f"`{source.name}`과 `{TOOLS}`를 Colab `/content`에 업로드합니다. "
             f"A 모델과 feature는 기존 Drive를 읽고 결과는 `{DESTINATION}`에 보관합니다. "
             "CUDA GPU 하나를 선택합니다. 실행 스위치는 기본 OFF입니다.")
    issue(output, pins, modules=modules, tag="v212", method=METHOD, notebook=NOTEBOOK, tools_name=TOOLS,
          config=CONFIG, guide=guide, destination=DESTINATION)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "a-bundle", "output"):
        parser.add_argument("--"+name, type=Path, required=True)
    build(**vars(parser.parse_args()))
