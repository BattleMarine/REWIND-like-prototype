"""Issue the default-off, fixed v2.2 paired hand Colab notebook."""

import argparse
from pathlib import Path

from rewind_archive.contracts import digest
from scripts.v211_build import MODULES, collect_pins, overlay_sources, pack, use_pinned_modules
from scripts.v220_hand import BRANCH, METHOD
from scripts.v220_run import CONTRACT, DESTINATION


NOTEBOOK = "v220_hand_upper_128_A_cuda.ipynb"
TOOLS = "v220_hand_upper_tools.zip"
CONFIG = '''RUN_V220 = False
DRIVE_PROJECT = "/content/drive/MyDrive/REWIND-like prototype"
RECOVER_WRITER_OWNER = ""
'''


def build(source, a_bundle, output):
    pins = collect_pins(source, a_bundle)
    if pins["trial_plan"]["branches"][METHOD] != BRANCH:
        raise ValueError("v220_cpu_trial_plan_branch")
    pins["v220"] = CONTRACT
    overlays = overlay_sources(pins, (*MODULES, "v220_upper.py", "v220_hand.py", "v220_eval.py", "v220_run.py"))
    use_pinned_modules(pins, overlays, "d926d25",
                       {"v211_run.py": "cd877c233a7d4f57be699b2387cf3d0c399f5a790f9343d9fd83388b21b55384"})
    guide = ("# v2.2 A hand · GT와 예측 상체 조건 비교\n\n"
             "GT:예측 상체 100:0 대 50:50, 각 seed 42·71·103, 각 1epoch 3,097 updates·batch 1·Adam 5e-5. "
             "A frontend/body 고정. 같은 seed는 clip·학습 noise를 공유합니다. 예측 상체를 검증 보관하고 재사용합니다. "
             "공식 validation 421클립·16,425프레임, 평가 seed 71, DDIM 10, FP32.\n\n"
             "호출당 8시간, read 8 GiB/write 2 GiB. I/O에 따라 여러 수동 호출이 필요합니다. "
             "오케스트레이션이 승인한 호출을 한 번씩 실행하세요. 기본 OFF입니다.\n\n"
             f"`{source.name}`과 `{TOOLS}`를 `/content`에 업로드하고 CUDA GPU 하나를 선택합니다. "
             f"출력 위치는 `{DESTINATION}`입니다.")
    pack(output, pins, overlays, tag="v220", method=METHOD, notebook=NOTEBOOK, tools_name=TOOLS,
         config=CONFIG, guide=guide, destination=DESTINATION,
         release_fields={"release": "v220_paired_hand_v1", "resume_inputs_sha256": digest(pins)})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "a-bundle", "output"):
        parser.add_argument("--"+name, type=Path, required=True)
    build(**vars(parser.parse_args()))
