"""Issue a finalist-only notebook from the sealed first nine trials."""

import argparse
from pathlib import Path

from rewind_archive.contracts import canonical, digest
from scripts.v211_build import MODULES, collect_pins, overlay_sources, pack, sha, use_pinned_modules
from scripts.v2_tuning_final_run import DESTINATIONS, SEEDS, STRATEGIES, scope
from scripts.v2_tuning_final_selection import screening_rows


TOOLS = "v2_tuning_final_tools.zip"
TAG = "v2_tuning_final"
CONFIG = '''RUN_V2_TUNING_FINAL = False
DRIVE_PROJECT = "/content/drive/MyDrive/REWIND-like prototype"
RECOVER_WRITER_OWNER = ""
'''


def build(source, a_bundle, output, method, screening_dirs, screening_report):
    pins = collect_pins(source, a_bundle)
    if pins["trial_plan"]["branches"][method] != STRATEGIES[method].BRANCH:
        raise ValueError("v2_tuning_final_cpu_plan_branch")
    qualified = screening_rows(pins["trial_plan"], screening_dirs, screening_report)
    raw = canonical(qualified)
    pins["final"] = {"method": method, "seeds": list(SEEDS), "shortlist": qualified["shortlist"],
                     "screening_sha256": sha(raw)}
    scope(pins)
    overlays = overlay_sources(pins, (*MODULES, "v212_lora.py", "v2_tuning_final_run.py"))
    use_pinned_modules(pins, overlays, "d926d25",
                       {"v211_run.py": "cd877c233a7d4f57be699b2387cf3d0c399f5a790f9343d9fd83388b21b55384"})
    notebook = f"{method.replace('.', '')}_final_seeds_137_181_cuda.ipynb"
    destination = DESTINATIONS[method]
    guide = (f"# {method} 최종 비교 · 추가 seed 137·181\n\n"
             "A에서 각각 시작하는 두 trial, 각 3,097 updates, 1epoch, batch 1, Adam 5e-5, FP32. "
             "평가 seed 71·DDIM 10·공식 validation 421클립·16,425프레임. "
             "호출당 최대 8시간, read 8 GiB, write 2 GiB. 승인된 호출만 실행합니다.\n\n"
             f"`{source.name}`과 `{TOOLS}`를 Colab `/content`에 올리고 CUDA GPU 하나를 선택합니다. "
             f"결과는 `{destination}`에 보관합니다. 실행 스위치는 기본 OFF입니다.\n\n"
             f"- 9회 선별 행 SHA `{pins['final']['screening_sha256']}`")
    pack(output, pins, overlays, tag=TAG, method=method, notebook=notebook, tools_name=TOOLS, config=CONFIG,
         guide=guide, destination=destination,
         release_fields={"release": "final_seeds_v1", "resume_inputs_sha256": digest(pins)},
         extra_files=(("qualified-screening.json", raw),))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "a-bundle", "output", "screening-report"):
        parser.add_argument("--"+name, type=Path, required=True)
    parser.add_argument("--method", choices=STRATEGIES, required=True)
    parser.add_argument("--screening-dirs", type=Path, nargs=3, required=True, metavar=("HEAD", "LORA", "BODY"))
    build(**vars(parser.parse_args()))
