"""Issue an inert Colab template for the main v2.4 comparison and pilot intake."""

import argparse
import json
from pathlib import Path

from scripts.v24_main_plan import R2_RECEIPT_SHA, canonical, sha, template


NOTEBOOK = "v24_main_comparison_plan_draft_20261123.ipynb"
TEMPLATE = "main-plan-template.json"


def cell(kind, name, source):
    result = {"cell_type": kind, "id": name, "metadata": {}, "source": source.splitlines(keepends=True)}
    if kind == "code":
        result.update(execution_count=None, outputs=[])
    return result


def build(output):
    source = Path(__file__).with_name("v24_main_plan.py").read_text(encoding="utf-8").split('\nif __name__ == "__main__":')[0]
    settings = '''ATTACH_VERIFIED_PILOT = False
PILOT_RESULT_DIRECTORY = "/content/v24_pilot_result"
PILOT_RELEASE_RECEIPT = "/content/v24_pilot_r2_release_receipt.json"
PILOT_RESULT_SHA256 = ""
OUTPUT_FILE = "/content/v24_main_plan_with_pilot.json"
'''
    code = source + '''
if ATTACH_VERIFIED_PILOT:
    plan = from_pilot(Path(PILOT_RESULT_DIRECTORY), Path(PILOT_RELEASE_RECEIPT), PILOT_RESULT_SHA256)
    print(json.dumps(publish(Path(OUTPUT_FILE), plan), ensure_ascii=False))
else:
    print("파일럿 회수 대기. 기본 설정에서는 입력을 읽거나 계획 파일을 쓰지 않습니다.")
'''
    lead = """# v2.4 본시험 계획 초안

초기 x0·v는 같은 seed 42·71·103의 6회다. 조건을 사전 확정하고 각각의 결과를 회수한 뒤
승인된 경우에만 seed 137·181로 4회를 추가한다.

이 노트북은 회수한 R2 파일럿 증거를 읽어 계획 초안에 붙인다. GPU 런타임이나 원본 데이터 캐시는 필요 없다.
두 설정 셀의 기본 상태는 OFF이며 본 학습 실행 경로는 제공하지 않는다.
"""
    compile(settings, "main template settings", "exec")
    compile(code, NOTEBOOK, "exec")
    doc = canonical({"nbformat": 4, "nbformat_minor": 5, "metadata": {"kernelspec": {
        "name": "python3", "display_name": "Python 3", "language": "python"}},
        "cells": [cell("markdown", "guide", lead), cell("code", "settings", settings), cell("code", "attach-pilot", code)]})
    plan = canonical(template()) + b"\n"
    files = {NOTEBOOK: doc, TEMPLATE: plan}
    output.mkdir()
    for name, raw in files.items():
        (output / name).write_bytes(raw)
    receipt = {"schema": "rewind_v24_main_template_release_v1", "status": "awaiting_pilot",
               "fixed_pilot_release_receipt_sha256": R2_RECEIPT_SHA,
               "files": {name: {"bytes": len(raw), "sha256": sha(raw)} for name, raw in files.items()}}
    (output / "receipt.json").write_bytes(canonical(receipt))
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(build(args.output)["files"]))
