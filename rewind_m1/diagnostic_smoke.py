"""고정 source stamp로 한 번 실행하는 합성 CPU 통합 진단 worker."""
import argparse
import hashlib
import math
import platform
import re
from pathlib import Path

import torch
from PIL import __version__ as pillow_version

from .data import content_hash
from .local_fixture import create_fixture
from .local_rgb import LocalStereoRGB
from .local_diagnostic import run_diagnostic
from .prediction_join import _read_file, _serialized
from .training import ROOT, read_checkpoint, source_identity


def run_smoke(output, expected_revision):
    output = Path(output)
    output.mkdir(exist_ok=False)
    summary = {"schema_version":"local_m1_cpu_smoke_v1", "status":"failed", "evaluation_kind":"synthetic_code_check",
        "expected_revision":expected_revision, "checks":{}, "environment":{"python":platform.python_version(),
        "torch":str(torch.__version__), "pillow":pillow_version, "cpu_threads":torch.get_num_threads()},
        "limitations":"source stamp comparison and synthetic code check; not Git history authentication or real data validation"}
    try:
        if not isinstance(expected_revision,str) or not re.fullmatch(r"[0-9a-fA-F]{40}",expected_revision):
            raise ValueError("고정 expected revision 40자리 필요")
        stamp = (ROOT/"SOURCE_REVISION.txt").read_text(encoding="utf-8").strip()
        if stamp != expected_revision:
            raise ValueError("SOURCE_REVISION stamp 불일치")
        if torch.get_num_threads() != 1:
            raise ValueError("CPU thread1 환경에서만 진단 실행")
        summary["source"] = source_identity()
        summary["checks"]["source_stamp_matches"] = True
        receipt = create_fixture(output/"fixture")
        fixture = output/"fixture"
        pose = _read_file(fixture/"features.json")
        before = content_hash(pose)
        report = run_diagnostic(LocalStereoRGB(fixture/"manifest.json", fixture), pose, receipt["joint_names"],
            receipt["rgb_manifest_sha256"], _read_file(fixture/"frontend_config.json"),
            _read_file(fixture/"regression_config.json"), _read_file(fixture/"groups.json"), output/"diagnostic", regression_steps=2)
        expected_stages = ["frontend","prediction_cache","evaluation_2d","prediction_join","regression"]
        if (report["status"] != "completed" or [s["name"] for s in report["stages"]] != expected_stages
                or any(s["status"] != "completed" for s in report["stages"])):
            raise ValueError("runner 통합 단계 미완료")
        for name, info in report["artifacts"].items():
            raw = (output/"diagnostic"/name).read_bytes()
            if len(raw) != info["size_bytes"] or hashlib.sha256(raw).hexdigest() != info["sha256"]:
                raise ValueError("runner 산출물 hash 불일치")
        frontend = _read_file(output/"diagnostic/frontend/report.json")
        checkpoint = read_checkpoint(output/"diagnostic/regression/last.pt")
        two_d_report = _read_file(output/"diagnostic/evaluation_2d.json")
        two_d = two_d_report["metrics"]["all"]["combined"]
        three_d = _read_file(output/"diagnostic/regression/report.json")["validation"]
        if len(frontend["training"]) != 2 or checkpoint["step"] != 2 or checkpoint["input_contract"]["input_source"] != "predicted_2d":
            raise ValueError("update/예측 입력 계약 불일치")
        if (two_d_report["split"] != "validation" or two_d_report["records"] != 2 or three_d["split"] != "validation"
                or two_d["paired_count"] != 8 or three_d["valid_joints"] != 4 or two_d["coverage"] != 1):
            raise ValueError("평가 개수/coverage 불일치")
        if not all(math.isfinite(v) for v in (two_d["mean_error_px"],three_d["mpjpe_mm"],three_d["mse_mm2"])):
            raise ValueError("평가 지표 비유한 값")
        predicted = _read_file(output/"diagnostic/predicted_features.json")
        if len(pose["records"]) != 4 or len(predicted["records"]) != len(pose["records"]):
            raise ValueError("fixture/예측 record 수 불일치")
        for original, row in zip(pose["records"], predicted["records"]):
            row.pop("prediction")
            if original != row:
                raise ValueError("원본 pose 변경")
        if content_hash(pose) != before:
            raise ValueError("원본 입력 변경")
        summary["checks"].update(runner_completed=True,artifact_hashes_match=True,updates_2_plus_2=True,
            predicted_input=True,validation_finite=True,pose_preserved=True)
        summary.update(status="passed", fixture_receipt=receipt, runner_report_sha256=hashlib.sha256((output/"diagnostic/report.json").read_bytes()).hexdigest(),
            metrics={"2d":two_d,"3d":three_d})
    except BaseException as exc:
        summary.update(status="interrupted" if isinstance(exc,KeyboardInterrupt) else "failed", error_type=type(exc).__name__)
        raise
    finally:
        with (output/"summary.json").open("xb") as handle:
            handle.write(_serialized(summary))
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--expected-revision",required=True)
    args = parser.parse_args(argv)
    try:
        result = run_smoke(args.output,args.expected_revision)
    except BaseException as exc:
        print(type(exc).__name__)
        return 1
    print(result["status"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
