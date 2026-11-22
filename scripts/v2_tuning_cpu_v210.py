"""Recompute a completed v2.1.0 GPU result on CPU from its native Drive markers and NPZs."""

import argparse
import json
import math
from pathlib import Path
import re
import tempfile
from types import SimpleNamespace

import numpy as np

from scripts.v2_gpu128_a_bundle_format import FINAL_BOUNDARY, SOURCE_SHA
from scripts.v2_tuning_cpu_common import ROOT_INDEX_SHA, check_json_pin, require, sha
from scripts.v210_eval import (CAP, COMPARE, DIAGNOSTIC, FEATURE, FINAL, GROUP_SCHEMA, SUMMARY_SCHEMA,
                               expected_frames, group_keys, reference_scores, segments)
from scripts.v210_metrics import aggregate
from scripts.archvis_pilot_bootstrap import extract_pinned_zip
from scripts.run_v2_cpu_cache import MANIFEST_NAME, read_manifest
from scripts.run_v2_gpu_training import training_config
from rewind_v1 import v2_contract


MODULE_PATH = Path(__file__).resolve().parent
GPU_RECEIPT_SHA = "ac766ffff915349fbbc3528337dec3fb3ac0d86152611d8bd9011f92c5eee2dd"


def read_file(root, filename, pin=None, cap=CAP):
    path = root / filename
    require(path.is_relative_to(root) and not path.is_symlink()
            and path.resolve(strict=True) == path and path.is_file(), f"unsafe evidence path: {filename}")
    size = path.stat().st_size
    require(0 < size <= cap, f"evidence size limit: {filename}")
    with path.open("rb") as handle:
        raw = handle.read(size + 1)
    require(len(raw) == size and (pin is None or pin == {"bytes": size, "sha256": sha(raw)}),
            f"evidence SHA or size mismatch: {filename}")
    return raw


def release_pins(receipt, receipt_sha):
    receipt = receipt.absolute()
    raw = read_file(receipt.parent, receipt.name, cap=2 * 1024**2)
    require(sha(raw) == receipt_sha == GPU_RECEIPT_SHA, "fixed dcd20c5 GPU release receipt SHA")
    pins = json.loads(raw)["pins"]
    require(pins["A_index_sha256"] == ROOT_INDEX_SHA and pins["base"]["source_sha256"] == SOURCE_SHA
            and pins["base"]["final_boundary_sha256"] == FINAL_BOUNDARY
            and set(pins["groups"]) == {str(i) for i in range(129, 147)}
            and pins["modules"].keys() == {"v210_eval.py", "v210_metrics.py"}, "GPU receipt lineage")
    for name, expected in pins["modules"].items():
        require(sha((MODULE_PATH / name).read_text(encoding="utf-8").encode()) == expected,
                f"Use the GPU release revision of {name}")
    return pins


def fixed_reference(project, pins):
    markers = {}
    for name, schema in ((DIAGNOSTIC, "rewind_v2_gpu_invocation_v1"),
                         (COMPARE, "rewind_v2_gpu_comparison_v1")):
        markers[name] = json.loads(read_file(project, name, pins["evidence"][name]))
        check_json_pin(markers[name], schema)
    diagnostic, comparison = markers[DIAGNOSTIC], markers[COMPARE]
    require(diagnostic["status"] == "completed" and diagnostic["comparison"] == comparison
            and diagnostic["checkpoint"]["content_sha256"] == FINAL, "A reference evidence binding")
    return reference_scores(diagnostic, comparison)


def validation(source_zip, pins, scratch):
    base = pins["base"]
    frozen = scratch / "source"
    extract_pinned_zip(source_zip, SOURCE_SHA, frozen, max_zip_bytes=32 * 1024**2,
                       max_output_bytes=192 * 1024**2, max_members=3000, revision=base["revision"])
    manifest = scratch / "manifest"
    extract_pinned_zip(frozen / base["embedded_manifest"], base["archive_sha256"], manifest,
                       max_zip_bytes=64 * 1024**2, max_output_bytes=256 * 1024**2, max_members=1)
    doc, spec = read_manifest(manifest / MANIFEST_NAME, base, frozen / "reference")
    plan = v2_contract.plan(doc, spec, training_config(42), scratch)
    sizes = {(r["sequence"], r["start_frame"]): r["real_frames"] for r in plan["clips"]}
    keys = {split: sorted((r["sequence"], r["start_frame"]) for r in plan["clips"] if r["split"] == split)
            for split in ("train", "validation")}
    return SimpleNamespace(clips=keys, real_frames=lambda key, split: sizes[key]), spec


def match_recorded(actual, expected):
    if isinstance(expected, dict):
        return isinstance(actual, dict) and actual.keys() == expected.keys() and all(
            match_recorded(actual[k], v) for k, v in expected.items())
    if type(expected) is float:
        return type(actual) in (int, float) and math.isclose(actual, expected, rel_tol=1e-6, abs_tol=1e-5)
    return type(actual) is type(expected) and actual == expected


def validate_segment(key, arrays, n):
    types = {"clip_start": ((n,), np.int64), "frame_index": ((n,), np.int64),
             "sequence_id": ((n,), np.str_), "seconds": ((n,), np.float64),
             "pred_device_mm": ((n, 62, 3), np.float32), "target_device_mm": ((n, 62, 3), np.float32),
             "target_valid": ((n, 62), np.bool_), "hand_local_pred_mm": ((n, 40, 3), np.float32),
             "hand_local_target_mm": ((n, 40, 3), np.float32), "hand_local_valid": ((n, 40), np.bool_)}
    require(arrays.keys() == types.keys(), "GPU NPZ array roster")
    for name, (shape, dtype) in types.items():
        value = arrays[name]
        require(value.shape == shape and value.dtype.type is dtype
                and (dtype is np.str_ or bool(np.isfinite(value).all())), f"GPU NPZ array: {name}")
    require(np.array_equal(arrays["seconds"], np.arange(key[1], key[1] + n, dtype=np.float64) / 25),
            "GPU NPZ nominal time axis mismatch")


def recompute(results, pins, data, spec, comparison, summary_sha):
    require(results.is_dir() and results.resolve(strict=True) == results, "results directory must be direct")
    raw = read_file(results, "summary.json")
    require(sha(raw) == summary_sha, "external GPU summary file SHA")
    summary = json.loads(raw)
    check_json_pin(summary, SUMMARY_SCHEMA)
    expected_basis = {"source": SOURCE_SHA, "code": pins["modules"], "checkpoint_receipt": FINAL,
                      "cache_boundary": FINAL_BOUNDARY, "feature_identity": FEATURE,
                      "weights": comparison["binding"]["model"], "eval_seed": 71, "precision": "fp32",
                      "upper": "predicted_frozen_body", "sampling_steps": 10}
    require(summary["binding"] == expected_basis and summary["counts"] == {"groups": 18, "clips": 421, "real_frames": 16425},
            "GPU summary parent / full validation coverage")
    names = spec["body_joint_names"] + [n for side in ("left", "right") for n in spec["hands"][side][1:]]
    stream, receipts, environments = [], [], []
    for i in sorted(map(int, pins["groups"])):
        keys = [key for split, key in group_keys(data, i) if split == "validation"]
        doc = json.loads(read_file(results, f"group-{i:05d}.json"))
        check_json_pin(doc, GROUP_SCHEMA)
        expected = {**expected_basis, "group": i, "feature_input": pins["groups"][str(i)]}
        require(doc["binding"] == expected and doc["keys"] == [list(key) for key in keys]
                and doc["real_frames"] == len(expected_frames(keys, data)), "GPU marker scope")
        asset = doc["artifact"]
        require(re.fullmatch(rf"group-{i:05d}-[0-9a-f]{{32}}\.npz", asset["name"]), "GPU artifact name")
        blob = read_file(results, asset["name"], {k: asset[k] for k in ("bytes", "sha256")})
        for key, arrays in segments(blob, keys, data, names):
            validate_segment(key, arrays, data.real_frames(key, "validation"))
            stream.append((key, arrays))
        receipts.append(doc["content_sha256"])
        environments.append(doc["environment"])
    require(receipts == summary["receipts"] and environments == summary["environments"]
            and [key for key, _ in stream] == data.clips["validation"]
            and len({(s, int(f)) for (s, _), value in stream for f in value["frame_index"]}) == 16425,
            "GPU receipt coverage or duplicate validation frames")
    recalculated = aggregate(stream, spec, comparison)
    require(match_recorded(recalculated, summary["metrics"]), "CPU metrics differ from the GPU report")
    return {"schema": "rewind_v2_tuning_cpu_v210_recount_v1", "status": "cpu_postprocess_passed",
            "gpu_summary_file_sha256": summary_sha, "gpu_summary_content_sha256": summary["content_sha256"],
            "counts": summary["counts"], "binding": expected_basis, "metrics": recalculated}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("project", "results", "source-zip", "gpu-receipt", "output"):
        parser.add_argument(f"--{name}", required=True, type=Path)
    parser.add_argument("--gpu-receipt-sha256", required=True)
    parser.add_argument("--gpu-summary-sha256", required=True)
    args = parser.parse_args()
    roots = (args.project.absolute(), args.results.absolute())
    output = args.output.absolute()
    require(not output.exists() and output.parent.is_dir()
            and all(not output.resolve().is_relative_to(root.resolve(strict=True)) for root in roots),
            "CPU report must be a new file outside the evidence roots")
    pins = release_pins(args.gpu_receipt, args.gpu_receipt_sha256)
    comp = fixed_reference(roots[0], pins)
    with tempfile.TemporaryDirectory() as scratch:
        data, spec = validation(args.source_zip, pins, Path(scratch))
        report = recompute(roots[1], pins, data, spec, comp, args.gpu_summary_sha256)
    with output.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, allow_nan=False, indent=2)
    print(json.dumps({"path": str(output), "sha256": sha(output.read_bytes())}))


if __name__ == "__main__":
    main()
