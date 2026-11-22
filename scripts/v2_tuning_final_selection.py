"""Verify the screening and finalist measurements before running the CPU selector."""

import argparse
import hashlib
import json
from pathlib import Path

from rewind_archive.contracts import canonical, digest
from rewind_m1.local_rgb import _regular_read
from rewind_prepare.temporal_selection import check_pin
from scripts.v211_trial import measurement
from scripts.v2_tuning_cpu_selection import BODY, INITIAL, summarize
from scripts.v2_tuning_final_run import DESTINATIONS, SEEDS, STRATEGIES, scope
from scripts.v213_body import SUMMARY_SCHEMA as V213_SUMMARY


SCHEMAS = {**{method: strategy.SUMMARY_SCHEMA for method, strategy in STRATEGIES.items()}, BODY[2]: V213_SUMMARY}
CAP = 2*1024**2
SCREENING_SCHEMA = "rewind_v2_tuning_qualified_screening_v1"


def read_file(path):
    raw = _regular_read(path, CAP)
    return json.loads(raw), hashlib.sha256(raw).hexdigest()


def read_summary(path, plan):
    doc, file_sha = read_file(path)
    row, basis = doc["measurement"], doc["binding"]
    method, seed = row["method"], row["seed"]
    check_pin(doc, SCHEMAS[method], doc["content_sha256"])
    if (doc["counts"] != {"clips": 421, "real_frames": 16425} or basis["method"] != method or basis["seed"] != seed
            or basis["plan"] != plan["content_sha256"] or basis["original_model"] != plan["common"]["selected_model_sha256"]
            or basis["source"] != plan["common"]["parent"]["source"]
            or row != measurement(plan, seed, basis["body_sha256"], doc["metrics"], method)):
        raise ValueError(f"v2_tuning_final_summary_scope: {path.name}")
    return row, basis, file_sha


def screening_rows(plan, folders, report_path):
    rows, files = [], []
    for method, folder in zip(BODY, folders, strict=True):
        for seed in INITIAL:
            path = folder/f"seed-{seed}-summary.json"
            row, _, file_sha = read_summary(path, plan)
            if (row["method"], row["seed"]) != (method, seed):
                raise ValueError("v2_tuning_screening_row_order")
            rows.append(row)
            files.append({"method": method, "seed": seed, "name": f"{folder.name}/{path.name}", "sha256": file_sha})
    reported, reported_sha = read_file(report_path)
    actual = summarize(plan, rows, phase="screening")

    def sorted_rows(result):
        return {**result, "rows": sorted(result["rows"], key=lambda row: (row["method"], row["seed"]))}

    if (actual["status"] != "shortlist_ready_proposed_only" or set(actual["shortlist"]) != set(STRATEGIES)
            or sorted_rows(actual) != sorted_rows(reported)):
        raise ValueError("v2_tuning_final_shortlist_or_original_screening_mismatch")
    return {"schema": SCREENING_SCHEMA, "trial_plan_sha256": plan["content_sha256"],
            "shortlist": actual["shortlist"], "original_screening_file_sha256": reported_sha,
            "summary_files": files, "rows": rows}


def select(releases, summaries, target_verdict=None):
    receipt_by_method = {}
    plan = qualified = qualified_sha = None
    for release in releases:
        receipt, _ = read_file(release/"receipt.json")
        current_plan, _ = read_file(release/"trial-plan.json")
        current_qualified, sha = read_file(release/"qualified-screening.json")
        pins = receipt["pins"]
        method = scope(pins)
        if (method in receipt_by_method or pins["trial_plan"] != current_plan or receipt["output_folder"] != DESTINATIONS[method]
                or receipt["resume_inputs_sha256"] != digest(pins) or pins["final"]["screening_sha256"] != sha
                or current_qualified["schema"] != SCREENING_SCHEMA or current_qualified["trial_plan_sha256"] != current_plan["content_sha256"]
                or pins["final"]["shortlist"] != current_qualified["shortlist"]):
            raise ValueError("v2_tuning_final_release_binding")
        if plan is not None and (plan != current_plan or qualified_sha != sha):
            raise ValueError("v2_tuning_final_release_pair_mismatch")
        plan, qualified, qualified_sha = current_plan, current_qualified, sha
        receipt_by_method[method] = receipt
    if set(receipt_by_method) != set(STRATEGIES):
        raise ValueError("v2_tuning_final_release_pair_required")
    original = summarize(plan, qualified["rows"], phase="screening")
    if original["status"] != "shortlist_ready_proposed_only" or original["shortlist"] != qualified["shortlist"]:
        raise ValueError("v2_tuning_final_qualified_screening")

    additions, files = [], []
    for path in summaries:
        row, basis, sha = read_summary(path, plan)
        method, seed = row["method"], row["seed"]
        if method not in receipt_by_method or seed not in SEEDS or basis["modules"] != receipt_by_method[method]["pins"]["modules"]:
            raise ValueError(f"v2_tuning_final_release_summary: {path.name}")
        additions.append(row)
        files.append({"method": method, "seed": seed, "name": path.name, "sha256": sha})
    result = summarize(plan, qualified["rows"]+additions, phase="final", target_verdict=target_verdict)
    return {**result, "trial_plan_sha256": plan["content_sha256"], "qualified_screening_sha256": qualified_sha,
            "final_summary_files": files}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--releases", type=Path, nargs=2, required=True)
    parser.add_argument("--summaries", type=Path, nargs="*", default=[])
    parser.add_argument("--target-verdict", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    verdict = read_file(args.target_verdict)[0] if args.target_verdict else None
    result = canonical(select(args.releases, args.summaries, verdict))+b"\n"
    if args.output:
        with args.output.open("xb") as stream:
            stream.write(result)
    else:
        print(result.decode("utf-8"), end="")
