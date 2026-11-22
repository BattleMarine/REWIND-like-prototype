"""Issue and verify the parent decision required to activate v2.1.4."""

import argparse
import hashlib
import json
import re
from pathlib import Path

from rewind_archive.contracts import canonical, digest
from rewind_m1.local_rgb import _regular_read
from rewind_prepare.temporal_selection import check_pin, seal
from scripts.v2_tuning_cpu_selection import ADDITIONAL, INITIAL, summarize


SCHEMA = "rewind_v214_verified_parent_decision_v1"
SCREENING_SHA = "344ad4f999afb21241ff6bd50daf99874bf46f54403677f254f0cfe25b11af65"
CONTROL_ROOTS = {"v2.1.1": "reports/v211_body_head128_a_v1", "v2.1.2": "reports/v212_body_lora128_a_v1"}
CONTROL_SCHEMAS = {"v2.1.1": "rewind_v211_validation_v1", "v2.1.2": "rewind_v212_validation_v1"}
FINAL_RELEASE_RECEIPTS = {"v2.1.1": "01fd02460f10ace4d6423cb829d234eb2c1e1b3e57bdcd22eb73e3787a9bdca1",
                          "v2.1.2": "fa8e0e2c85993d3a256fc383c5eb8a4a5af7456741c6317bd1ea49e6dc6fbd89"}
DECISION_NAME = "v214-selection-decision.json"
SHA_ENV = "REWIND_V214_DECISION_SHA256"


def valid_sha(value):
    return isinstance(value, str) and re.fullmatch("[0-9a-f]{64}", value) is not None


def verify(decision, plan):
    check_pin(decision, SCHEMA, decision["content_sha256"])
    qualification = decision["qualified_screening"]
    extra = decision["additional_rows"]
    if (decision["trial_plan_sha256"] != plan["content_sha256"] or digest(qualification) != SCREENING_SHA
            or decision["qualification_sha256"] != SCREENING_SHA or decision["final_release_receipts"] != FINAL_RELEASE_RECEIPTS
            or not valid_sha(decision["selection_report_file_sha256"]) or not valid_sha(decision["target_verdict_file_sha256"])
            or set((r["method"], r["seed"]) for r in extra) != {(m, s) for m in CONTROL_ROOTS for s in ADDITIONAL}
            or len(extra) != 4 or any(row["status"] != "completed" for row in extra)):
        raise ValueError("v214_exact_final_selection_inputs_required")
    selection = summarize(plan, qualification["rows"]+extra, phase="final", target_verdict=decision["target_verdict"])
    method = selection.get("selected_method")
    if (selection["status"] != "validation_candidate" or method not in CONTROL_ROOTS or method != decision["method"]
            or decision["selected_candidate"] != selection["candidate"]):
        raise ValueError("v214_final_parent_not_selected")
    rows = {str(row["seed"]): row for row in qualification["rows"] if row["method"] == method}
    summaries = {str(row["seed"]): row["sha256"] for row in qualification["summary_files"] if row["method"] == method}
    if set(decision["controls"]) != {str(seed) for seed in INITIAL}:
        raise ValueError("v214_matched_initial_controls_required")
    for seed, control in decision["controls"].items():
        if (control["summary_sha256"] != summaries[seed] or control["checkpoint_sha256"] != rows[seed]["checkpoint_sha256"]
                or not valid_sha(control["content_sha256"]) or not 0 < control["summary_bytes"] <= 2*1024**2):
            raise ValueError("v214_control_identity")
    return method


def read_verified(path, expected_sha256, plan):
    if not valid_sha(expected_sha256):
        raise ValueError("v214_reviewed_decision_sha256_required")
    raw = _regular_read(path, 2*1024**2)
    if hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise ValueError("v214_decision_file_sha256")
    decision = json.loads(raw)
    verify(decision, plan)
    return decision


def issue(trial_plan, releases, final_summaries, target_verdict, selection_report, initial_results):
    # The builder uses the same released-summary verifier as the final selector.
    from scripts.v2_tuning_final_selection import read_file, read_summary, select

    plan = read_file(trial_plan)[0]
    verdict, verdict_sha = read_file(target_verdict)
    receipts = {}
    for release in releases:
        receipt, sha = read_file(release/"receipt.json")
        receipts[receipt["pins"]["final"]["method"]] = sha
    if receipts != FINAL_RELEASE_RECEIPTS:
        raise ValueError("v214_final_release_receipts")
    qualified = read_file(releases[0]/"qualified-screening.json")[0]
    existing, selection_sha = read_file(selection_report)
    recomputed = select(releases, final_summaries, verdict)
    if existing != recomputed or existing["status"] != "validation_candidate" or existing.get("selected_method") not in CONTROL_ROOTS:
        raise ValueError("v214_reviewed_final_result_mismatch")
    method = existing["selected_method"]
    initial_dir = initial_results[list(CONTROL_ROOTS).index(method)]
    controls = {}
    for seed in INITIAL:
        path = initial_dir/f"seed-{seed}-summary.json"
        row, _, sha = read_summary(path, plan)
        if (row["method"], row["seed"]) != (method, seed):
            raise ValueError("v214_control_method_seed")
        doc, _ = read_file(path)
        controls[str(seed)] = {"summary_sha256": sha, "summary_bytes": path.stat().st_size,
                               "content_sha256": doc["content_sha256"], "checkpoint_sha256": row["checkpoint_sha256"]}
    decision = seal({"schema": SCHEMA, "trial_plan_sha256": plan["content_sha256"], "method": method,
                    "qualification_sha256": SCREENING_SHA, "qualified_screening": qualified,
                    "additional_rows": [read_summary(path, plan)[0] for path in final_summaries],
                    "target_verdict": verdict, "target_verdict_file_sha256": verdict_sha, "final_release_receipts": receipts,
                    "selection_report_file_sha256": selection_sha, "selected_candidate": existing["candidate"],
                    "controls": controls})
    verify(decision, plan)
    return decision


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("trial-plan", "target-verdict", "selection-report", "output"):
        parser.add_argument("--"+name, type=Path, required=True)
    parser.add_argument("--releases", type=Path, nargs=2, required=True)
    parser.add_argument("--final-summaries", type=Path, nargs=4, required=True)
    parser.add_argument("--initial-results", type=Path, nargs=2, required=True, metavar=("HEAD", "LORA"))
    args = vars(parser.parse_args())
    output = args.pop("output")
    raw = canonical(issue(**args))
    with output.open("xb") as stream:
        stream.write(raw)
    print(hashlib.sha256(raw).hexdigest())
