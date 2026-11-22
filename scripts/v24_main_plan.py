"""Draft the paired v2.4 main comparison, or attach verified R2 pilot evidence."""

import argparse
import hashlib
import json
import math
from pathlib import Path
import re


ARMS = ("x0", "v")
PRIMARY = (42, 71, 103)
ADDITIONAL = (137, 181)
R2_RECEIPT_SHA = "18b680fd7d32b8bf60be75ca1f7e26133d5d46fbeec35f1792504d53dd6a719d"
MIB = 1024**2
PILOT_FILES = {"numerical.json", "pilot-plan.json", "x0-pilot.json", "v-pilot.json", "pilot-summary.json",
               "x0-pilot-final.pt", "v-pilot-final.pt", "worker-result.json"}
ROLES = ("body", "hand_global", "hand_local")
SIDES = ("left", "right")


def canonical(value):
    return json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":")).encode()


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_small(path, cap):
    require(path.is_file() and not path.is_symlink(), f"Missing or linked input: {path.name}")
    with path.open("rb") as stream:
        raw = stream.read(cap + 1)
    require(0 < len(raw) <= cap, f"Input byte cap: {path.name}")
    return raw


def rows(seeds):
    return [{"arm": arm, "seed": seed, "status": "proposed", "authorization": "pending"}
            for seed in seeds for arm in ARMS]


def template():
    return {"schema": "rewind_v24_main_comparison_draft_v1", "status": "awaiting_pilot",
            "initial_paired_trials": rows(PRIMARY), "conditional_paired_trials": rows(ADDITIONAL),
            "same_seed_controls": {"parent": "fixed_v2_128_A", "seed_specific_equal_final_head_reinitialization": True,
                                   "x0_frame_head_both_arms": True, "same_clip_order_and_forward_noise": True,
                                   "same_x0_space_objective": True, "upstream_frontend_and_hand": "frozen",
                                   "evaluation_noise_seed": 71, "ddim_steps": 10, "device_mm": True},
            "evaluation_plan": {"selection_scope": "same_official_validation",
                                "expected_clips": 421, "expected_real_frames": 16425,
                                "primary_metric": "equal_scenario_body_and_hand_global_macro_mpjpe_mm",
                                "also_report": ["hand_local", "left_wrist", "right_wrist", "counts", "learning_curves"],
                                "independent_test_scope": "reserved"},
            "required_decisions": {"training_scope_and_manifest": None, "updates_per_arm": None,
                                   "batch_size_and_learning_rate": None, "validation_update_boundaries": None,
                                   "initialization_and_upstream_binding": None, "gpu_precision_and_environment": None,
                                   "read_write_scratch_and_time_budget": None, "convergence_and_stop_rule": None,
                                   "conditional_extension_criterion_and_budget": None},
            "activation_gates": ["verified_completed_pilot", "signed_common_main_protocol_and_budget",
                                 "implemented_and_reviewed_seed_parameterized_worker_and_resume",
                                 "user_approved_main_gpu_run"],
            "extension_gates": ["all_three_initial_seed_pairs_complete_and_comparable",
                                "predeclared_paired_extension_criterion_met",
                                "separate_budget_and_execution_approval"],
            "pilot": None}


def model_pin(path, expected, cap):
    require(path.is_file() and not path.is_symlink() and 0 < path.stat().st_size == expected["bytes"] <= cap,
            f"Invalid state size: {path.name}")
    total = 0
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(MIB):
            total += len(block)
            require(total <= cap, f"State growth: {path.name}")
            digest.update(block)
    require(expected == {"bytes": total, "sha256": digest.hexdigest()}, f"State SHA mismatch: {path.name}")
    return expected


def finite(value):
    require(type(value) in (int, float) and math.isfinite(value) and value >= 0, "Invalid pilot measurement")
    return value


def read_result_directory(pilot_root, receipt_path, result_sha):
    raw = read_small(receipt_path, MIB)
    require(sha(raw) == R2_RECEIPT_SHA, "Expected fixed R2 release receipt")
    pins = json.loads(raw)["pins"]
    root = pilot_root.absolute()
    require(root.resolve(strict=True) == root and root.is_dir(), "Expected direct pilot result directory")
    raw = read_small(root / "result.json", 64 * 1024)
    require(re.fullmatch("[0-9a-f]{64}", result_sha) and sha(raw) == result_sha, "Pilot result file SHA mismatch")
    result = json.loads(raw)
    require(result["schema"] == "rewind_v24_invocation_v1" and result["mode"] == "pilot"
            and result["status"] == "completed_pilot" and result["returncode"] == 0
            and result["limits"] == pins["limits"] and result["write_budget"] == pins["write_budget"]
            and result["source_sha256"] == pins["base"]["source_sha256"]
            and result["kit_sha256"] == pins["toolkit"]["sha256"]
            and set(result["files"]) == PILOT_FILES, "Pilot has not completed under the fixed R2 input and budget")
    documents, states = {}, {}
    for name, expected in result["files"].items():
        path = root / name
        if name.endswith(".pt"):
            states[name] = model_pin(path, expected, pins["limits"]["state_bytes"])
        else:
            data = read_small(path, pins["limits"]["report_bytes"])
            require(expected == {"bytes": len(data), "sha256": sha(data)}, f"Pilot report SHA mismatch: {name}")
            documents[name] = json.loads(data)
    return pins, result, documents, states


def score_values(score):
    entries = {role: score["official"][role] for role in ROLES}
    entries.update({f"{side}_wrist": score["sides"][side]["wrist"] for side in SIDES})
    values = {}
    for name, entry in entries.items():
        count = entry["valid_joints"]
        require(type(count) is int and count >= 0, "Invalid pilot joint count")
        measurement = entry["mpjpe_mm"]
        values[name] = {"mpjpe_mm": finite(measurement) if count else None, "valid_joints": count}
        require(count > 0 or measurement is None, "Pilot empty metric must be null")
    return values


def curve_values(arm, checkpoints, validation_keys):
    curve = []
    for point in checkpoints:
        key = f"{arm} update {point['update']}"
        require(point["scope"] == "pilot_validation_subset" and point["pooled"]["frames"] == 300
                and [clip["key"] for clip in point["clips"]] == validation_keys
                and all(clip["real_frames"] == 50 for clip in point["clips"]),
                f"Pilot validation scope: {key}")
        curve.append({"updates": point["update"], "pooled": score_values(point["pooled"]),
                      "per_scenario": {scene: score_values(scores) for scene, scores in point["per_scenario"].items()}})
    require([row["updates"] for row in curve] == [0, 12, 24], "Pilot eval cursor")
    return curve


def from_pilot(pilot_root, receipt_path, result_sha):
    pins, result, doc, states = read_result_directory(pilot_root, receipt_path, result_sha)
    binding = doc["pilot-plan.json"]["binding"]
    worker = doc["worker-result.json"]
    summary = doc["pilot-summary.json"]
    math_proof = doc["numerical.json"]
    require(worker["schema"] == "rewind_v24_worker_v1" and worker["status"] == "completed_pilot"
            and worker["phase"] == "finished" and bool(worker["environment"]["gpu"])
            and worker["environment"]["precision"] == "fp32" and worker["environment"]["tf32"] is False
            and worker["write_budget"] == pins["write_budget"]
            and worker["files"] == {k: v for k, v in result["files"].items() if k != "worker-result.json"}
            and doc["pilot-plan.json"]["binding_sha256"] == sha(canonical(binding))
            and math_proof["frame_x0_passthrough"] is True and math_proof["time_mask_preserved"] is True
            and all(finite(math_proof[name]) <= 1e-12 for name in ("x0_max_abs", "epsilon_max_abs",
                                                                 "DDIM10_output_max_abs", "masked_position_loss_abs_difference"))
            and summary["scope"] == "pilot_validation_subset" and summary["paired_noise_match"] is True
            and summary["frozen_upstream_state_sha256"] == binding["frozen_upstream_state_sha256"]
            and binding["seed"] == pins["scope"]["seed"] == 42 and binding["updates_per_arm"] == pins["scope"]["updates_per_arm"] == 24
            and len(binding["train_keys"]) == 24 and len(binding["validation_keys"]) == 6,
            "Pilot report scope, frozen upstream or state pins disagree")
    measurements = {}
    prior = None
    for arm in ARMS:
        result_arm = doc[f"{arm}-pilot.json"]
        checkpoints = result_arm["curve"]
        require(summary["arms"][arm]["validation_curve"] == checkpoints, "Pilot curve mismatch")
        require(states[f"{arm}-pilot-final.pt"] == result_arm["final_state"] == summary["arms"][arm]["final"],
                "Pilot final checkpoint pin")
        updates = result_arm["updates"]
        require([s["step"] for s in updates] == list(range(1, 25))
                and [s["key"] for s in updates] == binding["train_keys"], "Pilot update cursor mismatch")
        matched = [[s["initial_noise_sha256"] for s in updates],
                   [[[clip["key"], clip["noise_sha256"]] for clip in point["clips"]] for point in checkpoints]]
        require(prior is None or matched == prior, "Pilot paired inputs or evaluation noise differ")
        prior = matched
        measurements[arm] = {"fit_and_eval_and_save_seconds": finite(result_arm["fit_and_eval_seconds"]),
                             "curve": curve_values(arm, checkpoints, binding["validation_keys"]),
                             "final_state": states[f"{arm}-pilot-final.pt"]}
    draft = template()
    draft.update(status="pilot_verified_main_decisions_pending",
                 pilot={"result_sha256": result_sha, "release_receipt_sha256": R2_RECEIPT_SHA,
                        "environment": worker["environment"], "elapsed_seconds": finite(result["elapsed_seconds"]),
                        "worker_usage": worker["usage"], "arms": measurements, "scope": "six_clip_pilot_validation",
                        "measurements_note": "arm time includes fit, interim evaluation and save; full validation rate is unmeasured"})
    return draft


def publish(output, document):
    require(not output.exists() and not output.is_symlink(), "Use a new output file")
    payload = canonical(document) + b"\n"
    with output.open("xb") as stream:
        stream.write(payload)
    return {"path": str(output), "bytes": len(payload), "sha256": sha(payload)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot-directory", type=Path)
    parser.add_argument("--pilot-release-receipt", type=Path)
    parser.add_argument("--pilot-result-sha256")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    supplied = (args.pilot_directory, args.pilot_release_receipt, args.pilot_result_sha256)
    if any(supplied) and not all(supplied):
        parser.error("Provide all three pilot evidence arguments")
    draft = from_pilot(*supplied) if all(supplied) else template()
    print(json.dumps(publish(args.output, draft), ensure_ascii=False))


if __name__ == "__main__":
    main()
