"""Preflight trial matrix and CPU screening/final comparison for the body sweep."""

from collections import defaultdict
import math
import re
from statistics import mean, stdev

from scripts.v2_tuning_cpu_common import DESIGN, canonical, require, sha


INITIAL = (42, 71, 103)
ADDITIONAL = (137, 181)
BODY = ("v2.1.1", "v2.1.2", "v2.1.3")
SCENARIOS = ("ArchVisInterior_ArchVis_RT", "CityPark_Showcase", "DowntownWest_Demo_Environment")
METRICS = ("body_mpjpe_mm", "hand_global_mpjpe_mm")


def plan(identity, selected_models, parent_macro):
    common = {"parent": identity, "selected_model_sha256": selected_models, "resolution": [128, 128],
              "training_pairs": 119586, "training_clips": 3097, "validation_pairs": 16425,
              "validation_clips": 421, "evaluation_noise_seed": 71, "split": "official_validation",
              "input_2d": "predicted", "evaluation_upper": "predicted", "precision": "fp32"}
    trial = {"updates": 3097, "epoch": 1, "batch_size": 1, "learning_rate": 5e-5}
    phases = {
        "v2.1.0": {"train": False, "prerequisite": "full_validation_per_joint_prediction_export"},
        "v2.1.1": {"scope": "body_output_weight_bias", **trial},
        "v2.1.2": {"scope": "body_temporal_qkv", "adapter_layers": 3, "rank": 4,
                   "alpha": 4, "adapter_dropout": 0, "zero_initial_delta": True, **trial},
        "v2.1.3": {"scope": "body_all", **trial},
        "v2.1.4": {"prerequisite": "selected_v2.1.1_to_v2.1.3", "wrist_position_weight": 2,
                   "control_wrist_position_weight": 1, **trial},
        "v2.2": {"scope": "hand_all", "conditions": ["gt_100_pred_0", "gt_50_pred_50"], **trial},
        "v2.3": {"train": False, "ddim_steps": [10, 5, 2], "timed_repeats": 5,
                 "prerequisite": "same_device_full_validation_and_latency"},
        "v2.4": {"conditions": ["x0", "v"], "ddim_steps": 10,
                 "prerequisite": "fixed_initialization_update_budget_and_v_loss"},
    }
    body_rows = [{"method": method, "seed": seed} for method in BODY for seed in INITIAL]
    require(len({(row["method"], row["seed"]) for row in body_rows}) == 9
            and len(body_rows) + 2 * len(ADDITIONAL) == 13, "invalid trial matrix")
    result = {"schema": "rewind_v2_tuning_cpu_trial_plan_v1", "design_revision": DESIGN,
              "status": "proposed_cpu_contract_no_launch", "common": common, "parent_macro_mm": parent_macro,
              "initial_seeds": list(INITIAL), "conditional_additional_seeds": list(ADDITIONAL),
              "body_initial_trials": body_rows, "body_max_trials_if_extended": 13, "branches": phases,
              "unfixed": ["approved_gpu_budget", "evaluation_frequency", "target_metric_protocol",
                          "v2.2_extension_criterion", "v2.3_accuracy_limit", "v2.4_initialization_and_budget"]}
    return seal(result)


def seal(body):
    return {**body, "content_sha256": sha(canonical(body))}


def macro(parts):
    require(set(parts) == set(SCENARIOS), "score scenario coverage")
    numbers = []
    for name in SCENARIOS:
        require(set(parts[name]) == set(METRICS), "score joint groups")
        numbers.extend(parts[name][key] for key in METRICS)
    require(all(type(score) in (int, float) and math.isfinite(score) and score >= 0 for score in numbers),
            "score values must be finite nonnegative mm")
    return mean(numbers), mean([parts[name][METRICS[1]] for name in SCENARIOS])


def summarize(trial_plan, measurements, *, phase, target_verdict=None):
    """Evaluate already-collected rows. Missing, failed, and incomparable trials never score."""
    require(phase in ("screening", "final"), "invalid selection phase")
    body = {k: v for k, v in trial_plan.items() if k != "content_sha256"}
    require(trial_plan["schema"] == "rewind_v2_tuning_cpu_trial_plan_v1"
            and trial_plan == seal(body), "trial plan SHA mismatch")
    expected_pin = trial_plan["content_sha256"]
    parent = trial_plan["parent_macro_mm"]
    allowed_seeds = set(INITIAL + ADDITIONAL) if phase == "final" else set(INITIAL)
    scores, rows, seen = defaultdict(dict), [], set()
    for row in measurements:
        method, seed = row["method"], row["seed"]
        require(method in BODY and type(seed) is int and seed in allowed_seeds
                and row["trial_plan_sha256"] == expected_pin, "trial does not match the plan")
        key = (method, seed)
        require(key not in seen, "duplicate trial; resume must retain one identity")
        seen.add(key)
        status = row["status"]
        require(status in ("completed", "stopped", "failed", "missing"), "unknown trial status")
        summary = {"method": method, "seed": seed, "status": status}
        if status == "completed":
            require(row["comparison_common"] == trial_plan["common"]
                    and row["updates"] == trial_plan["branches"][method]["updates"]
                    and row["training"] == {name: trial_plan["branches"][method][name]
                                            for name in ("learning_rate", "batch_size", "epoch")}
                    and re.fullmatch("[0-9a-f]{64}", row["checkpoint_sha256"]), "trial common inputs or budget mismatch")
            overall, hand = macro(row["scenario_scores"])
            summary.update(macro_mm=overall, hand_global_macro_mm=hand, delta_parent_mm=overall - parent,
                           checkpoint_sha256=row["checkpoint_sha256"], scenario_scores=row["scenario_scores"])
            scores[method][seed] = summary
        else:
            require("scenario_scores" not in row, "incomplete trial supplied a score")
        rows.append(summary)
    screened, missing = aggregate(scores, BODY, INITIAL)
    if missing:
        return {"status": "awaiting_screening_rows", "missing": missing, "rows": rows, "methods": screened}
    ranking = sorted(screened, key=lambda method: screened[method]["mean_mm"])
    shortlist = ranking[:2]
    if screened[ranking[0]]["mean_mm"] >= parent:
        return {"status": "keep_parent_no_screening_improvement", "rows": rows, "methods": screened, "shortlist": []}
    if screened[ranking[1]]["mean_mm"] == screened[ranking[2]]["mean_mm"]:
        return {"status": "shortlist_tie_requires_decision", "rows": rows, "methods": screened}
    if phase == "screening":
        return {"status": "shortlist_ready_proposed_only", "rows": rows, "methods": screened, "shortlist": shortlist}
    require(all(row["seed"] in INITIAL or row["method"] in shortlist for row in rows), "extra seed outside shortlist")
    final, missing = aggregate(scores, shortlist, INITIAL + ADDITIONAL)
    result = {"rows": rows, "methods": final, "shortlist": shortlist}
    if missing:
        return {**result, "status": "awaiting_final_rows", "missing": missing}
    candidates = [method for method in shortlist if final[method]["mean_mm"] < parent
                  and final[method]["improved_seed_count"] >= 3]
    if candidates and (target_verdict is None or any(type(target_verdict.get(method)) is not bool for method in candidates)):
        return {**result, "status": "awaiting_target_metric_verdict", "candidates": candidates}
    approved = [method for method in candidates if target_verdict[method]]
    if not approved:
        return {**result, "status": "keep_parent"}
    approved.sort(key=lambda method: final[method]["mean_mm"])
    if len(approved) > 1 and final[approved[0]]["mean_mm"] == final[approved[1]]["mean_mm"]:
        return {**result, "status": "final_method_tie_requires_decision"}
    selected = approved[0]
    c = min(scores[selected].values(), key=lambda row: (row["macro_mm"], row["hand_global_macro_mm"], row["seed"]))
    return {**result, "status": "validation_candidate", "selected_method": selected, "candidate": c}


def aggregate(scores, methods, seeds):
    result, missing = {}, []
    for method in methods:
        missing.extend({"method": method, "seed": seed} for seed in seeds if seed not in scores[method])
        if all(seed in scores[method] for seed in seeds):
            values = [scores[method][seed]["macro_mm"] for seed in seeds]
            deltas = [scores[method][seed]["delta_parent_mm"] for seed in seeds]
            result[method] = {"seeds": list(seeds), "scores_mm": values, "mean_mm": mean(values),
                              "sample_stdev_mm": stdev(values), "range_mm": [min(values), max(values)],
                              "mean_delta_parent_mm": mean(deltas),
                              "improved_seed_count": sum(delta < 0 for delta in deltas)}
    return result, missing
