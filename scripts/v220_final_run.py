"""Extend v2.2 with matched seeds 137 and 181, reusing completed initial evidence."""

from functools import partial
from statistics import mean, stdev
import time

from rewind_archive.contracts import digest
from rewind_m1.training import seed_run
from scripts import v211_run, v220_hand
from scripts.v210_eval import CAP as EVAL_CAP, FEATURE, FINAL, original_json
from scripts.v211_progress import checkpoint_raw, fit_tip, folder
from scripts.v211_trial import SEEDS as INITIAL, TRAINING, TRAJECTORY, UPDATES, order, trajectory_seed
from scripts.v220_eval import COMPARISON, COUNTS, SUMMARY, evaluate, trial_scores
from scripts.v220_final_upper import INITIAL_ROOT, SEEDS, extend, initial_predictions
from scripts.v220_upper import ASSIGNMENT, CAP as BODY_CAP, CONTROL, MIXED, POLICIES, assignments
from scripts.run_v2_gpu_training import training_config
from scripts.v2_gpu_training_pipeline import CommonEvaluation
from scripts.v2_gpu_training_storage import EVAL_SEED, GIB, MIB, publish_json, read_sealed


DESTINATION = "reports/v220_hand_upper128_a_final_r2_v1"
SCHEMA = "rewind_v220_final_r2_invocation_v1"
FIVE = "rewind_v220_final_paired_comparison_v1"
IO_LIMITS = {"read_bytes": 16*GIB, "write_bytes": 8*GIB}


def io_budget(pins, parent_bytes):
    trials = len(SEEDS)*len(POLICIES)
    fit = trials*3*v220_hand.CAP
    recovery = trials*v220_hand.CAP
    delta = 130*BODY_CAP  # At most groups 0..129 can contain a train clip.
    evaluation = trials*18*EVAL_CAP
    read = {"parent_copy_and_readback": 2*parent_bytes,
            "A_features": sum(item["archive"]["bytes"]+1 for item in pins["groups"].values()),
            "initial_body_groups": 147*BODY_CAP, "delta_publish_and_verify": 3*delta,
            "fit_publish_and_readback": 2*fit, "verify_or_resume_one_fit_per_trial": recovery,
            "evaluation_publish_and_verify": 3*evaluation,
            "additional_evaluation_reverify_pass": evaluation,
            "extra_fit_publish_and_readback": 2*recovery, "terminal_and_preflight_margin": 256*MIB}
    write = {"parent_copy": parent_bytes, "parent_extract_reservation": training_config(42)["resources"]["archive_bytes"],
             "delta_publish": delta, "scheduled_fit_publish": fit, "evaluation_publish": evaluation,
             "extra_fit_publish": recovery, "terminal_and_preflight_margin": 256*MIB}
    bounds = {"read": read, "write": write}
    totals = {kind: sum(parts.values()) for kind, parts in bounds.items()}
    if any(total >= IO_LIMITS[f"{kind}_bytes"] for kind, total in totals.items()):
        raise ValueError("v220_final_single_invocation_budget")
    return {"limits": IO_LIMITS, "parent_archive_bytes": parent_bytes, "bounded_components_bytes": bounds,
            "bounded_total_bytes": totals,
            "headroom_bytes": {kind: IO_LIMITS[f"{kind}_bytes"]-total for kind, total in totals.items()}}


def pair(seed, control, mixed):
    a, left = trial_scores(control)
    b, right = trial_scores(mixed)
    if left.keys() != right.keys() or any(left[name]["body"] != right[name]["body"] for name in left):
        raise ValueError("v220_final_paired_frozen_body_mismatch")
    per_scenario = {name: {metric: right[name][metric]["mpjpe_mm"]-left[name][metric]["mpjpe_mm"]
                           for metric in ("hand_local", "hand_global")} for name in left}
    return {"seed": seed, "control": a, "mixed": b, "per_scenario_delta_mm": per_scenario,
            "delta_mm": {key: b[key]-a[key] for key in a},
            "summary_sha256": {CONTROL: control["content_sha256"], MIXED: mixed["content_sha256"]},
            "hand_checkpoint_sha256": {CONTROL: control["hand_checkpoint_sha256"], MIXED: mixed["hand_checkpoint_sha256"]}}


def combined(path, original, added, index, owner):
    pairs = original["pairs"]+added
    if tuple(row["seed"] for row in pairs) != INITIAL+SEEDS:
        raise ValueError("v220_final_requires_five_matched_seeds")
    frozen_body = [2*values["body_hand_global_macro_mm"]-values["hand_global_macro_mm"]
                   for row in pairs for values in (row["control"], row["mixed"])]
    if max(frozen_body)-min(frozen_body) > 1e-9:
        raise ValueError("v220_final_frozen_body_macro_changed")
    keys = pairs[0]["delta_mm"]
    result = {"initial_comparison_sha256": original["content_sha256"], "initial_cache_sha256": original["predicted_body_cache_sha256"],
              "extension_cache_sha256": index["content_sha256"], "pairs": pairs,
              "mean_delta_mm": {key: mean(p["delta_mm"][key] for p in pairs) for key in keys},
              "sample_stdev_delta_mm": {key: stdev(p["delta_mm"][key] for p in pairs) for key in keys}}
    if path.exists():
        saved = read_sealed(path, FIVE)
        if saved["content_sha256"] != digest({"schema": FIVE, **result}):
            raise ValueError("v220_final_combined_report_changed")
        return saved
    return publish_json(path, FIVE, result, owner)


def trials(project, work, target, pins, reference, pipe, ledger, owner, environment, report, strategy,
           *, bound_modules, seeds, original):
    data, plan = pipe.data, pins["trial_plan"]
    best = {role: item["sha256"] for role, item in pipe.progress["best"].items()}
    if seeds != SEEDS or best != plan["common"]["selected_model_sha256"] or pipe.cfg["teacher"]["sampling_steps"] != 10:
        raise ValueError("v220_final_scope_or_A_weights")
    common = {"method": strategy.METHOD, "plan": plan["content_sha256"], "parent_receipt": FINAL,
              "source": pins["base"]["source_sha256"], "module_shas": bound_modules, "original_model": best,
              "feature_identity": FEATURE, "feature_groups_sha256": digest(pins["groups"])}
    report["phase"] = "restore_A_features"
    start = time.monotonic()
    store, allocated = v211_run.restore_features(project, work, pins, pipe, ledger)
    report["features"] = {"bytes": allocated, "wall_seconds": time.monotonic()-start}
    first = data.clips["train"][0]
    cascade = CommonEvaluation(data, {"train": {first: store.load(first, "train")}}, pipe.cfg, pipe.device, 0)
    body = cascade.make("body")
    pipe._load_best("body", body)
    report["phase"] = "verify_initial_body_prediction_cache"
    before = time.monotonic()
    prior_sha = original["predicted_body_cache_sha256"]
    initial = {**common, "module_shas": pins["initial"]["module_shas"]}
    prior = initial_predictions(project/INITIAL_ROOT/"predicted-body", prior_sha, initial, body, data, len(cascade.body_names), ledger)
    report["initial_cache_verify_seconds"] = time.monotonic()-before
    report["phase"] = "extend_body_prediction_cache"
    planned = {seed: assignments(data.clips["train"], seed) for seed in seeds}
    scope = {**common, "initial_comparison_sha256": original["content_sha256"], "initial_cache_sha256": prior_sha}
    cache, index = extend(folder(target/"predicted-body-delta"), work, prior, store, cascade, body, planned, scope,
                          ledger, owner, environment, report.setdefault("body_prediction_cache", {}))
    body_pin = body.checkpoint_sha256
    del body, prior
    report["body_prediction_cache"]["index_sha256"] = index["content_sha256"]
    policy_dirs = {policy: folder(target/policy) for policy in POLICIES}
    reports = {}
    for seed in seeds:
        ordered = order(data.clips["train"],seed)
        if len(ordered) != UPDATES:
            raise ValueError("v220_final_train_clip_count")
        for policy in POLICIES:
            ledger.check()
            tag = f"{policy}/seed-{seed}"
            report.update(phase="final_hand_fit", active_trial=tag)
            timing = report["trials"][tag] = {"status": "in_progress"}
            dest = folder(policy_dirs[policy]/f"seed-{seed}")
            chosen = frozenset() if policy == CONTROL else planned[seed]
            basis = {**scope, "seed": seed, "policy": policy, "body": body_pin,
                     "predicted_body_cache_sha256": index["content_sha256"], "assignment": ASSIGNMENT,
                     "predicted_train_keys_sha256": digest(sorted(chosen)), "ordered_train_keys_sha256": digest(ordered),
                     "trajectory": TRAJECTORY, "training": TRAINING, "evaluation_seed": EVAL_SEED, "sampling_steps": 10}
            path = dest/"summary.json"
            if path.exists():
                summary = read_sealed(path, SUMMARY)
                tip = fit_tip(dest, basis, strategy.FIT)
                if (not tip or tip["step"] != UPDATES or summary["binding"] != {**basis,"fit":tip["content_sha256"]}
                        or summary["hand_checkpoint_sha256"] != tip["artifact"]["sha256"] or summary["counts"] != COUNTS
                        or summary["upper_counts"] != tip["upper_counts"]
                        or tip["upper_counts"] != strategy.source_counts(ordered, chosen, UPDATES)):
                    raise ValueError("v220_final_completed_trial_binding")
                checkpoint_raw(dest,tip,ledger,strategy.CAP)
            else:
                seed_run(trajectory_seed(seed,"model-initialization"))
                hand = strategy.make_hand(cascade,pipe,policy,body_pin)
                tip = strategy.fit(hand,ordered,chosen,dest,work,basis,store,cache,cascade,ledger,owner,environment,timing)
                report["phase"] = "final_hand_validation"
                summary = evaluate(cascade,hand,store,cache,dest,work,{**basis,"fit":tip["content_sha256"]},tip,reference,
                                   ledger,owner,environment,timing)
                del hand
            reports[policy,seed] = summary
            timing.update(status="completed", summary_sha256=summary["content_sha256"], upper_counts=summary["upper_counts"],
                          hand_checkpoint_sha256=summary["hand_checkpoint_sha256"])
    report.pop("active_trial",None)
    report["phase"] = "combined_five_seed_comparison"
    added = [pair(seed,reports[CONTROL,seed],reports[MIXED,seed]) for seed in seeds]
    report["comparison_sha256"] = combined(target/"comparison.json",original,added,index,owner)["content_sha256"]


def run(args, pins, report):
    report["phase"] = "verify_completed_initial_comparison"
    if (pins["final_seeds"] != list(SEEDS) or pins["trial_plan"]["conditional_additional_seeds"] != list(SEEDS)
            or pins["trial_plan"]["initial_seeds"] != list(INITIAL)):
        raise ValueError("v220_final_additional_seeds_only")
    pin = pins["initial"]["comparison"]
    budget = pins["io_budget"]
    if budget != io_budget(pins, budget["parent_archive_bytes"]):
        raise ValueError("v220_final_io_budget_pin")
    original = original_json(args.project.resolve(strict=True), f"{INITIAL_ROOT}/comparison.json",
                             {"bytes":pin["bytes"],"sha256":pin["sha256"]},COMPARISON)
    if original["content_sha256"] != pin["content_sha256"] or tuple(row["seed"] for row in original["pairs"]) != INITIAL:
        raise ValueError("v220_final_initial_comparison_binding")
    report.update(initial_comparison_sha256=original["content_sha256"],final_seeds=list(SEEDS),io_budget=budget)
    v211_run.run(args,pins,report,strategy=v220_hand,destination=DESTINATION,run_schema=SCHEMA,seeds=SEEDS,
                 trial_runner=partial(trials,original=original),io_limits=budget["limits"])


if __name__ == "__main__":
    v211_run.main(run,__doc__)
