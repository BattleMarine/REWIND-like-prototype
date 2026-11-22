"""Manual v2.1.4 wrist-weight comparison after selection approval."""

import os
import time
from functools import partial

from rewind_archive.contracts import digest
from rewind_m1.training import seed_run
from scripts import v211_run, v214_wrist
from scripts.v210_eval import FEATURE, FINAL
from scripts.v211_progress import checkpoint_raw, evaluate, fit, fit_tip, folder
from scripts.v211_trial import SEEDS, TRAINING, TRAJECTORY, UPDATES, order, trajectory_seed
from scripts.v214_decision import DECISION_NAME, SHA_ENV, read_verified
from scripts.v214_eval import control, load_predictions, publish_comparison, publish_pair
from scripts.v2_gpu_training_pipeline import CommonEvaluation
from scripts.v2_gpu_training_storage import read_sealed


DESTINATION = "reports/v214_wrist_weight128_a_v1"
SCHEMA = "rewind_v214_wrist_invocation_v1"


def trials(project, work, target, pins, reference, pipe, ledger, owner, environment, report, strategy,
           *, bound_modules, seeds, decision):
    data = pipe.data
    plan = pins["trial_plan"]
    selected = {role: pin["sha256"] for role, pin in pipe.progress["best"].items()}
    if selected != plan["common"]["selected_model_sha256"] or pipe.cfg["teacher"]["sampling_steps"] != 10 or seeds != SEEDS:
        raise ValueError("v214_selected_A_weights_or_scope")
    frozen = {"method": v214_wrist.METHOD, "parent_method": decision["method"], "wrist_position_weight": 2,
              "decision_sha256": pins["reviewed_decision"], "plan": plan["content_sha256"], "reference": FINAL,
              "source": pins["base"]["source_sha256"], "modules": bound_modules, "original_model": selected,
              "feature_identity": FEATURE, "feature_groups_sha256": digest(pins["groups"]), "trajectory": TRAJECTORY,
              "training": TRAINING, "trainable": list(strategy.TRAINABLE), "comparison_common": plan["common"]}
    names = data.spec["body_joint_names"] + [name for side in ("left", "right") for name in data.spec["hands"][side][1:]]
    preparation = {}
    report["phase"] = "verify_all_weight_1_controls"
    start = time.monotonic()
    for seed in seeds:
        ordered = order(data.clips["train"], seed)
        if len(ordered) != UPDATES or len(set(ordered)) != UPDATES:
            raise ValueError("v214_train_clip_coverage")
        basis = {**frozen, "seed": seed, "ordered_train_clips_sha256": digest(ordered)}
        old, old_summary = control(project, decision, seed, basis, data, names, ledger)
        preparation[seed] = ordered, basis, old, old_summary
    report["weight_1_verify_seconds"] = time.monotonic()-start
    store = cascade = None
    pairs = []
    for seed, (ordered, basis, old, old_summary) in preparation.items():
        ledger.check()
        target_seed = folder(target/f"seed-{seed}")
        timing = report["trials"][str(seed)] = {"status": "in_progress"}
        summary_path = target_seed/"summary.json"
        if summary_path.exists():
            summary = read_sealed(summary_path, v214_wrist.SUMMARY)
            tip = fit_tip(target_seed, basis, v214_wrist.FIT)
            if (not tip or tip["step"] != UPDATES or summary["binding"] != {**basis, "fit": tip["content_sha256"],
                    "body_sha256": tip["artifact"]["sha256"], "sampling_steps": 10}):
                raise ValueError("v214_completed_trial_binding")
            checkpoint_raw(target_seed, tip, ledger)
        else:
            if store is None:
                report["phase"] = "restore_A_features"
                start = time.monotonic()
                store, allocated = v211_run.restore_features(project, work, pins, pipe, ledger)
                first = data.clips["train"][0]
                cascade = CommonEvaluation(data, {"train": {first: store.load(first, "train")}}, pipe.cfg, pipe.device, 0)
                report["features"] = {"bytes": allocated, "wall_seconds": time.monotonic()-start}
            report["phase"] = "wrist_weight_2_fit"
            seed_run(trajectory_seed(seed, "model-initialization"))
            body = v214_wrist.make_body(cascade, pipe)
            tip = fit(body, ordered, target_seed, work, basis, store, cascade.body_idx, ledger, owner, environment,
                      timing, strategy=strategy)
            hand = cascade.make("hand", body.checkpoint_sha256)
            pipe._load_best("hand", hand)
            report["phase"] = "wrist_weight_2_validation"
            summary = evaluate(cascade, body, hand, store, target_seed, work,
                               {**basis, "fit": tip["content_sha256"], "body_sha256": body.checkpoint_sha256,
                                "sampling_steps": 10}, plan, reference, ledger, owner, environment, timing,
                               summary_schema=v214_wrist.SUMMARY)
            del body, hand
        report["phase"] = "matched_weight_comparison"
        new = load_predictions(target_seed, summary, data, names, ledger)
        pairs.append(publish_pair(target_seed/"comparison.json", old, new, old_summary, summary, seed, data.spec, owner))
        timing.update(status="completed", summary_sha256=summary["content_sha256"],
                      comparison_sha256=pairs[-1]["content_sha256"])
    comparison = publish_comparison(target/"comparison.json", pairs, decision["method"], pins["reviewed_decision"], owner)
    report["comparison_sha256"] = comparison["content_sha256"]


def run(args, pins, report):
    report["phase"] = "verify_reviewed_parent_decision"
    expected = os.getenv(SHA_ENV, "")
    decision = read_verified(args.work.resolve(strict=True).parent/DECISION_NAME, expected, pins["trial_plan"])
    report.update(parent_method=decision["method"], reviewed_decision_sha256=expected)
    v211_run.run(args, {**pins, "reviewed_decision": expected}, report, strategy=v214_wrist.strategy(decision["method"]),
                 destination=DESTINATION, run_schema=SCHEMA, trial_runner=partial(trials, decision=decision))


if __name__ == "__main__":
    v211_run.main(run, __doc__)
