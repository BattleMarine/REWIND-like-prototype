"""One manual CUDA invocation of v2.2 matched upper-condition hand trials."""

import time
import uuid
from pathlib import Path
from types import SimpleNamespace

from rewind_archive.contracts import digest
from rewind_m1.training import seed_run
from rewind_prepare.temporal_cache import _hash_file
from rewind_v1 import v2_retention as retention
from scripts.run_v2_bulk_cache import writer
from scripts.run_v2_gpu_training import identity_for, load_inputs, read_journal, training_config
from scripts.v210_eval import A, FEATURE, FINAL, input_file
from scripts.v211_progress import checkpoint_raw, fit_tip, folder
from scripts.v211_run import TrialLedger, collect_peak, initialize_cuda, inputs, main, record_failure, restore_features
from scripts.v211_trial import SEEDS, TRAINING, TRAJECTORY, UPDATES, order, trajectory_seed
from scripts.v220_eval import COUNTS, SUMMARY, evaluate, paired
from scripts.v220_hand import BRANCH, CAP, FIT, METHOD, fit, make_hand, source_counts
from scripts.v220_upper import ASSIGNMENT, CONTROL, POLICIES, assignments, prepare_upper
from scripts.v2_gpu_training_pipeline import CommonEvaluation, FullCache, FullData, Training
from scripts.v2_gpu_training_storage import EVAL_SEED, GIB, publish_json, read_sealed


DESTINATION = "reports/v220_hand_upper128_a_v1"
SCHEMA = "rewind_v220_invocation_v1"
CONTRACT = {"conditions": list(POLICIES), "seeds": list(SEEDS), "assignment": ASSIGNMENT,
            "mixed_predicted_clips": UPDATES//2, "evaluation_seed": EVAL_SEED, "sampling_steps": 10}


def trials(project, work, target, pins, reference, pipe, ledger, owner, report):
    data = pipe.data
    environment = report["environment"]
    best = {role: value["sha256"] for role, value in pipe.progress["best"].items()}
    plan_pin = pins["trial_plan"]["content_sha256"]
    if best != pins["trial_plan"]["common"]["selected_model_sha256"] or pipe.cfg["teacher"]["sampling_steps"] != 10:
        raise ValueError("v220_parent_weights_or_steps")
    report["phase"] = "restore_A_features"
    start = time.monotonic()
    store, allocated = restore_features(project, work, pins, pipe, ledger)
    report["features"] = {"bytes": allocated, "groups": 147, "wall_seconds": time.monotonic()-start}
    keys = data.clips["train"]
    exemplar = keys[0]
    cascade = CommonEvaluation(data, {"train": {exemplar: store.load(exemplar, "train")}}, pipe.cfg, pipe.device, 0)
    body = cascade.make("body")
    pipe._load_best("body", body)
    planned = {seed: assignments(keys, seed) for seed in SEEDS}
    common = {"method": METHOD, "plan": plan_pin, "parent_receipt": FINAL, "source": pins["base"]["source_sha256"],
              "module_shas": pins["modules"], "original_model": best, "feature_identity": FEATURE,
              "feature_groups_sha256": digest(pins["groups"])}
    report["phase"] = "prepare_A_body_prediction_cache"
    start = time.monotonic()
    stats = report["body_prediction_cache"] = {}
    cache, index = prepare_upper(folder(target/"predicted-body"), work, store, cascade, body, planned, common,
                                 ledger, owner, environment, stats)
    stats.update(wall_seconds=time.monotonic()-start, index_sha256=index["content_sha256"], counts=index["counts"])
    body_pin = body.checkpoint_sha256
    del body
    policy_dirs = {policy: folder(target/policy) for policy in POLICIES}
    summaries = {}
    for seed in SEEDS:
        ordered = order(keys, seed)
        if len(ordered) != UPDATES:
            raise ValueError("v220_order_coverage")
        for policy in POLICIES:
            ledger.check()
            report["phase"] = "hand_fit"
            tag = f"{policy}/seed-{seed}"
            report["active_trial"] = tag
            timing = report["trials"][tag] = {"status": "in_progress"}
            dest = folder(policy_dirs[policy]/f"seed-{seed}")
            selected = frozenset() if policy == CONTROL else planned[seed]
            basis = {**common, "seed": seed, "policy": policy, "body": body_pin,
                     "predicted_body_cache_sha256": index["content_sha256"], "assignment": ASSIGNMENT,
                     "predicted_train_keys_sha256": digest(sorted(selected)), "ordered_train_keys_sha256": digest(ordered),
                     "trajectory": TRAJECTORY, "training": TRAINING, "evaluation_seed": EVAL_SEED, "sampling_steps": 10}
            marker = dest/"summary.json"
            if marker.exists():
                summary = read_sealed(marker, SUMMARY)
                tip = fit_tip(dest, basis, FIT)
                if (not tip or tip["step"] != UPDATES or summary["binding"] != {**basis, "fit": tip["content_sha256"]}
                        or summary["hand_checkpoint_sha256"] != tip["artifact"]["sha256"] or summary["counts"] != COUNTS
                        or summary["upper_counts"] != tip["upper_counts"]
                        or tip["upper_counts"] != source_counts(ordered, selected, UPDATES)):
                    raise ValueError("v220_completed_trial_binding")
                checkpoint_raw(dest, tip, ledger, CAP)
            else:
                seed_run(trajectory_seed(seed, "model-initialization"))
                hand = make_hand(cascade, pipe, policy, body_pin)
                tip = fit(hand, ordered, selected, dest, work, basis, store, cache, cascade, ledger, owner, environment, timing)
                report["phase"] = "hand_validation"
                summary = evaluate(cascade, hand, store, cache, dest, work, {**basis, "fit": tip["content_sha256"]},
                                   tip, reference, ledger, owner, environment, timing)
                del hand
            summaries[policy, seed] = summary
            timing.update(status="completed", summary_sha256=summary["content_sha256"], upper_counts=summary["upper_counts"],
                          hand_checkpoint_sha256=summary["hand_checkpoint_sha256"])
    report.pop("active_trial", None)
    report["phase"] = "matched_comparison"
    comparison = paired(target/"comparison.json", summaries, index["content_sha256"], owner)
    report["comparison_sha256"] = comparison["content_sha256"]


def execute_parent(project, work, target, pins, cfg, limits, identity, device, ledger, owner, report):
    report["phase"] = "parent_validate_inputs"
    start = time.monotonic()
    step, reference = inputs(project, pins, identity, SimpleNamespace(METHOD=METHOD, BRANCH=BRANCH))
    base = pins["base"]
    doc, spec, plan = load_inputs(SimpleNamespace(invocation=work, workspace=work, slot="A"), base)
    journal, anchor, entries = read_journal(project, doc, spec, base, ledger)
    cache = FullCache(work/"cache-metadata-only", journal, anchor, entries, spec, cfg, ledger)
    try:
        report["phase"] = "parent_restore_checkpoint"
        data = FullData(doc, spec, cfg, plan, cache, ledger, base["source_sha256"])
        pipe = Training(cfg, data, work/"unused", device, work/"run", target, ledger, owner, identity)
        pipe.binding["environment"] = {k: v for k, v in step["environment"].items()
                                       if k not in {"device_name", "device_memory_bytes"}}
        package = step["archive"]
        restored = retention.restore_run(input_file(project/A/"checkpoints", package["name"]), package,
                                         pipe.root, limits, ledger, pipe.binding)
        if restored != step["last"]["sha256"]:
            raise ValueError("v220_A_checkpoint_sha256")
        pipe._resume(restored)
        if pipe.last != step["last"] or any(pipe.progress[k] != v for k, v in step["cursor"].items()):
            raise ValueError("v220_A_checkpoint_cursor")
        report["parent_prepare_seconds"] = time.monotonic()-start
        trials(project, work, target, pins, reference, pipe, ledger, owner, report)
    finally:
        cache.close()


def run(args, pins, report):
    device = initialize_cuda(args.started, report)
    report.update(phase="worker_input_validation", trials={}, startup_seconds=time.monotonic()-args.started)
    print(f"CUDA worker ready on {device} ({report['environment']['gpu']})", flush=True)
    project, work = args.project.resolve(strict=True), args.work.resolve(strict=True)
    if work.is_relative_to(project) or (project/"reports").resolve(strict=True) != project/"reports" or pins["v220"] != CONTRACT:
        raise ValueError("v220_workspace_or_contract")
    for name, sha in pins["modules"].items():
        if Path(name).name != name or _hash_file(Path(__file__).resolve().parent/name) != sha:
            raise ValueError("v220_overlay_module_sha256")
    pin = digest(pins)
    identity = identity_for("A", pins["base"])
    report.update(phase="drive_writer", release_inputs_sha256=pin)
    cfg = training_config(42)
    limits = {**cfg, "resources": {**cfg["resources"], "read_bytes": 8*GIB, "write_bytes": 2*GIB},
              "terminal_reserve_seconds": 300}
    target = folder(project/DESTINATION)
    with writer(target, args.recover_writer) as owner:
        diag = folder(target/"diagnostics")
        ledger = TrialLedger(folder(target/"usage"), {"method": METHOD, "parent": identity, "inputs": pin}, limits, owner)
        failure = None
        try:
            with ledger.session(args.started):
                execute_parent(project, work, target, pins, cfg, limits, identity, device, ledger, owner, report)
            report.update(phase="completed", status="completed")
        except BaseException as exc:
            failure = exc
            report["status"] = "stopped" if isinstance(exc, KeyboardInterrupt) else "paused" if isinstance(exc, TimeoutError) else "failed"
            record_failure(report, exc)
            raise
        finally:
            report["elapsed_seconds"] = time.monotonic()-args.started
            collect_peak(device, report)
            try:
                report["usage"] = ledger.snapshot()["invocations"][-1]
                name = f"invocation-{uuid.uuid4().hex}.json"
                publish_json(diag/name, SCHEMA, report, owner)
                report.update(diagnostic=name, diagnostic_verified=True)
            except Exception as exc:
                report.update(diagnostic_verified=False, diagnostic_error=str(exc)[:512])
                if failure is None:
                    report["phase"] = "drive_diagnostic_publish"
                    raise


if __name__ == "__main__":
    main(run, __doc__)
