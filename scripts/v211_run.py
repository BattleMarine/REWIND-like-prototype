"""One manual CUDA invocation of the fixed v2.1.1 A-head trials."""

import argparse
import json
import os
from pathlib import Path
import platform
import shutil
import tempfile
import time
import traceback
from types import SimpleNamespace
import uuid

import torch
from rewind_archive.contracts import canonical, digest
from rewind_m1.local_rgb import _regular_read
from rewind_m1.training import seed_run
from rewind_prepare.temporal_cache import _atomic, _hash_file
from rewind_prepare.temporal_selection import check_pin
from rewind_v1 import v2_retention as retention
from rewind_v1.frontend import frontend_fingerprint
from rewind_v1.full_data import FeatureStore
from scripts.run_v2_bulk_cache import writer
from scripts.run_v2_gpu_training import identity_for, load_inputs, read_journal, training_config
from scripts.v210_eval import A, COMPARE, DIAGNOSTIC, FEATURE, FINAL, RECEIPT, feature_group, input_file, original_json, reference_scores
from scripts import v211_trial
from scripts.v211_progress import checkpoint_raw, evaluate, fit, fit_tip, folder
from scripts.v211_trial import SEEDS, TRAINING, TRAJECTORY, UPDATES, measurement, order, trajectory_seed
from scripts.v2_gpu_training_pipeline import CommonEvaluation, FullCache, FullData, Training
from scripts.v2_gpu_training_storage import DriveLedger, GIB, MIB, publish_json, read_sealed


DESTINATION = "reports/v211_body_head128_a_v1"
RUN_SCHEMA = "rewind_v211_invocation_v1"
PLAN = "rewind_v2_tuning_cpu_trial_plan_v1"
LEGACY_INPUTS = {
    "v2.1.1": "b7175eb71d63127ae34b5bb28073536ad70181792828d9832ddd853c673041d9",
    "v2.1.2": "379c80f6943427bf93ea24b0429b48ac119987a5ce56f3665fcdcf7f22597002",
}


def resume_pins(pins, method):
    if "resume_modules" not in pins:
        return pins
    bound = {key: value for key, value in pins.items() if key != "resume_modules"}
    bound["modules"] = pins["resume_modules"]
    if digest(bound) != LEGACY_INPUTS[method]:
        raise ValueError("v2_tuning_legacy_resume_inputs")
    return bound


def exception_details(phase, exc):
    return {"phase": phase, "type": type(exc).__name__, "message": str(exc)[:1000],
            "traceback": "".join(traceback.format_exception(exc))[-16384:]}


def record_failure(report, exc):
    report.setdefault("error", str(exc)[:1000])
    if "failure" not in report:
        report["failure"] = exception_details(report["phase"], exc)


def initialize_cuda(started, report):
    report["phase"] = "cuda_validate"
    index = 0
    count = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if count != 1 or not 0 <= index < count or not 0 < time.monotonic()-started < 600:
        raise RuntimeError(f"v211_single_cuda_and_start_time (index={index}, visible={count})")
    report["phase"] = "cuda_initialize"
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.cuda.init()
    if torch.cuda.current_device() != index:
        raise RuntimeError(f"v2_tuning_unexpected_cuda_current_device (expected={index})")
    device = torch.device("cuda", index)
    properties = torch.cuda.get_device_properties(index)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    seed_run(42)
    report["environment"] = {"gpu": properties.name, "vram_bytes": properties.total_memory,
                             "python": platform.python_version(), "torch": str(torch.__version__),
                             "cuda": torch.version.cuda, "precision": "fp32", "tf32": False,
                             "deterministic_algorithms": True}
    report["phase"] = "cuda_allocator_probe"
    # A real, RNG-free allocation initializes the selected CUDA allocator even
    # when the allocator backend defers per-device setup until the first use.
    probe = torch.empty(1, device=device, dtype=torch.float32)
    del probe
    try:
        torch.cuda.reset_peak_memory_stats(index)
    except Exception as exc:
        report["cuda_peak"] = {"scope": "worker_lifetime",
                               "reset_failure": exception_details("cuda_peak_reset", exc)}
    else:
        report["cuda_peak"] = {"scope": "after_startup_probe"}
    return device


def collect_peak(device, report):
    try:
        report["peak_gpu_bytes"] = {"allocated": torch.cuda.max_memory_allocated(device.index),
                                   "reserved": torch.cuda.max_memory_reserved(device.index)}
    except Exception as exc:
        report["peak_gpu_bytes"] = None
        report["cuda_peak"]["read_failure"] = exception_details("cuda_peak_read", exc)


class TrialLedger(DriveLedger):
    def import_usage(self, usage):
        # The parent ZIP contains historical usage. Verify that receipt while
        # keeping the trial's new invocation accounting independent.
        check_pin(usage, "rewind_v2_usage_ledger_v1", usage["content_sha256"])
        if usage["binding"] != self.binding["parent"]:
            raise ValueError("v211_parent_usage_binding")
        self.current["parent_usage_receipt_sha256"] = usage["content_sha256"]


def inputs(project, pins, identity, strategy=v211_trial):
    diagnostic = original_json(project, DIAGNOSTIC, pins["evidence"][DIAGNOSTIC], "rewind_v2_gpu_invocation_v1")
    step = original_json(project, RECEIPT, pins["evidence"][RECEIPT], "rewind_v2_gpu_training_checkpoint_v1")
    comparison = original_json(project, COMPARE, pins["evidence"][COMPARE], "rewind_v2_gpu_comparison_v1")
    plan = pins["trial_plan"]
    check_pin(plan, PLAN, plan["content_sha256"])
    common = plan["common"]
    if (diagnostic["checkpoint"] != step or diagnostic["comparison"] != comparison or not step["completed"]
            or step["identity"] != identity or step["content_sha256"] != FINAL
            or step["cursor"] != {"stage": "hand", "epoch": 3, "batch": 0, "updates": 197961}
            or common["parent"] != identity or common["selected_model_sha256"] != comparison["binding"]["model"]
            or plan["parent_macro_mm"] != comparison["macro_score_mm"] or plan["initial_seeds"] != list(SEEDS)
            or plan["branches"][strategy.METHOD] != strategy.BRANCH):
        raise ValueError("v211_parent_or_trial_plan")
    return step, reference_scores(diagnostic, comparison)


def restore_features(project, work, pins, pipeline, ledger):
    frontend = pipeline._make_frontend()
    pipeline._load_best("frontend", frontend)
    if shutil.disk_usage(work).free < 4*GIB:
        raise OSError("v211_local_free_disk_below_4GiB")
    target = folder(work/"features")
    store = FeatureStore(target, pipeline.data, frontend_fingerprint(frontend.model))
    if store.identity != FEATURE or set(pins["groups"]) != {str(i) for i in range(147)}:
        raise ValueError("v211_feature_identity_or_coverage")
    allocated = 0
    for group in range(147):
        ledger.check()
        with tempfile.TemporaryDirectory(dir=work, prefix="features-") as directory:
            feature_group(project, pins, store, group, ledger, Path(directory))
            for staged in store.root.iterdir():
                allocated += sum(file.stat().st_size for file in staged.iterdir())
                if allocated > 2*GIB or (target/staged.name).exists():
                    raise ValueError("v211_feature_disk_or_duplicate")
                staged.rename(target/staged.name)
    store.root = target
    if set(store.names.values()) != {p.name for p in target.iterdir()}:
        raise ValueError("v211_feature_set")
    return store, allocated


def trials(project, work, target, pins, reference, pipe, ledger, owner, environment, report, strategy=v211_trial,
           *, bound_modules=None):
    plan = pins["trial_plan"]
    best = {role: info["sha256"] for role, info in pipe.progress["best"].items()}
    if best != plan["common"]["selected_model_sha256"] or pipe.cfg["teacher"]["sampling_steps"] != 10:
        raise ValueError("v211_selected_model_or_sampling")
    store = cascade = None
    checkpoint_cap = getattr(strategy, "CHECKPOINT_CAP", MIB)
    for seed in SEEDS:
        ledger.check()
        seed_run(trajectory_seed(seed, "model-initialization"))
        trial = folder(target/f"seed-{seed}")
        ordered = order(pipe.data.clips["train"], seed)
        if len(ordered) != UPDATES or len(set(ordered)) != UPDATES:
            raise ValueError("v211_train_coverage")
        basis = {"method": strategy.METHOD, "seed": seed, "plan": plan["content_sha256"], "reference": FINAL,
                 "source": pins["base"]["source_sha256"], "modules": bound_modules or pins["modules"], "original_model": best,
                 "feature_identity": FEATURE, "feature_groups_sha256": digest(pins["groups"]),
                 "ordered_train_clips_sha256": digest(ordered), "trajectory": TRAJECTORY,
                 "training": TRAINING, "trainable": list(strategy.TRAINABLE), "comparison_common": plan["common"]}
        timing = report["trials"][str(seed)] = {"status": "in_progress"}
        done = trial/"summary.json"
        if done.exists():
            summary = read_sealed(done, strategy.SUMMARY_SCHEMA)
            tip = fit_tip(trial, basis, strategy.FIT_SCHEMA)
            if (not tip or tip["step"] != UPDATES or summary["binding"] != {**basis, "fit": tip["content_sha256"],
                    "body_sha256": tip["artifact"]["sha256"], "sampling_steps": 10}
                    or summary["measurement"] != measurement(plan, seed, tip["artifact"]["sha256"],
                                                              summary["metrics"], strategy.METHOD)):
                raise ValueError("v211_completed_trial_binding")
            checkpoint_raw(trial, tip, ledger, checkpoint_cap)
            timing.update(status="completed_verified_summary_and_fit", summary_sha256=summary["content_sha256"],
                          measurement=summary["measurement"])
            continue
        if store is None:
            start = time.monotonic()
            store, allocated = restore_features(project, work, pins, pipe, ledger)
            first = pipe.data.clips["train"][0]
            cascade = CommonEvaluation(pipe.data, {"train": {first: store.load(first, "train")}}, pipe.cfg, pipe.device, 0)
            report["features"] = {"bytes": allocated, "groups": 147, "wall_seconds": time.monotonic()-start}
        body = cascade.make("body")
        pipe._load_best("body", body)
        tip = fit(body, ordered, trial, work, basis, store, cascade.body_idx, ledger, owner, environment, timing,
                  strategy=strategy)
        hand = cascade.make("hand", body.checkpoint_sha256)
        pipe._load_best("hand", hand)
        summary = evaluate(cascade, body, hand, store, trial, work,
                           {**basis, "fit": tip["content_sha256"], "body_sha256": body.checkpoint_sha256,
                            "sampling_steps": 10}, plan, reference, ledger, owner, environment, timing,
                           summary_schema=strategy.SUMMARY_SCHEMA, preflight_io=checkpoint_cap > MIB)
        timing.update(status="completed", summary_sha256=summary["content_sha256"], measurement=summary["measurement"])
        del body, hand


def run(args, pins, report, *, strategy=v211_trial, destination=DESTINATION, run_schema=RUN_SCHEMA):
    device = initialize_cuda(args.started, report)
    elapsed = time.monotonic()-args.started
    environment = report["environment"]
    report.update(phase="worker_input_validation", startup_seconds=elapsed, trials={})
    print(f"CUDA worker ready on {device} ({environment['gpu']}), peak scope={report['cuda_peak']['scope']}", flush=True)
    project, work = args.project.resolve(strict=True), args.work.resolve(strict=True)
    if work.is_relative_to(project) or (project/"reports").resolve(strict=True) != project/"reports":
        raise ValueError("v211_project_or_scratch")
    for name, sha in pins["modules"].items():
        if _hash_file(Path(__file__).resolve().parent/name) != sha:
            raise ValueError("v211_overlay_sha256")
    bound = resume_pins(pins, strategy.METHOD)
    report["release"] = {"executed_modules": pins["modules"], "executed_inputs_sha256": digest(pins),
                         "resume_inputs_sha256": digest(bound)}
    base = pins["base"]
    identity = identity_for("A", base)
    target = folder(project/destination)
    cfg = training_config(42)
    limits = {**cfg, "resources": {**cfg["resources"], "read_bytes": 8*GIB, "write_bytes": 2*GIB},
              "terminal_reserve_seconds": 300}
    report["phase"] = "drive_writer"
    with writer(target, args.recover_writer) as owner:
        diag = folder(target/"diagnostics")
        usage = folder(target/"usage")
        ledger = TrialLedger(usage, {"method": strategy.METHOD, "parent": identity, "inputs": digest(bound)}, limits, owner)
        failure = None
        try:
            with ledger.session(args.started):
                report["phase"] = "parent_validate_inputs"
                step, reference = inputs(project, pins, identity, strategy)
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
                        raise ValueError("v211_A_last_sha256")
                    pipe._resume(restored)
                    if pipe.last != step["last"] or any(pipe.progress[k] != v for k, v in step["cursor"].items()):
                        raise ValueError("v211_A_resume_cursor")
                    report["parent_prepare_seconds"] = time.monotonic()-args.started-elapsed
                    report["phase"] = "trials"
                    trials(project, work, target, pins, reference, pipe, ledger, owner, environment, report, strategy,
                           bound_modules=bound["modules"])
                finally:
                    cache.close()
            report.update(status="completed", phase="completed")
        except BaseException as exc:
            failure = exc
            report["status"] = ("stopped" if isinstance(exc, KeyboardInterrupt) else
                                "paused" if isinstance(exc, TimeoutError) else "failed")
            record_failure(report, exc)
            raise
        finally:
            report["elapsed_seconds"] = time.monotonic()-args.started
            collect_peak(device, report)
            try:
                report["usage"] = ledger.snapshot()["invocations"][-1]
                report["diagnostic"] = f"invocation-{uuid.uuid4().hex}.json"
                publish_json(diag/report["diagnostic"], run_schema, report, owner)
                report["diagnostic_verified"] = True
            except Exception as exc:
                report.update(diagnostic_verified=False, diagnostic_error=str(exc)[:512],
                              diagnostic_failure=exception_details("drive_diagnostic_publish", exc))
                if failure is None:
                    report["phase"] = "drive_diagnostic_publish"
                    raise


def main(run_worker=run, description=__doc__):
    parser = argparse.ArgumentParser(description=description)
    for name in ("project", "work", "pins"):
        parser.add_argument("--"+name, type=Path, required=True)
    parser.add_argument("--started", type=float, required=True)
    parser.add_argument("--recover-writer", default="")
    args = parser.parse_args()
    report = {"status": "failed", "phase": "read_worker_pins"}
    try:
        run_worker(args, json.loads(_regular_read(args.pins, 2*MIB)), report)
    except BaseException as exc:
        record_failure(report, exc)
        raise
    finally:
        _atomic(args.work/"result.json", canonical(report))


if __name__ == "__main__":
    main()
