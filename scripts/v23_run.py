"""Run one fixed A DDIM 10/5/2 CUDA accuracy and latency comparison."""

import argparse
from contextlib import contextmanager
import json
import os
from pathlib import Path
import platform
import shutil
import tempfile
import time
from types import SimpleNamespace

import torch

from rewind_archive.contracts import canonical
from rewind_m1.local_rgb import _regular_read
from rewind_m1.training import seed_run
from rewind_prepare.temporal_cache import _atomic, _hash_file
from rewind_v1 import v2_retention as retention
from rewind_v1.frontend import frontend_fingerprint
from rewind_v1.full_data import FeatureStore
from rewind_v1.v2_budget import Ledger
from scripts.run_v2_gpu_training import identity_for, load_inputs, read_journal, training_config
from scripts.v210_eval import A, COMPARE, DIAGNOSTIC, FEATURE, FINAL, RECEIPT, feature_group, input_file, original_json, reference_scores
from scripts.v2_gpu_training_comparison import shared_clips
from scripts.v2_gpu_training_pipeline import CommonEvaluation, FullCache, FullData, Training
from scripts.v2_gpu_training_storage import GIB, MIB
from scripts.v23_eval import STEPS, benchmark, prepare, quality


SECONDS = 3600
TERMINAL_RESERVE = 150
READ_CAP = 2 * GIB
WRITE_CAP = 4 * GIB
WORKSPACE_FREE = 4 * GIB
RESULT_CAP = 2 * MIB
DESTINATION = "v23_ddim128_a_write4g_v2"


def gpu(started, report):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    if (not torch.cuda.is_available() or torch.cuda.device_count() != 1
            or not 0 < time.monotonic() - started < 600):
        raise RuntimeError("v23_requires_single_visible_cuda_and_fresh_invocation")
    torch.set_num_threads(1)
    torch.cuda.init()
    device = torch.device("cuda:0")
    if torch.cuda.current_device() != device.index:
        raise RuntimeError("v23_unexpected_current_cuda_device")
    seed_run(42)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    props = torch.cuda.get_device_properties(device)
    report["environment"] = {"gpu": props.name, "vram_bytes": props.total_memory,
                             "cuda": torch.version.cuda, "torch": str(torch.__version__),
                             "python": platform.python_version(), "precision": "fp32", "tf32": False,
                             "deterministic_algorithms": torch.are_deterministic_algorithms_enabled()}
    # The allocator may not exist until the selected device receives its first allocation.
    probe = torch.empty(1, device=device, dtype=torch.float32)
    del probe
    try:
        torch.cuda.reset_peak_memory_stats(device)
    except RuntimeError as exc:
        report["memory_counter_error"] = str(exc)[:256]
    return device


def parent_evidence(project, pins):
    evidence = pins["evidence"]
    diagnostic = original_json(project, DIAGNOSTIC, evidence[DIAGNOSTIC], "rewind_v2_gpu_invocation_v1")
    step = original_json(project, RECEIPT, evidence[RECEIPT], "rewind_v2_gpu_training_checkpoint_v1")
    comparison = original_json(project, COMPARE, evidence[COMPARE], "rewind_v2_gpu_comparison_v1")
    identity = identity_for("A", pins["base"])
    if (diagnostic["status"] != "completed" or diagnostic["checkpoint"] != step or diagnostic["comparison"] != comparison
            or step["content_sha256"] != FINAL or step["completed"] is not True or step["identity"] != identity
            or step["cursor"] != {"stage": "hand", "epoch": 3, "batch": 0, "updates": 197961}):
        raise ValueError("v23_fixed_A_parent")
    return identity, step, reference_scores(diagnostic, comparison)


@contextmanager
def restored_parent(project, work, pins, identity, step, reference, limits, ledger, device):
    source = pins["base"]
    cfg = training_config(42)
    doc, spec, plan = load_inputs(SimpleNamespace(invocation=work, workspace=work, slot="A"), source)
    journal, anchor, entries = read_journal(project, doc, spec, source, ledger)
    cache = FullCache(work / "cache-metadata-only", journal, anchor, entries, spec, cfg, ledger)
    try:
        data = FullData(doc, spec, cfg, plan, cache, ledger, source["source_sha256"])
        pipe = Training(cfg, data, work / "unused", device, work / "run", work, ledger, lambda: None, identity)
        pipe.binding["environment"] = {k: v for k, v in step["environment"].items()
                                      if k not in {"device_name", "device_memory_bytes"}}
        package = step["archive"]
        last = retention.restore_run(input_file(project / A / "checkpoints", package["name"]), package,
                                     pipe.root, limits, ledger, pipe.binding)
        if last != step["last"]["sha256"]:
            raise ValueError("v23_A_last_checkpoint_pin")
        pipe._resume(last)
        if (pipe.last != step["last"] or any(pipe.progress[k] != v for k, v in step["cursor"].items())
                or pipe.cfg["teacher"]["sampling_steps"] != STEPS[0]
                or {role: pin["sha256"] for role, pin in pipe.progress["best"].items()} != reference["binding"]["model"]):
            raise ValueError("v23_A_selected_models")
        yield pipe
    finally:
        cache.close()


def fixed_features(project, work, pins, pipe, reference, ledger):
    frontend = pipe._make_frontend()
    pipe._load_best("frontend", frontend)
    store = FeatureStore(work / "unpopulated", pipe.data, frontend_fingerprint(frontend.model))
    del frontend
    if store.identity != FEATURE or sorted(map(int, pins["groups"])) != list(range(129, 147)):
        raise ValueError("v23_A_selected_feature_identity")
    clips = {}
    for group in range(129, 147):
        ledger.check()
        with tempfile.TemporaryDirectory(dir=work, prefix="features-") as scratch:
            keys = feature_group(project, pins, store, group, ledger, Path(scratch))
            if not clips.keys().isdisjoint(keys):
                raise ValueError("v23_duplicate_feature_clip")
            for key in keys:
                clips[key] = store.load(key, "validation")
    fixed = shared_clips(pipe.data.plan)
    if (list(clips) != pipe.data.clips["validation"] or len(clips) != 421
            or {name: list(key) for name, key in fixed.items()} != reference["binding"]["clips"]
            or any(key not in clips for key in fixed.values())):
        raise ValueError("v23_validation_clip_roster")
    return clips, tuple(fixed.values())


def verified_write(work, name, document):
    raw = canonical(document)
    if len(raw) > RESULT_CAP:
        raise ValueError("v23_result_size")
    path = work / name
    _atomic(path, raw)
    if path.read_bytes() != raw:
        raise OSError("v23_local_result_readback")
    return {"bytes": len(raw), "sha256": _hash_file(path)}


def run(args, pins, report):
    project, work = args.project.resolve(strict=True), args.work.resolve(strict=True)
    if (work.is_relative_to(project) or not (project / "reports").is_dir()
            or (project / "reports").resolve(strict=True) != project / "reports"
            or shutil.disk_usage(work).free < WORKSPACE_FREE):
        raise ValueError("v23_parent_or_scratch_preflight")
    for name, pin in pins["modules"].items():
        if _hash_file(Path(__file__).resolve().parent / name) != pin:
            raise ValueError("v23_fixed_module_sha256")
    fixed = pins["limits"]
    if fixed["worker_logical_write_bytes"] not in (2 * GIB, WRITE_CAP):
        raise ValueError("v23_fixed_worker_write_cap")
    report["phase"] = "initialize_cuda"
    device = gpu(args.started, report)
    report["phase"] = "fixed_parent_evidence"
    identity, step, reference = parent_evidence(project, pins)
    cfg = training_config(42)
    limits = {**cfg, "seconds_per_invocation": SECONDS, "external_timeout_seconds": SECONDS,
              "terminal_reserve_seconds": TERMINAL_RESERVE,
              "resources": {**cfg["resources"], "read_bytes": READ_CAP, "write_bytes": fixed["worker_logical_write_bytes"]}}
    report["logical_io_limits"] = {kind: limits["resources"][f"{kind}_bytes"] for kind in ("read", "write")}
    binding = {"parent": identity, "checkpoint_receipt": FINAL, "weights": reference["binding"]["model"],
               "source": pins["base"]["source_sha256"], "overlay": pins["modules"], "feature_identity": FEATURE,
               "quality_scope": "official_validation", "sampling_steps": list(STEPS), "eval_seed": 71}
    report["binding"] = binding
    report["timing"] = {}
    ledger = Ledger(work / "usage", identity, limits)
    with ledger.invocation("v23_fixed_A_ddim"):
        ledger.started = args.started
        try:
            report["phase"] = "restore_verified_A_parent"
            beginning = time.monotonic()
            with restored_parent(project, work, pins, identity, step, reference, limits, ledger, device) as pipe:
                report["timing"]["parent_prepare_seconds"] = time.monotonic() - beginning
                report["phase"] = "read_validation_features_once"
                beginning = time.monotonic()
                clips, fixed = fixed_features(project, work, pins, pipe, reference, ledger)
                report["timing"]["feature_restore_seconds"] = time.monotonic() - beginning
                cascade = CommonEvaluation(pipe.data, {"train": {fixed[0]: clips[fixed[0]]}}, pipe.cfg, device, 0)
                body = cascade.make("body")
                hand = cascade.make("hand", reference["binding"]["model"]["body"])
                pipe._load_best("body", body)
                pipe._load_best("hand", hand)
                report["phase"] = "timed_bundle"
                timing_inputs = {key: prepare(cascade, body, hand, key, clips[key]) for key in fixed}
                lat = benchmark(cascade, body, hand, timing_inputs, ledger.check)
                report["files"]["latency.json"] = verified_write(work, "latency.json",
                                                                {**lat, "binding": binding, "environment": report["environment"]})
                report["phase"] = "full_validation_quality"
                beginning = time.monotonic()
                scores = quality(cascade, body, hand, clips, fixed, reference, ledger.check)
                report["timing"]["quality_seconds"] = time.monotonic() - beginning
                noise = {tuple(item["key"]): item["initial_noise_sha256"] for item in scores["initial_noise_by_clip"]}
                if any(noise[key] != item.noise_sha256 for key, item in timing_inputs.items()):
                    raise ValueError("v23_quality_and_latency_noise_diverged")
                old = pins["reference_10"]
                scores["archived_v210_10_step"] = {**old,
                                                   "current_macro_delta_mm": scores["quality"]["10"]["macro_score_mm"] - old["macro_mm"]}
                report["files"]["quality.json"] = verified_write(work, "quality.json",
                                                                {**scores, "binding": binding, "environment": report["environment"]})
        finally:
            report["usage"] = dict(ledger.current)
            try:
                report["peak_gpu_bytes"] = {"allocated": torch.cuda.max_memory_allocated(device),
                                            "reserved": torch.cuda.max_memory_reserved(device)}
            except RuntimeError as exc:
                report["memory_read_error"] = str(exc)[:256]
    report.update(status="completed", phase="finished")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("project", "work", "pins"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--started", type=float, required=True)
    args = parser.parse_args()
    report = {"schema": "rewind_v23_worker_v1", "status": "failed", "phase": "preflight", "files": {}}
    try:
        run(args, json.loads(_regular_read(args.pins, 2 * MIB)), report)
    except BaseException as exc:
        if isinstance(exc, (TimeoutError, KeyboardInterrupt)):
            report["status"] = "stopped"
        report["failure"] = {"type": type(exc).__name__, "message": str(exc)[:800]}
        raise
    finally:
        report["worker_wall_seconds_since_invocation_start"] = time.monotonic() - args.started
        verified_write(args.work, "worker-result.json", report)


if __name__ == "__main__":
    main()
