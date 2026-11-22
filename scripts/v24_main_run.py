"""Pinned resumable worker for the proposed v2.4 paired body comparison."""

import argparse
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import time

import torch

from rewind_archive.contracts import canonical, digest
from rewind_m1.training import restore_rng, rng_state
from rewind_prepare.temporal_cache import _hash_file
from rewind_v1.frontend import frontend_fingerprint
from rewind_v1.full_data import FeatureStore
from rewind_v1.v2_budget import Ledger
from scripts.run_v2_gpu_training import training_config
from scripts.v210_eval import FEATURE, FINAL, feature_group, group_keys
from scripts.v210_metrics import aggregate
from scripts.v23_run import gpu, parent_evidence, restored_parent, verified_write
from scripts.v24_body import ARMS, LEARNING_RATE, draw, model_sha, setup, update
from scripts.v24_run import frame_probe, validate
from scripts.v2_gpu_training_pipeline import CommonEvaluation
from scripts.v2_gpu_training_storage import transfer_in


MIB = 1024**2
GIB = 1024*MIB
TRAJECTORY = "rewind_v24_main_group_stream_v1"
STATE_SCHEMA = "rewind_v24_main_state_v1"
RESULT_SCHEMA = "rewind_v24_main_invocation_v1"
WORKER_SCHEMA = "rewind_v24_main_worker_v1"
SEEDS = (42, 71, 103, 137, 181)
BOUNDARIES = (0, 768, 1536, 2304, 3097)
STATE_CAP = 192*MIB
REPORT_CAP = 2*MIB
SECONDS = 1500
WORKER_RESERVE = 240
PUBLISH_RESERVE = 180
READ_CAP = 2*GIB
WRITE_CAP = 6*GIB
BOOTSTRAP_AND_PUBLICATION_CAP = 512*MIB
SCRATCH_FREE = 4*GIB


def order_train(groups, seed):
    def rank(tag, item):
        return digest([TRAJECTORY, seed, tag, item])
    ordered = sorted(groups, key=lambda g: (len(groups[g]) != 24, rank("group", g)))
    return [(i, key) for i in ordered for key in sorted(groups[i], key=lambda k: rank("clip", k))]


def roster(data, seed):
    train, validation = {}, {}
    for group in range(147):
        entries = group_keys(data, group)
        train_keys = [k for split, k in entries if split == "train"]
        if train_keys:
            train[group] = train_keys
        validation.update((k, group) for split, k in entries if split == "validation")
    order = order_train(train, seed)
    if (len(order) != 3097 or {k for _, k in order} != set(data.clips["train"])
            or tuple(validation) != tuple(data.clips["validation"]) or len(validation) != 421
            or sum(data.real_frames(k, "validation") for k in validation) != 16425):
        raise ValueError("v24_main_official_roster")
    return order, validation


def pin_file(path):
    return {"bytes": path.stat().st_size, "sha256": _hash_file(path)}


def save(path, stage, binding):
    payload = {"schema": STATE_SCHEMA, "binding_sha256": digest(binding), "step": stage.step,
               "body": stage.model.state_dict(), "optimizer": stage.optimizer.state_dict(),
               "scheduler": stage.scheduler.state_dict(), "scaler": stage.scaler.state_dict(), "rng": rng_state()}
    with path.open("xb") as stream:
        torch.save(payload, stream)
    pin = pin_file(path)
    if not 0 < pin["bytes"] <= STATE_CAP:
        raise ValueError("v24_main_state_cap")
    return pin


def load(path, stage, binding, cursor):
    state = torch.load(path, map_location="cpu", weights_only=True)
    if (state["schema"], state["binding_sha256"], state["step"]) != (STATE_SCHEMA, digest(binding), cursor):
        raise ValueError("v24_main_resume_state_identity")
    stage.model.load_state_dict(state["body"], strict=True)
    stage.optimizer.load_state_dict(state["optimizer"])
    stage.scheduler.load_state_dict(state["scheduler"])
    stage.scaler.load_state_dict(state["scaler"])
    restore_rng(state["rng"])
    stage.step = cursor


def read_resume(project, resume, resume_sha, work, binding, cursor, ledger):
    receipt = resume.resolve(strict=True)
    parent = project / "reports" / "v24_main128_a_candidate"
    if (resume != receipt or receipt.name != "result.json" or receipt.parent.parent != parent
            or not receipt.parent.name.startswith("run-") or receipt.stat().st_size > 64*1024):
        raise ValueError("v24_main_resume_receipt_location")
    raw = receipt.read_bytes()
    if hashlib.sha256(raw).hexdigest() != resume_sha:
        raise ValueError("v24_main_resume_receipt_sha256")
    result = json.loads(raw)
    if (result["schema"] != RESULT_SCHEMA or result["status"] != "segment_complete"
            or result["returncode"] != 0 or result["after"] != cursor
            or result["binding_sha256"] != digest(binding)):
        raise ValueError("v24_main_resume_receipt_binding")
    pin = result["files"]["body-state.pt"]
    if not 0 < pin["bytes"] <= STATE_CAP:
        raise ValueError("v24_main_resume_state_cap")
    ledger.reserve("read", pin["bytes"]+1)
    ledger.reserve("write", pin["bytes"])
    target = work / "resume.pt"
    transfer_in(receipt.parent / "body-state.pt", target, pin, ledger)
    ledger.observed("write", pin["bytes"])
    return target


@contextmanager
def group_store(project, work, pins, store, group, ledger):
    ledger.check()
    wire_bytes = pins["groups"][str(group)]["archive"]["bytes"]
    ledger.reserve("write", wire_bytes + 32*MIB)
    with tempfile.TemporaryDirectory(dir=work, prefix="main-feature-") as folder:
        location = Path(folder)
        feature_group(project, pins, store, group, ledger, location)
        ledger.observed("write", wire_bytes + sum(p.stat().st_size for p in store.root.rglob("*") if p.is_file()))
        yield store


def evaluate(context, stage, arm, hand, requested, validation, reference, group_source, check):
    grouped = {}
    for key in requested:
        grouped.setdefault(validation[key], []).append(key)
    found = {}
    for group, keys in grouped.items():
        with group_source(group) as store:
            clips = {("validation", key): store.load(key, "validation") for key in keys}
            scored = validate(context, stage, arm, hand, clips, keys, check)
            found.update((tuple(row["key"]), row) for row in scored["clips"])
    rows = [found[k] for k in requested]
    metrics = aggregate([(tuple(row["key"]), row["scores"]) for row in rows], context.spec, reference)
    real_frames = sum(row["real_frames"] for row in rows)
    if real_frames != (16425 if len(requested) == 421 else 300):
        raise ValueError("v24_main_validation_frames")
    return {"update": stage.step, "scope": "official_validation" if len(requested) == 421 else "pilot_validation_subset",
            "frames": real_frames, "clips": len(requested), "metrics": metrics,
            "noise_and_roster_sha256": digest([[r["key"], r["real_frames"], r["noise_sha256"]] for r in rows])}


def compare_segment(args, project, work, pins, pipe, reference, ledger, report):
    from functools import partial
    from itertools import groupby

    if pipe.cfg["teacher"]["learning_rate"] != LEARNING_RATE:
        raise ValueError("v24_main_A_learning_rate_mismatch")
    order, validation = roster(pipe.data, args.seed)
    subset = [tuple(k) for k in pins["protocol"]["pilot_validation_keys"]]
    if len(set(subset)) != 6 or any(k not in validation or pipe.data.real_frames(k,"validation") != 50 for k in subset):
        raise ValueError("v24_main_fixed_interim_roster")
    frontend = pipe._make_frontend()
    pipe._load_best("frontend", frontend)
    frontend_sha = model_sha(frontend.model)
    store = FeatureStore(work / "unused-features", pipe.data, frontend_fingerprint(frontend.model))
    if store.identity != FEATURE:
        raise ValueError("v24_main_A_feature_binding")
    load_group = partial(group_store, project, work, pins, store, ledger=ledger)
    group, key = order[0]
    with load_group(group) as source:
        probe = source.load(key, "train")
    context = CommonEvaluation(pipe.data, {"train": {key: probe}}, pipe.cfg, pipe.device, 0)
    stage = context.make("body")
    pipe._load_best("body", stage)
    seed_args = {"seed": args.seed, "trajectory": TRAJECTORY}
    initialization = setup(stage, **seed_args)
    probe = probe.select_joints(context.body_idx)
    k, epsilon, _ = draw(stage.diffusion, 0, probe.target_mm.shape, **seed_args)
    frame = frame_probe(stage, probe, (k, epsilon))
    hand = context.make("hand", reference["binding"]["model"]["body"])
    pipe._load_best("hand", hand)
    hand_sha = model_sha(hand.model)
    common = {"protocol_sha256": pins["protocol_sha256"], "seed": args.seed, "parent_receipt_sha256": FINAL,
              "A_weights": reference["binding"]["model"], "feature_identity": FEATURE,
              "feature_group_pins_sha256": digest(pins["groups"]), "train_order_sha256": digest(order),
              "validation_roster_sha256": digest(list(validation)), "initialization": initialization,
              "frame_probe_sha256": frame, "frontend_state_sha256": frontend_sha, "hand_state_sha256": hand_sha,
              "environment": report["environment"]}
    binding = {"common": common, "arm": args.arm, "source_sha256": pins["base"]["source_sha256"], "module_sha256": pins["modules"]}
    report.update(binding_sha256=digest(binding), common_sha256=digest(common))
    if args.before:
        path = read_resume(project, args.resume, args.resume_sha, work, binding, args.before, ledger)
        load(path, stage, binding, args.before)
    scores, updates = [], []
    if args.before == 0:
        scores.append(evaluate(context, stage, args.arm, hand, subset, validation, reference, load_group, ledger.check))
    start = time.monotonic()
    for group, segment in groupby(order[args.before:args.after], key=lambda item: item[0]):
        with load_group(group) as source:
            for _, key in segment:
                ledger.check()
                clip = source.load(key, "train").select_joints(context.body_idx)
                k, epsilon, noise_sha = draw(stage.diffusion, stage.step, clip.target_mm.shape, **seed_args)
                stats = update(stage, args.arm, clip, (k, epsilon))
                updates.append({"step": stage.step, "key": list(key), "noise_sha256": noise_sha, **stats})
    report["training_seconds"] = time.monotonic()-start
    if stage.step != args.after:
        raise ValueError("v24_main_update_boundary")
    start = time.monotonic()
    keys = list(validation) if args.after == BOUNDARIES[-1] else subset
    scores.append(evaluate(context, stage, args.arm, hand, keys, validation, reference, load_group, ledger.check))
    report["boundary_evaluation_seconds"] = time.monotonic()-start
    if (model_sha(frontend.model) != frontend_sha or model_sha(hand.model) != hand_sha
            or any(m.training or any(p.requires_grad or p.grad is not None for p in m.parameters())
                   for m in (frontend.model, hand.model))):
        raise ValueError("v24_main_frozen_upstream_changed")
    ledger.reserve("write", STATE_CAP)
    report["files"]["body-state.pt"] = save(work / "body-state.pt", stage, binding)
    ledger.observed("write", report["files"]["body-state.pt"]["bytes"])
    report["files"]["segment.json"] = verified_write(work, "segment.json", {
        "binding": binding, "before": args.before, "after": args.after, "scores": scores, "updates": updates,
        "resume_result_sha256": args.resume_sha,
        "paired_update_sha256": digest([[r["step"], r["key"], r["noise_sha256"]] for r in updates])})


def run(args, pins, report):
    if (args.arm not in ARMS or args.seed not in SEEDS or (args.seed not in SEEDS[:3] and not args.extension)
            or (args.before, args.after) not in zip(BOUNDARIES, BOUNDARIES[1:])
            or bool(args.before) != bool(args.resume and args.resume_sha)
            or bool(args.resume) != bool(args.resume_sha)
            or args.protocol_sha != pins["protocol_sha256"] or digest(pins["protocol"]) != args.protocol_sha):
        raise ValueError("v24_main_protocol_or_segment")
    project, work = args.project.resolve(strict=True), args.work.resolve(strict=True)
    if work.is_relative_to(project) or shutil.disk_usage(work).free < SCRATCH_FREE:
        raise OSError("v24_main_scratch_preflight")
    if any(_hash_file(Path(__file__).resolve().parent / name) != pin for name, pin in pins["modules"].items()):
        raise ValueError("v24_main_module_hash")
    device = gpu(args.started, report)
    expected = pins["protocol"]["runtime"]
    if any(report["environment"][name] != value for name, value in expected.items()):
        raise ValueError("v24_main_gpu_environment_mismatch")
    identity, step, reference = parent_evidence(project, pins)
    cfg = training_config(42)
    limits = {**cfg, "seconds_per_invocation": SECONDS, "external_timeout_seconds": SECONDS,
              "terminal_reserve_seconds": WORKER_RESERVE,
              "resources": {**cfg["resources"], "read_bytes": READ_CAP,
                            "write_bytes": WRITE_CAP-BOOTSTRAP_AND_PUBLICATION_CAP}}
    ledger = Ledger(work / "usage", identity, limits)
    with ledger.invocation("v24_main_segment"):
        ledger.started = args.started
        try:
            ledger.reserve("write", pins["base"]["manifest_bytes"])
            with restored_parent(project, work, pins, identity, step, reference, limits, ledger, device) as pipe:
                ledger.observed("write", pins["base"]["manifest_bytes"])
                compare_segment(args, project, work, pins, pipe, reference, ledger, report)
        finally:
            report["usage"] = dict(ledger.current)
    report["status"] = "segment_complete"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("project", "work", "pins"):
        parser.add_argument("--"+key, type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    for key in ("seed", "before", "after"):
        parser.add_argument("--"+key, type=int, required=True)
    for key in ("arm", "protocol-sha"):
        parser.add_argument("--"+key, required=True)
    parser.add_argument("--resume-sha")
    parser.add_argument("--extension", action="store_true")
    parser.add_argument("--started", type=float, required=True)
    args = parser.parse_args()
    report = {"schema": WORKER_SCHEMA, "status": "failed", "files": {}, "seed": args.seed, "arm": args.arm,
              "before": args.before, "after": args.after, "protocol_sha256": args.protocol_sha}
    try:
        run(args, json.loads(args.pins.read_bytes()), report)
    except BaseException as exc:
        report["error"] = {"type": type(exc).__name__, "message": str(exc)[:800]}
        raise
    finally:
        report["elapsed_seconds"] = time.monotonic()-args.started
        verified_write(args.work, "worker-result.json", report)


if __name__ == "__main__":
    main()
