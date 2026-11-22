"""Evaluate the selected v2.1.1 body head with DDIM2 on the v2.3 roster."""

import argparse
import hashlib
import io
import json
import math
from pathlib import Path
import shutil
import time

import torch

from rewind_archive.contracts import canonical, digest
from rewind_m1.local_rgb import _regular_read
from rewind_prepare.temporal_cache import _hash_file
from rewind_v1.v2_budget import Ledger
from scripts.v210_metrics import aggregate
from scripts.v211_trial import HEAD, STATE, backbone_sha
from scripts.v2_gpu_training_pipeline import CommonEvaluation
from scripts.v23_eval import NOISE_OFFSETS, cascade_prediction, mpjpe_delta, prepare, score_arrays
from scripts.v23_run import (FEATURE, FINAL, MIB, READ_CAP, SECONDS, TERMINAL_RESERVE, WORKSPACE_FREE, WRITE_CAP,
                             fixed_features, gpu, parent_evidence, restored_parent, training_config, verified_write)


DESTINATION = "v25_ddim2_selected_head128_a_v1"
STEP = 2
COUNTS = {"clips": 421, "real_frames": 16425}
RESULTS = ("quality.json", "verdict.json")


def fingerprint(raw):
    return hashlib.sha256(raw).hexdigest()


def apply_head(body, raw, pin):
    if len(raw) != pin["bytes"] or fingerprint(raw) != pin["sha256"]:
        raise ValueError("v25_selected_head_file_pin")
    if body.model.training or any(p.requires_grad for p in body.model.parameters()) or backbone_sha(body) != pin["backbone_sha256"]:
        raise ValueError("v25_frozen_A_body_backbone")
    state = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=True)
    if (state["schema"] != STATE or state["step"] != 3097 or state["binding_sha256"] != pin["binding_sha256"]
            or set(state["head"]) != set(HEAD)):
        raise ValueError("v25_selected_head_state_binding")
    params = dict(body.model.named_parameters())
    for name in HEAD:
        value = state["head"][name]
        if value.shape != params[name].shape or value.dtype != torch.float32 or not torch.isfinite(value).all():
            raise ValueError("v25_selected_head_tensor")
    with torch.no_grad():
        for name in HEAD:
            params[name].copy_(state["head"][name])
            if not torch.equal(params[name].detach().cpu(), state["head"][name]):
                raise ValueError("v25_selected_head_copy")
    if backbone_sha(body) != pin["backbone_sha256"]:
        raise ValueError("v25_selected_head_changed_backbone")
    body.checkpoint_sha256 = digest({"A_body_sha256": body.checkpoint_sha256, "head_sha256": pin["sha256"]})


def evaluate(cascade, body, hand, clips, reference, check):
    scores, noise, seen = [], [], set()
    for key, clip in clips.items():
        check()
        ready = prepare(cascade, body, hand, key, clip)
        frames = {(key[0], frame) for frame in range(key[1], key[1]+ready.real_frames)}
        if not seen.isdisjoint(frames):
            raise ValueError("v25_duplicated_validation_frames")
        seen.update(frames)
        noise.append({"key": list(key), "real_frames": ready.real_frames, "initial_noise_sha256": ready.noise_sha256})
        prediction, local = cascade_prediction(cascade, body, hand, ready, STEP)
        scores.append((key, score_arrays(cascade, clip, ready, prediction, local)))
    if {"clips": len(clips), "real_frames": len(seen)} != COUNTS:
        raise ValueError("v25_validation_coverage")
    return aggregate(scores, cascade.spec, reference), noise


def compare(metrics, noise, baseline, environment):
    original_noise = baseline["initial_noise_by_clip"]
    if [(row["key"], row["real_frames"]) for row in noise] != [(row["key"], row["real_frames"]) for row in original_noise]:
        raise ValueError("v25_v23_validation_roster_mismatch")
    original_hash = baseline["noise_manifest_sha256"]
    actual_hash = fingerprint(canonical(noise))
    mismatches = [new["key"] for new, old in zip(noise, original_noise, strict=True)
                  if new["initial_noise_sha256"] != old["initial_noise_sha256"]]
    original = baseline["quality"][str(STEP)]
    for new, old in [(metrics["pooled"], original["pooled"]), *((metrics["per_scenario"][name], group)
                      for name, group in original["per_scenario"].items())]:
        if (new["clips"], new["frames"]) != (old["clips"], old["frames"]) or any(
                value["valid_joints"] != old["official"][metric]["valid_joints"] for metric, value in new["official"].items()):
            raise ValueError("v25_v23_validation_score_counts_mismatch")
    candidate, control = metrics["macro_score_mm"], original["macro_score_mm"]
    if not math.isfinite(candidate) or not math.isfinite(control):
        raise ValueError("v25_nonfinite_macro")
    macro = candidate-control
    equal_noise = not mismatches and actual_hash == original_hash
    same_runtime = all(environment[key] == baseline["environment"][key] for key in ("gpu", "torch", "cuda", "precision", "tf32"))
    outcome = "improved" if macro < 0 else "worse" if macro > 0 else "equal"
    return {"schema": "rewind_v25_verdict_v1", "criterion": "unrounded_equal_scenario_body_and_global_hand_macro_mpjpe_mm",
            "counts": COUNTS, "numeric_macro_result": outcome,
            "decision": outcome if equal_noise and same_runtime else "inconclusive",
            "comparison_status": "noise_mismatch" if not equal_noise else "matched" if same_runtime else "runtime_difference",
            "candidate_macro_mm": candidate, "v23_A_ddim2_macro_mm": control, "macro_delta_mm": macro,
            "pooled_delta_mm": mpjpe_delta(metrics["pooled"], original["pooled"]),
            "per_scenario_delta_mm": {name: mpjpe_delta(metrics["per_scenario"][name], group)
                                      for name, group in original["per_scenario"].items()},
            "noise": {"matches": equal_noise, "candidate_sha256": actual_hash, "v23_sha256": original_hash,
                      "mismatched_clip_keys": mismatches}, "same_recorded_gpu_torch_cuda_precision": same_runtime}


def run(args, pins, report):
    project, work = args.project.resolve(strict=True), args.work.resolve(strict=True)
    if (work.is_relative_to(project) or not (project/"reports").is_dir()
            or (project/"reports").resolve(strict=True) != project/"reports" or shutil.disk_usage(work).free < WORKSPACE_FREE):
        raise ValueError("v25_parent_or_scratch_preflight")
    if any(_hash_file(Path(__file__).resolve().parent/name) != value for name, value in pins["modules"].items()):
        raise ValueError("v25_overlay_sha256")
    raw = _regular_read(work/"v23-quality.json", MIB, pins["baseline"]["bytes"])
    if fingerprint(raw) != pins["baseline"]["sha256"]:
        raise ValueError("v25_v23_quality_file_pin")
    baseline = json.loads(raw)
    report["phase"] = "initialize_cuda"
    device = gpu(args.started, report)
    identity, step, reference = parent_evidence(project, pins)
    original = baseline["binding"]
    if (original["parent"] != identity or original["weights"] != reference["binding"]["model"]
            or original["checkpoint_receipt"] != FINAL or original["feature_identity"] != FEATURE):
        raise ValueError("v25_v23_parent_identity")
    cfg = training_config(42)
    limits = {**cfg, "seconds_per_invocation": SECONDS, "external_timeout_seconds": SECONDS,
              "terminal_reserve_seconds": TERMINAL_RESERVE,
              "resources": {**cfg["resources"], "read_bytes": READ_CAP, "write_bytes": WRITE_CAP}}
    ledger = Ledger(work/"usage", identity, limits)
    with ledger.invocation("v25_selected_head_ddim2"):
        ledger.started = args.started
        try:
            report["phase"] = "restore_fixed_A_and_validation_features"
            with restored_parent(project, work, pins, identity, step, reference, limits, ledger, device) as pipe:
                clips, _ = fixed_features(project, work, pins, pipe, reference, ledger)
                first = next(iter(clips))
                cascade = CommonEvaluation(pipe.data, {"train": {first: clips[first]}}, pipe.cfg, device, 0)
                body = cascade.make("body")
                pipe._load_best("body", body)
                report["phase"] = "apply_selected_seed71_head"
                head = pins["head"]
                apply_head(body, _regular_read(work/"selected-head.pt", MIB, head["bytes"]), head)
                hand = cascade.make("hand", body.checkpoint_sha256)
                pipe._load_best("hand", hand)
                binding = {"parent": identity, "parent_receipt": FINAL, "feature_identity": FEATURE,
                           "A_model_sha256": original["weights"], "effective_body_sha256": body.checkpoint_sha256,
                           "head": head, "baseline": pins["baseline"], "modules": pins["modules"],
                           "sampler": "DDIM", "body_steps": STEP, "hand_steps": STEP, "evaluation_seed": 71,
                           "noise_offsets": list(NOISE_OFFSETS), "split": "official_validation", "training_updates": 0}
                report["phase"] = "evaluate_official_validation"
                metrics, noise = evaluate(cascade, body, hand, clips, reference, ledger.check)
                verdict = compare(metrics, noise, baseline, report["environment"])
                quality = {"schema": "rewind_v25_quality_v1", "counts": COUNTS, "binding": binding,
                           "environment": report["environment"], "v23_environment": baseline["environment"],
                           "candidate": metrics, "v23_A_ddim2": baseline["quality"][str(STEP)], "initial_noise_by_clip": noise,
                           "noise_manifest_sha256": verdict["noise"]["candidate_sha256"]}
                report["files"]["quality.json"] = verified_write(work, "quality.json", quality)
                report["files"]["verdict.json"] = verified_write(work, "verdict.json", {**verdict, "binding": binding})
        finally:
            report["usage"] = dict(ledger.current)
    report.update(status="completed", phase="finished")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("project", "work", "pins"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--started", type=float, required=True)
    args = parser.parse_args()
    report = {"schema": "rewind_v25_worker_v1", "status": "failed", "phase": "preflight", "files": {}}
    try:
        run(args, json.loads(_regular_read(args.pins, 2*MIB)), report)
    except BaseException as exc:
        report["failure"] = {"type": type(exc).__name__, "message": str(exc)[:800]}
        raise
    finally:
        report["wall_seconds"] = time.monotonic()-args.started
        verified_write(args.work, "worker-result.json", report)


if __name__ == "__main__":
    main()
