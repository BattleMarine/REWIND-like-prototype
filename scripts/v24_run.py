"""Pilot only: one paired seed, 24 updates per arm and fixed subset validation."""

import argparse
import json
from pathlib import Path
import shutil
import tempfile
import time
from types import SimpleNamespace

import torch

from rewind_archive.contracts import canonical, digest
from rewind_m1.local_rgb import _regular_read
from rewind_m1.training import rng_state
from rewind_m3.cascade import UpperCondition
from rewind_prepare.temporal_cache import _hash_file
from rewind_v1.data import attach_predicted_hands, hand_clip
from rewind_v1.frontend import frontend_fingerprint
from rewind_v1.full_data import FeatureStore
from rewind_v1.v2_budget import Ledger
from scripts.run_v2_gpu_training import training_config
from scripts.v210_eval import FEATURE, FINAL, feature_group, group_keys
from scripts.v210_metrics import merge, sums
from scripts.v23_eval import score_arrays
from scripts.v23_run import gpu, parent_evidence, restored_parent, verified_write
from scripts.v24_body import (ARMS, EVAL_UPDATES, LEARNING_RATE, SEED, TRAJECTORY, UPDATES,
                              X0View, draw, model_sha, numerical, sample, setup, tensor_sha, update)
from scripts.v2_gpu_training_pipeline import CommonEvaluation
from scripts.v2_gpu_training_storage import GIB, MIB


SECONDS = 2700
RESERVE = 300
STATE_CAP = 192 * MIB
READ_CAP = 2 * GIB
WRITE_CAP = 3 * GIB
DESTINATION = "v24_diffusion_pilot128_a_v1"
SCOPE = "pilot_validation_subset"
TENSOR_FIELDS = ("features", "camera", "target_mm", "target_valid", "time_valid", "seconds")


def pilot_keys(data, reference):
    chosen_train, chosen_validation = [], []
    for scene in reference["binding"]["clips"]:
        train = [key for key in data.clips["train"] if key[0].startswith(scene + "/")]
        train.sort(key=lambda key: digest([TRAJECTORY, SEED, "train-roster", list(key)]))
        chosen_train.extend(train[:8])
        validation = [key for key in data.clips["validation"] if key[0].startswith(scene + "/")
                      and data.real_frames(key, "validation") == 50]
        if len(validation) < 2 or list(validation[0]) != reference["binding"]["clips"][scene]:
            raise ValueError("v24_expected_fixed_full_validation_clips")
        chosen_validation.extend(validation[:2])
    chosen_train.sort(key=lambda key: digest([TRAJECTORY, SEED, "train-order", list(key)]))
    if len(set(chosen_train)) != UPDATES or len(set(chosen_validation)) != 6 or set(chosen_train) & set(chosen_validation):
        raise ValueError("v24_pilot_split_or_roster")
    return chosen_train, chosen_validation


def features(project, work, pins, pipe, frontend, train_keys, validation_keys, ledger):
    store = FeatureStore(work / "unused-features", pipe.data, frontend_fingerprint(frontend.model))
    if store.identity != FEATURE:
        raise ValueError("v24_A_feature_binding")
    selected = {("train", key) for key in train_keys} | {("validation", key) for key in validation_keys}
    groups = {}
    for i in range(147):
        keys = [item for item in group_keys(pipe.data, i) if item in selected]
        if keys:
            groups[i] = keys
    if set(item for keys in groups.values() for item in keys) != selected:
        raise ValueError("v24_feature_selection_not_complete")
    found = {}
    for index, keys in groups.items():
        ledger.check()
        with tempfile.TemporaryDirectory(dir=work, prefix="v24-features-") as scratch:
            feature_group(project, pins, store, index, ledger, Path(scratch))
            for split, key in keys:
                found[split, key] = store.load(key, split)
    return found, {str(i): pins["groups"][str(i)]["archive"] for i in groups}


def feature_sha(clips):
    return tensor_sha((f"{split}/{key[0]}/{key[1]}/{field}", getattr(value, field))
                      for (split, key), value in clips.items() for field in TENSOR_FIELDS)


@torch.no_grad()
def frame_probe(stage, clip, noise):
    x, camera, time_valid = stage._inputs(clip)
    k, epsilon = (value.to(stage.device) for value in noise)
    valid = clip.target_valid.to(stage.device) & time_valid[..., None]
    clean = torch.where(valid[..., None], clip.target_mm.to(stage.device) / stage.config["target_scale_mm"], 0.)
    noisy = stage.diffusion.q_sample(clean, epsilon, k)
    stage.model.eval()
    direct, frame = X0View(stage.model, stage.diffusion, "x0").eval()(noisy, x, camera, k, time_valid)
    transformed, v_frame = X0View(stage.model, stage.diffusion, "v").eval()(noisy, x, camera, k, time_valid)
    if not torch.equal(frame, v_frame) or torch.equal(direct, transformed):
        raise ValueError("v24_frame_head_stays_x0_while_final_is_adapted")
    return tensor_sha((("frame_x0", frame),))


@torch.no_grad()
def validate(cascade, stage, arm, hand, clips, keys, check):
    stage.model.eval()
    candidate_sha = model_sha(stage.model)
    hand.spec["body_checkpoint_sha256"] = candidate_sha
    segments, evaluations = [], {}
    for key in keys:
        check()
        clip = clips["validation", key]
        local_clip = hand_clip(clip, cascade.spec)
        body_noise = cascade._noise(key, len(cascade.body_names), 100)
        hand_noise = cascade._noise(key, len(cascade.fingers), 200)
        body = sample(stage, arm, clip.select_joints(cascade.body_idx), body_noise)
        upper = body[..., cascade.upper_idx, :]
        valid = clip.time_valid.to(body.device)[..., None].expand(upper.shape[:-1])
        hand_pc = UpperCondition(upper, valid, "predicted_frozen_body", candidate_sha)
        local = hand.predict(local_clip, hand_noise, upper=hand_pc)
        pose = torch.cat((body, attach_predicted_hands(body, local, cascade.spec)), dim=2)
        n = int(clip.time_valid.sum())
        arrays = score_arrays(cascade, clip, SimpleNamespace(real_frames=n, hand_target=local_clip), pose, local)
        stats = sums(arrays, cascade.spec)
        segments.append({"key": list(key), "real_frames": n,
                         "noise_sha256": tensor_sha((("body", body_noise), ("hand", hand_noise))),
                         "scores": stats})
        evaluations.setdefault(key[0].split("/", 1)[0], []).append(stats)
    return {"scope": SCOPE, "body_state_sha256": candidate_sha,
            "per_scenario": {scene: merge(rows) for scene, rows in evaluations.items()},
            "pooled": merge([row["scores"] for row in segments]), "clips": segments}


def save_state(stage, arm, binding, work, ledger):
    name = f"{arm}-pilot-final.pt"
    ledger.reserve("write", STATE_CAP)
    path = work / name
    with path.open("xb") as output:
        torch.save({"schema": "rewind_v24_pilot_body_state_v1", "arm": arm, "binding": binding,
                    "step": stage.step, "body": stage.model.state_dict(), "optimizer": stage.optimizer.state_dict(),
                    "scheduler": stage.scheduler.state_dict(), "scaler": stage.scaler.state_dict(), "rng": rng_state()}, output)
    size = path.stat().st_size
    if not 0 < size <= STATE_CAP:
        raise ValueError("v24_pilot_state_size")
    ledger.observed("write", size)
    return name, {"bytes": size, "sha256": _hash_file(path)}


def pilot(project, work, pins, pipe, reference, ledger, report):
    teacher = pipe.cfg["teacher"]
    expected = {"dropout": 0.0, "diffusion_steps": 1000, "beta_start": .0001, "beta_end": .02, "sampling_steps": 10,
                "learning_rate": LEARNING_RATE, "frame_loss_weight": 1.0, "target_scale_mm": 100.,
                "velocity_loss_weight": 0.0}
    if any(teacher[key] != value for key, value in expected.items()):
        raise ValueError("v24_A_teacher_config")
    frontend = pipe._make_frontend()
    pipe._load_best("frontend", frontend)
    train, validation = pilot_keys(pipe.data, reference)
    clips, groups = features(project, work, pins, pipe, frontend, train, validation, ledger)
    cascade = CommonEvaluation(pipe.data, {"train": {train[0]: clips["train", train[0]]}}, pipe.cfg, pipe.device, 0)
    hand = cascade.make("hand", reference["binding"]["model"]["body"])
    pipe._load_best("hand", hand)
    upstream = {"frontend": model_sha(frontend.model), "hand": model_sha(hand.model), "features": feature_sha(clips)}
    binding = {"A_checkpoint_receipt": FINAL, "A_weights": reference["binding"]["model"], "feature_identity": FEATURE,
               "frozen_upstream_state_sha256": upstream, "source": pins["base"]["source_sha256"],
               "pilot_overlay": pins["modules"], "train_keys": train, "validation_keys": validation,
               "seed": SEED, "updates_per_arm": UPDATES, "validation_updates": EVAL_UPDATES, "learning_rate": LEARNING_RATE,
               "optimizer": "Adam", "trajectory": TRAJECTORY, "batch": 1,
               "teacher_config": pipe.cfg["teacher"], "eval_seed": 71}
    report["files"]["pilot-plan.json"] = verified_write(work, "pilot-plan.json",
                                                        {"binding": binding, "binding_sha256": digest(binding), "groups_read": groups})
    initial, probe, paired_noise, writeups = None, None, {}, {}
    for arm in ARMS:
        report["phase"] = f"pilot_{arm}"
        stage = cascade.make("body")
        pipe._load_best("body", stage)
        identity = setup(stage)
        first_clip = clips["train", train[0]].select_joints(cascade.body_idx)
        k, eps, _ = draw(stage.diffusion, 0, first_clip.target_mm.shape)
        fingerprint = frame_probe(stage, first_clip, (k, eps))
        if initial is None:
            initial, probe = identity, fingerprint
        elif initial != identity or probe != fingerprint:
            raise ValueError("v24_initialization_or_frame_output_differs")
        curve, updates = [dict(update=0, **validate(cascade, stage, arm, hand, clips, validation, ledger.check))], []
        start = time.monotonic()
        for index, key in enumerate(train):
            ledger.check()
            clip = clips["train", key].select_joints(cascade.body_idx)
            k, eps, sha = draw(stage.diffusion, index, clip.target_mm.shape)
            if arm == ARMS[0]:
                paired_noise[index] = sha
            elif paired_noise[index] != sha:
                raise ValueError("v24_paired_training_noise_diverged")
            stats = update(stage, arm, clip, (k, eps))
            updates.append({"step": stage.step, "key": list(key), "timestep": int(k.item()),
                            "initial_noise_sha256": sha, **stats})
            if stage.step in EVAL_UPDATES:
                curve.append(dict(update=stage.step, **validate(cascade, stage, arm, hand, clips, validation, ledger.check)))
        if [(point["update"], point["pooled"]["frames"]) for point in curve] != [(i, 300) for i in EVAL_UPDATES]:
            raise ValueError("v24_validation_scope_or_update_count")
        if any(model.training or any(p.requires_grad or p.grad is not None for p in model.parameters())
               for model in (frontend.model, hand.model)):
            raise ValueError("v24_upstream_grad_or_mode")
        if {"frontend": model_sha(frontend.model), "hand": model_sha(hand.model), "features": feature_sha(clips)} != upstream:
            raise ValueError("v24_upstream_tensor_state_changed")
        name, stat = save_state(stage, arm, binding, work, ledger)
        report["files"][name] = stat
        writeups[arm] = {"updates": updates, "curve": curve, "final_state": stat, "fit_and_eval_seconds": time.monotonic() - start}
        report["files"][f"{arm}-pilot.json"] = verified_write(work, f"{arm}-pilot.json", writeups[arm])
        del stage
    for left, right in zip(writeups["x0"]["curve"], writeups["v"]["curve"], strict=True):
        if [row["noise_sha256"] for row in left["clips"]] != [row["noise_sha256"] for row in right["clips"]]:
            raise ValueError("v24_paired_validation_noise_diverged")
    report["files"]["pilot-summary.json"] = verified_write(work, "pilot-summary.json",
        {"schema": "rewind_v24_diffusion_pilot_v1", "scope": SCOPE, "initialization": initial,
         "intermediate_A_x0_output_sha256": probe, "frozen_upstream_state_sha256": upstream,
         "loss_policy": {"final_position": "normalized masked x0 MSE for both arms",
                         "v_equivalent_timestep_weight": "1-alpha_bar", "additional_raw_v_mse_weight": 0,
                         "intermediate_frame_x0_weight": pipe.cfg["teacher"]["frame_loss_weight"]},
         "paired_noise_match": True, "arms": {arm: {"validation_curve": info["curve"], "final": info["final_state"]}
                                              for arm, info in writeups.items()}})


def run(args, pins, report):
    project, work = args.project.resolve(strict=True), args.work.resolve(strict=True)
    minimum_free = 4 * GIB if args.pilot else 256 * MIB
    if work.is_relative_to(project) or shutil.disk_usage(work).free < minimum_free:
        raise ValueError("v24_input_or_scratch")
    for name, sha in pins["modules"].items():
        if _hash_file(Path(__file__).resolve().parent / name) != sha:
            raise ValueError("v24_module_pin")
    report["files"]["numerical.json"] = verified_write(work, "numerical.json", numerical())
    if not args.pilot:
        report.update(status="completed_numerical", phase="finished")
        return
    report["phase"] = "initialize_cuda"
    # Reuse allocator initialization and FP32 controls from the fixed v2.3 release.
    device = gpu(args.started, report)
    identity, step, reference = parent_evidence(project, pins)
    cfg = training_config(42)
    limits = {**cfg, "seconds_per_invocation": SECONDS, "external_timeout_seconds": SECONDS,
              "terminal_reserve_seconds": RESERVE,
              "resources": {**cfg["resources"], "read_bytes": READ_CAP, "write_bytes": WRITE_CAP}}
    ledger = Ledger(work / "usage", identity, limits)
    with ledger.invocation("v24_pilot"):
        ledger.started = args.started
        try:
            report["phase"] = "restore_A_and_pilot_features"
            with restored_parent(project, work, pins, identity, step, reference, limits, ledger, device) as pipe:
                pilot(project, work, pins, pipe, reference, ledger, report)
        finally:
            report["usage"] = {key: value for key, value in ledger.current.items() if key != "status"}
    report.update(status="completed_pilot", phase="finished")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("project", "work", "pins"):
        parser.add_argument(f"--{name}", required=True, type=Path)
    parser.add_argument("--started", required=True, type=float)
    parser.add_argument("--pilot", action="store_true")
    args = parser.parse_args()
    report = {"schema": "rewind_v24_worker_v1", "status": "failed", "phase": "preflight", "files": {}}
    try:
        run(args, json.loads(_regular_read(args.pins, 2 * MIB)), report)
    except BaseException as exc:
        if isinstance(exc, (TimeoutError, KeyboardInterrupt)):
            report["status"] = "stopped"
        report["error"] = {"type": type(exc).__name__, "message": str(exc)[:1200]}
        raise
    finally:
        report["elapsed_seconds"] = time.monotonic() - args.started
        verified_write(args.work, "worker-result.json", report)


if __name__ == "__main__":
    main()
