"""Replay the pinned v2 128 A model on validation clips from the portable bundle."""

import argparse
from copy import deepcopy
import io
import json
import os
from pathlib import Path
import platform
import tempfile
import time
import zipfile

import numpy as np
from PIL import Image
import torch

from rewind_archive.contracts import canonical
from rewind_m1.local_rgb import _regular_read
from rewind_m1.rgb_frontend_artifacts import scale_xy
from rewind_m1.training import seed_run
from rewind_m3.cascade import UpperCondition
from rewind_prepare.extracted import IMAGE_SIZE
from rewind_v1.data import attach_predicted_hands, hand_clip
from rewind_v1.frontend import frontend_fingerprint
from rewind_v1.full_data import FeatureStore
from rewind_v1.runner import _read_checkpoint
from rewind_v1.v2_contract import SCENARIOS
from scripts.v2_gpu_training_comparison import draw_clip
from scripts.v2_gpu_training_pipeline import CommonEvaluation
from scripts.v2_gpu_training_storage import EVAL_SEED, MIB


TARGET_ARRAYS = ("ground_truth_3d_mm", "ground_truth_2d_px", "valid_3d", "valid_2d", "joint_names", "seconds")
PREDICTION_TOLERANCES = {"predicted_2d_px": 0.005, "predicted_3d_mm": 0.5}
ROLES = ("frontend", "body", "hand")
WARMSTART_SCHEMA = "rewind_v2_gpu128_stage_warmstart_v1"


def select_clips(data, published, scenario=None, sequence=None, start=None):
    if (sequence is None) != (start is None) or (sequence is not None and scenario is not None):
        raise ValueError("use --scenario, or --sequence and --start together")
    if set(published) != set(SCENARIOS):
        raise ValueError("v2_replay_reference_scenarios")
    if sequence is not None:
        selected = [(sequence.split("/", 1)[0], (sequence, start))]
    else:
        selected = [(name, tuple(published[name])) for name in SCENARIOS if scenario is None or name == scenario]
    if not selected:
        raise ValueError("v2_replay_empty_selection")
    for name, key in selected:
        if (name not in SCENARIOS or not key[0].startswith(name+"/") or key not in data.clips["validation"]
                or data.real_frames(key, "validation") != data.width or data.width != 50):
            raise ValueError("v2_replay_requires_complete_official_validation_clip")
    return selected


def score_3d(predicted, target, valid):
    distances = np.linalg.norm(predicted[valid].astype(np.float64)-target[valid].astype(np.float64), axis=-1)
    return {"valid_joints": int(distances.size), "mpjpe_mm": float(distances.mean()) if distances.size else None}


def compare_predictions(reference_zip, scenario, values, *, reproduction=True):
    with zipfile.ZipFile(reference_zip) as archive:
        member = archive.getinfo(f"{scenario}.npz")
        if not 0 < member.file_size <= 2*MIB:
            raise ValueError("v2_replay_reference_npz_size")
        raw = archive.read(member)
    with np.load(io.BytesIO(raw), allow_pickle=False) as archive:
        for name in TARGET_ARRAYS:
            reference = archive[name]
            if values[name].dtype != reference.dtype or not np.array_equal(values[name], reference):
                raise ValueError(f"v2_replay_reference_input_mismatch: {name}")
        result = {}
        for name, tolerance in PREDICTION_TOLERANCES.items():
            previous = archive[name]
            current = values[name]
            if previous.shape != current.shape or not np.isfinite(previous).all():
                raise ValueError(f"v2_replay_reference_prediction_invalid: {name}")
            differences = np.abs(current.astype(np.float64)-previous.astype(np.float64))
            maximum = float(differences.max())
            result[name] = {"max_abs_delta": maximum, "mean_abs_delta": float(differences.mean())}
            if reproduction:
                result[name].update(comparison_tolerance=tolerance, within_tolerance=maximum <= tolerance)
        result["status"] = ("within_tolerance" if all(result[n]["within_tolerance"] for n in PREDICTION_TOLERANCES)
                            else "outside_tolerance") if reproduction else "delta_from_original_model"
        return result


def load_warmstart(bundle, directory):
    """Bind an explicitly selected tune artifact to the exact original cascade."""
    directory = Path(directory)
    if directory.is_symlink() or not directory.is_dir() or directory.resolve() != directory.absolute():
        raise ValueError("v2_replay_warmstart_directory")
    report = json.loads(_regular_read(directory/"warmstart.json", 2*MIB))
    if report.get("status") != "completed" or report.get("schema") != WARMSTART_SCHEMA:
        raise ValueError("v2_replay_warmstart_report")
    binding, pin = report["binding"], report["checkpoint"]
    pipeline = bundle.pipeline
    best = pipeline.progress["best"]
    role = binding.get("stage")
    if role not in ROLES:
        raise ValueError("v2_replay_warmstart_role")
    expected_cfg = deepcopy(pipeline.cfg)
    cfg = binding["config"]
    policy = "frontend" if role == "frontend" else "teacher"
    expected_cfg["seed"] = cfg["seed"]
    expected_cfg[policy]["learning_rate"] = cfg[policy]["learning_rate"]
    if (binding["parent_identity"] != pipeline.identity or binding["parent_receipt_sha256"] != bundle.diagnostic["checkpoint"]["content_sha256"]
            or binding["parent_stage_checkpoint"] != best[role] or binding["upstream_checkpoints"] != {
                name: best[name]["sha256"] for name in ROLES[:ROLES.index(role)]}
            or binding["eval_seed"] != EVAL_SEED or binding["original_optimizer_loaded"] is not False or cfg != expected_cfg):
        raise ValueError("v2_replay_warmstart_parent_or_architecture")
    cap = pipeline.cfg["limits"]["checkpoint_bytes"]
    filename = directory/"warmstart.pt"
    if pin["name"] != filename.name or type(pin["bytes"]) is not int or not 0 < pin["bytes"] <= cap or filename.stat().st_size != pin["bytes"]:
        raise ValueError("v2_replay_warmstart_checkpoint_pin")
    checkpoint = _read_checkpoint(filename, pin["sha256"], cap)
    if checkpoint.get("schema") != WARMSTART_SCHEMA or canonical(checkpoint["binding"]) != canonical(binding):
        raise ValueError("v2_replay_warmstart_checkpoint_binding")
    weights = checkpoint["fit"]["model"]
    if not weights or any(not torch.is_tensor(t) or not torch.isfinite(t).all() for t in weights.values()):
        raise ValueError("v2_replay_warmstart_weights")
    return role, pin["sha256"], weights


def use_warmstart(stage, pin, weights):
    expected = stage.model.state_dict()
    if expected.keys() != weights.keys() or any(expected[n].shape != t.shape or expected[n].dtype != t.dtype
                                               for n, t in weights.items()):
        raise ValueError("v2_replay_warmstart_parameter_shape_or_dtype")
    stage.model.load_state_dict(weights, strict=True)
    stage.model.eval().requires_grad_(False)
    if hasattr(stage, "checkpoint_sha256"):
        stage.checkpoint_sha256 = pin


@torch.no_grad()
def predict_clip(pipeline, ledger, frontend, stages, body, hand, key, clip):
    data, device = pipeline.data, pipeline.device
    sequence, start = key
    ledger.check()
    body_pred = body.predict(clip.select_joints(stages.body_idx), stages._noise(key, len(stages.body_names), 100)).cpu()
    values = body_pred[..., stages.upper_idx, :]
    valid_upper = clip.time_valid[..., None].expand(values.shape[:-1])
    upper = UpperCondition(values, valid_upper, "predicted_frozen_body", body.checkpoint_sha256)
    ledger.check()
    hand_target = hand_clip(clip, data.spec)
    hand_local = hand.predict(hand_target, stages._noise(key, len(stages.fingers), 200), upper=upper).cpu()
    hand_global = attach_predicted_hands(body_pred, hand_local, data.spec)
    names = stages.body_names + stages.fingers
    indices = stages.body_idx + stages.finger_idx
    points = torch.cat((body_pred, hand_global), dim=2)[0].numpy()
    target = clip.target_mm[0, :, indices, :].numpy()
    valid = clip.target_valid[0, :, indices].numpy()
    images, xy, xy_gt, xy_mask = [], [], [], []
    for offset in range(0, data.width, 2):
        ledger.check()
        pair = [data.cache.frame((sequence, start+i), "validation") for i in range(offset, offset+2)]
        inputs = torch.stack([image for image, _ in pair])
        predicted = frontend.model(inputs.to(device))[1].cpu()
        images.extend(inputs.numpy())
        xy.extend(predicted.numpy())
        xy_gt.extend(meta["xy"].numpy() for _, meta in pair)
        xy_mask.extend(meta["xy_valid"].numpy() for _, meta in pair)
    xy, xy_gt, xy_mask = map(np.asarray, (xy, xy_gt, xy_mask))
    if not all(np.isfinite(v).all() for v in (xy, points, hand_local.numpy(), xy_gt, target)):
        raise ValueError("v2_replay_nonfinite_clip")
    inside = xy_mask & (xy_gt >= 0).all(-1) & (xy_gt <= 127).all(-1)
    xy_error = scale_xy(torch.from_numpy(xy-xy_gt), pipeline.frontend_size, IMAGE_SIZE).numpy()
    xy_distance = np.linalg.norm(xy_error[inside].astype(np.float64), axis=-1)
    body_count = len(stages.body_names)
    scores = {"frontend": {"valid_joint_views": int(xy_distance.size),
                           "mean_error_native_px": float(xy_distance.mean()) if xy_distance.size else None},
              "body": score_3d(points[:, :body_count], target[:, :body_count], valid[:, :body_count]),
              "hand_global": score_3d(points[:, body_count:], target[:, body_count:], valid[:, body_count:]),
              "hand_local": score_3d(hand_local[0].numpy(), hand_target.target_mm[0].numpy(), hand_target.target_valid[0].numpy())}
    return np.asarray(images), {"predicted_3d_mm": points, "ground_truth_3d_mm": target, "valid_3d": valid,
                                "joint_names": np.asarray(names), "predicted_2d_px": xy, "ground_truth_2d_px": xy_gt,
                                "valid_2d": xy_mask, "seconds": clip.seconds.numpy()}, scores


def reproduce(bundle, output, *, scenario=None, sequence=None, start=None, warmstart=None):
    pipeline, ledger = bundle.pipeline, bundle.ledger
    data, device = pipeline.data, pipeline.device
    published = bundle.comparison["binding"]
    pins = {role: pipeline.progress["best"][role]["sha256"] for role in ROLES}
    if list(pipeline.frontend_size) != [128, 128] or pins != published["model"] or published["eval_seed"] != EVAL_SEED:
        raise ValueError("v2_replay_model_or_noise_binding")
    tuned_role, tuned_pin, weights = load_warmstart(bundle, warmstart) if warmstart is not None else (None, None, None)
    effective_pins = {**pins, **({tuned_role: tuned_pin} if tuned_role else {})}
    selected = select_clips(data, published["clips"], scenario, sequence, start)
    seed_run(pipeline.cfg["seed"])
    frontend = pipeline._make_frontend()
    pipeline._load_best("frontend", frontend)
    if tuned_role == "frontend":
        use_warmstart(frontend, tuned_pin, weights)
    feature_store = FeatureStore(output, data, frontend_fingerprint(frontend.model))
    features, feature_timing = {}, {}
    for _, key in selected:
        started = time.monotonic()
        features[key] = data.clip(key, "validation", frontend.model, pipeline.frontend_size, feature_store.identity)
        feature_timing[key] = time.monotonic()-started
    first = selected[0][1]
    # The frozen constructor obtains its shared feature identity from the train map.
    stages = CommonEvaluation(data, {"train": {first: features[first]}, "validation": features}, pipeline.cfg, device, 0)
    body, hand = stages.make("body"), stages.make("hand", effective_pins["body"])
    pipeline._load_best("body", body)
    pipeline._load_best("hand", hand)
    if tuned_role in ("body", "hand"):
        use_warmstart(body if tuned_role == "body" else hand, tuned_pin, weights)
    del weights
    results = []
    for name, key in selected:
        started = time.monotonic()
        images, values, scores = predict_clip(pipeline, ledger, frontend, stages, body, hand, key, features[key])
        predicted_path, gif_path = output/f"{name}.npz", output/f"{name}.gif"
        np.savez_compressed(predicted_path, **values)
        files = [predicted_path.name]
        if values["valid_3d"].any():
            gif_args = tuple(values[k] for k in ("predicted_2d_px", "ground_truth_2d_px", "valid_2d", "predicted_3d_mm",
                                                 "ground_truth_3d_mm", "valid_3d", "joint_names"))
            draw_clip(gif_path, images, *gif_args, data.spec)
            with Image.open(gif_path) as image:
                image.save(output/f"{name}.png")
            files.extend((gif_path.name, f"{name}.png"))
        ledger.check()
        reference = (compare_predictions(bundle.comparison_path, name, values, reproduction=tuned_role is None)
                     if key == tuple(published["clips"][name]) else {"status": "not_published_clip"})
        results.append({"scenario": name, "sequence": key[0], "start_frame": key[1], "frames": data.width,
                        "scores_on_this_clip": scores, "published_comparison": reference,
                        "files": files, "feature_seconds": feature_timing[key],
                        "prediction_and_export_seconds": time.monotonic()-started})
    report = {"schema": "rewind_v2_gpu128_a_offline_prediction_v1", "status": "completed", "experiment": published["experiment"],
              "model_sha256": effective_pins, "published_comparison_sha256": bundle.comparison["content_sha256"],
              "model_selection": {"mode": "original" if tuned_role is None else "single_stage_experiment",
                                  "warmstart_stage": tuned_role, "original_model_sha256": pins},
              "runtime": {"python": platform.python_version(), "torch": str(torch.__version__), "device": str(device),
                          "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else platform.machine(),
                          "cuda_runtime": torch.version.cuda, "threads": torch.get_num_threads(), "precision": "fp32"},
              "published_runtime": bundle.diagnostic.get("environment"),
              "noise": {"eval_seed": EVAL_SEED, "body_offset": 100, "hand_offset": 200},
              "coordinates": {"xy": "128px input", "xy_score": "1024px native", "xyz": "device_mm",
                              "time": "nominal frame index / 25fps"}, "clips": results}
    (output/"prediction-report.json").write_bytes(canonical(report))
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("bundle", "work", "output"):
        parser.add_argument("--"+name, type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--seconds", type=int, default=1800)
    parser.add_argument("--scenario", choices=SCENARIOS)
    parser.add_argument("--sequence")
    parser.add_argument("--start", type=int)
    parser.add_argument("--warmstart", type=Path, help="Use one verified tune output directory (warmstart.json and warmstart.pt)")
    args = parser.parse_args(argv)
    if args.output.exists() or args.output.is_symlink():
        parser.error("--output must be a new directory")
    bundle, work, output = (getattr(args, n).resolve() for n in ("bundle", "work", "output"))
    if any(a.is_relative_to(b) or b.is_relative_to(a) for a, b in ((bundle, output), (work, output))):
        parser.error("--output must be separate from the bundle and workspace")
    if args.warmstart and any(a.is_relative_to(args.warmstart.resolve()) or args.warmstart.resolve().is_relative_to(a)
                              for a in (output, work)):
        parser.error("--warmstart must be separate from output and workspace")
    if (args.sequence is None) != (args.start is None) or (args.sequence is not None and args.scenario is not None):
        parser.error("use --scenario, or --sequence and --start together")
    from v2_gpu128_a_bundle_local import open_bundle

    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    with tempfile.TemporaryDirectory(prefix=".v2-predictions-", dir=output.parent) as staging:
        result = Path(staging)/"predictions"
        result.mkdir()
        with open_bundle(bundle, work, seconds=args.seconds, device=args.device) as local:
            reproduce(local, result, scenario=args.scenario, sequence=args.sequence, start=args.start, warmstart=args.warmstart)
        if output.exists() or output.is_symlink():
            raise FileExistsError(output)
        result.rename(output)
    print(output/"prediction-report.json")


if __name__ == "__main__":
    main()
