"""Controlled fixed-A DDIM quality and device-resident cascade latency."""

from dataclasses import dataclass
import hashlib
import statistics
import time

import numpy as np
import torch

from rewind_archive.contracts import canonical
from rewind_m3.cascade import UpperCondition
from rewind_v1.data import attach_predicted_hands, hand_clip
from scripts.v210_metrics import aggregate, sums


STEPS = (10, 5, 2)
ROUNDS = ((10, 5, 2), (5, 2, 10), (2, 10, 5), (2, 5, 10), (5, 10, 2))
NOISE_OFFSETS = (100, 200)


@dataclass(frozen=True)
class PreparedClip:
    body_input: tuple
    hand_input: tuple
    body_noise: torch.Tensor
    hand_noise: torch.Tensor
    noise_sha256: str
    real_frames: int
    hand_target: object


def prepare(cascade, body, hand, key, clip):
    n = int(clip.time_valid.sum())
    if clip.split != "validation" or not 1 <= n <= 50 or not torch.equal(clip.time_valid, torch.arange(50)[None] < n):
        raise ValueError("v23_validation_clip_mask")
    if not torch.equal(clip.seconds[0], torch.arange(key[1], key[1] + 50, dtype=torch.float64) / 25):
        raise ValueError("v23_clip_nominal_seconds")
    local = hand_clip(clip, cascade.spec)
    body_noise = cascade._noise(key, len(cascade.body_names), NOISE_OFFSETS[0])
    hand_noise = cascade._noise(key, len(cascade.fingers), NOISE_OFFSETS[1])
    digest = hashlib.sha256(canonical(list(key)))
    for noise in (body_noise, hand_noise):
        if noise.device.type != "cpu" or noise.dtype != torch.float32 or noise.shape[:2] != (1, 50):
            raise ValueError("v23_initial_noise_layout")
        digest.update(canonical(list(noise.shape)))
        digest.update(noise.contiguous().numpy().tobytes())
    return PreparedClip(body._inputs(clip.select_joints(cascade.body_idx)), hand._inputs(local),
                        body_noise.to(body.device), hand_noise.to(hand.device), digest.hexdigest(), n, local)


@torch.no_grad()
def cascade_prediction(cascade, body, hand, prepared, steps):
    positions = body.diffusion.sample(body.model, *prepared.body_input, prepared.body_noise, steps)
    prediction = positions * body.config["target_scale_mm"]
    upper = prediction[..., cascade.upper_idx, :]
    upper_valid = prepared.body_input[2][..., None].expand(upper.shape[:-1])
    condition = UpperCondition(upper, upper_valid, "predicted_frozen_body", body.checkpoint_sha256)
    conditioned = hand._condition(condition, training=False).eval()
    local = hand.diffusion.sample(conditioned, *prepared.hand_input, prepared.hand_noise, steps)
    local = local * hand.config["target_scale_mm"]
    attached = attach_predicted_hands(prediction, local, cascade.spec)
    return torch.cat((prediction, attached), dim=2), local


def score_arrays(cascade, clip, prepared, predicted, local):
    n = prepared.real_frames
    joints = cascade.body_idx + cascade.finger_idx
    outputs = {"pred_device_mm": predicted[0, :n].detach().cpu().numpy(),
               "target_device_mm": clip.target_mm[0, :n, joints].numpy(),
               "target_valid": clip.target_valid[0, :n, joints].numpy(),
               "hand_local_pred_mm": local[0, :n].detach().cpu().numpy(),
               "hand_local_target_mm": prepared.hand_target.target_mm[0, :n].numpy(),
               "hand_local_valid": prepared.hand_target.target_valid[0, :n].numpy()}
    if any(array.dtype.kind == "f" and not np.isfinite(array).all() for array in outputs.values()):
        raise ValueError("v23_nonfinite_score_array")
    return outputs


def quality(cascade, body, hand, clips, fixed_keys, reference, check):
    by_steps = {step: [] for step in STEPS}
    noise, fixed = [], {}
    seen_frames = set()
    for key, clip in clips.items():
        check()
        ready = prepare(cascade, body, hand, key, clip)
        frame_keys = {(key[0], frame) for frame in range(key[1], key[1] + ready.real_frames)}
        if not seen_frames.isdisjoint(frame_keys):
            raise ValueError("v23_duplicated_validation_frames")
        seen_frames.update(frame_keys)
        noise.append({"key": list(key), "real_frames": ready.real_frames, "initial_noise_sha256": ready.noise_sha256})
        for step in STEPS:
            check()
            pred, local = cascade_prediction(cascade, body, hand, ready, step)
            arrays = score_arrays(cascade, clip, ready, pred, local)
            by_steps[step].append((key, arrays))
            if key in fixed_keys:
                fixed.setdefault(step, []).append({"key": list(key), "scores": sums(arrays, cascade.spec)})
    if len(clips) != 421 or len(seen_frames) != 16425:
        raise ValueError("v23_validation_coverage")
    values = {str(step): aggregate(by_steps[step], cascade.spec, reference) for step in STEPS}
    baseline = values["10"]
    deltas = {}
    for step, values_at_step in values.items():
        deltas[step] = {"macro_mm": values_at_step["macro_score_mm"] - baseline["macro_score_mm"],
                        "pooled": mpjpe_delta(values_at_step["pooled"], baseline["pooled"]),
                        "per_scenario": {scene: mpjpe_delta(metric, baseline["per_scenario"][scene])
                                         for scene, metric in values_at_step["per_scenario"].items()}}
    return {"schema": "rewind_v23_quality_v1", "counts": {"clips": 421, "real_frames": 16425},
            "sampling_steps": list(STEPS), "evaluation_noise_seed": 71, "initial_noise_offsets": list(NOISE_OFFSETS),
            "same_initial_noise_per_arm": True, "noise_manifest_sha256": hashlib.sha256(canonical(noise)).hexdigest(),
            "initial_noise_by_clip": noise, "quality": values, "delta_from_current_10_step": deltas,
            "fixed_clip_sufficient_statistics": {str(step): rows for step, rows in fixed.items()}}


def mpjpe_delta(candidate, baseline):
    result = {role: metric["mpjpe_mm"] - baseline["official"][role]["mpjpe_mm"]
              for role, metric in candidate["official"].items()}
    for side, values in candidate["sides"].items():
        result[f"{side}_wrist"] = values["wrist"]["mpjpe_mm"] - baseline["sides"][side]["wrist"]["mpjpe_mm"]
    return result


def timing_summary(repetitions, frames):
    if len(repetitions) != 5 or any(set(row) != {"cuda_stream_ms", "synchronized_wall_ms"}
                                   or any(not 0 < t < float("inf") for t in row.values()) for row in repetitions):
        raise ValueError("v23_requires_five_finite_positive_timings")
    report = {}
    for scope in ("cuda_stream_ms", "synchronized_wall_ms"):
        values = [row[scope] for row in repetitions]
        middle = statistics.median(values)
        report[scope] = {"samples": values, "median_bundle_ms": middle, "min_bundle_ms": min(values),
                         "max_bundle_ms": max(values), "median_amortized_ms_per_frame": middle / frames,
                         "median_frames_per_second": frames * 1000 / middle}
    return report


def benchmark(cascade, body, hand, clips, check):
    device = body.device
    if device.type != "cuda" or hand.device != device or len(clips) != 3 or any(c.real_frames != 50 for c in clips.values()):
        raise ValueError("v23_requires_one_cuda_and_fixed_3x50_timing_bundle")

    def call(steps):
        for item in clips.values():
            cascade_prediction(cascade, body, hand, item, steps)

    # One full bundle per arm for warm-up. Warm-ups never enter the recorded repeats.
    for steps in STEPS:
        check()
        call(steps)
        torch.cuda.synchronize(device)
    samples = {steps: [] for steps in STEPS}
    for order in ROUNDS:
        for steps in order:
            check()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            stream = torch.cuda.current_stream(device)
            torch.cuda.synchronize(device)
            begin_wall = time.perf_counter()
            start.record(stream)
            call(steps)
            end.record(stream)
            torch.cuda.synchronize(device)
            wall_ms = (time.perf_counter() - begin_wall) * 1000
            samples[steps].append({"cuda_stream_ms": start.elapsed_time(end), "synchronized_wall_ms": wall_ms})
    timings = {str(steps): timing_summary(values, 150) for steps, values in samples.items()}
    baseline_ms = timings["10"]["synchronized_wall_ms"]["median_bundle_ms"]
    return {"schema": "rewind_v23_latency_v1", "scope": "device_resident_predicted_feature_to_body_hand_xyz",
            "precision": "fp32", "input_resolution": [128, 128], "batch": 1, "clip_frames": 50,
            "bundle_real_frames": 150, "warmups_per_arm": 1, "measurements_per_arm": 5,
            "round_order": ROUNDS, "keys_and_noise": [{"key": list(key), "noise_sha256": clip.noise_sha256}
                                                     for key, clip in clips.items()],
            "timings": timings, "wall_speedup_vs_10_steps": {step: baseline_ms / sc["synchronized_wall_ms"]["median_bundle_ms"]
                                                           for step, sc in timings.items()}}
