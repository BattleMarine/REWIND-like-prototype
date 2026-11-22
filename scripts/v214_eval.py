"""Compare wrist-weight runs with the original weight-1 validation artifacts."""

from collections import defaultdict
from statistics import mean, stdev

import numpy as np
from rewind_archive.contracts import digest
from scripts.v210_eval import group_keys, original_json, read_group
from scripts.v210_metrics import official
from scripts.v211_trial import SEEDS, TRAJECTORY, TRAINING, UPDATES
from scripts.v214_decision import CONTROL_ROOTS, CONTROL_SCHEMAS
from scripts.v2_gpu_training_storage import publish_json, read_sealed


PAIR = "rewind_v214_weight_pair_v1"
COMPARISON = "rewind_v214_weight_comparison_v1"
COUNTS = {"clips": 421, "real_frames": 16425}
METRICS = ("left_wrist", "right_wrist", "body_without_wrists", "hand_global", "hand_local")
TRUTH_FIELDS = ("sequence_id", "clip_start", "frame_index", "seconds", "target_device_mm", "target_valid",
                "hand_local_target_mm", "hand_local_valid")


def load_predictions(directory, summary, data, names, ledger):
    groups = summary["group_receipts"]
    if len(groups) != 18 or summary["counts"] != COUNTS:
        raise ValueError("v214_validation_summary_coverage")
    rows = []
    for group, expected in zip(range(129, 147), groups, strict=True):
        ledger.check()
        keys = [key for split, key in group_keys(data, group) if split == "validation"]
        items, receipt = read_group(directory, directory/f"group-{group:05d}.json", {**summary["binding"], "group": group},
                                    keys, data, names, ledger)
        if receipt["content_sha256"] != expected:
            raise ValueError("v214_validation_group_receipt_sha256")
        rows.extend(items)
    frames = {(key[0], int(frame)) for key, record in rows for frame in record["frame_index"]}
    if [key for key, _ in rows] != data.clips["validation"] or len(frames) != COUNTS["real_frames"]:
        raise ValueError("v214_validation_clip_or_frame_coverage")
    return rows


def control(project, decision, seed, basis, data, names, ledger):
    method = decision["method"]
    relative = f"{CONTROL_ROOTS[method]}/seed-{seed}"
    pin = decision["controls"][str(seed)]
    summary = original_json(project, f"{relative}/summary.json", {"bytes": pin["summary_bytes"], "sha256": pin["summary_sha256"]},
                            CONTROL_SCHEMAS[method])
    bind = summary["binding"]
    if (summary["content_sha256"] != pin["content_sha256"] or bind["body_sha256"] != pin["checkpoint_sha256"]
            or bind["method"] != method or bind["seed"] != seed or bind["trajectory"] != TRAJECTORY
            or bind["training"] != TRAINING or bind["sampling_steps"] != 10 or summary["measurement"]["updates"] != UPDATES
            or any(bind[name] != basis[name] for name in ("plan", "source", "original_model", "reference", "feature_identity",
                    "feature_groups_sha256", "ordered_train_clips_sha256", "comparison_common", "trainable"))):
        raise ValueError("v214_unmatched_weight_1_control")
    return load_predictions(project/relative, summary, data, names, ledger), summary


def sample_metrics(row, spec):
    pred, real, valid = (row[name] for name in ("pred_device_mm", "target_device_mm", "target_valid"))
    body = spec["body_joint_names"]
    wrists = [body.index(spec["wrists"][side]) for side in ("left", "right")]
    remaining = [index for index in range(len(body)) if index not in wrists]
    selectors = dict(zip(METRICS[:4], (*wrists, remaining, slice(len(body), None)), strict=True))
    result = {key: official(pred[:, indices], real[:, indices], valid[:, indices]) for key, indices in selectors.items()}
    result["hand_local"] = official(row["hand_local_pred_mm"], row["hand_local_target_mm"], row["hand_local_valid"])
    return result


def scores(buckets):
    def pooled(parts):
        result = {}
        for name in METRICS:
            distance = sum(row[name][0] for row in parts)
            count = sum(row[name][1] for row in parts)
            if count <= 0:
                raise ValueError("v214_no_valid_joints_for_metric")
            result[name] = {"mpjpe_mm": distance/count, "valid_joints": count}
        return result

    scenes = {name: pooled(rows) for name, rows in sorted(buckets.items())}
    if len(scenes) != 3:
        raise ValueError("v214_official_scenario_coverage")
    return {"scenarios": scenes, "pooled": pooled([part for rows in buckets.values() for part in rows]),
            "macro_mm": {name: mean(scene[name]["mpjpe_mm"] for scene in scenes.values()) for name in METRICS}}


def publish_pair(path, old, new, old_summary, new_summary, seed, spec, owner):
    left, right = defaultdict(list), defaultdict(list)
    for (key, a), (other, b) in zip(old, new, strict=True):
        if key != other or any(not np.array_equal(a[name], b[name]) for name in TRUTH_FIELDS):
            raise ValueError("v214_control_treatment_target_or_mask_mismatch")
        scenario = key[0].split("/", 1)[0]
        left[scenario].append(sample_metrics(a, spec))
        right[scenario].append(sample_metrics(b, spec))
    baseline, treatment = scores(left), scores(right)
    result = {"seed": seed, "weight_1": baseline, "weight_2": treatment,
              "weight_1_summary": old_summary["content_sha256"], "weight_2_summary": new_summary["content_sha256"],
              "macro_delta_mm": {name: treatment["macro_mm"][name]-baseline["macro_mm"][name] for name in METRICS}}
    return publish_or_verify(path, PAIR, result, owner)


def publish_or_verify(path, schema, result, owner):
    if path.exists():
        saved = read_sealed(path, schema)
        if saved["content_sha256"] != digest({"schema": schema, **result}):
            raise ValueError("v214_existing_pair_or_comparison_changed")
        return saved
    return publish_json(path, schema, result, owner)


def publish_comparison(path, pairs, method, decision_sha, owner):
    if tuple(row["seed"] for row in pairs) != SEEDS:
        raise ValueError("v214_missing_paired_trials")
    result = {"parent_method": method, "decision_sha256": decision_sha, "seeds": list(SEEDS),
              "pairs": pairs, "mean_macro_delta_mm": {name: mean(row["macro_delta_mm"][name] for row in pairs) for name in METRICS},
              "sample_stdev_delta_mm": {name: stdev(row["macro_delta_mm"][name] for row in pairs) for name in METRICS}}
    return publish_or_verify(path, COMPARISON, result, owner)
