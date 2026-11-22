"""Extend the completed v2.2 A-body cache with only newly assigned train clips."""

from pathlib import Path
import tempfile
import time
import uuid

import numpy as np
from rewind_archive.contracts import digest
from scripts.v210_eval import group_keys
from scripts.v211_progress import publish_artifact, require_io_room
from scripts.v211_trial import SEEDS as INITIAL
from scripts.v220_upper import ASSIGNMENT, CAP, GROUP, INDEX, assignments, read_group
from scripts.v2_gpu_training_storage import EVAL_SEED, publish_json, read_sealed


SEEDS = (137, 181)
INITIAL_ROOT = "reports/v220_hand_upper128_a_v1"


def cache_basis(common, body, plans):
    return {**common, "body": body.checkpoint_sha256, "noise_seed": EVAL_SEED, "noise_offset": 100,
            "sampling_steps": 10, "assignment": ASSIGNMENT,
            "assignment_shas": {str(seed): digest(sorted(keys)) for seed, keys in plans.items()}}


def initial_predictions(directory, expected, common, body, data, joints, ledger):
    index = read_sealed(directory/"index.json", INDEX)
    plans = {seed: assignments(data.clips["train"], seed) for seed in INITIAL}
    train = frozenset().union(*plans.values())
    base = cache_basis(common, body, plans)
    if (index["content_sha256"] != expected or index["binding"] != base
            or index["counts"] != {"train": len(train), "validation": 421}
            or index["prediction_tensor_bytes"] != (len(train)+421)*data.width*joints*3*4):
        raise ValueError("v220_final_initial_cache_index")
    items, receipts = {}, []
    for group in range(147):
        selected = [(split, key) for split, key in group_keys(data, group) if split == "validation" or key in train]
        if selected:
            ledger.check()
            rows, receipt = read_group(directory, directory/f"body-{group:05d}.json", {**base, "group": group},
                                       selected, data, joints, ledger)
            items.update(rows)
            receipts.append(receipt["content_sha256"])
    if receipts != index["group_receipts"] or len(items) != len(train)+421:
        raise ValueError("v220_final_initial_cache_group_chain")
    return items


def extend(directory, work, prior, store, cascade, body, plans, common, ledger, owner, environment, timing):
    data, joints = store.data, len(cascade.body_names)
    if body.model.training or any(p.requires_grad for p in body.model.parameters()) or tuple(plans) != SEEDS:
        raise ValueError("v220_final_frozen_body_or_scope")
    train = frozenset().union(*plans.values())
    expected = {("train", key) for key in train} | {("validation", key) for key in data.clips["validation"]}
    cache = {key: value for key, value in prior.items() if key in expected}
    missing = expected-cache.keys()
    if any(split != "train" for split, _ in missing):
        raise ValueError("v220_final_requires_original_validation_predictions")
    base = cache_basis(common, body, plans)
    base["delta_train_keys_sha256"] = digest(sorted(key for _,key in missing))
    receipts = []
    timing.update(reused_train=len(train)-len(missing), reused_validation=421, missing_train=len(missing),
                  computed_new_clips=0, reused_delta_groups=0, prediction_seconds=0.)
    for group in range(147):
        selected = [item for item in group_keys(data, group) if item in missing]
        if not selected:
            continue
        ledger.check()
        binding = {**base, "group": group}
        marker = directory/f"body-{group:05d}.json"
        if not marker.exists():
            require_io_room(ledger, read=3*CAP, write=CAP)
            start = time.monotonic()
            coordinates, masks = [], []
            for split, key in selected:
                ledger.check()
                clip = store.load(key, split)
                pred = body.predict(clip.select_joints(cascade.body_idx), cascade._noise(key, joints, 100)).cpu()
                coordinates.append(pred.numpy())
                masks.append(clip.time_valid.numpy())
                timing["computed_new_clips"] += 1
            timing["prediction_seconds"] += time.monotonic()-start
            with tempfile.TemporaryDirectory(dir=work, prefix="upper-delta-") as tmp:
                path = Path(tmp)/f"body-{group:05d}-{uuid.uuid4().hex}.npz"
                np.savez_compressed(path, coordinates_mm=np.concatenate(coordinates), time_valid=np.concatenate(masks))
                artifact = publish_artifact(path, directory, ledger, owner, CAP)
                publish_json(marker, GROUP, {"binding": binding, "keys": [[s,list(k)] for s,k in selected],
                                            "artifact": artifact, "environment": environment}, owner)
        else:
            timing["reused_delta_groups"] += 1
        items, receipt = read_group(directory, marker, binding, selected, data, joints, ledger)
        cache.update(items)
        receipts.append(receipt["content_sha256"])
    if cache.keys() != expected:
        raise ValueError("v220_final_delta_cache_coverage")
    result = {"binding": base, "group_receipts": receipts, "counts": {"train": len(train), "validation": 421},
              "new_train_clips": len(missing)}
    marker = directory/"index.json"
    if marker.exists():
        index = read_sealed(marker, INDEX)
        if index["content_sha256"] != digest({"schema": INDEX, **result}):
            raise ValueError("v220_final_delta_cache_index")
    else:
        index = publish_json(marker, INDEX, result, owner)
    return cache, index
