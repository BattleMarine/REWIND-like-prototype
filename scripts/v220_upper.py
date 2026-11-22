"""Paired clip assignments and verified shared A-body predictions for v2.2."""

import hashlib
import io
import tempfile
import time
import uuid
from pathlib import Path

import numpy as np
import torch
from rewind_archive.contracts import digest
from rewind_m1.local_rgb import _regular_read
from rewind_m3.cascade import UpperCondition
from rewind_v1.data import gt_upper_train
from scripts.v210_eval import group_keys, input_file
from scripts.v211_progress import publish_artifact, require_io_room
from scripts.v211_trial import SEEDS, UPDATES
from scripts.v2_gpu_training_storage import EVAL_SEED, MIB, publish_json, read_sealed


CONTROL, MIXED = "gt_100_pred_0", "gt_50_pred_50"
POLICIES = (CONTROL, MIXED)
ASSIGNMENT = "rewind_v220_balanced_clip_assignment_v1"
GROUP = "rewind_v220_A_body_prediction_group_v1"
INDEX = "rewind_v220_A_body_prediction_index_v1"
CAP = 2*MIB


def assignments(keys, seed):
    if len(set(keys)) != len(keys):
        raise ValueError("v220_train_keys_duplicate")
    ranked = sorted(keys, key=lambda key: (digest([ASSIGNMENT, seed, key]), key))
    return frozenset(ranked[:len(keys)//2])


def condition(clip, key, cascade, cache, selected, body_sha):
    if key not in selected:
        return gt_upper_train(clip, cascade.spec)
    body = cached_body(cache, "train", key, clip)
    upper = body[..., cascade.upper_idx, :]
    return UpperCondition(upper, clip.time_valid[..., None].expand(upper.shape[:-1]), "predicted_frozen_body", body_sha)


def cached_body(cache, split, key, clip):
    tensor, mask = cache[split, key]
    if clip.split != split or not torch.equal(mask, clip.time_valid):
        raise ValueError("v220_cached_body_clip_time_mask")
    return tensor


def read_group(directory, marker, binding, selected, data, joints, ledger):
    receipt = read_sealed(marker, GROUP)
    if receipt["binding"] != binding or receipt["keys"] != [[s, list(k)] for s, k in selected]:
        raise ValueError("v220_body_group_binding")
    package = receipt["artifact"]
    if not 0 < package["bytes"] <= CAP:
        raise ValueError("v220_body_group_size")
    ledger.reserve("read", package["bytes"])
    raw = _regular_read(input_file(directory, package["name"]), CAP, package["bytes"])
    ledger.observed("read", len(raw), persist=False)
    if hashlib.sha256(raw).hexdigest() != package["sha256"]:
        raise ValueError("v220_body_group_sha256")
    with np.load(io.BytesIO(raw), allow_pickle=False) as arrays:
        if set(arrays.files) != {"coordinates_mm", "time_valid"}:
            raise ValueError("v220_body_group_fields")
        coords, mask = arrays["coordinates_mm"], arrays["time_valid"]
    expected_mask = np.arange(data.width)[None, :] < np.asarray([data.real_frames(k, s) for s, k in selected])[:, None]
    if (coords.shape != (len(selected), data.width, joints, 3) or coords.dtype != np.float32
            or mask.dtype != np.bool_ or not np.array_equal(mask, expected_mask)
            or not np.isfinite(coords).all() or np.any(coords[~mask])):
        raise ValueError("v220_body_group_shape_mask_or_finite")
    return {item: (torch.from_numpy(coords[i:i+1]), torch.from_numpy(mask[i:i+1]))
            for i, item in enumerate(selected)}, receipt


def prepare_upper(directory, work, store, cascade, body, plans, basis, ledger, owner, environment, timing):
    data, joints = store.data, len(cascade.body_names)
    train = frozenset().union(*plans.values())
    if len(data.clips["train"]) != UPDATES or set(plans) != set(SEEDS) or any(len(x) != UPDATES//2 for x in plans.values()):
        raise ValueError("v220_upper_train_coverage")
    if body.model.training or any(p.requires_grad for p in body.model.parameters()):
        raise ValueError("v220_body_must_be_frozen")
    base = {**basis, "body": body.checkpoint_sha256, "noise_seed": EVAL_SEED, "noise_offset": 100,
            "sampling_steps": 10, "assignment": ASSIGNMENT,
            "assignment_shas": {str(seed): digest(sorted(keys)) for seed, keys in plans.items()}}
    cache, receipts = {}, []
    timing.update(prediction_seconds=0., publish_seconds=0., verify_seconds=0., reused_groups=0, new_groups=0)
    for group in range(147):
        selected = [(s, k) for s, k in group_keys(data, group) if s == "validation" or k in train]
        if not selected:
            continue
        ledger.check()
        binding = {**base, "group": group}
        marker = directory/f"body-{group:05d}.json"
        if marker.exists():
            timing["reused_groups"] += 1
        else:
            require_io_room(ledger, read=3*CAP, write=CAP)
            start = time.monotonic()
            coords, masks = [], []
            for split, key in selected:
                ledger.check()
                clip = store.load(key, split)
                predicted = body.predict(clip.select_joints(cascade.body_idx), cascade._noise(key, joints, 100)).cpu()
                coords.append(predicted.numpy())
                masks.append(clip.time_valid.numpy())
            timing["prediction_seconds"] += time.monotonic()-start
            start = time.monotonic()
            with tempfile.TemporaryDirectory(dir=work, prefix="upper-") as tmp:
                path = Path(tmp)/f"body-{group:05d}-{uuid.uuid4().hex}.npz"
                np.savez_compressed(path, coordinates_mm=np.concatenate(coords), time_valid=np.concatenate(masks))
                package = publish_artifact(path, directory, ledger, owner, CAP)
                publish_json(marker, GROUP, {"binding": binding, "keys": [[s, list(k)] for s, k in selected],
                                            "artifact": package, "environment": environment}, owner)
            timing["publish_seconds"] += time.monotonic()-start
            timing["new_groups"] += 1
        start = time.monotonic()
        items, receipt = read_group(directory, marker, binding, selected, data, joints, ledger)
        cache.update(items)
        receipts.append(receipt["content_sha256"])
        timing["verify_seconds"] += time.monotonic()-start
    expected = {("train", k) for k in train} | {("validation", k) for k in data.clips["validation"]}
    if set(cache) != expected or len(data.clips["validation"]) != 421:
        raise ValueError("v220_body_prediction_cache_coverage")
    summary = {"binding": base, "group_receipts": receipts,
               "counts": {"train": len(train), "validation": 421},
               "prediction_tensor_bytes": len(cache)*data.width*joints*3*4}
    marker = directory/"index.json"
    if marker.exists():
        saved = read_sealed(marker, INDEX)
        if saved["content_sha256"] != digest({"schema": INDEX, **summary}):
            raise ValueError("v220_body_prediction_cache_index")
    else:
        saved = publish_json(marker, INDEX, summary, owner)
    return cache, saved
