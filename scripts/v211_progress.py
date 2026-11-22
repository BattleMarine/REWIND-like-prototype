"""Single-writer publication and resume for v2.1.1 fit and validation."""

import hashlib
from pathlib import Path
import tempfile
import time
import uuid

import numpy as np
from rewind_archive.contracts import digest
from rewind_m1.local_rgb import _regular_read
from rewind_prepare.temporal_cache import _hash_file
from rewind_v1 import v2_retention as retention
from scripts.v210_eval import CAP, GROUP_SCHEMA, group_keys, read_group
from scripts.v210_metrics import aggregate, rows
from scripts.v211_trial import UPDATES, backbone_sha, begin_fit, load_state, measurement, save_state, update
from scripts.v2_gpu_training_storage import MIB, publish_json, read_sealed


FIT = "rewind_v211_fit_receipt_v1"
SUMMARY = "rewind_v211_validation_v1"
INTERVAL = 50


def folder(path):
    if path.is_symlink():
        raise ValueError(f"v211_directory_symlink: {path.name}")
    path.mkdir(exist_ok=True)
    if path.resolve(strict=True) != path:
        raise ValueError(f"v211_directory_path: {path.name}")
    return path


def publish_artifact(local, target, ledger, owner, cap):
    package = {"name": local.name, "bytes": local.stat().st_size, "sha256": _hash_file(local)}
    if not 0 < package["bytes"] <= cap:
        raise ValueError("v211_artifact_size")
    owner()
    retention.copy_verified(local, target/local.name, package, ledger, ledger.clock()+min(120, ledger.remaining(terminal=True)))
    return package


def fit_tip(target, binding):
    previous = None
    for marker in sorted(target.glob("fit-*.json")):
        item = read_sealed(marker, FIT)
        step = item["step"]
        if (type(step) is not int or not (previous["step"] if previous else 0) < step <= UPDATES
                or item["binding_sha256"] != digest(binding) or marker.name != f"fit-{step:05d}.json"
                or item["previous"] != (previous["content_sha256"] if previous else None)):
            raise ValueError("v211_fit_history")
        previous = item
    return previous


def checkpoint_raw(target, receipt, ledger):
    package = receipt["artifact"]
    name = package["name"]
    path = target/name
    if Path(name).name != name or not 0 < package["bytes"] <= MIB or path.resolve(strict=True) != path:
        raise ValueError("v211_fit_path")
    ledger.reserve("read", package["bytes"])
    raw = _regular_read(path, MIB, package["bytes"])
    ledger.observed("read", len(raw), persist=False)
    if hashlib.sha256(raw).hexdigest() != package["sha256"]:
        raise ValueError("v211_fit_sha256")
    return raw


def fit(stage, ordered, target, work, binding, store, body_indices, ledger, owner, environment, timing):
    start = time.monotonic()
    previous = fit_tip(target, binding)
    begin_fit(stage)
    expected_backbone = backbone_sha(stage)
    last_loss = None
    if previous:
        if previous["backbone_sha256"] != expected_backbone:
            raise ValueError("v211_parent_backbone_sha256")
        last_loss = load_state(stage, binding, checkpoint_raw(target, previous, ledger), previous["step"])
    timing.update(resumed_from=stage.step, training_seconds=0.0, fit_publish_seconds=0.0,
                  fit_prepare_seconds=time.monotonic()-start)

    def save():
        nonlocal previous
        start = time.monotonic()
        if stage.step <= (previous["step"] if previous else 0):
            return
        with tempfile.TemporaryDirectory(dir=work, prefix="fit-") as scratch:
            artifact = Path(scratch)/f"head-{stage.step:05d}-{uuid.uuid4().hex}.pt"
            save_state(stage, binding, artifact, last_loss)
            package = publish_artifact(artifact, target, ledger, owner, MIB)
            previous = publish_json(target/f"fit-{stage.step:05d}.json", FIT,
                                    {"step": stage.step, "binding_sha256": digest(binding), "artifact": package,
                                     "backbone_sha256": expected_backbone, "last_loss": last_loss,
                                     "previous": previous["content_sha256"] if previous else None,
                                     "environment": environment}, owner)
        timing["fit_publish_seconds"] += time.monotonic()-start
        print(f"v2.1.1 seed {binding['seed']} fit {stage.step}/{UPDATES}", flush=True)

    while stage.step < UPDATES:
        start = time.monotonic()
        try:
            ledger.check()
            clip = store.load(ordered[stage.step], "train").select_joints(body_indices)
        except TimeoutError:
            # The previous optimizer step is complete. Use the terminal reserve.
            save()
            raise
        last_loss = update(stage, clip, binding["seed"])
        timing["training_seconds"] += time.monotonic()-start
        if stage.step == 1 or stage.step % INTERVAL == 0 or stage.step == UPDATES:
            if stage.step == UPDATES and backbone_sha(stage) != expected_backbone:
                raise ValueError("v211_frozen_backbone_changed")
            save()
    stage.optimizer.zero_grad(set_to_none=True)
    stage.model.eval().requires_grad_(False)
    stage.checkpoint_sha256 = previous["artifact"]["sha256"]
    return previous


def evaluate(cascade, body, hand, store, target, work, basis, plan, reference, ledger, owner, environment, timing):
    data, names = store.data, cascade.body_names + cascade.fingers
    receipts, stream = [], []
    timing.update(validation_seconds=0.0, validation_publish_seconds=0.0, validation_verify_seconds=0.0)
    for group in range(129, 147):
        ledger.check()
        keys = [key for split, key in group_keys(data, group) if split == "validation"]
        marker = target/f"group-{group:05d}.json"
        binding = {**basis, "group": group}
        if not marker.exists():
            with tempfile.TemporaryDirectory(dir=work, prefix="eval-") as scratch:
                start = time.monotonic()
                batches = [rows(cascade, body, hand, store.load(key, "validation"), key, ledger) for key in keys]
                combined = {name: np.concatenate([row[name] for row in batches]) for name in batches[0]}
                timing["validation_seconds"] += time.monotonic()-start
                start = time.monotonic()
                path = Path(scratch)/f"group-{group:05d}-{uuid.uuid4().hex}.npz"
                np.savez_compressed(path, **combined, joint_names=np.asarray(names))
                artifact = publish_artifact(path, target, ledger, owner, CAP)
                publish_json(marker, GROUP_SCHEMA, {"binding": binding, "artifact": artifact,
                             "keys": [list(key) for key in keys], "real_frames": len(combined["frame_index"]),
                             "environment": environment}, owner)
                timing["validation_publish_seconds"] += time.monotonic()-start
        start = time.monotonic()
        parts, receipt = read_group(target, marker, binding, keys, data, names, ledger)
        stream.extend(parts)
        receipts.append(receipt["content_sha256"])
        timing["validation_verify_seconds"] += time.monotonic()-start
    if ([part[0] for part in stream] != data.clips["validation"] or len(stream) != 421
            or len({(key[0], int(frame)) for key, row in stream for frame in row["frame_index"]}) != 16425):
        raise ValueError("v211_validation_coverage")
    metrics = aggregate(stream, data.spec, reference)
    row = measurement(plan, basis["seed"], body.checkpoint_sha256, metrics)
    final = {"binding": basis, "counts": {"clips": 421, "real_frames": 16425}, "group_receipts": receipts,
             "metrics": metrics, "measurement": row}
    marker = target/"summary.json"
    if marker.exists():
        saved = read_sealed(marker, SUMMARY)
        if saved["content_sha256"] != digest({"schema": SUMMARY, **final}):
            raise ValueError("v211_summary_pin")
    else:
        saved = publish_json(marker, SUMMARY, final, owner)
    return saved
