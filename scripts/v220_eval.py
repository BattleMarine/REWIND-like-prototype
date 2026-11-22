"""Official validation and matched-seed hand comparisons using cached A body."""

import tempfile
import time
import uuid
from pathlib import Path
from statistics import mean, stdev
from types import SimpleNamespace

import numpy as np
from rewind_archive.contracts import digest
from scripts.v210_eval import CAP, GROUP_SCHEMA, group_keys, read_group
from scripts.v210_metrics import aggregate, rows
from scripts.v211_progress import publish_artifact, require_io_room
from scripts.v211_trial import SEEDS
from scripts.v220_upper import CONTROL, MIXED, cached_body
from scripts.v2_gpu_training_storage import publish_json, read_sealed


SUMMARY = "rewind_v220_hand_validation_v1"
COMPARISON = "rewind_v220_paired_comparison_v1"
COUNTS = {"clips": 421, "real_frames": 16425}


def validated_rows(cascade, frozen_body_pin, hand, store, key, cache, ledger):
    clip = store.load(key, "validation")
    body = cached_body(cache, "validation", key, clip)
    stage = SimpleNamespace(checkpoint_sha256=frozen_body_pin, predict=lambda *_: body)
    return rows(cascade, stage, hand, clip, key, ledger)


def evaluate(cascade, hand, store, cache, target, work, basis, fit, reference, ledger, owner, environment, timing):
    names = cascade.body_names + cascade.fingers
    data = store.data
    receipts, stream = [], []
    timing.update(validation_seconds=0., validation_publish_seconds=0., validation_verify_seconds=0.)
    for group in range(129, 147):
        ledger.check()
        keys = [key for split, key in group_keys(data, group) if split == "validation"]
        marker = target/f"group-{group:05d}.json"
        binding = {**basis, "group": group}
        if not marker.exists():
            require_io_room(ledger, read=3*CAP, write=CAP)
            start = time.monotonic()
            batches = [validated_rows(cascade, basis["body"], hand, store, key, cache, ledger) for key in keys]
            combined = {name: np.concatenate([batch[name] for batch in batches]) for name in batches[0]}
            timing["validation_seconds"] += time.monotonic()-start
            start = time.monotonic()
            with tempfile.TemporaryDirectory(dir=work, prefix="hand-eval-") as tmp:
                path = Path(tmp)/f"group-{group:05d}-{uuid.uuid4().hex}.npz"
                np.savez_compressed(path, **combined, joint_names=np.asarray(names))
                package = publish_artifact(path, target, ledger, owner, CAP)
                publish_json(marker, GROUP_SCHEMA, {"binding": binding, "artifact": package, "keys": [list(k) for k in keys],
                                                    "real_frames": len(combined["frame_index"]), "environment": environment}, owner)
            timing["validation_publish_seconds"] += time.monotonic()-start
        start = time.monotonic()
        items, receipt = read_group(target, marker, binding, keys, data, names, ledger)
        stream.extend(items)
        receipts.append(receipt["content_sha256"])
        timing["validation_verify_seconds"] += time.monotonic()-start
    if ([key for key, _ in stream] != data.clips["validation"] or len(stream) != COUNTS["clips"]
            or len({(key[0], int(frame)) for key, row in stream for frame in row["frame_index"]}) != COUNTS["real_frames"]):
        raise ValueError("v220_validation_coverage")
    metrics = aggregate(stream, data.spec, reference)
    return publish_json(target/"summary.json", SUMMARY, {"binding": basis, "counts": COUNTS, "group_receipts": receipts,
                        "hand_checkpoint_sha256": fit["artifact"]["sha256"], "upper_counts": fit["upper_counts"],
                        "metrics": metrics}, owner)


def trial_scores(doc):
    metrics = doc["metrics"]
    official = {name: value["official"] for name, value in metrics["per_scenario"].items()}
    summary = {f"{metric}_macro_mm": mean(s[metric]["mpjpe_mm"] for s in official.values())
               for metric in ("hand_local", "hand_global")}
    summary["body_hand_global_macro_mm"] = metrics["macro_score_mm"]
    return summary, official


def paired(path, trials, cache_sha, owner):
    pairs = []
    for seed in SEEDS:
        left, right = trials[CONTROL, seed], trials[MIXED, seed]
        control, a = trial_scores(left)
        mixed, b = trial_scores(right)
        if a.keys() != b.keys() or any(a[name]["body"] != b[name]["body"] for name in a):
            raise ValueError("v220_paired_frozen_body_mismatch")
        scenario_delta = {name: {metric: b[name][metric]["mpjpe_mm"]-a[name][metric]["mpjpe_mm"]
                                 for metric in ("hand_local", "hand_global")} for name in a}
        pairs.append({"seed": seed, "control": control, "mixed": mixed, "per_scenario_delta_mm": scenario_delta,
                      "delta_mm": {key: mixed[key]-control[key] for key in control},
                      "summary_sha256": {CONTROL: left["content_sha256"], MIXED: right["content_sha256"]},
                      "hand_checkpoint_sha256": {CONTROL: left["hand_checkpoint_sha256"], MIXED: right["hand_checkpoint_sha256"]}})
    keys = pairs[0]["delta_mm"]
    result = {"predicted_body_cache_sha256": cache_sha, "pairs": pairs,
              "mean_delta_mm": {key: mean(p["delta_mm"][key] for p in pairs) for key in keys},
              "sample_stdev_delta_mm": {key: stdev(p["delta_mm"][key] for p in pairs) for key in keys}}
    if path.exists():
        saved = read_sealed(path, COMPARISON)
        if saved["content_sha256"] != digest({"schema": COMPARISON, **result}):
            raise ValueError("v220_existing_paired_comparison")
        return saved
    return publish_json(path, COMPARISON, result, owner)
