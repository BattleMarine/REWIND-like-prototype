"""Compute clip-level errors from the fixed, recovered v2.1.0 archive."""

import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import re
import time
import zipfile

import numpy as np


INPUT_SHA = "d47a24ffb032b3de299396827755d840d0c9c59b5fc7a9105dbf406d4caf1f0f"
INPUT_BYTES = 37_004_795
SUMMARY_SHA = "796527609be8733fd44341e564641ff3b83314199fc55c7912418b81c2931da4"
JOINT_NAMES_SHA = "e86b5a9b04af4b492067b7063c0ab886046d276e8db54146895ded688d403dd2"
FILES = ("clip-errors.csv", "clip-errors.json")
GROUPS = tuple(range(129, 147))
EXPECTED = {"ArchVisInterior_ArchVis_RT": (59, 2236),
            "CityPark_Showcase": (248, 9620),
            "DowntownWest_Demo_Environment": (114, 4569)}
MIB = 1024**2
WORKER_SECONDS = 180
OUTPUT_CAP = 2 * MIB
METRICS = tuple(f"{side}_{group}" for side in ("left", "right")
                for group in ("wrist", "hand_local", "hand_global")) + ("hand_global",)
POLICY = {
    "units": "device_mm",
    "percentile": "linear interpolation (NumPy default), equal weight per eligible clip within scenario",
    "wrist_validity": "valid wrist for that side",
    "hand_local_validity": "valid same-side wrist and finger",
    "hand_global_validity": "valid finger, independently of wrist",
    "combined_hand_global": "pool valid left and right finger observations within a clip",
    "worst_one_per_scenario": "maximum combined hand_global MPJPE; tie: sequence_id ascending, clip_start ascending",
    "empty_metric": "null in JSON, empty in CSV; excluded from quantiles and selection",
}
ARRAYS = {
    "sequence_id": ((), np.str_), "clip_start": ((), np.int64), "frame_index": ((), np.int64),
    "seconds": ((), np.float64), "pred_device_mm": ((62, 3), np.float32),
    "target_device_mm": ((62, 3), np.float32), "target_valid": ((62,), np.bool_),
    "hand_local_pred_mm": ((40, 3), np.float32), "hand_local_target_mm": ((40, 3), np.float32),
    "hand_local_valid": ((40,), np.bool_),
}


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sealed(raw, schema):
    doc = json.loads(raw)
    body = {key: value for key, value in doc.items() if key != "content_sha256"}
    require(doc["schema"] == schema and doc["content_sha256"] == sha(canonical(body)), f"{schema}: invalid seal")
    return doc


def bounded_names(archive, count, byte_cap):
    infos = archive.infolist()
    names = {info.filename: info.file_size for info in infos}
    require(len(infos) == len(names) == count and sum(names.values()) <= byte_cap
            and all(not info.is_dir() and info.file_size > 0 and not info.flag_bits & 1
                    and re.fullmatch(r"[A-Za-z0-9_.-]+", info.filename) for info in infos), "invalid ZIP listing")
    return names


def group_arrays(raw, count):
    fields = ARRAYS.keys() | {"joint_names"}
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        require(bounded_names(archive, len(fields), 8 * MIB).keys() == {f"{key}.npy" for key in fields},
                "NPZ fields mismatch")
    with np.load(io.BytesIO(raw), allow_pickle=False) as archive:
        names = archive["joint_names"]
        require(names.shape == (62,) and names.dtype.type is np.str_
                and sha(canonical(names.tolist())) == JOINT_NAMES_SHA, "joint names/order mismatch")
        values = {key: archive[key] for key in ARRAYS}
    for key, (shape, dtype) in ARRAYS.items():
        value = values[key]
        require(value.shape == (count, *shape) and value.dtype.type is dtype
                and (dtype is np.str_ or bool(np.isfinite(value).all())), f"invalid NPZ array: {key}")
    return values, names.tolist()


def distances(predicted, target, valid):
    difference = predicted[valid].astype(np.float64) - target[valid].astype(np.float64)
    return np.linalg.norm(difference, axis=-1)


def clip_metrics(sequence, start, frames, names):
    pred, target, valid = (frames[key] for key in ("pred_device_mm", "target_device_mm", "target_valid"))
    local_pred, local_target, local_valid = (frames[key] for key in
                                            ("hand_local_pred_mm", "hand_local_target_mm", "hand_local_valid"))
    row = {"scenario": sequence.split("/", 1)[0], "sequence_id": sequence, "clip_start": start,
           "real_frames": len(pred)}
    all_global = []

    def add(name, values):
        row[f"{name}_valid_count"] = int(values.size)
        row[f"{name}_mpjpe_mm"] = float(values.mean()) if values.size else None

    for side, suffix, offset in (("left", "l", 0), ("right", "r", 20)):
        wrist = names.index(f"hand_{suffix}")
        fingers = slice(22 + offset, 42 + offset)
        relative = slice(offset, offset + 20)
        mask = valid[:, fingers]
        relative_valid = local_valid[:, relative]
        require(np.array_equal(relative_valid, mask & valid[:, wrist, None]), "relative mask differs from finger ∩ wrist")
        hand = distances(pred[:, fingers], target[:, fingers], mask)
        add(f"{side}_wrist", distances(pred[:, wrist], target[:, wrist], valid[:, wrist]))
        add(f"{side}_hand_local", distances(local_pred[:, relative], local_target[:, relative], relative_valid))
        add(f"{side}_hand_global", hand)
        all_global.append(hand)
    add("hand_global", np.concatenate(all_global))
    return row


def read_group(archive, listing, summary, index, group):
    marker_name = f"group-{group:05d}.json"
    require(listing[marker_name] <= 32 * 1024, "group marker too large")
    marker = sealed(archive.read(marker_name), "rewind_v210_prediction_group_v1")
    own_fields = {"group", "feature_input"}
    binding = {key: value for key, value in marker["binding"].items() if key not in own_fields}
    keys, count = marker["keys"], marker["real_frames"]
    require(binding == summary["binding"] and marker["binding"]["group"] == group
            and "feature_input" in marker["binding"]
            and marker["environment"] == summary["environments"][index]
            and marker["content_sha256"] == summary["receipts"][index]
            and 1 <= len(keys) <= 24 and len(keys) <= count <= 50 * len(keys), "group identity/scope mismatch")
    pin = marker["artifact"]
    name = pin["name"]
    require(re.fullmatch(rf"group-{group:05d}-[0-9a-f]{{32}}\.npz", name)
            and 0 < pin["bytes"] == listing[name] <= 8 * MIB, "group artifact name/size")
    raw = archive.read(name)
    require(sha(raw) == pin["sha256"], "group NPZ SHA mismatch")
    return keys, count, name, group_arrays(raw, count)


def analyze(input_zip):
    started = time.monotonic()
    require(input_zip.is_file() and not input_zip.is_symlink() and input_zip.stat().st_size == INPUT_BYTES,
            "expected the fixed input ZIP and size")
    rows, seen_clips, seen_frames = [], set(), set()
    with input_zip.open("rb") as stream:
        digest = hashlib.sha256()
        verified_bytes = 0
        for block in iter(lambda: stream.read(MIB), b""):
            verified_bytes += len(block)
            require(verified_bytes <= INPUT_BYTES, "input ZIP grew during verification")
            digest.update(block)
        require(verified_bytes == INPUT_BYTES and digest.hexdigest() == INPUT_SHA, "recovered input ZIP SHA mismatch")
        stream.seek(0)
        with zipfile.ZipFile(stream) as archive:
            listing = bounded_names(archive, 2 * len(GROUPS) + 1, 64 * MIB)
            require(listing["summary.json"] <= 64 * 1024, "summary too large")
            raw = archive.read("summary.json")
            require(sha(raw) == SUMMARY_SHA, "fixed GPU summary file SHA mismatch")
            summary = sealed(raw, "rewind_v210_completed_v1")
            require(summary["counts"] == {"groups": 18, "clips": 421, "real_frames": 16425}
                    and len(summary["receipts"]) == len(summary["environments"]) == len(GROUPS), "summary coverage")
            used = {"summary.json"}
            for index, group in enumerate(GROUPS):
                if time.monotonic() - started >= WORKER_SECONDS - 10:
                    raise TimeoutError("clip metrics budget")
                keys, count, name, (arrays, names) = read_group(archive, listing, summary, index, group)
                used.update((f"group-{group:05d}.json", name))
                offset = 0
                for sequence, start in keys:
                    key = (sequence, start)
                    require(key not in seen_clips and sequence.split("/", 1)[0] in EXPECTED, "duplicate/out-of-split clip")
                    seen_clips.add(key)
                    paired = (arrays["sequence_id"] == sequence) & (arrays["clip_start"] == start)
                    size = int(paired.sum())
                    stop = offset + size
                    require(1 <= size <= 50 and bool(paired[offset:stop].all()), "clip segment order or size")
                    segment = {key: value[offset:stop] for key, value in arrays.items()}
                    expected = np.arange(start, start + size, dtype=np.int64)
                    require(np.array_equal(segment["frame_index"], expected)
                            and np.array_equal(segment["seconds"], expected.astype(np.float64) / 25), "frame/time key mismatch")
                    frame_keys = {(sequence, int(frame)) for frame in expected}
                    require(seen_frames.isdisjoint(frame_keys), "duplicate sequence frame")
                    seen_frames.update(frame_keys)
                    rows.append(clip_metrics(sequence, start, segment, names))
                    offset = stop
                require(offset == count, "group has unaccounted frame rows")
            require(used == listing.keys() and len(rows) == 421 and len(seen_frames) == 16425,
                    "final source coverage mismatch")
    rows.sort(key=lambda row: (row["scenario"], row["sequence_id"], row["clip_start"]))
    return {"schema": "rewind_v210_clip_errors_v1",
            "input": {"bytes": INPUT_BYTES, "sha256": INPUT_SHA, "summary_file_sha256": SUMMARY_SHA,
                      "summary_content_sha256": summary["content_sha256"], "binding": summary["binding"]},
            "counts": summary["counts"], "metric_policy": POLICY, "scenarios": summarize(rows), "clips": rows,
            "official_selection_macro_mm": summary["metrics"]["macro_score_mm"]}


def summarize(rows):
    result = {}
    for scenario, (clip_count, frame_count) in EXPECTED.items():
        group = [row for row in rows if row["scenario"] == scenario]
        require(len(group) == clip_count and sum(row["real_frames"] for row in group) == frame_count,
                "scenario clip/frame scope")
        quantiles = {}
        for name in METRICS:
            eligible = [row[f"{name}_mpjpe_mm"] for row in group if row[f"{name}_mpjpe_mm"] is not None]
            p50, p90 = np.percentile(eligible, (50, 90), method="linear").tolist() if eligible else (None, None)
            quantiles[name] = {"eligible_clips": len(eligible), "p50_mm": p50, "p90_mm": p90,
                               "valid_observations": sum(row[f"{name}_valid_count"] for row in group)}
        worst = worst_global(group)
        result[scenario] = {"clips": len(group), "real_frames": frame_count,
                            "clip_equal_quantiles": quantiles, "worst_hand_global_clip": worst}
    return result


def worst_global(group):
    eligible = (row for row in group if row["hand_global_mpjpe_mm"] is not None)
    return min(eligible, key=lambda row: (-row["hand_global_mpjpe_mm"], row["sequence_id"], row["clip_start"]), default=None)


def write_outputs(report, output):
    columns = ["scenario", "sequence_id", "clip_start", "real_frames"] + [
        f"{name}_{suffix}" for name in METRICS for suffix in ("mpjpe_mm", "valid_count")]
    text = io.StringIO(newline="")
    writer = csv.DictWriter(text, fieldnames=columns, lineterminator="\n")
    writer.writeheader()
    writer.writerows(report["clips"])
    files = (text.getvalue().encode("utf-8"), canonical(report) + b"\n")
    output.mkdir()
    pins = {}
    for name, data in zip(FILES, files, strict=True):
        require(0 < len(data) <= OUTPUT_CAP, f"output byte cap: {name}")
        path = output / name
        with path.open("xb") as stream:
            stream.write(data)
        require(path.read_bytes() == data, f"output readback: {name}")
        pins[name] = {"bytes": len(data), "sha256": sha(data)}
    return pins


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    require(not args.output.exists() and not args.output.is_symlink(), "output directory must be new")
    started = time.monotonic()
    report = analyze(args.input)
    pins = write_outputs(report, args.output)
    print(json.dumps({"status": "passed", "elapsed_seconds": time.monotonic() - started,
                      "counts": report["counts"], "files": pins}))


if __name__ == "__main__":
    main()
