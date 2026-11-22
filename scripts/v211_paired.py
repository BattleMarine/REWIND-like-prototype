"""Compare fixed v2.1.0 and v2.1.1 validation clips from saved predictions."""

import argparse
from contextlib import contextmanager, ExitStack
import csv
import hashlib
import io
import json
from pathlib import Path
import re
import time
import zipfile

import numpy as np

if __package__:
    from . import v210_tail as tail
else:
    import v210_tail as tail


SEED_SUMMARY_SHA = {
    42: "32d9f1f61e140003dea811663ba5e6bb6e46226159be2f4097f83e50dd0b3352",
    71: "e88911c078a1d6d0fec3db843114f08dc84c338592dfe81bfa3d437f46e0b4f3",
    103: "bb46ceeb2a545b4bf26f7b0c3838ceb4732ea2d7711ce0425b001ee191504507",
}
PLAN_SHA = "7753d66479f5603a63816159d64a57fe050a31fdd1e2dd2279fce449d2185968"
METRICS = tail.METRICS + ("hand_local",)
FILES = ("paired-clips.csv", "paired-summary.json")
COUNTS = {"clips": 421, "real_frames": 16425}
MIB = tail.MIB
INPUT_BUDGET = 512 * MIB
OUTPUT_CAP = 4 * MIB
WORKER_SECONDS = 240
HARD_CLIPS = {
    "ArchVisInterior_ArchVis_RT": (
        "ArchVisInterior_ArchVis_RT/Day/rp_janna_rigged_002_ue4/SKM_MenReadingGlasses_Shape_01/013/Push__5_", 50),
    "CityPark_Showcase": (
        "CityPark_Showcase/Day/rp_sophia_rigged_003_ue4/SKM_MenReadingGlasses_Shape_01/023/Tut_Hip_Hop_Dance__11_", 100),
    "DowntownWest_Demo_Environment": (
        "DowntownWest_Demo_Environment/Day/rp_rin_rigged_007_ue4/SKM_MenReadingGlasses_Shape_01/014/Walk_Backward_Arc__4_", 0),
}
SAME_ARRAYS = ("sequence_id", "clip_start", "frame_index", "seconds", "target_valid", "hand_local_valid",
               "target_device_mm", "hand_local_target_mm")


class Budget:
    def __init__(self):
        self.started = time.monotonic()
        self.bytes = 0

    def charge(self, count):
        self.bytes += count
        tail.require(self.bytes <= INPUT_BUDGET, "paired input read cap")
        if time.monotonic() - self.started >= WORKER_SECONDS - 10:
            raise TimeoutError("paired validation budget exhausted")


class Evidence:
    def __init__(self, path, budget, archive=None, archive_sha=None):
        self.path, self.budget, self.archive, self.archive_sha = path, budget, archive, archive_sha
        self.read_names = set()

    def read(self, name, cap, pin=None):
        tail.require(re.fullmatch(r"(seed-\d+/)?[A-Za-z0-9_.-]+", name), "invalid evidence name")
        self.read_names.add(name)
        if self.archive is None:
            path = self.path / name
            tail.require(path.resolve(strict=True) == path and path.is_file(), f"missing or linked evidence: {name}")
            size = path.stat().st_size
            source = path.open
        else:
            info = self.archive.getinfo(name)
            size = info.file_size
            source = lambda mode: self.archive.open(info, mode=mode)
        with source("r" if self.archive is not None else "rb") as stream:
            tail.require(0 < size <= cap, f"invalid evidence size: {name}")
            self.budget.charge(size)
            raw = stream.read(size + 1)
        tail.require(len(raw) == size and (pin is None or pin == {"bytes": size, "sha256": tail.sha(raw)}),
                     f"evidence SHA or size mismatch: {name}")
        return raw


@contextmanager
def open_evidence(path, budget, *, base=False):
    path = path.absolute()
    tail.require(path.resolve(strict=True) == path, "evidence root path or link")
    if path.is_dir():
        yield Evidence(path, budget)
        return
    cap = tail.INPUT_BYTES if base else 192 * MIB
    size = path.stat().st_size
    tail.require(path.is_file() and 0 < size <= cap and (not base or size == tail.INPUT_BYTES), "archive size")
    with path.open("rb") as stream:
        digest = hashlib.sha256()
        total = 0
        while block := stream.read(MIB):
            total += len(block)
            tail.require(total <= size, "archive growth")
            budget.charge(len(block))
            digest.update(block)
        archive_sha = digest.hexdigest()
        tail.require(total == size and (not base or archive_sha == tail.INPUT_SHA), "fixed base archive SHA")
        stream.seek(0)
        with zipfile.ZipFile(stream) as archive:
            infos = archive.infolist()
            names = {item.filename for item in infos}
            tail.require(len(infos) == len(names) == (37 if base else 111)
                         and sum(item.file_size for item in infos) <= cap
                         and all(not item.is_dir() and not item.flag_bits & 1 for item in infos),
                         "unexpected recovery ZIP roster (package only listed files)")
            reader = Evidence(path, budget, archive, archive_sha)
            yield reader
            tail.require(reader.read_names == names, "unconsumed recovery members")


def summary_for(source, base=None, seed=None):
    path = "summary.json" if seed is None else f"seed-{seed}/summary.json"
    raw = source.read(path, 64 * 1024)
    expected = tail.SUMMARY_SHA if seed is None else SEED_SUMMARY_SHA[seed]
    tail.require(tail.sha(raw) == expected, f"fixed summary file SHA mismatch: {path}")
    if seed is None:
        doc = tail.sealed(raw, "rewind_v210_completed_v1")
        tail.require(doc["counts"] == {**COUNTS, "groups": 18}
                     and len(doc["receipts"]) == len(doc["environments"]) == len(tail.GROUPS), "A validation scope")
        return doc
    doc = tail.sealed(raw, "rewind_v211_validation_v1")
    basis, row = doc["binding"], doc["measurement"]
    common, original = basis["comparison_common"], base["binding"]
    tail.require(doc["counts"] == COUNTS and len(doc["group_receipts"]) == len(tail.GROUPS)
                 and basis["method"] == row["method"] == "v2.1.1" and basis["seed"] == row["seed"] == seed
                 and basis["plan"] == row["trial_plan_sha256"] == PLAN_SHA
                 and row["updates"] == 3097 and row["status"] == "completed"
                 and basis["body_sha256"] == row["checkpoint_sha256"]
                 and basis["reference"] == original["checkpoint_receipt"]
                 and basis["source"] == original["source"] and basis["feature_identity"] == original["feature_identity"]
                 and basis["sampling_steps"] == original["sampling_steps"]
                 and basis["original_model"] == common["selected_model_sha256"] == original["weights"]
                 and common == row["comparison_common"] and common["split"] == "official_validation"
                 and common["validation_clips"] == 421 and common["validation_pairs"] == 16425
                 and common["resolution"] == [128, 128] and common["evaluation_noise_seed"] == original["eval_seed"]
                 and common["precision"] == original["precision"] and common["evaluation_upper"] == "predicted"
                 and common["input_2d"] == "predicted"
                 and common["parent"]["cache_boundary"] == original["cache_boundary"], "variant validation binding")
    return doc


def read_group(source, summary, index, group, seed=None):
    prefix = "" if seed is None else f"seed-{seed}/"
    marker = tail.sealed(source.read(f"{prefix}group-{group:05d}.json", 32 * 1024),
                         "rewind_v210_prediction_group_v1")
    expected = {**summary["binding"], "group": group}
    if seed is None:
        expected["feature_input"] = marker["binding"]["feature_input"]
        receipt = summary["receipts"][index]
        tail.require(marker["environment"] == summary["environments"][index], "A group environment")
    else:
        receipt = summary["group_receipts"][index]
    keys, frames = marker["keys"], marker["real_frames"]
    tail.require(marker["content_sha256"] == receipt and marker["binding"] == expected
                 and 1 <= len(keys) <= 24 and len(keys) <= frames <= 50 * len(keys), "group receipt/binding")
    artifact = marker["artifact"]
    tail.require(re.fullmatch(rf"group-{group:05d}-[0-9a-f]{{32}}\.npz", artifact["name"]), "group NPZ name")
    raw = source.read(prefix + artifact["name"], 16 * MIB,
                      {key: artifact[key] for key in ("bytes", "sha256")})
    arrays, names = tail.group_arrays(raw, frames)
    clips = {}
    offset = 0
    for sequence, start in keys:
        key = (sequence, start)
        matching = (arrays["sequence_id"] == sequence) & (arrays["clip_start"] == start)
        count = int(matching.sum())
        stop = offset + count
        expected_frames = np.arange(start, start + count, dtype=np.int64)
        tail.require(key not in clips and 1 <= count <= 50 and bool(matching[offset:stop].all())
                     and np.array_equal(arrays["frame_index"][offset:stop], expected_frames)
                     and np.array_equal(arrays["seconds"][offset:stop], expected_frames.astype(np.float64) / 25),
                     "clip order/real frame identity")
        clips[key] = {name: values[offset:stop] for name, values in arrays.items()}
        offset = stop
    tail.require(offset == frames, "unaccounted group's frames")
    return clips, names


def clip_metrics(key, frames, names):
    row = tail.clip_metrics(*key, frames, names)
    counts = [row[f"{side}_hand_local_valid_count"] for side in ("left", "right")]
    means = [row[f"{side}_hand_local_mpjpe_mm"] for side in ("left", "right")]
    total = sum(counts)
    row["hand_local_valid_count"] = total
    row["hand_local_mpjpe_mm"] = sum(n * m for n, m in zip(counts, means, strict=True) if n) / total if total else None
    return row


def paired_row(seed, row, changes):
    pair = {name: row[name] for name in ("scenario", "sequence_id", "clip_start", "real_frames")}
    pair.update(seed=seed, fixed_difficult_clip=(row["sequence_id"], row["clip_start"]) in HARD_CLIPS.values())
    for metric in METRICS:
        count, score = f"{metric}_valid_count", f"{metric}_mpjpe_mm"
        tail.require(row[count] == changes[count], "valid count changed after mask alignment")
        before, after = row[score], changes[score]
        pair.update({count: row[count], f"{metric}_before_mm": before, f"{metric}_after_mm": after,
                     f"{metric}_delta_mm": after - before if row[count] else None})
    return pair


def compare_clips(seed, parent, parent_metrics, variant, names):
    tail.require(parent.keys() == variant.keys(), f"seed {seed}: clip key set mismatch")
    pairs = []
    for key, before in parent.items():
        after = variant[key]
        for field in SAME_ARRAYS:
            tail.require(np.array_equal(before[field], after[field]), f"seed {seed}: {key} {field} mismatch")
        pairs.append(paired_row(seed, parent_metrics[key], clip_metrics(key, after, names)))
    return pairs


def describe(pairs):
    result = {"clips": len(pairs), "real_frames": sum(row["real_frames"] for row in pairs), "metrics": {}}
    for metric in METRICS:
        deltas = [row[f"{metric}_delta_mm"] for row in pairs if row[f"{metric}_delta_mm"] is not None]
        counts = [row[f"{metric}_valid_count"] for row in pairs if row[f"{metric}_delta_mm"] is not None]
        quantiles = np.percentile(deltas, (50, 90), method="linear").tolist() if deltas else [None, None]
        result["metrics"][metric] = {
            "eligible_clips": len(deltas), "valid_observations": sum(counts),
            "clip_equal_mean_delta_mm": float(np.mean(deltas)) if deltas else None,
            "clip_equal_p50_delta_mm": quantiles[0], "clip_equal_p90_delta_mm": quantiles[1],
            "observation_equal_delta_mm": float(np.average(deltas, weights=counts)) if deltas else None,
            "improved_clips": sum(delta < 0 for delta in deltas), "worsened_clips": sum(delta > 0 for delta in deltas),
            "tied_clips": sum(delta == 0 for delta in deltas),
        }
    return result


def analyze(base_path, variants_path):
    budget = Budget()
    with ExitStack() as opened:
        base = opened.enter_context(open_evidence(base_path, budget, base=True))
        variant = opened.enter_context(open_evidence(variants_path, budget))
        original = summary_for(base)
        seeds = {seed: summary_for(variant, original, seed) for seed in SEED_SUMMARY_SHA}
        parent_rows, all_pairs, seen_frames = {}, {seed: [] for seed in seeds}, set()
        for index, group in enumerate(tail.GROUPS):
            before, names = read_group(base, original, index, group)
            tail.require(parent_rows.keys().isdisjoint(before), "duplicate validation clip across groups")
            metrics = {key: clip_metrics(key, arrays, names) for key, arrays in before.items()}
            parent_rows.update(metrics)
            for (sequence, _), arrays in before.items():
                entries = {(sequence, int(frame)) for frame in arrays["frame_index"]}
                tail.require(seen_frames.isdisjoint(entries), "duplicate validation frame across clips")
                seen_frames.update(entries)
            for seed, summary in seeds.items():
                after, variant_names = read_group(variant, summary, index, group, seed)
                tail.require(variant_names == names, "joint names changed")
                all_pairs[seed].extend(compare_clips(seed, before, metrics, after, names))
        tail.require(len(parent_rows) == 421 and len(seen_frames) == 16425, "full official validation scope")
        for scene, (count, frames) in tail.EXPECTED.items():
            rows = [row for row in parent_rows.values() if row["scenario"] == scene]
            tail.require(len(rows) == count and sum(row["real_frames"] for row in rows) == frames
                         and tail.worst_global(rows) == parent_rows[HARD_CLIPS[scene]], "fixed baseline hard clip or scene count")
        reports, pairs = {}, []
        for seed, rows in all_pairs.items():
            ordered = sorted(rows, key=lambda row: (row["sequence_id"], row["clip_start"]))
            pairs.extend(ordered)
            reports[str(seed)] = {
                "summary_file_sha256": SEED_SUMMARY_SHA[seed], "summary_content_sha256": seeds[seed]["content_sha256"],
                "body_checkpoint_sha256": seeds[seed]["binding"]["body_sha256"],
                "official_validation_macro_mm": seeds[seed]["metrics"]["macro_score_mm"],
                "whole_validation": describe(ordered),
                "scenarios": {scene: describe([row for row in ordered if row["scenario"] == scene]) for scene in tail.EXPECTED},
                "baseline_fixed_difficult_clips": [row for row in ordered if row["fixed_difficult_clip"]],
            }
        return {"schema": "rewind_v211_paired_clips_v1", "counts_per_seed": COUNTS,
                "seed_training_values": list(seeds), "evaluation_noise_seed": original["binding"]["eval_seed"],
                "scope": "official_validation_only", "delta": "after_minus_before_mm",
                "inputs": {"base_summary_sha256": tail.SUMMARY_SHA, "base_archive_sha256": base.archive_sha,
                           "variants_archive_sha256": variant.archive_sha},
                "baseline_official_validation_macro_mm": original["metrics"]["macro_score_mm"],
                "paired_metrics": METRICS, "metric_policy": tail.POLICY,
                "fixed_baseline_keys": HARD_CLIPS, "seeds": reports}, pairs, budget.bytes


def write_outputs(output, report, pairs):
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=list(pairs[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(pairs)
    data = [buffer.getvalue().encode("utf-8"), tail.canonical(report) + b"\n"]
    tail.require(all(0 < len(raw) <= OUTPUT_CAP for raw in data), "paired result file size")
    output.mkdir()
    result = {}
    for name, raw in zip(FILES, data, strict=True):
        path = output / name
        with path.open("xb") as stream:
            stream.write(raw)
        tail.require(path.read_bytes() == raw, "paired output readback")
        result[name] = {"bytes": len(raw), "sha256": tail.sha(raw)}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("base", "variants", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    tail.require(not args.output.exists() and all(not args.output.resolve().is_relative_to(path.resolve())
                 for path in (args.base, args.variants)), "new output outside evidence")
    report, pairs, io_bytes = analyze(args.base, args.variants)
    pins = write_outputs(args.output, report, pairs)
    print(json.dumps({"status": "passed", "rows": len(pairs), "read_bytes": io_bytes, "files": pins}))


if __name__ == "__main__":
    main()
