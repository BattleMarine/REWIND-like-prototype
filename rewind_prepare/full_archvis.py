"""Frozen ArchVis v1.2 inventory planning and bounded direct-file audit."""

import argparse
from collections import Counter, defaultdict
from contextlib import suppress
from pathlib import Path
import shutil
import sys
import time
import zipfile

from rewind_archive.contracts import canonical, digest
from rewind_archive.matching import contiguous_runs
from rewind_data.audit import read_json
from rewind_m1.local_rgb import _regular_read
from .converter import motion_entries, official_splits, sha
from .extracted import (
    MAX_MANIFEST_BYTES, MIB, ROLES, SCHEMA, SPLITS, ExtractedLimits,
    _Budget, _inventory, _local, prepare_extracted_t1,
)
from .spec import validate_spec
from .temporal_cache import _atomic, _locked
from .temporal_selection import BLOCKERS, CLIP, check_pin, seal


ENVIRONMENT = "ArchVisInterior_ArchVis_RT"
COUNTS = {"train": 20_888, "validation": 2_236}
INPUT_RECEIPT = "cab7d398a64aece59485580554b108bdc9abe9f732089d4c4c1ac1e480b4a53d"
INPUT_ZIP = "6256eb7bd148ec51bdb94627c68e54657be3808c607a12e13b68f7e479b15e97"
VALIDATION_KEYS = "b4c52d654e772771e553ce887f81f502ea80082e825bab63d02f7c6088ba2dee"
SNAPSHOT = "archvis_v12_frozen_inventory_v1"
SOURCE_PLAN = "direct_extracted_selection_plan_v1"
STAT_LIMITS = ExtractedLimits(16 * MIB, 120 * 60)
AUDIT_LIMITS = ExtractedLimits(65 * 1024 * MIB, 6 * 3600)
REPORT_LIMIT = 512 * MIB
FREE_RESERVE = 1024 * MIB
RECOVERY_FILES = ("snapshot.json", "coverage.json", "preflight.json", "source_plan.json",
                  "inventory_check.json", "audit_inventory_check.json", "preflight_status.json",
                  "audit_status.json", "notebook_status.json",
                  "verified/manifest.json")


def _read(path, maximum=MAX_MANIFEST_BYTES):
    return _regular_read(_local(path), maximum)


def _write(path, doc):
    raw = canonical(doc) + b"\n"
    if len(raw) > MAX_MANIFEST_BYTES:
        raise ValueError("v12_report_file_limit")
    _atomic(path, raw)


def clip_coverage(keys_by_split):
    """Tile contiguous runs exactly once, including a post-padded final clip."""
    rows, summary = [], {}
    for split, keys in keys_by_split.items():
        frames = defaultdict(list)
        for sequence, frame in keys:
            frames[sequence].append(frame)
        counts = Counter(dict.fromkeys(("runs", "frame_number_gaps", "interior_missing_frame_numbers",
                                        "complete_clips", "complete_frames", "short_runs",
                                        "short_run_frames", "tail_clips", "tail_frames", "padding_slots"), 0))
        for sequence, values in sorted(frames.items()):
            runs = contiguous_runs(values)
            counts["frame_number_gaps"] += len(runs) - 1
            counts["interior_missing_frame_numbers"] += runs[-1][1] - runs[0][0] + 1 - len(values)
            for first, last in runs:
                counts["runs"] += 1
                for start in range(first, last + 1, CLIP):
                    valid = min(CLIP, last - start + 1)
                    kind = "complete" if valid == CLIP else "short_run" if start == first else "tail"
                    counts[{"complete": "complete_clips", "short_run": "short_runs", "tail": "tail_clips"}[kind]] += 1
                    counts[kind + "_frames"] += valid
                    counts["padding_slots"] += CLIP - valid
                    rows.append({"split": split, "sequence": sequence, "run_start": first,
                                 "start_frame": start, "real_frames": valid, "kind": kind})
        covered = counts["complete_frames"] + counts["short_run_frames"] + counts["tail_frames"]
        summary[split] = {**counts, "sequences": len(frames), "eligible_frames": len(keys),
                          "strict_omitted_frames": covered - counts["complete_frames"],
                          "padded_real_frames": covered, "repeated_real_frames": 0,
                          "padded_clips": counts["complete_clips"] + counts["short_runs"] + counts["tail_clips"]}
    return summary, rows


def frozen_snapshot(inputs):
    receipt_raw = _read(inputs / "input_receipt.json", MIB)
    if sha(receipt_raw) != INPUT_RECEIPT:
        raise ValueError("v12_input_receipt_pin")
    input_receipt = read_json(receipt_raw)
    receipt = input_receipt["files"]
    assets = {}
    for name in ("previous_manifest.json", "inventory.json", "spec.json", "calibration.json",
                 "motion_map.json", "selection.json", "split/train.txt", "split/validation.txt", "split/test.txt"):
        raw = _read(inputs / name, 8 * MIB)
        if receipt[name] != {"sha256": sha(raw), "bytes": len(raw)}:
            raise ValueError("v12_input_asset_pin:" + name)
        assets[name] = raw
    spec = validate_spec(read_json(assets["spec.json"]))
    previous = read_json(assets["previous_manifest.json"])
    selection = read_json(assets["selection.json"])
    check_pin(selection, "archvis_temporal_selection_v1", input_receipt["selection_sha256"])
    check_pin(previous, SCHEMA, selection["prior_manifest_sha256"])
    provenance = previous["provenance"]
    for name, key in (("spec.json", "spec_sha256"), ("calibration.json", "calibration_sha256"),
                      ("motion_map.json", "motion_map_sha256"), ("inventory.json", "inventory_sha256")):
        if receipt[name]["sha256"] != provenance[key]:
            raise ValueError("v12_previous_asset_pin:" + name)
    budget = _Budget(STAT_LIMITS)
    owners, splits = official_splits(_local(inputs / "split", directory=True), spec, budget)
    if splits != provenance["split_files"] or previous["environment"] != ENVIRONMENT:
        raise ValueError("v12_previous_split_pin")
    available, duplicates, _ = _inventory(inputs / "inventory.json", provenance["inventory_sha256"],
                                          owners, ENVIRONMENT, budget)
    eligible = available - duplicates
    keys = {split: sorted(key for key in eligible if owners[key[0]] == split) for split in SPLITS}
    prior = {(row["sequence_id"], row["frame_index"]): row for row in previous["records"]}
    prior_validation = sorted(key for key, row in prior.items() if row["split"] == "validation")
    if (len(prior) != len(previous["records"]) or not prior.keys() <= eligible
            or any(owners[key[0]] != row["split"] for key, row in prior.items())
            or prior_validation != keys["validation"] or digest(prior_validation) != VALIDATION_KEYS
            or {split: len(values) for split, values in keys.items()} != COUNTS or len(duplicates) != 1):
        raise ValueError("v12_frozen_inventory_scope")
    sequences = {key[0] for key in eligible}
    motions = motion_entries(read_json(assets["motion_map.json"]), sequences, owners)
    sizes = Counter()
    counts = Counter()
    for row in prior.values():
        sizes[row["split"]] += sum(entry["size_bytes"] for entry in row["files"].values())
        counts[row["split"]] += 1
    estimate_train = round(sizes["train"] * len(keys["train"]) / counts["train"])
    coverage, clips = clip_coverage(keys)
    snapshot = seal({
        "schema": SNAPSHOT, "status": "inventory_only", "environment": ENVIRONMENT,
        "dataset_version": previous["dataset_version"], "counts": COUNTS,
        "train_ready": False, "temporal_ready": False, "blockers": list(BLOCKERS),
        "scope": "Frozen 2026-11-09 inventory keys. New uploads require a new reviewed snapshot.",
        "payload_validation": "not_run", "test_payload_reads": 0, "input_receipt_sha256": INPUT_RECEIPT,
        "input_zip_sha256": INPUT_ZIP, "inventory_sha256": provenance["inventory_sha256"],
        "previous_manifest_sha256": previous["content_sha256"], "validation_keys_sha256": VALIDATION_KEYS,
        "split_files": splits, "keys": keys, "excluded_duplicates": sorted(duplicates),
        "missing_official_archvis_sequences": {split: sorted(seq for seq, owner in owners.items()
            if owner == split and seq.startswith(ENVIRONMENT + "/") and seq not in sequences) for split in SPLITS},
        "motion_identity_status": dict(Counter(row["identity_status"] for row in motions.values())),
        "coverage": coverage,
        "cost_estimate": {"source_files": sum(COUNTS.values()) * len(ROLES),
                          "png_decodes": 2 * sum(COUNTS.values()), "pose_checks": sum(COUNTS.values()),
                          "previous_counts": dict(counts), "previous_bytes": dict(sizes),
                          "train_payload_bytes_extrapolated": estimate_train,
                          "validation_payload_bytes_previously_measured": sizes["validation"],
                          "total_payload_bytes_estimated": estimate_train + sizes["validation"],
                          "scope": "Prior train mean extrapolated and prior validation sizes. Raw payload only."},
    })
    clip_plan = seal({
        "schema": "archvis_v12_clip_coverage_plan_v1", "status": "frame_number_plan_only",
        "snapshot_sha256": snapshot["content_sha256"], "clip_width": CLIP, "summary": coverage,
        "frame_mask": "true for slots [0, real_frames), false for later slots",
        "padding": "right-pad with zero placeholders and exclude padded queries, keys, losses and metrics",
        "history": "reset at every clip start; never bridge gaps or sequences",
        "train_ready": False, "temporal_ready": False, "clips": clips,
    })
    return snapshot, clip_plan, prior


def _run_kwargs(inputs, snapshot, rgb_root, pose_root, output):
    return dict(rgb_root=rgb_root, pose_root=pose_root, output_dir=output, environment=ENVIRONMENT,
                dataset_version=snapshot["dataset_version"], split_dir=inputs / "split",
                motion_map=inputs / "motion_map.json", spec_path=inputs / "spec.json",
                calibration_path=inputs / "calibration.json", train_limit=None, seed=71,
                inventory=inputs / "inventory.json", inventory_sha256=snapshot["inventory_sha256"])


def _stat_report(plan, snapshot, prior, work):
    expected = [(sequence, frame, split) for split in SPLITS for sequence, frame in snapshot["keys"][split]]
    actual = [(sequence, frame, split) for sequence, frame, split, _ in plan["selection"]]
    changes = []
    changed_count = 0
    for sequence, frame, _, files in plan["selection"]:
        previous = prior.get((sequence, frame))
        if previous and any((path, size) != (previous["files"][role]["path"], previous["files"][role]["size_bytes"])
                            for role, path, size in files):
            changed_count += 1
            if len(changes) < 32:
                changes.append([sequence, frame])
    reserved = sum(plan["payload_bytes"].values()) + plan["metadata_bytes"] + plan["selected_file_stats"]
    blockers = []
    if (actual != expected or plan["validation_keys_sha256"] != VALIDATION_KEYS
            or digest(plan["duplicates"]) != digest(snapshot["excluded_duplicates"])
            or plan["inventory_extra_eligible"] or plan["inventory_missing_eligible"]):
        blockers.append("live_inventory_mismatch")
    if changed_count:
        blockers.append("prior_file_path_or_size_changed")
    if reserved >= AUDIT_LIMITS.max_read_bytes:
        blockers.append("read_budget")
    free = shutil.disk_usage(work).free
    if free < REPORT_LIMIT + FREE_RESERVE:
        blockers.append("local_report_disk_budget")
    return seal({
        "schema": "archvis_v12_stat_preflight_v1", "status": "blocked" if blockers else "ready_for_payload_audit",
        "snapshot_sha256": snapshot["content_sha256"], "source_plan_sha256": plan["content_sha256"],
        "blockers": blockers, "counts": plan["counts"], "source_bytes_stat": plan["payload_bytes"],
        "read_bytes_reserved": reserved, "audit_read_cap": AUDIT_LIMITS.max_read_bytes,
        "report_disk_cap": REPORT_LIMIT, "local_free_bytes": free, "local_free_reserve": FREE_RESERVE,
        "file_stats": plan["selected_file_stats"], "directory_entries": plan["directory_entries"],
        "prior_changed_keys_count": changed_count, "first_prior_changed_keys": changes,
        "outside_snapshot_eligible_keys": plan["inventory_extra_eligible"],
        "missing_snapshot_keys": plan["inventory_missing_eligible"], "excluded_duplicates": plan["duplicates"],
        "payload_validation": "not_run", "train_ready": False, "temporal_ready": False,
    })


def _verified_manifest(manifest, snapshot, prior):
    records = manifest["records"]
    expected = [(sequence, frame, split) for split in SPLITS for sequence, frame in snapshot["keys"][split]]
    if ([(row["sequence_id"], row["frame_index"], row["split"]) for row in records] != expected
            or manifest["train_ready"] is not False or manifest["temporal_ready"] is not False):
        raise ValueError("v12_manifest_scope")
    for row in records:
        _verify_prior(row, prior)


def _verify_prior(row, prior):
    key = row["sequence_id"], row["frame_index"]
    if key in prior:
        for role in ROLES:
            if row["files"][role] != prior[key]["files"][role]:
                raise ValueError(f"v12_prior_payload_identity_changed:{key[0]}#{key[1]}:{role}")


def _observe(work, status_path, status, snapshot, prior, started):
    expected = {tuple(key) for keys in snapshot["keys"].values() for key in keys}
    excluded = {tuple(key) for key in snapshot["excluded_duplicates"]}
    report_name = "inventory_check.json" if status["phase"] == "preflight" else "audit_inventory_check.json"

    def progress(event, **detail):
        if event == "inventory":
            extra = detail["usable"] - expected
            missing = expected - detail["usable"]
            new_duplicates = detail["duplicates"] - excluded
            inventory = {
                "status": "blocked" if extra or missing or new_duplicates else "matched",
                "snapshot_sha256": snapshot["content_sha256"], "directory_entries": detail["entries"],
                "outside_snapshot_eligible_keys": sorted(extra), "missing_snapshot_keys": sorted(missing),
                "new_duplicate_keys": sorted(new_duplicates), "excluded_duplicates": sorted(detail["duplicates"]),
                "incomplete_keys_count": len(detail["incomplete"]),
                "first_incomplete_keys": [[*key, roles] for key, roles in sorted(detail["incomplete"].items())[:32]],
                "ignored_entries": dict(detail["ignored"]), "payload_validation": "not_run",
            }
            _write(work / report_name, seal(inventory))
            if inventory["status"] == "blocked":
                raise ValueError("v12_live_inventory_mismatch")
            return
        if event == "file":
            status["current_file"] = {"key": detail["key"], "role": detail["role"], "path": detail["path"]}
        else:
            _verify_prior(detail["record"], prior)
        status.update({name: detail[name] for name in ("read_bytes", "pairs_verified", "pairs_total")})
        if event == "pair" and (detail["pairs_verified"] == 1 or detail["pairs_verified"] % 250 == 0):
            status["elapsed_seconds"] = round(time.monotonic() - started, 3)
            _write(status_path, status)
            print(canonical({name: status[name] for name in (
                "phase", "pairs_verified", "pairs_total", "read_bytes", "elapsed_seconds")}).decode(), flush=True)

    return progress


def run_phase(args):
    work = _local(args.work.parent, directory=True) / args.work.name
    if args.action != "plan":
        for source in (args.rgb_root, args.pose_root):
            if work.is_relative_to(_local(source, directory=True)):
                raise ValueError("v12_work_inside_sources")
    work.mkdir(exist_ok=True)
    work = _local(work, directory=True)
    with _locked(work):
        status_path = work / (args.action + "_status.json")
        if status_path.exists():
            raise ValueError("v12_phase_already_attempted_use_new_work")
        started = time.monotonic()
        status = {"status": "started", "phase": args.action, "train_ready": False, "temporal_ready": False}
        _write(status_path, status)
        try:
            snapshot, coverage, prior = frozen_snapshot(args.inputs)
            if args.action != "plan" and args.snapshot_sha256 != snapshot["content_sha256"]:
                raise ValueError("v12_expected_snapshot_pin")
            status["snapshot_sha256"] = snapshot["content_sha256"]
            if args.action in ("plan", "preflight"):
                _write(work / "snapshot.json", snapshot)
                _write(work / "coverage.json", coverage)
            if args.action == "plan":
                status.update(status="inventory_only", coverage_sha256=coverage["content_sha256"])
            else:
                pending = work / ".pending_audit"
                kwargs = _run_kwargs(args.inputs, snapshot, args.rgb_root, args.pose_root, pending)
                kwargs["on_progress"] = _observe(work, status_path, status, snapshot, prior, started)
                if args.action == "preflight":
                    plan = prepare_extracted_t1(**kwargs, limits=STAT_LIMITS, plan_only=True)
                    report = _stat_report(plan, snapshot, prior, work)
                    _write(work / "source_plan.json", plan)
                    _write(work / "preflight.json", report)
                    status.update(status=report["status"], preflight_sha256=report["content_sha256"],
                                  source_plan_sha256=plan["content_sha256"])
                else:
                    if (work / "verified").exists() or (work / "verified").is_symlink():
                        raise ValueError("v12_verified_manifest_already_exists")
                    plan = read_json(_read(work / "source_plan.json"))
                    check_pin(plan, SOURCE_PLAN, args.source_plan_sha256)
                    report = _stat_report(plan, snapshot, prior, work)
                    if report["blockers"]:
                        raise ValueError("v12_payload_preflight_blocked")
                    status["source_plan_sha256"] = args.source_plan_sha256
                    _write(status_path, status)
                    manifest = prepare_extracted_t1(**kwargs, limits=AUDIT_LIMITS,
                                                    expected_plan_sha256=args.source_plan_sha256)
                    _verified_manifest(manifest, snapshot, prior)
                    pending.rename(work / "verified")
                    status.pop("current_file", None)
                    status.update(status="full_file_audit_complete", manifest_sha256=manifest["content_sha256"],
                                  prior_pinned_pairs_verified=len(prior), file_audit=manifest["audit"])
        except BaseException as exc:
            status.update(status="stopped" if isinstance(exc, KeyboardInterrupt) else "failed",
                          failure={"type": type(exc).__name__, "reason": str(exc)[:512]})
            raise
        finally:
            status["elapsed_seconds"] = round(time.monotonic() - started, 3)
            if sys.platform == "linux":
                import resource
                status["peak_rss_bytes"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
            _write(status_path, status)
    print(canonical({k: v for k, v in status.items() if k != "file_audit"}).decode())
    return 1 if status["status"] == "blocked" else 0


def recover(work, output):
    work = _local(work, directory=True)
    _local(output.parent, directory=True)
    if shutil.disk_usage(output.parent).free < REPORT_LIMIT + FREE_RESERVE:
        raise ValueError("v12_recovery_free_space")
    with _locked(work):
        paths = [(name, work / name) for name in RECOVERY_FILES if (work / name).is_file()]
        if not paths or sum(_local(path).stat().st_size for _, path in paths) > REPORT_LIMIT // 2:
            raise ValueError("v12_recovery_budget")
        created = False
        try:
            with output.open("xb") as handle:
                created = True
                with zipfile.ZipFile(handle, "w", compression=zipfile.ZIP_DEFLATED) as archive:
                    entries = {}
                    for name, path in paths:
                        raw = _read(path)
                        archive.writestr(name, raw)
                        entries[name] = {"bytes": len(raw), "sha256": sha(raw)}
                    archive.writestr("recovery_receipt.json", canonical(seal({
                        "schema": "archvis_v12_preparation_recovery_v1", "files": entries,
                        "train_ready": False, "temporal_ready": False,
                    })) + b"\n")
                if handle.tell() > REPORT_LIMIT // 2:
                    raise ValueError("v12_recovery_zip_size")
        except BaseException:
            if created:
                with suppress(OSError):
                    output.unlink()
            raise
    print(sha(_read(output, REPORT_LIMIT // 2)))
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="action", required=True)
    for action in ("plan", "preflight", "audit", "recover"):
        sub = commands.add_parser(action)
        sub.add_argument("--work", type=Path, required=True)
        if action == "recover":
            sub.add_argument("--output", type=Path, required=True)
            continue
        sub.add_argument("--inputs", type=Path, required=True)
        if action != "plan":
            sub.add_argument("--rgb-root", type=Path, required=True)
            sub.add_argument("--pose-root", type=Path, required=True)
            sub.add_argument("--snapshot-sha256", required=True)
        if action == "audit":
            sub.add_argument("--source-plan-sha256", required=True)
    args = parser.parse_args()
    if sys.platform == "linux" and args.action in ("preflight", "audit"):
        import resource
        soft, hard = resource.getrlimit(resource.RLIMIT_AS)
        cap = 3 * 1024 * MIB
        resource.setrlimit(resource.RLIMIT_AS, (min(cap, soft) if soft != resource.RLIM_INFINITY else cap, hard))
    return recover(args.work, args.output) if args.action == "recover" else run_phase(args)


if __name__ == "__main__":
    sys.exit(main())
