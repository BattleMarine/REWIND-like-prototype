"""Audit the pinned ArchVis v1 selection and prepare evidence for human review."""

import argparse
from collections import Counter, defaultdict, deque
from contextlib import suppress
from dataclasses import asdict
import io
from pathlib import Path
import re
import shutil
import tempfile
import time
import zipfile

from PIL import Image, ImageDraw

from rewind_archive.contracts import canonical, digest
from rewind_archive.matching import identify
from rewind_data.audit import read_json
from rewind_m1.archvis_v0 import VALIDATION
from rewind_m1.local_rgb import _regular_read
from rewind_prepare.converter import motion_entries, sha, validate_pose_geometry
from rewind_prepare.extracted import IMAGE_SIZE, MAX_FILE_BYTES, MAX_MANIFEST_BYTES, MIB, ROLES, _local, _pose_targets, _raw_pose
from rewind_prepare.spec import validate_spec
from rewind_prepare.temporal_cache import (
    JOURNAL, MANIFEST, Limits, RawCache, _atomic, _check_part, _checked_plan, _hash_file, _inspect, _locked,
    prepare, restore,
)
from rewind_prepare.temporal_selection import check_pin, contiguous_clips, seal
from .data import REQUIRED_EVIDENCE, _checked_evidence, reviewed_clip
from .training import _source


SCHEMA = "archvis_v1_input_audit_v1"
REPORT = "audit.json"
CAMERAS = "camera_candidates.json"
PREFLIGHT = "preflight.json"
PREPARATION = "preparation.json"
INPUT_FILES = ("selection.json", "spec.json", "calibration.json", "previous_manifest.json", "inventory.json",
               "motion_map.json", "motion_review.json", "split/train.txt", "split/validation.txt", "split/test.txt")
TIME_FIELD = re.compile(r"time|timestamp|^pts$|^fps$|frame.?rate|presentation|capture.?clock", re.I)


def _read(path, maximum=MAX_MANIFEST_BYTES):
    return _regular_read(_local(path), maximum)


def _timer(seconds):
    if type(seconds) is not int or not 1 <= seconds <= 86400:
        raise ValueError("rewind_v1_audit_time_limit")
    start = time.monotonic()

    def check():
        if time.monotonic() - start >= seconds:
            raise TimeoutError("rewind_v1_audit_time_limit")

    return start, check


def _motion_review(raw, plan, owners, previous_review):
    doc = read_json(raw)
    if doc.get("schema_version") != "explicit_motion_map_v1":
        raise ValueError("rewind_v1_audit_motion_map")
    entries = doc["sequences"]
    by_id, statuses = defaultdict(set), Counter()
    for split in ("train", "validation"):
        group = {seq for seq, owner in owners.items() if owner == split}
        selected = {seq: entries[seq] for seq in group}
        motion_entries({"schema_version": "explicit_motion_map_v1", "sequences": selected}, group, owners)
        for entry in selected.values():
            by_id[entry["motion_id"]].add(split)
            statuses[split, entry["identity_status"]] += 1
    overlap = sorted(key for key, splits in by_id.items() if len(splits) > 1)
    candidates = []
    for item in previous_review.get("cross_split_candidates", ()):
        relevant = [row for row in item["sequences"] if owners.get(row["sequence"]) == row["split"]]
        if {row["split"] for row in relevant} == {"train", "validation"}:
            candidates.append({"normalized_display_name": item["normalized_display_name"], "sequences": relevant})
    return {"map_sha256": plan["motion_map_sha256"], "cross_split_declared_ids": overlap,
            "selected_cross_split_name_candidates_unverified": candidates,
            "sequence_identity_status": {f"{split}:{status}": count for (split, status), count in sorted(statuses.items())},
            "all_original_ids_verified_in_pinned_map": not any(status == "declared_unverified" for _, status in statuses),
            "scope": "pinned map IDs only; no inference from clip names"}


def _record_time(doc, key_id, fields, paths, samples):
    found = False
    for scope in ("pose", "camera_left", "camera_right", "device"):
        source = doc if scope == "pose" else doc[scope]
        for key, value in source.items():
            if TIME_FIELD.search(key):
                if len(key) > 100 or len(paths) >= 128 and (scope, key) not in paths:
                    raise ValueError("rewind_v1_audit_time_field_limit")
                paths.add((scope, key))
                fields[scope, key, type(value).__name__] += 1
                entry = {"key": key_id}
                if type(value) in (int, float, bool, type(None)):
                    entry["value"] = value
                elif isinstance(value, str):
                    entry.update(value=value[:120], truncated=len(value) > 120)
                else:
                    entry.update(value_omitted=True, value_type=type(value).__name__)
                first, last = samples.setdefault((scope, key), ([], deque(maxlen=3)))
                if len(first) < 3:
                    first.append(entry)
                last.append(entry)
                found = True
    return found


def _samples(clips, width):
    result = {}
    for split, starts in clips.items():
        seen = set()
        for sequence, start in starts:
            if split == "train" and sequence in seen:
                continue
            seen.add(sequence)
            frames = {start, start + width//2, start + width-1} if split == "train" else {start + width//2}
            for frame in frames:
                result[sequence, frame] = split
    return result


def _overlay(pair, fields, key):
    width = 384
    composed = Image.new("RGB", (width*2, width+24), "#111111")
    counts = {}
    for view, side in enumerate(("left", "right")):
        x0 = view*width
        with Image.open(io.BytesIO(pair[side]), formats=["PNG"]) as original:
            if list(original.size) != IMAGE_SIZE:
                raise ValueError("rewind_v1_audit_overlay_size")
            composed.paste(original.convert("RGB").resize((width, width), Image.Resampling.BILINEAR), (x0, 24))
        draw = ImageDraw.Draw(composed)
        draw.text((x0+4, 6), f"GT {side} | frame {key[1]}", fill="white")
        inside = 0
        for point, valid in zip(fields[view]["coordinates"], fields[view]["valid_mask"]):
            if not valid:
                continue
            x, y = point
            if 0 <= x <= IMAGE_SIZE[0]-1 and 0 <= y <= IMAGE_SIZE[1]-1:
                px, py = x0 + x*(width-1)/(IMAGE_SIZE[0]-1), 24 + y*(width-1)/(IMAGE_SIZE[1]-1)
                draw.ellipse((px-2, py-2, px+2, py+2), fill="#47eaff", outline="#002448")
                inside += 1
        counts[side] = {"inside": inside, "annotation_valid": sum(fields[view]["valid_mask"])}
    out = io.BytesIO()
    composed.save(out, format="PNG")
    return out.getvalue(), counts


def _cache_for_plan(cache, plan):
    expected = {key: plan[key] for key in ("spec_sha256", "calibration_sha256", "dataset_version",
                                         "inventory_sha256", "validation_keys_sha256")}
    manifest = cache.manifest
    if (manifest["selection_sha256"] != plan["content_sha256"] or manifest["binding"] != expected
            or manifest["source_kind"] != "native_rgb_png_and_pose_json"
            or manifest["preprocessing"] != "none_raw_source_bytes_v1"
            or set(cache.index) != {tuple(row["key"]) for row in plan["records"]}):
        raise ValueError("rewind_v1_audit_cache_binding")
    for row in plan["records"]:
        _, cached = cache.index[tuple(row["key"])]
        if cached["split"] != row["split"] or any(
            identify(cached["files"][role]["path"], "pose" if role == "pose" else "rgb") != (*row["key"], role)
            or any(cached["files"][role].get(k) != v for k, v in row["files"][role].items()) for role in ROLES
        ):
            raise ValueError("rewind_v1_audit_cache_pair")


def audit_cache(plan, cache, spec, calibration, motion_raw, output, *, max_read_bytes, max_report_bytes, max_seconds,
                previous_review=None):
    started, check = _timer(max_seconds)
    if type(max_read_bytes) is not int or max_read_bytes <= 0 or type(max_report_bytes) is not int or not 4*MIB <= max_report_bytes <= 128*MIB:
        raise ValueError("rewind_v1_audit_limits")
    _cache_for_plan(cache, plan)
    output = _local(output, directory=True)
    owners = {row["key"][0]: row["split"] for row in plan["records"]}
    keys = {split: sorted(tuple(row["key"]) for row in plan["records"] if row["split"] == split)
            for split in ("train", "validation")}
    width = plan["clip_frames"]
    clips = {"train": [tuple(row) for row in plan["train_clip_starts"]],
             "validation": sorted(clip for _, clip in contiguous_clips(keys["validation"], seed=plan["seed"], width=width))}
    sample_keys = _samples(clips, width)
    report = {"schema": SCHEMA, "status": "running", "selection_sha256": plan["content_sha256"],
              "cache_sha256": cache.manifest["content_sha256"], "spec_sha256": plan["spec_sha256"],
              "calibration_sha256": plan["calibration_sha256"], "source_sha256": digest(_source()),
              "train_ready": False, "blockers": list(REQUIRED_EVIDENCE),
              "validation_keys_sha256": plan["validation_keys_sha256"],
              "coverage": {split: {"selected_pairs": len(keys[split]), "number_contiguous_clips": len(clips[split]),
                                   "pairs_in_number_contiguous_clips": len(clips[split])*width,
                                   "clip_starts": clips[split]}
                           for split in keys},
              "clip_frames": width,
              "motion": _motion_review(motion_raw, plan, owners, previous_review or {}), "overlays": [],
              "overlay_scope": "GT sample only; start/middle/end of first selected train clip per sequence, midpoint per validation clip",
              "coordinate_convention": {"world_basis": "F_diag_1_minus1_1", "transform": "world_to_camera_column_vector",
                                        "translation_unit": "mm", "metric_anchor_scope": "adopted rig assumption"},
              "time": {"timestamp_verified": False, "stereo_sync_verified": False,
                       "nominal_fps_applied": False, "scope": "pose root and left/right camera and device object keys"}}
    counts, fields, paths, samples = Counter(), Counter(), set(), {}
    residual_max, residual_counts = {}, Counter()
    camera_rows, payload_groups = [], defaultdict(set)
    read_bytes = 0

    def publish(name, raw):
        used = sum(path.stat().st_size for path in output.iterdir() if path.is_file())
        if used + len(raw) > max_report_bytes:
            raise TimeoutError("rewind_v1_audit_report_limit")
        _atomic(output / name, raw)

    def save_report():
        report.update(pairs_audited=dict(counts), bytes_read_reserved=read_bytes,
                      elapsed_seconds=time.monotonic()-started, residual_max=residual_max,
                      residual_counts=dict(residual_counts))
        report["time"].update(
            observed_fields=[{"scope": scope, "name": key, "type": kind, "count": count}
                             for (scope, key, kind), count in sorted(fields.items())],
            field_value_samples=[{"scope": scope, "name": key, "first": first, "last": list(last)}
                                 for (scope, key), (first, last) in sorted(samples.items())],
            pairs_with_time_like_fields=counts["pairs_with_time_like_fields"],
        )
        publish(REPORT, canonical(seal(report)) + b"\n")

    try:
        save_report()
        for part in cache.manifest["parts"]:
            check()
            reserve = part["bytes"] + 1
            if read_bytes + reserve > max_read_bytes:
                raise TimeoutError("rewind_v1_audit_read_limit")
            read_bytes += reserve
            _check_part(cache.root / part["name"], part, check)
        for row in plan["records"]:
            check()
            sequence, frame = key = tuple(row["key"])
            report["last_attempted_key"] = key
            _, cached = cache.index[key]
            reserve = sum(item["size_bytes"] + 1 for item in cached["files"].values())
            if read_bytes + reserve > max_read_bytes:
                raise TimeoutError("rewind_v1_audit_read_limit")
            read_bytes += reserve
            pair = cache.read_pair(sequence, frame, split=row["split"])
            doc = _raw_pose(pair["pose"], frame)
            residuals = defaultdict(list)
            camera = validate_pose_geometry(doc, spec, calibration, residuals)
            if not all(residuals[name] for name in ("projection_px", "world_camera_raw_max_abs", "inverse_raw_max_abs")):
                raise ValueError("rewind_v1_audit_geometry_coverage")
            for name, values in residuals.items():
                if values:
                    residual_max[name] = max(residual_max.get(name, 0.), max(values))
                    residual_counts[name] += len(values)
            if _record_time(doc, key, fields, paths, samples):
                counts["pairs_with_time_like_fields"] += 1
            camera_rows.append({"key": key, "split": row["split"], "camera_pose": camera, "time_seconds": None})
            payload_groups[tuple(cached["files"][role]["sha256"] for role in ROLES)].add(row["split"])
            if key in sample_keys:
                name = f"gt-{row['split']}-{len(report['overlays']):02d}.png"
                raw, inside = _overlay(pair, _pose_targets(doc, spec), key)
                publish(name, raw)
                report["overlays"].append({"name": name, "key": key, "sha256": sha(raw), "visibility_counts": inside,
                                           "review_status": "manual_review_pending"})
            counts[row["split"]] += 1
            if (counts["train"] + counts["validation"]) % 50 == 0:
                save_report()
        camera_doc = seal({"schema": "archvis_v1_camera_candidates_v1", "selection_sha256": plan["content_sha256"],
                           "cache_sha256": report["cache_sha256"], "rows": camera_rows, "train_ready": False})
        publish(CAMERAS, canonical(camera_doc) + b"\n")
        report["camera_candidates_sha256"] = camera_doc["content_sha256"]
        report["motion"]["cross_split_exact_stereo_pose_payload_groups"] = sum(len(splits)>1 for splits in payload_groups.values())
        report["status"] = "blocked"
        report["decision"] = "external_review_required"
        report["numeric_geometry_passed"] = True
        save_report()
    except BaseException as exc:
        report.update(status="stopped" if isinstance(exc, (TimeoutError, KeyboardInterrupt)) else "failed",
                      failure={"type": type(exc).__name__, "reason": str(exc)[:512]})
        with suppress(BaseException):
            save_report()
        raise
    return seal(report)


def load_inputs(inputs, selection_sha256):
    receipt = read_json(_read(inputs / "input_receipt.json", MIB))
    if (receipt["schema"] != "archvis_v1_input_bundle_v1" or receipt["selection_sha256"] != selection_sha256
            or set(receipt["files"]) != set(INPUT_FILES)):
        raise ValueError("rewind_v1_audit_input_receipt")
    assets = {}
    for name in INPUT_FILES:
        raw = _read(inputs / name, 8*MIB)
        if receipt["files"][name] != {"bytes": len(raw), "sha256": sha(raw)}:
            raise ValueError("rewind_v1_audit_input_pin:" + name)
        assets[name] = raw
    plan = read_json(assets["selection.json"])
    _checked_plan(plan, selection_sha256)
    if plan["validation_keys_sha256"] != VALIDATION:
        raise ValueError("rewind_v1_audit_frozen_validation")
    for name, key in (("spec.json", "spec_sha256"), ("calibration.json", "calibration_sha256"),
                      ("motion_map.json", "motion_map_sha256"), ("inventory.json", "inventory_sha256")):
        if sha(assets[name]) != plan[key]:
            raise ValueError("rewind_v1_audit_selection_input_pin:" + name)
    previous = read_json(assets["previous_manifest.json"])
    check_pin(previous, "direct_extracted_stereo_v1", plan["prior_manifest_sha256"])
    if previous["selection"]["validation_keys_sha256"] != VALIDATION:
        raise ValueError("rewind_v1_audit_previous_validation")
    owners = {}
    if set(plan["split_files"]) != {"train", "validation", "test"}:
        raise ValueError("rewind_v1_audit_split_roles")
    for split, pin in plan["split_files"].items():
        raw = assets[f"split/{split}.txt"]
        rows = raw.decode("utf-8-sig").splitlines()
        if (pin != {"sha256": sha(raw), "rows": len(rows)} or len(set(rows)) != len(rows)
                or set(rows).intersection(owners)):
            raise ValueError("rewind_v1_audit_split_pin")
        owners.update(dict.fromkeys(rows, split))
    if any(owners.get(row["key"][0]) != row["split"] for row in plan["records"]):
        raise ValueError("rewind_v1_audit_split_membership")
    return plan, assets


def preflight(plan, rgb_root, pose_root, spec, output, limits):
    roots = {"rgb": _local(rgb_root, directory=True), "pose": _local(pose_root, directory=True)}
    _, check = _timer(min(900, int(limits.max_seconds)))
    missing, missing_count, reserved, pack_upper = [], 0, 0, 0
    for row in plan["records"]:
        check()
        try:
            [(_, entries)] = _inspect([row], roots, spec["limits"]["max_file_bytes"])
        except FileNotFoundError as exc:
            missing_count += 1
            if len(missing) < 32:
                missing.append({"key": row["key"], "missing_path": str(exc.filename)[-512:]})
            continue
        reserved += sum(size+1 for _, size, _ in entries.values())
        pack_upper += sum(size+128+2*len(pin["path"].encode("utf-8")) for _, size, pin in entries.values())
    pack_upper += 32*MIB
    passed = (missing_count == 0 and reserved <= limits.max_read_bytes and pack_upper <= limits.max_disk_bytes
              and shutil.disk_usage(output).free >= pack_upper)
    report = {"status": "ready_for_pack" if passed else "blocked", "selection_sha256": plan["content_sha256"],
              "scope": "stat only; zero source payload bytes", "missing_pairs": missing_count,
              "first_missing_pairs": missing, "source_bytes_reserved_estimate": reserved,
              "estimated_pack_and_metadata_bytes": pack_upper, "limits": asdict(limits)}
    _atomic(output / PREFLIGHT, canonical(report) + b"\n")
    if not passed:
        raise ValueError("rewind_v1_audit_source_preflight_blocked")
    return report


def run_prepare(args):
    if digest(_source()) != args.source_sha256:
        raise ValueError("rewind_v1_audit_source_pin")
    work = Path(args.work).absolute()
    work.mkdir(exist_ok=True)
    work = _local(work, directory=True)
    output = work / "audit"
    output.mkdir(exist_ok=True)
    cache_path = work / "cache"
    report = {"status": "started", "selection_sha256": args.selection_sha256, "source_sha256": args.source_sha256}
    with _locked(work):
        try:
            plan, inputs = load_inputs(args.inputs, args.selection_sha256)
            spec = validate_spec(read_json(inputs["spec.json"]))
            pack_limits = Limits(args.pack_read_bytes, args.pack_disk_bytes, args.pack_buffer_bytes,
                                 args.pack_seconds, args.workers)
            if bool(args.retained_cache) != bool(args.retained_cache_sha256):
                raise ValueError("rewind_v1_audit_retained_cache_pin_required")
            if (cache_path / MANIFEST).is_file() and not args.retained_cache:
                state = read_json(_read(cache_path / JOURNAL if (cache_path / JOURNAL).is_file() else output / PREPARATION))
                cache_pin = state["cache_sha256"]
                report["cache_mode"] = "local_workspace"
            elif args.retained_cache:
                cache_pin = args.retained_cache_sha256
                _cache_for_plan(RawCache(args.retained_cache, cache_pin), plan)
                report["cache_restore"] = restore(args.retained_cache, expected_sha256=cache_pin, output=cache_path,
                                                  max_disk_bytes=args.pack_disk_bytes, max_seconds=args.pack_seconds)
                report["cache_mode"] = "restored_pinned_cache"
            else:
                if args.rgb_root is None or args.pose_root is None:
                    raise ValueError("rewind_v1_audit_rgb_pose_roots_required")
                preflight(plan, args.rgb_root, args.pose_root, spec, output, pack_limits)
                state = prepare(plan, expected_sha256=args.selection_sha256, rgb_root=args.rgb_root,
                                pose_root=args.pose_root, spec_path=args.inputs / "spec.json",
                                calibration_path=args.inputs / "calibration.json", output=cache_path, limits=pack_limits)
                cache_pin = state["cache_sha256"]
                report["cache_mode"] = "created_from_pinned_selection"
            report["cache_sha256"] = cache_pin
            _atomic(output / PREPARATION, canonical(report) + b"\n")
            audit = audit_cache(plan, RawCache(cache_path, cache_pin), spec, read_json(inputs["calibration.json"]),
                                inputs["motion_map.json"], output, max_read_bytes=args.audit_read_bytes,
                                max_report_bytes=args.audit_report_bytes, max_seconds=args.audit_seconds,
                                previous_review=read_json(inputs["motion_review.json"]))
            report.update(status=audit["status"], audit_sha256=audit["content_sha256"])
        except BaseException as exc:
            report.update(status="stopped" if isinstance(exc, (KeyboardInterrupt, TimeoutError)) else "failed",
                          failure={"type": type(exc).__name__, "reason": str(exc)[:512]})
            with suppress(BaseException):
                _atomic(output / PREPARATION, canonical(report) + b"\n")
            raise
        _atomic(output / PREPARATION, canonical(report) + b"\n")
    return report


def approve_audit(work, audit_sha256, selection_sha256, evidence_path, evidence_sha256):
    audit = read_json(_read(work / "audit" / REPORT))
    check_pin(audit, SCHEMA, audit_sha256)
    if (audit["selection_sha256"] != selection_sha256 or audit["source_sha256"] != digest(_source())
            or audit["status"] != "blocked" or audit.get("numeric_geometry_passed") is not True
            or audit["motion"]["cross_split_declared_ids"] or audit["motion"]["cross_split_exact_stereo_pose_payload_groups"]):
        raise ValueError("rewind_v1_audit_unresolved_geometry_or_motion")
    evidence = read_json(_read(evidence_path))
    _checked_evidence(audit | {"content_sha256": selection_sha256}, evidence, evidence_sha256)
    if evidence.get("time_basis") != "recorded_timestamps":
        raise ValueError("rewind_v1_audit_recorded_timestamp_review_required")
    for key in REQUIRED_EVIDENCE:
        if sha(_read(evidence_path.parent / "review" / (key + ".json"), 4*MIB)) != evidence["review_artifact_sha256"][key]:
            raise ValueError("rewind_v1_audit_external_review_artifact_pin:" + key)
    cameras = read_json(_read(work / "audit" / CAMERAS))
    check_pin(cameras, "archvis_v1_camera_candidates_v1", audit["camera_candidates_sha256"])
    if (cameras["cache_sha256"], cameras["selection_sha256"]) != (audit["cache_sha256"], selection_sha256):
        raise ValueError("rewind_v1_audit_camera_candidate_binding")
    observed = {tuple(row["key"]): row["camera_pose"] for row in cameras["rows"]}
    for sequence, rows in evidence["frames"].items():
        for frame, row in rows.items():
            if observed.get((sequence, int(frame))) != row["camera_pose"]:
                raise ValueError("rewind_v1_audit_reviewed_camera_mismatch")
    for coverage in audit["coverage"].values():
        for sequence, start in coverage["clip_starts"]:
            reviewed_clip(evidence, sequence, range(start, start+audit["clip_frames"]), scale_mm=100.)
    return {"cache_sha256": audit["cache_sha256"], "audit_sha256": audit_sha256}


def recover_audit(work, output, max_bytes):
    work = _local(work, directory=True)
    output = Path(output).absolute()
    if (output.exists() or output.is_symlink() or output.is_relative_to(work)
            or type(max_bytes) is not int or not 1 <= max_bytes <= 256*MIB):
        raise ValueError("rewind_v1_audit_recovery_args")
    with _locked(work):
        candidates = [work / "audit" / name for name in (REPORT, CAMERAS, PREFLIGHT, PREPARATION)]
        candidates += [work / name for name in ("notebook_prepare_result.json", "notebook_training_result.json")]
        candidates += sorted((work / "audit").glob("gt-*.png"))
        candidates += [work / "cache" / name for name in (JOURNAL, MANIFEST)]
        paths = {path.relative_to(work).as_posix(): path for path in candidates if path.is_file()}
        if not paths or len(paths) > 128:
            raise ValueError("rewind_v1_audit_recovery_members")
        payloads = {}
        size = 4096
        for name, path in paths.items():
            raw = _read(path, min(MAX_MANIFEST_BYTES, max_bytes))
            size += len(raw) + 2*len(name.encode()) + 256
            if size > max_bytes:
                raise TimeoutError("rewind_v1_audit_recovery_size")
            payloads[name] = raw
        receipt = canonical({"schema": "archvis_v1_audit_recovery_v1", "files":
                             {name: {"bytes": len(raw), "sha256": sha(raw)} for name, raw in payloads.items()}}) + b"\n"
        if size + len(receipt) > max_bytes or shutil.disk_usage(output.parent).free < size + len(receipt):
            raise TimeoutError("rewind_v1_audit_recovery_size")
        with tempfile.NamedTemporaryFile(dir=output.parent, prefix=".audit-zip-", delete=False) as handle:
            temporary = Path(handle.name)
        try:
            with zipfile.ZipFile(temporary, "w") as archive:
                for name, raw in payloads.items():
                    archive.writestr(name, raw)
                archive.writestr("recovery_receipt.json", receipt)
            temporary.replace(output)
        finally:
            temporary.unlink(missing_ok=True)
    return {"output": str(output), "sha256": _hash_file(output), "bytes": output.stat().st_size}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest="command", required=True)
    prep = subs.add_parser("prepare")
    gate = subs.add_parser("check-training")
    rec = subs.add_parser("recover")
    for sub in (prep, gate, rec):
        sub.add_argument("--work", required=True, type=Path)
    prep.add_argument("--inputs", required=True, type=Path)
    for name in ("rgb-root", "pose-root", "retained-cache"):
        prep.add_argument("--" + name, type=Path)
    for name in ("selection-sha256", "source-sha256"):
        prep.add_argument("--" + name, required=True)
    prep.add_argument("--retained-cache-sha256")
    for name in ("pack-read-bytes", "pack-disk-bytes", "pack-buffer-bytes", "pack-seconds",
                 "workers", "audit-read-bytes", "audit-report-bytes", "audit-seconds"):
        prep.add_argument("--" + name, required=True, type=int)
    for name in ("audit-sha256", "selection-sha256", "evidence-sha256"):
        gate.add_argument("--" + name, required=True)
    gate.add_argument("--evidence-path", type=Path, required=True)
    rec.add_argument("--output", type=Path, required=True)
    rec.add_argument("--max-bytes", type=int, required=True)
    args = parser.parse_args(argv)
    if args.command == "prepare":
        result = run_prepare(args)
    elif args.command == "check-training":
        result = approve_audit(args.work, args.audit_sha256, args.selection_sha256, args.evidence_path, args.evidence_sha256)
    else:
        result = recover_audit(args.work, args.output, args.max_bytes)
    print(canonical(result).decode("utf-8"))


if __name__ == "__main__":
    main()
