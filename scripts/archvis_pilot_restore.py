"""Verify and restore the completed preparation from a pinned ArchVis runtime."""

from collections import Counter
import hashlib
from pathlib import Path
import re
import shutil
import stat
import zipfile

from rewind_archive.contracts import canonical, digest
from rewind_data.audit import read_json


RESTORE_MEMBERS = ("preflight.json", "prepare_report.json", "manifest/manifest.json")
RESTORE_FILE_LIMIT = 8 * 1024**2
PREPARATION_CODE = {
    "rewind_archive/contracts.py", "rewind_archive/matching.py", "rewind_data/audit.py",
    "rewind_m1/local_rgb.py", "rewind_prepare/converter.py", "rewind_prepare/coordinates.py",
    "rewind_prepare/extracted.py", "rewind_prepare/provenance.py", "rewind_prepare/spec.py",
}


def _restore_sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _restore_read(path, maximum=RESTORE_FILE_LIMIT):
    path = Path(path)
    info = path.lstat()
    _restore_require(stat.S_ISREG(info.st_mode) and 1 <= info.st_size <= maximum, "restore_file_size")
    for part in (path, *path.parents):
        component = part.lstat()
        if stat.S_ISLNK(component.st_mode) or getattr(component, "st_file_attributes", 0) & 0x400:
            raise ValueError("restore_link")
    with path.open("rb") as handle:
        raw = handle.read(maximum + 1)
    if len(raw) != info.st_size:
        raise ValueError("restore_file_size")
    return raw


def _restore_require(condition, reason):
    if not condition:
        raise ValueError(reason)


def _restore_seal(doc, schema):
    _restore_require(doc.get("schema") == schema and doc.get("content_sha256") ==
                     digest({k: v for k, v in doc.items() if k != "content_sha256"}), "restore_content_pin")


def verify_prepared(work, prepared_dir, *, expected_files, revision):
    """Check original preparation bytes and code locally, with no Drive payload reads."""
    work, prepared_dir = Path(work), Path(prepared_dir)
    _restore_require(set(expected_files) == set(RESTORE_MEMBERS)
                     and re.fullmatch("[0-9a-f]{40}", revision), "restore_expected_pins")
    raw = {name: _restore_read(prepared_dir / name) for name in RESTORE_MEMBERS}
    _restore_require(all(expected_files[n] == {"bytes": len(raw[n]), "sha256": _restore_sha(raw[n])}
                         for n in RESTORE_MEMBERS), "restore_file_pin")
    preflight, prepared, manifest = (read_json(raw[name]) for name in RESTORE_MEMBERS)
    plan, provenance = preflight["plan"], manifest["provenance"]
    _restore_seal(preflight, "archvis_pilot_preflight_v1")
    _restore_seal(plan, "direct_extracted_selection_plan_v1")
    _restore_seal(manifest, "direct_extracted_stereo_v1")
    _restore_require(
        preflight["status"] == "ready_for_manifest_approval" and prepared["status"] == "completed"
        and prepared["preflight_sha256"] == preflight["content_sha256"]
        and prepared["manifest_sha256"] == manifest["content_sha256"]
        and prepared["manifest_file_sha256"] == expected_files["manifest/manifest.json"]["sha256"]
        and prepared["manifest_file_bytes"] == expected_files["manifest/manifest.json"]["bytes"]
        and prepared["audit"] == manifest["audit"] and prepared["roots"] == plan["roots"]
        and prepared["validation_keys_sha256"] == manifest["selection"]["validation_keys_sha256"]
        == plan["validation_keys_sha256"]
        and provenance["selection_plan_sha256"] == plan["content_sha256"]
        and manifest["dataset_version"] == plan["dataset_version"]
        and manifest["environment"] == plan["environment"]
        and manifest["source_kind"] == "direct_extracted_files"
        and manifest["diagnostic_only"] is True and manifest["train_ready"] is False
        and manifest["temporal_ready"] is False
        and manifest["selection"]["train_limit"] == plan["train_limit"] == 1000
        and manifest["selection"]["seed"] == plan["seed"], "restore_prepare_chain")
    source = work / "source"
    stamp = _restore_read(source / "SOURCE_REVISION.txt", 42).decode("ascii").strip()
    _restore_require(stamp == provenance["revision"]["revision"] == revision
                     and provenance["revision"]["revision_kind"] == "source_revision_stamp"
                     and set(provenance["code_sha256"]) == PREPARATION_CODE, "restore_source_revision")
    _restore_require(all(_restore_sha(_restore_read(source / name, 2 * 1024**2)) == pin
                         for name, pin in provenance["code_sha256"].items()), "restore_source_code_pin")

    inputs = work / "inputs"
    pins = {"spec": provenance["spec_sha256"], "calibration": provenance["calibration_sha256"],
            "inventory": provenance["inventory_sha256"], "motion_map": provenance["motion_map_sha256"],
            "split_files": provenance["split_files"]}
    _restore_require(pins == plan["metadata_pins"], "restore_metadata_chain")
    for path, key in ((source / "configs/real_archvis62_v1.json", "spec"),
                      (inputs / "scaramuzza.json", "calibration"), (inputs / "inventory.json", "inventory"),
                      (inputs / "motion_map.json", "motion_map")):
        _restore_require(_restore_sha(_restore_read(path)) == pins[key], "restore_metadata_file_pin:" + key)
    owners = {}
    for split in ("train", "validation", "test"):
        payload = _restore_read(inputs / "split" / (split + ".txt"))
        rows = payload.decode("utf-8-sig").splitlines()
        _restore_require({"sha256": _restore_sha(payload), "rows": len(rows)} == pins["split_files"][split]
                         and not set(rows).intersection(owners) and len(rows) == len(set(rows)), "restore_split_pin")
        owners.update(dict.fromkeys(rows, split))
    motion = read_json(_restore_read(inputs / "motion_map.json"))["sequences"]
    selected, validation, keys, counts = [], [], set(), Counter()
    for row in manifest["records"]:
        seq, frame, split = row["sequence_id"], row["frame_index"], row["split"]
        _restore_require(split in ("train", "validation") and owners.get(seq) == split
                         and type(frame) is int and 0 <= frame < 10**12 and (seq, frame) not in keys
                         and row["motion_id"] == motion[seq]["motion_id"]
                         and manifest["motion_identities"][seq] == motion[seq], "restore_record_identity")
        keys.add((seq, frame))
        counts[split] += 1
        if split == "validation":
            validation.append([seq, frame])
        files = row["files"]
        _restore_require(set(files) == {"left", "right", "pose"}
                         and all(re.fullmatch("[0-9a-f]{64}", item["sha256"])
                                 for item in files.values()), "restore_record_files")
        selected.append([seq, frame, split, [[role, files[role]["path"], files[role]["size_bytes"]]
                                            for role in ("left", "right", "pose")]])
    _restore_require(selected == plan["selection"] and counts["train"] == 1000 and counts["validation"] > 0
                     and dict(counts) == plan["counts"] == manifest["audit"]["selected"]
                     and digest(validation) == prepared["validation_keys_sha256"], "restore_selection_chain")
    return {"status": "prepared_bytes_verified", "source_revision": revision,
            "preflight_sha256": preflight["content_sha256"], "manifest_sha256": manifest["content_sha256"],
            "validation_keys_sha256": prepared["validation_keys_sha256"], "counts": dict(counts),
            "mount_root": preflight["mount_root"], "roots": prepared["roots"],
            "current_drive_payloads_verified": False,
            "payload_check": "original initial evaluation and per-batch size/SHA/PNG checks"}, raw


def restore_prepared(work, prepared_dir, *, expected_files, revision):
    work = Path(work)
    report, raw = verify_prepared(work, prepared_dir, expected_files=expected_files, revision=revision)
    _restore_require(not any((work / n).exists() or (work / n).is_symlink()
                             for n in (*RESTORE_MEMBERS, "manifest")), "restore_destination_exists")
    (work / "manifest").mkdir()
    for name in RESTORE_MEMBERS:
        with (work / name).open("xb") as handle:
            handle.write(raw[name])
    return report


def _exact_directory(value):
    path = Path(value)
    _restore_require(path.is_absolute() and path.is_dir() and str(path.resolve(strict=True)) == value,
                     "restore_original_directory_missing_or_changed")
    return path


def confirm_original_roots(work, *, expected_files, revision):
    """Confirm the original directories exist at their recorded paths."""
    report, _ = verify_prepared(work, work, expected_files=expected_files, revision=revision)
    mount = _exact_directory(report["mount_root"])
    _restore_require(set(report["roots"]) == {"rgb", "pose"}, "restore_roots_roles")
    roots = [_exact_directory(value) for value in report["roots"].values()]
    _restore_require(len(set(roots)) == 2 and all(path.is_relative_to(mount) and path != mount for path in roots),
                     "restore_original_root_containment")
    return {**report, "status": "original_directories_confirmed",
            "scope": "exact directory existence only; no payload reads"}


def include_restore_report(recovery, report_path):
    """Extend the freshly generated recovery archive without changing the pinned runner."""
    recovery = Path(recovery)
    report_raw = _restore_read(report_path, 64 * 1024)
    temporary = recovery.with_suffix(".restore.partial")
    maximum = 128 * 1024**2
    with zipfile.ZipFile(recovery) as before:
        members = before.infolist()
        names = [item.filename for item in members]
        size = sum(item.file_size for item in members) + len(report_raw)
        _restore_require(len(set(names)) == len(names) <= 32 and "restore_report.json" not in names
                         and names.count("recovery_receipt.json") == 1 and size <= maximum,
                         "restore_recovery_members")
        _restore_require(shutil.disk_usage(recovery.parent).free >= size + 2 * 1024**2,
                         "restore_recovery_disk")
        receipt = read_json(before.read("recovery_receipt.json"))
        _restore_require(set(receipt["files"]) == set(names) - {"recovery_receipt.json"},
                         "restore_recovery_receipt")
        with zipfile.ZipFile(temporary, "x", compression=zipfile.ZIP_DEFLATED) as after:
            for item in members:
                if item.filename == "recovery_receipt.json":
                    continue
                raw = before.read(item)
                _restore_require(receipt["files"][item.filename] == {"bytes": len(raw), "sha256": _restore_sha(raw)},
                                 "restore_recovery_file_pin")
                after.writestr(item.filename, raw)
            receipt["files"]["restore_report.json"] = {"bytes": len(report_raw), "sha256": _restore_sha(report_raw)}
            after.writestr("restore_report.json", report_raw)
            after.writestr("recovery_receipt.json", canonical(receipt) + b"\n")
    with zipfile.ZipFile(temporary) as archive:
        _restore_require(archive.testzip() is None, "restore_recovery_crc")
    temporary.replace(recovery)
    return recovery
