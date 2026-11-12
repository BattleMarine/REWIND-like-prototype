"""Pinned v1.2 completed-audit input and separate nominal-clock training contract."""

from collections import Counter
import hashlib
import json
from pathlib import Path
import zipfile

from rewind_archive.contracts import digest
from rewind_m1.local_rgb import _regular_read
from rewind_prepare.converter import sha
from rewind_prepare.extracted import MAX_MANIFEST_BYTES, MIB, ROLES, SCHEMA, _local
from rewind_prepare.full_archvis import COUNTS, VALIDATION_KEYS, clip_coverage, frozen_snapshot
from rewind_prepare.temporal_cache import _hash_file
from rewind_prepare.temporal_selection import check_pin, contiguous_clips, seal


AUDIT_ZIP_SHA = "30c46b8d3259d9a48eae7f4b94465a5563e3ad6bf4fb74b6b88ef546da8e4c2f"
MANIFEST_SHA = "fc78e2a43e3b178f8b86abbe64c6221755ed4aa5bb309ea13e34d8cc08cb8877"
MANIFEST_FILE_SHA = "180af995fdc56d474396d71225d6eb2d02b8a90fe83d1e0d022bcfaf28443302"
COVERAGE_SHA = "3f3521c8e9bf8a7bff50ce8ebdf69e4abafbe691b6492d7464baf916ae3f125f"
RUN_ID = "V1.2-FULL-TRAIN-DIAGNOSTIC-001"
PREPROCESSING = "v12_float32_stereo_bilinear_align_corners_cache_v1"


def _record_keys(manifest):
    keys = {split: [] for split in COUNTS}
    for row in manifest["records"]:
        if row["split"] not in keys:
            raise ValueError("v12_training_split")
        key = row["sequence_id"], row["frame_index"]
        if (not isinstance(key[0], str) or type(key[1]) is not int or key[1] < 0
                or set(row["files"]) != set(ROLES)):
            raise ValueError("v12_training_record")
        for entry in row["files"].values():
            path = entry["path"]
            if (not isinstance(path, str) or path.startswith("/") or "\\" in path
                    or any(part in ("", ".", "..") or ":" in part for part in path.split("/"))
                    or type(entry["size_bytes"]) is not int or not 0 < entry["size_bytes"] <= 8*MIB
                    or not isinstance(entry["sha256"], str) or len(entry["sha256"]) != 64
                    or any(c not in "0123456789abcdef" for c in entry["sha256"])):
                raise ValueError("v12_training_file_identity")
        keys[row["split"]].append(key)
    if any(values != sorted(set(values)) for values in keys.values()):
        raise ValueError("v12_training_order_or_duplicate")
    if {split: len(values) for split, values in keys.items()} != COUNTS:
        raise ValueError("v12_training_counts")
    train, validation = (set(keys[split]) for split in COUNTS)
    if train & validation or {k[0] for k in train} & {k[0] for k in validation}:
        raise ValueError("v12_training_split_overlap")
    if digest(keys["validation"]) != VALIDATION_KEYS:
        raise ValueError("v12_training_frozen_validation")
    return keys


def load_audit(archive):
    """Verify the frozen recovery bytes, without rerunning source-file or journal audits."""
    archive = _local(archive)
    if archive.stat().st_size > 16*MIB or _hash_file(archive) != AUDIT_ZIP_SHA:
        raise ValueError("v12_training_audit_zip_pin")
    wanted = {"verified/manifest.json", "coverage.json", "snapshot.json"}
    payloads = {}
    with zipfile.ZipFile(archive) as bundle:
        members = bundle.infolist()
        if (len(members) != 14 or len({i.filename for i in members}) != len(members)
                or any(i.file_size > MAX_MANIFEST_BYTES for i in members)
                or sum(i.file_size for i in members) > 128*MIB):
            raise ValueError("v12_training_audit_zip_budget")
        receipt = json.loads(bundle.read("recovery_receipt.json"))
        check_pin(receipt, "archvis_v12_preparation_recovery_v1", receipt.get("content_sha256"))
        if (receipt["train_ready"] is not False or receipt["temporal_ready"] is not False
                or set(bundle.namelist()) != set(receipt["files"]) | {"recovery_receipt.json"}):
            raise ValueError("v12_training_receipt_scope")
        for name, pin in receipt["files"].items():
            info = bundle.getinfo(name)
            if info.file_size != pin["bytes"]:
                raise ValueError("v12_training_receipt_size")
            measured = hashlib.sha256()
            chunks = [] if name in wanted else None
            with bundle.open(name) as stream:
                while block := stream.read(MIB):
                    measured.update(block)
                    if chunks is not None:
                        chunks.append(block)
            if measured.hexdigest() != pin["sha256"]:
                raise ValueError("v12_training_receipt_sha")
            if chunks is not None:
                payloads[name] = b"".join(chunks)
    if sha(payloads["verified/manifest.json"]) != MANIFEST_FILE_SHA:
        raise ValueError("v12_training_manifest_file_pin")
    manifest = json.loads(payloads["verified/manifest.json"])
    check_pin(manifest, SCHEMA, MANIFEST_SHA)
    if (manifest["train_ready"] is not False or manifest["temporal_ready"] is not False
            or manifest["diagnostic_only"] is not True or manifest["distance_unit"] != "mm"
            or manifest["coordinate_frame"] != "device_relative"
            or manifest["image_size"] != [1024, 1024]):
        raise ValueError("v12_training_original_guards")
    keys = _record_keys(manifest)
    coverage = json.loads(payloads["coverage.json"])
    check_pin(coverage, "archvis_v12_clip_coverage_plan_v1", COVERAGE_SHA)
    summary, clips = clip_coverage(keys)
    if (coverage["summary"] != summary or coverage["clips"] != clips
            or coverage["train_ready"] is not False or coverage["temporal_ready"] is not False):
        raise ValueError("v12_training_coverage_pin")
    snapshot = json.loads(payloads["snapshot.json"])
    check_pin(snapshot, "archvis_v12_frozen_inventory_v1", coverage["snapshot_sha256"])
    if snapshot["keys"] != {split: [list(key) for key in values] for split, values in keys.items()}:
        raise ValueError("v12_training_snapshot_keys")
    return manifest, coverage, snapshot


def bind_inputs(inputs, manifest, snapshot):
    """Validate the old fixed input bundle; its producer revision remains historical."""
    inputs = _local(inputs, directory=True)
    regenerated, _, _ = frozen_snapshot(inputs)
    if digest(regenerated) != digest(snapshot):
        raise ValueError("v12_training_input_snapshot")
    spec_raw = _regular_read(inputs / "spec.json", MIB)
    if sha(spec_raw) != manifest["provenance"]["spec_sha256"]:
        raise ValueError("v12_training_spec_pin")
    spec = json.loads(spec_raw)
    if spec["joint_names"] != manifest["joint_names"]:
        raise ValueError("v12_training_spec_joints")
    return spec, spec_raw


def training_plan(manifest, coverage):
    keys = _record_keys(manifest)
    train = [row for row in coverage["clips"] if row["split"] == "train"]
    # Keep exactly the original strict complete-clip validation policy.
    validation = sorted(key for _, key in contiguous_clips(keys["validation"], seed=71, width=50))
    covered, repeated = set(), 0
    for row in train:
        real = row["real_frames"]
        if type(real) is not int or not 1 <= real <= 50:
            raise ValueError("v12_training_clip_real_frames")
        for frame in range(row["start_frame"], row["start_frame"]+real):
            key = row["sequence"], frame
            repeated += key in covered
            covered.add(key)
    if covered != set(keys["train"]) or repeated:
        raise ValueError("v12_training_exact_train_coverage")
    teacher_val = {(seq, i) for seq, start in validation for i in range(start, start+50)}
    if len(validation) != 29 or len(teacher_val) != 1450 or not teacher_val <= set(keys["validation"]):
        raise ValueError("v12_training_teacher_validation_coverage")
    updates = {"frontend": (len(keys["train"])+1)//2, "body": len(train), "hand": len(train)}
    return seal({"schema": "archvis_v12_training_plan_v1", "run_id": RUN_ID,
                 "audit_zip_sha256": AUDIT_ZIP_SHA, "manifest_sha256": MANIFEST_SHA,
                 "coverage_sha256": COVERAGE_SHA, "counts": COUNTS, "train_clips": train,
                 "teacher_validation_clips": [list(key) for key in validation],
                 "teacher_validation_keys_sha256": digest(sorted(teacher_val)),
                 "teacher_validation_excluded_keys": sorted(set(keys["validation"])-teacher_val),
                 "summary": coverage["summary"], "updates_per_epoch": updates,
                 "ten_epoch_total_updates": 10*sum(updates.values()),
                 "source_payload_bytes": sum(f["size_bytes"] for r in manifest["records"] for f in r["files"].values()),
                 "image_tensor_bytes": sum(COUNTS.values())*2*3*128*128*4})


def diagnostic_decision(config_sha256, plan_sha256):
    return seal({"schema": "archvis_v12_training_diagnostic_decision_v1", "run_id": RUN_ID,
                 "decision_date": "2026-11-12", "authorized_by": "user_via_orchestration",
                 "purpose": "full_archvis_train_expansion", "config_sha256": config_sha256,
                 "plan_sha256": plan_sha256, "audit_zip_sha256": AUDIT_ZIP_SHA,
                 "manifest_sha256": MANIFEST_SHA, "original_train_ready": False,
                 "original_temporal_ready": False, "nominal_fps": 25,
                 "clock_formula": "frame_index / 25", "recorded_timestamps_verified": False,
                 "stereo_sync_measured": False, "stereo_sync_basis": "shared_frame_key_assumption",
                 "original_motion_independence_verified": False, "validation_reference_only": True,
                 "independent_validation_verified": False, "velocity_loss_applied": False,
                 "foot_loss_applied": False, "overlay_89_review_scope": "historical_89_samples_only"})
