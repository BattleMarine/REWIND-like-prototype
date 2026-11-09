"""Metadata-only selection for the fixed-size ArchVis v1 temporal dataset."""

from collections import defaultdict

from rewind_archive.contracts import digest
from rewind_archive.matching import contiguous_runs, identify
from rewind_data.audit import read_json
from rewind_m1.local_rgb import _hash
from .converter import official_splits, sha
from .extracted import (
    MAX_MANIFEST_BYTES, ROLES, SPLITS, ExtractedLimits, _Budget, _inventory, _local,
)
from .spec import validate_spec


SCHEMA = "archvis_temporal_selection_v1"
CLIP = 50
TRAIN_COUNT = 1000
VALIDATION_COUNT = 2236
BLOCKERS = (
    "timestamp_and_stereo_sync_unverified", "camera_transform_and_coordinate_evidence_unverified",
    "image_pose_alignment_unverified", "original_motion_independence_unverified",
)


def seal(body):
    return {**body, "content_sha256": digest(body)}


def check_pin(doc, schema, expected):
    if (not _hash(expected) or doc.get("schema") != schema or doc.get("content_sha256") != expected
            or digest({key: value for key, value in doc.items() if key != "content_sha256"}) != expected):
        raise ValueError("temporal_document_pin")


def contiguous_clips(keys, *, seed, width):
    """Tile disjoint clips, preferring one clip per sequence before repeats."""
    frames = defaultdict(list)
    for sequence, frame in keys:
        frames[sequence].append(frame)
    clips = defaultdict(list)
    for sequence, items in frames.items():
        for first, last in contiguous_runs(sorted(items)):
            for start in range(first, last - width + 2, width):
                clips[sequence].append((sequence, start))
        clips[sequence].sort(key=lambda clip: (digest([seed, *clip]), clip))
    return sorted((item for group in clips.values() for item in enumerate(group)),
                  key=lambda item: (item[0], digest([seed, *item[1]]), item[1]))


def _nominal_paths(sequence, frame):
    return {side: f"{sequence}/fisheye_final_image/camera_{side}/final_{frame}.png"
            for side in ("left", "right")} | {"pose": f"{sequence}/json/frame_{frame}.json"}


def plan_selection(*, inventory, spec_path, split_dir, previous_manifest, previous_sha256, seed=71):
    if type(seed) is not int or not 0 <= seed < 2**64:
        raise ValueError("temporal_selection_seed")
    budget = _Budget(ExtractedLimits(96 * 1024**2, 60))
    previous = read_json(budget.read(_local(previous_manifest), MAX_MANIFEST_BYTES))
    check_pin(previous, "direct_extracted_stereo_v1", previous_sha256)
    provenance = previous["provenance"]
    spec_raw = budget.read(_local(spec_path), 256 * 1024)
    if sha(spec_raw) != provenance["spec_sha256"]:
        raise ValueError("temporal_spec_pin")
    owners, split_pins = official_splits(_local(split_dir, directory=True), validate_spec(read_json(spec_raw)), budget)
    if split_pins != provenance["split_files"]:
        raise ValueError("temporal_official_split_pin")
    available, duplicates, inventory_sha = _inventory(
        inventory, provenance["inventory_sha256"], owners, previous["environment"], budget)
    available -= duplicates
    prior = {}
    for row in previous["records"]:
        key = row["sequence_id"], row["frame_index"]
        if key in prior or owners.get(key[0]) != row["split"] or row["split"] not in SPLITS:
            raise ValueError("temporal_previous_split")
        prior[key] = row
    validation = sorted(key for key, row in prior.items() if row["split"] == "validation")
    if (len(validation) != VALIDATION_COUNT or not set(validation) <= available
            or digest(validation) != previous["selection"]["validation_keys_sha256"]):
        raise ValueError("temporal_validation_key_pin")
    train = {key for key in available if owners[key[0]] == "train"}
    candidates = contiguous_clips(train, seed=seed, width=CLIP)
    clips = sorted(clip for _, clip in candidates[:TRAIN_COUNT // CLIP])
    train_keys = [(sequence, index) for sequence, start in clips for index in range(start, start + CLIP)]
    rows = []
    for sequence, frame in train_keys + validation:
        key = sequence, frame
        previous_row = prior.get(key)
        paths = _nominal_paths(sequence, frame)
        files = previous_row["files"] if previous_row else {role: {"path": path} for role, path in paths.items()}
        if set(files) != set(ROLES) or any(
            identify(files[role]["path"], "pose" if role == "pose" else "rgb") != (sequence, frame, role)
            for role in ROLES
        ):
            raise ValueError("temporal_role_path")
        rows.append({"key": [sequence, frame], "split": owners[sequence], "files": files})
    return seal({
        "schema": SCHEMA, "status": "metadata_only" if len(train_keys) == TRAIN_COUNT else "insufficient_train_clips",
        "requested_train_frames": TRAIN_COUNT, "candidate_train_disjoint_clips": len(candidates),
        "seed": seed, "dataset_version": previous["dataset_version"],
        "environment": previous["environment"], "prior_manifest_sha256": previous_sha256,
        "inventory_sha256": inventory_sha, "split_files": split_pins, "spec_sha256": provenance["spec_sha256"],
        "calibration_sha256": provenance["calibration_sha256"], "motion_map_sha256": provenance["motion_map_sha256"],
        "validation_keys_sha256": digest(validation), "records": rows,
        "train_clip_starts": [list(clip) for clip in clips], "clip_frames": CLIP,
        "validation_disjoint_frame_number_clips": len(contiguous_clips(validation, seed=seed, width=CLIP)),
        "train_unique_frames": len(train_keys), "validation_unique_frames": len(validation),
        "excluded_inventory_duplicates": len(duplicates),
        "timestamp": None, "camera_pose": None, "train_ready": False, "temporal_ready": False,
        "blockers": list(BLOCKERS),
    })


def window_keys(sequence, frame, available, *, clip_start=0, width=CLIP):
    """Causal frame-number indexing only. Gaps and clip boundaries reset history."""
    keys = []
    for index in range(frame, max(frame - width, clip_start - 1), -1):
        key = sequence, index
        if key not in available:
            break
        keys.append(key)
    return [None] * (width - len(keys)) + list(reversed(keys))
