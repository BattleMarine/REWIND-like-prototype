"""ArchVis 직접 추출본의 T=1 입력 감사와 지연 로딩."""

import argparse
from collections import Counter, defaultdict
import copy
from dataclasses import asdict, dataclass
import os
from pathlib import Path
import stat
import time

from rewind_archive.contracts import PreparationError, canonical, digest
from rewind_archive.matching import identify
from rewind_data.audit import read_json
from rewind_m1.local_rgb import _hash, _no_link, _path, _regular_read, decode_rgb_payload
from .converter import motion_entries, normalize_pose_fields, official_splits, sha, validate_pose_geometry
from .provenance import revision_identity
from .spec import validate_spec


ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "direct_extracted_stereo_v1"
AUDIT_SCHEMA = "archvis_drive_metadata_join_20261109_v1"
SPLITS = ("train", "validation")
ROLES = ("left", "right", "pose")
MIB = 1024**2
IMAGE_SIZE = [1024, 1024]
MAX_MANIFEST_BYTES = 64 * MIB
MAX_FILE_BYTES = 8 * MIB
MAX_BATCH = 4


@dataclass(frozen=True)
class ExtractedLimits:
    max_read_bytes: int
    max_seconds: float
    max_entries: int = 200_000

    def __post_init__(self):
        if (type(self.max_read_bytes) is not int or self.max_read_bytes < 1
                or type(self.max_seconds) not in (int, float)
                or not 0 < self.max_seconds <= 24 * 3600
                or type(self.max_entries) is not int or not 1 <= self.max_entries <= 300_000):
            raise ValueError("extracted_limits")


class _Budget:
    def __init__(self, limits):
        if not isinstance(limits, ExtractedLimits):
            raise ValueError("extracted_limits_required")
        self.limits = limits
        self.started = time.monotonic()
        self.bytes = self.entries = 0

    def check(self):
        if time.monotonic() - self.started > self.limits.max_seconds:
            raise ValueError("extracted_elapsed_limit")

    def entry(self):
        self.check()
        self.entries += 1
        if self.entries > self.limits.max_entries:
            raise ValueError("extracted_directory_entry_limit")

    def read(self, path, maximum=MAX_FILE_BYTES, expected_size=None):
        self.check()
        # The regular-file reader uses one extra byte to detect growth.
        remaining = self.limits.max_read_bytes - self.bytes
        if remaining < 2 or _no_link(path).st_size >= remaining:
            raise ValueError("extracted_read_byte_limit")
        raw = _regular_read(path, min(maximum, remaining - 1), expected_size)
        self.bytes += len(raw)
        self.check()
        return raw


def _local(path, directory=False):
    path = Path(path).absolute()
    for component in (*reversed(path.parents), path):
        _no_link(component)
    if directory and not path.is_dir():
        raise ValueError("extracted_root_directory")
    return path.resolve(strict=True)


def _payload_path(root, relative):
    path = root
    _no_link(root)
    if root.resolve(strict=True) != root:
        raise ValueError("extracted_root_changed")
    for component in _path(relative).split("/"):
        path /= component
        _no_link(path)
    return path


def _source_index(root, environment, kind, owners, aliases, duplicates, budget, ignored):
    env_root = _payload_path(root, environment)
    if not env_root.is_dir():
        raise ValueError("extracted_environment_directory")

    def walk(path, depth):
        with os.scandir(path) as entries:
            for entry in entries:
                budget.entry()
                child = Path(entry.path)
                info = _no_link(child)
                if stat.S_ISDIR(info.st_mode):
                    if depth < 8:
                        yield from walk(child, depth + 1)
                    else:
                        ignored["outside_layout"] += 1
                elif stat.S_ISREG(info.st_mode):
                    yield child.relative_to(root).as_posix()
                else:
                    raise ValueError("extracted_special_file")

    for relative in walk(env_root, 1):
        try:
            sequence, frame, role = identify(relative, kind)
        except PreparationError as exc:
            if exc.reason != "unrecognized_member_name":
                raise
            ignored["outside_layout"] += 1
            continue
        if sequence not in owners:
            canonical_sequence = aliases.get(sequence.casefold())
            if canonical_sequence is not None and owners[canonical_sequence] in SPLITS:
                duplicates.add((canonical_sequence, frame))
            ignored["outside_official_split"] += 1
            continue
        if owners[sequence] == "test":
            ignored["test"] += 1
            continue
        yield (sequence, frame), role, _path(relative)


def _inventory(path, expected_sha256, owners, environment, budget):
    if path is None:
        if expected_sha256 is not None:
            raise ValueError("extracted_inventory_pin_without_file")
        return set(), set(), None
    raw = budget.read(_local(path))
    if not _hash(expected_sha256) or sha(raw) != expected_sha256:
        raise ValueError("extracted_inventory_pin")
    doc = read_json(raw)
    if doc.get("schema") != AUDIT_SCHEMA or doc.get("failed") or doc.get("capped"):
        raise ValueError("extracted_inventory_incomplete")
    expected, duplicates = set(), set()
    for row in doc["sequences"]:
        seq = _path(row["path"])
        if seq.split("/")[0] != environment or owners.get(seq) not in SPLITS:
            continue
        if row["split"] != owners[seq] or row["source"] != "train_val":
            raise ValueError("extracted_inventory_split_mismatch")
        for frame in row["frames"]:
            if type(frame) is not int or not 0 <= frame < 10**12 or (seq, frame) in expected:
                raise ValueError("extracted_inventory_frame_key")
            expected.add((seq, frame))
    for row in doc["duplicates"]:
        seq, sep, frame = row["key"].rpartition("#")
        if row["split"] == "train_val" and owners.get(seq) in SPLITS and seq.split("/")[0] == environment:
            if not sep or not frame.isascii() or not frame.isdecimal() or len(frame) > 12:
                raise ValueError("extracted_inventory_duplicate_key")
            duplicates.add((seq, int(frame)))
    return expected, duplicates, expected_sha256


def _raw_pose(raw, frame):
    doc = read_json(raw)
    if (not isinstance(doc, dict) or type(doc.get("frame")) not in (int, str)
            or str(doc["frame"]) != str(frame) or not isinstance(doc.get("joints"), dict)):
        raise ValueError("extracted_pose_frame_or_joints")
    return doc


def _pose_targets(doc, spec):
    fields, _, _, _ = normalize_pose_fields(doc, spec, Counter())
    wanted = [fields["camera_left_pts2d"], fields["camera_right_pts2d"], fields["device_pts3d"]]
    if not all(any(item["valid_mask"]) for item in wanted):
        raise ValueError("extracted_pose_missing_supervision")
    if any(abs(x) > 3.4028234663852886e38 for item in wanted
           for point in item["coordinates"] if point is not None for x in point):
        raise ValueError("extracted_pose_float32_range")
    return wanted


def _code_identity():
    paths = ("rewind_prepare/extracted.py", "rewind_prepare/converter.py",
             "rewind_prepare/coordinates.py", "rewind_prepare/spec.py", "rewind_prepare/provenance.py",
             "rewind_m1/local_rgb.py", "rewind_archive/contracts.py", "rewind_archive/matching.py",
             "rewind_data/audit.py")
    return {path: sha((ROOT / path).read_bytes()) for path in paths}


def prepare_extracted_t1(rgb_root, pose_root, *, environment, dataset_version, split_dir,
                         motion_map, spec_path, calibration_path, output_dir, limits,
                         train_limit=1000, seed=0, inventory=None, inventory_sha256=None,
                         plan_only=False, expected_plan_sha256=None, on_progress=None):
    """Plan from metadata or audit selected payloads and publish a pinned manifest."""

    if "/" in _path(environment) or not isinstance(dataset_version, str) or not dataset_version.strip() or len(dataset_version) > 128:
        raise ValueError("extracted_dataset_identity")
    if (train_limit is not None and (type(train_limit) is not int or train_limit < 1)
            or type(seed) is not int or not 0 <= seed < 2**64):
        raise ValueError("extracted_selection")
    budget = _Budget(limits)
    roots = {"rgb": _local(rgb_root, directory=True), "pose": _local(pose_root, directory=True)}
    output = _local(Path(output_dir).parent, directory=True) / Path(output_dir).name
    if output.exists() or output.is_symlink() or any(output.is_relative_to(root) for root in roots.values()):
        raise ValueError("extracted_new_output_outside_sources_required")
    spec_raw = budget.read(_local(spec_path), 256 * 1024)
    spec = validate_spec(read_json(spec_raw))
    owners, split_pins = official_splits(_local(split_dir, directory=True), spec, budget)
    calibration_raw = budget.read(_local(calibration_path))
    if sha(calibration_raw) != spec["pinned_sources"]["scaramuzza_sha256"]:
        raise ValueError("extracted_calibration_hash")
    calibration = read_json(calibration_raw)
    if calibration.get("size") != IMAGE_SIZE:
        raise ValueError("extracted_calibration_size")
    expected, duplicates, audit_pin = _inventory(inventory, inventory_sha256, owners, environment, budget)

    indexed, ignored = defaultdict(dict), Counter()
    aliases = {seq.casefold(): seq for seq in owners}
    if len(aliases) != len(owners):
        raise ValueError("extracted_official_split_case_alias")
    for kind, root in roots.items():
        for key, role, path in _source_index(root, environment, kind, owners, aliases, duplicates, budget, ignored):
            if role in indexed[key]:
                duplicates.add(key)
            indexed[key][role] = path
    incomplete = {key: sorted(set(ROLES) - roles.keys()) for key, roles in indexed.items() if set(roles) != set(ROLES)}
    usable = indexed.keys() - duplicates - incomplete.keys()
    if on_progress is not None:
        on_progress("inventory", entries=budget.entries, usable=usable, duplicates=duplicates,
                    incomplete=incomplete, ignored=ignored)
    train = sorted(key for key in usable if owners[key[0]] == "train")
    validation = sorted(key for key in usable if owners[key[0]] == "validation")
    if not train or not validation or (train_limit is not None and len(train) < train_limit):
        raise ValueError("extracted_insufficient_train_or_validation")
    motion_raw = budget.read(_local(motion_map))
    motion_doc = read_json(motion_raw)
    eligible_sequences = {sequence for sequence, _ in usable}
    motions = motion_entries(motion_doc, eligible_sequences, owners)
    if train_limit is not None:
        train = sorted(sorted(train, key=lambda key: (digest([seed, *key]), key))[:train_limit])
    selected = train + validation
    plan = None
    if plan_only or expected_plan_sha256 is not None:
        sizes = {}
        counts = Counter()
        payload_bytes = Counter()
        for sequence, frame in selected:
            counts[owners[sequence]] += 1
            for role in ROLES:
                budget.check()
                relative = indexed[sequence, frame][role]
                root = roots["pose" if role == "pose" else "rgb"]
                info = _no_link(_payload_path(root, relative))
                maximum = spec["limits"]["max_file_bytes"] if role == "pose" else MAX_FILE_BYTES
                if not stat.S_ISREG(info.st_mode) or not 1 <= info.st_size <= maximum:
                    raise ValueError("extracted_planned_file_size")
                sizes[sequence, frame, role] = info.st_size
                payload_bytes[owners[sequence]] += info.st_size
        body = {"schema": "direct_extracted_selection_plan_v1", "dataset_version": dataset_version,
                "environment": environment, "roots": {k: str(v) for k, v in roots.items()},
                "train_limit": train_limit, "seed": seed, "selection": [
                    [seq, frame, owners[seq], [[role, indexed[seq, frame][role], sizes[seq, frame, role]]
                                             for role in ROLES]] for seq, frame in selected],
                "validation_keys_sha256": digest(validation),
                "metadata_pins": {"spec": sha(spec_raw), "calibration": sha(calibration_raw),
                                  "split_files": split_pins, "motion_map": sha(motion_raw), "inventory": audit_pin},
                "counts": dict(counts), "payload_bytes": dict(payload_bytes), "metadata_bytes": budget.bytes,
                "directory_entries": budget.entries, "selected_file_stats": len(sizes),
                "duplicates": [list(key) for key in sorted(duplicates)],
                "inventory_missing_eligible": [list(key) for key in sorted(expected - duplicates - usable)],
                "inventory_extra_eligible": [list(key) for key in sorted(usable - expected)] if audit_pin else [],
                "payload_validation": "not_run"}
        plan = {**body, "content_sha256": digest(body)}
        if plan_only:
            return plan
        if plan["content_sha256"] != expected_plan_sha256:
            raise ValueError("extracted_selection_plan_changed")

    from PIL import Image

    records, exact_payloads = [], {}
    residual_max, residual_counts = {}, Counter()
    for sequence, frame in selected:
        payloads = {}
        for role in ROLES:
            relative = indexed[sequence, frame][role]
            root = roots["pose" if role == "pose" else "rgb"]
            if on_progress is not None:
                on_progress("file", key=(sequence, frame), role=role, path=relative,
                            read_bytes=budget.bytes, pairs_verified=len(records), pairs_total=len(selected))
            raw = budget.read(_payload_path(root, relative),
                              spec["limits"]["max_file_bytes"] if role == "pose" else MAX_FILE_BYTES,
                              sizes[sequence, frame, role] if plan is not None else None)
            payloads[role] = {"path": relative, "size_bytes": len(raw), "sha256": sha(raw)}
            if role == "pose":
                doc = _raw_pose(raw, frame)
                residuals = defaultdict(list)
                validate_pose_geometry(doc, spec, calibration, residuals)
                if not residuals["projection_px"]:
                    raise ValueError("extracted_pose_no_reprojection")
                _pose_targets(doc, spec)
                for name, values in residuals.items():
                    if values:
                        residual_max[name] = max(residual_max.get(name, 0.0), max(values))
                        residual_counts[name] += len(values)
            else:
                decode_rgb_payload(raw, IMAGE_SIZE, 1024**2, Image)
            budget.check()
        split = owners[sequence]
        signature = tuple(payloads[role]["sha256"] for role in ROLES)
        if exact_payloads.setdefault(signature, split) != split:
            raise ValueError("extracted_triplet_content_split_overlap")
        records.append({"sequence_id": sequence, "frame_index": frame, "split": split,
                        "motion_id": motions[sequence]["motion_id"], "files": payloads})
        if on_progress is not None:
            on_progress("pair", record=records[-1], read_bytes=budget.bytes,
                        pairs_verified=len(records), pairs_total=len(selected))

    body = {
        "schema": SCHEMA, "source_kind": "direct_extracted_files", "dataset_version": dataset_version,
        "environment": environment, "diagnostic_only": True, "train_ready": False,
        "temporal_ready": False, "distance_unit": "mm", "coordinate_frame": "device_relative",
        "image_size": IMAGE_SIZE, "joint_names": spec["joint_names"],
        "preprocessing_version": spec["preprocessing_version"],
        "motion_identities": {seq: motions[seq] for seq in sorted({seq for seq, _ in selected})},
        "selection": {"train_limit": train_limit, "seed": seed, "train_ordering": "seeded_key_sha256",
                      "validation": "all_eligible_official", "validation_keys_sha256": digest(validation)},
        "provenance": {"spec_sha256": sha(spec_raw), "calibration_sha256": sha(calibration_raw),
                       "split_files": split_pins, "motion_map_sha256": sha(motion_raw),
                       "inventory_sha256": audit_pin, "code_sha256": _code_identity(),
                       "revision": revision_identity(ROOT)},
        "audit": {"selected": dict(Counter(row["split"] for row in records)),
                  "eligible": dict(Counter(owners[seq] for seq, _ in usable)),
                  "missing_official_sequences": sorted(seq for seq, split in owners.items()
                      if split in SPLITS and seq.split("/")[0] == environment and seq not in eligible_sequences),
                  "inventory_pairs_missing_locally": [list(key) for key in sorted(expected - indexed.keys())],
                  "duplicates": [list(key) for key in sorted(duplicates)],
                  "incomplete": [[*key, incomplete[key]] for key in sorted(incomplete)],
                  "ignored_entries": dict(ignored), "directory_entries": budget.entries,
                  "read_bytes": budget.bytes, "limits": asdict(limits),
                  "max_geometry_residuals": residual_max, "geometry_checks": dict(residual_counts),
                  "image_pose_alignment": "key_and_numeric_annotation_only",
                  "unit_evidence_status": spec["coordinates"]["unit_evidence_status"],
                  "original_motion_status": dict(Counter(motions[seq]["identity_status"] for seq in eligible_sequences)),
                  "timing": "unverified"},
        "records": records,
    }
    if plan is not None:
        body["provenance"]["selection_plan_sha256"] = plan["content_sha256"]
    manifest = {**body, "content_sha256": digest(body)}
    encoded = canonical(manifest) + b"\n"
    budget.check()
    if len(encoded) > MAX_MANIFEST_BYTES:
        raise ValueError("extracted_manifest_size")
    output.mkdir()
    pending = output / "manifest.json.partial"
    with pending.open("xb") as handle:
        handle.write(encoded)
    pending.replace(output / "manifest.json")
    return manifest


class ExtractedStereoT1:
    """Pinned split, native stereo and independently masked 2D/3D supervision."""

    def __init__(self, manifest_path, rgb_root, pose_root, spec_path, *, expected_sha256, split,
                 read_metadata=None):
        if split not in SPLITS:
            raise ValueError("extracted_loader_split")
        read_metadata = read_metadata or _regular_read
        manifest = read_json(read_metadata(_local(manifest_path), MAX_MANIFEST_BYTES))
        body = {k: v for k, v in manifest.items() if k != "content_sha256"}
        if (not _hash(expected_sha256) or manifest.get("content_sha256") != expected_sha256
                or digest(body) != expected_sha256 or manifest.get("schema") != SCHEMA
                or manifest.get("source_kind") != "direct_extracted_files"
                or manifest.get("diagnostic_only") is not True or manifest.get("train_ready") is not False
                or manifest.get("temporal_ready") is not False
                or manifest.get("distance_unit") != "mm" or manifest.get("coordinate_frame") != "device_relative"
                or manifest.get("image_size") != IMAGE_SIZE):
            raise ValueError("extracted_manifest_identity")
        spec_raw = read_metadata(_local(spec_path), 256 * 1024)
        self._spec = validate_spec(read_json(spec_raw))
        if (sha(spec_raw) != manifest["provenance"]["spec_sha256"]
                or self._spec["joint_names"] != manifest["joint_names"]
                or self._spec["preprocessing_version"] != manifest["preprocessing_version"]):
            raise ValueError("extracted_loader_spec")
        keys, owners, motions, paths = set(), {}, {}, set()
        self._rows_by_split = {name: [] for name in SPLITS}
        for row in manifest["records"]:
            seq, frame, owner, motion = (row[k] for k in ("sequence_id", "frame_index", "split", "motion_id"))
            key = seq, frame
            if (owner not in SPLITS or not isinstance(motion, str) or not motion
                    or _path(seq).split("/")[0] != manifest["environment"] or len(seq.split("/")) != 6
                    or type(frame) is not int or not 0 <= frame < 10**12 or key in keys
                    or owners.setdefault(seq, owner) != owner or motions.setdefault(motion, owner) != owner):
                raise ValueError("extracted_loader_key_split_motion")
            keys.add(key)
            if set(row["files"]) != set(ROLES):
                raise ValueError("extracted_loader_roles")
            for role, item in row["files"].items():
                kind = "pose" if role == "pose" else "rgb"
                path = _path(item["path"])
                identity = kind, path.casefold()
                maximum = self._spec["limits"]["max_file_bytes"] if role == "pose" else MAX_FILE_BYTES
                if (identify(path, kind) != (*key, role) or identity in paths or not _hash(item["sha256"])
                        or type(item["size_bytes"]) is not int or not 1 <= item["size_bytes"] <= maximum):
                    raise ValueError("extracted_loader_payload_identity")
                paths.add(identity)
            self._rows_by_split[owner].append(row)
        self._rows = self._rows_by_split[split]
        if not self._rows or set(owners.values()) != set(SPLITS):
            raise ValueError("extracted_loader_split_empty")
        validation = sorted(key for key in keys if owners[key[0]] == "validation")
        if manifest["selection"]["validation_keys_sha256"] != digest(validation):
            raise ValueError("extracted_loader_validation_key_pin")
        motion_doc = {"schema_version": "explicit_motion_map_v1", "sequences": manifest["motion_identities"]}
        declared = motion_entries(motion_doc, set(owners), owners)
        if any(declared[row["sequence_id"]]["motion_id"] != row["motion_id"] for row in manifest["records"]):
            raise ValueError("extracted_loader_motion_identity")
        self._roots = {"rgb": _local(rgb_root, directory=True), "pose": _local(pose_root, directory=True)}
        self.split = split
        self.manifest_sha256 = expected_sha256
        self._identity = {key: manifest[key] for key in
                          ("dataset_version", "environment", "selection", "provenance", "audit", "joint_names")}
        self._identity["body_joint_names"] = self._spec["body_joint_names"]

    @property
    def identity(self):
        return copy.deepcopy(self._identity)

    def for_split(self, split):
        """Reuse the same validated manifest and roots without rereading payloads."""
        if split not in SPLITS:
            raise ValueError("extracted_loader_split")
        reader = copy.copy(self)
        reader.split, reader._rows = split, self._rows_by_split[split]
        return reader

    def __len__(self):
        return len(self._rows)

    def metadata(self, index):
        if type(index) is not int or not 0 <= index < len(self):
            raise ValueError("extracted_loader_index")
        return {**copy.deepcopy(self._rows[index]), "manifest_sha256": self.manifest_sha256,
                "source_kind": "direct_extracted_files", "time_seconds": None,
                "diagnostic_only": True, "temporal_ready": False, "train_ready": False}

    def _read(self, item, kind):
        raw = _regular_read(_payload_path(self._roots[kind], item["path"]),
                            item["size_bytes"], item["size_bytes"])
        if sha(raw) != item["sha256"]:
            raise ValueError("extracted_payload_hash_changed")
        return raw

    def load_batch(self, indices):
        if not isinstance(indices, (tuple, list)) or not 1 <= len(indices) <= MAX_BATCH:
            raise ValueError("extracted_loader_batch_limit")
        metadata = [self.metadata(i) for i in indices]
        import torch
        from PIL import Image

        images = torch.empty((len(indices), 2, 3, 1024, 1024), dtype=torch.float32, device="cpu")
        targets, masks = [], []
        for i, row in enumerate(metadata):
            files = row["files"]
            doc = _raw_pose(self._read(files["pose"], "pose"), row["frame_index"])
            fields = _pose_targets(doc, self._spec)
            targets.append([torch.tensor([v if v is not None else [0.] * dim for v in field["coordinates"]],
                                         dtype=torch.float32, device="cpu") for field, dim in zip(fields, (2, 2, 3))])
            masks.append([torch.tensor(field["valid_mask"], dtype=torch.bool, device="cpu") for field in fields])
            for view, role in enumerate(("left", "right")):
                raw = self._read(files[role], "rgb")
                pixels = torch.frombuffer(decode_rgb_payload(raw, IMAGE_SIZE, 1024**2, Image), dtype=torch.uint8)
                images[i, view].copy_(pixels.reshape(1024, 1024, 3).permute(2, 0, 1).float().div_(255.0))
        return {"images": images, "gt_xy": torch.stack([torch.stack(row[:2]) for row in targets]),
                "annotation_valid": torch.stack([torch.stack(row[:2]) for row in masks]),
                "gt_xyz_mm": torch.stack([row[2] for row in targets]),
                "target_valid": torch.stack([row[2] for row in masks]), "metadata": metadata}


def main(argv=None):
    parser = argparse.ArgumentParser(description="Audit a local ArchVis stereo/pose subset into a pinned T=1 manifest")
    for name in ("rgb-root", "pose-root", "split-dir", "motion-map", "spec-path", "calibration-path", "output-dir"):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--environment", default="ArchVisInterior_ArchVis_RT")
    parser.add_argument("--dataset-version", required=True)
    parser.add_argument("--max-read-bytes", required=True, type=int)
    parser.add_argument("--max-seconds", required=True, type=float)
    parser.add_argument("--max-entries", type=int, default=200_000)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--train-limit", type=int, default=1000)
    selection.add_argument("--full-train", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--inventory", type=Path)
    parser.add_argument("--inventory-sha256")
    args = vars(parser.parse_args(argv))
    args["limits"] = ExtractedLimits(*(args.pop(k) for k in ("max_read_bytes", "max_seconds", "max_entries")))
    if args.pop("full_train"):
        args["train_limit"] = None
    manifest = prepare_extracted_t1(**args)
    print(manifest["content_sha256"])


if __name__ == "__main__":
    main()
