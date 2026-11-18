"""Build v2 inputs from pinned local inventory evidence; never reads remote payloads."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import tempfile

from rewind_archive.contracts import canonical, digest, safe_path
from rewind_archive.matching import identify
from rewind_m1.local_rgb import _regular_read
from rewind_prepare.converter import official_splits
from rewind_prepare.spec import validate_spec
from rewind_prepare.temporal_selection import seal
from rewind_v1.v2_contract import SCENARIOS, SplitReader, config, plan, validate

MIB = 1024**2
LOADER_CAP = 128 * MIB
ROW_CAP = 16384
LIST_CAP = 512 * MIB
CHECKSUM = re.compile(r"[a-f0-9]{64}\Z")
ROLES = {"left", "right", "pose"}
SOURCES = {"bounded_payload_sha256", "prior_audit_same_object_version"}


def pinned_json(path, expected, limit):
    raw = _regular_read(path, limit)
    if hashlib.sha256(raw).hexdigest() != expected:
        raise ValueError("v2_producer_input_sha")
    return json.loads(raw)


def jsonl(path, expected):
    h = hashlib.sha256()
    if path.is_symlink() or path.stat().st_size > LIST_CAP:
        raise ValueError("v2_producer_list_size")
    with path.open("rb") as stream:
        total = 0
        while raw := stream.readline(ROW_CAP + 1):
            total += len(raw)
            if len(raw) > ROW_CAP or total > LIST_CAP:
                raise ValueError("v2_producer_list_limit")
            h.update(raw)
            yield json.loads(raw)
    if h.hexdigest() != expected:
        raise ValueError("v2_producer_list_sha")


def identity(row):
    object_id, version, size = row["file_id"], row["version"], row["size_bytes"]
    if (
        not isinstance(object_id, str) or not object_id or len(object_id) > 256
        or not isinstance(version, str) or not version.isascii() or not version.isdecimal()
        or len(version) > 32 or type(size) is not int or not 0 < size <= 8 * MIB
    ):
        raise ValueError("v2_metadata_object_identity")
    return object_id, version, size


def reconcile(rows, evidence, owners):
    grouped = {}
    problems = []
    counts = Counter()
    file_ids = set()
    seen_sequences = set()
    for row in rows:
        path = safe_path(row["path"])
        if path != row["path"]:
            raise ValueError("v2_noncanonical_inventory_path")
        seq, frame, role = identify(path, "pose" if path.endswith(".json") else "rgb")
        split = owners.get(seq)
        if seq.split("/")[0] not in SCENARIOS or split not in ("train", "validation"):
            problems.append({"reason": "outside_official_scope", "path": path})
            continue
        seen_sequences.add(seq)
        key = identity(row)
        pair = grouped.setdefault((seq, frame), {})
        if role in pair or key[0] in file_ids:
            problems.append({"reason": "duplicate", "path": path})
            continue
        file_ids.add(key[0])
        if row["scenario"] != seq.split("/")[0] or row["role"] != role:
            raise ValueError("v2_metadata_path_role")
        checksum = row.get("sha256")
        if row["checksum_source"] != ("drive_sha256Checksum" if checksum is not None else "missing"):
            raise ValueError("v2_metadata_checksum_source")
        if checksum is not None and (not isinstance(checksum, str) or not CHECKSUM.fullmatch(checksum)):
            raise ValueError("v2_bad_drive_sha256_checksum")
        prior = evidence.get(key)
        if prior and checksum and prior["sha256"] != checksum:
            problems.append({"reason": "checksum_conflict", "path": path})
        if checksum:
            origin = "drive_sha256Checksum"
        elif prior and prior["path"] == path:
            checksum, origin = prior["sha256"], prior["source"]
        else:
            origin = "missing"
            problems.append({"reason": "sha256_unavailable", "path": path, "size_bytes": key[2], "file_id": key[0], "version": key[1]})
        counts[origin] += 1
        pair[role] = {"path": path, "size_bytes": key[2], "sha256": checksum}

    records = []
    for (seq, frame), files in sorted(grouped.items()):
        absent = ROLES - files.keys()
        if absent:
            problems.append({"reason": "missing_roles", "key": [seq, frame], "roles": sorted(absent)})
        else:
            records.append({"sequence_id": seq, "frame_index": frame, "split": owners[seq], "files": files})
    official = sorted(s for s, split in owners.items() if s.split("/")[0] in SCENARIOS and split in ("train", "validation"))
    problems.extend({"reason": "missing_official_sequence", "sequence": s} for s in sorted(set(official) - seen_sequences))
    return records, official, problems, dict(counts)


def evidence_index(rows):
    result = {}
    for row in rows:
        key = identity(row)
        if key in result or row["source"] not in SOURCES or not CHECKSUM.fullmatch(row["sha256"]):
            raise ValueError("v2_invalid_checksum_evidence")
        if not CHECKSUM.fullmatch(row["evidence_sha256"]):
            raise ValueError("v2_unpinned_checksum_evidence")
        result[key] = row
    return result


def produce(receipt_path, receipt_sha, spec_path, spec_sha, splits, output, evidence_path=None, evidence_sha=None):
    receipt = pinned_json(receipt_path, receipt_sha, MIB)
    if receipt["schema"] != "rewind_v2_metadata_list_receipt_v1" or set(receipt["roots"]) != {"rgb", "pose"}:
        raise ValueError("v2_metadata_receipt_schema")
    if (evidence_path is None) != (evidence_sha is None):
        raise ValueError("v2_checksum_evidence_pin_required")
    spec = validate_spec(pinned_json(spec_path, spec_sha, MIB))
    owners, split_sources = official_splits(splits, spec, SplitReader())
    if receipt.get("reference_pins") != {"spec_file_sha256": spec_sha, "official_splits": split_sources}:
        raise ValueError("v2_metadata_reference_pin")
    evidence = evidence_index(jsonl(evidence_path, evidence_sha)) if evidence_path else {}
    metadata_complete = True

    def rows():
        nonlocal metadata_complete
        for role, listing in sorted(receipt["roots"].items()):
            metadata_complete &= listing["status"] == "complete" and listing["next_page_token_remaining"] is False
            name = listing["name"]
            if Path(name).name != name or name in (".", "..") or not listing["root_id"]:
                raise ValueError("v2_metadata_list_name")
            count = 0
            for row in jsonl(receipt_path.parent / name, listing["sha256"]):
                if row["path"].endswith(".json") != (role == "pose"):
                    raise ValueError("v2_metadata_wrong_root")
                count += 1
                yield row
            if count != listing["rows"]:
                raise ValueError("v2_metadata_list_count")

    records, sequences, problems, origins = reconcile(rows(), evidence, owners)
    if not metadata_complete:
        problems.append({"reason": "metadata_listing_incomplete"})
    inventory = seal({
        "schema": "rewind_v2_frozen_inventory_v1", "usable_keys": [[r["sequence_id"], r["frame_index"]] for r in records],
        "sequences": sequences, "failed": not metadata_complete, "capped": not metadata_complete,
        "missing": [p for p in problems if p["reason"].startswith("missing")],
        "duplicates": [p for p in problems if p["reason"] == "duplicate"],
        "excluded": [p for p in problems if not p["reason"].startswith("missing") and p["reason"] != "duplicate"],
        "metadata_receipt_file_sha256": receipt_sha,
    })
    provenance = {
        "producer_file_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "metadata_receipt_file_sha256": receipt_sha, "supplemental_evidence_file_sha256": evidence_sha,
        "spec_file_sha256": spec_sha, "checksum_file_counts": origins,
        "collection_change_check": receipt.get("change_check", "not_reported_by_collection"),
        "decode_audit": "not_run", "expected_coverage": "official sequences and union of observed frame keys",
        "all_roles_absent_frame_detection": "not_verified_by_metadata",
    }
    body = {
        "schema": "rewind_v2_audited_manifest_v1", "scenarios": list(SCENARIOS), "coordinates": "device_mm",
        "spec_sha256": digest(spec), "official_splits": split_sources, "records": records,
        "inventory": inventory, "provenance": provenance,
    }
    if not records:
        problems.append({"reason": "no_stereo_pose_pairs"})
    counts = Counter((r["sequence_id"].split("/")[0], r["split"]) for r in records)
    for scene in SCENARIOS:
        for split in ("train", "validation"):
            if not counts[scene, split]:
                problems.append({"reason": "empty_scenario_split", "scenario": scene, "split": split})
    missing_hashes = origins.get("missing", 0)
    missing_in_pairs = sum(p["sha256"] is None for r in records for p in r["files"].values())
    # JSON null is four bytes and a quoted 64-character hash is 66 bytes.
    measured = len(canonical(body)) + len(b',"content_sha256":"' + b'0'*64 + b'"')
    estimate = measured + 62 * missing_in_pairs
    cost = None
    candidate_sha = None
    if problems:
        status, manifest = "blocked_input_evidence", None
    else:
        candidate = validate(seal(body), spec, owners, split_sources)
        candidate_raw = canonical(candidate)
        candidate_sha = hashlib.sha256(candidate_raw).hexdigest()
        if len(candidate_raw) != measured:
            raise ValueError("v2_manifest_size_measurement")
        cost = plan(candidate, spec, config(), output.parent)
        if measured > LOADER_CAP:
            status, manifest = "blocked_loader_cap", None
        else:
            status, manifest = "metadata_pinned", candidate_raw
    if cost is None and records:
        cost = plan({**body, "content_sha256": None}, spec, config(), output.parent)
    report = {
        "schema": "rewind_v2_producer_report_v1", "status": status, "provenance": provenance,
        "candidate_json_bytes": measured, "estimated_json_bytes_with_all_hashes": estimate,
        "candidate_file_sha256": candidate_sha,
        "loader_cap_bytes": LOADER_CAP, "manifest_file_sha256": hashlib.sha256(manifest).hexdigest() if manifest else None,
        "pair_count": len(records), "missing_sha256_files": missing_hashes, "issues": dict(Counter(p["reason"] for p in problems)),
        "observed_pair_counts_by_scenario": {s: {p: counts[s, p] for p in ("train", "validation")} for s in SCENARIOS},
        "missing_sha256_bytes": sum(p["size_bytes"] for p in problems if p["reason"] == "sha256_unavailable"),
        "cost": cost,
    }
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    with tempfile.TemporaryDirectory(dir=output.parent) as temporary:
        stage = Path(temporary) / "ready"
        stage.mkdir()
        (stage / "report.json").write_bytes(canonical(seal(report)))
        (stage / "issues.jsonl").write_bytes(b"".join(canonical(p) + b"\n" for p in problems))
        if manifest:
            (stage / "frozen_inventory.json").write_bytes(canonical(inventory))
            (stage / "audited_manifest.json").write_bytes(manifest)
        stage.rename(output)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("receipt", "spec", "splits", "output", "evidence"):
        parser.add_argument("--" + name, type=Path, required=name != "evidence")
    for name in ("receipt-sha", "spec-sha", "evidence-sha"):
        parser.add_argument("--" + name, required=name != "evidence-sha")
    report = produce(**{
        {"receipt": "receipt_path", "spec": "spec_path", "evidence": "evidence_path"}.get(k, k): v
        for k, v in vars(parser.parse_args()).items()
    })
    print(canonical({k: v for k, v in report.items() if k != "cost"}).decode())
    return 0 if report["status"] == "metadata_pinned" else 2


if __name__ == "__main__":
    raise SystemExit(main())
