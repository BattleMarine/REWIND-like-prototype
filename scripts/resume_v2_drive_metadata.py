"""Verify and resume the fixed 2026-11-19 partial metadata result."""

import argparse
from contextlib import closing
import hashlib
import json
import os
from pathlib import Path
import shutil
import sqlite3
import tempfile
import time
import zipfile

from rewind_archive.contracts import canonical
from rewind_prepare.converter import official_splits
from rewind_prepare.spec import validate_spec
from rewind_v1.v2_contract import SplitReader
from scripts.list_v2_drive_metadata import Drive, FIELDS, collect, export
from scripts.produce_v2_inputs import pinned_json

ZIP_SHA = "ffd3c77b03d1bff8aad209d3f9e6f07ee555398c863e644c1f861cc140e919bf"
DB_SHA = "b8f03ce299ac71ef5133d1a770d3274ee3a73f9c8c85272e7db57c52cab17027"
DB_BYTES = 67805184
REMOVED = "1it9nMyJ-TxZeE3hB_aWDAh2Xcux9aBU9"
FAILED_PARENT = "1atCfluTn9ehgJQiv8myr0LbcORNhIN_m"
PRIOR_USAGE = [(2417, 33671553)]
TOTAL_SECONDS = 14400
RECOVERY_RESERVE = 300


def sha_file(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def validate_db(path, spec_path, spec_sha, splits):
    spec = validate_spec(pinned_json(spec_path, spec_sha, 1024**2))
    owners, split_pins = official_splits(splits, spec, SplitReader())
    pins = {"spec_file_sha256": spec_sha, "official_splits": split_pins}
    with closing(sqlite3.connect(path.resolve().as_uri() + "?mode=ro&immutable=1", uri=True)) as db:
        if db.execute("PRAGMA integrity_check").fetchall() != [("ok",)]:
            raise ValueError("v2_resume_sqlite_integrity")
        binding_raw, = db.execute("SELECT value FROM binding WHERE slot=1").fetchone()
        binding = json.loads(binding_raw)
        roots = binding["roots"]
        if binding_raw != canonical({"schema": "rewind_v2_drive_page_db_v1", "roots": roots, "pins": pins, "fields": FIELDS}):
            raise ValueError("v2_resume_binding")
        if set(roots) != {"pose", "rgb"} or db.execute("SELECT kind,id FROM queue WHERE path='' ORDER BY kind").fetchall() != sorted(roots.items()):
            raise ValueError("v2_resume_root_binding")
        if db.execute("SELECT requests,response_bytes FROM usage").fetchall() != PRIOR_USAGE:
            raise ValueError("v2_resume_usage")
        if db.execute("SELECT count(*) FROM entries").fetchone() != (79345,) or db.execute("SELECT count(*) FROM queue").fetchone() != (2418,):
            raise ValueError("v2_resume_cursor_counts")
        if db.execute("SELECT corpus,status FROM fences WHERE token IS NOT NULL").fetchall() != [("", "pending")]:
            raise ValueError("v2_resume_original_fence")
        if db.execute("SELECT 1 FROM completion").fetchone():
            raise ValueError("v2_resume_already_complete")
        if db.execute("SELECT id,token FROM queue WHERE done=0 ORDER BY kind,path LIMIT 1").fetchone() != (FAILED_PARENT, None):
            raise ValueError("v2_resume_original_failed_cursor")
        if any(db.execute(f"SELECT 1 FROM {table} WHERE id=?", (REMOVED,)).fetchone() for table in ("queue", "entries")):
            raise ValueError("v2_resume_removed_id_observed")
    return roots, owners, pins


def hydrate(archive, db_path, spec, spec_sha, splits):
    if archive.is_symlink() or archive.stat().st_size != 11621258 or sha_file(archive) != ZIP_SHA:
        raise ValueError("v2_resume_original_archive_pin")
    if db_path.exists() or db_path.is_symlink():
        raise FileExistsError(db_path)
    with tempfile.TemporaryDirectory(dir=db_path.parent) as temporary:
        candidate = Path(temporary) / "metadata.sqlite"
        with zipfile.ZipFile(archive) as bundle:
            if sorted(bundle.namelist()) != ["listing.stderr.txt", "listing.stdout.txt", "metadata.sqlite"]:
                raise ValueError("v2_resume_original_archive_members")
            if bundle.getinfo("metadata.sqlite").file_size != DB_BYTES:
                raise ValueError("v2_resume_original_db_size")
            with bundle.open("metadata.sqlite") as src, candidate.open("xb") as dst:
                shutil.copyfileobj(src, dst, length=1024**2)
        if sha_file(candidate) != DB_SHA:
            raise ValueError("v2_resume_original_db_pin")
        result = validate_db(candidate, spec, spec_sha, splits)
        candidate.rename(db_path)
    return result


def remaining_seconds(evidence_path, evidence_sha):
    if evidence_path is None or evidence_sha is None:
        raise ValueError("v2_resume_prior_wall_evidence_required")
    evidence = pinned_json(evidence_path, evidence_sha, 1024**2)
    prior = evidence.get("prior_wall_upper_seconds")
    if (evidence.get("schema") != "v2_metadata_prior_wall_upper_v1" or evidence.get("result_zip_sha256") != ZIP_SHA
            or type(prior) is not int or not 1 <= prior < TOTAL_SECONDS - RECOVERY_RESERVE
            or not isinstance(evidence.get("basis"), str) or not evidence["basis"].strip()):
        raise ValueError("v2_resume_invalid_prior_wall_evidence")
    return TOTAL_SECONDS - prior - RECOVERY_RESERVE


def save_recovery(source, directory, max_seconds):
    if not 1 <= max_seconds <= RECOVERY_RESERVE:
        raise ValueError("v2_recovery_time_limit")
    if source.is_symlink() or not source.is_file() or directory.is_symlink() or not directory.is_dir():
        raise ValueError("v2_recovery_paths")
    size = source.stat().st_size
    if size < 1 or shutil.disk_usage(directory).free < size:
        raise ValueError("v2_recovery_free_space")
    target = directory / source.name
    if target.exists() or target.is_symlink():
        raise FileExistsError(target)
    deadline = time.monotonic() + max_seconds

    def checked_hash(stream, copy=None):
        digest, total = hashlib.sha256(), 0
        while block := stream.read(1024**2):
            if time.monotonic() >= deadline or total + len(block) > size:
                raise ValueError("v2_recovery_copy_limit")
            total += len(block)
            digest.update(block)
            if copy is not None:
                copy.write(block)
        return total, digest.hexdigest()

    with tempfile.TemporaryDirectory(prefix=".v2-recovery-", dir=directory) as temp:
        staged = Path(temp) / target.name
        with source.open("rb") as stream, staged.open("xb") as copy:
            measured = checked_hash(stream, copy)
            copy.flush()
            os.fsync(copy.fileno())
        with staged.open("rb") as stream:
            checked = checked_hash(stream)
        if checked != measured or checked[0] != size or source.stat().st_size != size:
            raise ValueError("v2_recovery_readback_pin")
        if time.monotonic() >= deadline or target.exists() or target.is_symlink():
            raise ValueError("v2_recovery_publish_blocked")
        staged.rename(target)
    return {"path": str(target), "bytes": size, "sha256": checked[1], "status": "verified"}


def main():
    start = time.monotonic()
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=("verify", "resume"))
    for name in ("archive", "work", "spec", "splits", "elapsed-evidence"):
        p.add_argument("--" + name, type=Path, required=name != "elapsed-evidence")
    p.add_argument("--spec-sha", required=True)
    p.add_argument("--elapsed-evidence-sha")
    p.add_argument("--invocation-overhead-seconds", type=int, default=0)
    args = p.parse_args()
    args.work.mkdir()
    writable = args.work / "metadata.sqlite"
    roots, owners, pins = hydrate(args.archive, writable, args.spec, args.spec_sha, args.splits)
    report = {"status": "verified_partial", "parent_result_zip_sha256": ZIP_SHA, "original_db_sha256": DB_SHA,
              "remaining_requests": 20000 - PRIOR_USAGE[0][0], "remaining_response_bytes": 512*1024**2 - PRIOR_USAGE[0][1],
              "recovery_reserve_seconds": RECOVERY_RESERVE, "elapsed_evidence_sha256": args.elapsed_evidence_sha}
    try:
        if args.action == "resume":
            elapsed = remaining_seconds(args.elapsed_evidence, args.elapsed_evidence_sha)
            if args.invocation_overhead_seconds < 0:
                raise ValueError("v2_resume_invalid_overhead")
            max_seconds = int(elapsed - args.invocation_overhead_seconds - (time.monotonic() - start))
            if max_seconds < 1:
                raise ValueError("v2_resume_wall_exhausted")
            report["remaining_worker_seconds"] = max_seconds
            with closing(sqlite3.connect(writable)) as db:
                report.update(collect(db, Drive(), roots, owners, pins, max_requests=report["remaining_requests"],
                                      max_bytes=report["remaining_response_bytes"], max_seconds=max_seconds,
                                      resume_removal=REMOVED, resume_parent=FAILED_PARENT))
                if report["status"] == "complete":
                    provenance = {key: report[key] for key in ("parent_result_zip_sha256", "original_db_sha256", "elapsed_evidence_sha256")}
                    report["receipt_sha256"] = export(db, args.work / "listing", roots, pins, resume_provenance=provenance)
    except BaseException:
        report["status"] = "stopped_or_failed"
        raise
    finally:
        (args.work / "resume_report.json").write_bytes(canonical(report))
    print(canonical(report).decode())
    return 0 if report["status"] in ("complete", "verified_partial") else 2


if __name__ == "__main__":
    import sys
    if len(sys.argv) == 5 and sys.argv[1] == "save-recovery":
        print(canonical(save_recovery(Path(sys.argv[2]), Path(sys.argv[3]), int(sys.argv[4]))).decode())
    else:
        raise SystemExit(main())
