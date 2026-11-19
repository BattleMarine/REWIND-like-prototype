"""Read the fixed second metadata DB after a changes-only verification."""

import argparse
from collections import Counter
from contextlib import closing, contextmanager
from itertools import groupby
import json
from pathlib import Path
import shutil
import signal
import sqlite3
import tempfile
import time
import zipfile

from rewind_archive.contracts import canonical
from rewind_archive.matching import identify
from rewind_prepare.converter import official_splits
from rewind_prepare.spec import validate_spec
from rewind_v1.v2_contract import SCENARIOS, SplitReader
from scripts.list_v2_drive_metadata import Drive, FIELDS, allowed_removal
from scripts.produce_v2_inputs import CHECKSUM, identity, pinned_json
from scripts.resume_v2_drive_metadata import FAILED_PARENT, REMOVED, sha_file

ZIP_SHA = "0c0e685a079c68f7ac6c51874d45dc2e5feaad3120cc03cde0a0e3006e03800a"
DB_SHA = "d61aacaaecf38c427d6c62d31668edbb1be9d7bb2ce1fa5b1db0afd2a961dfc5"
ZIP_BYTES, DB_BYTES = 61030492, 380055552
SPEC_SHA = "32f1cfda25bc2b05745d2b534f0fd21eec0defbd379cd1d865c152e52c74b187"
MAX_REQUESTS, MAX_BYTES, MAX_SECONDS = 16, 8 * 1024**2, 120
ROLES = ("left", "right", "pose")


def read_only(path):
    return sqlite3.connect(path.resolve().as_uri() + "?mode=ro&immutable=1", uri=True)


def extract_db(archive, destination):
    if archive.is_symlink() or archive.stat().st_size != ZIP_BYTES or sha_file(archive) != ZIP_SHA:
        raise ValueError("v2_fence_archive_pin")
    if shutil.disk_usage(destination.parent).free < DB_BYTES * 2:
        raise ValueError("v2_fence_local_disk_space")
    with zipfile.ZipFile(archive) as z:
        if sorted(z.namelist()) != ["listing.stderr.txt", "listing.stdout.txt", "metadata.sqlite"]:
            raise ValueError("v2_fence_archive_members")
        if z.getinfo("metadata.sqlite").file_size != DB_BYTES:
            raise ValueError("v2_fence_db_size")
        with z.open("metadata.sqlite") as incoming, destination.open("xb") as output:
            shutil.copyfileobj(incoming, output, 1024**2)
    if sha_file(destination) != DB_SHA:
        raise ValueError("v2_fence_db_pin")


def db_contract(db, reference):
    spec = validate_spec(pinned_json(reference / "spec.json", SPEC_SHA, 1024**2))
    owners, split_pins = official_splits(reference / "split", spec, SplitReader())
    binding_raw, = db.execute("SELECT value FROM binding WHERE slot=1").fetchone()
    roots = json.loads(binding_raw)["roots"]
    expected = {"schema": "rewind_v2_drive_page_db_v1", "roots": roots,
                "pins": {"spec_file_sha256": SPEC_SHA, "official_splits": split_pins}, "fields": FIELDS}
    if binding_raw != canonical(expected) or set(roots) != {"pose", "rgb"}:
        raise ValueError("v2_fence_db_binding")
    if db.execute("PRAGMA quick_check").fetchall() != [("ok",)]:
        raise ValueError("v2_fence_db_integrity")
    if db.execute("SELECT kind,id FROM queue WHERE path='' ORDER BY kind").fetchall() != sorted(roots.items()):
        raise ValueError("v2_fence_root_binding")
    if db.execute("SELECT count(*) FROM queue").fetchone() != (11262,) or db.execute("SELECT 1 FROM queue WHERE done!=1 OR token IS NOT NULL LIMIT 1").fetchone():
        raise ValueError("v2_fence_incomplete_queue")
    if db.execute("SELECT count(*) FROM entries").fetchone() != (408033,):
        raise ValueError("v2_fence_db_entry_count")
    if db.execute("SELECT requests,response_bytes FROM usage").fetchall() != [(2417, 33671553), (8913, 137413582)]:
        raise ValueError("v2_fence_db_usage")
    if db.execute("SELECT 1 FROM completion").fetchone():
        raise ValueError("v2_fence_db_unexpected_completion")
    fences = db.execute("SELECT corpus,token,status FROM fences").fetchall()
    if len(fences) != 1 or fences[0][0] != "" or not fences[0][1] or fences[0][2] != "changed":
        raise ValueError("v2_fence_original_token")
    if any(db.execute(f"SELECT 1 FROM {table} WHERE id=? LIMIT 1", (REMOVED,)).fetchone() for table in ("queue", "entries")):
        raise ValueError("v2_fence_removed_object_observed")
    return owners, fences[0][0], fences[0][1]


def verify_changes(changes_page, corpus, initial_token, report, *, clock=time.monotonic):
    start = clock()
    token = initial_token
    consumed = {token}
    observed = None

    def check():
        if clock() - start >= MAX_SECONDS:
            raise TimeoutError("v2_fence_wall_limit")

    report.update(requests=0, response_bytes=0, change_count=0)
    try:
        while report["requests"] < MAX_REQUESTS and report["response_bytes"] < MAX_BYTES:
            check()
            report["requests"] += 1
            available = MAX_BYTES - report["response_bytes"]
            try:
                page, used = changes_page(corpus, token, check, available)
            except (ValueError, TimeoutError):
                raise
            except Exception as error:
                # HTTP error URLs can contain the old page cursor.
                raise RuntimeError("v2_fence_remote_" + type(error).__name__) from None
            if type(used) is not int or not 0 <= used <= available:
                raise ValueError("v2_fence_bytes_limit")
            report["response_bytes"] += used
            check()
            rows = page["changes"]
            if not isinstance(rows, list):
                raise ValueError("v2_fence_invalid_changes")
            for row in rows:
                report["change_count"] += 1
                if observed is not None or not isinstance(row, dict) or not allowed_removal(row, REMOVED, FAILED_PARENT):
                    change = row if isinstance(row, dict) else {}
                    change_type = change.get("changeType")
                    report["first_unexpected_change"] = {
                        "kind": "additional_change" if observed is not None else "unapproved_change",
                        "target_matches_approved_folder": change.get("fileId") == REMOVED,
                        "change_type": change_type if change_type in ("file", "drive") else "unknown",
                        "removed": change["removed"] if type(change.get("removed")) is bool else None,
                    }
                    raise ValueError("v2_fence_unexpected_change")
                observed = row
            following, end = page.get("nextPageToken"), page.get("newStartPageToken")
            if following is None and isinstance(end, str) and end and len(end) <= 16384:
                if observed is None:
                    raise ValueError("v2_fence_expected_removal_missing")
                report.update(allowed_change=observed, corpus=corpus, pagination_complete=True)
                return
            if end is not None or not isinstance(following, str) or not following or len(following) > 16384 or following in consumed:
                raise ValueError("v2_fence_incomplete_or_repeated_cursor")
            consumed.add(following)
            token = following
        raise TimeoutError("v2_fence_request_or_response_limit")
    finally:
        report["wall_seconds"] = clock() - start


def compare(db, owners, work):
    counts = {s: {split: Counter(dict.fromkeys(
        ("left", "right", "pose", "frame_keys", "clean_triplets", "missing_role_keys", "duplicate_role_keys",
         "missing_sha256_files", "missing_sha256_bytes", "missing_sequence_keys"), 0))
        for split in ("train", "validation")} for s in SCENARIOS}
    sequences = set()
    with tempfile.TemporaryDirectory(dir=work) as temp, closing(sqlite3.connect(Path(temp)/"keys.sqlite")) as index:
        index.executescript("""PRAGMA cache_size=-16384; PRAGMA temp_store=FILE;
            CREATE TABLE frames(seq TEXT, frame INTEGER, role TEXT, path TEXT, PRIMARY KEY(seq,frame,role,path)) WITHOUT ROWID;""")
        with index:
            for kind, path, object_id, raw in db.execute("SELECT kind,path,id,row FROM entries ORDER BY kind,path"):
                row = json.loads(raw)
                seq, frame, role = identify(path, kind)
                scene = seq.split("/")[0]
                split = owners.get(seq)
                if (kind not in ("rgb", "pose") or scene not in counts or split not in counts[scene]
                        or (role == "pose") != (kind == "pose")
                        or (row["path"], row["file_id"], row["scenario"], row["role"]) != (path, object_id, scene, role)):
                    raise ValueError("v2_fence_local_row_membership")
                _, _, size = identity(row)
                cell = counts[scene][split]
                cell[role] += 1
                checksum = row["sha256"]
                if checksum is None and row["checksum_source"] == "missing":
                    cell["missing_sha256_files"] += 1
                    cell["missing_sha256_bytes"] += size
                elif not isinstance(checksum, str) or not CHECKSUM.fullmatch(checksum) or row["checksum_source"] != "drive_sha256Checksum":
                    raise ValueError("v2_fence_local_checksum_metadata")
                sequences.add(seq)
                index.execute("INSERT INTO frames VALUES (?,?,?,?)", (seq, frame, role, path))
        issues_path = work / "issues.jsonl"
        with issues_path.open("xb") as issues:
            rows = index.execute("SELECT seq,frame,role,path FROM frames ORDER BY seq,frame,role,path")
            for (seq, frame), group in groupby(rows, key=lambda r: r[:2]):
                cell = counts[seq.split("/")[0]][owners[seq]]
                cell["frame_keys"] += 1
                paths = {role: [] for role in ROLES}
                for _, _, role, path in group:
                    paths[role].append(path)
                missing = [role for role in ROLES if not paths[role]]
                duplicates = {role: paths[role] for role in ROLES if len(paths[role]) > 1}
                cell["missing_role_keys"] += bool(missing)
                cell["duplicate_role_keys"] += bool(duplicates)
                if missing or duplicates:
                    issues.write(canonical({"sequence": seq, "frame": frame, "missing": missing, "duplicates": duplicates}) + b"\n")
                else:
                    cell["clean_triplets"] += 1
            for seq, split in sorted(owners.items()):
                scene = seq.split("/")[0]
                if scene in counts and split in counts[scene] and seq not in sequences:
                    counts[scene][split]["missing_sequence_keys"] += 1
                    issues.write(canonical({"sequence": seq, "reason": "no_observed_frame_key"}) + b"\n")
    return counts


@contextmanager
def remote_deadline():
    if not hasattr(signal, "setitimer"):
        yield
        return
    previous = signal.getsignal(signal.SIGALRM)

    def expired(_signal, _frame):
        raise TimeoutError("v2_fence_hard_wall_limit")

    signal.signal(signal.SIGALRM, expired)
    try:
        signal.setitimer(signal.ITIMER_REAL, MAX_SECONDS)
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


def run_check(db, owners, work, changes_page, corpus, token, report):
    with remote_deadline():
        verify_changes(changes_page, corpus, token, report["changes"])
    report["status"] = "change_verified_local_comparison_started"
    report["counts"] = compare(db, owners, work)
    totals = Counter()
    for group in report["counts"].values():
        for cell in group.values():
            totals.update(cell)
    report["totals"] = totals
    failure_keys = ("missing_role_keys", "duplicate_role_keys", "missing_sequence_keys")
    report["frame_correspondence_passed"] = not any(totals[k] for k in failure_keys)
    report["status"] = "comparison_complete"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("verify", "check"))
    for name in ("archive", "reference", "work"):
        parser.add_argument("--" + name, required=True, type=Path)
    args = parser.parse_args()
    args.work.mkdir()
    report = {"status": "started", "parent_zip_sha256": ZIP_SHA, "db_sha256": DB_SHA, "changes": {},
              "files_list_calls": 0, "start_token_requests": 0, "payload_reads": 0,
              "scope": "observed official train/validation frame keys; no decode or absent-from-all-roles inference"}
    try:
        with tempfile.TemporaryDirectory(dir=args.work) as temp:
            path = Path(temp) / "metadata.sqlite"
            extract_db(args.archive, path)
            with closing(read_only(path)) as db:
                owners, corpus, token = db_contract(db, args.reference)
                report["status"] = "verified_archive"
                if args.action == "check":
                    client = Drive()
                    run_check(db, owners, args.work, client.changes, corpus, token, report)
    except BaseException as exc:
        report["status_before_failure"] = report["status"]
        report["status"] = "stopped" if isinstance(exc, KeyboardInterrupt) else "blocked"
        report["reason"] = type(exc).__name__ + ": " + str(exc)[:500]
        raise
    finally:
        (args.work / "report.json").write_bytes(canonical(report))
    print(canonical(report).decode())


if __name__ == "__main__":
    main()
