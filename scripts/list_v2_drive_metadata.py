"""Explicit Colab Drive metadata listing with a local, transactional page cursor."""

import argparse
from contextlib import closing
import hashlib
import json
from pathlib import Path
import re
import sqlite3
import tempfile
import time

from rewind_archive.contracts import canonical, safe_path
from rewind_archive.matching import identify
from rewind_prepare.converter import official_splits
from rewind_prepare.spec import validate_spec
from rewind_v1.v2_contract import SCENARIOS, SplitReader
from scripts.produce_v2_inputs import MIB, pinned_json

FOLDER = "application/vnd.google-apps.folder"
SHORTCUT = "application/vnd.google-apps.shortcut"
FIELDS = "id,parents,name,mimeType,size,version,modifiedTime,sha256Checksum,trashed"
SCOPE = "https://www.googleapis.com/auth/drive.metadata.readonly"
BASE = "https://www.googleapis.com/drive/v3"
ID = re.compile(r"[A-Za-z0-9_-]{1,256}\Z")


def id_value(value):
    if not isinstance(value, str) or not ID.fullmatch(value):
        raise ValueError("v2_drive_metadata_id")
    return value


def folders(owners, kind):
    result = set()
    for sequence, split in owners.items():
        if split not in ("train", "validation") or sequence.split("/")[0] not in SCENARIOS:
            continue
        leaves = ("json",) if kind == "pose" else ("fisheye_final_image/camera_left", "fisheye_final_image/camera_right")
        for leaf in leaves:
            parts = (sequence + "/" + leaf).split("/")
            result.update("/".join(parts[:i]) for i in range(1, len(parts) + 1))
    return result


class Drive:
    def __init__(self):
        from google.auth import default
        from google.auth.transport.requests import AuthorizedSession
        credentials, _ = default(scopes=[SCOPE])
        self.session = AuthorizedSession(credentials, max_refresh_attempts=0)

    def get(self, suffix, params, check, byte_limit):
        with self.session.get(BASE + suffix, params=params, stream=True, timeout=(10, 25), allow_redirects=False) as response:
            response.raise_for_status()
            if response.status_code != 200:
                raise ValueError("v2_drive_metadata_http_status")
            body = bytearray()
            for block in response.iter_content(65536):
                check()
                if len(body) + len(block) > min(byte_limit, 8 * MIB):
                    raise ValueError("v2_drive_metadata_response_cap")
                body.extend(block)
        return json.loads(body), len(body)

    def root(self, folder, check, byte_limit):
        return self.get("/files/" + id_value(folder), {"fields": "id,mimeType,trashed,driveId", "supportsAllDrives": "true"}, check, byte_limit)

    def page(self, folder, token, check, byte_limit):
        params = {"q": f"'{id_value(folder)}' in parents and trashed=false", "spaces": "drive",
                  "pageSize": "1000", "supportsAllDrives": "true", "includeItemsFromAllDrives": "true",
                  "fields": f"nextPageToken,incompleteSearch,files({FIELDS})"}
        if token:
            params["pageToken"] = token
        return self.get("/files", params, check, byte_limit)

    def start(self, corpus, check, byte_limit):
        params = {"supportsAllDrives": "true"}
        if corpus:
            params["driveId"] = id_value(corpus)
        return self.get("/changes/startPageToken", params, check, byte_limit)

    def changes(self, corpus, token, check, byte_limit):
        params = {"pageToken": token, "pageSize": "1000", "supportsAllDrives": "true",
                  "includeItemsFromAllDrives": "true", "fields": "nextPageToken,newStartPageToken,changes(changeType)"}
        if corpus:
            params["driveId"] = id_value(corpus)
        return self.get("/changes", params, check, byte_limit)


def collect(db, client, roots, owners, pins, *, max_requests, max_seconds, max_bytes):
    if not 1 <= max_requests <= 50000 or not 1 <= max_seconds <= 28800 or not 1 <= max_bytes <= 512 * MIB:
        raise ValueError("v2_metadata_invocation_limits")
    if set(roots) != {"pose", "rgb"}:
        raise ValueError("v2_metadata_roots")
    for value in roots.values():
        id_value(value)
    expected = {kind: folders(owners, kind) for kind in roots}
    binding = canonical({"schema": "rewind_v2_drive_page_db_v1", "roots": roots, "pins": pins, "fields": FIELDS})
    db.executescript("""
      CREATE TABLE IF NOT EXISTS binding (slot INTEGER PRIMARY KEY CHECK(slot=1), value BLOB NOT NULL);
      CREATE TABLE IF NOT EXISTS queue (
        kind TEXT, path TEXT, id TEXT, token TEXT, done INTEGER DEFAULT 0,
        PRIMARY KEY(kind,path), UNIQUE(kind,id)
      );
      CREATE TABLE IF NOT EXISTS seen_tokens (kind TEXT, path TEXT, token TEXT, PRIMARY KEY(kind,path,token));
      CREATE TABLE IF NOT EXISTS entries (
        kind TEXT, path TEXT, id TEXT, row BLOB,
        PRIMARY KEY(kind,path), UNIQUE(kind,id)
      );
      CREATE INDEX IF NOT EXISTS queue_pending ON queue(done,kind,path);
      CREATE TABLE IF NOT EXISTS fences (corpus TEXT PRIMARY KEY, token TEXT, status TEXT);
      CREATE TABLE IF NOT EXISTS completion (slot INTEGER PRIMARY KEY CHECK(slot=1));
      CREATE TABLE IF NOT EXISTS usage (requests INTEGER, response_bytes INTEGER);
    """)
    saved = db.execute("SELECT value FROM binding").fetchone()
    if saved is not None and saved[0] != binding:
        raise ValueError("v2_drive_resume_binding")
    start = time.monotonic()
    requests = received = 0

    def check():
        if time.monotonic() - start >= max_seconds:
            raise TimeoutError("v2_metadata_wall_limit")

    def call(function, *args):
        nonlocal requests, received
        check()
        requests += 1
        data, used = function(*args, check, max_bytes - received)
        received += used
        check()
        return data

    try:
        if saved is None:
            # Root IDs and their folder type are verified before storing any cursor.
            if max_requests < 2 * len(roots):
                raise ValueError("v2_metadata_need_root_and_fence_requests")
            corpora = set()
            for root in roots.values():
                info = call(client.root, root)
                if info.get("id") != root or info.get("mimeType") != FOLDER or info.get("trashed") is not False:
                    raise ValueError("v2_metadata_root_not_folder")
                corpora.add(id_value(info["driveId"]) if info.get("driveId") else "")
            fences = []
            for corpus in sorted(corpora):
                token = call(client.start, corpus)["startPageToken"]
                if not isinstance(token, str) or not token or len(token) > 16384:
                    raise ValueError("v2_metadata_invalid_change_cursor")
                fences.append((corpus, token))
            with db:
                db.execute("INSERT INTO binding VALUES (1,?)", (binding,))
                db.executemany("INSERT INTO queue(kind,path,id) VALUES (?,'',?)", sorted(roots.items()))
                db.executemany("INSERT INTO fences VALUES (?,?,'pending')", fences)
        while requests < max_requests and received < max_bytes:
            pending = db.execute("SELECT kind,path,id,token FROM queue WHERE done=0 ORDER BY kind,path LIMIT 1").fetchone()
            if pending is None:
                break
            kind, prefix, parent, token = pending
            page = call(client.page, parent, token)
            if page.get("incompleteSearch"):
                raise ValueError("v2_drive_incomplete_search")
            following = page.get("nextPageToken")
            if following is not None and (not isinstance(following, str) or not following or len(following) > 16384):
                raise ValueError("v2_drive_invalid_page_cursor")
            with db:
                db.execute("BEGIN IMMEDIATE")
                current = db.execute("SELECT token,done FROM queue WHERE kind=? AND path=?", (kind, prefix)).fetchone()
                if current != (token, 0):
                    raise ValueError("v2_drive_concurrent_cursor_change")
                for item in page["files"]:
                    check()
                    if item.get("trashed") or item.get("parents") != [parent]:
                        raise ValueError("v2_drive_parent_membership")
                    leaf = item["name"]
                    if "/" in leaf or safe_path(leaf) != leaf:
                        raise ValueError("v2_drive_noncanonical_name")
                    path = f"{prefix}/{leaf}" if prefix else leaf
                    mime = item["mimeType"]
                    if path in expected[kind]:
                        if mime != FOLDER:
                            raise ValueError("v2_drive_expected_folder_type")
                        if db.execute("SELECT 1 FROM queue WHERE kind=? AND path=?", (kind, path)).fetchone():
                            raise ValueError("v2_drive_duplicate_folder_path")
                        db.execute("INSERT INTO queue(kind,path,id) VALUES (?,?,?)", (kind, path, id_value(item["id"])))
                    elif mime == FOLDER:
                        continue
                    else:
                        if prefix not in expected[kind]:
                            continue
                        is_payload_folder = prefix.endswith(("/camera_left", "/camera_right", "/json"))
                        if not is_payload_folder:
                            continue
                        if mime == SHORTCUT:
                            raise ValueError("v2_drive_payload_shortcut")
                        seq, _, role = identify(path, kind)
                        if owners.get(seq) not in ("train", "validation") or (role == "pose") != (kind == "pose"):
                            raise ValueError("v2_drive_unexpected_payload")
                        checksum = item.get("sha256Checksum")
                        normalized = {"scenario": seq.split("/")[0], "role": role, "path": path,
                                      "file_id": id_value(item["id"]), "size_bytes": int(item["size"]),
                                      "sha256": checksum, "checksum_source": "drive_sha256Checksum" if checksum else "missing",
                                      "modified_time": item.get("modifiedTime"), "version": item.get("version")}
                        if db.execute("SELECT 1 FROM entries WHERE kind=? AND path=?", (kind, path)).fetchone():
                            raise ValueError("v2_drive_duplicate_payload_path")
                        db.execute("INSERT INTO entries VALUES (?,?,?,?)", (kind, path, normalized["file_id"], canonical(normalized)))
                if following:
                    db.execute("INSERT INTO seen_tokens VALUES (?,?,?)", (kind, prefix, following))
                db.execute("UPDATE queue SET token=?,done=? WHERE kind=? AND path=?", (following, int(following is None), kind, prefix))
        pending = db.execute("SELECT count(*) FROM queue WHERE done=0").fetchone()[0]
        while not pending and requests < max_requests and received < max_bytes:
            if db.execute("SELECT 1 FROM fences WHERE status='changed'").fetchone():
                break
            fence = db.execute("SELECT corpus,token FROM fences WHERE status='pending' ORDER BY corpus LIMIT 1").fetchone()
            if fence is None:
                break
            corpus, token = fence
            page = call(client.changes, corpus, token)
            following = page.get("nextPageToken")
            if page["changes"]:
                status = "changed"
            elif following and following != token:
                status = "pending"
            elif not following and page.get("newStartPageToken"):
                status = "verified"
            else:
                raise ValueError("v2_metadata_incomplete_change_fence")
            with db:
                cur = db.execute("UPDATE fences SET token=?,status=? WHERE corpus=? AND token=? AND status='pending'",
                                 (following or token, status, corpus, token))
                if cur.rowcount != 1:
                    raise ValueError("v2_metadata_concurrent_fence")
    except TimeoutError:
        pass
    finally:
        with db:
            db.execute("INSERT INTO usage VALUES (?,?)", (requests, received))
    pending = db.execute("SELECT count(*) FROM queue WHERE done=0").fetchone()[0]
    missing = {k: sorted(expected[k] - {r[0] for r in db.execute("SELECT path FROM queue WHERE kind=?", (k,))}) for k in roots}
    initialized = db.execute("SELECT 1 FROM binding").fetchone() is not None
    fences = [row[0] for row in db.execute("SELECT status FROM fences")]
    if "changed" in fences:
        status = "source_changed"
    elif pending or not initialized or not fences or "pending" in fences:
        status = "partial"
    elif any(missing.values()):
        status = "missing_folders"
    else:
        status = "complete"
        with db:
            db.execute("INSERT OR IGNORE INTO completion VALUES (1)")
    return {"status": status, "pending_folders": pending, "missing_folders": missing,
            "requests_this_call": requests, "metadata_response_bytes_this_call": received}


def export(db, output, roots, pins):
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    binding = db.execute("SELECT value FROM binding").fetchone()
    expected = canonical({"schema": "rewind_v2_drive_page_db_v1", "roots": roots, "pins": pins, "fields": FIELDS})
    if binding != (expected,) or not db.execute("SELECT 1 FROM completion").fetchone():
        raise ValueError("v2_metadata_export_requires_complete_cursor")
    with tempfile.TemporaryDirectory(dir=output.parent) as temporary:
        stage = Path(temporary) / "ready"
        stage.mkdir()
        listings = {}
        for kind, root_id in sorted(roots.items()):
            name = kind + ".jsonl"
            count, h = 0, hashlib.sha256()
            with (stage / name).open("xb") as stream:
                for (row,) in db.execute("SELECT row FROM entries WHERE kind=? ORDER BY path,id", (kind,)):
                    data = row + b"\n"
                    stream.write(data)
                    h.update(data)
                    count += 1
            listings[kind] = {"name": name, "root_id": root_id, "rows": count, "sha256": h.hexdigest(),
                              "status": "complete", "next_page_token_remaining": False}
        receipt = {"schema": "rewind_v2_metadata_list_receipt_v1", "roots": listings, "reference_pins": pins,
                   "payload_downloads": 0, "scope": "official train/validation folder traversal",
                   "change_check": "no visible change entries through final corpus fence"}
        raw = canonical(receipt)
        (stage / "receipt.json").write_bytes(raw)
        stage.rename(output)
    return hashlib.sha256(raw).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("spec", "splits", "db", "output"):
        parser.add_argument("--" + name, type=Path, required=name != "output")
    for name in ("spec-sha", "rgb-root-id", "pose-root-id"):
        parser.add_argument("--" + name, required=True)
    for name in ("max-requests", "max-seconds", "max-bytes"):
        parser.add_argument("--" + name, required=True, type=int)
    args = parser.parse_args()
    spec = validate_spec(pinned_json(args.spec, args.spec_sha, MIB))
    owners, split_pins = official_splits(args.splits, spec, SplitReader())
    roots = {"rgb": args.rgb_root_id, "pose": args.pose_root_id}
    pins = {"spec_file_sha256": args.spec_sha, "official_splits": split_pins}
    with closing(sqlite3.connect(args.db)) as db:
        report = collect(db, Drive(), roots, owners, pins, max_requests=args.max_requests,
                         max_seconds=args.max_seconds, max_bytes=args.max_bytes)
        if args.output is not None and report["status"] == "complete":
            report["receipt_sha256"] = export(db, args.output, roots, pins)
    print(canonical(report).decode())
    return 0 if report["status"] == "complete" else 2


if __name__ == "__main__":
    raise SystemExit(main())
