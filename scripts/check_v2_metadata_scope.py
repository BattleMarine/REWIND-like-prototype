"""Changes-only inventory gate with independently pinned historical ancestry evidence."""

import argparse
from collections import Counter
from contextlib import closing
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import tempfile
import time

from rewind_archive.contracts import canonical
from scripts import check_v2_metadata_fence as parent
from scripts.list_v2_drive_metadata import Drive, FOLDER, SHORTCUT, allowed_removal, id_value
from scripts.produce_v2_inputs import pinned_json
from scripts.resume_v2_drive_metadata import FAILED_PARENT, REMOVED
from scripts.resume_v2_drive_metadata import sha_file

FILE_FIELDS = "id,parents,mimeType,trashed,driveId,ownedByMe,version,createdTime,modifiedTime,name,size,sha256Checksum"
MAX_DEPTH = 16
VERSION = re.compile(r"[1-9][0-9]{0,18}\Z")
DB_COMPLETED_UTC = "2026-11-19T04:18:37Z"
RFC3339 = re.compile(r"(\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d)(?:\.(\d{1,9}))?(Z|[+-]\d\d:\d\d)\Z")
RECOVERY_ARTIFACT = {
    "name": "v2_metadata_changes_20261119T043247Z_e81468933e0a.zip",
    "size": "2136",
    "sha256": "e8b95d1755d4b869216f3d530035fb742738d886e4d2359691af6acf661f04ca",
    "creation_lower_utc": "2026-11-19T04:32:54Z",
    "parents": ["1zAdr_RG7Z1_8KsviH5VxSB8A4qa6w-XD", "1-_IOAgfCOh2OBSTRWb9XeocKIkCuFo-W", "0AIF_mmJeNp7RUk9PVA"],
}


class ScopeDrive:
    """Expose changes.list and metadata files.get only."""

    def __init__(self):
        self.transport = Drive()

    def changes(self, token, check, remaining):
        return self.transport.get("/changes", {
            "pageToken": token, "pageSize": "1000", "includeRemoved": "true",
            "supportsAllDrives": "true", "includeItemsFromAllDrives": "true",
            "fields": f"nextPageToken,newStartPageToken,changes(changeType,fileId,driveId,removed,time,file({FILE_FIELDS}))",
        }, check, remaining)

    def file(self, object_id, check, remaining):
        return self.transport.get("/files/" + id_value(object_id), {
            "fields": FILE_FIELDS, "supportsAllDrives": "true",
        }, check, remaining)


class Budget:
    def __init__(self, output, clock=time.monotonic):
        self.output, self.clock = output, clock
        self.started = clock()
        output.update(requests=0, response_bytes=0, changes_list_requests=0, files_get_requests=0)

    def check(self):
        if self.clock() - self.started >= parent.MAX_SECONDS:
            raise TimeoutError("v2_scope_wall_limit")

    def request(self, name, function, argument):
        self.check()
        remaining = parent.MAX_BYTES - self.output["response_bytes"]
        if self.output["requests"] >= parent.MAX_REQUESTS or not remaining:
            raise TimeoutError("v2_scope_api_budget")
        self.output["requests"] += 1
        self.output[name + "_requests"] += 1
        try:
            result, amount = function(argument, self.check, remaining)
        except (TimeoutError, ValueError):
            raise
        except Exception as exc:
            status = getattr(getattr(exc, "response", None), "status_code", None)
            raise ValueError(f"v2_scope_{name}_http_{status}" if type(status) is int
                             else f"v2_scope_{name}_{type(exc).__name__}") from None
        if type(amount) is not int or not 0 <= amount <= remaining:
            raise ValueError("v2_scope_response_cap")
        self.output["response_bytes"] += amount
        self.check()
        return result


def all_changes(client, budget, token, rows):
    consumed = {token}
    while True:
        page = budget.request("changes_list", client.changes, token)
        events = page.get("changes")
        if not isinstance(events, list):
            raise ValueError("v2_scope_invalid_changes")
        rows.extend(events)
        following, terminal = page.get("nextPageToken"), page.get("newStartPageToken")
        if following is None and isinstance(terminal, str) and terminal and len(terminal) <= 16384:
            return terminal
        if terminal is not None or not isinstance(following, str) or not following or len(following) > 16384 or following in consumed:
            raise ValueError("v2_scope_incomplete_or_repeated_cursor")
        consumed.add(following)
        token = following


def timestamp(value):
    result = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if result.tzinfo is None:
        raise ValueError("v2_scope_timezone_required")
    return result.astimezone(timezone.utc)


def exact_utc(value):
    """Normalize offsets without dropping fractional precision."""
    match = RFC3339.fullmatch(value) if isinstance(value, str) else None
    if not match or match[3] == "-00:00":
        raise ValueError("v2_scope_unverified_time")
    seconds, fractional, zone = match.groups()
    if zone != "Z" and (int(zone[1:3]) > 23 or int(zone[4:]) > 59):
        raise ValueError("v2_scope_unverified_time")
    return timestamp(seconds + zone), int((fractional or "").ljust(9, "0"))


def evidence_index(raw, initial_token):
    if raw is None:
        return None
    cursor_sha = hashlib.sha256(initial_token.encode()).hexdigest()
    if (raw.get("schema") != "v2_external_parent_evidence_v1" or raw.get("parent_zip_sha256") != parent.ZIP_SHA
            or raw.get("saved_cursor_sha256") != cursor_sha or not isinstance(raw.get("basis"), str)
            or not raw["basis"].strip() or not isinstance(raw.get("items"), dict)):
        raise ValueError("v2_scope_evidence_binding")
    upper = raw.get("fence_start_upper_utc")
    return {"root": id_value(raw["my_drive_root_id"]), "items": raw["items"], "upper": timestamp(upper) if upper else None}


class Classifier:
    def __init__(self, db, client, budget, evidence, recovery_provenance=False):
        self.db, self.client, self.budget, self.evidence = db, client, budget, evidence
        self.local_cache, self.remote_cache = {}, {}
        self.changed_ids = Counter()
        self.recovery_provenance = recovery_provenance

    def membership(self, object_id):
        if object_id not in self.local_cache:
            found = []
            for table in ("entries", "queue"):
                for kind, path in self.db.execute(f"SELECT kind,path FROM {table} WHERE kind IN ('pose','rgb') AND id=?", (object_id,)):
                    found.append({"source": "root" if table == "queue" and not path else table, "kind": kind})
            self.local_cache[object_id] = found
        return self.local_cache[object_id]

    def metadata(self, object_id):
        id_value(object_id)
        if object_id not in self.remote_cache:
            try:
                self.remote_cache[object_id] = self.budget.request("files_get", self.client.file, object_id)
            except (ValueError, TimeoutError) as exc:
                self.remote_cache[object_id] = exc
        result = self.remote_cache[object_id]
        if isinstance(result, Exception):
            raise result
        if result.get("id") != object_id and object_id != "root":
            raise ValueError("v2_scope_metadata_identity")
        return result

    def node(self, object_id, item):
        if item.get("id") != object_id or item.get("driveId") or item.get("mimeType") == SHORTCUT:
            raise ValueError("v2_scope_shared_shortcut_or_identity")
        if item.get("trashed") is not False or item.get("ownedByMe") is not True or not item.get("mimeType"):
            raise ValueError("v2_scope_ownership_or_metadata_unverified")
        parents = item.get("parents", [])
        if not isinstance(parents, list) or len(parents) > 1:
            raise ValueError("v2_scope_ambiguous_parents")
        return [id_value(p) for p in parents]

    def historical(self, object_id, item, parent_id):
        earlier = self.evidence["items"].get(object_id)
        version = item.get("version")
        if (not isinstance(earlier, dict) or not isinstance(version, str) or not VERSION.fullmatch(version)
                or earlier.get("parent_id") != parent_id or earlier.get("version") != version
                or earlier.get("mime_type") != item["mimeType"]
                or not isinstance(earlier.get("source_sha256"), str)
                or not re.fullmatch(r"[a-f0-9]{64}", earlier["source_sha256"])):
            return False
        if earlier.get("kind") == "outside_before_saved_cursor":
            return True
        if earlier.get("kind") == "outside_creation_receipt" and self.evidence["upper"] is not None:
            try:
                created = timestamp(item["createdTime"])
                return (earlier["created_time"] == item["createdTime"]
                        and self.evidence["upper"] < created <= timestamp(earlier["observed_utc"]) <= datetime.now(timezone.utc))
            except (KeyError, TypeError, ValueError, AttributeError):
                return False
        return False

    def known_recovery(self, event, item, chain):
        artifact = RECOVERY_ARTIFACT
        parents = [node["id"] for node in chain[1:]]
        if (item.get("name") != artifact["name"] or item.get("size") != artifact["size"]
                or item.get("sha256Checksum") != artifact["sha256"] or parents != artifact["parents"]
                or self.changed_ids[item["id"]] != 1 or any(self.changed_ids[p] for p in parents)
                or not isinstance(item.get("version"), str) or not VERSION.fullmatch(item["version"])):
            return False
        try:
            return (timestamp(artifact["creation_lower_utc"]) <= timestamp(item["createdTime"])
                    <= timestamp(event["time"]) <= datetime.now(timezone.utc))
        except (KeyError, TypeError, ValueError, AttributeError):
            return False

    def observed_external_creation(self, event, item, chain):
        if (self.changed_ids[item["id"]] != 1 or any(self.changed_ids[node["id"]] for node in chain[1:])
                or not isinstance(item.get("version"), str) or not VERSION.fullmatch(item["version"])):
            return False
        try:
            created = exact_utc(item.get("createdTime"))
            changed = exact_utc(event.get("time"))
            modification = item.get("modifiedTime")
            return (exact_utc(DB_COMPLETED_UTC) < created == changed <= (datetime.now(timezone.utc), 0)
                    and (modification is None or exact_utc(modification) <= created))
        except ValueError:
            return False

    def classify(self, event):
        result = {"status": "blocked", "chain": []}
        if not isinstance(event, dict):
            return {**result, "reason": "invalid_change"}
        object_id = event.get("fileId")
        item = event.get("file")
        result.update(file_id=object_id, drive_id=event.get("driveId"), change_type=event.get("changeType"),
                      removed=event.get("removed"), event_time=event.get("time"), event_parent_ids=None)
        try:
            if isinstance(item, dict) and "parents" in item:
                parents = item["parents"]
                if not isinstance(parents, list):
                    raise ValueError("v2_scope_ambiguous_parents")
                result["event_parent_ids"] = [id_value(p) for p in parents]
            if event.get("changeType") != "file" or event.get("driveId"):
                raise ValueError("v2_scope_shared_or_non_file_change")
            id_value(object_id)
            result["object_membership"] = self.membership(object_id)
            result["event_parent_membership"] = {p: self.membership(p) for p in (result["event_parent_ids"] or [])}
            if result["object_membership"]:
                raise ValueError("v2_scope_known_dataset_object_changed")
            removal_fields = {k: event[k] for k in ("changeType", "fileId", "removed", "file") if k in event}
            if object_id == REMOVED:
                if allowed_removal(removal_fields, REMOVED, FAILED_PARENT):
                    return {**result, "status": "approved_removal", "reason": "exact_unobserved_folder_removal"}
                raise ValueError("v2_scope_approved_removal_shape")
            if any(result["event_parent_membership"].values()):
                raise ValueError("v2_scope_event_parent_in_dataset")
            if event.get("removed") is not False:
                raise ValueError("v2_scope_removed_or_historical_parent_unknown")
            if not isinstance(item, dict) or item.get("id") != object_id:
                raise ValueError("v2_scope_event_metadata_missing")
            if item.get("driveId") or item.get("mimeType") == SHORTCUT or item.get("trashed") is not False:
                raise ValueError("v2_scope_event_shared_shortcut_or_trashed")
            root_meta = self.metadata("root")
            root = id_value(root_meta.get("id"))
            if self.node(root, root_meta) or root_meta["mimeType"] != FOLDER or self.membership(root):
                raise ValueError("v2_scope_unverified_external_mydrive_root")
            if object_id == root:
                raise ValueError("v2_scope_mydrive_root_change")
            if self.evidence and self.evidence["root"] != root:
                raise ValueError("v2_scope_historical_root_mismatch")
            current, visited = object_id, set()
            all_history = self.evidence is not None
            object_metadata = None
            while current != root:
                if current in visited or len(visited) >= MAX_DEPTH:
                    raise ValueError("v2_scope_parent_cycle_or_depth")
                visited.add(current)
                local = self.membership(current)
                node_report = {"id": current, "membership": local}
                result["chain"].append(node_report)
                if local:
                    raise ValueError("v2_scope_current_ancestor_in_dataset")
                # The change response already covers the candidate. Save its GET for
                # a possible acceptance, then compare it against the change snapshot.
                metadata = item if current == object_id else self.metadata(current)
                parents = self.node(current, metadata)
                node_report["parents"] = parents
                if current == object_id:
                    object_metadata = metadata
                if current != object_id and metadata["mimeType"] != FOLDER:
                    raise ValueError("v2_scope_parent_not_folder")
                if len(parents) != 1:
                    raise ValueError("v2_scope_parent_chain_incomplete")
                historical = self.evidence is not None and self.historical(current, metadata, parents[0])
                node_report["historical_match"] = historical
                all_history &= historical
                current = parents[0]
            result["chain"].append({"id": root, "parents": [], "membership": [], "my_drive_root": True})
            recovery_signature = (object_metadata.get("name") == RECOVERY_ARTIFACT["name"]
                                  or object_metadata.get("sha256Checksum") == RECOVERY_ARTIFACT["sha256"])
            known_recovery = (self.recovery_provenance and object_metadata is not None
                              and self.known_recovery(event, object_metadata, result["chain"]))
            if recovery_signature and not known_recovery:
                raise ValueError("v2_scope_recovery_requires_its_pinned_identity")
            new_external = not recovery_signature and self.observed_external_creation(event, object_metadata, result["chain"])
            if known_recovery or all_history or new_external:
                current_metadata = self.metadata(object_id)
                if any(item.get(k) != current_metadata.get(k) for k in FILE_FIELDS.split(",")):
                    raise ValueError("v2_scope_event_metadata_changed")
            if known_recovery:
                return {**result, "status": "outside_proven", "reason": "verified_recovery_created_outside_after_db",
                        "proof_scope": "observed recovery creation and pinned DB impact; no assertion about hidden intermediate history",
                        "recovery_artifact_sha256": RECOVERY_ARTIFACT["sha256"]}
            if new_external:
                return {**result, "status": "outside_proven", "reason": "observed_external_creation_at_change_time_after_db",
                        "proof_scope": "observed creation event and frozen DB time",
                        "created_time": object_metadata["createdTime"], "verified_version": object_metadata["version"]}
            if not all_history:
                raise ValueError("v2_scope_historical_external_membership_unproven")
            return {**result, "status": "outside_proven", "reason": "outside_current_and_pinned_version_history"}
        except (ValueError, TimeoutError) as exc:
            return {**result, "reason": str(exc)}


def classify_interval(db, client, token, evidence, report, budget, *, recovery_provenance=False):
    classifier = Classifier(db, client, budget, evidence, recovery_provenance)
    rows, later = [], []
    try:
        end = all_changes(client, budget, token, rows)
        report["pagination_complete"] = True
    finally:
        classifier.changed_ids = Counter(row["fileId"] for row in rows
                                         if isinstance(row, dict) and isinstance(row.get("fileId"), str))
        report["events"] = []
        for row in rows:
            report["events"].append(classifier.classify(row))
        recovered = [event for event in report["events"] if event["reason"] == "verified_recovery_created_outside_after_db"]
        if len(recovered) > 1:
            for event in recovered:
                event.update(status="blocked", reason="v2_scope_ambiguous_recovery_artifact_identity")
        classes = Counter(r["status"] for r in report["events"])
        report["classification_totals"] = classes
    if classes["approved_removal"] != 1 or classes["blocked"]:
        raise ValueError("v2_scope_all_changes_not_proven")
    if classes["outside_proven"]:
        try:
            all_changes(client, budget, end, later)
            report["consistency_pagination_complete"] = True
        finally:
            report["during_inspection_events"] = [classifier.classify(row) for row in later]
        if later:
            raise ValueError("v2_scope_changes_during_parent_inspection")


def finish_comparison(db, owners, work, report):
    report["status"] = "change_verified_local_comparison_started"
    report["counts"] = parent.compare(db, owners, work)
    totals = Counter()
    for group in report["counts"].values():
        for cell in group.values():
            totals.update(cell)
    report["totals"] = totals
    failures = ("missing_role_keys", "duplicate_role_keys", "missing_sequence_keys")
    report["frame_correspondence_passed"] = not any(totals[k] for k in failures)
    report["status"] = "comparison_complete"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=("verify", "check"))
    for name in ("archive", "reference", "work", "external-evidence"):
        p.add_argument("--" + name, type=Path, required=name != "external-evidence")
    p.add_argument("--external-evidence-sha")
    args = p.parse_args()
    args.work.mkdir()
    report = {"status": "started", "parent_zip_sha256": parent.ZIP_SHA, "db_sha256": parent.DB_SHA,
              "metadata_budget": {}, "change_inspection": {}, "files_list_calls": 0, "payload_reads": 0,
              "start_token_requests": 0, "external_evidence_sha256": args.external_evidence_sha}
    try:
        recovery = args.reference/"v2_scope_recovery_043247Z.zip"
        recovery_provenance = recovery.exists()
        if recovery_provenance and (recovery.is_symlink() or recovery.stat().st_size != int(RECOVERY_ARTIFACT["size"])
                                    or sha_file(recovery) != RECOVERY_ARTIFACT["sha256"]):
            raise ValueError("v2_scope_recovery_artifact_pin")
        report["recovery_artifact_evidence_sha256"] = RECOVERY_ARTIFACT["sha256"] if recovery_provenance else None
        if (args.external_evidence is None) != (args.external_evidence_sha is None):
            raise ValueError("v2_scope_evidence_pin_required")
        raw = pinned_json(args.external_evidence, args.external_evidence_sha, 4*1024**2) if args.external_evidence else None
        with tempfile.TemporaryDirectory(dir=args.work) as temp:
            database = Path(temp)/"metadata.sqlite"
            parent.extract_db(args.archive, database)
            with closing(parent.read_only(database)) as db:
                owners, corpus, token = parent.db_contract(db, args.reference)
                if corpus != "":
                    raise ValueError("v2_scope_shared_corpus")
                evidence = evidence_index(raw, token)
                report["status"] = "verified_archive"
                if args.action == "check":
                    budget = Budget(report["metadata_budget"])
                    try:
                        with parent.remote_deadline():
                            classify_interval(db, ScopeDrive(), token, evidence, report["change_inspection"], budget,
                                              recovery_provenance=recovery_provenance)
                    finally:
                        report["metadata_budget"]["wall_seconds"] = budget.clock() - budget.started
                    finish_comparison(db, owners, args.work, report)
    except BaseException as exc:
        report["status_before_failure"] = report["status"]
        report["status"] = "stopped" if isinstance(exc, KeyboardInterrupt) else "blocked"
        report["reason"] = type(exc).__name__ + ": " + str(exc)[:400]
        raise
    finally:
        (args.work/"report.json").write_bytes(canonical(report))
    print(canonical(report).decode())


if __name__ == "__main__":
    main()
