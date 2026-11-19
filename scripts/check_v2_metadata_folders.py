"""Inspect only the nine folder IDs selected by the fixed scope report."""

import argparse
from collections import Counter
from contextlib import closing
import json
from pathlib import Path
import signal
import tempfile
import time
import zipfile

from rewind_archive.contracts import canonical, safe_path
from rewind_archive.matching import identify
from scripts import check_v2_metadata_fence as baseline
from scripts import check_v2_metadata_scope as scope
from scripts.list_v2_drive_metadata import Drive, FOLDER, SHORTCUT, folders, id_value
from scripts.resume_v2_drive_metadata import REMOVED, sha_file

REPORT_SHA = "a7cd85e64e05a750b6bab0b9082a88e4d0ae03cc53f317d28cf6fadae90fe934"
REPORT_SIZE = 10565
NEW_RECOVERY = "v2_metadata_scope_20261119T050017Z_ad41b1217c38.zip"
MAX_REQUESTS, MAX_BYTES, MAX_SECONDS = 128, 16*1024**2, 240
FIELDS = scope.FILE_FIELDS


def hits(db, object_id):
    return [{"table": table, "kind": kind, "path": path}
            for table in ("queue", "entries")
            for kind, path in db.execute(f"SELECT kind,path FROM {table} WHERE kind IN ('rgb','pose') AND id=?", (object_id,))]


def loaded_events(path, db):
    if path.is_symlink() or path.stat().st_size != REPORT_SIZE or sha_file(path) != REPORT_SHA:
        raise ValueError("v2_folders_report_zip_pin")
    with zipfile.ZipFile(path) as bundle:
        if sorted(bundle.namelist()) != ["report.json", "worker.stderr.txt", "worker.stdout.txt"]:
            raise ValueError("v2_folders_report_members")
        if bundle.getinfo("report.json").file_size > 65536:
            raise ValueError("v2_folders_report_size")
        report = json.loads(bundle.read("report.json"))
    inspection = report["change_inspection"]
    rows = inspection["events"]
    if (report["parent_zip_sha256"] != baseline.ZIP_SHA or report["db_sha256"] != baseline.DB_SHA
            or report["status"] != "blocked" or inspection["pagination_complete"] is not True or len(rows) != 17):
        raise ValueError("v2_folders_report_binding")
    ids, targets, others, approved = set(), [], [], []
    for event in rows:
        object_id = id_value(event["file_id"])
        if object_id in ids:
            raise ValueError("v2_folders_duplicate_event_id")
        ids.add(object_id)
        local = hits(db, object_id)
        observed = {(r["kind"], r["source"]) for r in event["object_membership"]}
        derived = {(r["kind"], "root" if r["table"] == "queue" and not r["path"] else r["table"]) for r in local}
        if observed != derived:
            raise ValueError("v2_folders_event_db_membership")
        if event["status"] == "approved_removal" and object_id == REMOVED and not local:
            approved.append(event)
        elif event["status"] == "blocked" and len(local) == 1 and local[0]["table"] == "queue":
            entry = local[0]
            expected_parent = event["event_parent_ids"]
            if entry["path"]:
                parent_path = entry["path"].rpartition("/")[0]
                row = db.execute("SELECT id FROM queue WHERE kind=? AND path=?", (entry["kind"], parent_path)).fetchone()
                if row is None or expected_parent != [row[0]]:
                    raise ValueError("v2_folders_prior_parent_pin")
            if not isinstance(expected_parent, list) or len(expected_parent) != 1:
                raise ValueError("v2_folders_prior_parent_missing")
            targets.append({**entry, "id": object_id, "event_parent_ids": expected_parent,
                            "parent_evidence": "db_queue" if entry["path"] else "previous_change_event_only"})
        elif event["status"] == "blocked" and not local:
            others.append(event)
        else:
            raise ValueError("v2_folders_unexpected_event_kind")
    if (len(targets), len(others), len(approved)) != (9, 7, 1):
        raise ValueError("v2_folders_selection_counts")
    return targets, others, approved[0]


class Metadata:
    def __init__(self, allowed):
        self.allowed = set(allowed)
        self.transport = Drive()

    def file(self, object_id, check, available):
        return self.transport.get("/files/" + id_value(object_id), {
            "fields": FIELDS, "supportsAllDrives": "true",
        }, check, available)

    def children(self, object_id, token, check, available):
        if object_id not in self.allowed:
            raise ValueError("v2_folders_unapproved_listing_id")
        params = {"q": f"'{id_value(object_id)}' in parents and trashed=false", "spaces": "drive", "pageSize": "1000",
                  "supportsAllDrives": "true", "includeItemsFromAllDrives": "true",
                  "fields": f"nextPageToken,incompleteSearch,files({FIELDS})"}
        if token:
            params["pageToken"] = token
        return self.transport.get("/files", params, check, available)


class Budget:
    def __init__(self, report):
        self.report, self.started = report, time.monotonic()
        report.update(requests=0, response_bytes=0, files_get=0, direct_files_list=0)

    def check(self):
        if time.monotonic() - self.started >= MAX_SECONDS:
            raise TimeoutError("v2_folders_wall_limit")

    def request(self, category, function, *args):
        self.check()
        left = MAX_BYTES - self.report["response_bytes"]
        if self.report["requests"] >= MAX_REQUESTS or left <= 0:
            raise TimeoutError("v2_folders_request_or_byte_limit")
        self.report["requests"] += 1
        self.report[category] += 1
        try:
            data, used = function(*args, self.check, left)
        except (ValueError, TimeoutError):
            raise
        except Exception as exc:
            code = getattr(getattr(exc, "response", None), "status_code", None)
            raise ValueError(f"v2_folders_http_{code}" if type(code) is int else type(exc).__name__) from None
        if type(used) is not int or not 0 <= used <= left:
            raise ValueError("v2_folders_response_byte_limit")
        self.report["response_bytes"] += used
        self.check()
        return data


def direct_baseline(db, kind, path):
    prefix = path + "/" if path else ""
    result = {}
    tables = ("queue", "entries") if path.endswith(("/json", "/camera_left", "/camera_right")) else ("queue",)
    for name in tables:
        columns = "path,id" if name == "queue" else "path,id,row"
        for row in db.execute(f"""SELECT {columns} FROM {name} WHERE kind=? AND path>=? AND path<?
                                  AND length(path)>? AND instr(substr(path,?),'/')=0""",
                              (kind, prefix, prefix+chr(0x10ffff), len(prefix), len(prefix)+1)):
            child_path, object_id, *raw = row
            remainder = child_path[len(prefix):]
            if not remainder or "/" in remainder:
                continue
            result[object_id] = {"id": object_id, "name": remainder, "path": child_path,
                                 "type": "folder" if name == "queue" else "file"}
            if raw:
                item = json.loads(raw[0])
                result[object_id].update(version=item.get("version"), size=str(item["size_bytes"]),
                                         sha256Checksum=item.get("sha256"), modifiedTime=item.get("modified_time"))
    return result


def filter_children(items, kind, path, expected_folders, owners):
    included, excluded, errors = {}, {}, []
    for item in items:
        name = item.get("name")
        if not isinstance(name, str) or "/" in name or safe_path(name) != name:
            raise ValueError("v2_folders_unsafe_child_name")
        child_path = f"{path}/{name}" if path else name
        item_type = "folder" if item["mimeType"] == FOLDER else "file"
        row = {**item, "type": item_type, "path": child_path}
        reason = None
        if child_path in expected_folders:
            if item_type != "folder":
                reason = "expected_folder_type"
        elif item_type == "folder" or not path.endswith(("/json", "/camera_left", "/camera_right")):
            excluded[row["id"]] = row
            continue
        else:
            try:
                seq, _, role = identify(child_path, kind)
                if item["mimeType"] == SHORTCUT or owners.get(seq) not in ("train", "validation") or (role == "pose") != (kind == "pose"):
                    reason = "invalid_payload_membership"
            except ValueError:
                reason = "invalid_payload_path"
        if reason:
            errors.append({"id": row["id"], "path": child_path, "reason": reason})
        else:
            included[row["id"]] = row
    return included, excluded, errors


def compare_children(previous, current, excluded, errors):
    changes, checksum_status = [], Counter()
    metadata_unverified = Counter()
    names = {}
    for new in current.values():
        names.setdefault(new["name"], []).append(new["id"])
    for name, ids in names.items():
        if len(ids) > 1:
            changes.append({"kind": "duplicate_current_name", "name": name, "ids": sorted(ids)})
    for object_id, old in previous.items():
        new = current.get(object_id)
        if not new:
            if object_id in excluded:
                outside = excluded[object_id]
                changes.append({"kind": "filtered_out", "id": object_id, "old_path": old["path"], "new_path": outside["path"]})
            elif names.get(old["name"]):
                changes.append({"kind": "replaced_at_name", "old_id": object_id, "path": old["path"], "new_ids": sorted(names[old["name"]])})
            else:
                changes.append({"kind": "not_in_direct_listing", "id": object_id, "old_path": old["path"]})
            continue
        if old["path"] != new["path"] or old["type"] != new["type"]:
            changes.append({"kind": "rename_or_type_change", "id": object_id, "old_path": old["path"], "new_path": new["path"],
                            "old_type": old["type"], "new_type": new["type"]})
        if old["type"] == "file" and new["type"] == "file":
            for key in ("version", "size", "sha256Checksum", "modifiedTime"):
                if old.get(key) is None or new.get(key) is None:
                    metadata_unverified[key] += 1
                if old.get(key) is not None and new.get(key) is not None and old[key] != new[key]:
                    changes.append({"kind": "file_metadata_changed", "id": object_id, "field": key,
                                    "previous": old[key], "current": new[key]})
            if old.get("sha256Checksum") and new.get("sha256Checksum"):
                checksum_status["matched_provider_sha" if old["sha256Checksum"] == new["sha256Checksum"] else "different_provider_sha"] += 1
            else:
                checksum_status["no_comparable_sha"] += 1
        elif old["type"] == new["type"] == "folder":
            metadata_unverified["folder_baseline_metadata_not_stored"] += 1
    for object_id in sorted(current.keys() - previous.keys()):
        changes.append({"kind": "added_to_filtered_listing", "id": object_id, "path": current[object_id]["path"]})
    membership_changes = [row for row in changes if row["kind"] != "file_metadata_changed"]
    return {"old_filtered_count": len(previous), "new_filtered_count": len(current),
            "observed_same_filtered_membership": not membership_changes and not errors,
            "changes": changes, "filter_errors": errors, "provider_sha_comparison": dict(checksum_status),
            "metadata_unverified_counts": dict(metadata_unverified),
            "payload_bytes_verified": False, "excluded_children_prior_state": "unknown",
            "current_excluded_children": [{"id": row["id"], "path": row["path"], "type": row["type"]} for row in excluded.values()]}


def child_pages(client, budget, target, output):
    token, cursors, ids = None, set(), set()
    while True:
        page = budget.request("direct_files_list", client.children, target, token)
        if page.get("incompleteSearch") or not isinstance(page.get("files"), list):
            raise ValueError("v2_folders_incomplete_direct_listing")
        for item in page["files"]:
            object_id = id_value(item["id"])
            if object_id in ids or item.get("parents") != [target] or item.get("trashed") is not False or item.get("driveId"):
                raise ValueError("v2_folders_unstable_direct_child")
            ids.add(object_id)
            output.append(item)
        token = page.get("nextPageToken")
        if token is None:
            return
        if not isinstance(token, str) or not token or len(token) > 16384 or token in cursors:
            raise ValueError("v2_folders_duplicate_or_invalid_page_token")
        cursors.add(token)


def snapshot(item):
    return {field: item.get(field) for field in FIELDS.split(",")}


class Inspector:
    def __init__(self, db, client, budget):
        self.db, self.client, self.budget = db, client, budget
        self.metadata, self.after = {}, {}

    def file(self, object_id, *, refresh=False):
        cache = self.after if refresh else self.metadata
        if object_id not in cache:
            try:
                item = self.budget.request("files_get", self.client.file, object_id)
                if not isinstance(item, dict) or (object_id != "root" and item.get("id") != object_id):
                    raise ValueError("v2_folders_get_identity")
                cache[object_id] = item
            except (ValueError, TimeoutError) as error:
                cache[object_id] = error
        item = cache[object_id]
        if isinstance(item, Exception):
            raise item
        return item

    def chain(self, object_id, output):
        root_metadata = self.file("root")
        root = id_value(root_metadata["id"])
        self.metadata.setdefault(root, root_metadata)
        known = set()
        current = object_id
        while True:
            if len(known) >= 16 or current in known:
                raise ValueError("v2_folders_parent_depth_or_cycle")
            known.add(current)
            item = self.file(current)
            output.append(snapshot(item))
            if item.get("driveId") or item.get("ownedByMe") is not True or item.get("mimeType") == SHORTCUT or item.get("trashed") is not False:
                raise ValueError("v2_folders_ambiguous_parent_metadata")
            name = item.get("name")
            if not isinstance(name, str) or "/" in name or safe_path(name) != name:
                raise ValueError("v2_folders_unverified_path_name")
            if current != object_id and item.get("mimeType") != FOLDER:
                raise ValueError("v2_folders_parent_is_not_folder")
            parents = item.get("parents", [])
            if current == root:
                if parents or item.get("mimeType") != FOLDER:
                    raise ValueError("v2_folders_invalid_root")
                return
            if not isinstance(parents, list) or len(parents) != 1:
                raise ValueError("v2_folders_missing_or_multiple_parent")
            current = id_value(parents[0])

    def inspect_folder(self, target, owners, expected, listings):
        object_id, kind, path = target["id"], target["kind"], target["path"]
        result = {**target, "status": "incomplete", "path_chain": []}
        raw = []
        try:
            item = snapshot(self.file(object_id))
            if item["mimeType"] != FOLDER or item["trashed"] is not False or item["driveId"]:
                raise ValueError("v2_folders_target_no_longer_folder")
            try:
                self.chain(object_id, result["path_chain"])
                result["path_chain_verified"] = True
            except (ValueError, TimeoutError, KeyError, TypeError) as error:
                if not result["path_chain"]:
                    result["path_chain"].append(item)
                result["path_chain_verified"] = False
                result["path_chain_error"] = str(error)[:240]
            parents = item["parents"]
            result["same_parent_as_reference"] = parents == target["event_parent_ids"] if isinstance(parents, list) and len(parents) == 1 else None
            result["same_basename_as_db"] = item["name"] == path.rsplit("/", 1)[-1] if path else None
            result["current_path"] = ("/".join(str(row["name"]) for row in reversed(result["path_chain"][:-1]))
                                      if result["path_chain_verified"] else None)
            db_root, = self.db.execute("SELECT id FROM queue WHERE kind=? AND path=''", (kind,)).fetchone()
            current_chain = result["path_chain"]
            if result["path_chain_verified"] and db_root in (row["id"] for row in current_chain):
                before_root = [row["name"] for row in current_chain[:next(i for i, row in enumerate(current_chain) if row["id"] == db_root)]]
                result["current_db_root_relative_path"] = "/".join(reversed(before_root))
            else:
                result["current_db_root_relative_path"] = None
            result["same_dataset_relative_path"] = result["current_db_root_relative_path"] == path if result["path_chain_verified"] else None
            child_pages(self.client, self.budget, object_id, raw)
            previous = direct_baseline(self.db, kind, path)
            included, excluded, errors = filter_children(raw, kind, path, expected[kind], owners)
            result.update(compare_children(previous, included, excluded, errors))
            for row in raw:
                listings.setdefault(row["id"], []).append({"folder_id": object_id, "name": row["name"]})
            result["status"] = "listing_compared" if result["path_chain_verified"] else "listing_compared_path_unverified"
        except (ValueError, TimeoutError, KeyError, TypeError) as error:
            result["reason"] = str(error)[:240]
        finally:
            result["received_direct_children"] = len(raw)
            if result["status"] == "incomplete":
                result["partial_direct_children"] = [{"id": row.get("id"), "name": row.get("name"), "type": row.get("mimeType")}
                                                     for row in raw]
        return result

    def inspect_other(self, event, dataset_ancestor_ids, observed_event_ids):
        result = {"id": event["file_id"], "event_time": event["event_time"], "event_removed": event["removed"],
                  "event_parent_ids": event.get("event_parent_ids"), "prior_reason": event["reason"], "path_chain": [],
                  "status": "unknown", "historical_membership": "unproven"}
        result["previously_observed_chain"] = [{"id": row.get("id"), "parents": row.get("parents")} for row in event.get("chain", [])]
        result["event_parent_db_membership"] = {p: hits(self.db, p) for p in (event.get("event_parent_ids") or [])}
        try:
            self.chain(event["file_id"], result["path_chain"])
            chain = result["path_chain"]
            result["current_path"] = "/".join(str(row["name"]) for row in reversed(chain[:-1]))
            result["matched_dataset_chain"] = [row["id"] for row in chain if hits(self.db, row["id"])]
            if event["file_id"] in dataset_ancestor_ids:
                result["status"] = "current_ancestor_of_inspected_dataset_folder"
            elif result["matched_dataset_chain"]:
                result["status"] = "currently_under_dataset"
            else:
                result["status"] = "currently_outside_dataset_historical_scope_unknown"
            item = chain[0]
            artifact = scope.RECOVERY_ARTIFACT
            recovery_event = event["file_id"] == "1Lexz55SEoGr_WebvN8LFeNoAOWbM-6Y2"
            if recovery_event or item["name"] in (artifact["name"], NEW_RECOVERY):
                is_new = item["name"] == NEW_RECOVERY and not recovery_event
                expected_name, size, pin = (NEW_RECOVERY, str(REPORT_SIZE), REPORT_SHA) if is_new else (
                    artifact["name"], artifact["size"], artifact["sha256"])
                fixed_chain = [row["id"] for row in chain[1:]]
                result["recovery_artifact"] = {
                    "reference": "050017_scope_report" if is_new else "043247_fence_report",
                    "name_matches": item["name"] == expected_name,
                    "size_matches": str(item["size"]) == size,
                    "sha256_matches": item["sha256Checksum"] == pin if item["sha256Checksum"] else None,
                    "parent_chain_matches_043247": fixed_chain == artifact["parents"] if not is_new else None,
                    "parent_chain_ids_also_in_old_change_report": sorted(set(fixed_chain) & observed_event_ids),
                }
                if not is_new:
                    result["recovery_artifact"]["prior_gate_would_reject_observed_changed_ancestors"] = bool(
                        set(fixed_chain) & observed_event_ids)
                    try:
                        result["recovery_artifact"]["creation_and_event_time_match_old_gate"] = (
                            scope.timestamp(artifact["creation_lower_utc"]) <= scope.timestamp(item["createdTime"])
                            <= scope.timestamp(event["event_time"]))
                    except (AttributeError, ValueError, TypeError, KeyError):
                        result["recovery_artifact"]["creation_and_event_time_match_old_gate"] = None
        except (ValueError, TimeoutError, KeyError, TypeError) as error:
            result["reason"] = str(error)[:240]
        return result

    def check_consistency(self, folder):
        changes, missing_versions = [], []
        for before in folder["path_chain"]:
            object_id = before["id"]
            if not before["version"]:
                missing_versions.append(object_id)
            try:
                after = snapshot(self.file(object_id, refresh=True))
                if after != before:
                    changes.append({"id": object_id, "fields": sorted(k for k in before if before[k] != after[k])})
            except (ValueError, TimeoutError) as error:
                changes.append({"id": object_id, "reason": str(error)[:200]})
        folder["consistency"] = {"observed_metadata_stable": not changes and not missing_versions,
                                 "differences": changes, "missing_prior_versions": missing_versions,
                                 "scope": "target/ancestor GET snapshots; child listings are not atomic"}
        if not folder["consistency"]["observed_metadata_stable"] and folder["status"] == "listing_compared":
            folder["status"] = "inconclusive_concurrency"


def run(db, owners, client, selections, report):
    targets, others, removal = selections
    budget = Budget(report["budget"])
    inspector = Inspector(db, client, budget)
    expected = {kind: folders(owners, kind) for kind in ("rgb", "pose")}
    reports, listings = [], {}
    report.update(targets=reports, other_events=[], approved_removal=removal, source_scope="nine direct listings only",
                  global_corpus_consistency="not_assessed")
    for target in targets:
        reports.append(inspector.inspect_folder(target, owners, expected, listings))
    for folder in reports:
        for change in folder.get("changes", []):
            if change["kind"] == "not_in_direct_listing":
                found = [where for where in listings.get(change["id"], []) if where["folder_id"] != folder["id"]]
                if found:
                    change["observed_directly_under_other_inspected_folders"] = found
    ancestors = {row["id"] for folder in reports for row in folder["path_chain"][1:]}
    event_ids = {row["file_id"] for row in others} | {row["id"] for row in targets} | {removal["file_id"]}
    for event in others:
        report["other_events"].append(inspector.inspect_other(event, ancestors, event_ids))
    for folder in reports:
        if folder["status"] in ("listing_compared", "listing_compared_path_unverified"):
            inspector.check_consistency(folder)
    report["budget"]["wall_seconds"] = time.monotonic() - budget.started
    report["target_status_counts"] = dict(Counter(row["status"] for row in reports))
    report["summary"] = summarize(targets, reports, db)
    report["status"] = "partial_scope_diagnostic" if any(row["status"] != "listing_compared" for row in reports) else "targeted_comparison_complete"


def summarize(targets, reports, db):
    compared = [row for row in reports if "new_filtered_count" in row]
    stable = [row for row in compared if row["status"] == "listing_compared"]
    provider = Counter(dict.fromkeys(("matched_provider_sha", "different_provider_sha", "no_comparable_sha"), 0))
    for row in compared:
        provider.update(row["provider_sha_comparison"])
    all_complete = len(stable) == len(targets)
    return {
        "target_count": len(targets), "listings_compared": len(compared), "listings_path_and_consistency_checked": len(stable),
        "baseline_old_filtered_total_all_targets": sum(len(direct_baseline(db, t["kind"], t["path"])) for t in targets),
        "old_filtered_total_compared_targets": sum(row["old_filtered_count"] for row in compared),
        "new_filtered_total_compared_targets": sum(row["new_filtered_count"] for row in compared),
        "observed_membership_difference_folders": sum(not row["observed_same_filtered_membership"] for row in compared),
        "observed_parent_difference_folders": sum(row.get("same_parent_as_reference") is False for row in reports),
        "unknown_parent_folders": len(targets) - sum(type(row.get("same_parent_as_reference")) is bool for row in reports),
        "observed_dataset_relative_path_difference_folders": sum(row.get("same_dataset_relative_path") is False for row in reports),
        "unknown_dataset_relative_path_folders": len(targets) - sum(type(row.get("same_dataset_relative_path")) is bool for row in reports),
        "all_filtered_memberships_observed_same": all(row["observed_same_filtered_membership"] for row in stable) if all_complete else None,
        "provider_sha_comparison_for_common_file_ids": dict(provider), "payload_bytes_verified": False,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=("verify", "check"))
    for name in ("archive", "scope-report", "reference", "work"):
        p.add_argument("--" + name, type=Path, required=True)
    args = p.parse_args()
    args.work.mkdir()
    report = {"status": "started", "parent_db_zip_sha256": baseline.ZIP_SHA, "scope_report_zip_sha256": REPORT_SHA,
              "budget": {}, "changes_list_calls": 0, "payload_reads": 0, "recursive_files_list_calls": 0}
    previous_handler = None
    try:
        with tempfile.TemporaryDirectory(dir=args.work) as tmp:
            path = Path(tmp)/"inventory.sqlite"
            baseline.extract_db(args.archive, path)
            with closing(baseline.read_only(path)) as db:
                owners, _, _ = baseline.db_contract(db, args.reference)
                selections = loaded_events(args.scope_report, db)
                report["selection"] = {"targets": [{**target, "baseline_direct_filtered_count": len(direct_baseline(db, target["kind"], target["path"]))}
                                                    for target in selections[0]], "other_ids": [e["file_id"] for e in selections[1]],
                                       "approved_removal_id": selections[2]["file_id"]}
                report["status"] = "verified_pinned_selection"
                if args.action == "check":
                    if hasattr(signal, "setitimer"):
                        def expired(_signum, _frame):
                            raise TimeoutError("v2_folders_hard_deadline")
                        previous_handler = signal.signal(signal.SIGALRM, expired)
                        signal.setitimer(signal.ITIMER_REAL, MAX_SECONDS)
                    run(db, owners, Metadata([r["id"] for r in selections[0]]), selections, report)
    except BaseException as error:
        report["status"] = "stopped" if isinstance(error, KeyboardInterrupt) else "partial_or_blocked"
        report["reason"] = type(error).__name__ + ": " + str(error)[:300]
        raise
    finally:
        if previous_handler is not None:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, previous_handler)
        (args.work/"report.json").write_bytes(canonical(report))
    print(canonical({"status": report["status"], "target_status_counts": report.get("target_status_counts")}).decode())


if __name__ == "__main__":
    main()
