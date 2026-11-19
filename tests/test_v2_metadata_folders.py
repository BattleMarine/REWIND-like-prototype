"""Local fixtures for the direct-folder diagnostic and its recovery notebook."""

import sqlite3
from unittest import TestCase

from rewind_archive.contracts import canonical
from rewind_v1.v2_contract import SCENARIOS
from scripts import build_v2_metadata_folders as builder
from scripts import check_v2_metadata_folders as job


def item(name, object_id, parent, *, mime=job.FOLDER, **extra):
    return {"name": name, "id": object_id, "parents": [parent] if parent else [],
            "mimeType": mime, "trashed": False, "ownedByMe": True, "version": "1", **extra}


class Client:
    def __init__(self, items, pages):
        self.items, self.pages = items, pages
        self.calls = []

    def file(self, object_id, check, free):
        self.calls.append(("get", object_id))
        result = self.items[object_id]
        if isinstance(result, list):
            return result.pop(0), 10
        if isinstance(result, Exception):
            raise result
        return result, 10

    def children(self, object_id, token, check, free):
        self.calls.append(("list", object_id, token))
        return self.pages[object_id, token], 100


class FolderTests(TestCase):
    def setUp(self):
        self.db = sqlite3.connect(":memory:")
        self.addCleanup(self.db.close)
        self.db.executescript("CREATE TABLE queue(kind,path,id); CREATE TABLE entries(kind,path,id,row)")
        self.seq = SCENARIOS[0]+"/Day/Actor/Model/001/motion"
        self.path = self.seq+"/json"
        self.owners = {self.seq: "train"}
        self.db.executemany("INSERT INTO queue VALUES (?,?,?)", [
            ("pose", "", "dataset"), ("pose", self.seq, "sequence"), ("pose", self.path, "leaf"),
        ])
        self.file = item("frame_00.json", "known", "leaf", mime="application/json", size="20", sha256Checksum="a"*64)
        self.db.execute("INSERT INTO entries VALUES (?,?,?,?)", ("pose", self.path+"/frame_00.json", "known",
                        canonical({"size_bytes": 20, "version": "1", "sha256": "a"*64})))

    def test_baseline_direct_not_recursive_and_nonleaf_does_not_read_entries(self):
        traced = []
        self.db.set_trace_callback(traced.append)
        self.assertEqual(job.direct_baseline(self.db, "pose", ""), {})
        self.assertFalse(any("FROM entries" in s for s in traced))
        self.assertEqual(list(job.direct_baseline(self.db, "pose", self.seq)), ["leaf"])
        self.assertEqual(list(job.direct_baseline(self.db, "pose", self.path)), ["known"])

    def test_compare_detects_replacement_rename_and_excluded_unknown_history(self):
        old = job.direct_baseline(self.db, "pose", self.path)
        new = item("frame_00.json", "replacement", "leaf", mime="application/json")
        extra = item("extraFolder", "outside", "leaf")
        current, outside, errors = job.filter_children([new, extra], "pose", self.path, {self.path}, self.owners)
        compared = job.compare_children(old, current, outside, errors)
        self.assertFalse(compared["observed_same_filtered_membership"])
        self.assertEqual({r["kind"] for r in compared["changes"]}, {"replaced_at_name", "added_to_filtered_listing"})
        self.assertEqual(compared["excluded_children_prior_state"], "unknown")
        self.assertEqual(compared["current_excluded_children"][0]["id"], "outside")
        renamed = dict(self.file, name="frame_01.json", sha256Checksum=None)
        current, excluded, errors = job.filter_children([renamed], "pose", self.path, {self.path}, self.owners)
        result = job.compare_children(old, current, excluded, errors)
        self.assertEqual(result["changes"][0]["kind"], "rename_or_type_change")
        self.assertEqual(result["provider_sha_comparison"]["no_comparable_sha"], 1)
        self.assertFalse(result["payload_bytes_verified"])

    def test_complete_direct_pages_and_incomplete_never_compare(self):
        child2 = item("frame_01.json", "new", "leaf", mime="application/json")
        client = Client({}, {("leaf", None): {"files": [self.file], "nextPageToken": "next"},
                             ("leaf", "next"): {"files": [child2]}})
        rows = []
        job.child_pages(client, job.Budget({}), "leaf", rows)
        self.assertEqual(client.calls, [("list", "leaf", None), ("list", "leaf", "next")])
        self.assertEqual(len(rows), 2)
        client.pages["leaf", "next"]["incompleteSearch"] = True
        rows.clear()
        with self.assertRaisesRegex(ValueError, "incomplete"):
            job.child_pages(client, job.Budget({}), "leaf", rows)
        self.assertEqual(rows, [self.file])

    def test_concurrent_target_version_is_inconclusive_and_partial_not_equal(self):
        leaf = item("json", "leaf", "sequence")
        moved = dict(leaf, version="2")
        root = item("My Drive", "myRoot", None)
        items = {"root": root, "myRoot": root, "leaf": [leaf, moved], "sequence": item("motion", "sequence", "dataset"),
                 "dataset": item("data", "dataset", "myRoot")}
        client = Client(items, {("leaf", None): {"files": [self.file]}})
        target = {"id": "leaf", "kind": "pose", "path": self.path, "event_parent_ids": ["sequence"]}
        report = {"budget": {}}
        job.run(self.db, self.owners, client, ([target], [], {"file_id": "deleted"}), report)
        compared = report["targets"][0]
        self.assertTrue(compared["observed_same_filtered_membership"])
        self.assertEqual(compared["status"], "inconclusive_concurrency")
        self.assertIsNone(report["summary"]["all_filtered_memberships_observed_same"])
        self.assertEqual(report["summary"]["provider_sha_comparison_for_common_file_ids"]["matched_provider_sha"], 1)
        self.assertEqual([call[1] for call in client.calls if call[0] == "list"], ["leaf"])

    def test_recovery_zip_comparison_and_unavailable_other_is_unknown(self):
        old = job.scope.RECOVERY_ARTIFACT
        reports, project, root = old["parents"]
        event_id = "1Lexz55SEoGr_WebvN8LFeNoAOWbM-6Y2"
        metadata = {"root": item("My Drive", root, None), root: item("My Drive", root, None),
                    reports: item("reports", reports, project), project: item("Project", project, root),
                    event_id: item(old["name"], event_id, reports, mime="application/zip", size=old["size"],
                                   sha256Checksum=old["sha256"], createdTime="2026-11-19T04:33:00Z"),
                    "lost": FileNotFoundError("secret-url")}
        inspector = job.Inspector(self.db, Client(metadata, {}), job.Budget({}))
        event = {"file_id": event_id, "event_time": "2026-11-19T04:33:17Z", "removed": False, "reason": "old",
                 "event_parent_ids": [reports]}
        result = inspector.inspect_other(event, set(), {reports})
        self.assertEqual(result["recovery_artifact"]["reference"], "043247_fence_report")
        self.assertTrue(result["recovery_artifact"]["sha256_matches"])
        self.assertTrue(result["recovery_artifact"]["prior_gate_would_reject_observed_changed_ancestors"])
        unknown = inspector.inspect_other(dict(event, file_id="lost", removed=True), set(), set())
        self.assertEqual(unknown["status"], "unknown")
        self.assertNotIn("secret-url", str(unknown))

    def test_missing_parent_is_unknown_not_changed_even_if_children_can_be_compared(self):
        leaf = dict(item("json", "leaf", "sequence"), parents=None)
        root = item("My Drive", "myRoot", None)
        client = Client({"root": root, "leaf": leaf}, {("leaf", None): {"files": [self.file]}})
        target = {"id": "leaf", "kind": "pose", "path": self.path, "event_parent_ids": ["sequence"]}
        result = {"budget": {}}
        job.run(self.db, self.owners, client, ([target], [], {"file_id": "deleted"}), result)
        self.assertEqual(result["targets"][0]["status"], "listing_compared_path_unverified")
        self.assertIsNone(result["targets"][0]["same_parent_as_reference"])
        self.assertIsNone(result["targets"][0]["same_dataset_relative_path"])
        self.assertEqual(result["summary"]["observed_dataset_relative_path_difference_folders"], 0)
        self.assertEqual(result["summary"]["unknown_dataset_relative_path_folders"], 1)

    def test_notebook_default_off_and_compiles(self):
        doc = builder.notebook({})
        self.assertIn("RUN_DIRECT_COMPARE = False", "".join(doc["cells"][1]["source"]))
        for c in doc["cells"][1:]:
            compile("".join(c["source"]), "folder-notebook", "exec")
