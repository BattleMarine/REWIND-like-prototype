"""Scope comparison with mocked Drive and an in-memory pinned inventory shape."""
import hashlib
import sqlite3
from unittest import TestCase
from unittest.mock import patch

from scripts import build_v2_metadata_scope as builder
from scripts import check_v2_metadata_scope as scope


def node(object_id, parent=None, *, folder=True, **extra):
    return {"id": object_id, "parents": [parent] if parent else [], "mimeType": scope.FOLDER if folder else "application/zip",
            "trashed": False, "ownedByMe": True, "version": "7", **extra}


def change(item):
    return {"changeType": "file", "removed": False, "fileId": item["id"], "file": item,
            "time": "2026-11-19T04:35:00Z"}


class Client:
    def __init__(self, pages, metadata):
        self.pages, self.metadata = pages, metadata
        self.calls = []

    def changes(self, token, check, remaining):
        self.calls.append(("changes", token))
        return self.pages[token], 30

    def file(self, item, check, remaining):
        self.calls.append(("get", item))
        response = self.metadata[item]
        if isinstance(response, Exception):
            raise response
        return response, 20


class ScopeTests(TestCase):
    def setUp(self):
        self.db = sqlite3.connect(":memory:")
        self.addCleanup(self.db.close)
        for name in ("entries", "queue"):
            self.db.execute(f"CREATE TABLE {name}(kind,path,id)")
        self.db.execute("INSERT INTO queue VALUES ('rgb','','datasetRoot')")
        self.db.execute("INSERT INTO entries VALUES ('rgb','known/path','datasetItem')")
        self.removal = {"changeType": "file", "fileId": scope.REMOVED, "removed": True}

    def run_interval(self, client, evidence=None, report=None, recovery=False):
        report = {} if report is None else report
        output = {}
        scope.classify_interval(self.db, client, "saved", evidence, report, scope.Budget(output), recovery_provenance=recovery)
        return report, output

    def test_reads_all_pages_and_reports_known_and_unknown_even_when_blocked(self):
        external = node("unproven", "parentA", folder=False)
        metadata = {"root": node("myRoot"), "unproven": external, "parentA": node("parentA", "myRoot")}
        client = Client({
            "saved": {"changes": [change(node("datasetItem", "myRoot", folder=False))], "nextPageToken": "second"},
            "second": {"changes": [change(external), self.removal], "newStartPageToken": "terminal"},
        }, metadata)
        report = {}
        with self.assertRaisesRegex(ValueError, "not_proven"):
            self.run_interval(client, report=report)
        self.assertEqual(client.calls[:2], [("changes", "saved"), ("changes", "second")])
        self.assertEqual([e["status"] for e in report["events"]], ["blocked", "blocked", "approved_removal"])
        self.assertIn("historical_external", report["events"][1]["reason"])
        self.assertNotIn(("get", "unproven"), client.calls)
        self.assertEqual(report["events"][1]["chain"][1]["parents"], ["myRoot"])

    def test_pinned_historical_chain_positive_and_delta_check(self):
        obj = node("obj", "folderA", folder=False)
        folder = node("folderA", "myRoot")
        evidence = scope.evidence_index({
            "schema": "v2_external_parent_evidence_v1", "parent_zip_sha256": scope.parent.ZIP_SHA,
            "saved_cursor_sha256": hashlib.sha256(b"saved").hexdigest(), "my_drive_root_id": "myRoot",
            "basis": "fixture earlier captured chain",
            "items": {i["id"]: {"kind": "outside_before_saved_cursor", "parent_id": i["parents"][0],
                                 "version": "7", "mime_type": i["mimeType"], "source_sha256": "a"*64} for i in (obj, folder)},
        }, "saved")
        client = Client({"saved": {"changes": [self.removal, change(obj)], "newStartPageToken": "terminal"},
                         "terminal": {"changes": [], "newStartPageToken": "delta"}}, {"root": node("myRoot"), "obj": obj, "folderA": folder})
        report, budget = self.run_interval(client, evidence)
        self.assertEqual(report["events"][1]["status"], "outside_proven")
        self.assertEqual(client.calls[-1], ("changes", "terminal"))
        self.assertEqual(budget["requests"], 5)
        client.pages["terminal"]["changes"] = [{"changeType": "file", "fileId": "late", "removed": True}]
        report = {}
        with self.assertRaisesRegex(ValueError, "during_parent_inspection"):
            self.run_interval(client, evidence, report)
        self.assertEqual(report["during_inspection_events"][0]["file_id"], "late")

    def test_actual_recovery_signature_fixture_is_positive_only_for_exact_external_chain(self):
        with patch.object(scope, "datetime", wraps=scope.datetime) as clock:
            clock.now.return_value = scope.timestamp("2026-11-19T05:00:00Z")
            artifact = scope.RECOVERY_ARTIFACT
            reports, project, root = artifact["parents"]
            obj = node("recoveryZip", reports, folder=False, name=artifact["name"], size=artifact["size"],
                       sha256Checksum=artifact["sha256"], createdTime=artifact["creation_lower_utc"])
            client = Client({"saved": {"changes": [change(obj), self.removal], "newStartPageToken": "terminal"},
                             "terminal": {"changes": [], "newStartPageToken": "delta"}},
                            {"root": node(root), reports: node(reports, project), project: node(project, root), "recoveryZip": obj})
            report, _ = self.run_interval(client, recovery=True)
            self.assertEqual(report["events"][0]["reason"], "verified_recovery_created_outside_after_db")
            with self.assertRaisesRegex(ValueError, "not_proven"):
                self.run_interval(client)
            obj["sha256Checksum"] = "f"*64
            with self.assertRaisesRegex(ValueError, "not_proven"):
                self.run_interval(client, recovery=True)

    def test_account_root_and_two_copies_of_recovery_are_blocked(self):
        reports, project, root = scope.RECOVERY_ARTIFACT["parents"]
        root_metadata = node(root)
        with patch.object(scope, "datetime", wraps=scope.datetime) as clock:
            clock.now.return_value = scope.timestamp("2026-11-19T05:00:00Z")
            artifact = scope.RECOVERY_ARTIFACT
            one, two = (node(key, reports, folder=False, name=artifact["name"], size=artifact["size"],
                             sha256Checksum=artifact["sha256"], createdTime=artifact["creation_lower_utc"]) for key in ("one", "two"))
            client = Client({"saved": {"changes": [self.removal, change(one), change(two), change(root_metadata)],
                                        "newStartPageToken": "terminal"}},
                            {"root": root_metadata, reports: node(reports, project), project: node(project, root), "one": one, "two": two})
            results = {}
            with self.assertRaisesRegex(ValueError, "not_proven"):
                self.run_interval(client, {"root": root, "items": {}}, report=results, recovery=True)
            self.assertEqual(results["events"][-1]["reason"], "v2_scope_mydrive_root_change")
            client.pages["saved"]["changes"].pop()
            with self.assertRaisesRegex(ValueError, "not_proven"):
                self.run_interval(client, {"root": root, "items": {}}, report=results, recovery=True)
            self.assertEqual([event["reason"] for event in results["events"][1:]],
                             ["v2_scope_ambiguous_recovery_artifact_identity"]*2)

    def test_permissions_move_and_nonordinary_parent_are_fail_closed(self):
        for reason, folder in (("403", PermissionError("secret-token")),
                               ("shared", node("folderA", "myRoot", driveId="team")),
                               ("shortcut", dict(node("folderA", "myRoot"), mimeType=scope.SHORTCUT)),
                               ("ambiguous", dict(node("folderA"), parents=["one", "two"]))):
            with self.subTest(reason=reason):
                obj = node("obj", "folderA", folder=False)
                client = Client({"saved": {"changes": [self.removal, change(obj)], "newStartPageToken": "terminal"}},
                                {"root": node("myRoot"), "obj": obj, "folderA": folder})
                report = {}
                with self.assertRaisesRegex(ValueError, "not_proven"):
                    self.run_interval(client, report=report)
                self.assertEqual(report["events"][1]["status"], "blocked")
                self.assertNotIn("secret-token", str(report))

    def test_page_budget_keeps_inspected_rows_without_completing(self):
        client = Client({"saved": {"changes": [self.removal], "nextPageToken": "next"}}, {})
        report = {}
        with patch.object(scope.parent, "MAX_REQUESTS", 1):
            with self.assertRaisesRegex(TimeoutError, "api_budget"):
                self.run_interval(client, report=report)
        self.assertNotIn("pagination_complete", report)
        self.assertEqual(report["events"][0]["status"], "approved_removal")

    def test_new_notebook_compiles_and_pins_optional_evidence(self):
        doc = builder.notebook({"external_evidence_sha256": None})
        for cell in doc["cells"][1:]:
            compile("".join(cell["source"]), "scope-notebook", "exec")
        self.assertIn("scripts.check_v2_metadata_scope", "".join(doc["cells"][2]["source"]))

    def creation_fixture(self, created="2026-11-19T04:35:00Z"):
        obj = node("newObject", "outsideFolder", folder=False, createdTime=created)
        return Client({
            "saved": {"changes": [self.removal, change(obj)], "newStartPageToken": "terminal"},
            "terminal": {"changes": [], "newStartPageToken": "end"},
        }, {"root": node("myRoot"), "outsideFolder": node("outsideFolder", "myRoot"), "newObject": dict(obj)})

    def test_generic_creation_without_external_evidence_checks_current_metadata_and_delta(self):
        client = self.creation_fixture("2026-11-19T13:35:00.000000000+09:00")
        with patch.object(scope, "datetime", wraps=scope.datetime) as clock:
            clock.now.return_value = scope.timestamp("2026-11-19T05:00:00Z")
            report, _ = self.run_interval(client)
            self.assertEqual(report["events"][1]["reason"], "observed_external_creation_at_change_time_after_db")
            self.assertEqual(client.calls[-2:], [("get", "newObject"), ("changes", "terminal")])
            client.metadata["newObject"]["version"] = "8"
            report = {}
            with self.assertRaisesRegex(ValueError, "not_proven"):
                self.run_interval(client, report=report)
            self.assertEqual(report["events"][1]["reason"], "v2_scope_event_metadata_changed")

    def test_generic_creation_rejects_time_or_ancestry_mismatch_and_recovery_bypass(self):
        for case in ("submicrosecond", "cutoff", "changed_ancestor", "duplicate_event", "modified_after", "recovery"):
            with self.subTest(case=case), patch.object(scope, "datetime", wraps=scope.datetime) as clock:
                clock.now.return_value = scope.timestamp("2026-11-19T05:00:00Z")
                client = self.creation_fixture()
                events = client.pages["saved"]["changes"]
                event, metadata = events[1], client.metadata["newObject"]
                item = event["file"]
                if case == "submicrosecond":
                    item["createdTime"] = "2026-11-19T04:35:00.000000001Z"
                elif case == "cutoff":
                    item["createdTime"] = event["time"] = scope.DB_COMPLETED_UTC
                elif case == "changed_ancestor":
                    events.append({"changeType": "file", "fileId": "outsideFolder", "removed": True})
                elif case == "duplicate_event":
                    events.append(event.copy())
                elif case == "modified_after":
                    item["modifiedTime"] = "2026-11-19T04:35:01Z"
                else:
                    item["name"] = scope.RECOVERY_ARTIFACT["name"]
                metadata.update(item)
                report = {}
                with self.assertRaisesRegex(ValueError, "not_proven"):
                    self.run_interval(client, report=report)
                self.assertEqual(report["events"][1]["status"], "blocked")
