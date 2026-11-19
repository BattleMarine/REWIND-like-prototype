import hashlib
from pathlib import Path
import sqlite3
import tempfile
import unittest
from contextlib import closing
from unittest.mock import patch

from rewind_archive.contracts import canonical
from rewind_v1 import v2_contract
from scripts import list_v2_drive_metadata as listing
from scripts import produce_v2_inputs as producer


def sha(data):
    return hashlib.sha256(data).hexdigest()


class MetadataService:
    def __init__(self, roots, owners):
        self.requests = []
        self.tree = {}
        self.missing_checksum = False
        self.changed = False
        self.incomplete = False
        for kind, root in roots.items():
            for folder in sorted(listing.folders(owners, kind)):
                parent, _, name = folder.rpartition("/")
                parent_id = sha((kind + parent).encode()) if parent else root
                self.tree.setdefault(parent_id, []).append(
                    {"id": sha((kind + folder).encode()), "name": name, "parents": [parent_id], "mimeType": listing.FOLDER}
                )
                extension = {"json": "frame_0.json", "camera_left": "final_0.png", "camera_right": "final_0.png"}
                if name in extension:
                    folder_id = sha((kind + folder).encode())
                    self.tree[folder_id] = [{"id": sha((folder + "/file").encode()), "name": extension[name],
                                             "parents": [folder_id], "mimeType": "application/octet-stream",
                                             "size": "100", "version": "7", "sha256Checksum": "a"*64}]

    def root(self, folder, check, limit):
        return {"id": folder, "mimeType": listing.FOLDER, "trashed": False}, 100

    def start(self, corpus, check, limit):
        return {"startPageToken": "start-1"}, 100

    def changes(self, corpus, token, check, limit):
        return {"changes": [{"changeType": "file"}] if self.changed else [], "newStartPageToken": "start-2"}, 100

    def page(self, parent, token, check, limit):
        self.requests.append((parent, token))
        rows = self.tree[parent]
        index = int(token or 0)
        row = dict(rows[index])
        if self.missing_checksum:
            row.pop("sha256Checksum", None)
        page = {"files": [row]}
        if self.incomplete:
            page["incompleteSearch"] = True
        if index + 1 < len(rows):
            page["nextPageToken"] = str(index + 1)
        return page, 300


class ProducerTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.work = Path(tmp.name)
        self.roots = {"pose": "pose_root", "rgb": "rgb_root"}
        self.owners = {
            "/".join((scene, "Day", "Avatar", "Actor", "001", split)): split
            for scene in v2_contract.SCENARIOS for split in ("train", "validation", "test")
        }
        self.split_pins = {}
        for split in ("train", "validation", "test"):
            sequences = sorted(s for s in self.owners if self.owners[s] == split)
            raw = ("\n".join(sequences) + "\n").encode()
            (self.work / (split + ".txt")).write_bytes(raw)
            self.split_pins[split] = {"sha256": sha(raw), "rows": len(sequences)}
        spec = {"joint_names": list(range(62)), "pinned_sources": {"split_sha256": {s: p["sha256"] for s, p in self.split_pins.items()}}}
        self.spec_path = self.work / "spec.json"
        self.spec_path.write_bytes(canonical(spec))
        self.pins = {"spec_file_sha256": sha(canonical(spec)), "official_splits": self.split_pins}
        self.client = MetadataService(self.roots, self.owners)

    def scan(self, db, limit, pins=None):
        return listing.collect(db, self.client, self.roots, self.owners, pins or self.pins,
                               max_requests=limit, max_seconds=20, max_bytes=2**20)

    def produce(self, evidence=None):
        with patch.object(producer, "validate_spec", side_effect=lambda x: x):
            return producer.produce(self.work / "listing/receipt.json", self.receipt_sha, self.spec_path,
                                    self.pins["spec_file_sha256"], self.work, self.work / "result",
                                    evidence, sha(evidence.read_bytes()) if evidence else None)

    def listing(self, missing=False):
        self.client.missing_checksum = missing
        database = self.work / "inventory.db"
        db = sqlite3.connect(database)
        self.assertEqual(self.scan(db, 4)["status"], "partial")
        db.close()
        with closing(sqlite3.connect(database)) as resumed:
            with self.assertRaisesRegex(ValueError, "resume_binding"):
                self.scan(resumed, 200, {"wrong": "pins"})
            self.assertEqual(self.scan(resumed, 300)["status"], "complete")
            self.receipt_sha = listing.export(resumed, self.work / "listing", self.roots, self.pins)
        self.assertEqual(len(set(self.client.requests)), len(self.client.requests))

    def test_resume_and_load_exact_manifest(self):
        self.listing()
        report = self.produce()
        self.assertEqual(report["status"], "metadata_pinned")
        self.assertEqual(report["pair_count"], 6)
        path = self.work / "result/audited_manifest.json"
        self.assertEqual(report["candidate_json_bytes"], path.stat().st_size)
        doc, _, _ = v2_contract.load(path, report["manifest_file_sha256"], self.spec_path,
                                     self.pins["spec_file_sha256"], self.work)
        self.assertEqual(doc["provenance"]["decode_audit"], "not_run")

    def test_absent_hashes_block_and_report_bytes(self):
        self.listing(missing=True)
        report = self.produce()
        self.assertEqual(report["status"], "blocked_input_evidence")
        self.assertEqual(report["missing_sha256_bytes"], 1800)
        self.assertFalse((self.work / "result/audited_manifest.json").exists())

    def test_conflicting_or_different_version_evidence_does_not_fill(self):
        self.listing(missing=True)
        evidence = self.work / "supplement.jsonl"
        rows = []
        for kind in self.roots:
            path = self.work / "listing" / (kind + ".jsonl")
            expected = sha(path.read_bytes())
            for row in producer.jsonl(path, expected):
                rows.append({**row, "sha256": "b"*64, "source": "bounded_payload_sha256", "evidence_sha256": "c"*64})
        rows[0]["version"] = "8"
        evidence.write_bytes(b"".join(canonical(row) + b"\n" for row in rows))
        self.assertEqual(self.produce(evidence)["missing_sha256_files"], 1)

    def test_changed_scope_and_incomplete_pages_do_not_export(self):
        with closing(sqlite3.connect(":memory:")) as db:
            self.client.incomplete = True
            with self.assertRaisesRegex(ValueError, "incomplete_search"):
                self.scan(db, 300)
            self.assertEqual(db.execute("SELECT count(*) FROM queue WHERE done=1").fetchone(), (0,))
            with self.assertRaisesRegex(ValueError, "complete_cursor"):
                listing.export(db, self.work / "rejected", self.roots, self.pins)
            self.client.incomplete = False
            self.client.changed = True
            self.assertEqual(self.scan(db, 300)["status"], "source_changed")
            with self.assertRaisesRegex(ValueError, "complete_cursor"):
                listing.export(db, self.work / "rejected", self.roots, self.pins)

    def test_oversize_candidate_preserves_exact_sha_without_runtime_pin(self):
        self.listing()
        with patch.object(producer, "LOADER_CAP", 1):
            report = self.produce()
        self.assertEqual(report["status"], "blocked_loader_cap")
        self.assertRegex(report["candidate_file_sha256"], r"^[a-f0-9]{64}$")
        self.assertIsNone(report["manifest_file_sha256"])
        self.assertFalse((self.work / "result/audited_manifest.json").exists())

    def test_request_failure_reuses_cursor_and_duplicate_path_blocks(self):
        with closing(sqlite3.connect(":memory:")) as db:
            self.assertEqual(self.scan(db, 4)["status"], "partial")
            pending = db.execute("SELECT id,token FROM queue WHERE done=0 ORDER BY kind,path LIMIT 1").fetchone()
            with patch.object(self.client, "page", side_effect=OSError("injected request failure")):
                with self.assertRaises(OSError):
                    self.scan(db, 300)
            self.assertEqual(db.execute("SELECT id,token FROM queue WHERE done=0 ORDER BY kind,path LIMIT 1").fetchone(), pending)
            duplicate = dict(self.client.tree[pending[0]][0])
            duplicate["id"] = "duplicate_folder"
            self.client.tree[pending[0]][int(pending[1])] = duplicate
            with self.assertRaisesRegex(ValueError, "duplicate_folder_path"):
                self.scan(db, 300)
            with self.assertRaisesRegex(ValueError, "complete_cursor"):
                listing.export(db, self.work / "rejected", self.roots, self.pins)

    def test_pinned_resume_records_exact_trashed_folder_then_checks_again(self):
        removed, parent = "discarded", "parent"
        event = {"changeType": "file", "fileId": removed, "removed": False,
                 "file": {"id": removed, "mimeType": listing.FOLDER, "parents": [parent], "trashed": True}}
        with closing(sqlite3.connect(":memory:")) as db:
            self.assertEqual(self.scan(db, 4)["status"], "partial")
            with patch.object(self.client, "changes", side_effect=[
                ({"changes": [event], "newStartPageToken": "after-removal"}, 100),
                ({"changes": [], "newStartPageToken": "final"}, 100),
            ]) as call:
                report = listing.collect(db, self.client, self.roots, self.owners, self.pins,
                                         max_requests=300, max_seconds=20, max_bytes=2**20,
                                         resume_removal=removed, resume_parent=parent)
                self.assertEqual(report["status"], "complete")
                self.assertEqual([args.args[1] for args in call.call_args_list], ["start-1", "after-removal"])
            listing.export(db, self.work / "resumed", self.roots, self.pins)
            receipt = producer.pinned_json(self.work / "resumed/receipt.json",
                                           sha((self.work / "resumed/receipt.json").read_bytes()), 2**20)
            self.assertEqual(receipt["allowed_changes"], [{**event, "corpus": "", "phase": "preflight"}])

    def test_resume_stops_before_folder_pages_on_other_change_or_wrong_parent(self):
        removed, parent = "discarded", "parent"
        allowed = {"changeType": "file", "fileId": removed, "removed": True}
        trashed = {"changeType": "file", "fileId": removed, "removed": False,
                   "file": {"id": removed, "mimeType": listing.FOLDER, "parents": ["wrong"], "trashed": True}}
        for events in ([allowed, {"changeType": "file", "fileId": "other", "removed": True}], [trashed]):
            with self.subTest(events=events), closing(sqlite3.connect(":memory:")) as db:
                self.scan(db, 4)
                with patch.object(self.client, "changes", return_value=({"changes": events, "newStartPageToken": "next"}, 100)):
                    with patch.object(self.client, "page", side_effect=AssertionError("must stop before listing")):
                        report = listing.collect(db, self.client, self.roots, self.owners, self.pins,
                                                 max_requests=300, max_seconds=20, max_bytes=2**20,
                                                 resume_removal=removed, resume_parent=parent)
                self.assertEqual(report["status"], "source_changed")

    def test_resume_wall_evidence_and_default_off_notebook(self):
        from scripts import build_v2_metadata_resume as builder, resume_v2_drive_metadata as resume
        with self.assertRaisesRegex(ValueError, "prior_wall_evidence_required"):
            resume.remaining_seconds(None, None)
        evidence = self.work / "elapsed.json"
        evidence.write_bytes(canonical({"schema": "v2_metadata_prior_wall_upper_v1",
                                        "result_zip_sha256": resume.ZIP_SHA, "prior_wall_upper_seconds": 1500,
                                        "basis": "fixture upper bound"}))
        self.assertEqual(resume.remaining_seconds(evidence, sha(evidence.read_bytes())), 12600)
        with self.assertRaises(ValueError):
            resume.remaining_seconds(evidence, "0"*64)
        doc = builder.notebook({"elapsed_evidence_sha256": None})
        settings = "".join(doc["cells"][1]["source"])
        compile(settings, "settings", "exec")
        compile("".join(doc["cells"][2]["source"]), "notebook", "exec")
        self.assertIn("RUN_RESUME = False", settings)
        self.assertIn("VERIFY_ONLY = False", settings)
        self.assertIn("PRIOR_WALL_UPPER_SECONDS = None", settings)
        self.assertIn("SAVE_RECOVERY_TO_DRIVE = False", settings)

    def test_recovery_copy_readback_preserves_local_and_refuses_overwrite(self):
        from scripts import resume_v2_drive_metadata as resume
        local = self.work / "result.zip"
        local.write_bytes(b"local backup bytes")
        destination = self.work / "copy"
        destination.mkdir()
        receipt = resume.save_recovery(local, destination, 3)
        saved = destination / local.name
        self.assertEqual(receipt["sha256"], sha(local.read_bytes()))
        self.assertEqual(saved.read_bytes(), local.read_bytes())
        with self.assertRaises(FileExistsError):
            resume.save_recovery(local, destination, 3)
        self.assertEqual(saved.read_bytes(), b"local backup bytes")
        self.assertEqual(local.read_bytes(), b"local backup bytes")

    def test_removed_event_is_not_allowed_again_in_final_fence(self):
        removed, parent = "discarded", "parent"
        event = {"changeType": "file", "fileId": removed, "removed": True}
        with closing(sqlite3.connect(":memory:")) as db:
            self.scan(db, 4)
            with patch.object(self.client, "changes", side_effect=[
                ({"changes": [event], "nextPageToken": "next-page"}, 100),
                ({"changes": [], "newStartPageToken": "after-removal"}, 100),
                ({"changes": [event], "newStartPageToken": "final"}, 100),
            ]):
                report = listing.collect(db, self.client, self.roots, self.owners, self.pins,
                                         max_requests=300, max_seconds=20, max_bytes=2**20,
                                         resume_removal=removed, resume_parent=parent)
            self.assertEqual(report["status"], "source_changed")
            self.assertEqual(db.execute("SELECT count(*) FROM resume_removals").fetchone(), (1,))
            with self.assertRaisesRegex(ValueError, "complete_cursor"):
                listing.export(db, self.work / "rejected", self.roots, self.pins)
