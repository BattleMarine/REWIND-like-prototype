"""Changes-only budget and post-fence data reconciliation fixtures."""

from contextlib import closing
import json
from pathlib import Path
import sqlite3
import tempfile
from unittest import TestCase
from unittest.mock import patch

from rewind_archive.contracts import canonical
from rewind_v1.v2_contract import SCENARIOS
from scripts import build_v2_metadata_fence as builder
from scripts import check_v2_metadata_fence as worker


class FenceTests(TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.work = Path(tmp.name)
        self.event = {"changeType": "file", "fileId": worker.REMOVED, "removed": True}

    def test_uses_saved_cursor_through_terminal_page(self):
        observed = []

        def get(corpus, token, check, remaining):
            observed.append((corpus, token, remaining))
            return ({"changes": [self.event], "nextPageToken": "t2"}, 90) if token == "saved" else (
                {"changes": [], "newStartPageToken": "final"}, 40)

        report = {}
        worker.verify_changes(get, "", "saved", report)
        self.assertEqual(observed, [("", "saved", worker.MAX_BYTES), ("", "t2", worker.MAX_BYTES-90)])
        self.assertEqual(report["requests"], 2)
        self.assertEqual(report["response_bytes"], 130)
        self.assertEqual(report["allowed_change"], self.event)

    def test_late_change_or_no_removal_never_runs_comparison(self):
        cases = (
            ([{"changes": []}], "expected_removal_missing"),
            ([{"changes": [self.event], "nextPageToken": "t2"}, {"changes": [self.event]}], "unexpected_change"),
            ([{"changes": [dict(self.event, fileId="other")]}], "unexpected_change"),
            ([{"changes": [], "nextPageToken": "saved"}], "repeated_cursor"),
        )
        for pages, reason in cases:
            with self.subTest(reason=reason):
                pages = [{**page, **({} if "nextPageToken" in page else {"newStartPageToken": "final"})} for page in pages]
                with patch.object(worker, "compare") as compare:
                    report = {"changes": {}}
                    with self.assertRaisesRegex(ValueError, reason):
                        worker.run_check(None, {}, self.work, lambda *args: (pages.pop(0), 20), "", "saved", report)
                    compare.assert_not_called()
                    if reason == "unexpected_change":
                        diagnosis = report["changes"]["first_unexpected_change"]
                        self.assertEqual(diagnosis["change_type"], "file")
                        self.assertNotIn("other", json.dumps(diagnosis))

    def test_request_response_and_clock_caps(self):
        report = {}
        with self.assertRaisesRegex(TimeoutError, "response_limit"):
            worker.verify_changes(lambda *args: ({"changes": [], "nextPageToken": str(report["requests"])}, 10), "", "saved", report)
        self.assertEqual(report["requests"], 16)
        with self.assertRaisesRegex(ValueError, "bytes_limit"):
            worker.verify_changes(lambda *args: ({}, worker.MAX_BYTES+1), "", "saved", {})
        ticks = iter((0, 121, 121))
        with self.assertRaisesRegex(TimeoutError, "wall_limit"):
            worker.verify_changes(lambda *args: self.fail("deadline before network"), "", "saved", {}, clock=lambda: next(ticks))

    def test_verified_fixture_reports_clean_missing_duplicate_and_sha(self):
        seq = SCENARIOS[0]+"/Day/Avatar/Actor/001/motion"
        absent = SCENARIOS[1]+"/Day/Avatar/Actor/001/motion"
        with closing(sqlite3.connect(":memory:")) as db:
            db.execute("CREATE TABLE entries(kind,path,id,row)")
            paths = [
                ("left", "00001"), ("right", "00001"), ("pose", "00001"),
                ("left", "00002"), ("left", "2"), ("pose", "00002"),
            ]
            for i, (role, frame) in enumerate(paths):
                path = seq+(f"/json/frame_{frame}.json" if role == "pose" else f"/fisheye_final_image/camera_{role}/final_{frame}.png")
                row = {"file_id": str(i), "path": path, "version": "1", "size_bytes": 23, "sha256": None,
                       "checksum_source": "missing", "scenario": SCENARIOS[0], "role": role}
                db.execute("INSERT INTO entries VALUES (?,?,?,?)", ("pose" if role == "pose" else "rgb", path, str(i), canonical(row)))
            report = {"changes": {}}
            worker.run_check(db, {seq: "train", absent: "validation"}, self.work,
                             lambda *args: ({"changes": [self.event], "newStartPageToken": "final"}, 20), "", "saved", report)
        self.assertEqual(report["status"], "comparison_complete")
        cell = report["counts"][SCENARIOS[0]]["train"]
        self.assertEqual((cell["frame_keys"], cell["clean_triplets"], cell["missing_role_keys"], cell["duplicate_role_keys"]), (2, 1, 1, 1))
        self.assertEqual(cell["missing_sha256_bytes"], 138)
        self.assertEqual(report["counts"][SCENARIOS[1]]["validation"]["missing_sequence_keys"], 1)
        self.assertFalse(report["frame_correspondence_passed"])
        self.assertEqual(report["totals"]["clean_triplets"], 1)
        first = json.loads((self.work/"issues.jsonl").read_bytes().splitlines()[0])
        self.assertEqual(first["missing"], ["right"])
        self.assertEqual(len(first["duplicates"]["left"]), 2)

    def test_notebook_is_default_off_and_compiles(self):
        doc = builder.notebook({})
        settings = "".join(doc["cells"][1]["source"])
        self.assertIn("RUN_CHANGES_AND_COMPARE = False", settings)
        self.assertIn("SAVE_RECOVERY_TO_DRIVE = False", settings)
        for c in doc["cells"][1:]:
            compile("".join(c["source"]), "fixed_cells", "exec")
