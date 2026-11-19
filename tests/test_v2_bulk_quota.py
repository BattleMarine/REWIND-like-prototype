import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from rewind_archive.contracts import canonical
from rewind_prepare.temporal_selection import seal
from scripts import build_v2_bulk_cache_budgeted as builder
from scripts.run_v2_bulk_cache import JOURNAL_SCHEMA
from scripts.v2_bulk_quota import CLOSURE, METADATA_CAP, DriveQuota, TimeQuota, baseline_entries


class QuotaTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)

    def quota(self):
        return TimeQuota(self.root, lambda: None, cap=45_000)

    def test_new_vm_charges_unsettled_reservation_and_reduces_next_watchdog_grant(self):
        first = self.quota()
        history = first.history(initialize=True, enforce_tip=True)
        ticket = first.reserve(history, 30)
        self.assertEqual(ticket["seconds"], 28_800)

        second = self.quota()
        with self.assertRaisesRegex(ValueError, "prior_tip_required"):
            second.history(enforce_tip=True)
        resume = second.history(required_tip=ticket["content_sha256"], enforce_tip=True)
        self.assertEqual(resume["charged_seconds"], 28_800)
        self.assertEqual(resume["unsettled"], 1)
        next_ticket = second.reserve(resume, 17)
        self.assertEqual(next_ticket["seconds"], 16_200)
        settlement = second.settle(next_ticket, 305.3)
        self.assertEqual(settlement["measured_seconds"], 306)
        self.assertEqual(settlement["seconds"], 306 + CLOSURE)
        after = first.history(required_tip=settlement["content_sha256"], enforce_tip=True)
        self.assertEqual(after["charged_seconds"], 28_800 + 306 + CLOSURE)
        remaining = first.reserve(after, 0)
        self.assertEqual(remaining["seconds"], 45_000-after["charged_seconds"])
        self.assertIsNone(first.reserve(first.history(), 0))

    def test_prior_tip_detects_rollback_and_missing_history_cannot_reset(self):
        quota = self.quota()
        # Simulate process death immediately after mkdir but before the origin marker.
        quota.root.mkdir()
        state = quota.history(initialize=True)
        ticket = quota.reserve(state, 0)
        settlement = quota.settle(ticket, 100)
        quota.path(0, "settle").unlink()
        with self.assertRaisesRegex(ValueError, "history_gap_or_rollback"):
            quota.history(required_tip=settlement["content_sha256"])
        quota.path(0, "reserve").unlink()
        with self.assertRaisesRegex(ValueError, "history_witness"):
            quota.history(initialize=True)
        quota.root.rmdir()
        with self.assertRaisesRegex(ValueError, "history_missing"):
            quota.history(initialize=True)

    def seed_baseline(self):
        entries = []
        for number in range(4):
            package = f"shard-{number:05d}-baseline.zip"
            boundary = seal({"schema": JOURNAL_SCHEMA, "index": number, "frames": 128,
                             "previous": entries[-1]["content_sha256"] if entries else None,
                             "package": {"name": package}})
            entries.append(boundary)
            (self.root / f"boundary-{number:05d}.json").write_bytes(canonical(boundary))
            if number:
                (self.root / package).write_bytes(b"pre-approved bytes")
        return entries

    def test_drive_counts_additional_published_and_interrupted_payloads(self):
        entries = self.seed_baseline()
        self.assertEqual(baseline_entries(self.root, entries[-1]["content_sha256"]), entries)
        with self.assertRaisesRegex(ValueError, "baseline_pin"):
            baseline_entries(self.root, "f"*64)
        (self.root / "shard-00004-committed.zip").write_bytes(b"abc")
        (self.root / "boundary-00004.json").write_bytes(b"de")
        staging = self.root / ".incoming-example"
        staging.mkdir()
        (staging / "payload.zip.partial").write_bytes(b"fghi")
        (self.root / ".publish-unfinished").write_bytes(b"j")
        quota = DriveQuota(self.root, entries, cap=METADATA_CAP+11)
        self.assertEqual(quota.used, 10)
        self.assertTrue(quota.admit(1))
        self.assertFalse(quota.admit(2))
        restarted = DriveQuota(self.root, entries)
        self.assertEqual(restarted.used, 10)

    def test_budget_notebook_default_off_and_paused_report(self):
        doc = builder.notebook({"source_name": "fixed.zip"}, {"scripts/example.py": "# pinned"})
        scope = {}
        with patch("builtins.print"):
            for cell in doc["cells"]:
                if cell["cell_type"] == "code":
                    exec(compile("".join(cell["source"]), "<notebook>", "exec"), scope)
        self.assertFalse(scope["RUN_APPROVED_CACHE"])
        cell = "".join(doc["cells"][-1]["source"])
        self.assertIn("scripts.run_v2_bulk_cache_budgeted", cell)
        self.assertIn("quota-grant.json", cell)
        self.assertNotIn("TARGET_TOTAL_SHARDS", cell)
        started = scope["time"].monotonic()
        (self.root / "quota-grant.json").write_bytes(canonical({"deadline": started+15}))
        self.assertTrue(0 < scope["diagnostic_seconds"](self.root, started) <= 15)
        (self.root / "result.json").write_bytes(canonical({"status": "paused", "budget_tip_sha256": "a"*64}))
        with patch("builtins.print"):
            scope["save_report"](self.root, None, scope["time"].monotonic(), None)
        report = json.loads((self.root / "diagnostic.json").read_bytes())
        self.assertEqual(report["status"], "paused")
        self.assertEqual(report["worker"]["budget_tip_sha256"], "a"*64)
