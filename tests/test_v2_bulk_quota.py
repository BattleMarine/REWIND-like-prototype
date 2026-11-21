import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from rewind_archive.contracts import canonical
from rewind_prepare.temporal_selection import seal
from rewind_v1 import v2_contract
from rewind_v1.v2_budget import Ledger
from scripts import build_v2_bulk_cache_budgeted as builder
from scripts.run_v2_bulk_cache import JOURNAL_SCHEMA, admit
from scripts.v2_bulk_quota import CLOSURE, METADATA_CAP, DriveQuota, TimeQuota, baseline_entries
from scripts.v2_cpu_cache_presets import NINETY_MINUTES, OVERNIGHT, SESSION_90, STANDARD, invocation_config


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

    def test_90min_lost_vm_keeps_previous_8hour_history_and_charges_only_5400(self):
        first = self.quota()
        old = first.reserve(first.history(initialize=True), 20)
        settled_old = first.settle(old, 445.6)
        current = self.quota()
        prior = current.history(required_tip=settled_old["content_sha256"], enforce_tip=True)
        self.assertEqual(prior["charged_seconds"], 536)
        short = current.reserve(prior, 20, session_cap=SESSION_90)
        self.assertEqual(short["seconds"], 5400)

        after_loss = self.quota()
        recovered = after_loss.history(required_tip=short["content_sha256"], enforce_tip=True)
        self.assertEqual(recovered["charged_seconds"], 536+5400)
        self.assertEqual(recovered["unsettled"], 1)
        next_short = after_loss.reserve(recovered, 20, session_cap=SESSION_90)
        settled = after_loss.settle(next_short, 5320)
        self.assertEqual(settled["seconds"], 5400)
        recovered = current.history(required_tip=settled["content_sha256"], enforce_tip=True)
        self.assertEqual(recovered["charged_seconds"], 536+5400+5400)

    def test_preset_keeps_standard_config_and_notebook_uses_90min_defaults(self):
        original = v2_contract.config()
        self.assertEqual(invocation_config(original, STANDARD)["resources"]["read_bytes"], 8*1024**3)
        short = invocation_config(original, NINETY_MINUTES)
        self.assertEqual(short["resources"]["read_bytes"], 112*1024**3)
        self.assertEqual(short["resources"]["write_bytes"], 24*1024**3)
        self.assertEqual(short["resources"]["rss_bytes"], original["resources"]["rss_bytes"])
        self.assertEqual(short["camera"], original["camera"])
        doc = builder.notebook({"source_name": "fixed.zip"}, {}, preset=NINETY_MINUTES)
        scope = {}
        with patch("builtins.print"):
            for cell in doc["cells"]:
                if cell["cell_type"] == "code":
                    exec(compile("".join(cell["source"]), "<notebook>", "exec"), scope)
        self.assertEqual(scope["LIMIT"], 5400)
        self.assertEqual(scope["PREFETCH_WORKERS"], 8)
        self.assertEqual(scope["EXPECTED_BUDGET_TIP_SHA256"], "")
        self.assertIn('["--preset", "90min"]', "".join(doc["cells"][-1]["source"]))

    def test_8h_preset_caps_deadline_and_unsettled_loss(self):
        cfg = invocation_config(v2_contract.config(), OVERNIGHT)
        self.assertEqual(cfg["resources"]["read_bytes"], 640*1024**3)
        self.assertEqual(cfg["resources"]["write_bytes"], 136*1024**3)
        self.assertEqual(cfg["resources"]["rss_bytes"], v2_contract.config()["resources"]["rss_bytes"])
        clock = [0.0]
        ledger = Ledger(self.root/"cutoff", {}, {**cfg, "seconds_per_invocation": 28800-CLOSURE},
                        clock=lambda: clock[0])
        sample = [{"files": {"left": {"size_bytes": 800}}}]
        with ledger.invocation("until-cutoff"):
            ledger.reserve("read", 8*1024**3)
            clock[0] = 28439
            self.assertIsNone(admit(sample, ledger, cfg))
            clock[0] = 28440
            self.assertEqual(admit(sample, ledger, cfg), "time")
        quota = self.quota()
        short = quota.reserve(quota.history(initialize=True), 0, session_cap=5400)
        settled = quota.settle(short, 300)
        long = quota.reserve(quota.history(required_tip=settled["content_sha256"]), 0, session_cap=28800)
        self.assertEqual(self.quota().history(required_tip=long["content_sha256"])["charged_seconds"], 28800+390)
        self.assertEqual(long["seconds"], 28800)

    def test_8h_notebook_default_off_with_no_copied_budget_tip(self):
        doc = builder.notebook({"source_name": "fixed.zip"}, {}, preset=OVERNIGHT)
        scope = {}
        with patch("builtins.print"):
            for cell in doc["cells"]:
                if cell["cell_type"] == "code":
                    exec(compile("".join(cell["source"]), "<notebook>", "exec"), scope)
        self.assertFalse(scope["RUN_APPROVED_CACHE"])
        self.assertEqual(scope["LIMIT"], 28800)
        self.assertEqual(scope["PREFETCH_WORKERS"], 8)
        self.assertEqual(scope["EXPECTED_BUDGET_TIP_SHA256"], "")
        self.assertIn('["--preset", "8h"]', "".join(doc["cells"][-1]["source"]))

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
