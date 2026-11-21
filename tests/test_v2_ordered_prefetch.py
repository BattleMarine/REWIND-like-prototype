import hashlib
from pathlib import Path
import tempfile
from threading import Event
import unittest
from unittest.mock import patch

from rewind_v1 import v2_contract
from rewind_v1.v2_budget import Ledger
from scripts import v2_ordered_prefetch as prefetch


class PrefetchTests(unittest.TestCase):
    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.root = Path(folder.name)
        self.ledger = Ledger(self.root/"usage", {"fixture": True}, v2_contract.config())
        self.rows = []
        for index in range(4):
            path = self.root/f"frame-{index}"
            raw = bytes([index])*7
            path.write_bytes(raw)
            self.rows.append({"files": {"left": {"path": path.name, "size_bytes": len(raw),
                                                "sha256": hashlib.sha256(raw).hexdigest()}}})

    def test_out_of_order_reads_preserve_order_and_window(self):
        second = Event()
        reader = prefetch.read_payload

        def reordered(path, expected, check, observed):
            if path.name == "frame-0":
                if not second.wait(timeout=3):
                    raise TimeoutError("fixture second worker never started")
            value = reader(path, expected, check, observed)
            if path.name == "frame-1":
                second.set()
            return value

        metrics = {}
        with (patch.object(prefetch, "WINDOW_BYTES", 14),
              patch.object(prefetch, "read_payload", side_effect=reordered),
              self.ledger.invocation("parallel"),
              prefetch.ordered_sources(self.rows, self.root, self.root, 8, self.ledger, metrics) as stream):
            self.assertEqual([item["left"][0] for item in stream], [0,1,2,3])
        self.assertEqual(metrics["peak_window_bytes"], 14)
        self.assertEqual(metrics["peak_window_frames"], 2)
        self.assertEqual(metrics["source_bytes"], 28)
        self.assertEqual(self.ledger.snapshot()["invocations"][-1]["read_bytes_observed"], 28)

    def test_failed_partial_read_and_already_fetched_bytes_are_charged(self):
        second = Event()

        def partial(path, expected, check, observed):
            if path.name == "frame-0":
                observed(3)
                if not second.wait(timeout=3):
                    raise TimeoutError("fixture second worker never started")
                raise OSError("partial input")
            observed(expected)
            second.set()
            return path.read_bytes()

        metrics = {}
        with (patch.object(prefetch, "read_payload", side_effect=partial),
              self.assertRaisesRegex(OSError, "partial input"),
              self.ledger.invocation("failed"),
              prefetch.ordered_sources(self.rows[:2], self.root, self.root, 4, self.ledger, metrics) as stream):
            list(stream)
        self.assertEqual(metrics["source_bytes"], 10)
        self.assertEqual(self.ledger.snapshot()["invocations"][-1]["read_bytes_observed"], 10)

    def test_decode_failure_never_publishes_and_still_settles_input(self):
        metrics = {}
        with (patch.object(prefetch, "decode", side_effect=ValueError("fixture corrupt pose")),
              self.assertRaisesRegex(ValueError, "corrupt pose"),
              self.ledger.invocation("decode_failed")):
            prefetch.publish_shard(self.root, 0, {}, self.rows, {}, self.root, self.root, self.ledger,
                                   workers=4, metrics=metrics)
        self.assertGreaterEqual(metrics["source_bytes"], 7)
        self.assertLessEqual(metrics["source_bytes"], 28)
        self.assertFalse((self.root/"shard-00000").exists())

    def test_hash_failure_and_one_worker_accounting(self):
        self.rows[0]["files"]["left"]["sha256"] = "0"*64
        metrics = {}
        with (self.assertRaisesRegex(ValueError, "v2_source_sha"), self.ledger.invocation("serial"),
              prefetch.ordered_sources(self.rows, self.root, self.root, 1, self.ledger, metrics) as stream):
            next(stream)
        self.assertEqual(metrics["source_bytes"], 7)
        self.assertEqual(metrics["peak_window_frames"], 1)

    def test_timeout_drains_inflight_without_reading_next_payload(self):
        started = Event()
        clock = [0.0]
        ledger = Ledger(self.root/"timeout-usage", {"fixture": True}, v2_contract.config(), clock=lambda: clock[0])

        def expired(path, expected, check, observed):
            if path.name == "frame-0":
                observed(2)
                clock[0] = ledger.cfg["seconds_per_invocation"]
                started.set()
                check()
            if not started.wait(timeout=3):
                raise TimeoutError("fixture did not advance time")
            check()
            self.fail("another read passed the deadline")

        metrics = {}
        with (patch.object(prefetch, "read_payload", side_effect=expired),
              self.assertRaises(TimeoutError), ledger.invocation("deadline"),
              prefetch.ordered_sources(self.rows, self.root, self.root, 4, ledger, metrics) as stream):
            next(stream)
        self.assertEqual(metrics["source_bytes"], 2)
        self.assertEqual(ledger.snapshot()["invocations"][-1]["read_bytes_observed"], 2)
