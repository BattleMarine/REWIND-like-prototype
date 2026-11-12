"""Regression from the frozen real 50-frame metadata, without original RGB/pose reads."""

from collections import Counter
import copy
import json
from pathlib import Path
from types import SimpleNamespace
from unittest import TestCase

from rewind_archive.contracts import canonical
from rewind_prepare.converter import sha
from rewind_v1.oracle_data import FRAMES, HISTORICAL, SEQUENCE, START, _check_clip_file_pins


class OracleFilePinTests(TestCase):
    def setUp(self):
        raw = (Path(__file__).parent / "fixtures/v14_clip_file_pins.json").read_bytes()
        fixture = json.loads(raw)
        self.assertEqual(sha(canonical(fixture)), "4b2cfd6b54bfa2ef349ea43344380b194414ac7285f3c8d00eb5ed6f8690f584")
        self.assertEqual(fixture["provenance"], {
            "input_zip_sha256": "6256eb7bd148ec51bdb94627c68e54657be3808c607a12e13b68f7e479b15e97",
            "selection_sha256": HISTORICAL["selection_sha256"], "cache_sha256": HISTORICAL["cache_sha256"]})
        self.plan = {"records": fixture["selection_records"]}
        self.index = {tuple(row["key"]): ("part-0000.zip", row) for row in fixture["cache_records"]}
        self.keys = [(SEQUENCE, i) for i in range(START, START+FRAMES)]
        self.assertEqual([tuple(row["key"]) for row in self.plan["records"]], self.keys)
        self.assertEqual(sorted(self.index), self.keys)

    def test_frozen_real_mixed_selection_accepts_measured_cache_fields(self):
        shapes = Counter(tuple(sorted(pin)) for row in self.plan["records"] for pin in row["files"].values())
        self.assertEqual(shapes, {("path",): 135, ("path", "sha256", "size_bytes"): 15})
        self.assertTrue(all(set(pin) == {"path", "sha256", "size_bytes"}
                            for _, row in self.index.values() for pin in row["files"].values()))
        # The old whole-dict equality rejects 45 genuine frames before reading their bytes.
        self.assertEqual(sum(row["files"] != self.index[tuple(row["key"])][1]["files"]
                             for row in self.plan["records"]), 45)
        _check_clip_file_pins(self.plan, SimpleNamespace(index=self.index), self.keys)

    def test_every_selection_path_size_and_sha_remains_enforced(self):
        checked = Counter()
        for expected in self.plan["records"]:
            key = tuple(expected["key"])
            for role, pin in expected["files"].items():
                for field, value in pin.items():
                    with self.subTest(frame=key[1], role=role, field=field):
                        changed = copy.deepcopy(self.index)
                        changed[key][1]["files"][role][field] = value+1 if field == "size_bytes" else value+"changed"
                        with self.assertRaisesRegex(ValueError, "v14_clip_file_pins"):
                            _check_clip_file_pins(self.plan, SimpleNamespace(index=changed), self.keys)
                        checked[field] += 1
        self.assertEqual(checked, {"path":150, "sha256":15, "size_bytes":15})

    def test_split_and_exact_file_roles_remain_enforced(self):
        key = self.keys[1]
        for operation in ("cache_split", "selection_split", "missing_role", "extra_role"):
            plan, index = copy.deepcopy(self.plan), copy.deepcopy(self.index)
            if operation == "cache_split": index[key][1]["split"] = "validation"
            elif operation == "selection_split": plan["records"][1]["split"] = "validation"
            elif operation == "missing_role": del index[key][1]["files"]["pose"]
            else: index[key][1]["files"]["extra"] = {"path":"unexpected"}
            with self.subTest(operation=operation), self.assertRaisesRegex(ValueError, "v14_clip_file_pins"):
                _check_clip_file_pins(plan, SimpleNamespace(index=index), self.keys)
