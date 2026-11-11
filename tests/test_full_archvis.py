"""Bounded fixtures for the ArchVis v1.2 preparation gate."""

from argparse import Namespace
from contextlib import redirect_stdout
import io
import json
import unittest
from unittest.mock import patch
import zipfile

from rewind_archive.contracts import digest
from rewind_prepare import full_archvis as full
from rewind_prepare.converter import sha
from rewind_prepare.temporal_selection import seal
import test_extracted_stereo


class FullArchVisTests(unittest.TestCase):
    def setUp(self):
        self.fixture = test_extracted_stereo.ExtractedStereoTests()
        self.addCleanup(self.fixture.doCleanups)
        self.fixture.setUp()
        self.root = self.fixture.base
        self.train, self.val = self.fixture.train, self.fixture.val
        inputs = self.root / "inputs"
        inputs.mkdir()
        for source, target in ((self.fixture.fixture.motion, "motion_map.json"),
                               (self.fixture.fixture.specpath, "spec.json"),
                               (self.fixture.fixture.calibration, "calibration.json")):
            (inputs / target).write_bytes(source.read_bytes())
        split = inputs / "split"
        split.mkdir()
        for file in self.fixture.fixture.splitdir.iterdir():
            (split / file.name).write_bytes(file.read_bytes())
        self.keys = {"train": [(self.train, frame) for frame in range(3)], "validation": [(self.val, 0)]}
        inventory = {"schema": "archvis_drive_metadata_join_20261109_v1", "failed": [], "capped": [],
                     "sequences": [{"path": seq, "split": name, "source": "train_val",
                                    "frames": [frame for _, frame in values]}
                                   for name, values in self.keys.items() for seq in (values[0][0],)],
                     "duplicates": []}
        (inputs / "inventory.json").write_text(json.dumps(inventory), encoding="utf-8")
        self.snapshot = seal({"schema": full.SNAPSHOT, "keys": self.keys, "dataset_version": "direct_fixture", "excluded_duplicates": [],
                              "inventory_sha256": sha((inputs / "inventory.json").read_bytes())})
        prior = self.fixture.prepare("prior", train_limit=1)
        prior = {(row["sequence_id"], row["frame_index"]): row for row in prior["records"]}
        self.enterContext(patch.object(full, "frozen_snapshot", return_value=(self.snapshot, {}, prior)))
        self.enterContext(patch.object(full, "ENVIRONMENT", "Room"))
        self.enterContext(patch.object(full, "VALIDATION_KEYS", digest(self.keys["validation"])))
        self.work = self.root / "work"
        self.args = dict(work=self.work, inputs=inputs, rgb_root=self.fixture.rgb,
                         pose_root=self.fixture.fixture.raw, snapshot_sha256=self.snapshot["content_sha256"])

    def run_phase(self, action, **kwargs):
        with redirect_stdout(io.StringIO()):
            return full.run_phase(Namespace(action=action, **self.args, **kwargs))

    def report(self, name):
        return json.loads((self.work / name).read_bytes())

    def preflight(self):
        self.assertEqual(self.run_phase("preflight"), 0)
        return self.report("preflight.json")["source_plan_sha256"]

    def test_plan_pin_audit_and_recovery(self):
        pin = self.preflight()
        self.assertEqual(self.report("preflight.json")["file_stats"], 12)
        self.assertFalse((self.work / "verified").exists())
        self.assertEqual(self.run_phase("audit", source_plan_sha256=pin), 0)
        report = self.report("audit_status.json")
        self.assertEqual(report["pairs_verified"], 4)
        self.assertEqual(report["prior_pinned_pairs_verified"], 2)
        self.assertFalse(report["train_ready"])
        self.assertFalse(self.report("verified/manifest.json")["temporal_ready"])
        output = self.root / "recover.zip"
        with redirect_stdout(io.StringIO()):
            full.recover(self.work, output)
        with zipfile.ZipFile(output) as archive:
            receipt = json.loads(archive.read("recovery_receipt.json"))
            self.assertIn("inventory_check.json", receipt["files"])
            self.assertIn("audit_inventory_check.json", receipt["files"])
            for name, item in receipt["files"].items():
                raw = archive.read(name)
                self.assertEqual(item, {"bytes": len(raw), "sha256": sha(raw)})

    def test_outside_snapshot_pair_is_reported_before_any_decode(self):
        self.fixture.add_pair(self.train, 6, 84)
        with self.assertRaisesRegex(ValueError, "v12_live_inventory_mismatch"):
            self.run_phase("preflight")
        report = self.report("inventory_check.json")
        self.assertEqual(report["outside_snapshot_eligible_keys"], [[self.train, 6]])
        self.assertEqual(report["payload_validation"], "not_run")
        self.assertFalse((self.work / "source_plan.json").exists())

    def test_same_size_bad_png_reports_path_and_keeps_manifest_unpublished(self):
        pin = self.preflight()
        bad_file = self.fixture.rgb / self.train / "fisheye_final_image/camera_left/final_0.png"
        good = bad_file.read_bytes()
        bad_file.write_bytes(b"bad!" + good[4:])
        with self.assertRaisesRegex(ValueError, "rgb_png_header"):
            self.run_phase("audit", source_plan_sha256=pin)
        self.assertEqual(self.report("audit_status.json")["current_file"],
                         {"key": [self.train, 0], "role": "left",
                          "path": self.train + "/fisheye_final_image/camera_left/final_0.png"})
        self.assertFalse((self.work / "verified").exists())

    def test_unchanged_size_and_geometry_cannot_replace_pinned_validation(self):
        pin = self.preflight()
        pose = self.fixture.fixture.raw / self.val / "json/frame_0.json"
        old = pose.read_bytes()
        changed = old.replace(b'": ', b'":\t', 1)
        self.assertNotEqual(old, changed)
        self.assertEqual(json.loads(old), json.loads(changed))
        pose.write_bytes(changed)
        with self.assertRaisesRegex(ValueError, "v12_prior_payload_identity_changed:.*#0:pose"):
            self.run_phase("audit", source_plan_sha256=pin)
        self.assertFalse((self.work / "verified").exists())

    def test_coverage_gaps_do_not_cross_or_repeat_frames(self):
        keys = {"train": [("seq", i) for i in [*range(56), *range(60, 65)]]}
        counts, rows = full.clip_coverage(keys)
        self.assertEqual(counts["train"]["padding_slots"], 89)
        self.assertEqual(counts["train"]["strict_omitted_frames"], 11)
        self.assertEqual(counts["train"]["interior_missing_frame_numbers"], 4)
        covered = [(r["sequence"], i) for r in rows for i in range(r["start_frame"],
                                                                  r["start_frame"] + r["real_frames"])]
        self.assertEqual(covered, keys["train"])


if __name__ == "__main__":
    unittest.main()
