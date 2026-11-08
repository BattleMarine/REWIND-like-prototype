"""Direct extracted files, without archive or synthetic training release markers."""

import json
import unittest

from PIL import Image

from rewind_prepare.converter import sha
from rewind_prepare.extracted import ExtractedLimits, ExtractedStereoT1, prepare_extracted_t1
import test_prepare


class ExtractedStereoTests(unittest.TestCase):
    def setUp(self):
        self.fixture = test_prepare.PreparationTests()
        self.addCleanup(self.fixture.doCleanups)
        self.fixture.setUp()
        self.base = self.fixture.base
        self.rgb = self.base / "rgb"
        self.train, self.val = self.fixture.seq, self.fixture.other
        for frame in (2, 0, 1):
            self.add_pair(self.train, frame, 72)
        self.add_pair(self.val, 0, 128)
        self.motions = self.fixture.motion_doc
        self.motions["sequences"][self.val] = {
            "motion_id": "different_motion", "identity_status": "declared_unverified",
            "evidence": "fixture declared validation motion",
        }
        self.write_motions()

    def add_pair(self, seq, frame, color):
        self.fixture.add_frame(seq, frame)
        for side in ("left", "right"):
            path = self.rgb / seq / "fisheye_final_image" / ("camera_" + side) / f"final_{frame}.png"
            path.parent.mkdir(parents=True, exist_ok=True)
            with Image.new("RGB", (1024, 1024), (color, 20, 30)) as image:
                image.save(path)

    def write_motions(self):
        self.fixture.motion.write_text(json.dumps(self.motions), encoding="utf-8")

    def prepare(self, name, **overrides):
        params = dict(environment="Room", dataset_version="direct_fixture",
                      split_dir=self.fixture.splitdir, motion_map=self.fixture.motion,
                      spec_path=self.fixture.specpath, calibration_path=self.fixture.calibration,
                      output_dir=self.base / name, limits=ExtractedLimits(128 * 1024**2, 90),
                      train_limit=None)
        params.update(overrides)
        return prepare_extracted_t1(self.rgb, self.fixture.raw, **params)

    def loader(self, name, manifest, split):
        return ExtractedStereoT1(self.base / name / "manifest.json", self.rgb,
                                 self.fixture.raw, self.fixture.specpath,
                                 expected_sha256=manifest["content_sha256"], split=split)

    def test_pilot_reproducible_full_keeps_same_validation_and_lazy_contract(self):
        pilot = self.prepare("pilot", train_limit=2, seed=9)
        repeated = self.prepare("repeat", train_limit=2, seed=9)
        full = self.prepare("full")
        self.assertEqual(pilot["content_sha256"], repeated["content_sha256"])
        self.assertEqual(pilot["audit"]["selected"], {"train": 2, "validation": 1})
        self.assertEqual(full["audit"]["selected"], {"train": 3, "validation": 1})
        self.assertEqual(pilot["records"][-1], full["records"][-1])
        self.assertEqual(pilot["selection"]["validation_keys_sha256"],
                         full["selection"]["validation_keys_sha256"])
        self.assertFalse(full["train_ready"])
        self.assertEqual(full["source_kind"], "direct_extracted_files")
        self.assertNotIn("tar_complete", full["records"][0]["files"]["left"])
        data = self.loader("full", full, "validation")
        self.assertEqual(len(data), 1)
        batch = data.load_batch([0])
        self.assertEqual(tuple(batch["images"].shape), (1, 2, 3, 1024, 1024))
        self.assertAlmostEqual(float(batch["images"][0, 0, 0, 0, 0]), 128 / 255)
        self.assertEqual(tuple(batch["gt_xy"].shape), (1, 2, 62, 2))
        self.assertEqual(batch["gt_xyz_mm"][0, 0].tolist(), [-30.0, 50.0, 700.0])
        self.assertTrue(batch["annotation_valid"].all())
        self.assertIsNone(batch["metadata"][0]["time_seconds"])

    def test_metadata_ignores_unselected_test_payload_and_quarantines_known_duplicate(self):
        test_seq = "Room/Day/person/glasses/003/Held_Out_Test"
        test_split = self.fixture.splitdir / "test.txt"
        test_split.write_bytes((test_seq + "\n").encode())
        self.fixture.spec["pinned_sources"]["split_sha256"]["test"] = sha(test_split.read_bytes())
        self.fixture.save_spec()
        self.add_pair(test_seq, 0, 155)
        (self.rgb / test_seq / "fisheye_final_image/camera_left/final_0.png").write_bytes(b"not a PNG")
        report = {"schema": "archvis_drive_metadata_join_20261109_v1", "failed": [], "capped": [],
                  "sequences": [{"path": self.train, "split": "train", "source": "train_val", "frames": [0, 1, 2, 7]}],
                  "duplicates": [{"split": "train_val", "key": self.train + "#0", "type": "left"}]}
        path = self.base / "inventory.json"
        path.write_text(json.dumps(report), encoding="utf-8")
        (self.rgb / self.train / "fisheye_final_image/camera_right/final_1.png").unlink()
        result = self.prepare("quarantine", inventory=path, inventory_sha256=sha(path.read_bytes()))
        self.assertEqual(result["audit"]["duplicates"], [[self.train, 0]])
        self.assertEqual(result["audit"]["incomplete"], [[self.train, 1, ["right"]]])
        self.assertEqual(result["audit"]["inventory_pairs_missing_locally"], [[self.train, 7]])
        self.assertEqual(result["audit"]["selected"], {"train": 1, "validation": 1})
        self.assertEqual(result["audit"]["ignored_entries"]["test"], 3)

    def test_full_selection_exceeds_old_small_release_record_limit(self):
        for frame in range(3, 35):
            self.add_pair(self.train, frame, 72)
        full = self.prepare("full_above_32")
        self.assertEqual(full["audit"]["selected"], {"train": 35, "validation": 1})
        self.assertEqual(len(self.loader("full_above_32", full, "train")), 35)

    def test_numeric_filename_alias_is_quarantined(self):
        source = self.rgb / self.train / "fisheye_final_image/camera_left/final_0.png"
        source.with_name("final_00.png").write_bytes(source.read_bytes())
        result = self.prepare("duplicate")
        self.assertEqual(result["audit"]["duplicates"], [[self.train, 0]])
        self.assertEqual(result["audit"]["selected"]["train"], 2)

    def test_motion_overlap_and_read_limit_reject_before_publication(self):
        self.motions["sequences"][self.val]["motion_id"] = self.motions["sequences"][self.train]["motion_id"]
        self.write_motions()
        with self.assertRaisesRegex(ValueError, "motion_split_overlap"):
            self.prepare("overlap")
        with self.assertRaisesRegex(ValueError, "extracted_read_byte_limit"):
            self.prepare("bounded", limits=ExtractedLimits(200, 10))
        self.assertFalse((self.base / "overlap").exists())
        self.assertFalse((self.base / "bounded").exists())

    def test_decode_and_geometry_fail_before_publication(self):
        path = self.rgb / self.train / "fisheye_final_image/camera_left/final_0.png"
        good = path.read_bytes()
        path.write_bytes(b"not a PNG")
        with self.assertRaisesRegex(ValueError, "rgb_png_header"):
            self.prepare("bad_png")
        path.write_bytes(good)
        pose = self.fixture.raw / self.train / "json/frame_0.json"
        doc = json.loads(pose.read_bytes())
        doc["joints"]["pelvis"]["camera_left_pts2d"][0] += 1
        pose.write_text(json.dumps(doc), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "annotation_projection_mismatch"):
            self.prepare("bad_geometry")
        self.assertFalse((self.base / "bad_png").exists())
        self.assertFalse((self.base / "bad_geometry").exists())

    def test_lazy_loader_detects_same_size_payload_change(self):
        result = self.prepare("pinned")
        path = self.fixture.raw / self.train / "json/frame_0.json"
        raw = path.read_bytes()
        path.write_bytes(raw.replace(b'"frame": "0"', b'"frame": "9"', 1))
        self.assertEqual(path.stat().st_size, len(raw))
        data = self.loader("pinned", result, "train")
        self.assertEqual(len(data), 3)
        with self.assertRaisesRegex(ValueError, "extracted_payload_hash_changed"):
            data.load_batch([0])


if __name__ == "__main__":
    unittest.main()
