import io
import json
import unittest
from unittest.mock import patch
import zipfile

from rewind_archive.contracts import canonical, digest
from rewind_prepare import temporal_cache as packs
from rewind_prepare.converter import sha
from rewind_prepare.temporal_selection import SCHEMA, seal
from rewind_v1 import audit
from rewind_v1.data import EVIDENCE_SCHEMA, REQUIRED_EVIDENCE
from scripts.build_archvis_v1_handoff import notebook_document
import test_extracted_stereo


class AuditTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixture = test_extracted_stereo.ExtractedStereoTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        fixture.add_pair(fixture.val, 1, 130)
        self.base = fixture.base
        self.spec_path = fixture.fixture.specpath
        self.calibration = fixture.fixture.calibration
        self.spec = json.loads(self.spec_path.read_bytes())
        prior = fixture.prepare("prior")
        records = [{"key": [row["sequence_id"], row["frame_index"]], "split": row["split"], "files": row["files"]}
                   for row in prior["records"] if row["frame_index"] < 2]
        self.motion_raw = fixture.fixture.motion.read_bytes()
        self.plan = seal({"schema": SCHEMA, "status": "metadata_only", "clip_frames": 2, "seed": 71,
                          "records": records, "train_ready": False, "temporal_ready": False, "blockers": ["fixture"],
                          "train_clip_starts": [[fixture.train, 0]], "dataset_version": "fixture",
                          "validation_keys_sha256": digest([[fixture.val, 0], [fixture.val, 1]]),
                          "inventory_sha256": "a"*64, "motion_map_sha256": sha(self.motion_raw),
                          "spec_sha256": sha(self.spec_path.read_bytes()), "calibration_sha256": sha(self.calibration.read_bytes())})
        self.work = self.base / "run"
        self.work.mkdir()
        self.out = self.work / "audit"
        self.out.mkdir()
        self.limits = packs.Limits(64*audit.MIB, 96*audit.MIB, 128*audit.MIB, 30, 2)

    def make_cache(self):
        with patch.multiple(packs, CLIP=2, TRAIN_COUNT=2, VALIDATION_COUNT=2, PART_FRAMES=2):
            state = packs.prepare(self.plan, expected_sha256=self.plan["content_sha256"],
                                  rgb_root=self.fixture.rgb, pose_root=self.fixture.fixture.raw,
                                  spec_path=self.spec_path, calibration_path=self.calibration,
                                  output=self.work/"cache", limits=self.limits)
        return packs.RawCache(self.work/"cache", state["cache_sha256"])

    def audit(self, cache, **kwargs):
        limits = dict(max_read_bytes=32*audit.MIB, max_report_bytes=16*audit.MIB, max_seconds=30)
        return audit.audit_cache(self.plan, cache, self.spec, json.loads(self.calibration.read_bytes()),
                                 self.motion_raw, self.out, **{**limits, **kwargs})

    def test_full_audit_blocks_without_evidence_and_recovers(self):
        cache = self.make_cache()
        result = self.audit(cache)
        self.assertEqual(result["status"], "blocked")
        self.assertEqual(result["pairs_audited"], {"train": 2, "validation": 2})
        self.assertTrue(result["numeric_geometry_passed"])
        self.assertFalse(result["train_ready"])
        self.assertEqual(result["time"]["pairs_with_time_like_fields"], 0)
        self.assertEqual(len(result["overlays"]), 3)
        self.assertFalse(result["motion"]["all_original_ids_verified_in_pinned_map"])
        (self.work / "notebook_prepare_result.json").write_bytes(canonical({"status": "stopped"}))
        (self.work / "notebook_training_result.json").write_bytes(canonical({"status": "not_started"}))
        target = self.base/"audit.zip"
        audit.recover_audit(self.work, target, 24*audit.MIB)
        with zipfile.ZipFile(target) as archive:
            self.assertIsNone(archive.testzip())
            receipt = json.loads(archive.read("recovery_receipt.json"))
            for name, pin in receipt["files"].items():
                raw = archive.read(name)
                self.assertEqual(pin, {"bytes": len(raw), "sha256": sha(raw)})
            self.assertFalse(any(name.startswith("cache/part-") for name in archive.namelist()))
            self.assertIn("notebook_prepare_result.json", receipt["files"])
            self.assertIn("notebook_training_result.json", receipt["files"])

        candidates = json.loads((self.out/audit.CAMERAS).read_bytes())
        self.assertTrue(all(row["time_seconds"] is None for row in candidates["rows"]))
        evidence_path = self.base / "evidence.json"
        reviews = self.base / "review"
        reviews.mkdir()
        review_pins, frames = {}, {}
        for name in REQUIRED_EVIDENCE:
            raw = canonical({"scope": "synthetic test", "name": name})
            (reviews / (name+".json")).write_bytes(raw)
            review_pins[name] = sha(raw)
        for row in candidates["rows"]:
            seq, frame = row["key"]
            frames.setdefault(seq, {})[str(frame)] = {"time_seconds": frame/30, "camera_pose": row["camera_pose"]}
        body = {"schema": EVIDENCE_SCHEMA, "selection_sha256": self.plan["content_sha256"],
                "review_artifact_sha256": review_pins, "frames": frames, "max_frame_delta_seconds": .05}
        arguments = self.work, result["content_sha256"], self.plan["content_sha256"], evidence_path
        nominal = seal({**body, "time_basis": "nominal_fps"})
        evidence_path.write_bytes(canonical(nominal))
        with self.assertRaisesRegex(ValueError, "recorded_timestamp"):
            audit.approve_audit(*arguments, nominal["content_sha256"])
        approved = seal({**body, "time_basis": "recorded_timestamps"})
        evidence_path.write_bytes(canonical(approved))
        self.assertEqual(audit.approve_audit(*arguments, approved["content_sha256"])["cache_sha256"],
                         cache.manifest["content_sha256"])
        frames[self.fixture.train]["1"]["camera_pose"]["left"]["translation"][0] += 1
        changed = seal({**body, "time_basis": "recorded_timestamps"})
        evidence_path.write_bytes(canonical(changed))
        with self.assertRaisesRegex(ValueError, "reviewed_camera_mismatch"):
            audit.approve_audit(*arguments, changed["content_sha256"])

    def test_preflight_and_read_cap(self):
        self.assertEqual(audit.preflight(self.plan, self.fixture.rgb, self.fixture.fixture.raw, self.spec,
                                         self.out, self.limits)["missing_pairs"], 0)
        cache = self.make_cache()
        with self.assertRaisesRegex(TimeoutError, "audit_read_limit"):
            self.audit(cache, max_read_bytes=1)
        report = json.loads((self.out/audit.REPORT).read_bytes())
        self.assertEqual(report["status"], "stopped")
        self.assertEqual(report["bytes_read_reserved"], 0)
        path = self.fixture.rgb / self.plan["records"][0]["files"]["left"]["path"]
        path.unlink()
        with self.assertRaisesRegex(ValueError, "source_preflight_blocked"):
            audit.preflight(self.plan, self.fixture.rgb, self.fixture.fixture.raw, self.spec, self.out, self.limits)
        self.assertEqual(json.loads((self.out/audit.PREFLIGHT).read_bytes())["missing_pairs"], 1)

    def test_time_candidate_samples_bounded(self):
        from collections import Counter
        fields, paths, samples = Counter(), set(), {}
        for frame in range(12):
            raw = {"frame": frame, "timestamp": frame*10, "camera_left": {"capture_time": "t"*200},
                   "camera_right": {"capture_time": [1, 2]}, "device": {}}
            self.assertTrue(audit._record_time(raw, ("seq", frame), fields, paths, samples))
        first, last = samples["pose", "timestamp"]
        self.assertEqual([row["value"] for row in first], [0, 10, 20])
        self.assertEqual([row["value"] for row in last], [90, 100, 110])
        self.assertTrue(samples["camera_left", "capture_time"][0][0]["truncated"])
        self.assertTrue(samples["camera_right", "capture_time"][0][0]["value_omitted"])

    def test_generated_notebook_without_external_approval_stays_blocked(self):
        cells = {cell["id"]: "".join(cell["source"]) for cell in notebook_document({})["cells"]}
        with patch("subprocess.run") as child, patch("sys.stdout", new_callable=io.StringIO) as output:
            exec(cells["conditional-training"].replace("RUN_TRAINING = False", "RUN_TRAINING = True"), {"PINS": {}})
            child.assert_not_called()
            self.assertIn("차단", output.getvalue())
