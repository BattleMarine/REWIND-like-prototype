import json
import unittest
from unittest.mock import patch

from rewind_archive.contracts import canonical
from rewind_prepare import temporal_cache as cache, temporal_selection as select
from rewind_prepare.converter import sha
import test_extracted_stereo


class TemporalPrepareTests(unittest.TestCase):
    def setUp(self):
        self.fixture = test_extracted_stereo.ExtractedStereoTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.base = self.fixture.base
        self.raw = self.fixture.fixture
        self.train, self.val = self.fixture.train, self.fixture.val
        self.inventory = self.base / "inventory.json"
        self.inventory.write_bytes(canonical({
            "schema": "archvis_drive_metadata_join_20261109_v1", "failed": [], "capped": [],
            "duplicates": [{"split": "train_val", "key": self.train + "#2"}],
            "sequences": [{"path": seq, "split": split, "source": "train_val", "frames": frames}
                          for seq, split, frames in ((self.train, "train", [0, 1, 2]), (self.val, "validation", [0]))],
        }))
        self.previous = self.fixture.prepare("prior", inventory=self.inventory,
                                            inventory_sha256=sha(self.inventory.read_bytes()))
        self.fixed = dict(CLIP=2, TRAIN_COUNT=2, VALIDATION_COUNT=1)
        for module in (cache, select):
            pins = patch.multiple(module, **self.fixed)
            pins.start()
            self.addCleanup(pins.stop)
        parts = patch.object(cache, "PART_FRAMES", 2)
        parts.start()
        self.addCleanup(parts.stop)
        self.plan = self.select()

    def select(self):
        return select.plan_selection(inventory=self.inventory, spec_path=self.raw.specpath, split_dir=self.raw.splitdir,
                                     previous_manifest=self.base / "prior/manifest.json",
                                     previous_sha256=self.previous["content_sha256"])

    def prepare(self, output, workers=1, max_parts=None, **overrides):
        caps = dict(max_read_bytes=4 * cache.MIB, max_disk_bytes=16 * cache.MIB,
                    max_buffer_bytes=96 * cache.MIB, max_seconds=30, workers=workers)
        return cache.prepare(self.plan, expected_sha256=self.plan["content_sha256"], rgb_root=self.fixture.rgb,
                             pose_root=self.raw.raw, spec_path=self.raw.specpath,
                             calibration_path=self.raw.calibration, output=self.base / output,
                             limits=cache.Limits(**{**caps, **overrides}), max_parts=max_parts)

    def test_clip_contract_and_shortfall(self):
        self.assertEqual(self.plan, self.select())
        self.assertEqual(self.plan["train_clip_starts"], [[self.train, 0]])
        self.assertEqual(self.plan["validation_disjoint_frame_number_clips"], 0)
        self.assertEqual(self.plan["excluded_inventory_duplicates"], 1)
        self.assertFalse(self.plan["train_ready"])
        present = {(self.train, 0), (self.train, 1), (self.train, 3), (self.val, 2)}
        self.assertEqual(select.window_keys(self.train, 3, present, width=3), [None, None, (self.train, 3)])
        self.assertEqual(select.window_keys(self.train, 1, present, clip_start=1, width=2),
                         [None, (self.train, 1)])
        with patch.object(select, "TRAIN_COUNT", 4):
            partial = self.select()
        self.assertEqual(partial["status"], "insufficient_train_clips")
        self.assertEqual(partial["train_unique_frames"], 2)

    def test_parallel_bytes_resume_restore_and_mutation(self):
        sequential = self.prepare("sequential")
        partial = self.prepare("parallel", workers=2, max_parts=1)
        self.assertEqual(partial["status"], "partial")
        self.assertFalse((self.base / "parallel" / cache.MANIFEST).exists())
        changed_dependency = {**cache.producer_identity(), "rewind_prepare/converter.py": "a" * 64}
        with patch.object(cache, "producer_identity", return_value=changed_dependency):
            with self.assertRaisesRegex(ValueError, "temporal_cache_resume_binding"):
                self.prepare("parallel", workers=2)
        resumed = self.prepare("parallel", workers=2)
        self.assertEqual(resumed["status"], "completed")
        self.assertEqual(resumed["parts"], sequential["parts"])
        self.assertEqual(resumed["cache_sha256"], sequential["cache_sha256"])
        self.assertEqual(resumed["read_bytes_reserved"], sequential["read_bytes_reserved"])
        target = self.base / "restored"
        args = dict(expected_sha256=resumed["cache_sha256"], output=target,
                    max_disk_bytes=16 * cache.MIB, max_seconds=10)
        self.assertGreater(cache.restore(self.base / "parallel", **args)["copied_bytes"], 0)
        self.assertEqual(cache.restore(self.base / "parallel", **args)["reused_parts"], 2)
        reader = cache.RawCache(target, resumed["cache_sha256"])
        raw = reader.read_pair(self.train, 1, split="train")
        self.assertEqual(raw["pose"], (self.raw.raw / self.train / "json/frame_1.json").read_bytes())
        with self.assertRaisesRegex(ValueError, "temporal_cache_pair_split"):
            reader.read_pair(self.train, 1, split="validation")
        pack = target / "part-0000.zip"
        data = pack.read_bytes()
        pack.write_bytes(bytes([data[0] ^ 1]) + data[1:])
        with self.assertRaisesRegex(ValueError, "temporal_cache_part_pin"):
            cache.restore(self.base / "parallel", **args)

    def test_failed_part_is_not_published_and_read_reservation_survives(self):
        partial = self.prepare("interrupted", workers=2, max_parts=1)
        with patch.object(cache, "_read_pair", side_effect=KeyboardInterrupt("stop")):
            with self.assertRaises(KeyboardInterrupt):
                self.prepare("interrupted", workers=2)
        root = self.base / "interrupted"
        progress = json.loads((root / cache.JOURNAL).read_bytes())
        self.assertEqual(progress["status"], "interrupted")
        self.assertGreater(progress["read_bytes_reserved"], partial["read_bytes_reserved"])
        self.assertFalse((root / cache.MANIFEST).exists())
        self.assertFalse((root / "part-0001.zip").exists())
        resumed = self.prepare("interrupted", workers=2)
        self.assertGreater(resumed["read_bytes_reserved"], progress["read_bytes_reserved"])
        self.assertEqual(resumed["status"], "completed")
        with self.assertRaisesRegex(ValueError, "temporal_cache_buffer_limit"):
            self.prepare("low_buffer", workers=2, max_buffer_bytes=1)
        with self.assertRaisesRegex(TimeoutError, "temporal_cache_read_limit"):
            self.prepare("low_read", max_read_bytes=1)
        with self.assertRaisesRegex(TimeoutError, "temporal_cache_disk_limit"):
            self.prepare("low_disk", max_disk_bytes=1)
