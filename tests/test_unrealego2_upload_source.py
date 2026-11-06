"""네트워크·Drive 없이 분할 원본 검증과 선택 staging의 실패 경계를 검사한다."""
import gzip
import hashlib
import io
import json
from pathlib import Path
import shutil
import tarfile
import tempfile
import unittest
from unittest.mock import patch
import zipfile

from scripts import unrealego2_upload_source as source


class UploadSourceTests(unittest.TestCase):
    def setUp(self):
        tmp_parent = Path(__file__).resolve().parents[1] / "outputs/test_tmp"
        tmp_parent.mkdir(parents=True, exist_ok=True)
        self.tmp = tempfile.TemporaryDirectory(dir=tmp_parent)
        self.addCleanup(self.tmp.cleanup)
        self.base = Path(self.tmp.name)
        self.env = "Scene"
        self.a = "Scene/Day/person/glasses/001/Train_A"
        self.b = "Scene/Day/person/glasses/002/Train_B"
        self.v = "Scene/Day/person/glasses/003/Validation"
        self.t = "Scene/Night/person/glasses/004/Test"
        self.split = self.base / "split.zip"
        self.make_split()
        self.parts = self.base / "parts"
        self.parts.mkdir()
        self.registry = self.base / "registry.json"
        self.category = "UnrealEgoData2_train_val_pose"
        self.limits = dict(max_output_bytes=1024**2, max_decompressed_bytes=1024**2,
                           max_members=1000, max_file_bytes=65536,
                           max_manifest_bytes=1024**2, reserve_free_bytes=0,
                           chunk_bytes=512)

    def make_split(self, train=None, validation=None, test=None):
        rows = {"train": train if train is not None else [self.a, self.b],
                "validation": validation if validation is not None else [self.v],
                "test": test if test is not None else [self.t]}
        with zipfile.ZipFile(self.split, "w") as z:
            for split, lines in rows.items():
                z.writestr("UnrealEgoData2/" + split + ".txt", "\n".join(lines) + "\n")

    def entries(self):
        return [(seq + "/json/frame_0.json", b'{"frame":0}')
                for seq in (self.a, self.b, self.v)]

    def tar(self, entries=None, fmt=tarfile.PAX_FORMAT):
        output = io.BytesIO()
        with tarfile.open(fileobj=output, mode="w", format=fmt) as tf:
            for name, body in (self.entries() if entries is None else entries):
                info = tarfile.TarInfo(name)
                if body is None:
                    info.type = tarfile.SYMTYPE
                    info.linkname = "../../outside"
                    tf.addfile(info)
                else:
                    info.size = len(body)
                    tf.addfile(info, io.BytesIO(body))
        return output.getvalue()

    def prepare(self, raw=None, entries=None, fmt=tarfile.PAX_FORMAT, count=3):
        compressed = gzip.compress(self.tar(entries, fmt) if raw is None else raw, mtime=0)
        self.write_compressed(compressed, count)
        return compressed

    def write_compressed(self, compressed, count=3):
        cuts = [len(compressed) * i // count for i in range(count + 1)]
        parts = []
        for i in range(count):
            name = f"Scene.tar.gz{i:03d}"
            (self.parts / name).write_bytes(compressed[cuts[i]:cuts[i+1]])
            parts.append(dict(index=i, name=name, size=None))
        registry = {"archives": [dict(category=self.category, name=self.env,
                      official_part_count=count, official_count_source={"fixture": True},
                      parts=list(reversed(parts)))]}
        self.registry.write_text(json.dumps(registry), encoding="utf-8")

    def scan(self, stage="stage", **kwargs):
        limits = {**self.limits, **kwargs.pop("limit_changes", {})}
        return source.scan_source(self.parts, self.category, self.env, self.split,
                                  self.base / stage, limits=limits,
                                  registry_path=self.registry, **kwargs)

    def assert_failure(self, reason, **kwargs):
        with self.assertRaises(source.SourceScanError) as caught:
            self.scan(**kwargs)
        self.assertEqual(caught.exception.reason, reason)
        report = json.loads((self.base / "stage/source_report.json").read_text())
        self.assertFalse(report["upload_ready"])
        self.assertEqual(report["status"], "failed")
        self.assertFalse((self.base / "stage/manifest.json").exists())
        return report

    def test_full_split_stream_stages_and_hashes_every_byte(self):
        compressed = self.prepare()
        report = self.scan()
        self.assertTrue(report["upload_ready"])
        self.assertTrue(report["gzip_crc_valid"])
        self.assertTrue(report["tar_complete"])
        self.assertFalse(report["official_archive_authenticity_verified"])
        self.assertEqual(sum(p["bytes_read"] for p in report["parts"]), len(compressed))
        self.assertEqual(report["missing_official_sequences"], [])
        self.assertEqual(report["status"], "complete")
        for item in report["files"]:
            actual = (self.base / "stage" / item["relative_path"]).read_bytes()
            self.assertEqual(item["size"], len(actual))
            self.assertEqual(item["sha256"], hashlib.sha256(actual).hexdigest())
        self.assertEqual(report, json.loads((self.base / "stage/manifest.json").read_text()))

    def test_selection_does_not_skip_crc_or_sequence_coverage(self):
        compressed = self.prepare()
        report = self.scan(selected_sequences=[self.a])
        self.assertEqual(len(report["files"]), 1)
        self.assertEqual(report["observed_sequence_count"], 3)
        self.assertTrue(report["upload_ready"])
        self.assertFalse((self.base / "stage" / self.b).exists())
        self.assertEqual(sum(p["bytes_read"] for p in report["parts"]), len(compressed))
        self.assertLess(report["output_bytes"], report["decompressed_bytes"])

    def test_prefix_selection_matches_whole_path_components(self):
        self.prepare()
        report = self.scan(sequence_prefixes=["Scene/Day/person/glasses/001"])
        self.assertEqual(report["selected_sequences"], [self.a])
        self.assertEqual(len(report["files"]), 1)

    def test_missing_sequence_and_unknown_file_are_not_upload_ready(self):
        self.prepare(entries=[(self.a + "/json/frame_0.json", b"{}"),
                              (self.a + "/readme.txt", b"info")])
        report = self.scan()
        self.assertFalse(report["upload_ready"])
        self.assertEqual(report["missing_official_sequences"], [self.b, self.v])
        self.assertEqual(report["unmatched_files"], [self.a + "/readme.txt"])

    def test_rgb_views_and_test_scope(self):
        self.category = "UnrealEgoData2_test_rgb"
        self.prepare(entries=[(self.t + "/fisheye_final_image/camera_left/final_0.png", b"left"),
                              (self.t + "/fisheye_final_image/camera_right/final_0.png", b"right")])
        report = self.scan()
        self.assertTrue(report["upload_ready"])
        self.assertEqual({r["split"] for r in report["files"]}, {"test"})
        self.assertEqual({r["view"] for r in report["files"]}, {"left", "right"})

    def test_missing_right_view_blocks_upload(self):
        self.category = "UnrealEgoData2_test_rgb"
        self.prepare(entries=[(self.t + "/fisheye_final_image/camera_left/final_0.png", b"left")])
        report = self.scan()
        self.assertFalse(report["upload_ready"])
        self.assertEqual(report["missing_stereo_views"],
                         [dict(sequence=self.t, frame=0, missing=["right"])])

    def test_pax_and_gnu_longname_paths(self):
        for index, fmt in enumerate((tarfile.PAX_FORMAT, tarfile.GNU_FORMAT)):
            with self.subTest(format=fmt):
                old = self.a
                self.a = self.a[:self.a.rindex("/")+1] + "Motion_" + "X" * 150
                self.make_split()
                self.prepare(fmt=fmt)
                report = self.scan(stage=f"long{index}")
                self.assertTrue(report["upload_ready"])
                self.assertIn(self.a + "/json/frame_0.json", [f["relative_path"] for f in report["files"]])
                self.a = old

    def test_crc_error_in_unselected_tail_still_fails(self):
        compressed = bytearray(self.prepare())
        compressed[-8] ^= 1
        self.write_compressed(bytes(compressed))
        report = self.assert_failure("BadGzipFile", selected_sequences=[self.a])
        self.assertFalse(report["gzip_crc_valid"])

    def test_missing_part_and_unexpected_part_fail_before_read(self):
        self.prepare()
        (self.parts / "Scene.tar.gz001").unlink()
        self.assert_failure("part_set_mismatch")
        self.prepare()
        (self.parts / "Scene.tar.gz003").write_bytes(b"extra")
        with self.assertRaisesRegex(source.SourceScanError, "part_set_mismatch"):
            self.scan(stage="extra")

    def test_tar_requires_two_zero_end_blocks(self):
        # One 512-byte header + padded payload + only a single zero block.
        self.prepare(raw=self.tar([(self.a + "/json/frame_0.json", b"{}")] )[:1536])
        self.assert_failure("incomplete_tar")

    def test_trailing_nonzero_data_and_missing_payload_fail(self):
        self.prepare(raw=self.tar() + b"garbage")
        self.assert_failure("nonzero_data_after_tar_end")
        self.prepare(raw=self.tar()[:520])
        with self.assertRaisesRegex(source.SourceScanError, "incomplete_tar"):
            self.scan(stage="truncated")

    def test_unsafe_paths_are_rejected_even_when_filtered_out(self):
        self.prepare(entries=self.entries() + [("../escaped", b"bad")])
        self.assert_failure("unsafe_path", selected_sequences=[self.a])
        self.assertFalse((self.base / "escaped").exists())

    def test_duplicate_paths_and_symlink_rejected(self):
        self.prepare(entries=self.entries() + [self.entries()[0]])
        self.assert_failure("duplicate_tar_path")
        self.prepare(entries=[(self.a + "/json/link", None)])
        with self.assertRaisesRegex(source.SourceScanError, "unsupported_tar_member_type"):
            self.scan(stage="symlink")

    def test_file_directory_collisions_in_both_orders(self):
        for i, entries in enumerate(([self.entries()[0], (self.a, b"bad")],
                                     [(self.a, b"bad"), self.entries()[0]])):
            self.prepare(entries=entries)
            with self.assertRaisesRegex(source.SourceScanError, "tar_file_directory_collision"):
                self.scan(stage=f"collision{i}")

    def test_file_and_decoded_and_output_budgets(self):
        for i, (changes, reason) in enumerate((
                ({"max_output_bytes": 1}, "output_bytes_limit"),
                ({"max_decompressed_bytes": 515}, "decompressed_bytes_limit"),
                ({"max_file_bytes": 1}, "file_bytes_limit"),
                ({"max_members": 1}, "members_limit"))):
            self.prepare()
            with self.assertRaisesRegex(source.SourceScanError, reason):
                self.scan(stage=f"budget{i}", limit_changes=changes)

    def test_free_disk_reserve_is_checked_before_staging(self):
        self.prepare()
        with patch.object(source.shutil, "disk_usage", return_value=shutil._ntuple_diskusage(1, 0, 1)):
            self.assert_failure("staging_free_space_limit")

    def test_stage_is_not_reused_or_overwritten(self):
        self.prepare()
        self.scan()
        before = (self.base / "stage/manifest.json").read_bytes()
        with self.assertRaisesRegex(source.SourceScanError, "stage_already_exists"):
            self.scan()
        self.assertEqual(before, (self.base / "stage/manifest.json").read_bytes())

    def test_split_overlap_and_invalid_selection_rejected(self):
        self.prepare()
        self.make_split(validation=[self.a])
        self.assert_failure("split_duplicate_or_overlap")
        self.make_split()
        with self.assertRaisesRegex(source.SourceScanError, "selection_not_official_or_duplicate"):
            self.scan(stage="badselection", selected_sequences=[self.t])

    def test_source_change_detected_after_complete_read(self):
        self.prepare()
        original = source._SplitReader.verify
        def change_then_verify(reader):
            reader.paths[-1].write_bytes(b"changed")
            original(reader)
        with patch.object(source._SplitReader, "verify", change_then_verify):
            self.assert_failure("source_changed_or_not_fully_read")

    def test_same_numeric_frame_with_distinct_name_rejected(self):
        self.prepare(entries=self.entries() + [(self.a + "/json/frame_00.json", b"{}")])
        self.assert_failure("duplicate_sequence_view_frame")

    def test_case_alias_in_implicit_parent_rejected(self):
        self.prepare(entries=self.entries() + [(self.a.replace("/person/", "/PERSON/") +
                                               "/json/frame_1.json", b"{}")])
        self.assert_failure("tar_case_alias")

    def test_unknown_sequence_and_cross_split_are_reported(self):
        wrong = "Scene/Day/person/glasses/999/Unknown"
        self.prepare(entries=self.entries() + [(wrong + "/json/frame_0.json", b"{}"),
                                               (self.t + "/json/frame_0.json", b"{}")])
        report = self.scan()
        self.assertFalse(report["upload_ready"])
        self.assertEqual(report["unexpected_sequences"], sorted([wrong, self.t]))
        self.assertEqual(len(report["unmatched_files"]), 2)
        self.assertFalse((self.base / "stage" / wrong).exists())

    def test_url_support_is_bounded_and_does_not_fallback(self):
        self.prepare()
        class Response(io.BytesIO):
            def geturl(self):
                return source.OFFICIAL_SPLIT_URL
        with patch.object(source.urllib.request, "urlopen",
                          return_value=Response(self.split.read_bytes())) as request:
            report = source.scan_source(self.parts, self.category, self.env,
                source.OFFICIAL_SPLIT_URL, self.base / "urlstage", limits=self.limits,
                registry_path=self.registry)
        request.assert_called_once()
        self.assertTrue(report["upload_ready"])
        with self.assertRaisesRegex(source.SourceScanError, "split_url_not_official"):
            source.scan_source(self.parts, self.category, self.env,
                "https://example.invalid/split.zip", self.base / "wrongurl",
                limits=self.limits, registry_path=self.registry)

    def test_manifest_is_not_published_after_write_failure(self):
        self.prepare()
        with patch.object(source.os, "fsync", side_effect=OSError("fixture disk failure")):
            with self.assertRaises(source.SourceScanError) as caught:
                self.scan()
        self.assertFalse(caught.exception.report["upload_ready"])
        self.assertFalse((self.base / "stage/manifest.json").exists())

    def test_partial_archive_does_not_claim_full_crc(self):
        compressed = self.prepare()
        self.write_compressed(compressed[:-5])
        report = self.assert_failure("EOFError")
        self.assertFalse(report["gzip_crc_valid"])

    def test_registry_order_is_validated(self):
        self.prepare()
        registry = json.loads(self.registry.read_text())
        registry["archives"][0]["parts"][0]["index"] = 100
        self.registry.write_text(json.dumps(registry))
        self.assert_failure("registry_part_order_invalid")

    def test_compressed_stat_sum_rejected_before_any_archive_read(self):
        compressed = self.prepare()
        with patch.object(source._SplitReader, "read", side_effect=AssertionError("must not read")):
            report = self.assert_failure("compressed_bytes_limit",
                limit_changes={"max_compressed_bytes": len(compressed) - 1})
        self.assertEqual(report["total_compressed_bytes"], len(compressed))
        self.assertEqual(report["files"], [])

    def test_exact_compressed_limit_and_progress_byte_cadence(self):
        compressed = self.prepare()
        events = []
        report = self.scan(limit_changes={"max_compressed_bytes": len(compressed)},
                           progress_callback=events.append, progress_bytes=1)
        self.assertTrue(report["upload_ready"])
        self.assertEqual(report["compressed_bytes"], len(compressed))
        self.assertEqual(events[0]["compressed_bytes"], 0)
        self.assertEqual(events[-1]["phase"], "read_complete")
        self.assertEqual(events[-1]["compressed_bytes"], len(compressed))
        self.assertGreater(len(events), 2)
        self.assertEqual([e["compressed_bytes"] for e in events],
                         sorted(e["compressed_bytes"] for e in events))
        self.assertTrue(all(e["total_compressed_bytes"] == len(compressed) for e in events))

    def test_actual_reader_never_exceeds_remaining_compressed_budget(self):
        self.prepare()
        reader = source._SplitReader(sorted(self.parts.iterdir()), 1024**2)
        self.addCleanup(reader.close)
        reader.max_compressed_bytes = 10
        with self.assertRaisesRegex(source.SourceScanError, "compressed_bytes_limit"):
            reader.read(100)
        self.assertEqual(reader.compressed_bytes, 10)
        self.assertEqual(sum(reader.sizes), 10)

    def test_growth_after_stat_is_not_read(self):
        self.prepare()
        path = self.parts / "Scene.tar.gz000"
        def grow(event):
            if event["compressed_bytes"] == 0:
                with path.open("ab") as f:
                    f.write(b"unexpected bytes")
        report = self.assert_failure("source_changed", progress_callback=grow)
        self.assertEqual(report["compressed_bytes"], 0)

    def test_progress_time_cadence_at_read_boundaries(self):
        self.prepare()
        events = []
        now = [0.0]
        with patch.object(source.time, "monotonic", side_effect=lambda: now[0]):
            reader = source._SplitReader(sorted(self.parts.iterdir()), 1024**2,
                                         events.append, progress_seconds=5)
            self.addCleanup(reader.close)
            reader.read(1)
            self.assertEqual(len(events), 1)
            now[0] = 5.1
            reader.read(1)
            self.assertEqual(len(events), 2)
            self.assertEqual(events[-1]["compressed_bytes"], 2)
            self.assertEqual(events[-1]["elapsed_seconds"], 5.1)

    def test_no_callback_means_no_default_output(self):
        from contextlib import redirect_stdout, redirect_stderr
        self.prepare()
        stdout, stderr = io.StringIO(), io.StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            self.scan()
        self.assertEqual(stdout.getvalue(), "")
        self.assertEqual(stderr.getvalue(), "")

    def test_callback_exception_preserves_failed_report(self):
        self.prepare()
        def stop(event):
            if event["compressed_bytes"]:
                raise KeyboardInterrupt()
        report = self.assert_failure("KeyboardInterrupt", progress_callback=stop, progress_bytes=1)
        self.assertGreater(report["compressed_bytes"], 0)

    def test_standalone_module_has_no_repository_imports(self):
        import ast
        tree = ast.parse(Path(source.__file__).read_text(encoding="utf-8"))
        imports = [n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)]
        self.assertFalse(any(n and n.startswith("rewind") for n in imports))


if __name__ == "__main__":
    unittest.main()
