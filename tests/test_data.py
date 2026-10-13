"""합성 fixture만 사용하는 데이터 계약·오류 경로 검증."""

import contextlib
import hashlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from rewind_data.__main__ import main
from rewind_data.audit import REQUIRED_FIELDS, audit_dataset, load_provenance
from rewind_data.skeleton import load_skeleton
from rewind_data.viewer import viewer_payload, write_outputs


def joint(x=1):
    return {field: [x + i for i in range(dim)] for field, dim in REQUIRED_FIELDS.items()}


def document(frame=0):
    return {"frame": str(frame), "joints": {"root": joint(), "tip": joint(2)},
            "camera_left": {"open3d_rot": [1, 2, 3]}, "camera_right": {}, "device": {"open3d_rot": [0.01, 0.02, 0.03]}}


class DataContractTests(unittest.TestCase):
    def setUp(self):
        self.temp_root = Path(__file__).resolve().parents[1] / "outputs" / "test_tmp"
        self.temp_root.mkdir(parents=True, exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=self.temp_root)
        self.base = Path(self.temp.name)
        self.root = self.base / "raw"
        self.root.mkdir()
        self.skeleton_path = self.base / "skeleton.json"
        self.skeleton_path.write_text(json.dumps({"id": "fixture", "status": "synthetic_fixture",
            "evidence": "test only", "groups": {"body": ["tip", "root"], "left_hand": ["tip"]},
            "edges": [["root", "tip"]]}), encoding="utf-8")
        self.skeleton = load_skeleton(self.skeleton_path)

    def tearDown(self):
        self.assertTrue(self.base.resolve().is_relative_to(self.temp_root.resolve()))
        self.temp.cleanup()

    def write(self, seq="a", frame=0, doc=None, filename=None, text=None):
        path = self.root / seq / (filename or f"frame_{frame}.json")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text if text is not None else json.dumps(doc if doc is not None else document(frame)), encoding="utf-8")
        return path

    def audit(self, provenance=None):
        return audit_dataset(self.root, self.skeleton, "fixture", provenance or {"data_kind": "synthetic_fixture"})

    def codes(self, result):
        return {i["code"] for i in result.report["issues"]}

    def test_numeric_order_and_multiple_sequence_boundaries(self):
        for i in (10, 2, 1):
            self.write("a", i)
        self.write("b", 0)
        result = self.audit()
        self.assertEqual([f.frame_index for f in result.sequences[0].frames], [1, 2, 10])
        self.assertEqual([f.frame_index for f in result.sequences[1].frames], [0])
        self.assertEqual(result.sequences[0].missing_frame_ranges, [[3, 9]])
        self.assertEqual(result.sequences[1].missing_frame_ranges, [])
        self.assertEqual([s["id"] for s in viewer_payload(result)["sequences"]], ["a", "b"])

    def test_duplicate_numeric_frame_not_silently_dropped(self):
        self.write(frame=2)
        self.write(frame=2, filename="frame_02.json")
        result = self.audit()
        self.assertIn("duplicate_frame", self.codes(result))
        self.assertEqual(len(result.sequences[0].frames), 2)
        self.assertEqual(len(next(i for i in result.report["issues"] if i["code"] == "duplicate_frame")["files"]), 2)
        with self.assertRaises(ValueError):
            result.require_clean()

    def test_malformed_json_duplicate_keys_and_nonfinite_constants(self):
        for seq, text in (("bad", "{oops"), ("keys", '{"joints": {}, "joints": {}}'),
                          ("nan", '{"a": NaN}'), ("inf", '{"a": Infinity}')):
            self.write(seq, text=text)
        result = self.audit()
        self.assertEqual(result.report["summary"]["parsed_file_count"], 0)
        self.assertTrue(all(i["file"] and i["location"] for i in result.report["issues"]))
        write_outputs(result, self.base / "out")  # invalid JSON에도 보고서/viewer 생성.
        json.loads((self.base / "out" / "manifest.json").read_text(encoding="utf-8"))

    def test_shape_type_and_finite_value_failures_with_locations(self):
        for bad in (None, [1, 2], [[1], 2, 3], [True, 2, 3], [1, "2", 3], [1e999, 2, 3], [10**400, 2, 3]):
            with self.subTest(bad=str(bad)[:50]):
                doc = document()
                doc["joints"]["tip"]["device_pts3d"] = bad
                self.write(doc=doc)
                result = self.audit()
                sample = result.sequences[0].frames[0]
                self.assertFalse(sample.field_valid_mask.get("tip", {}).get("device_pts3d", False))
                self.assertIsNone(sample.values.get("tip", {}).get("device_pts3d"))
                self.assertEqual(sample.select(["tip"])["coordinates"], [None])
                self.assertTrue(result.report["summary"]["error_count"])
                self.assertTrue(any(i["location"].endswith('.device_pts3d') or i["code"] == "invalid_json_or_io" for i in result.report["issues"]))

    def test_standard_json_overflow_is_rejected_as_invalid_vector(self):
        text = json.dumps(document()).replace('[2, 3, 4]', '[1e999, 3, 4]')
        self.write(text=text)
        result = self.audit()
        self.assertIn("invalid_vector", self.codes(result))
        write_outputs(result, self.base / "out")

    def test_missing_joint_and_field_get_null_and_false_mask(self):
        self.write(frame=0)
        doc = document(1)
        del doc["joints"]["tip"]
        del doc["joints"]["root"]["camera_left_pts2d"]
        self.write(frame=1, doc=doc)
        result = self.audit()
        self.assertTrue({"missing_joint", "missing_field"}.issubset(self.codes(result)))
        sample = result.sequences[0].frames[1]
        self.assertEqual(sample.select(["tip"])["valid_mask"], [False])
        self.assertFalse(sample.annotation_valid_mask["root"])
        self.assertTrue(sample.select(["root"])["valid_mask"][0])  # 3D field별 mask는 독립.

    def test_union_detects_missing_joint_in_first_frame(self):
        self.write(frame=0)
        doc = document(1)
        doc["joints"]["extra"] = joint()
        self.write(frame=1, doc=doc)
        result = self.audit()
        self.assertTrue(any(i["code"] == "missing_joint" and i["file"].endswith("frame_0.json") for i in result.report["issues"]))

    def test_selection_order_preserves_raw_extras_rotations_and_masks(self):
        doc = document()
        doc["joints"]["extra"] = joint(9)
        path = self.write(doc=doc)
        before = path.read_bytes()
        result = self.audit().require_clean()
        sample = result.sequences[0].frames[0]
        selected = sample.select(self.skeleton.groups["body"])
        self.assertEqual(selected["joint_names"], ["tip", "root"])
        self.assertEqual(selected["coordinates"], [[2, 3, 4], [1, 2, 3]])
        self.assertEqual(sample.raw_document, doc)
        self.assertEqual(path.read_bytes(), before)
        self.assertEqual(result.manifest["files"][0]["sha256"], hashlib.sha256(before).hexdigest())
        self.assertEqual(result.manifest["verification_state"]["rotation_representation"], "unknown")

    def test_offscreen_2d_and_gt3d_remain_valid_visibility_unknown(self):
        doc = document()
        doc["joints"]["root"]["camera_left_pts2d"] = [-100, 5000]
        self.write(doc=doc)
        sample = self.audit().require_clean().sequences[0].frames[0]
        self.assertTrue(sample.annotation_valid_mask["root"])
        self.assertEqual(sample.values["root"]["camera_left_pts2d"], [-100, 5000])
        self.assertEqual(sample.select(["root"])["visibility"], {"left": [None], "right": [None]})
        self.assertEqual(sample.select(["root"])["confidence"], {"left": [None], "right": [None]})

    def test_filename_and_document_frame_mismatch(self):
        self.write(frame=0, doc=document(1))
        self.write(filename="frame_bad.json")
        result = self.audit()
        self.assertTrue({"frame_id_mismatch", "unexpected_filename"}.issubset(self.codes(result)))

    def test_inventory_detects_missing_start_and_end_without_assuming_completeness(self):
        self.write(frame=1)
        result = self.audit({"data_kind": "synthetic_fixture", "observed_folder_inventory": {"a": [0, 1, 2]}})
        issue = next(i for i in result.report["issues"] if i["code"] == "folder_inventory_mismatch")
        self.assertEqual(issue["missing"], [0, 2])
        self.assertEqual(result.manifest["verification_state"]["sequence_completeness"], "unknown")

    def test_large_gap_is_bounded_ranges(self):
        self.write(frame=0)
        self.write(frame=1000000000)
        self.assertEqual(self.audit().sequences[0].missing_frame_ranges, [[1, 999999999]])

    def test_empty_root_invalid_schema_and_unverified_metadata(self):
        self.assertIn("empty_dataset", self.codes(self.audit()))
        self.write(doc={"frame": "0", "joints": []})
        self.assertIn("invalid_schema", self.codes(self.audit()))

    def test_skeleton_rejects_duplicate_names_and_unknown_edges(self):
        for mutate in (lambda d: d["groups"].update(body=["root", "root"]),
                       lambda d: d.update(edges=[["root", "unknown"]])):
            doc = json.loads(self.skeleton_path.read_text(encoding="utf-8"))
            mutate(doc)
            self.skeleton_path.write_text(json.dumps(doc), encoding="utf-8")
            with self.assertRaises(ValueError):
                load_skeleton(self.skeleton_path)

    def test_provenance_strips_queries_and_rejects_signed_download_urls(self):
        path = self.base / "provenance.json"
        path.write_text(json.dumps({"data_kind": "dataset_sample", "files": {"a/frame_0.json":
            "https://drive.google.com/file/d/abc/view?usp=drivesdk"}}), encoding="utf-8")
        self.assertEqual(load_provenance(path)["files"]["a/frame_0.json"], "https://drive.google.com/file/d/abc/view")
        path.write_text(json.dumps({"data_kind": "dataset_sample", "files": {"a/frame_0.json":
            "https://example.com/raw?sig=secret"}}), encoding="utf-8")
        with self.assertRaises(ValueError):
            load_provenance(path)

    def test_html_payload_escape_and_invalid_points_do_not_make_zero_bones(self):
        doc = document()
        hostile = '</script><script>alert(1)</script>'
        doc["joints"][hostile] = joint()
        doc["joints"]["tip"]["device_pts3d"] = None
        self.write(doc=doc)
        result = self.audit()
        out = self.base / "out"
        write_outputs(result, out)
        html = (out / "viewer.html").read_text(encoding="utf-8")
        self.assertNotIn(hostile, html)
        start = '<script id="pose-data" type="application/json">'
        payload = json.loads(html.split(start)[1].split('</script>')[0])
        seq = payload["sequences"][0]
        self.assertIsNone(seq["frames"][0]["points"][seq["joint_names"].index("tip")])
        self.assertIn('if(!p||!q)continue', html)
        self.assertNotIn('cdn.', html)

    def test_cli_exit_status_reports_error_and_never_writes_into_raw(self):
        self.write()
        args = ["--input", str(self.root), "--skeleton", str(self.skeleton_path), "--output", str(self.base / "out")]
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(main(args), 0)
        self.write(frame=2)
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(main(args), 1)
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as exc:
            main(args[:-1] + [str(self.root / "out")])
        self.assertEqual(exc.exception.code, 2)
        self.assertFalse((self.root / "out").exists())


if __name__ == "__main__":
    unittest.main()
