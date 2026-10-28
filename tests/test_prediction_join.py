"""합성 RGB cache와 GT 특징 문서 결합 계약 검사."""
import copy
import json
import unittest
from pathlib import Path
from unittest.mock import patch

try:
    import torch
    from PIL import Image
except ImportError:
    torch = None

if torch is not None:
    import test_local_rgb_predictions as rgb_fixtures
    from rewind_m1.data import DataContractError, content_hash, load_document, synthetic_document
    from rewind_m1.local_rgb_predictions import build_cache, load_weights
    from rewind_m1.prediction_join import MAX_BYTES, join_predictions, main


@unittest.skipUnless(torch is not None, "torch와 Pillow가 필요합니다")
class PredictionJoinTests(unittest.TestCase):
    def setUp(self):
        fixture = rgb_fixtures.LocalPredictionTests("test_prediction_reader_order_and_batch")
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        self.fixture = fixture
        model, checkpoint = load_weights(fixture.weights, fixture.loader, ["root", "tip"])
        self.cache = build_cache(fixture.loader, model, checkpoint, ["root", "tip"])
        self.hash = fixture.loader.manifest_sha256
        source = synthetic_document()
        self.doc = copy.deepcopy(source)
        self.doc["contract"]["image_sizes"] = [[8, 6], [8, 6]]
        self.doc["records"] = []
        for i, cached in enumerate(self.cache["records"]):
            row = copy.deepcopy(source["records"][i])
            row.update(sequence_id=cached["sequence_id"], frame_index=cached["frame_index"],
                       split=self.cache["record_splits"][i], motion_id=f"motion_{i}", timestamp_raw=f"time_{i}")
            self.doc["records"].append(row)

    def _rehash(self, cache):
        cache["content_sha256"] = content_hash({k: v for k, v in cache.items() if k != "content_sha256"})

    def test_join_preserves_pose_and_loads_predictions(self):
        before, cache_before = copy.deepcopy(self.doc), copy.deepcopy(self.cache)
        self.cache["records"].reverse()
        self.cache["record_splits"].reverse()
        self._rehash(self.cache)
        cache_before = copy.deepcopy(self.cache)
        result = join_predictions(self.doc, self.cache, self.hash)
        self.assertEqual(self.doc, before)
        self.assertEqual(self.cache, cache_before)
        expected = {(row["sequence_id"], row["frame_index"]): row["prediction"] for row in self.cache["records"]}
        for original, actual in zip(before["records"], result["records"]):
            prediction = actual.pop("prediction")
            self.assertEqual(original, actual)
            self.assertEqual(prediction, expected[(original["sequence_id"], original["frame_index"])])
        # 검사에서 수정한 result 대신 새 결과로 load_document를 실행한다.
        result = join_predictions(self.doc, self.cache, self.hash)
        contract, splits, _ = load_document(result, 1)
        _, original_splits, _ = load_document(self.doc, 1)
        self.assertEqual(contract.input_source, "predicted_2d")
        self.assertEqual(sum(len(split.anchors) for split in splits.values()), 3)
        for name in splits:
            self.assertEqual(splits[name].anchors, original_splits[name].anchors)
            self.assertTrue(torch.equal(splits[name].target_mm, original_splits[name].target_mm))
            self.assertTrue(torch.equal(splits[name].target_valid, original_splits[name].target_valid))
            self.assertTrue(torch.equal(splits[name].time_valid, original_splits[name].time_valid))
        provenance = result["manifest"]["prediction_join_provenance"]
        self.assertEqual(provenance, {"rgb_manifest_sha256": self.hash, "prediction_cache_sha256": self.cache["content_sha256"], "input_document_sha256": content_hash(self.doc)})
        with self.assertRaises(DataContractError): join_predictions(result, self.cache, self.hash)
        no_splits = copy.deepcopy(self.cache)
        no_splits.pop("record_splits")
        self._rehash(no_splits)
        self.assertEqual(len(join_predictions(self.doc, no_splits, self.hash)["records"]), 3)

    def test_reject_keys_provenance_and_real_guard(self):
        cases = []
        cache = copy.deepcopy(self.cache); cache["records"].pop(); cache["record_splits"].pop(); cases.append((self.doc, cache, self.hash))
        cache = copy.deepcopy(self.cache); cache["records"][1] = copy.deepcopy(cache["records"][0]); cases.append((self.doc, cache, self.hash))
        cache = copy.deepcopy(self.cache); cache["records"][0]["frame_index"] = 50; cases.append((self.doc, cache, self.hash))
        cache = copy.deepcopy(self.cache); cache["record_splits"][0] = "test"; cases.append((self.doc, cache, self.hash))
        for field, value in (("joint_names", ["tip", "root"]), ("image_sizes", [[9, 6], [9, 6]]), ("data_kind", "dataset_sample")):
            cache = copy.deepcopy(self.cache); cache[field] = value; cases.append((self.doc, cache, self.hash))
        cache = copy.deepcopy(self.cache); cache["records"][0]["prediction"]["frontend_version"] = "bad"; cases.append((self.doc, cache, self.hash))
        cases.append((self.doc, self.cache, "0" * 64))
        doc = copy.deepcopy(self.doc); doc["schema_version"] = "real_pose_candidate_v1"; cases.append((doc, self.cache, self.hash))
        doc = copy.deepcopy(self.doc); doc["contract"]["data_kind"] = "dataset_sample"; cases.append((doc, self.cache, self.hash))
        for missing in ("joints", "units", "coordinates", "calibration", "split", "rgb_pose_alignment"):
            doc = copy.deepcopy(self.doc)
            doc["contract"].update(data_kind="dataset_sample", verified={k: k != missing for k in ("joints", "units", "coordinates", "calibration", "split", "rgb_pose_alignment")},
                verification_evidence={k: "fixture only" for k in ("joints", "units", "coordinates", "calibration", "split", "rgb_pose_alignment")})
            cache = copy.deepcopy(self.cache); cache["data_kind"] = "dataset_sample"; cases.append((doc, cache, self.hash))
        doc = copy.deepcopy(self.doc); doc["manifest"]["prediction_cache_content_sha256"] = "a" * 64; cases.append((doc, self.cache, self.hash))
        doc = copy.deepcopy(self.doc); doc["records"][0]["prediction"] = None; cases.append((doc, self.cache, self.hash))
        doc = copy.deepcopy(self.doc); doc["records"][1] = copy.deepcopy(doc["records"][0]); cases.append((doc, self.cache, self.hash))
        for doc, cache, manifest_hash in cases:
            with self.subTest(doc=doc.get("schema_version"), cache=cache.get("data_kind")):
                if cache is not self.cache:
                    self._rehash(cache)
                with self.assertRaises(DataContractError): join_predictions(doc, cache, manifest_hash)

    def test_cli_bounds_strict_json_and_exclusive_output(self):
        root = self.fixture.root
        source, cache, output = root / "features.json", root / "cache.json", root / "result.json"
        source.write_text(json.dumps(self.doc), encoding="utf-8")
        cache.write_text(json.dumps(self.cache), encoding="utf-8")
        args = ["--features", str(source), "--cache", str(cache), "--manifest-sha256", self.hash, "--output", str(output)]
        main(args)
        data = output.read_bytes()
        with self.assertRaises(FileExistsError): main(args)
        self.assertEqual(output.read_bytes(), data)
        source.write_text('{"a":1,"a":2}', encoding="utf-8")
        with self.assertRaises(ValueError): main(args)
        with self.assertRaises(DataContractError): join_predictions({"records": [], "huge": "x" * MAX_BYTES}, self.cache, self.hash)
        too_many = copy.deepcopy(self.doc); too_many["records"] = too_many["records"] * 86
        with self.assertRaises(DataContractError): join_predictions(too_many, self.cache, self.hash)
        source.write_bytes(b"x" * (MAX_BYTES + 1))
        with self.assertRaises(DataContractError): main(args)

    def test_interrupted_write_cleanup(self):
        root = self.fixture.root
        source, cache, output = root / "features.json", root / "cache.json", root / "partial.json"
        source.write_text(json.dumps(self.doc), encoding="utf-8")
        cache.write_text(json.dumps(self.cache), encoding="utf-8")
        args = ["--features", str(source), "--cache", str(cache), "--manifest-sha256", self.hash, "--output", str(output)]
        original = Path.open
        class Interrupted:
            def __init__(self, handle): self.handle = handle
            def __enter__(self): return self
            def __exit__(self, *args): self.handle.close()
            def write(self, raw):
                self.handle.write(raw[:10])
                raise KeyboardInterrupt()
        def injected(path, *args, **kwargs):
            handle = original(path, *args, **kwargs)
            return Interrupted(handle) if path == output and args and args[0] == "xb" else handle
        with patch.object(Path, "open", injected):
            with self.assertRaises(KeyboardInterrupt): main(args)
        self.assertFalse(output.exists())
