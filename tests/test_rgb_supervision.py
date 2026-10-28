"""소형 RGB/GT supervision 연결과 heatmap gradient 검증."""
import copy
import unittest
from unittest.mock import patch

try:
    import torch
    from PIL import Image
except ImportError:
    torch = None

if torch is not None:
    import test_prediction_join as fixtures
    from rewind_m1.data import DataContractError
    from rewind_m1.frontend import StereoHeatmapNet, heatmap_loss
    from rewind_m1.local_rgb import RGBLimits
    from rewind_m1.rgb_supervision import LocalRGBSupervision


@unittest.skipUnless(torch is not None, "torch와 Pillow가 필요합니다")
class SupervisionTests(unittest.TestCase):
    def setUp(self):
        fixture = fixtures.PredictionJoinTests("test_join_preserves_pose_and_loads_predictions")
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        self.loader = fixture.fixture.loader
        self.doc = fixture.doc
        self.hash = fixture.hash
        for row in self.doc["records"]:
            for j, name in enumerate(("root", "tip")):
                row["joints"][name]["camera_left_pts2d"] = [1. + j, 2.]
                row["joints"][name]["camera_right_pts2d"] = [4. + j, 3.]

    def adapter(self, doc=None, **changes):
        args = dict(expected_manifest_sha256=self.hash, joint_names=["root", "tip"], split="train")
        args.update(changes)
        return LocalRGBSupervision(self.loader, self.doc if doc is None else doc, **args)

    def test_lazy_key_order_split_and_gradient(self):
        # RGB와 다른 pose 순서, GT annotation 누락 및 화면 밖은 서로 구분한다.
        self.doc["records"][0], self.doc["records"][1] = self.doc["records"][1], self.doc["records"][0]
        self.doc["records"][0]["joints"]["tip"].pop("camera_left_pts2d")
        self.doc["records"][0]["joints"]["tip"]["camera_right_pts2d"] = [-2., 3.]
        before = copy.deepcopy(self.doc)
        with patch.object(self.loader, "load_batch", wraps=self.loader.load_batch) as read:
            adapter = self.adapter()
            read.assert_not_called()
            batch = adapter.load_batch([0, 1])
            self.assertEqual(read.call_args.args[0], [1, 0])
        self.assertEqual(batch["images"].shape, (2, 2, 3, 6, 8))
        self.assertTrue(torch.equal(batch["images"], self.loader.load_batch([1, 0])["images"]))
        self.assertEqual(batch["gt_xy"].shape, (2, 2, 2, 2))
        self.assertEqual(batch["annotation_valid"].dtype, torch.bool)
        self.assertFalse(batch["annotation_valid"][0, 0, 1])
        self.assertEqual(batch["gt_xy"][0, 0, 1].tolist(), [0., 0.])
        self.assertTrue(batch["annotation_valid"][0, 1, 1])
        self.assertEqual(batch["gt_xy"][0, 1, 0].tolist(), [4., 3.])
        self.assertEqual(batch["gt_xy"][0, 0, 0].tolist(), [1., 2.])
        model = StereoHeatmapNet(2)
        logits, _, _ = model(batch["images"])
        loss, inside = heatmap_loss(logits, batch["gt_xy"], batch["annotation_valid"], (8, 6))
        self.assertFalse(inside[0, 1, 1])
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))
        self.assertEqual(self.doc, before)
        for i in range(2):
            self.assertEqual(batch["metadata"][i]["pose_record"], before["records"][i])
        metadata = adapter.metadata(0)
        metadata["pose_record"]["joints"].clear()
        metadata["rgb"]["record"]["views"].clear()
        batch["metadata"][0]["pose_record"]["joints"].clear()
        self.assertEqual(adapter.metadata(0)["pose_record"], before["records"][0])
        self.doc["records"][0]["joints"].clear()
        self.assertEqual(adapter.metadata(0)["pose_record"], before["records"][0])
        self.assertEqual(len(self.adapter(before, split="validation")), 1)

    def test_contract_keys_and_guards(self):
        for changes in ({"joint_names": ["tip", "root"]}, {"expected_manifest_sha256": "0" * 64}, {"split": "all"}, {"split": "test"}):
            with self.subTest(changes=changes), self.assertRaises(DataContractError): self.adapter(**changes)
        cases = []
        for schema in ("real_pose_candidate_v1", "bad"):
            doc = copy.deepcopy(self.doc); doc["schema_version"] = schema; cases.append(doc)
        for change in ({"input_source": "predicted_2d"}, {"image_sizes": [[9, 6], [9, 6]]}):
            doc = copy.deepcopy(self.doc); doc["contract"].update(change); cases.append(doc)
        for kind in ("missing", "duplicate", "extra", "split", "prediction"):
            doc = copy.deepcopy(self.doc)
            if kind == "missing": doc["records"].pop()
            elif kind == "duplicate": doc["records"][1] = copy.deepcopy(doc["records"][0])
            elif kind == "extra": doc["records"][0]["frame_index"] = 9
            elif kind == "split": doc["records"][0]["split"] = "test"
            else: doc["records"][0]["prediction"] = {}
            cases.append(doc)
        checks = ("joints", "units", "coordinates", "calibration", "split", "rgb_pose_alignment")
        guard_error = "실데이터 학습에는 관절/단위/좌표/calibration/split/RGB-pose 검증과 근거가 필요"
        doc = copy.deepcopy(self.doc)
        doc["contract"]["data_kind"] = "dataset_sample"
        with self.assertRaisesRegex(DataContractError, guard_error):
            self.adapter(doc)
        for missing in checks:
            doc = copy.deepcopy(self.doc)
            doc["contract"].update(data_kind="dataset_sample", verified={k: k != missing for k in checks},
                                   verification_evidence={k: "synthetic guard test only" for k in checks})
            with self.subTest(missing_verification=missing), self.assertRaisesRegex(DataContractError, guard_error):
                self.adapter(doc)
        doc = copy.deepcopy(self.doc); doc["padding"] = "x" * (4 * 1024**2); cases.append(doc)
        doc = copy.deepcopy(self.doc); doc["records"] *= 86; cases.append(doc)
        for doc in cases:
            with self.subTest(schema=doc["schema_version"]), self.assertRaises(DataContractError): self.adapter(doc)

    def test_indices_and_combined_budget_before_io(self):
        adapter = self.adapter()
        for indices in ([], [True], [-1], [2], [0] * 5):
            with self.subTest(indices=indices), self.assertRaises(DataContractError): adapter.load_batch(indices)
        # RGB 단독 1152bytes는 통과하지만 GT/mask 36bytes를 더하면 초과한다.
        self.loader._limits = RGBLimits(max_batch_tensor_bytes=1152)
        adapter = self.adapter()
        with patch.object(self.loader, "load_batch", side_effect=AssertionError("이미지 읽기 전에 거부해야 함")):
            with self.assertRaises(DataContractError): adapter.load_batch([0])
