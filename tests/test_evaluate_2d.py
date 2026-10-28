"""손계산 가능한 2D 평가와 기존 weights/cache 호환 검사."""
import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

try:
    import torch
except ImportError:
    torch = None

if torch is not None:
    from rewind_m1.data import DataContractError, content_hash, synthetic_document
    from rewind_m1.evaluate_2d import evaluate_cache, main


@unittest.skipUnless(torch is not None, "torch가 필요합니다")
class EvaluationTests(unittest.TestCase):
    def setUp(self):
        self.doc = synthetic_document()
        self.doc["records"] = self.doc["records"][:2]
        records = []
        for row in self.doc["records"]:
            row["split"] = "validation"
            for name, xy in (("root", [0., 0.]), ("tip", [2., 2.])):
                for side in ("left", "right"):
                    row["joints"][name][f"camera_{side}_pts2d"] = list(xy)
            prediction = {"joint_names":["root", "tip"], "frontend_version":"fixed_v1", "frontend_checkpoint_sha256":"a"*64, "preprocessing_version":"rgb_0_1_v1"}
            for side in ("left", "right"):
                prediction[side] = {"coordinates":[[0.,0.],[2.,2.]], "valid_mask":[True,True], "visibility":[None,None]}
            records.append({"sequence_id":row["sequence_id"], "frame_index":row["frame_index"], "prediction":prediction})
        records[0]["prediction"]["left"].update(coordinates=[[3.,4.],None], valid_mask=[True,False])
        records[0]["prediction"]["right"]["coordinates"] = [[2.,0.],[6.,2.]]
        records[1]["prediction"]["left"]["coordinates"][0] = [1.,0.]
        self.doc["records"][1]["joints"]["tip"].pop("camera_left_pts2d")
        self.doc["records"][1]["joints"]["root"]["camera_right_pts2d"] = [-2.,0.]
        records[1]["prediction"]["right"]["coordinates"] = [[-2.,3.],[8.,2.]]
        self.hash = "b"*64
        self.cache = {"schema_version":"predicted_2d_cache_v1", "data_kind":"synthetic_fixture", "input_source":"predicted_2d",
            "manifest_hash":self.hash, "image_sizes":[[64,48],[64,48]], "joint_names":["root","tip"],
            "frontend_version":"fixed_v1", "frontend_checkpoint_sha256":"a"*64, "preprocessing_version":"rgb_0_1_v1",
            "records":list(reversed(records)), "record_splits":["validation","validation"]}
        self.rehash()

    def rehash(self):
        self.cache["content_sha256"] = content_hash({k:v for k,v in self.cache.items() if k != "content_sha256"})

    def evaluate(self, **kwargs):
        args = dict(split="validation", groups={"body":["root"],"hand":["tip"],"overlap":["root","tip"]})
        args.update(kwargs)
        return evaluate_cache(self.doc,self.cache,self.hash,**args)

    def test_hand_calculated_weighting_overlap_and_immutability(self):
        before = copy.deepcopy((self.doc,self.cache))
        report = self.evaluate()
        self.assertIn("rewind_m1/evaluate_2d.py",report["source"]["code_sha256"])
        self.assertIn("revision",report["source"])
        all_metrics = report["metrics"]["all"]
        self.assertEqual(all_metrics["left"]["mean_error_px"],3.)
        self.assertEqual(all_metrics["right"]["mean_error_px"],3.75)
        combined = all_metrics["combined"]
        self.assertEqual(combined["mean_error_px"],3.5)
        self.assertEqual(combined["sum_error_px"],21.)
        self.assertEqual(combined["paired_count"],6)
        self.assertEqual(combined["gt_valid_count"],7)
        self.assertEqual(combined["gt_missing_count"],1)
        self.assertEqual(combined["pred_missing_on_gt_count"],1)
        self.assertAlmostEqual(combined["coverage"],6/7)
        self.assertEqual(combined["gt_out_of_frame_count"],1)
        self.assertEqual(combined["pred_out_of_frame_paired_count"],1)
        self.assertEqual(combined["gt_visibility_unknown_count"],7)
        self.assertEqual(combined["pred_visibility_unknown_on_gt_count"],7)
        self.assertEqual(report["metrics"]["body"]["combined"]["mean_error_px"],2.75)
        self.assertEqual(report["metrics"]["hand"]["combined"]["mean_error_px"],5.)
        self.assertEqual(self.evaluate(groups={})["metrics"]["all"], all_metrics)
        self.assertEqual((self.doc,self.cache),before)
        groups = {"one":["root"]}
        report = self.evaluate(groups=groups)
        report["groups"]["one"].clear()
        self.assertEqual(groups,{"one":["root"]})
        self.cache["records"][0]["prediction"]["right"]["visibility"][0] = False
        self.rehash()
        metric = self.evaluate()["metrics"]["all"]["combined"]
        self.assertEqual(metric["mean_error_px"],3.5)
        self.assertEqual(metric["pred_visibility_unknown_on_gt_count"],6)

    def test_missing_and_zero_gt(self):
        for row in self.cache["records"]:
            for side in ("left","right"):
                row["prediction"][side].update(coordinates=[None,None],valid_mask=[False,False])
        self.rehash()
        metric = self.evaluate()["metrics"]["all"]["combined"]
        self.assertEqual(metric["coverage"],0.)
        self.assertEqual(metric["missing_prediction_rate"],1.)
        self.assertIsNone(metric["mean_error_px"])
        for row in self.doc["records"]:
            for joint in row["joints"].values():
                joint.pop("camera_left_pts2d",None)
                joint.pop("camera_right_pts2d",None)
        metric = self.evaluate()["metrics"]["all"]["combined"]
        self.assertEqual(metric["sum_error_px"],0.)
        for key in ("mean_error_px","coverage","missing_prediction_rate"):
            self.assertIsNone(metric[key])

    def test_groups_split_and_contract_rejections(self):
        for groups in ({"all":["root"]},{"bad":["unknown"]},{"bad":["root","root"]}):
            with self.assertRaises(DataContractError): self.evaluate(groups=groups)
        with self.assertRaises(DataContractError): self.evaluate(split="train")
        with self.assertRaises(DataContractError): self.evaluate(split="test")
        self.doc["contract"]["data_kind"] = "dataset_sample"
        with self.assertRaisesRegex(DataContractError,"실데이터 학습에는"):
            self.evaluate()
        self.doc["contract"]["data_kind"] = "synthetic_fixture"
        self.hash = "c"*64
        with self.assertRaises(DataContractError): self.evaluate()
        self.hash = "b"*64
        self.cache["records"][0]["frame_index"] = 99; self.rehash()
        with self.assertRaises(DataContractError): self.evaluate()

    def test_explicit_test_and_train_diagnostic(self):
        for split in ("test","train"):
            for row in self.doc["records"]:
                row["split"] = split
            self.cache["record_splits"] = [split,split]
            self.rehash()
            report = self.evaluate(split=split,train_diagnostic=split == "train")
            self.assertEqual(report["split"],split)
            self.assertEqual(report["train_diagnostic"],split == "train")

    def test_cli_exclusive_and_interrupted_output(self):
        root = Path(__file__).resolve().parents[1]/"outputs"/"test_tmp"
        root.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(dir=root) as temp:
            root = Path(temp)
            for name, doc in (("features",self.doc),("cache",self.cache),("groups",{})):
                (root/f"{name}.json").write_text(json.dumps(doc),encoding="utf-8")
            output = root/"out.json"
            args = ["--features",str(root/"features.json"),"--cache",str(root/"cache.json"),"--groups",str(root/"groups.json"),
                "--manifest-sha256",self.hash,"--split","validation","--output",str(output)]
            main(args)
            original = output.read_bytes()
            with self.assertRaises(FileExistsError): main(args)
            self.assertEqual(output.read_bytes(),original)
            output.unlink()
            open_file = Path.open
            class Interrupted:
                def __init__(self, handle): self.handle=handle
                def __enter__(self): return self
                def __exit__(self,*args): self.handle.close()
                def write(self, raw):
                    self.handle.write(raw[:10]); raise KeyboardInterrupt()
            def injected(path,*args,**kwargs):
                handle=open_file(path,*args,**kwargs)
                return Interrupted(handle) if path==output and args and args[0]=="xb" else handle
            with patch.object(Path,"open",injected):
                with self.assertRaises(KeyboardInterrupt): main(args)
            self.assertFalse(output.exists())

    def test_export_cache_evaluation_compatibility(self):
        try:
            import PIL
        except ImportError:
            self.skipTest("Pillow가 필요합니다")
        import test_frontend_train as fixtures
        from rewind_m1.local_rgb_predictions import load_weights, build_cache
        fixture = fixtures.FrontendTrainTests("test_training_export_and_validation_isolation")
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        fixture.run_train("evaluation_export",config={**fixture.config,"steps":1})
        model, checkpoint = load_weights(fixture.root/"evaluation_export"/"frontend_weights.pt",fixture.loader,["root","tip"])
        cache = build_cache(fixture.loader,model,checkpoint,["root","tip"])
        report = evaluate_cache(fixture.doc,cache,fixture.hash,split="validation",groups={})
        metric = report["metrics"]["all"]["combined"]
        self.assertEqual(metric["paired_count"],4)
        self.assertEqual(metric["coverage"],1.)
        self.assertGreaterEqual(metric["mean_error_px"],0.)
