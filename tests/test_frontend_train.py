"""소량 CPU 학습, validation 격리와 weights 재사용 검사."""
import copy
import json
import unittest
from unittest.mock import patch

try:
    import torch
    from PIL import Image
except ImportError:
    torch = None

if torch is not None:
    import test_rgb_supervision as fixtures
    import rewind_m1.frontend_train as training
    from rewind_archive.contracts import digest
    from rewind_m1.data import DataContractError, InputContract
    from rewind_m1.frontend import read_prediction_cache
    from rewind_m1.local_rgb import LocalStereoRGB
    from rewind_m1.local_rgb_predictions import load_weights, build_cache


@unittest.skipUnless(torch is not None, "torch와 Pillow가 필요합니다")
class FrontendTrainTests(unittest.TestCase):
    def setUp(self):
        fixture = fixtures.SupervisionTests("test_lazy_key_order_split_and_gradient")
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        self.loader, self.doc, self.hash = fixture.loader, fixture.doc, fixture.hash
        self.root = self.loader._root
        self.config = dict(seed=17, steps=2, batch_size=1, learning_rate=.001, frontend_version="small_test_v1")

    def run_train(self, name, doc=None, config=None, loader=None):
        loader = self.loader if loader is None else loader
        return training.run_training(loader, self.doc if doc is None else doc, ["root", "tip"],
            loader.manifest_sha256, self.config if config is None else config, self.root/name)

    def test_training_export_and_validation_isolation(self):
        original = copy.deepcopy(self.doc)
        rng = torch.get_rng_state().clone()
        captured = {}
        save = training.save_frontend_weights
        def capture(model, meta, path):
            captured.update({k:v.detach().clone() for k,v in model.state_dict().items()})
            save(model, meta, path)
        with patch.object(training,"save_frontend_weights",side_effect=capture):
            report = self.run_train("one")
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        self.assertEqual(self.doc, original)
        self.assertNotEqual(report["initial_weights_sha256"], report["final_weights_sha256"])
        self.assertEqual(report["validation"]["status"], "evaluated")
        self.assertEqual(len(report["training"]), 2)
        train_keys = {tuple(k) for k in report["input"]["train_keys"]}
        for step in report["training"]:
            self.assertTrue(set(tuple(k) for k in step["keys"]) <= train_keys)
        changed = copy.deepcopy(self.doc)
        for r in changed["records"]:
            if r["split"] == "validation":
                for joint in r["joints"].values():
                    joint["camera_left_pts2d"] = [-10., -10.]
                    joint["camera_right_pts2d"] = [-10., -10.]
        other = self.run_train("two", changed)
        self.assertEqual(report["final_weights_sha256"], other["final_weights_sha256"])
        self.assertEqual(other["validation"]["status"], "not_evaluable")
        self.assertIsNone(other["validation"]["loss"])
        self.assertEqual(other["validation"]["zero_inside_batches"], 1)
        model, checkpoint = load_weights(self.root/"one"/"frontend_weights.pt", self.loader, ["root", "tip"])
        for name, tensor in model.state_dict().items():
            self.assertTrue(torch.equal(tensor, captured[name]))
        cache = build_cache(self.loader, model, checkpoint, ["root", "tip"])
        contract = InputContract.from_dict({**self.doc["contract"], "input_source":"predicted_2d",
            "frontend_version":checkpoint["frontend_version"], "frontend_checkpoint_sha256":checkpoint["frontend_checkpoint_sha256"],
            "preprocessing_version":"rgb_0_1_v1"})
        self.assertEqual(len(read_prediction_cache(cache, contract, self.hash)), 3)
        self.assertFalse(report["resume_supported"])
        self.assertIn("code_sha256", report["source"])

    def test_missing_validation_cli_and_existing_output(self):
        manifest = json.loads((self.root/"manifest.json").read_text())
        doc = copy.deepcopy(self.doc)
        for row in manifest["records"]:
            if row["split"] == "validation": row["split"] = "test"
        for row in doc["records"]:
            if row["split"] == "validation": row["split"] = "test"
        manifest["content_sha256"] = digest({k:v for k,v in manifest.items() if k != "content_sha256"})
        path = self.root/"test_manifest.json"; path.write_text(json.dumps(manifest), encoding="utf-8")
        features = self.root/"features.json"; features.write_text(json.dumps(doc), encoding="utf-8")
        config = self.root/"config.json"; config.write_text(json.dumps(self.config), encoding="utf-8")
        output = self.root/"cli"
        args = ["--manifest",str(path),"--root",str(self.root),"--features",str(features),"--joints","root","tip",
            "--manifest-sha256",manifest["content_sha256"],"--config",str(config),"--output",str(output)]
        training.main(args)
        report = json.loads((output/"report.json").read_text())
        self.assertEqual(report["validation"]["status"], "not_available")
        self.assertIsNone(report["validation"]["loss"])
        before = (output/"frontend_weights.pt").read_bytes()
        with self.assertRaises(FileExistsError): training.main(args)
        self.assertEqual((output/"frontend_weights.pt").read_bytes(), before)

    def test_guards_limits_and_interruption_cleanup(self):
        for key, value in (("seed",True),("seed",-1),("steps",True),("steps",101),("batch_size",False),
                ("learning_rate",float("nan")),("learning_rate",True),("frontend_version","x"*129),("unexpected",1)):
            config = {**self.config,key:value}
            with self.subTest(key=key), self.assertRaises(DataContractError): self.run_train("bad", config=config)
            self.assertFalse((self.root/"bad").exists())
        doc = copy.deepcopy(self.doc); doc["contract"]["data_kind"] = "dataset_sample"
        with self.assertRaisesRegex(DataContractError,"실데이터 학습에는"):
            self.run_train("guard",doc)
        with patch.object(training,"MAX_HEATMAP_ELEMENTS",1):
            with self.assertRaisesRegex(DataContractError,"heatmap element"):
                self.run_train("cap")
        with patch.object(training,"DEADLINE_SECONDS",-1):
            with self.assertRaisesRegex(DataContractError,"deadline"):
                self.run_train("deadline")
        self.assertFalse((self.root/"deadline").exists())
        def partial_save(model, meta, path):
            path.write_bytes(b"partial")
            raise KeyboardInterrupt()
        with patch.object(training,"save_frontend_weights",side_effect=partial_save):
            with self.assertRaises(KeyboardInterrupt): self.run_train("interrupt")
        self.assertFalse((self.root/"interrupt").exists())
        doc = copy.deepcopy(self.doc)
        for row in doc["records"]:
            if row["split"] == "train":
                for joint in row["joints"].values():
                    joint.pop("camera_left_pts2d", None)
                    joint.pop("camera_right_pts2d", None)
        with self.assertRaisesRegex(DataContractError,"supervise할 유효 2D 관절 없음"):
            self.run_train("no_labels",doc)
        self.assertFalse((self.root/"no_labels").exists())
