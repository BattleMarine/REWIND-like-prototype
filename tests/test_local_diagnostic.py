"""소량 end-to-end 실행 및 실패 산출물 보존 검사."""
import copy
import hashlib
import json
import os
import random
import unittest
from pathlib import Path
from unittest.mock import patch

try:
    import torch
    from PIL import Image
except ImportError:
    torch = None

if torch is not None:
    import test_rgb_supervision as fixtures
    import rewind_m1.local_diagnostic as runner
    from rewind_m1.data import DataContractError
    from rewind_m1.training import read_checkpoint


@unittest.skipUnless(torch is not None, "torch와 Pillow가 필요합니다")
class DiagnosticTests(unittest.TestCase):
    def setUp(self):
        fixture = fixtures.SupervisionTests("test_lazy_key_order_split_and_gradient")
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        self.loader,self.doc,self.hash = fixture.loader,fixture.doc,fixture.hash
        self.root = self.loader._root
        self.frontend = dict(seed=17,steps=2,batch_size=2,learning_rate=.001,frontend_version="runner_test_v1")
        project = Path(__file__).resolve().parents[1]
        self.config = json.loads((project/"configs/m1_single_synthetic.json").read_text())
        self.config.update(hidden=8,batch_size=2,window_size=2)

    def run_diagnostic(self, name, **changes):
        args = dict(loader=self.loader,pose_doc=self.doc,joint_names=["root","tip"],expected_manifest_sha256=self.hash,
            frontend_config=self.frontend,regression_config=self.config,groups={"body":["root"],"hand":["tip"]},
            output=self.root/name,regression_steps=2)
        args.update(changes)
        return runner.run_diagnostic(**args)

    def state(self):
        return (random.getstate(),torch.get_rng_state().clone(),torch.get_num_threads(),
            torch.are_deterministic_algorithms_enabled(),torch.is_deterministic_algorithms_warn_only_enabled(),
            os.environ.get("CUBLAS_WORKSPACE_CONFIG"))

    def assert_state(self, before):
        after = self.state()
        self.assertEqual(before[0],after[0])
        self.assertTrue(torch.equal(before[1],after[1]))
        self.assertEqual(before[2:],after[2:])

    def test_complete_pipeline_and_provenance(self):
        before = copy.deepcopy(self.doc)
        state = self.state()
        result = self.run_diagnostic("complete")
        self.assert_state(state)
        self.assertEqual(self.doc,before)
        self.assertEqual(result["status"],"completed")
        self.assertEqual([s["name"] for s in result["stages"]],
            ["frontend","prediction_cache","evaluation_2d","prediction_join","regression"])
        self.assertNotIn("report.json",result["artifacts"])
        for name, info in result["artifacts"].items():
            raw = (self.root/"complete"/name).read_bytes()
            self.assertEqual(info["sha256"],hashlib.sha256(raw).hexdigest())
            self.assertEqual(info["size_bytes"],len(raw))
        features = json.loads((self.root/"complete/predicted_features.json").read_text())
        for expected, row in zip(before["records"],features["records"]):
            row.pop("prediction")
            self.assertEqual(expected,row)
        checkpoint = read_checkpoint(self.root/"complete/regression/last.pt")
        self.assertEqual(checkpoint["step"],2)
        for key in ("optimizer","scheduler","scaler","rng","source","config"):
            self.assertIn(key,checkpoint)
        self.assertEqual(checkpoint["input_contract"]["input_source"],"predicted_2d")
        metrics = json.loads((self.root/"complete/regression/report.json").read_text())
        self.assertEqual(metrics["validation"]["split"],"validation")
        self.assertEqual(metrics["validation"]["input_source"],"predicted_2d")
        original = (self.root/"complete/report.json").read_bytes()
        with self.assertRaises(FileExistsError): self.run_diagnostic("complete")
        self.assertEqual((self.root/"complete/report.json").read_bytes(),original)

    def test_failure_and_interrupt_preserve_evidence_and_state(self):
        def fake_frontend(loader,doc,joints,hash_value,config,output):
            output.mkdir()
            (output/"report.json").write_text('{}',encoding="utf-8")
            (output/"frontend_weights.pt").write_bytes(b"fixture")
            random.seed(987)
            torch.manual_seed(123)
            torch.use_deterministic_algorithms(True,warn_only=True)
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
        for name, failure in (("failure",RuntimeError("injected")),("interrupted",KeyboardInterrupt())):
            before = self.state()
            def partial_cache(*args, **kwargs):
                (self.root/name/"predictions.json").write_bytes(b'{"partial":')
                raise failure
            with patch.object(runner,"run_training",side_effect=fake_frontend), patch.object(runner,"load_weights",return_value=(None,{})), patch.object(runner,"build_cache",side_effect=partial_cache):
                with self.assertRaises(type(failure)): self.run_diagnostic(name)
            self.assert_state(before)
            report = json.loads((self.root/name/"report.json").read_text())
            self.assertEqual(report["status"],"failed")
            self.assertEqual(report["failed_stage"],"prediction_cache")
            self.assertEqual(report["stages"][0]["status"],"completed")
            self.assertIn("frontend/frontend_weights.pt",report["artifacts"])
            self.assertIn("predictions.json",report["partial_artifacts"])
            self.assertNotIn("predictions.json",report["artifacts"])
            self.assertEqual((self.root/name/"predictions.json").read_bytes(),b'{"partial":')

    def test_restore_failure_never_reports_completion(self):
        with patch.object(runner,"run_training",side_effect=RuntimeError("injected stage failure")), \
             patch.object(runner.torch,"set_num_threads",side_effect=RuntimeError("injected restoration failure")):
            with self.assertRaises(runner.StateRestoreError): self.run_diagnostic("restore_failure")
        report = json.loads((self.root/"restore_failure/report.json").read_text())
        self.assertEqual(report["status"],"failed")
        self.assertEqual(report["failed_stage"],"restore_state")
        self.assertEqual(report["error_type"],"StateRestoreError")

    def test_preflight_guard_budgets_and_cli_existing(self):
        doc = copy.deepcopy(self.doc)
        doc["records"] = doc["records"][:2]
        with self.assertRaises(DataContractError): self.run_diagnostic("missing_validation",pose_doc=doc)
        self.assertFalse((self.root/"missing_validation").exists())
        for changes in ({"regression_steps":True}, {"regression_steps":101},
                {"regression_config":{**self.config,"window_size":21}}, {"groups":{"all":["root"]}}):
            with self.assertRaises(ValueError): self.run_diagnostic("invalid",**changes)
            self.assertFalse((self.root/"invalid").exists())
        doc = copy.deepcopy(self.doc); doc["contract"]["data_kind"] = "dataset_sample"
        with self.assertRaisesRegex(DataContractError,"실데이터 학습에는"):
            self.run_diagnostic("guard",pose_doc=doc)
        with patch.object(runner,"MAX_TENSOR_BYTES",1):
            with self.assertRaisesRegex(DataContractError,"tensor 추정"):
                self.run_diagnostic("budget")
        for name, value in (("features",self.doc),("frontend",self.frontend),("regression",self.config),("groups",{})):
            (self.root/f"{name}.json").write_text(json.dumps(value),encoding="utf-8")
        (self.root/"existing").mkdir()
        args = ["--manifest",str(self.root/"manifest.json"),"--root",str(self.root),"--features",str(self.root/"features.json"),
            "--frontend-config",str(self.root/"frontend.json"),"--regression-config",str(self.root/"regression.json"),
            "--groups",str(self.root/"groups.json"),"--manifest-sha256",self.hash,"--joints","root","tip","--output",str(self.root/"existing")]
        with self.assertRaises(FileExistsError): runner.main(args)
