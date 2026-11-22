"""Wrist loss, winner verification, activation and controlled metrics."""

import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import patch

import numpy as np
import torch
from rewind_archive.contracts import canonical, digest
from rewind_m1.training import seed_run
from rewind_prepare.temporal_selection import seal
from rewind_v1.data import Clip
from rewind_v1.training import TeacherRun
from scripts import v211_trial, v214_decision, v214_eval, v214_run, v214_wrist
from scripts.v2_tuning_cpu_selection import ADDITIONAL, BODY, INITIAL, SCENARIOS, plan, summarize


class WristTests(TestCase):
    def test_weighted_sum_respects_mask_and_xyz_weighted_denominator(self):
        pred = torch.tensor([[[[3., 0., 0.], [2., 0., 0.], [6., 0., 0.]]]], requires_grad=True)
        target = torch.zeros_like(pred)
        valid = torch.tensor([[[True, False, True]]])
        self.assertAlmostEqual(float(v214_wrist.position_loss(pred, target, valid, 1, (0, 1), 1).detach()), (9+36)/6)
        weighted = v214_wrist.position_loss(pred, target, valid, 1, (0, 1), 2)
        self.assertAlmostEqual(float(weighted.detach()), (2*9+36)/9)
        weighted.backward()
        self.assertAlmostEqual(float(pred.grad[0, 0, 0, 0]), 12/9)
        self.assertEqual(float(pred.grad[0, 0, 1, 0]), 0.)
        self.assertAlmostEqual(float(pred.grad[0, 0, 2, 0]), 12/9)

    def test_weight_1_replays_stock_teacher_and_weight_2_keeps_frame_loss(self):
        seed_run(42)
        directory = Path(__file__).resolve().parents[1]
        cfg = json.loads((directory/"configs/rewind_v12_full_training.json").read_bytes())["teacher"]
        spec = {"joints": 3, "edges": [(0, 1)], "source": "predicted_2d", "input_sha256": "a"*64}
        original = TeacherRun(cfg, **spec)
        state = {name: value.clone() for name, value in original.model.state_dict().items()}
        clip = Clip(torch.randn(1, 4, 42), torch.randn(1, 4, 2, 9), torch.randn(1, 4, 3, 3)*30,
                    torch.ones(1, 4, 3, dtype=torch.bool), torch.ones(1, 4, dtype=torch.bool),
                    torch.arange(4, dtype=torch.float64)[None]/25, "predicted_2d", split="train", input_sha256="a"*64)
        v211_trial.begin_fit(original)
        stock = v211_trial.update(original, clip, 42)
        for method, weight in (("v2.1.1", 1), ("v2.1.1", 2), ("v2.1.2", 2)):
            stage = v214_wrist.WristTeacher(cfg, **spec)
            stage.model.load_state_dict(state)
            stage.wrist_weight, stage.wrists = weight, (0, 1)
            adapter = v214_wrist.strategy(method)
            adapter.begin_fit(stage, 42)
            tuned = v211_trial.update(stage, clip, 42)
            self.assertEqual(stock["frame"], tuned["frame"])
            if weight == 1:
                self.assertEqual(stock, tuned)
                self.assertTrue(all(torch.equal(p, stage.model.state_dict()[n]) for n, p in original.model.state_dict().items()))
            else:
                self.assertNotEqual(stock["position"], tuned["position"])
                with tempfile.TemporaryDirectory() as scratch:
                    saved = Path(scratch)/"fit.pt"
                    binding = {"method": method, "wrist_position_weight": 2}
                    adapter.save_state(stage, binding, saved, tuned)
                    resumed = v214_wrist.WristTeacher(cfg, **spec)
                    resumed.model.load_state_dict(state)
                    resumed.wrist_weight, resumed.wrists = 2, (0, 1)
                    adapter.begin_fit(resumed, 42)
                    self.assertEqual(adapter.load_state(resumed, binding, saved.read_bytes(), 1), tuned)
                    expected = v211_trial.update(stage, clip, 42)
                    self.assertEqual(v211_trial.update(resumed, clip, 42), expected)
                    self.assertTrue(all(torch.equal(p, resumed.model.state_dict()[n]) for n, p in stage.model.state_dict().items()))

    def test_missing_decision_never_reaches_cuda_or_drive(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict("os.environ", {v214_decision.SHA_ENV: ""}):
            with patch.object(v214_run.v211_run, "run") as launch:
                with self.assertRaisesRegex(ValueError, "reviewed_decision_sha256_required"):
                    v214_run.run(SimpleNamespace(work=Path(tmp)), {"trial_plan": {}}, {})
                launch.assert_not_called()

    def test_decision_recomputes_winner_and_rejects_changed_control(self):
        trial_plan = plan({"source": "fixture"}, {}, 250.)

        def row(method, seed, gap):
            pair = {role: {"mpjpe_mm": 200.+gap} for role in ("body", "hand_global")}
            return v211_trial.measurement(trial_plan, seed, f"{seed:064x}", {"per_scenario": {
                scene: {"official": pair} for scene in SCENARIOS}}, method)

        initial = [row(method, seed, gap) for gap, method in enumerate(BODY) for seed in INITIAL]
        extra = [row(method, seed, gap) for gap, method in enumerate(BODY[:2]) for seed in ADDITIONAL]
        screening = {"rows": initial, "summary_files": [{"method": r["method"], "seed": r["seed"], "sha256": "b"*64}
                                                         for r in initial]}
        verdict = {"v2.1.1": False, "v2.1.2": True}
        selected = summarize(trial_plan, initial+extra, phase="final", target_verdict=verdict)
        controls = {str(seed): {"summary_sha256": "b"*64, "summary_bytes": 21, "content_sha256": "d"*64,
                                "checkpoint_sha256": f"{seed:064x}"} for seed in INITIAL}
        body = {"schema": v214_decision.SCHEMA, "trial_plan_sha256": trial_plan["content_sha256"], "method": "v2.1.2",
                "qualification_sha256": digest(screening), "qualified_screening": screening, "additional_rows": extra,
                "final_release_receipts": v214_decision.FINAL_RELEASE_RECEIPTS,
                "selection_report_file_sha256": "a"*64, "target_verdict_file_sha256": "c"*64, "target_verdict": verdict,
                "selected_candidate": selected["candidate"], "controls": controls}
        with patch.object(v214_decision, "SCREENING_SHA", digest(screening)), tempfile.TemporaryDirectory() as scratch:
            file = Path(scratch)/v214_decision.DECISION_NAME
            decision = seal(body)
            file.write_bytes(canonical(decision))
            self.assertEqual(v214_decision.read_verified(file, digest(decision), trial_plan)["method"], "v2.1.2")
            with self.assertRaisesRegex(ValueError, "decision_file_sha256"):
                v214_decision.read_verified(file, "e"*64, trial_plan)
            with self.assertRaisesRegex(ValueError, "final_parent_not_selected"):
                v214_decision.verify(seal({**body, "method": "v2.1.1"}), trial_plan)
            with self.assertRaisesRegex(ValueError, "exact_final_selection"):
                v214_decision.verify(seal({**body, "additional_rows": extra[:-1]}), trial_plan)
            with self.assertRaisesRegex(ValueError, "control_identity"):
                changed = dict(controls, **{"42": dict(controls["42"], summary_sha256="e"*64)})
                v214_decision.verify(seal({**body, "controls": changed}), trial_plan)

    def test_body_without_wrists_and_hand_metrics_are_distinct(self):
        pred = np.zeros((1, 4, 3), dtype=np.float32)
        pred[0, :, 0] = [10., 30., 3., 5.]
        row = {"pred_device_mm": pred, "target_device_mm": np.zeros_like(pred), "target_valid": np.ones((1, 4), dtype=bool),
               "hand_local_pred_mm": np.asarray([[[2., 0., 0.]]], dtype=np.float32),
               "hand_local_target_mm": np.zeros((1, 1, 3), dtype=np.float32), "hand_local_valid": np.ones((1, 1), dtype=bool)}
        spec = {"body_joint_names": ["L", "R", "trunk"], "wrists": {"left": "L", "right": "R"}}
        result = v214_eval.sample_metrics(row, spec)
        self.assertEqual(result, {"left_wrist": (10., 1), "right_wrist": (30., 1), "body_without_wrists": (3., 1),
                                  "hand_global": (5., 1), "hand_local": (2., 1)})
