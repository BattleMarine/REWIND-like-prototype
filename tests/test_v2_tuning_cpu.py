"""Regression checks for CPU mask math, trial guards, and subprocess settlement."""

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
from unittest import TestCase

import numpy as np

from scripts.v2_tuning_cpu_colab import run_cpu_child
from scripts.v2_tuning_cpu_reference import math_checks, wrist_metrics
from scripts.v2_tuning_cpu_selection import BODY, INITIAL, METRICS, SCENARIOS, plan, summarize
from scripts.v2_tuning_cpu_trials import resume_math, selection_fixture


class CpuContractsTests(TestCase):
    def test_wrist_decomposition_intersects_masks(self):
        truth = np.zeros((2, 3, 3), np.float64)
        pred = truth.copy()
        pred[:, 0, 0] = 2  # Wrist.
        pred[:, 1, 0] = 3  # One common observation.
        pred[:, 2, 0] = 7  # Invalid finger in the one frame with a valid wrist.
        mask = np.array([[True, True, False], [False, True, True]])
        metrics = wrist_metrics(pred, truth, mask, [0], [1, 2], 0)
        self.assertEqual(metrics["paired_observations"], 1)
        self.assertEqual(metrics["wrist_mpjpe_mm"], 2)
        self.assertEqual(metrics["paired_hand_local_mpjpe_mm"], 1)
        self.assertEqual(metrics["paired_hand_global_mpjpe_mm"], 3)
        self.assertEqual(metrics["mean_inner_product_mm2"], 2)

    def test_isolated_math(self):
        actual = math_checks({"diffusion_steps": 1000, "beta_start": 0.0001, "beta_end": 0.02})
        self.assertEqual(actual["lora_rank"], 4)
        self.assertEqual(actual["ddim_timesteps"]["2"], [1000, 1])
        self.assertLess(actual["x0_v_max_roundtrip_error"], 1e-12)

    def test_cpu_resume_and_selection_fixture(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertTrue(resume_math(Path(tmp))["same_loss_and_weights"])
        trial_plan = plan({"parent": "A"}, {}, 200)
        check = selection_fixture(trial_plan)
        self.assertEqual(check["screening_rows"], 9)
        self.assertEqual(check["extended_rows"], 13)

    def test_mismatched_contract_and_failed_score_are_rejected(self):
        trial_plan = plan({"parent": "A"}, {}, 200)
        good = {"method": BODY[0], "seed": INITIAL[0], "status": "completed",
                "trial_plan_sha256": trial_plan["content_sha256"], "checkpoint_sha256": "a" * 64,
                "comparison_common": trial_plan["common"], "updates": 3097,
                "training": {"learning_rate": 5e-5, "batch_size": 1, "epoch": 1},
                "scenario_scores": {scene: dict.fromkeys(METRICS, 199) for scene in SCENARIOS}}
        result = summarize(trial_plan, [good], phase="screening")
        self.assertEqual(len(result["missing"]), 8)
        self.assertEqual(result["status"], "awaiting_screening_rows")
        with self.assertRaisesRegex(ValueError, "duplicate"):
            summarize(trial_plan, [good, good], phase="screening")
        bad = deepcopy(good)
        bad["comparison_common"]["input_2d"] = "gt"
        with self.assertRaisesRegex(ValueError, "common inputs"):
            summarize(trial_plan, [bad], phase="screening")
        with self.assertRaisesRegex(ValueError, "incomplete trial supplied"):
            summarize(trial_plan, [{**good, "status": "failed"}], phase="screening")

    def test_child_failure_without_report_is_settled_at_reserved_cap(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            destination = root / "result"
            destination.mkdir()
            limits = {"seconds": 10, "read_bytes": 100, "scratch_bytes": 0}
            result = run_cpu_child("inputs", limits, root, root, root, destination,
                                   command=[sys.executable, "-c", "import sys; print('test'); sys.exit(7)"])
            report = result["result"]
            self.assertEqual((report["returncode"], report["status"], report["settlement"], report["project_read_charged_bytes"]),
                             (7, "failed", "reserved_upper_bound", 100))
            raw = (destination / "inputs/process-result.json").read_bytes()
            self.assertEqual(result["result_pin"]["sha256"], hashlib.sha256(raw).hexdigest())
            self.assertEqual(json.loads(raw)["files"]["process.log"]["name"], "process.log")
