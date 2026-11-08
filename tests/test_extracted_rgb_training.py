"""CPU fixture checks for the explicit direct-input RGB2D diagnostic."""

import copy
import json
import math
import unittest
from unittest.mock import patch

import torch

from rewind_archive.contracts import digest
from rewind_m1.extracted_rgb_cli import DirectRGBInference, RunLimits, run
from rewind_m1.extracted_rgb_training import (
    DirectRGBTrainer, RUN_SCHEMA, comparison_config_sha256, verify_promotion,
)
from rewind_m1.frontend import heatmap_loss
from rewind_m1.rgb_frontend_artifacts import resize_images, scale_xy
from rewind_m1.training import rng_state
import test_extracted_stereo


CONFIG = {"seed": 71, "batch_size": 2, "learning_rate": 1e-4, "scheduler_step": 51,
          "scheduler_gamma": 1.0, "amp": False, "training_image_size": [64, 64],
          "frontend_version": "direct_fixture", "max_native_tensor_bytes": 64 * 1024**2,
          "max_heatmap_elements": 2 * 1024**2, "loss_kind": "gaussian_cross_entropy"}


class DirectRGBTrainingTests(unittest.TestCase):
    def setUp(self):
        self.fixture = test_extracted_stereo.ExtractedStereoTests()
        self.addCleanup(self.fixture.doCleanups)
        self.fixture.setUp()
        self.base = self.fixture.base
        self.manifest = self.fixture.prepare("pilot", train_limit=3)
        self.config = self.base / "config.json"
        self.config.write_text(json.dumps(CONFIG), encoding="utf-8")
        self.limits = RunLimits(120, 10, 64 * 1024**2, 2 * 1024**3, 32 * 1024**2, 1024**3, 1024**3)

    def assert_state_equal(self, a, b):
        if isinstance(a, torch.Tensor):
            self.assertTrue(torch.equal(a, b))
        elif isinstance(a, dict):
            self.assertEqual(set(a), set(b))
            for key in a:
                self.assert_state_equal(a[key], b[key])
        elif isinstance(a, (list, tuple)):
            self.assertEqual(len(a), len(b))
            for x, y in zip(a, b):
                self.assert_state_equal(x, y)
        else:
            self.assertEqual(a, b)

    def execute(self, name, steps, **overrides):
        args = dict(manifest=self.base / "pilot/manifest.json", manifest_sha256=self.manifest["content_sha256"],
                    rgb_root=self.fixture.rgb, pose_root=self.fixture.fixture.raw, spec=self.fixture.fixture.specpath,
                    config=self.config, output=self.base / name, limits=self.limits, steps=steps, checkpoint_every=1)
        return run(**{**args, **overrides})

    def test_resume_at_mid_epoch_and_boundary_matches_uninterrupted(self):
        full = self.execute("uninterrupted", 5)
        part = self.execute("mid", 1)
        boundary = self.execute("boundary", 2, resume=self.base / "mid/last.pt",
                                resume_sha256=part["checkpoint"]["sha256"])
        resumed = self.execute("resumed", 5, resume=self.base / "boundary/last.pt",
                               resume_sha256=boundary["checkpoint"]["sha256"])
        self.assertEqual(boundary["coverage"]["completed_epochs"], 1)
        self.assertEqual(boundary["diagnostic_progress"]["last_completed_unique_train"], 3)
        self.assertEqual(part["diagnostic_progress"]["current_unique_train"], 2)
        a = torch.load(self.base / "uninterrupted/last.pt", weights_only=True)
        b = torch.load(self.base / "resumed/last.pt", weights_only=True)
        self.assert_state_equal(a, b)
        events = part["recent_updates"] + boundary["recent_updates"] + resumed["recent_updates"]
        self.assertEqual([x["keys"] for x in events], [x["keys"] for x in full["recent_updates"]])
        self.assertEqual(full["final_metrics"], resumed["final_metrics"])
        self.assertNotEqual(full["initial_weights_sha256"], full["final_weights_sha256"])
        self.assertGreater(full["diagnostic_progress"]["nonzero_gradient_updates"], 0)
        self.assertFalse(full["operational_pilot_passed"])
        self.assertGreater(full["resources"]["input_bytes_reserved"], 0)
        self.assertGreater(full["resources"]["peak_artifact_bytes"], 0)

        reader = self.fixture.loader("pilot", self.manifest, "train")
        sample = reader.load_batch([0])["images"]
        trainer = DirectRGBTrainer(CONFIG, reader)
        trainer.resume(b)
        model = DirectRGBInference(self.base / "resumed/weights.pt",
                                   expected_sha256=resumed["weights_file_sha256"])
        with torch.no_grad():
            _, xy, confidence = trainer.model(resize_images(sample, [64, 64]))
        predicted = model.predict(sample)
        self.assertTrue(torch.equal(predicted["coordinates"], scale_xy(xy, [64, 64], [1024, 1024])))
        self.assertTrue(torch.equal(predicted["confidence"], confidence))

    def test_evaluation_preserves_rng_sampler_and_mse_default(self):
        config = {k: v for k, v in CONFIG.items() if k != "loss_kind"}
        trainer = DirectRGBTrainer(config, self.fixture.loader("pilot", self.manifest, "train"))
        cursor, rng = trainer.sampler.state_dict(), rng_state()
        metrics = trainer.evaluate()
        self.assertEqual(metrics["loss_kind"], "mse")
        self.assertEqual(metrics["records"], 1)
        self.assertEqual(metrics["groups"]["body"]["valid"]["joint_views"], 44)
        self.assertEqual(metrics["groups"]["hands"]["valid"]["joint_views"], 80)
        self.assertEqual(cursor, trainer.sampler.state_dict())
        self.assert_state_equal(rng, rng_state())
        self.assertTrue(trainer.model.training)

    @patch("rewind_m1.extracted_rgb_training.CHUNK_SIZE", 4)
    def test_multi_chunk_resume_preserves_shuffle_bitmap_and_unique_coverage(self):
        for frame in range(3, 7):
            self.fixture.add_pair(self.fixture.train, frame, 72)
        manifest = self.fixture.prepare("multi_chunk", train_limit=7)
        reader = self.fixture.loader("multi_chunk", manifest, "train")

        def make_trainer():
            return DirectRGBTrainer(CONFIG, reader)

        uninterrupted = make_trainer()
        self.assertEqual(uninterrupted.sampler.sizes, {0: 4, 1: 3})
        expected = [uninterrupted.train_step()["keys"] for _ in range(9)]
        self.assertEqual([len(keys) for keys in expected[:8]], [2, 2, 2, 1, 2, 1, 2, 2])
        # Seed 71 visits the first chunk first in epoch 0 and the second first in epoch 1.
        self.assertTrue(all(key[1] < 4 for keys in expected[:2] for key in keys))
        self.assertTrue(all(key[1] >= 4 for keys in expected[4:6] for key in keys))
        for start in (0, 4):
            visited = [tuple(key) for keys in expected[start:start + 4] for key in keys]
            self.assertEqual(len(visited), 7)
            self.assertEqual(set(visited), {(self.fixture.train, frame) for frame in range(7)})

        resumed, observed = make_trainer(), []
        for target, unique in ((1, 2), (2, 4), (4, 0), (9, 2)):
            while resumed.step < target:
                observed.append(resumed.train_step()["keys"])
            state = copy.deepcopy(resumed.checkpoint())
            self.assertEqual(state["diagnostic_progress"]["current_unique_train"], unique)
            self.assertEqual(int(state["epoch_bitmap"], 16).bit_count(), unique)
            if target == 1:
                self.assertEqual(state["cursor"]["offset"], 2)
                tampered = {**state, "epoch_bitmap": "30"}  # same count, wrong chunk
                with self.assertRaisesRegex(ValueError, "direct_rgb_checkpoint_coverage"):
                    make_trainer().resume(tampered)
            if target == 2:
                self.assertEqual((state["cursor"]["shard_position"], state["cursor"]["offset"]), (1, 0))
            if target == 4:
                self.assertEqual(state["cursor"]["epoch"], 1)
                self.assertEqual(state["diagnostic_progress"]["last_completed_unique_train"], 7)
            resumed = make_trainer()
            resumed.resume(state)

        self.assertEqual(observed, expected)
        self.assertEqual(resumed.coverage()["completed_epochs"], 2)
        self.assertEqual(resumed.samples_seen, 16)
        self.assert_state_equal(uninterrupted.checkpoint(), resumed.checkpoint())

    def test_simulated_amp_skip_does_not_count_samples_or_optimizer_updates(self):
        class SimulatedOverflow:
            value = 2

            def scale(self, loss):
                return loss

            def unscale_(self, optimizer):
                pass

            def step(self, optimizer):
                pass

            def update(self):
                self.value = 1

            def get_scale(self):
                return self.value

        trainer = DirectRGBTrainer(CONFIG, self.fixture.loader("pilot", self.manifest, "train"))
        trainer.scaler = SimulatedOverflow()
        cursor = trainer.sampler.state_dict()
        event = trainer.train_step()
        self.assertTrue(event["amp_skipped"])
        self.assertEqual((trainer.step, trainer.samples_seen, trainer.attempts, trainer.skipped_updates),
                         (0, 0, 1, 1))
        self.assertEqual(trainer.sampler.state_dict(), cursor)
        self.assertEqual(trainer.progress()["current_unique_train"], 0)

    def test_budget_stop_and_full_requires_external_evidence(self):
        result = self.execute("bounded", 1, limits=RunLimits(
            120, 10, 8, 2 * 1024**3, 32 * 1024**2, 1024**3, 1024**3))
        self.assertEqual(result["status"], "bounded_stop")
        self.assertFalse((self.base / "bounded/weights.pt").exists())
        full = self.fixture.prepare("full")
        with self.assertRaisesRegex(ValueError, "direct_full_pilot_evidence_required"):
            DirectRGBTrainer(CONFIG, self.fixture.loader("full", full, "train"))


class LossAndPromotionTests(unittest.TestCase):
    def test_normalized_gaussian_cross_entropy_and_inside_mask(self):
        logits = torch.zeros((1, 2, 1, 2, 2), requires_grad=True)
        xy = torch.tensor([[[[0., 0.]], [[16., 0.]]]])
        valid = torch.ones((1, 2, 1), dtype=torch.bool)
        loss, inside = heatmap_loss(logits, xy, valid, (4, 4), loss_kind="gaussian_cross_entropy")
        self.assertAlmostEqual(float(loss.detach()), math.log(4), places=6)
        self.assertEqual(inside.tolist(), [[[True], [False]]])
        loss.backward()
        self.assertGreater(float(logits.grad[0, 0].norm()), 0)
        self.assertEqual(float(logits.grad[0, 1].norm()), 0)
        self.assertTrue(torch.equal(heatmap_loss(logits, xy, valid, (4, 4))[0],
                                    heatmap_loss(logits, xy, valid, (4, 4), loss_kind="mse")[0]))

    def test_full_gate_requires_review_actual_unique_coverage_and_same_payloads(self):
        identity = "a" * 64
        rows = [["seq", i, identity] for i in range(1000)]
        records = {(row[0], row[1]): row[2] for row in rows}
        binding = {"validation_keys_sha256": identity, "validation_payloads_sha256": identity,
                   "training_contract_sha256": identity,
                   "validation_count": 5, "data_identity": {"environment": "fixture"}}
        report = {"schema": RUN_SCHEMA, "status": "completed", "operational_pilot_passed": True,
                  "config_sha256": digest(CONFIG), "comparison_config_sha256": comparison_config_sha256(CONFIG),
                  "diagnostic_only": True, "train_ready": False,
                  "binding": {**binding, "phase": "pilot", "train_count": 1000},
                  "diagnostic_progress": {"last_completed_unique_train": 1000, "nonzero_gradient_updates": 50},
                  "coverage": {"completed_epochs": 1}, "checkpoint": {"sha256": identity, "step": 500},
                  "weights_file_sha256": identity, "initial_weights_sha256": identity,
                  "final_weights_sha256": "b" * 64, "pilot_train_records": rows}
        for phase, step in (("initial_metrics", 0), ("final_metrics", 500)):
            report[phase] = {s: {"records": n, "step": step, "loss_kind": CONFIG["loss_kind"],
                                 "representative_predictions": [{"key": ["fixture", 0]}],
                                 "groups": {k: {} for k in ("whole", "body", "hands")},
                                 **{k: 1.0 for k in ("loss", "native_mean_error_pixels",
                                     "center_mean_error_pixels", "mean_confidence", "mean_entropy")}}
                             for s, n in (("train", 1000), ("validation", 5))}
        review = {"schema": "direct_rgb2d_promotion_review_v1", "pilot_report_sha256": identity,
                  "decision": "approve_full_train", "reviewer": "fixture",
                  "learning_signal_evidence": "checked initial and final group metrics",
                  "representative_prediction_evidence": "compared spatial patterns with center"}
        verify_promotion(report, identity, review, CONFIG, binding, records)
        verify_promotion(report, identity, review,
                         {**CONFIG, "max_heatmap_elements": CONFIG["max_heatmap_elements"] * 2}, binding, records)
        with self.assertRaisesRegex(ValueError, "direct_full_successful_pilot_required"):
            verify_promotion(report, identity, review, {**CONFIG, "learning_rate": 0.01}, binding, records)
        with self.assertRaisesRegex(ValueError, "direct_full_review_required"):
            verify_promotion(report, identity, {}, CONFIG, binding, records)
        with self.assertRaisesRegex(ValueError, "direct_full_validation_or_data_changed"):
            verify_promotion(report, identity, review, CONFIG,
                             {**binding, "validation_payloads_sha256": "c" * 64}, records)
        with self.assertRaisesRegex(ValueError, "direct_full_pilot_train_payload_changed"):
            verify_promotion(report, identity, review, CONFIG, binding, {**records, ("seq", 0): "b" * 64})
        incomplete = copy.deepcopy(report)
        incomplete["diagnostic_progress"]["last_completed_unique_train"] = 999
        with self.assertRaisesRegex(ValueError, "direct_full_actual_coverage_or_update_missing"):
            verify_promotion(incomplete, identity, review, CONFIG, binding, records)


if __name__ == "__main__":
    unittest.main()
