"""Local v2.4 regression for paired controls and bounded state retention."""

import hashlib
from pathlib import Path
import tempfile
from types import SimpleNamespace
from unittest import TestCase

import torch

from rewind_m2.diffusion import DiffusionSchedule
from scripts import v24_body as pilot
from scripts.v24_colab import stream_v24_state


class V24PilotTests(TestCase):
    def test_x0_v_numerical_contract_and_frame_passthrough(self):
        proof = pilot.numerical()
        self.assertTrue(proof["frame_x0_passthrough"] and proof["time_mask_preserved"])
        self.assertEqual(proof["sampler_timesteps"][0], 1000)
        self.assertEqual(len(proof["sampler_timesteps"]), 10)
        self.assertLess(proof["DDIM10_output_max_abs"], 1e-12)
        self.assertLess(proof["masked_position_loss_abs_difference"], 1e-12)

    def test_both_arms_use_identical_head_and_per_step_noise_after_rng_use(self):
        weights, bias = torch.zeros((3, 512)), torch.zeros(3)
        first = pilot.common_head(weights, bias)
        schedule = DiffusionSchedule(1000, .0001, .02)
        k, eps, first_sha = pilot.draw(schedule, 0, (1, 50, 22, 3))
        torch.rand(200)
        second = pilot.common_head(weights, bias)
        kk, ee, second_sha = pilot.draw(schedule, 0, eps.shape)
        self.assertTrue(all(torch.equal(value, second[name]) for name, value in first.items()))
        self.assertTrue(torch.equal(k, kk) and torch.equal(eps, ee))
        self.assertEqual(first_sha, second_sha)
        self.assertNotEqual(first_sha, pilot.draw(schedule, 1, eps.shape)[2])

    def test_loss_rejects_training_on_validation(self):
        with self.assertRaisesRegex(ValueError, "scope"):
            pilot.loss(None, "x0", SimpleNamespace(split="validation"), None, None)

    def test_state_is_published_only_after_verified_source_readback(self):
        raw = b"checkpoint" * 100
        digest = hashlib.sha256(raw).hexdigest()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, target = root / "worker.pt", root / "retained.pt"
            source.write_bytes(raw)
            with self.assertRaisesRegex(ValueError, "SHA"):
                stream_v24_state(source, target, {"bytes": len(raw), "sha256": "0" * 64}, len(raw), lambda: None)
            self.assertFalse(target.exists())
            self.assertFalse(target.with_suffix(".partial").exists())
            pin = {"bytes": len(raw), "sha256": digest}
            self.assertEqual(stream_v24_state(source, target, pin, len(raw), lambda: None), pin)
            self.assertEqual(target.read_bytes(), raw)
