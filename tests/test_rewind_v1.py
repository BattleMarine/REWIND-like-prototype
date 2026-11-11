import hashlib
from dataclasses import replace
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import torch
from torch.nn import functional as F

from rewind_m1.frontend import heatmap_loss
from rewind_m1.training import seed_run
from rewind_m2.streaming import AnchorSampler
from rewind_m3.cascade import UpperCondition
from rewind_prepare.temporal_selection import SCHEMA, seal
from rewind_v1.attention import Equation4Block, kernel_attention
from rewind_v1.data import (
    Clip, EVIDENCE_SCHEMA, REQUIRED_EVIDENCE, _checked_evidence, attach_predicted_hands, gt_upper_train, hand_clip,
    load_clip,
)
from rewind_v1.device import require_a100, require_cuda_training
from rewind_v1.frontend import PaperHeatmapFrontend, frontend_fingerprint, predicted_features
from rewind_v1.teacher import motion_losses
from rewind_v1.training import TeacherRun
import test_extracted_stereo


class RewindV1Tests(unittest.TestCase):
    def setUp(self):
        seed_run(71)

    def test_equation_four_reference_mask_and_gradient(self):
        layer = Equation4Block(4, 1, 2, 0.)
        q, k, v = [torch.randn(2, 1, 5, 4, requires_grad=True) for _ in range(3)]
        valid = torch.tensor([[False, True, True, False, True], [False] * 5])
        pos = torch.arange(5)
        got = kernel_attention(q, k, v, valid, layer.rotary, pos, 2)
        actual_q, actual_k = F.elu(q) + 1, F.elu(k) + 1
        rotated_q, rotated_k = layer.rotary(actual_q, pos), layer.rotary(actual_k, pos)
        expected = torch.zeros_like(got)
        for j in (1, 2, 4):
            top = torch.zeros(4)
            bottom = torch.zeros(())
            for i in range(max(0, j - 2), j + 1):
                if valid[0, i]:
                    top += rotated_q[0, 0, j].dot(rotated_k[0, 0, i]) * v[0, 0, i]
                    bottom += actual_q[0, 0, j].dot(actual_k[0, 0, i])
            expected[0, 0, j] = top / bottom
        torch.testing.assert_close(got, expected)
        shifted = kernel_attention(q, k, v, valid, layer.rotary, pos + 29, 2)
        torch.testing.assert_close(got, shifted, atol=1e-5, rtol=1e-5)
        got.square().sum().backward()
        self.assertTrue(all(torch.isfinite(value.grad).all() for value in (q, k, v)))
        tiny = torch.full_like(q, -100., requires_grad=True)
        padded = torch.where(valid[:, None, :, None], v, float("nan"))
        tiny_result = kernel_attention(tiny, tiny, padded, valid, layer.rotary, pos, 2)
        self.assertTrue(torch.isfinite(tiny_result).all())
        self.assertEqual(tiny_result.abs().sum().item(), 0.)
        tiny_result.sum().backward()
        self.assertTrue(torch.isfinite(tiny.grad).all())
        almost_zero = torch.full_like(q, -12.)
        small = kernel_attention(almost_zero, almost_zero, v, valid, layer.rotary, pos, 0)
        positive = (F.elu(almost_zero[0, 0, 1]) + 1).square().sum()
        self.assertTrue(0 < positive < 1e-6)
        torch.testing.assert_close(small[0, 0, 1], v[0, 0, 1] * (positive / 1e-6))

    def test_frontend_observation_shape_training_and_frame_independence(self):
        model = PaperHeatmapFrontend(2)
        images = torch.rand(2, 2, 3, 32, 48)
        logits, xy, confidence = model(images)
        self.assertEqual(tuple(logits.shape), (2, 2, 2, 16, 24))
        self.assertEqual(tuple(xy.shape), (2, 2, 2, 2))
        self.assertEqual(tuple(confidence.shape), (2, 2, 2))
        target = torch.full_like(xy, 12.)
        mask = torch.ones_like(confidence, dtype=torch.bool)
        loss, _ = heatmap_loss(logits, target, mask, (48, 32), loss_kind="gaussian_cross_entropy")
        loss.backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))
        with self.assertRaisesRegex(ValueError, "eval_mode"):
            predicted_features(model, images)
        model.eval()
        features = predicted_features(model, images)
        self.assertEqual(tuple(features.shape), (2, 28))
        torch.testing.assert_close(features[:1], predicted_features(model, images[:1]), atol=1e-6, rtol=1e-6)

    def test_velocity_and_foot_masks_require_evidence(self):
        pred = torch.tensor([[[[0., 0., 0.]], [[1., 0., 0.]], [[float("nan"), 0., 0.]]]], requires_grad=True)
        mask = torch.tensor([[[True], [True], [False]]])
        time_valid = mask[..., 0]
        target = torch.zeros_like(pred)
        time = torch.tensor([[0., .5, float("nan")]], dtype=torch.float64)
        args = dict(scale_mm=1., frame_weight=1., velocity_weight=1., time_seconds=time)
        with self.assertRaisesRegex(ValueError, "world_and_contact"):
            motion_losses(pred, pred, target, mask, time_valid, foot_weight=1., **args)
        loss, terms = motion_losses(pred, pred, target, mask, time_valid, foot_weight=1.,
                                    world_prediction_mm=pred, foot_contact=mask, **args)
        self.assertAlmostEqual(float(terms["velocity"].detach()), 4 / 3)
        self.assertAlmostEqual(float(terms["foot"].detach()), 4 / 3)
        loss.backward()
        self.assertTrue(torch.isfinite(pred.grad).all())

    def test_body_hand_ddim_causality_and_resume(self):
        config = dict(hidden=8, heads=2, graph_layers=1, temporal_layers=2, attention_past_frames=4,
                      diffusion_steps=20, dropout=.1, beta_start=.0001, beta_end=.02, sampling_steps=3,
                      learning_rate=.001, scheduler_step=10, scheduler_gamma=.8, target_scale_mm=100.,
                      frame_loss_weight=1., velocity_loss_weight=.01)
        spec = dict(joints=2, edges=[(0, 1)], source="predicted_2d", input_sha256="a" * 64)
        valid = torch.ones(1, 50, 2, dtype=torch.bool)
        time = torch.arange(50).double()[None] / 30
        features, camera = torch.rand(1, 50, 28), torch.rand(1, 50, 2, 9)
        target = torch.rand(1, 50, 2, 3) * 100
        clip = Clip(features, camera, target, valid, valid[..., 0], time, "predicted_2d",
                    input_sha256=spec["input_sha256"])
        body = TeacherRun(config, **spec)
        with self.assertRaisesRegex(ValueError, "requires_train_split"):
            body.update(replace(clip, split="validation"))
        body.update(clip)
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as directory:
            path = Path(directory) / "body.pt"
            body.save(path)
            pin = hashlib.sha256(path.read_bytes()).hexdigest()
            body.update(clip)
            resumed = TeacherRun(config, **spec)
            resumed.resume(path, pin)
            resumed.update(clip)
            self.assertEqual(resumed.step, body.step)
            for name, value in body.model.state_dict().items():
                torch.testing.assert_close(resumed.model.state_dict()[name], value, atol=0, rtol=0)
            resumed.resume(path, pin)
            resumed.model.eval().requires_grad_(False)
            noise = torch.rand_like(target)
            stream = AnchorSampler(resumed.model, resumed.diffusion, 50, config["sampling_steps"])
            offline = resumed.predict(clip, noise)[0] / config["target_scale_mm"]
            for index in range(3):
                streamed = stream.push("clip", index, features[0, index], camera[0, index], noise[0, index])
                torch.testing.assert_close(streamed, offline[index], atol=1e-6, rtol=1e-5)
            gap = stream.push("clip", 9, features[0, 9], camera[0, 9], noise[0, 9])
            reset = stream.push("other", 9, features[0, 9], camera[0, 9], noise[0, 9])
            torch.testing.assert_close(gap, reset)
            self.assertEqual(len(stream.history), 1)
            mutated_features, mutated_camera, mutated_noise = features.clone(), camera.clone(), noise.clone()
            mutated_features[:, 25:] = .9
            mutated_camera[:, 25:] = .2
            mutated_noise[:, 25:] = .8
            changed = replace(clip, features=mutated_features, camera=mutated_camera, target_mm=target + 10000)
            upper = resumed.frozen_upper(clip, noise, [0])
            changed_upper = resumed.frozen_upper(changed, mutated_noise, [0])
            torch.testing.assert_close(upper.values_mm[:, :25], changed_upper.values_mm[:, :25])
            hand = TeacherRun(config, **spec, upper_joints=1, body_checkpoint_sha256=pin)
            gt = UpperCondition(target[:, :, :1], valid[:, :, :1], "gt_upper_train", None)
            hand.update(clip, upper=gt)
            hand.model.eval()
            with self.assertRaises(ValueError):
                hand.predict(clip, noise, upper=gt)
            actual = hand.predict(clip, noise, upper=upper)
            changed_prediction = hand.predict(changed, mutated_noise, upper=changed_upper)
            torch.testing.assert_close(actual[:, :25], changed_prediction[:, :25])

    def test_evidence_gated_raw_clip_adapter(self):
        fixture = test_extracted_stereo.ExtractedStereoTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        sequence = fixture.train
        spec_path = fixture.fixture.specpath
        plan = seal({"schema": SCHEMA, "train_clip_starts": [[sequence, 0]],
                     "records": [{"key": [sequence, index], "split": "train"} for index in (0, 1)],
                     "spec_sha256": hashlib.sha256(spec_path.read_bytes()).hexdigest()})
        camera = {side: {"rotation_matrix": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                         "translation": [offset, 0, 0]}
                  for side, offset in (("left", -20), ("right", 20))}
        evidence = seal({"schema": EVIDENCE_SCHEMA, "selection_sha256": plan["content_sha256"],
                         "review_artifact_sha256": dict.fromkeys(REQUIRED_EVIDENCE, "a" * 64),
                         "max_frame_delta_seconds": .05,
                         "frames": {sequence: {str(index): {"time_seconds": index / 30, "camera_pose": camera}
                                               for index in (0, 1)}}})
        cache = Mock()
        cache.manifest = {"selection_sha256": plan["content_sha256"], "content_sha256": "b" * 64}

        def read_pair(seq, index, *, split):
            self.assertEqual(split, "train")
            raw = {"pose": (fixture.fixture.raw / seq / f"json/frame_{index}.json").read_bytes()}
            for side in ("left", "right"):
                raw[side] = (fixture.rgb / seq / f"fisheye_final_image/camera_{side}/final_{index}.png").read_bytes()
            return raw

        cache.read_pair.side_effect = read_pair
        args = dict(split="train", selection_sha256=plan["content_sha256"],
                    evidence=evidence, evidence_sha256=evidence["content_sha256"], spec_path=spec_path, scale_mm=100.)
        with patch("rewind_v1.data.WINDOW_FRAMES", 2):
            with self.assertRaisesRegex(ValueError, "training_blocked"):
                load_clip(cache, plan, sequence, 0, **{**args, "evidence": None}, source="gt_2d_diagnostic")
            cache.read_pair.assert_not_called()
            gt = load_clip(cache, plan, sequence, 0, **args, source="gt_2d_diagnostic")
            frontend = PaperHeatmapFrontend(62).eval()
            prediction = load_clip(cache, plan, sequence, 0, **args, source="predicted_2d", frontend=frontend,
                                   frontend_size=[32, 32], frontend_sha256=frontend_fingerprint(frontend))
        self.assertEqual(tuple(gt.features.shape), (1, 2, 62 * 14))
        self.assertEqual(tuple(gt.camera.shape), (1, 2, 2, 9))
        self.assertEqual(tuple(prediction.target_valid.shape), (1, 2, 62))
        torch.testing.assert_close(gt.target_mm, prediction.target_mm)
        self.assertNotEqual(gt.input_sha256, prediction.input_sha256)
        self.assertFalse(torch.equal(gt.features, prediction.features))

    def test_hand_local_targets_and_predicted_wrist_attachment(self):
        spec = {"joint_names": ["left_wrist", "right_wrist", "left_tip", "right_tip"],
                "body_joint_names": ["left_wrist", "right_wrist"], "upper_joint_names": ["left_wrist", "right_wrist"],
                "hands": {"left": ["left_wrist", "left_tip"], "right": ["right_wrist", "right_tip"]},
                "wrists": {"left": "left_wrist", "right": "right_wrist"}}
        target = torch.tensor([[[[1., 2., 3.], [4., 5., 6.], [5., 7., 9.], [8., 9., 10.]]]])
        valid = torch.tensor([[[True, False, True, True]]])
        clip = Clip(torch.zeros(1, 1, 56), torch.zeros(1, 1, 2, 9), target, valid,
                    torch.ones(1, 1, dtype=torch.bool), torch.zeros(1, 1), "predicted_2d")
        local = hand_clip(clip, spec)
        self.assertEqual(local.target_valid.tolist(), [[[True, False]]])
        self.assertEqual(local.target_mm.tolist(), [[[[4., 5., 6.], [0., 0., 0.]]]])
        self.assertTrue(clip.target_valid[0, 0, 3])
        self.assertEqual(gt_upper_train(clip, spec).source, "gt_upper_train")
        with self.assertRaisesRegex(ValueError, "requires_train_split"):
            gt_upper_train(replace(clip, split="validation"), spec)
        predicted = attach_predicted_hands(target[..., :2, :] + 10, local.target_mm, spec)
        self.assertEqual(predicted.tolist(), [[[[15., 17., 19.], [14., 15., 16.]]]])

    def test_cuda_training_and_a100_benchmark_devices(self):
        reports = []
        with patch("torch.cuda.is_available", return_value=True), patch("torch.cuda.current_device", return_value=1):
            for name in ("Tesla T4", "NVIDIA GeForce RTX 4090", "NVIDIA L4"):
                with self.subTest(name=name), patch("torch.cuda.get_device_properties", return_value=SimpleNamespace(name=name, total_memory=16)) as props:
                    self.assertEqual(require_cuda_training(report=reports.append),
                                     {"device": "cuda:1", "gpu_name": name, "vram_bytes": 16})
                    props.assert_called_with(1)
                    self.assertEqual(require_cuda_training("cuda:0", report=reports.append)["device"], "cuda:0")
                    props.assert_called_with(0)
                    with self.assertRaisesRegex(RuntimeError, "a100_required"):
                        require_a100(report=reports.append)
            with patch("torch.cuda.get_device_properties", return_value=SimpleNamespace(name="NVIDIA A100-SXM4-40GB", total_memory=40)):
                self.assertEqual(require_a100(report=reports.append)["device"], "cuda:1")
        self.assertIn("Tesla T4", reports[0])
        self.assertIn("vram_bytes", reports[0])
        with patch("torch.cuda.is_available", return_value=False), patch("torch.cuda.get_device_properties") as props:
            with self.assertRaisesRegex(RuntimeError, "cuda_required"):
                require_cuda_training(report=reports.append)
            props.assert_not_called()
        self.assertIn('"cuda_available": false', reports[-1])
        with self.assertRaisesRegex(ValueError, "training_requires_cuda"):
            require_cuda_training("cpu")

    def test_missing_real_evidence(self):
        with self.assertRaisesRegex(ValueError, "timestamp_and_stereo_sync"):
            _checked_evidence({}, None, None)
