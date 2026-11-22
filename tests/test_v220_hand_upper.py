"""Check v2.2 clip pairing, condition provenance and prediction-cache integrity."""

from hashlib import sha256
from pathlib import Path
import tempfile
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import patch

import numpy as np
import torch
from rewind_v1.data import Clip
from scripts import v220_eval, v220_hand, v220_upper
from scripts.v211_trial import SEEDS, order
from scripts.v2_gpu_training_storage import publish_json


class Budget:
    def reserve(self, *_):
        pass

    def observed(self, *_, **__):
        pass


class UpperTests(TestCase):
    def test_exact_balanced_clip_policy_is_order_independent(self):
        keys = [("train", index) for index in range(3097)]
        assigned = {seed: v220_upper.assignments(keys, seed) for seed in SEEDS}
        for seed, selected in assigned.items():
            self.assertEqual(len(selected), 1548)
            self.assertEqual(selected, v220_upper.assignments(order(keys, seed), seed))
            self.assertEqual(v220_hand.source_counts(order(keys, seed), selected, 3097), {"gt": 1549, "predicted": 1548})
        self.assertEqual(len(set(assigned.values())), 3)

    def test_predicted_train_uses_only_body_time_mask_and_checks_provenance(self):
        time = torch.tensor([[True, True, False]])
        existing = torch.zeros(1, 3, 1, 3)
        clip = Clip(torch.zeros(1, 3, 14), torch.zeros(1, 3, 2, 9), existing, torch.zeros(1, 3, 1, dtype=torch.bool),
                    time, torch.arange(3)[None]/25, "predicted_2d", split="train", input_sha256="a"*64)
        key = ("train", 6)
        cache = {("train", key): (existing+1, time)}
        cascade = SimpleNamespace(spec={"joint_names": ["upper"], "upper_joint_names": ["upper"]}, upper_idx=[0])
        body_pin = "b"*64
        gt = v220_upper.condition(clip, key, cascade, cache, set(), body_pin)
        pred = v220_upper.condition(clip, key, cascade, cache, {key}, body_pin)
        self.assertEqual(gt.source, "gt_upper_train")
        self.assertFalse(gt.valid.any())
        self.assertTrue(torch.equal(pred.valid[..., 0], time))
        pred.require("predicted_frozen_body", {"checkpoint_sha256": body_pin})
        with self.assertRaises(ValueError):
            pred.require("predicted_frozen_body", {"checkpoint_sha256": "c"*64})
        with self.assertRaisesRegex(ValueError, "clip_time_mask"):
            v220_upper.cached_body({("train", key): (existing, ~time)}, "train", key, clip)
        teacher = object.__new__(v220_hand.HandTeacher)
        teacher.upper_policy = v220_upper.CONTROL
        with self.assertRaisesRegex(ValueError, "disallowed_in_control"):
            teacher._condition(pred, training=True)
        teacher.upper_policy = v220_upper.MIXED
        with patch.object(v220_hand.TeacherRun, "_condition", return_value="bound") as parent:
            self.assertEqual(teacher._condition(pred, training=True), "bound")
        parent.assert_called_once_with(pred, training=False)

    def test_group_readback_rejects_payload_and_order_drift(self):
        with tempfile.TemporaryDirectory() as scratch:
            root = Path(scratch).resolve()
            path, marker = root/"body.npz", root/"body.json"
            binding = {"parent": "fixed"}
            entries = [("train", ("seq", 0)), ("validation", ("other", 50))]
            coords = np.zeros((2, 3, 2, 3), dtype=np.float32)
            coords[:, 0] = 2
            mask = np.asarray([[True, True, False], [True, False, False]])
            np.savez_compressed(path, coordinates_mm=coords, time_valid=mask)
            raw = path.read_bytes()
            record = {"binding": binding, "keys": [[split, list(key)] for split, key in entries],
                      "artifact": {"name": path.name, "bytes": len(raw), "sha256": sha256(raw).hexdigest()}}
            publish_json(marker, v220_upper.GROUP, record, lambda: None)
            data = SimpleNamespace(width=3, real_frames=lambda _, split: 2 if split == "train" else 1)
            cache, _ = v220_upper.read_group(root, marker, binding, entries, data, 2, Budget())
            self.assertEqual(set(cache), set(entries))
            with self.assertRaisesRegex(ValueError, "group_binding"):
                v220_upper.read_group(root, marker, binding, entries[::-1], data, 2, Budget())
            path.write_bytes(raw[:-1] + bytes([raw[-1]^1]))
            with self.assertRaisesRegex(ValueError, "group_sha256"):
                v220_upper.read_group(root, marker, binding, entries, data, 2, Budget())

    def test_paired_comparison_reuses_one_body_and_reports_hand_metrics(self):
        def summary(score, body=100.):
            values = {"body": {"mpjpe_mm": body}, "hand_local": {"mpjpe_mm": score},
                      "hand_global": {"mpjpe_mm": 2*score}}
            return {"metrics": {"per_scenario": {"one": {"official": values}}, "macro_score_mm": (body+2*score)/2},
                    "content_sha256": "a"*64, "hand_checkpoint_sha256": "b"*64}
        trials = {(policy, seed): summary(80. if policy == v220_upper.CONTROL else 70.)
                  for policy in v220_upper.POLICIES for seed in SEEDS}
        with tempfile.TemporaryDirectory() as scratch:
            path = Path(scratch).resolve()/"comparison.json"
            result = v220_eval.paired(path, trials, "c"*64, lambda: None)
            self.assertEqual(result["mean_delta_mm"]["hand_local_macro_mm"], -10.)
            self.assertEqual(result["mean_delta_mm"]["hand_global_macro_mm"], -20.)
            self.assertEqual(v220_eval.paired(path, trials, "c"*64, lambda: None), result)
            trials[v220_upper.MIXED, SEEDS[0]] = summary(70., body=101.)
            with self.assertRaisesRegex(ValueError, "frozen_body_mismatch"):
                v220_eval.paired(path, trials, "c"*64, lambda: None)
