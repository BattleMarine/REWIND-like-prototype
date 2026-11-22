"""Verify the real fused-QKV path, compact restore and selection compatibility."""

import io
import json
from pathlib import Path
import tempfile
from unittest import TestCase

import torch
from rewind_m1.training import seed_run
from rewind_v1.data import Clip
from rewind_v1.training import TeacherRun
from scripts import v212_lora
from scripts.v211_trial import measurement, update
from scripts.v2_tuning_cpu_selection import SCENARIOS, plan, summarize


class LoraTests(TestCase):
    def test_fused_attention_adaptation_and_restore(self):
        seed_run(42)
        root = Path(__file__).resolve().parents[1]
        cfg = json.loads((root/"configs/rewind_v12_full_training.json").read_bytes())["teacher"]

        def stage():
            return TeacherRun(cfg, joints=2, edges=[(0, 1)], source="predicted_2d", input_sha256="c"*64)

        model = stage()
        model.model.requires_grad_(False)
        parent = {name: value.clone() for name, value in model.model.state_dict().items()}
        args = (torch.randn(1, 4, 2, 3), torch.randn(1, 4, 28), torch.randn(1, 4, 2, 9),
                torch.tensor([100]), torch.tensor([[True, True, True, False]]))
        noisy, features, camera, _, time = args
        clip = Clip(features, camera, noisy*50, time[..., None].expand(1, 4, 2), time,
                    torch.arange(4, dtype=torch.float64)[None]/25, "predicted_2d", split="train", input_sha256="c"*64)

        def embedding(teacher):
            teacher.model.eval()
            with torch.no_grad():
                return teacher.model(*args)

        parent_xy, parent_frame = embedding(model)
        base_sha = v212_lora.backbone_sha(model)
        before = torch.random.get_rng_state()
        v212_lora.begin_fit(model, 42)
        self.assertTrue(torch.equal(before, torch.random.get_rng_state()))
        weights = v212_lora.adapters(model)
        self.assertEqual(tuple(weights), v212_lora.TRAINABLE)
        self.assertEqual(sum(p.numel() for p in weights.values()), 24576)
        self.assertEqual({id(p) for p in model.optimizer.param_groups[0]["params"]}, {id(p) for p in weights.values()})
        self.assertTrue(torch.equal(parent_xy, embedding(model)[0]))
        self.assertEqual(v212_lora.backbone_sha(model), base_sha)
        self.assertTrue(all(torch.equal(block.qkv.weight, parent[f"temporal.{i}.qkv.weight"])
                            for i, block in enumerate(model.model.temporal)))

        first = update(model, clip, 42)
        self.assertTrue(all(float(p.grad.abs().sum()) > 0 for name, p in weights.items() if name.endswith(".up")))
        self.assertTrue(all(torch.count_nonzero(p.grad) == 0 for name, p in weights.items() if name.endswith(".down")))
        self.assertTrue(all(p.grad is None for p in model.model.parameters() if not p.requires_grad))
        tuned, frame = embedding(model)
        self.assertFalse(torch.equal(parent_xy, tuned))
        self.assertTrue(torch.equal(frame, parent_frame))

        with tempfile.TemporaryDirectory() as scratch:
            path = Path(scratch)/"lora.pt"
            binding = {"method": v212_lora.METHOD, "seed": 42}
            v212_lora.save_state(model, binding, path, first)
            raw = path.read_bytes()
            state = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=True)
            self.assertEqual(set(state["adapters"]), set(v212_lora.TRAINABLE))
            self.assertLess(len(raw), 1024**2)
            expected = update(model, clip, 42)
            restored = stage()
            restored.model.load_state_dict(parent, strict=True)
            v212_lora.begin_fit(restored, 42)
            with self.assertRaisesRegex(ValueError, "checkpoint_identity"):
                v212_lora.load_state(restored, {**binding, "seed": 71}, raw, 1)
            self.assertEqual(v212_lora.load_state(restored, binding, raw, 1), first)
            self.assertTrue(torch.equal(tuned, embedding(restored)[0]))
            self.assertEqual(update(restored, clip, 42), expected)
            self.assertEqual(v212_lora.backbone_sha(restored), base_sha)
            self.assertTrue(all(torch.equal(p, v212_lora.adapters(restored)[n]) for n, p in weights.items()))
            self.assertTrue(any(float(p.grad.abs().sum()) > 0 for name, p in weights.items() if name.endswith(".down")))

    def test_measurement_matches_cpu_trial_plan(self):
        trial_plan = plan({"source": "fixture"}, {}, 250.)
        self.assertEqual(trial_plan["branches"][v212_lora.METHOD], v212_lora.BRANCH)
        group = {role: {"mpjpe_mm": 200.} for role in ("body", "hand_global")}
        metrics = {"per_scenario": {name: {"official": group} for name in SCENARIOS}}
        row = measurement(trial_plan, 71, "a"*64, metrics, v212_lora.METHOD)
        result = summarize(trial_plan, [row], phase="screening")
        self.assertEqual(result["status"], "awaiting_screening_rows")
        self.assertEqual(result["rows"][0]["macro_mm"], 200.)
