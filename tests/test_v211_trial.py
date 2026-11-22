"""Head-only optimizer, restart boundary and CPU selection contract for v2.1.1."""

from hashlib import sha256
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import patch

import torch
from rewind_m1.training import rng_state, restore_rng
from scripts.v211_progress import FIT, checkpoint_raw, fit, fit_tip
from scripts.v211_trial import HEAD, backbone_sha, begin_fit, load_state, measurement, order, save_state, update
from scripts.v2_gpu_training_storage import publish_json
from scripts.v2_tuning_cpu_selection import METRICS, SCENARIOS, plan, summarize
from rewind_archive.contracts import digest


class Stage:
    device = torch.device("cpu")
    config = {"scheduler_step": 1, "scheduler_gamma": 1.0}

    def __init__(self):
        self.model = torch.nn.Sequential()
        self.model.add_module("backbone", torch.nn.Linear(4, 512))
        self.model.add_module("final_head", torch.nn.Linear(512, 3))

    def update(self, clip):
        self.optimizer.zero_grad(set_to_none=True)
        loss = (self.model(clip) - torch.randn(2, 3)).square().mean()
        loss.backward()
        self.optimizer.step()
        self.scheduler.step()
        self.step += 1
        return {"loss": float(loss.detach())}


class TrialTests(TestCase):
    def test_trajectory_is_independent_of_ambient_rng_and_only_head_updates(self):
        keys = [("train", n) for n in range(53)]
        first = order(keys, 42)
        self.assertCountEqual(first, keys)
        torch.rand(93)
        self.assertEqual(first, order(keys, 42))
        self.assertNotEqual(first, order(keys, 71))
        stage = Stage()
        initial = {k: v.clone() for k, v in stage.model.state_dict().items()}
        frozen = backbone_sha(stage)
        begin_fit(stage)
        self.assertEqual(tuple(n for n, p in stage.model.named_parameters() if p.requires_grad), HEAD)
        clip = torch.ones(2, 4)
        before = rng_state()["torch_cpu"]
        first_loss = update(stage, clip, 42)
        self.assertTrue(torch.equal(before, rng_state()["torch_cpu"]))
        self.assertEqual(backbone_sha(stage), frozen)
        self.assertTrue(all(not torch.equal(initial[name], stage.model.state_dict()[name]) for name in HEAD))
        with tempfile.TemporaryDirectory() as scratch:
            path = Path(scratch)/"head.pt"
            save_state(stage, {"trial": 42}, path, first_loss)
            expected_rng = rng_state()
            second_loss = update(stage, clip, 42)
            restored = Stage()
            restored.model.load_state_dict(initial)
            begin_fit(restored)
            with self.assertRaisesRegex(ValueError, "checkpoint_identity"):
                load_state(restored, {"trial": 71}, path.read_bytes(), 1)
            self.assertEqual(load_state(restored, {"trial": 42}, path.read_bytes(), 1), first_loss)
            self.assertTrue(torch.equal(rng_state()["torch_cpu"], expected_rng["torch_cpu"]))
            self.assertEqual(update(restored, clip, 42), second_loss)
            self.assertTrue(all(torch.equal(restored.model.state_dict()[n], stage.model.state_dict()[n]) for n in initial))
            self.assertEqual(restored.scheduler.state_dict(), stage.scheduler.state_dict())
            restore_rng(expected_rng)

    def test_receipt_chain_and_checkpoint_readback(self):
        class Budget:
            def reserve(self, *_):
                pass

            def observed(self, *_, **__):
                pass

        with tempfile.TemporaryDirectory() as scratch:
            target = Path(scratch).resolve()
            binding = {"seed": 71}
            payload = target/"head.pt"
            payload.write_bytes(b"model")
            artifact = {"name": payload.name, "bytes": 5, "sha256": sha256(b"model").hexdigest()}
            first = publish_json(target/"fit-00001.json", FIT, {"step": 1, "binding_sha256": digest(binding),
                                 "previous": None, "artifact": artifact}, lambda: None)
            final = publish_json(target/"fit-00050.json", FIT, {"step": 50, "binding_sha256": digest(binding),
                                 "previous": first["content_sha256"], "artifact": artifact}, lambda: None)
            self.assertEqual(fit_tip(target, binding), final)
            self.assertEqual(checkpoint_raw(target, final, Budget()), b"model")
            payload.write_bytes(b"break")
            with self.assertRaisesRegex(ValueError, "fit_sha256"):
                checkpoint_raw(target, final, Budget())
            (target/"fit-00001.json").unlink()
            with self.assertRaisesRegex(ValueError, "fit_history"):
                fit_tip(target, binding)

    def test_time_stop_publishes_completed_step_and_resumes_the_next_clip(self):
        class Budget:
            clock = staticmethod(time.monotonic)

            def __init__(self, checks):
                self.checks = checks

            def check(self):
                self.checks -= 1
                if self.checks < 0:
                    raise TimeoutError("fixture deadline")

            def remaining(self, **_):
                return 20

            def reserve(self, *_):
                pass

            def observed(self, *_, **__):
                pass

        class Store:
            def __init__(self):
                self.keys = []

            def load(self, key, _):
                self.keys.append(key)
                return SimpleNamespace(select_joints=lambda _: torch.full((2, 4), float(key)))

        with tempfile.TemporaryDirectory() as scratch, patch("scripts.v211_progress.UPDATES", 3):
            root = Path(scratch).resolve()
            first = Stage()
            parent = {name: weight.clone() for name, weight in first.model.state_dict().items()}
            keys, binding = [4, 1, 3], {"seed": 42}
            store, timing = Store(), {}
            with self.assertRaisesRegex(TimeoutError, "fixture deadline"):
                fit(first, keys, root, root, binding, store, [], Budget(2), lambda: None, {}, timing)
            self.assertEqual(store.keys, [4, 1])
            self.assertEqual(fit_tip(root, binding)["step"], 2)
            resumed = Stage()
            resumed.model.load_state_dict(parent)
            store = Store()
            end = fit(resumed, keys, root, root, binding, store, [], Budget(1), lambda: None, {}, timing)
            self.assertEqual(timing["resumed_from"], 2)
            self.assertEqual(store.keys, [3])
            self.assertEqual(end["step"], 3)
            self.assertEqual(resumed.checkpoint_sha256, end["artifact"]["sha256"])
            reference = Stage()
            reference.model.load_state_dict(parent)
            begin_fit(reference)
            for key in keys:
                update(reference, torch.full((2, 4), float(key)), 42)
            self.assertTrue(all(torch.equal(reference.model.state_dict()[n], resumed.model.state_dict()[n]) for n in parent))

    def test_measurement_consumed_by_actual_cpu_selector(self):
        trial_plan = plan({"source": "fixture"}, {}, 250.0)
        group = {metric.removesuffix("_mpjpe_mm"): {"mpjpe_mm": 200.0} for metric in METRICS}
        metrics = {"per_scenario": {name: {"official": group} for name in SCENARIOS}}
        row = measurement(trial_plan, 71, "a"*64, metrics)
        result = summarize(trial_plan, [row], phase="screening")
        self.assertEqual(result["status"], "awaiting_screening_rows")
        self.assertEqual(result["rows"][0]["macro_mm"], 200.0)
