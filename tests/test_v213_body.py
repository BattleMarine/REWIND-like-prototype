"""Full-body release parameters and the I/O pause boundary."""

from pathlib import Path
import tempfile
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import patch

import torch
from rewind_v1.v2_budget import Ledger, ResourceLimit
from scripts import v211_trial, v213_body
from scripts.v211_progress import fit, fit_tip
from scripts.v2_gpu_training_storage import MIB
from scripts.v2_tuning_cpu_selection import plan


class Stage:
    device = torch.device("cpu")
    config = {"scheduler_step": 1, "scheduler_gamma": 1.0}

    def __init__(self):
        self.model = torch.nn.Sequential()
        self.model.add_module("final_head", torch.nn.Linear(512, 3))

    def update(self, clip):
        self.optimizer.zero_grad(set_to_none=True)
        loss = (self.model(clip)-torch.randn(1, 3)).square().mean()
        loss.backward()
        self.optimizer.step()
        self.scheduler.step()
        self.step += 1
        return {"loss": float(loss.detach())}


class BodyTests(TestCase):
    def test_plan_and_checkpoint_size(self):
        self.assertEqual(plan({"source": "fixture"}, {}, 250.)["branches"][v213_body.METHOD], v213_body.BRANCH)
        self.assertGreater(v213_body.CHECKPOINT_CAP, v213_body.PARAMETER_VALUES*4*3)
        self.assertLess(v213_body.CHECKPOINT_INTERVAL, v211_trial.UPDATES)

    def test_io_pause_never_starts_unpublishable_update_and_can_resume(self):
        cap = MIB+1
        strategy = SimpleNamespace(**{name: getattr(v211_trial, name) for name in
                                      ("ARTIFACT_PREFIX", "FIT_SCHEMA", "begin_fit", "backbone_sha", "save_state", "load_state")},
                                   METHOD=v213_body.METHOD, CHECKPOINT_CAP=cap, CHECKPOINT_INTERVAL=2)
        cfg = {"seconds_per_invocation": 120, "terminal_reserve_seconds": 10, "cumulative_limits": None,
               "resources": {"read_bytes": 8*MIB, "write_bytes": cap, "rss_bytes": 8*1024*MIB}}
        keys = [42, 71, 103]
        seen = []

        def load(key, split):
            seen.append(key)
            return SimpleNamespace(select_joints=lambda _: torch.full((1, 512), key/103))

        store = SimpleNamespace(load=load)
        binding = {"seed": 42}
        root_stage = Stage()
        parent = {name: value.clone() for name, value in root_stage.model.state_dict().items()}
        with tempfile.TemporaryDirectory() as scratch, patch("scripts.v211_progress.UPDATES", 3):
            root = Path(scratch).resolve()
            ledger = Ledger(root/"usage", binding, cfg)
            for step, stage in ((1, root_stage), (2, Stage()), (3, Stage())):
                stage.model.load_state_dict(parent, strict=True)
                timing = {}
                if step < 3:
                    with self.assertRaisesRegex(ResourceLimit, "write_budget_pause_before_next_durable_unit"):
                        with ledger.invocation("fit"):
                            fit(stage, keys, root, root, binding, store, [], ledger, lambda: None, {}, timing, strategy=strategy)
                else:
                    with ledger.invocation("fit"):
                        fit(stage, keys, root, root, binding, store, [], ledger, lambda: None, {}, timing, strategy=strategy)
                self.assertEqual(timing["resumed_from"], step-1)
                self.assertEqual(fit_tip(root, binding)["step"], step)
                self.assertEqual(seen, keys[:step])
        reference = Stage()
        reference.model.load_state_dict(parent, strict=True)
        v211_trial.begin_fit(reference)
        for key in keys:
            v211_trial.update(reference, torch.full((1, 512), key/103), binding["seed"])
        self.assertTrue(all(torch.equal(reference.model.state_dict()[name], stage.model.state_dict()[name]) for name in parent))
