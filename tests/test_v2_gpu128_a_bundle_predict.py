"""Offline replay through the v2 model stack with a small synthetic cache."""

import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import zipfile

import numpy as np
import torch

from rewind_v1.run_data import FixtureData
from rewind_v1.run_stages import FrameStage
from rewind_v1.v2_contract import ROOT, SCENARIOS
from scripts.v2_gpu128_a_bundle_predict import compare_predictions, reproduce, select_clips
from scripts.v2_gpu_training_pipeline import FullData


class OfflineReplayTests(unittest.TestCase):
    def test_selection_rejects_training_and_short_clips(self):
        published = {name: [f"{name}/Validation", 7] for name in SCENARIOS}
        keys = [tuple(key) for key in published.values()]
        short = (f"{SCENARIOS[0]}/Short", 19)
        data = SimpleNamespace(width=50, clips={"validation": [*keys, short]},
                               real_frames=lambda k, _: 49 if k == short else 50)
        self.assertEqual([key for _, key in select_clips(data, published)], keys)
        self.assertEqual(select_clips(data, published, SCENARIOS[0]), [(SCENARIOS[0], keys[0])])
        for bad in ((f"{SCENARIOS[0]}/Train", 0), short):
            with self.assertRaisesRegex(ValueError, "complete_official_validation_clip"):
                select_clips(data, published, sequence=bad[0], start=bad[1])

    def test_cpu_replay_cascade_and_comparison_are_deterministic(self):
        scenario = SCENARIOS[0]
        sequence = f"{scenario}/Synthetic"
        key = (sequence, 0)
        spec = FixtureData().spec
        class Budget:
            def check(self):
                pass

        ledger = Budget()

        def frame(k, split):
            self.assertEqual(k[0], sequence)
            self.assertEqual(split, "validation")
            return torch.full((2, 3, 128, 128), (k[1]+1)/51), {
                "xy": torch.full((2, 4, 2), 40.), "xy_valid": torch.ones(2, 4, dtype=torch.bool),
                "target_mm": torch.arange(12, dtype=torch.float32).reshape(4, 3),
                "target_valid": torch.ones(4, dtype=torch.bool),
                "rotation64": torch.eye(3, dtype=torch.float64).repeat(2, 1, 1),
                "translation64": torch.tensor([[0., -80., 0.], [0., 80., 0.]], dtype=torch.float64)}

        records = [{"sequence_id": sequence, "frame_index": n, "split": "validation"} for n in range(50)]
        plan = {"clips": [{"sequence": sequence, "start_frame": 0, "split": "validation", "real_frames": 50}]}
        cache = SimpleNamespace(frame=frame, identity="fixture", metrics={})
        data = FullData({"records": records, "content_sha256": "fixture"}, spec, {"camera": "fixture"}, plan, cache, ledger, "fixture")
        cfg = json.loads((ROOT/"configs/rewind_v12_full_training.json").read_bytes())
        cfg["teacher"].update(hidden=8, heads=2, graph_layers=1, temporal_layers=1, diffusion_steps=20, sampling_steps=2)
        pins = dict(zip(("frontend", "body", "hand"), ("a"*64, "b"*64, "c"*64), strict=True))
        loaded = []

        def freeze(role, stage):
            loaded.append(role)
            stage.model.eval().requires_grad_(False)
            stage.checkpoint_sha256 = pins[role]

        device = torch.device("cpu")
        pipeline = SimpleNamespace(data=data, device=device, cfg=cfg, frontend_size=[128, 128],
                                   _make_frontend=lambda: FrameStage(cfg["frontend"], 4, device), _load_best=freeze,
                                   progress={"best": {role: {"sha256": sha} for role, sha in pins.items()}})
        binding = {"model": pins, "eval_seed": 71, "experiment": {},
                   "clips": {name: [f"{name}/Published", 0] for name in SCENARIOS}}
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for directory in ("first", "replayed"):
                (root/directory).mkdir()
            reference = root/"published.zip"
            bundle = SimpleNamespace(pipeline=pipeline, ledger=ledger, comparison_path=reference, diagnostic={},
                                     comparison={"binding": binding, "content_sha256": "fixture"})
            custom = reproduce(bundle, root/"first", sequence=key[0], start=key[1])
            self.assertEqual(custom["clips"][0]["published_comparison"], {"status": "not_published_clip"})
            with zipfile.ZipFile(reference, "w") as archive:
                archive.write(root/"first"/f"{scenario}.npz", f"{scenario}.npz")
            binding["clips"][scenario] = list(key)
            output = reproduce(bundle, root/"replayed", scenario=scenario)
            self.assertEqual(loaded, ["frontend", "body", "hand"]*2)
            scores = output["clips"][0]["scores_on_this_clip"]
            self.assertEqual(scores["body"]["valid_joints"], 100)
            self.assertEqual(output["clips"][0]["published_comparison"]["predicted_3d_mm"]["max_abs_delta"], 0.0)
            self.assertTrue((root/"replayed"/f"{scenario}.png").is_file())
            self.assertTrue((root/"replayed"/f"{scenario}.gif").is_file())
            from scripts.v2_gpu_training_pipeline import CommonEvaluation
            front = pipeline._make_frontend()
            freeze("frontend", front)
            clip = data.clip(key, "validation", front.model, [128, 128], "e"*64)
            cascade = CommonEvaluation(data, {"train": {key: clip}, "validation": {key: clip}}, cfg, device, 0)
            tuned_body = cascade.make("body")
            tuned_weights = {n: torch.zeros_like(p) if p.is_floating_point() else p for n,p in tuned_body.model.state_dict().items()}
            (root/"tuned").mkdir()
            with patch("scripts.v2_gpu128_a_bundle_predict.load_warmstart", return_value=("body", "d"*64, tuned_weights)):
                changed = reproduce(bundle, root/"tuned", scenario=scenario, warmstart=Path("fixture"))
            self.assertEqual(changed["model_sha256"]["body"], "d"*64)
            self.assertEqual(changed["model_sha256"]["hand"], pins["hand"])
            self.assertEqual(changed["model_selection"]["mode"], "single_stage_experiment")
            self.assertEqual(changed["clips"][0]["published_comparison"]["status"], "delta_from_original_model")
            self.assertEqual(pipeline.progress["best"]["body"]["sha256"], pins["body"])
            with np.load(root/"replayed"/f"{scenario}.npz", allow_pickle=False) as original:
                values = {name: original[name].copy() for name in original.files}
            values["predicted_3d_mm"][0, 0, 0] += 1
            self.assertEqual(compare_predictions(reference, scenario, values)["status"], "outside_tolerance")
            values["valid_3d"][0, 0] = False
            with self.assertRaisesRegex(ValueError, "reference_input_mismatch: valid_3d"):
                compare_predictions(reference, scenario, values)
