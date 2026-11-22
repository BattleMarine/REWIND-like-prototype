"""The v2.2 extension may only train new seeds and only predict missing inputs."""

from pathlib import Path
import tempfile
import time
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import patch

import torch
from scripts import v220_final_run, v220_final_upper
from scripts.v220_upper import CONTROL, MIXED
from scripts.v211_run import TrialLedger
from scripts.run_v2_gpu_training import training_config
from scripts.v2_gpu_training_storage import GIB


class Budget:
    current = {"read_bytes_reserved": 0, "write_bytes_reserved": 0}
    cfg = {"resources": {"read_bytes": 1024**3, "write_bytes": 1024**3}}

    def check(self):
        pass


class FinalHandTests(TestCase):
    def test_final_runtime_uses_original_comparison_and_two_new_seeds(self):
        seeds = v220_final_upper.SEEDS
        pins = {"final_seeds": list(seeds), "trial_plan": {"initial_seeds": [42,71,103], "conditional_additional_seeds": list(seeds)},
                "initial": {"comparison": {"sha256": "a"*64,"bytes": 99,"content_sha256": "b"*64}}, "groups": {}}
        pins["io_budget"] = v220_final_run.io_budget(pins, 435178624)
        result = {"content_sha256": "b"*64, "pairs": [{"seed": seed} for seed in (42,71,103)]}
        args = SimpleNamespace(project=Path(__file__).resolve().parent)
        with patch.object(v220_final_run,"original_json",return_value=result), patch.object(v220_final_run.v211_run,"run") as launch:
            v220_final_run.run(args,pins,{})
            self.assertEqual(launch.call_args.kwargs["seeds"], (137,181))
            self.assertEqual(launch.call_args.kwargs["destination"], v220_final_run.DESTINATION)
            self.assertEqual(launch.call_args.kwargs["io_limits"], {"read_bytes":16*GIB,"write_bytes":8*GIB})
            self.assertIs(launch.call_args.kwargs["trial_runner"].keywords["original"],result)
            with self.assertRaisesRegex(ValueError,"additional_seeds_only"):
                v220_final_run.run(args,{**pins,"final_seeds":[42,71,103]}, {})
            with self.assertRaisesRegex(ValueError,"io_budget_pin"):
                v220_final_run.run(args,{**pins,"io_budget":{**pins["io_budget"],"limits":{"read_bytes":1,"write_bytes":1}}}, {})
            self.assertEqual(launch.call_count,1)

    def test_one_ledger_invocation_covers_all_budgeted_components(self):
        pins = {"groups": {"fixture": {"archive": {"bytes":778972869}}}}
        budget = v220_final_run.io_budget(pins,435178624)
        cfg = training_config(42)
        cfg["resources"].update(budget["limits"])
        binding = {"io_limits":budget["limits"]}
        with tempfile.TemporaryDirectory() as scratch:
            ledger = TrialLedger(Path(scratch),binding,cfg,lambda: None)
            with ledger.session(time.monotonic()):
                for kind, parts in budget["bounded_components_bytes"].items():
                    for size in parts.values():
                        ledger.reserve(kind,size)
            row, = ledger.snapshot()["invocations"]
            self.assertEqual(row["status"],"settled")
            self.assertEqual(row["wall_seconds_reserved"],28800)
            for kind, amount in budget["bounded_total_bytes"].items():
                self.assertEqual(row[f"{kind}_bytes_reserved"],amount)
                self.assertGreater(budget["headroom_bytes"][kind],0)

    def test_delta_extension_shares_initial_train_and_validation_predictions(self):
        old, new = ("train",0),("train",50)
        validation = ("val",0)
        orders = {137:{old,new},181:{old}}
        data = SimpleNamespace(width=50,clips={"validation":[validation]},real_frames=lambda *_: 1)
        mask = torch.arange(50)[None]<1
        value = torch.zeros(1,50,1,3)
        prior = {("train",old):(value,mask),("validation",validation):(value,mask)}
        supplied = []

        def load(key, split):
            supplied.append((split,key))
            return SimpleNamespace(time_valid=mask,select_joints=lambda _: None)

        store = SimpleNamespace(data=data, load=load)
        cascade = SimpleNamespace(body_names=["fixture"],body_idx=[0],_noise=lambda *_: None)
        body = SimpleNamespace(model=SimpleNamespace(training=False,parameters=lambda:[]),checkpoint_sha256="c"*64,
                               predict=lambda *_: value)
        mocked = lambda _, group: [("train",old),("train",new),("validation",validation)] if group==0 else []
        read = {("train",new):(value,mask)}, {"content_sha256":"d"*64}
        with tempfile.TemporaryDirectory() as scratch:
            root = Path(scratch).resolve()
            with (patch.object(v220_final_upper,"group_keys",side_effect=mocked),
                  patch.object(v220_final_upper,"read_group",return_value=read),
                  patch.object(v220_final_upper,"publish_artifact",return_value={"sha256":"e"*64})):
                timing = {}
                cache, index = v220_final_upper.extend(root,root,prior,store,cascade,body,orders,{},Budget(),lambda:None,{},timing)
                self.assertEqual(supplied,[("train",new)])
                self.assertEqual(index["new_train_clips"],1)
                self.assertIs(cache["validation",validation][0], value)
                _, index2 = v220_final_upper.extend(root,root,prior,store,cascade,body,orders,{},Budget(),lambda:None,{},timing)
                self.assertEqual(supplied,[("train",new)])
                self.assertEqual(index2,index)
                self.assertEqual(timing["reused_delta_groups"],1)

    def test_five_seed_aggregation_keeps_original_pairs(self):
        def row(seed, delta):
            a = {"hand_global_macro_mm": 200., "body_hand_global_macro_mm": 175.}
            b = {"hand_global_macro_mm": 200.+delta, "body_hand_global_macro_mm": 175.+delta/2}
            return {"seed":seed,"control":a,"mixed":b,"delta_mm":{"hand_global_macro_mm":delta},
                    "summary_sha256":{CONTROL:"a"*64,MIXED:"b"*64}}
        original = {"pairs":[row(seed,-2) for seed in (42,71,103)],"content_sha256":"c"*64,"predicted_body_cache_sha256":"d"*64}
        added = [row(seed,-7) for seed in (137,181)]
        with tempfile.TemporaryDirectory() as scratch:
            marker = Path(scratch).resolve()/"comparison.json"
            index = {"content_sha256":"e"*64}
            result = v220_final_run.combined(marker,original,added,index,lambda:None)
            self.assertEqual(result["pairs"][:3], original["pairs"])
            self.assertEqual(result["mean_delta_mm"]["hand_global_macro_mm"],-4.)
            self.assertEqual(v220_final_run.combined(marker,original,added,index,lambda:None),result)
