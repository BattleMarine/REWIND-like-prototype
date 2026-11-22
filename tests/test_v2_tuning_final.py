"""Verify finalist isolation and the thirteen-row selection workflow."""

from pathlib import Path
import tempfile
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import patch

from rewind_archive.contracts import canonical, digest
from rewind_prepare.temporal_selection import seal
from scripts import v211_run, v211_trial, v2_tuning_final_run as runner
from scripts.v2_tuning_cpu_selection import BODY, INITIAL, SCENARIOS, plan, summarize
from scripts.v2_tuning_final_selection import SCHEMAS, screening_rows, select


class FinalTrialsTests(TestCase):
    def trial_plan(self):
        return plan({"source": "fixture"}, {"body": "b"*64}, 250.)

    def pins(self, method, trial_plan, screening_sha="a"*64):
        return {"trial_plan": trial_plan, "modules": {"worker": method},
                "final": {"method": method, "seeds": list(runner.SEEDS), "shortlist": list(runner.STRATEGIES),
                          "screening_sha256": screening_sha}}

    def test_fixed_runtime_routes_only_additional_seeds_to_separate_roots(self):
        trial_plan = self.trial_plan()
        args = SimpleNamespace()
        for method, strategy in runner.STRATEGIES.items():
            pins, report = self.pins(method, trial_plan), {}
            with patch.object(v211_run, "run") as launch:
                runner.run(args, pins, report)
            launch.assert_called_once_with(args, pins, report, strategy=strategy, destination=runner.DESTINATIONS[method],
                                           run_schema=runner.SCHEMA, seeds=(137, 181))
            self.assertTrue(runner.DESTINATIONS[method].endswith("_final_v1"))
            pins["final"]["seeds"] = list(INITIAL)
            with patch.object(v211_run, "run") as launch, self.assertRaisesRegex(ValueError, "final_trial_scope"):
                runner.run(args, pins, {})
            launch.assert_not_called()

    def test_shared_trials_iterate_only_requested_seed_nodes(self):
        trial_plan = self.trial_plan()
        data = SimpleNamespace(clips={"train": [("train", i) for i in range(v211_trial.UPDATES)]})
        with tempfile.TemporaryDirectory() as scratch:
            root = Path(scratch).resolve()
            for method, strategy in runner.STRATEGIES.items():
                target = root/runner.DESTINATIONS[method].split("/")[1]
                target.mkdir()
                pins = {**self.pins(method, trial_plan), "base": {"source_sha256": "fixture"}, "groups": {}}
                loaded, ordered = [], []
                best = {key: {"sha256": value} for key, value in trial_plan["common"]["selected_model_sha256"].items()}
                pipe = SimpleNamespace(data=data, progress={"best": best}, cfg={"teacher": {"sampling_steps": 10}},
                                       device="cpu", _load_best=lambda role, _: loaded.append(role))
                store = SimpleNamespace(load=lambda *_: None)
                cascade = SimpleNamespace(make=lambda *args: SimpleNamespace(), body_idx=[])

                def record_fit(stage, clips, _, _work, basis, *args, **kwargs):
                    ordered.append((basis["seed"], digest(clips)))
                    stage.checkpoint_sha256 = "a"*64
                    return {"content_sha256": "b"*64}

                with (patch.object(v211_run, "seed_run"), patch.object(v211_run, "restore_features", return_value=(store, 1)),
                      patch.object(v211_run, "CommonEvaluation", return_value=cascade), patch.object(v211_run, "fit", side_effect=record_fit),
                      patch.object(v211_run, "evaluate", return_value={"content_sha256": "c"*64, "measurement": {}})):
                    v211_run.trials(root, root, target, pins, {}, pipe, SimpleNamespace(check=lambda: None), lambda: None,
                                    {}, {"trials": {}}, strategy, seeds=runner.SEEDS)
                self.assertEqual({d.name for d in target.iterdir()}, {"seed-137", "seed-181"})
                self.assertEqual(loaded, ["body", "hand"]*2)
                self.assertEqual(ordered, [(seed, digest(v211_trial.order(data.clips["train"], seed))) for seed in runner.SEEDS])

    def test_final_selector_checks_release_hashes_and_waits_for_target_verdict(self):
        trial_plan, first_rows, extra = self.trial_plan(), [], []

        def write_summary(path, method, seed, score):
            scores = {role: {"mpjpe_mm": score} for role in ("body", "hand_global")}
            metrics = {"per_scenario": {scene: {"official": scores} for scene in SCENARIOS}}
            checkpoint = f"{seed:064x}"
            row = v211_trial.measurement(trial_plan, seed, checkpoint, metrics, method)
            basis = {"method": method, "seed": seed, "plan": trial_plan["content_sha256"],
                     "source": "fixture", "original_model": trial_plan["common"]["selected_model_sha256"],
                     "body_sha256": checkpoint, "modules": {"worker": method}}
            path.write_bytes(canonical(seal({"schema": SCHEMAS[method], "binding": basis, "measurement": row,
                                            "metrics": metrics, "counts": {"clips": 421, "real_frames": 16425}})))
            return row

        with tempfile.TemporaryDirectory() as scratch:
            root = Path(scratch).resolve()
            folders = [root/method for method in BODY]
            for position, (method, folder) in enumerate(zip(BODY, folders, strict=True)):
                folder.mkdir()
                for seed in INITIAL:
                    first_rows.append(write_summary(folder/f"seed-{seed}-summary.json", method, seed, 200.+position))
            report = root/"screening.json"
            report.write_bytes(canonical(summarize(trial_plan, first_rows, phase="screening")))
            rows = screening_rows(trial_plan, folders, report)
            releases = []
            for position, method in enumerate(runner.STRATEGIES):
                release = root/f"release-{method}"
                release.mkdir()
                releases.append(release)
                pins = self.pins(method, trial_plan, digest(rows))
                (release/"receipt.json").write_bytes(canonical({"pins": pins, "resume_inputs_sha256": digest(pins),
                                                               "output_folder": runner.DESTINATIONS[method]}))
                (release/"trial-plan.json").write_bytes(canonical(trial_plan))
                (release/"qualified-screening.json").write_bytes(canonical(rows))
                for seed in runner.SEEDS:
                    path = root/f"seed-{seed}-{method}.json"
                    write_summary(path, method, seed, 199.+position)
                    extra.append(path)
            waiting = select(releases, [])
            self.assertEqual(waiting["status"], "awaiting_final_rows")
            self.assertEqual(len(waiting["missing"]), 4)
            with self.assertRaisesRegex(ValueError, "final_release_summary"):
                select(releases, [folders[0]/"seed-42-summary.json"])
            self.assertEqual(select(releases, extra)["status"], "awaiting_target_metric_verdict")
            result = select(releases, extra, {"v2.1.1": True, "v2.1.2": True})
            self.assertEqual(result["selected_method"], "v2.1.1")
            self.assertEqual(len(result["rows"]), 13)
