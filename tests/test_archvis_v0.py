"""Regression coverage for fixed v0 preprocessing, recovery and inference."""

import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import torch

from rewind_archive.contracts import canonical, digest
from rewind_data.audit import read_json
from rewind_m1 import archvis_v0 as v0
from rewind_m1.archvis_v0_model import ContextHeatmapNet
from rewind_m1.extracted_rgb_training import DirectRGBTrainer
from rewind_m1.rgb_frontend_artifacts import scale_xy
from rewind_m1.training import seed_run
import test_extracted_stereo


class ArchVisV0Tests(unittest.TestCase):
    def setUp(self):
        fixture = test_extracted_stereo.ExtractedStereoTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        self.base = fixture.base
        manifest = fixture.prepare("v0_fixture", train_limit=3)
        self.reader = fixture.loader("v0_fixture", manifest, "train")
        pins = patch.multiple(v0, MANIFEST=manifest["content_sha256"],
                              VALIDATION=manifest["selection"]["validation_keys_sha256"],
                              COUNTS={"train": 3, "validation": 1}, EVALUATE=(0, 1, 2),
                              CHECKPOINT_EVERY=1)
        pins.start()
        self.addCleanup(pins.stop)

    def test_cache_is_exact_and_fails_closed_after_mutation(self):
        baseline = DirectRGBTrainer(v0.CONFIG, self.reader)
        directory = self.base / "cache"
        directory.mkdir()
        cache = v0._LocalBatchCache(directory)
        trainer = v0.ContextTrainer(self.reader, "cpu", None, cache)
        trainer.fill_cache()
        uncached = baseline._batch(0, "train", [2, 0])
        cached = trainer._batch(0, "train", [2, 0])
        for key in ("images", "gt_xy", "valid", "native_gt_xy"):
            self.assertTrue(torch.equal(uncached[key], cached[key]), key)
        self.assertEqual(uncached["keys"], cached["keys"])
        self.assertLess(cache.write_bytes * 3236 / 4, v0.CACHE_LIMIT)
        sample = directory / "0-2.pt"
        sample.write_bytes(sample.read_bytes() + b"x")
        with self.assertRaisesRegex(ValueError, "archvis_v0_cache_hash"):
            trainer._batch(0, "train", [2])

    def test_resume_repeats_only_unfinished_eval_and_matches_inference(self):
        reference = self.base / "reference"
        reference.mkdir()
        full = v0.run(self.reader, reference / "v0", "cpu")
        self.assertEqual(full["status"], "completed")

        interrupted = self.base / "interrupted"
        interrupted.mkdir()
        plan = {"schema": "archvis_1000_v0_plan_v1"}
        plan_pin = plan["content_sha256"] = digest(plan)
        (interrupted / "v0_plan.json").write_bytes(canonical(plan))
        original_evaluate = v0.ContextTrainer.evaluate

        def stop_during_epoch_one(trainer, split):
            if trainer.sampler.epoch == 1:
                raise TimeoutError("test evaluation interrupted")
            return original_evaluate(trainer, split)

        with patch.object(v0.ContextTrainer, "evaluate", stop_during_epoch_one):
            partial = v0.run(self.reader, interrupted / "v0", "cpu")
        self.assertEqual(partial["status"], "bounded_stop")
        self.assertEqual(partial["durable"]["step"], 2)
        self.assertEqual([item["epoch"] for item in partial["durable"]["journal"]["evaluations"]], [0])

        archive, receipt = v0.recover_run(interrupted)
        returned = self.base / "returned"
        with zipfile.ZipFile(archive) as bundle:
            self.assertIsNone(bundle.testzip())
            bundle.extractall(returned)
        fresh = self.base / "fresh"
        fresh.mkdir()
        pin = v0.restore_run(returned, fresh / "v0", plan_sha256=plan_pin)
        self.assertEqual(pin, receipt["files"]["last.pt"]["sha256"])
        resumed = v0.run(self.reader, fresh / "v0", "cpu", resume_sha256=pin)
        self.assertEqual(resumed["status"], "completed")
        self.assertEqual(resumed["durable"]["step"], full["durable"]["step"])
        self.assertEqual([item["epoch"] for item in resumed["durable"]["journal"]["evaluations"]], [0, 1, 2])
        state = v0.read_run_checkpoint(fresh / "v0/last.pt", resumed["durable"]["sha256"])
        expected = v0.read_run_checkpoint(reference / "v0/last.pt", full["durable"]["sha256"])
        for key, value in state["model"].items():
            self.assertTrue(torch.equal(value, expected["model"][key]), key)
        self.assertEqual(state["cursor"], expected["cursor"])
        self.assertEqual(state["scheduler"], expected["scheduler"])
        self.assertEqual(state["diagnostic_progress"], expected["diagnostic_progress"])
        self.assertTrue(torch.equal(state["rng"]["torch_cpu"], expected["rng"]["torch_cpu"]))
        self.assertGreater(state["diagnostic_progress"]["nonzero_gradient_updates"], 0)

        recovery, receipt = v0.recover_run(fresh)
        with zipfile.ZipFile(recovery) as bundle, tempfile.TemporaryDirectory(dir=self.base) as directory:
            self.assertIsNone(bundle.testzip())
            best = Path(directory) / "best.pt"
            best.write_bytes(bundle.read("best.pt"))
            model, contract = v0.load_context_checkpoint(best, receipt["files"]["best.pt"]["sha256"])
            with zipfile.ZipFile(io.BytesIO(bundle.read("predictions.zip"))) as demo:
                self.assertIsNone(demo.testzip())
                self.assertEqual(len(demo.namelist()), 5)
                predictions = read_json(demo.read("predictions.json"))
                self.assertEqual(predictions["joint_names"], contract["joint_names"])
                sample = DirectRGBTrainer(v0.CONFIG, self.reader)._batch(1, "validation", [0])
                with torch.no_grad():
                    native = scale_xy(model(sample["images"])[1], [64, 64], [1024, 1024])
                self.assertEqual(predictions["records"][0]["native_prediction_xy"], native[0].tolist())
            self.assertLess(6 * best.stat().st_size + 5 * v0.DEMO_LIMIT + 8 * v0.MIB, v0.ARTIFACT_LIMIT)

    def test_single_location_uses_distant_image_context(self):
        seed_run(71)
        model = ContextHeatmapNet(62)
        image = torch.rand((1, 2, 3, 64, 64), requires_grad=True)
        logits, xy, confidence = model(image)
        self.assertEqual(tuple(logits.shape), (1, 2, 62, 32, 32))
        self.assertTrue(torch.isfinite(xy).all() and torch.isfinite(confidence).all())
        logits[0, 0, 0, 0, 0].backward()
        self.assertGreater(float(image.grad[0, 0, :, -4:, -4:].abs().sum()), 0)

    def test_mid_epoch_resume_restores_optimizer_and_skips_initial_eval(self):
        output = self.base / "mid"
        original_step = v0.ContextTrainer.train_step

        def stop_after_first_update(trainer):
            if trainer.step == 1:
                raise TimeoutError("test mid-epoch stop")
            return original_step(trainer)

        with patch.object(v0.ContextTrainer, "train_step", stop_after_first_update):
            partial = v0.run(self.reader, output, "cpu")
        self.assertEqual(partial["durable"]["step"], 1)
        self.assertEqual(partial["durable"]["epoch"], 0)
        evaluated_steps = []
        original_evaluate = v0.ContextTrainer.evaluate

        def record_eval(trainer, split):
            evaluated_steps.append(trainer.step)
            return original_evaluate(trainer, split)

        with patch.object(v0.ContextTrainer, "evaluate", record_eval):
            resumed = v0.run(self.reader, output, "cpu", resume_sha256=partial["durable"]["sha256"])
        self.assertEqual(evaluated_steps, [2, 4])
        state = v0.read_run_checkpoint(output / "last.pt", resumed["durable"]["sha256"])
        self.assertEqual(state["samples_seen"], 6)
        self.assertTrue(all(float(entry["step"]) == 4 for entry in state["optimizer"]["state"].values()))
