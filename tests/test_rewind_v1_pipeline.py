from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import zipfile

import torch

from rewind_archive.contracts import canonical, digest
from rewind_prepare import temporal_cache as cache
from rewind_prepare.converter import sha
from rewind_prepare.temporal_selection import SCHEMA, seal
from rewind_v1.__main__ import main, recover
from rewind_v1.data import EVIDENCE_SCHEMA, REQUIRED_EVIDENCE
from rewind_v1.frontend import PaperHeatmapFrontend, frontend_fingerprint
from rewind_v1.run_data import ActualData, FixtureData
from rewind_v1.runner import Budget, LAST, REPORT
import test_extracted_stereo


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/rewind_v1_pipeline_synthetic.json"


class PipelineTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory(dir=ROOT)
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)

    def launch(self, name, *, stop=None, resume=None, config=CONFIG):
        output = self.root / name
        args = ["synthetic", "--config", str(config), "--config-sha256", sha(config.read_bytes()), "--output", str(output)]
        if stop:
            args += ["--stop-after-updates", str(stop)]
        if resume:
            args += ["--resume-sha256", resume]
        with redirect_stdout(io.StringIO()):
            main(args)
        return json.loads((output / REPORT).read_bytes())

    def assert_checkpoint_equal(self, left, right):
        def equal(a, b, location):
            if isinstance(a, torch.Tensor):
                self.assertTrue(torch.equal(a, b), location)
            elif isinstance(a, dict):
                self.assertEqual(set(a), set(b), location)
                for key in a:
                    equal(a[key], b[key], f"{location}/{key}")
            elif isinstance(a, (list, tuple)):
                self.assertEqual(len(a), len(b), location)
                for index, (first, second) in enumerate(zip(a, b)):
                    equal(first, second, f"{location}/{index}")
            else:
                self.assertEqual(a, b, location)
        states = [torch.load(self.root / name / LAST, map_location="cpu", weights_only=True) for name in (left, right)]
        equal(*states, LAST)

    def test_cli_pause_and_resume_matches_all_stages(self):
        full = self.launch("full")
        partial = self.launch("resume", stop=3)
        self.assertEqual((partial["status"], partial["cursor"]["stage"], partial["cursor"]["batch"]),
                         ("paused", "frontend", 3))
        for additional, stage, updates in ((5, "frontend", 8), (4, "body", 12), (1, "hand", 13)):
            partial = self.launch("resume", stop=additional, resume=partial["last_checkpoint"]["sha256"])
            self.assertEqual((partial["status"], partial["cursor"]["stage"], partial["cursor"]["updates"]),
                             ("paused", stage, updates))
        resumed = self.launch("resume", resume=partial["last_checkpoint"]["sha256"])
        self.assertEqual(resumed["status"], "completed")
        self.assertEqual(resumed["cursor"], full["cursor"])
        self.assert_checkpoint_equal("full", "resume")
        for folder in ("full", "resume"):
            state = torch.load(self.root / folder / LAST, map_location="cpu", weights_only=True)
            self.assertEqual(state["progress"]["updates"], 16)
            self.assertEqual(state["fit"]["scheduler"]["last_epoch"], 4)
        self.assertEqual(full["coverage"]["frontend_validation_frames"], 5)
        self.assertEqual(full["coverage"]["teacher_validation_frames"], 4)
        self.assertEqual(full["coverage"]["teacher_validation_excluded_keys"], [["fixture_val", 8]])
        self.assertEqual(full["cursor"]["history"][-1]["validation"]["upper_source"], "predicted_frozen_body")
        self.assertEqual(full["cursor"]["history"][-1]["validation"]["body_checkpoint_sha256"],
                         full["cursor"]["best"]["body"]["sha256"])

    def test_failed_evaluation_keeps_durable_cursor_and_recovery(self):
        full = self.launch("full")
        with patch("rewind_v1.runner.evaluate_frontend", side_effect=KeyboardInterrupt("stop before evaluation")):
            with self.assertRaisesRegex(KeyboardInterrupt, "stop before evaluation"):
                self.launch("stopped")
        run = self.root / "stopped"
        report = json.loads((run / REPORT).read_bytes())
        self.assertEqual(report["status"], "stopped")
        self.assertEqual(report["checkpoint_cursor"]["batch"], 4)
        self.assertEqual(report["checkpoint_cursor"]["epoch"], 0)
        archive = self.root / "recovery.zip"
        pin = recover(run, archive, 32 * 1024**2)["last_sha256"]
        with zipfile.ZipFile(archive) as bundle:
            self.assertIsNone(bundle.testzip())
            self.assertEqual(sha(bundle.read(LAST)), pin)
        resumed = self.launch("stopped", resume=pin)
        self.assertEqual(resumed["cursor"], full["cursor"])
        self.assert_checkpoint_equal("full", "stopped")

    def test_checkpoint_and_frozen_dependency_mutation_fail_closed(self):
        paused = self.launch("mutated", stop=8)
        root = self.root / "mutated"
        best = root / paused["cursor"]["best"]["frontend"]["name"]
        best.write_bytes(best.read_bytes() + b"x")
        with self.assertRaisesRegex(ValueError, "checkpoint_pin"):
            self.launch("mutated", resume=paused["last_checkpoint"]["sha256"])
        last = root / LAST
        last.write_bytes(last.read_bytes() + b"x")
        with self.assertRaisesRegex(ValueError, "checkpoint_pin"):
            self.launch("mutated", resume=paused["last_checkpoint"]["sha256"])

    def test_actual_guard_before_configuration_or_cache_reads(self):
        args = ["actual", "--output", str(self.root/"forbidden"), "--config", "missing",
                "--config-sha256", "unused", "--selection", "missing", "--selection-sha256", "unused",
                "--cache", "missing", "--cache-sha256", "unused", "--evidence", "missing",
                "--evidence-sha256", "unused", "--review-dir", "missing", "--spec", "missing",
                "--source-sha256", "unused"]
        with (
            patch("rewind_v1.__main__.require_a100", side_effect=RuntimeError("rewind_v1_a100_required")),
            patch("rewind_v1.__main__._regular_read") as read,
        ):
            with self.assertRaisesRegex(RuntimeError, "a100_required"):
                main(args)
            read.assert_not_called()
        self.assertFalse((self.root/"forbidden").exists())

    def test_actual_shaped_local_cache_and_review_pins(self):
        fixture = test_extracted_stereo.ExtractedStereoTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        fixture.add_pair(fixture.val, 1, 93)
        prior = fixture.prepare("v1_pipeline_prior")
        records = [{"key": [row["sequence_id"], row["frame_index"]], "files": row["files"], "split": row["split"]}
                   for row in prior["records"] if row["frame_index"] < 2]
        val_pin = digest([[fixture.val, 0], [fixture.val, 1]])
        plan = seal({"schema": SCHEMA, "status": "metadata_only", "clip_frames": 2, "seed": 71,
                     "train_ready": False, "temporal_ready": False, "records": records,
                     "train_clip_starts": [[fixture.train, 0]], "validation_keys_sha256": val_pin,
                     "spec_sha256": prior["provenance"]["spec_sha256"], "dataset_version": prior["dataset_version"],
                     "inventory_sha256": "a"*64, "calibration_sha256": prior["provenance"]["calibration_sha256"],
                     "blockers": ["synthetic_fixture"]})
        selected = self.root / "selection.json"
        selected.write_bytes(canonical(plan))
        review_dir = self.root / "reviews"
        review_dir.mkdir()
        for key in REQUIRED_EVIDENCE:
            (review_dir / (key + ".json")).write_bytes(canonical({"synthetic_fixture_review": key}))
        review_pins = {key: sha((review_dir/(key+".json")).read_bytes()) for key in REQUIRED_EVIDENCE}
        camera = {side: {"rotation_matrix": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                         "translation": [offset, 0, 0]}
                  for side, offset in (("left", -20), ("right", 20))}
        evidence = seal({"schema": EVIDENCE_SCHEMA, "selection_sha256": plan["content_sha256"],
                         "review_artifact_sha256": review_pins, "max_frame_delta_seconds": .05,
                         "frames": {seq: {str(i): {"time_seconds": i/30, "camera_pose": camera} for i in (0, 1)}
                                    for seq in (fixture.train, fixture.val)}})
        reviewed = self.root / "evidence.json"
        reviewed.write_bytes(canonical(evidence))
        cfg = json.loads(CONFIG.read_bytes())
        local = self.root / "local"
        with (
            patch.multiple(cache, CLIP=2, TRAIN_COUNT=2, VALIDATION_COUNT=2, PART_FRAMES=2),
            patch("rewind_v1.run_data.VALIDATION", val_pin),
            patch.object(ActualData, "width", 2),
            patch("rewind_v1.data.WINDOW_FRAMES", 2),
        ):
            packed = cache.prepare(plan, expected_sha256=plan["content_sha256"], rgb_root=fixture.rgb,
                                   pose_root=fixture.fixture.raw, spec_path=fixture.fixture.specpath,
                                   calibration_path=fixture.fixture.calibration, output=local,
                                   limits=cache.Limits(8*cache.MIB, 16*cache.MIB, 96*cache.MIB, 30, 2))
            args = SimpleNamespace(spec=fixture.fixture.specpath, selection=selected,
                                   selection_sha256=plan["content_sha256"], evidence=reviewed,
                                   evidence_sha256=evidence["content_sha256"], review_dir=review_dir,
                                   cache=self.root/"source-cache-absent", cache_local=local,
                                   cache_sha256=packed["cache_sha256"])
            data = ActualData(args, cfg, Budget(cfg["limits"]))
            self.assertEqual(data.restore_metrics["copied_bytes"], 0)
            self.assertEqual(data.coverage()["teacher_validation_frames"], 2)
            (image, target, mask), _ = data.frame((fixture.train, 0), "train", [32, 32])
            self.assertEqual((tuple(image.shape), tuple(target.shape), tuple(mask.shape)),
                             ((2, 3, 32, 32), (2, 62, 2), (2, 62)))
            frontend = PaperHeatmapFrontend(62).eval()
            clip = data.clip((fixture.train, 0), "train", frontend, [32, 32], frontend_fingerprint(frontend))
            self.assertEqual(tuple(clip.features.shape), (1, 2, 868))
            part = local/"part-0000.zip"
            original = part.read_bytes()
            part.write_bytes(b"\0" + original[1:])
            with self.assertRaisesRegex(ValueError, "part_pin"):
                ActualData(args, cfg, Budget(cfg["limits"]))
            part.write_bytes(original)
            (review_dir/(REQUIRED_EVIDENCE[0]+".json")).write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "review_pin"):
                ActualData(args, cfg, Budget(cfg["limits"]))

    def test_budget_split_and_notebook(self):
        cfg = json.loads(CONFIG.read_bytes())
        cfg["limits"]["max_updates"] = 15
        path = self.root / "too_small.json"
        path.write_text(json.dumps(cfg))
        with self.assertRaisesRegex(ValueError, "update_budget"):
            self.launch("bounded", config=path)
        self.assertFalse((self.root / "bounded" / LAST).exists())
        data = FixtureData()
        data.frames["validation"].append(data.frames["train"][0])
        with self.assertRaisesRegex(ValueError, "split_overlap"):
            data._validate_membership()
        notebook = json.loads((ROOT/"notebooks/archvis_v1_pipeline.ipynb").read_bytes())
        for cell in notebook["cells"]:
            if cell["cell_type"] == "code":
                compile("".join(cell["source"]), "v1 notebook", "exec")
