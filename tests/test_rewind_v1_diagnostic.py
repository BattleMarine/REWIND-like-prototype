import io
from pathlib import Path
import tempfile
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import patch

import torch

from rewind_archive.contracts import canonical
from rewind_prepare.temporal_selection import seal
from rewind_v1 import audit, diagnostic
from rewind_v1.__main__ import launch, main
from rewind_v1.runner import Pipeline
from scripts.build_archvis_v1_diagnostic import notebook_document


class DiagnosticTests(TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1])
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        (self.root / "audit").mkdir()
        self.hist = {**diagnostic.HISTORICAL, "selection_sha256": "a"*64, "cache_sha256": "b"*64}
        self.keys = [("train-fixture", 5), ("val-fixture", 9)]
        self.plan = {"content_sha256": self.hist["selection_sha256"], "spec_sha256": "c"*64,
                     "calibration_sha256": "d"*64, "validation_keys_sha256": "e"*64,
                     "records": [{"key": key, "split": split} for key, split in zip(self.keys, ("train", "validation"))]}
        self.cameras = seal({"schema": "archvis_v1_camera_candidates_v1",
                             "selection_sha256": self.hist["selection_sha256"], "cache_sha256": self.hist["cache_sha256"],
                             "train_ready": False, "rows": [{**row, "time_seconds": None} for row in self.plan["records"]]})
        self.hist["camera_candidates_sha256"] = self.cameras["content_sha256"]
        self.audit_doc = seal({"schema": audit.SCHEMA, "source_sha256": self.hist["audit_source_sha256"],
            "selection_sha256": self.hist["selection_sha256"], "cache_sha256": self.hist["cache_sha256"],
            "camera_candidates_sha256": self.hist["camera_candidates_sha256"], "spec_sha256": self.plan["spec_sha256"],
            "calibration_sha256": self.plan["calibration_sha256"], "validation_keys_sha256": self.plan["validation_keys_sha256"],
            "status": "blocked", "train_ready": False, "numeric_geometry_passed": True,
            "pairs_audited": {"train": 1, "validation": 1}, "overlays": [{"name": "gt.png", "sha256": "f"*64}],
            "time": {"timestamp_verified": False, "stereo_sync_verified": False, "nominal_fps_applied": False},
            "motion": {"all_original_ids_verified_in_pinned_map": False, "cross_split_declared_ids": [],
                       "cross_split_exact_stereo_pose_payload_groups": 0}})
        self.hist["audit_sha256"] = self.audit_doc["content_sha256"]
        self.review = seal({"schema": diagnostic.OVERLAY_SCHEMA, "audit_sha256": self.hist["audit_sha256"],
                            "reviewer": "user", "status": "no_issue_reported", "files": {"gt.png": "f"*64}})
        patcher = patch.multiple(diagnostic, HISTORICAL=self.hist, OVERLAY_COUNT=1, PAIR_COUNTS={"train": 1, "validation": 1})
        patcher.start()
        self.addCleanup(patcher.stop)
        self.write(self.root / "audit" / audit.REPORT, self.audit_doc)
        self.write(self.root / "audit" / audit.CAMERAS, self.cameras)
        self.decision = self.set_decision()

    @staticmethod
    def write(path, content):
        path.write_bytes(canonical(content))

    def set_decision(self, **changes):
        self.write(self.root / "overlay_review.json", self.review)
        decision = seal({"schema": diagnostic.DECISION_SCHEMA, **self.hist, **diagnostic.POLICY,
                         "overlay_review_sha256": self.review["content_sha256"], **changes})
        self.write(self.root / "diagnostic_decision.json", decision)
        return decision

    def test_prior_audit_is_pinned_without_current_source_equivalence(self):
        decision, report, rows = diagnostic.validate_bundle(self.root, self.decision["content_sha256"], self.plan)
        self.assertFalse(report["train_ready"])
        self.assertTrue(decision["validation_reference_only"])
        self.assertTrue(all(row["time_seconds"] is None for row in rows))
        for key, value in (("nominal_fps", 30), ("original_motion_verified", True), ("validation_reference_only", False)):
            changed = self.set_decision(**{key: value})
            with self.assertRaisesRegex(ValueError, key):
                diagnostic.validate_bundle(self.root, changed["content_sha256"], self.plan)
        self.set_decision()
        changed_audit = seal({**{k: v for k, v in self.audit_doc.items() if k != "content_sha256"}, "source_sha256": "0"*64})
        self.write(self.root / "audit" / audit.REPORT, changed_audit)
        with self.assertRaisesRegex(ValueError, "document_pin"):
            diagnostic.validate_bundle(self.root, self.decision["content_sha256"], self.plan)

    def test_resealed_overlay_substitution_is_rejected(self):
        self.review = seal({**{k: v for k, v in self.review.items() if k != "content_sha256"}, "files": {"gt.png": "a"*64}})
        decision = self.set_decision()
        with self.assertRaisesRegex(ValueError, "overlay_review"):
            diagnostic.validate_bundle(self.root, decision["content_sha256"], self.plan)

    def test_nominal_time_is_passed_only_via_diagnostic_context(self):
        data = object.__new__(diagnostic.DiagnosticData)
        data.identity = {"selection_sha256": self.hist["selection_sha256"], "decision_sha256": self.decision["content_sha256"]}
        data.plan, data.cache, data.spec_path, data.scale = self.plan, object(), self.root, 100.
        data.budget = type("Budget", (), {"check": lambda self: None})()
        data.camera = {("seq", index): torch.zeros(2, 9) for index in range(5, 7)}
        with patch.object(diagnostic, "_load_clip") as load:
            data.clip(("seq", 5), "train", None, [128, 128], "f"*64)
        call = load.call_args.kwargs
        self.assertEqual(call["context_sha256"], self.decision["content_sha256"])
        self.assertEqual(call["context_label"], "diagnostic_decision")
        self.assertNotIn("evidence", call)
        cameras, seconds = call["time_context"](range(5, 7))
        self.assertEqual(tuple(cameras.shape), (2, 2, 9))
        self.assertEqual(seconds, [0.2, 0.24])

    def test_strict_checkpoint_does_not_resume_into_diagnostic(self):
        pipeline = object.__new__(Pipeline)
        pipeline.root = self.root
        pipeline.cfg = {"limits": {"checkpoint_bytes": 1024}}
        pipeline.binding = {"input": {"run_kind": "nominal_25fps_diagnostic", "decision_sha256": "a"*64}}
        with patch("rewind_v1.runner._read_checkpoint", return_value={"binding": {"input": {"evidence_sha256": "a"*64}}}):
            with self.assertRaisesRegex(ValueError, "resume_binding"):
                pipeline._resume("b"*64)

    def test_cuda_guard_and_default_notebook_no_launch(self):
        args = ["diagnostic"]
        for name in ("output", "config", "selection", "spec", "cache", "cache-local", "diagnostic-bundle",
                     "config-sha256", "selection-sha256", "cache-sha256", "source-sha256", "decision-sha256"):
            args += ["--"+name, "missing"]
        with (
            patch("rewind_v1.__main__.require_cuda_training", side_effect=RuntimeError("CUDA required")),
            patch("rewind_v1.__main__._regular_read") as read,
        ):
            with self.assertRaisesRegex(RuntimeError, "CUDA required"):
                main(args)
            read.assert_not_called()
        cells = {cell["id"]: "".join(cell["source"]) for cell in notebook_document({})["cells"]}
        with patch("subprocess.run") as child, patch("sys.stdout", new_callable=io.StringIO):
            exec(cells["diagnostic"], {})
            exec(cells["recover"], {})
            child.assert_not_called()

    def test_all_real_training_commands_accept_t4_then_enforce_source_pin(self):
        for command in ("actual", "diagnostic", "epoch-expansion"):
            with (
                self.subTest(command=command),
                patch("torch.cuda.is_available", return_value=True),
                patch("torch.cuda.current_device", return_value=0),
                patch("torch.cuda.get_device_properties", return_value=SimpleNamespace(name="Tesla T4", total_memory=16*1024**3)),
                patch("rewind_v1.__main__._source", return_value={}),
                patch("rewind_v1.__main__._regular_read") as read,
                patch("sys.stdout", new_callable=io.StringIO),
            ):
                with self.assertRaisesRegex(ValueError, "launch_source_pin"):
                    launch(SimpleNamespace(command=command, device="cuda", source_sha256="invalid"))
                read.assert_not_called()
