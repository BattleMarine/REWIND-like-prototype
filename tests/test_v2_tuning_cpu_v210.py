"""Protect the dcd20c5 v2.1.0 reference schema at the CPU postprocessing edge."""

import json
from pathlib import Path
import tempfile
from unittest import TestCase

from scripts.v2_tuning_cpu_common import canonical, sha
from scripts.v2_tuning_cpu_selection import seal
from scripts.v2_tuning_cpu_v210 import COMPARE, DIAGNOSTIC, FINAL, fixed_reference, match_recorded


class V210ReferencesTests(TestCase):
    def test_preserves_chosen_hand_epoch_and_rejects_diagnostic_byte_change(self):
        earlier = {"mpjpe_mm": 50., "valid_joints": 18}
        selected = {"mpjpe_mm": 42., "valid_joints": 16}

        def row(epoch, metrics):
            return {"stage": "hand", "epoch": epoch, "validation": {
                "joint_metrics": {"hand_local": metrics},
                "per_scenario_reference": {"scene": {"joint_metrics": {"hand_local": metrics}}}}}

        comparison = seal({"schema": "rewind_v2_gpu_comparison_v1", "macro_score_mm": 150.})
        diagnostic = seal({"schema": "rewind_v2_gpu_invocation_v1", "status": "completed",
                           "comparison": comparison, "checkpoint": {"content_sha256": FINAL},
                           "checkpoint_cursor": {"best": {"hand": {"epoch": 2}},
                                                 "history": [row(1, earlier), row(2, selected)]}})
        with tempfile.TemporaryDirectory() as tmp:
            project = Path(tmp).resolve()
            pins = {"evidence": {}}
            for name, evidence in ((COMPARE, comparison), (DIAGNOSTIC, diagnostic)):
                raw = canonical(evidence)
                path = project / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(raw)
                pins["evidence"][name] = {"bytes": len(raw), "sha256": sha(raw)}
            picked = fixed_reference(project, pins)
            self.assertEqual(picked["hand_local_baseline"], {"pooled": selected, "per_scenario": {"scene": selected}})
            (project / DIAGNOSTIC).write_bytes(json.dumps(diagnostic, indent=2).encode())
            with self.assertRaisesRegex(ValueError, "SHA or size mismatch"):
                fixed_reference(project, pins)

    def test_missing_hand_local_summary_metric_cannot_pass(self):
        expected = {"prior_comparison": {"pooled_hand_local": {"delta_mm": 0.1, "valid_joint_delta": 0}}}
        self.assertFalse(match_recorded({"prior_comparison": {}}, expected))
        self.assertTrue(match_recorded(
            {"prior_comparison": {"pooled_hand_local": {"delta_mm": 0.1000002, "valid_joint_delta": 0}}}, expected))
