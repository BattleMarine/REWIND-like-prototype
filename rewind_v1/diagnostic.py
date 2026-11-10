"""V1-DIAGNOSTIC-001. The nominal clock is scoped to this approved diagnostic."""

from pathlib import Path

import torch

from rewind_data.audit import read_json
from rewind_m1.local_rgb import _regular_read
from rewind_m2.data import encode_camera
from rewind_prepare.extracted import MAX_MANIFEST_BYTES, _local
from rewind_prepare.temporal_selection import check_pin
from .audit import CAMERAS, REPORT, SCHEMA
from .data import _load_clip
from .run_data import _CacheData


DECISION_SCHEMA = "archvis_v1_nominal_diagnostic_decision_v1"
OVERLAY_SCHEMA = "archvis_v1_diagnostic_overlay_review_v1"
HISTORICAL = {
    "audit_sha256": "410148107c11c315ec006104ba12c247e3ab36a15af0d1442cc5017041ca6dbf",
    "audit_source_sha256": "32a15a39d07dd474284a1848ee6a78948cb3b338fe91da1a90fc73e495480d5c",
    "camera_candidates_sha256": "d80b11535020f6784b000b0cfead95ca1984c6463a0498b77a5dd4e277195fe9",
    "selection_sha256": "48e3c749c7d7710feb7fd61601b7406264837e7a67ceeb4a48787359825cafad",
    "cache_sha256": "70241b27f66565593f0c8de0aeb6fe1f94c3672f932d7609bc882fd24a21d7b0",
    "audit_recovery_zip_sha256": "836e75cceb43819bed5d3400455a0208c55e7b09d2f3ac5886d3ae535b60efa3",
}
POLICY = {
    "run_id": "V1-DIAGNOSTIC-001", "decision_date": "2026-11-10", "authorized_by": "user",
    "purpose": "one_epoch_diagnostic_training", "nominal_fps": 25,
    "time_basis": "derived_from_dataset_fps", "stereo_sync_basis": "shared_frame_key_and_user_visual_review_assumption",
    "recorded_timestamps_verified": False, "stereo_sync_measured": False, "original_motion_verified": False,
    "validation_reference_only": True, "independent_validation_verified": False,
}
PAIR_COUNTS = {"train": 1000, "validation": 2236}
OVERLAY_COUNT = 89
CACHE_PARTS = 65
CACHE_PART_BYTES = 7_554_249_675
PILOT_CONFIG = Path(__file__).resolve().parents[1] / "configs/rewind_v1_pipeline_pilot_candidate.json"


def _doc(path):
    return read_json(_regular_read(_local(path), MAX_MANIFEST_BYTES))


def validate_pilot(config):
    if config != _doc(PILOT_CONFIG):
        raise ValueError("rewind_v1_diagnostic_pilot_config")


def validate_bundle(bundle, decision_sha256, plan):
    decision = _doc(bundle / "diagnostic_decision.json")
    check_pin(decision, DECISION_SCHEMA, decision_sha256)
    for key, value in {**HISTORICAL, **POLICY}.items():
        if key not in decision or type(decision[key]) is not type(value) or decision[key] != value:
            raise ValueError("rewind_v1_diagnostic_decision:" + key)
    if plan["content_sha256"] != HISTORICAL["selection_sha256"]:
        raise ValueError("rewind_v1_diagnostic_selection")
    report = _doc(bundle / "audit" / REPORT)
    check_pin(report, SCHEMA, HISTORICAL["audit_sha256"])
    if (report["source_sha256"] != HISTORICAL["audit_source_sha256"]
            or report["selection_sha256"] != HISTORICAL["selection_sha256"]
            or report["cache_sha256"] != HISTORICAL["cache_sha256"]
            or report["camera_candidates_sha256"] != HISTORICAL["camera_candidates_sha256"]
            or report["spec_sha256"] != plan["spec_sha256"]
            or report["calibration_sha256"] != plan["calibration_sha256"]
            or report["validation_keys_sha256"] != plan["validation_keys_sha256"]
            or report["status"] != "blocked" or report["train_ready"] is not False
            or report["numeric_geometry_passed"] is not True or report["pairs_audited"] != PAIR_COUNTS
            or report["time"]["timestamp_verified"] is not False
            or report["time"]["stereo_sync_verified"] is not False
            or report["time"]["nominal_fps_applied"] is not False
            or report["motion"]["all_original_ids_verified_in_pinned_map"] is not False
            or report["motion"]["cross_split_declared_ids"]
            or report["motion"]["cross_split_exact_stereo_pose_payload_groups"]):
        raise ValueError("rewind_v1_diagnostic_audit_contract")
    review = _doc(bundle / "overlay_review.json")
    check_pin(review, OVERLAY_SCHEMA, decision["overlay_review_sha256"])
    overlays = {row["name"]: row["sha256"] for row in report["overlays"]}
    if (len(overlays) != OVERLAY_COUNT or len(report["overlays"]) != OVERLAY_COUNT
            or review["files"] != overlays or review["audit_sha256"] != HISTORICAL["audit_sha256"]
            or review["reviewer"] != "user" or review["status"] != "no_issue_reported"):
        raise ValueError("rewind_v1_diagnostic_overlay_review")
    cameras = _doc(bundle / "audit" / CAMERAS)
    check_pin(cameras, "archvis_v1_camera_candidates_v1", HISTORICAL["camera_candidates_sha256"])
    if (cameras["selection_sha256"], cameras["cache_sha256"], cameras["train_ready"]) != (
        HISTORICAL["selection_sha256"], HISTORICAL["cache_sha256"], False
    ):
        raise ValueError("rewind_v1_diagnostic_camera_binding")
    expected = {tuple(row["key"]): row["split"] for row in plan["records"]}
    observed = {tuple(row["key"]): row["split"] for row in cameras["rows"]}
    if (len(cameras["rows"]) != sum(PAIR_COUNTS.values()) or len(observed) != len(cameras["rows"])
            or observed != expected or any(row["time_seconds"] is not None for row in cameras["rows"])):
        raise ValueError("rewind_v1_diagnostic_camera_coverage")
    return decision, report, cameras["rows"]


class DiagnosticData(_CacheData):
    def _time_identity(self, args):
        if args.cache_sha256 != HISTORICAL["cache_sha256"]:
            raise ValueError("rewind_v1_diagnostic_cache_pin")
        decision, report, rows = validate_bundle(args.diagnostic_bundle, args.decision_sha256, self.plan)
        for split, starts in self.clips.items():
            coverage = report["coverage"][split]
            if (coverage["clip_starts"] != [list(clip) for clip in starts]
                    or coverage["selected_pairs"] != PAIR_COUNTS[split]
                    or coverage["pairs_in_number_contiguous_clips"] != len(starts)*self.width
                    or coverage["number_contiguous_clips"] != len(starts)
                    or report["clip_frames"] != self.width):
                raise ValueError("rewind_v1_diagnostic_clip_coverage")
        self.camera = {}
        for row in rows:
            self.budget.check()
            self.camera[tuple(row["key"])] = encode_camera(row["camera_pose"], self.scale)
        return {"run_kind": "nominal_25fps_diagnostic", **POLICY,
                "original_motion_independence_verified": False, "foot_loss_applied": False,
                "decision_sha256": decision["content_sha256"], "overlay_review_sha256": decision["overlay_review_sha256"],
                "prior_audit": HISTORICAL.copy(), "clock_formula": "frame_index / 25"}

    def _check_cache(self, manifest):
        parts = manifest["parts"]
        if len(parts) != CACHE_PARTS or sum(part["bytes"] for part in parts) != CACHE_PART_BYTES:
            raise ValueError("rewind_v1_diagnostic_cache_size")

    def clip(self, key, split, frontend, size, pin):
        sequence, start = key

        def nominal_context(frames):
            return torch.stack([self.camera[sequence, frame] for frame in frames]), [frame / 25 for frame in frames]

        return _load_clip(self.cache, self.plan, sequence, start, split=split,
                          selection_sha256=self.identity["selection_sha256"], context_sha256=self.identity["decision_sha256"],
                          time_context=nominal_context, spec_path=self.spec_path, scale_mm=self.scale,
                          source="predicted_2d", frontend=frontend, frontend_size=size, frontend_sha256=pin,
                          check_budget=self.budget.check, context_label="diagnostic_decision")
