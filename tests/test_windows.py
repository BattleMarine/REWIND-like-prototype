"""실제 dataset 접근 없이 causal window의 경계·불변성·오류 계약을 검사한다."""

import contextlib
import copy
import io
import json
import tempfile
import unittest
from dataclasses import FrozenInstanceError
from pathlib import Path

from rewind_data.audit import SequenceSamples, _validate_joints, audit_dataset
from rewind_data.skeleton import Skeleton
from rewind_data.window_fixture import fixture_report, main, synthetic_frame, synthetic_sequences
from rewind_data.windows import CausalWindowStream, WindowContractError, offline_windows

NAMES = ["tip", "root"]


def sequence(seq="A", indices=(0, 1, 2)):
    return SequenceSamples(seq, [synthetic_frame(seq, i) for i in indices], ["root", "tip"], [], [])


class WindowContractTests(unittest.TestCase):
    def test_left_padding_and_golden_current_past_window(self):
        windows = offline_windows([sequence()], 3, NAMES)
        self.assertEqual([w.to_dict()["frame_indices"] for w in windows], [[None, None, 0], [None, 0, 1], [0, 1, 2]])
        self.assertEqual(windows[0].time_valid, (False, False, True))
        first = windows[0].to_dict()
        self.assertEqual(first["fields"]["device_pts3d"]["coordinates"], [None, None, [[10, 11, 12], [0, 1, 2]]])
        self.assertEqual(first["fields"]["device_pts3d"]["valid_mask"], [None, None, [True, True]])
        self.assertEqual(first["visibility"]["left"], [None, None, [None, None]])

    def test_future_frame_changes_leave_previous_windows_identical(self):
        baseline = sequence(indices=(0, 1, 2, 3, 4))
        before = offline_windows([baseline], 3, NAMES)
        changed = copy.deepcopy(baseline)
        for f in changed.frames[2:]:
            f.raw_document["joints"]["tip"]["device_pts3d"] = [900, 901, 902]
            _validate_joints(f, ["root", "tip"])
        after = offline_windows([changed], 3, NAMES)
        self.assertEqual(before[:2], after[:2])
        self.assertNotEqual(before[2], after[2])

    def test_future_outside_joint_union_does_not_change_past_output_or_acceptance(self):
        temp_root = Path(__file__).resolve().parents[1] / "outputs" / "test_tmp"
        temp_root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=temp_root) as temp:
            base = Path(temp).resolve()
            self.assertTrue(base.is_relative_to(temp_root.resolve()))
            for version in ("before", "after"):
                raw = base / version / "A"
                raw.mkdir(parents=True)
                for i in (0, 1, 2):
                    doc = synthetic_frame("A", i).raw_document
                    if version == "after" and i == 2:
                        doc["joints"]["future_only_extra"] = copy.deepcopy(doc["joints"]["root"])
                    (raw / f"frame_{i}.json").write_text(json.dumps(doc), encoding="utf-8")
            skeleton = Skeleton({"id": "synthetic", "status": "synthetic_fixture", "evidence": "test",
                "groups": {"body": ["root", "tip"]}, "edges": [["root", "tip"]]}, "fixture")
            before = audit_dataset(base / "before", skeleton, provenance={"data_kind": "synthetic_fixture"})
            after = audit_dataset(base / "after", skeleton, provenance={"data_kind": "synthetic_fixture"})
            self.assertTrue(any(i["code"] == "missing_joint" for i in after.sequences[0].frames[0].issues))
            self.assertFalse(any(i["code"] == "missing_joint" for i in before.sequences[0].frames[0].issues))
            self.assertEqual(offline_windows(before.sequences, 3, NAMES), offline_windows(after.sequences, 3, NAMES))

    def test_offline_streaming_equal_across_sequences_and_gap(self):
        seqs = synthetic_sequences()
        state = CausalWindowStream(3, NAMES)
        stream = [state.push(f) for seq in seqs for f in seq.frames]
        batch = offline_windows(seqs, 3, NAMES)
        self.assertEqual([w.to_dict() for w in stream], [w.to_dict() for w in batch])
        self.assertEqual([w.boundary_reason for w in batch], ["start", "continue", "frame_gap", "continue", "sequence_change", "continue"])

    def test_same_id_container_boundary_matches_explicit_stream_reset(self):
        seqs = [sequence("A", (0, 1)), sequence("A", (0, 1))]
        state = CausalWindowStream(3, NAMES)
        streaming = [state.push(f) for f in seqs[0].frames]
        state.reset()
        streaming += [state.push(f) for f in seqs[1].frames]
        batch = offline_windows(seqs, 3, NAMES)
        self.assertEqual(streaming, batch)
        self.assertEqual(batch[2].to_dict()["frame_indices"], [None, None, 0])
        self.assertEqual(batch[2].boundary_reason, "manual_reset")

    def test_empty_container_boundary_matches_explicit_reset(self):
        seqs = [sequence("A", (0,)), sequence("B", ()), sequence("A", (1,))]
        state = CausalWindowStream(3, NAMES)
        first = state.push(seqs[0].frames[0])
        state.reset()  # frame이 없어도 명시적인 clip 경계를 처리.
        second = state.push(seqs[2].frames[0])
        self.assertEqual(offline_windows(seqs, 3, NAMES), [first, second])
        self.assertEqual(second.to_dict()["frame_indices"], [None, None, 1])

    def test_manual_reset_and_sequence_change_never_reuse_old_history(self):
        state = CausalWindowStream(3, NAMES)
        old = state.push(synthetic_frame("A", 7))
        state.reset()
        reset = state.push(synthetic_frame("A", 0))
        new = state.push(synthetic_frame("B", 99))
        self.assertEqual(reset.to_dict()["frame_indices"], [None, None, 0])
        self.assertEqual(new.to_dict()["frame_indices"], [None, None, 99])
        self.assertEqual(old.to_dict()["frame_indices"], [None, None, 7])

    def test_gap_starts_new_history_and_does_not_allocate_gap_length(self):
        state = CausalWindowStream(3, NAMES)
        state.push(synthetic_frame("A", 0))
        gap = state.push(synthetic_frame("A", 1000000000))
        self.assertEqual(gap.to_dict()["frame_indices"], [None, None, 1000000000])
        self.assertEqual(gap.boundary_reason, "frame_gap")
        self.assertEqual(state.push(synthetic_frame("A", 1000000001)).to_dict()["frame_indices"], [None, 1000000000, 1000000001])

    def test_duplicate_reverse_rejection_preserves_history(self):
        state = CausalWindowStream(3, NAMES)
        state.push(synthetic_frame("A", 0))
        before = state.push(synthetic_frame("A", 1))
        for index in (1, 0):
            with self.assertRaisesRegex(WindowContractError, "중복/역순"):
                state.push(synthetic_frame("A", index))
        self.assertEqual(state.push(synthetic_frame("A", 2)).to_dict()["frame_indices"], [0, 1, 2])
        self.assertEqual(before.to_dict()["frame_indices"], [None, 0, 1])

    def test_corrupt_current_in_new_sequence_does_not_clear_valid_history(self):
        state = CausalWindowStream(3, NAMES)
        state.push(synthetic_frame("A", 0))
        bad = synthetic_frame("B", 0)
        bad.values["tip"]["device_pts3d"] = [1, 2]
        with self.assertRaises(WindowContractError):
            state.push(bad)
        self.assertEqual(state.push(synthetic_frame("A", 1)).to_dict()["frame_indices"], [None, 0, 1])

    def test_corrupt_gap_frame_is_atomic_and_can_be_retried(self):
        state = CausalWindowStream(3, NAMES)
        state.push(synthetic_frame("A", 0))
        bad = synthetic_frame("A", 8)
        bad.field_valid_mask["tip"]["device_pts3d"] = "true"
        with self.assertRaises(WindowContractError):
            state.push(bad)
        self.assertEqual(state.push(synthetic_frame("A", 1)).to_dict()["frame_indices"], [None, 0, 1])
        retried = state.push(synthetic_frame("A", 8))
        self.assertEqual(retried.to_dict()["frame_indices"], [None, None, 8])

    def test_unparsed_structural_error_unknown_index_and_mismatch_are_rejected(self):
        for kind in ("unparsed", "fatal_issue", "index", "frame", "schema"):
            with self.subTest(kind=kind):
                state = CausalWindowStream(3, NAMES)
                state.push(synthetic_frame("A", 0))
                bad = synthetic_frame("A", 1)
                if kind == "unparsed": bad.raw_document = None
                elif kind == "fatal_issue": bad.issue("frame_id_mismatch", "$.frame", "fixture")
                elif kind == "index": bad.frame_index = None
                elif kind == "frame": bad.raw_document["frame"] = "2"
                else: bad.raw_document["joints"] = []
                with self.assertRaises(WindowContractError): state.push(bad)
                self.assertEqual(state.push(synthetic_frame("A", 1)).to_dict()["frame_indices"], [None, 0, 1])

    def test_missing_joint_and_invalid_field_keep_real_time_slot(self):
        windows = offline_windows(synthetic_sequences(), 3, NAMES)
        invalid = windows[1].to_dict()
        self.assertEqual(invalid["time_valid"], [False, True, True])
        self.assertEqual(invalid["fields"]["device_pts3d"]["coordinates"][-1], [None, [1, 2, 3]])
        self.assertEqual(invalid["fields"]["device_pts3d"]["valid_mask"][-1], [False, True])
        self.assertEqual(invalid["fields"]["camera_left_pts2d"]["valid_mask"][-1], [True, True])
        missing = windows[3].to_dict()
        self.assertTrue(missing["time_valid"][-1])
        self.assertTrue(all(not f["valid_mask"][-1][0] for f in missing["fields"].values()))

    def test_selected_absent_joint_and_explicit_fields(self):
        frame = synthetic_frame("A", 0)
        fields = ["device_pts3d", "camera_left_pts2d"]
        window = CausalWindowStream(2, ["unknown", "root"], fields).push(frame).to_dict()
        self.assertEqual(list(window["fields"]), fields)
        self.assertEqual(window["fields"]["device_pts3d"]["coordinates"][-1], [None, [0, 1, 2]])
        self.assertEqual(window["fields"]["device_pts3d"]["valid_mask"][-1], [False, True])

    def test_invalid_unselected_field_is_not_used_as_selected_mask(self):
        frame = synthetic_frame("A", 0)
        frame.raw_document["joints"]["tip"]["camera_left_pts2d"] = None
        _validate_joints(frame, ["root", "tip"])
        result = CausalWindowStream(1, NAMES, ["device_pts3d"]).push(frame).to_dict()
        self.assertEqual(result["fields"]["device_pts3d"]["valid_mask"], [[True, True]])
        self.assertEqual(result["invalid_annotations"], [[]])

    def test_mask_value_inconsistencies_are_rejected(self):
        for kind in ("true_null", "false_coordinate", "nonfinite", "bool_coordinate", "mask_missing"):
            with self.subTest(kind=kind):
                bad = synthetic_frame("A", 0)
                if kind == "true_null": bad.values["tip"]["device_pts3d"] = None
                elif kind == "false_coordinate": bad.field_valid_mask["tip"]["device_pts3d"] = False
                elif kind == "nonfinite": bad.values["tip"]["device_pts3d"] = [float("inf"), 0, 0]
                elif kind == "bool_coordinate": bad.values["tip"]["device_pts3d"] = [True, 0, 0]
                else: del bad.field_valid_mask["tip"]["device_pts3d"]
                with self.assertRaises(WindowContractError): CausalWindowStream(2, NAMES).push(bad)

    def test_stale_cache_after_raw_annotation_corruption_is_rejected_atomically(self):
        for changed in (None, [1, 2], [700, 701, 702]):
            with self.subTest(changed=changed):
                state = CausalWindowStream(3, NAMES)
                state.push(synthetic_frame("A", 0))
                bad = synthetic_frame("A", 1)
                bad.raw_document["joints"]["tip"]["device_pts3d"] = changed
                with self.assertRaisesRegex(WindowContractError, "불일치"):
                    state.push(bad)
                self.assertEqual(state.push(synthetic_frame("A", 1)).to_dict()["frame_indices"], [None, 0, 1])

    def test_timestamp_raw_and_negative_2d_have_no_inferred_time_or_visibility(self):
        frame = synthetic_frame("A", 0)
        frame.raw_document["timestamp"] = "unverified_stamp"
        frame.raw_document["joints"]["tip"]["camera_left_pts2d"] = [-50, 5000]
        _validate_joints(frame, NAMES)
        doc = CausalWindowStream(2, NAMES).push(frame).to_dict()
        self.assertEqual(doc["timestamp_raw"], [None, "unverified_stamp"])
        self.assertEqual(doc["timestamp_semantics"], "unknown")
        self.assertEqual(doc["fps"], "unknown")
        self.assertEqual(doc["fields"]["camera_left_pts2d"]["coordinates"][-1][0], [-50, 5000])
        self.assertEqual(doc["visibility"]["left"][-1], [None, None])

    def test_snapshot_isolation_from_input_returned_dict_and_other_window(self):
        state = CausalWindowStream(3, NAMES)
        first_frame = synthetic_frame("A", 0)
        first = state.push(first_frame)
        first_before = first.to_dict()
        first_frame.values["tip"]["device_pts3d"][0] = 999
        first_frame.raw_document["timestamp"] = "changed_after_push"
        mutated = first.to_dict()
        mutated["fields"]["device_pts3d"]["coordinates"][-1][0][0] = -999
        mutated["visibility"]["left"][-1][0] = True
        second = state.push(synthetic_frame("A", 1))
        self.assertEqual(first.to_dict(), first_before)
        self.assertEqual(second.to_dict()["fields"]["device_pts3d"]["coordinates"][-2][0], [10, 11, 12])
        self.assertIsNone(second.to_dict()["timestamp_raw"][-2])
        second_mutated = second.to_dict()
        second_mutated["fields"]["device_pts3d"]["valid_mask"][-2][0] = False
        self.assertEqual(first.to_dict(), first_before)
        with self.assertRaises(FrozenInstanceError): first.frames[-1].frame_index = 8
        with self.assertRaises(TypeError): first.frames[-1].fields[0].coordinates[0][0] = 88

    def test_configuration_is_validated_even_for_empty_batch(self):
        self.assertEqual(offline_windows([], 3, NAMES), [])
        for size in (0, -1, False, 2.5):
            with self.assertRaises(WindowContractError): offline_windows([], size, NAMES)
        for names in ([], ["tip", "tip"], "tip", [None]):
            with self.assertRaises(WindowContractError): CausalWindowStream(3, names)
        for fields in ([], ["unknown"], ["device_pts3d", "device_pts3d"]):
            with self.assertRaises(WindowContractError): offline_windows([], 3, NAMES, fields)

    def test_window_size_one_and_caller_selection_mutation(self):
        names, fields = list(NAMES), ["device_pts3d"]
        state = CausalWindowStream(1, names, fields)
        names.reverse(); fields.append("camera_left_pts2d")
        for i in range(4):
            doc = state.push(synthetic_frame("A", i)).to_dict()
            self.assertEqual(doc["joint_names"], NAMES)
            self.assertEqual(doc["frame_indices"], [i])
            self.assertEqual(list(doc["fields"]), ["device_pts3d"])

    def test_offline_rejects_reverse_duplicate_and_wrong_container(self):
        for seq in (sequence("A", (1, 0)), sequence("A", (0, 0))):
            with self.assertRaises(WindowContractError): offline_windows([seq], 3, NAMES)
        wrong = sequence("A")
        wrong.frames[1].sequence_id = "B"
        with self.assertRaises(WindowContractError): offline_windows([wrong], 3, NAMES)
        wrong.frames = None
        with self.assertRaises(WindowContractError): offline_windows([wrong], 3, NAMES)

    def test_fixture_cli_and_report_are_synthetic_without_dataset_paths(self):
        first, second = fixture_report(), fixture_report()
        self.assertEqual(first, second)
        self.assertEqual(first["data_kind"], "synthetic_fixture")
        self.assertFalse(first["real_data_accessed"])
        self.assertTrue(first["offline_streaming_equal"])
        self.assertEqual(len(first["windows"]), 6)
        self.assertTrue(all(w["fps"] == "unknown" for w in first["windows"]))
        text = io.StringIO()
        with contextlib.redirect_stdout(text): self.assertEqual(main(["--window-size", "3"]), 0)
        self.assertEqual(json.loads(text.getvalue()), first)


if __name__ == "__main__":
    unittest.main()
