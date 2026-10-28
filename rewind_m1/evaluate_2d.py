"""고정 예측 cache의 관절별 2D 픽셀 오차와 coverage 평가."""
import argparse
import copy
import math
from pathlib import Path

from rewind_data.audit import FrameSample, _validate_joints
from .data import DataContractError
from .prediction_join import PROVENANCE, _read_file, _serialized, join_predictions
from .training import source_identity


COUNTS = ("total_slots", "gt_valid_count", "gt_missing_count", "paired_count", "pred_missing_on_gt_count",
          "gt_out_of_frame_count", "pred_out_of_frame_paired_count", "gt_visibility_unknown_count",
          "pred_visibility_unknown_on_gt_count")


def _empty():
    return {**{key: 0 for key in COUNTS}, "sum_error_px": 0.}


def _finish(metric):
    count, gt = metric["paired_count"], metric["gt_valid_count"]
    coverage = count / gt if gt else None
    return {**metric, "mean_error_px": metric["sum_error_px"] / count if count else None,
            "coverage": coverage, "missing_prediction_rate": 1-coverage if gt else None}


def _outside(point, size):
    return not (0 <= point[0] <= size[0]-1 and 0 <= point[1] <= size[1]-1)


def evaluate_cache(pose_doc, cache_doc, expected_manifest_sha256, *, split, groups, train_diagnostic=False):
    if split not in ("train", "validation", "test") or type(train_diagnostic) is not bool or (split == "train" and not train_diagnostic):
        raise DataContractError("명시적 split 필요; train은 train_diagnostic=true 필요")
    _serialized(groups)
    if not isinstance(groups, dict) or len(groups) > 32:
        raise DataContractError("관절 group 사전 최대 32개 필요")
    joined = join_predictions(pose_doc, cache_doc, expected_manifest_sha256)
    names = joined["contract"]["joint_names"]
    for label, members in groups.items():
        if not isinstance(label, str) or not label.strip() or len(label.encode()) > 128 or label == "all":
            raise DataContractError("group 이름 오류; all은 예약됨")
        if not isinstance(members, list) or not members or any(not isinstance(n, str) or n not in names for n in members) or len(set(members)) != len(members):
            raise DataContractError("중복 없는 명시적 group 관절 목록 필요")
    group_names = {"all": list(names), **copy.deepcopy(groups)}
    metrics = {label: {side: _empty() for side in ("left", "right")} for label in group_names}
    selected = [r for r in joined["records"] if r["split"] == split]
    if not selected:
        raise DataContractError("선택 split이 비어 있음")
    sizes = joined["contract"]["image_sizes"]
    for row in selected:
        sample = FrameSample("evaluation", row["sequence_id"], row["frame_index"], raw_document={"joints": row["joints"]})
        _validate_joints(sample, names)
        for view, side in enumerate(("left", "right")):
            pred = row["prediction"][side]
            field = f"camera_{side}_pts2d"
            for index, name in enumerate(names):
                gt_valid = sample.field_valid_mask[name][field]
                pred_valid = pred["valid_mask"][index]
                delta = _empty()
                delta["total_slots"] = 1
                if gt_valid:
                    gt = sample.values[name][field]
                    delta["gt_valid_count"] = 1
                    delta["gt_visibility_unknown_count"] = 1
                    delta["gt_out_of_frame_count"] = int(_outside(gt, sizes[view]))
                    delta["pred_visibility_unknown_on_gt_count"] = int(pred.get("visibility", [None]*len(names))[index] is None)
                    if pred_valid:
                        point = pred["coordinates"][index]
                        delta["paired_count"] = 1
                        delta["pred_out_of_frame_paired_count"] = int(_outside(point, sizes[view]))
                        delta["sum_error_px"] = math.hypot(point[0]-gt[0], point[1]-gt[1])
                    else:
                        delta["pred_missing_on_gt_count"] = 1
                else:
                    delta["gt_missing_count"] = 1
                for label, members in group_names.items():
                    if name in members:
                        for key, value in delta.items():
                            metrics[label][side][key] += value
    for sides in metrics.values():
        combined = {key: sides["left"][key]+sides["right"][key] for key in (*COUNTS, "sum_error_px")}
        sides["combined"] = combined
        for side, metric in sides.items():
            if not math.isfinite(metric["sum_error_px"]):
                raise DataContractError("2D 픽셀 오차 합 비유한 값")
            sides[side] = _finish(metric)
    report = {"schema_version":"predicted_2d_evaluation_v1", "unit":"pixel", "alignment":"none", "aggregation":"paired_micro_mean",
        "split":split, "train_diagnostic":split == "train", "records":len(selected), "data_kind":joined["contract"]["data_kind"],
        "joint_names":names, "image_sizes":sizes, "groups":group_names, "group_overlap_policy":"independent_groups_do_not_sum; all_counts_each_joint_once",
        "gt_visibility_semantics":"unknown_in_current_m1_evaluation_contract; raw_source_may_contain_other_information",
        "provenance":joined["manifest"][PROVENANCE], "source":source_identity(),
        "frontend":{key: joined["contract"][key] for key in ("frontend_version", "frontend_checkpoint_sha256", "preprocessing_version")},
        "cache_split_cross_checked":"record_splits" in cache_doc, "metrics":metrics}
    _serialized(report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--split", choices=("validation", "test", "train"), required=True)
    parser.add_argument("--groups", type=Path, required=True)
    parser.add_argument("--train-diagnostic", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    result = evaluate_cache(_read_file(args.features), _read_file(args.cache), args.manifest_sha256,
        split=args.split, groups=_read_file(args.groups), train_diagnostic=args.train_diagnostic)
    created = False
    try:
        with args.output.open("xb") as handle:
            created = True
            handle.write(_serialized(result))
    except BaseException:
        if created:
            args.output.unlink(missing_ok=True)
        raise


if __name__ == "__main__":
    main()
