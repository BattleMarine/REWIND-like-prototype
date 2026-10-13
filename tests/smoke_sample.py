"""실제 dataset sample의 보존·선택·mask 계약을 검사하는 수동 smoke 진입점."""

import argparse
import hashlib
import json
import sys
from pathlib import Path

# script 직접 실행에서도 프로젝트 모듈을 사용한다. 데이터 절대 경로는 고정하지 않는다.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rewind_data.audit import audit_dataset, load_provenance
from rewind_data.skeleton import load_skeleton


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--skeleton", type=Path, required=True)
    parser.add_argument("--provenance", type=Path, required=True)
    parser.add_argument("--expected-frames", type=int, required=True)
    parser.add_argument("--expected-joints", type=int, required=True)
    args = parser.parse_args()
    before = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in args.input.rglob("*.json")}
    skeleton = load_skeleton(args.skeleton)
    result = audit_dataset(args.input, skeleton, "smoke_unverified", load_provenance(args.provenance)).require_clean()
    assert result.manifest["data_kind"] == "dataset_sample", "합성 fixture를 실제 sample로 집계하지 않습니다."
    assert result.report["summary"]["file_count"] == args.expected_frames
    negative_2d = 0
    for sequence in result.sequences:
        assert [f.frame_index for f in sequence.frames] == sorted(f.frame_index for f in sequence.frames)
        for f in sequence.frames:
            raw_joints = f.raw_document["joints"]
            assert len(raw_joints) == args.expected_joints
            assert set(f.values) >= set(raw_joints)
            for group in skeleton.groups.values():
                selected = f.select(group)
                assert selected["joint_names"] == group
                assert selected["coordinates"] == [raw_joints[n]["device_pts3d"] for n in group]
                assert all(selected["valid_mask"])
                assert all(v is None for side in selected["visibility"].values() for v in side)
            for name, fields in raw_joints.items():
                for side in ("left", "right"):
                    field = f"camera_{side}_pts2d"
                    if any(v < 0 for v in fields[field]):
                        negative_2d += 1
                        assert f.field_valid_mask[name][field]
                        assert f.field_valid_mask[name]["device_pts3d"]
                        assert f.values[name][field] == fields[field]
    assert before == {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in args.input.rglob("*.json")}
    assert all(value == "unknown" for value in result.manifest["verification_state"].values())
    print(json.dumps({"data_kind": "dataset_sample", "summary": result.report["summary"],
        "joint_count_per_frame": args.expected_joints, "negative_2d_observations_retained": negative_2d,
        "groups": {key: len(names) for key, names in skeleton.groups.items()},
        "unique_selected_joints": len(skeleton.joint_names), "raw_hashes_unchanged": True}, indent=2))


if __name__ == "__main__":
    main()
