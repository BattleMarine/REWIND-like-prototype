"""데이터 파일 없이 synthetic_fixture window 결과를 재현하는 진입점."""

import argparse
import json
from pathlib import Path

from .audit import FrameSample, REQUIRED_FIELDS, SequenceSamples, _validate_joints
from .windows import CausalWindowStream, offline_windows


def synthetic_frame(sequence_id: str, frame_index: int) -> FrameSample:
    joints = {name: {field: [frame_index + offset + axis for axis in range(dim)]
        for field, dim in REQUIRED_FIELDS.items()} for name, offset in (("root", 0), ("tip", 10))}
    doc = {"frame": str(frame_index), "joints": joints}
    sample = FrameSample(f"synthetic/{sequence_id}/frame_{frame_index}.json", sequence_id, frame_index,
                         raw_document=doc)
    _validate_joints(sample, ["root", "tip"])
    return sample


def synthetic_sequences() -> list[SequenceSamples]:
    sequences = []
    for seq, indices in (("fixture_A", (0, 1, 3, 4)), ("fixture_B", (0, 1))):
        frames = [synthetic_frame(seq, index) for index in indices]
        sequences.append(SequenceSamples(seq, frames, ["root", "tip"], [], []))
    # 실제 누락과 수치 invalid를 fixture에서 의도적으로 만든다.
    frames = sequences[0].frames
    frames[1].raw_document["joints"]["tip"]["device_pts3d"] = None
    del frames[-1].raw_document["joints"]["tip"]
    for f in (frames[1], frames[-1]):
        f.issues.clear()
        _validate_joints(f, ["root", "tip"])
    return sequences


def fixture_report(window_size: int = 3) -> dict:
    sequences = synthetic_sequences()
    names = ["tip", "root"]
    windows = offline_windows(sequences, window_size, names)
    state = CausalWindowStream(window_size, names)
    streaming = [state.push(f) for seq in sequences for f in seq.frames]
    if windows != streaming:
        raise AssertionError("offline/streaming window 불일치")
    return {"data_kind": "synthetic_fixture", "fixture_version": "0.1.0",
            "real_data_accessed": False, "offline_streaming_equal": True,
            "gap_policy": "reset", "window_size": window_size,
            "windows": [w.to_dict() for w in windows]}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="실제 데이터 없이 causal window 합성 계약을 실행")
    parser.add_argument("--window-size", type=int, default=3, help="현재 frame을 포함한 총 길이")
    parser.add_argument("--output", type=Path, help="새 합성 결과 JSON 저장 경로; 생략 시 stdout")
    args = parser.parse_args(argv)
    try:
        doc = fixture_report(args.window_size)
        text = json.dumps(doc, ensure_ascii=True, indent=2, allow_nan=False) + "\n"
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(text, encoding="utf-8")
            print(json.dumps({"data_kind": doc["data_kind"], "real_data_accessed": False,
                "offline_streaming_equal": True, "window_count": len(doc["windows"])}, ensure_ascii=True))
        else:
            print(text, end="")
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
