"""현재·과거의 검증된 annotation snapshot만 조립하는 causal window 계약."""

import re
from collections import deque
from dataclasses import dataclass
from typing import Iterable, Sequence

from .audit import FrameSample, REQUIRED_FIELDS, SequenceSamples, finite_number

CONTRACT_VERSION = "0.1.0"
ANNOTATION_ERRORS = {"missing_joint", "invalid_joint", "missing_field", "invalid_vector"}


class WindowContractError(ValueError):
    """History를 변경하지 않고 거부한 frame/설정 오류."""


@dataclass(frozen=True)
class FieldSnapshot:
    name: str
    coordinates: tuple[tuple[int | float, ...] | None, ...]
    valid_mask: tuple[bool, ...]


@dataclass(frozen=True)
class FrameSnapshot:
    sequence_id: str
    frame_index: int
    source_rel_path: str
    timestamp_raw: int | float | str | None
    fields: tuple[FieldSnapshot, ...]
    # 선택한 joint/field만의 현 프레임 진단. Sequence 전체 union issue를 복사하지 않는다.
    invalid_annotations: tuple[tuple[str, str], ...]


@dataclass(frozen=True)
class CausalWindow:
    joint_names: tuple[str, ...]
    field_names: tuple[str, ...]
    frames: tuple[FrameSnapshot | None, ...]
    boundary_reason: str

    @property
    def time_valid(self) -> tuple[bool, ...]:
        return tuple(frame is not None for frame in self.frames)

    def to_dict(self) -> dict:
        """호출자에게 새 nested list를 반환한다. 내부 snapshot과 alias하지 않는다."""
        fields = {}
        for index, name in enumerate(self.field_names):
            fields[name] = {
                "coordinates": [None if f is None else [None if p is None else list(p)
                    for p in f.fields[index].coordinates] for f in self.frames],
                "valid_mask": [None if f is None else list(f.fields[index].valid_mask) for f in self.frames],
            }
        return {
            "contract_version": CONTRACT_VERSION,
            "sequence_id": self.frames[-1].sequence_id,
            "current_frame_index": self.frames[-1].frame_index,
            "joint_names": list(self.joint_names), "window_size": len(self.frames),
            "boundary_reason": self.boundary_reason,
            "frame_indices": [None if f is None else f.frame_index for f in self.frames],
            "source_rel_paths": [None if f is None else f.source_rel_path for f in self.frames],
            "time_valid": list(self.time_valid), "fields": fields,
            "timestamp_raw": [None if f is None else f.timestamp_raw for f in self.frames],
            "timestamp_semantics": "unknown", "fps": "unknown", "units": "unknown",
            "visibility": {side: [None if f is None else [None] * len(self.joint_names)
                for f in self.frames] for side in ("left", "right")},
            "confidence": {side: [None if f is None else [None] * len(self.joint_names)
                for f in self.frames] for side in ("left", "right")},
            "invalid_annotations": [None if f is None else [list(pair) for pair in f.invalid_annotations]
                for f in self.frames],
            "input_source": "gt_annotation", "coordinate_transform": "none",
        }


def _names(values: Sequence[str], label: str) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, (list, tuple)):
        raise WindowContractError(f"{label}에는 명시적인 이름 list/tuple이 필요합니다.")
    result = tuple(values)
    if not result or any(not isinstance(n, str) or not n for n in result) or len(set(result)) != len(result):
        raise WindowContractError(f"{label}은 비어 있지 않은 중복 없는 이름 목록이어야 합니다.")
    return result


def _snapshot(sample: FrameSample, names: tuple[str, ...], fields: tuple[str, ...]) -> FrameSnapshot:
    if not isinstance(sample, FrameSample):
        raise WindowContractError("FrameSample 입력이 필요합니다.")

    def reject(detail):
        raise WindowContractError(f"{sample.source_rel_path}: {detail}")

    if not isinstance(sample.sequence_id, str) or not sample.sequence_id:
        reject("비어 있지 않은 sequence_id가 필요합니다.")
    if type(sample.frame_index) is not int or sample.frame_index < 0:
        reject("frame_index는 비음수 정수여야 합니다. bool/unknown 불허")
    if not isinstance(sample.source_rel_path, str) or not sample.source_rel_path:
        reject("source_rel_path가 필요합니다.")
    doc = sample.raw_document
    if not isinstance(doc, dict) or not isinstance(doc.get("joints"), dict) or not doc["joints"]:
        reject("파싱·schema가 유효한 원본 annotation 사전이 필요합니다.")
    frame = doc.get("frame")
    if type(frame) is int:
        frame_matches = frame == sample.frame_index
    else:
        frame_matches = isinstance(frame, str) and bool(re.fullmatch(r"[0-9]+", frame)) and frame.lstrip("0") == str(sample.frame_index).lstrip("0")
    if not frame_matches:
        reject("원본 frame 값과 frame_index가 다릅니다.")
    if not isinstance(sample.issues, list) or any(not isinstance(i, dict) or
        i.get("severity") not in ("warning", "error") or not isinstance(i.get("code"), str) for i in sample.issues):
        reject("잘못된 감사 issue 구조")
    for issue in sample.issues:
        if issue["severity"] == "error" and issue["code"] not in ANNOTATION_ERRORS:
            reject(f"구조/식별 오류: {issue['code']}")
    if not isinstance(sample.values, dict) or not isinstance(sample.field_valid_mask, dict):
        reject("field 값/mask 사전이 필요합니다.")
    timestamp = doc.get("timestamp")
    if timestamp is not None and not (isinstance(timestamp, str) or finite_number(timestamp)):
        reject("timestamp_raw는 유한한 숫자/string/None이어야 합니다. 시간 의미는 미검증")
    snapshots, invalid = [], []
    for field in fields:
        coordinates, masks = [], []
        for name in names:
            values = sample.values.get(name, {})
            validity = sample.field_valid_mask.get(name, {})
            if not isinstance(values, dict) or not isinstance(validity, dict):
                reject(f"{name}: 잘못된 field 값/mask 구조")
            # 선택 밖 관절의 sequence union 변화는 수용 여부/출력에 영향을 주지 않는다.
            if name in doc["joints"] and (field not in values or field not in validity):
                reject(f"{name}.{field}: 감사되지 않은 field (값/mask 누락)")
            value, valid = values.get(field), validity.get(field, False)
            if type(valid) is not bool:
                reject(f"{name}.{field}: valid mask는 bool이어야 합니다.")
            raw_joint = doc["joints"].get(name)
            raw_value = raw_joint.get(field) if isinstance(raw_joint, dict) else None
            raw_valid = isinstance(raw_value, list) and len(raw_value) == REQUIRED_FIELDS[field] and all(finite_number(v) for v in raw_value)
            if valid != raw_valid:
                reject(f"{name}.{field}: 원본 annotation과 cache mask 불일치")
            if valid:
                if name not in doc["joints"] or not isinstance(doc["joints"][name], dict):
                    reject(f"{name}.{field}: 원본에 없는 관절을 유효 처리할 수 없습니다.")
                if not isinstance(value, list) or len(value) != REQUIRED_FIELDS[field] or not all(finite_number(v) for v in value):
                    reject(f"{name}.{field}: true mask에 대응하는 유한한 좌표 배열이 필요합니다.")
                if value != raw_value:
                    reject(f"{name}.{field}: 원본 annotation과 cache 좌표 불일치")
                coordinates.append(tuple(value))
            else:
                if value is not None:
                    reject(f"{name}.{field}: false mask의 좌표는 None이어야 합니다.")
                coordinates.append(None)
                invalid.append((name, field))
            masks.append(valid)
        snapshots.append(FieldSnapshot(field, tuple(coordinates), tuple(masks)))
    return FrameSnapshot(sample.sequence_id, sample.frame_index, sample.source_rel_path,
                         timestamp, tuple(snapshots), tuple(invalid))


class CausalWindowStream:
    """Gap/새 sequence에서 초기화. 거부된 push는 기존 상태를 변경하지 않는다."""

    def __init__(self, window_size: int, joint_names: Sequence[str],
                 field_names: Sequence[str] = tuple(REQUIRED_FIELDS)):
        if type(window_size) is not int or window_size < 1:
            raise WindowContractError("window_size는 현재를 포함한 양의 정수 길이여야 합니다.")
        self._window_size = window_size
        self._joint_names = _names(joint_names, "joint_names")
        self._field_names = _names(field_names, "field_names")
        if any(n not in REQUIRED_FIELDS for n in self._field_names):
            raise WindowContractError("지원하지 않는 field 이름")
        self._history = deque(maxlen=window_size)
        self._pending_reason = "start"

    def reset(self):
        self._history.clear()
        self._pending_reason = "manual_reset"

    def push(self, sample: FrameSample) -> CausalWindow:
        current = _snapshot(sample, self._joint_names, self._field_names)
        # 모든 거부 조건을 먼저 검사한다. 이후부터 상태를 변경한다.
        reason = self._pending_reason if not self._history else "continue"
        if self._history:
            previous = self._history[-1]
            if previous.sequence_id != current.sequence_id:
                reason = "sequence_change"
            elif current.frame_index <= previous.frame_index:
                raise WindowContractError(f"{current.source_rel_path}: 중복/역순 frame (이전 {previous.frame_index}, 현재 {current.frame_index})")
            elif current.frame_index != previous.frame_index + 1:
                reason = "frame_gap"
        if reason in ("sequence_change", "frame_gap"):
            self._history.clear()
        self._history.append(current)
        self._pending_reason = "continue"
        padded = (None,) * (self._window_size - len(self._history)) + tuple(self._history)
        return CausalWindow(self._joint_names, self._field_names, padded, reason)


def offline_windows(sequences: Iterable[SequenceSamples], window_size: int,
                    joint_names: Sequence[str], field_names: Sequence[str] = tuple(REQUIRED_FIELDS)) -> list[CausalWindow]:
    """입력 순서를 정렬/수정하지 않는다. 각 SequenceSamples의 경계도 보존한다."""
    # 같은 ID의 별도 container/빈 clip 경계에서는 streaming 호출자도 reset해야 한다.
    result = []
    state = CausalWindowStream(window_size, joint_names, field_names)
    previous_container = None
    for sequence in sequences:
        if not isinstance(sequence, SequenceSamples) or not isinstance(sequence.sequence_id, str) or not sequence.sequence_id:
            raise WindowContractError("유효한 SequenceSamples container가 필요합니다.")
        if not isinstance(sequence.frames, (list, tuple)):
            raise WindowContractError("SequenceSamples.frames는 list/tuple이어야 합니다.")
        if sequence.sequence_id == previous_container or not sequence.frames:
            state.reset()
        for frame in sequence.frames:
            if not isinstance(frame, FrameSample) or frame.sequence_id != sequence.sequence_id:
                raise WindowContractError("FrameSample과 SequenceSamples의 sequence_id 불일치")
            result.append(state.push(frame))
        previous_container = sequence.sequence_id
    return result
