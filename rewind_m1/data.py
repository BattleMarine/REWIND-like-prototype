"""명시적 계약을 검증한 stereo 2D 특징과 GT 3D tensor 경로."""

import hashlib
import json
import re
from dataclasses import dataclass

import torch

from rewind_data.audit import FrameSample, _validate_joints, finite_number
from rewind_data.windows import CausalWindow, CausalWindowStream

MM_FACTOR = {"mm": 1.0, "cm": 10.0, "m": 1000.0}
VERIFICATIONS = ("joints", "units", "coordinates", "calibration", "split", "rgb_pose_alignment")


class DataContractError(ValueError):
    pass


def content_hash(doc) -> str:
    return hashlib.sha256(json.dumps(doc, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


@dataclass(frozen=True)
class InputContract:
    joint_names: tuple[str, ...]
    distance_unit: str
    image_sizes: tuple[tuple[int, int], tuple[int, int]]
    input_source: str
    data_kind: str
    frontend_version: str | None
    frontend_checkpoint_sha256: str | None
    preprocessing_version: str

    @classmethod
    def from_dict(cls, doc):
        if not isinstance(doc, dict):
            raise DataContractError("입력 계약 사전 필요")
        names = doc.get("joint_names")
        if not isinstance(names, list) or not names or any(not isinstance(n, str) or not n for n in names) or len(set(names)) != len(names):
            raise DataContractError("joint_names: 명시적인 중복 없는 이름 목록 필요")
        if doc.get("distance_unit") not in MM_FACTOR or doc.get("coordinate_frame") != "device_relative":
            raise DataContractError("단위(mm/cm/m)와 device_relative 좌표 계약 필요; unknown으로 학습 불가")
        sizes = doc.get("image_sizes")
        if not isinstance(sizes, list) or len(sizes) != 2 or any(not isinstance(s, list) or len(s) != 2 or
            any(type(v) is not int or v < 2 for v in s) for s in sizes):
            raise DataContractError("좌우 image_sizes=[[width,height],[width,height]] 필요")
        source, kind = doc.get("input_source"), doc.get("data_kind")
        if source not in ("gt_2d", "predicted_2d") or kind not in ("synthetic_fixture", "dataset_sample"):
            raise DataContractError("입력 출처 및 synthetic_fixture/dataset_sample 구분 필요")
        frontend = doc.get("frontend_version")
        weights = doc.get("frontend_checkpoint_sha256")
        if source == "predicted_2d" and (not isinstance(frontend, str) or not frontend):
            raise DataContractError("예측 입력은 frontend_version 필요")
        if source == "predicted_2d" and (not isinstance(weights, str) or not re.fullmatch(r"[0-9a-f]{64}", weights)):
            raise DataContractError("예측 입력은 frontend checkpoint SHA-256 필요")
        if source == "gt_2d" and (frontend is not None or weights is not None):
            raise DataContractError("GT 진단에 예측 frontend 버전을 붙이지 않음")
        prep = doc.get("preprocessing_version")
        if not isinstance(prep, str) or not prep:
            raise DataContractError("preprocessing_version 필요")
        if kind == "dataset_sample":
            checked, evidence = doc.get("verified", {}), doc.get("verification_evidence", {})
            if not isinstance(checked,dict) or not isinstance(evidence,dict): raise DataContractError("검증 상태/근거 사전 필요")
            if any(checked.get(k) is not True or not isinstance(evidence.get(k), str) or not evidence[k].strip() for k in VERIFICATIONS):
                raise DataContractError("실데이터 학습에는 관절/단위/좌표/calibration/split/RGB-pose 검증과 근거가 필요")
        return cls(tuple(names), doc["distance_unit"], tuple(tuple(s) for s in sizes), source, kind, frontend, weights, prep)


@dataclass
class TensorSplit:
    features: torch.Tensor  # N,T,14J: view별 xy,valid,confidence,known,visibility,known
    time_valid: torch.Tensor  # N,T
    target_mm: torch.Tensor  # N,T,J,3
    target_valid: torch.Tensor  # N,T,J; 0은 mask가 false인 수치 자리표시자
    anchors: list[tuple[str, int]]


def _observation(obs, names):
    if not isinstance(obs, dict) or obs.get("joint_names") != list(names):
        raise DataContractError("예측 관절 이름/순서 불일치 또는 예측 누락")
    for side in ("left", "right"):
        coords, masks = obs.get(side, {}).get("coordinates"), obs.get(side, {}).get("valid_mask")
        if not isinstance(coords, list) or not isinstance(masks, list) or len(coords) != len(names) or len(masks) != len(names):
            raise DataContractError("예측 좌표/mask shape 불일치")
        for p, valid in zip(coords, masks):
            if type(valid) is not bool or (valid and (not isinstance(p, list) or len(p) != 2 or not all(finite_number(v) for v in p))) or (not valid and p is not None):
                raise DataContractError("예측 좌표와 수치 유효 mask 불일치")
        for key in ("confidence", "visibility"):
            values = obs[side].get(key, [None] * len(names))
            if not isinstance(values, list) or len(values) != len(names):
                raise DataContractError(f"예측 {key} shape 불일치")
            for value in values:
                if value is not None and (not (type(value) is bool) if key == "visibility" else not (finite_number(value) and 0 <= value <= 1)):
                    raise DataContractError(f"잘못된 {key}")
    return obs


def pack_window(window: CausalWindow, contract: InputContract, predictions=None):
    if window.joint_names != contract.joint_names:
        raise DataContractError("Window/계약의 관절 순서 불일치")
    if contract.input_source == "gt_2d" and predictions is not None:
        raise DataContractError("GT 진단과 예측 입력 혼합 금지")
    doc = window.to_dict()
    t, j = len(window.frames), len(contract.joint_names)
    x, y = torch.zeros(t, j, 14), torch.zeros(t, j, 3)
    mask = torch.zeros(t, j, dtype=torch.bool)
    for index, frame in enumerate(window.frames):
        if frame is None:
            continue
        fields = {f.name: f for f in frame.fields}
        if "device_pts3d" not in fields:
            raise DataContractError("GT target field 누락")
        for k, (point, valid) in enumerate(zip(fields["device_pts3d"].coordinates, fields["device_pts3d"].valid_mask)):
            if valid:
                y[index, k] = torch.tensor(point) * MM_FACTOR[contract.distance_unit]
                mask[index, k] = True
        if contract.input_source == "predicted_2d":
            obs = _observation((predictions or {}).get((frame.sequence_id, frame.frame_index)), contract.joint_names)
            if obs.get("frontend_version") != contract.frontend_version or obs.get("frontend_checkpoint_sha256") != contract.frontend_checkpoint_sha256 or obs.get("preprocessing_version") != contract.preprocessing_version:
                raise DataContractError("예측 특징의 frontend/전처리 버전 불일치")
        else:
            obs = {side: {"coordinates": fields[f"camera_{side}_pts2d"].coordinates,
                "valid_mask": fields[f"camera_{side}_pts2d"].valid_mask} for side in ("left", "right")}
        for v, side in enumerate(("left", "right")):
            width, height = contract.image_sizes[v]
            view = obs[side]
            for k, (point, valid) in enumerate(zip(view["coordinates"], view["valid_mask"])):
                confidence = view.get("confidence", [None] * j)[k]
                visibility = view.get("visibility", [None] * j)[k]
                offset = v * 7
                if valid:
                    # 고정 영상 크기 기반. Sequence 통계/미래 입력/범위 clipping 사용 안 함.
                    x[index, k, offset:offset+2] = torch.tensor([2 * point[0] / (width-1) - 1, 2 * point[1] / (height-1) - 1])
                x[index, k, offset+2:offset+7] = torch.tensor([float(valid), confidence or 0., float(confidence is not None),
                    float(visibility) if visibility is not None else 0., float(visibility is not None)])
    if not torch.isfinite(x).all() or not torch.isfinite(y).all():
        raise DataContractError("float32 변환 이후 비유한 값")
    return x.flatten(1), torch.tensor(doc["time_valid"], dtype=torch.bool), y, mask


def load_document(doc, window_size):
    if not isinstance(doc, dict) or doc.get("schema_version") != "m1_features_v1":
        raise DataContractError("준비된 m1_features_v1 계약 필요; raw JSON 직접 학습 금지")
    contract = InputContract.from_dict(doc["contract"])
    manifest = doc.get("manifest", {})
    if not isinstance(manifest,dict) or not isinstance(manifest.get("id"), str) or not manifest["id"] or not manifest.get("dataset_version"):
        raise DataContractError("manifest id와 dataset_version 필요")
    records = doc.get("records")
    if not isinstance(records, list) or not records:
        raise DataContractError("비어 있지 않은 준비 record 목록 필요")
    owners, motion_owners, closed = {}, {}, set()
    previous_seq = None
    streams, packed, predictions = {}, {}, {}
    for r in records:
        if not isinstance(r,dict): raise DataContractError("record 사전 필요")
        seq, motion, split = r.get("sequence_id"), r.get("motion_id"), r.get("split")
        if not isinstance(seq, str) or not seq or not isinstance(motion, str) or not motion or split not in ("train", "validation", "test"):
            raise DataContractError("sequence/motion/split 식별 계약 필요")
        if owners.get(seq, split) != split or motion_owners.get(motion, split) != split:
            raise DataContractError("sequence 또는 원본 motion의 split 중복")
        owners[seq], motion_owners[motion] = split, split
        if seq != previous_seq:
            if seq in closed:
                raise DataContractError("Sequence record는 연속 묶음이어야 함; 자동 재정렬하지 않음")
            if previous_seq is not None:
                closed.add(previous_seq)
        previous_seq = seq
        index = r.get("frame_index")
        raw = {"frame": str(index), "joints": r.get("joints"), "timestamp": r.get("timestamp_raw")}
        if not isinstance(raw["joints"], dict) or not raw["joints"]:
            raise DataContractError("파싱된 joints 사전 필요")
        sample = FrameSample(f"{seq}/frame_{index}.json", seq, index, raw_document=raw)
        _validate_joints(sample, list(contract.joint_names))
        if seq not in streams:
            streams[seq] = CausalWindowStream(window_size, contract.joint_names)
        window = streams[seq].push(sample)
        if contract.input_source == "predicted_2d":
            predictions[(seq, index)] = r.get("prediction")
        features = pack_window(window, contract, predictions if contract.input_source == "predicted_2d" else None)
        packed.setdefault(split, []).append((features, (seq, index)))
    splits = {}
    for split, items in packed.items():
        tensors = [torch.stack([row[0][k] for row in items]) for k in range(4)]
        splits[split] = TensorSplit(*tensors, [row[1] for row in items])
    return contract, splits, content_hash(doc)


def synthetic_document():
    # 인위적인 좌표 대응이며 실제 camera projection이나 데이터 결과가 아니다.
    from rewind_data.audit import REQUIRED_FIELDS
    records = []
    for split in ("train", "validation"):
        for clip in range(2):
            for index in range(6):
                joints = {}
                for k, name in enumerate(("root", "tip")):
                    u, v = 8 + 4*index + k*5 + clip, 10 + 2*index + k*4
                    fields = {field: [0.] * dim for field, dim in REQUIRED_FIELDS.items()}
                    fields.update(camera_left_pts2d=[u, v], camera_right_pts2d=[u+3, v], device_pts3d=[u*1.5, v*2., u-v])
                    joints[name] = fields
                records.append({"sequence_id": f"{split}_{clip}", "motion_id": f"{split}_motion_{clip}", "split": split,
                    "frame_index": index, "joints": joints})
    return {"schema_version": "m1_features_v1", "contract": {"joint_names": ["root", "tip"], "distance_unit": "mm",
        "coordinate_frame": "device_relative", "image_sizes": [[64,48],[64,48]], "input_source": "gt_2d",
        "data_kind": "synthetic_fixture", "frontend_version": None, "preprocessing_version": "synthetic_pixel_v1"},
        "manifest": {"id": "synthetic_m1_v1", "dataset_version": "synthetic_fixture_v1"}, "records": records}
