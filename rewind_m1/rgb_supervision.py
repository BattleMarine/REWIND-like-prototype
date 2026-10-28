"""검증된 GT 특징 문서와 로컬 RGB를 키로 연결하는 소형 lazy API."""
import copy
import re

import torch

from rewind_data.audit import FrameSample, _validate_joints
from .data import DataContractError, content_hash, load_document
from .local_rgb import LocalStereoRGB
from .prediction_join import PROVENANCE, _identity, _records, _serialized


class LocalRGBSupervision:
    def __init__(self, loader, pose_doc, *, expected_manifest_sha256, joint_names, split):
        if not isinstance(loader, LocalStereoRGB):
            raise DataContractError("LocalStereoRGB 필요")
        if split not in ("train", "validation", "test"):
            raise DataContractError("명시적 단일 split 필요")
        if not isinstance(expected_manifest_sha256, str) or not re.fullmatch(r"[0-9a-f]{64}", expected_manifest_sha256) or loader.manifest_sha256 != expected_manifest_sha256:
            raise DataContractError("RGB manifest hash 불일치")
        _serialized(pose_doc)
        records = _records(pose_doc)
        if pose_doc.get("schema_version") != "m1_features_v1" or not isinstance(pose_doc.get("contract"), dict) or pose_doc["contract"].get("input_source") != "gt_2d":
            raise DataContractError("GT m1_features_v1 필요")
        manifest = pose_doc.get("manifest")
        if not isinstance(manifest, dict) or any(k in manifest for k in (PROVENANCE, "prediction_cache_sha256", "prediction_cache_content_sha256")) or any("prediction" in r for r in records if isinstance(r, dict)):
            raise DataContractError("예측 문서 또는 재결합 출처 금지")
        keys = [_identity(r) for r in records]
        if len(set(keys)) != len(keys) or len(loader) != len(records):
            raise DataContractError("RGB/pose 중복 또는 record 수 불일치")
        contract, _, _ = load_document(pose_doc, 1)
        if not isinstance(joint_names, list) or joint_names != list(contract.joint_names):
            raise DataContractError("명시적 관절 순서 불일치")
        rgb_rows = [loader.metadata(i) for i in range(len(loader))]
        rgb_indices = {_identity(m["record"]): i for i, m in enumerate(rgb_rows)}
        if len(rgb_indices) != len(rgb_rows) or set(rgb_indices) != set(keys):
            raise DataContractError("RGB/pose key 누락 또는 잉여")
        for key, row in zip(keys, records):
            meta = rgb_rows[rgb_indices[key]]
            sizes = [meta["record"]["views"][side]["image_size"] for side in ("left", "right")]
            if meta["data_kind"] != contract.data_kind or sizes != [list(s) for s in contract.image_sizes] or meta["record"]["split"] != row["split"]:
                raise DataContractError("RGB/pose data_kind, 영상 크기 또는 split 불일치")
        selected = [(r, rgb_indices[k]) for r, k in zip(records, keys) if r["split"] == split]
        if not selected:
            raise DataContractError("선택 split이 비어 있음")
        self._loader = loader
        # loader가 실제 적용하는 상한의 읽기만 사용한다. 공개 limits 접근자는 없다.
        self._limits = loader._limits
        self._contract = contract
        self._pose_hash = content_hash(pose_doc)
        self._rows = copy.deepcopy(selected)
        self._rgb_metadata = copy.deepcopy(rgb_rows)

    def __len__(self):
        return len(self._rows)

    def _index(self, index):
        if type(index) is not int or not 0 <= index < len(self):
            raise DataContractError("supervision index 범위 오류")
        return index

    def metadata(self, index):
        row, rgb_index = self._rows[self._index(index)]
        return {"pose_record": copy.deepcopy(row), "rgb": copy.deepcopy(self._rgb_metadata[rgb_index]),
            "pose_document_sha256": self._pose_hash, "joint_names": list(self._contract.joint_names),
            "image_sizes": [list(s) for s in self._contract.image_sizes], "preprocessing_version": "rgb_0_1_v1",
            "visibility": {side: [None] * len(self._contract.joint_names) for side in ("left", "right")}}

    def load_batch(self, indices):
        if not isinstance(indices, (list, tuple)) or not 1 <= len(indices) <= min(4, self._limits.max_batch_size):
            raise DataContractError("supervision batch 크기 상한")
        indices = [self._index(i) for i in indices]
        width, height = self._contract.image_sizes[0]
        joints = len(self._contract.joint_names)
        total_bytes = len(indices) * (2 * 3 * height * width * 4 + 2 * joints * (2 * 4 + 1))
        if total_bytes > self._limits.max_batch_tensor_bytes:
            raise DataContractError("RGB+GT 합산 tensor byte 상한")
        rgb_indices = [self._rows[i][1] for i in indices]
        batch = self._loader.load_batch(rgb_indices)
        xy = torch.zeros((len(indices), 2, joints, 2), dtype=torch.float32, device="cpu")
        valid = torch.zeros((len(indices), 2, joints), dtype=torch.bool, device="cpu")
        for b, index in enumerate(indices):
            row = self._rows[index][0]
            sample = FrameSample("supervision", row["sequence_id"], row["frame_index"], raw_document={"joints": row["joints"]})
            _validate_joints(sample, list(self._contract.joint_names))
            for v, side in enumerate(("left", "right")):
                field = f"camera_{side}_pts2d"
                for j, name in enumerate(self._contract.joint_names):
                    if sample.field_valid_mask[name][field]:
                        xy[b, v, j] = torch.tensor(sample.values[name][field], dtype=torch.float32, device="cpu")
                        valid[b, v, j] = True
        if not torch.isfinite(xy).all():
            raise DataContractError("float32 GT2D 변환 비유한 값")
        return {"images": batch["images"], "gt_xy": xy, "annotation_valid": valid,
                "metadata": [self.metadata(i) for i in indices]}
