"""원본 pose JSON을 보존하면서 수치 유효성과 시퀀스 이상을 검사한다."""

import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit

from . import __version__
from .skeleton import Skeleton

REQUIRED_FIELDS = {
    "camera_left_pts2d": 2,
    "camera_right_pts2d": 2,
    "camera_left_pts3d": 3,
    "camera_right_pts3d": 3,
    "device_pts3d": 3,
}
FRAME_NAME = re.compile(r"frame_([0-9]+)\.json\Z")
UNKNOWN_STATE = {k: "unknown" for k in (
    "units", "axes", "camera_transform_direction", "rotation_representation",
    "fps", "timestamp_semantics", "calibration", "split", "rgb_alignment",
    "visibility_semantics", "sequence_completeness", "dataset_version_verification",
)}


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def finite_number(value) -> bool:
    try:
        return type(value) in (int, float) and math.isfinite(value)
    except OverflowError:
        return False


def _pairs(pairs):
    obj = {}
    for key, value in pairs:
        if key in obj:
            raise ValueError(f"duplicate_json_key: {key}")
        obj[key] = value
    return obj


def _reject_constant(value):
    raise ValueError(f"nonfinite_json_constant: {value}")


def read_json(data: bytes):
    return json.loads(data.decode("utf-8-sig"), object_pairs_hook=_pairs, parse_constant=_reject_constant)


def canonical_drive_uri(uri: str | None) -> str | None:
    if uri is None:
        return None
    p = urlsplit(uri)
    if p.scheme != "https" or p.netloc != "drive.google.com" or not re.fullmatch(
        r"/(?:file/d|drive/folders)/[A-Za-z0-9_-]+(?:/view)?", p.path
    ):
        raise ValueError("출처에는 canonical Drive 파일/폴더 URL만 허용합니다.")
    return f"https://drive.google.com{p.path}"


def load_provenance(path: str | Path | None) -> dict:
    if path is None:
        return {"data_kind": "unknown", "files": {}}
    doc = read_json(Path(path).read_bytes())
    if not isinstance(doc, dict) or doc.get("data_kind") not in ("dataset_sample", "synthetic_fixture", "unknown"):
        raise ValueError("Provenance data_kind는 dataset_sample/synthetic_fixture/unknown 중 하나여야 합니다.")
    files = doc.get("files", {})
    if not isinstance(files, dict):
        raise ValueError("Provenance files는 상대 경로 → Drive URL 사전이어야 합니다.")
    doc["files"] = {key: canonical_drive_uri(uri) for key, uri in files.items()}
    doc["source_root_uri"] = canonical_drive_uri(doc.get("source_root_uri"))
    inventories = doc.get("observed_folder_inventory", {})
    if not isinstance(inventories, dict):
        raise ValueError("observed_folder_inventory는 sequence → frame 번호 목록 사전이어야 합니다.")
    for seq, numbers in inventories.items():
        if not isinstance(numbers, list) or any(type(n) is not int or n < 0 for n in numbers) or len(set(numbers)) != len(numbers):
            raise ValueError(f"잘못된 폴더 inventory: {seq}")
    # 자유 형식 metadata/인증 참조를 산출물에 복사하지 않는다.
    return {k: doc[k] for k in ("data_kind", "files", "source_root_uri", "observed_folder_inventory") if k in doc}


@dataclass
class FrameSample:
    source_rel_path: str
    sequence_id: str
    frame_index: int | None
    source_uri: str | None = None
    file_sha256: str | None = None
    size_bytes: int | None = None
    raw_document: dict | None = None
    values: dict = field(default_factory=dict)
    field_valid_mask: dict = field(default_factory=dict)
    issues: list[dict] = field(default_factory=list)

    def issue(self, code: str, location: str, detail: str, severity: str = "error"):
        self.issues.append({"severity": severity, "code": code, "file": self.source_rel_path,
                            "location": location, "detail": detail})

    @property
    def annotation_valid_mask(self) -> dict[str, bool]:
        """모든 필수 field의 수치 유효성. Visibility와 독립적이다."""
        return {name: all(fields.values()) for name, fields in self.field_valid_mask.items()}

    def select(self, names: list[str], field_name: str = "device_pts3d") -> dict:
        if field_name not in REQUIRED_FIELDS:
            raise ValueError(f"미지원 field: {field_name}")
        return {
            "joint_names": list(names),
            "coordinates": [self.values.get(n, {}).get(field_name) for n in names],
            "valid_mask": [self.field_valid_mask.get(n, {}).get(field_name, False) for n in names],
            "visibility": {"left": [None] * len(names), "right": [None] * len(names)},
            "confidence": {"left": [None] * len(names), "right": [None] * len(names)},
            "input_source": "gt_annotation",
            "coordinate_field": field_name,
            "units": "unknown",
        }


@dataclass
class SequenceSamples:
    sequence_id: str
    frames: list[FrameSample]
    joint_names: list[str]
    issues: list[dict]
    missing_frame_ranges: list[list[int]]


@dataclass
class AuditResult:
    sequences: list[SequenceSamples]
    manifest: dict
    report: dict

    def require_clean(self):
        if self.report["summary"]["error_count"]:
            raise ValueError("감사 오류가 있습니다. audit_report.json의 파일·관절 위치를 확인하세요.")
        return self


def _load_frame(path: Path, root: Path, provenance: dict) -> FrameSample:
    rel = path.relative_to(root).as_posix()
    match = FRAME_NAME.fullmatch(path.name)
    sample = FrameSample(rel, path.parent.relative_to(root).as_posix(), int(match[1]) if match else None,
                         provenance.get("files", {}).get(rel))
    if match is None:
        sample.issue("unexpected_filename", "$filename", "frame_<비음수 정수>.json 이름이 필요합니다.")
    if path.is_symlink() or not path.resolve().is_relative_to(root.resolve()):
        sample.issue("unsafe_source_path", "$file", "심볼릭 링크/입력 root 밖의 파일은 읽지 않습니다.")
        return sample
    try:
        data = path.read_bytes()
        sample.file_sha256, sample.size_bytes = sha256(data), len(data)
        doc = read_json(data)
    except (OSError, ValueError, UnicodeError) as exc:
        sample.issue("invalid_json_or_io", "$", str(exc))
        return sample
    if not isinstance(doc, dict) or not isinstance(doc.get("joints"), dict) or not doc["joints"]:
        sample.issue("invalid_schema", "$.joints", "비어 있지 않은 관절 이름 사전이 필요합니다.")
        return sample
    sample.raw_document = doc  # 모든 관절과 camera/device 필드를 그대로 보존.
    frame = doc.get("frame")
    frame_ok = (type(frame) is int and frame >= 0) or (isinstance(frame, str) and re.fullmatch(r"[0-9]+", frame))
    if not frame_ok or int(frame) != sample.frame_index:
        sample.issue("frame_id_mismatch", "$.frame", f"파일 번호 {sample.frame_index}와 frame 값 {frame!r} 불일치")
    for key in ("camera_left", "camera_right", "device"):
        if not isinstance(doc.get(key), dict):
            sample.issue("missing_pose_metadata", f"$.{key}", "카메라/device pose 사전 없음", "warning")
    return sample


def _validate_joints(sample: FrameSample, expected: list[str]):
    if sample.raw_document is None:
        return
    joints = sample.raw_document["joints"]
    for name in expected:
        entry = joints.get(name)
        sample.values[name], sample.field_valid_mask[name] = {}, {}
        location = f"$.joints[{json.dumps(name, ensure_ascii=False)}]"
        if not isinstance(entry, dict):
            sample.issue("missing_joint" if name not in joints else "invalid_joint", location,
                         "관절 annotation 사전이 없습니다.")
        for key, dim in REQUIRED_FIELDS.items():
            value = entry.get(key) if isinstance(entry, dict) else None
            valid = isinstance(value, list) and len(value) == dim
            if valid:
                valid = all(finite_number(v) for v in value)
            sample.field_valid_mask[name][key] = valid
            sample.values[name][key] = list(value) if valid else None
            if not valid and isinstance(entry, dict):
                sample.issue("missing_field" if key not in entry else "invalid_vector", f"{location}.{key}",
                             f"유한한 숫자 {dim}개 배열 필요 (bool/null/중첩 배열 불허)")


def code_hashes() -> dict[str, str]:
    base = Path(__file__).parent
    return {p.name: sha256(p.read_bytes()) for p in sorted(base.iterdir()) if p.suffix in (".py", ".html")}


def audit_dataset(root: str | Path, skeleton: Skeleton, dataset_version: str = "unknown",
                  provenance: dict | None = None) -> AuditResult:
    root = Path(root)
    if not root.is_dir():
        raise ValueError("입력 root가 디렉터리가 아닙니다.")
    provenance = provenance or {"data_kind": "unknown", "files": {}}
    grouped = defaultdict(list)
    for path in sorted(root.rglob("*.json")):
        sample = _load_frame(path, root, provenance)
        grouped[sample.sequence_id].append(sample)
    sequences, all_issues, records = [], [], []
    if not grouped:
        all_issues.append({"severity": "error", "code": "empty_dataset", "file": None,
                           "location": "$root", "detail": "JSON 파일을 찾지 못했습니다."})
    for seq_id, frames in sorted(grouped.items()):
        frames.sort(key=lambda f: (f.frame_index is None, f.frame_index or 0, f.source_rel_path))
        observed_names = sorted({n for f in frames if f.raw_document for n in f.raw_document["joints"]})
        expected = list(dict.fromkeys(observed_names + skeleton.joint_names))
        indices = Counter(f.frame_index for f in frames if f.frame_index is not None)
        seq_issues = []
        for frame, count in indices.items():
            if count > 1:
                seq_issues.append({"severity": "error", "code": "duplicate_frame", "file": None,
                                   "sequence": seq_id, "location": "$filename", "frame_index": frame,
                                   "files": [f.source_rel_path for f in frames if f.frame_index == frame],
                                   "detail": f"같은 숫자 frame 번호의 파일 {count}개"})
        ordered = sorted(indices)
        gaps = [[a + 1, b - 1] for a, b in zip(ordered, ordered[1:]) if b > a + 1]
        if gaps:
            seq_issues.append({"severity": "error", "code": "missing_frames", "file": None,
                               "sequence": seq_id, "location": "$filename", "ranges": gaps,
                               "detail": "관측된 최소·최대 frame 사이의 누락 (시퀀스 앞/뒤 완전성은 미확인)"})
        inventory = provenance.get("observed_folder_inventory", {}).get(seq_id)
        if inventory is not None and set(inventory) != set(indices):
            seq_issues.append({"severity": "error", "code": "folder_inventory_mismatch", "file": None,
                               "sequence": seq_id, "location": "$filename",
                               "missing": sorted(set(inventory) - set(indices)),
                               "extra": sorted(set(indices) - set(inventory)),
                               "detail": "관측한 Drive 폴더 목록과 로컬 파일 범위 불일치"})
        for f in frames:
            _validate_joints(f, expected)
            all_issues.extend(f.issues)
            doc = f.raw_document or {}
            records.append({"source_rel_path": f.source_rel_path, "source_uri": f.source_uri,
                            "sha256": f.file_sha256, "size_bytes": f.size_bytes,
                            "sequence_id": seq_id, "frame_index": f.frame_index,
                            "parse_ok": f.raw_document is not None,
                            "numerically_valid": f.raw_document is not None and not any(i["severity"] == "error" for i in f.issues),
                            "raw_joint_names": list(doc.get("joints", {})),
                            "field_valid_mask": f.field_valid_mask,
                            "annotation_valid_mask": f.annotation_valid_mask,
                            "visibility": "unknown", "confidence": "unknown",
                            "timestamp_raw": doc.get("timestamp") if type(doc.get("timestamp")) in (int, float, str) and
                            (not isinstance(doc.get("timestamp"), float) or math.isfinite(doc["timestamp"])) else None,
                            "timestamp_semantics": "unknown", "pose_metadata_present": {k: isinstance(doc.get(k), dict) for k in ("camera_left", "camera_right", "device")},
                            "error_count": sum(i["severity"] == "error" for i in f.issues)})
        all_issues.extend(seq_issues)
        sequences.append(SequenceSamples(seq_id, frames, observed_names, seq_issues, gaps))
    counts = Counter(i["severity"] for i in all_issues)
    summary = {"sequence_count": len(sequences), "file_count": len(records),
               "parsed_file_count": sum(f["parse_ok"] for f in records),
               "numerically_valid_file_count": sum(f["numerically_valid"] for f in records),
               "error_count": counts["error"], "warning_count": counts["warning"],
               "status": "errors" if counts["error"] else "numeric_checks_passed_semantics_unverified"}
    manifest = {"schema_version": "0.1.0", "preprocess_version": __version__,
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "dataset_version_declared": dataset_version, "data_kind": provenance.get("data_kind", "unknown"),
                "source_root_uri": provenance.get("source_root_uri"), "verification_state": dict(UNKNOWN_STATE),
                "coordinate_transform": "none; original annotation arrays",
                "input_source": "gt_annotation", "required_field_dimensions": REQUIRED_FIELDS,
                "skeleton": skeleton.document, "skeleton_file_sha256": skeleton.sha256,
                "code_file_sha256": code_hashes(), "observed_folder_inventory": provenance.get("observed_folder_inventory", {}),
                "sequence_rule": "one parent directory per sequence; no windows/padding/interpolation",
                "files": records}
    report = {"schema_version": "0.1.0", "summary": summary, "verification_state": dict(UNKNOWN_STATE),
              "issues": all_issues, "sequences": [{"sequence_id": s.sequence_id,
                  "frame_indices": [f.frame_index for f in s.frames], "joint_names": s.joint_names,
                  "missing_frame_ranges": s.missing_frame_ranges,
                  "observed_range_only": True} for s in sequences]}
    return AuditResult(sequences, manifest, report)
