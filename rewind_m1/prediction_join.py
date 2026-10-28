"""명시적 RGB manifest를 기준으로 GT 특징 문서에 예측 2D를 결합한다."""

import argparse
import copy
import json
import re
from pathlib import Path

from rewind_data.audit import read_json
from .data import DataContractError, InputContract, content_hash, load_document
from .frontend import read_prediction_cache


MAX_BYTES = 4 * 1024**2
MAX_RECORDS = 256
PROVENANCE = "prediction_join_provenance"


def _serialized(doc):
    """직렬화된 JSON 상한은 파일과 직접 API 모두에 적용한다."""
    encoder = json.JSONEncoder(ensure_ascii=True, allow_nan=False, separators=(",", ":"))
    parts, size = [], 0
    try:
        for part in encoder.iterencode(doc):
            size += len(part.encode("utf-8"))
            if size > MAX_BYTES:
                raise DataContractError("JSON byte 상한 초과")
            parts.append(part)
    except (TypeError, ValueError, RecursionError) as exc:
        raise DataContractError("JSON 직렬화 불가") from exc
    return "".join(parts).encode("utf-8")


def _records(doc):
    if not isinstance(doc, dict) or not isinstance(doc.get("records"), list) or not 1 <= len(doc["records"]) <= MAX_RECORDS:
        raise DataContractError("record 수 또는 형식 오류")
    return doc["records"]


def _identity(record):
    if not isinstance(record, dict):
        raise DataContractError("record 사전 필요")
    seq, frame = record.get("sequence_id"), record.get("frame_index")
    if not isinstance(seq, str) or not seq or type(frame) is not int or frame < 0:
        raise DataContractError("sequence/frame 식별 오류")
    return seq, frame


def join_predictions(pose_doc, cache_doc, expected_manifest_sha256):
    """원본을 보존하고 복사본의 입력 계약을 예측 2D로 변환한다."""
    _serialized(pose_doc)
    _serialized(cache_doc)
    pose_records, cache_records = _records(pose_doc), _records(cache_doc)
    if not isinstance(expected_manifest_sha256, str) or not re.fullmatch(r"[0-9a-f]{64}", expected_manifest_sha256):
        raise DataContractError("명시적 RGB manifest SHA-256 필요")
    if pose_doc.get("schema_version") != "m1_features_v1":
        raise DataContractError("m1_features_v1 GT 문서 필요")
    if not isinstance(pose_doc.get("contract"), dict) or pose_doc["contract"].get("input_source") != "gt_2d":
        raise DataContractError("GT 입력에서만 결합 가능")
    manifest = pose_doc.get("manifest")
    if not isinstance(manifest, dict) or PROVENANCE in manifest or any(k in manifest for k in ("rgb_manifest_sha256", "prediction_cache_sha256", "prediction_cache_content_sha256", "input_document_sha256")):
        raise DataContractError("이미 결합된 출처 또는 잘못된 manifest")
    if any(not isinstance(row, dict) or "prediction" in row for row in pose_records):
        raise DataContractError("기존 prediction 또는 잘못된 record")
    pose_keys = [_identity(row) for row in pose_records]
    cache_keys = [_identity(row) for row in cache_records]
    if len(set(pose_keys)) != len(pose_keys) or len(set(cache_keys)) != len(cache_keys):
        raise DataContractError("중복 sequence/frame")
    load_document(pose_doc, 1)

    result = copy.deepcopy(pose_doc)
    contract_data = result["contract"]
    contract_data.update(input_source="predicted_2d", frontend_version=cache_doc.get("frontend_version"),
                         frontend_checkpoint_sha256=cache_doc.get("frontend_checkpoint_sha256"),
                         preprocessing_version=cache_doc.get("preprocessing_version"))
    contract = InputContract.from_dict(contract_data)
    predictions = read_prediction_cache(cache_doc, contract, expected_manifest_sha256)
    if set(pose_keys) != set(cache_keys):
        raise DataContractError("pose와 예측의 누락 또는 잉여 frame")
    if "record_splits" in cache_doc:
        splits = cache_doc["record_splits"]
        if not isinstance(splits, list) or len(splits) != len(cache_keys) or any(s not in ("train", "validation", "test") for s in splits):
            raise DataContractError("cache split 형식 오류")
        by_key = dict(zip(cache_keys, splits))
        if any(by_key[key] != row["split"] for key, row in zip(pose_keys, pose_records)):
            raise DataContractError("pose/cache split 불일치")
    for row, key in zip(result["records"], pose_keys):
        row["prediction"] = copy.deepcopy(predictions[key])
    result["manifest"][PROVENANCE] = {"rgb_manifest_sha256": expected_manifest_sha256,
        "prediction_cache_sha256": cache_doc["content_sha256"], "input_document_sha256": content_hash(pose_doc)}
    _serialized(result)
    load_document(result, 1)
    return result


def _read_file(path):
    with Path(path).open("rb") as handle:
        raw = handle.read(MAX_BYTES + 1)
    if not raw or len(raw) > MAX_BYTES:
        raise DataContractError("입력 파일 byte 상한 오류")
    return read_json(raw)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    result = join_predictions(_read_file(args.features), _read_file(args.cache), args.manifest_sha256)
    raw = _serialized(result)
    created = False
    try:
        with args.output.open("xb") as handle:
            created = True
            handle.write(raw)
    except BaseException:
        if created:
            args.output.unlink(missing_ok=True)
        raise
    print(json.dumps({"records": len(result["records"]), "output": str(args.output), "content_sha256": content_hash(result)}))


if __name__ == "__main__":
    main()
