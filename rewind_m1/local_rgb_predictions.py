"""고정 front-end weights로 로컬 stereo RGB의 작은 예측 cache를 만든다."""
import argparse
import json
import re
from pathlib import Path

import torch

from .data import DataContractError, content_hash
from .frontend import StereoHeatmapNet, prediction_cache, weights_hash
from .local_rgb import LocalStereoRGB


MAX_RECORDS = 256
MAX_OUTPUT_BYTES = 4 * 1024**2
PREPROCESSING = "rgb_0_1_v1"


def validate_contract(checkpoint, loader, joint_names):
    if not isinstance(checkpoint, dict) or checkpoint.get("format_version") != "m1_frontend_weights_v1" or checkpoint.get("weights_only") is not True:
        raise DataContractError("frontend weights 형식 오류")
    config = checkpoint.get("model_config")
    if not isinstance(joint_names, list) or not joint_names or any(not isinstance(n, str) or not n for n in joint_names) or len(set(joint_names)) != len(joint_names):
        raise DataContractError("명시적 관절 순서 필요")
    if config != {"architecture":"small_cnn_v1", "joints":len(joint_names), "hidden_channels":16, "stride":2}:
        raise DataContractError("frontend 모델 설정 불일치")
    info = loader.metadata(0)
    size = info["record"]["views"]["left"]["image_size"]
    if (checkpoint.get("joint_names") != joint_names or checkpoint.get("image_sizes") != [size, size]
            or checkpoint.get("data_kind") != info["data_kind"] or checkpoint.get("preprocessing_version") != PREPROCESSING):
        raise DataContractError("frontend 입력 계약 불일치")
    if not isinstance(checkpoint.get("frontend_version"), str) or not checkpoint["frontend_version"].strip():
        raise DataContractError("frontend 버전 누락")
    for key in ("frontend_checkpoint_sha256", "manifest_hash"):
        if not isinstance(checkpoint.get(key), str) or not re.fullmatch(r"[0-9a-f]{64}", checkpoint[key]):
            raise DataContractError("frontend 출처 hash 오류")


def load_weights(path, loader, joint_names):
    """신뢰할 수 있는 weights-only 파일과 명시한 입력 계약을 대조한다."""
    checkpoint = torch.load(Path(path), map_location="cpu", weights_only=True)
    validate_contract(checkpoint, loader, joint_names)
    model = StereoHeatmapNet(len(joint_names))
    model.load_state_dict(checkpoint["model"], strict=True)
    if weights_hash(model) != checkpoint["frontend_checkpoint_sha256"]:
        raise DataContractError("frontend weights fingerprint 불일치")
    return model, checkpoint


def build_cache(loader, model, checkpoint, joint_names, batch_size=4, max_records=MAX_RECORDS, max_bytes=MAX_OUTPUT_BYTES):
    validate_contract(checkpoint, loader, joint_names)
    if not isinstance(model, StereoHeatmapNet) or model.joints != len(checkpoint["joint_names"]):
        raise DataContractError("frontend 모델과 관절 계약 불일치")
    if type(batch_size) is not int or not 1 <= batch_size <= 4:
        raise DataContractError("batch 크기는 1~4 필요")
    if type(max_records) is not int or not 1 <= max_records <= MAX_RECORDS or len(loader) > max_records:
        raise DataContractError("예측 record 상한 초과")
    if type(max_bytes) is not int or not 1 <= max_bytes <= MAX_OUTPUT_BYTES:
        raise DataContractError("예측 cache byte 상한 오류")
    before = weights_hash(model)
    if before != checkpoint["frontend_checkpoint_sha256"]:
        raise DataContractError("frontend weights fingerprint 불일치")
    mode = model.training
    records, splits, sources = [], [], []
    body = None
    try:
        model.eval()
        with torch.no_grad():
            start = 0
            while start < len(loader):
                split = loader.metadata(start)["record"]["split"]
                end = start + 1
                while end < len(loader) and end-start < batch_size and loader.metadata(end)["record"]["split"] == split:
                    end += 1
                batch = loader.load_batch(list(range(start, end)))
                identities = [(m["record"]["sequence_id"], m["record"]["frame_index"]) for m in batch["metadata"]]
                partial = prediction_cache(model, batch["images"], identities, checkpoint["joint_names"],
                    checkpoint["frontend_version"], checkpoint["preprocessing_version"], loader.manifest_sha256, checkpoint["data_kind"])
                if body is None:
                    body = {k:v for k,v in partial.items() if k not in ("records", "content_sha256")}
                    body["checkpoint_source_manifest_hash"] = checkpoint["manifest_hash"]
                records.extend(partial["records"])
                splits.extend([split] * len(identities))
                sources.extend([m["record"]["views"] for m in batch["metadata"]])
                body["records"] = records
                body["record_splits"] = splits
                body["record_sources"] = sources
                # 이전 기록도 포함해 출력 크기를 제한한다. 최종 hash 공간도 예약한다.
                if len(json.dumps(body, ensure_ascii=True, separators=(",", ":")).encode("utf-8")) + 96 > max_bytes:
                    raise DataContractError("예측 cache byte 상한 초과")
                start = end
        result = {**body, "content_sha256":content_hash(body)}
        if len(json.dumps(result, ensure_ascii=True, separators=(",", ":")).encode("utf-8")) > max_bytes:
            raise DataContractError("예측 cache byte 상한 초과")
        return result
    finally:
        model.train(mode)
        if weights_hash(model) != before:
            raise DataContractError("추론 중 frontend weights 변경")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--joints", nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args(argv)
    loader = LocalStereoRGB(args.manifest, args.root)
    model, checkpoint = load_weights(args.weights, loader, args.joints)
    cache = build_cache(loader, model, checkpoint, args.joints, args.batch_size)
    raw = (json.dumps(cache, ensure_ascii=True, separators=(",", ":")) + "\n").encode("utf-8")
    if len(raw) > MAX_OUTPUT_BYTES:
        raise DataContractError("예측 cache byte 상한 초과")
    created = False
    try:
        with args.output.open("xb") as output:
            created = True
            output.write(raw)
    except BaseException:
        if created:
            args.output.unlink(missing_ok=True)
        raise
    print(json.dumps({"records":len(cache["records"]), "output":str(args.output), "content_sha256":cache["content_sha256"]}))


if __name__ == "__main__":
    main()
