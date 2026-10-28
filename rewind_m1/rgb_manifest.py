"""검증된 소량 PNG 추출본에서 local_stereo_rgb_v1을 발급한다. 학습 승인 없음."""
import argparse
import copy
import os
import time
from pathlib import Path

from rewind_archive.contracts import digest, validate_manifest
from rewind_archive.extraction import code_hashes
from rewind_archive.matching import identify, verify_extraction, contiguous_runs
from rewind_data.audit import read_json
from .local_rgb import LocalStereoRGB, RGBLimits, _no_link, _path, _regular_read, _text
from .prediction_join import MAX_BYTES, _serialized
from .training import source_identity


MAX_FILES = 512
MAX_PAYLOAD_BYTES = 32 * 1024**2
MAX_ELAPSED_SECONDS = 30


def _safe_existing(path):
    path = Path(path).absolute()
    for component in (*reversed(path.parents), path):
        _no_link(component)
    return path.resolve(strict=True)


def _selection(selection):
    _serialized(selection)
    if not isinstance(selection, list) or not selection:
        raise ValueError("명시적 selection 목록 필요")
    keys, rows, sequences = set(), [], set()
    for item in selection:
        if not isinstance(item, dict) or set(item) != {"sequence_id", "frames", "split", "split_evidence"}:
            raise ValueError("selection 필드 오류")
        seq = _path(item["sequence_id"])
        if seq in sequences or item["split"] not in ("train", "validation", "test", "unknown") or not _text(item["split_evidence"]):
            raise ValueError("중복 sequence 또는 split 선언 오류")
        sequences.add(seq)
        frames = item["frames"]
        if not isinstance(frames, list) or not frames or any(type(f) is not int or not 0 <= f < 10**12 for f in frames) or any(b <= a for a,b in zip(frames, frames[1:])):
            raise ValueError("frames는 중복 없는 증가 정수 목록 필요")
        for frame in frames:
            keys.add((seq, frame))
            rows.append({"sequence_id":seq, "frame_index":frame, "split":item["split"], "split_evidence":item["split_evidence"]})
    if len(keys) > 256:
        raise ValueError("선택 frame 256개 상한")
    return keys, rows


def build_manifest(root, extractions, selection, image_size, output, *, manifest_id, allow_code_change=False):
    root = _safe_existing(root)
    if not root.is_dir():
        raise ValueError("공통 root 디렉터리 필요")
    output = Path(output).absolute()
    _safe_existing(output.parent)
    resolved_output = output.resolve(strict=False)
    if resolved_output.is_relative_to(root) or root.is_relative_to(resolved_output):
        raise ValueError("출력과 입력 root 겹침 금지")
    if not isinstance(extractions, (list, tuple)) or not 1 <= len(extractions) <= 4 or type(allow_code_change) is not bool:
        raise ValueError("추출본 목록 최대4개/명시 bool 옵션 필요")
    output.mkdir(exist_ok=False)
    pending, final = output/"manifest.pending.json", output/"manifest.json"
    started = time.monotonic()
    report = {"schema_version":"rgb_manifest_build_v1", "status":"rejected", "train_ready":False,
        "rgb_pose_alignment":"not_checked", "training_release":"not_implemented", "allow_code_change":allow_code_change,
        "sources":[], "issues":[], "unselected":[], "selection_sha256":None}

    def deadline():
        if time.monotonic()-started > MAX_ELAPSED_SECONDS:
            raise ValueError("builder cooperative deadline 초과")

    try:
        selected, rows = _selection(selection)
        report["selection_sha256"] = digest(selection)
        if not _text(manifest_id,128) or not isinstance(image_size,list) or len(image_size) != 2 or any(type(v) is not int or v < 4 for v in image_size):
            raise ValueError("명시 manifest id/image_size 필요")
        rgb_limits = RGBLimits()
        pixels = image_size[0]*image_size[1]
        if pixels > rgb_limits.max_pixels_per_view or pixels*2*3*4 > rgb_limits.max_batch_tensor_bytes:
            raise ValueError("PNG pixel/단일 tensor 상한")
        prepared, metadata_bytes, payload_bytes, files_count = [], 0, 0, 0
        seen_paths, source_versions, source_kinds = set(), set(), set()
        for path in extractions:
            deadline()
            path = _safe_existing(path)
            path = _safe_existing(path/"extraction_manifest.json") if path.is_dir() else path
            if not path.is_relative_to(root) or path in seen_paths:
                raise ValueError("추출본은 공통 root 아래 고유 경로여야 함")
            seen_paths.add(path)
            raw = _regular_read(path, MAX_BYTES-metadata_bytes)
            metadata_bytes += len(raw)
            doc = read_json(raw)
            if not isinstance(doc,dict) or not isinstance(doc.get("files"),list):
                raise ValueError("extraction files 목록 필요")
            if doc.get("schema_version") != "archive_extraction_v1" or doc.get("content_sha256") != digest({k:v for k,v in doc.items() if k != "content_sha256"}):
                raise ValueError("extraction schema/content hash 불일치")
            archive = validate_manifest(doc["archive_manifest"])
            if archive["kind"] != "rgb" or digest(archive) != doc["archive_manifest_sha256"]:
                raise ValueError("RGB archive kind/hash 불일치")
            source_versions.add(archive["dataset_version"])
            source_kinds.add(archive["source_scope"] == "synthetic_fixture")
            if len(source_versions) != 1 or len(source_kinds) != 1:
                raise ValueError("source dataset_version/data_kind 불일치")
            for entry in doc["files"]:
                files_count += 1
                if files_count > MAX_FILES or not isinstance(entry,dict) or type(entry.get("size_bytes")) is not int or not 1 <= entry["size_bytes"] <= 8*1024**2:
                    raise ValueError("추출 파일 수/개별 byte 상한")
                payload_bytes += entry["size_bytes"]
                if payload_bytes > MAX_PAYLOAD_BYTES:
                    raise ValueError("추출 payload 누적32MiB 상한")
                file = _safe_existing(path.parent/_path(entry["relative_path"]))
                if not file.is_relative_to(root):
                    raise ValueError("추출 파일 root 이탈")
            prepared.append((path,doc))
        report.update(metadata_bytes=metadata_bytes,payload_bytes=payload_bytes,files_count=files_count)
        current_code = code_hashes()
        verified = []
        for index, (path, original) in enumerate(prepared):
            deadline()
            source_id = f"source_{index}"
            report["sources"].append({"source_id":source_id,"path":path.relative_to(root).as_posix(),
                "original_code_sha256":original.get("code_sha256"),"current_code_sha256":current_code,
                "code_matches":original.get("code_sha256") == current_code})
            doc = verify_extraction(path, allow_code_change=allow_code_change)
            # 사전 상한을 검사한 보고서가 검증 중 바뀌지 않았는지 확인한다.
            if doc != original:
                raise ValueError("검증 중 extraction report 변경")
            if doc["archive_manifest"]["kind"] != "rgb":
                raise ValueError("RGB extraction만 허용")
            verified.append((source_id,path,doc))
            deadline()
        versions = {d["archive_manifest"]["dataset_version"] for _,_,d in verified}
        kinds = {"synthetic_fixture" if d["archive_manifest"]["source_scope"] == "synthetic_fixture" else "dataset_sample" for _,_,d in verified}
        if len(versions) != 1 or len(kinds) != 1:
            raise ValueError("source dataset_version/data_kind 불일치")
        by_key, sources = {}, {}
        for source_id,path,doc in verified:
            sources[source_id] = {"archive_manifest":copy.deepcopy(doc["archive_manifest"]), "extraction_manifest_sha256":doc["content_sha256"]}
            report["sources"][int(source_id.split("_")[-1])].update(extraction_manifest_sha256=doc["content_sha256"],
                whole_archive_integrity_verified=doc["whole_archive_integrity_verified"])
            for entry in doc["files"]:
                seq, frame, side = identify(entry["member"], "rgb")
                views = by_key.setdefault((seq,frame),{})
                if side in views:
                    report["issues"].append({"reason":"duplicate_key_side","sequence_id":seq,"frame_index":frame,"side":side})
                    continue
                path_value = (path.parent/entry["relative_path"]).relative_to(root).as_posix()
                views[side] = {"path":path_value, "member":entry["member"], "size_bytes":entry["size_bytes"], "sha256":entry["sha256"],
                    "image_size":list(image_size), "source_id":source_id, "tar_complete":entry["tar_complete"]}
        for key in sorted(set(by_key)|selected):
            missing = [side for side in ("left","right") if side not in by_key.get(key,{})]
            if key not in selected:
                report["unselected"].append({"sequence_id":key[0],"frame_index":key[1],"missing_views":missing})
            elif missing:
                report["issues"].append({"reason":"missing_selected_view","sequence_id":key[0],"frame_index":key[1],"missing_views":missing})
        if report["issues"]:
            raise ValueError("key/side 대응 실패")
        for row in rows:
            row["views"] = by_key[(row["sequence_id"],row["frame_index"])]
        manifest = {"schema_version":"local_stereo_rgb_v1","id":manifest_id,"dataset_version":next(iter(versions)),
            "data_kind":next(iter(kinds)),"image_size":list(image_size),"sources":sources,"records":rows}
        manifest["content_sha256"] = digest(manifest)
        pending.write_bytes(_serialized(manifest))
        loader = LocalStereoRGB(pending,root)
        for index in range(len(loader)):
            deadline()
            sample = loader.load_batch([index])
            del sample
            deadline()
        report.update(source=source_identity(),selected_records=len(rows),unselected_count=len(report["unselected"]),
            selected_runs={item["sequence_id"]:contiguous_runs(item["frames"]) for item in selection},
            manifest_sha256=manifest["content_sha256"],data_kind=manifest["data_kind"],
            verification_scope="all extraction payload hashes; selected stereo PNG decode only; no RGB-pose alignment or split truth proof")
        deadline()
        os.replace(pending,final)
        report["status"] = "created"
        return manifest,report
    except BaseException as exc:
        pending.unlink(missing_ok=True)
        final.unlink(missing_ok=True)
        report.update(status="rejected",error_type=type(exc).__name__,reason=str(exc)[:240])
        raise
    finally:
        report["elapsed_seconds"] = time.monotonic()-started
        try:
            (output/"report.json").write_bytes(_serialized(report))
        except BaseException:
            final.unlink(missing_ok=True)
            raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",type=Path,required=True)
    parser.add_argument("--extraction",type=Path,action="append",required=True)
    parser.add_argument("--selection",type=Path,required=True)
    parser.add_argument("--image-size",type=int,nargs=2,required=True)
    parser.add_argument("--id",required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--allow-code-change",action="store_true")
    args = parser.parse_args(argv)
    selection = read_json(_regular_read(_safe_existing(args.selection),MAX_BYTES))
    build_manifest(args.root,args.extraction,selection,args.image_size,args.output,manifest_id=args.id,allow_code_change=args.allow_code_change)


if __name__ == "__main__":
    main()
