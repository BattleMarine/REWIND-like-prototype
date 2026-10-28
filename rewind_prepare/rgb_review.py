"""RGB와 학습 차단 pose candidate의 소량 GT2D 검토 자료를 만든다."""
import argparse
import copy
import hashlib
import json
import re
import time
from pathlib import Path

from rewind_archive.contracts import digest
from rewind_archive.matching import identify
from rewind_data.audit import read_json
from rewind_m1.local_rgb import LocalStereoRGB, _no_link, _path, _regular_read
from .converter import validate_candidate
from .provenance import revision_identity


ROOT = Path(__file__).resolve().parents[1]
MAX_CANDIDATE_BYTES = 64 * 1024**2
MAX_JSON_BYTES = 4 * 1024**2
MAX_IMAGE_BYTES = 16 * 1024**2
MAX_OUTPUT_BYTES = 24 * 1024**2
MAX_SECONDS = 30


def _safe(path):
    path = Path(path).absolute()
    for part in (*reversed(path.parents), path):
        _no_link(part)
    return path.resolve(strict=True)


def _json_bytes(value):
    raw = json.dumps(value, ensure_ascii=True, allow_nan=False, separators=(",", ":")).encode("utf-8")
    if len(raw) > MAX_JSON_BYTES:
        raise ValueError("review JSON 4MiB 상한")
    return raw


def _key(row):
    if not isinstance(row,dict) or type(row.get("frame_index")) is not int or not 0 <= row["frame_index"] < 10**12:
        raise ValueError("sequence/frame 식별 오류")
    return _path(row["sequence_id"]), row["frame_index"]


def _mapping(rows):
    if not isinstance(rows,list) or not rows:
        raise ValueError("비어 있지 않은 record 목록 필요")
    result = {}
    for row in rows:
        key = _key(row)
        if key in result:
            raise ValueError("중복 sequence/frame")
        result[key] = row
    return result


def create_review(rgb_manifest, rgb_root, candidate_path, selection, output):
    manifest_path, rgb_root, candidate_path = map(_safe,(rgb_manifest,rgb_root,candidate_path))
    output = Path(output).absolute()
    _safe(output.parent)
    target = output.resolve(strict=False)
    if (target.is_relative_to(rgb_root) or rgb_root.is_relative_to(target)
            or any(p.is_relative_to(target) for p in (manifest_path,candidate_path))):
        raise ValueError("출력과 원본 입력 경로 겹침")
    output.mkdir(exist_ok=False)
    started = time.monotonic()
    report = {"schema_version":"rgb_pose_review_v1", "status":"rejected", "train_ready":False,
        "training_release":"not_implemented", "rgb_pose_alignment_confirmed":False, "frames":[]}

    def deadline():
        if time.monotonic()-started > MAX_SECONDS:
            raise ValueError("review cooperative deadline 초과")

    try:
        candidate_raw = _regular_read(candidate_path,MAX_CANDIDATE_BYTES)
        candidate = read_json(candidate_raw)
        validate_candidate(candidate)
        records = candidate["records"]
        if len(records) > candidate["preparation_spec"]["limits"]["max_files"]:
            raise ValueError("candidate record 수 상한")
        candidate_rows = _mapping(records)
        manifest = candidate.get("manifest")
        if not isinstance(manifest,dict) or manifest.get("data_kind") != candidate["contract"]["data_kind"]:
            raise ValueError("candidate manifest/contract data_kind 불일치")
        raw_files = manifest.get("raw_files")
        extraction_hash = manifest.get("source_hashes",{}).get("extraction_manifest")
        if not isinstance(extraction_hash,str) or not re.fullmatch(r"[0-9a-f]{64}",extraction_hash):
            raise ValueError("candidate extraction provenance hash 필요")
        if not isinstance(raw_files,list) or not raw_files or len(raw_files) > 256:
            raise ValueError("candidate raw_files 목록 필요")
        pose_sources = {}
        for item in raw_files:
            if not isinstance(item,dict) or item.get("member") in pose_sources:
                raise ValueError("candidate raw_files 중복/형식 오류")
            pose_sources[_path(item["member"])] = item
        _json_bytes(selection)
        if not isinstance(selection,list) or not 1 <= len(selection) <= 8 or any(set(row) != {"sequence_id","frame_index"} for row in selection if isinstance(row,dict)):
            raise ValueError("명시 selection 최대8 frame 필요")
        selected = _mapping(selection)
        manifest_raw = _regular_read(manifest_path,MAX_JSON_BYTES)
        loader = LocalStereoRGB(manifest_path,rgb_root)
        if read_json(manifest_raw).get("content_sha256") != loader.manifest_sha256:
            raise ValueError("검토 초기 RGB manifest 변경")
        rgb_rows = {_key(loader.metadata(i)["record"]):(i,loader.metadata(i)) for i in range(len(loader))}
        if any(key not in candidate_rows or key not in rgb_rows for key in selected):
            missing = [{"sequence_id":s,"frame_index":f,"missing_rgb":(s,f) not in rgb_rows,"missing_candidate":(s,f) not in candidate_rows} for s,f in selected if (s,f) not in rgb_rows or (s,f) not in candidate_rows]
            report["missing_selected"] = missing
            raise ValueError("선택 RGB/candidate 누락; 다른 frame 대체 금지")
        first_meta = loader.metadata(0)
        contract = candidate["contract"]
        if contract["data_kind"] != first_meta["data_kind"] or candidate["manifest"].get("dataset_version") != first_meta["dataset_version"]:
            raise ValueError("공통 dataset_version/data_kind 불일치")
        names = list(contract["joint_names"])
        report.update(joint_names=names,joint_definition_source="candidate_contract_only; RGB manifest has no joint definition",
            candidate_training_eligibility=copy.deepcopy(candidate["training_eligibility"]),
            provenance={"candidate_file_sha256":hashlib.sha256(candidate_raw).hexdigest(),"candidate_content_sha256":candidate["content_sha256"],
                "candidate_source_hashes":copy.deepcopy(candidate["manifest"].get("source_hashes",{})),
                "rgb_manifest_sha256":loader.manifest_sha256,"rgb_manifest_file_sha256":hashlib.sha256(manifest_raw).hexdigest(),
                "dataset_version":first_meta["dataset_version"],"data_kind":first_meta["data_kind"]},
            unselected={"rgb":[[s,f] for s,f in rgb_rows if (s,f) not in selected],"candidate":[[s,f] for s,f in candidate_rows if (s,f) not in selected]},
            unconfirmed=["human_RGB_pose_alignment_review","split_declaration_truth","original_motion_independence","timestamp_semantics",
                "units_and_camera_semantics","training_release"],
            visibility_semantics="unknown; annotation validity and image bounds are separate")
        total_image_bytes, physical_files = 0, set()
        for key in selected:
            index, meta = rgb_rows[key]
            row, rgb = candidate_rows[key], meta["record"]
            pose_split, rgb_split = row.get("split"), rgb["split"]
            if pose_split not in ("train","validation","test","unknown"):
                raise ValueError("candidate split 선언 오류")
            split_status = "unknown" if "unknown" in (pose_split,rgb_split) else "declared_match"
            if split_status != "unknown" and pose_split != rgb_split:
                raise ValueError("명시 split 충돌")
            if [rgb["views"][s]["image_size"] for s in ("left","right")] != contract["image_sizes"]:
                raise ValueError("candidate/RGB 원해상도 불일치")
            source = row["source"]
            if (not isinstance(source,dict) or pose_sources.get(source.get("member")) != source
                    or not isinstance(source.get("extraction_manifest_sha256"),str)
                    or not re.fullmatch(r"[0-9a-f]{64}",source["extraction_manifest_sha256"])
                    or source.get("extraction_manifest_sha256") != manifest.get("source_hashes",{}).get("extraction_manifest")
                    or type(source.get("size_bytes")) is not int or not 1 <= source["size_bytes"] <= candidate["preparation_spec"]["limits"]["max_file_bytes"]):
                raise ValueError("candidate record/source provenance 참조 불일치")
            if identify(source["member"],"pose") != (*key,"pose") or not isinstance(source.get("sha256"),str) or not re.fullmatch(r"[0-9a-f]{64}",source["sha256"]):
                raise ValueError("candidate pose source 식별/hash 오류")
            # 좌우 영상과 pose는 별개 파일/추출본이다. 서로의 source hash가 같아야 한다고 요구하지 않는다.
            for side in ("left","right"):
                view = rgb["views"][side]
                info = loader._file(view["path"]).stat()
                physical = (info.st_dev,info.st_ino)
                if physical in physical_files:
                    raise ValueError("선택 PNG의 동일 실제 파일 중복 참조")
                physical_files.add(physical)
                total_image_bytes += view["size_bytes"]
            if total_image_bytes > MAX_IMAGE_BYTES:
                raise ValueError("선택 PNG 누적16MiB 상한")
            report["frames"].append({"sequence_id":key[0],"frame_index":key[1],"candidate_split":pose_split,"rgb_split":rgb_split,
                "split_status":split_status,"split_evidence":rgb["split_evidence"],"motion_id":row["motion_id"],
                "motion_identity":copy.deepcopy(row.get("motion_identity")),"timestamp_raw":row.get("timestamp_raw"),"time":copy.deepcopy(row.get("time")),
                "pose_source":copy.deepcopy(source),"rgb_sources":copy.deepcopy(meta["sources"]),"views":{}})
        deadline()
        (output/"images").mkdir()
        for position,key in enumerate(selected):
            index,meta = rgb_rows[key]
            rgb, pose = meta["record"],candidate_rows[key]
            deadline()
            sample = loader.load_batch([index])
            del sample
            for side in ("left","right"):
                view = rgb["views"][side]
                raw = _regular_read(loader._file(view["path"]),view["size_bytes"],view["size_bytes"])
                if hashlib.sha256(raw).hexdigest() != view["sha256"]:
                    raise ValueError("검토 복사 전 PNG 변경")
                destination = f"images/frame_{position:02d}_{side}.png"
                (output/destination).write_bytes(raw)
                field = copy.deepcopy(pose["fields"][f"camera_{side}_pts2d"])
                size = view["image_size"]
                outside = [bool(valid and not(0 <= point[0] <= size[0]-1 and 0 <= point[1] <= size[1]-1)) for point,valid in zip(field["coordinates"],field["valid_mask"])]
                report["frames"][position]["views"][side] = {"image":destination,"image_size":size,"source":copy.deepcopy(view),
                    "coordinates":field["coordinates"],"valid_mask":field["valid_mask"],"outside_image":outside,
                    "valid_count":sum(field["valid_mask"]),"missing_count":sum(not v for v in field["valid_mask"]),"outside_count":sum(outside)}
            deadline()
        if hashlib.sha256(_regular_read(candidate_path,MAX_CANDIDATE_BYTES)).hexdigest() != report["provenance"]["candidate_file_sha256"]:
            raise ValueError("검토 중 candidate 원본 변경")
        if hashlib.sha256(_regular_read(manifest_path,MAX_JSON_BYTES)).hexdigest() != report["provenance"]["rgb_manifest_file_sha256"]:
            raise ValueError("검토 중 RGB manifest 원본 변경")
        source_paths = [Path(__file__),ROOT/"rewind_prepare/converter.py",ROOT/"rewind_m1/local_rgb.py",Path(__file__).with_name("rgb_review_template.html")]
        hashes = {p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths}
        report.update(status="review_ready",source={**revision_identity(ROOT),"code_hash_scope":list(hashes),"code_sha256":hashes},
            elapsed_seconds=time.monotonic()-started,selected_records=len(selected),copied_image_bytes=total_image_bytes)
        raw_report = _json_bytes(report)
        encoded = raw_report.decode().replace("&","\\u0026").replace("<","\\u003c").replace(">","\\u003e")
        template = Path(__file__).with_name("rgb_review_template.html").read_text(encoding="utf-8")
        html = template.replace("__REVIEW_JSON__",encoded).encode("utf-8")
        if len(html)+len(raw_report)+total_image_bytes > MAX_OUTPUT_BYTES:
            raise ValueError("검토 출력24MiB 상한")
        deadline()
        (output/"review.html").write_bytes(html)
        (output/"report.json").write_bytes(raw_report)
        return report
    except BaseException as exc:
        (output/"review.html").unlink(missing_ok=True)
        report.update(status="rejected",error_type=type(exc).__name__,reason=str(exc)[:240],elapsed_seconds=time.monotonic()-started)
        try:
            rejected = _json_bytes(report)
        except (ValueError,TypeError,RecursionError):
            rejected = _json_bytes({"schema_version":"rgb_pose_review_v1","status":"rejected","train_ready":False,
                "training_release":"not_implemented","rgb_pose_alignment_confirmed":False,"error_type":type(exc).__name__,
                "reason":str(exc)[:240],"report_compacted":True,"unselected_counts":{
                    k:len(v) for k,v in report.get("unselected",{}).items()},"elapsed_seconds":time.monotonic()-started})
        (output/"report.json").write_bytes(rejected)
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("rgb-manifest","rgb-root","candidate","selection","output"):
        parser.add_argument("--"+name,type=Path,required=True)
    args = parser.parse_args(argv)
    selection = read_json(_regular_read(_safe(args.selection),MAX_JSON_BYTES))
    create_review(args.rgb_manifest,args.rgb_root,args.candidate,selection,args.output)


if __name__ == "__main__":
    main()
