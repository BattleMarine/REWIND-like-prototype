"""추출본 재사용 검사와 이름 기반 stereo RGB/pose 매칭. 학습 승인 판정 없음."""

import hashlib
import json
import os
import re
import time
from pathlib import Path

from rewind_data.audit import read_json
from .contracts import Limits, PreparationError, canonical, digest, safe_path, validate_manifest
from .extraction import BENIGN_STOPS, code_hashes


def identify(member, kind):
    name = safe_path(member)
    pattern = r"(.+)/fisheye_final_image/camera_(left|right)/final_([0-9]+)\.png" if kind=="rgb" else r"(.+)/json/frame_([0-9]+)\.json"
    match = re.fullmatch(pattern,name)
    if not match:
        raise PreparationError("unrecognized_member_name")
    if kind=="rgb":
        sequence, view, frame = match.groups()
    else:
        sequence, frame = match.groups()
        view = "pose"
    if len(frame)>12:
        raise PreparationError("frame_index_limit")
    return sequence, int(frame), view


def verify_extraction(path, expected_manifest=None, allow_code_change=False):
    path = Path(path)
    if path.is_dir():
        path = path/"extraction_manifest.json"
    if path.is_symlink() or path.stat().st_size>8*1024**2:
        raise PreparationError("invalid_extraction_report")
    with path.open("rb") as f:
        raw = f.read(8*1024**2+1)
    if len(raw)>8*1024**2:
        raise PreparationError("invalid_extraction_report")
    report = read_json(raw)
    body = {k:v for k,v in report.items() if k!="content_sha256"}
    if report.get("schema_version")!="archive_extraction_v1" or report.get("content_sha256")!=digest(body):
        raise PreparationError("extraction_manifest_hash")
    manifest = validate_manifest(report["archive_manifest"])
    if digest(manifest)!=report["archive_manifest_sha256"] or (expected_manifest is not None and digest(validate_manifest(expected_manifest))!=digest(manifest)):
        raise PreparationError("archive_manifest_mismatch")
    if report["inspection_candidates_usable"] is not True or report["train_ready"] is not False or report["stop_reason"] not in BENIGN_STOPS|{"complete"}:
        raise PreparationError("unusable_extraction")
    if not allow_code_change and report["code_sha256"]!=code_hashes():
        raise PreparationError("extraction_code_changed")
    limits = Limits.from_dict(report["limits"])
    if len(report["files"])>limits.max_files or path.stat().st_size>report["report_disk_reserve_bytes"]:
        raise PreparationError("invalid_extraction_report")
    root = path.parent.resolve()
    total, seen = 0, set()
    started = time.monotonic()
    for entry in report["files"]:
        member = safe_path(entry["member"],limits.max_path_bytes)
        relative = safe_path(entry["relative_path"],limits.max_path_bytes+6)
        if relative != "files/"+member or relative.casefold() in seen or not entry["tar_complete"] or entry["kind"]!=manifest["kind"]:
            raise PreparationError("invalid_extraction_file")
        seen.add(relative.casefold())
        file = root/relative
        if any(p.is_symlink() for p in [file,*file.parents] if p.is_relative_to(root)) or not file.resolve().is_relative_to(root) or not file.is_file():
            raise PreparationError("missing_or_unsafe_extraction_file")
        if type(entry["size_bytes"]) is not int or not 0<=entry["size_bytes"]<=limits.max_file_bytes or file.stat().st_size!=entry["size_bytes"]:
            raise PreparationError("extracted_file_size")
        hash_ = hashlib.sha256()
        with file.open("rb") as f:
            before = os.fstat(f.fileno())
            remaining = entry["size_bytes"]
            if before.st_size!=remaining:
                raise PreparationError("extracted_file_size")
            while remaining:
                if time.monotonic()-started>limits.max_elapsed_seconds:
                    raise PreparationError("reuse_elapsed_limit")
                chunk = f.read(min(remaining,limits.io_chunk_bytes))
                if not chunk:
                    raise PreparationError("extracted_file_size")
                hash_.update(chunk)
                remaining -= len(chunk)
            after = os.fstat(f.fileno())
            current = file.stat()
            if (before.st_size,before.st_mtime_ns)!=(after.st_size,after.st_mtime_ns) or (current.st_dev,current.st_ino)!=(after.st_dev,after.st_ino):
                raise PreparationError("extracted_file_changed")
        if hash_.hexdigest()!=entry["sha256"]:
            raise PreparationError("extracted_file_hash")
        total += entry["size_bytes"]
    if total+path.stat().st_size>limits.max_disk_bytes or total!=report["stats"]["payload_disk_bytes"]:
        raise PreparationError("extraction_disk_size")
    return report


def contiguous_runs(values):
    runs = []
    for value in sorted(set(values)):
        if not runs or value!=runs[-1][1]+1:
            runs.append([value,value])
        else:
            runs[-1][1] = value
    return runs


def match_extractions(paths):
    if not paths or len(paths)>32:
        raise PreparationError("matching_inputs")
    by_sequence, reports, issues = {}, [], []
    versions = set()
    for path in paths:
        report = verify_extraction(path)
        manifest = report["archive_manifest"]
        versions.add(manifest["dataset_version"])
        reports.append({"archive_id":manifest["archive_id"],"source_scope":manifest["source_scope"],
            "extraction_manifest_sha256":digest(report),"stop_reason":report["stop_reason"],
            "whole_archive_integrity_verified":report["whole_archive_integrity_verified"]})
        for file in report["files"]:
            try:
                seq, frame, view = identify(file["member"],manifest["kind"])
            except PreparationError:
                issues.append({"kind":"unrecognized_member_name","member":file["member"]})
                continue
            item = by_sequence.setdefault(seq,{}).setdefault(frame,{})
            if view in item:
                issues.append({"kind":"duplicate_frame_view","sequence_id":seq,"frame_index":frame,"view":view})
                item[view] = None
            else:
                item[view] = {"archive_id":manifest["archive_id"],"member":file["member"],"sha256":file["sha256"]}
    if len(versions)>1:
        issues.append({"kind":"dataset_version_mismatch"})
    sequences = []
    for seq, frames in sorted(by_sequence.items()):
        observed = sorted(frames)
        complete, missing = [], []
        for frame in observed:
            absent = [view for view in ("left","right","pose") if not frames[frame].get(view)]
            if absent:
                missing.append({"frame_index":frame,"missing_views":absent})
            else:
                complete.append(frame)
        runs = contiguous_runs(observed)
        gaps = [[a[1]+1,b[0]-1] for a,b in zip(runs,runs[1:])]
        sequences.append({"sequence_id":seq,"split":"unknown","original_motion_verification":"unverified",
            "observed_frames":observed,"observed_runs":runs,"missing_frame_ranges_between_observations":gaps,
            "matched_triplet_frames":complete,"matched_triplet_runs":contiguous_runs(complete),"missing_views":missing,
            "sequence_start_end_completeness":"unknown","frames":[{"frame_index":i,"views":frames[i]} for i in observed]})
    report = {"schema_version":"archive_matching_v1","inputs":reports,"dataset_versions":sorted(versions),
        "issues":issues,"sequences":sequences,"matching_valid":not issues,"train_ready":False,
        "matched_triplet_count":sum(len(s["matched_triplet_frames"]) for s in sequences),
        "incomplete_frame_count":sum(len(s["missing_views"]) for s in sequences),
        "observed_frame_count":sum(len(s["observed_frames"]) for s in sequences),
        "pending_verifications":["joints","units","coordinates","calibration","timestamp","split","original_motion_duplicates","rgb_pose_content_alignment"]}
    return {**report,"content_sha256":digest(report)}
