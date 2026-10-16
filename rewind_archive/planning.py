"""완전 추출본에서 다음 대응 선택을 작성한다. 원본/네트워크를 읽지 않는다."""

from dataclasses import asdict
from glob import escape
from pathlib import Path

from .contracts import Limits, PreparationError, digest, validate_includes
from .matching import identify, verify_extraction


def discovery_globs(kind):
    if kind == "rgb":
        return ["*/fisheye_final_image/camera_*/final_*.png"]
    if kind == "pose":
        return ["*/json/frame_*.json"]
    raise PreparationError("discovery_kind")


def counterpart_plan(extractions, max_frames=8):
    """Hash 검사를 통과한 실제 관측 key만 사용한다. Split/content는 미검증."""
    if not isinstance(extractions,list) or not 1 <= len(extractions) <= 8 or type(max_frames) is not int or not 1 <= max_frames <= 16:
        raise PreparationError("counterpart_plan_limits")
    observed, inputs, versions = {}, [], set()
    for path in extractions:
        report = verify_extraction(path)
        versions.add(report["archive_manifest"]["dataset_version"])
        inputs.append({"extraction":str(Path(path)),"content_sha256":digest(report),"complete_files":len(report["files"])})
        for entry in report["files"]:
            sequence, frame, view = identify(entry["member"],entry["kind"])
            views = observed.setdefault((sequence,frame),set())
            if view in views:
                raise PreparationError("duplicate_frame_view")
            views.add(view)
    if len(versions)>1:
        raise PreparationError("dataset_version_mismatch")
    globs = {"rgb":[],"pose":[]}
    frames = []
    for (sequence,frame),views in sorted(observed.items())[:max_frames]:
        missing = sorted({"left","right","pose"}-views)
        frames.append({"sequence":sequence,"frame":frame,"observed_views":sorted(views),"missing_views":missing})
        seq = escape(sequence)
        if {"left","right"}-views:
            # 한 frame당 1 glob. 이미 있는 한쪽도 일치할 수 있으므로 재사용 목록과 대조한다.
            globs["rgb"].append(f"{seq}/fisheye_final_image/camera_*/final_{frame}.png")
        if "pose" not in views:
            globs["pose"].append(f"{seq}/json/frame_{frame}.json")
    for values in globs.values():
        if values:
            validate_includes(values)
    return {"schema_version":"archive_counterpart_plan_v1","verified_inputs":inputs,
        "frames":frames,"member_globs_by_kind":globs,"all_observed_frame_count":len(observed),
        "next_action":"targeted_counterpart_or_reuse" if frames else "stop_no_complete_sample",
        "split_verification":"unverified","content_alignment":"unverified","train_ready":False,
        "bounded_target_reachability":"unknown","resume_supported":False}


def claim_pilot_job(registry, job_id, selection, includes, limits, purpose, output):
    """Notebook 커널 내 at-most-once 시작. 실패도 소비하며 자동 retry하지 않는다."""
    includes = validate_includes(includes)
    if not isinstance(limits,Limits) or purpose not in ("discovery","targeted"):
        raise PreparationError("pilot_job_input")
    if not isinstance(job_id,str) or not job_id or len(job_id)>128:
        raise PreparationError("pilot_job_id")
    if job_id in registry:
        raise PreparationError("pilot_job_already_started_reuse_or_explicit_new_job")
    registry[job_id] = {"input_sha256":digest({"selection":selection,"includes":includes,
        "limits":asdict(limits),"purpose":purpose}),"output":str(output),"purpose":purpose}
    return registry[job_id]
