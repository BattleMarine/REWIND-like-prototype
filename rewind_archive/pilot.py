"""작은 준비 실행을 verify와 간결한 보고서에 연결한다. 학습을 시작하지 않는다."""

from pathlib import Path
import copy

from .contracts import Limits, PreparationError, digest
from .drive_metadata import environment, provenance, validate_metadata_result, write_new_json
from .extraction import prepare
from .matching import verify_extraction


PENDING = ["live_drive_colab_validation","joints","units","coordinates","calibration","timestamp",
    "split","original_motion_duplicates","stereo_rgb_pose_content_alignment","m1_actual_data_training"]


def run_pilot(manifest, output, includes, limits, authorized_session=None, base=".", metadata_report=None, mode="extract"):
    """새 output에서 1회 prepare 후 검사·pilot report 저장. 자동 retry/확대 없음."""
    if not isinstance(limits,Limits):
        raise PreparationError("limits_required")
    extraction=prepare(manifest,output,includes,limits,base=base,authorized_session=authorized_session,mode=mode)
    verification={"status":"not_usable","reason":None,"verified_complete_files":0}
    if extraction["inspection_candidates_usable"]:
        try:
            checked=verify_extraction(output,manifest)
            verification={"status":"passed","reason":None,"verified_complete_files":len(checked["files"])}
        except (PreparationError,OSError,ValueError,KeyError):
            verification={"status":"failed","reason":"extraction_verify_failed","verified_complete_files":0}
    report={"schema_version":"archive_pilot_report_v1","source":provenance(),"environment":environment(),
        "archive_id":manifest["archive_id"],"kind":manifest["kind"],"source_scope":manifest["source_scope"],
        "manifest_sha256":digest(manifest),"extraction_manifest_content_sha256":digest(extraction),
        "complete_files":len(extraction["files"]),"verification":verification,"stats":extraction["stats"],
        "stop_reason":extraction["stop_reason"],"failure_class":extraction["failure_class"],
        "inspection_candidates_usable":extraction["inspection_candidates_usable"] and verification["status"]=="passed",
        "gzip_eof_crc_valid":extraction["gzip_eof_crc_valid"],"tar_end_blocks_seen":extraction["tar_end_blocks_seen"],
        "whole_archive_integrity_verified":extraction["whole_archive_integrity_verified"],
        "sample_status":"verified_candidates" if verification["status"]=="passed" and extraction["files"] else "no_sample" if not extraction["files"] else "unusable_candidates",
        "sample_acquired":verification["status"]=="passed" and bool(extraction["files"]),
        "source_validation_pending":extraction["source_validation_pending"],
        "operation":mode,"observed_regular_members":len(extraction["member_observations"]),
        "member_observations_usable":extraction["member_observations_usable"],
        "member_observation_trust":extraction["member_observation_trust"],
        "train_ready":False,"pending_verifications":PENDING,"scope":"single_archive_preparation_only"}
    if metadata_report is not None:
        report["metadata_report_content_sha256"]=metadata_report["content_sha256"]
        report["metadata_stage_stats"]=copy.deepcopy(metadata_report["stats"])
        report["metadata_stage_limits"]=copy.deepcopy(metadata_report["limits"])
    report["content_sha256"]=digest(report)
    raw_path=Path(output)/"pilot_report.json"
    # 기존 max_disk_bytes 안의 report reserve에서 compact report도 사용한다.
    from .contracts import canonical
    existing=(Path(output)/"extraction_manifest.json").stat().st_size
    if existing+len(canonical(report))+1>extraction["report_disk_reserve_bytes"]:
        raise PreparationError("pilot_report_reserve")
    write_new_json(raw_path,report)
    return report


def run_metadata_pilot(result, output, includes, limits, authorized_session, mode="extract"):
    """성공한 metadata 결과의 hash/code를 대조한 뒤에만 media 경로를 시작한다."""
    manifest=validate_metadata_result(result)
    return run_pilot(manifest,output,includes,limits,authorized_session,
        metadata_report=result["report"],mode=mode)
