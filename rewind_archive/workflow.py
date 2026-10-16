"""Notebook의 단일 설정·사전 요약·1회 실행. 이 모듈은 인증을 시작하지 않는다."""

import copy
import re
from dataclasses import asdict
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED

from rewind_data.audit import read_json
from .contracts import Limits, PreparationError, canonical, digest, validate_includes
from .drive_metadata import MetadataLimits, build_drive_manifest, provenance, save_metadata_bundle, validate_selection
from .extraction import code_hashes
from .matching import verify_extraction
from .pilot import run_metadata_pilot


NOTEBOOK_API_VERSION = 2
CONFIG_FIELDS = {"job_id","reason","mode","purpose","selection","includes","limits","metadata_limits","previous_result"}


def read_report(path, maximum=8*1024**2):
    path=Path(path)
    if path.is_symlink() or path.stat().st_size>maximum:
        raise PreparationError("workflow_report_size_or_link")
    with path.open("rb") as f:
        raw=f.read(maximum+1)
    if len(raw)>maximum:
        raise PreparationError("workflow_report_size_or_link")
    doc=read_json(raw)
    if not isinstance(doc,dict) or doc.get("content_sha256")!=digest({k:v for k,v in doc.items() if k!="content_sha256"}):
        raise PreparationError("workflow_report_hash")
    return doc


def load_history(outputs):
    """현재 소스 작업 폴더의 작은 job 기록만 읽는다. 자동 원본/ZIP 탐색 없음."""
    result={}
    for i,path in enumerate(sorted(Path(outputs).glob("archive_job_*/job.json"))):
        if i>=128:
            raise PreparationError("workflow_history_limit")
        record=read_report(path,256*1024)
        if record.get("schema_version")!="archive_workflow_job_v1":
            raise PreparationError("workflow_history_schema")
        result[record["job_id"]]=record
    return result


def make_preflight(config, history):
    if not isinstance(config,dict) or set(config)!=CONFIG_FIELDS:
        raise PreparationError("workflow_config_fields")
    config=copy.deepcopy(config)
    if not isinstance(config["job_id"],str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,64}",config["job_id"]):
        raise PreparationError("workflow_job_id")
    reason=config["reason"]
    if not isinstance(reason,str) or not reason.strip() or len(reason.encode("utf-8"))>512 or any(ord(c)<32 for c in reason):
        raise PreparationError("workflow_reason_required")
    mode,purpose=config["mode"],config["purpose"]
    if (mode,purpose) not in {("scan","prefix_inventory"),("extract","discovery"),("extract","targeted"),("review","review")}:
        raise PreparationError("workflow_mode_purpose")
    limits=Limits.from_dict(config["limits"])
    metadata=MetadataLimits.from_dict(config["metadata_limits"])
    # 이 notebook 재작성 범위에서는 기존 개발 예산을 늘리지 않는다.
    if any(value>asdict(Limits())[key] for key,value in asdict(limits).items()) or any(value>asdict(MetadataLimits())[key] for key,value in asdict(metadata).items()):
        raise PreparationError("workflow_budget_increase_not_supported")
    config["limits"]=asdict(limits)
    config["metadata_limits"]=asdict(metadata)
    if not isinstance(config["previous_result"],str) or len(config["previous_result"].encode("utf-8"))>4096:
        raise PreparationError("workflow_previous_result")
    if mode=="review":
        if not config["previous_result"]:
            raise PreparationError("workflow_previous_result_required")
    else:
        validate_selection(config["selection"])
        if len(config["selection"]["parts"])>min(metadata.max_files,metadata.max_requests):
            raise PreparationError("workflow_metadata_selection_budget")
        if mode=="scan" and config["includes"]==[]:
            pass
        else:
            validate_includes(config["includes"])
        if purpose=="discovery" and limits.max_files>2:
            raise PreparationError("workflow_discovery_set_max_files_at_most_2")
    effective={k:config[k] for k in ("mode","purpose","selection","includes","limits","metadata_limits")}
    fingerprint=digest(effective)
    repeats=[r["job_id"] for r in history.values() if r.get("effective_sha256")==fingerprint]
    selection_hash=digest(config["selection"])
    same_source=[r["job_id"] for r in history.values() if r.get("selection_sha256")==selection_hash]
    return {"config":config,"config_sha256":digest(config),"effective_sha256":fingerprint,
        "summary":{"job_id":config["job_id"],"reason":reason,"mode":mode,"purpose":purpose,
            "selection":config["selection"],"includes":config["includes"],"limits":config["limits"],
            "metadata_limits":config["metadata_limits"],"same_config_previous_jobs":repeats,
            "same_selection_previous_jobs":same_source,"job_id_already_used":config["job_id"] in history,
            "source_size_version_modified_time":"not_checked_until_metadata" if mode!="review" else "no_network",
            "reads_from_byte_zero":mode!="review","gzip_resume_or_seek":False,
            "includes_meaning":"storage_filter_not_compressed_seek; scan_stores_no_payload",
            "payload_storage_enabled":mode=="extract","full_source_listing_claimed":False,
            "previous_result":config["previous_result"],"train_ready":False}}


def claim_workflow_job(config, preflight, history):
    current=make_preflight(config,history)
    if current["config_sha256"]!=preflight.get("config_sha256") or current["effective_sha256"]!=preflight.get("effective_sha256") or preflight.get("config")!=current["config"]:
        raise PreparationError("workflow_stale_preflight_run_summary_again")
    config=current["config"]
    if config["mode"]=="review":
        raise PreparationError("workflow_review_has_no_network_job")
    if config["job_id"] in history:
        raise PreparationError("workflow_job_already_started")
    previous=[r for r in history.values() if r.get("effective_sha256")==current["effective_sha256"]]
    if previous and any(r["config"]["reason"].strip()==config["reason"].strip() for r in previous):
        raise PreparationError("workflow_repeat_requires_new_id_and_changed_reason")
    record={"schema_version":"archive_workflow_job_v1","job_id":config["job_id"],"config":config,
        "config_sha256":current["config_sha256"],"effective_sha256":current["effective_sha256"],
        "selection_sha256":digest(config["selection"]),"repeat_of":[r["job_id"] for r in previous],
        "status":"started","stage":"before_authentication","source":provenance(),"code_sha256":code_hashes(),
        "reads_from_byte_zero":True,"train_ready":False}
    history[config["job_id"]]=record
    return record


def write_job_record(path, record):
    body={k:v for k,v in record.items() if k!="content_sha256"}
    raw=canonical({**body,"content_sha256":digest(body)})+b"\n"
    if len(raw)>256*1024:
        raise PreparationError("workflow_job_report_bytes_limit")
    Path(path).write_bytes(raw)


def run_workflow_job(config, preflight, history, outputs, session_factory):
    """명시한 호출 1회. Claim 전 인증 없음. 실패도 job을 소비하고 자동 retry하지 않는다."""
    record=claim_workflow_job(config,preflight,history)
    config=record["config"]
    root=Path(outputs)/("archive_job_"+config["job_id"])
    root.mkdir(parents=True,exist_ok=False)
    record["output"]=str(root.resolve())
    write_job_record(root/"job.json",record)
    session=None
    try:
        record["stage"]="authentication"
        session=session_factory()
        record["stage"]="metadata"
        metadata=build_drive_manifest(config["selection"],session,MetadataLimits.from_dict(config["metadata_limits"]))
        save_metadata_bundle(metadata,root/"metadata")
        if not metadata["report"]["manifest_ready"]:
            record.update(status="metadata_failed",stop_reason=metadata["report"]["stop_reason"])
        else:
            record["source_identity_after_metadata"]=metadata["manifest"]
            write_job_record(root/"job.json",record)
            # 선택 ID만의 preflight와 실제 metadata snapshot을 구분해 실행 직전에 표시한다.
            print({"source_identity_after_metadata":metadata["manifest"],"reads_from_byte_zero":True})
            record["stage"]="pilot"
            pilot=run_metadata_pilot(metadata,root/"pilot",config["includes"],Limits.from_dict(config["limits"]),session,mode=config["mode"])
            record.update(status="finished",stop_reason=pilot["stop_reason"],sample_acquired=pilot["sample_acquired"],
                observed_regular_members=pilot["observed_regular_members"])
    except KeyboardInterrupt:
        record.update(status="interrupted",stop_reason="interrupted")
    except Exception as exc:
        # 인증/transport exception 원문에는 token이나 URL이 있을 수 있다.
        record.update(status="failed",stop_reason=exc.reason if isinstance(exc,PreparationError) else "workflow_execution_failed")
    finally:
        if session is not None:
            try:session.close()
            except Exception:pass
        write_job_record(root/"job.json",record)
    return root,record


def review_run(path):
    """보고서 hash 및 가능한 경우 현재 코드의 완전 파일 hash 검사. 원본 요청 없음."""
    root=Path(path)
    result={"root":str(root),"network_requested":False,"train_ready":False}
    if (root/"job.json").is_file():
        record=read_report(root/"job.json",256*1024)
        result["job"]={k:record[k] for k in ("job_id","config","status","stage","stop_reason") if k in record}
    extraction_root=root/"pilot" if (root/"pilot").is_dir() else root
    if (extraction_root/"extraction_manifest.json").is_file():
        report=read_report(extraction_root/"extraction_manifest.json")
        result["extraction"]={k:report[k] for k in ("operation","stop_reason","inspection_candidates_usable",
            "source_validation_pending","stats","member_observations","member_observation_trust",
            "member_observation_cap_reached","full_source_listing_claimed") if k in report}
        result["complete_stored_files"]=len(report["files"])
        try:
            verify_extraction(extraction_root)
            result["file_verification"]="passed_current_code"
        except (PreparationError,OSError,ValueError,KeyError):
            result["file_verification"]="not_verified_current_code_or_unusable"
    if (extraction_root/"pilot_report.json").is_file():
        pilot=read_report(extraction_root/"pilot_report.json")
        result["sample_acquired"]=pilot.get("sample_acquired",False)
        result["sample_status"]=pilot.get("sample_status","legacy_report")
    if len(result)==3:
        raise PreparationError("workflow_no_result_reports")
    return result


def save_result_zip(path):
    root=Path(path).resolve()
    review_run(root)  # 잘못 고른 source/다른 디렉터리를 결과로 압축하지 않는다.
    paths=[]
    allowed={"job.json","pilot_job.json","metadata","pilot","files","extraction_manifest.json","pilot_report.json"}
    for i,p in enumerate(root.rglob("*")):
        if i>=4096 or p.is_symlink() or not p.resolve().is_relative_to(root) or p.relative_to(root).parts[0] not in allowed:
            raise PreparationError("workflow_result_paths")
        if p.is_file():paths.append(p)
    if len(paths)>256 or sum(p.stat().st_size for p in paths)>17*1024**2:
        raise PreparationError("workflow_result_bundle_limit")
    target=root.with_suffix(".zip")
    if target.is_symlink():raise PreparationError("workflow_result_paths")
    if not target.exists():
        with ZipFile(target,"x",ZIP_DEFLATED) as bundle:
            for p in paths:bundle.write(p,(Path(root.name)/p.relative_to(root)).as_posix())
    return target
