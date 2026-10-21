"""Notebook의 단일 설정·사전 요약·1회 실행. 이 모듈은 인증을 시작하지 않는다."""

import copy
import re
from dataclasses import asdict
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED

from rewind_data.audit import read_json
from .contracts import Limits, PreparationError, canonical, digest, validate_includes, validate_exact_targets, target_summary
from .drive_metadata import MetadataLimits, build_drive_manifest, provenance, save_metadata_bundle, validate_selection
from .extraction import code_hashes
from .matching import verify_extraction
from .pilot import run_metadata_pilot
from .profiles import POSE_COUNTERPART_PROFILE, POSE_PROFILES, CACHE_PROFILE, CACHE_PROFILES, WARM128_PROFILE, TARGET_PAIR_PROFILES, validate_profile_limits
from .diagnostics import request_failure, safe_request_failure
from .request_policy import validate_request_policy, policy_description, validate_no_hidden_retries
from .prefix_cache import validate_input_spec, export_cache
from .cache_policy import WARM128_POLICY,validate_expectation,summary as policy_summary


NOTEBOOK_API_VERSION = 8
LEGACY_CONFIG_FIELDS = {"job_id","reason","mode","purpose","selection","includes","limits","metadata_limits","previous_result"}
PREVIOUS_CONFIG_FIELDS = LEGACY_CONFIG_FIELDS | {"profile","exact_targets"}
POLICY_CONFIG_FIELDS = PREVIOUS_CONFIG_FIELDS | {"request_policy"}
CACHE_CONFIG_FIELDS = POLICY_CONFIG_FIELDS | {"cache_input"}
CONFIG_FIELDS = CACHE_CONFIG_FIELDS | {"cache_expectation"}


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
    if not isinstance(config,dict) or set(config) not in (CONFIG_FIELDS,CACHE_CONFIG_FIELDS,POLICY_CONFIG_FIELDS,PREVIOUS_CONFIG_FIELDS,LEGACY_CONFIG_FIELDS):
        raise PreparationError("workflow_config_fields")
    config=copy.deepcopy(config)
    config.setdefault("profile",None)
    config.setdefault("exact_targets",[])
    # 누락된 legacy 필드는 config hash/유효 fingerprint까지 그대로 보존한다.
    request_policy=validate_request_policy(config.get("request_policy"))
    cache_input=validate_input_spec(config.get("cache_input"))
    expectation=validate_expectation(config.get("cache_expectation"),required=config.get("profile")==WARM128_PROFILE)
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
    if config["profile"] is None:
        # 이전9필드/정규화된 legacy는 기존 smoke 이하만. 고정 preset 이름을 붙이지 않는다.
        if any(value>asdict(Limits())[key] for key,value in asdict(limits).items()):
            raise PreparationError("workflow_budget_increase_not_supported")
    else:
        validate_profile_limits(config["profile"],limits)
    if any(value>asdict(MetadataLimits())[key] for key,value in asdict(metadata).items()):
        raise PreparationError("workflow_budget_increase_not_supported")
    config["exact_targets"]=validate_exact_targets(config["exact_targets"],config["includes"],limits,"extract" if mode=="review" else mode)
    if config["profile"] in TARGET_PAIR_PROFILES and mode!="review" and (mode!="extract" or purpose!="targeted" or len(config["exact_targets"])!=2):
        raise PreparationError("counterpart_profile_requires_targeted_pair")
    if config["profile"] in POSE_PROFILES and (not isinstance(config["selection"],dict) or config["selection"].get("kind")!="pose"):
        raise PreparationError("pose_profile_requires_pose_kind")
    if request_policy and (config["profile"] not in POSE_PROFILES or
            (mode!="review" and (mode,purpose)!=("extract","targeted")) or len(config["exact_targets"])!=2):
        raise PreparationError("request_policy_requires_pose_targeted_pair")
    if cache_input is not None and config["profile"] not in CACHE_PROFILES:raise PreparationError("cache_profile_required")
    if expectation is not None and config["profile"]!=WARM128_PROFILE:raise PreparationError("warm_profile_required")
    if config["profile"]==WARM128_PROFILE:
        if cache_input is None:raise PreparationError("warm_cache_required")
        if cache_input["manifest_sha256"]!=expectation["manifest_sha256"]:raise PreparationError("cache_expected_manifest_mismatch")
    if config["profile"] in CACHE_PROFILES and (config["selection"].get("total_parts")!=1 or len(config["selection"].get("parts",[]))!=1):
        raise PreparationError("cache_single_drive_part")
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
    if config["profile"] is not None or config["exact_targets"]:
        effective.update(profile=config["profile"],exact_targets=config["exact_targets"])
    if request_policy:effective["request_policy"]=request_policy
    if config["profile"] in CACHE_PROFILES:effective["cache_input_manifest_sha256"]=cache_input["manifest_sha256"] if cache_input else None
    if expectation:effective["cache_expectation"]=expectation
    fingerprint=digest(effective)
    repeats=[r["job_id"] for r in history.values() if r.get("effective_sha256")==fingerprint]
    selection_hash=digest(config["selection"])
    same_source=[r["job_id"] for r in history.values() if r.get("selection_sha256")==selection_hash]
    return {"config":config,"config_sha256":digest(config),"effective_sha256":fingerprint,
        "summary":{"job_id":config["job_id"],"reason":reason,"mode":mode,"purpose":purpose,
            "selection":config["selection"],"includes":config["includes"],"limits":config["limits"],
            "profile":config["profile"] if config["profile"] is not None else "custom_limits",
            "exact_targets":config["exact_targets"],
            "request_policy":policy_description(request_policy),
            "cache_input":cache_input,"cache_expectation":expectation,"prefix_cache_enabled":config["profile"] in CACHE_PROFILES,
            "cache_policy":policy_summary(WARM128_POLICY) if config["profile"]==WARM128_PROFILE else None,
            "cache_limits":({"max_logical_compressed_bytes":96*1024**2,"max_local_io_bytes":196*1024**2,
                "max_cache_disk_bytes":128*1024**2,"max_catalog_bytes":6*1024**2,
                "max_catalog_sequences":128,"max_catalog_entries":16000,"export_max_seconds":60,
                "export_max_local_read_bytes":384*1024**2} if config["profile"]==CACHE_PROFILE else
                {"max_logical_compressed_bytes":128*1024**2,"max_local_io_bytes":260*1024**2,
                "max_cache_disk_bytes":160*1024**2,"max_catalog_bytes":6*1024**2,"max_catalog_sequences":256,
                "max_catalog_entries":16000,"max_range_audit_rows":1024,"export_max_seconds":60,
                "export_max_local_read_bytes":512*1024**2} if config["profile"]==WARM128_PROFILE else None),
            "metadata_entry_limit_bytes":min(limits.max_metadata_bytes,64*1024) if config["profile"] in POSE_PROFILES else limits.max_metadata_bytes,
            "stop_when_targets_stored":bool(config["exact_targets"]),
            "total_body_upper_bound_bytes":limits.max_transfer_bytes+metadata.max_transfer_bytes if mode!="review" else 0,
            "total_cooperative_stage_upper_bound_seconds":limits.max_elapsed_seconds+metadata.max_elapsed_seconds if mode!="review" else 0,
            "total_request_upper_bound":limits.max_ranges+metadata.max_requests if mode!="review" else 0,
            "excluded_costs":"authentication; whole wire; OS/RSS hard limits; separate result ZIP; cache upload/export/permanent storage",
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
        "profile":config["profile"] if config["profile"] is not None else "custom_limits","limits":config["limits"],
        **target_summary(config["exact_targets"],[],True),
        "reads_from_byte_zero":True,"train_ready":False}
    record["request_failure"]=None
    record["request_policy"]=policy_description(config.get("request_policy"))
    history[config["job_id"]]=record
    return record


def write_job_record(path, record):
    body={k:v for k,v in record.items() if k!="content_sha256"}
    raw=canonical({**body,"content_sha256":digest(body)})+b"\n"
    if len(raw)>256*1024:
        raise PreparationError("workflow_job_report_bytes_limit")
    Path(path).write_bytes(raw)


def run_workflow_job(config, preflight, history, outputs, session_factory, cache_upload_report=None):
    """명시한 호출 1회. Claim 전 인증 없음. 실패도 job을 소비하고 자동 retry하지 않는다."""
    record=claim_workflow_job(config,preflight,history)
    config=record["config"]
    root=Path(outputs)/("archive_job_"+config["job_id"])
    root.mkdir(parents=True,exist_ok=False)
    record["output"]=str(root.resolve())
    write_job_record(root/"job.json",record)
    session=None
    try:
        if config["profile"] in CACHE_PROFILES and cache_upload_report is not None:
            record["stage"]="cache_upload_record"
            if len(canonical(cache_upload_report))>4096 or cache_upload_report.get("manifest_sha256")!=(config.get("cache_input") or {}).get("manifest_sha256"):
                raise PreparationError("cache_upload_report_mismatch")
            record["cache_upload_report"]=copy.deepcopy(cache_upload_report)
        record["stage"]="authentication"
        session=session_factory()
        if config.get("request_policy") or config["profile"] in CACHE_PROFILES:validate_no_hidden_retries(session)
        record["stage"]="metadata"
        metadata=build_drive_manifest(config["selection"],session,MetadataLimits.from_dict(config["metadata_limits"]))
        save_metadata_bundle(metadata,root/"metadata")
        if not metadata["report"]["manifest_ready"]:
            record.update(status="metadata_failed",stop_reason=metadata["report"]["stop_reason"],
                request_failure=safe_request_failure(metadata["report"].get("request_failure")))
        else:
            record["source_identity_after_metadata"]=metadata["manifest"]
            write_job_record(root/"job.json",record)
            # 선택 ID만의 preflight와 실제 metadata snapshot을 구분해 실행 직전에 표시한다.
            print({"source_identity_after_metadata":metadata["manifest"],"reads_from_byte_zero":True})
            record["stage"]="pilot"
            pilot=run_metadata_pilot(metadata,root/"pilot",config["includes"],Limits.from_dict(config["limits"]),session,
                mode=config["mode"],profile=config["profile"],exact_targets=config["exact_targets"],request_policy=config.get("request_policy"),cache_input=config.get("cache_input"),cache_expectation=config.get("cache_expectation"))
            record.update(status="finished",stop_reason=pilot["stop_reason"],sample_acquired=pilot["sample_acquired"],
                request_failure=safe_request_failure(pilot.get("request_failure")),
                request_retry=pilot["request_retry"],
                observed_regular_members=pilot["observed_regular_members"],profile=pilot["profile"],limits=pilot["limits"],
                **{k:pilot[k] for k in ("exact_targets","stored_targets","completed_targets","missing_targets","targets_complete","targets_status")})
            for key in ("prefix_cache","observation_catalog","cache_limits","cache_policy","timing_scope"):
                if key in pilot:record[key]=pilot[key]
    except KeyboardInterrupt:
        record.update(status="interrupted",stop_reason="interrupted")
    except Exception as exc:
        # 인증/transport exception 원문에는 token이나 URL이 있을 수 있다.
        record.update(status="failed",stop_reason=exc.reason if isinstance(exc,PreparationError) else "workflow_execution_failed")
        record["request_failure"]=request_failure(exc,"authentication" if record["stage"]=="authentication" else "workflow",
            "authentication_setup" if record["stage"]=="authentication" else "unknown")
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
        result["job"]["request_failure"]=safe_request_failure(record.get("request_failure"))
        for key in ("request_policy","request_retry"):
            if key in record:result["job"][key]=record[key]
        result["request_failure"]=safe_request_failure(record.get("request_failure"))
        result["request_failure_recorded"]="request_failure" in record
    extraction_root=root/"pilot" if (root/"pilot").is_dir() else root
    if (extraction_root/"extraction_manifest.json").is_file():
        report=read_report(extraction_root/"extraction_manifest.json")
        result["extraction"]={k:report[k] for k in ("operation","stop_reason","inspection_candidates_usable",
            "source_validation_pending","stats","member_observations","member_observation_trust",
            "member_observation_cap_reached","full_source_listing_claimed","profile","limits",
            "last_regular_observation","regular_members_seen","metadata_entry_limit_bytes",
            "exact_targets","stored_targets","completed_targets","missing_targets","targets_complete","targets_status") if k in report}
        result["complete_stored_files"]=len(report["files"])
        result["extraction"]["request_failure"]=safe_request_failure(report.get("request_failure"))
        for key in ("prefix_cache","observation_catalog","cache_limits","cache_policy","timing_scope"):
            if key in report:result["extraction"][key]=report[key]
        if "observation_catalog" in report:
            summary=report["observation_catalog"]
            if summary["relative_path"]!="observation_catalog.json":raise PreparationError("catalog_report_path")
            catalog=read_report(extraction_root/"observation_catalog.json",6*1024**2)
            if catalog["content_sha256"]!=summary["content_sha256"]:raise PreparationError("catalog_report_hash")
        if "request_retry" in report:result["extraction"]["request_retry"]=report["request_retry"]
        result["request_failure"]=safe_request_failure(report.get("request_failure"))
        result["request_failure_recorded"]="request_failure" in report
        try:
            verify_extraction(extraction_root)
            result["file_verification"]="passed_current_code"
        except (PreparationError,OSError,ValueError,KeyError):
            result["file_verification"]="not_verified_current_code_or_unusable"
    if (extraction_root/"pilot_report.json").is_file():
        pilot=read_report(extraction_root/"pilot_report.json")
        result["sample_acquired"]=pilot.get("sample_acquired",False)
        result["sample_status"]=pilot.get("sample_status","legacy_report")
        result["request_failure"]=safe_request_failure(pilot.get("request_failure"))
        result["request_failure_recorded"]="request_failure" in pilot
        result["request_retry"]=pilot.get("request_retry")
        result["request_retry_recorded"]="request_retry" in pilot
        result["target_verification"]={k:pilot[k] for k in ("completed_targets","missing_targets","targets_complete","targets_status") if k in pilot}
    if len(result)==3:
        raise PreparationError("workflow_no_result_reports")
    return result


def save_result_zip(path):
    root=Path(path).resolve()
    review_run(root)  # 잘못 고른 source/다른 디렉터리를 결과로 압축하지 않는다.
    paths=[]
    extraction_root=root/"pilot" if (root/"pilot").is_dir() else root
    allowed={"job.json","pilot_job.json","metadata","pilot","files","extraction_manifest.json","pilot_report.json","observation_catalog.json","cache_export_report.json"}
    for i,p in enumerate(root.rglob("*")):
        if p.is_symlink() or not p.resolve().is_relative_to(root):raise PreparationError("workflow_result_paths")
        if p==extraction_root/"cache" or p.is_relative_to(extraction_root/"cache") or p==extraction_root/"prefix_cache.zip":continue
        if i>=4096 or p.relative_to(root).parts[0] not in allowed:
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


def save_cache_zip(path):
    root=Path(path);pilot=root/"pilot" if (root/"pilot").is_dir() else root
    return export_cache(pilot/"cache")
