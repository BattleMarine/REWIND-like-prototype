"""명시한 Drive part의 metadata만 검증해 archive manifest를 만든다."""

import os
import hashlib
import platform
import re
import subprocess
import tempfile
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

from rewind_data.audit import read_json
from . import __version__
from .contracts import Budget, PreparationError, canonical, digest, validate_manifest
from .extraction import code_hashes
from .sources import read_metadata_body, metadata_diagnostic, session_opener


FIELDS = "id,name,size,version,modifiedTime,mimeType,trashed,capabilities(canDownload)"


@dataclass(frozen=True)
class MetadataLimits:
    max_files: int = 8
    max_requests: int = 8
    max_response_bytes: int = 16384
    max_transfer_bytes: int = 131072
    max_elapsed_seconds: float = 30.0
    request_timeout_seconds: float = 5.0
    io_chunk_bytes: int = 4096

    def __post_init__(self):
        for key, value in asdict(self).items():
            if key.endswith("seconds"):
                if type(value) not in (int,float) or not 0<value<float("inf"):
                    raise PreparationError("metadata_limits")
            elif type(value) is not int or value<1:
                raise PreparationError("metadata_limits")
        if self.max_files>32 or self.max_requests>64 or self.max_response_bytes>65536 or self.max_transfer_bytes>2097152 or not 512<=self.io_chunk_bytes<=65536:
            raise PreparationError("metadata_limits")
        if self.request_timeout_seconds>min(30,self.max_elapsed_seconds):
            raise PreparationError("metadata_limits")

    @classmethod
    def from_dict(cls, doc):
        if not isinstance(doc,dict) or set(doc)!=set(asdict(cls())):
            raise PreparationError("metadata_limits_fields")
        return cls(**doc)


def provenance():
    root=Path(__file__).resolve().parents[1]
    revision,dirty="unversioned",None
    stamp=root/"SOURCE_REVISION.txt"
    if stamp.exists():
        text=stamp.read_text(encoding="utf-8").strip()
        if re.fullmatch(r"[0-9a-fA-F]{40}",text):
            revision=text
    elif (root/".git").exists():
        try:
            revision=subprocess.run(["git","rev-parse","HEAD"],cwd=root,capture_output=True,text=True,check=True,timeout=5).stdout.strip()
            dirty=bool(subprocess.run(["git","status","--porcelain"],cwd=root,capture_output=True,text=True,check=True,timeout=5).stdout.strip())
        except (OSError,subprocess.SubprocessError):
            pass
    return {"extractor_version":__version__,"revision":revision,"dirty":dirty,"code_sha256":code_hashes()}


def environment():
    return {"python":platform.python_version(),"system":platform.system(),"machine":platform.machine(),"phase_device":"cpu"}


def validate_selection(selection):
    if not isinstance(selection,dict) or set(selection)!={"schema_version","archive_id","dataset_version","kind","total_parts","parts"} or selection["schema_version"]!="drive_part_selection_v1":
        raise PreparationError("drive_selection_fields")
    parts=selection["parts"]
    if not isinstance(parts,list) or not 1<=len(parts)<=32:
        raise PreparationError("drive_selection_parts")
    placeholders=[]
    ids=set()
    stems=set()
    for part in parts:
        if not isinstance(part,dict) or set(part)!={"index","name","file_id"}:
            raise PreparationError("drive_selection_part_fields")
        file_id=part["file_id"]
        if not isinstance(file_id,str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,200}",file_id) or file_id in ids:
            raise PreparationError("drive_selection_id")
        ids.add(file_id)
        name=part["name"]
        match=re.fullmatch(r"(.+\.tar\.gz)[0-9]{3}",name) if isinstance(name,str) else None
        if not match:
            raise PreparationError("drive_selection_archive_name")
        stems.add(match.group(1))
        if len(stems)>1:
            raise PreparationError("drive_selection_archive_stem")
        placeholders.append({"index":part["index"],"name":part["name"],"size_bytes":1,"version":"selection_validation_only",
            "source":{"type":"drive","file_id":file_id},"identity":{"version":"1","modified_time":"selection_validation_only"}})
    # 원본 값을 추측한 manifest를 반환하지 않는다. 기존 규격의 index/name/identity 검사만 재사용한다.
    validate_manifest({"schema_version":"archive_manifest_v1","archive_id":selection["archive_id"],"dataset_version":selection["dataset_version"],
        "kind":selection["kind"],"source_scope":"user_drive","total_parts":selection["total_parts"],"parts":placeholders})
    return selection


def validate_file(doc, expected):
    if not isinstance(doc,dict):
        raise PreparationError("metadata_object_required")
    required={"id","name","size","version","modifiedTime","mimeType","trashed","capabilities"}
    if not required.issubset(doc):
        raise PreparationError("metadata_required_fields")
    if doc["id"]!=expected["file_id"] or doc["name"]!=expected["name"]:
        raise PreparationError("metadata_identity_mismatch")
    mime=doc["mimeType"]
    if not isinstance(mime,str) or not re.fullmatch(r"[A-Za-z0-9!#$&^_.+-]{1,100}/[A-Za-z0-9!#$&^_.+-]{1,100}",mime):
        raise PreparationError("metadata_mime_type")
    if mime.lower().startswith("application/vnd.google-apps."):
        raise PreparationError("metadata_non_blob")
    if doc["trashed"] is not False:
        raise PreparationError("metadata_trashed_or_unknown")
    caps=doc["capabilities"]
    if not isinstance(caps,dict) or caps.get("canDownload") is not True:
        raise PreparationError("metadata_download_unavailable")
    for key in ("size","version"):
        value=doc[key]
        if not isinstance(value,str) or not re.fullmatch(r"[1-9][0-9]{0,18}",value) or int(value)>2**63-1:
            raise PreparationError("metadata_"+key)
    modified=doc["modifiedTime"]
    if not isinstance(modified,str) or not re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d{1,9})?(?:Z|[+-]\d{2}:\d{2})",modified):
        raise PreparationError("metadata_modified_time")
    try:
        datetime.fromisoformat(modified.replace("Z","+00:00"))
    except ValueError:
        raise PreparationError("metadata_modified_time") from None
    return {"id":doc["id"],"name":doc["name"],"size":doc["size"],"version":doc["version"],"modifiedTime":modified,
        "mimeType":mime,"trashed":False,"capabilities":{"canDownload":True}}


def build_drive_manifest(selection, authorized_session, limits, clock=time.monotonic):
    """Media 없이 성공 시에만 manifest 반환. 실패는 비밀을 담지 않는 report로 기록한다."""
    if not isinstance(limits,MetadataLimits):
        raise PreparationError("metadata_limits_required")
    budget=Budget(limits,clock=clock)
    manifest=None
    files=[]
    request_audit=[]
    reason="unknown"
    selection_hash=None
    try:
        validate_selection(selection)
        selection_hash=digest(selection)
        if len(selection["parts"])>limits.max_files:
            raise PreparationError("metadata_files_limit")
        if authorized_session is None:
            raise PreparationError("drive_session_required")
        opener=session_opener(authorized_session)
        validated=[]
        for part in selection["parts"]:
            budget.request("metadata", limits.max_requests)
            record={"part_index":part["index"],"bytes_received":0,"status":"pending"}
            request_audit.append(record)
            received=hashlib.sha256()
            def on_chunk(chunk):
                record["bytes_received"]+=len(chunk)
                received.update(chunk)
            response=None
            try:
                url=f"https://www.googleapis.com/drive/v3/files/{part['file_id']}?fields={FIELDS}"
                response=opener(url,{"Accept-Encoding":"identity"},budget.timeout())
                record["response"]=metadata_diagnostic(response)
                raw=read_metadata_body(response,budget,limits.max_response_bytes,on_chunk)
                try:
                    doc=read_json(raw)
                except (ValueError,UnicodeError):
                    raise PreparationError("metadata_json") from None
                snapshot=validate_file(doc,part)
                validated.append({"index":part["index"],"name":snapshot["name"],"size_bytes":int(snapshot["size"]),
                    "version":snapshot["version"],"source":{"type":"drive","file_id":snapshot["id"]},
                    "identity":{"version":snapshot["version"],"modified_time":snapshot["modifiedTime"]}})
                files.append(snapshot)
                budget.files+=1
                budget.check_time()
                record["status"]="validated"
            except PreparationError as exc:
                record["status"],record["reason"]="failed",exc.reason
                raise
            finally:
                record["sha256_received_bytes"]=received.hexdigest()
                if response is not None:
                    response.close()
        manifest={"schema_version":"archive_manifest_v1","archive_id":selection["archive_id"],"dataset_version":selection["dataset_version"],
            "kind":selection["kind"],"source_scope":"user_drive","total_parts":selection["total_parts"],"parts":validated}
        validate_manifest(manifest)
        budget.check_time()
        reason="complete"
    except PreparationError as exc:
        manifest=None;reason=exc.reason
    except KeyboardInterrupt:
        manifest=None;reason="interrupted"
    except Exception:
        manifest=None;reason="metadata_request_or_input_failed"
    elapsed=budget.clock()-budget.started
    report={"schema_version":"drive_metadata_report_v1","phase":"metadata_only","manifest_ready":manifest is not None,
        "stop_reason":reason,"selection_sha256":selection_hash,"manifest_sha256":digest(manifest) if manifest else None,
        "source":provenance(),"environment":environment(),"limits":asdict(limits),
        "stats":{"requests":budget.ranges,"transfer_bytes":budget.transfer_bytes,"validated_files":budget.files,"elapsed_seconds":elapsed},
        "validated_metadata":files,"request_audit":request_audit,"media_requested_by_builder":False,"train_ready":False}
    report["content_sha256"]=digest(report)
    return {"manifest":manifest,"report":report}


def write_new_json(path, doc):
    path=Path(path)
    temporary=None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent,suffix=".tmp",delete=False) as f:
            temporary=Path(f.name)
            f.write(canonical(doc)+b"\n")
        if os.name=="nt":
            os.rename(temporary,path)
        else:
            os.link(temporary,path)
            temporary.unlink()
        temporary=None
    except OSError:
        raise PreparationError("bundle_write_failed") from None
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def validate_metadata_result(result):
    if not isinstance(result,dict) or set(result)!={"manifest","report"} or not isinstance(result["report"],dict):
        raise PreparationError("metadata_result_invalid")
    report=result["report"]
    if report.get("schema_version")!="drive_metadata_report_v1" or report.get("content_sha256")!=digest({k:v for k,v in report.items() if k!="content_sha256"}):
        raise PreparationError("metadata_report_hash")
    if report.get("manifest_ready") is not True or report.get("stop_reason")!="complete" or report.get("train_ready") is not False:
        raise PreparationError("metadata_manifest_unavailable")
    manifest=validate_manifest(result["manifest"])
    if report.get("manifest_sha256")!=digest(manifest) or report["source"]["code_sha256"]!=code_hashes():
        raise PreparationError("metadata_manifest_or_code_changed")
    if manifest["source_scope"]!="user_drive" or any(p["source"]["type"]!="drive" for p in manifest["parts"]):
        raise PreparationError("metadata_manifest_type")
    return manifest


def save_metadata_bundle(result, output):
    """실패는report만 남기고 성공은manifest/report를 새 디렉터리에 기록한다."""
    if result["manifest"] is not None:
        validate_metadata_result(result)
    output=Path(output)
    try:
        output.mkdir(parents=True,exist_ok=False)
    except OSError:
        raise PreparationError("bundle_output_exists_or_unavailable") from None
    manifest_path=output/"archive_manifest.json"
    try:
        if result["manifest"] is not None:
            write_new_json(manifest_path,result["manifest"])
        write_new_json(output/"metadata_report.json",result["report"])
    except PreparationError:
        if manifest_path.exists():
            manifest_path.unlink()
        raise
    return output
