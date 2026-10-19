"""고정 크기 gzip 해제와 tar header/metadata/payload의 순차 선택 추출."""

import fnmatch
import hashlib
import os
import tarfile
import tempfile
import zlib
from dataclasses import asdict
from pathlib import Path

from . import __version__
from .contracts import Budget, Limits, PreparationError, canonical, digest, safe_path, validate_manifest, validate_includes, validate_exact_targets, target_summary
from .profiles import POSE_COUNTERPART_PROFILE, TARGET_PAIR_PROFILES, validate_profile_limits
from .sources import SplitStream
from .diagnostics import safe_request_failure


BENIGN_STOPS = {"prefix_exhausted", "incomplete_header", "incomplete_payload", "incomplete_padding",
    "transfer_bytes_limit", "decompressed_bytes_limit", "elapsed_limit", "files_limit", "members_limit",
    "metadata_bytes_limit", "metadata_entry_bytes_limit", "ranges_limit", "file_bytes_limit", "payload_disk_bytes_limit", "interrupted", "observed_members_limit", "targets_complete"}


MAX_OBSERVED_MEMBERS = 128


def report_reserve_bytes(manifest, includes, limits, exact_targets=None):
    # name JSON escape 최악은 UTF-8 경로 bytes의 2배. 관측목록은 pilot에 복사하지 않는다.
    observation_reserve = min(MAX_OBSERVED_MEMBERS,limits.max_members)*(2*limits.max_path_bytes+512)
    return (16384+2*len(canonical(manifest))+len(canonical(includes))+limits.max_ranges*1024
        +limits.max_files*(3*limits.max_path_bytes+1024)+observation_reserve
        +2*(2*limits.max_path_bytes+512)  # last regular 행을 extraction/pilot에 각각 예약
        +(6*len(canonical(exact_targets))+1024 if exact_targets else 0))


def code_hashes():
    return {p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(Path(__file__).parent.glob("*.py"))}


class GzipReader:
    def __init__(self, source, budget):
        self.source, self.budget = source, budget
        self.z = zlib.decompressobj(31)
        self.pending = self.buffer = b""
        self.finished = self.gzip_eof_crc_valid = False
        self.drain = False

    def fill(self):
        b, limits = self.budget, self.budget.limits
        while not self.buffer and not self.finished:
            b.check_time()
            if not self.pending and not self.drain:
                self.pending = self.source.read()
                if not self.pending:
                    self.finished = True
                    return
            remaining = limits.max_decompressed_bytes-b.decompressed_bytes
            if remaining <= 0:
                raise PreparationError("decompressed_bytes_limit")
            try:
                maximum = min(limits.io_chunk_bytes,remaining)
                output = self.z.decompress(self.pending, maximum)
            except zlib.error:
                raise PreparationError("gzip_corrupt") from None
            self.pending = self.z.unconsumed_tail
            self.drain = len(output)==maximum
            b.decompressed_bytes += len(output)
            b.peak_output_chunk_bytes = max(b.peak_output_chunk_bytes,len(output))
            self.buffer = output
            if self.z.eof:
                self.gzip_eof_crc_valid = True
                if self.z.unused_data or not self.source.at_end:
                    raise PreparationError("trailing_or_concatenated_gzip")
                self.pending = b""
                self.finished = True

    def read(self, size):
        if not 0 <= size <= self.budget.limits.io_chunk_bytes:
            raise PreparationError("read_buffer_limit")
        chunks, remaining = [], size
        while remaining:
            self.fill()
            if not self.buffer:
                break
            take = min(remaining,len(self.buffer))
            chunks.append(self.buffer[:take])
            self.buffer = self.buffer[take:]
            remaining -= take
        return b"".join(chunks)

    def exact(self, size, reason):
        data = self.read(size)
        if len(data) != size:
            raise PreparationError(reason)
        return data


def pax_fields(data):
    result = {}
    position = 0
    while position < len(data):
        space = data.find(b" ",position,min(len(data),position+12))
        if space < 0 or not data[position:space].isdigit():
            raise PreparationError("invalid_pax")
        length = int(data[position:space])
        end = position+length
        if end > len(data) or end <= space+1 or data[end-1:end] != b"\n":
            raise PreparationError("invalid_pax")
        record = data[space+1:end-1]
        if b"=" not in record:
            raise PreparationError("invalid_pax")
        key, value = record.split(b"=",1)
        try:
            key, value = key.decode("utf-8"), value.decode("utf-8")
        except UnicodeError:
            raise PreparationError("invalid_pax_encoding") from None
        if key in result or key.startswith(("GNU.sparse", "SCHILY")) or key == "linkpath":
            raise PreparationError("unsupported_pax")
        if key not in ("path","size","mtime","atime","ctime","uid","gid","uname","gname"):
            raise PreparationError("unsupported_pax")
        result[key] = value
        position = end
    return result


def read_payload(reader, size, sink=None, on_chunk=None):
    remaining = size
    while remaining:
        chunk = reader.read(min(remaining,reader.budget.limits.io_chunk_bytes))
        if not chunk:
            raise PreparationError("incomplete_payload")
        if sink is not None:
            sink(chunk)
        if on_chunk is not None:
            on_chunk(chunk)
        remaining -= len(chunk)
    padding = (-size)%512
    if padding:
        if reader.exact(padding,"incomplete_padding") != bytes(padding):
            raise PreparationError("nonzero_tar_padding")


def prepare(manifest, output, includes, limits, base=".", authorized_session=None, clock=None, mode="extract", profile=None, exact_targets=None):
    """단일 writer용 새 출력. 중단에도 완전한 후보와 inspection report를 남긴다."""
    validate_manifest(manifest)
    if not isinstance(limits,Limits):
        raise PreparationError("limits_required")
    if mode not in ("extract","scan"):
        raise PreparationError("preparation_mode")
    includes = [] if mode=="scan" and includes==[] else validate_includes(includes)
    exact_targets = validate_exact_targets(exact_targets,includes,limits,mode)
    if profile is not None:
        validate_profile_limits(profile,limits)
        if profile in TARGET_PAIR_PROFILES and (mode!="extract" or len(exact_targets)!=2):
            raise PreparationError("counterpart_profile_requires_two_exact_targets")
        if profile==POSE_COUNTERPART_PROFILE and manifest["kind"]!="pose":
            raise PreparationError("pose_profile_requires_pose_kind")
    metadata_entry_limit = min(limits.max_metadata_bytes,64*1024) if profile==POSE_COUNTERPART_PROFILE else limits.max_metadata_bytes
    budget = Budget(limits,**({"clock":clock} if clock else {}))
    # Metadata report 자체의 logical bytes도 disk 상한 안에 예약한다.
    reserve = report_reserve_bytes(manifest,includes,limits,exact_targets)
    if reserve >= limits.max_disk_bytes:
        raise PreparationError("disk_report_reserve")
    output = Path(output).resolve()
    output.mkdir(parents=True,exist_ok=False)
    partial_dir = output/".partials"
    partial_dir.mkdir()
    files, active, reason = [], None, "unknown"
    failure = None
    observations, observation_cap_reached = [], False
    last_regular, regular_count = None, 0
    tar_eof = False
    source = reader = None
    try:
        source = SplitStream(manifest,base,budget,authorized_session)
        reader = GzipReader(source,budget)
        pending_name, pending_pax = None, {}
        seen = set()
        while True:
            header = reader.read(512)
            if not header:
                raise PreparationError("prefix_exhausted")
            if len(header) != 512:
                raise PreparationError("incomplete_header")
            if header == bytes(512):
                if reader.exact(512,"incomplete_header") != bytes(512) or pending_name is not None or pending_pax:
                    raise PreparationError("invalid_tar_end")
                tar_eof = True
                # Tar 종료 뒤의 zero padding도 읽어 gzip trailer/CRC까지 검증한다.
                while tail := reader.read(limits.io_chunk_bytes):
                    if tail.strip(b"\0"):
                        raise PreparationError("nonzero_tar_tail")
                reason = "complete" if reader.gzip_eof_crc_valid else "prefix_exhausted"
                break
            budget.add("members",1,limits.max_members)
            try:
                info = tarfile.TarInfo.frombuf(header,"utf-8","strict")
            except (tarfile.HeaderError,UnicodeError,ValueError):
                raise PreparationError("invalid_tar_header") from None
            if info.size < 0:
                raise PreparationError("invalid_tar_size")
            safe_path(info.name,limits.max_path_bytes)
            if info.type in (tarfile.GNUTYPE_LONGNAME,tarfile.XHDTYPE,tarfile.XGLTYPE):
                if profile==POSE_COUNTERPART_PROFILE and info.size>metadata_entry_limit:
                    raise PreparationError("metadata_entry_bytes_limit")
                budget.add("metadata_bytes",info.size,limits.max_metadata_bytes)
                chunks = []
                read_payload(reader,info.size,chunks.append)
                data = b"".join(chunks)
                if info.type == tarfile.GNUTYPE_LONGNAME:
                    if pending_name is not None or not data.endswith(b"\0") or b"\0" in data[:-1]:
                        raise PreparationError("invalid_gnu_longname")
                    try:
                        pending_name = safe_path(data[:-1].decode("utf-8"),limits.max_path_bytes)
                    except UnicodeError:
                        raise PreparationError("invalid_name_encoding") from None
                else:
                    values = pax_fields(data)
                    if info.type == tarfile.XGLTYPE:
                        if set(values)&{"path","size"}:
                            raise PreparationError("unsupported_global_pax")
                    elif pending_pax:
                        raise PreparationError("duplicate_pax_header")
                    else:
                        pending_pax = values
                continue
            name = safe_path(pending_pax.get("path",pending_name or info.name),limits.max_path_bytes)
            size = info.size
            if "size" in pending_pax:
                if not re_size(pending_pax["size"]):
                    raise PreparationError("invalid_pax_size")
                size = int(pending_pax["size"])
            pending_name, pending_pax = None, {}
            key = name.casefold()
            if key in seen:
                raise PreparationError("duplicate_tar_path")
            seen.add(key)
            if info.type == tarfile.DIRTYPE:
                if size != 0:
                    raise PreparationError("directory_payload")
                continue
            if info.type not in (tarfile.REGTYPE,tarfile.AREGTYPE):
                raise PreparationError("links_or_special_file")
            selected = any(fnmatch.fnmatchcase(name,pattern) for pattern in includes)
            regular_count += 1
            observed = {"header_ordinal":budget.members,"regular_ordinal":regular_count,
                "member":name,"size_bytes":size,"selected":selected,"payload_complete":False,
                "payload_bytes_returned":0,"stored":False}
            last_regular = observed
            if len(observations)>=min(MAX_OBSERVED_MEMBERS,limits.max_members):
                observation_cap_reached = True
                if mode=="scan":
                    raise PreparationError("observed_members_limit")
            else:
                observations.append(observed)
            def observe(chunk):
                if observed is not None:
                    observed["payload_bytes_returned"] += len(chunk)
            active = {"member":name,"expected_bytes":size,"selected":selected,"written_bytes":0}
            if size > limits.max_file_bytes:
                raise PreparationError("file_bytes_limit")
            if mode=="scan" or not selected:
                read_payload(reader,size,on_chunk=observe)
                if observed is not None:
                    observed["payload_complete"] = True
                active = None
                continue
            if budget.files >= limits.max_files:
                raise PreparationError("files_limit")
            if budget.payload_disk_bytes+size+reserve > limits.max_disk_bytes:
                raise PreparationError("payload_disk_bytes_limit")
            target = output/"files"/name
            target.parent.mkdir(parents=True,exist_ok=True)
            if not target.resolve().is_relative_to(output) or any(p.is_symlink() for p in [target,*target.parents] if p.is_relative_to(output)):
                raise PreparationError("unsafe_output_path")
            fingerprint = hashlib.sha256()
            temporary = None
            try:
                with tempfile.NamedTemporaryFile(dir=partial_dir, suffix=".partial",delete=False) as f:
                    temporary = Path(f.name)
                    def write(chunk):
                        budget.check_time()
                        f.write(chunk)
                        fingerprint.update(chunk)
                        active["written_bytes"] += len(chunk)
                        budget.peak_payload_disk_bytes = max(budget.peak_payload_disk_bytes,budget.payload_disk_bytes+active["written_bytes"])
                    read_payload(reader,size,write,on_chunk=observe)
                    if observed is not None:
                        observed["payload_complete"] = True
                budget.check_time()
                # Hard link 생성은 기존 파일을 덮어쓰지 않는다. 같은 filesystem 안에서만 사용.
                if os.name == "nt":
                    os.rename(temporary,target)  # Windows에서는 기존 destination이 있으면 실패한다.
                else:
                    os.link(temporary,target)
                    temporary.unlink()
                temporary = None
                files.append({"member":name,"relative_path":"files/"+name,"size_bytes":size,"sha256":fingerprint.hexdigest(),
                    "tar_complete":True,"kind":manifest["kind"]})
                if observed is not None:
                    observed["stored"] = True
                budget.files += 1
                budget.payload_disk_bytes += size
                active = None
                if exact_targets and set(exact_targets).issubset({entry["member"] for entry in files}):
                    reason = "targets_complete"
                    break  # 다음 tar header/Range를 읽지 않는다. 받은 Range의 선수신은 별도다.
            finally:
                if temporary is not None and temporary.exists():
                    temporary.unlink()
    except PreparationError as exc:
        reason = exc.reason
        failure = safe_request_failure(exc.request_failure)
    except KeyboardInterrupt:
        reason = "interrupted"
    except OSError:
        reason = "output_or_read_error"
    finally:
        partial_dir.rmdir()
    complete_source = len(manifest["parts"])==manifest["total_parts"]
    malformed_complete_tar = bool(reader and reader.gzip_eof_crc_valid and not tar_eof and reason in {"prefix_exhausted","incomplete_header","incomplete_payload","incomplete_padding"})
    pending_validation = bool(source and any(getattr(s,"pending_validation",False) for s in source.sources))
    usable = (reason=="complete" or reason in BENIGN_STOPS) and not malformed_complete_tar and not pending_validation
    failure_class = "complete_declared_input" if reason=="complete" else ("bounded_stop" if reason.endswith("_limit") or reason in {"interrupted","targets_complete"} else "incomplete_contiguous_prefix") if usable else "invalid_or_changed_source"
    report = {"schema_version":"archive_extraction_v1","extractor_version":__version__,"code_sha256":code_hashes(),
        "archive_manifest":manifest,"archive_manifest_sha256":digest(manifest),"selection":includes,"limits":asdict(limits),
        "stop_reason":reason,"request_failure":failure,"failure_class":failure_class,"active_incomplete_member":active,"files":files,"range_audit":source.audit if source else [],
        "stats":budget.stats(),"compressed_bytes_received":source.compressed_bytes_received if source else 0,
        "gzip_eof_crc_valid":reader.gzip_eof_crc_valid if reader else False,"tar_end_blocks_seen":tar_eof,
        "declared_source_complete":complete_source,
        "whole_archive_integrity_verified":reason=="complete" and complete_source and manifest["source_scope"]!="official_prefix_local",
        "inspection_candidates_usable":usable,"source_validation_pending":pending_validation,"train_ready":False,
        "operation":mode,"member_observations":observations,
        "last_regular_observation":last_regular,"regular_members_seen":regular_count,
        "metadata_entry_limit_bytes":metadata_entry_limit,
        "member_observation_scope":"reached_regular_headers_only_not_a_seek_index",
        "member_observation_limit":min(MAX_OBSERVED_MEMBERS,limits.max_members),
        "member_observation_cap_reached":observation_cap_reached,
        "member_observations_usable":usable,
        "member_observation_trust":"bounded_observation_only" if usable else "untrusted_failed_source_or_parser",
        "full_source_listing_claimed":False,
        "profile":profile if profile is not None else "custom_limits",
        **target_summary(exact_targets,files,usable),
        "report_disk_reserve_bytes":reserve}
    raw = canonical({**report,"content_sha256":digest(report)})+b"\n"
    if len(raw)>reserve:
        raise PreparationError("report_reserve_exceeded")
    (output/"extraction_manifest.json").write_bytes(raw)
    return report


def re_size(value):
    return isinstance(value,str) and value.isascii() and value.isdigit() and len(value)<=20
