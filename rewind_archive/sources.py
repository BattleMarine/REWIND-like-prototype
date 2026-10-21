"""순차 Range 읽기. 토큰/인증 session은 런타임에만 주입한다."""

import hashlib
import json
import os
import re
import urllib.error
import urllib.request
from pathlib import Path

from .contracts import PreparationError
from .diagnostics import attach_failure, request_failure, safe_request_failure
from .request_policy import (PRE_OPEN_TIMEOUT_SECONDS, MAX_RETRIES_PER_PRE, MAX_RETRIES_TOTAL,
    RETRY_DELAY_SECONDS, eligible_open_timeout, mark_open_timeout, validate_no_hidden_retries)


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise PreparationError("redirect_rejected")


def http_open(url, headers, timeout):
    request = urllib.request.Request(url, headers=headers)
    # 환경 proxy·Cookie·redirect를 사용하지 않는다. TLS 인증서 검증은 기본값이다.
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect())
    try:
        return opener.open(request, timeout=timeout)
    except PreparationError as exc:
        raise attach_failure(exc,"media","request_open")
    except urllib.error.HTTPError as exc:
        diagnostic = metadata_diagnostic(exc)
        close_response(exc,"media","http_status_rejected",preserve_active=True)
        raise PreparationError("http_status_rejected", diagnostic,request_failure(exc,"media","request_open")) from None
    except (OSError, urllib.error.URLError) as exc:
        raise PreparationError("http_open_failed",request_failure=request_failure(exc,"media","request_open")) from None


class SessionResponse:
    def __init__(self, response):
        self.response = response
        self.status, self.headers = response.status_code, response.headers
        if not callable(getattr(self.headers,"get",None)):
            raise TypeError("response_headers_mapping_required")

    def read1(self, size):
        read1 = getattr(self.response.raw,"read1",None)
        if not callable(read1):
            raise PreparationError("authorized_transport_read1_required")
        return read1(size, decode_content=False)

    def set_timeout(self, timeout):
        fp = getattr(self.response.raw,"_fp",None)
        sock = getattr(getattr(getattr(fp,"fp",None),"raw",None),"_sock",None)
        if sock is not None:
            sock.settimeout(timeout)

    def close(self):
        self.response.close()


def session_opener(session, operation="unknown"):
    if getattr(session,"_max_refresh_attempts",0) != 0:
        raise PreparationError("session_retry_policy")
    def open_response(url, headers, timeout):
        try:
            response=session.get(url, headers=headers, stream=True, allow_redirects=False, timeout=timeout)
        except Exception as exc:
            # Session 예외에 token/URL이 들어 있더라도 로그에 전달하지 않는다.
            error=PreparationError("authorized_request_failed",request_failure=request_failure(exc,operation,"request_open"))
            raise mark_open_timeout(error,exc,operation) from None
        try:
            return SessionResponse(response)
        except Exception as exc:
            try:response.close()
            except Exception:pass
            raise PreparationError("authorized_request_failed",request_failure=request_failure(exc,operation,"response_wrap")) from None
    return open_response


def close_response(response, operation, reason, preserve_active=False):
    try:response.close()
    except Exception as exc:
        if preserve_active:return
        if reason=="output_or_read_error" and not isinstance(exc,OSError):reason="workflow_execution_failed"
        raise PreparationError(reason,request_failure=request_failure(exc,operation,"response_close")) from None


def read_one(response, size, budget, on_chunk, operation="unknown"):
    timeout = budget.timeout()
    try:
        if hasattr(response,"set_timeout"):
            response.set_timeout(timeout)
        sock = getattr(getattr(getattr(response, "fp", None), "raw", None), "_sock", None)
        if sock is not None:
            sock.settimeout(timeout)
        chunk = response.read1(size)
    except PreparationError as exc:
        raise attach_failure(exc,operation,"body_read")
    except Exception as exc:
        try:partial = getattr(exc,"partial",b"")
        except Exception:partial=b""
        if isinstance(partial,bytes):
            budget.transfer_bytes += len(partial)
            on_chunk(partial)
        raise PreparationError("http_body_failed",request_failure=request_failure(exc,operation,"body_read")) from None
    budget.transfer_bytes += len(chunk)
    on_chunk(chunk)
    if budget.transfer_bytes > budget.limits.max_transfer_bytes:
        raise PreparationError("transfer_bytes_limit")
    budget.check_time()
    return chunk


def read_body(response, expected, budget, on_chunk, operation="unknown"):
    data = bytearray()
    while len(data) < expected:
        chunk = read_one(response,min(expected-len(data),budget.limits.io_chunk_bytes),budget,on_chunk,operation)
        if not chunk:
            raise PreparationError("short_http_body")
        data.extend(chunk)
    return bytes(data)


def metadata_diagnostic(response):
    """값이 정해진 status/헤더 분류만 기록한다. 원본 URL/헤더/본문은 기록하지 않는다."""
    length=response.headers.get("Content-Length")
    encoding=response.headers.get("Content-Encoding")
    transfer=response.headers.get("Transfer-Encoding")
    content_type=response.headers.get("Content-Type")
    return {"status":response.status if type(response.status) is int and 100<=response.status<=599 else None,
        "content_length":None if length is None else int(length) if isinstance(length,str) and re.fullmatch(r"[0-9]{1,6}",length) else "invalid",
        "content_encoding":"absent" if encoding is None else "identity" if isinstance(encoding,str) and encoding.strip().lower()=="identity" else "unsupported",
        "transfer_encoding":"absent" if transfer is None else "chunked" if isinstance(transfer,str) and transfer.strip().lower()=="chunked" else "unsupported",
        "content_type":"absent" if content_type is None else "json" if isinstance(content_type,str) and content_type.split(";",1)[0].strip().lower()=="application/json" else "other"}


def read_metadata_body(response, budget, maximum, on_chunk, protocol_reason="metadata_protocol", limit_reason="metadata_response_bytes_limit", operation="unknown"):
    try:
        diagnostic=metadata_diagnostic(response)
        length=diagnostic["content_length"]
        if diagnostic["status"]!=200 or diagnostic["transfer_encoding"]=="unsupported" or (length is not None and diagnostic["transfer_encoding"]!="absent"):
            raise PreparationError(protocol_reason)
        if diagnostic["content_encoding"]=="unsupported":
            raise PreparationError("metadata_content_encoding")
        if length is not None:
            if length=="invalid":
                raise PreparationError(protocol_reason)
            if not 0<length<=maximum:
                raise PreparationError(limit_reason)
            if length>budget.limits.max_transfer_bytes-budget.transfer_bytes:
                raise PreparationError("transfer_bytes_limit")
    except PreparationError as exc:
        raise attach_failure(exc,operation,"response_validation")
    if length is not None:
        return read_body(response,length,budget,on_chunk,operation)
    data=bytearray()
    while True:
        # EOF 확인용 1byte도 budget을 초과해 읽지 않는다. Exact cap은 미확정으로 거부한다.
        remaining=maximum-len(data)
        global_remaining=budget.limits.max_transfer_bytes-budget.transfer_bytes
        if global_remaining<=0:
            raise PreparationError("transfer_bytes_limit")
        if remaining<=0:
            raise PreparationError(limit_reason)
        chunk=read_one(response,min(budget.limits.io_chunk_bytes,remaining,global_remaining),budget,on_chunk,operation)
        if not chunk:
            return bytes(data)
        data.extend(chunk)


class LocalSource:
    def __init__(self, part, base):
        self.part = part
        self.path = (Path(base)/part["source"]["path"]).resolve()

    def read_range(self, start, end, budget, on_chunk):
        budget.request("local")
        try:
            with self.path.open("rb") as f:
                before = os.fstat(f.fileno())
                if before.st_size != self.part["size_bytes"] or before.st_mtime_ns != self.part["identity"]["mtime_ns"]:
                    raise PreparationError("source_changed")
                f.seek(start)
                chunk = f.read(end-start+1)
                budget.transfer_bytes += len(chunk)
                on_chunk(chunk)
                after = os.fstat(f.fileno())
                current = self.path.stat()
                if (after.st_size, after.st_mtime_ns) != (before.st_size, before.st_mtime_ns) or (current.st_dev,current.st_ino) != (after.st_dev,after.st_ino):
                    raise PreparationError("source_changed")
        except OSError:
            raise PreparationError("local_read_failed") from None
        budget.check_time()
        if len(chunk) != end-start+1:
            raise PreparationError("short_local_body")
        return chunk


class HttpSource:
    def __init__(self, part, opener=http_open):
        self.part, self.opener = part, opener
        self.url = part["source"]["url"]
        self.last_diagnostic = None

    def read_range(self, start, end, budget, on_chunk):
        self.last_diagnostic = None
        headers = {"Range":f"bytes={start}-{end}","Accept-Encoding":"identity"}
        identity = self.part["identity"]
        if identity.get("etag"):
            headers["If-Match"] = identity["etag"]
        if identity.get("last_modified"):
            headers["If-Unmodified-Since"] = identity["last_modified"]
        budget.request("media")
        response = self.opener(self.url, headers, budget.timeout())
        stage="response_validation"
        failed=True
        try:
            self.last_diagnostic = metadata_diagnostic(response)
            if response.status != 206:
                # Range를 무시한 200 본문은 읽지 않는다.
                raise PreparationError("range_status_not_206")
            expected = end-start+1
            content_range = response.headers.get("Content-Range", "")
            if content_range != f"bytes {start}-{end}/{self.part['size_bytes']}":
                raise PreparationError("content_range_mismatch")
            if response.headers.get("Content-Length") != str(expected):
                raise PreparationError("content_length_mismatch")
            if response.headers.get("Content-Encoding", "identity").lower() != "identity":
                raise PreparationError("unexpected_content_encoding")
            for field, header in (("etag","ETag"),("last_modified","Last-Modified")):
                if field in identity and response.headers.get(header) != identity[field]:
                    raise PreparationError("source_changed")
            stage="body_read"
            chunk=read_body(response, expected, budget, on_chunk,"media")
            failed=False
            return chunk
        except PreparationError as exc:
            raise attach_failure(exc,"media",stage)
        except Exception as exc:
            reason="output_or_read_error" if isinstance(exc,OSError) else "workflow_execution_failed"
            raise PreparationError(reason,request_failure=request_failure(exc,"media",stage)) from None
        finally:
            close_response(response,"media","output_or_read_error",preserve_active=failed)


class DriveSource(HttpSource):
    """Colab AuthorizedSession 주입. 이 모듈은 인증 시작·token 저장을 하지 않는다."""
    def __init__(self, part, authorized_session):
        if authorized_session is None:
            raise PreparationError("drive_session_required")
        self.part, self.opener = part, session_opener(authorized_session,"media")
        self.authorized_session = authorized_session
        file_id = part["source"]["file_id"]
        self.url = f"https://www.googleapis.com/drive/v3/files/{file_id}?alt=media"
        self.metadata_url = f"https://www.googleapis.com/drive/v3/files/{file_id}?fields=size,version,modifiedTime"
        self.last_diagnostic = None
        self.pending_validation = False

    METADATA_RESPONSE_BYTES = 16384

    def check_metadata(self, budget, operation="metadata_pre", media_bytes=0):
        if not budget._request_policy or operation!="metadata_pre":
            return self._check_metadata_once(budget,operation)
        validate_no_hidden_retries(self.authorized_session)
        attempt=0;previous_failure=None;retry_event=None
        while True:
            try:
                # 대기 후 실제 시도 직전의 검사 실패도 앞선 timeout을 보존한다.
                budget.reserve_requests(3)
                if media_bytes+2*self.METADATA_RESPONSE_BYTES > budget.limits.max_transfer_bytes-budget.transfer_bytes:
                    raise PreparationError("transfer_bytes_limit")
                return self._check_metadata_once(budget,operation,
                    timeout=min(PRE_OPEN_TIMEOUT_SECONDS,budget.check_time()),retry_event=retry_event)
            except PreparationError as exc:
                decisions={"ranges_limit":"request_budget","transfer_bytes_limit":"transfer_budget","elapsed_limit":"time_budget"}
                if retry_event is not None and not retry_event.get("retry_started",False) and exc.reason in decisions:
                    retry_event["decision"]=decisions[exc.reason]
                    raise previous_failure from None
                if not eligible_open_timeout(exc):raise
                event={"metadata_audit_index":len(budget.metadata_response_audit)-1,
                    "attempt":attempt+1,"request_failure":safe_request_failure(exc.request_failure),
                    "decision":"waiting","retry_started":False}
                budget.retry_events.append(event)  # 전역 retry3회 + 마지막 실패1행 이하
                if attempt>=MAX_RETRIES_PER_PRE:event["decision"]="per_pre_limit"
                elif budget.retry_attempts>=MAX_RETRIES_TOTAL:event["decision"]="global_limit"
                else:
                    try:
                        budget.reserve_requests(3)
                        if media_bytes+2*self.METADATA_RESPONSE_BYTES>budget.limits.max_transfer_bytes-budget.transfer_bytes:
                            raise PreparationError("transfer_bytes_limit")
                        if budget.check_time()<=RETRY_DELAY_SECONDS:raise PreparationError("elapsed_limit")
                    except PreparationError as limit:
                        event["decision"]=decisions[limit.reason]
                    else:
                        before=budget.clock()
                        try:budget._sleeper(RETRY_DELAY_SECONDS)
                        finally:budget.retry_wait_seconds+=max(0.0,budget.clock()-before)
                        previous_failure=exc;retry_event=event;attempt+=1
                        continue
                raise  # 첫 timeout과 실패 audit를 상한 오류로 바꾸지 않는다.

    def _check_metadata_once(self, budget, operation, timeout=None, retry_event=None):
        remaining=budget.request("metadata")
        # 정책 경로는 마지막 시간 검사와 요청 계수를 같은 commit에서 처리한다.
        # 이 뒤 get()까지 추가 check_time이 없어 미발송 retry/benign elapsed 경계가 없다.
        open_timeout=None if timeout is None else min(timeout,remaining)
        if retry_event is not None:
            budget.retry_attempts+=1
            retry_event.update(decision="retry_started",retry_started=True)
        if not hasattr(budget,"metadata_response_audit"):
            budget.metadata_response_audit=[]
        audit={"operation":operation,"response":None,"bytes_received":0,"status":"pending"}
        budget.metadata_response_audit.append(audit)
        response=None;stage="request_open";failed=True
        try:
            response=session_opener(self.authorized_session,operation)(self.metadata_url,{"Accept-Encoding":"identity"},
                budget.timeout() if open_timeout is None else open_timeout)
            stage="response_validation"
            audit["response"]=metadata_diagnostic(response)
            def received(chunk):
                audit["bytes_received"]+=len(chunk)
            stage="body_read"
            raw = read_metadata_body(response,budget,self.METADATA_RESPONSE_BYTES,received,protocol_reason="drive_metadata_response",operation=operation)
            stage="response_validation"
            try:
                doc = json.loads(raw)
            except (ValueError,UnicodeError):
                raise PreparationError("drive_metadata_response") from None
            if not isinstance(doc,dict):
                raise PreparationError("drive_metadata_response")
            if doc.get("size") != str(self.part["size_bytes"]) or doc.get("version") != self.part["identity"]["version"] or doc.get("modifiedTime") != self.part["identity"]["modified_time"]:
                raise PreparationError("source_changed")
            audit["status"]="complete"
            failed=False
        except PreparationError as exc:
            attach_failure(exc,operation,stage)
            audit.update(status="failed",reason=exc.reason,request_failure=safe_request_failure(exc.request_failure))
            raise
        except Exception as exc:
            reason="output_or_read_error" if isinstance(exc,OSError) else "workflow_execution_failed"
            failure=request_failure(exc,operation,stage)
            audit.update(status="failed",reason=reason,request_failure=failure)
            raise PreparationError(reason,request_failure=failure) from None
        finally:
            if response is not None:
                try:close_response(response,operation,"output_or_read_error",preserve_active=failed)
                except PreparationError as exc:
                    audit.update(status="failed",reason=exc.reason,request_failure=safe_request_failure(exc.request_failure))
                    raise

    def read_range(self, start, end, budget, on_chunk):
        self.last_diagnostic = None
        # 전후 metadata 둘 다 끝낼 여유를 media를 열기 전에 예약한다.
        budget.reserve_requests(3)
        if end-start+1+2*self.METADATA_RESPONSE_BYTES > budget.limits.max_transfer_bytes-budget.transfer_bytes:
            raise PreparationError("transfer_bytes_limit")
        self.check_metadata(budget,"metadata_pre",end-start+1)
        self.pending_validation = True
        chunk = super().read_range(start,end,budget,on_chunk)
        self.check_metadata(budget,"metadata_post")
        self.pending_validation = False
        return chunk


class CachedDriveSource(DriveSource):
    def __init__(self,part,session,cache):
        super().__init__(part,session)
        self.cache=cache;self.cache_checked=False;self.cache_window_pending=False

    def finish_cached_validation(self,budget):
        if not self.cache_window_pending:return
        budget.reserve_requests(1)
        if self.METADATA_RESPONSE_BYTES>budget.limits.max_transfer_bytes-budget.transfer_bytes:
            raise PreparationError("transfer_bytes_limit")
        self._check_metadata_once(budget,"metadata_post")
        self.pending_validation=False;self.cache_window_pending=False

    def read_range(self,start,end,budget,on_chunk):
        if start>=self.cache.input_bytes:
            chunk=super().read_range(start,end,budget,on_chunk)
            self.cache.append(start,chunk)  # 원본 post 검사가 끝난 범위만 저장한다.
            return chunk
        if not self.cache_checked:
            budget.reserve_requests(2)
            if 2*self.METADATA_RESPONSE_BYTES>budget.limits.max_transfer_bytes-budget.transfer_bytes:
                raise PreparationError("transfer_bytes_limit")
            self._check_metadata_once(budget,"metadata_pre")
            self.cache_checked=True;self.pending_validation=True;self.cache_window_pending=True
        chunk=self.cache.read(start,end)
        if end+1==self.cache.input_bytes:self.finish_cached_validation(budget)
        on_chunk(chunk)
        return chunk


class SplitStream:
    def __init__(self, manifest, base, budget, authorized_session=None, cache=None):
        self.manifest, self.budget = manifest, budget
        self.cache=cache
        self.sources = []
        for part in manifest["parts"]:
            kind = part["source"]["type"]
            self.sources.append(LocalSource(part,base) if kind=="local" else HttpSource(part) if kind=="http" else
                CachedDriveSource(part,authorized_session,cache) if cache else DriveSource(part,authorized_session))
        self.part_index = self.offset = 0
        self.audit = []
        self.compressed_bytes_received = 0

    @property
    def at_end(self):
        return self.part_index == len(self.sources)

    def read(self):
        budget = self.budget
        budget.check_time()
        if self.at_end:
            return b""
        part = self.manifest["parts"][self.part_index]
        remaining = budget.limits.max_transfer_bytes-budget.transfer_bytes
        cached=bool(self.cache and self.offset<self.cache.input_bytes)
        if remaining <= 0 and not cached:
            raise PreparationError("transfer_bytes_limit")
        source = self.sources[self.part_index]
        if cached:
            remaining=self.cache.input_bytes-self.offset
        elif isinstance(source, DriveSource):
            budget.reserve_requests(3)
            remaining -= 2*source.METADATA_RESPONSE_BYTES
            if remaining <= 0:
                raise PreparationError("transfer_bytes_limit")
        else:
            budget.reserve_requests(1)
        size = min(budget.limits.io_chunk_bytes if isinstance(source,LocalSource) else budget.limits.media_range_bytes,
            part["size_bytes"]-self.offset, remaining)
        if self.cache:
            from .prefix_cache import LOGICAL_BYTES
            if budget.logical_compressed_bytes>=LOGICAL_BYTES:raise PreparationError("logical_input_limit")
            size=source.cache.cap(self.offset,min(size,LOGICAL_BYTES-budget.logical_compressed_bytes))
        record = {"part_index":self.part_index,"start":self.offset,"end":self.offset+size-1,"bytes_received":0,"status":"pending"}
        self.audit.append(record)
        if self.cache:record["input_source"]="cache" if cached else "drive"
        hash_ = hashlib.sha256()
        def received(chunk):
            record["bytes_received"] += len(chunk)
            self.compressed_bytes_received += len(chunk)
            hash_.update(chunk)
        try:
            chunk = source.read_range(self.offset,self.offset+size-1,budget,received)
            record["status"] = "complete"
        except PreparationError as exc:
            record["status"], record["reason"] = "failed", exc.reason
            if exc.diagnostic is not None:
                record["response"] = exc.diagnostic
            if exc.request_failure is not None:
                record["request_failure"]=safe_request_failure(exc.request_failure)
            raise
        finally:
            if getattr(source,"last_diagnostic",None) is not None and "response" not in record:
                record["response"] = source.last_diagnostic
            record["sha256_received_bytes"] = hash_.hexdigest()
        budget.peak_input_chunk_bytes = max(budget.peak_input_chunk_bytes,len(chunk))
        if self.cache:budget.logical_compressed_bytes+=len(chunk)
        self.offset += len(chunk)
        if self.offset == part["size_bytes"]:
            self.part_index += 1
            self.offset = 0
        return chunk
