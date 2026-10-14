"""순차 Range 읽기. 토큰/인증 session은 런타임에만 주입한다."""

import hashlib
import json
import os
import re
import urllib.error
import urllib.request
from pathlib import Path

from .contracts import PreparationError


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise PreparationError("redirect_rejected")


def http_open(url, headers, timeout):
    request = urllib.request.Request(url, headers=headers)
    # 환경 proxy·Cookie·redirect를 사용하지 않는다. TLS 인증서 검증은 기본값이다.
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect())
    try:
        return opener.open(request, timeout=timeout)
    except urllib.error.HTTPError as exc:
        exc.close()
        raise PreparationError("http_status_rejected") from None
    except (OSError, urllib.error.URLError):
        raise PreparationError("http_open_failed") from None


class SessionResponse:
    def __init__(self, response):
        self.response = response
        self.status, self.headers = response.status_code, response.headers

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


def session_opener(session):
    if getattr(session,"_max_refresh_attempts",0) != 0:
        raise PreparationError("session_retry_policy")
    def open_response(url, headers, timeout):
        try:
            return SessionResponse(session.get(url, headers=headers, stream=True, allow_redirects=False, timeout=timeout))
        except Exception:
            # Session 예외에 token/URL이 들어 있더라도 로그에 전달하지 않는다.
            raise PreparationError("authorized_request_failed") from None
    return open_response


def read_body(response, expected, budget, on_chunk):
    data = bytearray()
    while len(data) < expected:
        timeout = budget.timeout()
        # 연결 socket 접근이 가능하면 각 body read의 잔여 시간도 줄인다.
        if hasattr(response,"set_timeout"):
            response.set_timeout(timeout)
        sock = getattr(getattr(getattr(response, "fp", None), "raw", None), "_sock", None)
        if sock is not None:
            sock.settimeout(timeout)
        try:
            chunk = response.read1(min(expected-len(data), budget.limits.io_chunk_bytes))
        except PreparationError:
            raise
        except Exception as exc:
            partial = getattr(exc,"partial",b"")
            if isinstance(partial,bytes):
                budget.transfer_bytes += len(partial)
                on_chunk(partial)
            raise PreparationError("http_body_failed") from None
        if not chunk:
            raise PreparationError("short_http_body")
        # 반환된 바이트는 timeout 후에도 기록한다. 다음 read 전에 중단한다.
        budget.transfer_bytes += len(chunk)
        on_chunk(chunk)
        if budget.transfer_bytes > budget.limits.max_transfer_bytes:
            raise PreparationError("transfer_bytes_limit")
        budget.check_time()
        data.extend(chunk)
    return bytes(data)


class LocalSource:
    def __init__(self, part, base):
        self.part = part
        self.path = (Path(base)/part["source"]["path"]).resolve()

    def read_range(self, start, end, budget, on_chunk):
        budget.check_time()
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

    def read_range(self, start, end, budget, on_chunk):
        headers = {"Range":f"bytes={start}-{end}","Accept-Encoding":"identity"}
        identity = self.part["identity"]
        if identity.get("etag"):
            headers["If-Match"] = identity["etag"]
        if identity.get("last_modified"):
            headers["If-Unmodified-Since"] = identity["last_modified"]
        response = self.opener(self.url, headers, budget.timeout())
        try:
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
            return read_body(response, expected, budget, on_chunk)
        finally:
            response.close()


class DriveSource(HttpSource):
    """Colab AuthorizedSession 주입. 이 모듈은 인증 시작·token 저장을 하지 않는다."""
    def __init__(self, part, authorized_session):
        if authorized_session is None:
            raise PreparationError("drive_session_required")
        self.part, self.opener = part, session_opener(authorized_session)
        file_id = part["source"]["file_id"]
        self.url = f"https://www.googleapis.com/drive/v3/files/{file_id}?alt=media"
        self.metadata_url = f"https://www.googleapis.com/drive/v3/files/{file_id}?fields=size,version,modifiedTime"

    def check_metadata(self, budget):
        budget.add("ranges",1,budget.limits.max_ranges)
        response = self.opener(self.metadata_url, {"Accept-Encoding":"identity"}, budget.timeout())
        try:
            length = response.headers.get("Content-Length", "")
            if response.status != 200 or not re.fullmatch(r"[0-9]{1,5}", length) or not 1 <= int(length) <= 16384:
                raise PreparationError("drive_metadata_response")
            if int(length) > budget.limits.max_transfer_bytes-budget.transfer_bytes:
                raise PreparationError("transfer_bytes_limit")
            raw = read_body(response,int(length),budget,lambda chunk:None)
            try:
                doc = json.loads(raw)
            except (ValueError,UnicodeError):
                raise PreparationError("drive_metadata_response") from None
            if not isinstance(doc,dict):
                raise PreparationError("drive_metadata_response")
            if doc.get("size") != str(self.part["size_bytes"]) or doc.get("version") != self.part["identity"]["version"] or doc.get("modifiedTime") != self.part["identity"]["modified_time"]:
                raise PreparationError("source_changed")
        finally:
            response.close()

    def read_range(self, start, end, budget, on_chunk):
        self.check_metadata(budget)
        if end-start+1 > budget.limits.max_transfer_bytes-budget.transfer_bytes:
            raise PreparationError("transfer_bytes_limit")
        chunk = super().read_range(start,end,budget,on_chunk)
        self.check_metadata(budget)
        return chunk


class SplitStream:
    def __init__(self, manifest, base, budget, authorized_session=None):
        self.manifest, self.budget = manifest, budget
        self.sources = []
        for part in manifest["parts"]:
            kind = part["source"]["type"]
            self.sources.append(LocalSource(part,base) if kind=="local" else HttpSource(part) if kind=="http" else DriveSource(part,authorized_session))
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
        if remaining <= 0:
            raise PreparationError("transfer_bytes_limit")
        size = min(budget.limits.io_chunk_bytes, part["size_bytes"]-self.offset, remaining)
        budget.add("ranges",1,budget.limits.max_ranges)
        record = {"part_index":self.part_index,"start":self.offset,"end":self.offset+size-1,"bytes_received":0,"status":"pending"}
        self.audit.append(record)
        hash_ = hashlib.sha256()
        def received(chunk):
            record["bytes_received"] += len(chunk)
            self.compressed_bytes_received += len(chunk)
            hash_.update(chunk)
        try:
            chunk = self.sources[self.part_index].read_range(self.offset,self.offset+size-1,budget,received)
            record["status"] = "complete"
        except PreparationError as exc:
            record["status"], record["reason"] = "failed", exc.reason
            raise
        finally:
            record["sha256_received_bytes"] = hash_.hexdigest()
        budget.peak_input_chunk_bytes = max(budget.peak_input_chunk_bytes,len(chunk))
        self.offset += len(chunk)
        if self.offset == part["size_bytes"]:
            self.part_index += 1
            self.offset = 0
        return chunk
