"""예외 문자열/속성/chain 없이 실제 타입을 분류하는 작은 allowlist 진단."""

import http.client
import sys
import socket
import ssl
import urllib.error

from .contracts import PreparationError


CATEGORIES=frozenset({"authentication","connect_timeout","read_timeout","timeout","tls",
    "connection","transport","http","incomplete_body","request_other","os_error","unknown"})
LIBRARIES=frozenset({"requests","urllib3","google_auth","stdlib","unknown"})
OPERATIONS=frozenset({"metadata_initial","metadata_pre","metadata_post","media","authentication","workflow","unknown"})
STAGES=frozenset({"request_open","response_wrap","body_read","response_validation","response_close","authentication_setup","unknown"})

# 순서 중요: 구체적인 timeout/TLS와 TransportError를 일반 부모보다 먼저 판별한다.
OPTIONAL_TYPES=(
    ("requests.exceptions","requests",(("ConnectTimeout","connect_timeout"),("ReadTimeout","read_timeout"),
        ("SSLError","tls"),("Timeout","timeout"),("ConnectionError","connection"),
        ("HTTPError","http"),("RequestException","request_other"))),
    ("urllib3.exceptions","urllib3",(("SSLError","tls"),("NewConnectionError","connection"),
        ("ReadTimeoutError","read_timeout"),("ConnectTimeoutError","connect_timeout"),
        ("TimeoutError","timeout"),("IncompleteRead","incomplete_body"),("HTTPError","transport"))),
    ("google.auth.exceptions","google_auth",(("MutualTLSChannelError","tls"),
        ("TransportError","transport"),("GoogleAuthError","authentication"))),
)


def classify_exception(exc):
    for module_name,library,entries in OPTIONAL_TYPES:
        # 실제 exception을 만든 라이브러리는 이미 로드되어 있다. 진단 때문에 새 import를 시작하지 않는다.
        module=sys.modules.get(module_name)
        if module is None:continue
        for name,category in entries:
            cls=getattr(module,name,None)  # 고정 allowlist 이름만 조회한다.
            if isinstance(cls,type) and isinstance(exc,cls):return category,library
    for cls,category in ((ssl.SSLError,"tls"),(TimeoutError,"timeout"),
        (http.client.IncompleteRead,"incomplete_body"),(urllib.error.HTTPError,"http"),
        (ConnectionError,"connection"),(socket.gaierror,"connection"),
        (urllib.error.URLError,"transport"),(OSError,"os_error")):
        if isinstance(exc,cls):return category,"stdlib"
    return "unknown","unknown"


def allowed(value,choices):
    return value if isinstance(value,str) and value in choices else "unknown"


def safe_request_failure(value):
    if value is None:return None
    doc=value if isinstance(value,dict) else {}
    return {"schema_version":"request_failure_v1","category":allowed(doc.get("category"),CATEGORIES),
        "library":allowed(doc.get("library"),LIBRARIES),"operation":allowed(doc.get("operation"),OPERATIONS),
        "stage":allowed(doc.get("stage"),STAGES)}


def request_failure(exc,operation,stage):
    if isinstance(exc,PreparationError) and exc.request_failure is not None:
        return safe_request_failure(exc.request_failure)
    category,library=classify_exception(exc)
    return safe_request_failure({"category":category,"library":library,"operation":operation,"stage":stage})


def attach_failure(exc,operation,stage):
    exc.request_failure=request_failure(exc,operation,stage)
    return exc
