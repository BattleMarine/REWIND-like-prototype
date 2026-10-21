"""사전 metadata 개시 timeout만 다루는 고정 opt-in 정책. 숨은 retry는 거부한다."""

from .contracts import PreparationError
from .diagnostics import classify_exception


METADATA_PRE_POLICY = "metadataPreTimeout10sRetry1Global3_v1"
PRE_OPEN_TIMEOUT_SECONDS = 10.0
MAX_RETRIES_PER_PRE = 1
MAX_RETRIES_TOTAL = 3
RETRY_DELAY_SECONDS = 0.5
_OPEN_TIMEOUT = object()


def validate_request_policy(name):
    if name is not None and (not isinstance(name,str) or name != METADATA_PRE_POLICY):
        raise PreparationError("unknown_request_policy")
    return name


def policy_description(name):
    validate_request_policy(name)
    return {"name":name,"scope":"metadata_pre_request_open" if name else "disabled",
        "pre_open_timeout_seconds":PRE_OPEN_TIMEOUT_SECONDS if name else None,
        "max_retries_per_pre":MAX_RETRIES_PER_PRE if name else 0,
        "max_retries_total":MAX_RETRIES_TOTAL if name else 0,
        "retry_delay_seconds":RETRY_DELAY_SECONDS if name else 0.0}


def retry_summary(budget):
    return {**policy_description(budget._request_policy),"retries_started":budget.retry_attempts,
        "wait_seconds":budget.retry_wait_seconds,"events":[dict(e) for e in budget.retry_events],
        "exhausted":any(e["decision"] in {"per_pre_limit","global_limit"} for e in budget.retry_events)}


def validate_no_hidden_retries(session):
    # opt-in 경로에서만 실제 선택된 HTTPS adapter와 내부 인증 session까지 확인한다.
    # requests Retry(total=0)는 connect/read/status 등의 개별 값보다 우선한다.
    seen=set()
    def check(current, depth):
        if depth>2 or id(current) in seen:
            raise PreparationError("session_retry_policy")
        seen.add(id(current))
        if getattr(current,"_max_refresh_attempts",0) != 0:
            raise PreparationError("session_retry_policy")
        adapter_getter=getattr(current,"get_adapter",None)
        if not callable(adapter_getter):raise PreparationError("session_retry_policy")
        adapters=[adapter_getter("https://www.googleapis.com/drive/v3/files/")]
        mounted=getattr(current,"adapters",None)
        if mounted is not None:adapters.extend(mounted.values())
        # file ID/query·OAuth endpoint에 더 구체적으로 mount한 adapter도 우회하지 못한다.
        for adapter in adapters:
            total=getattr(getattr(adapter,"max_retries",None),"total",None)
            if total is not False and not (type(total) is int and total==0):
                raise PreparationError("session_retry_policy")
        auth_session=getattr(current,"_auth_request_session",None)
        if auth_session is not None:check(auth_session,depth+1)
        auth_request=getattr(current,"_auth_request",None)
        if auth_request is not None:
            request_session=getattr(auth_request,"session",None)
            if request_session is None:raise PreparationError("session_retry_policy")
            if id(request_session) not in seen:check(request_session,depth+1)
    try:check(session,0)
    except PreparationError:raise
    except Exception:raise PreparationError("session_retry_policy") from None


def mark_open_timeout(error, original, operation):
    # 보고용 dict는 retry 권한이 아니다. 실제 get() catch 경계에서 직접 타입을 확인한다.
    category,library=classify_exception(original)
    if operation!="metadata_pre" or category not in {"connect_timeout","read_timeout"} or library not in {"requests","urllib3"}:
        return error
    try:
        if getattr(original,"response",None) is not None:return error
        partial=getattr(original,"partial",None)
        if partial is not None and (type(partial) is not bytes or partial!=b""):return error
    except Exception:return error
    error._open_timeout_token=_OPEN_TIMEOUT
    return error


def eligible_open_timeout(error):
    return isinstance(error,PreparationError) and getattr(error,"_open_timeout_token",None) is _OPEN_TIMEOUT
