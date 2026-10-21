"""실제 requests/urllib3 타입과 AuthorizedSession adapter 설정으로 retry 경계를 검사한다. 네트워크 없음."""

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--dependency-dir",type=Path)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
    if args.dependency_dir:sys.path.insert(0,str(args.dependency_dir.resolve()))
    import requests,urllib3,google.auth
    from requests import exceptions as re
    from urllib3 import exceptions as ue
    from google.auth.transport.requests import AuthorizedSession,Request
    from google.auth.credentials import AnonymousCredentials
    from rewind_archive.contracts import PreparationError
    from rewind_archive.request_policy import eligible_open_timeout,validate_no_hidden_retries
    from rewind_archive.sources import session_opener
    secret="SECRET_exception_url_token_body"
    response=requests.Response();response.status_code=403;response._content=secret.encode()
    partial=re.ReadTimeout(secret);partial.partial=b"x"
    forged=PreparationError("authorized_request_failed",request_failure={"category":"read_timeout","library":"requests",
        "operation":"metadata_pre","stage":"request_open"})
    spoof=type("ReadTimeout",(RuntimeError,),{"__module__":"requests.exceptions"})(secret)
    wrapped=RuntimeError(secret);wrapped.__cause__=re.ReadTimeout(secret)
    cases=[("requests_connect",re.ConnectTimeout(secret),True),
        ("requests_read",re.ReadTimeout(secret),True),
        ("urllib3_connect",ue.ConnectTimeoutError(secret),True),
        ("urllib3_read",ue.ReadTimeoutError(None,secret,secret),True),
        ("requests_generic_timeout",re.Timeout(secret),False),
        ("requests_response_attached",re.ReadTimeout(secret,response=response),False),
        ("requests_partial",partial,False),("stdlib_timeout",TimeoutError(secret),False),
        ("requests_tls",re.SSLError(secret),False),("requests_connection",re.ConnectionError(secret),False),
        ("urllib3_retry_wrapper",ue.MaxRetryError(None,secret,ue.ConnectTimeoutError(secret)),False),
        ("diagnostic_forgery",forged,False),("class_name_spoof",spoof,False),("cause_wrapper",wrapped,False)]
    results=[]
    with patch("socket.create_connection",side_effect=AssertionError("no network")):
        for name,error,expected in cases:
            def get(*a,**k):raise error
            with_error=None
            try:session_opener(SimpleNamespace(get=get),"metadata_pre")("hidden",{},1)
            except PreparationError as exc:with_error=exc
            assert with_error is not None and eligible_open_timeout(with_error)==expected,name
            assert secret not in json.dumps(with_error.request_failure)
            results.append({"case":name,"retry_eligible":expected,"request_failure":with_error.request_failure})
        for operation in ("metadata_initial","metadata_post","media"):
            try:session_opener(SimpleNamespace(get=lambda *a,**k:(_ for _ in ()).throw(re.ReadTimeout(secret))),operation)("hidden",{},1)
            except PreparationError as exc:assert not eligible_open_timeout(exc)
            results.append({"case":operation,"retry_eligible":False})
        # 실제 google-auth 기본 내부 auth session은 max_retries3이므로 명시적으로 고정해야 한다.
        unsafe=AuthorizedSession(AnonymousCredentials(),max_refresh_attempts=0)
        try:
            try:validate_no_hidden_retries(unsafe)
            except PreparationError as exc:assert exc.reason=="session_retry_policy"
            else:raise AssertionError("hidden auth retry accepted")
        finally:unsafe.close()
        auth_http=requests.Session();auth_http.trust_env=False
        auth_http.mount("https://",requests.adapters.HTTPAdapter(max_retries=0))
        safe=AuthorizedSession(AnonymousCredentials(),max_refresh_attempts=0,auth_request=Request(session=auth_http))
        safe.trust_env=False;safe.mount("https://",requests.adapters.HTTPAdapter(max_retries=0))
        try:
            validate_no_hidden_retries(safe)
            safe.mount("https://www.googleapis.com/drive/v3/files/a-specific-file",requests.adapters.HTTPAdapter(max_retries=1))
            try:validate_no_hidden_retries(safe)
            except PreparationError:pass
            else:raise AssertionError("more-specific retry adapter accepted")
        finally:safe.close();auth_http.close()
        results.extend([{"case":"default_auth_retry3_rejected"},{"case":"explicit_auth_and_media_retry0_accepted"},
            {"case":"more_specific_retry_adapter_rejected"}])
    report={"versions":{"requests":requests.__version__,"urllib3":urllib3.__version__,"google_auth":google.auth.__version__},
        "network_requests":0,"cases":results,"case_count":len(results)}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({"case_count":len(results),"network_requests":0}))


if __name__=="__main__":main()
