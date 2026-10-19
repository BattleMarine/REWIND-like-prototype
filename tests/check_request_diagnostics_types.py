"""설치된 requests/urllib3/google-auth 실제 예외 타입만 사용. 네트워크/skip 없음."""

import argparse
import json
import sys
from pathlib import Path
from unittest.mock import patch


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--dependency-dir",type=Path)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
    if args.dependency_dir:sys.path.insert(0,str(args.dependency_dir.resolve()))
    import requests, urllib3, google.auth
    from requests import exceptions as re
    from urllib3 import exceptions as ue
    from google.auth import exceptions as ge
    from rewind_archive.contracts import PreparationError, canonical
    from rewind_archive.diagnostics import request_failure
    from rewind_archive.sources import session_opener

    secret="SECRET_raw_url_token_headers_body_args_classname"
    prepared=requests.Request("POST","https://private.invalid/?token="+secret,headers={"Authorization":secret},data=secret).prepare()
    response=requests.Response();response.status_code=403;response.headers["Set-Cookie"]=secret;response._content=secret.encode()
    def req(cls):return cls(secret,request=prepared,response=response)
    cases=[
        ("requests_connect_timeout",req(re.ConnectTimeout),"connect_timeout","requests"),
        ("requests_read_timeout",req(re.ReadTimeout),"read_timeout","requests"),
        ("requests_timeout",req(re.Timeout),"timeout","requests"),
        ("requests_tls",req(re.SSLError),"tls","requests"),
        ("requests_connection",req(re.ConnectionError),"connection","requests"),
        ("requests_proxy",req(re.ProxyError),"connection","requests"),
        ("requests_http_error",req(re.HTTPError),"http","requests"),
        ("requests_generic",req(re.RequestException),"request_other","requests"),
        ("urllib3_connect_timeout",ue.ConnectTimeoutError(secret),"connect_timeout","urllib3"),
        ("urllib3_read_timeout",ue.ReadTimeoutError(None,secret,secret),"read_timeout","urllib3"),
        ("urllib3_new_connection",ue.NewConnectionError(None,secret),"connection","urllib3"),
        ("urllib3_dns",ue.NameResolutionError(secret,None,OSError(secret)),"connection","urllib3"),
        ("urllib3_tls",ue.SSLError(secret),"tls","urllib3"),
        ("urllib3_protocol",ue.ProtocolError(secret),"transport","urllib3"),
        ("urllib3_retry_wrapper",ue.MaxRetryError(None,secret,RuntimeError(secret)),"transport","urllib3"),
        ("google_refresh",ge.RefreshError(secret),"authentication","google_auth"),
        ("google_credentials",ge.DefaultCredentialsError(secret),"authentication","google_auth"),
        ("google_oauth",ge.OAuthError(secret),"authentication","google_auth"),
        ("google_transport",ge.TransportError(secret),"transport","google_auth"),
        ("google_mutual_tls",ge.MutualTLSChannelError(secret),"tls","google_auth"),
    ]
    def poison(self):raise AssertionError("formatting forbidden")
    Hostile=type(secret,(RuntimeError,),{"__str__":poison,"__repr__":poison,"__module__":"requests.exceptions"})
    spoof=type("ReadTimeout",(RuntimeError,),{"__module__":"requests.exceptions"})
    wrapped=RuntimeError(secret);wrapped.__cause__=req(re.ReadTimeout)
    cases += [("unknown_hostile",Hostile(),"unknown","unknown"),("unknown_spoof",spoof(secret),"unknown","unknown"),
        ("unknown_wrapper",wrapped,"unknown","unknown")]
    results=[]
    with patch("socket.create_connection",side_effect=AssertionError("no network")):
        for case,error,category,library in cases:
            failure=request_failure(error,"media","request_open")
            assert failure["category"]==category and failure["library"]==library,case
            assert len(canonical(failure))<=256 and secret not in json.dumps(failure)
            calls=[]
            class Session:
                def get(self,*args,**kwargs):calls.append(True);raise error
            try:session_opener(Session(),"metadata_initial")("not opened",{},1)
            except PreparationError as caught:
                assert caught.reason=="authorized_request_failed"
                assert caught.request_failure["category"]==category
                assert caught.request_failure["stage"]=="request_open"
                assert caught.request_failure["operation"]=="metadata_initial"
            else:raise AssertionError(case)
            assert calls==[True]
            results.append({"case":case,"request_failure":failure,"single_attempt":True})
    report={"actual_google_requests":0,"actual_authentication":False,"versions":{"requests":requests.__version__,
        "urllib3":urllib3.__version__,"google_auth":google.auth.__version__},"cases":results}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({"actual_type_cases_passed":len(results),"actual_google_requests":0,"output":str(args.output)}))


if __name__=="__main__":main()
