"""실제 전송 라이브러리+AnonymousCredentials+loopback 전용 별도 검사. Skip하지 않는다."""

import argparse
import gzip
import json
import sys
import threading
from dataclasses import replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--dependency-dir",type=Path,help="기존 격리 설치 경로; 자동 설치하지 않음")
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    root=Path(__file__).resolve().parents[1]
    sys.path.insert(0,str(root))
    if args.dependency_dir:
        sys.path.insert(0,str(args.dependency_dir.resolve()))
    try:
        import requests, urllib3, google.auth
        from google.auth.credentials import AnonymousCredentials
        from google.auth.transport.requests import AuthorizedSession
        from requests.adapters import HTTPAdapter
    except ImportError:
        parser.error("실제 requests/urllib3/google-auth 설치 환경 또는 --dependency-dir이 필요합니다.")
    from rewind_archive.contracts import Limits, canonical
    from rewind_archive.drive_metadata import MetadataLimits, build_drive_manifest
    from rewind_archive.fixture import SEQUENCE, tar_bytes
    from rewind_archive.pilot import run_metadata_pilot

    args.output.mkdir(parents=True,exist_ok=False)
    compressed=gzip.compress(tar_bytes([(f"{SEQUENCE}/json/frame_{i}.json",canonical({"frame":i})) for i in (2,0,1)]),mtime=0)
    cut=(len(compressed)+2)//3
    data={f"part_{i}":compressed[n:n+cut] for i,n in enumerate(range(0,len(compressed),cut))}
    selection={"schema_version":"drive_part_selection_v1","archive_id":"loopback_chunked","dataset_version":"synthetic_chunked_v1",
        "kind":"pose","total_parts":len(data),"parts":[{"index":i,"name":f"pose.tar.gz{i:03d}","file_id":fid} for i,fid in enumerate(data)]}

    class Handler(BaseHTTPRequestHandler):
        protocol_version="HTTP/1.1"
        def log_message(self,*args):pass
        def do_GET(self):
            query=parse_qs(urlsplit(self.path).query)
            fid=urlsplit(self.path).path.split("/")[-1]
            assert fid in data
            media=query.get("alt")==["media"]
            self.server.calls.append({"media":media,"mode":self.server.mode})
            if media:
                start,end=map(int,self.headers["Range"][6:].split("-"))
                body=data[fid][start:end+1]
                self.send_response(206)
                if self.server.mode=="media_chunked":
                    self.send_header("Transfer-Encoding","chunked")
                else:self.send_header("Content-Length",str(len(body)))
                self.send_header("Content-Range",f"bytes {start}-{end}/{len(data[fid])}")
                self.end_headers()
                try:
                    if self.server.mode=="media_chunked":self.wfile.write(f"{len(body):X}\r\n".encode()+body+b"\r\n0\r\n\r\n")
                    else:self.wfile.write(body)
                    self.wfile.flush()
                except (OSError,ValueError):pass
                return
            index=int(fid.split("_")[-1])
            body=canonical({"id":fid,"name":f"pose.tar.gz{index:03d}","size":str(len(data[fid])),"version":"11",
                "modifiedTime":"2026-10-15T00:00:00.000Z","mimeType":"application/octet-stream","trashed":False,"capabilities":{"canDownload":True}})
            mode=self.server.mode
            if any(c["media"] for c in self.server.calls):
                mode={"post_no_zero":"no_zero","post_cap":"oversized","post_slow":"slow"}.get(mode,mode)
            if mode=="oversized":body=body[:-1]+b',"padding":"'+b'x'*18000+b'"}'
            if mode=="invalid_json":body=b'{"id":'
            status=401 if mode=="auth" else 200
            self.send_response(status)
            self.send_header("Transfer-Encoding","chunked")
            self.send_header("Content-Type","application/json; charset=UTF-8")
            if mode=="encoding":self.send_header("Content-Encoding","gzip")
            if mode=="cl_te":self.send_header("Content-Length",str(len(body)))
            self.end_headers()
            try:
                if mode=="malformed":self.wfile.write(b"NOTHEX\r\n");self.wfile.flush();self.close_connection=True;return
                if mode=="truncated":
                    self.wfile.write(f"{len(body):X}\r\n".encode()+body[:10]);self.wfile.flush();self.close_connection=True;return
                width=1 if mode=="slow" else 17
                for n in range(0,len(body),width):
                    if self.server.stop.is_set():self.close_connection=True;return
                    chunk=body[n:n+width]
                    self.wfile.write(f"{len(chunk):X}\r\n".encode()+chunk+b"\r\n");self.wfile.flush()
                    if mode=="slow" and self.server.stop.wait(.02):self.close_connection=True;return
                if mode=="no_zero":self.close_connection=True;return
                self.wfile.write(b"0\r\n\r\n");self.wfile.flush()
            except (OSError,ValueError):self.close_connection=True

    server=ThreadingHTTPServer(("127.0.0.1",0),Handler)
    server.daemon_threads=True
    server.handle_error=lambda *args:None
    server.calls=[];server.mode="normal";server.stop=threading.Event()
    thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
    class Adapter(HTTPAdapter):
        def send(self,request,**kwargs):
            original=urlsplit(request.url)
            assert original.hostname=="www.googleapis.com"
            request.url=f"http://127.0.0.1:{server.server_port}{original.path}?{original.query}"
            return super().send(request,**kwargs)
    cases=[]
    session=AuthorizedSession(AnonymousCredentials(),max_refresh_attempts=0)
    session.trust_env=False
    session.mount("https://",Adapter(max_retries=0))
    try:
        for mode in ("normal","response_cap","global_cap","malformed","truncated","no_zero","invalid_json","auth","encoding","cl_te","slow","media_chunked","post_no_zero","post_cap","post_slow"):
            server.mode=mode;server.calls=[]
            limits=MetadataLimits()
            if mode=="response_cap":limits=replace(limits,max_response_bytes=40)
            if mode=="global_cap":limits=replace(limits,max_transfer_bytes=40)
            if mode=="slow":limits=replace(limits,max_elapsed_seconds=.2,request_timeout_seconds=.1)
            result=build_drive_manifest(selection,session,limits)
            record={"mode":mode,"reason":result["report"]["stop_reason"],"metadata_stats":result["report"]["stats"]}
            assert not any(c["media"] for c in server.calls)
            if mode in ("normal","media_chunked","post_no_zero","post_cap","post_slow"):
                assert result["report"]["manifest_ready"]
                pilot_limits=Limits() if mode!="post_slow" else Limits(max_elapsed_seconds=.2,request_timeout_seconds=.1)
                pilot=run_metadata_pilot(result,args.output/mode,["**/json/*.json"],pilot_limits,session)
                record.update(pilot_reason=pilot["stop_reason"],pilot_stats=pilot["stats"],files=pilot["complete_files"],train_ready=pilot["train_ready"])
                if mode=="normal":
                    assert pilot["verification"]["status"]=="passed" and pilot["complete_files"]==3
                    assert json.loads((args.output/mode/"pilot_report.json").read_text())==pilot
                    assert len(pilot["stats"]["metadata_response_audit"])==2*len(data)
                elif mode=="media_chunked":assert pilot["stop_reason"]=="content_length_mismatch" and pilot["complete_files"]==0
                elif mode=="post_no_zero":assert pilot["stop_reason"]=="http_body_failed" and pilot["complete_files"]==0
                elif mode=="post_cap":assert pilot["stop_reason"]=="metadata_response_bytes_limit" and pilot["complete_files"]==0
                else:assert pilot["stop_reason"] in ("elapsed_limit","http_body_failed") and pilot["complete_files"]==0
                core=json.loads((args.output/mode/"extraction_manifest.json").read_text())
                report_bytes=sum((args.output/mode/n).stat().st_size for n in ("extraction_manifest.json","pilot_report.json"))
                assert report_bytes<=core["report_disk_reserve_bytes"]
            else:
                assert result["manifest"] is None
                if mode in ("malformed","truncated","no_zero"):assert record["reason"]=="http_body_failed"
                if mode=="invalid_json":assert record["reason"]=="metadata_json"
                if mode in ("auth","cl_te"):assert record["reason"]=="metadata_protocol" and record["metadata_stats"]["transfer_bytes"]==0
                if mode=="encoding":assert record["reason"]=="metadata_content_encoding"
                if mode=="response_cap":assert record["reason"]=="metadata_response_bytes_limit" and record["metadata_stats"]["transfer_bytes"]==40
                if mode=="global_cap":assert record["reason"]=="transfer_bytes_limit" and record["metadata_stats"]["transfer_bytes"]==40
                if mode=="slow":
                    assert record["reason"] in ("elapsed_limit","http_body_failed")
                    assert 0<record["metadata_stats"]["transfer_bytes"]<100
                    assert record["metadata_stats"]["elapsed_seconds"]<1
            cases.append(record)
    finally:
        server.stop.set();session.close();server.shutdown();server.server_close();thread.join()
    report={"scope":"real transport libraries; AnonymousCredentials; loopback only","versions":{"requests":requests.__version__,"urllib3":urllib3.__version__,"google_auth":google.auth.__version__},
        "actual_google_requests":0,"actual_authentication":False,"cases":cases}
    (args.output/"transport_report.json").write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({"cases_passed":len(cases),"actual_google_requests":0,"output":str(args.output)},indent=2))


if __name__=="__main__":main()
