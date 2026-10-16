"""실제 requests/urllib3/AuthorizedSession을 사용한 loopback 요청 예산 회귀."""

import argparse
import gzip
import json
import random
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--dependency-dir",type=Path)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--mode",choices=("extract","scan"),default="extract")
    args=parser.parse_args()
    root=Path(__file__).resolve().parents[1]
    sys.path.insert(0,str(root))
    if args.dependency_dir:sys.path.insert(0,str(args.dependency_dir.resolve()))
    try:
        import requests, urllib3, google.auth
        from google.auth.credentials import AnonymousCredentials
        from google.auth.transport.requests import AuthorizedSession
        from requests.adapters import HTTPAdapter
    except ImportError:
        parser.error("기존 실제 전송 라이브러리 환경이 필요합니다. 자동 설치하지 않습니다.")
    from rewind_archive.contracts import Limits, canonical
    from rewind_archive.fixture import SEQUENCE, tar_bytes
    from rewind_archive.matching import verify_extraction
    from rewind_archive.pilot import run_pilot

    args.output.mkdir(parents=True,exist_ok=False)
    # Default8MiB보다 긴, 거의 압축되지 않는 synthetic archive. 원본 데이터 접근 없음.
    data=gzip.compress(tar_bytes([(f"{SEQUENCE}/json/frame_{i}.json",random.Random(i).randbytes(2*1024**2)) for i in range(5)]),mtime=0)
    class Handler(BaseHTTPRequestHandler):
        protocol_version="HTTP/1.1"
        def log_message(self,*args):pass
        def do_GET(self):
            query=parse_qs(urlsplit(self.path).query)
            media=query.get("alt")==["media"]
            mode=self.server.mode
            self.server.calls.append("media" if media else "metadata")
            if media:
                start,end=map(int,self.headers["Range"][6:].split("-"))
                self.server.ranges.append([start,end])
                status=int(mode[6:]) if mode.startswith("media_") else 206
                body=data[start:end+1] if status==206 else b"do not read error body or secrets"
                self.send_response(status)
                self.send_header("Content-Length",str(len(body)))
                self.send_header("Content-Range",f"bytes {start}-{end}/{len(data)}")
            else:
                post=self.server.calls[-2:]==["media","metadata"]
                version="changed" if mode=="post_change" and post else "1"
                body=canonical({"size":str(len(data)),"version":version,"modifiedTime":"fixed"})
                self.send_response(200)
                self.send_header("Content-Length",str(len(body)))
            self.end_headers()
            try:self.wfile.write(body);self.wfile.flush()
            except (OSError,ValueError):self.close_connection=True
    server=ThreadingHTTPServer(("127.0.0.1",0),Handler)
    server.daemon_threads=True;server.handle_error=lambda *args:None
    thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
    class Adapter(HTTPAdapter):
        def send(self,request,**kwargs):
            source=urlsplit(request.url)
            assert source.hostname=="www.googleapis.com"
            request.url=f"http://127.0.0.1:{server.server_port}{source.path}?{source.query}"
            return super().send(request,**kwargs)
    manifest={"schema_version":"archive_manifest_v1","archive_id":"loopback_request_budget",
        "dataset_version":"synthetic_v1","kind":"pose","source_scope":"synthetic_fixture","total_parts":1,
        "parts":[{"index":0,"name":"fixture.tar.gz000","size_bytes":len(data),"version":"1",
            "source":{"type":"drive","file_id":"loopback_only"},"identity":{"version":"1","modified_time":"fixed"}}]}
    session=AuthorizedSession(AnonymousCredentials(),max_refresh_attempts=0)
    session.trust_env=False;session.mount("https://",Adapter(max_retries=0))
    cases=[]
    try:
        for mode in ("normal","post_change","media_200","media_401","media_403","media_429"):
            server.mode=mode;server.calls=[];server.ranges=[]
            pilot=run_pilot(manifest,args.output/mode,[] if args.mode=="scan" else ["*/json/*.json"],Limits(),session,mode=args.mode)
            stats=pilot["stats"]
            core=json.loads((args.output/mode/"extraction_manifest.json").read_text())
            assert stats["http_requests_total"]==len(server.calls)==stats["ranges"]
            assert stats["media_requests"]==server.calls.count("media")
            assert stats["metadata_requests"]==server.calls.count("metadata")
            assert stats["transfer_bytes"]<=8*1024**2 and stats["ranges"]<=128
            assert stats["local_reads"]==0
            assert not any((args.output/mode).rglob("*.partial"))
            if mode=="normal":
                assert pilot["stop_reason"]=="transfer_bytes_limit",pilot
                assert stats["ranges"]<128 and stats["media_requests"]<=33
                assert stats["metadata_requests"]==2*stats["media_requests"]
                assert stats["peak_input_chunk_bytes"]==256*1024
                assert stats["peak_output_chunk_bytes"]<=64*1024
                assert core["compressed_bytes_received"]>7*1024**2
                assert server.ranges[0][0]==0
                assert all(a[1]+1==b[0] for a,b in zip(server.ranges,server.ranges[1:]))
                assert server.ranges[-1][1]-server.ranges[-1][0]+1<256*1024
                if args.mode=="extract":
                    assert len(verify_extraction(args.output/mode)["files"])>=3
                    assert pilot["sample_acquired"]
                else:
                    assert pilot["complete_files"]==0 and not pilot["sample_acquired"]
                    assert not (args.output/mode/"files").exists()
                    assert sum(r["payload_complete"] for r in core["member_observations"])>=3
                    assert core["member_observations_usable"]
                assert not pilot["source_validation_pending"]
            elif mode=="post_change":
                assert pilot["stop_reason"]=="source_changed"
                assert len(server.calls)==3 and pilot["source_validation_pending"]
                assert not pilot["inspection_candidates_usable"] and not pilot["sample_acquired"]
                assert not core["member_observations_usable"]
            else:
                assert pilot["stop_reason"]=="range_status_not_206"
                assert len(server.calls)==2 and core["compressed_bytes_received"]==0
                assert core["range_audit"][0]["response"]["status"]==int(mode[6:])
                assert not pilot["inspection_candidates_usable"]
            report_bytes=sum((args.output/mode/n).stat().st_size for n in ("extraction_manifest.json","pilot_report.json"))
            assert report_bytes<=core["report_disk_reserve_bytes"]
            cases.append({"mode":mode,"stop_reason":pilot["stop_reason"],"complete_files":pilot["complete_files"],
                "stats":stats,"last_range":server.ranges[-1],"source_validation_pending":pilot["source_validation_pending"]})
    finally:
        session.close();server.shutdown();server.server_close();thread.join()
    report={"scope":"AnonymousCredentials loopback only","operation":args.mode,"actual_google_requests":0,"actual_authentication":False,
        "versions":{"requests":requests.__version__,"urllib3":urllib3.__version__,"google_auth":google.auth.__version__},"cases":cases}
    (args.output/"request_transport_report.json").write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({"cases_passed":len(cases),"actual_google_requests":0,"normal":{k:v for k,v in cases[0].items() if k!="stats"},
        "stats":{k:v for k,v in cases[0]["stats"].items() if k!="metadata_response_audit"}},indent=2))


if __name__=="__main__":main()
