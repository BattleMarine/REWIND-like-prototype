"""안전한 예외 분류/단계·비밀 비노출·보고서/ZIP 전달. 실제 인증/요청 없음."""

import gzip
import http.client
import io
import json
import ssl
import tempfile
import unittest
import urllib.error
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from zipfile import ZipFile

from rewind_archive.contracts import PreparationError, canonical, digest
from rewind_archive.diagnostics import request_failure, safe_request_failure
from rewind_archive.drive_metadata import MetadataLimits
from rewind_archive.fixture import SEQUENCE, tar_bytes
from rewind_archive.profiles import POSE_COUNTERPART_PROFILE, profile_limits
from rewind_archive.sources import session_opener, http_open
from rewind_archive.workflow import make_preflight, run_workflow_job, review_run, save_result_zip
from test_drive_metadata import FakeDrive, Raw

SECRET="SECRET_url_token_header_body_classname"
A=f"{SEQUENCE}/json/frame_1.json";B=f"{SEQUENCE}/json/frame_10.json"


class DiagnosticTests(unittest.TestCase):
    def setUp(self):
        parent=Path(__file__).resolve().parents[1]/"outputs/request_diagnostic_test_tmp"
        parent.mkdir(parents=True,exist_ok=True)
        self.tmp=tempfile.TemporaryDirectory(dir=parent);self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name)
        self.config={"job_id":"diagnostic_test","reason":"bounded mock diagnostic","mode":"extract","purpose":"targeted",
            "profile":POSE_COUNTERPART_PROFILE,"selection":{"schema_version":"drive_part_selection_v1","archive_id":"fake_pose",
            "dataset_version":"mock_v1","kind":"pose","total_parts":1,"parts":[{"index":0,"name":"pose.tar.gz000","file_id":"fake"}]},
            "includes":[A,B],"exact_targets":[A,B],"limits":asdict(profile_limits(POSE_COUNTERPART_PROFILE)),
            "metadata_limits":asdict(MetadataLimits()),"previous_result":""}
        self.data=gzip.compress(tar_bytes([(A,b'{"mock":1}'),(B,b'{"mock":10}')]),mtime=0)

    def session(self):
        session=FakeDrive(self.config["selection"],{"fake":self.data});session.close=lambda:None
        return session

    def run_job(self, session):
        return run_workflow_job(self.config,make_preflight(self.config,{}),{},self.root,lambda:session)

    def reports(self,root):
        result={}
        for p in root.rglob("*.json"):
            if "/files/" not in p.as_posix():result[p.name]=json.loads(p.read_text())
        self.assertNotIn(SECRET,json.dumps(result))
        return result

    def assert_zip_and_review(self,root,failure):
        review=review_run(root)
        self.assertEqual(review["request_failure"],failure)
        self.assertTrue(review["request_failure_recorded"])
        target=save_result_zip(root)
        with ZipFile(target) as z:
            self.assertIsNone(z.testzip())
            for name in z.namelist():
                if name.endswith(".json"):self.assertNotIn(SECRET,z.read(name).decode())
        self.assertLessEqual(len(canonical(failure)),256)

    def fail_call(self,session,number,error):
        original=session.get;calls=[0]
        def get(url,**kwargs):
            calls[0]+=1
            if calls[0]==number:raise error
            return original(url,**kwargs)
        session.get=get
        return calls

    def test_stdlib_type_allowlist_and_unknown_never_formats_exception(self):
        for error,category in ((TimeoutError(SECRET),"timeout"),(ConnectionResetError(SECRET),"connection"),
            (ssl.SSLError(SECRET),"tls"),(urllib.error.URLError(SECRET),"transport"),
            (http.client.IncompleteRead(b"x"),"incomplete_body")):
            self.assertEqual(request_failure(error,"media","body_read")["category"],category)
        def poison(self):raise AssertionError("exception must not be formatted")
        Unknown=type(SECRET,(RuntimeError,),{"__str__":poison,"__repr__":poison})
        failure=request_failure(Unknown(),"media","request_open")
        self.assertEqual(failure["category"],"unknown");self.assertEqual(failure["library"],"unknown")
        self.assertNotIn(SECRET,json.dumps(failure))

    def test_diagnostic_values_and_keys_are_sanitized(self):
        value={"category":[SECRET],"library":SECRET,"operation":SECRET,"stage":SECRET,"raw":SECRET}
        safe=safe_request_failure(value)
        self.assertEqual(set(safe),{"schema_version","category","library","operation","stage"})
        self.assertNotIn(SECRET,json.dumps(safe))
        self.assertTrue(all(v=="unknown" for k,v in safe.items() if k!="schema_version"))
        # 최대 길이 allowlist 값과 실패 record를 모든 요청에 가정해도 기존 예약 안이다.
        worst={"schema_version":"request_failure_v1","category":"connect_timeout","library":"google_auth",
            "operation":"metadata_initial","stage":"response_validation"}
        self.assertLessEqual(len(canonical(worst)),256)
        response={"status":599,"content_length":999999,"content_encoding":"unsupported",
            "transfer_encoding":"unsupported","content_type":"absent"}
        metadata={"operation":"metadata_initial","response":response,"bytes_received":16384,
            "status":"failed","reason":"metadata_response_bytes_limit","request_failure":worst}
        media={"part_index":999,"start":67108864,"end":67108864,"bytes_received":262144,"status":"failed",
            "reason":"authorized_request_failed","response":response,"request_failure":worst,"sha256_received_bytes":"f"*64}
        self.assertLess(4*len(canonical(metadata))+len(canonical(media)),3*1024)

    def test_initial_metadata_open_failure_reaches_job_review_zip(self):
        session=self.session();calls=self.fail_call(session,1,ConnectionResetError(SECRET))
        root,record=self.run_job(session);reports=self.reports(root)
        failure=record["request_failure"]
        self.assertEqual(record["stop_reason"],"authorized_request_failed")
        self.assertEqual((failure["operation"],failure["stage"]),("metadata_initial","request_open"))
        self.assertEqual(failure["category"],"connection")
        self.assertEqual(reports["metadata_report.json"]["request_failure"],failure)
        self.assertEqual(reports["metadata_report.json"]["request_audit"][0]["request_failure"],failure)
        self.assertEqual(calls[0],1);self.assertFalse((root/"pilot").exists())
        self.assert_zip_and_review(root,failure)

    def test_response_wrapping_has_own_stage_same_stop_reason(self):
        closed=[]
        session=SimpleNamespace(get=lambda *a,**k:SimpleNamespace(close=lambda:closed.append(True)))
        with self.assertRaises(PreparationError) as raised:session_opener(session,"media")("hidden",{},1)
        self.assertEqual(raised.exception.reason,"authorized_request_failed")
        self.assertEqual(raised.exception.request_failure["stage"],"response_wrap")
        self.assertEqual(closed,[True])
        self.assertNotIn(SECRET,json.dumps(raised.exception.request_failure))

    def test_pre_metadata_failure_media_zero_pending_false(self):
        session=self.session();calls=self.fail_call(session,2,TimeoutError(SECRET))
        root,record=self.run_job(session);reports=self.reports(root)
        extraction=reports["extraction_manifest.json"]
        self.assertEqual(record["request_failure"]["operation"],"metadata_pre")
        self.assertEqual(extraction["stats"]["media_requests"],0)
        self.assertFalse(extraction["source_validation_pending"])
        self.assertFalse(extraction["inspection_candidates_usable"])
        self.assertIsNone(extraction["stats"]["metadata_response_audit"][-1]["response"])
        self.assertEqual(calls[0],2)
        self.assert_zip_and_review(root,record["request_failure"])

    def test_media_request_no_response_status_not_invented(self):
        session=self.session();calls=self.fail_call(session,3,ConnectionRefusedError(SECRET))
        root,record=self.run_job(session);reports=self.reports(root)
        failure=record["request_failure"];extraction=reports["extraction_manifest.json"]
        self.assertEqual((failure["category"],failure["operation"],failure["stage"]),("connection","media","request_open"))
        audit=extraction["range_audit"][-1]
        self.assertEqual(audit["bytes_received"],0);self.assertNotIn("response",audit)
        self.assertEqual(audit["request_failure"],failure)
        self.assertEqual(reports["pilot_report.json"]["request_failure"],failure)
        self.assertTrue(extraction["source_validation_pending"])
        self.assertFalse(extraction["inspection_candidates_usable"])
        self.assertEqual(calls[0],3)
        self.assert_zip_and_review(root,failure)

    def test_post_metadata_failure_keeps_received_media_unusable(self):
        session=self.session();calls=self.fail_call(session,4,TimeoutError(SECRET))
        root,record=self.run_job(session);reports=self.reports(root)
        extraction=reports["extraction_manifest.json"];failure=record["request_failure"]
        self.assertEqual((failure["operation"],failure["stage"]),("metadata_post","request_open"))
        self.assertGreater(extraction["range_audit"][-1]["bytes_received"],0)
        self.assertEqual(extraction["range_audit"][-1]["response"]["status"],206)
        self.assertIsNone(extraction["stats"]["metadata_response_audit"][-1]["response"])
        self.assertTrue(extraction["source_validation_pending"])
        self.assertFalse(record["targets_complete"])
        self.assertEqual(calls[0],4)
        self.assert_zip_and_review(root,failure)

    def test_partial_body_bytes_counted_and_secret_body_not_logged(self):
        session=self.session();original=session.get
        class Partial(Raw):
            def read1(self,size,decode_content=False):
                if self.tell()==0:return super().read(8)
                raise http.client.IncompleteRead(SECRET.encode(),100)
        def get(url,**kwargs):
            response=original(url,**kwargs)
            if "alt=media" in url:
                response.raw=Partial(self.data)
                def close():raise ConnectionResetError(SECRET)
                response.close=close  # 본문 실패를 cleanup 오류가 덮어쓰면 안 된다.
            return response
        session.get=get
        root,record=self.run_job(session);reports=self.reports(root)
        failure=record["request_failure"];extraction=reports["extraction_manifest.json"]
        self.assertEqual(record["stop_reason"],"http_body_failed")
        self.assertEqual((failure["category"],failure["operation"],failure["stage"]),("incomplete_body","media","body_read"))
        self.assertEqual(extraction["compressed_bytes_received"],8+len(SECRET))
        self.assertEqual(extraction["range_audit"][-1]["bytes_received"],8+len(SECRET))
        self.assertFalse(extraction["inspection_candidates_usable"])
        self.assertTrue(extraction["source_validation_pending"])
        self.assertFalse(list(root.rglob("*.partial")))
        self.assert_zip_and_review(root,failure)

    def test_post_metadata_body_read_failure_phase_not_media(self):
        session=self.session();original=session.get;calls=[0]
        class Broken(Raw):
            def read1(self,*args,**kwargs):raise ConnectionResetError(SECRET)
        def get(url,**kwargs):
            calls[0]+=1;response=original(url,**kwargs)
            if calls[0]==4:
                response.raw=Broken(b"")
                def close():raise ConnectionResetError(SECRET)
                response.close=close
            return response
        session.get=get
        root,record=self.run_job(session);reports=self.reports(root)
        self.assertEqual(record["stop_reason"],"http_body_failed")
        self.assertEqual((record["request_failure"]["operation"],record["request_failure"]["stage"]),("metadata_post","body_read"))
        self.assertTrue(reports["extraction_manifest.json"]["source_validation_pending"])
        self.assert_zip_and_review(root,record["request_failure"])

    def test_keyboard_interrupt_is_not_overwritten_by_close_failure(self):
        session=self.session();original=session.get
        class Interrupted(Raw):
            def read1(self,*args,**kwargs):raise KeyboardInterrupt()
        def get(url,**kwargs):
            response=original(url,**kwargs)
            if "alt=media" in url:
                response.raw=Interrupted(b"")
                def close():raise ConnectionResetError(SECRET)
                response.close=close
            return response
        session.get=get
        root,record=self.run_job(session);reports=self.reports(root)
        self.assertEqual(record["stop_reason"],"interrupted")
        self.assertIsNone(record["request_failure"])
        self.assertTrue(reports["extraction_manifest.json"]["source_validation_pending"])
        self.assertFalse(reports["extraction_manifest.json"]["inspection_candidates_usable"])

    def test_authentication_setup_failure_is_typed_not_assumed_auth_cause(self):
        def fail():raise TimeoutError(SECRET)
        root,record=run_workflow_job(self.config,make_preflight(self.config,{}),{},self.root,fail)
        self.assertEqual(record["stop_reason"],"workflow_execution_failed")
        self.assertEqual(record["request_failure"]["category"],"timeout")
        self.assertEqual(record["request_failure"]["operation"],"authentication")
        self.assertEqual(record["request_failure"]["stage"],"authentication_setup")
        self.reports(root);self.assert_zip_and_review(root,record["request_failure"])

    def test_normal_and_legacy_missing_diagnostics_remain_readable(self):
        root,record=self.run_job(self.session());reports=self.reports(root)
        self.assertIsNone(record["request_failure"]);self.assertTrue(record["targets_complete"])
        self.assert_zip_and_review(root,None)
        for name in ("job.json","pilot/extraction_manifest.json","pilot/pilot_report.json"):
            path=root/name;doc=json.loads(path.read_text());doc.pop("request_failure")
            body={k:v for k,v in doc.items() if k!="content_sha256"}
            path.write_bytes(canonical({**body,"content_sha256":digest(body)}))
        review=review_run(root)
        self.assertIsNone(review["request_failure"]);self.assertFalse(review["request_failure_recorded"])

    def test_initial_response_validation_and_close_failures_are_sanitized(self):
        for phase in ("response_validation","response_close"):
            with self.subTest(phase=phase):
                self.config["job_id"]="job_"+phase
                session=self.session();original=session.get
                class Headers(dict):
                    def get(self,*args):raise RuntimeError(SECRET)
                def get(url,**kwargs):
                    response=original(url,**kwargs)
                    if phase=="response_validation":response.headers=Headers()
                    else:
                        def close():raise ConnectionResetError(SECRET)
                        response.close=close
                    return response
                session.get=get
                root,record=self.run_job(session);self.reports(root)
                self.assertEqual(record["stop_reason"],"metadata_request_or_input_failed")
                self.assertEqual(record["request_failure"]["stage"],phase)
                self.assert_zip_and_review(root,record["request_failure"])

    def test_stdlib_http_open_keeps_response_and_exception_separate(self):
        error=urllib.error.HTTPError("https://hidden/"+SECRET,403,SECRET,{"Content-Type":SECRET},None)
        opener=SimpleNamespace(open=lambda *a,**k:None)
        with patch("urllib.request.build_opener",return_value=opener),patch.object(opener,"open",side_effect=error):
            with self.assertRaises(PreparationError) as raised:http_open("https://hidden",{},1)
        exc=raised.exception
        self.assertEqual(exc.reason,"http_status_rejected")
        self.assertEqual(exc.diagnostic["status"],403)
        self.assertEqual(exc.request_failure["category"],"http")
        self.assertNotIn(SECRET,json.dumps([exc.diagnostic,exc.request_failure]))


if __name__=="__main__":unittest.main()
