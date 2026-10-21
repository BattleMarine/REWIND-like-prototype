"""retry 상태기계/예산/보고서의 네트워크 없는 검증. 실제 타입 경계는 별도 checker로 검사한다."""

import copy
import gzip
import http.client
import json
import tempfile
import unittest
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from zipfile import ZipFile

from rewind_archive.contracts import Budget, Limits, PreparationError, canonical, digest
from rewind_archive.drive_metadata import MetadataLimits, build_drive_manifest
from rewind_archive.extraction import prepare
from rewind_archive.fixture import SEQUENCE, tar_bytes
from rewind_archive.profiles import POSE_COUNTERPART_PROFILE, profile_limits
from rewind_archive.request_policy import (METADATA_PRE_POLICY, retry_summary, validate_no_hidden_retries,
    eligible_open_timeout)
from rewind_archive.sources import DriveSource, session_opener
from rewind_archive.workflow import make_preflight, run_workflow_job, review_run, save_result_zip
from test_drive_metadata import FakeDrive, Raw

SECRET="SECRET_exception_url_token_body"
A=f"{SEQUENCE}/json/frame_1.json";B=f"{SEQUENCE}/json/frame_10.json"


class Clock:
    def __init__(self):self.now=0.;self.waits=[]
    def __call__(self):return self.now
    def sleep(self,seconds):self.waits.append(seconds);self.now+=seconds


class RetryTests(unittest.TestCase):
    def setUp(self):
        parent=Path(__file__).resolve().parents[1]/"outputs/retry_test_tmp";parent.mkdir(parents=True,exist_ok=True)
        tmp=tempfile.TemporaryDirectory(dir=parent);self.addCleanup(tmp.cleanup);self.root=Path(tmp.name)
        self.selection={"schema_version":"drive_part_selection_v1","archive_id":"retry_fixture","dataset_version":"mock_v1",
            "kind":"pose","total_parts":1,"parts":[{"index":0,"name":"pose.tar.gz000","file_id":"fake"}]}
        self.data=gzip.compress(tar_bytes([(A,b'{"mock":1}'),(B,b'{"mock":10}')]),mtime=0)
        self.clock=Clock()
        self.session=FakeDrive(self.selection,{"fake":self.data})
        self.session.close=lambda:None
        self.session.get_adapter=lambda url:SimpleNamespace(max_retries=SimpleNamespace(total=0))
        self.part={"source":{"file_id":"fake"},"size_bytes":len(self.data),
            "identity":{"version":"9","modified_time":"2026-10-14T00:00:00.000Z"}}
        self.config={"job_id":"retry_test","reason":"bounded metadata timeout fixture","mode":"extract","purpose":"targeted",
            "profile":POSE_COUNTERPART_PROFILE,"request_policy":METADATA_PRE_POLICY,"selection":self.selection,
            "includes":[A,B],"exact_targets":[A,B],"limits":asdict(profile_limits(POSE_COUNTERPART_PROFILE)),
            "metadata_limits":asdict(MetadataLimits()),"previous_result":""}
        # 상태기계 단위 테스트에서만 타입 판별 결과를 주입한다. 위조 dict 검사는 실제 판별로 수행한다.
        self.type_patch=patch("rewind_archive.request_policy.classify_exception",return_value=("read_timeout","requests"))

    def budget(self,**changes):
        if "max_elapsed_seconds" in changes:
            changes.setdefault("request_timeout_seconds",min(5,changes["max_elapsed_seconds"]))
        return Budget(replace(Limits(),**changes),clock=self.clock,sleeper=self.clock.sleep,request_policy=METADATA_PRE_POLICY)

    def fail_calls(self,numbers,error=None,elapsed=0):
        original=self.session.get;self.attempts=[]
        def get(url,**kwargs):
            self.attempts.append((url,kwargs));self.clock.now+=elapsed
            if len(self.attempts) in numbers:raise error or RuntimeError(SECRET)
            return original(url,**kwargs)
        self.session.get=get

    def read(self,budget):return DriveSource(self.part,self.session).read_range(0,len(self.data)-1,budget,lambda b:None)

    def test_transient_success_counts_every_attempt_and_wait(self):
        budget=self.budget();self.fail_calls({1})
        with self.type_patch:self.assertEqual(self.read(budget),self.data)
        self.assertEqual((budget.ranges,budget.metadata_requests,budget.media_requests,budget.retry_attempts),(4,3,1,1))
        self.assertEqual(self.clock.waits,[.5]);self.assertEqual(budget.retry_wait_seconds,.5)
        self.assertEqual([k[1]["timeout"] for k in self.attempts],[10.,10.,5.,5.])
        self.assertEqual([a["status"] for a in budget.metadata_response_audit],["failed","complete","complete"])
        expected=len(self.data)+2*len(canonical(self.session.docs["fake"]))
        self.assertEqual(budget.transfer_bytes,expected)
        self.assertEqual(budget.retry_events[0]["decision"],"retry_started")

    def test_per_pre_exhaustion_preserves_timeout_and_no_media(self):
        budget=self.budget();self.fail_calls({1,2})
        source=DriveSource(self.part,self.session)
        with self.type_patch,self.assertRaises(PreparationError) as err:source.read_range(0,len(self.data)-1,budget,lambda b:None)
        self.assertEqual(err.exception.reason,"authorized_request_failed")
        self.assertEqual((budget.ranges,budget.media_requests,budget.transfer_bytes,budget.retry_attempts),(2,0,0,1))
        self.assertFalse(source.pending_validation)
        self.assertEqual(budget.retry_events[-1]["decision"],"per_pre_limit");self.assertTrue(retry_summary(budget)["exhausted"])

    def test_global_three_retries_across_parts_and_event_cap(self):
        budget=self.budget()
        original=self.session.get;fail=[True]
        def get(url,**kwargs):
            if "alt=media" not in url and fail[0]:fail[0]=False;raise RuntimeError(SECRET)
            return original(url,**kwargs)
        self.session.get=get
        with self.type_patch:
            for i in range(4):
                file_id=f"fake_{i}"
                self.session.data[file_id]=self.data
                self.session.docs[file_id]={**self.session.docs["fake"],"id":file_id}
                part=copy.deepcopy(self.part);part["source"]["file_id"]=file_id
                source=DriveSource(part,self.session)
                fail[0]=True
                if i<3:self.assertEqual(source.read_range(0,len(self.data)-1,budget,lambda b:None),self.data)
                else:
                    with self.assertRaises(PreparationError):source.read_range(0,len(self.data)-1,budget,lambda b:None)
        self.assertEqual((budget.retry_attempts,budget.media_requests,budget.ranges),(3,3,13))
        self.assertEqual(len(budget.retry_events),4);self.assertEqual(budget.retry_events[-1]["decision"],"global_limit")
        self.assertEqual(sum(self.clock.waits),1.5)

    def test_retry_reserves_future_media_and_post_requests(self):
        budget=self.budget(max_ranges=3);self.fail_calls({1})
        with self.type_patch,self.assertRaises(PreparationError):self.read(budget)
        self.assertEqual((len(self.attempts),budget.ranges,budget.retry_attempts),(1,1,0))
        self.assertEqual(budget.retry_events[-1]["decision"],"request_budget");self.assertEqual(self.clock.waits,[])
        budget=self.budget(max_ranges=4);self.session.get=self.session.__class__.get.__get__(self.session);self.fail_calls({1})
        with patch("rewind_archive.request_policy.classify_exception",return_value=("connect_timeout","urllib3")):
            self.read(budget)
        self.assertEqual(budget.ranges,4)

    def test_initial_reservation_rejects_without_any_attempt(self):
        budget=self.budget(max_ranges=2);self.fail_calls({1})
        with self.assertRaises(PreparationError) as err:self.read(budget)
        self.assertEqual(err.exception.reason,"ranges_limit");self.assertEqual(self.attempts,[])

    def test_transfer_reservation_does_not_start_request(self):
        budget=self.budget(max_transfer_bytes=len(self.data)+32767);self.fail_calls({1})
        with self.assertRaises(PreparationError) as err:self.read(budget)
        self.assertEqual(err.exception.reason,"transfer_bytes_limit");self.assertEqual(self.attempts,[])

    def test_timeout_clipped_and_wait_refused_when_remaining_too_short(self):
        budget=self.budget(max_elapsed_seconds=1);self.fail_calls({1},elapsed=.6)
        with self.type_patch,self.assertRaises(PreparationError):self.read(budget)
        self.assertEqual(self.attempts[0][1]["timeout"],1.)
        self.assertEqual(self.clock.waits,[]);self.assertEqual(budget.retry_events[-1]["decision"],"time_budget")

    def test_actual_wait_and_deadline_rechecked_before_retry(self):
        budget=self.budget(max_elapsed_seconds=1);self.fail_calls({1})
        def oversleep(seconds):self.clock.sleep(seconds+1)
        budget._sleeper=oversleep
        with self.type_patch,self.assertRaises(PreparationError):self.read(budget)
        self.assertEqual(budget.retry_wait_seconds,1.5);self.assertEqual(len(self.attempts),1)
        self.assertEqual(budget.retry_events[-1]["decision"],"time_budget")

    def test_clock_advances_each_check_before_second_request_reservation(self):
        # sleep 후 while/reserve/timeout/request 경계마다 시간이 진행한다.
        for expire_at in (1,2,3,4,5):
            with self.subTest(expire_at=expire_at):
                self.clock=Clock();budget=self.budget(max_elapsed_seconds=1)
                original_get=FakeDrive.get.__get__(self.session);attempts=[]
                def get(url,**kwargs):attempts.append(url);raise RuntimeError(SECRET)
                self.session.get=get
                def sleep(seconds):
                    self.clock.sleep(seconds)
                    checks=[0]
                    def ticking():
                        checks[0]+=1
                        return 2. if checks[0]>=expire_at else self.clock.now
                    budget.clock=ticking
                budget._sleeper=sleep
                with self.type_patch,self.assertRaises(PreparationError) as err:self.read(budget)
                self.assertEqual(err.exception.reason,"authorized_request_failed")
                if expire_at<=4:  # 실제 wait 계측의 clock 호출1회 + 3개 사전 검사
                    self.assertEqual(len(attempts),1);self.assertEqual(budget.retry_attempts,0)
                    self.assertEqual(budget.retry_events[-1]["decision"],"time_budget")
                    self.assertFalse(budget.retry_events[-1]["retry_started"])
                else:
                    self.assertEqual(len(attempts),2);self.assertEqual(budget.retry_attempts,1)
                    self.assertTrue(budget.retry_events[0]["retry_started"])
                    self.assertEqual(budget.retry_events[-1]["decision"],"per_pre_limit")
                self.session.get=original_get
                self.type_patch=patch("rewind_archive.request_policy.classify_exception",return_value=("read_timeout","requests"))

    def test_post_timeout_never_retried_and_pending_retained(self):
        budget=self.budget();self.fail_calls({3});source=DriveSource(self.part,self.session)
        with self.type_patch,self.assertRaises(PreparationError) as err:source.read_range(0,len(self.data)-1,budget,lambda b:None)
        self.assertEqual(err.exception.request_failure["operation"],"metadata_post")
        self.assertTrue(source.pending_validation);self.assertEqual(budget.retry_attempts,0);self.assertEqual(len(self.attempts),3)

    def test_source_changed_after_successful_retry_is_unusable(self):
        original=self.session.get;calls=[0]
        def get(url,**kwargs):
            calls[0]+=1
            if calls[0]==1:raise RuntimeError(SECRET)
            if calls[0]==4:self.session.docs["fake"]["version"]="10"
            return original(url,**kwargs)
        self.session.get=get
        with self.type_patch,patch("rewind_archive.contracts.time.sleep",self.clock.sleep):
            report=prepare({"schema_version":"archive_manifest_v1","archive_id":"fake_pose","dataset_version":"mock_v1",
                "kind":"pose","source_scope":"user_drive","total_parts":1,"parts":[{"index":0,"name":"pose.tar.gz000",
                "version":"9",**self.part,"source":{"type":"drive","file_id":"fake"}}]},self.root/"changed",[A,B],
                profile_limits(POSE_COUNTERPART_PROFILE),authorized_session=self.session,profile=POSE_COUNTERPART_PROFILE,
                exact_targets=[A,B],request_policy=METADATA_PRE_POLICY,clock=self.clock,sleeper=self.clock.sleep)
        self.assertEqual(report["stop_reason"],"source_changed");self.assertFalse(report["inspection_candidates_usable"])
        self.assertTrue(report["source_validation_pending"]);self.assertFalse(report["targets_complete"])
        self.assertEqual(report["request_retry"]["retries_started"],1);self.assertEqual(calls[0],4)

    def test_partial_metadata_body_counts_bytes_and_no_retry(self):
        original=self.session.get
        class Partial(Raw):
            def read1(self,size,decode_content=False):raise http.client.IncompleteRead(b"123",12)
        def get(url,**kwargs):
            response=original(url,**kwargs);response.raw=Partial(b"");return response
        self.session.get=get;budget=self.budget()
        with self.type_patch,self.assertRaises(PreparationError) as err:self.read(budget)
        self.assertEqual(err.exception.reason,"http_body_failed");self.assertEqual(budget.transfer_bytes,3)
        self.assertEqual(budget.retry_attempts,0);self.assertEqual(budget.ranges,1)

    def test_http_403_and_429_are_never_retried(self):
        for status in (403,429):
            with self.subTest(status=status):
                self.session.protocol={"fake":(status,{"Content-Length":"10"})};budget=self.budget()
                with self.type_patch,self.assertRaises(PreparationError):self.read(budget)
                self.assertEqual(budget.ranges,1);self.assertEqual(budget.retry_events,[])
                self.assertEqual(budget.metadata_response_audit[0]["response"]["status"],status)

    def test_media_open_or_partial_body_never_retried(self):
        for body in (False,True):
            session=self.session;original=FakeDrive.get.__get__(session)
            class Partial(Raw):
                def read1(self,*a,**k):raise http.client.IncompleteRead(b"xyz",10)
            def get(url,**kwargs):
                if "alt=media" in url and not body:raise RuntimeError(SECRET)
                response=original(url,**kwargs)
                if "alt=media" in url:response.raw=Partial(b"")
                return response
            session.get=get;budget=self.budget();source=DriveSource(self.part,session)
            with self.type_patch,self.assertRaises(PreparationError):source.read_range(0,len(self.data)-1,budget,lambda b:None)
            self.assertTrue(source.pending_validation);self.assertEqual(budget.media_requests,1)
            self.assertEqual(budget.retry_attempts,0)

    def test_response_wrapping_failure_never_retried(self):
        self.session.get=lambda *a,**k:SimpleNamespace(close=lambda:None)
        budget=self.budget()
        with self.type_patch,self.assertRaises(PreparationError) as err:self.read(budget)
        self.assertEqual(err.exception.request_failure["stage"],"response_wrap");self.assertEqual(budget.ranges,1)

    def test_forged_diagnostic_and_stdlib_timeout_are_not_eligible(self):
        failure={"category":"read_timeout","library":"requests","operation":"metadata_pre","stage":"request_open"}
        for error in (PreparationError("authorized_request_failed",request_failure=failure),TimeoutError(SECRET),
                RuntimeError(SECRET)):
            self.session.get=lambda *a,**k:(_ for _ in ()).throw(error)
            budget=self.budget()
            with self.assertRaises(PreparationError) as err:self.read(budget)
            self.assertFalse(eligible_open_timeout(err.exception));self.assertEqual(budget.ranges,1)

    def test_hidden_adapter_and_auth_retries_rejected_before_network(self):
        for current in (SimpleNamespace(),SimpleNamespace(_max_refresh_attempts=1),
                SimpleNamespace(get_adapter=lambda _:SimpleNamespace(max_retries=SimpleNamespace(total=3))),
                SimpleNamespace(get_adapter=self.session.get_adapter,_auth_request=SimpleNamespace()),
                SimpleNamespace(get_adapter=self.session.get_adapter,_auth_request_session=SimpleNamespace())):
            with self.subTest(current=type(current)),self.assertRaises(PreparationError):validate_no_hidden_retries(current)
        validate_no_hidden_retries(self.session)
        self.session._max_refresh_attempts=1
        root,record=run_workflow_job(self.config,make_preflight(self.config,{}),{},self.root,lambda:self.session)
        self.assertEqual(record["stop_reason"],"session_retry_policy");self.assertEqual(self.session.calls,[])

    def test_policy_fingerprint_scope_and_legacy_config_hashes(self):
        legacy={k:v for k,v in self.config.items() if k!="request_policy"}
        old=make_preflight(legacy,{})
        self.assertEqual(old["config"],legacy);self.assertEqual(old["config_sha256"],digest(legacy))
        null=make_preflight({**legacy,"request_policy":None},{})
        new=make_preflight(self.config,{})
        self.assertEqual(old["effective_sha256"],null["effective_sha256"])
        self.assertNotEqual(old["effective_sha256"],new["effective_sha256"])
        self.assertEqual(new["summary"]["limits"],legacy["limits"])
        self.assertEqual(new["summary"]["request_policy"]["max_retries_total"],3)
        for name in ("bad",{},False):
            with self.assertRaises(PreparationError):make_preflight({**self.config,"request_policy":name},{})
        with self.assertRaises(PreparationError):make_preflight({**self.config,"profile":"counterpart64MiB_v1"},{})
        review=make_preflight({**self.config,"mode":"review","purpose":"review","previous_result":"old"},{})
        self.assertEqual(review["summary"]["request_policy"]["name"],METADATA_PRE_POLICY)

    def test_legacy_default_uses_five_seconds_and_zero_retries(self):
        budget=Budget(Limits(),clock=self.clock,sleeper=self.clock.sleep);self.fail_calls({1})
        with self.type_patch,self.assertRaises(PreparationError):self.read(budget)
        self.assertEqual(len(self.attempts),1);self.assertEqual(self.attempts[0][1]["timeout"],5.)
        self.assertEqual(budget.retry_attempts,0)

    def test_initial_metadata_has_separate_budget_and_zero_retries(self):
        self.fail_calls({1})
        with self.type_patch:result=build_drive_manifest(self.selection,self.session,MetadataLimits())
        self.assertFalse(result["report"]["manifest_ready"]);self.assertEqual(len(self.attempts),1)
        self.assertEqual(result["report"]["stats"]["requests"],1)
        self.assertLessEqual(self.attempts[0][1]["timeout"],5.)

    def test_success_history_and_final_failure_null_through_reports_review_zip(self):
        self.fail_calls({2})
        with self.type_patch,patch("rewind_archive.sources.RETRY_DELAY_SECONDS",0):
            root,record=run_workflow_job(self.config,make_preflight(self.config,{}),{},self.root,lambda:self.session)
        self.assertTrue(record["targets_complete"]);self.assertIsNone(record["request_failure"])
        self.assertEqual(record["request_retry"]["retries_started"],1)
        review=review_run(root);self.assertIsNone(review["request_failure"])
        self.assertEqual(review["request_retry"],record["request_retry"])
        with ZipFile(save_result_zip(root)) as z:
            self.assertIsNone(z.testzip())
            for name in z.namelist():
                if name.endswith(".json"):
                    value=json.loads(z.read(name));self.assertNotIn(SECRET,json.dumps(value))
                    if name.endswith(("extraction_manifest.json","pilot_report.json","job.json")):
                        self.assertIsNone(value["request_failure"]);self.assertEqual(value["request_retry"],record["request_retry"])
        self.assertFalse(list(root.rglob("*.partial")))


if __name__=="__main__":unittest.main()
