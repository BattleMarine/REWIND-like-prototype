"""요청 예약·이전 limits·관측 기반 대응 계획과 notebook 작업 중복 방지."""

import gzip
import io
import json
import tempfile
import unittest
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace

from rewind_archive.contracts import Budget, Limits, PreparationError, canonical, validate_includes
from rewind_archive.fixture import SEQUENCE, create_fixture, tar_bytes
from rewind_archive.matching import verify_extraction
from rewind_archive.pilot import run_pilot
from rewind_archive.planning import claim_pilot_job, counterpart_plan, discovery_globs
from rewind_archive.sources import DriveSource, SplitStream


class PlanningTests(unittest.TestCase):
    def setUp(self):
        base=Path(__file__).resolve().parents[1]/"outputs"/"archive_planning_test_tmp"
        base.mkdir(parents=True,exist_ok=True)
        self.tmp=tempfile.TemporaryDirectory(dir=base)
        self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name)

    def test_limits_legacy_json_and_independent_range_io(self):
        old=asdict(Limits());old.pop("media_range_bytes")
        self.assertEqual(Limits.from_dict(old).media_range_bytes,256*1024)
        self.assertEqual(Limits.from_dict(asdict(Limits())),Limits())
        with self.assertRaises(PreparationError):Limits.from_dict({**old,"other":1})
        with self.assertRaises(PreparationError):replace(Limits(),media_range_bytes=2*1024**2)

    def test_preflight_18_camera_globs_and_invalid_36_before_work(self):
        globs=[f"{SEQUENCE}/fisheye_final_image/camera_*/final_{i}.png" for i in range(18)]
        self.assertEqual(validate_includes(globs),globs)
        for bad in (globs+globs,[],["../*.png"],["a\\*.png"],["same","same"]):
            with self.assertRaises(PreparationError):validate_includes(bad)

    def test_discovery_observed_complete_keys_counterparts_and_reuse(self):
        fixture=self.root/"fixture";create_fixture(fixture)
        manifest=json.loads((fixture/"rgb_manifest.json").read_text())
        out=self.root/"discovery"
        pilot=run_pilot(manifest,out,discovery_globs("rgb"),replace(Limits(),max_files=2),base=fixture)
        self.assertTrue(pilot["sample_acquired"])
        self.assertEqual(pilot["complete_files"],2)
        plan=counterpart_plan([out])
        report=verify_extraction(out)
        actual={entry["member"] for entry in report["files"]}
        self.assertEqual(sum(len(f["observed_views"]) for f in plan["frames"]),len(actual))
        self.assertTrue(plan["member_globs_by_kind"]["pose"])
        self.assertFalse(plan["train_ready"])
        self.assertEqual(plan["bounded_target_reachability"],"unknown")
        entry=report["files"][0]
        (out/entry["relative_path"]).write_bytes(b"corrupt")
        with self.assertRaises(PreparationError):counterpart_plan([out])

    def test_zero_files_is_no_sample_and_no_target_proposal(self):
        fixture=self.root/"fixture";create_fixture(fixture)
        manifest=json.loads((fixture/"pose_manifest.json").read_text())
        out=self.root/"empty"
        pilot=run_pilot(manifest,out,["missing/*"],Limits(),base=fixture)
        self.assertEqual(pilot["verification"]["status"],"passed")
        self.assertFalse(pilot["sample_acquired"])
        self.assertEqual(pilot["sample_status"],"no_sample")
        plan=counterpart_plan([out])
        self.assertEqual(plan["next_action"],"stop_no_complete_sample")
        self.assertEqual(plan["member_globs_by_kind"],{"rgb":[],"pose":[]})

    def test_same_job_rejected_even_changed_path_inputs_and_explicit_new_id(self):
        jobs={}
        claim_pilot_job(jobs,"first",{},["**"],Limits(),"discovery",self.root/"one")
        with self.assertRaisesRegex(PreparationError,"already_started"):
            claim_pilot_job(jobs,"first",{"changed":True},["other/*"],Limits(),"targeted",self.root/"two")
        claim_pilot_job(jobs,"explicit_new",{},["**"],Limits(),"targeted",self.root/"two")
        self.assertEqual(len(jobs),2)

    def test_legacy_report_limits_can_be_verified(self):
        fixture=self.root/"fixture";create_fixture(fixture)
        manifest=json.loads((fixture/"pose_manifest.json").read_text())
        out=self.root/"old"
        run_pilot(manifest,out,["**"],Limits(),base=fixture)
        from rewind_archive.contracts import digest
        path=out/"extraction_manifest.json"
        doc=json.loads(path.read_text());doc["limits"].pop("media_range_bytes")
        body={k:v for k,v in doc.items() if k!="content_sha256"}
        path.write_bytes(canonical({**body,"content_sha256":digest(body)}))
        self.assertGreater(len(verify_extraction(out)["files"]),0)


class DriveReservationTests(unittest.TestCase):
    def part(self):
        return {"size_bytes":1024,"source":{"type":"drive","file_id":"fake"},
            "identity":{"version":"1","modified_time":"unchanged"}}

    def test_metadata_request_and_byte_reservation_before_network(self):
        class NoNetwork:
            def get(self,*args,**kwargs):raise AssertionError("network must not start")
        source=DriveSource(self.part(),NoNetwork())
        for limits,reason in ((replace(Limits(),max_ranges=2),"ranges_limit"),
            (replace(Limits(),max_transfer_bytes=32768),"transfer_bytes_limit")):
            budget=Budget(limits)
            with self.assertRaisesRegex(PreparationError,reason):source.read_range(0,511,budget,lambda c:None)
            self.assertEqual(budget.ranges,0)

    def test_postcheck_interruption_is_unusable_and_no_retry(self):
        raw=gzip.compress(tar_bytes([(f"{SEQUENCE}/json/frame_0.json",b"payload")]),mtime=0)
        class Raw(io.BytesIO):
            def read1(self,size,decode_content=False):return super().read1(size)
        class Session:
            def __init__(self,clock=None):self.calls=0;self.clock=clock
            def get(self,url,**kwargs):
                self.calls+=1
                if self.calls==3:
                    if self.clock is None:raise OSError("secret must not appear")
                    self.clock[0]=2
                if "alt=media" in url:
                    body=raw;headers={"Content-Length":str(len(raw)),"Content-Range":f"bytes 0-{len(raw)-1}/{len(raw)}"};status=206
                else:
                    body=canonical({"size":str(len(raw)),"version":"1","modifiedTime":"unchanged"})
                    headers={"Content-Length":str(len(body))};status=200
                response=SimpleNamespace(status_code=status,headers=headers,raw=Raw(body))
                response.close=response.raw.close
                return response
        manifest={"schema_version":"archive_manifest_v1","archive_id":"fake","dataset_version":"fake",
            "kind":"pose","source_scope":"synthetic_fixture","total_parts":1,
            "parts":[{"index":0,"name":"fake.tar.gz000","size_bytes":len(raw),"version":"1",
                "source":{"type":"drive","file_id":"fake"},"identity":{"version":"1","modified_time":"unchanged"}}]}
        session=Session();stream=SplitStream(manifest,".",Budget(Limits()),session)
        with self.assertRaisesRegex(PreparationError,"authorized_request_failed"):stream.read()
        self.assertTrue(stream.sources[0].pending_validation)
        self.assertEqual(session.calls,3)
        self.assertEqual(stream.budget.media_requests,1)
        self.assertEqual(stream.budget.metadata_requests,2)
        # elapsed_limit은 평소 benign이지만 후검사가 미완료면 추출 전체는 unusable.
        from rewind_archive.extraction import prepare
        base=Path(__file__).resolve().parents[1]/"outputs"/"archive_planning_test_tmp"
        base.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(dir=base) as tmp:
            clock=[0]
            out=Path(tmp)/"elapsed_postcheck"
            report=prepare(manifest,out,["**"],replace(Limits(),max_elapsed_seconds=1,request_timeout_seconds=.5),
                authorized_session=Session(clock),clock=lambda:clock[0])
            self.assertEqual(report["stop_reason"],"elapsed_limit")
            self.assertTrue(report["source_validation_pending"])
            self.assertFalse(report["inspection_candidates_usable"])
            with self.assertRaisesRegex(PreparationError,"unusable"):verify_extraction(out)


if __name__=="__main__":unittest.main()
