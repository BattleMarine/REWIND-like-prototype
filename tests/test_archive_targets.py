"""고정 profile·literal 목표 즉시중단·legacy notebook·실패/상한 회귀."""

import copy
import gzip
import json
import random
import tempfile
import unittest
from dataclasses import FrozenInstanceError, asdict, replace
from pathlib import Path
from unittest.mock import patch

from rewind_archive.contracts import Limits, PreparationError, digest
from rewind_archive.drive_metadata import MetadataLimits
from rewind_archive.extraction import prepare
from rewind_archive.fixture import SEQUENCE, local_manifest, tar_bytes
from rewind_archive.matching import verify_extraction
from rewind_archive.pilot import run_pilot
from rewind_archive.profiles import PROFILES, SMOKE_PROFILE, COUNTERPART_PROFILE, profile_limits, validate_profile_limits
from rewind_archive.sources import SplitStream
from rewind_archive.workflow import make_preflight, run_workflow_job, review_run, save_result_zip
from test_drive_metadata import FakeDrive


A=f"{SEQUENCE}/json/frame_1.json"
B=f"{SEQUENCE}/json/frame_10.json"


class TargetTests(unittest.TestCase):
    def setUp(self):
        root=Path(__file__).resolve().parents[1]/"outputs/target_test_tmp"
        root.mkdir(parents=True,exist_ok=True)
        self.tmp=tempfile.TemporaryDirectory(dir=root);self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name);self.counter=0

    def source(self,entries=None,raw=None):
        self.counter+=1;base=self.root/f"source{self.counter}"
        raw=gzip.compress(tar_bytes(entries),mtime=0) if raw is None else raw
        return local_manifest(base,raw,"pose"),base

    def config(self):
        return {"job_id":"counterpart_001","reason":"explicit bounded two counterpart files",
            "mode":"extract","purpose":"targeted","profile":COUNTERPART_PROFILE,
            "selection":{"schema_version":"drive_part_selection_v1","archive_id":"fake","dataset_version":"synthetic_v1",
                "kind":"pose","total_parts":1,"parts":[{"index":0,"name":"fake.tar.gz000","file_id":"fake"}]},
            "includes":[A,B],"exact_targets":[A,B],"limits":asdict(profile_limits(COUNTERPART_PROFILE)),
            "metadata_limits":asdict(MetadataLimits()),"previous_result":""}

    def test_profiles_immutable_config_matches_and_tampering_rejected(self):
        with self.assertRaises(TypeError):PROFILES[COUNTERPART_PROFILE]=Limits()
        with self.assertRaises(FrozenInstanceError):profile_limits(COUNTERPART_PROFILE).max_files=3
        path=Path(__file__).resolve().parents[1]/"configs/archive_limits_counterpart64MiB_v1.json"
        self.assertEqual(json.loads(path.read_text()),asdict(profile_limits(COUNTERPART_PROFILE)))
        self.assertEqual(profile_limits(SMOKE_PROFILE),Limits())
        for profile,limits in (("made_up",Limits()),(SMOKE_PROFILE,replace(Limits(),max_files=2)),
            (COUNTERPART_PROFILE,replace(profile_limits(COUNTERPART_PROFILE),max_transfer_bytes=64*1024**2+1)),
            (COUNTERPART_PROFILE,replace(profile_limits(COUNTERPART_PROFILE),max_files=1))):
            with self.subTest(profile=profile):
                with self.assertRaises(PreparationError):validate_profile_limits(profile,limits)

    def test_invalid_exact_targets_rejected_before_output_or_source(self):
        manifest,base=self.source([(A,b"a"),(B,b"b")])
        cases=[([A,B],[A,A]),([A,B],[A.upper(),A]),([A,B],[A,"../escape"]),
            (["*/json/*.json"],[A,B]),([A],[A,B]),([A,B],[A,B+"*"]),([A,B],"not_list")]
        for includes,targets in cases:
            with self.subTest(targets=targets):
                with self.assertRaises(PreparationError):
                    prepare(manifest,self.root/"invalid",includes,Limits(),base,exact_targets=targets)
                self.assertFalse((self.root/"invalid").exists())

    def test_both_targets_stop_before_next_header_or_source_read(self):
        # 다음 header는 unsafe다. 거기까지 진행하지 않고 source 추가 호출도 하지 않는다.
        manifest,base=self.source([(A,b"a"),(B,b"b"),("../unsafe",random.Random(1).randbytes(5000))])
        calls=[];original=SplitStream.read
        def read_once(stream):
            calls.append(stream.offset)
            if len(calls)>1:raise AssertionError("unexpected next source read")
            return original(stream)
        with patch.object(SplitStream,"read",read_once):
            report=prepare(manifest,self.root/"complete",[A,B],Limits(),base,exact_targets=[A,B])
        self.assertEqual(report["stop_reason"],"targets_complete")
        self.assertEqual(report["failure_class"],"bounded_stop")
        self.assertEqual(report["completed_targets"],[A,B]);self.assertEqual(report["missing_targets"],[])
        self.assertTrue(report["targets_complete"])
        self.assertEqual(report["stats"]["members"],2)
        self.assertEqual(len(calls),1)
        self.assertFalse(report["whole_archive_integrity_verified"])
        self.assertEqual(len(verify_extraction(self.root/"complete")["files"]),2)

    def test_expanded_profile_reaches_targets_after_unselected_eight_mib(self):
        entries=[(f"skip_{i}",random.Random(i).randbytes(2*1024**2)) for i in range(4)]
        entries += [(A,b"first"),(B,b"second"),("tail",random.Random(9).randbytes(1024**2))]
        manifest,base=self.source(entries)
        smoke=prepare(manifest,self.root/"smoke",[A,B],Limits(),base,profile=SMOKE_PROFILE,exact_targets=[A,B])
        self.assertFalse(smoke["targets_complete"])
        self.assertEqual(smoke["missing_targets"],[A,B])
        expanded=prepare(manifest,self.root/"expanded",[A,B],profile_limits(COUNTERPART_PROFILE),base,
            profile=COUNTERPART_PROFILE,exact_targets=[A,B])
        self.assertEqual(expanded["stop_reason"],"targets_complete")
        self.assertGreater(expanded["stats"]["transfer_bytes"],8*1024**2)
        self.assertLess(expanded["stats"]["transfer_bytes"],sum(p["size_bytes"] for p in manifest["parts"]))
        self.assertEqual(expanded["stats"]["files"],2)
        self.assertEqual(expanded["profile"],COUNTERPART_PROFILE)

    def test_first_target_retained_second_partial_and_absent_reported(self):
        data=gzip.compress(tar_bytes([(A,b"first"),(B,random.Random(7).randbytes(4000))]),mtime=0)
        # 선언 prefix가 부족한 경우이며 CRC가 깨진 완전 stream과 구분한다.
        manifest,base=self.source(raw=data[:800])
        report=prepare(manifest,self.root/"partial",[A,B],Limits(),base,exact_targets=[A,B])
        self.assertEqual(report["completed_targets"],[A])
        self.assertEqual(report["missing_targets"],[B])
        self.assertFalse(report["targets_complete"])
        self.assertFalse(list((self.root/"partial").rglob("*.partial")))
        self.assertTrue(report["inspection_candidates_usable"])

    def test_source_change_invalidates_stored_targets_and_observations(self):
        raw=gzip.compress(tar_bytes([(A,random.Random(1).randbytes(3000)),(B,random.Random(2).randbytes(3000))]),mtime=0)
        config=self.config();session=FakeDrive(config["selection"],{"fake":raw})
        original=session.get;media_count=[0]
        def changing_get(url,**kwargs):
            if "alt=media" in url:
                media_count[0]+=1
                if media_count[0]==2:session.docs["fake"]["version"]="10"
            return original(url,**kwargs)
        session.get=changing_get;session.close=lambda:None
        # 작은 Range에서 first를 게시한 후 second의 postcheck에 변경을 주입한다.
        config.update(profile=None,limits=asdict(replace(Limits(),media_range_bytes=4096)))
        root,record=run_workflow_job(config,make_preflight(config,{}),{},self.root,lambda:session)
        report=json.loads((root/"pilot/extraction_manifest.json").read_text())
        self.assertEqual(report["stop_reason"],"source_changed")
        self.assertEqual(report["stored_targets"],[A]);self.assertEqual(report["completed_targets"],[])
        self.assertEqual(report["missing_targets"],[A,B]);self.assertEqual(report["targets_status"],"unusable")
        self.assertFalse(report["member_observations_usable"])
        self.assertFalse(record["targets_complete"])

    def test_target_after_observation_cap_and_report_reserve_zip(self):
        entries=[(f"skip_{i}",b"") for i in range(140)]+[(A,b"first"),(B,b"second")]
        raw=gzip.compress(tar_bytes(entries),mtime=0)
        config=self.config();session=FakeDrive(config["selection"],{"fake":raw});session.close=lambda:None
        root,record=run_workflow_job(config,make_preflight(config,{}),{},self.root,lambda:session)
        self.assertTrue(record["targets_complete"])
        report=json.loads((root/"pilot/extraction_manifest.json").read_text())
        self.assertTrue(report["member_observation_cap_reached"])
        self.assertEqual(len(report["member_observations"]),128)
        self.assertEqual(report["stats"]["members"],142)
        report_bytes=sum((root/"pilot"/n).stat().st_size for n in ("extraction_manifest.json","pilot_report.json"))
        self.assertLessEqual(report_bytes,report["report_disk_reserve_bytes"])
        self.assertLessEqual(sum(p.stat().st_size for p in (root/"pilot").rglob("*") if p.is_file()),16*1024**2)
        self.assertTrue(save_result_zip(root).is_file())
        self.assertTrue(review_run(root)["target_verification"]["targets_complete"])

    def test_workflow_legacy_normalization_stale_and_old_fingerprint(self):
        legacy=self.config()
        legacy.pop("profile");legacy.pop("exact_targets")
        legacy.update(mode="scan",purpose="prefix_inventory",includes=[],limits=asdict(Limits()))
        history={};snapshot=make_preflight(legacy,history)
        old_effective={k:legacy[k] for k in ("mode","purpose","selection","includes","limits","metadata_limits")}
        self.assertEqual(snapshot["effective_sha256"],digest(old_effective))
        self.assertIsNone(snapshot["config"]["profile"])
        calls=[0]
        raw=gzip.compress(tar_bytes([(A,b"abc")]),mtime=0)
        def session_factory():calls[0]+=1;return FakeDrive(legacy["selection"],{"fake":raw})
        root,record=run_workflow_job(legacy,snapshot,history,self.root,session_factory)
        self.assertEqual(record["status"],"finished")
        self.assertEqual(record["profile"],"custom_limits")
        changed=copy.deepcopy(legacy);changed["reason"]="changed after preflight"
        with self.assertRaises(PreparationError):run_workflow_job(changed,snapshot,history,self.root,session_factory)
        self.assertEqual(calls[0],1)
        old_record={"job_id":"v2_old","config":legacy,"effective_sha256":digest(old_effective)}
        self.assertEqual(make_preflight(legacy,{"v2_old":old_record})["summary"]["same_config_previous_jobs"],["v2_old"])

    def test_workflow_profile_target_and_changed_budget_guards(self):
        config=self.config();snapshot=make_preflight(config,{})
        self.assertEqual(snapshot["summary"]["total_request_upper_bound"],776)
        self.assertEqual(snapshot["summary"]["total_body_upper_bound_bytes"],64*1024**2+128*1024)
        self.assertEqual(snapshot["summary"]["total_cooperative_stage_upper_bound_seconds"],330)
        for change in ({"profile":"made_up"},{"mode":"scan","purpose":"prefix_inventory"},{"exact_targets":[A]},
            {"includes":[A]},{"limits":asdict(Limits())},
            {"metadata_limits":{**config["metadata_limits"],"max_transfer_bytes":131073}}):
            bad=copy.deepcopy(config);bad.update(change)
            with self.subTest(change=change):
                with self.assertRaises(PreparationError):make_preflight(bad,{})
        calls=[]
        changed=copy.deepcopy(config);changed["exact_targets"]=[B,A]
        with self.assertRaises(PreparationError):
            run_workflow_job(changed,snapshot,{},self.root,lambda:calls.append("auth"))
        self.assertEqual(calls,[])
        review=copy.deepcopy(config);review.update(mode="review",purpose="review",previous_result=str(self.root))
        summary=make_preflight(review,{})["summary"]
        self.assertEqual(summary["total_request_upper_bound"],0)
        self.assertFalse(summary["payload_storage_enabled"])

    def test_pilot_file_verification_failure_never_marks_targets_complete(self):
        manifest,base=self.source([(A,b"a"),(B,b"b")])
        with patch("rewind_archive.pilot.verify_extraction",side_effect=PreparationError("extracted_file_hash")):
            report=run_pilot(manifest,self.root/"verifyfail",[A,B],Limits(),base=base,exact_targets=[A,B])
        self.assertFalse(report["targets_complete"])
        self.assertEqual(report["stored_targets"],[A,B])
        self.assertEqual(report["completed_targets"],[])
        self.assertEqual(report["missing_targets"],[A,B])
        self.assertEqual(report["targets_status"],"unusable")


if __name__=="__main__":unittest.main()
