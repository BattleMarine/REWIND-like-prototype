"""Pose metadata 밀도 재현과 독립 상한·마지막 1행 진단 회귀. 외부 요청 없음."""

import copy
import gzip
import io
import json
import random
import tarfile
import tempfile
import unittest
from dataclasses import asdict, replace
from pathlib import Path
from unittest.mock import patch

from rewind_archive.contracts import Budget, Limits, PreparationError
from rewind_archive.drive_metadata import MetadataLimits
from rewind_archive.extraction import GzipReader, prepare
from rewind_archive.fixture import SEQUENCE, local_manifest, tar_bytes
from rewind_archive.profiles import COUNTERPART_PROFILE, POSE_COUNTERPART_PROFILE, SMOKE_PROFILE, profile_limits
from rewind_archive.workflow import make_preflight, run_workflow_job, save_result_zip
from test_drive_metadata import FakeDrive

A=f"{SEQUENCE}/json/frame_1.json"
B=f"{SEQUENCE}/json/frame_10.json"


def pax_data(count, uname=None, long_paths=True):
    output=io.BytesIO()
    with tarfile.open(fileobj=output,mode="w",format=tarfile.PAX_FORMAT) as tar:
        for i in range(count):
            name=("p"*102+"/") if long_paths else ""
            info=tarfile.TarInfo(name+f"unused_{i}.json")
            if uname is not None:info.pax_headers={"uname":"u"*uname}
            payload=b'{"mock":true}'
            info.size=len(payload);tar.addfile(info,io.BytesIO(payload))
        for name in (B,A,"not_needed_after_targets"):
            info=tarfile.TarInfo(name);info.size=2;tar.addfile(info,io.BytesIO(b"{}"))
    return gzip.compress(output.getvalue(),mtime=0)


class PoseProfileTests(unittest.TestCase):
    def setUp(self):
        base=Path(__file__).resolve().parents[1]/"outputs/pose_profile_test_tmp"
        base.mkdir(parents=True,exist_ok=True)
        self.tmp=tempfile.TemporaryDirectory(dir=base);self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name);self.counter=0
        self.limits=profile_limits(POSE_COUNTERPART_PROFILE)

    def config(self):
        return {"job_id":"pose_density","reason":"bounded mock pose metadata density",
            "mode":"extract","purpose":"targeted","profile":POSE_COUNTERPART_PROFILE,
            "selection":{"schema_version":"drive_part_selection_v1","archive_id":"fake_pose",
                "dataset_version":"mock_v1","kind":"pose","total_parts":1,
                "parts":[{"index":0,"name":"pose.tar.gz000","file_id":"fake_pose"}]},
            "includes":[A,B],"exact_targets":[A,B],"limits":asdict(self.limits),
            "metadata_limits":asdict(MetadataLimits()),"previous_result":""}

    def extract(self,raw,profile=POSE_COUNTERPART_PROFILE,mode="extract"):
        self.counter+=1;base=self.root/f"input{self.counter}"
        manifest=local_manifest(base,raw,"pose")
        out=self.root/f"result{self.counter}"
        report=prepare(manifest,out,[A,B] if mode=="extract" else [],profile_limits(profile),base,
            profile=profile,exact_targets=[A,B] if mode=="extract" else [])
        return report,out

    def test_fixed_pose_preset_and_old_presets_unchanged(self):
        expected=replace(profile_limits(COUNTERPART_PROFILE),max_members=10000,
            max_metadata_bytes=1024**2,max_decompressed_bytes=512*1024**2)
        self.assertEqual(self.limits,expected)
        self.assertEqual(profile_limits(SMOKE_PROFILE),Limits())
        self.assertEqual(profile_limits(COUNTERPART_PROFILE).max_members,1000)
        path=Path(__file__).resolve().parents[1]/"configs/archive_limits_poseCounterpart64MiB_v1.json"
        self.assertEqual(json.loads(path.read_text()),asdict(self.limits))

    def test_profile_guards_kind_mode_targets_limits_and_review(self):
        config=self.config()
        for change in ({"selection":{**config["selection"],"kind":"rgb"}},
            {"mode":"scan","purpose":"prefix_inventory"},{"exact_targets":[A]},
            {"limits":asdict(replace(self.limits,max_members=10001))}):
            bad=copy.deepcopy(config);bad.update(change)
            with self.subTest(change=change):
                with self.assertRaises(PreparationError):make_preflight(bad,{})
        config.update(mode="review",purpose="review",previous_result=str(self.root))
        self.assertEqual(make_preflight(config,{})["summary"]["total_request_upper_bound"],0)
        raw=gzip.compress(tar_bytes([(A,b"{}"),(B,b"{}")]),mtime=0)
        manifest=local_manifest(self.root/"rgb",raw,"rgb")
        with self.assertRaisesRegex(PreparationError,"pose_kind"):
            prepare(manifest,self.root/"invalid",[A,B],self.limits,self.root/"rgb",
                profile=POSE_COUNTERPART_PROFILE,exact_targets=[A,B])
        self.assertFalse((self.root/"invalid").exists())

    def test_actual_pax_density_old_header_stop_new_targets_and_zip(self):
        raw=pax_data(600)
        old,_=self.extract(raw,COUNTERPART_PROFILE)
        self.assertEqual(old["stop_reason"],"members_limit")
        self.assertEqual(old["stats"]["members"],1000)
        self.assertEqual(old["missing_targets"],[A,B])
        self.assertGreater(old["last_regular_observation"]["regular_ordinal"],128)
        config=self.config();session=FakeDrive(config["selection"],{"fake_pose":raw})
        session.close=lambda:None
        out,record=run_workflow_job(config,make_preflight(config,{}),{},self.root,lambda:session)
        report=json.loads((out/"pilot/extraction_manifest.json").read_text())
        self.assertTrue(record["targets_complete"])
        self.assertGreater(report["stats"]["members"],1000)
        self.assertGreater(report["stats"]["metadata_bytes"],64*1024)
        self.assertEqual(len(report["member_observations"]),128)
        self.assertEqual(report["regular_members_seen"],602)
        self.assertEqual(report["last_regular_observation"]["member"],A)
        self.assertTrue(report["last_regular_observation"]["stored"])
        self.assertFalse(report["full_source_listing_claimed"])
        size=sum((out/"pilot"/name).stat().st_size for name in ("extraction_manifest.json","pilot_report.json"))
        self.assertLessEqual(size,report["report_disk_reserve_bytes"])
        self.assertLessEqual(sum(p.stat().st_size for p in (out/"pilot").rglob("*") if p.is_file()),16*1024**2)
        self.assertTrue(save_result_zip(out).is_file())

    def test_actual_cumulative_metadata_old_cap_new_pass(self):
        raw=pax_data(350,uname=220,long_paths=False)
        old,_=self.extract(raw,COUNTERPART_PROFILE)
        self.assertEqual(old["stop_reason"],"metadata_bytes_limit")
        new,_=self.extract(raw)
        self.assertEqual(new["stop_reason"],"targets_complete")
        self.assertGreater(new["stats"]["metadata_bytes"],65536)

    def test_one_mib_cumulative_metadata_cap_is_independent(self):
        report,_=self.extract(pax_data(34,uname=32720,long_paths=False))
        self.assertEqual(report["stop_reason"],"metadata_bytes_limit")
        self.assertLessEqual(report["stats"]["metadata_bytes"],1024**2)
        self.assertGreater(report["stats"]["metadata_bytes"],1000000)
        self.assertLess(report["stats"]["members"],10000)
        self.assertFalse(report["targets_complete"])

    def test_single_metadata_64k_rejected_before_payload_allocation(self):
        raw=pax_data(1,uname=65536,long_paths=False)
        with patch("rewind_archive.extraction.read_payload",side_effect=AssertionError("must not allocate metadata payload")):
            report,_=self.extract(raw)
        self.assertEqual(report["stop_reason"],"metadata_entry_bytes_limit")
        self.assertEqual(report["stats"]["metadata_bytes"],0)
        self.assertEqual(report["metadata_entry_limit_bytes"],65536)
        old,_=self.extract(raw,COUNTERPART_PROFILE)
        self.assertEqual(old["stop_reason"],"metadata_bytes_limit")

    def test_ten_thousand_header_cap_with_small_compressed_fixture(self):
        output=io.BytesIO()
        with tarfile.open(fileobj=output,mode="w",format=tarfile.USTAR_FORMAT) as tar:
            for i in range(10001):
                info=tarfile.TarInfo(f"dir_{i}");info.type=tarfile.DIRTYPE;tar.addfile(info)
        compressed=gzip.compress(output.getvalue(),mtime=0)
        self.assertLess(len(compressed),128*1024)
        report,_=self.extract(compressed)
        self.assertEqual(report["stop_reason"],"members_limit")
        self.assertEqual(report["stats"]["members"],10000)
        self.assertEqual(report["stats"]["metadata_bytes"],0)
        self.assertIsNone(report["last_regular_observation"])

    def test_512m_decompression_boundary_with_bounded_prefix_state(self):
        # 이미 skip한 prefix는 예산 상태로 표현한다. 512MiB 원본을 만들지 않는다.
        budget=Budget(self.limits);budget.decompressed_bytes=self.limits.max_decompressed_bytes-65536
        compressed=gzip.compress(bytes(2*65536),mtime=0)
        class Source:
            at_end=False
            def read(self):self.at_end=True;return compressed
        reader=GzipReader(Source(),budget)
        self.assertEqual(len(reader.read(65536)),65536)
        with self.assertRaisesRegex(PreparationError,"decompressed_bytes_limit"):reader.read(1)
        self.assertEqual(budget.decompressed_bytes,512*1024**2)
        self.assertEqual(budget.peak_output_chunk_bytes,65536)

    def test_pose_partial_and_source_change_keep_last_row_untrusted(self):
        config=self.config()
        raw=gzip.compress(tar_bytes([(A,b"{}"),(B,random.Random(7).randbytes(400000))]),mtime=0)
        partial,_=self.extract(raw[:800])
        self.assertEqual(partial["completed_targets"],[A]);self.assertEqual(partial["missing_targets"],[B])
        self.assertFalse(partial["last_regular_observation"]["payload_complete"])
        self.assertTrue(partial["member_observations_usable"])
        session=FakeDrive(config["selection"],{"fake_pose":raw});original=session.get;count=[0]
        def changing(url,**kwargs):
            if "alt=media" in url:
                count[0]+=1
                if count[0]==2:session.docs["fake_pose"]["version"]="10"
            return original(url,**kwargs)
        session.get=changing;session.close=lambda:None
        out,record=run_workflow_job(config,make_preflight(config,{}),{},self.root,lambda:session)
        report=json.loads((out/"pilot/extraction_manifest.json").read_text())
        self.assertEqual(report["stop_reason"],"source_changed")
        self.assertFalse(report["member_observations_usable"])
        self.assertEqual(report["member_observation_trust"],"untrusted_failed_source_or_parser")
        self.assertIsNotNone(report["last_regular_observation"])
        self.assertEqual(record["completed_targets"],[])

    def test_scan_still_stops_at_128_with_one_extra_last_header(self):
        raw=gzip.compress(tar_bytes([(f"item_{i}",b"{}") for i in range(150)]),mtime=0)
        manifest=local_manifest(self.root/"scan_input",raw,"pose")
        report=prepare(manifest,self.root/"scan",[],Limits(),self.root/"scan_input",mode="scan")
        self.assertEqual(report["stop_reason"],"observed_members_limit")
        self.assertEqual(len(report["member_observations"]),128)
        self.assertEqual(report["last_regular_observation"]["regular_ordinal"],129)
        self.assertFalse(report["last_regular_observation"]["payload_complete"])
        self.assertEqual(report["files"],[])


if __name__=="__main__":unittest.main()
