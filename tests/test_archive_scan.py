"""Bounded tar 순서 관측과 payload 저장 없는 scan 회귀."""

import contextlib
import gzip
import io
import json
import random
import tarfile
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from rewind_archive.__main__ import main
from rewind_archive.contracts import Limits, canonical
from rewind_archive.extraction import prepare, MAX_OBSERVED_MEMBERS
from rewind_archive.fixture import SEQUENCE, local_manifest, tar_bytes
from rewind_archive.pilot import run_pilot


class ScanTests(unittest.TestCase):
    def setUp(self):
        base=Path(__file__).resolve().parents[1]/"outputs/scan_test_tmp"
        base.mkdir(parents=True,exist_ok=True)
        self.tmp=tempfile.TemporaryDirectory(dir=base);self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name)
        self.limits=Limits(io_chunk_bytes=512)

    def setup_source(self, entries=None, raw=None, split=None):
        data=raw if raw is not None else gzip.compress(tar_bytes(entries),mtime=0)
        manifest=local_manifest(self.root/"source",data,"pose",split_bytes=split)
        return manifest,self.root/"source"

    def assert_bounded(self, report, output):
        self.assertLessEqual(sum(p.stat().st_size for p in output.rglob("*") if p.is_file()),report["limits"]["max_disk_bytes"])
        self.assertFalse(list(output.rglob("*.partial")))
        self.assertLessEqual(len(report["member_observations"]),MAX_OBSERVED_MEMBERS)

    def test_scan_preserves_regular_tar_order_and_stores_zero(self):
        entries=[(f"{SEQUENCE}/json/frame_{i}.json",bytes([i])*20) for i in (10,1,5)]
        manifest,base=self.setup_source(entries)
        out=self.root/"scan"
        pilot=run_pilot(manifest,out,[],self.limits,base=base,mode="scan")
        report=json.loads((out/"extraction_manifest.json").read_text())
        observed=report["member_observations"]
        self.assertEqual([r["member"] for r in observed],[name for name,_ in entries])
        self.assertEqual([r["regular_ordinal"] for r in observed],[1,2,3])
        self.assertTrue(all(r["payload_complete"] and not r["selected"] and not r["stored"] for r in observed))
        self.assertEqual(pilot["observed_regular_members"],3)
        self.assertEqual(pilot["complete_files"],0)
        self.assertFalse(pilot["sample_acquired"])
        self.assertFalse((out/"files").exists())
        self.assertFalse(report["full_source_listing_claimed"])
        self.assertTrue(report["member_observations_usable"])
        self.assert_bounded(report,out)

    def test_extract_observes_unselected_and_selected_without_changing_filter(self):
        manifest,base=self.setup_source([(f"{SEQUENCE}/json/frame_{i}.json",b"abc") for i in (2,0,1)])
        report=prepare(manifest,self.root/"extract",["*/frame_0.json"],self.limits,base)
        obs=report["member_observations"]
        self.assertEqual([r["selected"] for r in obs],[False,True,False])
        self.assertEqual([r["stored"] for r in obs],[False,True,False])
        self.assertTrue(all(r["payload_complete"] for r in obs))
        self.assertEqual(len(report["files"]),1)

    def test_partial_payload_header_is_observed_but_not_complete(self):
        raw=tar_bytes([(f"{SEQUENCE}/json/frame_0.json",random.Random(9).randbytes(4000))])
        # 완전 gzip이어도 tar payload가 잘렸으면 목록 전체의 신뢰를 거부한다.
        manifest,base=self.setup_source(raw=gzip.compress(raw[:1024],mtime=0))
        report=prepare(manifest,self.root/"partial",[],self.limits,base,mode="scan")
        self.assertEqual(report["stop_reason"],"incomplete_payload")
        self.assertEqual(len(report["member_observations"]),1)
        self.assertFalse(report["member_observations"][0]["payload_complete"])
        self.assertFalse(report["member_observations_usable"])
        self.assertEqual(report["member_observation_trust"],"untrusted_failed_source_or_parser")

    def test_observation_cap_stops_instead_of_unbounded_records(self):
        manifest,base=self.setup_source([(f"item_{i}",b"") for i in range(150)])
        report=prepare(manifest,self.root/"cap",[],self.limits,base,mode="scan")
        self.assertEqual(report["stop_reason"],"observed_members_limit")
        self.assertEqual(len(report["member_observations"]),128)
        self.assertTrue(report["member_observation_cap_reached"])
        self.assertTrue(report["member_observations_usable"])
        self.assert_bounded(report,self.root/"cap")
        # 목록 cap이 기존 선택 추출의 도달 범위를 바꾸지 않는다.
        extracted=prepare(manifest,self.root/"extract_after_cap",["item_149"],self.limits,base)
        self.assertEqual(extracted["stop_reason"],"complete")
        self.assertEqual([f["member"] for f in extracted["files"]],["item_149"])
        self.assertEqual(len(extracted["member_observations"]),128)
        self.assertTrue(extracted["member_observation_cap_reached"])

    def test_crc_or_unsafe_failure_invalidates_prior_observations(self):
        raw=gzip.compress(tar_bytes([("first",random.Random(1).randbytes(1200)),("second",b"a")]),mtime=0)
        damaged=bytearray(raw);damaged[-8]^=1
        manifest,base=self.setup_source(raw=bytes(damaged))
        report=prepare(manifest,self.root/"crc",[],self.limits,base,mode="scan")
        self.assertEqual(report["stop_reason"],"gzip_corrupt")
        self.assertFalse(report["member_observations_usable"])
        # Link/special은 이전 regular의 진단 기록이 있어도 usable로 남기지 않는다.
        buffer=io.BytesIO()
        with tarfile.open(fileobj=buffer,mode="w",format=tarfile.USTAR_FORMAT) as tar:
            first=tarfile.TarInfo("safe");first.size=1;tar.addfile(first,io.BytesIO(b"a"))
            link=tarfile.TarInfo("unsafe");link.type=tarfile.SYMTYPE;link.linkname="safe";tar.addfile(link)
        manifest=local_manifest(self.root/"unsafe_source",gzip.compress(buffer.getvalue(),mtime=0),"pose")
        report=prepare(manifest,self.root/"unsafe",[],self.limits,self.root/"unsafe_source",mode="scan")
        self.assertEqual(len(report["member_observations"]),1)
        self.assertEqual(report["stop_reason"],"links_or_special_file")
        self.assertFalse(report["member_observations_usable"])

    def test_escape_worst_case_report_and_pilot_fit_disk_reserve(self):
        # UTF-8 byte/path 상한과 따옴표 JSON escape를 함께 검사한다.
        entries=[('"'*480+f"_{i}",b"") for i in range(128)]
        manifest,base=self.setup_source(entries)
        out=self.root/"escaped"
        run_pilot(manifest,out,[],replace(self.limits,max_path_bytes=500),base=base,mode="scan")
        report=json.loads((out/"extraction_manifest.json").read_text())
        self.assertEqual(len(report["member_observations"]),128)
        self.assertLessEqual(sum((out/name).stat().st_size for name in ("extraction_manifest.json","pilot_report.json")),report["report_disk_reserve_bytes"])
        self.assert_bounded(report,out)

    def test_byte_cap_and_cli_scan_do_not_store_payload(self):
        manifest,base=self.setup_source([("first",random.Random(3).randbytes(4000))])
        report=prepare(manifest,self.root/"bounded",[],replace(self.limits,max_transfer_bytes=700),base,mode="scan")
        self.assertEqual(report["stop_reason"],"transfer_bytes_limit")
        self.assertTrue(report["member_observations_usable"])
        self.assertFalse(report["member_observations"][0]["payload_complete"])
        limits_path=self.root/"limits.json";limits_path.write_bytes(canonical(self.limits.__dict__))
        manifest_path=base/"manifest.json";manifest_path.write_bytes(canonical(manifest))
        with contextlib.redirect_stdout(io.StringIO()):
            status=main(["scan","--manifest",str(manifest_path),"--limits",str(limits_path),"--output",str(self.root/"cli")])
        self.assertEqual(status,0)
        self.assertFalse((self.root/"cli/files").exists())


if __name__=="__main__":unittest.main()
