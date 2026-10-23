"""임시 스트리밍 fixture만 사용하는 v2→v3,192MiB/769chunk 및 실패 경계 검사."""
import copy
import gzip
import hashlib
import io
import json
import random
import tarfile
import tempfile
import unittest
from dataclasses import asdict,replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from zipfile import ZipFile,ZIP_STORED
import http.client

from rewind_archive.contracts import Budget,PreparationError,canonical,digest
from rewind_archive.profiles import WARM128_PROFILE,WARM192_PROFILE,profile_limits
from rewind_archive.cache_policy import WARM128_POLICY,WARM192_POLICY,validate_expectation
from rewind_archive.prefix_cache import input_spec,export_cache,open_bundle,read_manifest,resolve_policy
from rewind_archive.extraction import prepare,code_hashes,report_reserve_bytes
from rewind_archive.catalog import CATALOG_BYTES
from rewind_archive.fixture import SEQUENCE
from rewind_archive.drive_metadata import MetadataLimits
from rewind_archive.workflow import make_preflight,save_result_zip
from rewind_archive.request_policy import METADATA_PRE_POLICY
from test_drive_metadata import FakeDrive,Raw

M=1024**2;C=128*M
A=f"{SEQUENCE}/json/frame_1.json";B=f"{SEQUENCE}/json/frame_10.json"


class Warm192Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        base=Path(__file__).resolve().parents[1]/"outputs/warm192_test_tmp";base.mkdir(parents=True,exist_ok=True)
        cls.tmp=tempfile.TemporaryDirectory(dir=base);cls.base=Path(cls.tmp.name)
        cls.full=cls.base/"fixture.gz";payload=random.Random(19223).randbytes(M)
        cls.payload_sha=hashlib.sha256(payload).hexdigest()
        with gzip.open(cls.full,"wb",compresslevel=1) as gz,tarfile.open(fileobj=gz,mode="w|",format=tarfile.GNU_FORMAT) as tar:
            for i in range(200):
                name=A if i==188 else B if i==189 else f"{SEQUENCE}/json/frame_{i+100}.json"
                info=tarfile.TarInfo(name);info.size=len(payload);tar.addfile(info,io.BytesIO(payload))
        cls.selection={"schema_version":"drive_part_selection_v1","archive_id":"warm192_fixture","dataset_version":"mock_v1",
            "kind":"pose","total_parts":1,"parts":[{"index":0,"name":"pose.tar.gz000","file_id":"fake"}]}
        cls.manifest={"schema_version":"archive_manifest_v1","archive_id":"warm192_fixture","dataset_version":"mock_v1",
            "kind":"pose","source_scope":"user_drive","total_parts":1,"parts":[{"index":0,"name":"pose.tar.gz000",
                "version":"9","size_bytes":cls.full.stat().st_size,"source":{"type":"drive","file_id":"fake"},
                "identity":{"version":"9","modified_time":"2026-10-14T00:00:00.000Z"}}]}
        cls.bundle=cls.base/"v2.zip";chunks=[];h=hashlib.sha256();offset=0
        # 이전96MiB cache의 짧은 chunk 경계를 유지한 실제와 동일한513개 배치.
        ends=list(range(262144,100561588,262144))+[100561588]
        ends+=list(range(100561588+262144,C,262144))+[C]
        with cls.full.open("rb") as source,ZipFile(cls.bundle,"x",ZIP_STORED) as z:
            with z.open("prefix.bin","w") as dest:
                for end in ends:
                    data=source.read(end-offset);dest.write(data);h.update(data)
                    chunks.append({"start":offset,"end":end-1,"sha256":hashlib.sha256(data).hexdigest()});offset=end
            body={"schema_version":"bounded_prefix_cache_v2","policy_id":WARM128_POLICY.name,"archive_manifest":cls.manifest,
                "archive_manifest_sha256":digest(cls.manifest),"prefix_bytes":C,"prefix_sha256":h.hexdigest(),"chunks":chunks,
                "status":"usable_prefix","producer_code_sha256":code_hashes()}
            doc={**body,"content_sha256":digest(body)};z.writestr("manifest.json",canonical(doc)+b"\n")
        assert len(chunks)==513
        cls.spec=input_spec(cls.bundle,policy=WARM192_POLICY)
        cls.expected={"manifest_sha256":doc["content_sha256"],"prefix_sha256":h.hexdigest(),"prefix_bytes":C}

    @classmethod
    def tearDownClass(cls):cls.tmp.cleanup()

    def setUp(self):
        tmp=tempfile.TemporaryDirectory(dir=self.base);self.addCleanup(tmp.cleanup);self.root=Path(tmp.name)
        self.session=FakeDrive(self.selection,{"fake":b""});self.session.close=lambda:None
        self.session.docs["fake"]["size"]=str(self.full.stat().st_size)
        self.session.get_adapter=lambda _:SimpleNamespace(max_retries=SimpleNamespace(total=0))
        original=self.session.get
        def get(url,**kwargs):
            if "alt=media" not in url:return original(url,**kwargs)
            self.session.calls.append((url,kwargs));start,end=map(int,kwargs["headers"]["Range"].split("=")[1].split("-"))
            with self.full.open("rb") as f:f.seek(start);body=f.read(end-start+1)
            raw=Raw(body);return SimpleNamespace(status_code=206,raw=raw,close=raw.close,
                headers={"Content-Length":str(len(body)),"Content-Range":f"bytes {start}-{end}/{self.full.stat().st_size}"})
        self.session.get=get

    def run_prepare(self,targets=None,**kwargs):
        targets=targets or [A,B]
        return prepare(self.manifest,self.root/"pilot",targets,profile_limits(WARM192_PROFILE),
            profile=WARM192_PROFILE,exact_targets=targets,authorized_session=self.session,
            cache_input=self.spec,cache_expectation=self.expected,**kwargs)

    def test_01_suffix_target_success_starts_at128mib(self):
        e=self.run_prepare();self.assertTrue(e["targets_complete"])
        self.assertEqual({f["sha256"] for f in e["files"]},{self.payload_sha})
        first=next(row for row in e["range_audit"] if row["input_source"]=="drive")
        self.assertEqual(first["start"],C);self.assertLessEqual(e["stats"]["ranges"],770)
        self.assertEqual(e["prefix_cache"]["schema_version"],"bounded_prefix_cache_v3")

    def test_02_769chunks_192mib_export_reimport_and_old_reader_reject(self):
        missing=[f"{SEQUENCE}/json/frame_{i}.json" for i in (99998,99999)]
        e=self.run_prepare(missing)
        self.assertEqual(e["stop_reason"],"logical_input_limit");self.assertTrue(e["inspection_candidates_usable"])
        self.assertEqual((e["stats"]["media_requests"],e["stats"]["metadata_requests"]),(256,514))
        self.assertEqual((e["prefix_cache"]["prefix_bytes"],e["prefix_cache"]["chunks"]),(192*M,769))
        bundle,report=export_cache(self.root/"pilot/cache")
        self.assertGreater(report["local_read_bytes"],576*M);self.assertLess(report["local_read_bytes"],640*M)
        self.assertGreater(bundle.stat().st_size,192*M);self.assertLess(bundle.stat().st_size,224*M)
        with self.assertRaises(PreparationError):input_spec(bundle,policy=WARM128_POLICY)
        spec=input_spec(bundle,policy=WARM192_POLICY);stream,z=open_bundle(bundle,policy=WARM192_POLICY)
        try:doc=read_manifest(z,policy=WARM192_POLICY)
        finally:z.close();stream.close()
        expected={"manifest_sha256":doc["content_sha256"],"prefix_sha256":doc["prefix_sha256"],"prefix_bytes":doc["prefix_bytes"]}
        w=prepare(self.manifest,self.root/"again",missing,profile_limits(WARM192_PROFILE),profile=WARM192_PROFILE,
            exact_targets=missing,authorized_session=self.session,cache_input=spec,cache_expectation=expected)
        self.assertEqual((w["stats"]["media_requests"],w["stats"]["metadata_requests"]),(0,2))
        self.assertGreater(w["stats"]["local_io_bytes"],384*M);self.assertLess(w["stats"]["local_io_bytes"],388*M)
        self.assertEqual(w["prefix_cache"]["status"],"usable_prefix")
        self.assertTrue(save_result_zip(self.root/"again").exists())

    def test_03_required128_input_and_fixed_expectation(self):
        config={"job_id":"warm192","reason":"bounded fixture","mode":"extract","purpose":"targeted","profile":WARM192_PROFILE,
            "request_policy":METADATA_PRE_POLICY,"cache_input":self.spec,"cache_expectation":self.expected,"selection":self.selection,
            "includes":[A,B],"exact_targets":[A,B],"limits":asdict(profile_limits(WARM192_PROFILE)),
            "metadata_limits":asdict(MetadataLimits()),"previous_result":""}
        self.assertEqual(make_preflight(config,{})["summary"]["total_request_upper_bound"],781)
        for value in (None,{**self.expected,"prefix_bytes":C-1}):
            with self.assertRaises(PreparationError):make_preflight({**config,"cache_expectation":value},{})
        with self.assertRaises(PreparationError):make_preflight({**config,"cache_input":None},{})
        with patch.dict(self.expected,{"prefix_sha256":"0"*64}):e=self.run_prepare()
        self.assertEqual(e["stop_reason"],"cache_expectation_mismatch");self.assertEqual(self.session.calls,[])

    def test_04_source_change_and_post_failure_stay_unusable(self):
        self.session.docs["fake"]["version"]="10";e=self.run_prepare()
        self.assertEqual(e["stop_reason"],"source_changed");self.assertEqual(e["prefix_cache"]["status"],"unusable")
        self.assertEqual(e["stats"]["media_requests"],0)

    def test_05_partial_media_is_never_appended_or_salvaged(self):
        original=self.session.get
        class Partial(Raw):
            def read1(self,*a,**k):raise http.client.IncompleteRead(b"abc",100)
        def get(url,**kwargs):
            response=original(url,**kwargs)
            if "alt=media" in url:response.raw=Partial(b"")
            return response
        self.session.get=get;e=self.run_prepare()
        self.assertEqual(e["stop_reason"],"http_body_failed");self.assertTrue(e["source_validation_pending"])
        self.assertEqual(e["prefix_cache"]["prefix_bytes"],C);self.assertEqual(e["prefix_cache"]["status"],"unusable")

    def test_06_import_deadline_and_live_post_failure(self):
        calls=[0]
        def clock():calls[0]+=1;return 421. if calls[0]>=6 else 0.
        e=self.run_prepare(clock=clock);self.assertFalse(e["inspection_candidates_usable"])
        self.assertEqual(e["observation_catalog"]["trust"],"untrusted_failed_source_or_parser")
        original=self.session.get;count=[0]
        def get(url,**kwargs):
            count[0]+=1
            if count[0]==2:raise TimeoutError("SECRET")
            return original(url,**kwargs)
        self.session.get=get
        e=prepare(self.manifest,self.root/"post",[A,B],profile_limits(WARM192_PROFILE),profile=WARM192_PROFILE,
            exact_targets=[A,B],authorized_session=self.session,cache_input=self.spec,cache_expectation=self.expected)
        self.assertTrue(e["source_validation_pending"]);self.assertEqual(e["prefix_cache"]["status"],"unusable")

    def test_07_aggregate_output_budget_and_long_paths(self):
        lim=profile_limits(WARM192_PROFILE)
        reserve=report_reserve_bytes(self.manifest,[A,B],lim,[A,B],METADATA_PRE_POLICY,WARM192_POLICY)+CATALOG_BYTES
        self.assertLess(reserve+2*lim.max_file_bytes,lim.max_disk_bytes)
        long=["a/"+'"'*1010+str(i)+'.json' for i in (1,2)]
        worst=report_reserve_bytes(self.manifest,long,lim,long,METADATA_PRE_POLICY,WARM192_POLICY)+CATALOG_BYTES
        self.assertGreater(worst+2*lim.max_file_bytes,lim.max_disk_bytes)
        with patch("rewind_archive.extraction.report_reserve_bytes",return_value=lim.max_disk_bytes-CATALOG_BYTES-1):e=self.run_prepare()
        self.assertEqual(e["stop_reason"],"payload_disk_bytes_limit");self.assertFalse(e["targets_complete"])
        self.assertLess((self.root/"pilot/extraction_manifest.json").stat().st_size+e["observation_catalog"]["bytes"],lim.max_disk_bytes)

    def test_08_policy_caps_and_retry_headroom(self):
        self.assertEqual(WARM192_POLICY.audit_rows,WARM192_POLICY.max_chunks+1)
        self.assertEqual(3*256+2+3,profile_limits(WARM192_PROFILE).max_ranges)
        with self.assertRaises(PreparationError):resolve_policy(replace(WARM192_POLICY,logical_bytes=1024*M))
        self.assertEqual(validate_expectation(self.expected,policy=WARM192_POLICY)["prefix_bytes"],C)
        self.assertEqual(profile_limits(WARM128_PROFILE).max_ranges,392)


if __name__=="__main__":unittest.main()
