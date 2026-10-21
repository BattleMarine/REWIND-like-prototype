"""실데이터와 분리된 완전 gzip에서 v1→v2 warm 확장·128MiB·513chunk 경계를 검증한다."""
import copy
import gzip
import hashlib
import json
import random
import tempfile
import unittest
from dataclasses import asdict,replace
from pathlib import Path
from types import SimpleNamespace
from zipfile import ZipFile

from rewind_archive.contracts import Budget,PreparationError,canonical,digest
from rewind_archive.profiles import CACHE_PROFILE,WARM128_PROFILE,profile_limits
from rewind_archive.cache_policy import WARM128_POLICY,validate_expectation
from rewind_archive.prefix_cache import PrefixCache,input_spec,export_cache,resolve_policy,read_manifest,open_bundle
from rewind_archive.extraction import prepare,code_hashes,report_reserve_bytes
from rewind_archive.catalog import Catalog,CATALOG_BYTES
from rewind_archive.fixture import SEQUENCE,tar_bytes
from rewind_archive.workflow import make_preflight,review_run,save_result_zip
from rewind_archive.drive_metadata import MetadataLimits
from rewind_archive.request_policy import METADATA_PRE_POLICY
from test_drive_metadata import FakeDrive

A=f"{SEQUENCE}/json/frame_1.json";B=f"{SEQUENCE}/json/frame_10.json"
C=100561588;M=1024**2


class Warm128Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        parent=Path(__file__).resolve().parents[1]/"outputs/warm128_test_tmp";parent.mkdir(parents=True,exist_ok=True)
        cls.tmp=tempfile.TemporaryDirectory(dir=parent);cls.base=Path(cls.tmp.name)
        payload=random.Random(1941).randbytes(M)
        items=[(f"{SEQUENCE}/json/frame_{i+100}.json",payload) for i in range(134)]
        items[125]=(A,payload);items[126]=(B,payload)
        cls.data=gzip.compress(tar_bytes(items),compresslevel=1,mtime=0)
        cls.payload_sha=hashlib.sha256(payload).hexdigest()
        cls.selection={"schema_version":"drive_part_selection_v1","archive_id":"warm128_fixture","dataset_version":"mock_v1",
            "kind":"pose","total_parts":1,"parts":[{"index":0,"name":"pose.tar.gz000","file_id":"fake"}]}
        cls.manifest={"schema_version":"archive_manifest_v1","archive_id":"warm128_fixture","dataset_version":"mock_v1",
            "kind":"pose","source_scope":"user_drive","total_parts":1,"parts":[{"index":0,"name":"pose.tar.gz000",
                "version":"9","size_bytes":len(cls.data),"source":{"type":"drive","file_id":"fake"},
                "identity":{"version":"9","modified_time":"2026-10-14T00:00:00.000Z"}}]}
        cache=PrefixCache(cls.base,cls.manifest,Budget(profile_limits(CACHE_PROFILE)));cache.load(None)
        for start in range(0,C,262144):cache.append(start,cls.data[start:min(start+262144,C)])
        cache.finish(True,code_hashes());bundle,_=export_cache(cls.base/"cache")
        cls.spec=input_spec(bundle);doc=json.loads((cls.base/"cache/manifest.json").read_text())
        cls.expected={"manifest_sha256":doc["content_sha256"],"prefix_bytes":C,"prefix_sha256":doc["prefix_sha256"]}

    @classmethod
    def tearDownClass(cls):cls.tmp.cleanup()

    def setUp(self):
        self.tmp_case=tempfile.TemporaryDirectory(dir=self.base);self.addCleanup(self.tmp_case.cleanup)
        self.root=Path(self.tmp_case.name)
        self.session=FakeDrive(self.selection,{"fake":self.data});self.session.close=lambda:None
        self.session.get_adapter=lambda _:SimpleNamespace(max_retries=SimpleNamespace(total=0))

    def run_prepare(self,targets=None,spec=None,expected=None,**kwargs):
        return prepare(self.manifest,self.root/"pilot",targets or [A,B],profile_limits(WARM128_PROFILE),
            profile=WARM128_PROFILE,exact_targets=targets or [A,B],authorized_session=self.session,
            cache_input=spec or self.spec,cache_expectation=expected or self.expected,**kwargs)

    def test_01_warm_suffix_targets_and_exact_first_offset(self):
        e=self.run_prepare()
        self.assertTrue(e["targets_complete"]);self.assertFalse(e["source_validation_pending"])
        self.assertEqual({f["sha256"] for f in e["files"]},{self.payload_sha})
        first=next(k["headers"]["Range"] for url,k in self.session.calls if "alt=media" in url)
        self.assertTrue(first.startswith(f"bytes={C}-"))
        self.assertLessEqual(e["stats"]["ranges"],389);self.assertEqual(e["prefix_cache"]["schema_version"],"bounded_prefix_cache_v2")
        self.assertGreater(e["stats"]["cache_import_seconds"],0)
        self.assertGreater(e["stats"]["cache_read_seconds"],0)
        self.assertGreater(e["stats"]["remote_range_seconds"],0)
        self.assertGreaterEqual(e["stats"]["cache_live_check_seconds"],0)  # 짧은 fixture는 clock 해상도 이하일 수 있다.
        self.assertIn("gzip",e["timing_scope"]["unsegmented"])

    def test_02_logical128_513chunks_roundtrip_and_old_reader_rejection(self):
        missing=[f"{SEQUENCE}/json/frame_{i}.json" for i in (99997,99998)]
        e=self.run_prepare(missing)
        self.assertEqual(e["stop_reason"],"logical_input_limit");self.assertTrue(e["inspection_candidates_usable"])
        self.assertEqual(e["prefix_cache"]["prefix_bytes"],128*M);self.assertEqual(e["prefix_cache"]["chunks"],513)
        self.assertEqual((e["stats"]["media_requests"],e["stats"]["metadata_requests"]),(129,260))
        self.assertEqual(len(e["range_audit"]),513);self.assertEqual(e["stats"]["logical_compressed_bytes"],128*M)
        bundle,export=export_cache(self.root/"pilot/cache")
        self.assertGreater(export["zip_bytes"],128*M);self.assertLess(export["zip_bytes"],160*M)
        self.assertGreater(export["local_read_bytes"],384*M);self.assertLess(export["local_read_bytes"],512*M)
        with self.assertRaises(PreparationError):input_spec(bundle)
        spec=input_spec(bundle,policy=WARM128_POLICY)
        stream,z=open_bundle(bundle,policy=WARM128_POLICY)
        try:doc=read_manifest(z,policy=WARM128_POLICY)
        finally:z.close();stream.close()
        expected={"manifest_sha256":doc["content_sha256"],"prefix_sha256":doc["prefix_sha256"],"prefix_bytes":doc["prefix_bytes"]}
        self.session.calls=[]
        warm=prepare(self.manifest,self.root/"again",missing,profile_limits(WARM128_PROFILE),profile=WARM128_PROFILE,
            exact_targets=missing,authorized_session=self.session,cache_input=spec,cache_expectation=expected)
        self.assertEqual(warm["stats"]["media_requests"],0);self.assertEqual(warm["stats"]["metadata_requests"],2)
        self.assertGreater(warm["stats"]["local_io_bytes"],256*M);self.assertLess(warm["stats"]["local_io_bytes"],260*M)
        self.assertEqual(warm["prefix_cache"]["status"],"usable_prefix")
        # 단일 카탈로그를 포함한 결과ZIP에는 큰 cache 본체를 넣지 않는다.
        self.assertTrue(save_result_zip(self.root/"again").is_file())

    def test_03_expectation_mismatch_has_no_media_or_live_request(self):
        wrong={**self.expected,"prefix_sha256":"0"*64}
        e=self.run_prepare(expected=wrong)
        self.assertEqual(e["stop_reason"],"cache_expectation_mismatch")
        self.assertFalse(e["inspection_candidates_usable"]);self.assertEqual(self.session.calls,[])
        self.assertFalse((self.root/"pilot/cache/prefix.partial").exists())

    def test_04_live_source_change_and_cached_post_failure_are_unusable(self):
        self.session.docs["fake"]["version"]="10"
        e=self.run_prepare();self.assertEqual(e["stop_reason"],"source_changed")
        self.assertEqual(e["prefix_cache"]["status"],"unusable")
        self.assertEqual(e["stats"]["media_requests"],0)

    def test_05_no_cold_or_tiny_cache_fallback_preflight_and_fingerprints(self):
        config={"job_id":"warm_fixed","reason":"bounded warm fixture","mode":"extract","purpose":"targeted",
            "profile":WARM128_PROFILE,"request_policy":METADATA_PRE_POLICY,"cache_input":self.spec,
            "cache_expectation":self.expected,"selection":self.selection,"includes":[A,B],"exact_targets":[A,B],
            "limits":asdict(profile_limits(WARM128_PROFILE)),"metadata_limits":asdict(MetadataLimits()),"previous_result":""}
        good=make_preflight(config,{})
        self.assertEqual(good["summary"]["total_request_upper_bound"],400)
        for change in ({"cache_input":None},{"cache_expectation":None},
                {"cache_expectation":{**self.expected,"prefix_bytes":WARM128_POLICY.min_input_bytes-1}}):
            with self.assertRaises(PreparationError):make_preflight({**config,**change},{})
        altered=make_preflight({**config,"cache_expectation":{**self.expected,"prefix_sha256":"0"*64}},{})
        self.assertNotEqual(good["effective_sha256"],altered["effective_sha256"])
        with self.assertRaises(PreparationError):prepare(self.manifest,self.root/"no_cache",[A,B],profile_limits(WARM128_PROFILE),
            profile=WARM128_PROFILE,exact_targets=[A,B],cache_expectation=self.expected,authorized_session=self.session)

    def test_06_policy_cap_forgery_and_catalog256_report_reserve(self):
        old=resolve_policy()
        with self.assertRaises(PreparationError):resolve_policy(replace(old,disk_bytes=100000*M))
        catalog=Catalog(max_sequences=256,policy=WARM128_POLICY)
        for i in range(16000):
            catalog.add({"member":f"{'s'*900}{i%256}/json/frame_{i}.json","header_ordinal":20000,"regular_ordinal":16000,
                "size_bytes":10**20-1,"payload_complete":False,"payload_bytes_returned":10**20-1,"stored":False})
        self.assertEqual(len(catalog.sequences),256);self.assertLess(len(canonical(catalog.document(False))),CATALOG_BYTES)
        limits=profile_limits(WARM128_PROFILE)
        reserve=report_reserve_bytes(self.manifest,[A,B],limits,[A,B],METADATA_PRE_POLICY,WARM128_POLICY)
        self.assertLess(reserve+CATALOG_BYTES+2*limits.max_file_bytes,limits.max_disk_bytes)
        self.assertGreater(WARM128_POLICY.audit_rows,limits.max_ranges)

    def test_07_post_timeout_stays_pending_and_not_reusable(self):
        original=self.session.get;count=[0]
        def get(url,**kwargs):
            count[0]+=1
            if count[0]==2:raise TimeoutError("SECRET")
            return original(url,**kwargs)
        self.session.get=get;e=self.run_prepare()
        self.assertTrue(e["source_validation_pending"]);self.assertFalse(e["inspection_candidates_usable"])
        self.assertEqual(e["prefix_cache"]["status"],"unusable")
        self.assertEqual(e["stats"]["media_requests"],0)

    def test_08_import_interruption_keeps_catalog_untrusted(self):
        calls=[0]
        def clock():
            calls[0]+=1
            return 421. if calls[0]>=6 else 0.
        e=self.run_prepare(clock=clock)
        self.assertFalse(e["inspection_candidates_usable"])
        self.assertEqual(e["prefix_cache"]["status"],"unusable")
        self.assertEqual(e["observation_catalog"]["trust"],"untrusted_failed_source_or_parser")


if __name__=="__main__":unittest.main()
