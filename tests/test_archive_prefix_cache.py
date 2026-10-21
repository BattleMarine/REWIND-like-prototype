"""네트워크 없는 cold/export/import/warm/후속수신과 cache/catalog 경계 검증."""

import copy
import gzip
import hashlib
import http.client
import json
import random
import struct
import stat
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from zipfile import ZipFile,ZIP_STORED,ZIP_DEFLATED,ZipInfo

from rewind_archive.contracts import Budget,PreparationError,canonical,digest
from rewind_archive.catalog import Catalog,CATALOG_BYTES
from rewind_archive.extraction import prepare,code_hashes,report_reserve_bytes
from rewind_archive.fixture import SEQUENCE,tar_bytes
from rewind_archive.profiles import CACHE_PROFILE,profile_limits
from rewind_archive.prefix_cache import (PrefixCache,input_spec,export_cache,open_bundle,MANIFEST_BYTES,
    LOCAL_IO_BYTES,EXPORT_IO_BYTES)
from rewind_archive.workflow import make_preflight,run_workflow_job,review_run,save_result_zip
from rewind_archive.drive_metadata import MetadataLimits
from test_drive_metadata import FakeDrive,Raw

A=f"{SEQUENCE}/json/frame_1.json";B=f"{SEQUENCE}/json/frame_10.json"


class CacheTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rng=random.Random(735)
        cls.data=gzip.compress(tar_bytes([(f"{SEQUENCE}/json/frame_{i}.json",rng.randbytes(8192)) for i in range(100,300)]
            +[(A,b'{"target":1}'),(B,b'{"target":10}')]),mtime=0)

    def setUp(self):
        base=Path(__file__).resolve().parents[1]/"outputs/prefix_cache_test_tmp";base.mkdir(parents=True,exist_ok=True)
        tmp=tempfile.TemporaryDirectory(dir=base);self.addCleanup(tmp.cleanup);self.root=Path(tmp.name)
        self.selection={"schema_version":"drive_part_selection_v1","archive_id":"cache_fixture","dataset_version":"mock_v1",
            "kind":"pose","total_parts":1,"parts":[{"index":0,"name":"pose.tar.gz000","file_id":"fake"}]}
        self.manifest={"schema_version":"archive_manifest_v1","archive_id":"cache_fixture","dataset_version":"mock_v1",
            "kind":"pose","source_scope":"user_drive","total_parts":1,"parts":[{"index":0,"name":"pose.tar.gz000",
            "version":"9","source":{"type":"drive","file_id":"fake"},"size_bytes":len(self.data),
            "identity":{"version":"9","modified_time":"2026-10-14T00:00:00.000Z"}}]}
        self.session=self.make_session();self.number=0

    def make_session(self):
        session=FakeDrive(self.selection,{"fake":self.data});session.close=lambda:None
        session.get_adapter=lambda _:SimpleNamespace(max_retries=SimpleNamespace(total=0))
        return session

    def run_prepare(self,spec=None,**kwargs):
        self.number+=1
        out=self.root/str(self.number)
        report=prepare(self.manifest,out,[A,B],profile_limits(CACHE_PROFILE),authorized_session=self.session,
            profile=CACHE_PROFILE,exact_targets=[A,B],cache_input=spec,**kwargs)
        return out,report

    def cold_prefix(self):
        with patch("rewind_archive.prefix_cache.LOGICAL_BYTES",512*1024):out,report=self.run_prepare()
        self.assertEqual(report["stop_reason"],"logical_input_limit")
        self.assertEqual(report["prefix_cache"]["prefix_bytes"],512*1024)
        bundle,_=export_cache(out/"cache")
        return out,report,input_spec(bundle)

    def test_cold_warm_full_output_and_catalog_identical(self):
        cold,e=self.run_prepare();self.assertTrue(e["targets_complete"])
        self.assertEqual(e["prefix_cache"]["status"],"usable_prefix")
        self.assertEqual(e["stats"]["local_io_bytes"],0)
        self.assertEqual(e["stats"]["cache_write_bytes"],e["prefix_cache"]["prefix_bytes"])
        bundle,export=export_cache(cold/"cache");self.assertFalse(export["network_requested"])
        self.session=self.make_session();warm,w=self.run_prepare(input_spec(bundle))
        self.assertTrue(w["targets_complete"]);self.assertEqual(w["stats"]["media_requests"],0)
        self.assertEqual(w["stats"]["metadata_requests"],2)
        self.assertEqual(w["stats"]["logical_compressed_bytes"],e["stats"]["logical_compressed_bytes"])
        self.assertEqual([(f["member"],f["sha256"]) for f in e["files"]],[(f["member"],f["sha256"]) for f in w["files"]])
        self.assertEqual((cold/"observation_catalog.json").read_bytes(),(warm/"observation_catalog.json").read_bytes())
        self.assertTrue(w["stats"]["local_io_bytes"]>2*w["prefix_cache"]["prefix_bytes"])
        self.assertFalse(w["source_validation_pending"])

    def test_partial_prefix_replay_then_only_suffix_network(self):
        cold,e,spec=self.cold_prefix();self.session=self.make_session()
        warm,w=self.run_prepare(spec)
        self.assertTrue(w["targets_complete"])
        media=[k["headers"]["Range"] for url,k in self.session.calls if "alt=media" in url]
        self.assertTrue(media[0].startswith("bytes=524288-"),media)
        self.assertEqual(w["prefix_cache"]["input_prefix_bytes"],524288)
        self.assertEqual(w["stats"]["logical_compressed_bytes"],w["prefix_cache"]["prefix_bytes"])
        self.assertEqual(w["prefix_cache"]["prefix_bytes"],sum(r["bytes_received"] for r in w["range_audit"]))
        self.assertLess(w["stats"]["transfer_bytes"],w["stats"]["logical_compressed_bytes"])

    def test_same_bounded_prefix_warm_has_media_zero(self):
        _,e,spec=self.cold_prefix();self.session=self.make_session()
        with patch("rewind_archive.prefix_cache.LOGICAL_BYTES",524288):_,w=self.run_prepare(spec)
        self.assertEqual(w["stats"]["media_requests"],0);self.assertEqual(w["stats"]["metadata_requests"],2)
        self.assertEqual(w["stop_reason"],"logical_input_limit");self.assertTrue(w["inspection_candidates_usable"])
        self.assertFalse(w["targets_complete"])

    def test_early_target_stop_still_postchecks_warm_source(self):
        # 완전cache를 갖고 선택 경로를 앞부분으로 바꾸면 early stop에도 post 검사가 필요하다.
        cold,e=self.run_prepare();bundle,_=export_cache(cold/"cache");self.session=self.make_session()
        targets=[f"{SEQUENCE}/json/frame_{i}.json" for i in (100,101)]
        out=self.root/"early"
        w=prepare(self.manifest,out,targets,profile_limits(CACHE_PROFILE),profile=CACHE_PROFILE,
            exact_targets=targets,authorized_session=self.session,cache_input=input_spec(bundle))
        self.assertTrue(w["targets_complete"]);self.assertEqual(w["stats"]["media_requests"],0)
        self.assertEqual(w["stats"]["metadata_requests"],2);self.assertFalse(w["source_validation_pending"])

    def test_live_pre_source_change_blocks_cached_payload(self):
        _,_,spec=self.cold_prefix();self.session=self.make_session();self.session.docs["fake"]["version"]="10"
        _,w=self.run_prepare(spec)
        self.assertEqual(w["stop_reason"],"source_changed");self.assertFalse(w["inspection_candidates_usable"])
        self.assertEqual(w["stats"]["media_requests"],0);self.assertEqual(w["prefix_cache"]["status"],"unusable")

    def test_live_post_failure_keeps_pending_unusable_and_no_bypass(self):
        _,_,spec=self.cold_prefix();self.session=self.make_session();original=self.session.get;calls=[0]
        def get(url,**kwargs):
            calls[0]+=1
            if calls[0]==2:raise TimeoutError("SECRET")
            return original(url,**kwargs)
        self.session.get=get
        out,w=self.run_prepare(spec)
        self.assertTrue(w["source_validation_pending"]);self.assertFalse(w["inspection_candidates_usable"])
        self.assertEqual(w["prefix_cache"]["status"],"unusable")
        bundle,_=export_cache(out/"cache");self.session=self.make_session();_,again=self.run_prepare(input_spec(bundle))
        self.assertEqual(again["stop_reason"],"cache_unusable_input");self.assertEqual(self.session.calls,[])

    def rewrite_bundle(self,spec,mutate=None,files=None,compression=ZIP_STORED):
        with ZipFile(spec["path"]) as z:values={n:z.read(n) for n in z.namelist()}
        if mutate:
            doc=json.loads(values["manifest.json"]);mutate(doc)
            doc["content_sha256"]=digest({k:v for k,v in doc.items() if k!="content_sha256"});values["manifest.json"]=canonical(doc)
        if files:values.update(files)
        self.number+=1;path=self.root/(f"altered_{self.number}.zip")
        with ZipFile(path,"x",compression) as z:
            for n,v in values.items():z.writestr(n,v)
        return path

    def test_input_identity_gap_chunk_hash_and_full_hash_fail_atomically(self):
        _,_,spec=self.cold_prefix()
        changes=[lambda d:d["archive_manifest"]["parts"][0]["identity"].update(version="100"),
            lambda d:d["chunks"][0].update(start=1),lambda d:d["chunks"][0].update(sha256="0"*64),
            lambda d:d.update(prefix_sha256="0"*64)]
        for change in changes:
            path=self.rewrite_bundle(spec,mutate=change)
            altered=input_spec(path) if change not in changes[:2] else {"path":str(path),"manifest_sha256":json.loads(ZipFile(path).read("manifest.json"))["content_sha256"]}
            out,e=self.run_prepare(altered)
            self.assertFalse(e["inspection_candidates_usable"]);self.assertEqual(e["prefix_cache"]["status"],"unusable")
            self.assertFalse((out/"cache/prefix.partial").exists());self.assertEqual((out/"cache/prefix.bin").stat().st_size,0)

    def test_payload_tamper_detected_without_network(self):
        _,_,spec=self.cold_prefix()
        path=self.rewrite_bundle(spec,files={"prefix.bin":b"x"*524288})
        self.session=self.make_session();_,e=self.run_prepare(input_spec(path))
        self.assertEqual(e["stop_reason"],"cache_chunk_hash_mismatch");self.assertEqual(self.session.calls,[])

    def test_bad_zip_members_compression_oversize_manifest_and_central_directory(self):
        _,_,spec=self.cold_prefix()
        paths=[self.rewrite_bundle(spec,files={"../outside":b"x"}),self.rewrite_bundle(spec,compression=ZIP_DEFLATED),
            self.rewrite_bundle(spec,files={"manifest.json":b"x"*(MANIFEST_BYTES+1)})]
        for path in paths:
            with self.assertRaises(PreparationError):input_spec(path)
        raw=bytearray(Path(spec["path"]).read_bytes());raw[-12:-10]=struct.pack("<H",65535)
        bad=self.root/"central.zip";bad.write_bytes(raw)
        with patch("rewind_archive.prefix_cache.ZipFile",side_effect=AssertionError("must reject before parsing")):
            with self.assertRaises(PreparationError):open_bundle(bad)

    def test_local_io_and_deadline_limits_during_import(self):
        _,_,spec=self.cold_prefix();budget=Budget(profile_limits(CACHE_PROFILE));budget.local_io_bytes=LOCAL_IO_BYTES
        out=self.root/"io";out.mkdir();cache=PrefixCache(out,self.manifest,budget)
        with self.assertRaises(PreparationError) as e:cache.load(spec)
        self.assertEqual(e.exception.reason,"cache_local_io_limit");self.assertFalse((out/"cache/prefix.partial").exists())
        budget=Budget(profile_limits(CACHE_PROFILE));budget.clock=lambda:budget.started+421
        out=self.root/"time";out.mkdir();cache=PrefixCache(out,self.manifest,budget)
        with self.assertRaises(PreparationError) as e:cache.load(spec)
        self.assertEqual(e.exception.reason,"elapsed_limit")

    def test_network_request_reservation_before_new_media(self):
        # 실제 preset은 유지하고 clock/요청 상태만 fixture에서 경계로 만든다.
        original=Budget.request
        def request(budget,kind,maximum=None):
            value=original(budget,kind,maximum)
            if kind=="metadata":budget.ranges=budget.limits.max_ranges
            return value
        with patch.object(Budget,"request",request):_,e=self.run_prepare()
        self.assertEqual(e["prefix_cache"]["status"],"unusable")
        self.assertTrue(e["source_validation_pending"]);self.assertFalse(e["inspection_candidates_usable"])

    def test_import_deadline_or_interrupt_never_promotes_empty_cache(self):
        _,_,spec=self.cold_prefix()
        for interrupt in (False,True):
            calls=[0]
            def clock():
                calls[0]+=1
                if calls[0]>=6:
                    if interrupt:raise KeyboardInterrupt()
                    return 421.
                return 0.
            if interrupt:
                # 중단을 한 번만 주입해 보고서의 elapsed 측정은 계속 가능하게 한다.
                def clock():
                    calls[0]+=1
                    if calls[0]==6:raise KeyboardInterrupt()
                    return 0.
            _,e=self.run_prepare(spec,clock=clock)
            self.assertFalse(e["inspection_candidates_usable"])
            self.assertEqual(e["prefix_cache"]["status"],"unusable")
            self.assertEqual(e["observation_catalog"]["trust"],"untrusted_failed_source_or_parser")

    def test_warm_before_live_precheck_deadline_is_unusable(self):
        _,_,spec=self.cold_prefix()
        original=PrefixCache.load
        def loaded(cache,value):
            original(cache,value)
            cache.budget.clock=lambda:cache.budget.started+421
        with patch.object(PrefixCache,"load",loaded):_,report=self.run_prepare(spec)
        self.assertEqual(report["stop_reason"],"elapsed_limit");self.assertFalse(report["inspection_candidates_usable"])
        self.assertEqual(report["prefix_cache"]["status"],"unusable")

    def test_catalog_cap_and_failure_trust_without_payload_duplication(self):
        catalog=Catalog(max_entries=1,max_sequences=1)
        obs={"member":A,"header_ordinal":11,"regular_ordinal":1,"size_bytes":55,"payload_complete":False,"payload_bytes_returned":2,"stored":False}
        catalog.add(obs);catalog.add({**obs,"member":B});obs.update(payload_complete=True,stored=True)
        doc=catalog.document(False)
        self.assertTrue(doc["truncated"]);self.assertEqual(len(doc["entries"]),1)
        self.assertTrue(doc["entries"][0]["payload_complete"]);self.assertEqual(doc["entries"][0]["frame"],1)
        self.assertEqual(doc["entries"][0]["trust"],"untrusted_failed_source_or_parser")
        self.assertFalse(doc["seek_index"]);self.assertEqual(doc["sequences"],[A.split("/json/")[0]])

    def test_partial_media_never_appended_and_first_body_error_preserved(self):
        original=self.session.get;media=[0]
        class Broken(Raw):
            def read1(self,*a,**k):raise http.client.IncompleteRead(b"abc",100)
        def get(url,**kwargs):
            response=original(url,**kwargs)
            if "alt=media" in url:
                media[0]+=1
                if media[0]==2:
                    response.raw=Broken(b"")
                    response.close=lambda:(_ for _ in ()).throw(RuntimeError("SECRET"))
            return response
        self.session.get=get;out,e=self.run_prepare()
        self.assertEqual(e["stop_reason"],"http_body_failed")
        self.assertEqual(e["request_failure"]["category"],"incomplete_body")
        self.assertEqual(e["prefix_cache"]["prefix_bytes"],262144)
        self.assertEqual(e["range_audit"][-1]["bytes_received"],3)
        self.assertEqual(e["prefix_cache"]["status"],"unusable")
        self.assertNotIn("SECRET",(out/"extraction_manifest.json").read_text())

    def test_media_elapsed_failure_is_not_salvaged_by_cache_postcheck(self):
        original=self.session.get
        class Elapsed(Raw):
            def read1(self,*a,**k):raise PreparationError("elapsed_limit")
        def get(url,**kwargs):
            response=original(url,**kwargs)
            if "alt=media" in url:response.raw=Elapsed(b"")
            return response
        self.session.get=get;_,e=self.run_prepare()
        self.assertTrue(e["source_validation_pending"]);self.assertFalse(e["inspection_candidates_usable"])
        self.assertEqual(e["prefix_cache"]["status"],"unusable")
        self.assertEqual(e["stats"]["metadata_requests"],1)

    def test_parser_and_gzip_crc_failure_mark_cache_and_catalog_unusable(self):
        for invalid in (gzip.compress(b"x"*1024),self.data[:-8]+bytes([self.data[-8]^255])+self.data[-7:]):
            self.data=invalid;self.manifest["parts"][0]["size_bytes"]=len(invalid);self.session=self.make_session()
            targets=[f"{SEQUENCE}/json/frame_{i}.json" for i in (99999,99998)]
            self.number+=1
            e=prepare(self.manifest,self.root/str(self.number),targets,profile_limits(CACHE_PROFILE),profile=CACHE_PROFILE,
                exact_targets=targets,authorized_session=self.session)
            self.assertIn(e["stop_reason"],("invalid_tar_header","gzip_corrupt"))
            self.assertEqual(e["prefix_cache"]["status"],"unusable")
            self.assertEqual(e["observation_catalog"]["trust"],"untrusted_failed_source_or_parser")

    def test_catalog_worst_rows_fit_reserved_output_disk(self):
        catalog=Catalog()
        for i in range(16000):
            obs={"member":f"{'s'*900}{i%128}/json/frame_{i}.json","header_ordinal":16000,"regular_ordinal":16000,
                "size_bytes":10**20-1,"payload_complete":False,"payload_bytes_returned":10**20-1,"stored":False}
            catalog.add(obs)
        raw=canonical(catalog.document(False));self.assertLess(len(raw),CATALOG_BYTES)
        limits=profile_limits(CACHE_PROFILE)
        self.assertLess(report_reserve_bytes(self.manifest,[A,B],limits,[A,B])+CATALOG_BYTES+2*limits.max_file_bytes,limits.max_disk_bytes)
        catalog.add({**obs,"member":"other/json/frame_1.json"});catalog.add({**obs,"member":"unclassified.txt"})
        self.assertTrue(catalog.truncated);self.assertEqual(catalog.unclassified,1)

    def test_symlink_duplicate_and_traversal_zip_guards(self):
        _,_,spec=self.cold_prefix()
        with ZipFile(spec["path"]) as source:manifest=source.read("manifest.json");prefix=source.read("prefix.bin")
        for case in ("duplicate","traversal","symlink"):
            p=self.root/(case+".zip")
            with ZipFile(p,"x",ZIP_STORED) as z:
                z.writestr("manifest.json",manifest)
                if case=="symlink":
                    info=ZipInfo("prefix.bin");info.external_attr=(stat.S_IFLNK|0o777)<<16;z.writestr(info,prefix)
                else:z.writestr("manifest.json" if case=="duplicate" else "../prefix.bin",prefix)
            with self.assertRaises(PreparationError):input_spec(p)
        with patch("pathlib.Path.is_symlink",return_value=True):
            with self.assertRaises(PreparationError):export_cache(self.root/"1/cache")

    def test_export_failure_cleans_partial_and_result_zip_remains_available(self):
        config={"job_id":"export_failure","reason":"export cleanup fixture","mode":"extract","purpose":"targeted","profile":CACHE_PROFILE,
            "request_policy":None,"cache_input":None,"selection":self.selection,"includes":[A,B],"exact_targets":[A,B],
            "limits":asdict(profile_limits(CACHE_PROFILE)),"metadata_limits":asdict(MetadataLimits()),"previous_result":""}
        root,record=run_workflow_job(config,make_preflight(config,{}),{},self.root,lambda:self.session)
        with patch("rewind_archive.prefix_cache.EXPORT_IO_BYTES",1):
            with self.assertRaises(PreparationError):export_cache(root/"pilot/cache")
        self.assertFalse((root/"pilot/prefix_cache.zip.partial").exists())
        self.assertTrue(save_result_zip(root).is_file())

    def test_workflow_preflight_report_refs_review_and_result_zip_separate_cache(self):
        config={"job_id":"cache_cold","reason":"cache fixture","mode":"extract","purpose":"targeted","profile":CACHE_PROFILE,
            "request_policy":None,"cache_input":None,"selection":self.selection,"includes":[A,B],"exact_targets":[A,B],
            "limits":asdict(profile_limits(CACHE_PROFILE)),"metadata_limits":asdict(MetadataLimits()),"previous_result":""}
        pre=make_preflight(config,{})
        self.assertTrue(pre["summary"]["prefix_cache_enabled"]);self.assertEqual(pre["summary"]["total_request_upper_bound"],1165)
        root,record=run_workflow_job(config,pre,{},self.root,lambda:self.session)
        self.assertTrue(record["targets_complete"]);self.assertIn("observation_catalog",record)
        export_cache(root/"pilot/cache");result=save_result_zip(root)
        with ZipFile(result) as z:
            self.assertTrue(any(n.endswith("observation_catalog.json") for n in z.namelist()))
            self.assertFalse(any(n.endswith("prefix.bin") or n.endswith("prefix_cache.zip") for n in z.namelist()))
        reviewed=review_run(root);self.assertEqual(reviewed["extraction"]["observation_catalog"]["entries"],202)
        self.assertLess((root/"job.json").stat().st_size,256*1024)

    def test_near_96mib_export_import_and_replay_stays_inside_separate_limits(self):
        out=self.root/"large";out.mkdir();budget=Budget(profile_limits(CACHE_PROFILE))
        manifest=copy.deepcopy(self.manifest);manifest["parts"][0]["size_bytes"]=100*1024**2
        cache=PrefixCache(out,manifest,budget);cache.load(None);chunk=b"bounded-prefix-fixture".ljust(262144,b"x")
        for i in range(384):cache.append(i*262144,chunk)
        cache.finish(True,code_hashes());bundle,report=export_cache(out/"cache")
        self.assertGreater(report["local_read_bytes"],288*1024**2)
        self.assertLess(report["local_read_bytes"],EXPORT_IO_BYTES)
        warm=self.root/"large_warm";warm.mkdir();other=Budget(profile_limits(CACHE_PROFILE));loaded=PrefixCache(warm,manifest,other)
        loaded.load(input_spec(bundle))
        for row in loaded.chunks:self.assertEqual(loaded.read(row["start"],row["end"]),chunk)
        self.assertLess(other.local_io_bytes,LOCAL_IO_BYTES);loaded.finish(True,code_hashes())
        again,again_report=export_cache(out/"cache")
        self.assertEqual(again,bundle);self.assertTrue(again_report["existing_export_reused"])


if __name__=="__main__":unittest.main()
