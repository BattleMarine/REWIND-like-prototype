"""원본 요청 없이 v3 192MiB 재생과 짧은 합성 suffix의 v4 선택/EOF/실패 검사."""
import gzip
import hashlib
import io
import json
import random
import tarfile
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from zipfile import ZipFile, ZIP_STORED

from rewind_archive.cache_policy import TAIL352_POLICY, WARM192_POLICY
from rewind_archive.contracts import canonical, digest, PreparationError
from rewind_archive.drive_metadata import MetadataLimits
from rewind_archive.extraction import prepare, code_hashes, report_reserve_bytes
from rewind_archive.catalog import Catalog
from rewind_archive.fixture import SEQUENCE
from rewind_archive.prefix_cache import input_spec, export_cache
from rewind_archive.profiles import TAIL352_PROFILE, WARM192_PROFILE, profile_limits
from rewind_archive.workflow import make_preflight, review_run, save_result_zip, NOTEBOOK_API_VERSION
from rewind_archive.request_policy import METADATA_PRE_POLICY
from test_drive_metadata import FakeDrive, Raw

M=1024**2
C=192*M
A=f"{SEQUENCE}/json/frame_1.json"
B=f"{SEQUENCE}/json/frame_10.json"


class Tail352Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        parent=Path(__file__).resolve().parents[1]/'outputs/tail352_test_tmp'
        parent.mkdir(parents=True,exist_ok=True)
        cls.tmp=tempfile.TemporaryDirectory(dir=parent);cls.base=Path(cls.tmp.name)
        cls.full=cls.base/'fixture.gz'
        block=random.Random(35229).randbytes(M)
        with gzip.open(cls.full,'wb',compresslevel=1) as gz, tarfile.open(fileobj=gz,mode='w|') as tar:
            for i in range(193):
                info=tarfile.TarInfo(f'{SEQUENCE}/json/frame_{i+100}.json');info.size=len(block)
                tar.addfile(info,io.BytesIO(block))
            for name,body in [(A,b'{"fixture":1}'),(B,b'{"fixture":10}')]:
                info=tarfile.TarInfo(name);info.size=len(body);tar.addfile(info,io.BytesIO(body))
        cls.selection={'schema_version':'drive_part_selection_v1','archive_id':'tail_fixture',
            'dataset_version':'mock_v1','kind':'pose','total_parts':1,
            'parts':[{'index':0,'name':'pose.tar.gz000','file_id':'fake'}]}
        cls.manifest={'schema_version':'archive_manifest_v1','archive_id':'tail_fixture',
            'dataset_version':'mock_v1','kind':'pose','source_scope':'user_drive','total_parts':1,
            'parts':[{'index':0,'name':'pose.tar.gz000','version':'9','size_bytes':cls.full.stat().st_size,
                'source':{'type':'drive','file_id':'fake'},
                'identity':{'version':'9','modified_time':'2026-10-14T00:00:00.000Z'}}]}
        cls.bundle=cls.base/'v3.zip';rows=[];h=hashlib.sha256();offset=0
        # 이전 prefix의 비정렬 chunk 경계를 유지해 실제 v3와 같은769행을 구성한다.
        ends=list(range(262144,100561588,262144))+[100561588]
        ends+=list(range(100561588+262144,128*M,262144))+[128*M]
        ends+=list(range(128*M+262144,C,262144))+[C]
        with cls.full.open('rb') as src, ZipFile(cls.bundle,'x',ZIP_STORED) as z:
            with z.open('prefix.bin','w') as dst:
                for end in ends:
                    data=src.read(end-offset);dst.write(data);h.update(data)
                    rows.append({'start':offset,'end':end-1,'sha256':hashlib.sha256(data).hexdigest()});offset=end
            body={'schema_version':WARM192_POLICY.schema,'policy_id':WARM192_POLICY.name,
                'archive_manifest':cls.manifest,'archive_manifest_sha256':digest(cls.manifest),
                'prefix_bytes':C,'prefix_sha256':h.hexdigest(),'chunks':rows,
                'status':'usable_prefix','producer_code_sha256':code_hashes()}
            doc={**body,'content_sha256':digest(body)};z.writestr('manifest.json',canonical(doc)+b'\n')
        assert len(rows)==769
        cls.spec=input_spec(cls.bundle,policy=TAIL352_POLICY)
        cls.expected={'manifest_sha256':doc['content_sha256'],'prefix_sha256':h.hexdigest(),'prefix_bytes':C}

    @classmethod
    def tearDownClass(cls):cls.tmp.cleanup()

    def setUp(self):
        tmp=tempfile.TemporaryDirectory(dir=self.base);self.addCleanup(tmp.cleanup);self.root=Path(tmp.name)
        self.session=FakeDrive(self.selection,{'fake':b''});self.session.close=lambda:None
        self.session.docs['fake']['size']=str(self.full.stat().st_size)
        self.session.get_adapter=lambda _:SimpleNamespace(max_retries=SimpleNamespace(total=0))
        original=self.session.get
        def get(url,**kwargs):
            if 'alt=media' not in url:return original(url,**kwargs)
            self.session.calls.append((url,kwargs));start,end=map(int,kwargs['headers']['Range'].split('=')[1].split('-'))
            with self.full.open('rb') as f:f.seek(start);body=f.read(end-start+1)
            raw=Raw(body)
            return SimpleNamespace(status_code=206,raw=raw,close=raw.close,
                headers={'Content-Length':str(len(body)),'Content-Range':f'bytes {start}-{end}/{self.full.stat().st_size}'})
        self.session.get=get

    def run_prepare(self,targets=None,expectation=None):
        targets=targets or [A,B]
        return prepare(self.manifest,self.root/'pilot',targets,profile_limits(TAIL352_PROFILE),
            profile=TAIL352_PROFILE,exact_targets=targets,authorized_session=self.session,
            cache_input=self.spec,cache_expectation=expectation or self.expected)

    def test_suffix_pair_hash_json_and_early_stop(self):
        e=self.run_prepare();self.assertTrue(e['targets_complete'])
        first=next(row for row in e['range_audit'] if row['input_source']=='drive')
        self.assertEqual(first['start'],C)
        self.assertEqual(e['stop_reason'],'targets_complete');self.assertFalse(e['whole_archive_integrity_verified'])
        review=review_run(self.root)
        self.assertTrue(review['selected_json_verification']['selected_pair_passed'])
        self.assertTrue(save_result_zip(self.root).is_file())

    def test_missing_pair_eof_crc_and_v4_export(self):
        missing=[f'{SEQUENCE}/json/frame_{i}.json' for i in (99998,99999)]
        e=self.run_prepare(missing)
        self.assertEqual(e['stop_reason'],'complete');self.assertTrue(e['gzip_eof_crc_valid'])
        self.assertTrue(e['whole_archive_integrity_verified']);self.assertFalse(e['targets_complete'])
        self.assertEqual(e['prefix_cache']['prefix_bytes'],self.full.stat().st_size)
        self.assertFalse(review_run(self.root)['selected_json_verification']['selected_pair_passed'])
        bundle,report=export_cache(self.root/'pilot/cache')
        self.assertLess(report['local_read_bytes'],TAIL352_POLICY.export_io_bytes)
        self.assertEqual(input_spec(bundle,policy=TAIL352_POLICY)['manifest_sha256'],report['manifest_sha256'])
        with self.assertRaises(PreparationError):input_spec(bundle,policy=WARM192_POLICY)

    def test_changed_source_and_pinned_hash_refuse(self):
        self.session.docs['fake']['version']='10'
        e=self.run_prepare();self.assertEqual(e['prefix_cache']['status'],'unusable')
        self.assertEqual(e['stats']['media_requests'],0)

    def test_wrong_input_hash_has_no_remote_fallback(self):
        e=self.run_prepare(expectation={**self.expected,'prefix_sha256':'0'*64})
        self.assertEqual(e['stop_reason'],'cache_expectation_mismatch');self.assertEqual(self.session.calls,[])

    def test_catalog_truncation_does_not_block_target_and_invalid_json_fails(self):
        with patch('rewind_archive.extraction.Catalog',lambda **kw:Catalog(max_entries=1,**kw)):
            e=self.run_prepare()
        self.assertTrue(e['observation_catalog']['truncated']);self.assertTrue(e['targets_complete'])
        entry=e['files'][0];path=self.root/'pilot'/entry['relative_path']
        bad=b'!'*entry['size_bytes'];path.write_bytes(bad)
        entry['sha256']=hashlib.sha256(bad).hexdigest()
        e['content_sha256']=digest({k:v for k,v in e.items() if k!='content_sha256'})
        (self.root/'pilot/extraction_manifest.json').write_bytes(canonical(e)+b'\n')
        result=review_run(self.root)
        self.assertEqual(result['file_verification'],'passed_current_code')
        self.assertFalse(result['selected_json_verification']['selected_pair_passed'])
        self.assertEqual(result['selected_json_verification']['invalid_members'],[entry['member']])

    def test_crc_failure_stays_unusable(self):
        original=self.session.get
        def corrupt(url,**kwargs):
            response=original(url,**kwargs)
            if 'alt=media' in url:
                start,end=map(int,kwargs['headers']['Range'].split('=')[1].split('-'))
                if end==self.full.stat().st_size-1:
                    with self.full.open('rb') as f:f.seek(start);data=f.read(end-start+1)
                    raw=Raw(data[:-1]+bytes([data[-1]^1]));response.raw=raw;response.close=raw.close
            return response
        self.session.get=corrupt
        e=self.run_prepare([f'{SEQUENCE}/json/frame_{i}.json' for i in (99998,99999)])
        self.assertFalse(e['inspection_candidates_usable']);self.assertFalse(e['whole_archive_integrity_verified'])
        self.assertEqual(e['prefix_cache']['status'],'unusable')

    def test_real_budget_preflight_and_report_room(self):
        config={'job_id':'tail352_fixture','reason':'bounded fixture','mode':'extract','purpose':'targeted',
            'profile':TAIL352_PROFILE,'request_policy':METADATA_PRE_POLICY,'cache_input':self.spec,
            'cache_expectation':self.expected,'selection':self.selection,'includes':[A,B],'exact_targets':[A,B],
            'limits':asdict(profile_limits(TAIL352_PROFILE)),'metadata_limits':asdict(MetadataLimits()),'previous_result':''}
        self.assertEqual(NOTEBOOK_API_VERSION,10)
        p=make_preflight(config,{})['summary'];self.assertEqual(p['total_request_upper_bound'],1924)
        self.assertEqual((368145316-C+262143)//262144,637)
        self.assertEqual(637*3+2+3,profile_limits(TAIL352_PROFILE).max_ranges)
        limits=profile_limits(TAIL352_PROFILE)
        reserved=report_reserve_bytes(self.manifest,[A,B],limits,[A,B],METADATA_PRE_POLICY,TAIL352_POLICY)+6*M
        self.assertLess(reserved+8*M,limits.max_disk_bytes)
        for value in (None,{**self.expected,'prefix_bytes':C-1}):
            with self.assertRaises(PreparationError):make_preflight({**config,'cache_expectation':value},{})


if __name__=='__main__':unittest.main()
