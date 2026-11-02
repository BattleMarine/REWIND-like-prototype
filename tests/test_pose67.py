"""작은 warm 캐시 fixture로 고정67 profile을 검사한다. 실제 원격/234MiB 복제 없음."""
from contextlib import ExitStack
from dataclasses import asdict,replace
import gzip
import hashlib
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from zipfile import ZipFile,ZIP_STORED

from rewind_archive.contracts import canonical,digest,PreparationError,validate_includes,validate_exact_targets
from rewind_archive import cache_policy,prefix_cache,profiles
from rewind_archive.drive_metadata import MetadataLimits
from rewind_archive.extraction import prepare,code_hashes
from rewind_archive.profiles import TARGET67_PROFILE,TAIL352_PROFILE,profile_limits,target67_names
from rewind_archive.workflow import make_preflight,review_run,save_result_zip
from rewind_archive.prefix_cache import input_spec,export_cache
from rewind_archive.request_policy import METADATA_PRE_POLICY
from test_drive_metadata import FakeDrive

ROOT=Path(__file__).resolve().parents[1]


class Pose67Tests(unittest.TestCase):
    def setUp(self):
        parent=ROOT/'outputs/test_tmp';parent.mkdir(parents=True,exist_ok=True)
        self.tmp=tempfile.TemporaryDirectory(dir=parent);self.addCleanup(self.tmp.cleanup);self.base=Path(self.tmp.name)
        self.output=self.base/'job'
        self.targets=target67_names()
        # 정책의 byte 한도만 test 범위에서 축소한다. 실제 profile의 대상/파일수/전송cap은 바꾸지 않는다.
        self.policy=replace(cache_policy.TAIL352_POLICY,logical_bytes=1024**2,disk_bytes=2*1024**2,
                            local_io_bytes=4*1024**2,export_io_bytes=6*1024**2,min_input_bytes=32)
        stack=ExitStack();self.addCleanup(stack.close)
        warm=(*cache_policy.WARM_POLICIES,self.policy)
        stack.enter_context(patch.object(cache_policy,'TAIL352_POLICY',self.policy))
        stack.enter_context(patch.object(cache_policy,'WARM_POLICIES',warm))
        stack.enter_context(patch.object(prefix_cache,'WARM_POLICIES',warm))

    def fixture(self,missing=()):
        from rewind_archive.fixture import tar_bytes
        payloads=[(name,canonical({'frame':int(name.rsplit('_',1)[1].split('.')[0]),'fixture_only':True})) for name in self.targets if name not in missing]
        self.compressed=gzip.compress(tar_bytes(payloads),mtime=0)
        self.selection={'schema_version':'drive_part_selection_v1','archive_id':'pose67_small_fixture','dataset_version':'synthetic_test_only',
            'kind':'pose','total_parts':1,'parts':[{'index':0,'name':'pose.tar.gz000','file_id':'fake'}]}
        self.session=FakeDrive(self.selection,{'fake':self.compressed})
        self.session.get_adapter=lambda _:SimpleNamespace(max_retries=SimpleNamespace(total=0));self.session.close=lambda:None
        manifest={'schema_version':'archive_manifest_v1','archive_id':self.selection['archive_id'],'dataset_version':self.selection['dataset_version'],
            'kind':'pose','source_scope':'user_drive','total_parts':1,'parts':[{'index':0,'name':'pose.tar.gz000','size_bytes':len(self.compressed),
            'version':'9','source':{'type':'drive','file_id':'fake'},'identity':{'version':'9','modified_time':'2026-10-14T00:00:00.000Z'}}]}
        size=min(97,len(self.compressed)-1);body=self.compressed[:size]
        doc={'schema_version':self.policy.schema,'policy_id':self.policy.name,'archive_manifest':manifest,'archive_manifest_sha256':digest(manifest),
            'prefix_bytes':size,'prefix_sha256':hashlib.sha256(body).hexdigest(),
            'chunks':[{'start':0,'end':size-1,'sha256':hashlib.sha256(body).hexdigest()}],
            'status':'usable_prefix','producer_code_sha256':code_hashes()}
        doc['content_sha256']=digest(doc)
        cache=self.base/'warm.zip'
        with ZipFile(cache,'x',ZIP_STORED) as z:z.writestr('prefix.bin',body);z.writestr('manifest.json',canonical(doc)+b'\n')
        self.expected={'manifest_sha256':doc['content_sha256'],'prefix_sha256':doc['prefix_sha256'],'prefix_bytes':size}
        expectation_patch=patch.object(profiles,'TARGET67_CACHE_EXPECTATION',self.expected);expectation_patch.start();self.addCleanup(expectation_patch.stop)
        self.spec=input_spec(cache,policy=self.policy);self.manifest=manifest
        self.config={'job_id':'pose67_fixture','reason':'bounded67 fixture only','mode':'extract','purpose':'targeted',
            'selection':self.selection,'profile':TARGET67_PROFILE,'includes':self.targets,'exact_targets':self.targets,
            'limits':asdict(profile_limits(TARGET67_PROFILE)),'metadata_limits':asdict(MetadataLimits()),'previous_result':'',
            'request_policy':METADATA_PRE_POLICY,'cache_input':self.spec,'cache_expectation':self.expected}

    def run_prepare(self):
        return prepare(self.manifest,self.output/'pilot',self.targets,profile_limits(TARGET67_PROFILE),authorized_session=self.session,
                       profile=TARGET67_PROFILE,exact_targets=self.targets,cache_input=self.spec,cache_expectation=self.expected,
                       request_policy=METADATA_PRE_POLICY)

    def test_exact67_warm_first_new_range_and_json_review(self):
        self.fixture();preflight=make_preflight(self.config,{})
        self.assertEqual(len(preflight['config']['exact_targets']),67)
        report=self.run_prepare();self.assertTrue(report['targets_complete']);self.assertEqual(len(report['files']),67)
        ranges=[k['headers']['Range'] for url,k in self.session.calls if 'alt=media' in url]
        self.assertTrue(ranges[0].startswith('bytes='+str(self.expected['prefix_bytes'])+'-'))
        self.assertEqual(report['prefix_cache']['input_prefix_bytes'],self.expected['prefix_bytes'])
        checked=review_run(self.output);result=checked['selected_json_verification']
        self.assertTrue(result['selected_targets_passed']);self.assertFalse(result['selected_pair_passed'])
        self.assertTrue(save_result_zip(self.output).is_file())
        exported,export_report=export_cache(self.output/'pilot/cache',policy=self.policy)
        self.assertEqual(input_spec(exported,policy=self.policy)['manifest_sha256'],export_report['manifest_sha256'])
        entry=report['files'][0];path=self.output/'pilot'/entry['relative_path']
        invalid=b'!'*entry['size_bytes'];path.write_bytes(invalid);entry['sha256']=hashlib.sha256(invalid).hexdigest()
        sealed={**report,'content_sha256':digest({k:v for k,v in report.items() if k!='content_sha256'})}
        (self.output/'pilot/extraction_manifest.json').write_bytes(canonical(sealed)+b'\n')
        self.assertFalse(review_run(self.output)['selected_json_verification']['selected_targets_passed'])

    def test_partial_eof_missing_is_not_all_targets_complete(self):
        absent=self.targets[-3:];self.fixture(absent);report=self.run_prepare()
        self.assertEqual(report['stop_reason'],'complete');self.assertTrue(report['gzip_eof_crc_valid'])
        self.assertEqual(report['missing_targets'],absent);self.assertFalse(report['targets_complete'])
        self.assertFalse(review_run(self.output)['selected_json_verification']['selected_targets_passed'])

    def test_budget_stop_preserves_partial_reports_and_missing(self):
        self.fixture()
        from rewind_archive.contracts import Budget
        original=Budget.add
        def limited(budget,field,amount,maximum):
            if field=='members' and budget.members>=10:raise PreparationError('members_limit')
            return original(budget,field,amount,maximum)
        with patch.object(Budget,'add',limited):report=self.run_prepare()
        self.assertEqual(report['stop_reason'],'members_limit');self.assertFalse(report['targets_complete'])
        self.assertTrue(report['inspection_candidates_usable']);self.assertTrue(report['missing_targets'])
        self.assertTrue(save_result_zip(self.output).is_file())

    def test_old_cap_pair_and_fixed67_rejection_no_cold(self):
        self.fixture()
        with self.assertRaises(PreparationError):validate_includes(self.targets)
        with self.assertRaises(PreparationError):validate_exact_targets(self.targets,self.targets,profile_limits(TARGET67_PROFILE),'extract')
        changed={**self.config,'profile':TAIL352_PROFILE,'limits':asdict(profile_limits(TAIL352_PROFILE))}
        with self.assertRaises(PreparationError):make_preflight(changed,{})
        wrong={**self.config,'exact_targets':self.targets[:-1],'includes':self.targets[:-1]}
        with self.assertRaisesRegex(PreparationError,'fixed_target_set'):make_preflight(wrong,{})
        with self.assertRaises(PreparationError):make_preflight({**self.config,'cache_input':None},{})
        with self.assertRaisesRegex(PreparationError,'pinned_234MiB'):make_preflight({**self.config,'cache_expectation':{**self.expected,'prefix_sha256':'0'*64}},{})
        self.assertEqual(self.session.calls,[])
        # 핀은 맞아도 ZIP body가 변조된 경우 import 검사가 거부하고 원격으로 대체하지 않는다.
        cache=Path(self.spec['path']);raw=bytearray(cache.read_bytes());offset=raw.find(self.compressed[:30]);self.assertGreater(offset,0);raw[offset]^=1;cache.write_bytes(raw)
        report=self.run_prepare();self.assertFalse(report['inspection_candidates_usable']);self.assertEqual(self.session.calls,[])


if __name__=='__main__':unittest.main()
