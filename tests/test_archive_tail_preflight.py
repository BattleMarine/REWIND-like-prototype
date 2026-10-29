"""사용자 인계용 소형 사전검사. 파일/Drive/대형 fixture 생성 없음."""
import unittest
from dataclasses import asdict
from rewind_archive.cache_policy import TAIL352_POLICY, WARM192_POLICY, validate_expectation
from rewind_archive.profiles import TAIL352_PROFILE, WARM192_PROFILE, profile_limits
from rewind_archive.workflow import make_preflight, NOTEBOOK_API_VERSION
from rewind_archive.drive_metadata import MetadataLimits
from rewind_archive.contracts import PreparationError
from rewind_archive.request_policy import METADATA_PRE_POLICY


class TailPreflightTests(unittest.TestCase):
    def test_api_and_old_budget(self):
        self.assertEqual(NOTEBOOK_API_VERSION,10)
        self.assertEqual(profile_limits(WARM192_PROFILE).max_ranges,773)
        self.assertEqual(WARM192_POLICY.logical_bytes,192*1024**2)

    def test_input_is_required_and_pinned(self):
        with self.assertRaises(PreparationError):validate_expectation(None,required=True,policy=TAIL352_POLICY)
        with self.assertRaises(PreparationError):validate_expectation({'manifest_sha256':'0'*64,
            'prefix_sha256':'1'*64,'prefix_bytes':192*1024**2-1},policy=TAIL352_POLICY)

    def test_budget_summary(self):
        target='scene/Day/actor/item/003/Getting_Up__1_/json/frame_{}.json'
        pair=[target.format(i) for i in (1,10)]
        config={'job_id':'tail_preflight_fixture','reason':'offline preflight','mode':'extract','purpose':'targeted',
            'profile':TAIL352_PROFILE,'request_policy':METADATA_PRE_POLICY,
            'cache_input':{'path':'fixture-not-opened.zip','manifest_sha256':'0'*64},
            'cache_expectation':{'manifest_sha256':'0'*64,'prefix_sha256':'1'*64,'prefix_bytes':192*1024**2},
            'selection':{'schema_version':'drive_part_selection_v1','archive_id':'fixture','dataset_version':'mock_v1',
                'kind':'pose','total_parts':1,'parts':[{'index':0,'name':'scene.tar.gz000','file_id':'fake'}]},
            'includes':pair,'exact_targets':pair,'limits':asdict(profile_limits(TAIL352_PROFILE)),
            'metadata_limits':asdict(MetadataLimits()),'previous_result':''}
        result=make_preflight(config,{})['summary']
        self.assertEqual(result['total_request_upper_bound'],1924)
        self.assertEqual(profile_limits(TAIL352_PROFILE).max_elapsed_seconds,1200)
        self.assertEqual((368145316-192*1024**2+262143)//262144,637)
        with self.assertRaises(PreparationError):make_preflight({**config,'cache_input':None},{})


if __name__=='__main__':unittest.main()
