"""작은 warm fixture로 고정269 profile·기존계약·부분회수를 검사한다."""
from dataclasses import asdict
import json
from pathlib import Path
import unittest
from unittest.mock import patch

from rewind_archive import profiles
from rewind_archive.contracts import PreparationError,validate_includes,Budget,canonical,digest
from rewind_archive.extraction import prepare
from rewind_archive.workflow import make_preflight,review_run,save_result_zip
import test_pose67 as fixtures


class Pose269Tests(unittest.TestCase):
    def setUp(self):
        self.helper=fixtures.Pose67Tests();self.helper.setUp();self.addCleanup(self.helper.doCleanups)
        self.helper.targets=profiles.target269_names()

    def fixture(self,missing=()):
        self.helper.fixture(missing)
        patcher=patch.object(profiles,'TARGET269_CACHE_EXPECTATION',self.helper.expected)
        patcher.start();self.addCleanup(patcher.stop)
        self.helper.config.update(profile=profiles.TARGET269_PROFILE,
            limits=asdict(profiles.profile_limits(profiles.TARGET269_PROFILE)))

    def prepare(self):
        h=self.helper
        return prepare(h.manifest,h.output/'pilot',h.targets,profiles.profile_limits(profiles.TARGET269_PROFILE),
            authorized_session=h.session,profile=profiles.TARGET269_PROFILE,exact_targets=h.targets,
            cache_input=h.spec,cache_expectation=h.expected,request_policy=h.config['request_policy'])

    def test_269_warm_first_new_range_json_and_result_over256_files(self):
        self.fixture();h=self.helper
        self.assertEqual(len(make_preflight(h.config,{})['config']['exact_targets']),269)
        report=self.prepare();self.assertEqual(len(report['files']),269);self.assertTrue(report['targets_complete'])
        media=[kw['headers']['Range'] for url,kw in h.session.calls if 'alt=media' in url]
        self.assertTrue(media[0].startswith('bytes='+str(h.expected['prefix_bytes'])+'-'))
        self.assertTrue(review_run(h.output)['selected_json_verification']['selected_targets_passed'])
        self.assertTrue(save_result_zip(h.output).is_file())
        import hashlib
        entry=report['files'][0];invalid=b'!'*entry['size_bytes']
        (h.output/'pilot'/entry['relative_path']).write_bytes(invalid);entry['sha256']=hashlib.sha256(invalid).hexdigest()
        sealed={**report,'content_sha256':digest({k:v for k,v in report.items() if k!='content_sha256'})}
        (h.output/'pilot/extraction_manifest.json').write_bytes(canonical(sealed)+b'\n')
        checked=review_run(h.output)
        self.assertEqual(checked['file_verification'],'passed_current_code')
        self.assertFalse(checked['selected_json_verification']['selected_targets_passed'])

    def test_old32_67_wrong_target_pin_and_no_cold_fail_before_requests(self):
        self.fixture();h=self.helper
        with self.assertRaises(PreparationError):validate_includes(h.targets)
        for bad in ({**h.config,'profile':profiles.TARGET67_PROFILE,'limits':asdict(profiles.profile_limits(profiles.TARGET67_PROFILE))},
            {**h.config,'includes':h.targets[:-1],'exact_targets':h.targets[:-1]},
            {**h.config,'cache_input':None},
            {**h.config,'cache_expectation':{**h.expected,'prefix_sha256':'0'*64}}):
            with self.assertRaises(PreparationError):make_preflight(bad,{})
        self.assertEqual(h.session.calls,[])

    def test_budget_stop_missing_and_partial_zip(self):
        self.fixture();h=self.helper;original=Budget.add
        def limited(budget,field,amount,maximum):
            if field=='members' and budget.members>=10:raise PreparationError('members_limit')
            return original(budget,field,amount,maximum)
        with patch.object(Budget,'add',limited):report=self.prepare()
        self.assertEqual(report['stop_reason'],'members_limit')
        self.assertTrue(report['missing_targets']);self.assertFalse(report['targets_complete'])
        self.assertTrue(save_result_zip(h.output).is_file())

    def test_catalog_truncation_does_not_stop_targets(self):
        self.fixture()
        import rewind_archive.catalog as catalog
        with patch.object(catalog,'MAX_ENTRIES',2):report=self.prepare()
        self.assertTrue(report['observation_catalog']['truncated'])
        self.assertTrue(report['targets_complete'])
        self.assertTrue(review_run(self.helper.output)['selected_json_verification']['selected_targets_passed'])
