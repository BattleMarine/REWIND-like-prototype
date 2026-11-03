"""남은8 고정계약·warm 경계·40/41MiB scope·기존계약 보존 회귀."""
from dataclasses import asdict
import json
from pathlib import Path
import unittest
from unittest.mock import patch
from rewind_archive import profiles
from rewind_archive.contracts import PreparationError,Budget
from rewind_archive.extraction import prepare
from rewind_archive.workflow import make_preflight,review_run,save_result_zip
import test_pose67 as fixtures


class Pose8Tests(unittest.TestCase):
    def setUp(self):
        self.h=fixtures.Pose67Tests();self.h.setUp();self.addCleanup(self.h.doCleanups)
        self.h.targets=profiles.target8_names();self.h.fixture()
        p=patch.object(profiles,'TARGET8_CACHE_EXPECTATION',self.h.expected);p.start();self.addCleanup(p.stop)
        self.h.config.update(profile=profiles.TARGET8_PROFILE,limits=asdict(profiles.profile_limits(profiles.TARGET8_PROFILE)))

    def run_prepare(self):
        h=self.h
        return prepare(h.manifest,h.output/'pilot',h.targets,profiles.profile_limits(profiles.TARGET8_PROFILE),
            authorized_session=h.session,profile=profiles.TARGET8_PROFILE,exact_targets=h.targets,
            cache_input=h.spec,cache_expectation=h.expected,request_policy=h.config['request_policy'])

    def test_exact8_warm_range_hash_json_review_and_zip(self):
        h=self.h;self.assertEqual(len(make_preflight(h.config,{})['config']['exact_targets']),8)
        report=self.run_prepare();self.assertTrue(report['targets_complete']);self.assertEqual(len(report['files']),8)
        media=[kw['headers']['Range'] for url,kw in h.session.calls if 'alt=media' in url]
        self.assertTrue(media[0].startswith('bytes='+str(h.expected['prefix_bytes'])+'-'))
        self.assertTrue(review_run(h.output)['selected_json_verification']['selected_targets_passed'])
        self.assertTrue(save_result_zip(h.output).is_file())

    def test_exact_pin_and_cold_rejection_before_requests(self):
        h=self.h
        for bad in ({**h.config,'exact_targets':h.targets[:-1],'includes':h.targets[:-1]},
                    {**h.config,'exact_targets':profiles.target269_names()[:8],'includes':profiles.target269_names()[:8]},
                    {**h.config,'cache_input':None},
                    {**h.config,'cache_expectation':{**h.expected,'prefix_sha256':'0'*64}},
                    {**h.config,'limits':{**h.config['limits'],'max_disk_bytes':41*1024**2}}):
            with self.assertRaises(PreparationError):make_preflight(bad,{})
        self.assertEqual(h.session.calls,[])

    def test_partial_budget_result_kept(self):
        original=Budget.add
        def limited(b,field,amount,maximum):
            if field=='members' and b.members>=2:raise PreparationError('members_limit')
            return original(b,field,amount,maximum)
        with patch.object(Budget,'add',limited):r=self.run_prepare()
        self.assertFalse(r['targets_complete']);self.assertTrue(r['missing_targets'])
        self.assertFalse(review_run(self.h.output)['selected_json_verification']['selected_targets_passed'])
        self.assertTrue(save_result_zip(self.h.output).is_file())

    def test_output_caps_scoped_to8_legacy_profiles_unchanged(self):
        for p in (profiles.TARGET67_PROFILE,profiles.TARGET269_PROFILE,profiles.TAIL352_PROFILE):
            self.assertEqual(profiles.profile_limits(p).max_disk_bytes,32*1024**2)
        self.assertEqual(profiles.profile_limits(profiles.TARGET8_PROFILE).max_disk_bytes,40*1024**2)
        root=self.h.output;root.mkdir();pilot=root/'pilot';pilot.mkdir();(pilot/'extraction_manifest.json').write_text('{}')
        extra=pilot/'padding.bin'
        with extra.open('wb') as f:f.truncate(34*1024**2)
        import rewind_archive.workflow as wf
        with patch.object(wf,'review_run'),patch.object(wf,'read_report',return_value={'profile':profiles.TARGET8_PROFILE}):
            target=save_result_zip(root);self.assertTrue(target.exists());target.unlink()
        for profile in (profiles.TARGET269_PROFILE,profiles.TARGET67_PROFILE):
            with patch.object(wf,'review_run'),patch.object(wf,'read_report',return_value={'profile':profile,'cache_policy':{'name':'pose_prefix_tail352_v4'}}):
                with self.assertRaisesRegex(PreparationError,'bundle_limit'):save_result_zip(root)
        with extra.open('wb') as f:f.truncate(41*1024**2)
        with patch.object(wf,'review_run'),patch.object(wf,'read_report',return_value={'profile':profiles.TARGET8_PROFILE}):
            with self.assertRaisesRegex(PreparationError,'bundle_limit'):save_result_zip(root)

    def test_legacy_staging_max_kept_and8_wrapper_scope(self):
        import rewind_archive.cache_staging as s
        kwargs=dict(expected_bytes=s.TARGET8_CACHE_BYTES,expected_sha256=s.TARGET8_CACHE_SHA256,
            readonly_root='unused',local_root='unused',forbidden_roots=(),consumed_job='unused',report_path='unused',validator=lambda p:p)
        with self.assertRaisesRegex(PreparationError,'expected_pin'):s.stage_file('unused','unused',**kwargs)
        with self.assertRaisesRegex(PreparationError,'size_scope'):s.stage_file('unused','unused',**kwargs,max_bytes=s.TARGET8_CACHE_BYTES+1)
        with patch.object(s,'stage_file',return_value={}) as stage:
            s.stage_pose8_cache('source','target',source_root='source-root',input_root='inputs',workspace='work',report_path='report',readonly_root='drive',local_root='local')
        self.assertEqual(stage.call_args.kwargs['max_bytes'],250718584)
        self.assertEqual(stage.call_args.kwargs['expected_sha256'],s.TARGET8_CACHE_SHA256)
