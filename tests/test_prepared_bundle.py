"""분할 archive→실제 queue→bundle의 합성 로컬 통합과 거부 경계."""
import contextlib
import copy
import gzip
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image
from rewind_archive.contracts import canonical, digest
from rewind_archive.fixture import local_manifest, tar_bytes
from rewind_archive_index.preparation import _source
from rewind_prepare_queue.core import initialize_queue, make_plan, run_queue, hashed
from rewind_prepared_bundle.core import build_bundle, verify_bundle, LIMITS, MARKER
from rewind_prepared_bundle.__main__ import main
import test_prepare as pose_fixture

ROOT = Path(__file__).resolve().parents[1]


@unittest.skipUnless(importlib.util.find_spec('indexed_gzip'), '기존 requirements-index 환경 필요')
class PreparedBundleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fixture = pose_fixture.PreparationTests()
        cls.fixture.setUp()
        cls.addClassCleanup(cls.fixture.doCleanups)
        f = cls.fixture
        cls.base = f.base
        f.add_frame(f.seq,10)
        f.add_frame(f.other,2)
        motion = copy.deepcopy(f.motion_doc)
        motion['sequences'][f.other] = {'motion_id':'declared_test_b','identity_status':'declared_unverified','evidence':'명시 합성 fixture'}
        f.motion.write_bytes(canonical(motion))
        cls.selection = cls.base/'selection.json'
        cls.selection.write_bytes(canonical([
            {'sequence_id':f.other, 'frames':[2], 'split':'validation', 'split_evidence':'합성 split 파일 선언'},
            {'sequence_id':f.seq, 'frames':[0,10], 'split':'train', 'split_evidence':'합성 split 파일 선언'}]))
        buffer = io.BytesIO()
        Image.new('RGB',(1024,1024),(11,23,47)).save(buffer,format='PNG')
        png = buffer.getvalue()
        keys = [(f.seq,10),(f.other,2),(f.seq,0)]
        jobs = []
        for name,kind,job_keys in [('left','rgb',keys),('right','rgb',keys),('pose_a','pose',keys[:2]),('pose_b','pose',keys[2:])]:
            entries = []
            for seq,index in job_keys:
                member = f'{seq}/json/frame_{index}.json' if kind == 'pose' else f'{seq}/fisheye_final_image/camera_{name}/final_{index}.png'
                payload = (f.raw/member).read_bytes() if kind == 'pose' else png
                entries.append((member,payload))
            folder = 'raw_'+name
            manifest = local_manifest(cls.base/folder,gzip.compress(tar_bytes(entries),mtime=0),kind)
            manifest['archive_id'] = 'fixture_'+name
            jobs.append({'id':name,'manifest':manifest,'source_base':folder,
                         'source_sha256':digest(_source(manifest,cls.base/folder)), 'targets':[n for n,_ in entries],
                         'index_options':{'batch_members':1,'max_batches':8,'spacing':65536}})
        cls.plan = make_plan(jobs)
        cls.planpath = cls.base/'plan.json'; cls.planpath.write_bytes(canonical(cls.plan))
        cls.queue = cls.base/'queue'
        initialize_queue(cls.plan,cls.queue)
        cls.state = run_queue(cls.plan,cls.queue,base=cls.base)
        if cls.state['status'] != 'prepared_not_training_release':
            raise AssertionError(cls.state)

    def setUp(self):
        self.output = self.base/self._testMethodName

    def build(self, **kwargs):
        f = self.fixture
        args = dict(plan_path=self.planpath, queue_root=self.queue, selection_path=self.selection,
                    spec_path=f.specpath, calibration_path=f.calibration, split_dir=f.splitdir,
                    motion_map_path=f.motion, output=self.output, manifest_id='synthetic_bundle')
        args.update(kwargs)
        return build_bundle(**args)

    @contextlib.contextmanager
    def altered(self,path,fn,*,hashed_doc=False):
        raw = path.read_bytes()
        doc = json.loads(raw)
        fn(doc)
        if hashed_doc:
            doc = hashed(doc)
        path.write_bytes(canonical(doc))
        try:
            yield
        finally:
            path.write_bytes(raw)

    def rejected(self, reason):
        with self.assertRaisesRegex(Exception,reason):
            self.build()
        self.assertFalse((self.output/MARKER).exists())
        self.assertEqual(json.loads((self.output/'report.json').read_bytes())['status'],'failed')
        with self.assertRaises(Exception):
            verify_bundle(self.output)
        with self.assertRaises(FileExistsError):
            self.build()

    def test_complete_multi_pose_bundle_stable_keys_and_original_bytes(self):
        before = (self.queue/'state.json').read_bytes()
        marker = self.build()
        self.assertEqual(marker,verify_bundle(self.output))
        self.assertEqual((self.queue/'state.json').read_bytes(),before)
        report = json.loads((self.output/'report.json').read_bytes())
        self.assertFalse(report['train_ready'])
        self.assertEqual(report['evidence_scope'],'synthetic_integration_only')
        self.assertEqual(len(report['pose_candidates']),2)
        self.assertIn('independent_validation_not_verified', report['blockers'])
        self.assertIn('candidate_to_learning_shard_not_implemented', report['blockers'])
        self.assertEqual(marker['input_sha256'],report['input_sha256'])
        provenance = json.loads((self.output/'provenance.json').read_bytes())
        keys = [(r['sequence_id'],r['frame_index']) for r in provenance['records']]
        self.assertEqual(keys,sorted(keys)); self.assertEqual(len(keys),3)
        for row,job in zip(self.state['jobs'],self.plan['jobs']):
            original = (self.queue/row['output']/'extraction_manifest.json').read_bytes()
            self.assertEqual((self.output/'inputs'/job['id']/'extraction_manifest.json').read_bytes(),original)
        for job in ('pose_a','pose_b'):
            candidate = json.loads((self.output/'pose'/job/'candidate.json').read_bytes())
            self.assertEqual(candidate['contract']['data_kind'],'dataset_sample')
            self.assertFalse(candidate['training_eligibility']['train_ready'])

    def test_original_archive_parts_not_required(self):
        with contextlib.ExitStack() as stack:
            for job in self.plan['jobs']:
                folder = self.base/job['source_base']
                moved = self.base/(job['source_base']+'_hidden')
                folder.rename(moved)
                stack.callback(moved.rename,folder)
            self.build()
            verify_bundle(self.output)

    def test_partial_queue_refused(self):
        with self.altered(self.queue/'state.json',lambda d:d.update(status='bounded_job_stop'),hashed_doc=True):
            self.rejected('bundle_queue_not_completed')

    def test_selection_subset_refused(self):
        with self.altered(self.selection,lambda d:d[1].update(frames=[0])):
            self.rejected('bundle_selection_must_equal_complete_queue')

    def test_explicit_split_mismatch(self):
        with self.altered(self.selection,lambda d:d[1].update(split='test')):
            self.rejected('bundle_selection_split_mismatch')

    def test_calibration_tamper(self):
        with self.altered(self.fixture.calibration,lambda d:d.update(affine=[2,0,0])):
            self.rejected('bundle_calibration_pin')

    def test_motion_cross_job_split_leak(self):
        with self.altered(self.fixture.motion,lambda d:d['sequences'][self.fixture.other].update(motion_id='declared_test_a')):
            self.rejected('motion_split_overlap')

    def test_matching_view_tamper(self):
        with self.altered(self.queue/'matching.json',lambda d:d['frames'][0]['views']['left'].update(sha256='0'*64),hashed_doc=True):
            self.rejected('bundle_matching_identity')

    def test_source_pin_tamper(self):
        job = self.plan['jobs'][0]
        plan = copy.deepcopy(self.plan)
        plan['jobs'][0]['source_sha256'] = '0'*64
        plan = hashed(plan)
        # 잘못된 plan을 state까지 재발급하지 않고 pinned plan 단계에서 거부한다.
        path = self.base/'bad_plan.json';path.write_bytes(canonical(plan))
        with self.assertRaisesRegex(ValueError,'bundle_pinned_plan_changed'):
            self.build(plan_path=path)

    def test_payload_missing_and_mutation_preserved(self):
        row = self.state['jobs'][0]
        doc = json.loads((self.queue/row['output']/'extraction_manifest.json').read_bytes())
        path = self.queue/row['output']/doc['files'][0]['relative_path']
        raw = path.read_bytes()
        path.write_bytes(bytes([raw[0]^1])+raw[1:])
        try:
            self.rejected('extracted_file_hash')
        finally:
            path.write_bytes(raw)
        self.assertTrue((self.output/'metadata/plan.json').is_file())

    def test_payload_bound_before_copy(self):
        with patch.dict(LIMITS,{'rgb_payload_bytes':1}):
            self.rejected('bundle_payload_limit')
        self.assertFalse((self.output/'inputs').exists())

    def test_rgb_job_bound_and_pose_input_bound_before_copy(self):
        with patch.dict(LIMITS,{'rgb_jobs':1}):
            self.rejected('bundle_rgb_job_limit')
        # 별도 출력에서 converter의 더 좁은 spec 한계도 preflight로 거부한다.
        self.output = self.base/(self._testMethodName+'_pose')
        with self.altered(self.fixture.specpath,lambda d:d['limits'].update(max_total_input_bytes=1)):
            self.rejected('prepare_input_byte_limit|bundle_pose_converter_input_limit')
        self.assertFalse((self.output/'inputs').exists())

    def test_missing_payload(self):
        row = self.state['jobs'][0]
        doc = json.loads((self.queue/row['output']/'extraction_manifest.json').read_bytes())
        path = self.queue/row['output']/doc['files'][0]['relative_path']
        raw = path.read_bytes();path.unlink()
        try:
            self.rejected('bundle_payload_missing_size')
        finally:
            path.write_bytes(raw)

    def test_mixed_scope_before_payload_verification(self):
        # 실제 archive 생성이 아니라 입력 계약 거부 검사다. plan/state pin까지는 일관되게 만든다.
        paths = [self.planpath,self.queue/'plan.json',self.queue/'state.json',self.queue/'matching.json']
        originals = [p.read_bytes() for p in paths]
        plan = copy.deepcopy(self.plan)
        plan['jobs'][-1]['manifest']['source_scope'] = 'local_unverified'
        plan = hashed(plan)
        state = copy.deepcopy(self.state);state['plan_sha256'] = plan['content_sha256']
        matching = json.loads(originals[3]);matching['plan_sha256'] = plan['content_sha256'];matching = hashed(matching)
        state['matching']['content_sha256'] = matching['content_sha256'];state = hashed(state)
        for p,d in zip(paths,[plan,plan,state,matching]):
            p.write_bytes(canonical(d))
        try:
            self.rejected('bundle_mixed_source_scope')
        finally:
            for p,raw in zip(paths,originals):p.write_bytes(raw)

    def test_resource_stops_preserve_reason(self):
        with patch.dict(LIMITS,{'elapsed_seconds':0}):
            self.rejected('bundle_elapsed_limit')
        self.output = self.base/(self._testMethodName+'_rss')
        with patch.dict(LIMITS,{'rss_bytes':0}):
            self.rejected('bundle_rss_limit')

    def test_bounded_inventory_checks_before_sort(self):
        from rewind_prepared_bundle.core import inventory, Guard
        # 4097번째 항목을 소비한 뒤 정렬에 도달하지 않고 중단해야 한다.
        self.output.mkdir()
        with patch.object(Path,'rglob',return_value=iter([self.output]*4097)):
            with self.assertRaisesRegex(ValueError,'bundle_inventory_limit'):
                inventory(self.output,Guard())

    def test_inventory_order_uses_posix_strings_across_platforms(self):
        from rewind_prepared_bundle.core import inventory, Guard
        for name in ('a/z.txt','B/y.txt'):
            path = self.output/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(b'x')
        rows,_ = inventory(self.output,Guard())
        self.assertEqual([r['path'] for r in rows],['B/y.txt','a/z.txt'])

    def test_input_change_during_conversion_no_marker(self):
        from rewind_prepared_bundle.core import convert
        raw = self.selection.read_bytes()
        def changed(*args,**kwargs):
            result = convert(*args,**kwargs)
            self.selection.write_bytes(raw+b' ')
            return result
        try:
            with patch('rewind_prepared_bundle.core.convert',side_effect=changed):
                self.rejected('bundle_input_changed')
        finally:
            self.selection.write_bytes(raw)
        self.assertTrue((self.output/'pose/pose_a/candidate.json').is_file())

    def test_completion_file_tamper_and_extra_file(self):
        self.build()
        extra = self.output/'extra.txt';extra.write_text('extra')
        with self.assertRaisesRegex(ValueError,'bundle_inventory_mismatch'):
            verify_bundle(self.output)
        extra.unlink()
        path = self.output/'pose/pose_b/candidate.json'
        path.write_bytes(path.read_bytes()+b' ')
        with self.assertRaisesRegex(ValueError,'bundle_inventory_mismatch'):
            verify_bundle(self.output)

    def test_cli_build_and_verify(self):
        f = self.fixture
        argv = ['build']
        for key,value in [('plan',self.planpath),('queue',self.queue),('selection',self.selection),('spec',f.specpath),
                          ('calibration',f.calibration),('split-dir',f.splitdir),('motion-map',f.motion),('output',self.output),('id','fixture_cli')]:
            argv.extend(['--'+key,str(value)])
        with contextlib.redirect_stdout(io.StringIO()):
            result = main(argv)
            self.assertEqual(result,main(['verify','--root',str(self.output)]))

    def test_existing_or_overlapping_output_refused(self):
        with self.assertRaisesRegex(ValueError,'bundle_output_input_overlap'):
            self.build(output=self.queue/'bundle')
        self.output.mkdir()
        keep = self.output/'keep.txt';keep.write_text('keep')
        with self.assertRaises(FileExistsError):
            self.build()
        self.assertEqual(keep.read_text(),'keep')


if __name__ == '__main__':
    unittest.main()
