"""완료 bundle reader의 T1 수치·mask·출처·lazy 경계 검사."""
import contextlib
import copy
import io
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from rewind_archive.contracts import canonical, digest
from rewind_bundle_data import PreparedBundleReader, ReaderLimits
from rewind_bundle_data.integrity import sha
from rewind_data.audit import REQUIRED_FIELDS
from rewind_data.windows import CausalWindow, FieldSnapshot, FrameSnapshot
from rewind_m1.data import InputContract, pack_window
from rewind_prepared_bundle.core import verify_bundle

ROOT = Path(__file__).resolve().parents[1]


def marker_pin(root):
    return json.loads((root/'COMPLETE.json').read_bytes())['content_sha256']


def reseal(root):
    marker = json.loads((root/'COMPLETE.json').read_bytes())
    for row in marker['files']:
        raw = (root/row['path']).read_bytes();row.update(size_bytes=len(raw),sha256=sha(raw))
    marker['total_bytes'] = sum(r['size_bytes'] for r in marker['files'])
    marker['content_sha256'] = digest({k:v for k,v in marker.items() if k != 'content_sha256'})
    (root/'COMPLETE.json').write_bytes(canonical(marker))


@unittest.skipUnless(importlib.util.find_spec('indexed_gzip'),'기존 requirements-index 환경 필요')
class BundleDataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from test_prepared_bundle import PreparedBundleTests
        PreparedBundleTests.setUpClass()
        cls.addClassCleanup(PreparedBundleTests.doClassCleanups)
        case = PreparedBundleTests('test_complete_multi_pose_bundle_stable_keys_and_original_bytes');case.setUp()
        cls.fixture_bundle = case.base/'reader_fixture_bundle'
        case.build(output=cls.fixture_bundle)

    def setUp(self):
        parent = ROOT/'outputs/test_tmp';parent.mkdir(parents=True,exist_ok=True)
        self.tmp = tempfile.TemporaryDirectory(dir=parent);self.addCleanup(self.tmp.cleanup)
        self.base = Path(self.tmp.name);self.bundle = self.base/'bundle'
        shutil.copytree(self.fixture_bundle,self.bundle)

    def reader(self,**kwargs):
        args = dict(completion_sha256=marker_pin(self.bundle),split='train',diagnostic_only=True)
        args.update(kwargs)
        return PreparedBundleReader(self.bundle,**args)

    def test_preserved_cross_job_batch_pack_window_rgb_and_mask_parity(self):
        reader = self.reader();batch = reader.load_batch([1,0])
        self.assertEqual(list(batch['images'].shape),[2,2,3,1024,1024])
        self.assertEqual([m['frame_index'] for m in batch['metadata']],[10,0])
        self.assertEqual(reader.identity['sample_keys'][0][-1],0)
        for b,meta in enumerate(batch['metadata']):
            row = json.loads((self.bundle/meta['pose']['candidate']).read_bytes())['records'][meta['pose']['record_index']]
            snapshots = tuple(FieldSnapshot(name,tuple(None if p is None else tuple(p) for p in row['fields'][name]['coordinates']),tuple(row['fields'][name]['valid_mask'])) for name in REQUIRED_FIELDS)
            frame = FrameSnapshot(row['sequence_id'],row['frame_index'],row['source']['member'],row['timestamp_raw'],snapshots,())
            names = tuple(reader.identity['joint_names'])
            window = CausalWindow(names,tuple(REQUIRED_FIELDS),(frame,),'diagnostic_independent_frame')
            expected = pack_window(window,SimpleNamespace(joint_names=names,distance_unit='mm',input_source='gt_2d',image_sizes=[[1024,1024]]*2))
            for actual,wanted in zip((batch['features'],batch['time_valid'],batch['target_mm'],batch['target_valid']),expected):
                self.assertTrue(torch.equal(actual[b],wanted))
            for v,side in enumerate(('left','right')):
                f = row['fields']['camera_'+side+'_pts2d']
                self.assertTrue(torch.equal(batch['annotation_valid'][b,v],torch.tensor(f['valid_mask'])))
                self.assertTrue(torch.equal(batch['gt_xy'][b,v],torch.tensor(f['coordinates'])))
        self.assertEqual(batch['identity']['data_kind'],'dataset_sample')
        self.assertEqual(batch['identity']['source_scope'],'synthetic_fixture')
        self.assertFalse(batch['identity']['train_ready'])
        with self.assertRaises(Exception):InputContract.from_dict(reader._contract)

    def test_lazy_parse_cache_and_requested_png_decode(self):
        from rewind_m1.local_rgb import LocalStereoRGB
        with patch.object(LocalStereoRGB,'_decode_into',side_effect=AssertionError('no decode yet')):
            reader = self.reader()
            self.assertIsNone(reader._cached)
            reader.load_batch([0],include_rgb=False)
            reader.load_batch([0],include_rgb=False)
            self.assertEqual(reader._integrity.parsed_reads['pose/pose_b/candidate.json'],2) # 생성 감사1 + 요청1
            reader.load_batch([1],include_rgb=False)
            self.assertEqual(reader._cached_path,'pose/pose_a/candidate.json')
        original = LocalStereoRGB._decode_into
        calls = []
        def tracked(instance,view,*args):
            calls.append(view['member']);return original(instance,view,*args)
        with patch.object(LocalStereoRGB,'_decode_into',tracked):
            reader.load_batch([0])
        self.assertEqual(len(calls),2)
        self.assertTrue(all('final_0.png' in name for name in calls))

    def test_split_limits_and_explicit_diagnostic_gate(self):
        with self.assertRaisesRegex(ValueError,'diagnostic_only'):self.reader(diagnostic_only=False)
        with self.assertRaisesRegex(ValueError,'explicit_split'):self.reader(split=None)
        with self.assertRaisesRegex(ValueError,'completion_pin'):self.reader(completion_sha256='0'*64)
        val = self.reader(split='validation')
        self.assertEqual(len(val),1)
        with self.assertRaisesRegex(ValueError,'reader_index'):val.load_batch([1])
        with self.assertRaisesRegex(ValueError,'candidate|document_limit'):self.reader(limits=ReaderLimits(max_candidate_bytes=1))
        reader = self.reader(limits=ReaderLimits(max_batch_tensor_bytes=32*1024**2))
        with self.assertRaisesRegex(ValueError,'tensor_limit'):reader.load_batch([0,1])
        with self.assertRaisesRegex(ValueError,'batch_size'):reader.load_batch([0,1,0])
        with torch.device('meta'):
            batch = reader.load_batch([0],include_rgb=False)
        self.assertTrue(all(v.device.type == 'cpu' for v in batch.values() if isinstance(v,torch.Tensor)))

    def test_marker_verifier_accept_reject_parity(self):
        self.assertEqual(verify_bundle(self.bundle)['content_sha256'],self.reader().identity['completion_sha256'])
        path = self.bundle/'COMPLETE.json';original = path.read_bytes()
        marker = json.loads(original);marker['files'].reverse()
        marker['content_sha256'] = digest({k:v for k,v in marker.items() if k != 'content_sha256'})
        path.write_bytes(canonical(marker))
        for verifier in (lambda:verify_bundle(self.bundle),self.reader):
            with self.assertRaises(Exception):verifier()
        path.write_bytes(original)
        mutations = [('extra.txt',b'extra'),('metadata/spec.json',b'bad'),('rgb/manifest.json',None)]
        for name,data in mutations:
            path = self.bundle/name;original = path.read_bytes() if path.exists() else None
            if data is None:path.unlink()
            else:path.write_bytes(data)
            for verifier in (lambda:verify_bundle(self.bundle),self.reader):
                with self.assertRaises(Exception):verifier()
            if original is None:path.unlink()
            else:path.write_bytes(original)
        # marker와 파일 목록을 함께 바꾸더라도 입력 metadata pin 불일치는 거부한다.
        path = self.bundle/'metadata/selection.json';path.write_bytes(path.read_bytes()+b' ');reseal(self.bundle)
        for verifier in (lambda:verify_bundle(self.bundle),self.reader):
            with self.assertRaises(Exception):verifier()

    def test_resealed_candidate_raw_mismatch_and_release_false(self):
        path = self.bundle/'pose/pose_b/candidate.json';original = path.read_bytes()
        doc = json.loads(original);doc['records'][0]['fields']['device_pts3d']['coordinates'][0][0] += 1
        doc['content_sha256'] = digest({k:v for k,v in doc.items() if k != 'content_sha256'})
        path.write_bytes(canonical(doc));reseal(self.bundle)
        with self.assertRaisesRegex(ValueError,'raw_field_or_unit'):self.reader()
        path.write_bytes(original);reseal(self.bundle)
        doc = json.loads((self.bundle/'COMPLETE.json').read_bytes());doc['train_ready'] = True
        doc['content_sha256'] = digest({k:v for k,v in doc.items() if k != 'content_sha256'})
        (self.bundle/'COMPLETE.json').write_bytes(canonical(doc))
        for verifier in (lambda:verify_bundle(self.bundle),self.reader):
            with self.assertRaises(Exception):verifier()

    def test_cached_candidate_raw_and_reference_tamper_rejected(self):
        reader = self.reader();reader.load_batch([0],include_rgb=False)
        row = reader.metadata(0);job = row['pose']['job_id']
        rawpath = self.bundle/'inputs'/job/'files'/row['views']['pose']['member']
        original = rawpath.read_bytes();rawpath.write_bytes(original+b' ')
        with self.assertRaisesRegex(ValueError,'file_size|file_hash'):reader.load_batch([0],include_rgb=False)
        rawpath.write_bytes(original)
        path = self.bundle/'metadata/motion_map.json';path.write_bytes(path.read_bytes()+b' ')
        with self.assertRaisesRegex(ValueError,'file_size|file_hash'):reader.load_batch([0],include_rgb=False)

    def test_metadata_cli_has_no_torch_model_or_training_import(self):
        output = self.base/'inspect'
        script = "import sys,json; from rewind_bundle_data.__main__ import main; main(sys.argv[1:]); assert 'torch' not in sys.modules; assert not any(n.endswith(('training','models','rgb_shard_training')) for n in sys.modules); print('NO_MODEL_IMPORT')"
        result = subprocess.run([sys.executable,'-X','utf8','-c',script,'inspect','--root',str(self.bundle),
            '--completion-sha256',marker_pin(self.bundle),'--split','train','--diagnostic-only','--output',str(output)],
            cwd=ROOT,capture_output=True,text=True,encoding='utf-8',timeout=60)
        self.assertEqual(result.returncode,0,result.stderr)
        self.assertIn('NO_MODEL_IMPORT',result.stdout)
        self.assertEqual(json.loads((output/'report.json').read_bytes())['status'],'passed')


class NewFixtureMaskTests(unittest.TestCase):
    @unittest.skipUnless(__import__('importlib').util.find_spec('indexed_gzip'),'기존 index 환경 필요')
    def test_new_raw_fixture_invalid_and_outside_masks_survive_converter_and_reader(self):
        import test_prepare
        from test_prepared_bundle import PreparedBundleTests
        original = test_prepare.PreparationTests.add_frame
        def altered(instance,seq,index):
            path = original(instance,seq,index)
            if index == 0:
                doc = json.loads(path.read_bytes());first,second = instance.spec['joint_names'][:2]
                doc['joints'][first].update(camera_left_pts2d=[-5.,1200.],camera_left_pts3d=None,camera_right_pts2d=None)
                doc['joints'][second].update(device_pts3d=None,camera_left_pts3d=None,camera_right_pts3d=None)
                path.write_bytes(canonical(doc))
            return path
        with patch.object(test_prepare.PreparationTests,'add_frame',altered):
            PreparedBundleTests.setUpClass()
        try:
            case = PreparedBundleTests('test_complete_multi_pose_bundle_stable_keys_and_original_bytes');case.setUp()
            case.output = case.base/'mask_bundle';marker = case.build()
            reader = PreparedBundleReader(case.output,completion_sha256=marker['content_sha256'],split='train',diagnostic_only=True)
            batch = reader.load_batch([0],include_rgb=False)
            self.assertEqual(batch['gt_xy'][0,0,0].tolist(),[-5.,1200.])
            self.assertTrue(batch['annotation_valid'][0,0,0]);self.assertFalse(batch['annotation_valid'][0,1,0])
            self.assertFalse(batch['target_valid'][0,0,1]);self.assertEqual(batch['target_mm'][0,0,1].tolist(),[0.,0.,0.])
            self.assertIsNone(reader.metadata(0)['time_seconds'])
        finally:
            PreparedBundleTests.doClassCleanups()
