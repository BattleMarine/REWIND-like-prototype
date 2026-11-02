"""합성 독립 frame chunk의 native RGB/GT 공급과 선언/실제 PNG 검사."""
import contextlib
import copy
import hashlib
import io
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

import torch

from rewind_archive.contracts import digest
from rewind_m1.data import content_hash
from rewind_m1.local_fixture import create_fixture
from rewind_m1.local_rgb import LocalStereoRGB
from rewind_m1.rgb_supervision import LocalRGBSupervision
from rewind_m1.rgb_shards import (DEFAULT_LIMITS, RGBEpochSampler, RGBShardDataset,
    build_catalog, read_catalog, validate_catalog)
from rewind_m1.rgb_shards_cli import main

ROOT = Path(__file__).resolve().parents[1]


def signed(doc):
    return {**doc,'content_sha256':content_hash({k:v for k,v in doc.items() if k!='content_sha256'})}


def fixture(base):
    """원본 train sequence 두 chunk와 별도 validation sequence. 시간 의미는 부여하지 않는다."""
    inputs = base/'inputs'; create_fixture(inputs)
    pose = json.loads((inputs/'features.json').read_bytes())
    rgb = json.loads((inputs/'manifest.json').read_bytes())
    specs = []
    for name, split, offset in (('train_a','train',0),('train_b','train',2),('validation','validation',0)):
        p, r = copy.deepcopy(pose), copy.deepcopy(rgb)
        p['records'] = [row for row in p['records'] if row['split']==split]
        r['records'] = [row for row in r['records'] if row['split']==split]
        for row in p['records']: row['frame_index'] += offset
        for row in r['records']:
            row['frame_index'] += offset
            if offset:
                for side,view in row['views'].items():
                    original = inputs/view['path']
                    member = f"{row['sequence_id']}/fisheye_final_image/camera_{side}/final_{row['frame_index']}.png"
                    shutil.copyfile(original,inputs/member)
                    view['member'] = view['path'] = member
        r['id'] = 'rgb_'+name
        r['content_sha256'] = digest({k:v for k,v in r.items() if k!='content_sha256'})
        for file,doc in ((name+'_pose.json',p),(name+'_rgb.json',r)):
            (base/file).write_text(json.dumps(doc),encoding='utf8')
        specs.append({'id':name,'pose':name+'_pose.json','rgb_manifest':name+'_rgb.json','rgb_root':'inputs'})
    return specs


class RGBShardTests(unittest.TestCase):
    def setUp(self):
        target = ROOT/'outputs'/'test_tmp'; target.mkdir(parents=True,exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=target); self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.specs = fixture(self.base)
        self.catalog = build_catalog(self.specs,self.base)

    def test_same_sequence_disjoint_chunks_and_separate_validation(self):
        a,b,v = [r['sequences'][0] for r in self.catalog['shards']]
        self.assertEqual(a['sequence_id'],b['sequence_id'])
        self.assertEqual(a['motion_id'],b['motion_id'])
        self.assertEqual((a['first_frame'],a['last_frame'],b['first_frame'],b['last_frame']),(0,1,2,3))
        self.assertNotEqual(v['sequence_id'],a['sequence_id'])
        self.assertNotEqual(v['motion_id'],a['motion_id'])
        self.assertEqual(v['split'],'validation')

    def test_native_tensor_coordinates_and_annotation_masks_match_legacy(self):
        dataset = RGBShardDataset(self.catalog,self.base)
        for i,spec in enumerate(self.specs):
            split = 'validation' if i==2 else 'train'
            pose = json.loads((self.base/spec['pose']).read_bytes())
            loader = LocalStereoRGB(self.base/spec['rgb_manifest'],self.base/spec['rgb_root'])
            legacy = LocalRGBSupervision(loader,pose,expected_manifest_sha256=self.catalog['shards'][i]['rgb']['content_sha256'],
                joint_names=['root','tip'],split=split)
            actual = dataset.load(i,split).load_batch([0,1]); expected = legacy.load_batch([0,1])
            for key in ('images','gt_xy','annotation_valid'):
                self.assertTrue(torch.equal(actual[key],expected[key]),(i,key))
            self.assertEqual(tuple(actual['images'].shape),(2,2,3,6,8))
            self.assertEqual(actual['metadata'],expected['metadata'])
        self.assertEqual(actual['gt_xy'][1,1,1].tolist(),[9.,3.])
        self.assertTrue(actual['annotation_valid'][1,1,1])

    def test_one_shard_cache_and_lazy_png(self):
        with patch.object(LocalStereoRGB,'_decode_into',side_effect=AssertionError('metadata must not decode')):
            catalog = build_catalog(self.specs,self.base)
            dataset = RGBShardDataset(catalog,self.base)
            first = dataset.load(0,'train')
            self.assertIs(first,dataset.load(0,'train'))
            self.assertEqual(dataset._key,(0,'train'))
            del first
            dataset.load(1,'train')
            self.assertEqual(dataset._key,(1,'train'))
            report = dataset.verify_all()
            self.assertEqual(report['png_samples_verified'],0)
            self.assertFalse(report['full_png_verified'])
            self.assertIsNone(dataset._cached)
        report = dataset.verify_all(full_png=True)
        self.assertEqual(report['metadata_records'],{'train':4,'validation':2})
        self.assertEqual(report['png_samples_verified'],6)
        self.assertFalse(report['train_ready'])
        self.assertFalse(report['temporal_ready'])

    def test_png_corruption_is_lazy_and_full_inspection_rejects(self):
        rgb = json.loads((self.base/'train_a_rgb.json').read_bytes())
        path = self.base/'inputs'/rgb['records'][0]['views']['left']['path']
        raw = bytearray(path.read_bytes()); raw[-1] ^= 1; path.write_bytes(raw)
        catalog = build_catalog(self.specs,self.base)
        dataset = RGBShardDataset(catalog,self.base)
        self.assertFalse(dataset.verify_all()['full_png_verified'])
        with self.assertRaises(ValueError): dataset.verify_all(full_png=True)
        self.assertIsNone(dataset._cached)

    def test_descriptor_overlap_motion_split_and_duplicate_paths_rejected(self):
        for kind, error in (('overlap','rgb_shard_chunk_overlap'),('seq_motion','rgb_shard_sequence_motion_split_changed'),
                            ('seq_split','rgb_shard_sequence_motion_split_changed'),
                            ('motion_split','rgb_shard_motion_split_overlap'),('duplicate_path','rgb_shard_duplicate_metadata_path')):
            doc = copy.deepcopy(self.catalog)
            if kind=='overlap': doc['shards'][1]['sequences'][0].update(first_frame=1,last_frame=2)
            if kind=='seq_motion': doc['shards'][1]['sequences'][0]['motion_id']='other'
            if kind=='seq_split': doc['shards'][1]['sequences'][0]['split']='validation'
            if kind=='motion_split': doc['shards'][2]['sequences'][0]['motion_id']=doc['shards'][0]['sequences'][0]['motion_id']
            if kind=='duplicate_path': doc['shards'][1]['pose']['path']=doc['shards'][0]['pose']['path']
            with self.assertRaisesRegex(ValueError,error): validate_catalog(signed(doc))

    def test_payload_hash_key_and_version_mismatch(self):
        for kind in ('pose_hash','rgb_hash','key','version'):
            catalog = copy.deepcopy(self.catalog)
            file = self.base/self.specs[0]['pose']
            original = file.read_bytes()
            if kind=='pose_hash':
                file.write_bytes(original+b' ')
                error='rgb_shard_pose_hash'
            elif kind=='rgb_hash':
                catalog['shards'][0]['rgb']['sha256']='a'*64
                error='rgb_shard_rgb_file_hash'
            else:
                doc=json.loads(original)
                if kind=='key':
                    doc['records'][0]['frame_index']=9
                    error='shard_frame_order'
                else:
                    doc['manifest']['dataset_version']='other'
                    error='rgb_shard_pose_dataset_identity'
                file.write_text(json.dumps(doc),encoding='utf8')
                raw=file.read_bytes()
                catalog['shards'][0]['pose'].update(size_bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
            try:
                with self.assertRaisesRegex(ValueError,error): RGBShardDataset(signed(catalog),self.base).load(0,'train')
            finally: file.write_bytes(original)

    def test_limits_before_parsing_or_supervision_allocation(self):
        with patch('rewind_m1.rgb_shards.json.loads',side_effect=AssertionError('must reject before parse')):
            with self.assertRaisesRegex(ValueError,'shard_byte_limit'):
                build_catalog(self.specs,self.base,{**DEFAULT_LIMITS,'max_pose_bytes':1})
        with patch('rewind_m1.rgb_shards.LocalRGBSupervision',side_effect=AssertionError('must not allocate')):
            for key,limit in (('max_records',1),('max_payload_bytes',1),('max_catalog_bytes',10),('max_shards',1)):
                with self.assertRaises(ValueError):
                    build_catalog(self.specs,self.base,{**DEFAULT_LIMITS,key:limit})
        giant = self.base/'giant.json'; giant.write_bytes(b'x'*(DEFAULT_LIMITS['max_catalog_bytes']+1))
        with patch('rewind_m1.rgb_shards.json.loads',side_effect=AssertionError('must not parse giant catalog')):
            with self.assertRaisesRegex(ValueError,'shard_byte_limit'): read_catalog(giant)

    def test_incremental_catalog_limit_stops_before_later_inputs(self):
        maximum = len(json.dumps(self.catalog,sort_keys=True,separators=(',',':')).encode('utf8'))-30
        specs = self.specs+[{'id':'must_not_read','pose':'missing.json','rgb_manifest':'missing_rgb.json','rgb_root':'inputs'}]
        with self.assertRaisesRegex(ValueError,'rgb_shard_catalog_byte_limit'):
            build_catalog(specs,self.base,{**DEFAULT_LIMITS,'max_catalog_bytes':maximum})

    def test_sampler_tail_coverage_and_resume_bound_to_rgb_catalog(self):
        sampler = RGBEpochSampler(self.catalog,'train',71)
        seen = []
        shard, indices = sampler.peek(1); seen += [(shard,i) for i in indices]; sampler.commit(1)
        restored = RGBEpochSampler(self.catalog,'train',71); restored.load_state_dict(sampler.state_dict())
        self.assertEqual(restored.peek(3),sampler.peek(3))
        while restored.epoch==0:
            shard, indices = restored.peek(3)
            seen += [(shard,i) for i in indices]; restored.commit(3)
        self.assertEqual(len(seen),4); self.assertEqual(len(set(seen)),4)
        changed = copy.deepcopy(self.catalog); changed['shards'][0]['rgb']['sha256']='a'*64
        other = RGBEpochSampler(signed(changed),'train',71)
        with self.assertRaisesRegex(ValueError,'shard_cursor_identity'): other.load_state_dict(sampler.state_dict())

    def test_split_offsets_are_pose_order_not_rgb_order(self):
        p=json.loads((self.base/'train_a_pose.json').read_bytes())
        r=json.loads((self.base/'train_a_rgb.json').read_bytes())
        p['records'] += json.loads((self.base/'validation_pose.json').read_bytes())['records']
        r['records'] += json.loads((self.base/'validation_rgb.json').read_bytes())['records']
        r['records'].reverse()
        r['content_sha256']=digest({k:v for k,v in r.items() if k!='content_sha256'})
        (self.base/'train_a_pose.json').write_text(json.dumps(p),encoding='utf8')
        (self.base/'train_a_rgb.json').write_text(json.dumps(r),encoding='utf8')
        catalog=build_catalog(self.specs[:2],self.base)
        dataset=RGBShardDataset(catalog,self.base)
        train=dataset.load(0,'train')
        self.assertEqual([train.metadata(i)['pose_record']['frame_index'] for i in range(len(train))],[0,1])
        del train
        validation=dataset.load(0,'validation')
        self.assertEqual([validation.metadata(i)['pose_record']['frame_index'] for i in range(len(validation))],[0,1])
        sampler=RGBEpochSampler(catalog,'validation',4,shuffle=False)
        self.assertEqual(sampler.peek(4),(0,[0,1]))

    def test_cli_declarations_metadata_and_full_png_scopes(self):
        specs=self.base/'specs.json'; specs.write_text(json.dumps(self.specs),encoding='utf8')
        catalog=self.base/'catalog.json'
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(main(['build','--specs',str(specs),'--base',str(self.base),'--output',str(catalog)]),0)
        for flags, full in (([],False),(['--metadata'],False),(['--full-png'],True)):
            with contextlib.redirect_stdout(io.StringIO()) as out:
                self.assertEqual(main(['inspect','--catalog',str(catalog),*flags]),0)
            report=json.loads(out.getvalue()); self.assertEqual(report['full_png_verified'],full)
            self.assertFalse(report['train_ready'])
        self.assertEqual(read_catalog(catalog),self.catalog)


if __name__ == '__main__': unittest.main()
