"""공통 frame 선언과 기존 image 경로 호환. Decoder/학습/원격 실행 없음."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

from rewind_archive.contracts import digest
from rewind_m1.frame_input import from_image_metadata,validate_frame_input


def metadata(sequence='Fixture/Day/person/avatar/001/Motion',size=None):
    archive={'schema_version':'archive_manifest_v1','archive_id':'fixture','dataset_version':'fixture_v1','kind':'rgb',
        'source_scope':'synthetic_fixture','total_parts':1,'parts':[{'index':0,'name':'archive000','size_bytes':1,'version':'fixture',
        'source':{'type':'local','path':'archive000'},'identity':{'mtime_ns':0}}]}
    return {'data_kind':'synthetic_fixture','dataset_version':'fixture_v1','sources':{'fixture':{'archive_manifest':archive,'extraction_manifest_sha256':'a'*64}},
        'record':{'sequence_id':sequence,'frame_index':7,'split':'train','split_evidence':'fixture only',
        'views':{side:{'image_size':size or [8,6],'member':sequence+'/fisheye_final_image/camera_'+side+'/final_7.png',
            'sha256':str(i+1)*64,'size_bytes':128,'source_id':'fixture'} for i,side in enumerate(('left','right'))}}}


def signed(record):
    value=copy.deepcopy(record);value['content_sha256']=digest({k:v for k,v in value.items() if k!='content_sha256'})
    return value


def timing(pts):return {'status':'provided','timestamp_seconds':None,'pts':pts,'time_base':{'numerator':1,'denominator':90000},'clock_id':'fixture_clock','reason':None}


class FrameContractTests(unittest.TestCase):
    def setUp(self):self.record=from_image_metadata(metadata())

    def test_image_unknown_values_not_generated_from_frame_number(self):
        record=validate_frame_input(self.record)
        for view in record['views'].values():
            self.assertEqual(view['timing']['status'],'unknown')
            for key in ('timestamp_seconds','pts','time_base','clock_id'):self.assertIsNone(view['timing'][key])
            self.assertTrue(view['timing']['reason'])
        self.assertEqual(record['mode'],'independent_frame');self.assertFalse(record['temporal_ready']);self.assertFalse(record['train_ready'])
        self.assertIsNone(record['geometry']['camera_pose']['left']['matrix'])

    def test_provided_pts_is_validated_without_materializing_seconds(self):
        for side in ('left','right'):self.record['views'][side]['timing']=timing(90001)
        self.record['stereo_sync']={'status':'provided','evidence':'declared same hardware clock','max_skew_seconds':.001}
        result=validate_frame_input(signed(self.record))
        self.assertIsNone(result['views']['left']['timing']['timestamp_seconds'])
        self.assertFalse(result['temporal_ready'])

    def test_missing_view_wrong_side_member_identity_and_hash_rejected(self):
        changes=[lambda d:d['views'].pop('right'),lambda d:d['views']['right'].update(side='left'),
            lambda d:d['views']['right']['source_identity'].update(member=d['views']['left']['source_identity']['member']),
            lambda d:d['views']['left']['source_identity'].update(payload_sha256='wrong'),
            lambda d:d['views']['left']['source_identity'].update(member='./'+d['views']['left']['source_identity']['member'])]
        for change in changes:
            doc=copy.deepcopy(self.record);change(doc)
            with self.assertRaises(ValueError):validate_frame_input(signed(doc))
        doc=copy.deepcopy(self.record);doc['frame_index']=8
        with self.assertRaises(ValueError):validate_frame_input(doc)
        doc=copy.deepcopy(self.record);doc['content_sha256']='0'*64
        with self.assertRaisesRegex(ValueError,'frame_input_hash'):validate_frame_input(doc)

    def test_bad_or_unexplained_time_rejected(self):
        for change in (lambda t:t.update(timestamp_seconds=0),lambda t:t.update(reason=None)):
            doc=copy.deepcopy(self.record);change(doc['views']['left']['timing'])
            with self.assertRaises(ValueError):validate_frame_input(signed(doc))
        changes=[(lambda t:t.update(pts=True),'pts_timebase'),(lambda t:t.update(pts=1.25),'pts_timebase'),
            (lambda t:t.update(time_base=None),'pts_timebase_pair'),
            (lambda t:t['time_base'].update(denominator=0),'pts_timebase'),
            (lambda t:t['time_base'].update(numerator=-1),'pts_timebase'),
            (lambda t:t.update(timestamp_seconds=3),'time_disagreement'),(lambda t:t.update(clock_id=''),'time_status')]
        for change,reason in changes:
            doc=copy.deepcopy(self.record);doc['views']['left']['timing']=timing(90000);change(doc['views']['left']['timing'])
            with self.assertRaisesRegex(ValueError,'frame_input_'+reason):validate_frame_input(signed(doc))
        doc=copy.deepcopy(self.record);doc['views']['left']['timing']=timing(90000)
        doc['views']['left']['timing']['timestamp_seconds']=float('inf')
        # 비유한 입력은 canonical hash를 만들 수 없으므로 시간 validator의 오류를 직접 확인한다.
        with self.assertRaisesRegex(ValueError,'frame_input_timestamp'):validate_frame_input(doc)
        doc=copy.deepcopy(self.record);doc['views']['left']['timing']=timing(1000000000)
        doc['views']['left']['timing'].update(timestamp_seconds=1000000000.5,time_base={'numerator':1,'denominator':1})
        with self.assertRaisesRegex(ValueError,'frame_input_time_disagreement'):validate_frame_input(signed(doc))

    def test_clock_mismatch_and_excess_stereo_skew_rejected(self):
        for side in ('left','right'):self.record['views'][side]['timing']=timing(90000)
        self.record['stereo_sync']={'status':'provided','evidence':'fixture only','max_skew_seconds':.0001}
        self.record['views']['right']['timing']['clock_id']='other'
        with self.assertRaisesRegex(ValueError,'sync_mismatch'):validate_frame_input(signed(self.record))
        self.record['views']['right']['timing']=timing(91000)
        with self.assertRaisesRegex(ValueError,'sync_mismatch'):validate_frame_input(signed(self.record))

    def test_geometry_declarations_and_reflected_matrix(self):
        pose=self.record['geometry']['camera_pose']['left']
        pose.update(status='provided',matrix=[[-1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]],direction='world_to_camera',
                    translation_unit='mm',coordinate_frame='reflected_world_F',reason=None)
        result=validate_frame_input(signed(self.record));self.assertEqual(result['geometry']['camera_pose']['left']['matrix'][0][0],-1)
        pose['direction']='unknown'
        with self.assertRaisesRegex(ValueError,'pose_contract'):validate_frame_input(signed(self.record))
        pose['direction']='world_to_camera';pose['matrix'][3]=[0,0,1,0]
        with self.assertRaisesRegex(ValueError,'pose_matrix'):validate_frame_input(signed(self.record))

    def test_legacy_path_and_image_dimension_ranges_preserved(self):
        result=from_image_metadata(metadata(sequence='L'*600,size=[262144,4]))
        self.assertEqual(len(result['sequence_id']),600);self.assertEqual(result['views']['left']['image_size'],[262144,4])

    def test_video_identity_is_metadata_only_never_temporal_ready(self):
        for index,side in enumerate(('left','right')):
            self.record['views'][side]['source_identity']={'kind':'video_stream','asset_sha256':'c'*64,'stream_index':index,'source_frame_index':12,'decoder_version':'declaration_only'}
        result=validate_frame_input(signed(self.record));self.assertFalse(result['temporal_ready'])
        self.record['temporal_ready']=True
        with self.assertRaisesRegex(ValueError,'independent_only'):validate_frame_input(signed(self.record))


try:
    import torch
    from PIL import Image
except (ImportError,OSError):HAS_IMAGES=False
else:HAS_IMAGES=True


@unittest.skipUnless(HAS_IMAGES,'torch와 Pillow 필요')
class ImageFrameIntegrationTests(unittest.TestCase):
    def test_loader_metadata_batch_and_lazy_compatibility(self):
        from unittest.mock import patch
        from rewind_m1.local_fixture import create_fixture
        from rewind_m1.local_rgb import LocalStereoRGB
        root=Path(__file__).resolve().parents[1]/'outputs/frame_input_20261102/developer/test_tmp';root.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=root) as temporary:
            directory=Path(temporary)/'fixture';create_fixture(directory)
            manifest=directory/'manifest.json'
            # create_fixture의 실제 반환 manifest 이름을 고정 계약에서 사용한다.
            if not manifest.exists():manifest=directory/'rgb_manifest.json'
            loader=LocalStereoRGB(manifest,directory)
            original=copy.deepcopy(loader._records[0])
            with patch.object(loader,'_decode_into',side_effect=AssertionError('metadata must not decode')):
                meta=loader.metadata(0);validate_frame_input(meta['frame_input'])
            self.assertEqual(meta['record'],original)
            meta['frame_input']['views']['left']['timing']['reason']='changed'
            self.assertNotEqual(loader.metadata(0)['frame_input']['views']['left']['timing']['reason'],'changed')
            one=loader[0];batch=loader.load_batch([0,1])
            self.assertEqual(set(one),{'images','metadata'});self.assertEqual(set(batch),{'images','metadata'})
            self.assertTrue(torch.equal(one['images'],batch['images'][0]))
            for item in batch['metadata']:self.assertFalse(validate_frame_input(item['frame_input'])['temporal_ready'])


if __name__=='__main__':unittest.main()
