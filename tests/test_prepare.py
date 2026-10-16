"""실제준비 converter의 단위·좌표·split·출처·mask·차단 계약 검사."""
import contextlib
import copy
import io
import json
from pathlib import Path
import tempfile
import unittest

from rewind_archive.contracts import digest
from rewind_prepare.converter import convert,sha,validate_candidate
from rewind_prepare.coordinates import camera_transform,camera_to_world,world_to_camera,project,determinant
from rewind_prepare.spec import validate_spec
from rewind_prepare.__main__ import main
from rewind_prepare.provenance import revision_identity

ROOT=Path(__file__).resolve().parents[1]


class PreparationTests(unittest.TestCase):
    def setUp(self):
        base=ROOT/'outputs/test_tmp';base.mkdir(parents=True,exist_ok=True)
        self.temp=tempfile.TemporaryDirectory(dir=base);self.addCleanup(self.temp.cleanup)
        self.base=Path(self.temp.name)
        self.raw=self.base/'files';self.raw.mkdir()
        self.spec=json.loads((ROOT/'configs/real_archvis62_v1.json').read_text())
        self.seq='Room/Day/person/glasses/001/Motion_A'
        self.other='Room/Day/person/glasses/002/Motion_B'
        self.splitdir=self.base/'split';self.splitdir.mkdir()
        for s,text in {'train':self.seq,'validation':self.other,'test':'Other/Day/person/glasses/003/Motion_C'}.items():
            raw=(text+'\n').encode();(self.splitdir/(s+'.txt')).write_bytes(raw)
            self.spec['pinned_sources']['split_sha256'][s]=sha(raw)
        self.calib={'size':[1024,1024],'image_center':[512.,512.],'polynomialW2C':[500.,100.],'affine':[1.,0.,0.]}
        self.calibration=self.base/'calibration.json';self.calibration.write_text(json.dumps(self.calib))
        self.spec['pinned_sources']['scaramuzza_sha256']=sha(self.calibration.read_bytes())
        self.specpath=self.base/'spec.json';self.save_spec()
        self.motion=self.base/'motion.json'
        self.motion_doc={'schema_version':'explicit_motion_map_v1','sequences':{self.seq:{'motion_id':'declared_test_a','identity_status':'declared_unverified','evidence':'test declaration, original motion unknown'}}}
        self.motion.write_text(json.dumps(self.motion_doc))
        self.manifest=self.base/'extraction.json'
        self.add_frame(self.seq,0)
        self.reseal()

    def save_spec(self):self.specpath.write_text(json.dumps(self.spec))

    def add_frame(self,seq,index):
        d={'frame':str(index),'joints':{},'device':{'open3d_trans':[0.,0.,0.],'open3d_rot':[0.,0.,0.]}}
        for side,sign in [('left',1),('right',-1)]:
            d['camera_'+side]={'trans':[0.,sign*6.,0.],'rot':[0.,0.,0.],'open3d_trans':[0.,-sign*6.,0.],
                              'open3d_rot':[0.,0.,0.],'device_pts3d':[sign*6.,0.,0.]}
        for k,name in enumerate(self.spec['joint_names']):
            dev=[k*.1-3.,5.+index,70.+k*.3]
            entry={'trans':[dev[2],dev[0],-dev[1]],'rot':[0.,0.,0.],'device_pts3d':dev}
            for side,sign in [('left',1),('right',-1)]:
                c=[dev[0]-sign*6.,dev[1],dev[2]]
                entry['camera_'+side+'_pts3d']=c
                entry['camera_'+side+'_pts2d']=project(c,self.calib)
            d['joints'][name]=entry
        path=self.raw/seq/'json'/f'frame_{index}.json';path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(d))
        return path

    def reseal(self):
        files=[]
        for p in sorted(self.raw.rglob('*.json')):
            files.append({'member':p.relative_to(self.raw).as_posix(),'sha256':sha(p.read_bytes()),'size_bytes':p.stat().st_size,'kind':'pose','tar_complete':True})
        em={'schema_version':'archive_extraction_v1','inspection_candidates_usable':True,'failure_class':'bounded_stop',
            'archive_manifest':{'kind':'pose','archive_id':'test_pose','dataset_version':'test_v1'},'files':files,'whole_archive_integrity_verified':False}
        em['content_sha256']=digest(em);self.manifest.write_text(json.dumps(em))

    def prepare(self):return convert(self.raw,self.manifest,self.splitdir,self.motion,self.specpath,self.calibration)

    def modify(self,index,fn):
        path=self.raw/self.seq/'json'/f'frame_{index}.json'
        doc=json.loads(path.read_text());fn(doc);path.write_text(json.dumps(doc));self.reseal()

    def test_adopted_spec_and_single_mm_conversion(self):
        candidate,report=self.prepare();r=candidate['records'][0]
        self.assertEqual(candidate['schema_version'],'real_pose_candidate_v1')
        self.assertEqual(candidate['contract']['data_kind'],'dataset_sample')
        self.assertEqual(candidate['contract']['distance_unit'],'mm')
        self.assertEqual(r['fields']['device_pts3d']['coordinates'][0],[-30.,50.,700.])
        self.assertEqual(len(r['observation_features']),868)
        self.assertEqual(r['observation_features'][3:7],[0,0,0,0])
        self.assertEqual(r['camera_pose']['left']['translation'],[-60.,0.,0.])
        self.assertAlmostEqual(determinant(r['camera_pose']['left']['rotation_matrix']),1.)
        self.assertFalse(report['train_ready']);self.assertFalse(candidate['training_eligibility']['train_ready'])
        self.assertIn('required_train_validation_splits_missing',report['blockers'])
        self.assertIn('rgb_pose_alignment_not_checked',report['blockers'])
        self.assertIn('original_motion_identity_unverified',report['blockers'])
        for change in ({'mm_per_raw_unit':1.},{'raw_unit':'mm'},{'transform':'no_reflection'}):
            altered=copy.deepcopy(self.spec);altered['coordinates'].update(change)
            with self.assertRaises(ValueError):validate_spec(altered)

    def test_nontrivial_rotation_roundtrip_and_center_is_not_extrinsic_t(self):
        import math
        camera={'trans':[12.,-6.,10.],'rot':[-35.,22.,11.],'open3d_rot':[11.,35.,-22.],
                'open3d_trans':[12.,6.,10.],'device_pts3d':[0.,0.,0.]}
        result,t=camera_transform(camera,10.,1e-6)
        point=[20.,18.,-7.]
        transformed=world_to_camera(point,result['rotation_matrix'],t)
        inverse=camera_to_world(transformed,result['rotation_matrix'],t)
        self.assertTrue(all(abs(a-b)<1e-12 for a,b in zip(point,inverse)))
        self.assertNotEqual(result['translation'],camera['trans'])
        self.assertAlmostEqual(determinant(result['rotation_matrix']),1.)

    def test_missing_wrist_local_mask_preserves_global_gt(self):
        self.modify(0,lambda d:d['joints']['hand_l'].update(device_pts3d=None))
        candidate,_=self.prepare();r=candidate['records'][0]
        self.assertEqual(r['hand_target']['valid_mask'][:20],[False]*20)
        self.assertTrue(all(r['hand_target']['valid_mask'][20:]))
        index=self.spec['joint_names'].index('index_01_l')
        self.assertTrue(r['fields']['device_pts3d']['valid_mask'][index])
        self.assertIsNone(r['fields']['device_pts3d']['coordinates'][self.spec['joint_names'].index('hand_l')])

    def test_missing_2d_and_invalid_field_keep_other_masks(self):
        self.modify(0,lambda d:d['joints']['pelvis'].update(camera_left_pts2d=[1.]))
        candidate,report=self.prepare();r=candidate['records'][0]
        self.assertFalse(r['fields']['camera_left_pts2d']['valid_mask'][0])
        self.assertIsNone(r['fields']['camera_left_pts2d']['coordinates'][0])
        self.assertTrue(r['fields']['device_pts3d']['valid_mask'][0])
        self.assertEqual(r['observation_features'][:7],[0.]*7)
        self.assertEqual(report['field_counts']['camera_left_pts2d_missing_or_invalid'],1)

    def test_negative_2d_and_nonpositive_depth_are_valid_not_clipped(self):
        def mutate(d):
            joint=d['joints']['pelvis'];joint['device_pts3d']=[-20.,0.,-100.];joint['trans']=[-100.,-20.,0.]
            for side,offset in [('left',6.),('right',-6.)]:
                point=[-20.-offset,0.,-100.]
                joint['camera_'+side+'_pts3d']=point;joint['camera_'+side+'_pts2d']=project(point,self.calib)
        self.modify(0,mutate)
        candidate,report=self.prepare();record=candidate['records'][0]
        self.assertLess(record['fields']['camera_left_pts2d']['coordinates'][0][0],0.)
        self.assertLess(record['observation_features'][0],-1.)
        self.assertTrue(record['fields']['camera_left_pts2d']['valid_mask'][0])
        self.assertTrue(record['fields']['device_pts3d']['valid_mask'][0])
        self.assertEqual(record['visibility']['left'][0],None)
        self.assertEqual(report['field_counts']['negative_2d_joint_views'],2)

    def test_frame_gaps_reset_and_future_change_does_not_affect_previous(self):
        self.add_frame(self.seq,1);self.add_frame(self.seq,4);self.reseal()
        before,_=self.prepare()
        self.modify(4,lambda d:d.update(timestamp='uninterpreted-original-time'))
        after,report=self.prepare()
        for old,new in zip(before['records'][:2],after['records'][:2]):
            old['source'].pop('extraction_manifest_sha256');new['source'].pop('extraction_manifest_sha256')
            self.assertEqual(old,new)  # global manifest hash는 바뀌지만 모델 입력/좌표는 불변이다.
        self.assertEqual([r['time']['history_reset'] for r in after['records']],[True,False,True])
        self.assertEqual(after['records'][2]['time']['gap_before'],2)
        self.assertEqual(after['records'][2]['timestamp_raw'],'uninterpreted-original-time')
        self.assertIsNone(after['records'][2]['time']['time_seconds'])
        self.assertEqual(report['sequence_summary'][self.seq]['contiguous_runs'],[[0,1],[4,4]])

    def test_hash_pins_manifest_tamper_and_duplicate_frames_reject(self):
        original_split=(self.splitdir/'train.txt').read_bytes()
        (self.splitdir/'train.txt').write_text(self.seq+'\nchanged')
        with self.assertRaisesRegex(ValueError,'split_hash'):self.prepare()
        (self.splitdir/'train.txt').write_bytes(original_split)
        raw=next(self.raw.rglob('*.json'));raw.write_text(raw.read_text()+' ')
        with self.assertRaisesRegex(ValueError,'payload_hash'):self.prepare()
        self.reseal()
        (raw.parent/'frame_00.json').write_bytes(raw.read_bytes());self.reseal()
        with self.assertRaisesRegex(ValueError,'duplicate_sequence_frame'):self.prepare()

    def test_split_motion_overlap_and_missing_explicit_id_reject(self):
        self.add_frame(self.other,0);self.reseal()
        with self.assertRaisesRegex(ValueError,'motion_map'):self.prepare()
        self.motion_doc['sequences'][self.other]=copy.deepcopy(self.motion_doc['sequences'][self.seq])
        self.motion.write_text(json.dumps(self.motion_doc))
        with self.assertRaisesRegex(ValueError,'motion_split_overlap'):self.prepare()

    def test_side_scale_coordinate_and_baseline_mismatch_reject(self):
        for kind in ('side','scale','baseline','device'):
            with self.subTest(kind=kind):
                raw=next(self.raw.rglob('*.json'));original=raw.read_bytes()
                def mutate(d):
                    if kind=='side':
                        j=d['joints']['pelvis'];j['camera_left_pts3d'],j['camera_right_pts3d']=j['camera_right_pts3d'],j['camera_left_pts3d']
                    elif kind=='scale':d['joints']['pelvis']['device_pts3d']=[x*10 for x in d['joints']['pelvis']['device_pts3d']]
                    elif kind=='baseline':d['camera_left']['trans'][1]=20
                    else:d['device']['open3d_rot'][0]=10
                self.modify(0,mutate)
                with self.assertRaises(ValueError):self.prepare()
                raw.write_bytes(original);self.reseal()

    def test_candidate_integrity_and_ready_promotion_reject(self):
        candidate,_=self.prepare()
        modified=copy.deepcopy(candidate);modified['records'][0]['fields']['device_pts3d']['coordinates'][0][0]+=1
        with self.assertRaisesRegex(ValueError,'hash'):validate_candidate(modified)
        modified=copy.deepcopy(candidate);modified['training_eligibility']['train_ready']=True
        modified['content_sha256']=digest({k:v for k,v in modified.items() if k!='content_sha256'})
        with self.assertRaisesRegex(ValueError,'ready_flag'):validate_candidate(modified)

    def test_budgets_and_no_overwrite_or_raw_output(self):
        self.spec['limits']['max_files']=1;self.save_spec();self.add_frame(self.seq,1);self.reseal()
        with self.assertRaisesRegex(ValueError,'file_count_limit'):self.prepare()
        self.spec['limits']['max_files']=256;self.spec['limits']['max_total_input_bytes']=10;self.save_spec()
        with self.assertRaisesRegex(ValueError,'input_byte_limit'):self.prepare()
        self.spec['limits']['max_total_input_bytes']=33554432;self.save_spec()
        args=['--input',str(self.raw),'--extraction-manifest',str(self.manifest),'--split-dir',str(self.splitdir),
              '--motion-map',str(self.motion),'--spec',str(self.specpath),'--calibration',str(self.calibration)]
        out=self.base/'prepared'
        with contextlib.redirect_stdout(io.StringIO()):self.assertEqual(main(args+['--output',str(out)]),0)
        original=(out/'candidate.json').read_bytes()
        with contextlib.redirect_stderr(io.StringIO()):
            for target in (out,self.raw/'prepared'):
                with self.assertRaises(SystemExit):main(args+['--output',str(target)])
        self.assertEqual((out/'candidate.json').read_bytes(),original)

    def test_cli_failure_report_has_no_candidate_and_raw_is_unchanged(self):
        raw=next(self.raw.rglob('*.json'));before=raw.read_bytes()
        self.calibration.write_text('{}')
        out=self.base/'rejected'
        with contextlib.redirect_stderr(io.StringIO()),self.assertRaises(SystemExit):
            main(['--input',str(self.raw),'--extraction-manifest',str(self.manifest),'--split-dir',str(self.splitdir),
                  '--motion-map',str(self.motion),'--spec',str(self.specpath),'--calibration',str(self.calibration),'--output',str(out)])
        self.assertEqual(json.loads((out/'report.json').read_text())['status'],'rejected')
        self.assertFalse((out/'candidate.json').exists());self.assertEqual(before,raw.read_bytes())

    def test_existing_training_schema_guard_rejects_real_candidate(self):
        try:
            from rewind_m1.data import load_document
            from rewind_m2.data import load_document as load_m2
            from rewind_m3.data import prepare as load_m3
        except ImportError:self.skipTest('PyTorch guard 확인은 별도 환경 필요')
        candidate,_=self.prepare()
        for loader,args in [(load_document,(candidate,4)),(load_m2,(candidate,4,100.)),(load_m3,(candidate,4,100.))]:
            with self.assertRaisesRegex(ValueError,'m1_features_v1'):loader(*args)

    def test_source_zip_revision_stamp_and_unversioned_hash_scope(self):
        stamp=self.base/'SOURCE_REVISION.txt';stamp.write_text('a'*40)
        result=revision_identity(self.base)
        self.assertEqual(result['revision'],'a'*40)
        self.assertEqual(result['revision_kind'],'source_revision_stamp')
        self.assertIsNone(result['dirty'])
        stamp.write_text('wrong')
        with self.assertRaisesRegex(ValueError,'stamp_invalid'):revision_identity(self.base)


if __name__=='__main__':unittest.main()
