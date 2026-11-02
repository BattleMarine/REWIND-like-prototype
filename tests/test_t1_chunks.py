"""원본 sequence를 유지한 독립T1 특징 chunk. 기존 temporal 정책과 구분한다."""
import contextlib
import copy
import io
import json
from pathlib import Path
import tempfile
import unittest

import torch

from test_shard_training import equal
from rewind_m1.data import content_hash, load_document, synthetic_document
from rewind_m1.prediction_join import join_predictions
from rewind_m1.readiness import inspect_catalog, main as readiness_main
from rewind_m1.shards import (DEFAULT_LIMITS, SCHEMA, T1_SCHEMA, T1_POLICY, EpochSampler, ShardDataset,
    build_catalog, build_t1_catalog, validate_catalog)
from rewind_m1.shard_training import ShardTrainer
from rewind_m1.shard_cli import export_checkpoint, main, read_checkpoint
from rewind_m1.training import save_checkpoint

ROOT=Path(__file__).resolve().parents[1]


def signed(doc): return {**doc,'content_sha256':content_hash({k:v for k,v in doc.items() if k!='content_sha256'})}


def feature_fixture(base):
    """합성 예측 cache를 기존 join에 넣는다. 실제 신경망 정확도 결과가 아니다."""
    paths=[]
    for name,sequence,frames in (('train_a','train_0',range(3)),('train_b','train_0',range(3,6)),
                                  ('validation','validation_0',range(3))):
        pose=synthetic_document()
        pose['records']=[r for r in pose['records'] if r['sequence_id']==sequence and r['frame_index'] in frames]
        if name=='train_a': pose['records'][1]['joints']['tip']['device_pts3d']=None
        names=pose['contract']['joint_names']; records=[]
        for row in pose['records']:
            pred={'joint_names':names,'frontend_version':'fixed_synthetic_prediction_v1',
                  'frontend_checkpoint_sha256':'a'*64,'preprocessing_version':'synthetic_predicted_native_v1'}
            for side in ('left','right'):
                pred[side]={'coordinates':[[v+.25 for v in row['joints'][n]['camera_'+side+'_pts2d']] for n in names],
                            'valid_mask':[True]*len(names),'confidence':[.5]*len(names),'visibility':[None]*len(names)}
            records.append({'sequence_id':sequence,'frame_index':row['frame_index'],'prediction':pred})
        manifest_hash=content_hash({'fixture':name,'frames':list(frames)})
        cache=signed({'schema_version':'predicted_2d_cache_v1','data_kind':'synthetic_fixture','input_source':'predicted_2d',
             'manifest_hash':manifest_hash,'image_sizes':[[64,48],[64,48]],'joint_names':names,
             'frontend_version':'fixed_synthetic_prediction_v1','frontend_checkpoint_sha256':'a'*64,
             'preprocessing_version':'synthetic_predicted_native_v1','records':records,
             'record_splits':[r['split'] for r in pose['records']]})
        doc=join_predictions(pose,cache,manifest_hash)
        path=name+'.json'; (base/path).write_text(json.dumps(doc),encoding='utf8'); paths.append(path)
    return paths


class T1ChunkTests(unittest.TestCase):
    def setUp(self):
        parent=ROOT/'outputs'/'test_tmp'; parent.mkdir(parents=True,exist_ok=True)
        self.temp=tempfile.TemporaryDirectory(dir=parent); self.addCleanup(self.temp.cleanup)
        self.base=Path(self.temp.name); self.paths=feature_fixture(self.base)
        self.catalog=build_t1_catalog(self.paths,self.base)
        self.config={'model':'single_frame','window_size':1,'hidden':8,'dropout':.2,'batch_size':2,
            'learning_rate':.005,'seed':73,'target_scale_mm':100.,'amp':False,'scheduler_step':2,'scheduler_gamma':.8}

    def test_distinct_schema_policy_and_legacy_rejects_multiple_chunks(self):
        self.assertEqual(self.catalog['schema_version'],T1_SCHEMA)
        self.assertEqual(self.catalog['window_size'],1)
        with self.assertRaisesRegex(ValueError,'shard_sequence_crosses_shards'):
            build_catalog(self.paths,self.base,1)
        sampler=EpochSampler(self.catalog,'train',73)
        self.assertEqual(sampler.state_dict()['policy'],T1_POLICY)
        old=build_catalog([self.paths[0],self.paths[2]],self.base,1)
        self.assertEqual(old['schema_version'],SCHEMA)
        self.assertEqual(EpochSampler(old,'train',73).state_dict()['policy'],EpochSampler.POLICY)

    def test_original_keys_prediction_targets_masks_provenance_preserved(self):
        dataset=ShardDataset(self.catalog,self.base)
        for i,path in enumerate(self.paths):
            raw=(self.base/path).read_bytes(); doc=json.loads(raw); expected=load_document(doc,1)
            actual=dataset.load(i)
            self.assertEqual(actual[0],expected[0]); self.assertEqual(actual[2],content_hash(doc))
            for split in expected[1]:
                a,b=actual[1][split],expected[1][split]
                self.assertEqual(a.anchors,b.anchors)
                for key in ('features','time_valid','target_mm','target_valid'):
                    self.assertTrue(torch.equal(getattr(a,key),getattr(b,key)))
            self.assertEqual((self.base/path).read_bytes(),raw)
            self.assertIn('prediction_join',str(doc['manifest']))
        self.assertFalse(dataset.load(0)[1]['train'].target_valid[1,0,1])

    def test_overlap_motion_split_and_prediction_contract_rejected(self):
        for kind,error in (('overlap','t1_chunk_overlap'),('motion','t1_sequence_motion_split_changed'),
                           ('split','t1_sequence_motion_split_changed'),('motion_leak','shard_motion_split_overlap'),
                           ('window','t1_requires_window_one')):
            doc=copy.deepcopy(self.catalog)
            if kind=='overlap': doc['shards'][1]['sequences'][0].update(first_frame=2,last_frame=4)
            if kind=='motion': doc['shards'][1]['sequences'][0]['motion_id']='other_motion'
            if kind=='split': doc['shards'][1]['sequences'][0]['split']='validation'
            if kind=='motion_leak': doc['shards'][2]['sequences'][0]['motion_id']=doc['shards'][0]['sequences'][0]['motion_id']
            if kind=='window': doc['window_size']=2
            with self.assertRaisesRegex(ValueError,error): validate_catalog(signed(doc))
        path=self.base/self.paths[1]; original=path.read_bytes(); doc=json.loads(original)
        doc['contract']['frontend_checkpoint_sha256']='b'*64; path.write_text(json.dumps(doc),encoding='utf8')
        with self.assertRaisesRegex(ValueError,'shard_contract_mismatch'): build_t1_catalog(self.paths,self.base)

    def test_trainer_and_export_require_single_frame_window_one(self):
        for config in ({**self.config,'model':'temporal'},{**self.config,'window_size':2}):
            with self.assertRaisesRegex(ValueError,'t1_requires_single_frame_window_one'):
                ShardTrainer(config,self.catalog,self.base)
        trainer=ShardTrainer(self.config,self.catalog,self.base)
        checkpoint=copy.deepcopy(trainer.checkpoint()); checkpoint['config']['model']='temporal'
        with self.assertRaisesRegex(ValueError,'t1_requires_single_frame_window_one'):
            export_checkpoint(checkpoint,'c'*64,self.base/'must_not_exist.pt')
        self.assertFalse((self.base/'must_not_exist.pt').exists())

    def test_incremental_catalog_budget_stops_before_later_files(self):
        maximum=len(json.dumps(self.catalog,sort_keys=True,separators=(',',':')).encode())-30
        with self.assertRaisesRegex(ValueError,'shard_catalog_byte_limit'):
            build_t1_catalog(self.paths+['must_not_read.json'],self.base,{**DEFAULT_LIMITS,'max_catalog_bytes':maximum})

    def test_t1_resume_exact_across_chunk_epoch_and_tail(self):
        trainer=ShardTrainer(self.config,self.catalog,self.base); paths={}
        for _ in range(9):
            trainer.train_step()
            if trainer.step in (1,2,4):
                path=self.base/f'cut_{trainer.step}.pt'; save_checkpoint(trainer.checkpoint(),path); paths[trainer.step]=path
        expected=copy.deepcopy(trainer.checkpoint())
        self.assertEqual(expected['samples_seen'],14)
        self.assertEqual(expected['cursor']['epoch'],2)
        for cut,path in paths.items():
            restored=ShardTrainer(self.config,self.catalog,self.base); restored.resume(read_checkpoint(path)[0])
            while restored.step<9: restored.train_step()
            self.assertTrue(equal(expected,restored.checkpoint()),cut)

    def test_explicit_cli_build_train_resume_readiness_export(self):
        catalog_path=self.base/'catalog.json'
        with contextlib.redirect_stdout(io.StringIO()):
            args=['build-t1','--base',str(self.base),'--output',str(catalog_path)]
            for p in self.paths: args += ['--shard',p]
            self.assertEqual(readiness_main(args),0)
        report=inspect_catalog(catalog_path,True)
        self.assertEqual(report['supported_path'],'local_m1_t1_feature_chunks')
        self.assertFalse(report['temporal_ready']); self.assertFalse(report['full_training_ready'])
        config=self.base/'config.json'; config.write_text(json.dumps(self.config),encoding='utf8')
        output=self.base/'run'
        with contextlib.redirect_stdout(io.StringIO()):
            main(['train','--config',str(config),'--catalog',str(catalog_path),'--output',str(output),'--steps','1'])
            main(['train','--config',str(config),'--catalog',str(catalog_path),'--output',str(output),'--resume',str(output/'last.pt'),
                  '--epochs','1','--export-weights',str(output/'weights.pt')])
            main(['evaluate','--checkpoint',str(output/'last.pt'),'--catalog',str(catalog_path),'--output',str(output/'eval.json')])
        state,_=read_checkpoint(output/'last.pt'); self.assertEqual((state['step'],state['samples_seen']),(4,6))

    def test_t1_real_declarations_still_do_not_release_training(self):
        from rewind_m1.data import VERIFICATIONS
        catalog=copy.deepcopy(self.catalog)
        catalog['contract'].update(data_kind='dataset_sample',verified=dict.fromkeys(VERIFICATIONS,True),
                                   verification_evidence=dict.fromkeys(VERIFICATIONS,'declared_only'))
        with self.assertRaisesRegex(ValueError,'shard_dataset_sample_release_not_implemented'):
            ShardTrainer(self.config,signed(catalog),self.base)


if __name__=='__main__': unittest.main()
