"""Small meaningful fixtures for pairing, anchors, resumption and result restoration."""

import ast
from copy import deepcopy
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import patch

import torch

from rewind_m1.training import rng_state
from rewind_prepare.converter import sha
from rewind_v1 import camera_training as train,camera_training_recovery as recovery,camera_training_eval as evaluation,camera_training_data as feature_data
from rewind_archive.contracts import canonical,digest
from rewind_prepare.temporal_selection import seal
from rewind_v1.data import Clip
from rewind_v1.run_data import FixtureData
from scripts.build_v15_camera_training import notebook_document


ROOT=Path(__file__).resolve().parents[1]


class CameraTrainingTests(TestCase):
    def setUp(self):
        (ROOT/'outputs/test_tmp').mkdir(parents=True,exist_ok=True)
        tmp=tempfile.TemporaryDirectory(dir=ROOT/'outputs/test_tmp');self.addCleanup(tmp.cleanup);self.root=Path(tmp.name)
        self.cfg=deepcopy(train.POLICY);self.cfg.update(updates_per_teacher=2,milestones=[0,1,2],checkpoint_every=1)
        self.cfg['teacher'].update(hidden=8,heads=2,graph_layers=1,temporal_layers=1,diffusion_steps=20,sampling_steps=2)
        self.cfg['limits'].update(train_seconds=240,checkpoint_bytes=8*1024**2,output_bytes=64*1024**2)
        self.spec=FixtureData().spec;self.identity={'feature_identity':'a'*64};self.clips={};self.meta={}
        for seq,start in [('train_a',0),('train_b',50)]:
            key=(seq,start);features=torch.arange(4*56,dtype=torch.float32).reshape(1,4,56)/100
            self.clips[key]=Clip(features,torch.zeros(1,4,2,9),torch.arange(48,dtype=torch.float32).reshape(1,4,4,3),
                torch.ones(1,4,4,dtype=torch.bool),torch.ones(1,4,dtype=torch.bool),torch.arange(4,dtype=torch.float64)[None]/25,
                'predicted_2d',input_sha256='a'*64)
            self.meta[key]={'sequence':seq,'start':start,'frames':list(range(start,start+4)),
                'rotation':torch.eye(3,dtype=torch.float64).expand(4,2,3,3).tolist(),
                'translation_mm':[[[-1000.-i,60.,20.],[-1000.-i,-60.,20.]] for i in range(4)]}

    def make(self,name):
        (self.root/name).parent.mkdir(parents=True,exist_ok=True)
        return train.PairedTraining(self.cfg,self.spec,self.clips,self.meta,self.identity,'cpu',self.root/name,fixture=True)

    def test_origin_shift_stream_save_reset_and_no_future(self):
        m=self.meta['train_a',0];origin=train.CameraOrigin('first_midpoint')
        whole,audit=origin.apply(m['rotation'],m['translation_mm'],m['sequence'],m['start'],m['frames'])
        shifted=train.shifted_metadata(m,[100000,0,0]);other=train.CameraOrigin('first_midpoint')
        shifted_camera,_=other.apply(shifted['rotation'],shifted['translation_mm'],m['sequence'],0,m['frames'])
        torch.testing.assert_close(whole,shifted_camera,atol=1e-6,rtol=0)
        stream=train.CameraOrigin('first_midpoint');a,_=stream.apply(m['rotation'][:2],m['translation_mm'][:2],m['sequence'],0,[0,1])
        restored=train.CameraOrigin('first_midpoint',json.loads(json.dumps(stream.state)))
        b,_=restored.apply(m['rotation'][2:],m['translation_mm'][2:],m['sequence'],0,[2,3])
        self.assertTrue(torch.equal(whole,torch.cat((a,b))))
        with self.assertRaises(ValueError):restored.apply(m['rotation'],m['translation_mm'],'new_sequence',0,m['frames'])
        restored.reset();restored.apply(m['rotation'],m['translation_mm'],'new_sequence',0,m['frames'])
        self.assertEqual(audit['state']['first_frame'],0)

    def test_same_initial_state_same_noise_and_pair_failure_rollback(self):
        runner=self.make('paired');before={a:train.model_fingerprint(runner.stages[a]['body'].model) for a in train.ARMS}
        self.assertEqual(len(set(before.values())),1);rng=train.rng_fingerprint()
        with patch.object(runner.stages['first_midpoint']['body'],'update',side_effect=RuntimeError('half pair')):
            with self.assertRaisesRegex(RuntimeError,'half pair'):runner.update_pair()
        self.assertEqual(runner.position['total_pairs'],0);self.assertEqual(train.rng_fingerprint(),rng)
        self.assertEqual(before,{a:train.model_fingerprint(runner.stages[a]['body'].model) for a in train.ARMS})
        observed=[]
        for arm in train.ARMS:
            original=runner.stages[arm]['body']._update
            def wrapped(*args,_original=original,**kwargs):observed.append(train.rng_fingerprint());return _original(*args,**kwargs)
            self.enterContext(patch.object(runner.stages[arm]['body'],'_update',side_effect=wrapped))
        runner.update_pair();self.assertEqual(observed[0],observed[1])

    def test_interrupt_after_cursor_commit_rolls_back_history_and_position(self):
        runner=self.make('cursor');before=train.model_fingerprint(runner.stages['absolute']['body'].model)
        class InterruptHistory(list):
            def append(self,row):super().append(row);raise KeyboardInterrupt('after cursor')
        runner.history=InterruptHistory()
        with self.assertRaises(KeyboardInterrupt):runner.update_pair()
        self.assertEqual(runner.position,{'role':'body','pair':0,'total_pairs':0});self.assertEqual(runner.history,[])
        self.assertEqual(train.model_fingerprint(runner.stages['absolute']['body'].model),before)

    def test_partial_snapshot_is_not_committed_and_resume_recreates_both(self):
        runner=self.make('snapshot');original=runner.publish
        def fail_second(path,*args,**kwargs):
            if path.name.startswith('first_midpoint-'):raise RuntimeError('second snapshot failed')
            return original(path,*args,**kwargs)
        with patch.object(runner,'publish',side_effect=fail_second):result=runner.run(stop_after_pairs=0)
        self.assertEqual(result['status'],'failed');self.assertEqual(runner.snapshots,{})
        restored=self.make('snapshot');restored.resume(runner.last['sha256']);restored.run(stop_after_pairs=0)
        self.assertEqual(set(restored.snapshots),{'absolute/body/0','first_midpoint/body/0'})

    def test_body_boundary_hand_midpoint_and_complete_resume_exact(self):
        full=self.make('full');expected=full.run();self.assertEqual(expected['status'],'complete')
        part=self.make('part');part.run(stop_after_pairs=1)
        restored=self.make('part');restored.resume(part.last['sha256']);restored.run(stop_after_pairs=2)
        self.assertEqual(restored.position['role'],'hand');self.assertEqual(restored.position['pair'],0)
        hand=self.make('part');hand.resume(restored.last['sha256']);hand.run(stop_after_pairs=3)
        self.assertEqual(hand.position['role'],'hand');self.assertEqual(hand.position['pair'],1)
        final=self.make('part');final.resume(hand.last['sha256']);final.run()
        for arm in train.ARMS:
            for role in ('body','hand'):
                self.assertEqual(train.model_fingerprint(full.stages[arm][role].model),train.model_fingerprint(final.stages[arm][role].model))
        complete=self.make('part');complete.resume(final.last['sha256'])
        self.assertEqual(complete.position['role'],'complete')
        for arm in train.ARMS:
            self.assertEqual(complete.stages[arm]['hand'].step,2)
            self.assertEqual(complete.stages[arm]['hand'].spec['body_checkpoint_sha256'],complete.snapshots[f'{arm}/body/2']['sha256'])

    def test_resume_environment_anchor_and_missing_snapshot_rejected(self):
        run=self.make('resume');run.run(stop_after_pairs=1)
        with patch.object(torch,'__version__','different-version'):
            other=self.make('resume')
            with self.assertRaisesRegex(ValueError,'resume_binding'):other.resume(run.last['sha256'])
        other=self.make('resume');other.anchors[next(iter(other.anchors))]['anchor_world_mm'][0]+=1
        with self.assertRaisesRegex(ValueError,'resume_binding'):other.resume(run.last['sha256'])
        path=run.root/next(iter(run.snapshots.values()))['name'];path.unlink()
        with self.assertRaises(Exception):self.make('resume').resume(run.last['sha256'])

    def test_recovery_restores_checkpoint_and_features_excludes_raw(self):
        run=self.make('run/train');run.run(stop_after_pairs=1)
        root=self.root/'run';(root/'features').mkdir();(root/'features/train-00.pt').write_bytes(b'derived-only')
        (root/'raw_train_cache').mkdir();(root/'raw_train_cache/original.zip').write_bytes(b'raw')
        archive=recovery.package(root,self.root/'result.zip');target=self.root/'fresh-runtime'
        result=recovery.restore(Path(archive['archive']),archive['sha256'],target)
        self.assertEqual(result['status'],'verified');self.assertTrue((target/'train/last.pt').is_file())
        self.assertTrue((target/'features/train-00.pt').is_file());self.assertFalse((target/'raw_train_cache').exists())
        relocated=train.PairedTraining(self.cfg,self.spec,self.clips,self.meta,self.identity,'cpu',target/'train',fixture=True)
        relocated.resume(run.last['sha256']);self.assertEqual(relocated.position,run.position)

    def test_notebook_defaults_compile_and_bounded_interrupt(self):
        doc=notebook_document({})
        for c in doc['cells']:
            if c['cell_type']!='code':continue
            src=''.join(c['source']);compile(src,c['id'],'exec')
            if c['id']!='bootstrap':exec(src,{'stage':lambda *a:self.fail('default stage'),'execute':lambda *a:self.fail('default execute')})
        text=''.join(doc['cells'][1]['source']);self.assertIn('signal.SIGINT',text);self.assertIn('timeout=60',text)
        self.assertIn('finally:',text);self.assertIn("['package'",text);self.assertIn("['publish'",text)

    def test_pilot_resets_models_and_rng_for_main(self):
        pilot=self.make('pilot');initial_rng=pilot.initial_rng;initial=deepcopy(pilot.initial);pilot.run()
        main=self.make('main')
        self.assertEqual(main.initial,initial);self.assertEqual(main.initial_rng,initial_rng)
        self.assertNotEqual(train.model_fingerprint(pilot.stages['absolute']['body'].model),main.initial['body'])

    def test_cumulative_budget_does_not_reset_on_resume(self):
        runner=self.make('budget');runner.elapsed=self.cfg['limits']['train_cumulative_seconds']
        result=runner.run();self.assertEqual(result['status'],'cumulative_budget_exhausted')
        other=self.make('budget');other.resume(runner.last['sha256']);again=other.run()
        self.assertEqual(again['position']['total_pairs'],0);self.assertEqual(again['elapsed_training_seconds'],self.cfg['limits']['train_cumulative_seconds'])

    def test_operational_stage_failure_always_retains_once_no_retry(self):
        code=''.join(notebook_document({})['cells'][1]['source']);node=next(x for x in ast.parse(code).body if isinstance(x,ast.FunctionDef) and x.name=='stage')
        for fail in (None,'run','package','publish'):
            calls=[]
            def execute(args,timeout):
                calls.append(args[0])
                if args[0]==fail:raise RuntimeError('fixture '+fail)
                return {'attempt':'attempt'} if args[0]=='backup-prepare' else {'archive':'local.zip','sha256':'f'*64,'bytes':10} if args[0]=='package' else {'status':'complete'}
            root=self.root/('op-'+str(fail));root.mkdir();drive=self.root/'drive';drive.mkdir(exist_ok=True)
            env={'execute':execute,'RUN':root,'WORK':self.root,'INPUTS':self.root,'EVIDENCE':self.root,'DRIVE':drive,'DEVICE':'cpu',
                'MODEL_LOCAL':self.root/'model.zip','RECOVERY':self.root,'Path':Path,'json':json,'uuid':__import__('uuid'),
                'POLICY':train.POLICY,'GAUGE':{'model_drive_relative':'model','rgb_drive_relative':'rgb','pose_drive_relative':'pose'},
                'PINS':{'source_code_sha256':'f'*64,'reference':{'name':'city.zip'},'control_reference':{'name':'control.zip'}}}
            exec(compile(ast.Module(body=[node],type_ignores=[]),'stage-fixture','exec'),env)
            with patch.dict('sys.modules',{'google':SimpleNamespace(),'google.colab':SimpleNamespace(drive=SimpleNamespace(mount=lambda *a:None))}),patch.object(Path,'is_mount',return_value=True):
                if fail:
                    with self.assertRaises(RuntimeError):env['stage']('train',7200)
                else:env['stage']('train',7200)
            self.assertEqual(calls.count('package'),1);self.assertEqual(calls.count('publish'),0 if fail=='package' else 1)

    def test_fresh_runtime_eval_raw_rebuild_reuses_features(self):
        root=self.root/'restored_eval';root.mkdir();entry={'sequence':'fixed','frames':[{'frame':0,'files':{'left':{'sha256':'f'*64}}}]}
        doc=seal({'schema':'v15_eval_features_v1','source_sha256':digest(train.source_identity()),'clips':[{'entry':entry}]})
        (root/'eval_features.json').write_bytes(canonical(doc))
        with patch.object(evaluation.gauge,'fixed_plan',return_value={'content_sha256':'a'*64}), \
                patch.object(evaluation.cross,'prepare',return_value={'clips':[entry]}), \
                patch.object(evaluation.infer.view,'_models',side_effect=AssertionError('no F9 feature recomputation')):
            result=evaluation.prepare_eval(self.root,{},self.root/'model',self.root,self.root,root,torch.device('cpu'))
        self.assertEqual(result,doc);self.assertEqual((root/'eval_features.json').read_bytes(),canonical(doc))

    def test_selected_part_restore_never_fetches_unselected_parts_and_checks_reuse(self):
        # Stub only historical metadata decoding; real bounded copy/hash/reuse paths execute.
        starts=[['seq-'+str(i),0] for i in range(20)];index={};records=[];parts=[];source=self.root/'source-cache';source.mkdir()
        for i in range(65):
            raw=('part-'+str(i)).encode();part={'name':f'part-{i:04d}.zip','bytes':len(raw),'sha256':sha(raw)};parts.append(part)
            if i<20:(source/part['name']).write_bytes(raw)
        for i,(seq,start) in enumerate(starts):
            for frame in range(50):
                row={'key':[seq,frame],'split':'train','files':{r:{'path':f'{seq}/{frame}/{r}'} for r in ('left','right','pose')}}
                records.append(row);index[seq,frame]=(parts[i]['name'],row)
        plan={'content_sha256':feature_data.HISTORICAL['selection_sha256'],'train_clip_starts':starts,'records':records}
        manifest={'selection_sha256':plan['content_sha256'],'parts':parts};(source/feature_data.MANIFEST).write_text(json.dumps(manifest))
        cache=SimpleNamespace(manifest=manifest,index=index)
        local=self.root/'partial-cache'
        with patch.object(feature_data,'load_inputs',return_value=(plan,{})),patch.object(feature_data,'validate_bundle',return_value=({}, {}, [])), \
                patch.object(feature_data,'RawCache',return_value=cache),patch.object(feature_data,'CACHE_PART_BYTES',sum(p['bytes'] for p in parts)):
            report=feature_data.restore_train_parts(self.root,self.root,source,local,self.root/'restore.json')
            self.assertEqual(len(report['parts']),20);self.assertEqual(len(list(local.glob('part-*.zip'))),20)
            self.assertEqual(json.loads((local/feature_data.MANIFEST).read_bytes()),manifest)
            reused=feature_data.restore_train_parts(self.root,self.root,source,local,self.root/'reuse.json')
            self.assertEqual(reused['copied_bytes'],0);self.assertEqual(reused['verified_local_bytes'],report['copied_bytes'])
            p=local/parts[0]['name'];p.write_bytes(b'x'*p.stat().st_size)
            with self.assertRaises(ValueError):feature_data.restore_train_parts(self.root,self.root,source,local,self.root/'bad.json')

    def test_final_eval_paired_shift_prefix_and_separate_train_fit(self):
        runner=self.make('final');runner.run();root=self.root/'evaldata';root.mkdir();rows=[]
        for number in range(6):
            meta=deepcopy(self.meta['train_a',0]);meta.update(sequence='regression_'+str(number),frames=list(range(50)))
            meta['rotation']=torch.eye(3,dtype=torch.float64).expand(50,2,3,3).tolist()
            meta['translation_mm']=[[[-1000.-i,60.,20.],[-1000.-i,-60.,20.]] for i in range(50)]
            camera,_=train.CameraOrigin('absolute').apply(meta['rotation'],meta['translation_mm'],meta['sequence'],0,meta['frames'])
            tensors={'features':torch.arange(50*56,dtype=torch.float32).reshape(50,56)/1000,'camera':camera,
                'target':torch.arange(50*4*3,dtype=torch.float32).reshape(50,4,3)/10,'target_valid':torch.ones(50,4,dtype=torch.bool),
                'left_xy':torch.zeros(50,4,2),'right_xy':torch.zeros(50,4,2),'left_xy_valid':torch.ones(50,4,dtype=torch.bool),'right_xy_valid':torch.ones(50,4,dtype=torch.bool)}
            pin=feature_data._save_tensor(root/f'eval-{number:02d}.pt',tensors,4*1024**2)
            rows.append({'metadata':meta,'entry':{'number':number,'sequence':meta['sequence'],'frames':[{'frame':f} for f in range(50)]},'tensor':pin})
        doc=seal({'schema':'v15_eval_features_v1','source_sha256':digest(train.source_identity()),'clips':rows})
        (root/'eval_features.json').write_bytes(canonical(doc))
        def render(folder,*args,**kwargs):
            from PIL import Image
            p=folder/'comparison.gif';Image.new('RGB',(8,8),'blue').save(p,format='GIF')
            return [{'name':p.name,'bytes':p.stat().st_size,'sha256':sha(p.read_bytes())}]
        with patch.object(evaluation.gauge,'render_comparison',side_effect=render):
            result=evaluation.evaluate(runner,root,self.root/'evaluation',expected_pin=doc['content_sha256'])
        self.assertEqual(len(result['clips']),6);self.assertEqual(result['train_fit']['key'],list(runner.keys[0]))
        for clip in result['clips']:
            centered=clip['arms']['first_midpoint'];self.assertTrue(centered['normalized_input_within_tolerance'])
            self.assertEqual(centered['paired_shift_status'],'within_repeat_aware_tolerance')
            self.assertIn('body_centroid_bias_mean_mm',centered['centroid'])
