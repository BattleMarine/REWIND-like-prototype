"""M1 합성 계약 검사. 실제 dataset·Colab을 사용하지 않는다."""

import contextlib
import copy
import io
import json
import tempfile
import unittest
import hashlib
from unittest.mock import patch
from pathlib import Path

try:
    import torch
except ImportError:
    torch = None

if torch is not None:
    from rewind_data.window_fixture import synthetic_frame
    from rewind_data.windows import CausalWindowStream
    from rewind_m1.data import InputContract, DataContractError, load_document, pack_window, synthetic_document, content_hash
    from rewind_m1.frontend import StereoHeatmapNet, heatmap_loss, prediction_cache, read_prediction_cache, weights_hash
    from rewind_m1.metrics import masked_mse, metric_sums
    from rewind_m1.models import make_model
    from rewind_m1.training import Trainer, read_checkpoint, save_checkpoint, source_identity, require_torch_api
    from rewind_m1.__main__ import main
    from rewind_m1.predicted_fixture import prepare_fixture


@unittest.skipIf(torch is None,"M1 학습 검증에는 별도 PyTorch 환경 필요 (M0는 stdlib)")
class M1Tests(unittest.TestCase):
    def setUp(self):
        self.doc=synthetic_document()
        self.config=json.loads((Path(__file__).resolve().parents[1]/'configs/m1_single_synthetic.json').read_text())
        self.temp_root=Path(__file__).resolve().parents[1]/'outputs'/'test_tmp'
        self.temp_root.mkdir(parents=True,exist_ok=True)
        self.temp=tempfile.TemporaryDirectory(dir=self.temp_root)
        self.addCleanup(self.temp.cleanup)
        self.base=Path(self.temp.name).resolve()
        self.assertTrue(self.base.is_relative_to(self.temp_root.resolve()))

    def test_tensor_padding_mask_and_fixed_normalization(self):
        contract,splits,_=load_document(self.doc,3)
        d=splits['train']
        self.assertEqual(tuple(d.features.shape),(12,3,28))
        self.assertEqual(d.time_valid[0].tolist(),[False,False,True])
        self.assertEqual(d.features[0,:2].abs().sum().item(),0)
        self.assertFalse(d.target_valid[0,:2].any())
        self.assertAlmostEqual(d.features[0,-1,0].item(),2*8/63-1,places=6)
        # GT confidence/visibility는 unknown이며 known bit로 분리한다.
        self.assertEqual(d.features[0,-1,3:7].tolist(),[0,0,0,0])
        self.assertEqual(contract.input_source,'gt_2d')

    def test_real_data_and_unknown_units_are_gated(self):
        for change in ({'data_kind':'dataset_sample'},{'distance_unit':'unknown'},{'image_sizes':[[0,48],[64,48]]}):
            d=copy.deepcopy(self.doc);d['contract'].update(change)
            with self.assertRaises(DataContractError): load_document(d,3)
        self.doc['schema_version']='raw_json'
        with self.assertRaises(DataContractError): load_document(self.doc,3)

    def test_confirmed_unit_conversion_and_split_overlap(self):
        baseline=load_document(self.doc,3)[1]['train'].target_mm
        cm=copy.deepcopy(self.doc);cm['contract']['distance_unit']='cm'
        self.assertTrue(torch.equal(load_document(cm,3)[1]['train'].target_mm,baseline*10))
        overlap=copy.deepcopy(self.doc)
        overlap['records'][-1]['motion_id']=overlap['records'][0]['motion_id']
        with self.assertRaisesRegex(DataContractError,'split 중복'): load_document(overlap,3)

    def test_missing_2d_does_not_invalidate_3d(self):
        self.doc['records'][0]['joints']['root']['camera_left_pts2d']=None
        d=load_document(self.doc,3)[1]['train']
        self.assertEqual(d.features[0,-1,2].item(),0)
        self.assertTrue(d.target_valid[0,-1,0])
        self.doc['records'][0]['joints']['root']['device_pts3d']=None
        d=load_document(self.doc,3)[1]['train']
        self.assertFalse(d.target_valid[0,-1,0])
        self.assertEqual(d.target_mm[0,-1,0].tolist(),[0,0,0])

    def test_masked_nan_loss_reduction_gradient_and_empty_batch(self):
        p=torch.tensor([[[3.,4.,0.],[float('nan')]*3]],requires_grad=True)
        y=torch.zeros_like(p); y[0,1]=float('nan')
        m=torch.tensor([[True,False]])
        loss=masked_mse(p,y,m,1.)
        self.assertAlmostEqual(float(loss.detach()),25/3,places=5)
        loss.backward()
        self.assertEqual(p.grad[0,1].tolist(),[0,0,0])
        self.assertAlmostEqual(metric_sums(p.detach(),y,m)['distance_sum_mm'],5.)
        for scale in (0,-1,float('inf'),True):
            with self.assertRaises(ValueError): masked_mse(p,y,m,scale)
        with self.assertRaisesRegex(ValueError,'유효 GT'): masked_mse(p,y,torch.zeros_like(m),1.)
        with self.assertRaises(ValueError): metric_sums(p,y,m.long())

    def test_small_gradient_and_overfit_both_models(self):
        for kind in ('single_frame','temporal'):
            cfg={**self.config,'model':kind,'dropout':0.}
            t=Trainer(cfg)
            initial=t.evaluate('train')['normalized_mse']
            for _ in range(60): t.train_step()
            self.assertLess(t.evaluate('train')['normalized_mse'],initial*.1)
            self.assertTrue(any(p.grad is not None and p.grad.abs().sum()>0 for p in t.model.parameters()))

    def test_checkpoint_resume_is_bitwise_equal_with_rng_and_scheduler(self):
        full=Trainer(self.config)
        for _ in range(14): full.train_step()
        expected=copy.deepcopy(full.checkpoint())
        part=Trainer(self.config)
        for _ in range(6): part.train_step()
        path=self.base/'resume.pt';save_checkpoint(part.checkpoint(),path)
        resumed=Trainer(self.config);resumed.resume(read_checkpoint(path))
        for _ in range(8): resumed.train_step()
        actual=resumed.checkpoint()
        for k,v in expected['model'].items(): self.assertTrue(torch.equal(v,actual['model'][k]),k)
        for key in ('scheduler','scaler','step','samples_seen','config'): self.assertEqual(expected[key],actual[key])
        self.assertTrue(torch.equal(expected['rng']['torch_cpu'],actual['rng']['torch_cpu']))
        self.assertEqual(expected['rng']['python'],actual['rng']['python'])
        for i,state in expected['optimizer']['state'].items():
            for k,v in state.items(): self.assertTrue(torch.equal(v,actual['optimizer']['state'][i][k]))

    def test_resume_rejects_manifest_config_and_code_changes(self):
        trainer=Trainer(self.config);checkpoint=copy.deepcopy(trainer.checkpoint())
        for key in ('manifest_hash','config','source'):
            altered=copy.deepcopy(checkpoint)
            if key=='source': altered[key]['code_sha256']['rewind_data/windows.py']='bad'
            else: altered[key]='bad'
            with self.assertRaises(ValueError): trainer.resume(altered)
        self.assertIn('rewind_data/windows.py',source_identity()['code_sha256'])
        self.assertIn('rewind_data/audit.py',source_identity()['code_sha256'])

    def test_temporal_causality_and_padding(self):
        contract,splits,_=load_document(self.doc,3)
        cfg={**self.config,'model':'temporal','dropout':0.}
        model=make_model(cfg,28,2).eval()
        x,time=splits['train'].features[:2].clone(),splits['train'].time_valid[:2]
        before=model(x,time);x[:,-1]+=100
        after=model(x,time)
        self.assertTrue(torch.equal(before[:,:-1],after[:,:-1]))
        x=splits['train'].features[:2].clone();x[~time]=999
        self.assertTrue(torch.equal(model(x,time),before))

    def test_fixed_window_streaming_reset_matches_batch_model(self):
        contract=InputContract.from_dict(self.doc['contract'])
        cfg={**self.config,'model':'temporal','dropout':0.}
        model=make_model(cfg,28,2).eval()
        frames=[synthetic_frame('A',i) for i in (0,1,4)]+[synthetic_frame('B',0)]
        state=CausalWindowStream(3,contract.joint_names)
        windows=[state.push(f) for f in frames]
        batch=torch.stack([pack_window(w,contract)[0] for w in windows])
        valid=torch.stack([pack_window(w,contract)[1] for w in windows])
        expected=model(batch,valid)[:,-1]
        for i,w in enumerate(windows):
            x,t,_,_=pack_window(w,contract)
            self.assertTrue(torch.allclose(model(x[None],t[None])[0,-1],expected[i],atol=1e-6))
        state.reset();fresh=pack_window(state.push(frames[0]),contract)
        self.assertTrue(torch.allclose(model(fresh[0][None],fresh[1][None])[0,-1],expected[0]))

    def test_prediction_cache_provenance_and_no_gt_fallback(self):
        model=StereoHeatmapNet(2)
        images=torch.rand(2,2,3,48,64)
        manifest_hash='b'*64
        cache=prediction_cache(model,images,[('A',0),('A',1)],['root','tip'],'cnn_v1','rgb_v1',manifest_hash,'synthetic_fixture')
        cfg={**self.doc['contract'],'input_source':'predicted_2d','frontend_version':'cnn_v1',
            'frontend_checkpoint_sha256':weights_hash(model),'preprocessing_version':'rgb_v1'}
        contract=InputContract.from_dict(cfg)
        observations=read_prediction_cache(cache,contract,manifest_hash)
        window=CausalWindowStream(3,contract.joint_names).push(synthetic_frame('A',0))
        x,_,_,_=pack_window(window,contract,observations)
        self.assertEqual(x[-1,4].item(),1.) # 예측 confidence known
        self.assertEqual(x[-1,6].item(),0.) # visibility unknown
        with self.assertRaises(DataContractError): pack_window(window,contract,{})
        changed=copy.deepcopy(cache);changed['records'][0]['prediction']['left']['confidence'][0]=.9
        with self.assertRaises(DataContractError): read_prediction_cache(changed,contract,manifest_hash)
        with self.assertRaises(DataContractError): read_prediction_cache(cache,contract,'different_manifest')
        bad=copy.deepcopy(cache);bad['records'][0]['prediction']['frontend_version']='other'
        bad['content_sha256']=content_hash({k:v for k,v in bad.items() if k!='content_sha256'})
        with self.assertRaises(DataContractError): read_prediction_cache(bad,contract,manifest_hash)

    def test_predicted_features_document_loads_only_explicit_predictions(self):
        d=copy.deepcopy(self.doc);weights='a'*64
        d['contract'].update(input_source='predicted_2d',frontend_version='fixture_pred',frontend_checkpoint_sha256=weights)
        for r in d['records']:
            r['prediction']={'joint_names':['root','tip'],'frontend_version':'fixture_pred','frontend_checkpoint_sha256':weights,
                'preprocessing_version':d['contract']['preprocessing_version']}
            for side in ('left','right'):
                r['prediction'][side]={'coordinates':[[2.,3.],[4.,5.]],'valid_mask':[True,True],
                    'confidence':[.7,.2],'visibility':[None,None]}
        self.assertEqual(load_document(d,3)[0].input_source,'predicted_2d')
        del d['records'][0]['prediction']
        with self.assertRaises(DataContractError): load_document(d,3)

    def test_frontend_heatmap_mask_preserves_3d_and_gradient(self):
        model=StereoHeatmapNet(2);images=torch.rand(1,2,3,32,32)
        logits,xy,confidence=model(images)
        self.assertEqual(tuple(xy.shape),(1,2,2,2));self.assertTrue((confidence>=0).all())
        targets=torch.tensor([[[[5.,6.],[-10.,20.]],[[5.,6.],[float('nan'),0.]]]])
        annotation=torch.tensor([[[True,True],[True,False]]]);before=annotation.clone()
        loss,supervision=heatmap_loss(logits,targets,annotation,(32,32))
        self.assertEqual(supervision.tolist(),[[[True,False],[True,False]]])
        self.assertTrue(torch.equal(annotation,before))
        loss.backward();self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters()))
        with self.assertRaises(DataContractError): heatmap_loss(logits,targets,torch.zeros_like(annotation),(32,32))
        for sigma in (float('nan'),float('inf'),0):
            with self.assertRaises(DataContractError): heatmap_loss(logits,targets,annotation,(32,32),sigma)

    def test_training_test_tuning_and_invalid_anchor_rejected(self):
        trainer=Trainer(self.config)
        with self.assertRaisesRegex(ValueError,'최종 평가'): trainer.evaluate('test')
        bad=copy.deepcopy(self.doc)
        for joint in bad['records'][0]['joints'].values(): joint['device_pts3d']=None
        with self.assertRaisesRegex(ValueError,'전부 누락'): Trainer(self.config,bad)

    def test_model_cache_document_regression_chain_train_only(self):
        weights_path=self.base/'frontend_weights.pt'
        doc,cache,manifest,report=prepare_fixture(2,weights_path)
        weights=torch.load(weights_path,weights_only=True)
        restored=StereoHeatmapNet(weights['model_config']['joints'])
        restored.load_state_dict(weights['model'])
        self.assertTrue(weights['weights_only'])
        self.assertEqual(weights_hash(restored),cache['frontend_checkpoint_sha256'])
        self.assertEqual(weights['frontend_version'],cache['frontend_version'])
        self.assertEqual(report['frontend_training_splits'],['train'])
        self.assertTrue(report['frontend_weights_fixed']);self.assertTrue(report['gt3d_and_mask_preserved'])
        self.assertEqual(cache['manifest_hash'],content_hash(manifest))
        trainer=Trainer(self.config,doc)
        trainer.train_step()
        result=trainer.evaluate()
        self.assertEqual(result['input_source'],'predicted_2d')
        self.assertEqual(result['evaluation_kind'],'synthetic_code_check')
        without_prediction=copy.deepcopy(doc);del without_prediction['records'][0]['prediction']
        with self.assertRaises(DataContractError): Trainer(self.config,without_prediction)

    def test_cli_resume_history_and_evaluation_provenance(self):
        config=self.base/'config.json';config.write_text(json.dumps(self.config))
        out=self.base/'run'
        args=['train','--config',str(config),'--output',str(out)]
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(main(args+['--steps','4']),0)
            original_hash=hashlib.sha256((out/'last.pt').read_bytes()).hexdigest()
            self.assertEqual(main(args+['--steps','7','--resume',str(out/'last.pt')]),0)
        report=json.loads((out/'report.json').read_text())
        self.assertEqual([(s['initial_step'],s['final_step']) for s in report['segments']],[(0,4),(4,7)])
        self.assertEqual([p['step'] for p in report['log']],[4,7])
        self.assertEqual(report['segments'][-1]['resume_checkpoint_sha256'],original_hash)
        self.assertNotEqual(original_hash,hashlib.sha256((out/'last.pt').read_bytes()).hexdigest())
        evaluation=self.base/'evaluation.json'
        with contextlib.redirect_stdout(io.StringIO()): main(['evaluate','--checkpoint',str(out/'last.pt'),'--output',str(evaluation)])
        result=json.loads(evaluation.read_text())
        self.assertEqual(result['metrics']['evaluation_kind'],'synthetic_code_check')
        self.assertEqual(len(result['checkpoint_sha256']),64)
        self.assertEqual(result['checkpoint_step'],7)

    def test_cli_periodic_report_and_stale_report_recovery(self):
        config=self.base/'config.json';config.write_text(json.dumps(self.config))
        out=self.base/'run'
        args=['train','--config',str(config),'--output',str(out)]
        with contextlib.redirect_stdout(io.StringIO()): main(args+['--steps','4'])
        old_report=(out/'report.json').read_bytes()
        # 주기 checkpoint 저장 후 report 쓰기 직전에 끊기는 상황을 만든다.
        import rewind_m1.__main__ as cli
        real_save=cli.save_report
        def interrupt_report(report,path):
            self.assertEqual(report['step'],10)
            self.assertEqual(read_checkpoint(out/'last.pt')['step'],10)
            raise RuntimeError('simulated interruption')
        with patch.object(cli,'save_report',side_effect=interrupt_report), contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(RuntimeError,'simulated'):
                main(args+['--steps','14','--resume',str(out/'last.pt')])
        self.assertEqual((out/'report.json').read_bytes(),old_report)
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(main(args+['--steps','14','--resume',str(out/'last.pt')]),0)
        report=json.loads((out/'report.json').read_text())
        recovered=report['segments'][-2]
        self.assertTrue(recovered['report_recovery'])
        self.assertEqual(recovered['missing_log_steps'],{'after':4,'through':10})
        self.assertEqual(recovered['log'],[])
        full=Trainer(self.config)
        for _ in range(14): full.train_step()
        actual=read_checkpoint(out/'last.pt')
        for k,v in full.checkpoint()['model'].items(): self.assertTrue(torch.equal(v,actual['model'][k]))
        # 다음 구간에서는 정상 주기 report와 checkpoint가 같은 위치로 저장된다.
        observed=[]
        def inspect_report(report,path):
            observed.append(report['step'])
            self.assertEqual(read_checkpoint(out/'last.pt')['step'],report['step'])
            real_save(report,path)
        with patch.object(cli,'save_report',side_effect=inspect_report), contextlib.redirect_stdout(io.StringIO()):
            main(args+['--steps','21','--resume',str(out/'last.pt')])
        self.assertIn(20,observed)
        self.assertEqual(json.loads((out/'report.json').read_text())['step'],21)

    def test_minimum_api_guard_and_source_zip_revision(self):
        # 버전 표기 검사이며 새 PyTorch 실행 검증을 뜻하지 않는다.
        with patch.object(torch,'__version__','2.8.0'): require_torch_api()
        with patch.object(torch,'__version__','2.2.0'):
            with self.assertRaises(ValueError): require_torch_api()
        import rewind_m1.training as training
        stamp=self.base/'SOURCE_REVISION.txt';stamp.write_text('a'*40)
        (self.base/'rewind_m1').mkdir();(self.base/'rewind_data').mkdir()
        (self.base/'rewind_m1'/'module.py').write_text('pass\n')
        with patch.object(training,'ROOT',self.base):
            source=source_identity()
            self.assertEqual(source['revision'],'a'*40)
            self.assertEqual(len(source['code_sha256']['rewind_m1/module.py']),64)
            stamp.write_text('invalid')
            with self.assertRaises(ValueError): source_identity()

    def test_predicted_fixture_cli_checkpoint_resumes_same_output(self):
        from rewind_m1.predicted_fixture import main as predicted_main
        config=self.base/'config.json';config.write_text(json.dumps(self.config))
        out=self.base/'predicted'
        with contextlib.redirect_stdout(io.StringIO()):
            predicted_main(['--config',str(config),'--output',str(out),'--frontend-steps','2','--steps','3'])
            original_hash=hashlib.sha256((out/'last.pt').read_bytes()).hexdigest()
            main(['train','--config',str(config),'--output',str(out),'--dataset',str(out/'features.json'),
                '--resume',str(out/'last.pt'),'--steps','5'])
        report=json.loads((out/'report.json').read_text())
        self.assertEqual([(s['initial_step'],s['final_step']) for s in report['segments']],[(0,3),(3,5)])
        self.assertEqual(report['segments'][-1]['resume_checkpoint_sha256'],original_hash)
        self.assertEqual(report['validation']['input_source'],'predicted_2d')
        frontend=json.loads((out/'frontend_report.json').read_text())
        self.assertTrue(frontend['frontend_weights_only'])


if __name__=='__main__': unittest.main()
