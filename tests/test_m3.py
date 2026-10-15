"""손목 상대 계약·고정 body·조건 누출·cascade causality와 완전 재개 검사."""
import contextlib
import copy
import io
import json
from pathlib import Path
import tempfile
import unittest

try:
    import torch
except ImportError:
    torch = None

if torch is not None:
    from rewind_m1.training import save_checkpoint,rng_state
    from rewind_m1.frontend import weights_hash
    from rewind_m2.training import Trainer as BodyTrainer
    from rewind_m2.streaming import anchor_noise
    from rewind_m3.data import synthetic_document,body_document,prepare,DataContractError
    from rewind_m3.cascade import reconstruct,BodyCache,prediction_hash,UpperCondition,FrozenBody
    from rewind_m3.models import BoundHand
    from rewind_m3.metrics import cascade_metrics
    from rewind_m3.training import Trainer,read_checkpoint
    from rewind_m3.streaming import CascadeStream
    from rewind_m3.__main__ import main
    from rewind_m3.fixture import main as fixture_main

ROOT = Path(__file__).resolve().parents[1]


@unittest.skipIf(torch is None,'M3 검증에는 별도 PyTorch 환경 필요')
class M3Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        (ROOT/'outputs/test_tmp').mkdir(parents=True,exist_ok=True)
        cls.shared = tempfile.TemporaryDirectory(dir=ROOT/'outputs/test_tmp')
        cls.body_path = Path(cls.shared.name)/'body.pt'
        cls.doc = synthetic_document()
        body_cfg = json.loads((ROOT/'configs/m3_body_synthetic.json').read_text())
        body = BodyTrainer(body_cfg,body_document(cls.doc))
        for _ in range(30): body.train_step()
        save_checkpoint(body.checkpoint(),cls.body_path)

    @classmethod
    def tearDownClass(cls):
        cls.shared.cleanup()

    def setUp(self):
        self.config = json.loads((ROOT/'configs/m3_hand_upper_synthetic.json').read_text())
        self.temp = tempfile.TemporaryDirectory(dir=ROOT/'outputs/test_tmp')
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)

    def trainer(self,**changes):
        return Trainer({**self.config,**changes},self.body_path,self.doc)

    def test_named_coordinate_contract_missing_wrist_and_roles(self):
        contract,roles,fingers,splits,_,_,_,_ = prepare(self.doc,4,100.)
        self.assertEqual(contract.joint_names,tuple(roles['body_joint_names']+fingers))
        self.assertEqual(splits['train'].coordinate_frame,'wrist_relative_parallel_device')
        self.assertEqual(self.doc['contract']['coordinate_frame'],'device_relative')
        self.assertEqual(splits['train'].target_mm[0,-1,0].tolist(),[-5,5,2])
        missing = copy.deepcopy(self.doc)
        missing['records'][0]['joints']['wrist_l']['device_pts3d'] = None
        d = prepare(missing,4,100.)[3]['train']
        self.assertEqual(d.target_valid[0,-1].tolist(),[False,False,True,True])
        self.assertTrue(d.global_target_valid[0,-1].all())
        self.assertEqual(d.target_mm[0,-1,:2].abs().sum(),0)
        for change in ('duplicate','wrong_left','upper_finger','false_coordinate'):
            bad = copy.deepcopy(self.doc)
            if change=='duplicate': bad['m3_contract']['wrists']['right'] = 'wrist_l'
            if change=='wrong_left': bad['m3_contract']['hands']['left'][0] = 'wrist_r'
            if change=='upper_finger': bad['m3_contract']['upper_joint_names'].append('index_l')
            if change=='false_coordinate': bad['contract']['coordinate_frame'] = 'wrist_relative'
            with self.assertRaises(DataContractError,msg=change): prepare(bad,4,100.)

    def test_reconstruct_left_right_masks_unique_count_and_body_wrist_error(self):
        t = self.trainer()
        d = t.splits['validation']
        body,local = d.body_target_mm[:,-1],d.target_mm[:,-1]
        whole,global_fingers = reconstruct(body,local,t.roles,t.fingers)
        self.assertEqual(whole.shape,(12,10,3))
        self.assertTrue(torch.equal(global_fingers,d.global_target_mm[:,-1]))
        metrics = cascade_metrics(body,local,global_fingers,d,t.roles,t.fingers)
        self.assertEqual(metrics['whole_body_unique']['valid_joints'],120)
        self.assertEqual(metrics['whole_body_unique']['mpjpe_mm'],0)
        shifted = body.clone();shifted[:,4,0] += 10
        _,moved = reconstruct(shifted,local,t.roles,t.fingers)
        self.assertTrue(torch.equal(moved[:,:2],global_fingers[:,:2]+torch.tensor([10,0,0])))
        self.assertTrue(torch.equal(moved[:,2:],global_fingers[:,2:]))
        metrics = cascade_metrics(shifted,local,moved,d,t.roles,t.fingers)
        self.assertEqual(metrics['hand_wrist_relative']['mpjpe_mm'],0)
        self.assertEqual(metrics['left_hand_global']['mpjpe_mm'],10)
        self.assertEqual(metrics['left_body_wrist']['mpjpe_mm'],10)

    def test_sources_missing_cache_and_tampering_reject_no_gt_fallback(self):
        t = self.trainer()
        d = t.splits['train']
        predicted = t.condition(d,'predicted_frozen_body')
        with self.assertRaisesRegex(ValueError,'혼용'):
            t.batch_loss(torch.tensor([0]),torch.tensor([1000]),torch.randn(1,4,4,3),predicted)
        gt = t.condition(d,'gt_upper_train')
        with self.assertRaisesRegex(ValueError,'혼용'): gt.require('predicted_frozen_body',t.frozen.identity)
        with self.assertRaises(ValueError): t.predict(condition_source='none')
        saved = copy.deepcopy(t.body_cache)
        t.body_cache.source = 'gt_upper_diagnostic'
        with self.assertRaises(ValueError): t.predict()
        t.body_cache = saved
        t.body_cache.values_mm[('validation_0',0)][0,0] += 1
        with self.assertRaises(ValueError): t.predict()
        t.body_cache = t.frozen.predict()
        del t.body_cache.values_mm[('validation_0',0)]
        t.body_cache.payload_sha256 = prediction_hash(t.body_cache.values_mm)
        with self.assertRaisesRegex(ValueError,'GT fallback'): t.predict()

    def test_prediction_is_independent_of_gt_3d_targets(self):
        t = self.trainer(dropout=0.)
        before = t.predict()
        body_hash = t.body_cache.payload_sha256
        # 실제 body sampling 경로의 GT만 바꿔도 prediction cache는 불변이어야 한다.
        for data in t.frozen.trainer.splits.values():
            data.target_mm += 9999
        cache = t.frozen.predict()
        self.assertEqual(cache.payload_sha256,body_hash)
        for d in t.splits.values():
            d.body_target_mm += 8888;d.target_mm -= 5555;d.global_target_mm += 6666
        t.body_cache = cache
        after = t.predict()
        for field in ('body_mm','local_mm','global_fingers_mm','whole_body_mm'):
            self.assertTrue(torch.equal(before[field],after[field]),field)
        diagnostic = t.predict(condition_source='gt_upper_diagnostic')
        self.assertFalse(torch.equal(after['local_mm'],diagnostic['local_mm']))
        self.assertEqual(diagnostic['attachment_source'],'predicted_frozen_body_wrist')

    def test_small_overfit_finite_gradients_and_frozen_body(self):
        for mode in ('upper','none'):
            t = self.trainer(conditioning=mode,dropout=0.)
            before = t.evaluate('train')['normalized_mse']
            body_hash = weights_hash(t.frozen.model)
            for _ in range(100): t.train_step()
            final = t.evaluate('train')['normalized_mse']
            self.assertLess(final,before*.1,(mode,before,final))
            t.frozen.assert_unchanged()
            self.assertEqual(body_hash,weights_hash(t.frozen.model))
            self.assertTrue(all(torch.isfinite(p.grad).all() for p in t.model.parameters() if p.grad is not None))
            if mode=='upper':
                self.assertGreater(float(t.model.upper_encoder[0].weight.grad.abs().sum()),0)
            else:
                self.assertFalse(hasattr(t.model,'upper_encoder'))
                with self.assertRaises(ValueError): t.predict(condition_source='gt_upper_diagnostic')

    def test_upper_padding_nan_and_full_sampler_future_causality(self):
        t = self.trainer(dropout=0.)
        d = t.splits['train'];t.model.eval()
        x,camera,time = d.features[3:5].clone(),t.cameras['train'][3:5].clone(),d.time_valid[3:5]
        condition = t.condition(d,'predicted_frozen_body')
        upper = condition.values_mm[3:5].clone()/100
        valid = condition.valid[3:5].clone()
        noise = torch.randn(2,4,4,3)
        before = t.diffusion.sample(BoundHand(t.model,upper,valid).eval(),x,camera,time,noise,4)
        upper[:,2:] += 100;x[:,2:] += 100;camera[:,2:] -= 20;noise[:,2:] += 30
        after = t.diffusion.sample(BoundHand(t.model,upper,valid).eval(),x,camera,time,noise,4)
        self.assertTrue(torch.equal(before[:,:2],after[:,:2]))
        first = t.condition(d,'gt_upper_train')
        upper = first.values_mm[:1].clone();valid = first.valid[:1].clone()
        upper[~valid] = float('nan')
        y,_ = t.model(torch.randn(1,4,4,3),d.features[:1],t.cameras['train'][:1],torch.tensor([1000]),d.time_valid[:1],upper=upper,upper_valid=valid)
        self.assertTrue(torch.isfinite(y).all())

    def test_whole_cascade_future_input_perturbation(self):
        t = self.trainer(dropout=0.)
        before = t.predict('train')['whole_body_mm'].clone()
        for i,(_,index) in enumerate(t.splits['train'].anchors):
            if index>=3:
                t.splits['train'].features[i] += 100
                t.cameras['train'][i] -= 30
                t.frozen.trainer.splits['train'].features[i] += 100
                t.frozen.trainer.cameras['train'][i] -= 30
        t.body_cache = t.frozen.predict()
        after = t.predict('train')['whole_body_mm']
        for i,(_,index) in enumerate(t.splits['train'].anchors):
            if index<3: self.assertTrue(torch.equal(before[i],after[i]),i)

    def test_cascade_streaming_offline_reset_gap_and_bounded_history(self):
        t = self.trainer(dropout=0.);t.model.eval()
        stream = CascadeStream(t)
        first_row = None
        for split in ('train','validation'):
            expected = t.predict(split)['whole_body_mm']
            d,bd = t.splits[split],t.frozen.trainer.splits[split]
            for i,(seq,index) in enumerate(d.anchors):
                row = (seq,index,bd.features[i,-1],d.features[i,-1],t.frozen.trainer.cameras[split][i,-1],t.cameras[split][i,-1])
                if first_row is None: first_row = row
                actual = stream.push(*row)['whole_body_mm']
                self.assertTrue(torch.allclose(actual,expected[i],atol=8e-5,rtol=1e-5),(seq,index,float((actual-expected[i]).abs().max())))
        stream.reset()
        fresh = stream.push(*first_row)['whole_body_mm']
        independent = CascadeStream(t).push(*first_row)['whole_body_mm']
        self.assertTrue(torch.equal(fresh,independent))
        with self.assertRaisesRegex(ValueError,'중복'): stream.push(*first_row)
        gap = (first_row[0],9,*first_row[2:])
        self.assertTrue(torch.equal(stream.push(*gap)['whole_body_mm'],CascadeStream(t).push(*gap)['whole_body_mm']))
        stream.reset()
        for i in range(36):
            stream.push('long',i,*first_row[2:])
            self.assertLessEqual(len(stream.history),4);self.assertLessEqual(len(stream.body.history),4)

    def test_hand_checkpoint_exact_full_resume_and_body_identity_guards(self):
        full = self.trainer()
        for _ in range(12): full.train_step()
        expected = copy.deepcopy(full.checkpoint())
        part = self.trainer()
        for _ in range(5): part.train_step()
        path = self.base/'resume.pt';save_checkpoint(part.checkpoint(),path)
        resumed = self.trainer();resumed.resume(read_checkpoint(path))
        for _ in range(7): resumed.train_step()
        actual = resumed.checkpoint()
        for name,value in expected['model'].items(): self.assertTrue(torch.equal(value,actual['model'][name]),name)
        for key in ('scheduler','scaler','step','samples_seen','config','frozen_body','hand_contract','m3_contract'):
            self.assertEqual(expected[key],actual[key])
        self.assertTrue(torch.equal(expected['rng']['torch_cpu'],actual['rng']['torch_cpu']))
        self.assertEqual(expected['rng']['python'],actual['rng']['python'])
        for index,state in expected['optimizer']['state'].items():
            for key,value in state.items(): self.assertTrue(torch.equal(value,actual['optimizer']['state'][index][key]))
        for field in ('frozen_body','hand_contract','manifest_hash','source'):
            bad = copy.deepcopy(actual)
            if field=='frozen_body': bad[field]['checkpoint_sha256'] = 'changed'
            elif field=='hand_contract': bad[field]['coordinate_frame'] = 'device_relative'
            elif field=='source': bad[field]['code_sha256']['rewind_m3/models.py'] = 'changed'
            else: bad[field] = 'changed'
            with self.assertRaises(ValueError,msg=field): resumed.resume(bad)

    def test_frozen_checkpoint_contract_mismatch_and_mutation_detection(self):
        wrong = copy.deepcopy(self.doc)
        wrong['m3_contract']['body_joint_names'] = list(reversed(wrong['m3_contract']['body_joint_names']))
        with self.assertRaises(ValueError): FrozenBody(self.body_path,wrong)
        t = self.trainer()
        with torch.no_grad(): next(t.frozen.model.parameters()).add_(1)
        with self.assertRaisesRegex(ValueError,'변경'): t.frozen.assert_unchanged()

    def test_cli_fixture_three_paths_and_hand_resume(self):
        out = self.base/'pipeline'
        with contextlib.redirect_stdout(io.StringIO()):
            fixture_main(['--output',str(out),'--body-steps','2','--hand-steps','2'])
            args = ['train','--config',str(ROOT/'configs/m3_hand_upper_synthetic.json'),'--dataset',str(out/'whole.json'),
                    '--body-checkpoint',str(out/'body/last.pt'),'--output',str(out/'hand_upper'),'--resume',str(out/'hand_upper/last.pt'),'--steps','4']
            self.assertEqual(main(args),0)
        comparison = json.loads((out/'comparison.json').read_text())
        ids = []
        for source,result in comparison['results'].items():
            metrics = result['metrics']
            self.assertEqual(metrics['condition_source'],source)
            self.assertEqual(metrics['attachment_source'],'predicted_frozen_body_wrist')
            self.assertEqual(metrics['evaluation_kind'],'synthetic_code_check')
            ids.append(metrics['frozen_body']['checkpoint_sha256'])
        self.assertEqual(len(set(ids)),1)
        report = json.loads((out/'hand_upper/report.json').read_text())
        self.assertEqual(report['step'],4)
        self.assertEqual([(x['initial_step'],x['final_step']) for x in report['segments']],[(0,2),(2,4)])
        # 같은 config/data이지만 다른 body weights의 실제 CLI 평가를 거부한다.
        other = torch.load(out/'body/last.pt',weights_only=True)
        first_key = next(iter(other['model']))
        other['model'][first_key] = other['model'][first_key]+.01
        other_path = out/'other_body.pt';save_checkpoint(other,other_path)
        wrong_output = out/'wrong_body_evaluation.json'
        with contextlib.redirect_stdout(io.StringIO()),contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                main(['evaluate','--checkpoint',str(out/'hand_upper/last.pt'),'--body-checkpoint',str(other_path),
                      '--dataset',str(out/'whole.json'),'--output',str(wrong_output)])
        self.assertFalse(wrong_output.exists())

    def test_evaluation_preserves_rng_and_test_gate(self):
        t = self.trainer();rng = rng_state()
        t.evaluate()
        self.assertTrue(torch.equal(rng['torch_cpu'],rng_state()['torch_cpu']))
        self.assertEqual(rng['python'],rng_state()['python'])
        with self.assertRaisesRegex(ValueError,'최종 평가'): t.evaluate('test')
        with self.assertRaisesRegex(ValueError,'CPU'): Trainer(self.config,self.body_path,self.doc,device='cuda')


if __name__=='__main__': unittest.main()
