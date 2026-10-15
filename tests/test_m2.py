"""작은 합성 teacher의 수학·mask·causality·재개 계약 검사."""
import contextlib
import copy
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

try:
    import torch
except ImportError:
    torch = None

if torch is not None:
    from rewind_m2.data import synthetic_document,load_document,encode_camera,DataContractError
    from rewind_m2.diffusion import DiffusionSchedule
    from rewind_m2.training import Trainer,validate_config,position_losses,read_checkpoint
    from rewind_m2.streaming import AnchorSampler,anchor_noise,keyed_frame_noise
    from rewind_m1.training import save_checkpoint,rng_state
    from rewind_m2.__main__ import main

ROOT = Path(__file__).resolve().parents[1]


@unittest.skipIf(torch is None,'M2 검증에는 별도 PyTorch 환경 필요')
class M2Tests(unittest.TestCase):
    def setUp(self):
        self.config = json.loads((ROOT/'configs/m2_body_synthetic.json').read_text())
        self.doc = synthetic_document()
        (ROOT/'outputs/test_tmp').mkdir(parents=True,exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=ROOT/'outputs/test_tmp')
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)

    def trainer(self, **changes):
        return Trainer({**self.config,**changes},self.doc)

    def inputs(self, trainer):
        d = trainer.splits['train']
        g = torch.Generator().manual_seed(111)
        noise = torch.randn(d.target_mm[:2].shape,generator=g)
        return noise,d.features[:2].clone(),trainer.cameras['train'][:2].clone(),torch.tensor([999,400]),d.time_valid[:2].clone()

    def test_schedule_q_sample_endpoints_and_oracle_ddim(self):
        schedule = DiffusionSchedule(32,.0001,.02)
        self.assertAlmostEqual(float(schedule.beta[0]),.0001)
        self.assertAlmostEqual(float(schedule.beta[-1]),.02)
        self.assertEqual(float(schedule.alpha_bar[0]),1.)
        self.assertTrue((schedule.alpha_bar[1:]<schedule.alpha_bar[:-1]).all())
        clean,noise = torch.randn(2,4,2,3),torch.randn(2,4,2,3)
        self.assertTrue(torch.equal(schedule.q_sample(clean,noise,torch.zeros(2,dtype=torch.long)),clean))
        k = torch.tensor([1,32])
        a = schedule.alpha(k,clean)
        self.assertTrue(torch.equal(schedule.q_sample(clean,noise,k),a.sqrt()*clean+(1-a).sqrt()*noise))
        class Oracle(torch.nn.Module):
            def forward(self,*args):
                return clean,clean
        model = Oracle().eval()
        valid = torch.ones(2,4,dtype=torch.bool)
        for steps in (1,2,4,10,32):
            result = schedule.sample(model,None,None,valid,noise,steps)
            self.assertTrue(torch.equal(result,clean),steps)
        for bad in (0,33,True):
            with self.assertRaises(ValueError): schedule.timesteps(bad)

    def test_camera_skeleton_contract_and_mask_packing(self):
        contract,splits,cameras,_,edges = load_document(self.doc,4,100.)
        self.assertEqual(edges,[(0,1)])
        self.assertEqual(tuple(cameras['train'].shape),(12,4,2,9))
        self.assertEqual(cameras['train'][0,:3].abs().sum(),0)
        self.assertEqual(cameras['train'][0,-1,0,:6].tolist(),[1,0,0,0,1,0])
        self.assertAlmostEqual(float(cameras['train'][0,-1,0,6]),-.06,places=6)
        self.assertEqual(splits['train'].features[0,-1,3:7].tolist(),[0,0,0,0])
        for mutate in ('reflection','nan','missing','definition','edge','real'):
            bad = copy.deepcopy(self.doc)
            if mutate=='reflection': bad['records'][0]['camera_pose']['left']['rotation_matrix'][0][0] = -1
            elif mutate=='nan': bad['records'][0]['camera_pose']['left']['translation'][0] = float('nan')
            elif mutate=='missing': del bad['records'][0]['camera_pose']
            elif mutate=='definition': bad['m2_contract']['camera']['transform'] = 'unknown'
            elif mutate=='edge': bad['m2_contract']['skeleton']['edges'] = [['root','other']]
            elif mutate=='real': bad['contract']['data_kind'] = 'dataset_sample'
            with self.assertRaises(DataContractError,msg=mutate): load_document(bad,4,100.)

    def test_masked_nan_before_q_sample_and_loss_gradient(self):
        trainer = self.trainer(dropout=0.)
        d = trainer.splits['train']
        d.target_valid[0,-1,1] = False
        d.target_mm[0,-1,1] = float('nan')
        noise = torch.randn(1,4,2,3)
        loss,_,_ = trainer.batch_loss(d,trainer.cameras['train'],torch.tensor([0]),torch.tensor([1000]),noise)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in trainer.model.parameters() if p.grad is not None))
        self.assertTrue(trainer.model.camera_encoder[0].weight.grad.abs().sum()>0)
        p = torch.tensor([[[[3.,4.,0.],[float('nan')]*3]]],requires_grad=True)
        y = torch.zeros_like(p);y[0,0,1] = float('nan')
        valid = torch.tensor([[[True,False]]]);time = torch.tensor([[True]])
        value,_,_ = position_losses(p,p,y,valid,time,1.,1.)
        self.assertAlmostEqual(float(value.detach()),50/3,places=5)
        value.backward()
        self.assertEqual(p.grad[0,0,1].tolist(),[0,0,0])

    def test_denoiser_and_sampler_future_perturbation_padding_rope_shift(self):
        trainer = self.trainer(dropout=0.)
        model,schedule = trainer.model.eval(),trainer.diffusion
        data = trainer.splits['train']
        # 과거가 채워진 window에서 미래 2D/camera/noise를 동시에 변경한다.
        x = data.features[3:5].clone()
        cam = trainer.cameras['train'][3:5].clone()
        valid = data.time_valid[3:5].clone()
        noise = torch.randn(2,4,2,3)
        k = torch.tensor([1,1000])
        before = model(noise,x,cam,k,valid)[0]
        shifted = model(noise,x,cam,k,valid,positions=torch.arange(4)+100)[0]
        self.assertTrue(torch.allclose(before,shifted,atol=2e-6,rtol=1e-5))
        n2,x2,c2 = noise.clone(),x.clone(),cam.clone()
        n2[:,2:] += 100;x2[:,2:] -= 100;c2[:,2:] += 50
        after = model(n2,x2,c2,k,valid)[0]
        self.assertTrue(torch.equal(before[:,:2],after[:,:2]))
        sample = schedule.sample(model,x,cam,valid,noise,4)
        shifted_sample = schedule.sample(model,x,cam,valid,noise,4,positions=torch.arange(4)+100)
        self.assertTrue(torch.allclose(sample,shifted_sample,atol=3e-6,rtol=1e-5))
        sample_after = schedule.sample(model,x2,c2,valid,n2,4)
        self.assertTrue(torch.equal(sample[:,:2],sample_after[:,:2]))
        noise,x,cam,k,valid = self.inputs(trainer)
        padded = model(noise,x,cam,k,valid)[0]
        noise[~valid] = float('nan');x[~valid] = float('nan');cam[~valid] = float('nan')
        self.assertTrue(torch.equal(padded,model(noise,x,cam,k,valid)[0]))
        self.assertFalse(padded[~valid].any())

    def test_streaming_matches_independent_anchor_windows_reset_gap_and_bounded_history(self):
        trainer = self.trainer(dropout=0.)
        model,schedule = trainer.model.eval(),trainer.diffusion
        stream = AnchorSampler(model,schedule,4,4)
        gen = torch.Generator().manual_seed(100)
        rows,history = [],[]
        sequence_frames = [('A',i) for i in range(55)]+[('A',70),('B',0),('B',1)]
        for sequence,index in sequence_frames:
            rows.append((sequence,index,torch.randn(28,generator=gen),trainer.cameras['train'][0,-1].clone(),torch.randn(2,3,generator=gen)))
        previous = None
        for seq,index,x,c,n in rows:
            if previous is None or seq!=previous[0] or index!=previous[1]+1:
                history = []
            history = (history+[(x,c,n)])[-4:]
            packed = [torch.zeros(1,4,*a.shape) for a in (x,c,n)]
            for slot,row in enumerate(history,4-len(history)):
                for out,value in zip(packed,row): out[0,slot] = value
            valid = torch.tensor([[False]*(4-len(history))+[True]*len(history)])
            expected = schedule.sample(model,packed[0],packed[1],valid,packed[2],4)[0,-1]
            actual = stream.push(seq,index,x,c,n)
            self.assertTrue(torch.allclose(expected,actual,atol=1e-6))
            self.assertLessEqual(len(stream.history),4)
            previous = (seq,index)
        with self.assertRaisesRegex(ValueError,'중복'):
            stream.push(*rows[-1])
        self.assertEqual(stream.frame_index,1)
        stream.reset()
        fresh = stream.push(*rows[0])
        independent = AnchorSampler(model,schedule,4,4).push(*rows[0])
        self.assertTrue(torch.equal(fresh,independent))

    def test_synthetic_overfit_gradient_and_eval_does_not_consume_rng(self):
        trainer = self.trainer(dropout=0.)
        initial = trainer.evaluate('train')['normalized_mse']
        rng = rng_state()
        trainer.evaluate()
        self.assertTrue(torch.equal(rng['torch_cpu'],rng_state()['torch_cpu']))
        self.assertEqual(rng['python'],rng_state()['python'])
        for _ in range(100): trainer.train_step()
        final = trainer.evaluate('train')['normalized_mse']
        self.assertLess(final,initial*.1,(initial,final))
        for prefix in ('camera_encoder','time_encoder','spatial','temporal','frame_head','final_head'):
            self.assertTrue(any(p.grad is not None and p.grad.abs().sum()>0 for name,p in trainer.model.named_parameters() if name.startswith(prefix)),prefix)

    def test_frame_keyed_noise_independent_of_future_order_and_anchor(self):
        trainer = self.trainer()
        data = trainer.splits['train']
        noise = anchor_noise(data,99)
        self.assertTrue(torch.equal(noise[0,-1],noise[1,-2]))
        self.assertTrue(torch.equal(noise[0,-1],keyed_frame_noise('train_0',0,2,99)))
        before = noise[0,-1].clone()
        keyed_frame_noise('future',999,2,99)
        self.assertTrue(torch.equal(before,keyed_frame_noise('train_0',0,2,99)))
        self.assertFalse(noise[0,:-1].any())

    def test_checkpoint_exact_resume_full_state_and_identity_guards(self):
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
        for key in ('scheduler','scaler','step','samples_seen','config','m2_contract','environment'):
            self.assertEqual(expected[key],actual[key])
        self.assertTrue(torch.equal(expected['rng']['torch_cpu'],actual['rng']['torch_cpu']))
        self.assertEqual(expected['rng']['python'],actual['rng']['python'])
        for index,state in expected['optimizer']['state'].items():
            for key,value in state.items(): self.assertTrue(torch.equal(value,actual['optimizer']['state'][index][key]))
        for field in ('manifest_hash','source','config','diffusion'):
            bad = copy.deepcopy(actual)
            if field=='source': bad[field]['code_sha256']['rewind_m2/models.py'] = 'changed'
            elif field=='diffusion': bad[field]['alpha_bar'][1] = .5
            else: bad[field] = 'changed'
            with self.assertRaises(ValueError,msg=field): resumed.resume(bad)

    def test_cli_periodic_resume_evaluation_and_recovery(self):
        config = self.base/'config.json';config.write_text(json.dumps(self.config))
        out = self.base/'run'
        args = ['train','--config',str(config),'--output',str(out)]
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(main(args+['--steps','3']),0)
            prior = (out/'report.json').read_bytes()
            self.assertEqual(main(args+['--steps','5','--resume',str(out/'last.pt')]),0)
            (out/'report.json').write_bytes(prior)
            self.assertEqual(main(args+['--steps','6','--resume',str(out/'last.pt')]),0)
            self.assertEqual(main(['evaluate','--checkpoint',str(out/'last.pt'),'--output',str(out/'evaluation.json')]),0)
        report = json.loads((out/'report.json').read_text())
        self.assertEqual(report['step'],6)
        self.assertTrue(report['segments'][-2]['report_recovery'])
        self.assertEqual(report['validation']['evaluation_kind'],'synthetic_code_check')
        self.assertEqual(json.loads((out/'evaluation.json').read_text())['checkpoint_step'],6)

    def test_config_and_test_gates(self):
        for changes in ({'ddim_eta':1},{'hidden':30},{'schedule':'cosine_unknown'},{'sampling_steps':1001},{'attention_past_frames':-1}):
            with self.assertRaises(ValueError): validate_config({**self.config,**changes})
        with self.assertRaisesRegex(ValueError,'최종 평가'): self.trainer().evaluate('test')


if __name__=='__main__':
    unittest.main()
