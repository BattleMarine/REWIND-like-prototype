"""합성 CPU shard 학습·재개·중단 검증. 실제 자료/원격 서비스는 사용하지 않는다."""
import contextlib
import copy
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from rewind_m1.artifacts import load_weights
from rewind_m1.data import DataContractError, load_document, synthetic_document
from rewind_m1.metrics import metric_sums
from rewind_m1.shards import ShardDataset, build_catalog, validate_catalog
from rewind_m1.shard_training import ShardTrainer
from rewind_m1.shard_cli import main, read_checkpoint
from rewind_m1.training import rng_state, save_checkpoint

ROOT = Path(__file__).resolve().parents[1]


def equal(a, b):
    if isinstance(a, torch.Tensor):
        return isinstance(b, torch.Tensor) and torch.equal(a, b)
    if isinstance(a, dict):
        return isinstance(b, dict) and set(a) == set(b) and all(equal(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return type(a) is type(b) and len(a) == len(b) and all(equal(x, y) for x,y in zip(a,b))
    return a == b


class ShardTrainingTests(unittest.TestCase):
    def setUp(self):
        target = ROOT/'outputs'/'test_tmp'
        target.mkdir(parents=True, exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=target)
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        doc = synthetic_document()
        paths = []
        for sequence in ('train_0','train_1','validation_0','validation_1'):
            part = copy.deepcopy(doc)
            part['records'] = [r for r in part['records'] if r['sequence_id'] == sequence]
            if sequence == 'validation_0':
                part['records'][0]['joints']['tip']['device_pts3d'] = None
            path = sequence+'.json'
            (self.base/path).write_text(json.dumps(part), encoding='utf8')
            paths.append(path)
        self.catalog = build_catalog(paths, self.base, 3)
        (self.base/'catalog.json').write_text(json.dumps(self.catalog), encoding='utf8')
        self.config = {'model':'single_frame','window_size':3,'hidden':8,'dropout':.2,'batch_size':4,
            'learning_rate':.005,'seed':31,'target_scale_mm':100.,'amp':False,'scheduler_step':2,'scheduler_gamma':.8}
        (self.base/'config.json').write_text(json.dumps(self.config), encoding='utf8')

    def trainer(self, model='single_frame'):
        return ShardTrainer({**self.config,'model':model}, self.catalog, self.base)

    def test_uninterrupted_resume_mid_shard_boundary_epoch_both_models(self):
        for model in ('single_frame','temporal'):
            trainer = self.trainer(model)
            checkpoints = {}
            for _ in range(9):
                trainer.train_step()
                if trainer.step in (1,2,4):
                    path = self.base/f'{model}_{trainer.step}.pt'
                    save_checkpoint(trainer.checkpoint(), path)
                    checkpoints[trainer.step] = path
            expected = copy.deepcopy(trainer.checkpoint())
            self.assertEqual(expected['samples_seen'], 28)
            self.assertEqual(expected['cursor']['epoch'], 2)
            for cut, path in checkpoints.items():
                restored = self.trainer(model)
                state, _ = read_checkpoint(path)
                restored.resume(state)
                for _ in range(9-cut):
                    restored.train_step()
                self.assertTrue(equal(expected, restored.checkpoint()), (model,cut))

    def test_validation_sum_count_and_rng_mode(self):
        trainer = self.trainer()
        trainer.train_step()
        before = copy.deepcopy(trainer.checkpoint())
        result = trainer.evaluate()
        self.assertTrue(trainer.model.training)
        self.assertTrue(equal(before, trainer.checkpoint()))
        dataset = ShardDataset(self.catalog, self.base)
        expected = {'distance_sum_mm':0.,'squared_sum_mm2':0.,'valid_joints':0}
        trainer.model.eval()
        with torch.no_grad():
            for i in (2,3):
                data = dataset.load(i)[1]['validation']
                prediction = trainer.model(data.features, data.time_valid)[:,-1]*100.
                values = metric_sums(prediction, data.target_mm[:,-1], data.target_valid[:,-1])
                for key in expected:
                    expected[key] += values[key]
                del data
        self.assertEqual(result['records'], 12)
        self.assertEqual(result['valid_joints'], 23)
        self.assertAlmostEqual(result['mpjpe_mm'], expected['distance_sum_mm']/23, places=5)
        self.assertAlmostEqual(result['normalized_mse'], expected['squared_sum_mm2']/(23*3*10000), places=7)
        trainer.model.eval()
        rng = rng_state()
        trainer.evaluate()
        self.assertFalse(trainer.model.training)
        self.assertTrue(equal(rng, rng_state()))

    def test_failed_load_and_optimizer_do_not_advance_cursor(self):
        for failure in ('load','optimizer'):
            trainer = self.trainer()
            cursor = trainer.sampler.state_dict()
            target = trainer.dataset if failure == 'load' else trainer.optimizer
            method = 'load' if failure == 'load' else 'step'
            with patch.object(target, method, side_effect=OSError('injected_failure')):
                with self.assertRaises(OSError):
                    trainer.train_step()
            self.assertEqual(trainer.sampler.state_dict(), cursor)
            self.assertEqual(trainer.step, 0)
            self.assertEqual(trainer.samples_seen, 0)
            with self.assertRaisesRegex(DataContractError, 'shard_failed_state_cannot_checkpoint'):
                trainer.checkpoint()

    def test_nonfinite_loss_rejected_before_backward(self):
        trainer = self.trainer()
        with patch('rewind_m1.shard_training.masked_mse', return_value=torch.tensor(float('nan'),requires_grad=True)):
            with self.assertRaisesRegex(DataContractError, 'shard_nonfinite_loss'):
                trainer.train_step()
        self.assertEqual(trainer.step, 0)

    def test_amp_skip_cursor_scheduler_step_stay_and_scaler_records(self):
        trainer = self.trainer()
        trainer.scaler.get_scale = unittest.mock.Mock(side_effect=[8.,4.])
        trainer.scaler.step = unittest.mock.Mock()
        trainer.scaler.update = unittest.mock.Mock()
        cursor = trainer.sampler.state_dict()
        scheduler = copy.deepcopy(trainer.scheduler.state_dict())
        model = copy.deepcopy(trainer.model.state_dict())
        result = trainer.train_step()
        self.assertFalse(result['updated'])
        self.assertTrue(equal(model, trainer.model.state_dict()))
        self.assertEqual(trainer.sampler.state_dict(), cursor)
        self.assertEqual(scheduler, trainer.scheduler.state_dict())
        self.assertEqual((trainer.step, trainer.samples_seen, trainer.attempts, trainer.skipped_updates), (0,0,1,1))

    def test_cached_shard_is_not_reloaded_per_batch(self):
        trainer = self.trainer()
        from rewind_m1.shards import load_document as original
        with patch('rewind_m1.shards.load_document', wraps=original) as load:
            trainer.train_step()
            trainer.train_step()
            self.assertEqual(load.call_count, 1)
            trainer.train_step()
            self.assertEqual(load.call_count, 2)

    def test_resume_bad_counters_identity_and_rng_no_partial_mutation(self):
        trainer = self.trainer()
        trainer.train_step()
        expected = copy.deepcopy(trainer.checkpoint())
        for kind in ('samples','source','rng','shape','cursor','optimizer','scheduler'):
            bad = copy.deepcopy(expected)
            if kind == 'samples': bad['samples_seen'] += 1
            if kind == 'source': bad['source']['code_sha256'] = {}
            if kind == 'rng': bad['rng']['torch_cpu'] = torch.tensor([1],dtype=torch.uint8)
            if kind == 'shape': bad['model']['net.0.weight'] = torch.zeros(1,1)
            if kind == 'cursor': bad['cursor']['offset'] = 1
            if kind == 'optimizer':
                next(iter(bad['optimizer']['state'].values()))['exp_avg'] = torch.zeros(1)
            if kind == 'scheduler': bad['scheduler']['last_epoch'] += 1
            with self.assertRaises((ValueError,RuntimeError)):
                trainer.resume(bad)
            self.assertTrue(equal(expected, trainer.checkpoint()), kind)

    def test_missing_split_config_and_real_release_gates(self):
        with self.assertRaisesRegex(DataContractError, 'shard_model_window_mismatch'):
            ShardTrainer({**self.config,'window_size':2}, self.catalog, self.base)
        real = copy.deepcopy(self.catalog)
        from rewind_m1.data import VERIFICATIONS, content_hash
        real['contract'].update(data_kind='dataset_sample',verified=dict.fromkeys(VERIFICATIONS,True),
                                verification_evidence=dict.fromkeys(VERIFICATIONS,'declared'))
        real['content_sha256'] = content_hash({k:v for k,v in real.items() if k!='content_sha256'})
        with self.assertRaisesRegex(DataContractError, 'shard_dataset_sample_release_not_implemented'):
            ShardTrainer(self.config, real, self.base)
        only_train = build_catalog(['train_0.json'], self.base, 3)
        with self.assertRaisesRegex(DataContractError, 'shard_training_requires_train_validation'):
            ShardTrainer(self.config, only_train, self.base)

    def cli_train(self, output, *args):
        with contextlib.redirect_stdout(io.StringIO()):
            return main(['train','--config',str(self.base/'config.json'),'--catalog',str(self.base/'catalog.json'),
                         '--output',str(output), *args])

    def test_cli_train_resume_evaluate_export_and_log_bound(self):
        output = self.base/'run'
        self.cli_train(output, '--epochs','1','--log-limit','2')
        checkpoint, _ = read_checkpoint(output/'last.pt')
        self.assertEqual(checkpoint['step'], 4)
        self.assertEqual(checkpoint['samples_seen'], 12)
        self.cli_train(output,'--resume',str(output/'last.pt'),'--steps','6','--export-weights',str(output/'weights.pt'))
        checkpoint, identity = read_checkpoint(output/'last.pt')
        self.assertEqual(checkpoint['samples_seen'], 18)
        report = json.loads((output/'report.json').read_bytes())
        self.assertEqual(report['status'], 'completed_requested_target')
        with contextlib.redirect_stdout(io.StringIO()):
            main(['evaluate','--checkpoint',str(output/'last.pt'),'--catalog',str(self.base/'catalog.json'),
                  '--output',str(output/'eval.json')])
        args = dict(expected_contract=self.catalog['contract'],expected_catalog_sha256=self.catalog['content_sha256'],
            expected_source={k:checkpoint['source'][k] for k in ('revision','code_sha256')},
            expected_checkpoint_sha256=identity,expected_model_config={k:self.config[k] for k in
                    ('model','window_size','hidden','dropout','target_scale_mm')})
        model, _ = load_weights(output/'weights.pt', **args)
        reference = self.trainer()
        reference.resume(checkpoint)
        reference.model.eval()
        x,valid = torch.rand(2,3,28),torch.ones(2,3,dtype=torch.bool)
        self.assertTrue(torch.equal(model(x,valid),reference.model(x,valid)))
        # export는 원본 특징이 사라진 뒤에도 checkpoint만으로 동작한다.
        for path in self.base.glob('train_*.json'): path.unlink()
        with contextlib.redirect_stdout(io.StringIO()):
            main(['export','--checkpoint',str(output/'last.pt'),'--output',str(output/'separate.pt')])

    def test_cli_interrupted_preserves_durable_and_resumes(self):
        output = self.base/'interrupt'
        original = ShardTrainer.train_step
        def interrupted(trainer):
            if trainer.step == 1:
                with torch.no_grad():
                    next(trainer.model.parameters()).add_(123.)
                raise KeyboardInterrupt('injected_interrupt')
            return original(trainer)
        with patch.object(ShardTrainer,'train_step',interrupted):
            with self.assertRaises(KeyboardInterrupt):
                self.cli_train(output,'--steps','3')
        checkpoint, _ = read_checkpoint(output/'last.pt')
        self.assertEqual(checkpoint['step'], 1)
        report = json.loads((output/'report.json').read_bytes())
        self.assertEqual(report['status'], 'interrupted')
        self.assertEqual(report['durable_checkpoint']['step'], 1)
        self.cli_train(output,'--resume',str(output/'last.pt'),'--steps','3')
        self.assertEqual(read_checkpoint(output/'last.pt')[0]['step'], 3)

    def test_cli_attempt_bound_and_validation_final_test_guard(self):
        output = self.base/'bounded'
        self.cli_train(output,'--epochs','2','--max-attempts','1')
        report = json.loads((output/'report.json').read_bytes())
        self.assertEqual(report['status'], 'bounded_stop')
        self.assertEqual(report['coverage']['completed_epochs'], 0)
        self.assertEqual(report['coverage']['current_epoch_samples'], 4)
        trainer = self.trainer()
        # test split이 실제 존재하는 합성 특징을 별도로 만든다.
        doc = synthetic_document()
        doc['records'] = [r for r in doc['records'] if r['sequence_id']=='validation_0']
        for r in doc['records']:
            r.update(sequence_id='test_0',motion_id='test_motion_0',split='test')
        (self.base/'test.json').write_text(json.dumps(doc),encoding='utf8')
        catalog = build_catalog([r['path'] for r in self.catalog['shards']]+['test.json'], self.base,3)
        trainer = ShardTrainer(self.config,catalog,self.base)
        with self.assertRaisesRegex(DataContractError,'shard_test_requires_final_test'):
            trainer.evaluate('test')
        self.assertEqual(trainer.evaluate('test',True)['records'],6)

    def test_cli_existing_output_identity_and_evaluate_collision_preserved(self):
        output = self.base/'protected'
        self.cli_train(output,'--steps','1')
        before_checkpoint = (output/'last.pt').read_bytes()
        before_report = (output/'report.json').read_bytes()
        other = self.base/'other'
        self.cli_train(other,'--steps','2')
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                self.cli_train(output,'--resume',str(other/'last.pt'),'--steps','3')
        self.assertEqual(before_checkpoint,(output/'last.pt').read_bytes())
        self.assertEqual(before_report,(output/'report.json').read_bytes())
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                main(['evaluate','--checkpoint',str(output/'last.pt'),'--catalog',str(self.base/'catalog.json'),
                      '--output',str(output/'last.pt')])
        self.assertEqual(before_checkpoint,(output/'last.pt').read_bytes())

    def test_budget_during_validation_preserves_durable_and_skips_full_eval(self):
        output = self.base/'timeout'
        original = ShardTrainer.evaluate
        def deadline(trainer, *args, **kwargs):
            trainer.check_budget = lambda: (_ for _ in ()).throw(TimeoutError('injected_deadline'))
            return original(trainer, *args, **kwargs)
        with patch.object(ShardTrainer,'evaluate',deadline):
            self.cli_train(output,'--steps','1')
        report = json.loads((output/'report.json').read_bytes())
        self.assertEqual(report['status'],'bounded_stop')
        self.assertEqual(read_checkpoint(output/'last.pt')[0]['step'],1)
        self.assertEqual(report['durable_checkpoint']['step'],1)

    def test_cli_log_tail_is_bounded(self):
        output = self.base/'logs'
        self.cli_train(output,'--steps','7','--log-limit','2')
        report = json.loads((output/'report.json').read_bytes())
        self.assertEqual(len(report['log']),2)
        self.assertEqual([e['step'] for e in report['log']],[6,7])


if __name__ == '__main__':
    unittest.main()
