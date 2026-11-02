"""준비된 합성 M1 특징 shard의 headless 학습·정확 재개. 실제 release는 미지원."""
import copy
import math
import os
import platform

import torch

from .data import InputContract
from .metrics import masked_mse, metric_sums
from .models import make_model
from .shards import EpochSampler, ShardDataset, _integer, _require, validate_catalog
from .training import (require_torch_api, restore_rng, rng_state, seed_run,
                       source_identity, validate_config)

FORMAT = 'm1_shard_checkpoint_v1'


class ShardTrainer:
    def __init__(self, config, catalog, base, device='cpu', check_budget=None):
        require_torch_api()
        self.config = validate_config(config)
        self.check_budget = check_budget or (lambda: None)
        self.catalog = validate_catalog(catalog)
        _require(config['window_size'] == self.catalog['window_size'], 'shard_model_window_mismatch')
        _require(config['batch_size'] <= self.catalog['limits']['max_records'], 'shard_batch_size_limit')
        _require(config['seed'] <= 2**32-1, 'shard_seed_limit')
        self.contract = InputContract.from_dict(self.catalog['contract'])
        # 카탈로그의 검증 문자열을 새 학습 release 권한으로 사용하지 않는다.
        _require(self.contract.data_kind == 'synthetic_fixture', 'shard_dataset_sample_release_not_implemented')
        _require(config['hidden'] <= 1024 and len(self.contract.joint_names) <= 256, 'shard_model_size_limit')
        self.dataset = ShardDataset(self.catalog, base)
        self.counts = self.dataset.verify_all(self.check_budget)
        _require({'train', 'validation'} <= set(self.counts), 'shard_training_requires_train_validation')
        self.device = torch.device(device)
        _require(self.device.type in ('cpu', 'cuda') and
                 (self.device.type != 'cuda' or torch.cuda.is_available()), 'shard_device_unavailable')
        _require(not config['amp'] or self.device.type == 'cuda', 'shard_amp_requires_cuda')
        if self.device.type == 'cuda':
            _require(os.environ.get('CUBLAS_WORKSPACE_CONFIG', ':4096:8') in (':4096:8', ':16:8'),
                     'shard_cuda_workspace')
        seed_run(config['seed'])
        self.model = make_model(config, len(self.contract.joint_names)*14, len(self.contract.joint_names)).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config['learning_rate'])
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, config['scheduler_step'], config['scheduler_gamma'])
        self.scaler = torch.amp.GradScaler('cuda', enabled=config['amp'])
        self.sampler = EpochSampler(self.catalog, 'train', config['seed'])
        self.step = self.samples_seen = self.attempts = self.skipped_updates = 0
        self.poisoned = False
        self.source = source_identity()
        self.environment = {'python':platform.python_version(), 'torch':str(torch.__version__),
            'device':str(self.device), 'cuda_runtime':torch.version.cuda, 'threads':torch.get_num_threads(),
            'amp':config['amp'], 'sampler_random_implementation':'python_random_v1'}
        if self.device.type == 'cuda':
            props = torch.cuda.get_device_properties(self.device)
            self.environment.update(gpu_name=props.name, gpu_total_memory_bytes=props.total_memory,
                gpu_capability=list(torch.cuda.get_device_capability(self.device)),
                cublas_workspace_config=os.environ['CUBLAS_WORKSPACE_CONFIG'])

    def _position_counts(self, sampler=None):
        sampler = sampler or self.sampler
        order = sampler._order()
        batch = self.config['batch_size']
        _require(sampler.offset % batch == 0, 'shard_cursor_batch_alignment')
        samples = sampler.epoch*sum(sampler.sizes.values())
        steps = sampler.epoch*sum(math.ceil(n/batch) for n in sampler.sizes.values())
        samples += sum(sampler.sizes[i] for i in order[:sampler.shard_position]) + sampler.offset
        steps += sum(math.ceil(sampler.sizes[i]/batch) for i in order[:sampler.shard_position]) + sampler.offset//batch
        return steps, samples

    def coverage(self):
        return {'completed_epochs': self.sampler.epoch, 'epoch_samples':self.counts['train'],
                'current_epoch_samples':self.samples_seen % self.counts['train'],
                'samples_seen':self.samples_seen, 'cursor':self.sampler.state_dict(),
                'whole_archive_coverage': 'not_measured'}

    def train_step(self):
        _require(not self.poisoned, 'shard_trainer_failed_reload_checkpoint_required')
        shard, indices = self.sampler.peek(self.config['batch_size'])
        # 같은 shard는 재사용한다. 새 shard로 교체할 때 loader가 이전 내부 cache를 먼저 비운다.
        self.attempts += 1
        try:
            _, splits, _ = self.dataset.load(shard)
            data = splits['train']
            keys = [list(data.anchors[i]) for i in indices]
            x = data.features[indices].to(self.device)
            valid = data.time_valid[indices].to(self.device)
            y = data.target_mm[indices, -1].to(self.device)
            mask = data.target_valid[indices, -1].to(self.device)
            self.model.train()
            self.optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=self.device.type, dtype=torch.float16, enabled=self.config['amp']):
                prediction = self.model(x, valid)[:, -1].float()*self.config['target_scale_mm']
                loss = masked_mse(prediction, y, mask, self.config['target_scale_mm'])
            _require(torch.isfinite(loss).item(), 'shard_nonfinite_loss')
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            finite = all(p.grad is None or torch.isfinite(p.grad).all().item() for p in self.model.parameters())
            _require(finite or self.config['amp'], 'shard_nonfinite_gradient')
            old_scale = self.scaler.get_scale()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            if self.scaler.get_scale() < old_scale:
                self.skipped_updates += 1
                return {'updated':False, 'step':self.step, 'amp_skipped':True, 'batch_samples':len(indices)}
            _require(all(torch.isfinite(p).all().item() for p in self.model.parameters()), 'shard_nonfinite_weights')
            _require(all(not isinstance(value, torch.Tensor) or torch.isfinite(value).all().item()
                         for state in self.optimizer.state.values() for value in state.values()),
                     'shard_nonfinite_optimizer_state')
            self.scheduler.step()
            self.sampler.commit(self.config['batch_size'])
            self.step += 1
            self.samples_seen += len(indices)
            return {'updated':True, 'step':self.step, 'loss':float(loss.detach()),
                    'batch_samples':len(indices), 'keys':keys, 'coverage':self.coverage()}
        except BaseException:
            # 부분 optimizer 변경이 있을 수 있다. 이 인스턴스의 checkpoint는 절대 게시하지 않는다.
            self.poisoned = True
            raise

    def evaluate(self, split='validation', final_test=False):
        _require(not self.poisoned, 'shard_trainer_failed_reload_checkpoint_required')
        _require(split in self.counts, 'shard_evaluation_split_missing')
        _require(split != 'test' or final_test is True, 'shard_test_requires_final_test')
        previous_mode, rng = self.model.training, rng_state()
        sums = {'distance_sum_mm':0., 'squared_sum_mm2':0., 'valid_joints':0}
        records = 0
        self.dataset.clear()
        try:
            self.model.eval()
            with torch.no_grad():
                for index, descriptor in enumerate(self.catalog['shards']):
                    self.check_budget()
                    if not any(s['split'] == split for s in descriptor['sequences']):
                        continue
                    _, splits, _ = self.dataset.load(index)
                    data = splits[split]
                    for start in range(0, len(data.anchors), self.config['batch_size']):
                        self.check_budget()
                        end = min(start+self.config['batch_size'], len(data.anchors))
                        x = data.features[start:end].to(self.device)
                        valid = data.time_valid[start:end].to(self.device)
                        prediction = self.model(x, valid)[:, -1].float()*self.config['target_scale_mm']
                        values = metric_sums(prediction, data.target_mm[start:end,-1].to(self.device),
                                             data.target_valid[start:end,-1].to(self.device))
                        for key in sums:
                            sums[key] += values[key]
                        records += end-start
                        del x, valid, prediction
                    del data, splits
                    self.dataset.clear()
        finally:
            self.dataset.clear()
            self.model.train(previous_mode)
            restore_rng(rng)
        count = sums['valid_joints']
        _require(count > 0, 'shard_evaluation_no_valid_joint')
        return {**sums, 'records':records, 'split':split, 'mpjpe_mm':sums['distance_sum_mm']/count,
                'mse_mm2':sums['squared_sum_mm2']/(3*count),
                'normalized_mse':sums['squared_sum_mm2']/(3*count*self.config['target_scale_mm']**2),
                'evaluation_kind':'synthetic_code_check', 'input_source':self.contract.input_source,
                'full_training_ready':False}

    def checkpoint(self):
        _require(not self.poisoned, 'shard_failed_state_cannot_checkpoint')
        _require(self._position_counts() == (self.step, self.samples_seen), 'shard_checkpoint_position')
        return {'format_version':FORMAT, 'config':copy.deepcopy(self.config), 'catalog':copy.deepcopy(self.catalog),
                'source':copy.deepcopy(self.source), 'environment':copy.deepcopy(self.environment),
                'model':self.model.state_dict(), 'optimizer':self.optimizer.state_dict(),
                'scheduler':self.scheduler.state_dict(), 'scaler':self.scaler.state_dict(), 'rng':rng_state(),
                'cursor':self.sampler.state_dict(), 'step':self.step, 'samples_seen':self.samples_seen,
                'attempts':self.attempts, 'skipped_updates':self.skipped_updates,
                'training_scope':'synthetic_fixture_only', 'full_training_ready':False}

    def resume(self, checkpoint):
        _require(isinstance(checkpoint, dict) and checkpoint.get('format_version') == FORMAT and
                 checkpoint.get('training_scope') == 'synthetic_fixture_only' and
                 checkpoint.get('full_training_ready') is False, 'shard_checkpoint_format')
        _require(checkpoint['config'] == self.config and checkpoint['catalog'] == self.catalog,
                 'shard_resume_config_catalog')
        _require(checkpoint['source']['code_sha256'] == self.source['code_sha256'] and
                 checkpoint['environment'] == self.environment, 'shard_resume_code_environment')
        sampler = EpochSampler(self.catalog, 'train', self.config['seed'])
        sampler.load_state_dict(checkpoint['cursor'])
        for key in ('step', 'samples_seen', 'attempts', 'skipped_updates'):
            _require(_integer(checkpoint[key], 0, 2**63-1), 'shard_checkpoint_counters')
        _require(self._position_counts(sampler) == (checkpoint['step'], checkpoint['samples_seen']) and
                 checkpoint['attempts'] == checkpoint['step']+checkpoint['skipped_updates'], 'shard_checkpoint_position')
        # 모든 state를 별도 객체에서 먼저 검증한다. 실패한 resume가 현재 학습을 일부 덮지 않는다.
        model = copy.deepcopy(self.model)
        expected_state = model.state_dict()
        supplied = checkpoint['model']
        _require(isinstance(supplied, dict) and set(supplied) == set(expected_state), 'shard_checkpoint_model')
        _require(all(isinstance(supplied[k], torch.Tensor) and supplied[k].shape == v.shape and
                     supplied[k].dtype == v.dtype and torch.isfinite(supplied[k]).all().item()
                     for k,v in expected_state.items()), 'shard_checkpoint_model')
        model.load_state_dict(checkpoint['model'], strict=True)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.config['learning_rate'])
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, self.config['scheduler_step'], self.config['scheduler_gamma'])
        scaler = torch.amp.GradScaler('cuda', enabled=self.config['amp'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        scaler.load_state_dict(checkpoint['scaler'])
        _require(scheduler.last_epoch == checkpoint['step'] and
                 scheduler.step_size == self.config['scheduler_step'] and
                 scheduler.gamma == self.config['scheduler_gamma'], 'shard_checkpoint_scheduler')
        for param in model.parameters():
            state = optimizer.state.get(param, {})
            if checkpoint['step'] == 0:
                _require(not state, 'shard_checkpoint_optimizer')
                continue
            _require(set(state) == {'step', 'exp_avg', 'exp_avg_sq'} and
                     isinstance(state['step'], torch.Tensor) and state['step'].numel() == 1 and
                     state['step'].item() == checkpoint['step'], 'shard_checkpoint_optimizer')
            _require(all(isinstance(state[k], torch.Tensor) and state[k].shape == param.shape and
                         state[k].dtype == param.dtype and torch.isfinite(state[k]).all().item()
                         for k in ('exp_avg', 'exp_avg_sq')), 'shard_checkpoint_optimizer')
        previous_rng = rng_state()
        try:
            restore_rng(checkpoint['rng'])
        finally:
            restore_rng(previous_rng)
        self.model, self.optimizer, self.scheduler, self.scaler = model, optimizer, scheduler, scaler
        restore_rng(checkpoint['rng'])
        self.sampler = sampler
        for key in ('step', 'samples_seen', 'attempts', 'skipped_updates'):
            setattr(self, key, checkpoint[key])
        self.poisoned = False
