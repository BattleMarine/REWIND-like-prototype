"""고정 실제102페어의 단일 pass CPU 진단. 정식 학습 gate를 해제하지 않는다."""
import argparse
import copy
import io
import json
import math
from pathlib import Path
import platform
import time
from types import SimpleNamespace
from zipfile import ZipFile

import psutil
import torch
from torch.nn import functional as F
from PIL import Image, __version__ as pillow_version

from rewind_archive.contracts import canonical, digest, safe_path
from rewind_archive.matching import identify
from rewind_data.audit import read_json
from rewind_prepare.converter import convert, validate_candidate
from .data import TensorSplit, content_hash
from .frontend import (StereoHeatmapNet, heatmap_loss, prediction_cache, weights_hash,
                       save_frontend_weights, read_prediction_cache)
from .local_diagnostic import _preserve_state
from .metrics import masked_mse, metric_sums
from .models import SingleFrameRegressor
from .real_gt2d_diagnostic import DiagnosticData, diagnostic_source_identity, equal_state, sha, tensor_sha
from .real_rgb_diagnostic import RESIZE, PREPROCESS, predicted_data as _predicted_data, errors_2d
from .training import seed_run, rng_state, restore_rng, save_checkpoint

ROOT = Path(__file__).resolve().parents[1]
ZIP_SHA = 'ec11b7eb5c4a69f76acb2e5f5045d8c7c560096a6dd33b2249e5d3a6a3e1dcae'
CONFIG_SHA = '414279e6af3049dd88400e425162f898d940b4e00f714872f6225291e0f2b42c'
VERSION = 'small_cnn_real_rgb102_64_diagnostic_v1'
MIB = 1024**2
CONFIG = {'batch_size': 2, 'learning_rate': 1e-4, 'frontend_seed': 71, 'regression_seed': 37,
          'updates_per_stage': 51, 'passes': 1, 'hidden': 32, 'dropout': 0.1,
          'target_scale_mm': 1000., 'scheduler': 'StepLR_step51_gamma1_constant',
          'sampler': 'sorted_unique_keys_without_replacement', 'device': 'cpu', 'precision': 'float32', 'threads': 1}
LIMITS = {'seconds': 600, 'rss_bytes': 2*1024*MIB, 'output_bytes': 512*MIB,
          'zip_bytes': 300*MIB, 'selected_read_bytes': 512*MIB, 'source_logical_read_bytes': 1024*MIB, 'updates': 102}


def require(value, reason):
    if not value:
        raise ValueError(reason)


def file_sha(path):
    import hashlib
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        while raw := stream.read(65536):
            h.update(raw)
    return h.hexdigest()


class Budget:
    def __init__(self, output):
        self.output, self.started = output, time.monotonic()
        self.read_bytes, self.peak_rss, self.updates = 0, 0, 0
        self.source_sha_bytes = 0

    def check(self):
        self.peak_rss = max(self.peak_rss, psutil.Process().memory_info().rss)
        require(self.peak_rss <= LIMITS['rss_bytes'], 'rgb102_rss_budget')
        require(time.monotonic()-self.started <= LIMITS['seconds'], 'rgb102_time_budget')
        require(self.updates <= LIMITS['updates'], 'rgb102_update_budget')
        require(self.source_sha_bytes+self.read_bytes <= LIMITS['source_logical_read_bytes'], 'rgb102_source_logical_read_budget')
        total = sum(p.stat().st_size for p in self.output.rglob('*') if p.is_file())
        require(total <= LIMITS['output_bytes']-4*MIB, 'rgb102_output_budget')

    def read(self, archive, name):
        self.check()
        info = archive.getinfo(name)
        require(0 <= info.file_size <= 8*MIB and self.read_bytes+info.file_size <= LIMITS['selected_read_bytes'], 'rgb102_member_read_budget')
        raw = archive.read(name)  # 사용 member를 끝까지 읽어 CRC도 검사한다.
        self.read_bytes += len(raw)
        require(len(raw) == info.file_size, 'rgb102_member_size_changed')
        return raw


def validate_inventory(archive):
    names = set()
    rows = archive.infolist()
    require(len(rows) <= 512 and sum(i.file_size for i in rows) <= 300*MIB, 'rgb102_zip_inventory_budget')
    for info in rows:
        require(not info.is_dir() and info.filename == safe_path(info.filename) and
                info.filename.casefold() not in names and not (info.flag_bits & 1) and
                (info.external_attr >> 16) & 0o170000 in (0, 0o100000), 'rgb102_zip_path_or_type')
        names.add(info.filename.casefold())


def load_input(path, output, budget, *, diagnostic_only):
    require(diagnostic_only is True, 'explicit_diagnostic_only_required')
    require(path.is_file() and not path.is_symlink() and path.stat().st_size <= LIMITS['zip_bytes'], 'rgb102_fixed_zip_size')
    require(file_sha(path) == ZIP_SHA, 'rgb102_fixed_zip_sha256')
    budget.source_sha_bytes = path.stat().st_size
    budget.check()
    with ZipFile(path) as z:
        validate_inventory(z)
        config_raw = budget.read(z, 'inputs/config.json')
        config = read_json(config_raw)
        require(digest(config) == CONFIG_SHA and len(config['pairs']) == 102, 'rgb102_fixed_config')
        run = read_json(budget.read(z, 'run/report.json'))
        require(run['all_targets_content_audited'] is True and run['complete_pairs'] == 102 and
                run['train_ready'] is False and run['config_sha256'] == CONFIG_SHA, 'rgb102_source_run_incomplete')
        pins = read_json(budget.read(z, 'inputs/input_manifest.json'))['files']
        # 소량 pose/metadata만 선택 추출해 기존 converter로 raw 수치 감사를 다시 실행한다.
        audit_root = output/'input_audit'
        for name, pin in pins.items():
            if not name.startswith(('pose/', 'metadata/')):
                continue
            raw = budget.read(z, 'inputs/'+name)
            require(len(raw) == pin['bytes'] and sha(raw) == pin['sha256'], 'rgb102_pinned_pose_metadata')
            target = audit_root/safe_path(name)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(raw)
        original = validate_candidate(read_json((audit_root/'pose/candidate.json').read_bytes()))
        require(original['content_sha256'] == config['candidate_sha256'] and
                original['contract']['data_kind'] == 'dataset_sample' and original['contract']['distance_unit'] == 'mm' and
                original['training_eligibility']['train_ready'] is False and original['training_eligibility']['blockers'], 'rgb102_candidate_scope')
        candidate, audit = convert(audit_root/'pose/files', audit_root/'pose/extraction_manifest.json', audit_root/'metadata/split',
            audit_root/'metadata/motion_map.json', ROOT/'configs/real_archvis62_v1.json', audit_root/'metadata/scaramuzza.json')
        require(candidate['records'] == original['records'] and candidate['contract'] == original['contract'] and
                candidate['training_eligibility'] == original['training_eligibility'], 'rgb102_raw_candidate_contract_changed')
        (output/'pose_audit.json').write_bytes(canonical(audit)+b'\n')
        rows = {(r['sequence_id'], r['frame_index']): r for r in candidate['records']}
        keys = sorted((p['sequence'], p['frame']) for p in config['pairs'])
        require(len(set(keys)) == 102 and set(keys) == set(rows) and all(r['split'] == 'train' for r in rows.values()), 'rgb102_unique_train102')
        rgb = {identify(r['name'], 'rgb'): r for r in run['files']}
        require(len(run['files']) == len(rgb) == 204, 'rgb102_unique_views204')
        images, gt, valid, source_views = [], [], [], []
        for pair in sorted(config['pairs'], key=lambda p: (p['sequence'], p['frame'])):
            key = (pair['sequence'], pair['frame'])
            views, points, masks, sources = [], [], [], {}
            for side in ('left', 'right'):
                expected, entry = pair['rgb'][side], rgb[(*key, side)]
                require(all(entry[k] == expected[k] for k in ('name', 'size', 'sha256', 'offset', 'ordinal')), 'rgb102_g9_payload_contract')
                raw = budget.read(z, 'run/'+safe_path(entry['path']))
                require(len(raw) == expected['size'] and sha(raw) == expected['sha256'], 'rgb102_png_payload_hash')
                with Image.open(io.BytesIO(raw)) as image:
                    require(image.format == 'PNG' and image.mode == 'RGB' and image.size == (1024, 1024), 'rgb102_native_png_contract')
                    image.load()
                    native = torch.frombuffer(bytearray(image.tobytes()), dtype=torch.uint8).reshape(1024, 1024, 3).permute(2, 0, 1).float().div_(255)
                    small = F.interpolate(native[None], size=(64, 64), mode='bilinear', align_corners=True, antialias=False)[0]
                    views.append(small)
                    del native
                field = rows[key]['fields']['camera_'+side+'_pts2d']
                points.append([point if ok else [0., 0.] for point, ok in zip(field['coordinates'], field['valid_mask'])])
                masks.append(field['valid_mask'])
                sources[side] = {k: expected[k] for k in ('name', 'sha256', 'size', 'offset', 'ordinal')}
                budget.check()
            images.append(torch.stack(views)); gt.append(points); valid.append(masks)
            source_views.append({'sequence_id': key[0], 'frame_index': key[1], 'views': sources})
    images, gt, valid = torch.stack(images), torch.tensor(gt, dtype=torch.float32), torch.tensor(valid, dtype=torch.bool)
    targets = torch.tensor([[[(p if ok else [0., 0., 0.]) for p, ok in zip(rows[k]['fields']['device_pts3d']['coordinates'], rows[k]['fields']['device_pts3d']['valid_mask'])]] for k in keys])
    target_mask = torch.tensor([[rows[k]['fields']['device_pts3d']['valid_mask']] for k in keys], dtype=torch.bool)
    require(images.shape == (102, 2, 3, 64, 64) and target_mask.any(-1).all(), 'rgb102_tensor_contract')
    manifest = {'schema': 'real_rgb102_diagnostic_input_v1', 'diagnostic_only': True, 'train_ready': False,
        'source_zip_sha256': ZIP_SHA, 'source_config_sha256': CONFIG_SHA, 'candidate_sha256': original['content_sha256'],
        'joint_names': candidate['contract']['joint_names'], 'source_views': source_views, 'resize': RESIZE,
        'image_tensor_sha256': tensor_sha(images), 'gt2d_sha256': tensor_sha(gt), 'annotation_mask_sha256': tensor_sha(valid),
        'target_sha256': tensor_sha(targets), 'target_mask_sha256': tensor_sha(target_mask), 'keys': keys,
        'blockers': candidate['training_eligibility']['blockers'], 'whole_archive_integrity_verified': False}
    # GT2D 특징은 regression 입력에 제공하지 않는다. predicted_data는 아래 GT3D/mask만 사용한다.
    tensors = TensorSplit(features=torch.zeros(102, 1, 868), time_valid=torch.ones(102, 1, dtype=torch.bool),
                          target_mm=targets, target_valid=target_mask, anchors=keys)
    require(tensors.features.shape == (102, 1, 868) and tensors.time_valid.shape == (102, 1) and
            tensors.time_valid.dtype == torch.bool and tensors.target_mm.shape == (102, 1, 62, 3) and
            tensors.target_mm.dtype == torch.float32 and tensors.target_valid.shape == (102, 1, 62) and
            tensors.target_valid.dtype == torch.bool, 'rgb102_named_tensor_shape_dtype')
    data = DiagnosticData(tensors, {'joint_names': manifest['joint_names'], 'manifest_sha256': digest(manifest)},
                          candidate['contract'], list(manifest['blockers']))
    return images, gt, valid, data, manifest


def predicted_data(predictions, data, cache):
    return _predicted_data(predictions, data, cache, frontend_version=VERSION)


class Stage:
    def __init__(self, role, identity):
        require(role in ('frontend', 'regression'), 'rgb102_stage_role')
        self.role, self.identity, self.cursor, self.step = role, identity, 0, 0
        seed_run(CONFIG[role+'_seed'])
        self.model = StereoHeatmapNet(62) if role == 'frontend' else SingleFrameRegressor(868, 62, hidden=32, dropout=0.1)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=CONFIG['learning_rate'])
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, 51, 1.0)
        self.scaler = torch.amp.GradScaler('cuda', enabled=False)
        self.source = diagnostic_source_identity()
        self.environment = {'python': platform.python_version(), 'torch': str(torch.__version__), 'pillow': pillow_version,
                            'device': 'cpu', 'precision': 'float32', 'threads': torch.get_num_threads(), 'amp': False}

    def checkpoint(self):
        return {'format_version': 'real_rgb102_diagnostic_checkpoint_v1', 'role': 'diagnostic_only_rgb102_'+self.role,
            'diagnostic_only': True, 'train_ready': False, 'config': CONFIG, 'data_identity': self.identity,
            'model': self.model.state_dict(), 'optimizer': self.optimizer.state_dict(), 'scheduler': self.scheduler.state_dict(),
            'scaler': self.scaler.state_dict(), 'rng': rng_state(), 'step': self.step, 'cursor': self.cursor,
            'samples_seen': self.cursor, 'source': self.source, 'environment': self.environment}

    def restore(self, state):
        expected = self.checkpoint()
        for k in ('format_version', 'role', 'diagnostic_only', 'train_ready', 'config', 'data_identity', 'source', 'environment'):
            require(state[k] == expected[k], 'rgb102_resume_identity_or_role')
        require(type(state['step']) is int and 0 <= state['step'] <= 51 and
                state['cursor'] == state['samples_seen'] == state['step']*2, 'rgb102_resume_cursor')
        for k in ('model', 'optimizer', 'scheduler', 'scaler'):
            getattr(self, k).load_state_dict(state[k])
        self.step, self.cursor = state['step'], state['cursor']
        restore_rng(state['rng'])

    def update(self, loss, budget):
        budget.check()
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradients = [p.grad for p in self.model.parameters()]
        require(torch.isfinite(loss).item() and all(g is not None and torch.isfinite(g).all().item() for g in gradients), 'rgb102_nonfinite_gradient')
        norm = math.sqrt(sum(float(g.square().sum()) for g in gradients))
        require(norm > 0 and math.isfinite(norm), 'rgb102_zero_gradient')
        self.optimizer.step(); self.scheduler.step()
        self.step += 1; self.cursor += 2; budget.updates += 1
        require(all(torch.isfinite(p).all().item() for p in self.model.parameters()), 'rgb102_nonfinite_weights')
        budget.check()
        return {'step': self.step, 'cursor': self.cursor, 'loss': float(loss.detach()), 'gradient_norm_l2': norm}


def frontend_eval(model, images, gt, valid, keys, names, manifest, budget):
    model.eval()
    loss_sum, inside_total, records, cache = 0., 0, [], None
    with torch.no_grad():
        for start in range(0, 102, 2):
            budget.check()
            logits = model(images[start:start+2])[0]
            loss, inside = heatmap_loss(logits, gt[start:start+2]*torch.tensor(RESIZE['gt_to_network_scale']), valid[start:start+2], (64, 64))
            count = int(inside.sum()); loss_sum += float(loss)*count; inside_total += count
            part = prediction_cache(model, images[start:start+2], keys[start:start+2], names, VERSION, PREPROCESS, digest(manifest), 'dataset_sample')
            records.extend(part['records']); cache = part
    cache['records'] = records
    cache['content_sha256'] = content_hash({k: v for k, v in cache.items() if k != 'content_sha256'})
    native = copy.deepcopy(cache); native['image_sizes'] = [[1024, 1024]]*2
    for row in native['records']:
        for side in ('left', 'right'):
            row['prediction'][side]['coordinates'] = [[x*1023/63, y*1023/63] for x, y in row['prediction'][side]['coordinates']]
    native['content_sha256'] = content_hash({k: v for k, v in native.items() if k != 'content_sha256'})
    return {'heatmap_mse': loss_sum/inside_total, 'inside_joint_views': inside_total,
            'annotation_valid_joint_views': int(valid.sum()), 'evaluation_kind': 'train_fit_diagnostic'}, cache, native


def regression_eval(model, data, budget):
    model.eval(); predictions = []
    with torch.no_grad():
        for start in range(0, 102, 2):
            budget.check()
            predictions.append(model(data.features[start:start+2], data.time_valid[start:start+2])[:, -1]*1000.)
    predictions = torch.cat(predictions)
    sums = metric_sums(predictions, data.target_mm[:, -1], data.target_valid[:, -1])
    loss = masked_mse(predictions, data.target_mm[:, -1], data.target_valid[:, -1], 1000.)
    return {'normalized_mse': float(loss), 'mpjpe_mm': sums['distance_sum_mm']/sums['valid_joints'],
            'valid_joints': sums['valid_joints'], 'evaluation_kind': 'train_fit_diagnostic'}, predictions


def native_predictions(native, data, manifest, weight_hash):
    contract = dict(data.contract, input_source='predicted_2d', frontend_version=VERSION,
                    frontend_checkpoint_sha256=weight_hash, preprocessing_version=PREPROCESS)
    mapped = read_prediction_cache(native, SimpleNamespace(**contract), digest(manifest))
    require(set(mapped) == set(data.tensors.anchors), 'rgb102_missing_prediction_no_gt_fallback')
    return mapped


def validate_cache(cache, data, manifest, weights_path):
    require(cache.get('schema_version') == 'real_rgb102_diagnostic_cache_v1' and
            cache.get('diagnostic_only') is True and cache.get('train_ready') is False and
            cache.get('content_sha256') == digest({k: v for k, v in cache.items() if k != 'content_sha256'}), 'rgb102_cache_schema_hash')
    require(cache['resize'] == RESIZE and cache['input_manifest_sha256'] == digest(manifest) and
            cache['weights_file_sha256'] == file_sha(weights_path), 'rgb102_cache_provenance')
    state = torch.load(weights_path, map_location='cpu', weights_only=True)
    expected = {'format_version': 'm1_frontend_weights_v1', 'weights_only': True,
        'model_config': {'architecture': 'small_cnn_v1', 'joints': 62, 'hidden_channels': 16, 'stride': 2},
        'joint_names': manifest['joint_names'], 'image_sizes': [[64, 64]]*2, 'frontend_version': VERSION,
        'preprocessing_version': PREPROCESS, 'frontend_checkpoint_sha256': cache['weights_state_sha256'],
        'data_kind': 'dataset_sample', 'manifest_hash': digest(manifest)}
    require(all(state.get(k) == value for k, value in expected.items()), 'rgb102_weights_contract')
    model = StereoHeatmapNet(62); model.load_state_dict(state['model'])
    require(weights_hash(model) == cache['weights_state_sha256'], 'rgb102_weights_state_hash')
    native = native_predictions(cache['native_cache'], data, manifest, cache['weights_state_sha256'])
    contract = dict(data.contract, input_source='predicted_2d', frontend_version=VERSION,
        frontend_checkpoint_sha256=cache['weights_state_sha256'], preprocessing_version=PREPROCESS, image_sizes=[[64, 64]]*2)
    network = read_prediction_cache(cache['network_cache'], SimpleNamespace(**contract), digest(manifest))
    require(set(network) == set(native), 'rgb102_cache_keys')
    for key in network:
        for side in ('left', 'right'):
            a, b = network[key][side], native[key][side]
            require(all(a[k] == b[k] for k in ('valid_mask', 'confidence', 'visibility')), 'rgb102_cache_mask_or_confidence')
            require(b['coordinates'] == [[p[0]*1023/63, p[1]*1023/63] if ok else None
                    for p, ok in zip(a['coordinates'], a['valid_mask'])], 'rgb102_cache_native_transform')
    return native


def run(source_zip, output, *, diagnostic_only):
    require(diagnostic_only is True, 'explicit_diagnostic_only_required')
    source_zip, output = Path(source_zip).absolute(), Path(output).absolute()
    require(not output.exists() and not source_zip.is_relative_to(output), 'rgb102_new_output_required')
    output.mkdir(parents=True)
    budget = Budget(output)
    report = {'schema': 'real_rgb102_train_fit_diagnostic_v1', 'status': 'started', 'diagnostic_only': True,
        'train_ready': False, 'evaluation_kind': 'train_fit_diagnostic', 'data_kind': 'dataset_sample',
        'input_source': 'predicted_2d', 'temporal_experiment': False, 'validation_or_test_run': False,
        'remote_requests': 0, 'limits': LIMITS, 'config': CONFIG, 'trace': {'frontend': [], 'regression': []},
        'coverage': {}, 'uninterrupted_resume_equivalence_tested': False}
    def save(name, value):
        (output/name).write_bytes(canonical(value)+b'\n')
    try:
        with _preserve_state():
            torch.set_num_threads(1)
            images, gt, valid, data, manifest = load_input(source_zip, output, budget, diagnostic_only=diagnostic_only)
            save('input_manifest.json', manifest)
            report.update(input_manifest_sha256=digest(manifest), blockers=data.blockers,
                          prepared_input_shape=list(images.shape), target_shape=list(data.tensors.target_mm.shape))
            front = Stage('frontend', manifest)
            initial_hash = weights_hash(front.model)
            report.update(source=front.source, environment=front.environment)
            fit, _, initial_native = frontend_eval(front.model, images, gt, valid, data.tensors.anchors, manifest['joint_names'], manifest, budget)
            report['initial_frontend_fit'] = fit
            report['initial_2d'] = errors_2d(native_predictions(initial_native, data, manifest, initial_hash), data, gt, valid)
            for stage in ('frontend', 'regression'):
                if stage == 'regression':
                    front.model.eval()
                    fit, network, native = frontend_eval(front.model, images, gt, valid, data.tensors.anchors, manifest['joint_names'], manifest, budget)
                    report['final_frontend_fit'] = fit
                    save_frontend_weights(front.model, network, output/'frontend_weights.pt')
                    weight_state = torch.load(output/'frontend_weights.pt', map_location='cpu', weights_only=True)
                    restored_front = StereoHeatmapNet(62); restored_front.load_state_dict(weight_state['model'])
                    _, repeat_network, repeat_native = frontend_eval(restored_front, images, gt, valid, data.tensors.anchors, manifest['joint_names'], manifest, budget)
                    report['frontend_weights_reload_cache_equal'] = network == repeat_network and native == repeat_native
                    require(report['frontend_weights_reload_cache_equal'], 'rgb102_frontend_reload_mismatch')
                    cache = {'schema_version': 'real_rgb102_diagnostic_cache_v1', 'diagnostic_only': True, 'train_ready': False,
                        'network_cache': network, 'native_cache': native, 'resize': RESIZE, 'input_manifest_sha256': digest(manifest),
                        'weights_file_sha256': file_sha(output/'frontend_weights.pt'), 'weights_state_sha256': weights_hash(front.model)}
                    cache['content_sha256'] = digest(cache); save('prediction_cache.json', cache)
                    cache = read_json((output/'prediction_cache.json').read_bytes())
                    predictions = validate_cache(cache, data, manifest, output/'frontend_weights.pt')
                    pred_data = predicted_data(predictions, data, cache)
                    report['predicted_data_identity'] = pred_data.identity
                    report['final_2d'] = errors_2d(predictions, data, gt, valid)
                    trainer = Stage('regression', pred_data.identity)
                    report['initial_regression_fit'] = regression_eval(trainer.model, pred_data.tensors, budget)[0]
                else:
                    trainer = front
                before_hash = weights_hash(trainer.model)
                before_parameters = {name: value.detach().clone() for name, value in trainer.model.named_parameters()}
                seen = []
                for start in range(0, 102, 2):
                    require(trainer.cursor == start and trainer.step < 51, 'rgb102_one_pass_cursor')
                    trainer.model.train()
                    if stage == 'frontend':
                        loss = heatmap_loss(trainer.model(images[start:start+2])[0], gt[start:start+2]*torch.tensor(RESIZE['gt_to_network_scale']), valid[start:start+2], (64, 64))[0]
                    else:
                        t = pred_data.tensors
                        loss = masked_mse(trainer.model(t.features[start:start+2], t.time_valid[start:start+2])[:, -1]*1000., t.target_mm[start:start+2, -1], t.target_valid[start:start+2, -1], 1000.)
                    report['trace'][stage].append(trainer.update(loss, budget)); seen.extend(data.tensors.anchors[start:start+2])
                    if trainer.step == 25:
                        checkpoint_path = output/(stage+'_025.pt')
                        save_checkpoint(trainer.checkpoint(), checkpoint_path)
                        state = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
                        restored = Stage(stage, trainer.identity); restored.restore(state)
                        report[stage+'_resume_loaded_state_equal'] = equal_state(state, restored.checkpoint())
                        require(report[stage+'_resume_loaded_state_equal'], 'rgb102_resume_state_mismatch')
                        trainer = restored
                require(seen == data.tensors.anchors and len(set(seen)) == 102, 'rgb102_coverage_failed')
                report['coverage'][stage] = {'keys_seen': seen, 'unique_frames': 102, 'samples_seen': trainer.cursor, 'updates': trainer.step, 'passes': 1}
                report[stage+'_weights_changed'] = weights_hash(trainer.model) != before_hash
                report[stage+'_weights_delta_l2'] = math.sqrt(sum(float((p.detach()-before_parameters[name]).square().sum())
                    for name, p in trainer.model.named_parameters()))
                require(report[stage+'_weights_changed'], 'rgb102_weights_unchanged')
                save_checkpoint(trainer.checkpoint(), output/(stage+'_051.pt'))
                if stage == 'frontend':
                    front = trainer
                else:
                    fit, prediction = regression_eval(trainer.model, pred_data.tensors, budget)
                    report['final_regression_fit'] = fit
                    weights = {'format_version': 'real_rgb102_regression_weights_v1', 'diagnostic_only': True, 'train_ready': False,
                        'model': trainer.model.state_dict(), 'config': CONFIG, 'data_identity': trainer.identity, 'blockers': data.blockers}
                    save_checkpoint(weights, output/'regression_weights.pt')
                    loaded = torch.load(output/'regression_weights.pt', map_location='cpu', weights_only=True)
                    restored_regression = SingleFrameRegressor(868, 62, 32, 0.1); restored_regression.load_state_dict(loaded['model'])
                    repeated = regression_eval(restored_regression, pred_data.tensors, budget)[1]
                    report['regression_weights_reload_prediction_equal'] = torch.equal(prediction, repeated)
                    require(report['regression_weights_reload_prediction_equal'], 'rgb102_regression_reload_mismatch')
            report['finite_nonzero_gradients'] = all(math.isfinite(r['gradient_norm_l2']) and r['gradient_norm_l2'] > 0 for trace in report['trace'].values() for r in trace)
            report['frontend_loss_decreased'] = report['final_frontend_fit']['heatmap_mse'] < report['initial_frontend_fit']['heatmap_mse']
            report['regression_loss_decreased'] = report['final_regression_fit']['normalized_mse'] < report['initial_regression_fit']['normalized_mse']
            require(budget.updates == 102 and report['finite_nonzero_gradients'], 'rgb102_update_count_or_gradient')
            budget.check(); report['status'] = 'completed_diagnostic'
    except BaseException as exc:
        report.update(status='failed_or_interrupted', error_type=type(exc).__name__)
        raise
    finally:
        report.update(updates_executed=budget.updates, elapsed_seconds=time.monotonic()-budget.started,
            observed_peak_rss=budget.peak_rss, selected_zip_read_bytes=budget.read_bytes,
            source_zip_sha256_read_bytes=budget.source_sha_bytes,
            source_logical_read_bytes=budget.source_sha_bytes+budget.read_bytes,
            read_accounting_scope='source ZIP full SHA bytes + selected uncompressed member bytes; excludes ZIP headers/compressed backend reads and output audit/checkpoint reload',
            output_bytes_before_report=sum(p.stat().st_size for p in output.rglob('*') if p.is_file()))
        save('report.json', report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-zip', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--diagnostic-only', action='store_true')
    args = parser.parse_args(argv)
    report = run(args.source_zip, args.output, diagnostic_only=args.diagnostic_only)
    print(json.dumps({k: report[k] for k in ('status', 'updates_executed', 'elapsed_seconds', 'initial_frontend_fit',
        'final_frontend_fit', 'initial_2d', 'final_2d', 'initial_regression_fit', 'final_regression_fit', 'train_ready')}, indent=2))


if __name__ == '__main__':
    main()
