"""고정102 입력·cache/weights·진단역할 계약 검사. 학습 update를 추가하지 않는다."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

import torch
from rewind_archive.contracts import digest
from rewind_m1 import real_rgb102_diagnostic as rgb
from rewind_m1.frontend import StereoHeatmapNet, save_frontend_weights, weights_hash
from rewind_m1.local_diagnostic import _preserve_state
from rewind_m1.real_gt2d_diagnostic import equal_state

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT/'outputs/test_rgb102_pairs_001_20261103/rgb102_pairs_001_result.zip'


@unittest.skipUnless(SOURCE.is_file(), '검토된 실제102 입력 ZIP 필요')
class RGB102DiagnosticTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.context = _preserve_state(); cls.context.__enter__()
        cls.addClassCleanup(cls.context.__exit__, None, None, None)
        torch.set_num_threads(1)
        parent = ROOT/'outputs/test_tmp'; parent.mkdir(parents=True, exist_ok=True)
        cls.temp = tempfile.TemporaryDirectory(dir=parent); cls.addClassCleanup(cls.temp.cleanup)
        cls.output = Path(cls.temp.name)
        cls.budget = rgb.Budget(cls.output)
        cls.images, cls.gt, cls.valid, cls.data, cls.manifest = rgb.load_input(SOURCE, cls.output, cls.budget, diagnostic_only=True)
        cls.stage = rgb.Stage('frontend', cls.manifest)
        cls.fit, network, native = rgb.frontend_eval(cls.stage.model, cls.images, cls.gt, cls.valid,
            cls.data.tensors.anchors, cls.manifest['joint_names'], cls.manifest, cls.budget)
        cls.weights = cls.output/'weights.pt'; save_frontend_weights(cls.stage.model, network, cls.weights)
        cls.cache = {'schema_version': 'real_rgb102_diagnostic_cache_v1', 'diagnostic_only': True, 'train_ready': False,
            'network_cache': network, 'native_cache': native, 'resize': rgb.RESIZE, 'input_manifest_sha256': digest(cls.manifest),
            'weights_file_sha256': rgb.file_sha(cls.weights), 'weights_state_sha256': weights_hash(cls.stage.model)}
        cls.cache['content_sha256'] = digest(cls.cache)

    def reseal(self, cache):
        for k in ('network_cache', 'native_cache'):
            cache[k]['content_sha256'] = digest({key: val for key, val in cache[k].items() if key != 'content_sha256'})
        cache['content_sha256'] = digest({key: val for key, val in cache.items() if key != 'content_sha256'})
        return cache

    def test_all102_native_sources_and_small_tensor_masks_resize(self):
        self.assertEqual(tuple(self.images.shape), (102, 2, 3, 64, 64))
        self.assertEqual(tuple(self.data.tensors.target_mm.shape), (102, 1, 62, 3))
        self.assertEqual(self.data.tensors.target_mm.dtype, torch.float32)
        self.assertEqual(self.data.tensors.time_valid.shape, (102, 1))
        self.assertEqual(self.data.tensors.time_valid.dtype, torch.bool)
        self.assertEqual(self.data.tensors.target_valid.dtype, torch.bool)
        candidate = json.loads((self.output/'input_audit/pose/candidate.json').read_bytes())
        by_key = {(r['sequence_id'], r['frame_index']): r for r in candidate['records']}
        for i, key in enumerate(self.data.tensors.anchors):
            field = by_key[key]['fields']['device_pts3d']
            self.assertTrue(torch.equal(self.data.tensors.target_valid[i, 0], torch.tensor(field['valid_mask'])))
            self.assertTrue(torch.equal(self.data.tensors.target_mm[i, 0], torch.tensor([
                p if ok else [0., 0., 0.] for p, ok in zip(field['coordinates'], field['valid_mask'])])))
        self.assertEqual(len(set(self.data.tensors.anchors)), 102)
        self.assertEqual(len(self.manifest['source_views']), 102)
        self.assertEqual(rgb.RESIZE['gt_to_network_scale'], [63/1023]*2)
        self.assertLess(self.images.numel()*self.images.element_size(), 16*1024**2)
        self.assertFalse(self.data.tensors.features.any())
        self.assertEqual(self.data.contract['training_release'], 'not_implemented')
        self.assertTrue(self.data.blockers)
        self.assertTrue(self.valid.all())

    def test_disk_weights_and_cache_reload_exact_native_transform(self):
        predictions = rgb.validate_cache(copy.deepcopy(self.cache), self.data, self.manifest, self.weights)
        self.assertEqual(set(predictions), set(self.data.tensors.anchors))
        predicted = rgb.predicted_data(predictions, self.data, self.cache)
        self.assertEqual(predicted.contract['input_source'], 'predicted_2d')
        self.assertTrue(torch.equal(predicted.tensors.target_mm, self.data.tensors.target_mm))
        self.assertTrue(torch.equal(predicted.tensors.target_valid, self.data.tensors.target_valid))

    def test_cache_transform_hash_and_missing_key_rejected(self):
        bad = copy.deepcopy(self.cache); bad['native_cache']['records'].pop()
        with self.assertRaises(Exception):
            rgb.validate_cache(self.reseal(bad), self.data, self.manifest, self.weights)
        bad = copy.deepcopy(self.cache); bad['native_cache']['records'][0]['prediction']['left']['coordinates'][0][0] += 1
        with self.assertRaisesRegex(ValueError, 'native_transform'):
            rgb.validate_cache(self.reseal(bad), self.data, self.manifest, self.weights)
        bad = copy.deepcopy(self.cache); bad['content_sha256'] = '0'*64
        with self.assertRaisesRegex(ValueError, 'schema_hash'):
            rgb.validate_cache(bad, self.data, self.manifest, self.weights)

    def test_wrong_weights_contract_and_weight_bytes_rejected(self):
        raw = torch.load(self.weights, map_location='cpu', weights_only=True)
        raw['data_kind'] = 'synthetic_fixture'
        wrong = self.output/'wrong_weights.pt'; torch.save(raw, wrong)
        bad = copy.deepcopy(self.cache); bad['weights_file_sha256'] = rgb.file_sha(wrong)
        with self.assertRaisesRegex(ValueError, 'weights_contract'):
            rgb.validate_cache(self.reseal(bad), self.data, self.manifest, wrong)
        with self.assertRaisesRegex(ValueError, 'provenance'):
            rgb.validate_cache(self.cache, self.data, self.manifest, wrong)

    def test_no_gt_fallback_and_gt_feature_changes_do_not_change_predictions(self):
        predictions = rgb.validate_cache(self.cache, self.data, self.manifest, self.weights)
        before = rgb.predicted_data(predictions, self.data, self.cache)
        changed = copy.deepcopy(self.data); changed.tensors.features.fill_(999)
        after = rgb.predicted_data(predictions, changed, self.cache)
        self.assertTrue(torch.equal(before.tensors.features, after.tensors.features))
        predictions.pop(next(iter(predictions)))
        with self.assertRaisesRegex(ValueError, 'predictions_required'):
            rgb.predicted_data(predictions, self.data, self.cache)

    def test_checkpoint_cursor_role_full_state_and_fixed_config(self):
        stage = rgb.Stage('frontend', self.manifest)
        state = copy.deepcopy(stage.checkpoint())
        restored = rgb.Stage('frontend', self.manifest); restored.restore(state)
        self.assertTrue(equal_state(state, restored.checkpoint()))
        bad = copy.deepcopy(state); bad['cursor'] = 1
        with self.assertRaisesRegex(ValueError, 'resume_cursor'):
            restored.restore(bad)
        bad = copy.deepcopy(state); bad['role'] = 'diagnostic_only_regression'
        with self.assertRaisesRegex(ValueError, 'resume_identity_or_role'):
            restored.restore(bad)
        self.assertEqual(rgb.CONFIG['learning_rate'], 1e-4)
        self.assertEqual(stage.optimizer.param_groups[0]['lr'], 1e-4)
        self.assertEqual([key for i in range(0, 102, 2) for key in self.data.tensors.anchors[i:i+2]], self.data.tensors.anchors)

    def test_explicit_diagnostic_and_source_pin_required_general_gate_stays_closed(self):
        from rewind_m1.data import load_document, DataContractError
        with self.assertRaisesRegex(ValueError, 'explicit_diagnostic_only'):
            rgb.load_input(SOURCE, self.output, self.budget, diagnostic_only=False)
        with self.assertRaises(DataContractError):
            load_document(json.loads((self.output/'input_audit/pose/candidate.json').read_bytes()), 1)
