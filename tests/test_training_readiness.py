"""제한된 로컬 특징 shard/순회/추론 파일 계약. 학습·네트워크는 실행하지 않는다."""
import contextlib
import copy
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import torch

from rewind_m1.artifacts import export_weights, load_weights, _state_hash
from rewind_m1.data import DataContractError, content_hash, load_document, synthetic_document
from rewind_m1.models import make_model
from rewind_m1.readiness import inspect_catalog, main
from rewind_m1.shards import (DEFAULT_LIMITS, EpochSampler, ShardDataset, build_catalog,
                              read_catalog, validate_catalog)

ROOT = Path(__file__).resolve().parents[1]


def signed(doc):
    return {**doc, 'content_sha256': content_hash({k:v for k,v in doc.items() if k != 'content_sha256'})}


class ReadinessTests(unittest.TestCase):
    def setUp(self):
        target = ROOT/'outputs'/'test_tmp'
        target.mkdir(parents=True, exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=target)
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.documents = []
        for index, split in enumerate(('train', 'validation')):
            doc = synthetic_document()
            doc['records'] = [r for r in doc['records'] if r['split'] == split]
            self.documents.append(doc)
            self.write(f'{index}.json', doc)
        self.catalog = build_catalog(['0.json', '1.json'], self.base, window_size=3)
        self.write('catalog.json', self.catalog)

    def write(self, name, doc):
        (self.base/name).write_text(json.dumps(doc), encoding='utf8')

    def test_tensor_equivalence_and_one_cache(self):
        dataset = ShardDataset(self.catalog, self.base)
        self.assertIsNone(dataset._cached)
        for index, doc in enumerate(self.documents):
            actual = dataset.load(index)
            expected = load_document(doc, 3)
            self.assertEqual(actual[0], expected[0])
            for split in expected[1]:
                a, b = actual[1][split], expected[1][split]
                self.assertEqual(a.anchors, b.anchors)
                for key in ('features', 'time_valid', 'target_mm', 'target_valid'):
                    self.assertTrue(torch.equal(getattr(a, key), getattr(b, key)))
            self.assertEqual(dataset._index, index)
            self.assertIs(dataset.load(index), actual)
        self.assertEqual(dataset.verify_all(), {'train':12, 'validation':12})
        self.assertIsNone(dataset._cached)

    def test_hash_change_and_relative_escape(self):
        (self.base/'0.json').write_bytes(b'{}')
        with self.assertRaisesRegex(DataContractError, 'shard_payload_identity'):
            ShardDataset(self.catalog, self.base).load(0)
        for path in ('../0.json', '/0.json', 'C:/0.json', './0.json', 'a//b', 'a\\b'):
            doc = copy.deepcopy(self.catalog)
            doc['shards'][0]['path'] = path
            with self.assertRaisesRegex(DataContractError, 'shard_relative_path'):
                validate_catalog(signed(doc))

    def test_payload_declaration_and_contract_mismatch(self):
        import hashlib
        for key in ('records', 'contract', 'manifest'):
            doc = copy.deepcopy(self.documents[0])
            if key == 'records':
                doc[key][0]['motion_id'] = 'other_motion'
                error = 'shard_sequence_identity_changed'
            elif key == 'contract':
                doc[key]['preprocessing_version'] = 'other'
                error = 'shard_contract_mismatch'
            else:
                doc[key]['dataset_version'] = 'other'
                error = 'shard_dataset_identity_mismatch'
            self.write('0.json', doc)
            raw = (self.base/'0.json').read_bytes()
            catalog = copy.deepcopy(self.catalog)
            catalog['shards'][0].update(sha256=hashlib.sha256(raw).hexdigest(), size_bytes=len(raw))
            with self.assertRaisesRegex(DataContractError, error):
                ShardDataset(signed(catalog), self.base).load(0)

    def test_split_leak_sequence_boundary_and_dataset_version(self):
        changed = copy.deepcopy(self.documents[1])
        changed['records'][0]['motion_id'] = self.documents[0]['records'][0]['motion_id']
        # 같은 sequence의 일관성을 보존한 실제 split 누수 입력.
        sequence = changed['records'][0]['sequence_id']
        for row in changed['records']:
            if row['sequence_id'] == sequence:
                row['motion_id'] = changed['records'][0]['motion_id']
        self.write('1.json', changed)
        with self.assertRaisesRegex(DataContractError, 'shard_motion_split_overlap'):
            build_catalog(['0.json', '1.json'], self.base, 3)
        self.write('1.json', self.documents[0])
        with self.assertRaisesRegex(DataContractError, 'shard_sequence_crosses_shards'):
            build_catalog(['0.json', '1.json'], self.base, 3)
        changed = copy.deepcopy(self.documents[1])
        changed['manifest']['dataset_version'] = 'different'
        self.write('1.json', changed)
        with self.assertRaisesRegex(DataContractError, 'shard_dataset_identity_mismatch'):
            build_catalog(['0.json', '1.json'], self.base, 3)

    def test_limits_before_tensor_allocation(self):
        from unittest.mock import patch
        for key, value, error in (('max_records', 1, 'shard_record_limit'),
                                  ('max_shard_bytes', 10, 'shard_byte_limit'),
                                  ('max_tensor_elements', 10, 'shard_tensor_element_limit')):
            limits = {**DEFAULT_LIMITS, key:value}
            with patch('rewind_m1.shards.load_document', side_effect=AssertionError('must not allocate')):
                with self.assertRaisesRegex(DataContractError, error):
                    build_catalog(['0.json', '1.json'], self.base, 3, limits)

    def test_epoch_exact_coverage_and_resume(self):
        sampler = EpochSampler(self.catalog, 'train', 73)
        seen = []
        for _ in range(2):
            shard, indices = sampler.peek(5)
            self.assertEqual((shard, indices), sampler.peek(5))
            seen += indices
            sampler.commit(5)
        resumed = EpochSampler(self.catalog, 'train', 73)
        resumed.load_state_dict(sampler.state_dict())
        self.assertEqual(resumed.peek(5), sampler.peek(5))
        seen += resumed.peek(5)[1]
        resumed.commit(5)
        self.assertEqual(sorted(seen), list(range(12)))
        self.assertEqual(resumed.epoch, 1)
        order = EpochSampler(self.catalog, 'train', 73, shuffle=False)
        self.assertEqual(order.peek(5), (0, list(range(5))))

    def test_cursor_identity_and_position_rejection(self):
        sampler = EpochSampler(self.catalog, 'train', 3)
        for key, value in (('offset', 12), ('epoch', True), ('seed', 9), ('shuffle', 1),
                           ('catalog_sha256', '0'*64), ('shard_position', 9)):
            state = sampler.state_dict()
            state[key] = value
            with self.assertRaisesRegex(DataContractError, 'shard_cursor_'):
                sampler.load_state_dict(state)
        self.assertEqual(sampler.epoch, 0)
        self.assertEqual(sampler.offset, 0)

    def test_multiple_train_shards_once_per_epoch(self):
        paths = []
        for clip in range(2):
            doc = copy.deepcopy(self.documents[0])
            doc['records'] = [r for r in doc['records'] if r['sequence_id'] == f'train_{clip}']
            name = f'train_{clip}.json'
            self.write(name, doc)
            paths.append(name)
        catalog = build_catalog(paths+['1.json'], self.base, 3)
        sampler = EpochSampler(catalog, 'train', 13)
        seen = []
        while sampler.epoch == 0:
            shard, indices = sampler.peek(4)
            seen.extend((shard, i) for i in indices)
            sampler.commit(4)
        self.assertEqual(len(seen), 12)
        self.assertEqual(len(set(seen)), 12)
        self.assertEqual(set(i for i,_ in seen), {0,1})

    def test_missing_target_anchor_is_blocked(self):
        doc = copy.deepcopy(self.documents[0])
        for joint in doc['records'][0]['joints'].values():
            joint['device_pts3d'] = None
        self.write('0.json', doc)
        with self.assertRaisesRegex(DataContractError, 'shard_anchor_without_target'):
            build_catalog(['0.json', '1.json'], self.base, 3)

    def test_predicted_feature_contract_and_target_mask_preserved(self):
        for index, original in enumerate(self.documents):
            doc = copy.deepcopy(original)
            doc['contract'].update(input_source='predicted_2d', frontend_version='synthetic_prediction_fixture',
                                   frontend_checkpoint_sha256='a'*64)
            for row in doc['records']:
                row['prediction'] = {'joint_names':doc['contract']['joint_names'],
                    'frontend_version':'synthetic_prediction_fixture', 'frontend_checkpoint_sha256':'a'*64,
                    'preprocessing_version':doc['contract']['preprocessing_version']}
                for side in ('left', 'right'):
                    row['prediction'][side] = {'coordinates':[row['joints'][name]['camera_'+side+'_pts2d']
                        for name in doc['contract']['joint_names']], 'valid_mask':[True, True]}
            self.write(f'{index}.json', doc)
        catalog = build_catalog(['0.json', '1.json'], self.base, 3)
        actual = ShardDataset(catalog, self.base).load(0)[1]['train']
        original = load_document(self.documents[0], 3)[1]['train']
        self.assertTrue(torch.equal(actual.target_valid, original.target_valid))
        self.assertTrue(torch.equal(actual.target_mm, original.target_mm))
        self.assertEqual(catalog['contract']['input_source'], 'predicted_2d')

    def test_readiness_declarations_vs_payload_and_no_training_approval(self):
        declaration = inspect_catalog(self.base/'catalog.json')
        self.assertFalse(declaration['full_payload_read'])
        self.assertFalse(declaration['full_training_ready'])
        checked = inspect_catalog(self.base/'catalog.json', True)
        self.assertTrue(checked['full_payload_read'])
        self.assertEqual(checked['verified_records'], {'train':12, 'validation':12})
        self.assertFalse(checked['full_training_ready'])
        (self.base/'0.json').unlink()
        self.assertFalse(inspect_catalog(self.base/'catalog.json')['full_payload_read'])
        with self.assertRaises(DataContractError):
            inspect_catalog(self.base/'catalog.json', True)

    def test_real_string_evidence_does_not_issue_full_readiness(self):
        catalog = copy.deepcopy(self.catalog)
        catalog['contract']['data_kind'] = 'dataset_sample'
        keys = ('joints', 'units', 'coordinates', 'calibration', 'split', 'rgb_pose_alignment')
        catalog['contract']['verified'] = dict.fromkeys(keys, True)
        catalog['contract']['verification_evidence'] = dict.fromkeys(keys, 'declared')
        self.write('catalog.json', signed(catalog))
        report = inspect_catalog(self.base/'catalog.json')
        self.assertFalse(report['full_training_ready'])
        self.assertIn('original_motion_identity_not_independently_verified', report['blockers'])

    def test_headless_cli_build_inspect(self):
        with contextlib.redirect_stdout(io.StringIO()) as output:
            result = main(['build', '--base', str(self.base), '--shard', '0.json', '--shard', '1.json',
                           '--window-size', '3', '--output', str(self.base/'new.json')])
        self.assertEqual(result, 0)
        self.assertTrue(json.loads(output.getvalue())['build_payloads_fully_read'])
        self.assertEqual(read_catalog(self.base/'new.json'), self.catalog)

    def artifact(self, architecture='single_frame'):
        config = {'model':architecture, 'window_size':3, 'hidden':8, 'dropout':.1, 'target_scale_mm':100.}
        model = make_model(config, 28, 2).eval()
        source = {'revision':'local_test', 'code_sha256':{'models.py':'a'*64}}
        path = self.base/(architecture+'.pt')
        rng = torch.get_rng_state().clone()
        before = {k:v.clone() for k,v in model.state_dict().items()}
        export_weights(model, config, self.catalog['contract'], self.catalog['content_sha256'], source, 7, path,
                       checkpoint_sha256='c'*64)
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        self.assertFalse(model.training)
        self.assertTrue(all(torch.equal(v, model.state_dict()[k]) for k,v in before.items()))
        return model, source, path

    def test_weights_prediction_roundtrip_and_rng_preserved(self):
        for architecture in ('single_frame', 'temporal'):
            model, source, path = self.artifact(architecture)
            rng = torch.get_rng_state().clone()
            loaded, meta = load_weights(path, expected_contract=self.catalog['contract'],
                    expected_catalog_sha256=self.catalog['content_sha256'], expected_source=source,
                    expected_checkpoint_sha256='c'*64,
                    expected_model_config={'model':architecture,'window_size':3,'hidden':8,'dropout':.1,'target_scale_mm':100.})
            self.assertTrue(torch.equal(rng, torch.get_rng_state()))
            x, valid = torch.rand(2,3,28), torch.ones(2,3,dtype=torch.bool)
            self.assertTrue(torch.equal(model(x, valid), loaded(x, valid)))
            self.assertEqual(meta['training_step'], 7)
            self.assertFalse(loaded.training)

    def test_weights_identity_version_shape(self):
        _, source, path = self.artifact()
        expected = dict(expected_contract=self.catalog['contract'],
                        expected_catalog_sha256=self.catalog['content_sha256'], expected_source=source,
                        expected_checkpoint_sha256='c'*64,
                        expected_model_config={'model':'single_frame','window_size':3,'hidden':8,'dropout':.1,'target_scale_mm':100.})
        for key, value in (('expected_source', {**source, 'revision':'other'}),
                           ('expected_catalog_sha256', 'b'*64), ('expected_contract', {}),
                           ('expected_checkpoint_sha256', 'd'*64), ('expected_model_config', {})):
            with self.assertRaisesRegex(DataContractError, 'weights_expected_identity'):
                load_weights(path, **{**expected, key:value})
        original = torch.load(path, weights_only=True)
        changed = copy.deepcopy(original)
        changed['format_version'] = 'other'
        torch.save(changed, path)
        with self.assertRaisesRegex(DataContractError, 'weights_format'):
            load_weights(path, **expected)
        changed = copy.deepcopy(original)
        changed['model']['net.0.weight'] = changed['model']['net.0.weight'][:1]
        changed['state_sha256'] = _state_hash(changed['model'])
        torch.save(changed, path)
        with self.assertRaisesRegex(ValueError, 'weights_model_shape'):
            load_weights(path, **expected)

    def test_import_without_ui_trainer_cli(self):
        code = "import sys; import rewind_m1.artifacts; assert not any(k in sys.modules for k in ['rewind_m1.training','rewind_m1.__main__','rewind_m1.frontend_train','rewind_prepare.rgb_review']); print('headless_ok')"
        result = subprocess.run([sys.executable, '-X', 'utf8', '-c', code], cwd=ROOT,
                                capture_output=True, text=True, check=True)
        self.assertEqual(result.stdout.strip(), 'headless_ok')

    def test_export_mode_state_and_serialized_budget(self):
        from unittest.mock import patch
        config = {'model':'single_frame','window_size':3,'hidden':8,'dropout':.1,'target_scale_mm':100.}
        model = make_model(config, 28, 2).train()
        source = {'revision':'local_test','code_sha256':{'models.py':'a'*64}}
        rng = torch.get_rng_state().clone()
        with patch('rewind_m1.artifacts.MAX_BYTES', 1):
            with self.assertRaisesRegex(DataContractError, 'weights_serialized_byte_limit'):
                export_weights(model, config, self.catalog['contract'], self.catalog['content_sha256'],
                               source, 0, self.base/'over.pt', checkpoint_sha256='c'*64)
        self.assertFalse((self.base/'over.pt').exists())
        self.assertTrue(model.training)
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))


if __name__ == '__main__':
    unittest.main()
