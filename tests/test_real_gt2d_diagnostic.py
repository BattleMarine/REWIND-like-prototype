"""보존된 실제2sample의 진단 입구/일반 학습 차단 회귀. 학습 update는 수행하지 않는다."""
import copy
import hashlib
import json
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
PREPARED=ROOT/'outputs/real_stereo_sample_20261102/root/prepared'
try:
    import torch
    from PIL import Image
except (ImportError,OSError):
    AVAILABLE=False
else:
    AVAILABLE=(PREPARED/'pose/candidate.json').is_file()
if AVAILABLE:
    import rewind_m1.real_gt2d_diagnostic as diagnostic
    from rewind_archive.contracts import canonical,digest
    from rewind_m1.data import InputContract,load_document,DataContractError
    from rewind_m1.local_diagnostic import _preserve_state
    from rewind_m1.training import read_checkpoint,save_checkpoint,seed_run,Trainer


@unittest.skipUnless(AVAILABLE,'검토된 local Serena 실제 sample과 PyTorch가 필요')
class RealGTDiagnosticTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.candidate=json.loads((PREPARED/'pose/candidate.json').read_bytes())
        cls.pins={'candidate_sha256':hashlib.sha256((PREPARED/'pose/candidate.json').read_bytes()).hexdigest(),
                  'rgb_manifest_sha256':hashlib.sha256((PREPARED/'rgb/manifest.json').read_bytes()).hexdigest()}
        with _preserve_state():cls.data=diagnostic.load_diagnostic(PREPARED,diagnostic_only=True,**cls.pins)

    def mutate(self,change):
        value=copy.deepcopy(self.candidate);change(value)
        value['content_sha256']=digest({k:v for k,v in value.items() if k!='content_sha256'});raw=canonical(value)
        original=diagnostic.read
        def reading(path,*args):return raw if Path(path)==PREPARED/'pose/candidate.json' else original(path,*args)
        with patch.object(diagnostic,'read',reading):
            return diagnostic.load_diagnostic(PREPARED,diagnostic_only=True,**{**self.pins,'candidate_sha256':hashlib.sha256(raw).hexdigest()})

    def test_explicit_approval_and_fixed_pin_required(self):
        for value in (False,None,1):
            with self.assertRaisesRegex(ValueError,'diagnostic_only'):diagnostic.load_diagnostic(PREPARED,diagnostic_only=value,**self.pins)
        with self.assertRaisesRegex(ValueError,'pin_mismatch'):diagnostic.load_diagnostic(PREPARED,diagnostic_only=True,**{**self.pins,'candidate_sha256':'0'*64})

    def test_shape_units_masks_and_outside_points_preserved(self):
        self.assertEqual(tuple(self.data.tensors.features.shape),(2,1,868))
        self.assertEqual(tuple(self.data.tensors.target_mm.shape),(2,1,62,3))
        self.assertEqual(self.data.identity['data_kind'],'dataset_sample')
        self.assertEqual(self.data.identity['target_unit'],'mm')
        self.assertEqual(self.data.identity['outside_image_counts'],[{'left':22,'right':20},{'left':0,'right':0}])
        self.assertEqual(int(self.data.tensors.target_valid.sum()),124)
        self.assertTrue((self.data.tensors.features[0,0].reshape(62,14)[:,[0,1,7,8]].abs()>1).any())
        self.assertFalse(self.data.identity['train_ready'])

    def test_duplicate_and_nontrain_records_rejected(self):
        with self.assertRaisesRegex(ValueError,'unique_frames'):self.mutate(lambda d:d['records'].__setitem__(1,copy.deepcopy(d['records'][0])))
        with self.assertRaisesRegex(ValueError,'train_only'):self.mutate(lambda d:d['records'][1].update(split='validation'))

    def test_feature_cache_leak_or_raw_target_change_rejected(self):
        with self.assertRaisesRegex(ValueError,'feature_cache'):self.mutate(lambda d:d['records'][0]['observation_features'].__setitem__(0,123.))
        with self.assertRaisesRegex(ValueError,'raw_field_or_unit'):self.mutate(lambda d:d['records'][0]['fields']['device_pts3d']['coordinates'][0].__setitem__(0,99999.))

    def test_joint_order_and_raw_provenance_rejected(self):
        with self.assertRaises(ValueError):self.mutate(lambda d:d['contract']['joint_names'].reverse())
        with self.assertRaisesRegex(ValueError,'pose_source'):self.mutate(lambda d:d['records'][0]['source'].update(sha256='0'*64))

    def test_general_loader_and_contract_still_reject_candidate(self):
        with self.assertRaises(DataContractError):load_document(self.candidate,1)
        with self.assertRaises(DataContractError):InputContract.from_dict(self.candidate['contract'])
        with self.assertRaises(DataContractError):Trainer(diagnostic.CONFIG,self.candidate)
        self.assertNotIn('verified',self.candidate['contract'])
        self.assertFalse(self.candidate['training_eligibility']['train_ready'])

    def test_diagnostic_checkpoint_role_and_identity_rejected_by_normal_reader(self):
        parent=ROOT/'outputs/real_gt2d_diagnostic_20261102/test_tmp';parent.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=parent) as temporary,_preserve_state():
            trainer=diagnostic.DiagnosticTrainer(self.data,diagnostic.CONFIG);checkpoint=trainer.checkpoint()
            path=Path(temporary)/'diagnostic.pt';save_checkpoint(checkpoint,path)
            with self.assertRaisesRegex(ValueError,'checkpoint'):read_checkpoint(path)
            wrong=copy.deepcopy(checkpoint);wrong['role']='normal_training'
            with self.assertRaisesRegex(ValueError,'checkpoint_role'):trainer.resume(wrong)
            wrong=copy.deepcopy(checkpoint);wrong['data_identity']['features_sha256']='0'*64
            with self.assertRaisesRegex(ValueError,'resume_identity'):trainer.resume(wrong)
            trainer.resume(torch.load(path,map_location='cpu',weights_only=True))
            self.assertTrue(diagnostic.equal_state(checkpoint,trainer.checkpoint()))

    def test_cpu_single_frame_independence_and_config_limits(self):
        with _preserve_state():
            trainer=diagnostic.DiagnosticTrainer(self.data,diagnostic.CONFIG)
            checks=diagnostic.independence(trainer)
            self.assertTrue(all(v for k,v in checks.items() if k.endswith('_equal')))
            for options in ({'window_size':2},{'model':'temporal'},{'amp':True},{'hidden':65},{'batch_size':3}):
                with self.assertRaisesRegex(ValueError,'cpu_limits'):diagnostic.DiagnosticTrainer(self.data,{**diagnostic.CONFIG,**options})

    def test_global_rng_threads_and_determinism_are_restored(self):
        before=(random.getstate(),torch.get_rng_state().clone(),torch.get_num_threads(),torch.are_deterministic_algorithms_enabled())
        with _preserve_state():seed_run(999);torch.rand(3);random.random()
        self.assertEqual(random.getstate(),before[0]);self.assertTrue(torch.equal(torch.get_rng_state(),before[1]))
        self.assertEqual(torch.get_num_threads(),before[2]);self.assertEqual(torch.are_deterministic_algorithms_enabled(),before[3])

    def test_source_identity_includes_all_used_packages(self):
        source=diagnostic.diagnostic_source_identity()['code_sha256']
        for key in ('rewind_prepare/converter.py','rewind_archive/matching.py','rewind_m1/models.py','rewind_data/windows.py'):
            self.assertEqual(source[key],hashlib.sha256((ROOT/key).read_bytes()).hexdigest())


if __name__=='__main__':unittest.main()
