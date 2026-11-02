"""실자료 승인 없이 development_fixture로 release adapter와 공통 RGB 코어 검증."""
import copy
import importlib.util
import json
import subprocess
import sys
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from rewind_archive.contracts import canonical,digest
from rewind_bundle_data.integrity import sha
from rewind_m1.rgb_frontend_artifacts import resize_images,scale_xy
from rewind_rgb_release.contracts import (admit,audit_candidate,CATEGORIES,facts,SCOPE,validate_policy,validate_review,proposal_from_data)
from rewind_rgb_release.training import ReleasedRGBTrainer
from rewind_rgb_release.weights import checkpoint,export_checkpoint,ReleasedInference
from rewind_rgb_release.__main__ import main
from rewind_m1.training import save_checkpoint
from test_shard_training import equal

ROOT = Path(__file__).resolve().parents[1]


@unittest.skipUnless(importlib.util.find_spec('indexed_gzip'),'기존 index test환경 필요')
class RGBReleaseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from test_bundle_data import BundleDataTests
        BundleDataTests.setUpClass();cls.addClassCleanup(BundleDataTests.doClassCleanups)
        cls.bundle = BundleDataTests.fixture_bundle
        cls.pin = json.loads((cls.bundle/'COMPLETE.json').read_bytes())['content_sha256']
        previous = torch.get_num_threads();torch.set_num_threads(1);cls.addClassCleanup(torch.set_num_threads,previous)

    def setUp(self):
        parent = ROOT/'outputs/test_tmp';parent.mkdir(parents=True,exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=parent);self.addCleanup(self.temp.cleanup);self.base = Path(self.temp.name)
        self.config = {'seed':43,'batch_size':1,'learning_rate':.005,'scheduler_step':2,'scheduler_gamma':.8,
            'amp':False,'training_image_size':[8,8],'frontend_version':'development_fixture_release_test',
            'max_native_tensor_bytes':32*1024**2,'max_heatmap_elements':32*1024**2}
        self.request = {'mode':'development_fixture','bundle':str(self.bundle),'completion_sha256':self.pin}

    def trainer(self):return ReleasedRGBTrainer(self.config,self.request)

    def test_full_resume_all_state_validation_and_export(self):
        full = self.trainer()
        for _ in range(4):full.train_step()
        expected = copy.deepcopy(full.checkpoint());metrics = full.evaluate()
        self.assertEqual(metrics['records'],1);self.assertEqual(metrics['evaluation_kind'],'development_fixture_rgb2d_t1_only')
        partial = self.trainer()
        for _ in range(2):partial.train_step()
        state = copy.deepcopy(partial.checkpoint())
        self.assertEqual(state['training_scope'],'development_fixture_rgb2d_t1_only')
        self.assertEqual(state['catalog']['contract']['data_kind'],'dataset_sample')
        resumed = self.trainer();resumed.resume(state)
        for _ in range(2):resumed.train_step()
        self.assertTrue(equal(expected,resumed.checkpoint()))
        path = self.base/'last.pt';save_checkpoint(resumed.checkpoint(),path);stored,pin = checkpoint(path)
        weights = self.base/'weights.pt';export_checkpoint(stored,pin,weights)
        inference = ReleasedInference(weights,expected_file_sha256=sha(weights.read_bytes()))
        native = resumed.dataset.load(0,'train').load_batch([0])['images']
        with torch.no_grad():wanted = scale_xy(resumed.model.eval()(resize_images(native,[8,8]))[1],[8,8],[1024,1024])
        self.assertTrue(torch.equal(inference.predict(native)['coordinates'],wanted))
        self.assertFalse(inference.predict(native)['training_authorization'])
        self.assertEqual(inference.metadata['release_binding'],resumed.binding)
        previous_dtype = torch.get_default_dtype()
        try:
            torch.set_default_dtype(torch.float64)
            with torch.device('meta'):
                offline = ReleasedInference(weights,expected_file_sha256=sha(weights.read_bytes()))
            self.assertTrue(all(t.device.type == 'cpu' and t.dtype == torch.float32 for t in offline.model.parameters()))
        finally:torch.set_default_dtype(previous_dtype)
        result = subprocess.run([sys.executable,'-c',"import sys;import rewind_rgb_release.weights;assert 'rewind_m1.training' not in sys.modules;assert 'rewind_rgb_release.training' not in sys.modules;print('offline_ok')"],
            cwd=ROOT,capture_output=True,text=True,timeout=60)
        self.assertEqual(result.returncode,0,result.stderr)

    def test_release_fixture_scope_and_resume_identity_rejections(self):
        request = {**self.request,'mode':'reviewed_real','release':'not-issued','review':'not-issued','policy':'not-issued','expected_policy_sha256':'0'*64}
        with self.assertRaisesRegex(ValueError,'synthetic_not_real'):admit(request,self.config)
        trainer = self.trainer();trainer.train_step();state = copy.deepcopy(trainer.checkpoint())
        for mutate in (lambda d:d.update(training_scope='synthetic_fixture_only'),
                       lambda d:d['catalog']['binding'].update(policy_sha256='0'*64),
                       lambda d:d['source']['code_sha256'].update(fake='0'*64),
                       lambda d:d['environment'].update(torch='changed'),
                       lambda d:d['config'].update(seed=9)):
            bad = copy.deepcopy(state);mutate(bad)
            with self.assertRaises(Exception):self.trainer().resume(bad)
        with self.assertRaisesRegex(ValueError,'no_test'):trainer.evaluate('test',True)

    def test_reader_batch_parity_and_failed_update_cursor(self):
        trainer = self.trainer();native = trainer.dataset.load(0,'train').load_batch([0])
        batch = trainer._batch(0,'train',[0])
        self.assertTrue(torch.equal(batch['images'],resize_images(native['images'],[8,8])))
        self.assertTrue(torch.equal(batch['native_gt_xy'],native['gt_xy']))
        self.assertTrue(torch.equal(batch['valid'],native['annotation_valid']))
        cursor = trainer.sampler.state_dict()
        with patch.object(trainer,'_batch',side_effect=ValueError('bounded_failure')):
            with self.assertRaisesRegex(ValueError,'bounded_failure'):trainer.train_step()
        self.assertEqual(trainer.sampler.state_dict(),cursor)
        with self.assertRaises(Exception):trainer.checkpoint()

    def review_fixture(self):
        _,data = facts(self.bundle,self.pin);coverage = digest(data['key_coverage'])
        artifact = self.base/'fixture_notes.txt';artifact.write_text('development fixture only; no actual data review',encoding='utf-8')
        document = {'schema_version':'rgb2d_human_review_v1','scope':SCOPE,'completion_sha256':self.pin,'key_coverage_sha256':coverage,
            'evidence':{c:{'artifact':'fixture_notes.txt','artifact_sha256':sha(artifact.read_bytes()),
                'reviewer':'development_fixture_not_real_approval','date':'2026-11-02',
                'conclusion':'adopted_assumptions_for_rgb2d_t1' if c == 'units_geometry' else 'reviewed_for_rgb2d_t1',
                'limitations':['fixture only'],'key_coverage_sha256':coverage} for c in CATEGORIES},
            'reviewed_motion_map':{s:{'original_motion_id':'fixture:'+s,'candidate_motion_id':r['motion_id'],'evidence_note':'fixture only'}
                for s,r in data['original_motion_map']['sequences'].items()},
            'geometry_statement':{'verified_facts':['fixture hashes checked'],'adopted_assumptions':['fixture camera convention']}}
        path = self.base/'fixture_review.json';path.write_bytes(canonical(document))
        return data,document,path

    def test_review_evidence_coverage_motion_leak_and_hash(self):
        data,doc,path = self.review_fixture();self.assertTrue(validate_review(path,data))
        seqs = list(doc['reviewed_motion_map'])
        doc['reviewed_motion_map'][seqs[1]]['original_motion_id'] = doc['reviewed_motion_map'][seqs[0]]['original_motion_id']
        path.write_bytes(canonical(doc))
        with self.assertRaisesRegex(ValueError,'motion_split_leak'):validate_review(path,data)
        data,doc,path = self.review_fixture();doc['key_coverage_sha256'] = '0'*64;path.write_bytes(canonical(doc))
        with self.assertRaisesRegex(ValueError,'coverage'):validate_review(path,data)
        data,doc,path = self.review_fixture();(self.base/'fixture_notes.txt').write_text('changed')
        with self.assertRaisesRegex(ValueError,'evidence_hash'):validate_review(path,data)

    def test_external_policy_pin_and_exact_allowlist_unit_contract(self):
        # 권한 없는 가상 payload의 predicate 검사다. 실제자료/실제release는 만들지 않는다.
        proposal = {'content_sha256':'1'*64,'code_sha256':{'fixture':'2'*64},
                    'review':{'review_file_sha256':'3'*64,'reviewers':['fixture_no_actual_approval']}}
        allow = {'release_file_sha256':'4'*64,'release_content_sha256':'1'*64,'review_file_sha256':'3'*64,
                 'code_sha256':digest(proposal['code_sha256']),'reviewers':['fixture_no_actual_approval']}
        policy = {'schema_version':'local_rgb2d_review_policy_v1','trust_model':'external_operator_pinned_allowlist','scope':SCOPE,'allowed_releases':[allow]}
        path = self.base/'predicate_fixture_policy.json';path.write_bytes(canonical(policy));pin = sha(path.read_bytes())
        self.assertEqual(validate_policy(path,pin,proposal,'4'*64),pin)
        with self.assertRaisesRegex(ValueError,'policy_pin'):validate_policy(path,'0'*64,proposal,'4'*64)
        with self.assertRaisesRegex(ValueError,'allowlist'):validate_policy(path,pin,proposal,'5'*64)

    def test_reviewed_real_admission_join_with_explicit_mock_facts(self):
        # 실제 fixture archive를 relabel하지 않는다. 승인 조립만 가상 facts로 검사한다.
        data,review,path = self.review_fixture();data = copy.deepcopy(data);data['source_scope'] = 'official_server'
        releases = self.base/'mock_release';releases.mkdir();trust = self.base/'external_mock_trust';trust.mkdir()
        proposal = proposal_from_data(data,self.config,path)
        release = releases/'release.json';release.write_bytes(canonical(proposal));release_sha = sha(release.read_bytes())
        allow = {'release_file_sha256':release_sha,'release_content_sha256':proposal['content_sha256'],
            'review_file_sha256':proposal['review']['review_file_sha256'],'code_sha256':digest(proposal['code_sha256']),
            'reviewers':proposal['review']['reviewers']}
        policy = {'schema_version':'local_rgb2d_review_policy_v1','trust_model':'external_operator_pinned_allowlist','scope':SCOPE,'allowed_releases':[allow]}
        policy_path = trust/'mock_predicate_policy.json';policy_path.write_bytes(canonical(policy))
        request = {**self.request,'mode':'reviewed_real','release':str(release),'review':str(path),'policy':str(policy_path),
                   'expected_policy_sha256':sha(policy_path.read_bytes())}
        with patch('rewind_rgb_release.contracts.facts',return_value=({'mock_readers':True},data)):
            _,binding = admit(request,self.config)
            self.assertEqual(binding['mode'],'reviewed_real')
            self.assertEqual(binding['data']['source_scope'],'official_server')
            with patch('rewind_rgb_release.contracts.code_identity',return_value={'changed':'0'*64}):
                with self.assertRaisesRegex(ValueError,'payload_bundle_code_review'):admit(request,self.config)
            raw = release.read_bytes();release.write_bytes(raw+b' ')
            with self.assertRaisesRegex(ValueError,'allowlist'):admit(request,self.config)
            release.write_bytes(raw)
            original = path.read_bytes();changed = json.loads(original)
            changed['geometry_statement']['adopted_assumptions'].append('changed')
            path.write_bytes(canonical(changed))
            with self.assertRaisesRegex(ValueError,'payload_bundle_code_review'):admit(request,self.config)
            path.write_bytes(original)
            policy_path.write_bytes(b'{}')
            with self.assertRaisesRegex(ValueError,'policy_pin'):admit(request,self.config)

    def test_cli_fixture_train_resume_validate_export_and_durable_failure(self):
        config = self.base/'config.json';config.write_bytes(canonical(self.config))
        def run(command,out,*extra):
            return main([command,'--bundle',str(self.bundle),'--completion-sha256',self.pin,'--config',str(config),
                         '--output',str(self.base/out),*extra])
        first = run('development-fixture-train','first','--steps','1')
        self.assertEqual(first['training_scope'],'development_fixture_rgb2d_t1_only')
        run('development-fixture-train','resumed','--steps','3','--resume',str(self.base/'first/last.pt'))
        result = run('development-fixture-validate','validation','--checkpoint',str(self.base/'resumed/last.pt'))
        self.assertEqual(result['metrics']['records'],1)
        main(['export','--checkpoint',str(self.base/'resumed/last.pt'),'--output',str(self.base/'cli_weights.pt')])
        from rewind_rgb_release.training import ReleasedRGBTrainer
        original = ReleasedRGBTrainer.train_step
        def failed(instance):
            if instance.step == 1:raise ValueError('fixture_stop_after_durable')
            return original(instance)
        with patch.object(ReleasedRGBTrainer,'train_step',failed):
            with self.assertRaisesRegex(ValueError,'fixture_stop_after_durable'):
                run('development-fixture-train','failed','--steps','3')
        durable,_ = checkpoint(self.base/'failed/last.pt');self.assertEqual(durable['step'],1)
        self.assertEqual(json.loads((self.base/'failed/report.json').read_bytes())['status'],'failed')
