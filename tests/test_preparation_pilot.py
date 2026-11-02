"""자동 준비 통합 preset과 REPORT 없는 실패/중단 회수. 실제 인증·외부 요청 없음."""
import contextlib
import gzip
import hashlib
import io
import json
from pathlib import Path
import tarfile
import unittest
from unittest.mock import patch
import zipfile

from rewind_archive_index.preparation_pilot import run_preparation_integration,recover_preparation_integration,INTEGRATION_LIMITS
from rewind_archive_index.source import IndexFailure
from test_persistent_index_extension import SYNTHETIC_PROVENANCE
import test_persistent_index_extension as fixtures


@unittest.skipUnless(fixtures.importlib.util.find_spec('indexed_gzip') and fixtures.importlib.util.find_spec('psutil'),'install requirements-index.txt')
class PreparationIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.case=fixtures.ExtensionTests();self.case.setUpClass();self.case.setUp();self.addCleanup(self.case.doCleanups)
        self.folder,self.index,self.anchor=self.case.baseline(1048576)
        with tarfile.open(fileobj=io.BytesIO(gzip.decompress(self.case.data))) as archive:
            raw=archive.extractfile('sequence/json/frame_24.json').read()
        self.targets=self.anchor['selected']+[{'name':'sequence/json/frame_24.json','sha256':hashlib.sha256(raw).hexdigest()}]
        self.kwargs=dict(run_id='integration',workspace=self.folder/'auto-work',input_snapshot_directory=self.folder/'B-input',
            output_snapshot_directory=self.folder/'auto-snapshot',manifest=self.case.manifest,anchor=self.anchor,targets=self.targets,
            provenance=SYNTHETIC_PROVENANCE,session_factory=lambda:self.case.session)

    def recover(self):return recover_preparation_integration(workspace=self.kwargs['workspace'],run_id='integration',provenance=SYNTHETIC_PROVENANCE)

    def execute(self):
        with contextlib.redirect_stdout(io.StringIO()):return run_preparation_integration(**self.kwargs)

    def test_complete_preset_shared_budget_cold_restore_reuse_and_recovery(self):
        before=self.index._snapshot();result=self.execute();self.assertEqual(result['status'],'passed',result.get('error'))
        self.assertEqual(self.index._snapshot(),before)
        self.assertEqual(result['reuse']['source_read_bytes'],0)
        self.assertLessEqual(result['transport']['phase_response_body_bytes'],8*1024**2)
        self.assertLessEqual(result['transport']['phase_http_requests'],120)
        self.assertLessEqual(result['resources']['source_read_bytes'],32*1024**2)
        self.assertLessEqual(result['resources']['returned_decoded_bytes'],32*1024**2)
        self.assertEqual(result['limits'],INTEGRATION_LIMITS.__dict__)
        with patch('sqlite3.connect',side_effect=AssertionError('no DB recovery')),patch('requests.sessions.Session.request',side_effect=AssertionError('no network')):
            bundle=self.recover();again=self.recover()
        self.assertTrue(again['reused'])
        with zipfile.ZipFile(bundle['path']) as archive:
            report=json.loads(archive.read('report.json'));self.assertEqual(report['status'],'passed')
            first=json.loads(archive.read('first-extraction.json'))['persistent_index']
            restored=json.loads(archive.read('restored-extraction.json'))['persistent_index']
            self.assertGreater(first['after']['index_coverage'],first['before']['index_coverage'])
            self.assertEqual(restored['restore']['receipt'],first['head']['receipts'][-1])
            self.assertEqual(restored['scans'],[])
            self.assertEqual(sum(n.startswith('selected/') for n in archive.namelist()),3)
            self.assertFalse(any(n.endswith('.sqlite') for n in archive.namelist()))
        with self.assertRaisesRegex(IndexFailure,'prepare_pilot_new_snapshot_directory_required'):self.execute()

    def test_factory_interrupt_and_normal_error_are_recoverable_without_report(self):
        for kind in (KeyboardInterrupt,RuntimeError):
            with self.subTest(kind=kind.__name__):
                self.kwargs['workspace']=self.folder/kind.__name__
                original=kind('secret-token URL')
                def factory():raise original
                self.kwargs['session_factory']=factory
                if kind is KeyboardInterrupt:
                    with self.assertRaises(KeyboardInterrupt) as caught:self.execute()
                    self.assertIs(caught.exception,original)
                else:self.assertEqual(self.execute()['status'],'failed')
                recovered=self.recover()
                with zipfile.ZipFile(recovered['path']) as archive:
                    report=json.loads(archive.read('report.json'))
                    self.assertEqual(report['status'],'interrupted' if kind is KeyboardInterrupt else 'failed')
                    self.assertEqual(report['failure_stage']['name'],'session_factory')
                    self.assertEqual(report['transport'],{'status':'unknown'})
                    self.assertNotIn(b'secret-token',archive.read('report.json'))

    def test_mid_request_interruption_recovers_verified_identity_request_marker(self):
        interrupted=KeyboardInterrupt()
        with patch('rewind_archive_index.preparation.PreparationIndex.scan',side_effect=interrupted):
            with self.assertRaises(KeyboardInterrupt) as caught:self.execute()
        self.assertIs(caught.exception,interrupted)
        bundle=self.recover()
        with zipfile.ZipFile(bundle['path']) as archive:
            self.assertIn('first-request.json',archive.namelist())
            request=json.loads(archive.read('first-request.json'));self.assertEqual(request['status'],'interrupted')
        Path(bundle['path']).unlink()
        path=next((self.kwargs['workspace']/'live-first').rglob('request-*.json'))
        doc=json.loads(path.read_bytes());doc['provenance']={**SYNTHETIC_PROVENANCE,'manifest_sha256':'f'*64};path.write_text(json.dumps(doc))
        with self.assertRaisesRegex(IndexFailure,'recovery_provenance_mismatch'):self.recover()

    def test_missing_or_broken_marker_produces_diagnostic_not_success(self):
        self.kwargs['workspace'].mkdir()
        marker=self.kwargs['workspace']/'integration-auto.json'
        for data in (None,b'{broken'):
            if data is not None:marker.write_bytes(data)
            result=self.recover()
            with zipfile.ZipFile(result['path']) as archive:self.assertEqual(json.loads(archive.read('report.json'))['status'],'offline_diagnostic')
            Path(result['path']).unlink()

    def test_partial_request_recovers_manifest_and_keeps_snapshot(self):
        import rewind_archive_index.preparation_pilot as module
        original=module.prepare
        def one_batch(*args,**kwargs):
            kwargs['persistent_index']={**kwargs['persistent_index'],'max_batches':1}
            return original(*args,**kwargs)
        with patch.object(module,'prepare',one_batch):result=self.execute()
        self.assertEqual(result['status'],'partial')
        self.assertTrue(list(self.kwargs['output_snapshot_directory'].glob('g*.zip')))
        bundle=self.recover()
        with zipfile.ZipFile(bundle['path']) as archive:
            self.assertIn('first-extraction.json',archive.namelist())
            self.assertEqual(json.loads(archive.read('report.json'))['status'],'partial')


if __name__=='__main__':unittest.main()
