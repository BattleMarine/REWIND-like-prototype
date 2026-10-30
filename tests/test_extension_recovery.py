"""인증·원본 읽기 없는 중단 기록과 디스크 회수 계약."""
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile

from rewind_archive_index import ArchiveIndex, IndexFailure
from rewind_archive_index.drive import make_drive_source
from rewind_archive_index.extension_pilot import run_extension
from rewind_archive_index.extension_records import atomic_record, recover_report, RECORD_LIMIT, FILE_LIMIT
from rewind_archive_index.snapshots import export_snapshot
from test_persistent_index_extension import SYNTHETIC_PROVENANCE


@unittest.skipUnless(importlib.util.find_spec('indexed_gzip'), 'install requirements-index.txt')
class InterruptedRecords(unittest.TestCase):
    def setUp(self):
        parent=Path(__file__).resolve().parents[1]/'outputs/extension_record_tests';parent.mkdir(exist_ok=True)
        self.tmp=tempfile.TemporaryDirectory(dir=parent);self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name)
        self.manifest={'schema_version':'archive_manifest_v1','archive_id':'diagnostic','dataset_version':'synthetic_v1','kind':'pose',
            'source_scope':'user_drive','total_parts':1,'parts':[{'index':0,'name':'pose.tar.gz000','size_bytes':12345,'version':'9',
            'source':{'type':'drive','file_id':'fake_no_requests'},'identity':{'version':'9','modified_time':'2026-10-30T00:00:00.000Z'}}]}
        index=ArchiveIndex(self.root/'B.sqlite');index.create(make_drive_source(self.manifest))
        snapshot=export_snapshot(index,self.root/'B-input')
        self.anchor={'schema':'persistent_drive_pilot_anchor_v1','runtime_id':'other-runtime','receipt':snapshot['receipt'],
            'selected':[{'name':'frame.json','sha256':'0'*64}]}

    def kwargs(self,run_id,factory):
        return dict(run_id=run_id,workspace=self.root/run_id,input_snapshot_directory=self.root/'B-input',
            output_snapshot_directory=self.root/(run_id+'-output'),manifest=self.manifest,anchor=self.anchor,
            session_factory=factory,provenance=SYNTHETIC_PROVENANCE)

    def test_six_injected_paths_record_initial_pins_and_b_summary(self):
        cases=[('factory',KeyboardInterrupt),('factory',SystemExit),('factory',GeneratorExit),
               ('factory',RuntimeError),('runtime',KeyboardInterrupt),('index',KeyboardInterrupt)]
        for number,(location,kind) in enumerate(cases):
            with self.subTest(location=location,kind=kind.__name__):
                run_id='case'+str(number);exc=kind('SECRET https://invalid/?token=hidden')
                class Session:
                    def close(self):pass
                def factory():
                    saved=json.loads((self.root/run_id/(run_id+'-C.json')).read_bytes())
                    self.assertEqual(saved['execution_metadata'],SYNTHETIC_PROVENANCE)
                    self.assertIn('restored_b_checkpoint',saved)
                    self.assertEqual(saved['stage']['name'],'session_factory')
                    if location=='factory':raise exc
                    return Session()
                kwargs=self.kwargs(run_id,factory)
                target='DriveRuntime' if location=='runtime' else 'ArchiveIndex'
                context=patch('rewind_archive_index.extension_pilot.'+target) if location!='factory' else patch.dict({}, {})
                with context as mocked:
                    if location!='factory':
                        mocked.side_effect=exc
                        if location=='index':mocked._engine_version.return_value='1.10.3'
                    if kind is RuntimeError:
                        returned=run_extension(**kwargs);self.assertEqual(returned['status'],'failed')
                    else:
                        with self.assertRaises(kind) as caught:run_extension(**kwargs)
                        self.assertIs(caught.exception,exc)
                raw=(self.root/run_id/(run_id+'-C.json')).read_bytes();saved=json.loads(raw)
                self.assertNotIn(b'SECRET',raw);self.assertNotIn(b'http',raw)
                self.assertEqual(saved['status'],'failed' if kind is RuntimeError else 'interrupted')
                self.assertEqual(saved['transport_recording'],'not_recorded')
                if kind is not RuntimeError:self.assertEqual(saved['interruption']['type'],kind.__name__)
                self.assertEqual([r['name'] for r in saved['stages'][:6]],['initial','initial','restore_b','restore_b','b_summary','b_summary'])
                self.assertEqual([r['edge'] for r in saved['stages'][:6]],['pre','post']*3)
                with self.assertRaisesRegex(IndexFailure,'pilot_run_already_consumed'):run_extension(**kwargs)
                bundle=recover_report(workspace=self.root/run_id,run_id=run_id,provenance=SYNTHETIC_PROVENANCE)
                with zipfile.ZipFile(bundle['path']) as archive:self.assertEqual(json.loads(archive.read('report.json'))['status'],saved['status'])

    def test_cleanup_interruption_does_not_replace_original(self):
        original=KeyboardInterrupt();secondary=SystemExit('secret')
        class Session:
            def close(self):raise secondary
        with patch('rewind_archive_index.extension_pilot.DriveRuntime',side_effect=original):
            with self.assertRaises(KeyboardInterrupt) as caught:run_extension(**self.kwargs('cleanup',Session))
        self.assertIs(caught.exception,original)
        saved=json.loads((self.root/'cleanup/cleanup-C.json').read_bytes())
        self.assertEqual(saved['status'],'interrupted');self.assertEqual(saved['interruption']['type'],'KeyboardInterrupt')
        self.assertEqual(saved['cleanup_exception']['type'],'SystemExit')

    def test_initial_record_failure_never_starts_auth(self):
        calls=[]
        with patch('rewind_archive_index.extension_pilot.atomic_record',side_effect=OSError()):
            with self.assertRaises(OSError):run_extension(**self.kwargs('initial',lambda:calls.append(1)))
        self.assertEqual(calls,[])


class OfflineRecovery(unittest.TestCase):
    def setUp(self):
        parent=Path(__file__).resolve().parents[1]/'outputs/extension_recovery_tests';parent.mkdir(exist_ok=True)
        self.tmp=tempfile.TemporaryDirectory(dir=parent);self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name)
        self.marker=self.root/'recover-C.json'

    def report(self,status='started'):
        return {'schema':'persistent_drive_extension_report_v1','run_id':'recover','phase':'C','status':status,
                'execution_metadata':SYNTHETIC_PROVENANCE,'stages':[],'scans':[]}

    def recover(self,**kwargs):return recover_report(workspace=self.root,run_id='recover',provenance=SYNTHETIC_PROVENANCE,**kwargs)

    def test_all_statuses_without_report_global_and_no_db_open(self):
        for status in ('started','failed','partial','interrupted','passed'):
            with self.subTest(status=status):
                atomic_record(self.marker,self.report(status))
                db=self.root/'recover-C.sqlite';db.write_bytes(b'never open this')
                before=hashlib.sha256(self.marker.read_bytes()).hexdigest()
                with patch('sqlite3.connect',side_effect=AssertionError('DB forbidden')),patch('requests.sessions.Session.request',side_effect=AssertionError('network forbidden')):
                    result=self.recover()
                with zipfile.ZipFile(result['path']) as archive:
                    self.assertEqual(set(archive.namelist()),{'report.json','recovery.json'})
                    self.assertEqual(json.loads(archive.read('report.json'))['status'],status)
                reused=self.recover();self.assertTrue(reused['reused'])
                self.assertEqual(hashlib.sha256(self.marker.read_bytes()).hexdigest(),before)
                self.assertEqual(db.read_bytes(),b'never open this')
                Path(result['path']).unlink()

    def test_missing_broken_and_legacy_marker_are_diagnostic(self):
        for raw in (None,b'{broken',b'{"status":"started"}',b'{"execution_metadata":{"bad":1},"run_id":"recover","phase":"C"}',b'x'*(RECORD_LIMIT+1)):
            if raw is None:self.marker.unlink(missing_ok=True)
            else:self.marker.write_bytes(raw)
            result=self.recover()
            with zipfile.ZipFile(result['path']) as archive:
                value=json.loads(archive.read('report.json'));self.assertEqual(value['status'],'offline_diagnostic')
                self.assertFalse(value['execution_provenance_verified'])
            Path(result['path']).unlink()

    def test_wrong_run_phase_pin_and_existing_bundle_rejected(self):
        for key,value in (('run_id','other'),('phase','B'),('execution_metadata',{**SYNTHETIC_PROVENANCE,'manifest_sha256':'f'*64})):
            report=self.report();report[key]=value;atomic_record(self.marker,report)
            with self.assertRaises(IndexFailure):self.recover()
        atomic_record(self.marker,self.report());result=self.recover();old=Path(result['path']).read_bytes()
        with self.assertRaises(IndexFailure):recover_report(workspace=self.root,run_id='recover',provenance={**SYNTHETIC_PROVENANCE,'anchor_sha256':'f'*64})
        self.assertEqual(Path(result['path']).read_bytes(),old)

    def test_nested_structure_damage_yields_bounded_diagnostic(self):
        for field,value in (('scans',None),('scans',[{'result':'broken'}]),('stages',{'broken':True}),('phase_resources',{'calls':None})):
            with self.subTest(field=field,value=value):
                report=self.report();report[field]=value;atomic_record(self.marker,report)
                result=self.recover()
                with zipfile.ZipFile(result['path']) as archive:
                    self.assertEqual(set(archive.namelist()),{'report.json','recovery.json'})
                    summary=json.loads(archive.read('report.json'))
                    self.assertEqual(summary['status'],'offline_diagnostic')
                    self.assertEqual(summary['reason'],'marker_structure_invalid')
                Path(result['path']).unlink()

    def test_selected_path_hash_json_count_size_and_symlink_guards(self):
        target=self.root/'recover-C-selected/new/files/x.json';target.parent.mkdir(parents=True);target.write_bytes(b'{"x":1}')
        entry={'name':'x.json','relative_path':'new/files/x.json','sha256':hashlib.sha256(target.read_bytes()).hexdigest()}
        report=self.report();report['extraction']={'files':[entry]};atomic_record(self.marker,report)
        result=self.recover()
        with zipfile.ZipFile(result['path']) as archive:self.assertIn('selected/new/files/x.json',archive.namelist())
        Path(result['path']).unlink()
        for bad in ('../outside.json','new/files/../../../outside.json','/tmp/x.json','new\\files\\x.json'):
            report['extraction']['files']=[{**entry,'relative_path':bad}];atomic_record(self.marker,report)
            with self.assertRaises(IndexFailure):self.recover()
        report['extraction']['files']=[{**entry,'sha256':'0'*64}];atomic_record(self.marker,report)
        with self.assertRaises(IndexFailure):self.recover()
        report['extraction']['files']=[entry]*4;atomic_record(self.marker,report)
        with self.assertRaises(IndexFailure):self.recover()
        report['extraction']['files']=[entry];atomic_record(self.marker,report);target.write_bytes(b'x'*(FILE_LIMIT+1))
        with self.assertRaises(IndexFailure):self.recover()
        target.unlink()
        try:target.symlink_to(self.marker)
        except OSError:return  # Windows에서 링크 권한이 없으면 별도 모의 검사로 확인한다.
        with self.assertRaises(IndexFailure):self.recover()

    def test_atomic_replace_failure_preserves_previous_marker(self):
        original=self.report();atomic_record(self.marker,original,initial=True)
        with patch('rewind_archive_index.extension_records.os.replace',side_effect=OSError('injected')):
            with self.assertRaises(OSError):atomic_record(self.marker,self.report('failed'))
        self.assertEqual(json.loads(self.marker.read_bytes()),original)

    @unittest.skipUnless(importlib.util.find_spec('indexed_gzip') and importlib.util.find_spec('psutil'), 'install requirements-index.txt')
    def test_passed_recovery_preserves_c_evidence_and_verified_files(self):
        import test_persistent_index_extension as fixture_module
        case=fixture_module.ExtensionTests();case.setUpClass();case.setUp();self.addCleanup(case.doCleanups)
        folder,index,anchor=case.baseline();report=case.execute(folder,anchor)
        self.assertEqual(report['status'],'passed',report.get('error'))
        result=recover_report(workspace=folder/'C-work',run_id='extend',provenance=SYNTHETIC_PROVENANCE)
        with zipfile.ZipFile(result['path']) as archive:
            saved=json.loads(archive.read('report.json'))
            for key in ('old_coverage','old_catalog_exact_prefix','verification_status','cold_cache_clear','restore_c'):
                self.assertIn(key,saved)
            self.assertEqual(saved['restore_c']['receipt'],report['snapshot']['receipt'])
            self.assertGreater(saved['after']['index_coverage'],saved['old_coverage'])
            self.assertTrue(saved['persisted_index_validation']['cold_new_member_first'])
            self.assertIn('source_read_trace',saved['cold_new_extraction']['stats'])
            self.assertEqual(json.loads(archive.read('anchor.json'))['selected'],report['selected'])
            selected=[name for name in archive.namelist() if name.startswith('selected/')]
            self.assertEqual(len(selected),3)
            self.assertFalse(any(name.endswith('.sqlite') for name in archive.namelist()))


if __name__=='__main__':unittest.main()
