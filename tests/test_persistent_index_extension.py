"""고정 coverage 밖 누적 확장과 C snapshot의 새 handle 검증. 실제 Google 호출 없음."""
import hashlib
import importlib.util
import json
from pathlib import Path
import random
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from rewind_archive_index import ArchiveIndex,IndexFailure
from rewind_archive_index.drive import DriveRuntime,make_drive_source
from rewind_archive_index.extension_pilot import run_extension
from rewind_archive_index.snapshots import export_snapshot
from test_drive_metadata import FakeDrive
from test_persistent_archive_index import fixture


@unittest.skipUnless(importlib.util.find_spec('indexed_gzip') and importlib.util.find_spec('psutil'),'install requirements-index.txt')
class ExtensionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        randomizer=random.Random(3010)
        cls.data=fixture([(f'sequence/json/frame_{i}.json',json.dumps({'i':i,'pose':randomizer.randbytes(32768).hex()}).encode()) for i in range(180)])

    def setUp(self):
        parent=Path(__file__).resolve().parents[1]/'outputs/persistent_extension_tests';parent.mkdir(parents=True,exist_ok=True)
        self.tmp=tempfile.TemporaryDirectory(dir=parent);self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name)
        selection={'parts':[{'index':0,'name':'pose.tar.gz000','file_id':'fake'}]}
        self.session=FakeDrive(selection,{'fake':self.data});self.session.close=lambda:None
        self.session.get_adapter=lambda _:SimpleNamespace(max_retries=SimpleNamespace(total=0))
        self.manifest={'schema_version':'archive_manifest_v1','archive_id':'extension','dataset_version':'synthetic_v1','kind':'pose',
            'source_scope':'user_drive','total_parts':1,'parts':[{'index':0,'name':'pose.tar.gz000','size_bytes':len(self.data),
            'version':'9','source':{'type':'drive','file_id':'fake'},'identity':{'version':'9','modified_time':'2026-10-14T00:00:00.000Z'}}]}

    def baseline(self,spacing=65536):
        folder=self.root/str(spacing);folder.mkdir()
        index=ArchiveIndex(folder/'B.sqlite',runtime=DriveRuntime(self.session));index.create(make_drive_source(self.manifest),spacing)
        index.scan(4,verify_end=False);a=export_snapshot(index,folder/'B-input')
        index.scan(4,verify_end=False);b=export_snapshot(index,folder/'B-input',known_anchor=a['receipt'])
        state,_=index._snapshot()
        anchor={'schema':'persistent_drive_pilot_anchor_v1','runtime_id':'fake-runtime-B','receipt':b['receipt'],
            'selected':[{'name':r['name'],'sha256':r['sha256']} for r in state['members'][:2]]}
        return folder,index,anchor

    def execute(self,folder,anchor,run_id='extend'):
        return run_extension(run_id=run_id,workspace=folder/'C-work',input_snapshot_directory=folder/'B-input',
            output_snapshot_directory=folder/'C-output',manifest=self.manifest,anchor=anchor,session_factory=lambda:self.session)

    def test_multiple_restart_spacings_extend_and_cold_snapshot_read(self):
        for spacing in (65536,262144,1048576):
            with self.subTest(spacing=spacing):
                folder,index,anchor=self.baseline(spacing);before=index._snapshot()
                result=self.execute(folder,anchor)
                self.assertEqual(result['status'],'passed',result)
                self.assertGreater(result['after']['boundary'],result['old_coverage'])
                self.assertGreater(result['after']['index_coverage'],result['old_coverage'])
                self.assertGreaterEqual(result['new_member']['offset'],result['old_coverage'])
                self.assertLessEqual(result['new_member']['end'],result['after']['index_coverage'])
                self.assertEqual(index._snapshot(),before)
                self.assertEqual(result['snapshot']['receipt']['parent_snapshot_id'],anchor['receipt']['snapshot_id'])
                self.assertTrue(result['persisted_index_validation']['cold_new_member_first'])
                self.assertGreater(result['persisted_index_validation']['new_read_min'],0)
                self.assertLessEqual(result['transport']['phase_response_body_bytes'],8*1024**2)
                self.assertLessEqual(result['transport']['phase_http_requests'],120)
                self.assertLessEqual(result['phase_resources']['read_bytes'],32*1024**2)
                self.assertLessEqual(result['phase_resources']['decoded_bytes'],32*1024**2)
                with self.assertRaisesRegex(IndexFailure,'pilot_run_already_consumed'):self.execute(folder,anchor)

    def test_same_coverage_never_passes_from_member_count_only(self):
        folder,index,anchor=self.baseline()
        original=ArchiveIndex.scan
        def fixed(instance,*args,**kwargs):
            state,blob=instance._snapshot();coverage=state['index_coverage'];result=original(instance,*args,**kwargs)
            db=instance._connect()
            try:
                current,new_blob=instance._load(db);current['index_coverage']=coverage;raw=__import__('rewind_archive_index.source',fromlist=['canonical']).canonical(current)
                db.execute('UPDATE checkpoint SET state=?,state_sha=?',(raw,hashlib.sha256(raw).hexdigest()));db.commit()
            finally:db.close()
            return result
        with patch.object(ArchiveIndex,'scan',fixed):result=self.execute(folder,anchor)
        self.assertEqual(result['status'],'partial');self.assertNotIn('snapshot',result)

    def test_changed_source_and_budget_preserve_failed_or_partial_report(self):
        folder,index,anchor=self.baseline();self.session.docs['fake']['version']='new'
        failed=self.execute(folder,anchor);self.assertEqual(failed['status'],'failed');self.assertEqual(failed['error'],'source_changed')
        self.assertTrue((folder/'C-work/extend-C.json').exists())
        self.assertNotIn('snapshot',failed)

    def test_budget_exhaustion_is_partial_not_success(self):
        from rewind_archive_index.drive import DriveLimits
        folder,index,anchor=self.baseline()
        with patch('rewind_archive_index.extension_pilot.PILOT_DRIVE_LIMITS',DriveLimits(max_body_bytes=16384,max_requests=3,max_seconds=300)):
            result=self.execute(folder,anchor)
        self.assertEqual(result['status'],'partial');self.assertNotIn('snapshot',result)

    def test_failed_verification_keeps_published_c_and_b_unchanged(self):
        folder,index,anchor=self.baseline();original=ArchiveIndex.extract
        def fail(instance,names,output):
            if 'selected' in str(output):raise IndexFailure('injected_cold_failure')
            return original(instance,names,output)
        with patch.object(ArchiveIndex,'extract',fail):result=self.execute(folder,anchor)
        self.assertEqual(result['status'],'failed');self.assertIn('snapshot',result)
        self.assertTrue(Path(result['snapshot']['receipt_path']).is_file())
        self.assertEqual(index.status()['generation'],anchor['receipt']['generation'])

    def test_old_restart_read_cannot_pass_even_with_growing_blob(self):
        folder,index,anchor=self.baseline();original=ArchiveIndex.extract
        def old_trace(instance,names,output):
            result=original(instance,names,output)
            if str(output).endswith('selected\\new') or str(output).endswith('selected/new'):
                result['stats']['source_read_trace'][0]['start']=0
            return result
        with patch.object(ArchiveIndex,'extract',old_trace):result=self.execute(folder,anchor)
        self.assertEqual(result['status'],'failed');self.assertEqual(result['error'],'extension_new_restart_not_used')
        self.assertIn('snapshot',result)


if __name__=='__main__':unittest.main()
