"""실제 API 없이 portable metadata·strict Range·cache·final identity를 검사한다."""
import copy
from dataclasses import replace
import gzip
import importlib.util
import io
from pathlib import Path
import random
import tempfile
from types import SimpleNamespace
import unittest

from rewind_archive_index import ArchiveIndex,IndexFailure,IndexLimits
from rewind_archive_index.drive import DriveRuntime,DriveLimits,make_drive_source
from rewind_archive_index.source import Budget
from test_drive_metadata import FakeDrive
from test_persistent_archive_index import fixture,png


@unittest.skipUnless(importlib.util.find_spec('indexed_gzip'),'install requirements-index.txt')
class PersistentDriveTests(unittest.TestCase):
    def setUp(self):
        parent=Path(__file__).resolve().parents[1]/'outputs/persistent_drive_tests';parent.mkdir(parents=True,exist_ok=True)
        self.tmp=tempfile.TemporaryDirectory(dir=parent);self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name)
        raw=fixture([(f'frame_{i}.png',png(i)) for i in range(8)])
        pieces=[raw[:len(raw)//2],raw[len(raw)//2:]]
        self.selection={'parts':[{'index':i,'name':f'pose.tar.gz{i:03d}','file_id':f'fake_{i}'} for i in range(2)]}
        self.session=FakeDrive(self.selection,{f'fake_{i}':b for i,b in enumerate(pieces)})
        self.session.get_adapter=lambda _:SimpleNamespace(max_retries=SimpleNamespace(total=0))
        manifest={'schema_version':'archive_manifest_v1','archive_id':'remote','dataset_version':'synthetic_v1','kind':'pose',
            'source_scope':'user_drive','total_parts':2,'parts':[{'index':i,'name':f'pose.tar.gz{i:03d}',
            'size_bytes':len(b),'version':'9','source':{'type':'drive','file_id':f'fake_{i}'},
            'identity':{'version':'9','modified_time':'2026-10-14T00:00:00.000Z'}} for i,b in enumerate(pieces)]}
        self.spec=make_drive_source(manifest)
        self.runtime=DriveRuntime(self.session)
        self.idx=ArchiveIndex(self.root/'index.sqlite',runtime=self.runtime);self.idx.create(self.spec,65536)

    def test_remote_scan_restore_extract_cache_and_stats(self):
        first=self.idx.scan(3)
        self.assertTrue(first['stats']['network_requested']);self.assertTrue(first['stats']['remote_final_identity_checked'])
        self.assertTrue(first['network_requested'])
        self.assertGreater(first['stats']['remote_cache_hits'],0)
        new=ArchiveIndex(self.idx.path,runtime=DriveRuntime(self.session))
        second=new.scan(3);self.assertEqual(second['members'],6)
        r=new.extract(['frame_4.png'],self.root/'selected');self.assertTrue(r['stats']['network_requested'])
        self.assertLessEqual(r['stats']['remote_cache_resident_bytes'],4*1024**2)
        self.assertNotIn('path',str(self.spec));self.assertNotIn('session',str(self.spec))
        self.assertFalse(second['complete'])
        with self.assertRaisesRegex(IndexFailure,'drive_full_crc_deferred'):new.verify()

    def test_same_runtime_budget_is_shared_across_scan_and_extract(self):
        self.idx.scan(3);before=self.runtime.transfer.ranges
        self.idx.extract(['frame_0.png'],self.root/'shared-selected')
        self.assertGreater(self.runtime.transfer.ranges,before)
        stats=self.runtime.stats();self.assertEqual(stats['phase_http_requests'],self.runtime.transfer.ranges)
        self.assertEqual(stats['remote_http_requests'],self.runtime.transfer.ranges-before)
        self.assertLessEqual(stats['phase_response_body_bytes'],self.runtime.limits.max_body_bytes)

    def test_identity_changed_on_cached_source_refuses_commit(self):
        self.idx.scan(2);before=self.idx.status()
        self.session.docs['fake_0']['version']='changed'
        with self.assertRaisesRegex(IndexFailure,'source_changed'):self.idx.scan(2)
        self.assertEqual(self.idx.status(),before);self.assertEqual(self.runtime.cache_bytes,0)

    def test_bad_range_never_cached_and_budget_fails_closed(self):
        self.session.protocol['fake_0']=(200,{})
        with self.assertRaisesRegex(IndexFailure,'range_status_not_206'):self.idx.scan(2)
        self.assertEqual(self.idx.status()['generation'],0);self.assertEqual(self.runtime.cache_bytes,0)
        self.session.protocol.clear()
        limited=DriveRuntime(self.session,DriveLimits(max_requests=5))
        idx=ArchiveIndex(self.idx.path,runtime=limited)
        result=idx.scan(3)
        self.assertTrue(result['checkpoint_deferred']);self.assertEqual(result['generation'],0)

    def test_final_metadata_change_refuses_cached_checkpoint(self):
        self.idx.scan(2);before=self.idx.status();original=self.runtime.finish
        def change(budget):self.session.docs['fake_1']['version']='new';return original(budget)
        self.runtime.finish=change
        with self.assertRaisesRegex(IndexFailure,'source_changed'):self.idx.scan(2)
        self.assertEqual(self.idx.status(),before)

    def test_source_order_and_total_rejected(self):
        from rewind_archive_index.drive import validate_drive_spec
        bad=copy.deepcopy(self.spec);bad['parts'].reverse()
        with self.assertRaisesRegex(IndexFailure,'drive_part_order'):validate_drive_spec(bad)
        bad=copy.deepcopy(self.spec);bad['total_bytes']-=1
        with self.assertRaisesRegex(IndexFailure,'drive_total_size'):validate_drive_spec(bad)
