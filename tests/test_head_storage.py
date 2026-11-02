"""Mount HEAD의 hardlink/fsync 비의존·충돌·불완전 기록·이전본 보존 검사."""
import errno
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from rewind_archive.contracts import canonical,digest
from rewind_archive_index.head_storage import publish_head
import rewind_archive_index.head_storage as storage
from rewind_archive_index.preparation import _head
from rewind_archive_index.source import IndexFailure


def head(generation):
    receipt={'schema':'persistent_archive_snapshot_v1','snapshot_id':str(generation+1)*64,
        'filename':f'g{generation}-'+str(generation+1)*64+'.zip','bytes':100,'source_sha256':'a'*64,
        'generation':generation,'db_sha256':'b'*64,'lineage':'c'*32,
        'parent_snapshot_id':str(generation)*64 if generation else None}
    body={'schema':'archive_prepare_head_v1','source_sha256':'a'*64,'receipts':[receipt]}
    return {**body,'content_sha256':digest(body)}


class MountedHeadTests(unittest.TestCase):
    def setUp(self):
        root=Path(__file__).resolve().parents[1]/'outputs/head_storage_tests';root.mkdir(exist_ok=True)
        self.tmp=tempfile.TemporaryDirectory(dir=root);self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name)
        self.path=self.root/('a'*64+'-head.json');self.state={};self.events=[]

    def publish(self,value,expected,progress=None):
        return publish_head(self.path,value,expected,state=self.state,progress=progress or self.events.append)

    def test_initial_and_update_without_link_or_fsync(self):
        proxy=SimpleNamespace(replace=storage.os.replace,link=lambda *a:(_ for _ in ()).throw(OSError(errno.EPERM,'no link')),
            fsync=lambda *a:(_ for _ in ()).throw(OSError(95,'no fsync')))
        with patch.object(storage,'os',proxy):
            self.publish(head(0),None);old=self.path.read_bytes();self.publish(head(1),head(0))
        self.assertEqual(_head(self.path,'a'*64,None),head(1))
        self.assertEqual(Path(self.state['previous_path']).read_bytes(),old)
        self.assertTrue(self.state['verified']);self.assertFalse(self.state['mount_atomicity_claimed'])
        self.assertFalse(self.state['durability_proven'])
        self.assertIn('head_create_pre',self.events);self.assertIn('head_replace_pre',self.events)

    def test_stale_expected_and_initial_race_never_overwrite(self):
        self.publish(head(0),None);old=self.path.read_bytes()
        with self.assertRaisesRegex(IndexFailure,'prepare_head_cas_conflict'):self.publish(head(1),head(2))
        self.assertEqual(self.path.read_bytes(),old)
        self.path.unlink()
        def race(operation):
            if operation=='head_create_pre':self.path.write_bytes(old)
        with self.assertRaises(FileExistsError):self.publish(head(1),None,race)
        self.assertEqual(self.path.read_bytes(),old)

    def test_replace_unsupported_preserves_old_backup_and_closed_staging(self):
        self.publish(head(0),None);old=self.path.read_bytes()
        with patch.object(storage.os,'replace',side_effect=OSError(95,'unsupported')):
            with self.assertRaises(OSError):self.publish(head(1),head(0))
        self.assertEqual(self.path.read_bytes(),old)
        self.assertEqual(Path(self.state['previous_path']).read_bytes(),old)
        self.assertEqual(json.loads(Path(self.state['staging_path']).read_bytes()),head(1))
        self.assertEqual(self.state['operation'],'head_replace_pre')

    def test_incomplete_replace_is_rejected_and_previous_head_remains(self):
        self.publish(head(0),None);old=self.path.read_bytes()
        def torn(source,target):Path(target).write_bytes(b'{broken')
        with patch.object(storage.os,'replace',torn):
            with self.assertRaisesRegex(IndexFailure,'prepare_head_invalid_json'):self.publish(head(1),head(0))
        with self.assertRaises(ValueError):_head(self.path,'a'*64,None)
        self.assertEqual(Path(self.state['previous_path']).read_bytes(),old)

    def test_partial_initial_write_is_kept_and_never_read_as_head(self):
        original=Path.open;target=self.path
        class Partial:
            def __init__(self,stream):self.stream=stream
            def __enter__(self):return self
            def __exit__(self,*args):self.stream.close()
            def write(self,data):self.stream.write(data[:12]);self.stream.flush();raise OSError(errno.ENOSPC,'full')
        def opening(path,*args,**kwargs):
            result=original(path,*args,**kwargs)
            return Partial(result) if path==target and args==('xb',) else result
        with patch.object(Path,'open',opening):
            with self.assertRaises(OSError):self.publish(head(0),None)
        self.assertTrue(self.path.exists())
        with self.assertRaises(ValueError):_head(self.path,'a'*64,None)
        with self.assertRaisesRegex(IndexFailure,'prepare_head_cas_conflict'):self.publish(head(0),None)

    def test_backup_collision_does_not_overwrite_head_or_backup(self):
        self.publish(head(0),None);old=self.path.read_bytes()
        backup=self.path.with_name(self.path.stem+'-previous-'+hashlib.sha256(old).hexdigest()+'.json')
        backup.write_bytes(b'incomplete')
        with self.assertRaisesRegex(IndexFailure,'prepare_head_backup_conflict'):self.publish(head(1),head(0))
        self.assertEqual(self.path.read_bytes(),old);self.assertEqual(backup.read_bytes(),b'incomplete')

    def test_cas_recheck_during_staging_preserves_competing_head(self):
        self.publish(head(0),None)
        competitor=canonical(head(2))+b'\n'
        def race(operation):
            if operation=='head_cas_pre':self.path.write_bytes(competitor)
        with self.assertRaisesRegex(IndexFailure,'prepare_head_cas_conflict'):self.publish(head(1),head(0),race)
        self.assertEqual(self.path.read_bytes(),competitor)


@unittest.skipUnless(importlib.util.find_spec('indexed_gzip') and importlib.util.find_spec('psutil'),'install requirements-index.txt')
class PreparationHeadDiagnostics(unittest.TestCase):
    def test_head_errno_and_operation_saved_without_raw_message(self):
        from test_index_preparation import AutomaticPreparationTests
        case=AutomaticPreparationTests();case.setUp();self.addCleanup(case.doCleanups)
        original=Path.open
        def opening(path,*args,**kwargs):
            if path.name.endswith('-head.json') and args==('xb',):raise PermissionError(errno.EACCES,'SECRET token URL')
            return original(path,*args,**kwargs)
        with patch.object(Path,'open',opening):report=case.execute()
        failure=report['persistent_index']['failure']
        self.assertEqual(failure,{'type':'PermissionError','errno':errno.EACCES,'operation':'head_create_pre'})
        self.assertEqual(report['persistent_index']['scans'],[])
        self.assertEqual(report['persistent_index']['failure_stage'],'head_create_pre')
        self.assertNotIn(b'SECRET',canonical(report))


if __name__=='__main__':unittest.main()
