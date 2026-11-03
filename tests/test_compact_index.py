"""명시적 새 DB 복사본 compaction과 실패 partial 보존. 원격 없음."""
import hashlib,json
from pathlib import Path
import sqlite3,tempfile
import unittest
from unittest.mock import patch
from rewind_archive_index.compact import compact_copy
from rewind_archive_index.source import IndexFailure
import test_persistent_snapshots as fixtures


class CompactTests(unittest.TestCase):
    def setUp(self):
        self.helper=fixtures.SnapshotTests();self.helper.setUp();self.addCleanup(self.helper.doCleanups)
        self.helper.idx.scan(2)
        self.export=self.helper.export_a()
        self.source=self.helper.idx.path
        self.pin=hashlib.sha256(self.source.read_bytes()).hexdigest()

    def test_semantics_anchor_source_preserved_and_no_automatic_head(self):
        out=self.helper.root/'compact'
        report=compact_copy(self.source,out,expected_sha256=self.pin)
        self.assertFalse(report['official_resume_head']);self.assertFalse(report['new_source_coverage'])
        self.assertTrue(report['checkpoint_state_blob_anchor_bytes_identical'])
        self.assertTrue(report['source_unchanged'])
        self.assertEqual(hashlib.sha256(self.source.read_bytes()).hexdigest(),self.pin)
        with self.assertRaises(IndexFailure):compact_copy(self.source,out,expected_sha256=self.pin)

    def test_wrong_pin_and_sql_failure_leave_report_and_partial(self):
        with self.assertRaisesRegex(IndexFailure,'compact_source_pin'):
            compact_copy(self.source,self.helper.root/'wrong',expected_sha256='0'*64)
        original=sqlite3.connect
        def broken(path,*args,**kwargs):
            db=original(path,*args,**kwargs)
            if str(path).endswith('compact.partial.sqlite?mode=ro'):
                db.close();raise OSError('private detail')
            return db
        out=self.helper.root/'partial'
        with patch('rewind_archive_index.compact.sqlite3.connect',broken),self.assertRaises(OSError):
            compact_copy(self.source,out,expected_sha256=self.pin)
        report=json.loads((out/'report.json').read_bytes())
        self.assertTrue(report['partial_preserved']);self.assertTrue((out/'compact.partial.sqlite').exists())
        self.assertNotIn('private',json.dumps(report))

    def test_missing_source_sidecar_and_time_error_preserve_report(self):
        missing=self.helper.root/'missing.sqlite';out=self.helper.root/'missing_report'
        with self.assertRaisesRegex(IndexFailure,'compact_source_limit'):
            compact_copy(missing,out,expected_sha256='0'*64)
        self.assertEqual(json.loads((out/'report.json').read_bytes())['reason'],'compact_source_limit')
        sidecar=self.source.with_name(self.source.name+'-wal');sidecar.write_bytes(b'active')
        with self.assertRaisesRegex(IndexFailure,'compact_active_sidecar'):
            compact_copy(self.source,self.helper.root/'wal_report',expected_sha256=self.pin)
        sidecar.unlink()
        with patch('rewind_archive_index.compact.SECONDS',0),self.assertRaisesRegex(IndexFailure,'compact_time_limit'):
            compact_copy(self.source,self.helper.root/'time_report',expected_sha256=self.pin)
        self.assertTrue((self.helper.root/'time_report/report.json').is_file())

    def test_interrupt_after_vacuum_preserves_partial_and_report(self):
        original=sqlite3.connect
        def interrupted(path,*args,**kwargs):
            if str(path).endswith('compact.partial.sqlite?mode=ro'):raise KeyboardInterrupt()
            return original(path,*args,**kwargs)
        out=self.helper.root/'interrupted'
        with patch('rewind_archive_index.compact.sqlite3.connect',interrupted),self.assertRaises(KeyboardInterrupt):
            compact_copy(self.source,out,expected_sha256=self.pin)
        report=json.loads((out/'report.json').read_bytes())
        self.assertEqual(report['error_type'],'KeyboardInterrupt')
        self.assertTrue(report['partial_preserved'])
