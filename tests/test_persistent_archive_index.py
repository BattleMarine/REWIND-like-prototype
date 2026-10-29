"""실제 물리 split·재프로세스 복원·원자성·tar/gzip 완료 경계 검사."""
import copy
from dataclasses import replace
import gzip
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import random
import sqlite3
import struct
import subprocess
import sys
import tarfile
import tempfile
import unittest
from unittest.mock import patch
import zlib

from rewind_archive_index import ArchiveIndex, IndexFailure, IndexLimits, make_source
from rewind_archive_index.source import canonical,digest


def png(seed):
    rng=random.Random(seed)
    raw=b''.join(b'\0'+rng.randbytes(256*3) for _ in range(128))
    def chunk(kind,data):return struct.pack('>I',len(data))+kind+data+struct.pack('>I',zlib.crc32(kind+data)&0xffffffff)
    return b'\x89PNG\r\n\x1a\n'+chunk(b'IHDR',struct.pack('>IIBBBBB',256,128,8,2,0,0,0))+chunk(b'IDAT',zlib.compress(raw))+chunk(b'IEND',b'')


def fixture(entries,format=tarfile.PAX_FORMAT):
    stream=io.BytesIO()
    with tarfile.open(fileobj=stream,mode='w',format=format) as tar:
        for name,data in entries:
            info=tarfile.TarInfo(name);info.size=len(data);tar.addfile(info,io.BytesIO(data))
    return gzip.compress(stream.getvalue(),compresslevel=1)


@unittest.skipUnless(importlib.util.find_spec('indexed_gzip'),'install requirements-index.txt')
class PersistentIndexTests(unittest.TestCase):
    def setUp(self):
        parent=Path(__file__).resolve().parents[1]/'outputs/persistent_index_tests'
        parent.mkdir(exist_ok=True,parents=True)
        self.tmp=tempfile.TemporaryDirectory(dir=parent);self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name)
        self.limits=IndexLimits(max_read_bytes=32*1024**2,max_decoded_bytes=32*1024**2)
        self.entries=[]
        for i,frame in enumerate([10,2,1,11,3,20,4,21,5,22,6,23]):
            base=f'scene/Day/actor/item/{i:03d}/'+('long_motion_'*12 if i==3 else 'motion')
            self.entries.append((f'{base}/fisheye_final_image/camera_left/final_{frame}.png',png(i)))
            self.entries.append((f'{base}/json/frame_{frame}.json',json.dumps({'frame':frame,'pose':[i,2,3]}).encode()))
        self.data=fixture(self.entries)
        self.parts=[]
        for i,start in enumerate(range(0,len(self.data),170123)):
            p=self.root/f'archive.tar.gz{i:03d}';p.write_bytes(self.data[start:start+170123]);self.parts.append(p)
        self.spec=make_source(self.parts,'mixed_fixture','mixed')
        self.idx=ArchiveIndex(self.root/'index.sqlite',self.limits)
        self.idx.create(self.spec,spacing=65536)

    def test_batches_fresh_process_resume_extract_and_physical_order(self):
        first=self.idx.scan(5);self.assertFalse(first['complete']);self.assertEqual(first['new_members'],5)
        cmd=[sys.executable,'-m','rewind_archive_index','scan',str(self.idx.path),'--max-new-members','4']
        env=dict(os.environ);env['PYTHONPATH']=os.pathsep.join(sys.path)
        r=subprocess.run(cmd,cwd=Path(__file__).resolve().parents[1],env=env,capture_output=True,text=True)
        self.assertEqual(r.returncode,0,r.stdout+r.stderr)
        middle=json.loads(r.stdout);self.assertEqual(middle['members'],9)
        self.assertGreater(middle['generation'],first['generation'])
        result=self.idx.scan(100)
        self.assertTrue(result['complete']);self.assertTrue(result['gzip_crc_valid'])
        names=[n for n,d in self.entries]
        state,_=self.idx._snapshot();self.assertEqual([r['name'] for r in state['members']],names)
        report=self.idx.extract([names[-1],names[0]],self.root/'selected')
        self.assertEqual([r['name'] for r in report['files']],[names[0],names[-1]])
        self.assertEqual([r['sha256'] for r in report['files']],
            [hashlib.sha256(self.entries[i][1]).hexdigest() for i in (0,len(self.entries)-1)])
        self.assertTrue(any(t['part']>0 for t in report['stats']['source_read_trace']))
        with self.assertRaisesRegex(IndexFailure,'member_not_indexed'):self.idx.lookup('unknown.json')
        self.assertFalse(report['existing_archive_verify_compatible'])

    def test_stale_writer_regression_and_atomic_failed_commit(self):
        before,blob=self.idx._snapshot();self.idx.scan(2)
        with self.assertRaisesRegex(IndexFailure,'stale_generation'):self.idx._commit(0,before,blob)
        latest,blob=self.idx._snapshot();bad=copy.deepcopy(latest);bad['members']=[];bad['boundary']=0
        with self.assertRaisesRegex(IndexFailure,'progress_regression'):self.idx._commit(latest['generation'],bad,blob)
        snapshot=self.idx.status()
        with patch.object(ArchiveIndex,'_commit',side_effect=KeyboardInterrupt):
            with self.assertRaises(KeyboardInterrupt):self.idx.scan(2)
        self.assertEqual(self.idx.status(),snapshot)

    def test_source_mutation_order_gap_checksum_and_optional_hash(self):
        bad=copy.deepcopy(self.spec);bad['parts'][0]['index']=1
        with self.assertRaisesRegex(IndexFailure,'source_order'):ArchiveIndex(self.root/'bad.sqlite').create(bad)
        audit=self.idx.audit_source_sha();self.assertEqual(audit['parts'][0]['sha256'],hashlib.sha256(self.parts[0].read_bytes()).hexdigest())
        self.parts[-1].write_bytes(self.parts[-1].read_bytes()+b'x')
        with self.assertRaisesRegex(IndexFailure,'source_changed'):self.idx.status()

    def test_duplicate_member_and_crc_failure_never_complete(self):
        duplicate=self.root/'dup.gz';duplicate.write_bytes(fixture([('a.json',b'{}'),('a.json',b'{}')]))
        idx=ArchiveIndex(self.root/'dup.sqlite',self.limits);idx.create(make_source([duplicate],'dup','pose'),65536)
        with self.assertRaisesRegex(IndexFailure,'duplicate_tar_member'):idx.scan(10)
        self.assertEqual(idx.status()['members'],0)
        bad=self.root/'crc.gz';bad.write_bytes(self.data[:-1]+bytes([self.data[-1]^1]))
        idx=ArchiveIndex(self.root/'crc.sqlite',self.limits);idx.create(make_source([bad],'crc','mixed'),65536)
        try:idx.scan(100)
        except Exception:pass
        self.assertFalse(idx.status()['complete'])

    def test_crc_pending_then_explicit_verify_and_truncated_source(self):
        result=self.idx.scan(100,verify_end=False)
        self.assertTrue(result['catalog_complete']);self.assertFalse(result['complete'])
        verified=self.idx.verify();self.assertTrue(verified['complete'])
        self.assertNotIn('stats',self.idx.verify())  # 완료후 재검증은 원본을 다시 읽지 않는다.
        part=self.root/'prefix.gz';part.write_bytes(self.data[:len(self.data)//2])
        idx=ArchiveIndex(self.root/'prefix.sqlite',self.limits);idx.create(make_source([part],'prefix','mixed'),65536)
        for _ in range(3):idx.scan(5,verify_end=False)
        self.assertFalse(idx.status()['complete'])

    def test_mid_file_decoded_budget_preserves_complete_boundary(self):
        limited=ArchiveIndex(self.idx.path,replace(self.limits,max_decoded_bytes=120000))
        result=limited.scan(100,verify_end=False)
        self.assertEqual(result['stop_reason'],'decoded_bytes_limit')
        state,_=self.idx._snapshot();self.assertGreater(len(state['members']),0)
        for row in state['members']:
            expected=dict(self.entries)[row['name']];self.assertEqual(row['sha256'],hashlib.sha256(expected).hexdigest())
        self.assertTrue(self.idx.scan(100)['complete'])

    def test_gnu_longname_pending_boundary_and_checkpoint_hash(self):
        p=self.root/'gnu.gz';entries=[('long_'*30+'/frame_10.json',b'{}'),('frame_2.json',b'{"x":2}')]
        p.write_bytes(fixture(entries,tarfile.GNU_FORMAT))
        idx=ArchiveIndex(self.root/'gnu.sqlite',self.limits);idx.create(make_source([p],'gnu','pose'),65536)
        self.assertEqual(idx.scan(1)['members'],1)
        self.assertTrue(idx.scan(10)['complete'])
        db=sqlite3.connect(idx.path)
        try:db.execute("UPDATE checkpoint SET gzidx_sha='invalid'");db.commit()
        finally:db.close()
        with self.assertRaisesRegex(IndexFailure,'checkpoint_checksum'):idx.status()

    def test_native_read_budget_mid_member_retains_safe_checkpoint(self):
        limited=ArchiveIndex(self.idx.path,replace(self.limits,max_read_bytes=786432))
        result=limited.scan(100,verify_end=False)
        self.assertEqual(result['stop_reason'],'read_bytes_limit')
        self.assertGreater(result['members'],0)
        state,_=self.idx._snapshot()
        self.assertEqual(state['boundary'],state['members'][-1]['end'])
        self.idx.extract([state['members'][-1]['name']],self.root/'budget-selected')
        self.assertTrue(self.idx.scan(100)['complete'])

    def test_budget_during_pax_metadata_replays_chain_from_last_boundary(self):
        limited=ArchiveIndex(self.idx.path,replace(self.limits,max_decoded_bytes=512))
        result=limited.scan(10,verify_end=False)
        self.assertEqual(result['members'],0);self.assertEqual(result['boundary'],0)
        self.assertTrue(self.idx.scan(100)['complete'])

    def test_engine_version_and_noncontiguous_snapshot_rejected(self):
        self.idx.scan(3)
        state,blob=self.idx._snapshot();bad=copy.deepcopy(state);bad['members'][1]['chain_start']+=512
        with self.assertRaisesRegex(IndexFailure,'catalog_order_gap'):self.idx._commit(state['generation'],bad,blob)
        db=sqlite3.connect(self.idx.path)
        try:
            state['engine_version']='future-unsupported';raw=canonical(state)
            db.execute('UPDATE checkpoint SET state=?,state_sha=?',(raw,hashlib.sha256(raw).hexdigest()));db.commit()
        finally:db.close()
        with self.assertRaisesRegex(IndexFailure,'index_engine_version_changed'):self.idx.scan(3)

    def test_sql_update_abort_rolls_back_state_blob_generation(self):
        self.idx.scan(2)
        before=self.idx._snapshot()
        db=sqlite3.connect(self.idx.path)
        try:
            db.execute("CREATE TRIGGER fail_checkpoint BEFORE UPDATE ON checkpoint BEGIN SELECT RAISE(ABORT,'test_abort'); END")
            db.commit()
        finally:db.close()
        with self.assertRaises(sqlite3.IntegrityError):self.idx.scan(3)
        self.assertEqual(self.idx._snapshot(),before)
        db=sqlite3.connect(self.idx.path)
        try:db.execute('DROP TRIGGER fail_checkpoint');db.commit()
        finally:db.close()
        self.assertTrue(self.idx.scan(100)['complete'])

    def test_torn_tail_two_zero_and_nonzero_after_tar_not_complete(self):
        # gzip CRC가 정상이더라도 tar 끝의2개 zero block을 만족해야 한다.
        raw=io.BytesIO()
        with tarfile.open(fileobj=raw,mode='w') as tar:
            info=tarfile.TarInfo('a.json');info.size=2;tar.addfile(info,io.BytesIO(b'{}'))
        for label,data in [('one-zero',raw.getvalue()[:1536]),('nonzero-tail',raw.getvalue()+b'bad')]:
            p=self.root/(label+'.gz');p.write_bytes(gzip.compress(data))
            idx=ArchiveIndex(self.root/(label+'.sqlite'),self.limits);idx.create(make_source([p],label,'pose'),65536)
            try:idx.scan(20)
            except (IndexFailure,ValueError):pass
            self.assertFalse(idx.status()['complete'])

    def test_optional_strong_hash_detects_metadata_preserved_change(self):
        p=self.root/'audit-only.gz';p.write_bytes(fixture([('a.json',b'{}')]))
        expected=hashlib.sha256(p.read_bytes()).hexdigest()
        spec=make_source([p],'audit','pose',expected_sha256=[expected])
        idx=ArchiveIndex(self.root/'audit.sqlite',self.limits);idx.create(spec,65536)
        meta=p.stat();data=p.read_bytes();p.write_bytes(data[:-1]+bytes([data[-1]^1]));os.utime(p,ns=(meta.st_atime_ns,meta.st_mtime_ns))
        self.assertFalse(idx.status()['complete'])  # 기본metadata검사에대한의도된한계.
        with self.assertRaisesRegex(IndexFailure,'source_sha_changed'):idx.audit_source_sha()


if __name__=='__main__':unittest.main()
