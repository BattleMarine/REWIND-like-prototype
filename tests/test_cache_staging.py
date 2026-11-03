"""고정 chunk staging의 copy/hash/실패 보존·경로 계약. 실제 Drive/Colab0."""
import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from rewind_archive.cache_staging import stage_file,CHUNK_BYTES
from rewind_archive.contracts import PreparationError

ROOT=Path(__file__).resolve().parents[1]


class CacheStagingTests(unittest.TestCase):
    def setUp(self):
        parent=ROOT/'outputs/test_tmp';parent.mkdir(parents=True,exist_ok=True)
        self.temp=tempfile.TemporaryDirectory(dir=parent);self.addCleanup(self.temp.cleanup)
        self.base=Path(self.temp.name);self.drive=self.base/'drive';self.local=self.base/'local'
        self.drive.mkdir();self.local.mkdir()
        self.source=self.drive/'cache.zip';self.target=self.local/'cache.zip'
        self.data=b'fixed cache bytes\n'*(CHUNK_BYTES//8)
        self.source.write_bytes(self.data)
        self.pin=hashlib.sha256(self.data).hexdigest()
        self.report=self.local/'stage.json';self.job=self.local/'job'

    def run_stage(self,**changes):
        args=dict(expected_bytes=len(self.data),expected_sha256=self.pin,readonly_root=self.drive,local_root=self.local,
            forbidden_roots=(self.local/'source',self.local/'inputs'),consumed_job=self.job,report_path=self.report,
            validator=lambda p:{'path':str(p),'checked':True})
        args.update(changes)
        return stage_file(self.source,self.target,**args)

    def last(self):return json.loads(self.report.read_bytes())['attempts'][-1]

    def test_streaming_copy_local_hash_source_unchanged_and_exact_reuse(self):
        before=(self.source.stat().st_mtime_ns,self.source.read_bytes())
        result=self.run_stage()
        self.assertEqual(self.target.read_bytes(),self.data)
        self.assertEqual(result['cache_input']['path'],str(self.target))
        self.assertEqual(self.last()['source_read_bytes'],len(self.data))
        self.assertEqual(self.last()['local_verify_bytes'],len(self.data))
        staged_mtime=self.target.stat().st_mtime_ns
        self.run_stage()
        self.assertEqual(self.last()['source_read_bytes'],0)
        self.assertEqual(self.target.stat().st_mtime_ns,staged_mtime)
        self.assertEqual((self.source.stat().st_mtime_ns,self.source.read_bytes()),before)
        self.assertFalse(self.job.exists())

    def test_bad_source_and_existing_mismatch_no_overwrite_or_fallback(self):
        self.source.write_bytes(b'x'*len(self.data))
        with self.assertRaisesRegex(PreparationError,'source_sha256'):self.run_stage()
        self.assertFalse(self.target.exists());self.assertTrue(self.last()['partial_preserved'])
        # 기존 target 불일치는 별도 경로에서 원본을 읽거나 덮어쓰지 않는다.
        self.source.write_bytes(self.data)
        self.target=self.local/'different.zip';self.target.write_bytes(b'wrong')
        self.report=self.local/'different.json'
        with self.assertRaisesRegex(PreparationError,'local_size'):self.run_stage()
        self.assertEqual(self.target.read_bytes(),b'wrong');self.assertEqual(self.last()['source_read_bytes'],0)

    def test_read_error_preserves_partial_and_refuses_automatic_resume(self):
        real_open=Path.open;source=self.source
        class Broken:
            def __init__(self,stream):self.stream=stream;self.calls=0
            def __enter__(self):return self
            def __exit__(self,*args):self.stream.close()
            def read(self,n):
                self.calls+=1
                if self.calls==2:raise OSError('private transport detail')
                return self.stream.read(n)
        def opened(path,*args,**kwargs):
            stream=real_open(path,*args,**kwargs)
            return Broken(stream) if path==source and args and args[0]=='rb' else stream
        with patch.object(Path,'open',opened),self.assertRaises(OSError):self.run_stage()
        self.assertEqual(self.last()['written_bytes'],CHUNK_BYTES)
        self.assertTrue(self.last()['partial_preserved'])
        self.assertNotIn('private',json.dumps(self.last()))
        with self.assertRaisesRegex(PreparationError,'partial_exists'):self.run_stage()
        self.assertFalse(self.target.exists())

    def test_timeout_after_source_read_accounts_bytes_and_validator_time_on_reuse(self):
        now=[0.];real_open=Path.open;source=self.source
        class Slow:
            def __init__(self,stream):self.stream=stream
            def __enter__(self):return self
            def __exit__(self,*args):self.stream.close()
            def read(self,n):
                result=self.stream.read(n);now[0]=301.;return result
        def opened(path,*args,**kwargs):
            stream=real_open(path,*args,**kwargs)
            return Slow(stream) if path==source and args and args[0]=='rb' else stream
        with patch.object(Path,'open',opened),self.assertRaisesRegex(PreparationError,'time_limit'):
            self.run_stage(clock=lambda:now[0])
        self.assertEqual(self.last()['source_read_bytes'],CHUNK_BYTES)
        self.assertEqual(self.last()['written_bytes'],0)
        self.target=self.local/'reused.zip';self.target.write_bytes(self.data);self.report=self.local/'reuse.json';now[0]=0.
        def validator(path):now[0]=301.;return {'path':str(path)}
        with self.assertRaisesRegex(PreparationError,'time_limit'):
            self.run_stage(clock=lambda:now[0],validator=validator)
        self.assertEqual(self.last()['source_read_bytes'],0)

    def test_paths_job_links_hardlinks_and_validator_rejection(self):
        self.job.mkdir()
        with self.assertRaisesRegex(PreparationError,'consumed_job'):self.run_stage()
        self.job.rmdir()
        self.target=self.job/'cache.zip'
        with self.assertRaisesRegex(PreparationError,'inside_job'):self.run_stage()
        self.assertFalse(self.job.exists())
        self.target=self.local/'source/cache.zip'
        with self.assertRaisesRegex(PreparationError,'source_or_input_tree'):self.run_stage()
        self.target=self.local/'hardlink.zip';self.target.write_bytes(self.data)
        with patch('rewind_archive.cache_staging.os.path.samefile',return_value=True),self.assertRaisesRegex(PreparationError,'same_file'):
            self.run_stage()
        self.target=self.local/'good.zip'
        def reject(path):raise PreparationError('test_input_spec_rejection')
        with self.assertRaisesRegex(PreparationError,'input_spec_rejection'):self.run_stage(validator=reject)
        self.assertFalse(self.target.exists());self.assertTrue(self.last()['partial_preserved'])

    def test_symbolic_link_path_rejected(self):
        link=self.local/'link.zip'
        self.target=link
        original=Path.is_symlink
        def is_link(path):return path==link or original(path)
        with patch.object(Path,'is_symlink',is_link),self.assertRaisesRegex(PreparationError,'cache_stage_link'):
            self.run_stage()

    @unittest.skipUnless((ROOT/'outputs/test_pose_missing67_001_20261102/prefix_cache (3).zip').exists(),'실제 고정 cache 필요')
    def test_existing_input_spec_counted_metadata_matches_default(self):
        from rewind_archive.prefix_cache import input_spec
        from rewind_archive.cache_policy import TAIL352_POLICY
        cache=ROOT/'outputs/test_pose_missing67_001_20261102/prefix_cache (3).zip'
        reads=[]
        def counted(stream,size):
            data=stream.read(size);reads.append(len(data));return data
        self.assertEqual(input_spec(cache,policy=TAIL352_POLICY),input_spec(cache,policy=TAIL352_POLICY,read_callback=counted))
        self.assertGreater(sum(reads),0);self.assertLess(sum(reads),2*1024**2)
