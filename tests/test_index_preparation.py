"""실제 prepare/CLI→지속형 인덱스→기존 검증/consumer 연결. 외부 요청 없음."""
import contextlib
import copy
from dataclasses import replace, asdict
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from rewind_archive.contracts import Limits, PreparationError, canonical, digest
from rewind_archive.extraction import prepare
from rewind_archive.fixture import create_fixture, local_manifest
from rewind_archive.matching import verify_extraction, match_extractions
from rewind_archive.__main__ import main
from rewind_archive_index.preparation import PreparationContext, PreparationIndex, backend_hashes
from rewind_archive_index import ArchiveIndex, IndexFailure
from test_persistent_archive_index import fixture


@unittest.skipUnless(importlib.util.find_spec('indexed_gzip') and importlib.util.find_spec('psutil'),'install requirements-index.txt')
class AutomaticPreparationTests(unittest.TestCase):
    def setUp(self):
        parent=Path(__file__).resolve().parents[1]/'outputs/index_preparation_tests';parent.mkdir(exist_ok=True)
        self.tmp=tempfile.TemporaryDirectory(dir=parent);self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name)
        create_fixture(self.root/'raw');self.base=self.root/'raw'
        self.manifest=json.loads((self.base/'pose_manifest.json').read_bytes())
        from rewind_archive.fixture import SEQUENCE
        self.targets=[SEQUENCE+'/json/frame_1.json',SEQUENCE+'/json/frame_0.json']
        self.limits=Limits();self.options={'workspace':str(self.root/'work'),'snapshot_directory':str(self.root/'snapshots'),
            'batch_members':1,'max_batches':8,'spacing':65536}

    def execute(self,output='prepared',targets=None,options=None,**kwargs):
        wanted=targets or self.targets
        return prepare(self.manifest,self.root/output,wanted,self.limits,base=self.base,exact_targets=wanted,
            persistent_index=options or self.options,**kwargs)

    def test_real_prepare_restore_reuse_verify_and_matching(self):
        first=self.execute();self.assertTrue(first['targets_complete']);self.assertFalse(first['train_ready'])
        verified=verify_extraction(self.root/'prepared',self.manifest)
        self.assertEqual(verified['failure_class'],'bounded_stop')
        self.assertEqual([r['member'] for r in verified['files']],list(reversed(self.targets)))
        matching=match_extractions([self.root/'prepared']);self.assertFalse(matching['train_ready'])
        self.assertEqual(matching['observed_frame_count'],2)
        fresh={**self.options,'workspace':str(self.root/'fresh')}
        second=self.execute('restored',options=fresh)
        self.assertEqual(second['persistent_index']['scans'],[])
        self.assertIn('restore',second['persistent_index'])
        self.assertEqual(second['files'],first['files'])
        with patch.object(PreparationIndex,'extract',side_effect=AssertionError('reuse must not read source')):
            reused=self.execute()
        self.assertTrue(reused['reuse']['verified']);self.assertEqual(reused['request_stats']['source_read_bytes'],0)
        file=self.root/'prepared'/first['files'][0]['relative_path'];file.write_bytes(b'changed')
        with self.assertRaises(PreparationError):self.execute()

    def test_partial_batch_budget_persists_and_next_request_resumes(self):
        opts={**self.options,'max_batches':1}
        partial=self.execute(options=opts);self.assertEqual(partial['persistent_index']['status'],'partial')
        self.assertFalse(partial['targets_complete']);verify_extraction(self.root/'prepared',self.manifest)
        saved=partial['persistent_index']['head']['receipts'][-1]
        resumed=self.execute('continued');self.assertTrue(resumed['targets_complete'])
        self.assertEqual(resumed['persistent_index']['before']['generation'],saved['generation'])

    def test_eof_missing_and_no_progress_stop_without_looping(self):
        missing=self.targets[0].replace('frame_1','frame_999')
        result=self.execute(targets=[missing]);self.assertFalse(result['targets_complete'])
        self.assertLessEqual(len(result['persistent_index']['scans']),6)
        self.assertTrue(result['persistent_index']['after']['catalog_complete']);self.assertFalse(result['persistent_index']['after']['complete'])
        opts={**self.options,'workspace':str(self.root/'other'),'snapshot_directory':str(self.root/'other-snap')}
        def no_progress(index,*args,**kwargs):return {**index.status(),'new_members':0,'stop_reason':'no_progress'}
        with patch.object(PreparationIndex,'scan',no_progress):stopped=self.execute('no-progress',options=opts)
        self.assertEqual(stopped['persistent_index']['stop_reason'],'index_no_progress')
        self.assertEqual(len(stopped['persistent_index']['scans']),1)

    def test_opt_in_profile_cache_mode_and_cli_contracts(self):
        for extra in ({'profile':'unknown'},{'cache_input':{}},{'mode':'scan'}):
            with self.assertRaises(PreparationError):self.execute(**extra)
        with self.assertRaises(PreparationError):prepare(self.manifest,self.root/'no-target',self.targets,self.limits,persistent_index=self.options)
        opts=self.root/'options.json';opts.write_bytes(canonical(self.options))
        limits=self.root/'limits.json';limits.write_bytes(canonical(asdict(self.limits)))
        args=['extract','--manifest',str(self.base/'pose_manifest.json'),'--limits',str(limits),'--output',str(self.root/'cli'),
              '--persistent-index',str(opts)]
        for target in self.targets:args+=['--include',target,'--exact-target',target]
        with contextlib.redirect_stdout(io.StringIO()):self.assertEqual(main(args),0)
        verify_extraction(self.root/'cli',self.manifest)

    def test_backend_hash_validation_and_explicit_allow_code_change(self):
        self.execute()
        with patch('rewind_archive_index.preparation.backend_hashes',return_value={'changed':'0'*64}):
            with self.assertRaisesRegex(PreparationError,'persistent_backend_code_changed'):verify_extraction(self.root/'prepared',self.manifest)
            verify_extraction(self.root/'prepared',self.manifest,allow_code_change=True)
            with self.assertRaises(PreparationError):self.execute()

    def test_head_failure_preserves_previous_and_refuses_unpublished_local_branch(self):
        import rewind_archive_index.preparation as module
        original=module.atomic_record
        def fail(path,value,**kwargs):
            if str(path).endswith('-head.json') and len(value.get('receipts',[]))>1:raise OSError('injected')
            return original(path,value,**kwargs)
        with patch.object(module,'atomic_record',fail):result=self.execute()
        self.assertEqual(result['persistent_index']['status'],'failed')
        previous=json.loads(next((self.root/'snapshots').glob('*-head.json')).read_bytes())
        self.assertEqual(len(previous['receipts']),1)
        self.assertGreater(len(list((self.root/'snapshots').glob('g*.zip'))),1)
        retry=self.execute('retry');self.assertEqual(retry['persistent_index']['stop_reason'],'prepare_local_anchor_conflict')
        self.assertEqual(json.loads(next((self.root/'snapshots').glob('*-head.json')).read_bytes()),previous)

    def test_original_interruption_survives_cleanup_failures(self):
        import rewind_archive_index.preparation as module
        original=module.atomic_record;interrupted=KeyboardInterrupt('secret')
        def save(path,value,**kwargs):
            if value.get('status')=='interrupted':raise SystemExit('cleanup-secret')
            return original(path,value,**kwargs)
        with patch.object(PreparationIndex,'scan',side_effect=interrupted),patch.object(module,'atomic_record',save):
            with self.assertRaises(KeyboardInterrupt) as caught:self.execute()
        self.assertIs(caught.exception,interrupted)
        self.assertTrue(list((self.root/'snapshots').glob('g*.zip')))
        self.assertFalse(list((self.root/'snapshots').glob('*-writer.lock')))

    def test_adapter_enters_existing_converter_and_keeps_training_blocked(self):
        import test_prepare as fixture_module
        case=fixture_module.PreparationTests();case.setUp();self.addCleanup(case.doCleanups)
        entries=[(p.relative_to(case.raw).as_posix(),p.read_bytes()) for p in case.raw.rglob('*.json')]
        self.manifest=local_manifest(self.base/'converter',fixture(entries),'pose');self.base=self.base/'converter'
        targets=[name for name,data in entries];self.execute(targets=targets)
        from rewind_prepare.converter import convert
        candidate,report=convert(self.root/'prepared/files',self.root/'prepared/extraction_manifest.json',case.splitdir,case.motion,case.specpath,case.calibration)
        self.assertEqual(candidate['contract']['training_release'],'not_implemented')

    def test_source_and_head_tampering_and_writer_guard(self):
        self.execute()
        head=next((self.root/'snapshots').glob('*-head.json'));original=head.read_bytes();value=json.loads(original)
        value['source_sha256']='0'*64;head.write_bytes(canonical(value))
        failed=self.execute('bad-head');self.assertFalse(failed['inspection_candidates_usable'])
        head.write_bytes(original)
        lock=head.with_name(head.name.replace('-head.json','-writer.lock'));lock.write_bytes(b'locked')
        with self.assertRaisesRegex(IndexFailure,'prepare_writer_active'):self.execute('locked')
        lock.unlink()
        part=self.base/self.manifest['parts'][0]['source']['path'];part.write_bytes(part.read_bytes()+b'x')
        with self.assertRaisesRegex(IndexFailure,'prepare_local_identity_changed'):self.execute('changed')

    def test_drive_shared_context_across_scan_restore_extract_and_reuse(self):
        import test_persistent_index_extension as fixture_module
        case=fixture_module.ExtensionTests();case.setUpClass();case.setUp();self.addCleanup(case.doCleanups)
        folder,index,anchor=case.baseline(1048576);before=index._snapshot()
        self.manifest=case.manifest;self.limits=replace(self.limits,max_elapsed_seconds=300.,max_ranges=120)
        state=index._snapshot()[0];targets=[state['members'][1]['name'],'sequence/json/frame_24.json']
        opts={**self.options,'batch_members':16,'spacing':1048576,'anchor':anchor['receipt'],'input_snapshot_directory':str(folder/'B-input')}
        context=PreparationContext(self.limits,case.session)
        first=self.execute(targets=targets,options=opts,persistent_context=context)
        self.assertTrue(first['targets_complete'],first['persistent_index'].get('stop_reason'))
        runtime=context.runtime;body=runtime.transfer.transfer_bytes;read=context.resources.read
        runtime.cache.clear();runtime.cache_bytes=0
        second=self.execute('fresh-drive',targets=targets,options={**opts,'workspace':str(self.root/'fresh-work')},persistent_context=context)
        self.assertTrue(second['targets_complete']);self.assertIs(runtime,context.runtime)
        self.assertGreater(runtime.transfer.transfer_bytes,body);self.assertGreater(context.resources.read,read)
        self.assertEqual(second['persistent_index']['scans'],[])
        self.assertLessEqual(runtime.transfer.transfer_bytes,8*1024**2);self.assertLessEqual(runtime.transfer.ranges,120)
        self.assertLessEqual(context.resources.read,32*1024**2);self.assertLessEqual(context.resources.decoded,32*1024**2)
        reused=self.execute(targets=targets,options=opts,persistent_context=context)
        self.assertTrue(reused['reuse']['verified']);self.assertEqual(index._snapshot(),before)

    def test_shared_resource_exhaustion_blocks_next_scan_and_preserves_head(self):
        context=PreparationContext(self.limits)
        first=self.execute(options={**self.options,'max_batches':1},persistent_context=context)
        self.assertFalse(first['targets_complete']);context.resources.read=32*1024**2
        report=self.execute('exhausted',persistent_context=context)
        self.assertEqual(report['persistent_index']['stop_reason'],'prepare_read_limit')
        self.assertFalse(report['targets_complete']);self.assertEqual(report['persistent_index']['scans'],[])

    def test_caller_member_disk_file_path_and_metadata_limits_are_not_relaxed(self):
        self.limits=replace(self.limits,max_members=1)
        report=self.execute();self.assertFalse(report['targets_complete'])
        self.assertEqual(report['persistent_index']['resources']['scanned_members'],1)
        self.assertEqual(report['persistent_index']['stop_reason'],'prepare_members_limit')
        self.assertEqual(report['persistent_index']['failure_stage'],'scan_post')
        self.limits=replace(Limits(),max_disk_bytes=1024*1024)
        with self.assertRaisesRegex(IndexFailure,'prepare_disk_limit'):self.execute('tiny-disk')
        self.limits=replace(Limits(),max_file_bytes=1)
        failed=self.execute('tiny-file',options={**self.options,'workspace':str(self.root/'tiny-file-work'),
            'snapshot_directory':str(self.root/'tiny-file-snap')});self.assertFalse(failed['inspection_candidates_usable'])
        self.assertEqual(failed['persistent_index']['stop_reason'],'file_bytes_limit')
        self.limits=replace(Limits(),max_metadata_bytes=1)
        with self.assertRaisesRegex(IndexFailure,'prepare_metadata_limit_unsupported'):self.execute('tiny-meta')


if __name__=='__main__':unittest.main()
