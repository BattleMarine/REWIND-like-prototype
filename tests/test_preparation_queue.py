"""로컬 합성 분할gzip queue. 원격/실제 데이터 학습/자동 삭제 없음."""
import contextlib
import copy
from dataclasses import replace
import gzip
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from rewind_archive.contracts import Limits, canonical, digest
from rewind_archive.fixture import SEQUENCE, local_manifest, png_bytes, tar_bytes
from rewind_data.window_fixture import synthetic_frame
from rewind_archive_index.preparation import _source, prepare_indexed
from rewind_prepare_queue.core import (MAXIMUM, _state, hashed, initialize_queue, make_plan,
    mark_consumed, read, run_queue, validate_plan, writer)
from rewind_prepare_queue.__main__ import main

ROOT = Path(__file__).resolve().parents[1]


@unittest.skipUnless(importlib.util.find_spec('indexed_gzip') and importlib.util.find_spec('psutil'),
                     '기존 requirements-index 환경 필요')
class PreparationQueueTests(unittest.TestCase):
    def setUp(self):
        parent = ROOT/'outputs'/'test_tmp'
        parent.mkdir(parents=True,exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=parent)
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        jobs = []
        for side in ('left','right','pose'):
            folder = 'raw_'+side
            kind = 'pose' if side == 'pose' else 'rgb'
            entries = []
            for frame in (10,0,2,1):
                member = (f'{SEQUENCE}/json/frame_{frame}.json' if kind == 'pose' else
                          f'{SEQUENCE}/fisheye_final_image/camera_{side}/final_{frame}.png')
                payload = canonical(synthetic_frame(SEQUENCE,frame).raw_document) if kind == 'pose' else png_bytes()
                entries.append((member,payload))
            manifest = local_manifest(self.base/folder,gzip.compress(tar_bytes(entries),mtime=0),kind)
            manifest['archive_id'] = 'fixture_'+side
            target = next(name for name,_ in entries if name.endswith('frame_1.json' if kind == 'pose' else 'final_1.png'))
            jobs.append({'id':side,'manifest':manifest,'source_base':folder,
                         'source_sha256':digest(_source(manifest,self.base/folder)), 'targets':[target],
                         'index_options':{'batch_members':1,'max_batches':8,'spacing':65536}})
        self.plan = make_plan(jobs)
        self.root = self.base/'queue'
        initialize_queue(self.plan,self.root)

    def test_actual_prepare_order_resume_and_hash_reuse(self):
        state = run_queue(self.plan,self.root,base=self.base,max_jobs=1)
        self.assertEqual((state['cursor'],state['status']),(1,'bounded_job_stop'))
        first = copy.deepcopy(state)
        with patch('rewind_prepare_queue.core.prepare_indexed',wraps=prepare_indexed) as prepare:
            state = run_queue(self.plan,self.root,base=self.base)
            self.assertEqual([c.args[0]['archive_id'] for c in prepare.call_args_list],['fixture_right','fixture_pose'])
        self.assertEqual(state['cursor'],3)
        self.assertEqual(state['matching']['matched_triplet_count'],1)
        self.assertEqual(state['status'],'prepared_not_training_release')
        self.assertFalse(state['train_ready'])
        self.assertEqual(state['jobs'][0]['output'],first['jobs'][0]['output'])
        self.assertEqual(state['charged']['http_requests'],0)
        self.assertEqual(state['charged']['body_bytes'],0)
        self.assertEqual(state['charged'],state['observed'])
        before = copy.deepcopy(state['charged'])
        with patch('rewind_prepare_queue.core.prepare_indexed',side_effect=AssertionError('no source reread')):
            reused = run_queue(self.plan,self.root,base=self.base)
        for key in ('read_bytes','decoded_bytes','members','http_requests','body_bytes'):
            self.assertEqual(reused['charged'][key],before[key])
        self.assertEqual(reused['content_sha256'],read(self.root/'state.json')['content_sha256'])

    def test_partial_retry_new_output_resumes_same_index(self):
        plan = copy.deepcopy(self.plan)
        plan['jobs'][0]['index_options']['max_batches'] = 1
        plan = hashed(plan)
        root = self.base/'partial'
        initialize_queue(plan,root)
        state = run_queue(plan,root,base=self.base)
        self.assertEqual(state['cursor'],0)
        self.assertEqual(state['jobs'][0]['status'],'partial')
        first_output = root/state['jobs'][0]['attempts'][0]['output']
        original = (first_output/'extraction_manifest.json').read_bytes()
        refused = run_queue(plan,root,base=self.base)
        self.assertEqual(refused['status'],'explicit_retry_required')
        for key in MAXIMUM:
            if key != 'elapsed_seconds': self.assertEqual(refused['charged'][key],state['charged'][key])
        self.assertGreaterEqual(refused['charged']['elapsed_seconds'],state['charged']['elapsed_seconds'])
        self.assertLessEqual(refused['charged']['elapsed_seconds'],plan['budgets']['elapsed_seconds'])
        for _ in range(4):
            state = run_queue(plan,root,base=self.base,retry=True,max_jobs=1)
            if state['cursor']: break
        self.assertEqual(state['cursor'],1)
        self.assertEqual((first_output/'extraction_manifest.json').read_bytes(),original)
        self.assertNotEqual(state['jobs'][0]['output'],state['jobs'][0]['attempts'][0]['output'])
        latest = read(root/state['jobs'][0]['output']/'extraction_manifest.json')
        self.assertGreater(latest['persistent_index']['before']['generation'],1)

    def test_actual_backend_budget_is_reduced_and_cumulative(self):
        plan = copy.deepcopy(self.plan)
        plan['budgets'].update(read_bytes=1024,decoded_bytes=1024,members=1)
        plan = hashed(plan)
        root = self.base/'bounded'
        initialize_queue(plan,root)
        def checked(*args,**kwargs):
            context = kwargs['context']
            self.assertLessEqual(context.resources.index_limits.max_read_bytes,1024)
            self.assertLessEqual(context.resources.index_limits.max_decoded_bytes,1024)
            self.assertLessEqual(args[3].max_members,1)
            return prepare_indexed(*args,**kwargs)
        with patch('rewind_prepare_queue.core.prepare_indexed',checked):
            state = run_queue(plan,root,base=self.base)
        self.assertLessEqual(state['charged']['read_bytes'],1024)
        self.assertLessEqual(state['charged']['decoded_bytes'],1024)
        self.assertLessEqual(state['charged']['members'],1)
        again = run_queue(plan,root,base=self.base,retry=True)
        self.assertIn(again['status'],('budget_exhausted','unknown_cost_blocked'))

    def test_python_failure_keeps_full_reservation_and_no_retry(self):
        with patch('rewind_prepare_queue.core.prepare_indexed',side_effect=OSError('unknown_partial_io')):
            state = run_queue(self.plan,self.root,base=self.base)
        self.assertEqual(state['status'],'unknown_cost_blocked')
        self.assertEqual(state['cursor'],0)
        self.assertIsNotNone(state['inflight'])
        self.assertEqual(state['charged']['read_bytes'],MAXIMUM['read_bytes'])
        with patch('rewind_prepare_queue.core.prepare_indexed',side_effect=AssertionError('must not retry')):
            self.assertEqual(run_queue(self.plan,self.root,base=self.base,retry=True)['charged'],state['charged'])

    def test_subprocess_crash_keeps_reservation_after_process_restart(self):
        plan_path = self.base/'plan.json'
        plan_path.write_bytes(canonical(self.plan))
        deps = ROOT/'outputs/rgb_index_probe_20261029/deps'
        code = "import os,sys; from pathlib import Path; sys.path[:0]=[sys.argv[1],sys.argv[2]]; import rewind_prepare_queue.core as c; c.prepare_indexed=lambda *a,**k: os._exit(9); c.run_queue(c.read(sys.argv[3]),Path(sys.argv[4]),base=Path(sys.argv[5]))"
        result = subprocess.run([sys.executable,'-X','utf8','-c',code,str(ROOT),str(deps),str(plan_path),
                                 str(self.root),str(self.base)],cwd=ROOT,capture_output=True,text=True)
        self.assertEqual(result.returncode,9,result.stderr)
        with patch('rewind_prepare_queue.core.prepare_indexed',side_effect=AssertionError('crash retry prohibited')):
            state = run_queue(self.plan,self.root,base=self.base,retry=True)
        self.assertEqual(state['status'],'unknown_cost_blocked')
        self.assertEqual(state['next_attempt'],2)
        self.assertEqual(state['charged']['read_bytes'],MAXIMUM['read_bytes'])

    def test_normal_new_process_restores_queue_and_finishes(self):
        run_queue(self.plan,self.root,base=self.base,max_jobs=1)
        plan_path = self.base/'normal_plan.json'; plan_path.write_bytes(canonical(self.plan))
        deps = ROOT/'outputs/rgb_index_probe_20261029/deps'
        code = "import sys; from pathlib import Path; sys.path[:0]=[sys.argv[1],sys.argv[2]]; import rewind_prepare_queue.core as c; s=c.run_queue(c.read(sys.argv[3]),Path(sys.argv[4]),base=Path(sys.argv[5])); assert s['cursor']==3 and not s['train_ready']; print(s['status'])"
        result = subprocess.run([sys.executable,'-X','utf8','-c',code,str(ROOT),str(deps),str(plan_path),
                                 str(self.root),str(self.base)],cwd=ROOT,capture_output=True,text=True)
        self.assertEqual(result.returncode,0,result.stderr)
        self.assertEqual(result.stdout.strip(),'prepared_not_training_release')
        self.assertEqual(read(self.root/'state.json')['cursor'],3)

    def test_drive_budget_injection_without_remote_request(self):
        plan = copy.deepcopy(self.plan)
        for j,job in enumerate(plan['jobs']):
            manifest = job['manifest']; manifest['source_scope'] = 'user_drive'
            for i,part in enumerate(manifest['parts']):
                part['source'] = {'type':'drive','file_id':f'fixture_{j}_{i}'}
                part['identity'] = {'version':'1','modified_time':'fixture_time'}
            job['source_sha256'] = digest(_source(manifest,self.base/job['source_base']))
        plan['budgets'].update(read_bytes=2048,decoded_bytes=2048,members=2,http_requests=7,body_bytes=4096)
        plan = hashed(plan)
        root = self.base/'mock_drive'; initialize_queue(plan,root)
        def inspect_limits(*args,**kwargs):
            context = kwargs['context']; limits = args[3]
            self.assertEqual((limits.max_ranges,limits.max_transfer_bytes),(7,4096))
            self.assertEqual((context.resources.index_limits.max_read_bytes,
                              context.resources.index_limits.max_decoded_bytes),(2048,2048))
            context.bind(_source(args[0],kwargs['base']))
            self.assertEqual((context.runtime.limits.max_requests,context.runtime.limits.max_body_bytes),(7,4096))
            raise OSError('injected_before_any_remote_request')
        with patch('rewind_prepare_queue.core.prepare_indexed',inspect_limits):
            state = run_queue(plan,root,base=self.base,authorized_session=object())
        self.assertEqual(state['status'],'unknown_cost_blocked')
        self.assertEqual((state['charged']['http_requests'],state['charged']['body_bytes']),(7,4096))

    def test_completed_payload_mutation_blocks_without_advancing(self):
        state = run_queue(self.plan,self.root,base=self.base,max_jobs=1)
        report = read(self.root/state['jobs'][0]['output']/'extraction_manifest.json')
        payload = self.root/state['jobs'][0]['output']/report['files'][0]['relative_path']
        payload.write_bytes(b'changed')
        with self.assertRaises(ValueError): run_queue(self.plan,self.root,base=self.base)
        state = read(self.root/'state.json')
        self.assertEqual(state['cursor'],1)
        self.assertEqual(state['status'],'unknown_cost_blocked')

    def test_source_mutation_plan_change_and_state_loss_do_not_reset(self):
        root = self.base/'lost'
        initialize_queue(self.plan,root)
        (root/'state.json').rename(root/'state_preserved.json')
        with self.assertRaisesRegex(ValueError,'queue_state_missing_no_automatic_reset'):
            run_queue(self.plan,root,base=self.base)
        self.assertFalse((root/'state.json').exists())
        changed = copy.deepcopy(self.plan)
        changed['jobs'].reverse()
        with self.assertRaisesRegex(ValueError,'queue_pinned_plan_changed'):
            run_queue(hashed(changed),self.root,base=self.base)
        part = self.base/self.plan['jobs'][0]['source_base']/self.plan['jobs'][0]['manifest']['parts'][0]['source']['path']
        part.write_bytes(part.read_bytes()+b'x')
        with self.assertRaises(ValueError): run_queue(self.plan,self.root,base=self.base)
        self.assertEqual(read(self.root/'state.json')['cursor'],0)

    def test_plan_rejects_duplicate_missing_keys_version_and_widening(self):
        for kind in ('duplicate','missing','version','budget','backend'):
            plan = copy.deepcopy(self.plan)
            if kind == 'duplicate': plan['jobs'][1]['targets'] = plan['jobs'][0]['targets']
            if kind == 'missing': plan['jobs'].pop()
            if kind == 'version': plan['jobs'][0]['manifest']['dataset_version'] = 'other'
            if kind == 'budget': plan['budgets']['read_bytes'] += 1
            if kind == 'backend': plan['backend_limits']['max_decompressed_bytes'] += 1
            with self.assertRaises(ValueError): validate_plan(hashed(plan))

    def test_disk_refusal_and_consumption_preserve_sources_outputs(self):
        state = run_queue(self.plan,self.root,base=self.base)
        before = {p: p.read_bytes() for p in self.root.rglob('*') if p.is_file() and 'files' in p.parts}
        marked = mark_consumed(self.plan,self.root,'left','a'*64)
        self.assertEqual(marked['jobs'][0]['consumed'],'a'*64)
        with self.assertRaisesRegex(ValueError,'queue_consumer_conflict'):
            mark_consumed(self.plan,self.root,'left','b'*64)
        self.assertTrue(all(p.read_bytes()==raw for p,raw in before.items()))
        plan = copy.deepcopy(self.plan); plan['max_disk_bytes'] = 4*1024**2; plan = hashed(plan)
        root = self.base/'disk'
        initialize_queue(plan,root)
        (root/'preserved.bin').write_bytes(b'x'*(3*1024**2))
        with self.assertRaisesRegex(ValueError,'queue_disk_exhausted_preserved'):
            run_queue(plan,root,base=self.base)
        self.assertTrue((root/'preserved.bin').exists())
        stopped = read(root/'state.json')
        self.assertEqual(stopped['cursor'],0)
        self.assertEqual(stopped['status'],'unknown_cost_blocked')
        self.assertEqual(stopped['failure']['reason'],'queue_disk_exhausted_preserved')

    def test_single_writer_and_explicit_init(self):
        with writer(self.root):
            with self.assertRaisesRegex(ValueError,'queue_writer_active'):
                run_queue(self.plan,self.root,base=self.base)
        with self.assertRaises(FileExistsError): initialize_queue(self.plan,self.root)
        with self.assertRaisesRegex(ValueError,'queue_initialize_required'):
            run_queue(self.plan,self.base/'not_initialized',base=self.base)

    def test_cli_plan_init_run_inspect_consumed(self):
        jobs = copy.deepcopy(self.plan['jobs'])
        for job in jobs: job.pop('source_sha256')
        jobs_path = self.base/'jobs.json'; jobs_path.write_bytes(canonical(jobs))
        plan_path = self.base/'cli_plan.json'; root = self.base/'cli_queue'
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(main(['plan','--jobs',str(jobs_path),'--base',str(self.base),'--output',str(plan_path)]),0)
            self.assertEqual(main(['init','--plan',str(plan_path),'--root',str(root)]),0)
            self.assertEqual(main(['run','--plan',str(plan_path),'--root',str(root),'--base',str(self.base)]),0)
            self.assertEqual(main(['inspect','--plan',str(plan_path),'--root',str(root)]),0)
            self.assertEqual(main(['consumed','--plan',str(plan_path),'--root',str(root),'--job','pose','--consumer-sha256','c'*64]),0)
        self.assertEqual(read(root/'state.json')['cursor'],3)


if __name__ == '__main__': unittest.main()
