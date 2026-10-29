"""실제 인증/게시 없이 고정 A/B phase와 marker·합산 예산을 확인한다."""
import json
import importlib.util
import os
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch
import unittest
import test_persistent_drive_index as fixtures
from rewind_archive_index import IndexFailure
from rewind_archive_index import pilot


@unittest.skipUnless(importlib.util.find_spec('indexed_gzip') and importlib.util.find_spec('psutil'),'install requirements-index.txt')
class PilotTests(unittest.TestCase):
    setUp=fixtures.PersistentDriveTests.setUp

    def manifest(self):
        return {'schema_version':'archive_manifest_v1','archive_id':'remote','dataset_version':'synthetic_v1','kind':'pose',
            'source_scope':'user_drive','total_parts':2,'parts':[{'index':i,'name':f'pose.tar.gz{i:03d}','size_bytes':p['size'],
                'version':p['version'],'source':{'type':'drive','file_id':p['file_id']},
                'identity':{'version':p['version'],'modified_time':p['modified_time']}} for i,p in enumerate(self.spec['parts'])]}

    def test_failure_marker_blocks_auth_and_automatic_retry(self):
        calls=[]
        def fail():calls.append(1);raise RuntimeError('secret must not be logged')
        args={'phase':'A','run_id':'once','workspace':self.root/'runtime','snapshot_directory':self.root/'mount',
            'manifest':self.manifest(),'session_factory':fail}
        report=pilot.run_phase(**args);self.assertEqual(report['status'],'failed')
        with self.assertRaisesRegex(IndexFailure,'pilot_run_already_consumed'):pilot.run_phase(**args)
        self.assertEqual(len(calls),1)
        self.assertNotIn('secret',json.dumps(report))

    def test_ab_phase_json_hash_snapshot_parent_and_single_runtime(self):
        # 같은 실제 gzip/tar fixture 생성기로 JSON payload를 준비한다.
        from test_persistent_archive_index import fixture
        from test_drive_metadata import FakeDrive
        from types import SimpleNamespace
        data=fixture([(f'frame_{i}.json',b'{"x":1}') for i in range(40)])
        chunks=[data[:len(data)//2],data[len(data)//2:]]
        for i,b in enumerate(chunks):self.session.data[f'fake_{i}']=b;self.session.docs[f'fake_{i}']['size']=str(len(b));self.spec['parts'][i]['size']=len(b)
        self.spec['total_bytes']=len(data);manifest=self.manifest()
        self.session.close=lambda:None
        original=pilot.DriveRuntime
        with patch.object(pilot,'DriveRuntime',wraps=original) as runtime:
            a=pilot.run_phase(phase='A',run_id='pilot',workspace=self.root/'a',snapshot_directory=self.root/'mount',manifest=manifest,session_factory=lambda:self.session)
            self.assertEqual(a['status'],'passed',a);self.assertEqual(runtime.call_count,1)
        anchor=json.loads(__import__('pathlib').Path(a['anchor_path']).read_text())
        with self.assertRaisesRegex(IndexFailure,'pilot_fresh_runtime_required'):
            pilot.run_phase(phase='B',run_id='pilot',workspace=self.root/'b',snapshot_directory=self.root/'mount',manifest=manifest,session_factory=lambda:self.session,anchor=anchor)
        with patch.object(pilot,'RUNTIME_ID','new-process-test'):
            b=pilot.run_phase(phase='B',run_id='pilot',workspace=self.root/'b',snapshot_directory=self.root/'mount',manifest=manifest,session_factory=lambda:self.session,anchor=anchor)
        self.assertEqual(b['status'],'passed',b)
        self.assertEqual(b['snapshot']['receipt']['parent_snapshot_id'],a['snapshot']['receipt']['snapshot_id'])
        self.assertEqual(b['selected'],a['selected'])
        for report in (a,b):
            self.assertLessEqual(report['transport']['phase_response_body_bytes'],8*1024**2)
            self.assertLessEqual(report['transport']['phase_http_requests'],120)
            self.assertFalse(report['full_archive_crc_requested'])

    def test_actual_new_process_b_restores_scans_and_publishes_child(self):
        from test_persistent_archive_index import fixture
        from rewind_archive_index import pilot as module
        data=fixture([(f'frame_{i}.json',b'{"pose":[1,2,3]}') for i in range(40)])
        chunks=[data[:len(data)//2],data[len(data)//2:]]
        for i,b in enumerate(chunks):
            self.session.data[f'fake_{i}']=b;self.session.docs[f'fake_{i}']['size']=str(len(b));self.spec['parts'][i]['size']=len(b)
        self.spec['total_bytes']=len(data);self.session.close=lambda:None
        manifest=self.manifest();mount=self.root/'mount'
        a=module.run_phase(phase='A',run_id='real-process',workspace=self.root/'a',snapshot_directory=mount,manifest=manifest,session_factory=lambda:self.session)
        self.assertEqual(a['status'],'passed',a)
        anchor=json.loads(Path(a['anchor_path']).read_text())
        conf={'manifest':manifest,'anchor':anchor,'data':{k:v.hex() for k,v in self.session.data.items()},'mount':str(mount),'workspace':str(self.root/'b')}
        cfg=self.root/'child.json';cfg.write_text(json.dumps(conf))
        script='''import json,sys
from pathlib import Path
from types import SimpleNamespace
from test_drive_metadata import FakeDrive
from rewind_archive_index.pilot import run_phase
c=json.loads(Path(sys.argv[1]).read_text());selection={'parts':[{'index':p['index'],'name':p['name'],'file_id':p['source']['file_id']} for p in c['manifest']['parts']]}
s=FakeDrive(selection,{k:bytes.fromhex(v) for k,v in c['data'].items()});s.get_adapter=lambda _:SimpleNamespace(max_retries=SimpleNamespace(total=0));s.close=lambda:None
r=run_phase(phase='B',run_id='real-process',workspace=c['workspace'],snapshot_directory=c['mount'],manifest=c['manifest'],session_factory=lambda:s,anchor=c['anchor']);print(json.dumps(r));sys.exit(r['status']!='passed')
'''
        env=dict(os.environ);env['PYTHONPATH']=os.pathsep.join(sys.path)
        child=subprocess.run([sys.executable,'-c',script,str(cfg)],env=env,capture_output=True,text=True)
        self.assertEqual(child.returncode,0,child.stdout+child.stderr)
        b=json.loads(child.stdout);self.assertEqual(b['after']['members'],32)
        self.assertEqual(b['selected'],a['selected'])
        self.assertEqual(b['snapshot']['receipt']['parent_snapshot_id'],a['snapshot']['receipt']['snapshot_id'])
