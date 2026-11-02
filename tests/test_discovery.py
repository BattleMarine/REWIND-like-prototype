"""실제 원격 없이 source pin·누적 장부·snapshot 재개·중단 회수를 확인한다."""
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from rewind_archive.contracts import canonical,digest
from rewind_archive_index.core import ArchiveIndex
from rewind_discovery.runner import run,DiscoveryLimits
from rewind_discovery.recovery import recover
from test_drive_metadata import FakeDrive
from test_persistent_archive_index import fixture,png

ROOT=Path(__file__).resolve().parents[1]


@unittest.skipUnless(importlib.util.find_spec('indexed_gzip'),'기존 index 환경 필요')
class DiscoveryTests(unittest.TestCase):
    def setUp(self):
        parent=ROOT/'outputs/test_tmp';parent.mkdir(parents=True,exist_ok=True)
        self.tmp=tempfile.TemporaryDirectory(dir=parent);self.addCleanup(self.tmp.cleanup);self.base=Path(self.tmp.name)
        self.seqs=['Room/Day/person/glasses/001/Motion_A','Room/Day/person/glasses/002/Motion_B']
        entries=[]
        for seq in self.seqs:
            for frame in (2,0,3,1):
                for side in ('right','left'):
                    entries.append((f'{seq}/fisheye_final_image/camera_{side}/final_{frame}.png',png(frame)))
        self.data=fixture(entries)
        self.selection={'schema_version':'drive_part_selection_v1','archive_id':'discovery_fixture_rgb',
            'dataset_version':'synthetic_test_only','kind':'rgb','total_parts':1,
            'parts':[{'index':0,'name':'scene.tar.gz000','file_id':'fake_rgb'}]}
        self.expected=[{'index':0,'name':'scene.tar.gz000','file_id':'fake_rgb','size_bytes':len(self.data),
                        'modified_time':'2026-10-14T00:00:00.000Z'}]
        self.split_paths={};self.split_pins={}
        import hashlib
        for split,seq in zip(('train','validation','test'),[*self.seqs,'Other/Day/person/glasses/003/Motion_C']):
            path=self.base/(split+'.txt');path.write_bytes((seq+'\n').encode());self.split_paths[split]=path
            self.split_pins[split]=hashlib.sha256(path.read_bytes()).hexdigest()
        pose={'schema_version':'fixture_observation_only','seek_index':False,'full_source_listing_claimed':False,
            'truncated':False,'sequences':self.seqs,'entries':[{'kind':'pose','payload_complete':True,'sequence_ref':v,'frame':f}
                for v in (0,1) for f in (0,1,2,3)]}
        pose['content_sha256']=digest(pose);self.pose=self.base/'pose_catalog.json';self.pose.write_bytes(canonical(pose))
        self.pose_sha=hashlib.sha256(self.pose.read_bytes()).hexdigest()
        self.limits=DiscoveryLimits(body_bytes=4*1024**2,requests=256,read_bytes=8*1024**2,decoded_bytes=8*1024**2,
                                    seconds=60,reserve_seconds=5,batch_members=4,batches=2)
        self.sessions=[]

    def session(self):
        session=FakeDrive(self.selection,{'fake_rgb':self.data})
        session.get_adapter=lambda _:SimpleNamespace(max_retries=SimpleNamespace(total=0))
        session.close=lambda:None;self.sessions.append(session);return session

    def execute(self,name='job',**kwargs):
        args=dict(job_id=name,selection=self.selection,expected_metadata=self.expected,
            split_paths=self.split_paths,split_pins=self.split_pins,pose_catalog=self.pose,pose_sha256=self.pose_sha,
            workspace=self.base/name,snapshot_directory=self.base/(name+'_snapshots'),session_factory=self.session,limits=self.limits)
        args.update(kwargs);return run(**args)

    def test_incremental_snapshots_resume_cumulative_budget_and_keys(self):
        from rewind_discovery.runner import DriveRuntime
        runtimes=[]
        def factory(*a,**kw):
            obj=DriveRuntime(*a,**kw);runtimes.append(obj);return obj
        with patch('rewind_discovery.runner.DriveRuntime',factory):first=self.execute()
        self.assertEqual(first['status'],'completed_requested_batches');self.assertEqual(len(first['events']),2)
        self.assertEqual(first['events'][-1]['generation'],2);self.assertEqual(first['after']['members'],8)
        self.assertEqual(len(runtimes),1)
        metadata=json.loads((self.base/'job/metadata.json').read_bytes())['report']['stats']
        self.assertEqual(first['body_bytes'],metadata['transfer_bytes']+runtimes[0].transfer.transfer_bytes)
        self.assertEqual(first['requests'],metadata['requests']+runtimes[0].transfer.ranges)
        self.assertLessEqual(first['body_bytes'],self.limits.body_bytes)
        receipt_path=Path(first['last_receipt_path'])
        import hashlib
        anchor={'receipt_path':receipt_path,'receipt_file_sha256':hashlib.sha256(receipt_path.read_bytes()).hexdigest(),'anchor':first['last_receipt']}
        resumed=self.execute('resumed',resume=anchor)
        self.assertEqual(resumed['before']['members'],8);self.assertEqual(resumed['after']['members'],16)
        self.assertEqual(resumed['discovered']['keys']['stereo_pose_observed'],{'train':4,'validation':4,'test':0,'unknown':0})
        self.assertEqual(resumed['actual_extracted_triplets'],0)
        result=recover(self.base/'resumed',self.base/'resumed_snapshots',self.base/'recovery.zip',job_id='resumed',selection_sha256=digest(self.selection))
        self.assertEqual(result['last_receipt'],resumed['last_receipt'])
        with self.assertRaises(FileExistsError):self.execute('job')

    def test_interrupt_keeps_previous_receipt_and_recovers_without_session(self):
        original=ArchiveIndex.scan;calls=0
        def interrupt(index,*a,**kw):
            nonlocal calls
            calls+=1
            if calls==2:raise KeyboardInterrupt('secret token must not appear')
            return original(index,*a,**kw)
        with patch.object(ArchiveIndex,'scan',interrupt):
            with self.assertRaises(KeyboardInterrupt):self.execute()
        report=json.loads((self.base/'job/report.json').read_bytes())
        self.assertEqual(report['status'],'interrupted');self.assertIsNotNone(report['last_receipt'])
        self.assertNotIn('secret',json.dumps(report))
        result=recover(self.base/'job',self.base/'job_snapshots',self.base/'interrupt.zip',job_id='job',selection_sha256=digest(self.selection))
        self.assertFalse(result['network_requested'])
        missing=recover(self.base/'not_started',self.base/'absent_snapshots',self.base/'missing_report.zip',
                        job_id='not_started',selection_sha256=digest(self.selection))
        self.assertFalse(missing['report_present']);self.assertIsNone(missing['last_receipt'])

    def test_metadata_split_and_anchor_change_fail_without_media(self):
        expected=copy.deepcopy(self.expected);expected[0]['size_bytes']+=1
        report=self.execute(expected_metadata=expected)
        self.assertEqual(report['status'],'failed');self.assertEqual(report['failure']['reason'],'discovery_source_metadata_changed')
        self.assertFalse(any('alt=media' in u for u,_ in self.sessions[-1].calls))
        pins={**self.split_pins,'train':'0'*64};report=self.execute('split_bad',split_pins=pins)
        self.assertEqual(report['failure']['reason'],'discovery_split_pin')
        self.assertEqual(len(self.sessions),1)
        report=self.execute('anchor_bad',resume={'anchor':{'untrusted':True},'receipt_path':self.base/'missing','receipt_file_sha256':'0'*64})
        self.assertEqual(report['status'],'failed')

    def test_small_transfer_cap_never_resets_across_batches(self):
        from dataclasses import replace
        result=self.execute(limits=replace(self.limits,body_bytes=65536,batches=4))
        self.assertIn(result['status'],('failed','bounded_stop'))
        self.assertLessEqual(result['body_bytes'],65536)
        self.assertLessEqual(result['requests'],self.limits.requests)
