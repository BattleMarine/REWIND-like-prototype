"""작은 다중 RGB/pose archive의 산출×2 계획·영속 재개·과거 추출·실패 예약."""
import copy,gzip,hashlib,io,json,random,tarfile,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
from rewind_archive.contracts import canonical
from rewind_archive_index.source import make_source,SplitSource,Budget,IndexLimits,IndexFailure
from rewind_fleet.planning import plan,seal
from rewind_fleet.runner import run,recover_pending
from rewind_fleet.storage import Store,sha
from rewind_fleet.scanner import extract
from rewind_fleet import window

ROOT=Path(__file__).resolve().parents[1]
class FleetTests(unittest.TestCase):
    def setUp(self):
        tmp=tempfile.TemporaryDirectory(dir=ROOT/'outputs/test_tmp');self.addCleanup(tmp.cleanup);self.base=Path(tmp.name)
        self.inv={'schema':'fleet_inventory_v1','archives':[],'official_registry_complete':True};self.sources={};self.payloads={};self.est={}
        for kind in ('rgb','pose'):
            data=io.BytesIO();wanted={};rng=random.Random(11)
            with tarfile.open(fileobj=data,mode='w',format=tarfile.PAX_FORMAT) as tar:
                for i in range(5):
                    name=f'{kind}/sequence/frame_{i}.bin';raw=rng.randbytes(4000);wanted[name]=raw
                    item=tarfile.TarInfo(name);item.size=len(raw);tar.addfile(item,io.BytesIO(raw))
            raw=data.getvalue();compressed=gzip.compress(raw,mtime=0);p=self.base/(kind+'.gz');p.write_bytes(compressed)
            a={'key':kind,'kind':kind,'parts':[{'index':0,'name':kind+'.gz000','file_id':kind,'size':len(compressed),'version':'1','modified_time':'2026-11-03T00:00:00Z'}],
                'runtime_versions_verified':True,'official_part_count':1,'classification':'observed_contiguous_official_unknown'}
            self.inv['archives'].append(a);self.sources[kind]=make_source([p],archive_id=kind,kind=kind,dataset_version='fixture')
            self.payloads[kind]=wanted
            self.est[kind]={'decoded_bytes':len(raw),'members':5,'source_bytes_per_second':5*1024**2,'http_requests_per_second':1,
                'artifact_bytes_per_second':20*1024**2,'python_base_rss_bytes':256*1024**2,'replay_fraction':1,
                'catalog_bytes_per_member':1024,'seconds_per_segment':.1,'failure_residue_bytes':8*1024**2,
                'basis':'exact small fixture bytes; conservative one replay and scalar throughput assumptions'}
        self.inv=seal(self.inv);self.plan=plan(self.inv,self.est,['rgb','pose'],verification='full_bounded',storage_backend='local')
    def execute(self,**kwargs):
        return run(self.inv,self.plan,self.base/'store',kwargs.pop('workspace',self.base/'local'),owner='test',single_writer=True,
            local_sources=self.sources,**kwargs)
    def test_computed_times2_multiple_archives_resume_and_old_extract(self):
        first=self.execute(max_segments=1);self.assertFalse(first['complete']);self.assertIsNone(first['pending'])
        result=self.execute(max_segments=16,workspace=self.base/'new_runtime')
        self.assertTrue(result['complete'],result)
        self.assertTrue(all(result['reconciled_usage'][k]<=self.plan['cap'][k] for k in result['usage']))
        store=Store(self.base/'store');ledger=store.committed(store.head())
        for kind in ('rgb','pose'):
            for name,expected in self.payloads[kind].items():
                b=Budget(IndexLimits(max_read_bytes=4*1024**2,max_decoded_bytes=4*1024**2))
                got=extract(store,ledger['archives'][kind],self.base/'lookup',sha(canonical(self.sources[kind])),name,SplitSource(self.sources[kind],b),b,1024**2)
                self.assertEqual(got,expected)
    def test_resealed_cap_change_and_unknown_plan_blocked(self):
        bad=copy.deepcopy(self.plan);bad.pop('content_sha256');bad['cap']['source_bytes']*=100;bad=seal(bad)
        with self.assertRaisesRegex(IndexFailure,'recalculation'):run(self.inv,bad,self.base/'bad',self.base/'local',owner='test',single_writer=True)
        unknown=plan(self.inv,{},['rgb'])
        with self.assertRaisesRegex(IndexFailure,'incomplete'):run(self.inv,unknown,self.base/'unknown',self.base/'local',owner='test',single_writer=True)
    def test_hardloss_reservation_is_not_reset_and_explicit_recovery(self):
        with patch('rewind_fleet.runner.scan',side_effect=KeyboardInterrupt()):result=self.execute(max_segments=1)
        self.assertIsNotNone(result['pending']);used=dict(result['usage'])
        with self.assertRaisesRegex(IndexFailure,'pending'):self.execute(max_segments=1)
        store=Store(self.base/'store');head=store.head();token=result['pending']['token']
        with self.assertRaises(IndexFailure):recover_pending(store,head,token)
        key=recover_pending(store,head,token,owner_confirmed_stopped=True)
        self.assertEqual(store.committed(key)['usage'],used);self.assertIsNone(store.committed(key)['pending'])
    def test_source_order_identity_change_keeps_last_catalog(self):
        first=self.execute(max_segments=1);head=first['head'];store=Store(self.base/'store');archive=store.committed(head)['archives']['rgb']
        changed=copy.deepcopy(self.sources);changed['rgb']['archive_id']='other';self.sources=changed
        with self.assertRaisesRegex(IndexFailure,'fleet_source_identity'):self.execute(max_segments=1)
        self.assertEqual(store.head(),head);self.assertEqual(store.committed(store.head())['archives']['rgb'],archive)

    def test_mock_drive_multiple_archives_uses_global_budget(self):
        from types import SimpleNamespace
        from test_drive_metadata import FakeDrive
        inv=copy.deepcopy(self.inv);inv.pop('content_sha256');bodies={};selection={'parts':[]}
        for a in inv['archives']:
            p=a['parts'][0];p.update(version='9',modified_time='2026-10-14T00:00:00.000Z')
            bodies[p['file_id']]=Path(self.sources[a['key']]['parts'][0]['path']).read_bytes()
            selection['parts'].append({'file_id':p['file_id'],'name':p['name']})
        inv=seal(inv);cost=plan(inv,self.est,['rgb','pose'],verification='full_bounded',storage_backend='local')
        session=FakeDrive(selection,bodies);session.get_adapter=lambda _:SimpleNamespace(max_retries=SimpleNamespace(total=0));session.close=lambda:None
        result=run(inv,cost,self.base/'drive-store',self.base/'drive-local',owner='drive',single_writer=True,max_segments=16,session_factory=lambda:session)
        self.assertTrue(result['complete'],result);self.assertGreater(result['reconciled_usage']['requests'],0)
        self.assertLessEqual(result['reconciled_usage']['requests'],cost['cap']['requests']);self.assertGreater(result['reconciled_usage']['source_bytes'],0)

    def test_catalog_only_never_starts_automatic_full_crc_pass(self):
        self.plan=plan(self.inv,self.est,['rgb','pose'],verification='catalog_only',storage_backend='local')
        result=self.execute(max_segments=16)
        self.assertFalse(result['complete']);self.assertFalse(result['dataset_complete'])
        self.assertEqual(len(result['events']),2)
        self.assertTrue(all(e['state']['trailer_crc_match'] and not e['state']['complete'] for e in result['events']))

    def test_truncated_trailer_keeps_unknown_reservation_and_no_complete(self):
        p=Path(self.sources['rgb']['parts'][0]['path']);p.write_bytes(p.read_bytes()[:-10])
        self.sources['rgb']=make_source([p],archive_id='rgb',kind='rgb',dataset_version='fixture')
        result=self.execute(max_segments=1)
        self.assertEqual(result['stop_reason'],'fleet_trailer_crc_or_size');self.assertFalse(result['complete'])
        self.assertIsNotNone(result['pending'])
