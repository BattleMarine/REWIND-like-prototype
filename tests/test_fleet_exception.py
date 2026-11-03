"""고정 calibration 한도의 실제 runner 정산·재개와 오류 경계."""
import gzip,hashlib,io,json,random,tarfile,tempfile,unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from rewind_archive.contracts import canonical
from rewind_archive_index.source import Budget,IndexLimits,IndexFailure,SplitSource,make_source
from rewind_archive_index.drive import DriveRuntime,DriveLimits
from rewind_fleet import scanner,runner
from rewind_fleet.planning import seal,calibration_plan,source
from rewind_fleet.storage import Store,sha
from test_drive_metadata import FakeDrive

MIB=1024**2
ROOT=Path(__file__).resolve().parents[1]

class FleetExceptionTests(unittest.TestCase):
    def setUp(self):
        parent=ROOT/'outputs/test_tmp';parent.mkdir(exist_ok=True,parents=True)
        tmp=tempfile.TemporaryDirectory(dir=parent);self.addCleanup(tmp.cleanup);self.base=Path(tmp.name)
        data=io.BytesIO();rng=random.Random(71);self.wanted={}
        with tarfile.open(fileobj=data,mode='w',format=tarfile.USTAR_FORMAT) as tar:
            for i in range(12):
                name=f'seq/rgb/frame_{i}.png';raw=rng.randbytes(400000);self.wanted[name]=raw
                item=tarfile.TarInfo(name);item.size=len(raw);tar.addfile(item,io.BytesIO(raw))
        self.compressed=gzip.compress(data.getvalue(),mtime=0);self.assertEqual(gzip.decompress(self.compressed),data.getvalue())
        chunks=[self.compressed[:1700003],self.compressed[1700003:]]
        self.bodies={f'p{i}':raw for i,raw in enumerate(chunks)}
        a={'key':'rgb','kind':'rgb','parts':[{'index':i,'file_id':f'p{i}','name':f'data.gz{i:03}',
            'size':len(raw),'version':'9','modified_time':'2026-10-14T00:00:00.000Z'} for i,raw in enumerate(chunks)],
            'runtime_versions_verified':True,'official_part_count':2,'classification':'observed_contiguous_official_unknown'}
        self.inv=seal({'schema':'fleet_inventory_v1','archives':[a],'official_registry_complete':True})
        self.plan=calibration_plan(self.inv,'rgb',storage_backend='local');self.spec=source(a)
        self.sessions=[]
    def session(self):
        fake=FakeDrive({'parts':[{'file_id':p['file_id'],'name':p['name']} for p in self.inv['archives'][0]['parts']]},self.bodies)
        fake.get_adapter=lambda _:SimpleNamespace(max_retries=SimpleNamespace(total=0));fake.close=lambda:None
        self.sessions.append(fake);return fake
    def execute(self,workspace='first'):
        return runner.run(self.inv,self.plan,self.base/'store',self.base/workspace,owner='exception_test',single_writer=True,
            max_segments=1,session_factory=self.session)
    def test_runner_partial_settlement_new_workspace_resume_old_extract(self):
        first=self.execute();self.assertIsNone(first['pending'],first)
        self.assertEqual(first['events'][0]['event']['reason'],'read_bytes_limit')
        self.assertEqual(first['events'][0]['state']['ordinal'],1)
        self.assertLess(first['reconciled_usage']['source_bytes'],first['usage']['source_bytes'])
        second=self.execute('new_workspace');self.assertIsNone(second['pending'],second)
        self.assertEqual(second['stop_reason'],'no_committed_progress')
        self.assertEqual(second['events'][0]['state']['ordinal'],1)
        store=Store(self.base/'store');ledger=store.committed(store.head());ref=ledger['archives']['rgb']
        self.assertEqual(ref,store.committed(first['head'])['archives']['rgb'])
        for key,event_key in [('source_bytes','actual_source_bytes'),('requests','actual_requests')]:
            self.assertEqual(second['reconciled_usage'][key],first['reconciled_usage'][key]+second['events'][0][event_key])
        budget=Budget(IndexLimits(max_read_bytes=8*MIB,max_decoded_bytes=32*MIB,max_seconds=110,max_rss_bytes=1024**3))
        runtime=DriveRuntime(self.session(),DriveLimits(max_body_bytes=16*MIB,max_requests=158,max_seconds=110,range_bytes=MIB,max_cache_bytes=4*MIB))
        name=next(iter(self.wanted));got=scanner.extract(store,ref,self.base/'extract',sha(canonical(self.spec)),name,runtime.source(self.spec,budget),budget,MIB)
        runtime.finish(budget);self.assertEqual(got,self.wanted[name])
        self.assertTrue(all(second['reconciled_usage'][k]<=self.plan['cap'][k] for k in runner.COUNTERS))
        self.proof={'first':first,'second':second,'extraction':{'sha256':hashlib.sha256(got).hexdigest(),'bytes':len(got),
            'logical_read_bytes':budget.read_bytes,'runtime':runtime.stats(),'separate_read_only_budget':True},'fixture_bytes':len(self.compressed)}
    def test_corruption_and_observation_failure_keep_pending(self):
        changed=bytearray(self.bodies['p0']);changed[10]=(changed[10]&~6)|6;self.bodies['p0']=bytes(changed)
        with patch.object(DriveRuntime,'stats',side_effect=RuntimeError('secret')):result=self.execute()
        event=result['events'][0];self.assertIsNotNone(result['pending']);self.assertEqual(event['phase'],'scan')
        self.assertEqual(event['exception_chain'][0]['type'],'ZranError');self.assertIsNone(event['source_reason'])
        self.assertIn('observation_error',event['observations']['budget']);self.assertNotIn('secret',json.dumps(result))
        self.assertEqual(Store(self.base/'store').committed(result['head'])['archives'],{})
    def test_checkpoint_failure_preserves_primary_and_last_archive(self):
        with patch.object(scanner,'export',side_effect=IndexFailure('checkpoint_injected')):result=self.execute('checkpoint')
        event=result['events'][0];self.assertIsNotNone(result['pending']);self.assertTrue(event['checkpoint_failed'])
        self.assertEqual(event['scan_primary'][0]['reason'],'read_bytes_limit')
        self.assertEqual(Store(self.base/'store').committed(result['head'])['archives'],{})
    def test_stale_source_reason_never_hides_corruption_in_checkpoint(self):
        # 첫 정상 read stop 뒤 checkpoint 재시작의 독립 손상 오류를 복원하지 않는다.
        budget=Budget(IndexLimits(max_read_bytes=8*MIB,max_decoded_bytes=32*MIB,max_rss_bytes=1024**3))
        budget.active_read_limit=4*MIB
        runtime=DriveRuntime(self.session(),DriveLimits(max_body_bytes=16*MIB,max_requests=158,max_seconds=110,range_bytes=MIB,max_cache_bytes=4*MIB))
        db=scanner.cache(Store(self.base/'local-store'),None,self.base/'cache','probe');self.addCleanup(db.close)
        original=scanner.open_engine;calls=[]
        def engine(*args):
            calls.append(1)
            if len(calls)>1:raise RuntimeError('independent_engine_failure')
            return original(*args)
        # coverage 부족을 강제해 checkpoint engine을 호출한다. 과거 reason은 재사용하지 않는다.
        original_decode=__import__('rewind_fleet.window',fromlist=['decode']).decode
        def decode(raw,**kw):
            header,points=original_decode(raw,**kw)
            return (header,[]) if kw.get('allow_empty') else (header,points)
        with patch.object(scanner,'open_engine',side_effect=engine),patch('rewind_fleet.window.decode',side_effect=decode):
            with self.assertRaises(RuntimeError) as caught:scanner.scan(scanner.initial(self.spec),None,runtime.source(self.spec,budget),budget,MIB,db)
        self.assertIsNone(budget.last_source_failure);self.assertEqual(caught.exception.fleet_scan_primary.reason,'read_bytes_limit')

    def test_zero_progress_settles_without_archive_head(self):
        original=runner.scan
        def limited(state,pack,reader,budget,*args,**kwargs):
            budget.active_read_limit=65536 # 첫 완전 파일 이전의 중단 경계 주입.
            return original(state,pack,reader,budget,*args,**kwargs)
        with patch.object(runner,'scan',side_effect=limited):result=self.execute()
        self.assertEqual(result['stop_reason'],'no_committed_progress');self.assertIsNone(result['pending'])
        self.assertEqual(result['events'][0]['state']['cursor'],0)
        self.assertEqual(Store(self.base/'store').committed(result['head'])['archives'],{})
        self.assertLess(result['reconciled_usage']['source_bytes'],result['usage']['source_bytes'])

    def test_decoded_stop_before_first_member_settles_without_head(self):
        original=runner.scan
        def limited(state,pack,reader,budget,*args,**kwargs):
            budget.limits=replace(budget.limits,max_decoded_bytes=65536)
            return original(state,pack,reader,budget,*args,**kwargs)
        with patch.object(runner,'scan',side_effect=limited):result=self.execute()
        self.assertIsNone(result['pending']);self.assertEqual(result['events'][0]['event']['reason'],'decoded_bytes_limit')
        self.assertEqual(Store(self.base/'store').committed(result['head'])['archives'],{})

    def test_checkpoint_hard_read_limit_does_not_publish_partial(self):
        original_engine=scanner.open_engine;calls=[]
        def engine(reader,*args):
            calls.append(1)
            if len(calls)>1:
                while True:reader.seek(0);reader.read(65536)
            return original_engine(reader,*args)
        original_decode=__import__('rewind_fleet.window',fromlist=['decode']).decode
        def decode(raw,**kw):
            header,points=original_decode(raw,**kw)
            return (header,[]) if kw.get('allow_empty') else (header,points)
        with patch.object(scanner,'open_engine',side_effect=engine),patch('rewind_fleet.window.decode',side_effect=decode):result=self.execute()
        event=result['events'][0];self.assertIsNotNone(result['pending']);self.assertTrue(event['checkpoint_failed'])
        self.assertEqual(event['exception_chain'][0]['reason'],'read_bytes_limit')
        self.assertEqual(event['scan_primary'][0]['reason'],'read_bytes_limit')
        self.assertEqual(event['observations']['budget']['source_read_bytes'],8*MIB)
        self.assertEqual(Store(self.base/'store').committed(result['head'])['archives'],{})

    def test_finish_failure_never_promotes_and_keeps_old_archive(self):
        first=self.execute();store=Store(self.base/'store');old=store.committed(first['head'])['archives']['rgb']
        with patch.object(DriveRuntime,'finish',side_effect=IndexFailure('metadata_injected')):result=self.execute('finish')
        self.assertIsNotNone(result['pending']);self.assertEqual(result['events'][0]['phase'],'source_finish')
        self.assertEqual(store.committed(result['head'])['archives']['rgb'],old)

    def test_later_checkpoint_failure_keeps_existing_archive(self):
        first=self.execute();store=Store(self.base/'store');old=store.committed(first['head'])['archives']['rgb']
        # 다음 scan의 checkpoint 실패를 runner 경계에 주입한다. source IO/한도는 늘리지 않는다.
        primary=IndexFailure('read_bytes_limit');failure=IndexFailure('checkpoint_injected')
        failure.fleet_checkpoint_failure=True;failure.fleet_scan_primary=primary
        with patch.object(runner,'scan',side_effect=failure):result=self.execute('later-checkpoint')
        event=result['events'][0];self.assertIsNotNone(result['pending']);self.assertTrue(event['checkpoint_failed'])
        self.assertEqual(store.committed(result['head'])['archives']['rgb'],old)
        for k in runner.COUNTERS:
            self.assertEqual(result['usage'][k],first['reconciled_usage'][k]+event['reservation'][k])

    def test_corruption_primary_survives_source_and_session_cleanup(self):
        changed=bytearray(self.bodies['p0']);changed[10]=(changed[10]&~6)|6;self.bodies['p0']=bytes(changed)
        original_source=DriveRuntime.source;original_session=self.session
        def reader(runtime,*args):
            obj=original_source(runtime,*args);close=obj.close;closed=[]
            def fail_once():
                close()
                if not closed:closed.append(True);raise IndexFailure('source_cleanup_injected')
            obj.close=fail_once;return obj
        def session():
            obj=original_session()
            def fail():raise IndexFailure('session_cleanup_injected')
            obj.close=fail;return obj
        self.session=session
        with patch.object(DriveRuntime,'source',reader),patch.object(scanner,'export',side_effect=AssertionError('fatal must not export')):result=self.execute()
        event=result['events'][0];self.assertIsNotNone(result['pending'])
        self.assertEqual(event['exception_chain'][0]['type'],'ZranError')
        self.assertEqual([chain[0]['reason'] for chain in event['cleanup_errors']],['source_cleanup_injected','session_cleanup_injected'])

if __name__=='__main__':unittest.main()
