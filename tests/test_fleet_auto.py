"""명시 auto profile의 연속 저장·다중 archive·협력 중단/재개."""
import copy,gzip,hashlib,io,json,random,tarfile,tempfile,unittest
import ast
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from rewind_archive.contracts import canonical
from rewind_archive_index.source import Budget,IndexLimits,SplitSource,IndexFailure,make_source
from rewind_fleet import planning,runner,scanner
from rewind_fleet.control import coverage,request_stop
from rewind_fleet.storage import Store,sha
from test_drive_metadata import FakeDrive

MIB=1024**2
ROOT=Path(__file__).resolve().parents[1]

class AutoFleetTests(unittest.TestCase):
    def setUp(self):
        root=ROOT/'outputs/test_tmp';root.mkdir(parents=True,exist_ok=True)
        tmp=tempfile.TemporaryDirectory(dir=root);self.addCleanup(tmp.cleanup);self.base=Path(tmp.name)
        archives=[];self.sources={};self.wanted={};self.est={};self.bodies={}
        for kind in ('rgb','pose'):
            data=io.BytesIO();rng=random.Random(82);wanted={}
            with tarfile.open(fileobj=data,mode='w',format=tarfile.USTAR_FORMAT) as tar:
                for i in range(400):
                    name=f'{kind}/seq/frame_{i}.bin';raw=rng.randbytes(40000)
                    if i in (0,199,399):wanted[name]=raw
                    item=tarfile.TarInfo(name);item.size=len(raw);tar.addfile(item,io.BytesIO(raw))
            compressed=gzip.compress(data.getvalue(),mtime=0);file=self.base/(kind+'.gz');file.write_bytes(compressed)
            self.bodies[kind]=compressed;self.sources[kind]=make_source([file],archive_id=kind,kind=kind,dataset_version='auto_fixture')
            self.wanted[kind]=wanted
            archives.append({'key':kind,'kind':kind,'parts':[{'index':0,'file_id':kind,'name':kind+'.gz000','size':len(compressed),
                'version':'9','modified_time':'2026-10-14T00:00:00.000Z'}],'runtime_versions_verified':True,'official_part_count':1,
                'classification':'observed_contiguous_official_unknown'})
            self.est[kind]={'decoded_bytes':len(data.getvalue()),'members':400,'source_bytes_per_second':5*MIB,
                'http_requests_per_second':1,'artifact_bytes_per_second':20*MIB,'python_base_rss_bytes':512*MIB,
                'replay_fraction':1,'catalog_bytes_per_member':1024,'seconds_per_segment':.2,'failure_residue_bytes':8*MIB,
                'basis':'exact generated 400x40000 fixture; one source replay prior, measured tests still bounded by plan'}
        self.inv=planning.seal({'schema':'fleet_inventory_v1','archives':archives,'official_registry_complete':True})
        self.plan=planning.plan(self.inv,self.est,['rgb','pose'],execution_profile=planning.AUTO_PROFILE,storage_backend='local')
    def execute(self,**kw):
        return runner.run(self.inv,self.plan,self.base/'store',kw.pop('workspace',self.base/'work'),owner='auto_test',single_writer=True,
            max_segments=kw.pop('max_segments',32),session_seconds=120,local_sources=self.sources,**kw)
    def check_extract(self):
        store=Store(self.base/'store');ledger=store.committed(store.head())
        for kind in self.wanted:
            for name,raw in self.wanted[kind].items():
                b=Budget(IndexLimits(max_read_bytes=8*MIB,max_decoded_bytes=32*MIB,max_rss_bytes=1024**3))
                got=scanner.extract(store,ledger['archives'][kind],self.base/'extract',sha(canonical(self.sources[kind])),name,SplitSource(self.sources[kind],b),b,MIB)
                self.assertEqual(hashlib.sha256(raw).digest(),hashlib.sha256(got).digest())
    def test_one_call_more_than_ten_segments_two_archives_and_old_sha(self):
        result=self.execute();self.assertIsNone(result['pending'],result)
        self.assertEqual(result['stop_reason'],'all_selected_catalogs_finished',result)
        self.assertGreaterEqual(result['progress']['segments'],10)
        self.assertEqual(result['progress']['files'],800);self.assertLessEqual(len(result['events']),64)
        self.check_extract();c=coverage(self.base/'store')
        self.assertTrue(c['catalog_complete']);self.assertFalse(c['full_integrity_complete']);self.assertIsNone(c['pair_count'])
        self.assertEqual(sum(a['files'] for a in c['archives']),800)
        self.assertTrue(all(result['reconciled_usage'][k]<=self.plan['cap'][k] for k in runner.COUNTERS))
        self.proof={'report':result,'coverage':c}
    def test_stop_after_segment_resume_new_workspace_no_budget_reset(self):
        original=runner.scan;calls=[]
        def stop_after(*args,**kwargs):
            output=original(*args,**kwargs);calls.append(1)
            if len(calls)==3:request_stop(self.base/'store',self.plan['content_sha256'])
            return output
        with patch.object(runner,'scan',side_effect=stop_after):first=self.execute()
        self.assertEqual(first['stop_reason'],'stop_requested');self.assertIsNone(first['pending']);self.assertEqual(len(calls),3)
        stopped=self.execute(workspace=self.base/'still_stopped');self.assertFalse(stopped['events'])
        result=self.execute(workspace=self.base/'new_runtime',resume_stopped=True)
        self.assertEqual(result['stop_reason'],'all_selected_catalogs_finished',result)
        self.assertEqual(result['progress']['files'],800)
        for key in ('source_bytes','requests'):
            actual=sum(e['actual_source_bytes' if key=='source_bytes' else 'actual_requests'] for e in result['events'])
            self.assertEqual(result['reconciled_usage'][key],first['reconciled_usage'][key]+actual)
        self.check_extract()
        self.proof={'first':first,'resumed':result,'coverage':coverage(self.base/'store')}
    def test_profile_pin_no_progress_stops_once_and_pending_not_refunded(self):
        bad=copy.deepcopy(self.plan);bad.pop('content_sha256');bad['engine']['buffer_size']=2*MIB;bad=planning.seal(bad)
        with self.assertRaisesRegex(IndexFailure,'recalculation'):
            runner.run(self.inv,bad,self.base/'bad',self.base/'badwork',owner='test',single_writer=True)
        original=runner.scan;calls=[]
        def no_progress(state,pack,reader,budget,*args,**kw):
            calls.append(1);budget.active_read_limit=65536
            return original(state,pack,reader,budget,*args,**kw)
        with patch.object(runner,'scan',side_effect=no_progress):result=self.execute()
        self.assertEqual(result['stop_reason'],'no_committed_progress');self.assertEqual(len(calls),1);self.assertIsNone(result['pending'])

    def test_mock_drive_multi_segment_actual_transfer_settlement(self):
        def session():
            fake=FakeDrive({'parts':[{'file_id':p['file_id'],'name':p['name']} for a in self.inv['archives'] for p in a['parts']]},self.bodies)
            fake.get_adapter=lambda _:SimpleNamespace(max_retries=SimpleNamespace(total=0));fake.close=lambda:None
            return fake
        result=runner.run(self.inv,self.plan,self.base/'drive-store',self.base/'drive-work',owner='mock_auto',single_writer=True,
            max_segments=32,session_seconds=120,session_factory=session)
        self.assertEqual(result['stop_reason'],'all_selected_catalogs_finished',result);self.assertIsNone(result['pending'])
        self.assertEqual(result['progress']['files'],800);self.assertGreaterEqual(result['progress']['segments'],10)
        self.assertEqual(result['reconciled_usage']['source_bytes'],sum(e['actual_source_bytes'] for e in result['events']))
        self.assertEqual(result['reconciled_usage']['requests'],sum(e['actual_requests'] for e in result['events']))
        self.assertTrue(all(result['reconciled_usage'][k]<=self.plan['cap'][k] for k in runner.COUNTERS))
        self.proof={'report':result,'coverage':coverage(self.base/'drive-store')}

    def test_recent_events_bounded_at64_including_failure(self):
        # 비용/관측 계약만 검사하는 fake scan. 이 catalog를 실제 데이터 증거로 쓰지 않는다.
        from rewind_fleet.window import encode
        est=copy.deepcopy(self.est)
        for e in est.values():
            e['seconds_per_segment']=60;e['catalog_bytes_per_member']=4096
            e['basis']='fake scan event retention fixture; explicit scheduling and 4KiB catalog allowance'
        self.plan=planning.plan(self.inv,est,['rgb','pose'],execution_profile=planning.AUTO_PROFILE,storage_backend='local')
        calls=[]
        def fake(state,pack,reader,budget,*args,**kw):
            calls.append(1)
            if len(calls)==66:raise IndexFailure('injected_after65')
            end=state['cursor']+512;n=state['ordinal'];new={**state,'cursor':end,'boundary':end,'ordinal':n+1}
            row={'ordinal':n,'chain_start':state['cursor'],'name':f'fake/{n}','offset':state['cursor'],'size':512,'end':end,'sha256':'0'*64,'type':'file'}
            raw=encode((reader.size,end,MIB,32768),[[[10,0,0,0],b'']])
            return new,[row],raw,{'reason':None,'decoded_advanced':512,'stats':budget.stats()}
        with patch.object(runner,'scan',side_effect=fake):result=self.execute(max_segments=66)
        self.assertEqual(len(calls),66,result['stop_reason']);self.assertEqual(len(result['events']),64)
        self.assertEqual(result['progress']['segments'],65);self.assertIsNotNone(result['pending'])
        self.assertEqual(result['events'][-1]['reason'],'injected_after65')
        self.assertEqual(coverage(self.base/'store')['last_run']['stop_reason'],'injected_after65')

    def test_auto_corrupt_deflate_keeps_reservation_no_retry(self):
        p=Path(self.sources['rgb']['parts'][0]['path']);raw=bytearray(p.read_bytes());raw[10]=(raw[10]&~6)|6;p.write_bytes(raw)
        self.sources['rgb']=make_source([p],archive_id='rgb',kind='rgb',dataset_version='auto_fixture')
        result=self.execute();self.assertIsNotNone(result['pending']);self.assertEqual(len(result['events']),1)
        self.assertEqual(result['events'][0]['exception_chain'][0]['type'],'ZranError')
        self.assertEqual(result['usage'],result['reconciled_usage']);self.assertEqual(result['progress']['files'],0)

class AutoFrontendTests(unittest.TestCase):
    def setUp(self):
        from test_fleet import FleetTests
        fixture=FleetTests();fixture.setUp();self.addCleanup(fixture.doCleanups)
        self.base=fixture.base;self.inv=fixture.inv;self.sources=fixture.sources
        self.plan=planning.plan(self.inv,fixture.est,['rgb','pose'],execution_profile=planning.AUTO_PROFILE,storage_backend='local')
    def test_session_boundary_stops_before_reservation_or_source(self):
        with patch.object(runner,'time',SimpleNamespace(monotonic=lambda:100)):
            result=runner.run(self.inv,self.plan,self.base/'store',self.base/'work',owner='short',single_writer=True,
                max_segments=100,session_seconds=10,session_factory=lambda:self.fail('no source'))
        self.assertEqual(result['stop_reason'],'session_time_boundary');self.assertIsNone(result['pending']);self.assertFalse(result['events'])
    def test_cli_auto_run_stop_resume_coverage_same_plan(self):
        from rewind_fleet.__main__ import main
        from contextlib import redirect_stdout
        for name,doc in [('inventory',self.inv),('plan',self.plan)]:(self.base/(name+'.json')).write_bytes(canonical(doc))
        args=['--inventory',str(self.base/'inventory.json'),'--plan',str(self.base/'plan.json'),'--plan-sha256',self.plan['content_sha256'],
            '--store',str(self.base/'store'),'--workspace',str(self.base/'work'),'--owner','cli','--single-writer','--session-seconds','120']
        # 실제 CLI parser→core, 테스트만 local fixture source로 주입한다.
        original=runner.run
        def execute(*a,**kw):return original(*a,**kw,local_sources=self.sources)
        with patch('rewind_fleet.__main__.run',side_effect=execute),redirect_stdout(io.StringIO()):
            main(['auto-run',*args,'--max-segments','1','--output',str(self.base/'first.json')])
            main(['stop','--store',str(self.base/'store'),'--plan-sha256',self.plan['content_sha256']])
            main(['resume',*args,'--output',str(self.base/'second.json')])
            main(['coverage','--store',str(self.base/'store')])
        result=json.loads((self.base/'second.json').read_bytes());self.assertIsNone(result['pending'])
        self.assertEqual(result['stop_reason'],'all_selected_catalogs_finished')
    def test_thin_notebook_compiles_and_status_default_has_no_auth(self):
        nb=json.loads((ROOT/'notebooks/fleet_auto_index.ipynb').read_bytes());scope={}
        for cell in nb['cells']:
            if cell['cell_type']=='code':ast.parse(''.join(cell['source']))
        code=[''.join(c['source']) for c in nb['cells'] if c['cell_type']=='code']
        exec(code[0],scope);self.assertEqual(scope['ACTION'],'status')
        scope.update(PROJECT_DIR=ROOT,STORE=self.base/'missing')
        from contextlib import redirect_stdout
        with patch('rewind_fleet.__main__.authorized_factory',side_effect=AssertionError('no auth')),redirect_stdout(io.StringIO()):exec(code[1],scope)

    def test_notebook_launch_returns_and_stop_works_with_active_worker(self):
        nb=json.loads((ROOT/'notebooks/fleet_auto_index.ipynb').read_bytes())
        code=[''.join(c['source']) for c in nb['cells'] if c['cell_type']=='code'];scope={};exec(code[0],scope)
        scope.update(PROJECT_DIR=ROOT,STORE=self.base/'store',WORKSPACE=self.base/'worker',ACTION='auto-run',EXPECTED_PLAN_SHA256=self.plan['content_sha256'])
        child=SimpleNamespace(pid=123,poll=lambda:None)
        from contextlib import redirect_stdout
        with patch('subprocess.Popen',return_value=child) as launch,patch('rewind_fleet.__main__.main') as cli,redirect_stdout(io.StringIO()):
            exec(code[1],scope);self.assertEqual(launch.call_count,1);cli.assert_not_called()
            scope['ACTION']='stop';exec(code[1],scope);self.assertEqual(cli.call_args[0][0][0],'stop')
            scope['ACTION']='resume'
            with self.assertRaisesRegex(ValueError,'worker_already_running'):exec(code[1],scope)

    def test_tiny_members_lazy_index_creates_bounded_checkpoint(self):
        raw=io.BytesIO();rng=random.Random(7)
        with tarfile.open(fileobj=raw,mode='w') as tar:
            for i in range(1000):
                data=rng.randbytes(2048);item=tarfile.TarInfo(f'f/{i}');item.size=len(data);tar.addfile(item,io.BytesIO(data))
        p=self.base/'tiny.gz';p.write_bytes(gzip.compress(raw.getvalue(),mtime=0))
        a=copy.deepcopy(self.inv['archives'][0]);a['parts'][0]['size']=p.stat().st_size
        inv=planning.seal({'schema':'fleet_inventory_v1','archives':[a],'official_registry_complete':True})
        est={**self.plan['archive_plans'][0]['estimates'],'members':1000,'decoded_bytes':len(raw.getvalue()),'replay_fraction':2,
             'basis':'exact tiny1000x2048 fixture; bounded checkpoint materialization test'}
        plan=planning.plan(inv,{'rgb':est},['rgb'],storage_backend='local',execution_profile=planning.AUTO_PROFILE)
        spec=make_source([p],archive_id='tiny',kind='rgb')
        result=runner.run(inv,plan,self.base/'tiny-store',self.base/'tiny-work',owner='tiny',single_writer=True,max_segments=2,
            session_seconds=120,local_sources={'rgb':spec})
        self.assertIsNone(result['pending'],result);self.assertEqual(result['progress']['files'],128)

if __name__=='__main__':unittest.main()
