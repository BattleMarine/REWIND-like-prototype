"""명시 survey allocation, quota 영속/준비 재사용/metadata-only 회수."""
import ast,copy,gzip,io,json,random,sys,tarfile,tempfile,unittest,zipfile
from pathlib import Path
from types import SimpleNamespace,ModuleType
from unittest.mock import patch
from rewind_archive.contracts import canonical
from rewind_archive_index.source import make_source,IndexFailure
from rewind_fleet import planning,runner,scanner
from rewind_fleet.survey import request,prepared,load_json,stage_zip,recover
from rewind_fleet.storage import Store,sha
from rewind_fleet.control import coverage,request_stop
from test_drive_metadata import FakeDrive

ROOT=Path(__file__).resolve().parents[1]

class SurveyTests(unittest.TestCase):
    def setUp(self):
        root=ROOT/'outputs/test_tmp';root.mkdir(exist_ok=True,parents=True)
        temp=tempfile.TemporaryDirectory(dir=root);self.addCleanup(temp.cleanup);self.base=Path(temp.name)
        archives=[];self.sources={};self.bodies={}
        for kind,count in [('rgb',130),('pose',130),('depth',5)]:
            data=io.BytesIO();rng=random.Random(19)
            with tarfile.open(fileobj=data,mode='w',format=tarfile.USTAR_FORMAT) as t:
                for i in range(count):
                    raw=rng.randbytes(16000);info=tarfile.TarInfo(f'{kind}/frame{i}');info.size=len(raw);t.addfile(info,io.BytesIO(raw))
            p=self.base/(kind+'.gz');self.bodies[kind]=gzip.compress(data.getvalue(),mtime=0);p.write_bytes(self.bodies[kind])
            self.sources[kind]=make_source([p],archive_id=kind,kind=kind if kind!='depth' else 'mixed')
            archives.append({'key':kind,'name':'scene','kind':kind,'split_scope':'train_val','parts':[{'index':0,'file_id':kind,
                'name':kind+'.gz000','size':p.stat().st_size,'version':None,'modified_time':'2026-10-14T00:00:00.000Z'}],
                'runtime_versions_verified':False,'classification':'observed_contiguous_official_unknown','official_part_count':1})
        self.observed=planning.seal({'schema':'fleet_inventory_v1','archives':archives,'official_registry_complete':True})
        inv=copy.deepcopy(self.observed);inv.pop('content_sha256')
        for a in inv['archives']:
            a['runtime_versions_verified']=True
            for p in a['parts']:p['version']='9'
        self.inv=planning.seal(inv);self.plan=planning.survey_plan(self.inv,['rgb','pose','depth'],storage_backend='local')
    def session(self):
        fake=FakeDrive({'parts':[{'file_id':k,'name':k+'.gz000'} for k in self.bodies]},self.bodies)
        fake.close=lambda:None;fake.get_adapter=lambda _:SimpleNamespace(max_retries=SimpleNamespace(total=0))
        return fake
    def run_survey(self,**kwargs):
        return runner.run(self.inv,self.plan,self.base/'store',kwargs.pop('workspace',self.base/'work'),owner='survey_test',
            single_writer=True,max_segments=kwargs.pop('max_segments',20),session_seconds=None,local_sources=self.sources,**kwargs)
    def test_two_scan_quota_then_next_archive_no_full_completion(self):
        first=self.run_survey(max_segments=1);self.assertEqual(first['survey_segments'],{'rgb':1})
        result=self.run_survey(workspace=self.base/'new_workspace')
        self.assertEqual(result['stop_reason'],'survey_allocations_finished',result)
        self.assertEqual(result['survey_segments'],{'rgb':2,'pose':2,'depth':1})
        self.assertFalse(result['dataset_complete']);self.assertFalse(result['whole_archive_estimate_complete'])
        self.assertEqual(result['progress']['files'],261)
        self.assertEqual([e['archive'] for e in result['events']],['rgb','pose','pose','depth'])
        c=coverage(self.base/'store');self.assertTrue(c['survey_complete']);self.assertFalse(c['catalog_complete'])
        self.assertEqual(c['next_action'],'survey_finished_review_measurements_for_full_plan')
        self.assertEqual(c['survey_archives_finished'],3)
        again=self.run_survey(workspace=self.base/'third');self.assertFalse(again['events']);self.assertEqual(again['survey_segments'],result['survey_segments'])
        self.assertTrue(all(result['reconciled_usage'][k]<=self.plan['cap'][k] for k in runner.COUNTERS if k!='seconds'))
        self.assertIsNone(result['reconciled_usage']['seconds']);self.assertGreater(result['observed_seconds'],0)
    def test_restore_segment_does_not_use_scan_quota(self):
        original=runner.cache;calls=[]
        def cache(*a,**kw):
            calls.append(1)
            return None if len(calls)==1 else original(*a,**kw)
        with patch.object(runner,'cache',side_effect=cache):r=self.run_survey(max_segments=2)
        self.assertEqual(r['progress']['segments'],2);self.assertEqual(r['survey_segments'],{'rgb':1})
    def test_policy_double_and_unknown_full_gate_unchanged(self):
        p=request(self.observed)
        for k,v in p['approved_base'].items():self.assertEqual(p['approved_cap'][k],2*v if v is not None else None)
        self.assertEqual(p['pin_base']['requests'],3);self.assertEqual(p['pin_cap']['requests'],6)
        unpinned=planning.survey_plan(self.observed,p['selected']);self.assertFalse(unpinned['complete_estimate'])
        full=planning.plan(self.inv,{},['rgb'],execution_profile=planning.AUTO_PROFILE);self.assertFalse(full['complete_estimate'])
        bad=copy.deepcopy(self.plan);bad.pop('content_sha256');bad['segments_per_archive']=3;bad=planning.seal(bad)
        with self.assertRaisesRegex(IndexFailure,'recalculation'):runner.run(self.inv,bad,self.base/'bad',self.base/'badwork',owner='bad',single_writer=True)
    def test_pin_once_persistent_reuse_and_conflict_blocked(self):
        policy=request(self.observed);calls=[]
        def factory():calls.append(1);return self.session()
        root=self.base/'prepared';manifest=prepared(self.observed,policy,root,session_factory=factory,source_sha256='a'*64)
        self.assertEqual(len(calls),1);self.assertEqual(manifest['pin_stats']['requests'],3)
        same=prepared(self.observed,policy,root,session_factory=lambda:self.fail('must not repin'),source_sha256='a'*64);self.assertEqual(same,manifest)
        with self.assertRaisesRegex(IndexFailure,'source_conflict'):prepared(self.observed,policy,root,source_sha256='b'*64)
        plan=load_json(root/'plan.json');self.assertEqual(plan['content_sha256'],manifest['plan_sha256'])
        changed=copy.deepcopy(plan);changed['cap']['source_bytes']+=1;(root/'plan.json').write_bytes(canonical(changed))
        with self.assertRaises(IndexFailure):prepared(self.observed,policy,root,session_factory=lambda:self.fail('must not repin'),source_sha256='a'*64)
    def test_failed_prepare_preserves_pin_reservation_no_repin(self):
        policy=request(self.observed);root=self.base/'failed'
        def fail():raise IndexFailure('fixture_auth_failure')
        with self.assertRaises(IndexFailure):prepared(self.observed,policy,root,session_factory=fail)
        self.assertTrue((root/'PREPARING.json').exists());self.assertTrue((root/'PREPARE_ERROR.json').exists())
        with self.assertRaisesRegex(IndexFailure,'no_automatic_repin'):prepared(self.observed,policy,root,session_factory=lambda:self.fail('no repin'))
    def test_mock_drive_survey_and_recovery_has_no_pack_or_source(self):
        result=runner.run(self.inv,self.plan,self.base/'drive-store',self.base/'drive-work',owner='survey',single_writer=True,
            max_segments=20,session_seconds=None,session_factory=self.session)
        self.assertTrue(result['survey_complete'],result);self.assertGreater(result['reconciled_usage']['requests'],0)
        work=self.base/'drive-work';(work/'worker-test.log').write_text('normal progress\nBearer topsecret\nsk-proj-secret\n',encoding='utf8')
        p=self.base/'recovery.zip';meta=recover(p,store=self.base/'drive-store',workspace=work,policy=request(self.observed))
        with zipfile.ZipFile(p) as z:
            self.assertFalse(any('/objects/' in n or n.endswith('.gz') for n in z.namelist()))
            text=z.read('logs/worker-test.log.redacted_tail.txt').decode();self.assertNotIn('topsecret',text);self.assertNotIn('sk-proj',text)
            self.assertEqual(z.testzip(),None)
        self.assertFalse(meta['source_or_pack_data_included'])
        self.assertLessEqual(meta['logical_io_bytes'],32*1024**2)
    def test_source_bundle_reuse_exact_and_traversal_rejected(self):
        p=self.base/'source.zip'
        with zipfile.ZipFile(p,'x') as z:z.writestr('rewind_fleet/a.py','x=1\n')
        target=self.base/'source';stage_zip(p,sha(p.read_bytes()),target);stage_zip(p,sha(p.read_bytes()),target)
        (target/'extra').write_bytes(b'x')
        with self.assertRaises(IndexFailure):stage_zip(p,sha(p.read_bytes()),target)
        bad=self.base/'bad.zip'
        with zipfile.ZipFile(bad,'x') as z:z.writestr('../escape','bad')
        with self.assertRaises(Exception):stage_zip(bad,sha(bad.read_bytes()),self.base/'badtarget')

    def test_after_six_hours_observed_time_never_forces_stop(self):
        clock=[0]
        def elapsed():clock[0]+=23000;return clock[0]
        with patch.object(runner,'time',SimpleNamespace(monotonic=elapsed)):
            result=self.run_survey()
        self.assertEqual(result['stop_reason'],'survey_allocations_finished',result)
        self.assertIsNone(result['pending']);self.assertIsNone(result['usage']['seconds'])
        self.assertGreater(result['seconds'],21600);self.assertGreater(result['observed_seconds'],21600)
        c=coverage(self.base/'store');self.assertIsNone(c['remaining']['seconds']);self.assertEqual(c['time_policy'],'observe_only')
        self.assertGreater(c['observed_seconds'],21600)

    def test_transfer_time_is_observed_but_request_timeout_stays_five(self):
        from rewind_fleet.timing import ObservedTransferBudget,ObservedTransferLimits,ObservedIndexBudget,ObservedIndexLimits
        b=ObservedTransferBudget(ObservedTransferLimits());b.started-=86400
        self.assertEqual(b.timeout(),5);self.assertGreater(b.stats()['elapsed_seconds'],21600)
        index=ObservedIndexBudget(ObservedIndexLimits());index.start-=86400;index.check()

class SurveyNotebookTests(unittest.TestCase):
    def setUp(self):
        root=ROOT/'outputs/test_tmp';root.mkdir(exist_ok=True,parents=True)
        tmp=tempfile.TemporaryDirectory(dir=root);self.addCleanup(tmp.cleanup);self.base=Path(tmp.name)
        nb=json.loads((ROOT/'notebooks/fleet_drive_survey_READY.ipynb').read_bytes())
        self.code=[''.join(c['source']) for c in nb['cells'] if c['cell_type']=='code']
        for code in self.code:ast.parse(code)
    def test_worker_launch_timer_stop_no_total_time_cap(self):
        from contextlib import redirect_stdout
        scope={};exec(self.code[0],scope)
        scope.update(BOOTSTRAP_OK=True,ACTION='prepare_and_start',LOCAL_ROOT=self.base,WORKSPACE=self.base/'work',SOURCE_DIR=ROOT,
            STORE=self.base/'store',PREPARED_DIR=self.base/'prepared',INPUT_DIR=self.base/'input',SOURCE_ZIP_SHA256='a'*64)
        manifest={'inventory_sha256':'b'*64,'plan_sha256':'c'*64}
        scope.update(prepared=lambda *a,**kw:manifest,load_json=lambda p:{},POLICY={},main=lambda args:commands.append(args),
            authorized_factory=lambda:self.fail('fake prepare must not open auth'),psutil=SimpleNamespace(virtual_memory=lambda:SimpleNamespace(available=3*1024**3)))
        colab=ModuleType('google.colab');colab.auth=SimpleNamespace(authenticate_user=lambda:None);colab.drive=SimpleNamespace(mount=lambda _:None)
        ip=ModuleType('IPython.display');screens=[];ip.display=screens.append;ip.HTML=lambda text:text
        worker=SimpleNamespace(pid=7,poll=lambda:None);commands=[]
        with patch.dict(sys.modules,{'google.colab':colab,'IPython.display':ip}),patch('subprocess.Popen',return_value=worker) as popen,patch('shutil.disk_usage',return_value=SimpleNamespace(free=5*1024**3)),redirect_stdout(io.StringIO()):
            exec(self.code[2],scope)
            self.assertIsNone(scope['FAILED']);argv=popen.call_args[0][0]
            self.assertIn('-B',argv);self.assertEqual(argv[argv.index('--session-seconds')+1],'none')
            self.assertTrue(any('setInterval' in s for s in screens));self.assertFalse(any('21600' in s for s in screens))
            scope['survey_stop']();self.assertEqual(commands[-1][0],'stop')
    def test_bootstrap_failure_recover_works_without_prepared_variables(self):
        from contextlib import redirect_stdout
        scope={'LOCAL_ROOT':self.base,'FAILED':{'stage':'fixture','type':'ValueError'}}
        colab=ModuleType('google.colab');colab.files=SimpleNamespace(download=lambda p:None)
        with patch.dict(sys.modules,{'google.colab':colab}),redirect_stdout(io.StringIO()):exec(self.code[3],scope)
        zips=list(self.base.glob('*.zip'));self.assertEqual(len(zips),1)
        with zipfile.ZipFile(zips[0]) as z:self.assertIsNone(z.testzip());self.assertIn('bootstrap_failure.json',z.namelist())
    def test_request_file_matches_time_policy_and_90_sources(self):
        observed=load_json(ROOT/'configs/fleet_inventory_observed.json');policy=load_json(ROOT/'configs/fleet_survey_request.json')
        self.assertEqual(policy,request(observed));self.assertEqual(len(policy['selected']),90)
        self.assertEqual(policy['pin_base']['requests'],354);self.assertEqual(policy['pin_cap']['requests'],708)
        self.assertIsNone(policy['approved_cap']['seconds']);self.assertEqual(policy['time_policy'],'observe_only')

    def test_354_metadata_only_pin_with_708_cap(self):
        observed=load_json(ROOT/'configs/fleet_inventory_observed.json')
        parts=[p for a in observed['archives'] for p in a['parts']]
        fake=FakeDrive({'parts':[{'file_id':p['file_id'],'name':p['name']} for p in parts]})
        for p in parts:fake.docs[p['file_id']].update(size=str(p['size']),modifiedTime=p['modified_time'])
        runtime=planning.pin(observed,fake,[a['key'] for a in observed['archives']],max_requests=708,max_seconds=None,survey_allocation=True)
        self.assertEqual(runtime['metadata_pin_stats']['requests'],354);self.assertIsNone(runtime['metadata_pin_stats']['seconds_limit'])
        self.assertEqual(len(fake.calls),354);self.assertTrue(all('alt=media' not in url for url,kw in fake.calls))
        self.assertTrue(all(kw['timeout']<=5 for url,kw in fake.calls))

    def test_loaded_project_modules_must_match_verified_source_dir(self):
        tree=ast.parse(self.code[1]);scope={'sys':sys,'Path':Path}
        funcs=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='require_source_modules']
        self.assertEqual(len(funcs),1)
        exec(compile(ast.Module(body=funcs,type_ignores=[]),'source_guard','exec'),scope)
        scope['require_source_modules'](ROOT)
        with patch.dict(sys.modules,{'rewind_fleet.stale':SimpleNamespace(__file__=str(ROOT.parent/'old.py'))}):
            with self.assertRaisesRegex(ValueError,'fresh_runtime'):scope['require_source_modules'](ROOT)

if __name__=='__main__':unittest.main()
