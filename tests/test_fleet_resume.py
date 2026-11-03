"""고정 pending1건의 source 승계·전액예약·46quota 보존과 충돌 차단."""
import ast,copy,gzip,hashlib,io,json,random,sys,tarfile,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
from types import ModuleType,SimpleNamespace
from rewind_archive.contracts import canonical
from rewind_archive_index.source import make_source,IndexFailure
from rewind_fleet import planning,scanner,resume
from rewind_fleet.survey import request as survey_request,publish_json,prepared
from rewind_fleet.storage import Store,sha
from rewind_fleet.control import request_stop

ROOT=Path(__file__).resolve().parents[1]

class ResumeTests(unittest.TestCase):
    def setUp(self):
        parent=ROOT/'outputs/test_tmp';parent.mkdir(parents=True,exist_ok=True)
        temp=tempfile.TemporaryDirectory(dir=parent);self.addCleanup(temp.cleanup);self.base=Path(temp.name)
        data=io.BytesIO();rng=random.Random(88)
        with tarfile.open(fileobj=data,mode='w') as t:
            for i in range(5):
                raw=rng.randbytes(4000);item=tarfile.TarInfo(f'seq/frame{i}');item.size=len(raw);t.addfile(item,io.BytesIO(raw))
        source=self.base/'data.gz';source.write_bytes(gzip.compress(data.getvalue(),mtime=0))
        keys=[f'done{i:02}' for i in range(46)]+['pending','zz_last'];archives=[];self.sources={}
        for key in keys:
            self.sources[key]=make_source([source],archive_id=key,kind='pose')
            archives.append({'key':key,'name':key,'split_scope':'train_val','kind':'pose','classification':'observed_contiguous_official_unknown',
                'runtime_versions_verified':True,'parts':[{'index':0,'file_id':key,'name':key+'.gz000','size':source.stat().st_size,
                'version':'9','modified_time':'2026-10-14T00:00:00.000Z'}],'official_part_count':1})
        self.inv=planning.seal({'schema':'fleet_inventory_v1','archives':archives,'official_registry_complete':True})
        observed=copy.deepcopy(self.inv);observed.pop('content_sha256')
        for a in observed['archives']:a['runtime_versions_verified']=False;a['parts'][0]['version']=None
        self.observed=planning.seal(observed);self.policy=survey_request(self.observed)
        self.plan=planning.survey_plan(self.inv,keys,storage_backend='local')
        self.prepared=self.base/'prepared';self.prepared.mkdir()
        manifest=planning.seal({'schema':'fleet_prepared_survey_v1','inventory_sha256':self.inv['content_sha256'],
            'plan_sha256':self.plan['content_sha256'],'request_sha256':self.policy['content_sha256'],'source_zip_sha256':'a'*64,
            'complete':True,'whole_archive_estimate_complete':False})
        for name,doc in [('PREPARED.json',manifest),('runtime_inventory.json',self.inv),('plan.json',self.plan)]:publish_json(self.prepared/name,doc)
        self.store=Store(self.base/'store');refs={}
        for i,key in enumerate(keys[:46]):
            state=scanner.initial(self.sources[key]);state.update(files=80 if i<45 else 152,directories=0,ordinal=80 if i<45 else 152)
            refs[key]=self.store.commit({'schema':'fleet_archive_commit_v1','state':state})
        pending={'archive':'pending','token':'c'*32,'reserved':{'source_bytes':64*1024**2,'requests':1000,'seconds':None,'artifact_io_bytes':16*1024**2},'owner':'old_worker'}
        ledger={'schema':'fleet_ledger_v1','plan_sha256':self.plan['content_sha256'],'parent':None,'pending':pending,'settlement':None,
            'usage':{'source_bytes':64*1024**2+10000,'requests':1010,'seconds':None,'artifact_io_bytes':16*1024**2+10000},
            'archive_usage':{'pending':copy.deepcopy(pending['reserved'])},'selected_archives':keys,'survey_quota':2,
            'survey_segments':{k:2 for k in keys[:46]},'statuses':{k:{'eof':False,'complete':False} for k in keys[:46]},
            'archives':refs,'complete':False,'train_ready':False,'time_policy':'observe_only','scope':'survey','plan_cap':self.plan['cap'],
            'progress':{'segments':92,'files':3752,'directories':0,'decoded_advanced':0},'observed_seconds':800}
        self.head=self.store.commit(ledger);self.store.publish(self.head,None);self.original=ledger
        case=planning.seal({'schema':'fleet_survey_resume_case_v1','head':self.head,'pending':pending,'usage':ledger['usage'],
            'survey_segments':ledger['survey_segments'],'progress':ledger['progress'],'prepared_content_sha256':manifest['content_sha256'],
            'prepared_file_sha256':sha((self.prepared/'PREPARED.json').read_bytes()),'old_source_zip_sha256':'a'*64,
            'original_request_sha256':self.policy['content_sha256'],'inventory_sha256':self.inv['content_sha256'],
            'plan_sha256':self.plan['content_sha256'],'owner_stopped_evidence':{'worker_exit_code':0,'lease_absent':True}})
        self.request=resume.resume_request(case,'b'*64)
    def snapshot(self):
        return {str(p.relative_to(self.base)):sha(p.read_bytes()) for p in self.base.rglob('*') if p.is_file()}
    def preflight(self,**kw):return resume.preflight(self.store.root,self.prepared,self.request,new_source_sha256=kw.get('source','b'*64))
    def apply(self):return resume.apply_once(self.store.root,self.prepared,self.request,new_source_sha256='b'*64)
    def test_read_only_preflight_and_exact_pins(self):
        before=self.snapshot();r=self.preflight();self.assertEqual(r['phase'],'original_pending')
        self.assertEqual(len(r['survey_segments']),46);self.assertEqual(self.snapshot(),before)
        with self.assertRaises(IndexFailure):self.preflight(source='d'*64)
        with self.assertRaisesRegex(IndexFailure,'source_conflict'):prepared(self.observed,self.policy,self.prepared,source_sha256='b'*64)
    def test_recovery_keeps_full_usage_quota_archives_and_prepared(self):
        files={n:sha((self.prepared/n).read_bytes()) for n in ('PREPARED.json','runtime_inventory.json','plan.json')}
        receipt=self.apply();after=self.store.committed(self.store.head())
        self.assertEqual(after,{**self.original,'parent':self.head,'pending':None,'recovery':'unknown_consumption_charged_full_reservation'})
        self.assertEqual(receipt['recovered_head'],self.store.head());self.assertTrue(receipt['unknown_reservation_charged_in_full'])
        self.assertEqual(files,{n:sha((self.prepared/n).read_bytes()) for n in files})
        self.assertEqual(self.preflight()['phase'],'ready_once')
        self.assertEqual(self.apply(),receipt) # 완료receipt를 다시 읽을 뿐 예약을 다시 처리하지 않는다.
    def test_same_plan_normal_resume_preserves_46_completed_quotas(self):
        r=resume.resume_once(self.store.root,self.prepared,self.request,new_source_sha256='b'*64,workspace=self.base/'work',owner='new_worker',local_sources=self.sources)
        self.assertEqual(r['stop_reason'],'survey_allocations_finished',r);self.assertIsNone(r['pending'])
        self.assertEqual([e['archive'] for e in r['events']],['pending','zz_last'])
        self.assertEqual({k:r['survey_segments'][k] for k in self.original['survey_segments']},self.original['survey_segments'])
        for k in ('source_bytes','requests','artifact_io_bytes'):self.assertGreaterEqual(r['reconciled_usage'][k],self.original['usage'][k])
        self.assertEqual(r['plan_sha256'],self.plan['content_sha256'])
        final=self.store.committed(self.store.head())
        self.assertEqual({k:final['archives'][k] for k in self.original['archives']},self.original['archives'])
        with self.assertRaisesRegex(IndexFailure,'already_launched'):self.preflight()
    def test_conflicting_head_lease_and_prepared_block_without_mutation(self):
        with self.store.lease('other'):
            with self.assertRaisesRegex(IndexFailure,'writer_active'):self.preflight()
        original=(self.prepared/'PREPARED.json').read_bytes();(self.prepared/'PREPARED.json').write_bytes(original+b' ')
        with self.assertRaisesRegex(IndexFailure,'prepared_conflict'):self.preflight()
        (self.prepared/'PREPARED.json').write_bytes(original)
        doc=copy.deepcopy(self.original);doc['parent']=self.head;head=self.store.commit(doc);self.store.publish(head,self.head)
        with self.assertRaisesRegex(IndexFailure,'head_changed'):self.preflight()
    def test_crash_between_intent_and_recovery_is_blocked_no_retry(self):
        with patch.object(resume,'recover_pending',side_effect=RuntimeError('injected')):
            with self.assertRaises(RuntimeError):self.apply()
        self.assertEqual(self.store.head(),self.head);self.assertEqual(self.store.committed(self.head)['pending'],self.original['pending'])
        with self.assertRaisesRegex(IndexFailure,'incomplete_migration'):self.apply()
    def test_crash_after_recovery_before_complete_marker_is_blocked(self):
        write=resume._Context.write
        def fail_complete(ctx,name,doc):
            if 'COMPLETE' in name:raise RuntimeError('injected')
            return write(ctx,name,doc)
        with patch.object(resume._Context,'write',fail_complete):
            with self.assertRaises(RuntimeError):self.apply()
        current=self.store.committed(self.store.head());self.assertIsNone(current['pending']);self.assertEqual(current['usage'],self.original['usage'])
        with self.assertRaisesRegex(IndexFailure,'incomplete_migration'):self.preflight()

    def test_error_cli_and_recovery_include_fixed_request_and_markers(self):
        request_file=self.base/'request.json';publish_json(request_file,self.request)
        output=self.base/'work'/'attempt.json'
        from contextlib import redirect_stdout
        with redirect_stdout(io.StringIO()),self.assertRaises(SystemExit):
            resume.main(['--store',str(self.store.root),'--prepared',str(self.prepared),'--request',str(request_file),
                '--request-sha256',self.request['content_sha256'],'--new-source-sha256','d'*64,'--output',str(output)])
        diagnostic=Path(str(output)+'.error.json');self.assertTrue(diagnostic.is_file())
        self.apply();zipfile_path=self.base/'recovery.zip'
        result=resume.recover_resume(zipfile_path,root=self.store.root,prepared_root=self.prepared,workspace=output.parent,
            request_file=request_file,diagnostic_file=diagnostic)
        import zipfile
        with zipfile.ZipFile(zipfile_path) as z:
            names=z.namelist();self.assertIn('resume/request.json',names);self.assertIn('resume/diagnostic.json',names)
            self.assertTrue(any('RESUME-INTENT' in n for n in names));self.assertTrue(any('RESUME-COMPLETE' in n for n in names))
            self.assertFalse(any('/objects/' in n for n in names));self.assertIsNone(z.testzip())
        self.assertLessEqual(result['logical_io_bytes'],32*1024**2)

    def test_completed_receipt_path_still_enforces_storage_cap(self):
        self.apply();original_guard=resume._storage_guard;calls=[]
        def no_storage(ctx,plan):
            calls.append(1);original_guard(ctx,plan)
            ctx.store.write_check=lambda n:(_ for _ in ()).throw(IndexFailure('fleet_resume_storage_limit'))
        with patch.object(resume,'_storage_guard',side_effect=no_storage):
            with self.assertRaisesRegex(IndexFailure,'storage_limit'):
                resume.resume_once(self.store.root,self.prepared,self.request,new_source_sha256='b'*64,workspace=self.base/'work',owner='new',local_sources=self.sources)
        self.assertEqual(len(calls),1);self.assertIsNone(self.store.committed(self.store.head())['pending'])
    def test_new_stop_during_recovery_is_not_acknowledged_or_ignored(self):
        original_run=resume.run
        def stopped(*args,**kwargs):
            request_stop(self.store.root,self.plan['content_sha256'])
            self.assertFalse(kwargs['resume_stopped'])
            return original_run(*args,**kwargs)
        with patch.object(resume,'run',side_effect=stopped):
            result=resume.resume_once(self.store.root,self.prepared,self.request,new_source_sha256='b'*64,
                workspace=self.base/'work',owner='new_worker',local_sources=self.sources)
        self.assertEqual(result['stop_reason'],'stop_requested');self.assertFalse(result['events'])
        self.assertIsNone(result['pending']);self.assertEqual(result['usage'],self.original['usage'])
        self.assertEqual(result['survey_segments'],self.original['survey_segments'])
        self.assertNotIn('stop_acknowledged',self.store.committed(self.store.head()))

class ResumeNotebookTests(unittest.TestCase):
    def setUp(self):
        parent=ROOT/'outputs/test_tmp';parent.mkdir(parents=True,exist_ok=True)
        tmp=tempfile.TemporaryDirectory(dir=parent);self.addCleanup(tmp.cleanup);self.base=Path(tmp.name)
        nb=json.loads((ROOT/'notebooks/fleet_survey_resume_READY.ipynb').read_bytes())
        self.code=[''.join(c['source']) for c in nb['cells'] if c['cell_type']=='code']
        for code in self.code:ast.parse(code)
    def receive(self):
        tree=ast.parse(self.code[1]);funcs=[n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='receive_verified_bundles']
        scope={'hashlib':hashlib};exec(compile(ast.Module(body=funcs,type_ignores=[]),'upload_helper','exec'),scope)
        return scope['receive_verified_bundles']
    def test_two_bundles_can_be_uploaded_one_at_a_time_and_reused(self):
        expected={'source.zip':sha(b'source'),'input.zip':sha(b'input')};calls=[]
        chunks=[{'source (1).zip':b'source'},{'input (2).zip':b'input'}]
        def upload():calls.append(1);return chunks.pop(0)
        self.receive()(self.base,expected,upload);self.assertEqual(len(calls),2)
        self.receive()(self.base,expected,lambda:self.fail('valid files must be reused'))
    def test_upload_sha_conflict_is_blocked_without_overwrite(self):
        expected={'source.zip':sha(b'source'),'input.zip':sha(b'input')}
        with self.assertRaisesRegex(ValueError,'unknown_sha'):self.receive()(self.base,expected,lambda:{'source.zip':b'wrong'})
        self.assertFalse((self.base/'source.zip').exists())
        (self.base/'source.zip').write_bytes(b'wrong')
        with self.assertRaisesRegex(ValueError,'existing_bundle_sha'):self.receive()(self.base,expected,lambda:self.fail('must not upload'))
    def test_both_renamed_uploads_and_cancel(self):
        expected={'source.zip':sha(b'source'),'input.zip':sha(b'input')}
        self.receive()(self.base,expected,lambda:{'renamed-A.zip':b'input','renamed-B.zip':b'source'})
        self.assertEqual((self.base/'source.zip').read_bytes(),b'source');self.assertEqual((self.base/'input.zip').read_bytes(),b'input')
        empty=self.base/'empty';empty.mkdir()
        with self.assertRaisesRegex(ValueError,'cancelled'):self.receive()(empty,expected,lambda:{})
    def test_worker_resume_command_is_pinned_and_nonblocking_stop_available(self):
        from contextlib import redirect_stdout
        scope={};exec(self.code[0],scope)
        scope.update(BOOTSTRAP_OK=True,LOCAL_ROOT=self.base,WORKSPACE=self.base/'work',SOURCE_DIR=ROOT,STORE=self.base/'store',
            PREPARED_DIR=self.base/'prepared',INPUT_DIR=self.base/'input',SOURCE_ZIP_SHA256='b'*64,EXPECTED_RESUME_REQUEST_SHA256='d'*64,
            PLAN_SHA256='e'*64,psutil=SimpleNamespace(virtual_memory=lambda:SimpleNamespace(available=3*1024**3)),main=lambda args:commands.append(args))
        colab=ModuleType('google.colab');colab.auth=SimpleNamespace(authenticate_user=lambda:None);colab.drive=SimpleNamespace(mount=lambda _:None)
        ip=ModuleType('IPython.display');screens=[];ip.display=screens.append;ip.HTML=lambda text:text
        worker=SimpleNamespace(pid=7,poll=lambda:None);commands=[]
        with patch.dict(sys.modules,{'google.colab':colab,'IPython.display':ip}),patch('subprocess.Popen',return_value=worker) as popen,patch('shutil.disk_usage',return_value=SimpleNamespace(free=5*1024**3)),redirect_stdout(io.StringIO()):
            exec(self.code[2],scope);self.assertIsNone(scope['FAILED'])
            argv=popen.call_args[0][0];self.assertIn('rewind_fleet.resume',argv);self.assertIn('--apply-once',argv)
            self.assertIn('-B',argv);self.assertEqual(argv[argv.index('--new-source-sha256')+1],'b'*64)
            self.assertEqual(argv[argv.index('--request-sha256')+1],'d'*64);self.assertNotIn('--session-seconds',argv)
            self.assertTrue(any('setInterval' in screen for screen in screens));scope['survey_stop']();self.assertEqual(commands[-1][0],'stop')
    def test_fixed_error_codes_and_setup_failure_are_recovered(self):
        from contextlib import redirect_stdout
        scope={};exec(self.code[0],scope)
        self.assertEqual(scope['safe_notebook_failure'](ValueError('uploaded_bundle_unknown_sha'),'bootstrap')['reason'],'uploaded_bundle_unknown_sha')
        self.assertIsNone(scope['safe_notebook_failure'](ValueError('Bearer secret'),'bootstrap')['reason'])
        scope.update(BOOTSTRAP_OK=True,ACTION='preflight',WORKSPACE=self.base/'work',STORE=self.base/'store',
            PREPARED_DIR=self.base/'prepared',LOCAL_ROOT=self.base,INPUT_DIR=self.base/'input',RESUME_REQUEST={},
            preflight=lambda *a,**kw:(_ for _ in ()).throw(IndexFailure('fleet_resume_old_head_changed_review_required')))
        colab=ModuleType('google.colab');colab.auth=SimpleNamespace(authenticate_user=lambda:None);colab.drive=SimpleNamespace(mount=lambda _:None)
        colab.files=SimpleNamespace(download=lambda _:None)
        calls=[]
        def recovery(output,**kw):calls.append(kw);return {'path':str(output)}
        scope['recover_resume']=recovery
        with patch.dict(sys.modules,{'google.colab':colab}),redirect_stdout(io.StringIO()):
            exec(self.code[2],scope);self.assertTrue(scope['SETUP_DIAGNOSTIC'].is_file())
            exec(self.code[3],scope)
        self.assertEqual(calls[0]['diagnostic_file'],scope['SETUP_DIAGNOSTIC'])
        self.assertEqual(json.loads(scope['SETUP_DIAGNOSTIC'].read_bytes())['reason'],'fleet_resume_old_head_changed_review_required')
    def test_settings_clear_old_diagnostics_but_preserve_worker(self):
        worker=object();scope={'SETUP_DIAGNOSTIC':'old','PREFLIGHT_FILE':'old','WORKER':worker}
        exec(self.code[0],scope)
        self.assertIsNone(scope['SETUP_DIAGNOSTIC']);self.assertIsNone(scope['PREFLIGHT_FILE']);self.assertIs(scope['WORKER'],worker)

if __name__=='__main__':unittest.main()
