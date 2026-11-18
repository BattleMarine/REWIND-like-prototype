"""Meaningful local-only v2 boundary, cache, recovery and learning fixtures."""
from copy import deepcopy
import io
import json
from pathlib import Path
import tempfile
import subprocess
import sys
from unittest import TestCase
from unittest.mock import patch

import torch

from rewind_archive.contracts import canonical,digest
from rewind_prepare.converter import sha
from rewind_prepare.temporal_selection import seal
from rewind_v1 import v2_contract as contract,v2_cache as cache,v2_retention as retention,v2_training as training
from rewind_v1.v2_budget import Ledger,ResourceLimit
from rewind_v1 import v2_space as space
from rewind_v1.run_data import FixtureData
from rewind_v1.runner import _read_checkpoint
from scripts.build_v2 import notebook


class V2Tests(TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        root=contract.ROOT/'outputs/test_tmp';root.mkdir(parents=True,exist_ok=True)
        tmp=tempfile.TemporaryDirectory(dir=root);self.addCleanup(tmp.cleanup);self.root=Path(tmp.name)
        self.cfg=contract.config();self.spec=FixtureData().spec;self.owners={};records=[]
        for scenario in contract.SCENARIOS:
            for split in ('train','validation'):
                seq=scenario+'/Day/avatar/motion/003/'+split;self.owners[seq]=split
                for frame in range(2):
                    files={}
                    for role in ('left','right','pose'):
                        name=seq+(f'/json/frame_{frame}.json' if role=='pose' else f'/fisheye_final_image/camera_{role}/final_{frame}.png')
                        p=self.root/'raw'/name;p.parent.mkdir(parents=True,exist_ok=True);raw=(role+str(frame)).encode();p.write_bytes(raw)
                        files[role]={'path':name,'size_bytes':len(raw),'sha256':sha(raw)}
                    records.append({'sequence_id':seq,'frame_index':frame,'split':split,'files':files})
        records.sort(key=lambda r:(r['sequence_id'],r['frame_index']))
        self.inventory=seal({'schema':'rewind_v2_frozen_inventory_v1','usable_keys':[[r['sequence_id'],r['frame_index']] for r in records],
            'sequences':sorted(self.owners),'failed':False,'capped':False,'missing':[],'duplicates':[],'excluded':[]})
        self.split_sources={'fixture':True}
        self.doc=seal({'schema':'rewind_v2_audited_manifest_v1','scenarios':list(contract.SCENARIOS),'spec_sha256':digest(self.spec),
            'official_splits':self.split_sources,'coordinates':'device_mm','records':records,'inventory':self.inventory})
        self.bind={'manifest':self.doc['content_sha256'],'source':'fixture'}
        self.ledger=Ledger(self.root/'usage',self.bind,self.cfg)
        for module in (cache,retention):
            p=patch.object(module,'CHUNK',2);p.start();self.addCleanup(p.stop)

    def build_frame(self,raw,row,spec):
        j=len(spec['joint_names']);frame=row['frame_index']
        return torch.full((2,3,128,128),(frame%10)/10),{'xy':torch.ones(2,j,2)*30,'xy_valid':torch.ones(2,j,dtype=torch.bool),
            'target_mm':torch.arange(j*3,dtype=torch.float32).reshape(j,3)+frame,'target_valid':torch.ones(j,dtype=torch.bool),
            'camera':torch.zeros(2,9),'rotation64':torch.eye(3,dtype=torch.float64).repeat(2,1,1),
            'translation64':torch.tensor([[-100000000.01+frame,60,0],[-100000000.01+frame,-60,0]],dtype=torch.float64)}

    def prepare(self,max_shards=None):
        with self.ledger.invocation('prepare'):
            return cache.prepare(self.doc,self.spec,self.cfg,self.root/'raw',self.root/'raw',self.root/'cache',self.ledger,frame_builder=self.build_frame,max_shards=max_shards)

    def test_time_boundary_resource_status_ledger_resume_and_import(self):
        clock=[0.];ledger=Ledger(self.root/'clock',self.bind,self.cfg,lambda:clock[0])
        with self.assertRaises(TimeoutError),ledger.invocation('prepare'):
            ledger.reserve('read',20);ledger.observed('read',10);clock[0]=27480;ledger.check()
        first=ledger.read();self.assertEqual(first['invocations'][0]['status'],'paused_time')
        self.assertEqual(first['invocations'][0]['read_bytes_observed'],10)
        with ledger.invocation('again'):
            self.assertEqual(ledger.totals()['read_bytes_observed'],10)
            with self.assertRaises(ValueError):ledger.observed('read',1)
            clock[0]+=28800
            with self.assertRaises(TimeoutError):ledger.remaining(terminal=True)
        other=Ledger(self.root/'other',self.bind,self.cfg)
        with other.invocation('restore'):
            usage=json.loads(ledger.path.read_bytes());other.import_usage(usage);before=other.totals();other.import_usage(usage)
            self.assertEqual(other.totals()['read_bytes_observed'],before['read_bytes_observed'])
        with self.assertRaises(ResourceLimit),self.ledger.invocation('bytes'):self.ledger.reserve('read',self.cfg['resources']['read_bytes']+1)
        self.assertEqual(self.ledger.read()['invocations'][-1]['status'],'paused_resource')
        bad=deepcopy(self.cfg);bad['cumulative_limits']={}
        with self.assertRaises(ValueError):Ledger(self.root/'bad',{},bad)

    def test_manifest_path_split_coverage_and_cost_plan(self):
        contract.validate(self.doc,self.spec,self.owners,self.split_sources)
        def rejected(change):
            d=deepcopy(self.doc);d.pop('content_sha256');change(d);d=seal(d)
            with self.assertRaises(ValueError):contract.validate(d,self.spec,self.owners,self.split_sources)
        rejected(lambda d:d['records'][0]['files']['left'].update(path=d['records'][0]['files']['right']['path']))
        rejected(lambda d:d['records'].pop())
        rejected(lambda d:d['records'][0].update(split='test'))
        p=contract.plan(self.doc,self.spec,self.cfg,self.root)
        self.assertEqual(p['image_bytes'],len(self.doc['records'])*393216);self.assertTrue(p['resource_fit'])
        tiny=deepcopy(self.cfg);tiny['resources']['read_bytes']=10
        self.assertGreater(contract.plan(self.doc,self.spec,tiny,self.root)['minimum_source_read_invocations'],1)

    def test_first_rename_interrupt_reconciles_and_old_catalog_never_shrinks(self):
        real=cache.catalog;calls=[0]
        def interrupted(*args):
            calls[0]+=1
            if calls[0]==2:raise KeyboardInterrupt
            return real(*args)
        with patch.object(cache,'catalog',side_effect=interrupted),self.assertRaises(KeyboardInterrupt):self.prepare()
        self.assertTrue((self.root/'cache/shard-00000').is_dir())
        self.prepare();old=(self.root/'cache/catalog.json').read_bytes();r=cache.receipt
        def fail_second(root,n,*args):
            if n==1:raise KeyboardInterrupt
            return r(root,n,*args)
        with patch.object(cache,'receipt',side_effect=fail_second),self.assertRaises(KeyboardInterrupt):self.prepare()
        self.assertEqual((self.root/'cache/catalog.json').read_bytes(),old)

    def test_partial_reuse_corruption_quarantine_and_restore(self):
        partial=self.prepare(2);self.assertEqual(partial['completed_frames'],4)
        with self.ledger.invocation('copy'):one=retention.user_copy_cache(self.root/'cache',self.root/'retained',self.doc,self.spec,self.cfg,self.ledger)
        with self.ledger.invocation('copy'):two=retention.user_copy_cache(self.root/'cache',self.root/'retained',self.doc,self.spec,self.cfg,self.ledger)
        self.assertEqual(one['copied_shards'],2);self.assertEqual(two['copied_shards'],0)
        remote_sha=sha((self.root/'retained/catalog.json').read_bytes())
        with self.ledger.invocation('restore'):
            restored=retention.restore_cache(self.root/'retained',remote_sha,self.root/'restored',self.doc,self.spec,self.cfg,self.ledger)
        self.assertEqual(restored['status'],'local_partial')
        corrupt=self.root/'cache/shard-00000/targets.pt';corrupt.write_bytes(b'corrupt')
        with self.assertRaises(ValueError):self.prepare()
        before=sha((corrupt.parent/'receipt.json').read_bytes());quarantined=cache.quarantine(self.root/'cache','shard-00000',before)
        self.assertTrue(Path(quarantined).exists())
        with self.ledger.invocation('restore'):
            retention.restore_cache(self.root/'retained',remote_sha,self.root/'cache',self.doc,self.spec,self.cfg,self.ledger)
        self.prepare()

    def test_final_copy_failure_never_publishes_catalog_or_partial(self):
        self.prepare(1);real=retention.copy_verified
        with patch.object(retention,'copy_verified',side_effect=OSError('injected_copy')):
            with self.assertRaises(OSError),self.ledger.invocation('copy'):
                retention.user_copy_cache(self.root/'cache',self.root/'retained',self.doc,self.spec,self.cfg,self.ledger)
        self.assertFalse((self.root/'retained/catalog.json').exists())
        with self.ledger.invocation('copy'):
            self.assertEqual(retention.user_copy_cache(self.root/'cache',self.root/'retained',self.doc,self.spec,self.cfg,self.ledger)['copied_shards'],1)

    def test_retained_catalog_monotone_binding_and_large_prefix_progress(self):
        self.prepare(1);short=(self.root/'cache/catalog.json').read_bytes();self.prepare(2)
        with self.ledger.invocation('copy'):retention.user_copy_cache(self.root/'cache',self.root/'retained',self.doc,self.spec,self.cfg,self.ledger)
        remote=self.root/'retained/catalog.json';before=remote.read_bytes()
        (self.root/'cache/catalog.json').write_bytes(short)
        with self.assertRaisesRegex(ValueError,'downgrade'),self.ledger.invocation('copy'):
            retention.user_copy_cache(self.root/'cache',self.root/'retained',self.doc,self.spec,self.cfg,self.ledger)
        self.assertEqual(remote.read_bytes(),before)
        self.prepare(3)
        small=deepcopy(self.cfg);small['resources']['read_bytes']=1800000
        limited=Ledger(self.root/'limited',self.bind,small)
        with limited.invocation('copy'):
            result=retention.user_copy_cache(self.root/'cache',self.root/'retained',self.doc,self.spec,small,limited)
        self.assertEqual(result['copied_shards'],1);self.assertEqual(result['verified_shards'],3)
        after=remote.read_bytes();changed=deepcopy(self.doc);changed['content_sha256']='f'*64
        with self.assertRaises(ValueError),self.ledger.invocation('copy'):
            retention.user_copy_cache(self.root/'cache',self.root/'retained',changed,self.spec,self.cfg,self.ledger)
        self.assertEqual(remote.read_bytes(),after)

    def test_restore_second_recheck_interrupt_keeps_catalog_and_empty_catalog(self):
        self.prepare(0)
        with self.ledger.invocation('copy'):
            empty=retention.user_copy_cache(self.root/'cache',self.root/'retained',self.doc,self.spec,self.cfg,self.ledger)
        self.assertEqual(empty['verified_shards'],0)
        self.prepare(2)
        with self.ledger.invocation('copy'):retention.user_copy_cache(self.root/'cache',self.root/'retained',self.doc,self.spec,self.cfg,self.ledger)
        catsha=sha((self.root/'retained/catalog.json').read_bytes())
        with self.ledger.invocation('restore'):retention.restore_cache(self.root/'retained',catsha,self.root/'restored',self.doc,self.spec,self.cfg,self.ledger)
        old=(self.root/'restored/catalog.json').read_bytes();real=retention.receipt
        def interrupted(root,n,*args):
            if n==1:raise KeyboardInterrupt
            return real(root,n,*args)
        with patch.object(retention,'receipt',side_effect=interrupted),self.assertRaises(KeyboardInterrupt),self.ledger.invocation('restore'):
            retention.restore_cache(self.root/'retained',catsha,self.root/'restored',self.doc,self.spec,self.cfg,self.ledger)
        self.assertEqual((self.root/'restored/catalog.json').read_bytes(),old)

    def test_immutable_ledger_generation_and_tamper_rejection(self):
        with self.ledger.invocation('one'):self.ledger.reserve('read',1)
        old=self.ledger.path;before=old.read_bytes()
        again=Ledger(self.root/'usage',self.bind,self.cfg)
        with again.invocation('two'):again.reserve('read',2)
        self.assertEqual(old.read_bytes(),before);self.assertNotEqual(old,again.path)
        self.assertEqual(again.read()['invocations'][1]['read_bytes_reserved'],2)
        with again.invocation('compact'):
            for _ in range(12):again.persist()
        self.assertLessEqual(len(list((self.root/'usage').glob('ledger-????????.json'))),4)
        (self.root/'usage/.publish-uncommitted').write_bytes(b'incomplete')
        reopened=Ledger(self.root/'usage',self.bind,self.cfg);self.assertEqual(reopened.generation,again.generation)
        raw=bytearray(again.path.read_bytes());raw[-2]^=1;again.path.write_bytes(raw)
        with self.assertRaises((ValueError,json.JSONDecodeError)):again.read()
        with self.assertRaises((ValueError,json.JSONDecodeError)):Ledger(self.root/'usage',self.bind,self.cfg).read()

    def test_ledger_crosses_100000_and_snapshot_is_one_bounded_read(self):
        with self.ledger.invocation('baseline'):pass
        doc=self.ledger.snapshot();doc.pop('content_sha256');doc['generation']=99999
        folder=self.root/'high-generation';folder.mkdir();(folder/'ledger-00099999.json').write_bytes(canonical(seal(doc)))
        high=Ledger(folder,self.bind,self.cfg)
        with high.invocation('next'):high.reserve('read',1)
        self.assertGreaterEqual(high.generation,100000)
        import rewind_v1.v2_budget as budget
        original=budget._regular_read
        with patch.object(budget,'_regular_read',wraps=original) as read:
            snapshot=high.snapshot();self.assertEqual(snapshot['generation'],high.generation)
            self.assertEqual(read.call_count,1);self.assertEqual(read.call_args.args[1],8*1024**2)
        high.path.write_bytes(b'x'*(8*1024**2+1))
        with self.assertRaises(ValueError):high.snapshot()

    def test_action_space_does_not_reserve_installed_cache_twice(self):
        self.prepare(2);plan=contract.plan(self.doc,self.spec,self.cfg,self.root)
        paths={'work':self.root/'work','cache':self.root/'cache','retained':self.root/'remote'}
        from types import SimpleNamespace
        # Far less than full working space, enough for one incremental copy and staging.
        with patch.object(space.shutil,'disk_usage',return_value=SimpleNamespace(free=128*1024**2)):
            result=space.requirements('copy-cache',plan,self.cfg,paths)
            self.assertTrue(result['fit']);self.assertLess(sum(v['additional_bytes'] for v in result['volumes']),128*1024**2)
            prepared=space.requirements('prepare',plan,self.cfg,paths)
            self.assertTrue(prepared['fit'])
        with patch.object(space.shutil,'disk_usage',return_value=SimpleNamespace(free=1)):
            self.assertFalse(space.requirements('copy-cache',plan,self.cfg,paths)['fit'])

    def test_windows_actual_cli_second_invocation_preserves_usage(self):
        splits=self.root/'splits';splits.mkdir();spec=deepcopy(self.spec);sources={}
        for split in ('train','validation','test'):
            seqs=sorted(s for s,p in self.owners.items() if p==split)
            if split=='test':seqs=[contract.SCENARIOS[0]+'/Day/avatar/motion/003/test']
            raw=('\n'.join(seqs)+'\n').encode();(splits/(split+'.txt')).write_bytes(raw);sources[split]={'sha256':sha(raw),'rows':len(seqs)}
        spec['pinned_sources']={'split_sha256':{s:p['sha256'] for s,p in sources.items()}}
        doc=deepcopy(self.doc);doc.pop('content_sha256');doc.update(spec_sha256=digest(spec),official_splits=sources)
        # Use production128-pair shards across subprocesses, not the small in-process seam.
        seq=doc['records'][0]['sequence_id']
        for frame in range(2,130):
            files={}
            for role in ('left','right','pose'):
                name=seq+(f'/json/frame_{frame}.json' if role=='pose' else f'/fisheye_final_image/camera_{role}/final_{frame}.png')
                path=self.root/'raw'/name;raw=(role+str(frame)).encode();path.write_bytes(raw)
                files[role]={'path':name,'size_bytes':len(raw),'sha256':sha(raw)}
            doc['records'].append({'sequence_id':seq,'frame_index':frame,'split':self.owners[seq],'files':files})
        doc['records'].sort(key=lambda r:(r['sequence_id'],r['frame_index']))
        inventory=deepcopy(doc['inventory']);inventory.pop('content_sha256');inventory['usable_keys']=[[r['sequence_id'],r['frame_index']] for r in doc['records']]
        doc['inventory']=seal(inventory);doc=seal(doc)
        manifest=self.root/'manifest.json';manifest.write_bytes(canonical(doc));sp=self.root/'spec.json';sp.write_bytes(canonical(spec))
        args=[sys.executable,'-m','rewind_v1.v2_cli','plan','--config-sha256',sha((contract.ROOT/'configs/rewind_v2.json').read_bytes()),
            '--manifest',str(manifest),'--manifest-sha256',sha(manifest.read_bytes()),'--spec',str(sp),'--spec-sha256',sha(sp.read_bytes()),
            '--splits',str(splits),'--work',str(self.root/'cli'),'--source-sha256',digest(contract.source())]
        for _ in range(2):
            completed=subprocess.run(args,cwd=contract.ROOT,capture_output=True,text=True,timeout=60)
            self.assertEqual(completed.returncode,0,completed.stderr)
        ledger=Ledger(self.root/'cli/usage',{'manifest':doc['content_sha256'],'config':digest(self.cfg),'source':digest(contract.source())},self.cfg)
        self.assertEqual(len(ledger.read()['invocations']),2)
        # Actual CLI copy-cache call, expand the local prefix, then call it again.
        if space.available_ram()<self.cfg['resources']['rss_bytes']:self.skipTest('Host RAM below configured4GiB; CLI plan history checked')
        self.doc,self.spec=doc,spec
        production=patch.object(cache,'CHUNK',128);production.start();self.addCleanup(production.stop)
        self.prepare(1)
        copy_args=list(args);copy_args[3]='copy-cache';copy_args += ['--cache',str(self.root/'cache'),'--retained',str(self.root/'cli-retained')]
        first=subprocess.run(copy_args,cwd=contract.ROOT,capture_output=True,text=True,timeout=60)
        self.assertEqual(first.returncode,0,first.stderr);self.assertEqual(json.loads(first.stdout)['copied_shards'],1)
        self.prepare(2)
        second=subprocess.run(copy_args,cwd=contract.ROOT,capture_output=True,text=True,timeout=60)
        self.assertEqual(second.returncode,0,second.stderr);self.assertEqual(json.loads(second.stdout)['copied_shards'],1)
        self.assertEqual(json.loads(second.stdout)['verified_shards'],2)

    def test_camera_float64_anchor_padding_reset_and_local_io(self):
        self.prepare();plan=contract.plan(self.doc,self.spec,self.cfg,self.root)
        with self.ledger.invocation('inspect'):
            data=training.Data(self.doc,self.spec,self.cfg,plan,self.root/'cache',self.ledger)
            net=torch.nn.Linear(1,1).eval().requires_grad_(False)
            def features(model,images):return torch.ones(len(images),len(self.spec['joint_names'])*14)
            with patch.object(training,'predicted_features',side_effect=features):
                key=data.clips['train'][0];clip=data.clip(key,'train',net,[128,128],'f'*64)
                self.assertTrue(torch.equal(clip.camera[:,2:],torch.zeros_like(clip.camera[:,2:])))
                self.assertEqual(clip.time_valid.sum(),2);self.assertAlmostEqual(float(clip.camera[0,0,0,6]),0.,places=5)
                self.assertAlmostEqual(clip.timings['camera_anchor_world_mm'][0],100000000.01,places=5)
                other=data.clip(data.clips['train'][1],'train',net,[128,128],'f'*64)
                self.assertTrue(torch.equal(clip.camera,other.camera))
            data.cache.close();self.assertGreater(self.ledger.totals()['local_read_logical_bytes'],0)

    def test_notebook_worker_deadlines_and_training_gate(self):
        cfg=self.cfg
        self.assertEqual(cfg['seconds_per_invocation'],28800)
        self.assertEqual(cfg['terminal_reserve_seconds'],cfg['package_seconds']+cfg['copy_seconds']+cfg['checkpoint_grace_seconds'])
        self.assertGreater(cfg['external_timeout_seconds'],28800)
        doc=notebook({'source_name':'source.zip','source_sha256':'a'*64,'revision':'b'*40,'source_identity':'c'*64})
        for cell in doc['cells']:
            if cell['cell_type']=='code':compile(''.join(cell['source']),'v2notebook','exec')
        self.assertIn('RUN_OPERATION=False',''.join(doc['cells'][-1]['source']))
        self.assertIn('USER_EXPLICIT_CACHE_COPY=False',''.join(doc['cells'][-1]['source']))
        with self.assertRaisesRegex(ValueError,'decisions_pending'):contract.training_ready(cfg)

    def tiny_runtime(self,*args):
        cfg=json.loads((contract.ROOT/'configs/rewind_v12_full_training.json').read_bytes());cfg['epochs']={r:1 for r in ('frontend','body','hand')}
        cfg['teacher'].update(hidden=8,heads=2,graph_layers=1,temporal_layers=1,diffusion_steps=20,sampling_steps=2)
        cfg['limits']['checkpoint_every']=1;return cfg

    def test_all_stages_resume_exact_and_new_vm_result_restore(self):
        self.prepare();plan=contract.plan(self.doc,self.spec,self.cfg,self.root)
        def pipeline(name,features):
            run=self.root/name;run.mkdir(exist_ok=True)
            pipeline=training.V2Pipeline(self.cfg,self.doc,self.spec,plan,self.root/'cache',self.root/features,torch.device('cpu'),run,self.ledger)
            self.addCleanup(pipeline.data.cache.close);return pipeline
        with patch.object(training,'runtime_config',side_effect=self.tiny_runtime):
            with self.ledger.invocation('train-fixture'):
                whole=pipeline('whole','whole-features');whole.run();whole.data.cache.close()
            with self.ledger.invocation('train-fixture'):
                partial=pipeline('partial','partial-features');partial.run(stop_after_updates=2);pin=partial.last['sha256'];partial.data.cache.close()
                archive=self.root/'partial.zip';archive_pin=retention.package_run(partial.root,archive,self.cfg,self.ledger)
            with self.ledger.invocation('restore-fixture'):
                probe=pipeline('probe','restored-features')
                restored_pin=retention.restore_run(archive,archive_pin,self.root/'restored-run',self.cfg,self.ledger,probe.binding);probe.data.cache.close()
                restored=pipeline('restored-run','restored-features');restored.run(restored_pin);restored.data.cache.close()
                a=_read_checkpoint(whole.root/'last.pt',whole.last['sha256'],256*1024**2)
                b=_read_checkpoint(restored.root/'last.pt',restored.last['sha256'],256*1024**2)
                def exact(x,y):
                    if torch.is_tensor(x):self.assertTrue(torch.equal(x,y))
                    elif isinstance(x,dict):
                        self.assertEqual(x.keys(),y.keys())
                        for k in x:exact(x[k],y[k])
                    elif isinstance(x,(tuple,list)):
                        self.assertEqual(len(x),len(y))
                        for u,v in zip(x,y):exact(u,v)
                    else:self.assertEqual(x,y)
                exact(a['fit'],b['fit'])
                for key,tensor in a['fit']['model'].items():self.assertTrue(torch.equal(tensor,b['fit']['model'][key]))
                self.assertTrue(torch.equal(a['fit']['rng']['torch_cpu'],b['fit']['rng']['torch_cpu']))
                self.assertEqual(a['progress']['updates'],b['progress']['updates'])
                wrong=deepcopy(probe.binding);wrong['environment']['device_name']='differentGPU'
                with self.assertRaisesRegex(ValueError,'binding_environment'):retention.restore_run(archive,archive_pin,self.root/'wrong',self.cfg,self.ledger,wrong)
