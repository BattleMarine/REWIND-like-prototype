"""Local v1.2 connection regressions; fixtures are not full-data/GPU results."""

import copy
from dataclasses import replace
import io
import json
from pathlib import Path
import tempfile
from unittest import TestCase
from unittest.mock import patch
from types import SimpleNamespace
import time
from contextlib import redirect_stdout

from PIL import Image
import torch

from rewind_archive.contracts import canonical, digest
from rewind_m1.training import seed_run
from rewind_prepare.converter import sha
from rewind_prepare.full_archvis import clip_coverage
from rewind_v1 import full_cache, full_contract
from rewind_v1 import full_cache_retention, full_recovery
from rewind_v1 import full_cli
from scripts.build_archvis_v12_training import notebook_document
from rewind_v1.data import frame_image
from rewind_v1.full_data import pad_clip
from rewind_v1.teacher import V1Body, V1Hand, motion_losses
from rewind_v1.full_training import FullPipeline
from rewind_v1.run_data import FixtureData
from rewind_v1.runner import Budget
import test_prepare


ROOT = Path(__file__).resolve().parents[1]


class FullTrainingTests(TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(dir=ROOT / "outputs/test_tmp")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.spec = json.loads((ROOT / "configs/real_archvis62_v1.json").read_bytes())
        self.cfg = json.loads((ROOT / "configs/rewind_v1_epoch_expansion.json").read_bytes())
        self.cfg["teacher"].update(hidden=8, heads=2, graph_layers=1, temporal_layers=1,
                                   diffusion_steps=20, attention_past_frames=4)
        seed_run(71)

    def source_fixture(self):
        fixture = test_prepare.PreparationTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        records = []
        for frame in range(3):
            fixture.add_frame(fixture.seq, frame)
            files = {"pose": fixture.raw / fixture.seq / f"json/frame_{frame}.json"}
            for side in ("left", "right"):
                file = fixture.raw / fixture.seq / f"fisheye_final_image/camera_{side}/final_{frame}.png"
                file.parent.mkdir(parents=True, exist_ok=True)
                image = Image.frombytes("RGB", (1024, 1024), bytes(range(256))*(1024*1024*3//256))
                image.putpixel((17, 19), (255, 0, 0))
                image.save(file)
                files[side] = file
            records.append({"sequence_id": fixture.seq, "frame_index": frame, "split": "train",
                            "files": {role: {"path": file.relative_to(fixture.raw).as_posix(), "size_bytes": file.stat().st_size,
                                             "sha256": sha(file.read_bytes())} for role, file in files.items()}})
        return fixture, {"records": records}, canonical(fixture.spec)

    def test_atomic_chunks_resume_exact_resize_and_source_corruption(self):
        fixture, manifest, spec_raw = self.source_fixture()
        output = self.root / "cache"
        with patch.object(full_cache, "CHUNK", 2):
            first = full_cache.prepare_frames(manifest, spec_raw, fixture.raw, fixture.raw, output, max_chunks=1)
            self.assertEqual((first["status"], first["completed_frames"]), ("paused", 2))
            original_read = full_cache.PreparationBudget.read_source
            read_frames = []
            def observed(budget, root, entry):
                read_frames.append(entry["path"])
                return original_read(budget, root, entry)
            with patch.object(full_cache.PreparationBudget, "read_source", observed):
                final = full_cache.prepare_frames(manifest, spec_raw, fixture.raw, fixture.raw, output)
            self.assertEqual(final["status"], "completed")
            self.assertEqual(len(read_frames), 3)
            self.assertTrue(all("_2." in name for name in read_frames))
            budget = full_cache.PreparationBudget()
            cached = full_cache.FrameCache(output, manifest, spec_raw, budget)
            self.addCleanup(cached.close)
            row = manifest["records"][0]
            raw = {role: (fixture.raw / entry["path"]).read_bytes() for role, entry in row["files"].items()}
            image, target = cached.frame((fixture.seq, 0), "train")
            self.assertTrue(torch.equal(image, frame_image(raw, [128, 128])))
            self.assertGreater(float(((image*255)-(image*255).round()).abs().max()), .001)
            torch.testing.assert_close(target["camera"][0, -3:], torch.tensor([-.6, 0., 0.]))
            with self.assertRaisesRegex(ValueError, "split"):
                cached.frame((fixture.seq, 0), "validation")
            # Cached source is not reread, but a changed prepared chunk must be rejected.
            cached.close()
            (output / "chunk-00000/images.npy").write_bytes(b"damaged")
            with self.assertRaises(ValueError):
                full_cache.FrameCache(output, manifest, spec_raw, full_cache.PreparationBudget())
        # SHA checking applies to every new source byte consumed.
        entry = manifest["records"][0]["files"]["pose"]
        file = fixture.raw / entry["path"]
        raw = bytearray(file.read_bytes()); raw[-2] ^= 1; file.write_bytes(raw)
        with self.assertRaisesRegex(ValueError, "source_sha"):
            full_cache.PreparationBudget().read_source(fixture.raw, entry)

    def test_partial_write_failure_leaves_only_previous_committed_chunk(self):
        fixture, manifest, spec_raw = self.source_fixture()
        output = self.root / "cache"
        with patch.object(full_cache, "CHUNK", 2):
            full_cache.prepare_frames(manifest, spec_raw, fixture.raw, fixture.raw, output, max_chunks=1)
            with patch.object(full_cache.torch, "save", side_effect=OSError("fixture partial write")):
                with self.assertRaisesRegex(OSError, "partial write"):
                    full_cache.prepare_frames(manifest, spec_raw, fixture.raw, fixture.raw, output)
            self.assertTrue((output / "chunk-00000/receipt.json").is_file())
            self.assertFalse((output / "chunk-00001").exists())
            self.assertFalse(list(output.glob(".pending-*")))
            final = full_cache.prepare_frames(manifest, spec_raw, fixture.raw, fixture.raw, output)
            self.assertEqual(final["completed_frames"], 3)

    def test_padding_prefix_loss_gradient_and_future_invariance(self):
        real, joints = 3, 4
        features, camera = torch.randn(real, joints*14), torch.randn(real, 2, 9)
        target, valid = torch.randn(real, joints, 3)*100, torch.ones(real, joints, dtype=torch.bool)
        valid[1, 1] = False
        clip = pad_clip(features, camera, target, valid, sequence="fixture", start=11, split="train", input_sha256="f"*64)
        model = V1Body(self.cfg["teacher"], joints, [(0, 1), (1, 2), (2, 3)])
        noisy, k = torch.randn(1, 50, joints, 3), torch.tensor([7])
        def run(length, altered=False):
            model.zero_grad(set_to_none=True)
            f, c, x, y = clip.features.clone(), clip.camera.clone(), noisy.clone(), clip.target_mm.clone()
            if altered:
                f[:, real:] = 12345
                c[:, real:] = -7890
                x[:, real:] = 4567
                y[:, real:] = 1e9
            out, intermediate = model(x[:, :length], f[:, :length], c[:, :length], k, clip.time_valid[:, :length])
            loss, metrics = motion_losses(out, intermediate, y[:, :length], clip.target_valid[:, :length],
                                          clip.time_valid[:, :length], scale_mm=100., frame_weight=1.)
            loss.backward()
            return out.detach(), loss.detach(), {n:p.grad.detach().clone() for n,p in model.named_parameters() if p.grad is not None}
        short = run(real)
        for result in (run(50), run(50, True)):
            torch.testing.assert_close(short[0], result[0][:, :real], atol=2e-5, rtol=2e-5)
            torch.testing.assert_close(short[1], result[1], atol=1e-6, rtol=1e-6)
            for name in short[2]:
                torch.testing.assert_close(short[2][name], result[2][name], atol=2e-5, rtol=2e-5)
        # Explicit denominator: two squared-error heads, valid xyz components only.
        ones = torch.ones_like(clip.target_mm)
        loss, _ = motion_losses(ones, ones, torch.zeros_like(ones), clip.target_valid, clip.time_valid,
                                scale_mm=100., frame_weight=1.)
        self.assertEqual(float(loss), 2.)

    def test_gap_sequence_boundaries_and_no_duplicate_real_coverage(self):
        keys = {"train": [("a", i) for i in list(range(3))+list(range(4, 57))] + [("b", i) for i in range(2)],
                "validation": [("v", i) for i in range(50)]}
        summary, rows = clip_coverage(keys)
        train = [r for r in rows if r["split"] == "train"]
        self.assertEqual([(r["sequence"], r["start_frame"], r["real_frames"]) for r in train],
                         [("a", 0, 3), ("a", 4, 50), ("a", 54, 3), ("b", 0, 2)])
        self.assertEqual(summary["train"]["frame_number_gaps"], 1)
        covered = [(r["sequence"], i) for r in train for i in range(r["start_frame"], r["start_frame"]+r["real_frames"])]
        self.assertEqual(sorted(covered), sorted(keys["train"]))
        self.assertEqual(len(covered), len(set(covered)))

    def test_baseline_seeds_and_config_are_preserved(self):
        baseline=json.loads((ROOT/'configs/rewind_v1_epoch_expansion.json').read_bytes())
        full=json.loads((ROOT/'configs/rewind_v12_full_training.json').read_bytes())
        for key in ('seed','epochs','frontend','teacher'):self.assertEqual(full[key],baseline[key])
        self.assertEqual(full['limits']['max_updates'],115720)

    def test_disk_features_and_stochastic_pipeline_resume_match(self):
        cfg = copy.deepcopy(self.cfg)
        cfg["epochs"] = dict.fromkeys(cfg["epochs"], 1)
        cfg["frontend"]["size"] = [32, 32]
        cfg["limits"].update(max_updates=100, checkpoint_every=1)
        def experiment(name):
            data = FixtureData()
            data.real_frames = lambda key, split: data.width
            data.budget = Budget(cfg["limits"])
            folder = self.root / name
            folder.mkdir(exist_ok=True)
            return FullPipeline(cfg, digest(cfg), data, torch.device("cpu"), folder, data.budget, self.root / (name+"-features"))
        full = experiment("full")
        with patch('rewind_v1.runner.seed_run',wraps=seed_run) as seeded:
            expected = full.run()
        self.assertEqual([call.args[0] for call in seeded.call_args_list],[71,72,73])
        # Stop inside body after frontend; the next invocation must reuse sealed disk features.
        first = experiment("resume").run(stop_after_updates=5)
        resumed = experiment("resume")
        with patch.object(resumed.data, "clip", side_effect=AssertionError("feature regeneration")):
            actual = resumed.run(first["last_checkpoint"]["sha256"])
        self.assertEqual(actual["status"], "completed")
        self.assertEqual(actual["cursor"], expected["cursor"])
        left = torch.load(self.root / "full/last.pt", weights_only=True, map_location="cpu")
        right = torch.load(self.root / "resume/last.pt", weights_only=True, map_location="cpu")
        for name, tensor in left["fit"]["model"].items():
            self.assertTrue(torch.equal(tensor, right["fit"]["model"][name]), name)
        for key, state in left["fit"]["optimizer"]["state"].items():
            for name, tensor in state.items():
                self.assertTrue(torch.equal(tensor, right["fit"]["optimizer"]["state"][key][name]))
        self.assertTrue(torch.equal(left["fit"]["rng"]["torch_cpu"], right["fit"]["rng"]["torch_cpu"]))
        archive = self.root / "training.zip"
        recovered = full_recovery.recover(self.root / "resume", archive)
        with patch.object(full_recovery, "validate_full_config"):
            restored = full_recovery.restore(archive, recovered["sha256"], self.root / "cold")
        data = FixtureData(); data.real_frames = lambda key, split: data.width; data.budget = Budget(cfg["limits"])
        cold = FullPipeline(cfg, digest(cfg), data, torch.device("cpu"), self.root / "cold", data.budget,
                            self.root / "resume-features")
        self.assertEqual(cold.run(restored["last_sha256"])["cursor"], actual["cursor"])

    def test_hand_padding_prefix_loss_and_gradient_with_missing_upper(self):
        real, joints, upper_joints = 2, 3, 2
        model = V1Hand(self.cfg["teacher"], joints, [(0,1), (1,2)], upper_joints)
        noisy, features, camera = torch.randn(1,50,joints,3), torch.randn(1,50,14*joints), torch.randn(1,50,2,9)
        upper, upper_valid = torch.randn(1,50,upper_joints,3), torch.ones(1,50,upper_joints,dtype=torch.bool)
        upper_valid[:,1,0] = False
        mask, target = torch.arange(50)[None]<real, torch.randn(1,50,joints,3)*100
        valid = mask[...,None].expand(-1,-1,joints).clone(); valid[:,0,1] = False
        def run(length, mutate=False):
            model.zero_grad(set_to_none=True)
            values = [v.clone() for v in (noisy,features,camera,upper)]
            if mutate:
                for value in values: value[:,real:] = float("nan")
            out, intermediate = model(*(v[:,:length] for v in values[:3]), torch.tensor([5]), mask[:,:length],
                                      upper=values[3][:,:length],upper_valid=upper_valid[:,:length])
            loss,_=motion_losses(out,intermediate,target[:,:length],valid[:,:length],mask[:,:length],scale_mm=100.,frame_weight=1.)
            loss.backward()
            return out.detach(),loss.detach(),[p.grad.clone() for p in model.parameters() if p.grad is not None]
        small=run(real)
        for large in (run(50),run(50,True)):
            torch.testing.assert_close(small[0],large[0][:,:real],atol=2e-5,rtol=2e-5)
            torch.testing.assert_close(small[1],large[1],atol=1e-6,rtol=1e-6)
            for a,b in zip(small[2],large[2]): torch.testing.assert_close(a,b,atol=2e-5,rtol=2e-5)

    def test_user_cache_package_determinism_partial_restore_and_wrong_pin(self):
        fixture,manifest,spec_raw=self.source_fixture()
        cache,retained=self.root/'cache',self.root/'retained';retained.mkdir()
        with patch.object(full_cache,'CHUNK',2),patch.object(full_cache_retention,'CHUNK',2):
            full_cache.prepare_frames(manifest,spec_raw,fixture.raw,fixture.raw,cache,max_chunks=1)
            pack=full_cache_retention.pack_chunk(cache,manifest,spec_raw,0,self.root/'one.zip')
            catalog=full_cache_retention.catalog(full_cache.cache_binding(manifest,spec_raw),[pack])
            (retained/catalog['chunks'][0]['name']).write_bytes((self.root/'one.zip').read_bytes())
            catalog_raw=canonical(catalog)+b'\n';catalog_path=retained/'catalog.json';catalog_path.write_bytes(catalog_raw)
            result=full_cache_retention.restore_cache(retained,catalog_path,sha(catalog_raw),manifest,spec_raw,self.root/'restored-cache')
            self.assertEqual((result['status'],result['completed_frames'],result['source_rgb_pose_bytes']),('partial',2,0))
            # Zip metadata is deterministic even after new filesystem mtimes on restoration.
            again=full_cache_retention.pack_chunk(self.root/'restored-cache',manifest,spec_raw,0,self.root/'two.zip')
            self.assertEqual(pack['sha256'],again['sha256'])
            final=full_cache.prepare_frames(manifest,spec_raw,fixture.raw,fixture.raw,self.root/'restored-cache')
            self.assertEqual(final['completed_frames'],3)
            with self.assertRaisesRegex(ValueError,'older_catalog'):
                full_cache_retention.restore_cache(retained,catalog_path,sha(catalog_raw),manifest,spec_raw,self.root/'restored-cache')
            with self.assertRaisesRegex(ValueError,'file_pin'):
                full_cache_retention.restore_cache(retained,catalog_path,'0'*64,manifest,spec_raw,self.root/'wrong')

    def test_remote_ledger_terminal_reserve_copy_failure_conflict_and_corruption(self):
        archive_path=self.root/'archive.zip';archive_path.write_bytes(b'a'*150)
        archive={'output':str(archive_path),'bytes':150,'sha256':sha(archive_path.read_bytes()),'last_sha256':'f'*64}
        identity={'source':'fixture','phase':'train'}
        with patch.object(full_recovery,'ARCHIVE_LIMIT',200),patch.object(full_recovery,'REMOTE_LIMIT',1000):
            with full_recovery.RemoteResults(self.root,identity) as remote:
                remote_root=remote.root
                with self.assertRaises(OSError):
                    with full_recovery.RemoteResults(self.root,identity): pass
                with patch('scripts.v11_recovery._measure',side_effect=OSError('fixture copy failure')):
                    with self.assertRaisesRegex(OSError,'copy failure'): remote.publish(archive,periodic=True)
                self.assertTrue(archive_path.is_file())
                self.assertEqual(sum(r['bytes'] for r in remote.ledger['reservations']),150)
                for _ in range(4): remote.reserve(archive,periodic=True)
                with self.assertRaises(full_recovery.RemoteBudgetExhausted): remote.reserve(archive,periodic=True)
                terminal=remote.publish(archive,periodic=False)
                self.assertEqual(terminal['state'],'verified')
                self.assertEqual(sum(r['bytes'] for r in remote.ledger['reservations']),900)
            # Reservation accounting survives a fresh connection; corruption is not reset.
            with full_recovery.RemoteResults(self.root,identity) as reopened:
                self.assertEqual(len(reopened.ledger['reservations']),6)
                with self.assertRaises(full_recovery.RemoteBudgetExhausted):reopened.ensure_terminal_room()
            stale=b'{"token":"old","identity":{}}'
            (remote_root/'lease.json').write_bytes(stale)
            with self.assertRaises(FileExistsError):
                with full_recovery.RemoteResults(self.root,identity): pass
            self.assertEqual((remote_root/'lease.json').read_bytes(),stale)
            (remote_root/'lease.json').unlink()
            (remote_root/'ledger.json').write_bytes(b'broken')
            with self.assertRaises(ValueError):
                with full_recovery.RemoteResults(self.root,identity): pass
            self.assertEqual((remote_root/'ledger.json').read_bytes(),b'broken')

    def test_remote_budget_exhaustion_pauses_at_saved_cursor(self):
        cfg=copy.deepcopy(self.cfg);cfg['epochs']=dict.fromkeys(cfg['epochs'],1)
        cfg['frontend']['size']=[32,32];cfg['limits'].update(max_updates=100,checkpoint_every=1)
        data=FixtureData();data.budget=Budget(cfg['limits']);data.real_frames=lambda key,split:data.width
        output=self.root/'run';output.mkdir()
        def exhausted(*args,**kwargs):raise full_recovery.RemoteBudgetExhausted('fixture final reserve')
        pipeline=FullPipeline(cfg,digest(cfg),data,torch.device('cpu'),output,data.budget,self.root/'features',exhausted)
        pipeline.backup_last_time=time.monotonic()-3601
        report=pipeline.run()
        self.assertEqual(report['status'],'paused')
        self.assertEqual(report['cursor'],report['checkpoint_cursor'])
        self.assertTrue((output/'last.pt').is_file())
        self.assertEqual(report['failure']['type'],'RemoteBudgetExhausted')

    def test_notebook_compile_default_noop_and_cuda_failure_final_retention(self):
        notebook=notebook_document({})
        for i,cell in enumerate(notebook['cells']):
            if cell['cell_type']=='code':
                code=''.join(cell['source']);compile(code,f'notebook cell{i}','exec')
                if i!=1:
                    def forbidden(*args,**kwargs):raise AssertionError('default executed operation')
                    scope={'mount_drive':forbidden,'child':forbidden,'subprocess':SimpleNamespace(run=forbidden)}
                    exec(code,scope)
        cfg=self.cfg;work=self.root/'work';output=self.root/'run';drive=self.root/'drive';drive.mkdir()
        args=SimpleNamespace(config_sha256='a'*64,source_sha256='b'*64,work=work,output=output,
            features=self.root/'features',cache=self.root/'cache',inputs=self.root/'inputs',drive_root=drive,device='cuda')
        with patch.object(full_cli,'_context',return_value=(cfg,{'records':[]},b'{}',{})), \
             patch.object(full_cli.shutil,'disk_usage',return_value=SimpleNamespace(free=100*1024**3)), \
             patch.object(full_cli,'require_cuda_training',side_effect=RuntimeError('fixture CUDA guard')):
            with self.assertRaisesRegex(RuntimeError,'CUDA guard'):full_cli.launch_train(args)
        receipts=list(drive.rglob('receipt.json'))
        self.assertEqual(len(receipts),1)
        receipt=json.loads(receipts[0].read_bytes())
        self.assertEqual((receipt['state'],receipt['kind']),('verified','status'))

    def test_user_cache_copy_cell_with_local_fixture_only(self):
        fixture,manifest,spec_raw=self.source_fixture()
        cache=self.root/'cache';drive=self.root/'fake-drive';drive.mkdir()
        notebook=notebook_document({})
        code=''.join(notebook['cells'][5]['source']).replace('USER_COPY_COMPLETED_CACHE = False','USER_COPY_COMPLETED_CACHE = True',1)
        import os,sys,hashlib
        scope={'SOURCE':ROOT,'CACHE':cache,'WORK':self.root,'DRIVE_ROOT':drive,'AUDIT':self.root/'fixture.zip',
               'INPUTS':self.root,'mount_drive':lambda:None,'Path':Path,'sys':sys,'time':time,'os':os,
               'hashlib':hashlib,'tempfile':tempfile}
        with patch.object(full_cache,'CHUNK',2),patch.object(full_cache_retention,'CHUNK',2), \
             patch.object(full_contract,'load_audit',return_value=(manifest,{},{})), \
             patch.object(full_contract,'bind_inputs',return_value=(fixture.spec,spec_raw)):
            full_cache.prepare_frames(manifest,spec_raw,fixture.raw,fixture.raw,cache)
            with redirect_stdout(io.StringIO()):exec(code,scope)
            self.assertEqual(len(list(drive.rglob('chunk-*.zip'))),2)
            catalogs=sorted(drive.rglob('catalog-*.json'))
            self.assertEqual(len(catalogs),2)
            latest=catalogs[-1];doc=json.loads(latest.read_bytes())
            self.assertEqual(doc['completed_frames'],3)
            result=full_cache_retention.restore_cache(latest.parent,latest,sha(latest.read_bytes()),
                                                       manifest,spec_raw,self.root/'restored')
            self.assertEqual(result['status'],'completed')
            with redirect_stdout(io.StringIO()):exec(code,scope)
            self.assertEqual(scope['written'],0)
            corrupted=next(drive.rglob('chunk-*.zip'));raw=bytearray(corrupted.read_bytes());raw[-1]^=1;corrupted.write_bytes(raw)
            with redirect_stdout(io.StringIO()),self.assertRaisesRegex(ValueError,'mismatch'):
                exec(code,scope)
            self.assertEqual(corrupted.read_bytes(),bytes(raw))

    def test_teacher_calls_reset_between_sequences(self):
        model=V1Body(self.cfg['teacher'],4,[(0,1),(1,2),(2,3)]).eval()
        def sample():return (torch.randn(1,3,4,3),torch.randn(1,3,56),torch.randn(1,3,2,9),
                            torch.tensor([3]),torch.ones(1,3,dtype=torch.bool))
        first,second=sample(),sample()
        with torch.no_grad():
            before=model(*first)
            model(*second)
            after=model(*first)
        for a,b in zip(before,after):self.assertTrue(torch.equal(a,b))

    def test_final_archive_refreshes_status_after_periodic_budget_pause(self):
        drive=self.root/'drive';drive.mkdir()
        args=SimpleNamespace(config_sha256='a'*64,source_sha256='b'*64,work=self.root/'work',output=self.root/'run',
            features=self.root/'features',cache=self.root/'cache',inputs=self.root/'inputs',drive_root=drive,device='cuda',
            resume_sha256=None,stop_after_updates=None)
        original_publish=full_recovery.RemoteResults.publish
        def publish(remote,archive,*,periodic):
            if periodic:raise full_recovery.RemoteBudgetExhausted('fixture periodic budget')
            return original_publish(remote,archive,periodic=False)
        def pipeline(cfg,config_sha,data,device,output,budget,features,backup):
            def run(*args):
                (output/'run.json').write_bytes(canonical({'status':'running'}))
                try:backup(output,None,periodic=True)
                except full_recovery.RemoteBudgetExhausted:
                    (output/'run.json').write_bytes(canonical({'status':'paused'}))
                return {'status':'paused','cursor':{'updates':0},'last_checkpoint':None,'planned_updates':115720}
            return SimpleNamespace(run=run)
        with patch.object(full_cli,'_context',return_value=(self.cfg,{'records':[]},b'{}',{})), \
             patch.object(full_cli.shutil,'disk_usage',return_value=SimpleNamespace(free=100*1024**3)), \
             patch.object(full_cli,'require_cuda_training',return_value={'device':'cpu'}), \
             patch.object(full_cli.torch.cuda,'reset_peak_memory_stats'), \
             patch.object(full_cli,'FullData',return_value=SimpleNamespace(cache=SimpleNamespace(close=lambda:None))), \
             patch.object(full_cli,'FullPipeline',side_effect=pipeline), \
             patch.object(full_recovery.RemoteResults,'publish',new=publish):
            result=full_cli.launch_train(args)
        self.assertEqual(result['status'],'paused')
        import zipfile
        remote_archive=next(drive.rglob('training.zip'))
        with zipfile.ZipFile(remote_archive) as archive:self.assertEqual(json.loads(archive.read('run.json'))['status'],'paused')
        failed_local=list((args.work/'recovery').rglob('training.zip'))
        self.assertEqual(len(failed_local),1)
        with zipfile.ZipFile(failed_local[0]) as archive:self.assertEqual(json.loads(archive.read('run.json'))['status'],'running')

    def test_preparation_failure_archive_includes_durable_chunk_progress(self):
        drive=self.root/'drive';drive.mkdir()
        args=SimpleNamespace(config_sha256='a'*64,source_sha256='b'*64,work=self.root/'work',output=self.root/'cache',
            inputs=self.root/'inputs',drive_root=drive,rgb_root=self.root/'raw',pose_root=self.root/'raw',max_chunks=None)
        durable={'status':'failed','completed_frames':128,'source_bytes_this_invocation':1234,
                 'chunks':[{'number':0,'receipt_sha256':'c'*64}]}
        def fail(*args,**kwargs):
            (self.root/'cache/preparation.json').write_bytes(canonical(durable))
            raise OSError('fixture later chunk failure')
        with patch.object(full_cli,'_context',return_value=(self.cfg,{'records':[]},b'{}',{})), \
             patch.object(full_cli.shutil,'disk_usage',return_value=SimpleNamespace(free=100*1024**3)), \
             patch.object(full_cli,'prepare_frames',side_effect=fail):
            with self.assertRaisesRegex(OSError,'later chunk'):full_cli.launch_prepare(args)
        import zipfile
        with zipfile.ZipFile(next(drive.rglob('training.zip'))) as archive:
            self.assertEqual(json.loads(archive.read('run.json')),durable)
            self.assertEqual(json.loads(archive.read('launch_error.json'))['type'],'OSError')
