"""고정 실제 g16의 오프라인 checkpoint 회귀. 새 원본 coverage/원격 실행 없음."""
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from zipfile import ZipFile

from rewind_archive.contracts import canonical,digest
from rewind_archive_index.core import ArchiveIndex
from rewind_archive_index.drive import DriveRuntime
from rewind_archive_index.source import IndexFailure,IndexLimits,MIB
from rewind_archive_index.snapshots import import_snapshot,export_snapshot,SnapshotLimits
from rewind_discovery.runner import PreindexLimits,preindex_storage_plan
from rewind_discovery.preindex import sha_file
from rewind_discovery.recovery import recover
import test_preindex as preindex_fixtures

ROOT=Path(__file__).resolve().parents[1]
RESULT=ROOT/'outputs/test_archvis_preindex_g9_001_20261103/archvis_rgb_preindex_g9_001_result.zip'
RESULT_SHA='ddc3762406e35ac6faa2353f3e7afff1bb9c4c2e58377b9e834bcabd7a049423'
SNAPSHOT='2842dea84a61251340880455393d822d4cf0b0e3275a801cf37a72638160be4c'
RECEIPT_SHA='c8533610fa5927cf9909ba6e2dc9258a023468f15071ba084ec78780c51bb316'


@unittest.skipUnless(RESULT.exists(),'보존된 실제 g16 결과 ZIP 필요')
class G16StorageTests(unittest.TestCase):
    def test_actual_g16_old_failure_new_commit_snapshot_recover_within768(self):
        parent=ROOT/'outputs/test_tmp';parent.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(dir=parent) as temp:
            base=Path(temp);inputs=base/'inputs';inputs.mkdir();work=base/'work';work.mkdir();snapshots=base/'snapshots'
            self.assertEqual(sha_file(RESULT),RESULT_SHA)
            with ZipFile(RESULT) as z:
                filename='g16-'+SNAPSHOT+'.zip'
                raw=z.read('snapshot/'+filename);receipt_raw=z.read('snapshot/'+filename+'.receipt.json')
            self.assertEqual(hashlib.sha256(raw).hexdigest(),SNAPSHOT)
            self.assertEqual(hashlib.sha256(receipt_raw).hexdigest(),RECEIPT_SHA)
            (inputs/filename).write_bytes(raw);receipt_path=inputs/(filename+'.receipt.json');receipt_path.write_bytes(receipt_raw)
            receipt=json.loads(receipt_raw)
            with patch.object(DriveRuntime,'source',side_effect=AssertionError('offline only')):
                import_snapshot(receipt_path,work/'index.sqlite',expected_source_sha256=receipt['source_sha256'],
                    expected_snapshot_id=SNAPSHOT,min_generation=16,known_anchor=receipt,index_limits=IndexLimits(max_disk_bytes=768*MIB))
                idx=ArchiveIndex(work/'index.sqlite',IndexLimits(max_disk_bytes=151060480))
                state,blob=idx._snapshot();before=sha_file(idx.path)
                self.assertEqual(len(state['members']),494)
                estimate=3*(idx.path.stat().st_size+len(canonical(state))+len(blob)+16384)
                self.assertGreater(estimate,151060480)
                with self.assertRaisesRegex(IndexFailure,'checkpoint_disk_limit'):
                    idx._commit(state['generation'],state,blob)
                self.assertEqual(sha_file(idx.path),before)
                limits=PreindexLimits();plan=preindex_storage_plan(limits,idx.limits,idx.path.stat().st_size,0,receipt['bytes'])
                self.assertLessEqual(plan['required_total_bytes'],768*MIB)
                self.assertGreater(plan['checkpoint_budget_bytes'],estimate)
                idx.limits=replace(idx.limits,max_disk_bytes=plan['checkpoint_budget_bytes'])
                idx._commit(state['generation'],state,blob)  # fixture상 generation만 증가, 새 scan이 아니다.
                after,_=idx._snapshot()
                self.assertEqual(after['members'],state['members']);self.assertEqual(after['source'],state['source'])
                exported=export_snapshot(idx,snapshots,known_anchor=receipt,limits=SnapshotLimits(max_disk_bytes=plan['checkpoint_budget_bytes']))
                report={'job_id':'offline_g16_fixture','selection_sha256':digest({'fixture_only':True}),
                    'status':'offline_same_state_commit_no_new_coverage','last_receipt':exported['receipt']}
                (work/'report.json').write_bytes(canonical(report))
                recovered=recover(work,snapshots,base/'result.zip',job_id=report['job_id'],
                    selection_sha256=report['selection_sha256'],disk_limit=768*MIB)
                total=sum(p.stat().st_size for p in work.rglob('*') if p.is_file())+sum(p.stat().st_size for p in snapshots.rglob('*') if p.is_file())+recovered['bytes']
                self.assertLessEqual(total,768*MIB)
                summary={'schema':'offline_g16_storage_regression_v1','original_result_sha256':RESULT_SHA,
                    'original_snapshot_id':SNAPSHOT,'members':494,'old_checkpoint_cap':151060480,
                    'same_state_peak_estimate':estimate,'plan':plan,'same_state_commit_export_recover':'passed',
                    'new_source_coverage':False,'remote_requests':0,'work_snapshot_recovery_bytes':total}
                output=ROOT/'outputs/preindex_g16_20261103/developer';output.mkdir(parents=True,exist_ok=True)
                (output/'g16_offline_regression.json').write_bytes(canonical(summary)+b'\n')
            self.assertEqual(sha_file(RESULT),RESULT_SHA)


class SnapshotCapacityTests(unittest.TestCase):
    def test_opt_in_cache16_and_invalid_cache_rejection(self):
        helper=preindex_fixtures.PreindexTests();helper.setUp();self.addCleanup(helper.doCleanups)
        with self.assertRaisesRegex(IndexFailure,'discovery_runtime_cache_policy'):
            helper.execute(runtime_cache_bytes=32*MIB)
        report=helper.execute(runtime_cache_bytes=16*MIB)
        self.assertEqual(report['status'],'completed_requested_batches')
        self.assertEqual(report['runtime_cache_bytes'],16*MIB)
        self.assertEqual(report['runtime_range_bytes'],MIB)
        self.assertLessEqual(report['body_bytes'],helper.limits.body_bytes)
        helper.recover(report)


class G16NotebookTests(unittest.TestCase):
    def test_actual_g16_notebook_preparation_and_factory_contract(self):
        import ast
        import sys
        from types import ModuleType,SimpleNamespace
        import test_discovery_notebook_upload as fixture
        notebook=json.loads((ROOT/'notebooks/archvis_rgb_preindex_g16.ipynb').read_bytes())
        old=json.loads((ROOT/'notebooks/archvis_rgb_preindex.ipynb').read_bytes())
        for c in notebook['cells']:
            if c['cell_type']=='code':ast.parse(''.join(c['source']))
        def factory(cell):
            return next(n for n in ast.parse(''.join(cell['source'])).body if isinstance(n,ast.FunctionDef) and n.name=='session_factory')
        self.assertEqual(ast.dump(factory(notebook['cells'][4])),ast.dump(factory(old['cells'][4])))
        helper=fixture.NotebookUploadTests();helper.setUp();self.addCleanup(helper.doCleanups)
        context=helper.context()
        psutil=ModuleType('psutil');psutil.virtual_memory=lambda:SimpleNamespace(available=4*1024**3)
        with patch.dict(sys.modules,{'psutil':psutil}),patch('shutil.disk_usage',return_value=SimpleNamespace(free=4*1024**3)):
            helper.execute(''.join(notebook['cells'][2]['source']),context,[{'s':helper.source},{'i':helper.input}])
            self.assertFalse(Path(context['WORKSPACE']).exists())
            helper.execute(''.join(notebook['cells'][5]['source']),context)
        self.assertEqual(len(helper.downloads),1)

    @unittest.skipUnless((ROOT/'outputs/preindex_g16_20261103/developer/inputs').exists(),'고정 g16 입력 필요')
    def test_g16_fixed_inputs_opt_in_and_old_input_default4(self):
        from rewind_discovery.preindex import preflight
        config,args=preflight(ROOT/'outputs/preindex_g16_20261103/developer/inputs',
            'ef9a45fd4074f9af0282b64e1940b5021a5632e44ce1e21ccb3840aa1ec424f4')
        self.assertEqual(args['resume']['anchor']['generation'],16)
        self.assertEqual(args['runtime_cache_bytes'],16*MIB)
        self.assertEqual(config['baseline']['keys']['stereo']['train'],224)
        self.assertEqual(config['baseline']['keys']['stereo_pose_observed']['train'],102)
        old,old_args=preflight(ROOT/'outputs/preindex_20261103/developer/inputs',
            '1a94db146a28aeddac51fc7cf59e82661358ff15ed3158aeda422de5d0ea7d44')
        self.assertEqual(old_args['runtime_cache_bytes'],4*MIB)

    def test_db_above_export_cap_stops_before_export_and_recovers_previous(self):
        helper=preindex_fixtures.PreindexTests();helper.setUp();self.addCleanup(helper.doCleanups)
        original=ArchiveIndex.scan
        def oversized(index,*args,**kwargs):
            result=original(index,*args,**kwargs)
            with index.path.open('r+b') as stream:stream.truncate(64*MIB+4096)
            return result
        with patch.object(ArchiveIndex,'scan',oversized),patch('rewind_discovery.runner.export_snapshot') as export:
            report=helper.execute()
        self.assertEqual(report['status'],'bounded_stop')
        self.assertEqual(report['failure']['reason'],'snapshot_db_limit')
        self.assertEqual(report['last_receipt'],helper.resume['anchor'])
        self.assertNotEqual(report['working_discovered'],report['durable_discovered'])
        export.assert_not_called()
        helper.recover(report)
