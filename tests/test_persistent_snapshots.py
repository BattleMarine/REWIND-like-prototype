"""모의 Drive source를 다른 local runtime 경로로 exact receipt 복원한다."""
import copy
import hashlib
import json
import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import unittest
from zipfile import ZipFile,ZIP_DEFLATED

import test_persistent_drive_index as fixtures
from rewind_archive_index import ArchiveIndex,IndexFailure
from rewind_archive_index.drive import DriveRuntime
from rewind_archive_index.snapshots import export_snapshot,import_snapshot,read_receipt


@unittest.skipUnless(importlib.util.find_spec('indexed_gzip') and importlib.util.find_spec('psutil'),'install requirements-index.txt')
class SnapshotTests(unittest.TestCase):
    setUp=fixtures.PersistentDriveTests.setUp

    def export_a(self):
        self.idx.scan(3)
        return export_snapshot(self.idx,self.root/'mount')

    def restore(self,export,target,**extra):
        a=export['receipt']
        return import_snapshot(export['receipt_path'],target,expected_source_sha256=a['source_sha256'],
            expected_snapshot_id=a['snapshot_id'],min_generation=a['generation'],**extra)

    def test_new_runtime_restore_extend_parent_and_existing_member_hash(self):
        a=self.export_a();bpath=self.root/'runtime-b/index.sqlite'
        self.restore(a,bpath,known_anchor=a['receipt'])
        b=ArchiveIndex(bpath,runtime=DriveRuntime(self.session));self.assertEqual(b.lookup('frame_0.png'),self.idx.lookup('frame_0.png'))
        self.assertEqual(b.scan(3)['members'],6)
        b.extract(['frame_0.png'],self.root/'selected-b')
        out=export_snapshot(b,self.root/'mount',known_anchor=a['receipt'])
        self.assertEqual(out['receipt']['parent_snapshot_id'],a['receipt']['snapshot_id'])
        self.assertEqual(out['receipt']['lineage'],a['receipt']['lineage'])
        with self.assertRaisesRegex(IndexFailure,'snapshot_stale'):
            self.restore(a,self.root/'stale.sqlite',known_anchor=out['receipt'])

    def test_fresh_process_snapshot_restore_has_no_runtime_path_dependency(self):
        a=self.export_a();config=self.root/'handoff.json';config.write_text(json.dumps(a),encoding='utf-8')
        script='''import json,sys;from pathlib import Path
from rewind_archive_index.snapshots import import_snapshot
from rewind_archive_index import ArchiveIndex
a=json.loads(Path(sys.argv[1]).read_text());r=a['receipt']
import_snapshot(a['receipt_path'],sys.argv[2],expected_source_sha256=r['source_sha256'],expected_snapshot_id=r['snapshot_id'],min_generation=r['generation'],known_anchor=r)
i=ArchiveIndex(sys.argv[2]);print(json.dumps({'status':i.status(),'member':i.lookup('frame_0.png')}))
'''
        env=dict(os.environ);env['PYTHONPATH']=os.pathsep.join(sys.path)
        r=subprocess.run([sys.executable,'-c',script,str(config),str(self.root/'fresh-runtime/index.sqlite')],env=env,capture_output=True,text=True)
        self.assertEqual(r.returncode,0,r.stdout+r.stderr)
        self.assertEqual(json.loads(r.stdout)['member']['sha256'],self.idx.lookup('frame_0.png')['sha256'])

    def test_corruption_target_reuse_version_and_receipt_mix_refused(self):
        a=self.export_a();target=self.root/'restored.sqlite';self.restore(a,target)
        with self.assertRaisesRegex(IndexFailure,'snapshot_target_exists'):self.restore(a,target)
        path=Path(a['receipt_path']);receipt=read_receipt(path)
        bad=copy.deepcopy(receipt);bad['schema']='unsupported';path.write_text(json.dumps(bad))
        with self.assertRaisesRegex(IndexFailure,'snapshot_receipt_schema'):self.restore(a,self.root/'bad.sqlite')
        path.write_text(json.dumps(receipt));bundle=path.parent/receipt['filename']
        with bundle.open('ab') as f:f.write(b'broken')
        with self.assertRaisesRegex(IndexFailure,'snapshot_bundle_size'):self.restore(a,self.root/'bad.sqlite')

    def test_known_same_generation_branch_conflict_and_parent_rejected(self):
        a=self.export_a();other=copy.deepcopy(a['receipt']);other['snapshot_id']='0'*64;other['filename']=f"g{other['generation']}-{other['snapshot_id']}.zip"
        with self.assertRaisesRegex(IndexFailure,'snapshot_generation_conflict'):self.restore(a,self.root/'bad.sqlite',known_anchor=other)
        with self.assertRaisesRegex(IndexFailure,'snapshot_not_advanced'):export_snapshot(self.idx,self.root/'mount')

    def test_zip_bomb_and_traversal_rejected_before_payload_load(self):
        a=self.export_a();path=Path(a['receipt_path']);bad=copy.deepcopy(a['receipt'])
        temp=self.root/'evil.zip'
        with ZipFile(temp,'w',ZIP_DEFLATED) as z:
            z.writestr('index.sqlite',b'0'*(1024**2));z.writestr('../manifest.json',b'{}')
        bad['snapshot_id']=hashlib.sha256(temp.read_bytes()).hexdigest();bad['filename']=f"g{bad['generation']}-{bad['snapshot_id']}.zip";bad['bytes']=temp.stat().st_size
        os.rename(temp,path.parent/bad['filename']);evil=path.parent/'evil.receipt.json';evil.write_text(json.dumps(bad))
        with self.assertRaisesRegex(IndexFailure,'snapshot_zip_members'):
            import_snapshot(evil,self.root/'evil.sqlite',expected_source_sha256=bad['source_sha256'],expected_snapshot_id=bad['snapshot_id'],min_generation=0)


if __name__=='__main__':unittest.main()
