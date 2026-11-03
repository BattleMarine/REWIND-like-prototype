import gzip,io,json,random,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
from rewind_fleet import window
from rewind_fleet.storage import Store,sha
from rewind_archive_index.source import make_source,SplitSource,Budget,IndexLimits,IndexFailure

ROOT=Path(__file__).resolve().parents[1]
class WindowStorageTests(unittest.TestCase):
    def setUp(self):
        tmp=tempfile.TemporaryDirectory(dir=ROOT/'outputs/test_tmp');self.addCleanup(tmp.cleanup);self.root=Path(tmp.name)
    def test_middle_bit_dictionary_split_new_handle_no_origin_read(self):
        r=random.Random(9);raw=bytes(r.randrange(64) for _ in range(2*1024**2));compressed=gzip.compress(raw,mtime=0)
        initial=io.BytesIO(compressed);g=window.open_engine(initial,65536);g.read(1400000);pack=window.export(g);g.close()
        header,points=window.decode(pack);i=next(i for i,(p,d) in enumerate(points) if p[1]>800000 and p[2]>0)
        p=points[i][0];subset=window.encode(header,[points[0],points[i]])
        cut=p[0];paths=[self.root/'0',self.root/'1'];paths[0].write_bytes(compressed[:cut]);paths[1].write_bytes(compressed[cut:])
        spec=make_source(paths,archive_id='bit',kind='rgb',dataset_version='fixture');b=Budget(IndexLimits(max_read_bytes=8*1024**2))
        source=SplitSource(spec,b);cursor=p[1]+177;g=window.open_engine(source,65536,subset,cursor)
        got=g.read(300000);new=window.export(g);g.close();source.close()
        self.assertEqual(got,raw[cursor:cursor+300000]);self.assertGreater(p[2],0)
        self.assertGreaterEqual(min(x['start'] for x in b.trace),p[0]-1);self.assertEqual({x['part'] for x in b.trace},{0,1})
        self.assertLess(len(subset),len(pack));self.assertLess(len(new),window.MAX_PACK)
    def test_orphan_stage_and_explicit_corrupt_object_repair(self):
        store=Store(self.root/'store');raw=b'normal immutable object';key=sha(raw)
        with patch('rewind_fleet.storage.os.rename',side_effect=OSError('simulated publish failure')):
            with self.assertRaises(OSError):store.put(raw)
        stages=list((store.root/'objects').glob('.stage-*'));self.assertEqual(len(stages),1)
        self.assertEqual(store.put(raw),key);self.assertTrue(stages[0].exists())
        bad=b'replacement';other=sha(bad);(store.root/'objects'/other).write_bytes(b'partial')
        with self.assertRaises(IndexFailure):store.put(bad)
        with self.assertRaises(IndexFailure):store.repair_object(other,bad)
        self.assertEqual(store.repair_object(other,bad,owner_confirmed_stopped=True),other)
        self.assertTrue(list(store.root.glob('quarantine-'+other+'-*')))
    def test_marker_key_relative_path_and_head_schema(self):
        store=Store(self.root/'store')
        for name in ('../outside','/absolute','a/../outside','C:outside','a\\b'):
            with self.assertRaises(IndexFailure):store.read(name,100)
        with patch.object(store,'read') as read:
            with self.assertRaises(IndexFailure):store.committed('../bad')
            read.assert_not_called()
        (store.root/'HEAD.json').write_text(json.dumps({'schema':'other','commit':'0'*64}))
        with self.assertRaisesRegex(IndexFailure,'head_schema'):store.head()

    def test_incremental_catalog_cache_does_not_revisit_old_chain(self):
        from rewind_fleet.scanner import cache
        from rewind_archive.contracts import canonical
        store=Store(self.root/'store')
        def commit(parent,n):return store.commit({'schema':'fleet_archive_commit_v1','parent':parent,
            'catalog':store.put(canonical({'rows':[{'name':str(n),'ordinal':n}]}))})
        first=commit(None,0);db=cache(store,first,self.root/'local','catalog');db.close()
        second=commit(first,1);calls=[];original=store.committed
        def observed(key):calls.append(key);return original(key)
        with patch.object(store,'committed',observed):db=cache(store,second,self.root/'local','catalog')
        self.assertEqual(db.execute('SELECT COUNT(*) FROM rows').fetchone()[0],2);db.close();self.assertNotIn(first,calls)

    def test_stale_lease_needs_exact_token_and_owner_stop_confirmation(self):
        from rewind_fleet.runner import release_stale_lease
        root=self.root/'store';root.mkdir();p=root/'WRITER.lease';p.write_text(json.dumps({'owner':'old','token':'known'}))
        with self.assertRaises(IndexFailure):release_stale_lease(root,'known')
        with self.assertRaises(IndexFailure):release_stale_lease(root,'wrong',owner_confirmed_stopped=True)
        self.assertTrue(p.exists());release_stale_lease(root,'known',owner_confirmed_stopped=True);self.assertFalse(p.exists())
