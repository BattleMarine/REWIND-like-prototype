import ast,copy,json,sys,tempfile,unittest
from pathlib import Path
from types import ModuleType,SimpleNamespace
from unittest.mock import patch
from rewind_fleet.planning import inventory,pin,plan,calibration_plan,validate
from rewind_fleet.__main__ import main
from rewind_fleet.runner import run
import test_indexed_pairs_notebook as auth_fixtures
import test_fleet
from test_drive_metadata import FakeDrive

ROOT=Path(__file__).resolve().parents[1]
NB=json.loads((ROOT/'notebooks/fleet_index_workbench.ipynb').read_bytes())
SETTINGS,PREP,INSPECT,PLAN,RUN,RECOVER=[''.join(c['source']) for c in NB['cells'][1:]]
class FrontendTests(unittest.TestCase):
    def setUp(self):
        tmp=tempfile.TemporaryDirectory(dir=ROOT/'outputs/test_tmp');self.addCleanup(tmp.cleanup);self.root=Path(tmp.name)
    def test_official_inventory90_depth_and_unknown_full_plan(self):
        inv=validate(json.loads((ROOT/'configs/fleet_inventory_observed.json').read_bytes()),'fleet_inventory_v1')
        self.assertEqual(len(inv['archives']),90);self.assertEqual(inv['observed_files'],354)
        self.assertEqual(inv['known_compressed_bytes'],3094462001717);self.assertTrue(inv['official_registry_complete'])
        self.assertEqual(sum(a['kind']=='depth' for a in inv['archives']),30)
        cost=plan(inv,{},[a['key'] for a in inv['archives']]);self.assertFalse(cost['complete_estimate']);self.assertIsNone(cost['cap'])
    def test_pin54_parts_uses_existing32_metadata_cap_without_media(self):
        files=[{'id':f'p{i}','name':f'Scene.tar.gz{i:03}','size_bytes':100,'modified_time':'2026-10-14T00:00:00.000Z'} for i in range(54)]
        inv=inventory({'archives':[{'category':'UnrealEgoData2_train_val_rgb','name':'Scene','files':files}]})
        session=FakeDrive({'parts':[{'file_id':f['id'],'name':f['name']} for f in files]},{f['id']:b'x'*100 for f in files})
        got=pin(inv,session,[inv['archives'][0]['key']],max_requests=54)
        self.assertEqual(got['metadata_pin_stats']['requests'],54);self.assertEqual(got['metadata_pin_stats']['archive_media_body_bytes'],0)
        self.assertTrue(all(p['version']=='9' for p in got['archives'][0]['parts']));self.assertFalse(any('alt=media' in u for u,k in session.calls))
    def test_fixed_calibration_starts_without_whole_decoded_guess(self):
        h=test_fleet.FleetTests();h.setUp();self.addCleanup(h.doCleanups)
        cost=calibration_plan(h.inv,'rgb',storage_backend='local');self.assertTrue(cost['complete_estimate'])
        self.assertEqual(cost['scope'],'calibration');self.assertEqual(cost['automatic_segments'],1)
        result=run(h.inv,cost,self.root/'store',self.root/'local',owner='probe',single_writer=True,max_segments=1,local_sources=h.sources)
        self.assertIsNone(result['pending']);self.assertFalse(result['dataset_complete'])
        self.assertIn('source_read_bytes',result['events'][0]['event']['stats'])
    def test_notebook_inventory_plan_auth_gate_and_independent_recovery(self):
        inputs=self.root/'inputs';inputs.mkdir()
        for name,src in [('inventory.json','fleet_inventory_observed.json'),('estimates.json','fleet_estimates_template.json')]:
            (inputs/name).write_bytes((ROOT/'configs'/src).read_bytes())
        context={};exec(SETTINGS,context);context.update(SOURCE_ROOT=str(ROOT),INPUT_ROOT=str(inputs),WORKSPACE=str(self.root/'work'),RECOVERY_DIRECTORY=str(self.root))
        with patch('rewind_fleet.__main__.authorized_factory',side_effect=AssertionError('unexpected auth')):
            exec(INSPECT,context);exec(PLAN,context);self.assertFalse(context['COST_PLAN']['complete_estimate'])
            context['ACTION']='index'
            with self.assertRaises(AssertionError):exec(RUN,context)
        downloads=[];colab=ModuleType('google.colab');colab.files=SimpleNamespace(download=downloads.append)
        from rewind_fleet.storage import Store
        store=Store(self.root/'store');head=store.commit({'pending':{'token':'pending-token'}});store.publish(head,None)
        (store.root/'WRITER.lease').write_text(json.dumps({'owner':'previous','token':'lease-token'}))
        context['STORE_DIR']=str(store.root)
        with patch.dict(sys.modules,{'google.colab':colab}):exec(RECOVER,context)
        self.assertEqual(len(downloads),1);self.assertTrue(Path(downloads[0]).is_file())
        from zipfile import ZipFile
        with ZipFile(downloads[0]) as z:
            state=json.loads(z.read('recovery.json'))['store_state']
            self.assertEqual(state['head'],head);self.assertEqual(state['pending']['token'],'pending-token');self.assertEqual(state['lease']['token'],'lease-token')
        self.assertTrue((store.root/'WRITER.lease').exists())
        self.assertNotIn('authenticate_user',RECOVER);self.assertNotIn('authorized_factory',RECOVER)
    def test_cli_existing_output_rejected_before_plan_or_auth(self):
        target=self.root/'existing.json';target.write_text('{}')
        with patch('rewind_fleet.__main__.authorized_factory',side_effect=AssertionError('auth')):
            with self.assertRaisesRegex(ValueError,'new_output'):
                main(['run','--inventory','missing','--plan','missing','--plan-sha256','0'*64,'--store','missing','--workspace','missing',
                    '--owner','test','--max-segments','1','--session-seconds','60','--output',str(target)])

@unittest.skipUnless(auth_fixtures.session_tests.available(),'보존 transport_deps 필요')
class FleetAuthTests(auth_fixtures.PairNotebookSessionTests):
    def factory(self,env):
        tree=ast.parse((ROOT/'rewind_fleet/__main__.py').read_text(encoding='utf8'))
        parent=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='authorized_factory')
        node=next(n for n in parent.body if isinstance(n,ast.FunctionDef) and n.name=='factory')
        exec(compile(ast.Module(body=[node],type_ignores=[]),'fleet_actual_factory','exec'),env);return env['factory']
