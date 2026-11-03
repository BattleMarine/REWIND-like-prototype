"""실제269 notebook 셀 준비·회수와 고정235MiB 입력 preflight. 원격0."""
import ast,json
from pathlib import Path
import unittest
import test_pose67_notebook as fixtures
from rewind_archive.contracts import digest
from rewind_archive.profiles import target269_names

ROOT=Path(__file__).resolve().parents[1]
NB=json.loads((ROOT/'notebooks/pose_missing269.ipynb').read_bytes())
PREP,PREFLIGHT,RUN,RECOVERY=(''.join(c['source']) for c in NB['cells'][2:])


class Pose269NotebookTests(unittest.TestCase):
    def setUp(self):
        self.helper=fixtures.Pose67NotebookTests();self.helper.setUp();self.addCleanup(self.helper.doCleanups)
        h=self.helper;h.targets=target269_names();h.target_sha=digest(h.targets)
        h.input=fixtures.archive({'targets.json':json.dumps(h.targets),'config.json':json.dumps({
            'job':{'exact_targets':h.targets,'includes':h.targets,'cache_expectation':h.expected},
            'targets_content_sha256':h.target_sha,'cache_zip_sha256':fixtures.sha(h.cache),'cache_zip_bytes':len(h.cache)})})

    def test_three_zip_missing_cache_no_mutation_normal_reuse_and_preparation_recovery(self):
        h=self.helper;context=h.context()
        with self.assertRaisesRegex(AssertionError,'cache ZIP'):
            h.execute(PREP,context,[{'s':h.source},{'i':h.input},{}])
        self.assertFalse(Path(context['SOURCE_ROOT']).exists())
        h.execute(PREP,context,[{'s':h.source},{'i':h.input},{'c':h.cache}])
        before=Path(context['CACHE_INPUT']).stat().st_mtime_ns
        h.execute(PREP,context,[{'s':h.source},{'i':h.input},{'c':h.cache}])
        self.assertEqual(Path(context['CACHE_INPUT']).stat().st_mtime_ns,before)
        h.execute(RECOVERY,context)
        self.assertEqual(h.downloads,[context['RECOVERY_ZIP']])
        self.assertFalse(Path(context['WORKSPACE']).exists())

    def test_ast_and_reviewed_auth_factory_same(self):
        for c in NB['cells']:
            if c['cell_type']=='code':ast.parse(''.join(c['source']))
        old=json.loads((ROOT/'notebooks/pose_missing67.ipynb').read_bytes())
        def factory(code):return next(n for n in ast.parse(code).body if isinstance(n,ast.FunctionDef) and n.name=='session_factory')
        self.assertEqual(ast.dump(factory(RUN)),ast.dump(factory(''.join(old['cells'][4]['source']))))
        self.assertNotIn('authenticate_user',RECOVERY)
        self.assertIn('pose269_review.json',RECOVERY)

    @unittest.skipUnless((ROOT/'outputs/g25_next_20261103/developer/pose_inputs/config.json').exists(),'고정269 입력 필요')
    def test_actual235_pin_and_269_preflight(self):
        from rewind_archive.workflow import make_preflight
        from rewind_archive.prefix_cache import input_spec
        from rewind_archive.cache_policy import TAIL352_POLICY
        config=json.loads((ROOT/'outputs/g25_next_20261103/developer/pose_inputs/config.json').read_bytes())
        job=config['job'];job['cache_input']=input_spec(ROOT/'outputs/test_pose_missing67_001_20261102/prefix_cache (3).zip',policy=TAIL352_POLICY)
        result=make_preflight(job,{})
        self.assertEqual(len(result['config']['exact_targets']),269)
        self.assertEqual(result['config']['cache_expectation']['prefix_bytes'],246415360)
