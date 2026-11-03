"""실제 coverage 조회에서 미탐색/부분/pending/EOF/빈 survey 집계를 검증한다."""
import copy,hashlib,tempfile,unittest
from pathlib import Path
from rewind_fleet.control import coverage
from rewind_fleet.storage import Store

ROOT=Path(__file__).resolve().parents[1]

class SurveyCoverageTests(unittest.TestCase):
    def setUp(self):
        root=ROOT/'outputs/test_tmp';root.mkdir(parents=True,exist_ok=True)
        tmp=tempfile.TemporaryDirectory(dir=root);self.addCleanup(tmp.cleanup);self.base=Path(tmp.name)

    def check_coverage(self,selected,segments,statuses,*,pending=None):
        store=Store(self.base/'store');archives={}
        for key,status in statuses.items():
            archives[key]=store.commit({'schema':'fleet_archive_commit_v1','state':{
                'ordinal':10,'files':8,'directories':2,'cursor':4096,'eof':bool(status.get('eof')),
                'trailer_crc_match':bool(status.get('eof')),'complete':False}})
        usage={'source_bytes':12345,'requests':12,'seconds':None,'artifact_io_bytes':6789}
        ledger={'schema':'fleet_ledger_v1','plan_sha256':'a'*64,'pending':pending,'settlement':None,
            'selected_archives':selected,'survey_segments':segments,'survey_quota':2,'statuses':statuses,
            'archives':archives,'complete':False,'usage':usage,'archive_usage':{},'time_policy':'observe_only',
            'observed_seconds':23456,'plan_cap':{'source_bytes':20000,'requests':100,'seconds':None,'artifact_io_bytes':20000}}
        original=copy.deepcopy(ledger);head=store.commit(ledger);store.publish(head,None)
        def hashes():return {p.relative_to(store.root).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in store.root.rglob('*') if p.is_file()}
        before=hashes();report=coverage(store.root)
        self.assertEqual(report['errors'],[])
        self.assertEqual(report['pending'],pending);self.assertEqual(report['head'],head)
        self.assertEqual(report['usage_reserved_or_charged'],usage);self.assertEqual(report['usage_reconciled'],usage)
        self.assertIsNone(report['remaining']['seconds']);self.assertEqual(report['observed_seconds'],23456)
        self.assertFalse(report['full_integrity_complete']);self.assertIsNone(report['pair_count'])
        self.assertEqual(store.head(),head);self.assertEqual(store.committed(head),original);self.assertEqual(hashes(),before)
        return report

    def test_partial_and_unvisited_archives_are_unfinished(self):
        for status in ({'eof':False},{'eof':None},{}):
            with self.subTest(status=status):
                # 각 fixture마다 실제 독립 store를 사용한다.
                self.base=self.base/'case';report=self.check_coverage(['a','unvisited'],{'a':1},{'a':status})
                self.assertEqual(report['survey_archives_finished'],0);self.assertFalse(report['survey_complete'])
                self.assertFalse(report['catalog_complete'])

    def test_46_finished_44_unvisited_and_pending_preserved(self):
        selected=[f'archive{i}' for i in range(90)]
        pending={'archive':selected[46],'token':'b'*32,'reserved':{'source_bytes':64*1024**2,'requests':1000,'seconds':None,'artifact_io_bytes':16*1024**2},'owner':'fixture'}
        report=self.check_coverage(selected,{k:2 for k in selected[:46]},{k:{'eof':False} for k in selected[:46]},pending=pending)
        self.assertEqual(report['survey_archives_finished'],46);self.assertFalse(report['survey_complete'])
        self.assertEqual(report['next_action'],'explicit_pending_recovery_required')
        self.assertEqual(len(report['archives']),46)

    def test_eof_finishes_archive_below_quota(self):
        report=self.check_coverage(['eof','partial'],{'eof':1,'partial':1},{'eof':{'eof':True},'partial':{'eof':False}})
        self.assertEqual(report['survey_archives_finished'],1);self.assertFalse(report['survey_complete'])

    def test_all_quotas_finished_is_not_catalog_or_integrity_completion(self):
        report=self.check_coverage(['a','b'],{'a':2,'b':2},{'a':{'eof':False},'b':{'eof':False}})
        self.assertEqual(report['survey_archives_finished'],2);self.assertTrue(report['survey_complete'])
        self.assertFalse(report['catalog_complete']);self.assertEqual(report['next_action'],'survey_finished_review_measurements_for_full_plan')

    def test_all_eof_finishes_survey_without_full_integrity_promotion(self):
        report=self.check_coverage(['a','b'],{'a':1},{'a':{'eof':True},'b':{'eof':True}})
        self.assertEqual(report['survey_archives_finished'],2);self.assertTrue(report['survey_complete'])
        self.assertTrue(report['catalog_complete'])

    def test_empty_selection_never_completes_survey(self):
        report=self.check_coverage([],{},{});self.assertEqual(report['survey_archives_finished'],0)
        self.assertFalse(report['survey_complete']);self.assertFalse(report['catalog_complete'])
        self.assertNotEqual(report['next_action'],'survey_finished_review_measurements_for_full_plan')

if __name__=='__main__':unittest.main()
