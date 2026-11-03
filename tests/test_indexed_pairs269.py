"""고정269 key/identity의 합성17chunk·부분회수·예산격리 검증. 원격0."""
from dataclasses import asdict
import json
from pathlib import Path
import unittest
from unittest.mock import patch
from zipfile import ZipFile
from rewind_archive.matching import identify
from rewind_archive.profiles import target269_names
from rewind_archive_index.source import IndexFailure
from rewind_discovery import pairs
import test_indexed_pairs as fixtures


class Pair269Tests(unittest.TestCase):
    def fixture(self):
        h=fixtures.IndexedPairsTests();h.setUp();self.addCleanup(h.doCleanups)
        keys=[identify(n,'pose')[:2] for n in target269_names()];h.build(269,keys=keys)
        h.config.update(schema=pairs.RGB269_PROFILE,job_id='rgb269_pairs_001',limits=asdict(pairs.RGB269Limits()))
        receipt=h.config['snapshot_receipt']
        for name,value in [('RGB269_SOURCE_SHA',h.config['source_sha256']),('RGB269_SNAPSHOT_SHA',receipt['snapshot_id']),
                ('RGB269_RECEIPT_SHA',h.config['receipt_file_sha256']),('RGB269_GENERATION',receipt['generation'])]:
            p=patch.object(pairs,name,value);p.start();self.addCleanup(p.stop)
        h.seal();return h

    def test_269_whole17_chunks_scan_forbidden_and_recovery(self):
        h=self.fixture();config,pin=pairs.preflight(h.inputs);self.assertEqual(len(config['pairs']),269)
        result=h.run_job();self.assertEqual(result['status'],'completed');self.assertEqual(result['complete_pairs'],269)
        self.assertEqual(len(result['files']),538);self.assertEqual(len(result['chunks']),17)
        self.assertEqual(result['chunks'][-1]['expected_files'],26);self.assertTrue(result['index_unchanged'])
        self.assertFalse(result['scan_performed']);self.assertTrue(result['all_targets_content_audited'])
        h.recovery()

    def test_changed_target_identity_cold_input_and_budget_blocked(self):
        h=self.fixture()
        original=json.loads(json.dumps(h.config))
        for field,value in [('job_id','another_job'),('source_sha256','0'*64),('receipt_file_sha256','0'*64),
                            ('limits',{**h.config['limits'],'body_bytes':897*1024**2}),
                            ('schema','indexed_stereo_pairs_v1')]:
            h.config={**original,field:value};h.seal()
            with self.assertRaises((IndexFailure,KeyError)):pairs.preflight(h.inputs)
        h.config=json.loads(json.dumps(original));h.config['pairs'][0]['frame']=99999;h.seal()
        with self.assertRaises(IndexFailure):pairs.preflight(h.inputs)
        self.assertEqual(h.session.calls,[])
        with self.assertRaises(IndexFailure):pairs.PairLimits(body_bytes=896*1024**2)
        self.assertEqual(asdict(pairs.PairLimits())['decoded_bytes'],256*1024**2)

    def test_269_interruption_partial_recovery_without_reexecution(self):
        h=self.fixture();original=pairs.ExtractOnlyIndex.extract;calls=[]
        def interrupted(index,names,out):
            calls.append(names)
            if len(calls)==2:raise KeyboardInterrupt()
            return original(index,names,out)
        with patch.object(pairs.ExtractOnlyIndex,'extract',interrupted):result=h.run_job()
        self.assertEqual(result['complete_pairs'],0) # archive stores left view first; first chunk has no complete stereo
        self.assertEqual(result['chunks'][0]['status'],'content_audited');self.assertEqual(result['stop_reason'],'KeyboardInterrupt')
        self.assertTrue(result['index_unchanged']);before=len(h.session.calls);h.recovery();self.assertEqual(len(h.session.calls),before)

    def test_269_cumulative_http_cap_survives_17chunk_plan(self):
        h=self.fixture();original=pairs.DriveLimits
        def bounded(**kwargs):return original(**{**kwargs,'max_requests':7})
        with patch.object(pairs,'DriveLimits',bounded):result=h.run_job()
        self.assertEqual(result['status'],'partial_or_failed');self.assertLessEqual(result['requests'],8)
        self.assertGreaterEqual(len(result['chunks']),2);self.assertEqual(result['chunks'][0]['status'],'content_audited')
        self.assertEqual(result['target_pairs'],269);self.assertTrue(result['index_unchanged'])
        before=len(h.session.calls);h.recovery();self.assertEqual(len(h.session.calls),before)
