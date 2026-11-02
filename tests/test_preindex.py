"""opt-in 계측·사전인덱싱 누적/중단 회귀. 실제 원격 요청 없음."""
from dataclasses import asdict, replace
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from rewind_archive.sources import session_opener
from rewind_archive_index.core import ArchiveIndex
from rewind_archive_index.source import IndexFailure
from rewind_discovery.runner import PREINDEX_PROFILE, PreindexLimits, DiscoveryLimits, run
from rewind_discovery.recovery import recover
from rewind_discovery.timing import Timing
from rewind_archive.contracts import digest
import test_discovery as fixtures


class TimingTests(unittest.TestCase):
    def test_fake_clock_headers_raw_read_and_close_contract_with_and_without_hook(self):
        clock = SimpleNamespace(wall=0., cpu=0.)
        timing = Timing(lambda: clock.wall, lambda: clock.cpu)
        closed, calls = [], []
        class Raw:
            def read1(self, size, decode_content):
                calls.append((size, decode_content)); clock.wall += 3; clock.cpu += .5
                return b'abc'
        response = SimpleNamespace(status_code=200, headers={}, raw=Raw(), close=lambda: closed.append(True))
        def get(url, **kwargs):
            calls.append(kwargs); clock.wall += 2; clock.cpu += .25
            return response
        session = SimpleNamespace(get=get, _max_refresh_attempts=0, _rewind_request_timing=timing)
        wrapped = session_opener(session)('not_recorded', {'test': 'value'}, 4)
        self.assertEqual(wrapped.read1(3), b'abc'); wrapped.close()
        self.assertEqual(calls, [{'headers': {'test': 'value'}, 'stream': True, 'allow_redirects': False, 'timeout': 4}, (3, False)])
        self.assertEqual(closed, [True])
        self.assertEqual(timing.calls['http_request_to_headers']['wall_seconds'], 2)
        self.assertEqual(timing.calls['http_body_read1']['wall_seconds'], 3)
        self.assertEqual(timing.snapshot()['process_cpu_seconds'], .75)
        before = timing.snapshot()['calls']
        del session._rewind_request_timing
        wrapped = session_opener(session)('not_recorded', {}, 4)
        self.assertEqual(wrapped.read1(3), b'abc'); wrapped.close()
        self.assertEqual(timing.snapshot()['calls'], before)
        self.assertEqual(closed, [True, True])

    def test_failed_call_records_time_without_exception_content(self):
        clock = SimpleNamespace(wall=0., cpu=0.)
        timer = Timing(lambda: clock.wall, lambda: clock.cpu)
        def fail():
            clock.wall += 5; clock.cpu += .2
            raise OSError('private-secret')
        with self.assertRaises(OSError):
            timer.call('snapshot_export', fail)
        self.assertEqual(timer.calls['snapshot_export']['failed_calls'], 1)
        self.assertEqual(timer.calls['snapshot_export']['wall_seconds'], 5)
        self.assertNotIn('private', json.dumps(timer.snapshot()))


@unittest.skipUnless(importlib.util.find_spec('indexed_gzip'), '보존 indexed_gzip 환경 필요')
class PreindexTests(unittest.TestCase):
    def setUp(self):
        self.helper = fixtures.DiscoveryTests()
        self.helper.setUp(); self.addCleanup(self.helper.doCleanups)
        first = self.helper.execute('anchor')
        receipt_path = Path(first['last_receipt_path'])
        import hashlib
        self.resume = {'receipt_path': receipt_path, 'receipt_file_sha256': hashlib.sha256(receipt_path.read_bytes()).hexdigest(),
                       'anchor': first['last_receipt']}
        self.limits = PreindexLimits(body_bytes=4*1024**2, requests=256, read_bytes=8*1024**2,
            decoded_bytes=8*1024**2, seconds=60, reserve_seconds=5, batch_members=4, batches=2)

    def execute(self, **kwargs):
        return self.helper.execute('preindex', limits=kwargs.pop('limits', self.limits),
            resume=self.resume, profile=PREINDEX_PROFILE, **kwargs)

    def recover(self, report):
        base = self.helper.base
        result = recover(base/'preindex', base/'preindex_snapshots', base/'recovery.zip', job_id='preindex',
            selection_sha256=digest(self.helper.selection), disk_limit=self.limits.disk_bytes,
            input_snapshot_directory=self.resume['receipt_path'].parent)
        self.assertEqual(result['last_receipt'], report['last_receipt'])
        self.assertFalse(result['network_requested'])

    def test_explicit_profile_defaults_old_caps_and_resume_metrics(self):
        self.assertEqual(DiscoveryLimits().body_bytes, 256*1024**2)
        self.assertEqual(PreindexLimits().body_bytes, 512*1024**2)
        with self.assertRaises(IndexFailure):
            DiscoveryLimits(body_bytes=512*1024**2)
        with self.assertRaises(IndexFailure):
            self.helper.execute('bad', limits=self.limits, resume=self.resume)
        result = self.execute()
        self.assertEqual(result['status'], 'completed_requested_batches')
        self.assertEqual(result['before']['members'], 8)
        self.assertEqual(result['durable_members'], 16)
        self.assertTrue(result['validation_rgb_stereo_found'])
        self.assertEqual(len(result['scan_attempts']), 2)
        timing = result['timing']['calls']
        self.assertEqual(timing['scan']['calls'], 2)
        self.assertEqual(timing['snapshot_export']['calls'], 2)
        self.assertGreater(timing['http_body_read1']['calls'], 0)
        self.assertEqual(result['read_bytes'], sum(a['stats']['source_read_bytes'] for a in result['scan_attempts']))
        self.assertEqual(result['cumulative_cache']['hits'], sum(a['stats']['remote_cache_hits'] for a in result['scan_attempts']))
        self.assertTrue(all('snapshot_timing' in e for e in result['events']))
        self.assertIn('http_calls',result['scan_attempts'][0]['timing_delta'])
        self.assertEqual(result['durable_discovered'],result['working_discovered'])
        self.assertFalse(hasattr(self.helper.sessions[-1], '_rewind_request_timing'))
        self.recover(result)

    def test_failed_scan_preserves_io_timing_and_last_normal_receipt(self):
        original = ArchiveIndex.scan
        calls = []
        def interrupted(index, *args, **kwargs):
            result = original(index, *args, **kwargs)
            calls.append(True)
            if len(calls) == 2:
                raise KeyboardInterrupt()
            return result
        with patch.object(ArchiveIndex, 'scan', interrupted), self.assertRaises(KeyboardInterrupt):
            self.execute()
        report = json.loads((self.helper.base/'preindex/report.json').read_bytes())
        self.assertEqual(report['status'], 'interrupted')
        self.assertEqual(report['timing']['calls']['scan']['failed_calls'], 1)
        self.assertEqual(report['scan_attempts'][-1]['status'], 'failed_or_interrupted')
        self.assertGreater(report['scan_attempts'][-1]['stats']['source_read_bytes'], 0)
        self.assertEqual(report['durable_members'], 12)
        self.recover(report)

    def test_snapshot_space_stop_is_bounded_and_previous_receipt_recovers(self):
        def failed_publish(*args,**kwargs):
            (self.helper.base/'preindex_snapshots'/'partial_unpublished.zip').write_bytes(b'x'*1024**2)
            raise IndexFailure('snapshot_disk_limit')
        with patch('rewind_discovery.runner.export_snapshot', side_effect=failed_publish):
            report = self.execute()
        self.assertEqual(report['status'], 'bounded_stop')
        self.assertEqual(report['last_receipt'], self.resume['anchor'])
        self.assertNotEqual(report['working_discovered'],report['durable_discovered'])
        self.assertEqual(report['timing']['calls']['snapshot_export']['failed_calls'], 1)
        self.assertTrue((self.helper.base/'preindex_snapshots'/'partial_unpublished.zip').exists())
        self.recover(report)

    def test_small_disk_pre_scan_reserve_stops_and_recovery_still_fits(self):
        self.limits=replace(self.limits,disk_bytes=8*1024**2)
        report=self.execute(limits=self.limits)
        self.assertEqual(report['status'],'bounded_stop')
        self.assertEqual(report['failure']['reason'],'discovery_scan_snapshot_recovery_reserve')
        self.assertEqual(report['last_receipt'],self.resume['anchor'])
        self.assertEqual(report['scan_attempts'],[])
        self.recover(report)

    def test_identity_failure_stays_failed_not_bounded(self):
        expected = [dict(self.helper.expected[0], size_bytes=self.helper.expected[0]['size_bytes']+1)]
        report = self.execute(expected_metadata=expected)
        self.assertEqual(report['status'], 'failed')
        self.assertEqual(report['failure']['reason'], 'discovery_source_metadata_changed')
        self.assertEqual(report['last_receipt'], self.resume['anchor'])
        self.recover(report)
