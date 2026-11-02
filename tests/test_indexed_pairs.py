"""합성102 stereo와 mock Drive로 누적 budget·무scan·partial 회수를 검사한다."""
import copy
from dataclasses import asdict
import hashlib
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from zipfile import ZipFile

from PIL import Image
from rewind_archive.contracts import canonical, digest
from rewind_archive_index.core import ArchiveIndex
from rewind_archive_index.drive import DriveRuntime, make_drive_source
from rewind_archive_index.snapshots import export_snapshot
from rewind_archive_index.source import IndexLimits
from rewind_discovery import pairs
import test_prepare as pose_fixtures
from test_drive_metadata import FakeDrive
from test_persistent_archive_index import fixture

ROOT = Path(__file__).resolve().parents[1]


class IndexedPairsTests(unittest.TestCase):
    def setUp(self):
        parent = ROOT/'outputs/test_tmp'
        parent.mkdir(parents=True, exist_ok=True)
        tmp = tempfile.TemporaryDirectory(dir=parent)
        self.addCleanup(tmp.cleanup)
        self.base = Path(tmp.name)
        helper = pose_fixtures.PreparationTests('test_adopted_spec_and_single_mm_conversion')
        helper.setUp()
        self.addCleanup(helper.doCleanups)
        self.template = helper.prepare()[0]
        self.seq = helper.seq

    def build(self, count=4, *, bad_png=False, limit_changes=None):
        self.inputs = self.base/'inputs'
        self.inputs.mkdir()
        self.work = self.base/'work'
        buffer = io.BytesIO()
        Image.new('RGB', (1024, 1024), (50, 80, 100)).save(buffer, format='PNG')
        self.png = buffer.getvalue()
        rows = []
        for side in ('left', 'right'):
            for frame in range(count):
                name = f'{self.seq}/fisheye_final_image/camera_{side}/final_{frame}.png'
                rows.append((name, b'bad PNG payload' if bad_png and side == 'left' and frame == 1 else self.png))
        self.compressed = fixture(rows)
        selection = {'parts': [{'index': 0, 'name': 'rgb.tar.gz000', 'file_id': 'fake'}]}
        self.session = FakeDrive(selection, {'fake': self.compressed})
        self.session.get_adapter = lambda _: SimpleNamespace(max_retries=SimpleNamespace(total=0))
        self.closed = []
        self.session.close = lambda: self.closed.append(True)
        manifest = {'schema_version': 'archive_manifest_v1', 'archive_id': 'pair_mock', 'dataset_version': 'synthetic_pair_fixture',
            'kind': 'rgb', 'source_scope': 'user_drive', 'total_parts': 1, 'parts': [{'index': 0, 'name': 'rgb.tar.gz000',
            'size_bytes': len(self.compressed), 'version': '9', 'source': {'type': 'drive', 'file_id': 'fake'},
            'identity': {'version': '9', 'modified_time': '2026-10-14T00:00:00.000Z'}}]}
        spec = make_drive_source(manifest)
        idx = ArchiveIndex(self.base/'index.sqlite', IndexLimits(max_read_bytes=32*1024**2, max_decoded_bytes=32*1024**2),
                           runtime=DriveRuntime(self.session))
        idx.create(spec, spacing=65536)
        idx.scan(count*2)
        state, _ = idx._snapshot()
        exported = export_snapshot(idx, self.inputs/'snapshot')
        receipt = exported['receipt']
        candidate = copy.deepcopy(self.template)
        candidate['records'] = []
        config_pairs = []
        for frame in range(count):
            pose = copy.deepcopy(self.template['records'][0])
            pose['frame_index'] = frame
            candidate['records'].append(pose)
            config_pairs.append({'sequence': self.seq, 'frame': frame, 'rgb': {
                side: next(r for r in state['members'] if r['name'] == f'{self.seq}/fisheye_final_image/camera_{side}/final_{frame}.png')
                for side in ('left', 'right')}})
        candidate['content_sha256'] = digest({k: v for k, v in candidate.items() if k != 'content_sha256'})
        (self.inputs/'pose').mkdir()
        (self.inputs/'pose/candidate.json').write_bytes(canonical(candidate))
        limits = asdict(pairs.PairLimits())
        limits.update(limit_changes or {})
        self.config = {'schema': 'indexed_stereo_pairs_v1', 'job_id': 'synthetic_pairs', 'pairs': config_pairs,
            'snapshot_receipt': receipt, 'receipt_file_sha256': pairs.sha_file(exported['receipt_path']),
            'rgb_manifest': manifest, 'source_sha256': digest(spec), 'candidate_sha256': candidate['content_sha256'],
            'train_ready': False, 'limits': limits}
        self.seal()
        self.session.calls.clear()

    def seal(self):
        (self.inputs/'config.json').write_bytes(canonical(self.config))
        pins = {p.relative_to(self.inputs).as_posix(): {'bytes': p.stat().st_size, 'sha256': pairs.sha_file(p)}
                for p in self.inputs.rglob('*') if p.is_file() and p.name != 'input_manifest.json'}
        (self.inputs/'input_manifest.json').write_bytes(canonical({'schema': 'pinned_pair_inputs_v1', 'files': pins}))

    def run_job(self):
        with patch.object(ArchiveIndex, 'scan', side_effect=AssertionError('no production scan')), \
             patch.object(ArchiveIndex, '_commit', side_effect=AssertionError('no production mutation')):
            return pairs.run(self.inputs, self.work, lambda: self.session)

    def recovery(self):
        output = self.base/'result.zip'
        before = len(self.session.calls)
        result = pairs.recover(self.inputs, self.work, output, expected_config_sha256=digest(self.config))
        self.assertEqual(len(self.session.calls), before)
        self.assertFalse(result['network_requested'])
        with ZipFile(output) as z:
            self.assertIsNone(z.testzip())
        return output

    def test_full102_seven_chunks_original32_cap_and_shared_stats(self):
        self.build(102)
        result = self.run_job()
        self.assertEqual(result['status'], 'completed', result.get('stop_reason'))
        self.assertEqual(result['complete_pairs'], 102)
        self.assertEqual(len(result['files']), 204)
        self.assertEqual([c['expected_files'] for c in result['chunks']], [32]*6+[12])
        self.assertTrue(result['index_unchanged'])
        self.assertTrue(result['all_targets_content_audited'])
        self.assertFalse(result['human_visual_alignment_confirmed'])
        self.assertEqual(result['returned_decoded_bytes'], len(self.png)*204)
        self.assertEqual(result['source_read_bytes'], sum(c['stats']['source_read_bytes'] for c in result['chunks']))
        self.assertGreaterEqual(result['requests'], result['chunks'][-1]['stats']['phase_http_requests'])
        self.assertEqual(len(list((self.work/'overlays').glob('*.png'))), 2)
        self.recovery()
        with self.assertRaises(FileExistsError):
            self.run_job()
        self.assertEqual(self.closed, [True])

    def test_shared_decode_cap_preserves_prior_chunk_and_current_partial(self):
        self.build(4, limit_changes={'batch_files': 2, 'decoded_bytes': len(self._png())*3})
        result = self.run_job()
        self.assertEqual(result['status'], 'partial_or_failed')
        self.assertEqual(result['stop_reason'], 'decoded_bytes_limit')
        self.assertEqual(result['chunks'][0]['status'], 'content_audited')
        self.assertEqual(len(result['files']), 2)
        self.assertLessEqual(result['returned_decoded_bytes'], self.config['limits']['decoded_bytes'])
        output = self.recovery()
        with ZipFile(output) as z:
            self.assertTrue(any(n.startswith('run/chunks/001/files/') for n in z.namelist()))

    def _png(self):
        stream = io.BytesIO()
        Image.new('RGB', (1024, 1024), (50, 80, 100)).save(stream, format='PNG')
        return stream.getvalue()

    def test_identity_changed_fails_before_media_and_recovers(self):
        self.build()
        self.session.docs['fake']['version'] = '10'
        result = self.run_job()
        self.assertEqual(result['stop_reason'], 'pair_source_identity_changed')
        self.assertFalse(any('alt=media' in url for url, kw in self.session.calls))
        self.assertTrue(result['index_unchanged'])
        self.assertEqual(result['complete_pairs'], 0)
        self.recovery()

    def test_png_bad_payload_is_not_a_complete_pair(self):
        self.build(bad_png=True)
        result = self.run_job()
        self.assertEqual(result['status'], 'partial_or_failed')
        self.assertEqual(result['stage'], 'png_audit')
        self.assertFalse(result['all_targets_content_audited'])
        self.assertEqual(len(result['files']), 1)
        self.recovery()

    def test_shared_http_cap_is_not_reset_at_chunk_boundary(self):
        self.build(limit_changes={'batch_files': 2, 'requests': 7})
        result = self.run_job()
        self.assertEqual(result['status'], 'partial_or_failed')
        self.assertLessEqual(result['requests'], 7)
        self.assertLessEqual(result['body_bytes'], self.config['limits']['body_bytes'])
        metadata = json.loads((self.work/'metadata.json').read_bytes())
        self.assertEqual(result['body_bytes'], metadata['report']['stats']['transfer_bytes']+
                         result['chunks'][-1]['stats']['phase_response_body_bytes'])
        self.assertTrue(result['index_unchanged'])
        self.assertGreaterEqual(len(result['chunks']), 2)
        self.assertEqual(result['chunks'][0]['status'], 'content_audited')
        self.recovery()

    def test_payload_changed_before_png_audit_has_explicit_hash_failure(self):
        self.build()
        original = pairs.ExtractOnlyIndex.extract
        def changed(index, names, output):
            report = original(index, names, output)
            (output/report['files'][0]['relative_path']).write_bytes(b'changed')
            return report
        with patch.object(pairs.ExtractOnlyIndex, 'extract', changed):
            result = self.run_job()
        self.assertEqual(result['stop_reason'], 'pair_png_payload_hash')
        self.assertEqual(result['complete_pairs'], 0)
        self.recovery()

    def test_cleanup_close_and_db_hash_errors_still_save_and_recover(self):
        self.build(1)
        def close():
            raise OSError('private detail must not be recorded')
        self.session.close = close
        original = pairs.sha_file
        counts = []
        def failing_hash(path):
            if Path(path) == self.work/'index.sqlite':
                counts.append(path)
                if len(counts) == 2:
                    raise OSError('private DB detail')
            return original(path)
        with patch.object(pairs, 'sha_file', failing_hash):
            result = self.run_job()
        self.assertEqual({e['stage'] for e in result['cleanup_errors']}, {'session_close', 'index_after_sha256'})
        self.assertEqual(result['status'], 'partial_or_failed')
        self.assertFalse(result['index_unchanged'])
        self.assertNotIn('private', json.dumps(result))
        self.assertTrue((self.work/'report.json').is_file())
        self.recovery()

    def test_mismatched_index_row_no_auth_and_input_tamper_rejected(self):
        self.build()
        self.config['pairs'][0]['rgb']['left']['sha256'] = 'a'*64
        self.seal()
        result = self.run_job()
        self.assertEqual(result['stop_reason'], 'pair_index_exact_rows')
        self.assertEqual(self.session.calls, [])
        self.assertEqual(self.closed, [])
        (self.inputs/'pose/candidate.json').write_bytes(b'{}')
        with self.assertRaisesRegex(Exception, 'pair_input_pin'):
            pairs.preflight(self.inputs)

    def test_interruption_after_one_chunk_preserves_report_and_recovery(self):
        self.build(limit_changes={'batch_files': 2})
        original = pairs.ExtractOnlyIndex.extract
        calls = []
        def interrupted(index, names, output):
            calls.append(names)
            if len(calls) == 2:
                raise KeyboardInterrupt()
            return original(index, names, output)
        with patch.object(pairs.ExtractOnlyIndex, 'extract', interrupted):
            result = self.run_job()
        self.assertEqual(result['stop_reason'], 'KeyboardInterrupt')
        self.assertEqual(result['chunks'][0]['status'], 'content_audited')
        self.assertTrue(result['index_unchanged'])
        self.recovery()
