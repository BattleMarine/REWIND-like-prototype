"""Linux memory-availability metric regressions, without Drive or inference work."""

from pathlib import Path
import tempfile
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import mock_open,patch

from rewind_v1 import cross_cli


class CrossPreflightTests(TestCase):
    def setUp(self):
        (Path(__file__).resolve().parents[1]/'outputs/test_tmp').mkdir(parents=True,exist_ok=True)
        temporary=tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]/'outputs/test_tmp')
        self.addCleanup(temporary.cleanup);self.root=Path(temporary.name)

    def run_preflight(self,text):
        with patch.object(Path,'open',mock_open(read_data=text)), \
             patch.object(cross_cli.shutil,'disk_usage',return_value=SimpleNamespace(free=16*1024**3)), \
             patch.object(cross_cli.os,'sysconf',create=True,side_effect=AssertionError('free pages must not be used')):
            return cross_cli.preflight(self.root)

    def test_high_available_low_free_is_accepted(self):
        result=self.run_preflight('MemTotal: 87518494 kB\nMemFree: 2617500 kB\nMemAvailable: 84462880 kB\n')
        self.assertEqual(result['available_ram_bytes'],84462880*1024)
        self.assertEqual(result['ram_measurement'],'Linux /proc/meminfo MemAvailable')
        self.assertIn('cgroup headroom not measured',result['ram_scope'])

    def test_low_available_is_rejected_and_four_gib_boundary_is_unchanged(self):
        for kb in (1048576,4194303):
            with self.subTest(kb=kb),self.assertRaisesRegex(MemoryError,'cross_initial_available_ram_4GiB'):
                self.run_preflight(f'MemFree: 1000 kB\nMemAvailable: {kb} kB\n')
        self.assertEqual(self.run_preflight('MemAvailable: 4194304 kB\n')['available_ram_bytes'],4*1024**3)

    def test_missing_invalid_or_ambiguous_metric_does_not_fallback_to_free(self):
        for text in ('MemFree: 84462880 kB\n','MemAvailable: 10 MB\n','MemAvailable: -1 kB\n',
                     'MemAvailable: 4194304 kB\nMemAvailable: 84462880 kB\n','x'*65537):
            with self.subTest(text=text[:70]),self.assertRaisesRegex(RuntimeError,'cross_available_ram_'):
                self.run_preflight(text)

    def test_unreadable_proc_is_an_explicit_failure(self):
        with patch.object(Path,'open',side_effect=FileNotFoundError('/proc/meminfo')):
            with self.assertRaisesRegex(RuntimeError,'cross_available_ram_unavailable'):
                cross_cli._linux_available_ram()
