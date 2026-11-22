"""CPU fixtures for cold worker CUDA startup and preserved failure evidence."""

import json
from pathlib import Path
import sys
import tempfile
import time
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import patch

import torch
from rewind_archive.contracts import digest
from scripts import v211_run


class ColdCUDA:
    def __init__(self):
        self.events = []
        self.initialized = False
        self.allocated = False
        self.visible = 1
        self.index = 0
        self.fail_reset = self.fail_peak_read = self.fail_alloc = False

    def is_available(self):
        return True

    def device_count(self):
        return self.visible

    def init(self):
        self.events.append("init")
        self.initialized = True

    def current_device(self):
        assert self.initialized
        self.events.append("current_device")
        return self.index

    def get_device_properties(self, index):
        assert self.initialized and index == self.index
        self.events.append("properties")
        return SimpleNamespace(name="fixture CUDA GPU", total_memory=1024)

    def allocate(self, elements, *, device, dtype):
        assert self.initialized and device.index == self.index and elements > 0 and dtype == torch.float32
        self.events.append("real_allocation")
        if self.fail_alloc:
            raise RuntimeError("allocator probe failed")
        self.allocated = True
        return object()

    def reset_peak_memory_stats(self, index):
        self.events.append("peak_reset")
        if self.fail_reset or not self.allocated or index != self.index:
            raise RuntimeError("Invalid device argument")

    def max_memory_allocated(self, index):
        assert self.allocated and index == self.index
        if self.fail_peak_read:
            raise RuntimeError("peak read failed")
        return 128

    def max_memory_reserved(self, index):
        assert self.allocated and index == self.index
        return 256

    def torch_api(self):
        return SimpleNamespace(cuda=self, empty=self.allocate, device=torch.device, float32=torch.float32,
                               version=SimpleNamespace(cuda="fixture"), __version__="fixture",
                               backends=SimpleNamespace(cuda=SimpleNamespace(matmul=SimpleNamespace()),
                                                        cudnn=SimpleNamespace()))


class CudaStartupTests(TestCase):
    def init(self, cuda, report):
        with patch.object(v211_run, "torch", cuda.torch_api()), patch.object(v211_run, "seed_run"):
            return v211_run.initialize_cuda(time.monotonic()-1, report)

    def test_cold_allocator_precedes_optional_peak_reset(self):
        cuda, report = ColdCUDA(), {}
        with self.assertRaisesRegex(RuntimeError, "Invalid device argument"):
            cuda.reset_peak_memory_stats(0)
        cuda.events.clear()
        device = self.init(cuda, report)
        self.assertEqual(device, torch.device("cuda:0"))
        self.assertEqual(cuda.events, ["init", "current_device", "properties", "real_allocation", "peak_reset"])
        self.assertEqual(report["environment"]["gpu"], "fixture CUDA GPU")
        self.assertEqual(report["cuda_peak"]["scope"], "after_startup_probe")
        with patch.object(v211_run, "torch", cuda.torch_api()):
            v211_run.collect_peak(device, report)
        self.assertEqual(report["peak_gpu_bytes"], {"allocated": 128, "reserved": 256})

    def test_peak_errors_remain_observational(self):
        cuda, report = ColdCUDA(), {}
        cuda.fail_reset = cuda.fail_peak_read = True
        device = self.init(cuda, report)
        self.assertEqual(report["cuda_peak"]["scope"], "worker_lifetime")
        reset = report["cuda_peak"]["reset_failure"]
        self.assertEqual(reset["phase"], "cuda_peak_reset")
        self.assertIn("Invalid device argument", reset["traceback"])
        with patch.object(v211_run, "torch", cuda.torch_api()):
            v211_run.collect_peak(device, report)
        self.assertIsNone(report["peak_gpu_bytes"])
        self.assertIn("peak read failed", report["cuda_peak"]["read_failure"]["traceback"])
        self.assertNotIn("failure", report)

    def test_early_fatal_error_keeps_phase_and_traceback_in_worker_result(self):
        cuda = ColdCUDA()
        cuda.fail_alloc = True
        with tempfile.TemporaryDirectory() as scratch:
            path = Path(scratch)
            pins_path = path/"pins.json"
            pins_path.write_text("{}", encoding="utf-8")
            argv = ["worker", "--project", str(path), "--work", str(path), "--pins", str(pins_path),
                    "--started", str(time.monotonic()-1)]
            with patch.object(sys, "argv", argv), patch.object(v211_run, "torch", cuda.torch_api()), patch.object(v211_run, "seed_run"):
                with self.assertRaisesRegex(RuntimeError, "allocator probe failed"):
                    v211_run.main()
            result = json.loads((path/"result.json").read_bytes())
        self.assertEqual(result["failure"]["phase"], "cuda_allocator_probe")
        self.assertEqual(result["failure"]["type"], "RuntimeError")
        self.assertIn("RuntimeError: allocator probe failed", result["failure"]["traceback"])
        self.assertNotIn("peak_reset", cuda.events)

    def test_wrong_visible_or_current_device_fails_before_allocator(self):
        cuda, report = ColdCUDA(), {}
        cuda.visible = 2
        with self.assertRaisesRegex(RuntimeError, "index=0, visible=2"):
            self.init(cuda, report)
        self.assertEqual(report["phase"], "cuda_validate")
        self.assertEqual(cuda.events, [])
        cuda.visible, cuda.index = 1, 1
        with self.assertRaisesRegex(RuntimeError, "unexpected_cuda_current_device"):
            self.init(cuda, report)
        self.assertEqual(report["phase"], "cuda_initialize")
        self.assertEqual(cuda.events, ["init", "current_device"])

    def test_resume_alias_accepts_only_the_pinned_original_inputs(self):
        prior = {"base": "fixed", "modules": {"worker": "legacy"}}
        revised = {**prior, "modules": {"worker": "revised"}, "resume_modules": prior["modules"]}
        with patch.dict(v211_run.LEGACY_INPUTS, {"v2.1.1": digest(prior)}):
            self.assertEqual(v211_run.resume_pins(revised, "v2.1.1"), prior)
            self.assertEqual(v211_run.resume_pins(prior, "v2.1.1"), prior)
            for variant in ({**revised, "base": "changed"}, {**revised, "resume_modules": {"worker": "wrong"}}):
                with self.assertRaisesRegex(ValueError, "legacy_resume_inputs"):
                    v211_run.resume_pins(variant, "v2.1.1")
