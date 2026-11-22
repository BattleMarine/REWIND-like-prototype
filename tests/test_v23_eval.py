"""Check fixed timing scope and warmup/repeat protocol without a CUDA runtime."""

from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import Mock, patch

import torch

from scripts import v23_eval as audit


class V23TimingTests(TestCase):
    def test_warmups_excluded_and_each_arm_gets_five_synchronized_bundles(self):
        device = torch.device("cuda:0")
        model = SimpleNamespace(device=device)
        inputs = {(f"scene-{i}/clip", 0): SimpleNamespace(real_frames=50, noise_sha256=str(i)) for i in range(3)}
        event = Mock()
        event.elapsed_time.return_value = 12
        with (patch.object(audit, "cascade_prediction") as predict,
              patch.object(torch.cuda, "Event", return_value=event),
              patch.object(torch.cuda, "current_stream", return_value=object()),
              patch.object(torch.cuda, "synchronize") as sync):
            result = audit.benchmark(None, model, model, inputs, lambda: None)
        expected = [steps for steps in audit.STEPS for _ in range(3)]
        expected += [steps for order in audit.ROUNDS for steps in order for _ in range(3)]
        self.assertEqual([call.args[-1] for call in predict.call_args_list], expected)
        self.assertEqual(sync.call_count, 33)
        self.assertEqual(result["bundle_real_frames"], 150)
        for item in result["timings"].values():
            self.assertEqual(item["cuda_stream_ms"]["samples"], [12] * 5)
            self.assertEqual(item["cuda_stream_ms"]["median_amortized_ms_per_frame"], .08)

    def test_summary_rejects_absent_or_nonfinite_measurement(self):
        good = {"cuda_stream_ms": 4.0, "synchronized_wall_ms": 5.0}
        with self.assertRaisesRegex(ValueError, "five"):
            audit.timing_summary([good] * 4, 150)
        with self.assertRaisesRegex(ValueError, "five"):
            audit.timing_summary([good] * 4 + [dict(good, cuda_stream_ms=float("nan"))], 150)
