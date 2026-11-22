"""Regression tests for the metrics R2 notebook's Drive receipt chaining."""

import contextlib
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest

from scripts.v24_main_chain import V24_OUTPUT_FOLDER, V24_RESULT_SCHEMA, run_v24_chain, v24_chain_digest


class ChainTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.project = Path(tmp.name).resolve()
        self.parent = self.project / "reports" / V24_OUTPUT_FOLDER
        self.parent.mkdir(parents=True)
        protocol = {"boundaries": [0, 768, 1536, 2304, 3097], "initial_seeds": [42, 71, 103],
                    "conditional_seeds": [137, 181], "official_validation_update": 3097,
                    "runtime": {"gpu": "Tesla T4", "precision": "fp32", "tf32": False},
                    "budgets": {"seconds": 1500, "worker_reserve_seconds": 240, "publish_reserve_seconds": 180,
                                "report_bytes": 2**21, "state_bytes": 2**28, "worker_logical_read_bytes": 2**31,
                                "all_logical_write_bytes": 6*2**30, "scratch_free_bytes": 2**32}}
        self.pins = {"protocol": protocol, "protocol_sha256": v24_chain_digest(protocol),
                     "base": {"source_sha256": "a"*64}, "modules": {"v24_main_run.py": "b"*64}}
        self.calls = []
        self.results = []
        self.volume = 0

    @staticmethod
    def write(path, data):
        raw = data if isinstance(data, bytes) else json.dumps(data, sort_keys=True).encode()
        path.write_bytes(raw)
        return {"bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}

    def stage(self, project, arm, seed, before, after, resume, resume_sha, protocol_sha, extension, pins,
              *, fail=False, wrong_count=False):
        self.calls.append((before, after, resume, resume_sha))
        self.volume += 1
        folder = self.parent / f"run-{self.volume:032x}"
        folder.mkdir()
        if fail:
            return {"path": str(folder / "result.json"), "sha256": "f"*64, "status": "failed"}
        common = {"seed": seed, "protocol_sha256": protocol_sha, "environment": pins["protocol"]["runtime"]}
        binding = {"arm": arm, "common": common, "module_sha256": pins["modules"], "source_sha256": pins["base"]["source_sha256"]}
        receipt = {"status": "segment_complete", "arm": arm, "seed": seed, "before": before, "after": after,
                   "protocol_sha256": protocol_sha, "binding_sha256": v24_chain_digest(binding), "common_sha256": v24_chain_digest(common)}
        scores = []
        for step in ([0, after] if before == 0 else [after]):
            final = step == 3097
            scope, clips, frames = ("official_validation", 421, 16424 if wrong_count else 16425) if final else ("pilot_validation_subset", 6, 300)
            scores.append({"scope": scope, "clips": clips, "frames": frames, "update": step,
                           "noise_and_roster_sha256": "c"*64, "metrics": {"pooled": {"clips": clips, "frames": frames},
                                                                         "per_scenario": {name: {} for name in "abc"}}})
        updates = [[step, ["clip", step], "d"*64] for step in range(before + 1, after + 1)]
        segment = {"binding": binding, "before": before, "after": after, "resume_result_sha256": resume_sha or None,
                   "scores": scores, "paired_update_sha256": v24_chain_digest(updates),
                   "updates": [{"step": step, "key": key, "noise_sha256": noise} for step, key, noise in updates]}
        files = {"segment.json": self.write(folder / "segment.json", segment), "body-state.pt": self.write(folder / "body-state.pt", b"state")}
        marker = {**receipt, "schema": "rewind_v24_main_worker_v1", "files": files.copy()}
        files["worker-result.json"] = self.write(folder / "worker-result.json", marker)
        path = folder / "result.json"
        pin = self.write(path, {**receipt, "schema": V24_RESULT_SCHEMA, "files": files, "path": str(folder),
                                "resume_result_sha256": resume_sha or None, "elapsed_seconds": 45,
                                "returncode": 0, "budgets": pins["protocol"]["budgets"]})
        handoff = {"path": str(path), "status": "segment_complete", **pin}
        self.results.append(handoff)
        return handoff

    def run_chain(self, boundaries, runner=None, anchor=None, arm="v", seed=42, clock=lambda: 0):
        with contextlib.redirect_stdout(io.StringIO()):
            return run_v24_chain(str(self.project), arm, seed, boundaries, anchor["path"] if anchor else "",
                                 anchor["sha256"] if anchor else "", self.pins["protocol_sha256"], False, self.pins,
                                 run_segment=runner or self.stage, prepare_project=Path, clock=clock)

    def test_stops_on_middle_failure_and_resumes_explicit_verified_anchor(self):
        def stop(*args):
            return self.stage(*args, fail=args[3] == 768)
        with self.assertRaisesRegex(RuntimeError, "failed"):
            self.run_chain([0, 768, 1536, 2304, 3097], stop)
        self.assertEqual([r[:2] for r in self.calls], [(0, 768), (768, 1536)])
        anchor = self.results[0]
        self.assertEqual(self.calls[1][2:], (anchor["path"], anchor["sha256"]))
        accepted = self.run_chain([768, 1536, 2304, 3097], anchor=anchor)
        self.assertEqual(self.calls[2][2:], (anchor["path"], anchor["sha256"]))
        self.assertEqual(self.calls[3][2:], (accepted[0]["path"], accepted[0]["sha256"]))
        self.assertEqual(accepted[-1]["scope"], "official_validation")

    def test_bad_drive_readback_stops_before_next_stage(self):
        def corrupt(*args):
            handoff = self.stage(*args)
            Path(handoff["path"]).with_name("segment.json").write_bytes(b"changed after publication")
            return handoff
        with self.assertRaisesRegex(ValueError, "SHA mismatch"):
            self.run_chain([0, 768, 1536], corrupt)
        self.assertEqual(len(self.calls), 1)

    def test_wrong_anchor_sha_or_arm_stops_before_worker(self):
        self.run_chain([0, 768])
        anchor = self.results[0]
        with self.assertRaisesRegex(ValueError, "SHA mismatch"):
            self.run_chain([768, 1536], anchor={**anchor, "sha256": "f"*64})
        with self.assertRaisesRegex(ValueError, "approved stage"):
            self.run_chain([768, 1536], anchor=anchor, arm="x0")
        self.assertEqual(len(self.calls), 1)

    def test_plan_and_overall_budget_rejected_before_worker(self):
        for invalid in ((), (0, 1536), (1536, 768), (0, 768, 768), (768, 1536)):
            with self.assertRaises(ValueError):
                self.run_chain(invalid)
        with self.assertRaises(ValueError):
            self.run_chain([0, 768], seed=137)
        ticks = iter([0, 181])
        with self.assertRaisesRegex(TimeoutError, "Insufficient overall time"):
            self.run_chain([0, 768], clock=lambda: next(ticks))
        self.assertEqual(self.calls, [])

    def test_rejects_final_result_with_wrong_frame_count(self):
        prefix = self.run_chain([0, 768, 1536, 2304])
        with self.assertRaisesRegex(ValueError, "sample count"):
            self.run_chain([2304, 3097], runner=lambda *args: self.stage(*args, wrong_count=True), anchor=prefix[-1])
        self.assertEqual(len(self.calls), 4)
