import copy
import io
import json
from pathlib import Path
import tempfile
from unittest import TestCase
from unittest.mock import patch
import zipfile

from PIL import Image
import torch

from rewind_archive.contracts import canonical
from rewind_prepare.temporal_selection import seal
from scripts import visualize_archvis_v1 as view


class VisualizationTests(TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory(dir=view.ROOT)
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.spec = {"joint_names": ["left_wrist", "right_wrist", "left_tip", "right_tip"],
                     "body_joint_names": ["left_wrist", "right_wrist"], "upper_joint_names": ["left_wrist", "right_wrist"],
                     "hands": {"left": ["left_wrist", "left_tip"], "right": ["right_wrist", "right_tip"]},
                     "edges": [["left_wrist", "right_wrist"], ["left_wrist", "left_tip"], ["right_wrist", "right_tip"]]}

    def checkpoint_zip(self, *, wrong_dependency=False, full=False, wrong_manifest=False):
        config = json.loads((view.ROOT / ("configs/rewind_v12_full_training.json" if full else "configs/rewind_v1_epoch_expansion.json")).read_bytes())
        source = {path.relative_to(view.ROOT).as_posix(): view.sha(path.read_bytes())
                  for package in ("rewind_v1", "rewind_m1", "rewind_m2", "rewind_m3", "rewind_prepare")
                  for path in (view.ROOT / package).glob("*.py")}
        binding = {"format": "rewind_v1_pipeline_v1", "input": {"run_id": view.RUN_ID, "spec_sha256": view.SPEC_SHA,
                   "selection_sha256": view.HISTORICAL["selection_sha256"], "cache_sha256": view.HISTORICAL["cache_sha256"]},
                   "config": config, "config_sha256": "a"*64, "source_code_sha256": source,
                   "environment": {"torch": "fixture"}, "coverage": {"excluded": [("fixture", 0)]}}
        if full:
            decision = view.diagnostic_decision("a"*64,"b"*64)
            binding.update(format=view.FULL_FORMAT, input={"run_id":view.FULL_RUN_ID,"spec_sha256":view.SPEC_SHA,
                "manifest_sha256":"f"*64 if wrong_manifest else view.MANIFEST_SHA,"plan_sha256":"b"*64,
                "diagnostic_decision":decision,"decision_sha256":decision["content_sha256"]})
            binding["coverage"].update(train_unique_frames=20888,frontend_validation_frames=2236,teacher_validation_frames=1450)
        files, best, contents, dependencies = {}, {}, {}, {}
        for role, epoch in (("frontend", 9), ("body", 10 if full else 9), ("hand", 9 if full else 8)):
            name = f"best-{role}-e{epoch:03d}.pt"
            state = {"binding": binding, "stage": role, "epoch": epoch, "score": 1.,
                     "dependencies": copy.deepcopy(dependencies), "model": torch.nn.Linear(1, 1).state_dict()}
            if wrong_dependency and role == "hand":
                state["dependencies"]["body"] = "f"*64
            buffer = io.BytesIO()
            torch.save(state, buffer)
            raw = buffer.getvalue()
            pin = {"name": name, "bytes": len(raw), "sha256": view.sha(raw)}
            files[name], best[role], contents[name] = pin, {**pin, "epoch": epoch, "score": 1.}, raw
            dependencies[role] = pin["sha256"]
        report = {**binding, "status": "completed", "cursor": {"updates": 115720 if full else 5400, "best": best}}
        contents["run.json"] = canonical(report)
        files["run.json"] = {"bytes": len(contents["run.json"]), "sha256": view.sha(contents["run.json"])}
        contents["recovery_receipt.json"] = canonical(seal({"schema":"archvis_v12_training_recovery_v1","files":files})
                                                       if full else {"schema": "rewind_v1_recovery_v1", "files": files})
        archive = self.root / (f"{'full' if full else 'old'}-{wrong_dependency}-{wrong_manifest}.zip")
        with zipfile.ZipFile(archive, "x") as bundle:
            for name, raw in contents.items():
                bundle.writestr(name, raw)
        return archive, view.sha(archive.read_bytes()), report, files

    def test_loader_checks_dependencies_and_normalizes_json_coverage(self):
        archive, pin, _, _ = self.checkpoint_zip()
        with patch.multiple(view, **{name: lambda *args: torch.nn.Linear(1, 1)
                                     for name in ("PaperHeatmapFrontend", "V1Body", "V1Hand")}):
            _, pins, _, _ = view._models(archive, self.spec, True, training_sha256=pin, run_id=view.RUN_ID)
            self.assertEqual({role: value["epoch"] for role, value in pins.items()}, {"frontend": 9, "body": 9, "hand": 8})
            archive, pin, _, _ = self.checkpoint_zip(wrong_dependency=True)
            with self.assertRaisesRegex(ValueError, "best_binding"):
                view._models(archive, self.spec, True, training_sha256=pin, run_id=view.RUN_ID)
        with self.assertRaisesRegex(ValueError, "explicit_training_identity"):
            view._models(archive, self.spec, True, run_id=view.RUN_ID)

    def test_milestone_keeps_recorded_upstream_dependencies_and_receipt_pins(self):
        _, _, report, files = self.checkpoint_zip()
        pin = copy.deepcopy(report["cursor"]["best"]["hand"])
        pin.update(name="milestone-hand-e005.pt", epoch=5,
                   dependencies={role: report["cursor"]["best"][role]["sha256"] for role in ("frontend", "body")})
        files[pin["name"]] = {key: pin[key] for key in ("bytes", "sha256")}
        report["cursor"]["milestones"] = {"hand": {"5": pin}}
        chosen = view._selected_pins(report, files, True, 5)
        self.assertEqual(chosen["body"]["epoch"], 9)
        self.assertEqual(chosen["hand"]["epoch"], 5)
        pin["dependencies"]["body"] = "f"*64
        with self.assertRaisesRegex(ValueError, "milestone_dependencies"):
            view._selected_pins(report, files, True, 5)
        files[report["cursor"]["best"]["body"]["name"]]["sha256"] = "f"*64
        with self.assertRaisesRegex(ValueError, "checkpoint_pin"):
            view._selected_pins(report, files, True, None)

    def test_two_dimensional_comparison_accepts_frontend_only_and_common_3d_scale(self):
        buffer = io.BytesIO()
        Image.new("RGB", (16, 16)).save(buffer, format="PNG")
        raw = dict.fromkeys(("left", "right"), buffer.getvalue())
        points = [[[10., 10.]]*4]*2
        image = view.comparison_panel(raw, points, points, points, None, None, None, self.spec, 1,
                                      {"frontend": 9}, "Audited diagnostic sample")
        self.assertEqual(image.size, (1128, 888))
        small, large = view._shared_3d_groups(([[0., 0., 0.], [1., 0., 0.]],
                                             [[0., 0., 0.], [10., 0., 0.]]), 360)
        self.assertAlmostEqual((large[1][0]-large[0][0])/(small[1][0]-small[0][0]), 10.)

    def test_full_loader_uses_own_manifest_receipt_and_dependency_contract(self):
        with patch.multiple(view, **{name:lambda *args:torch.nn.Linear(1,1)
                                    for name in ("PaperHeatmapFrontend","V1Body","V1Hand")}):
            archive,pin,_,_ = self.checkpoint_zip(full=True)
            _,pins,_,_ = view._models(archive,self.spec,True,training_sha256=pin,run_id=view.FULL_RUN_ID)
            self.assertEqual({r:p['epoch'] for r,p in pins.items()},{'frontend':9,'body':10,'hand':9})
            for changes,error in (({'wrong_dependency':True},'best_binding'),({'wrong_manifest':True},'v12_view_training_binding')):
                archive,pin,_,_ = self.checkpoint_zip(full=True,**changes)
                with self.assertRaisesRegex(ValueError,error):
                    view._models(archive,self.spec,True,training_sha256=pin,run_id=view.FULL_RUN_ID)
        with self.assertRaisesRegex(ValueError,'explicit_training_identity'):
            view._models(archive,self.spec,True,run_id=view.FULL_RUN_ID)
