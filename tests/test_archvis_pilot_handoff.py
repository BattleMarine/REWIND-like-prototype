"""Local synthetic validation of the staged ArchVis handoff and recovery."""

from contextlib import ExitStack
import json
from pathlib import Path
import stat
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import zipfile

from rewind_archive.contracts import canonical
from rewind_prepare.converter import sha
from scripts import archvis_pilot as pilot
from scripts import build_archvis_pilot_handoff as builder
from scripts.archvis_pilot_bootstrap import extract_pinned_zip
import test_extracted_stereo


ROOT = Path(__file__).resolve().parents[1]


class BootstrapTests(unittest.TestCase):
    def setUp(self):
        parent = ROOT / "outputs/test_tmp"
        parent.mkdir(exist_ok=True, parents=True)
        tmp = tempfile.TemporaryDirectory(dir=parent)
        self.addCleanup(tmp.cleanup)
        self.base = Path(tmp.name)

    def make_zip(self, name, rows):
        path = self.base / name
        with zipfile.ZipFile(path, "x") as archive:
            for name, mode, content in rows:
                info = zipfile.ZipInfo(name)
                info.create_system = 3
                info.external_attr = mode << 16
                archive.writestr(info, content)
        return path

    def unpack(self, archive, name, **kwargs):
        options = dict(max_zip_bytes=65536, max_output_bytes=4096, max_members=8)
        options.update(kwargs)
        pin = options.pop("pin", sha(archive.read_bytes()))
        return extract_pinned_zip(archive, pin, self.base / name, **options)

    def test_pinned_source_revision_and_no_overwrite(self):
        revision = "a" * 40
        archive = self.make_zip("valid.zip", [("SOURCE_REVISION.txt", stat.S_IFREG, revision),
                                ("scripts/work.py", stat.S_IFREG, "pass\n")])
        self.unpack(archive, "safe", revision=revision)
        self.assertEqual((self.base / "safe/scripts/work.py").read_bytes(), b"pass\n")
        with self.assertRaisesRegex(ValueError, "zip_pin_or_destination"):
            self.unpack(archive, "safe", revision=revision)
        with self.assertRaisesRegex(ValueError, "zip_sha256_mismatch"):
            self.unpack(archive, "wrong_pin", pin="b" * 64)
        with self.assertRaisesRegex(ValueError, "zip_revision_mismatch"):
            self.unpack(archive, "wrong_rev", revision="b" * 40)
        with self.assertRaisesRegex(ValueError, "zip_output_limit"):
            self.unpack(archive, "too_large", max_output_bytes=5)

    def test_traversal_case_duplicate_symlink_and_implicit_file_dir_conflict(self):
        for i, rows in enumerate((
            [("../escape", stat.S_IFREG, "bad")],
            [("A/file", stat.S_IFREG, "one"), ("a/FILE", stat.S_IFREG, "two")],
            [("alias", stat.S_IFLNK, "../escape")],
            [("directory/child", stat.S_IFREG, "one"), ("directory", stat.S_IFREG, "two")],
        )):
            archive = self.make_zip(f"unsafe{i}.zip", rows)
            with self.assertRaises(ValueError):
                self.unpack(archive, f"unsafe{i}")
            self.assertFalse((self.base / f"unsafe{i}").exists())


class WorkflowTests(unittest.TestCase):
    def setUp(self):
        self.fixture = test_extracted_stereo.ExtractedStereoTests()
        self.addCleanup(self.fixture.doCleanups)
        self.fixture.setUp()
        self.base = self.fixture.base
        self.work = self.base / "work"
        inputs = self.work / "inputs"
        (inputs / "split").mkdir(parents=True)
        spec = self.work / "source/configs/real_archvis62_v1.json"
        spec.parent.mkdir(parents=True)
        spec.write_bytes(self.fixture.fixture.specpath.read_bytes())
        inventory = {
            "schema": "archvis_drive_metadata_join_20261109_v1", "failed": [], "capped": [],
            "sequences": [
                {"path": self.fixture.train, "split": "train", "source": "train_val", "frames": [0, 1, 2]},
                {"path": self.fixture.val, "split": "validation", "source": "train_val", "frames": [0]},
            ],
            "duplicates": [{"split": "train_val", "key": self.fixture.train + "#0", "type": "left"}]}
        data = {f"split/{s}.txt": (self.fixture.fixture.splitdir / f"{s}.txt").read_bytes()
                for s in ("train", "validation", "test")}
        data.update({"scaramuzza.json": self.fixture.fixture.calibration.read_bytes(),
                     "inventory.json": canonical(inventory), "motion_map.json": self.fixture.fixture.motion.read_bytes(),
                     "motion_review.json": b"{}", "mse.json": canonical({**builder.CONFIG, "loss_kind": "mse"}),
                     "ce.json": canonical({**builder.CONFIG, "loss_kind": "gaussian_cross_entropy"})})
        receipt = {"schema": "archvis_pilot_inputs_v1", "spec_sha256": sha(spec.read_bytes()),
                   "files": {name: {"bytes": len(raw), "sha256": sha(raw)} for name, raw in data.items()}}
        data["input_receipt.json"] = canonical(receipt)
        for name, raw in data.items():
            (inputs / name).write_bytes(raw)
        self.asset_data = data
        stack = self.enterContext(ExitStack())
        stack.enter_context(patch.object(pilot, "ENVIRONMENT", "Room"))
        stack.enter_context(patch.object(pilot, "TRAIN_COUNT", 2))
        stack.enter_context(patch.object(pilot, "INVENTORY_SHA256", sha(data["inventory.json"])))
        stack.enter_context(patch.object(builder, "INVENTORY_SHA256", sha(data["inventory.json"])))
        self.limits = dict(max_seconds=60, max_attempts=10, max_read_bytes=32 * pilot.MIB,
                           max_rss_bytes=1024 * pilot.MIB, max_disk_bytes=16 * pilot.MIB,
                           max_cuda_allocated_bytes=pilot.MIB, max_cuda_reserved_bytes=pilot.MIB)

    def plan(self):
        return pilot.preflight(self.work, self.base, self.fixture.rgb, self.fixture.fixture.raw,
                               max_seconds=20, max_entries=1000)

    def test_frozen_selection_cost_prepare_same_manifest_and_recovery(self):
        with patch("rewind_prepare.extracted.decode_rgb_payload", side_effect=AssertionError("no decode in preflight")):
            plan = self.plan()
        self.assertEqual(plan["status"], "ready_for_manifest_approval")
        self.assertEqual(plan["plan"]["counts"], {"train": 2, "validation": 1})
        self.assertEqual(plan["plan"]["duplicates"], [[self.fixture.train, 0]])
        train = plan["plan"]["payload_bytes"]["train"] + 6
        validation = plan["plan"]["payload_bytes"]["validation"] + 3
        self.assertEqual(plan["cost"]["fresh_two_run_data_bytes"], 2 * (3 * train + 2 * validation))
        self.assertEqual(plan["cost"]["resume_same_epoch_data_bytes_upper_each"], 2 * train + validation)
        prepared = pilot.prepare(self.work, plan["content_sha256"], max_read_bytes=32 * pilot.MIB,
                                 max_seconds=30, max_entries=1000)
        self.assertEqual(prepared["status"], "completed")
        frozen = pilot.comparison_plan(self.work, device="cpu", limits=self.limits)
        self.assertGreater(frozen["read_headroom_pct_each"]["mse"], 0)
        launched = []

        def fake_child(args, cwd, timeout):
            self.assertEqual(cwd, self.work / "source")
            self.assertEqual(timeout, 75)
            manifest_pin = args[args.index("--manifest-sha256") + 1]
            config = json.loads(Path(args[args.index("--config") + 1]).read_bytes())
            out = Path(args[args.index("--output") + 1])
            launched.append((manifest_pin, config))
            out.mkdir()
            (out / "report.json").write_bytes(canonical({
                "status": "completed", "binding": {"manifest_sha256": manifest_pin}, "config": config,
            }))
            return SimpleNamespace(returncode=0)

        result = pilot.compare(self.work, frozen["content_sha256"], launch=fake_child)
        self.assertEqual(result["status"], "completed")
        self.assertEqual([pin for pin, _ in launched], [prepared["manifest_sha256"]] * 2)
        self.assertEqual({k for k in launched[0][1] if launched[0][1][k] != launched[1][1][k]}, {"loss_kind"})
        with zipfile.ZipFile(pilot.recover(self.work)) as archive:
            self.assertIsNone(archive.testzip())
            names = set(archive.namelist())
            self.assertTrue({"preflight.json", "prepare_report.json", "mse/report.json", "ce/report.json",
                             "manifest/manifest.json", "recovery_receipt.json"} <= names)
            receipt = json.loads(archive.read("recovery_receipt.json"))
            self.assertEqual(receipt["files"]["manifest/manifest.json"]["sha256"], prepared["manifest_file_sha256"])

    def test_stat_change_stops_preparation_before_png_payloads(self):
        plan = self.plan()
        path = self.fixture.rgb / self.fixture.train / "fisheye_final_image/camera_left/final_1.png"
        with path.open("ab") as handle:
            handle.write(b"x")
        with patch("rewind_prepare.extracted.decode_rgb_payload", side_effect=AssertionError("no decode after drift")):
            with self.assertRaisesRegex(ValueError, "extracted_selection_plan_changed"):
                pilot.prepare(self.work, plan["content_sha256"], max_read_bytes=32 * pilot.MIB,
                              max_seconds=30, max_entries=1000)
        self.assertFalse((self.work / "manifest").exists())
        with zipfile.ZipFile(pilot.recover(self.work)) as archive:
            self.assertEqual(json.loads(archive.read("prepare_report.json"))["status"], "failed")

    def test_discovery_stops_at_category_roots_and_reports_ambiguous_matches(self):
        mount = self.base / "mount"
        for parent, role in (("main", "rgb"), ("copy", "rgb"), ("main", "pose")):
            category = mount / parent / pilot.CATEGORY[role]
            (category / "Room").mkdir(parents=True)
            # A nested category inside a hit must never be treated as a candidate.
            (category / "Room" / pilot.CATEGORY["pose"] / "Room").mkdir(parents=True)
        result = pilot.discover_roots(mount, max_entries=20, max_depth=4)
        self.assertIsNone(result["stopped"])
        self.assertEqual({k: len(v) for k, v in result["candidates"].items()}, {"rgb": 2, "pose": 1})
        bounded = pilot.discover_roots(mount, max_entries=1, max_depth=4)
        self.assertEqual(bounded["stopped"], "discovery_budget")

    def test_first_run_stop_recovers_last_checkpoint_without_starting_ce(self):
        plan = self.plan()
        pilot.prepare(self.work, plan["content_sha256"], max_read_bytes=32 * pilot.MIB,
                      max_seconds=30, max_entries=1000)
        frozen = pilot.comparison_plan(self.work, device="cpu", limits=self.limits)
        launched = []

        def bounded_child(args, **_):
            output = Path(args[args.index("--output") + 1])
            launched.append(output.name)
            output.mkdir()
            (output / "last.pt").write_bytes(b"fixture checkpoint")
            (output / "report.json").write_bytes(canonical({"status": "bounded_stop"}))
            return SimpleNamespace(returncode=2)

        result = pilot.compare(self.work, frozen["content_sha256"], launch=bounded_child)
        self.assertEqual(launched, ["mse"])
        self.assertEqual(result["status"], "stopped_with_results")
        with zipfile.ZipFile(pilot.recover(self.work)) as archive:
            self.assertEqual(archive.read("mse/last.pt"), b"fixture checkpoint")
            self.assertNotIn("ce/report.json", archive.namelist())

    def ready_packages(self, name, *, source_spec=None, assets=None):
        source = self.base / (name + "_source.zip")
        inputs = self.base / (name + "_inputs.zip")
        revision = "e" * 40
        with zipfile.ZipFile(source, "x") as archive:
            archive.writestr("SOURCE_REVISION.txt", revision)
            archive.writestr("scripts/archvis_pilot.py", "# local fixture")
            archive.writestr("configs/real_archvis62_v1.json", source_spec if source_spec is not None
                             else self.fixture.fixture.specpath.read_bytes())
        with zipfile.ZipFile(inputs, "x") as archive:
            for member, raw in (self.asset_data if assets is None else assets).items():
                archive.writestr(member, raw)
        return source, inputs, revision

    def test_ready_notebook_pins_assets_without_self_reference(self):
        source, inputs, revision = self.ready_packages("valid")
        output = self.base / "ready.ipynb"
        pins = builder.build_ready(source, inputs, revision, output)
        doc = json.loads(output.read_bytes())
        bootstrap = "".join(doc["cells"][1]["source"])
        self.assertIn(pins["source"]["sha256"], bootstrap)
        self.assertIn(pins["inputs"]["sha256"], bootstrap)
        self.assertNotIn("MANIFEST_CONTENT_SHA256", bootstrap)
        self.assertEqual([c["id"] for c in doc["cells"]],
                         ["guide", "bootstrap", "discover", "preflight", "prepare", "budget", "train", "recover"])
        for cell in doc["cells"]:
            if cell["cell_type"] == "code":
                compile("".join(cell["source"]), cell["id"], "exec")

    def test_ready_rejects_archive_spec_line_endings_before_notebook_creation(self):
        spec = self.fixture.fixture.specpath.read_bytes().replace(b"\r\n", b"\n").rstrip(b"\n")
        assets = self.asset_data.copy()
        receipt = json.loads(assets["input_receipt.json"])
        receipt["spec_sha256"] = sha(spec + b"\r\n")
        assets["input_receipt.json"] = canonical(receipt)
        source, inputs, revision = self.ready_packages("line_endings", source_spec=spec + b"\n", assets=assets)
        output = self.base / "mismatched_ready.ipynb"
        with self.assertRaisesRegex(ValueError, "archvis_ready_source_spec_pin_mismatch"):
            builder.build_ready(source, inputs, revision, output)
        self.assertFalse(output.exists())

    def test_ready_rejects_altered_official_files_with_resealed_receipts(self):
        for i, name in enumerate(("split/validation.txt", "scaramuzza.json")):
            assets = self.asset_data.copy()
            assets[name] += b"\n"
            receipt = json.loads(assets["input_receipt.json"])
            receipt["files"][name] = {"bytes": len(assets[name]), "sha256": sha(assets[name])}
            assets["input_receipt.json"] = canonical(receipt)
            source, inputs, revision = self.ready_packages(f"official{i}", assets=assets)
            output = self.base / f"mismatched_official{i}.ipynb"
            with self.assertRaisesRegex(ValueError, "archvis_ready_official_pin_mismatch"):
                builder.build_ready(source, inputs, revision, output)
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
