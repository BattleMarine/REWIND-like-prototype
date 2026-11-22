"""Independent packaging regressions for the 128 B/C deliverables."""

from copy import deepcopy
import json
from pathlib import Path
import tempfile
from unittest import TestCase
from unittest.mock import patch
import zipfile

from scripts import build_v2_variant_bundle as builder
from scripts import v2_gpu128_a_bundle_format as fmt


def seal(schema, **fields):
    body = {"schema": schema, **fields}
    return {**body, "content_sha256": fmt.sha(fmt.canonical(body))}


def zip_files(path, files):
    with zipfile.ZipFile(path, "w") as archive:
        for name, raw in files.items():
            archive.writestr(name, raw)
    return {"name": path.name, **builder._pin(path.read_bytes())}


class VariantBundleBuilderTests(TestCase):
    def setUp(self):
        scratch = tempfile.TemporaryDirectory()
        self.addCleanup(scratch.cleanup)
        self.root = Path(scratch.name)
        self.base = {"run_id": "test", "slot": "A", "seed": 42, "source": "source",
                     "manifest": "manifest", "cache_boundary": fmt.FINAL_BOUNDARY, "eval_seed": 71}
        self.clips = {"scenario": ["scenario/clip", 0]}
        self.base_diagnostic = {"identity": self.base, "cache_shards": 1063, "counts": {"scenario": 136011}}
        self.base_comparison = {"binding": {"clips": self.clips, "model": {"frontend": "A-weight"}}}
        self.tools = {"v2_variant.py": b"# launcher", "tools/v2_variant_runtime.py": b"# runtime",
                      "GUIDE_KO.md": "한국어 안내".encode()}

    def fixture(self, slot, *, status="completed", feature=None, **override_identity):
        work = self.root / slot
        work.mkdir()
        seed = builder.SEEDS[slot]
        identity = {**self.base, "slot": slot, "seed": seed, **override_identity}
        files = {f"best-{role}-e003.pt": f"{slot}-{role}".encode() for role in builder.ROLES}
        last = b"final state"
        files["last.pt"] = last
        roster = {name: builder._pin(raw) for name, raw in files.items()}
        best = {role: {"name": f"best-{role}-e003.pt", **roster[f"best-{role}-e003.pt"]}
                for role in builder.ROLES}
        checkpoint = work / "source-checkpoint.zip"
        zip_pin = zip_files(checkpoint, {**files, "recovery_receipt.json": fmt.canonical(
            seal("rewind_v2_result_recovery_v1", files=roster))})
        step = seal("rewind_v2_gpu_training_checkpoint_v1", completed=True, cursor=builder.CURSOR,
                    identity=identity, last=roster["last.pt"], archive=zip_pin)
        binding = {"experiment": identity, "eval_seed": 71, "clips": self.clips,
                   "model": {role: best[role]["sha256"] for role in builder.ROLES}}
        comparison_zip = work / "source-comparison.zip"
        metrics = {"binding": binding, "models": best, "scenario_scores": {"scenario": {}}, "macro_score_mm": 1}
        compare_pin = zip_files(comparison_zip, {"comparison_metrics.json": fmt.canonical(metrics),
                                                "scenario.gif": b"gif data", "scenario.npz": b"array data"})
        comparison = seal("rewind_v2_gpu_comparison_v1", binding=binding, archive=compare_pin,
                          scores=metrics["scenario_scores"], macro_score_mm=metrics["macro_score_mm"])
        diagnostic = seal("rewind_v2_gpu_invocation_v1", **{**self.base_diagnostic, "identity": identity},
                          slot=slot, status=status, checkpoint=step, trained_updates=builder.CURSOR["updates"],
                          checkpoint_cursor={**builder.CURSOR, "best": best, "milestones": {}},
                          common_clips=self.clips, comparison=comparison,
                          timing={"feature_identity": feature or builder.FEATURE_IDENTITIES[slot]})
        marker_path = work / "original-comparison.json"
        marker_path.write_bytes(fmt.canonical(comparison))
        diagnostic_path = work / "original-diagnostic.json"
        diagnostic_path.write_bytes(fmt.canonical(diagnostic))
        return {"slot": slot, "base": self.root / "A.zip", "diagnostic": diagnostic_path, "checkpoint": checkpoint,
                "comparison": marker_path, "comparison_zip": comparison_zip, "output": work / "portable.zip"}

    def build(self, args, *, set_diagnostic_pin=True, set_comparison_pin=True):
        pins = deepcopy(builder.DIAGNOSTIC_SHA)
        comparisons = deepcopy(builder.COMPARISON_SHA)
        if set_diagnostic_pin:
            pins[args["slot"]] = fmt.sha(args["diagnostic"].read_bytes())
        if set_comparison_pin:
            comparisons[args["slot"]] = fmt.sha(args["comparison"].read_bytes())
        with (patch.object(builder, "DIAGNOSTIC_SHA", pins),
              patch.object(builder, "COMPARISON_SHA", comparisons),
              patch.object(builder, "_base_metadata", return_value=(self.base_diagnostic, self.base_comparison, {})),
              patch.object(builder, "_tools", return_value=self.tools)):
            return builder.build(**args)

    def test_b_and_c_publish_distinct_cache_free_bundles_with_original_evidence(self):
        for slot, seed in builder.SEEDS.items():
            with self.subTest(slot=slot):
                args = self.fixture(slot)
                report = self.build(args)
                external = args["output"].read_bytes()
                self.assertEqual(report["sha256"], fmt.sha(external))
                self.assertEqual(report["bytes"], len(external))
                with zipfile.ZipFile(args["output"]) as archive:
                    manifest = json.loads(archive.read("variant.json"))
                    fmt.check_json_pin(manifest, builder.SCHEMA)
                    self.assertEqual((manifest["slot"], manifest["seed"]), (slot, seed))
                    original = {"diagnostic.json": args["diagnostic"], "checkpoint.zip": args["checkpoint"],
                                "comparison.json": args["comparison"], "comparison.zip": args["comparison_zip"]}
                    expected = {**self.tools, **{name: path.read_bytes() for name, path in original.items()}}
                    self.assertEqual(set(archive.namelist()), set(expected) | {"variant.json"})
                    self.assertEqual(manifest["members"], {name: builder._pin(raw) for name, raw in expected.items()})
                    for name, raw in expected.items():
                        self.assertEqual(archive.read(name), raw)
                with self.assertRaises(FileExistsError):
                    self.build(args)
                self.assertEqual(args["output"].read_bytes(), external)

    def test_rejects_incomplete_or_wrong_seed_even_with_sealed_receipts(self):
        for args in (self.fixture("B", status="stopped"), self.fixture("C", seed=71)):
            with self.subTest(slot=args["slot"]), self.assertRaisesRegex(ValueError, "Completed slot"):
                self.build(args)
            self.assertFalse(args["output"].exists())

    def test_rejects_wrong_cache_and_a_features(self):
        for args, message in ((self.fixture("B", cache_boundary="wrong"), "image cache"),
                              (self.fixture("C", feature=fmt.FEATURE_IDENTITY), "feature identity")):
            with self.subTest(slot=args["slot"]), self.assertRaisesRegex(ValueError, message):
                self.build(args)
            self.assertFalse(args["output"].exists())

    def test_requires_the_fixed_original_diagnostic_and_a_zip(self):
        args = self.fixture("B")
        with self.assertRaisesRegex(ValueError, "Fixed slot diagnostic"):
            self.build(args, set_diagnostic_pin=False)
        with self.assertRaisesRegex(ValueError, "Fixed slot comparison"):
            self.build(args, set_comparison_pin=False)
        args["base"].write_bytes(b"wrong A bundle")
        with self.assertRaisesRegex(ValueError, "Fixed A bundle SHA-256"):
            builder._base_metadata(args["base"])
        self.assertFalse(args["output"].exists())

    def test_archive_payload_tampering_fails_without_publishing_partial_output(self):
        args = self.fixture("B")
        source = args["comparison_zip"]
        source.write_bytes(source.read_bytes().replace(b"gif data", b"bad data"))
        with self.assertRaises((ValueError, zipfile.BadZipFile)):
            self.build(args)
        self.assertFalse(args["output"].exists())
        self.assertEqual(list(args["output"].parent.glob(".v2-variant-*")), [])

    def test_shared_tools_normalize_crlf_and_require_the_fixed_bytes(self):
        scripts = self.root / "scripts"
        docs = self.root / "docs"
        scripts.mkdir()
        docs.mkdir()
        (docs / "V2_GPU128_BC_PORTABLE.md").write_bytes(self.tools["GUIDE_KO.md"])
        (scripts / "v2_variant.py").write_bytes(self.tools["v2_variant.py"])
        (scripts / "v2_variant_runtime.py").write_bytes(self.tools["tools/v2_variant_runtime.py"])
        expected = {}
        for suffix in builder.COMMON_TOOLS:
            name = f"v2_gpu128_a_bundle_{suffix}.py"
            data = f"# {suffix}\n".encode()
            (scripts / name).write_bytes(data.replace(b"\n", b"\r\n"))
            expected[f"tools/{name}"] = data
        pins = {name: builder._pin(raw) for name, raw in expected.items()}
        with patch.object(builder, "ROOT", self.root):
            self.assertEqual(builder._tools(pins), {**self.tools, **expected})
            (scripts / "v2_gpu128_a_bundle_format.py").write_bytes(b"# modified")
            with self.assertRaisesRegex(ValueError, "Fixed A code pin"):
                builder._tools(pins)
