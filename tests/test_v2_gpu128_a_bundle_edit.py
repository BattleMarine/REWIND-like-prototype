"""Edited diffusion execution with deterministic source identity and repeat-call guard."""

import json
from pathlib import Path
import runpy
import shutil
import sys
import tempfile
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import patch

import torch

from rewind_v1.v2_contract import ROOT
from scripts import v2_gpu128_a_bundle_edit as edit
from scripts import v2_gpu128_a_bundle_train_full as train


class EditableSourceTests(TestCase):
    def setUp(self):
        scratch = tempfile.TemporaryDirectory()
        self.addCleanup(scratch.cleanup)
        self.root = Path(scratch.name).resolve()
        self.frozen = self.root/"frozen"
        (self.frozen/"rewind_m2").mkdir(parents=True)
        self.module = "rewind_m2/diffusion.py"
        self.original = (ROOT/self.module).read_text(encoding="utf-8")
        (self.frozen/self.module).write_text(self.original, encoding="utf-8")
        (self.frozen/"spec.json").write_text("{}", encoding="utf-8")
        self.fork = self.root/"fork"
        shutil.copytree(self.frozen, self.fork)

    def snapshot(self, amount, name):
        source = self.root/name
        shutil.copytree(self.frozen, source)
        changed = self.original.replace("return a.sqrt()*clean+(1-a).sqrt()*noise", f"return a.sqrt()*clean+(1-a).sqrt()*noise + {amount}")
        self.assertNotEqual(changed, self.original)
        (self.fork/self.module).write_text(changed, encoding="utf-8")
        snapshot = edit.apply_snapshot(source, self.fork, train.SOURCE_SHA)
        self.assertEqual(edit.validate_pin(snapshot, source, train.SOURCE_SHA), snapshot["source_sha256"])
        return source, snapshot

    def test_real_diffusion_executes_edited_q_sample_with_snapshot_and_resume_guard(self):
        def run(root):
            schedule = runpy.run_path(str(root/self.module))["DiffusionSchedule"](20, .001, .02)
            return schedule.q_sample(torch.zeros((1,2,1,3)), torch.zeros((1,2,1,3)), torch.tensor([10]))

        one, first = self.snapshot(.25, "one")
        same, identical = self.snapshot(.25, "same")
        two, changed = self.snapshot(.5, "two")
        self.assertTrue(torch.equal(run(self.frozen), torch.zeros(1,2,1,3)))
        self.assertTrue(torch.equal(run(one), torch.full((1,2,1,3), .25)))
        self.assertTrue(torch.equal(run(two), torch.full((1,2,1,3), .5)))
        self.assertEqual(first, identical)
        self.assertEqual(run(same).tolist(), run(one).tolist())
        output = self.root/"run"
        output.mkdir()
        (output/"experiment.json").write_bytes(edit.canonical({"source":first["source_sha256"]}))
        edit.require_resume_source(json.loads((output/"experiment.json").read_bytes()), identical["source_sha256"])
        proof = self.root/"pin.json"
        proof.write_bytes(edit.canonical(changed))

        def should_not_open(*_args, **_kwargs):
            self.fail("Changed source must fail before requesting the bundle")

        with patch.object(train.contract,"ROOT",two), patch.dict(sys.modules, {"v2_gpu128_a_bundle_local":
            SimpleNamespace(open_bundle=should_not_open)}), self.assertRaisesRegex(ValueError,"portable_full_run_source_changed"):
            train.main(["--bundle",str(self.root/"bundle.zip"), "--work",str(self.root/"unused"),"--output",str(output),
                        "--resume-sha256","f"*64,"--source-identity-file",str(proof),"--editable-origin",str(self.fork)])
        self.assertFalse((self.root/"unused").exists())

    def test_manifest_assets_are_not_editable(self):
        (self.fork/"spec.json").write_text("[]",encoding="utf-8")
        with self.assertRaisesRegex(ValueError,"editable_source_out_of_scope: spec.json"):
            edit.apply_snapshot(self.frozen,self.fork,train.SOURCE_SHA)
