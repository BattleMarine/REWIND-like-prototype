"""Fixed variant evidence, shared input isolation and native checkpoint recovery."""

from copy import deepcopy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import patch
import zipfile

import torch

from scripts import v2_gpu128_a_bundle_format as fmt
from scripts import v2_variant as launch
from scripts import v2_variant_runtime as runtime
from scripts.run_v2_gpu_training import identity_for, training_config
from rewind_prepare.temporal_selection import seal
from rewind_v1 import v2_retention as retention
from rewind_v1.v2_budget import Ledger


def pin(raw):
    return {"bytes": len(raw), "sha256": fmt.sha(raw)}


class VariantTests(TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.source_pins = {"source_sha256": fmt.SOURCE_SHA, "manifest_content_sha256": "1"*64}

    def write_variant(self, slot, checkpoint=None, model_pins=None, last=None):
        identity = identity_for(slot,self.source_pins)
        models = model_pins or {r: str(i)*64 for i,r in enumerate(sorted(runtime.ROLES),1)}
        root = self.root/slot
        root.mkdir()
        payload = {"checkpoint.zip": checkpoint or (slot+" archive").encode(), "comparison.zip": (slot+" compare").encode()}
        step = seal({"schema": "rewind_v2_gpu_training_checkpoint_v1", "identity": identity, "completed": True,
                     "cursor": runtime.FINAL_CURSOR, "archive": {"name":f"{slot}-actual.zip", **pin(payload["checkpoint.zip"])},
                     "last": last, "environment": {"device":"cuda:0", "device_name":"GPU", "device_memory_bytes":123}})
        comparison = seal({"schema": "rewind_v2_gpu_comparison_v1", "archive": {"name":f"{slot}-comparison.zip", **pin(payload["comparison.zip"])},
                           "binding": {"experiment": identity, "eval_seed": 71, "model": models}})
        diagnostic = seal({"schema":"rewind_v2_gpu_invocation_v1", "slot":slot, "identity":identity, "status":"completed",
                           "checkpoint":step, "comparison":comparison})
        payload.update({"tools/v2_variant_runtime.py": b"pass", "diagnostic.json":fmt.canonical(diagnostic),
                        "comparison.json":fmt.canonical(comparison)})
        self.write_payload(root,slot,payload)
        expected = {"seed":identity["seed"], "step":step["content_sha256"],
                    **{n:fmt.sha(payload[n]) for n in ("diagnostic.json","comparison.json")}}
        return root, expected, step, comparison

    def write_payload(self, root, slot, payload):
        payload = {**{f"tools/{name}": b"pass" for name in launch.BASE_TOOLS}, **payload}
        for name, raw in payload.items():
            path = root/name
            path.parent.mkdir(parents=True,exist_ok=True)
            path.write_bytes(raw)
        raw = fmt.canonical(seal({"schema":launch.SCHEMA,"seed":launch.SEEDS[slot],"slot":slot,
                                  "members":{n:pin(raw) for n,raw in payload.items()}}))
        (root/"variant.json").write_bytes(raw)
        self.assertEqual(launch.variant_manifest(raw)["slot"],slot)

    def test_two_seeds_keep_their_own_evidence_and_do_not_fallback(self):
        b, eb, _, _ = self.write_variant("B")
        c, ec, _, _ = self.write_variant("C")
        with patch.object(runtime,"EVIDENCE",{"B":eb,"C":ec}), fmt.BundleView(b) as bv, fmt.BundleView(c) as cv:
            for slot, seed, view in (("B",71,bv),("C",103,cv)):
                diagnostic, step, comparison = runtime.selected_evidence(view,self.source_pins)
                self.assertEqual((slot,seed),(diagnostic["slot"],step["identity"]["seed"]))
                self.assertEqual(comparison["binding"]["experiment"]["seed"],seed)
            (c/"checkpoint.zip").unlink()
            with self.assertRaisesRegex(ValueError,"checkpoint.zip"):
                runtime.selected_evidence(cv,self.source_pins)
            swapped = {k:v.read_bytes() for k in ("comparison.json","diagnostic.json") for v in (b/k,)}
            swapped.update({"checkpoint.zip":b"wrong", "comparison.zip":b"wrong", "tools/v2_variant_runtime.py":b"pass"})
            self.write_payload(c,"C",swapped)
            with self.assertRaisesRegex(ValueError,"variant_original_evidence_sha256"):
                runtime.selected_evidence(cv,self.source_pins)

    def test_base_reader_uses_only_index_and_fixed_source_receipt(self):
        receipt = fmt.canonical({"pins":self.source_pins})
        index = fmt.canonical(seal({"schema":fmt.SCHEMA,"members":{fmt.SOURCE_RECEIPT:pin(receipt),
                                                                   "A-private-checkpoint.zip":pin(b"unread")}}))
        reads = []

        def read(name, *_):
            reads.append(name)
            return {fmt.INDEX:index,fmt.SOURCE_RECEIPT:receipt}[name]

        with patch.object(runtime,"BASE_INDEX_SHA256",fmt.sha(index)), patch.object(fmt,"SOURCE_RECEIPT_SHA",fmt.sha(receipt)):
            self.assertEqual(runtime.base_inputs(SimpleNamespace(read=read))[1],self.source_pins)
        self.assertEqual(reads,[fmt.INDEX,fmt.SOURCE_RECEIPT])

    def test_selected_native_archive_restore_validates_parent_and_best(self):
        torch.set_num_threads(1)
        cfg = training_config(103)
        identity = identity_for("C",self.source_pins)
        ledger = Ledger(self.root/"usage",identity,cfg)
        source = self.root/"native"
        source.mkdir()
        binding = {"experiment":identity,"environment":{"device":"cuda:0"}}
        best = {}
        for role in runtime.ROLES:
            name = f"best-{role}-e003.pt"
            raw = role.encode()
            (source/name).write_bytes(raw)
            best[role] = {"name":name,**pin(raw)}
        torch.save({"binding":binding,"progress":{**runtime.FINAL_CURSOR,"best":best}},source/"last.pt")
        archived = self.root/"native.zip"
        with ledger.invocation("native"):
            retention.package_run(source,archived,cfg,ledger)
        root,evidence,step,comparison = self.write_variant("C", archived.read_bytes(), {r:p["sha256"] for r,p in best.items()},
                                                        {"name":"last.pt",**pin((source/"last.pt").read_bytes())})
        work = self.root/"work"
        work.mkdir()
        model = SimpleNamespace(root=work/"run", binding=deepcopy(binding))

        def resume(expected):
            raw = (model.root/"last.pt").read_bytes()
            self.assertEqual(fmt.sha(raw), expected)
            model.progress = torch.load(model.root/"last.pt",map_location="cpu",weights_only=True)["progress"]
            model.last = {"name":"last.pt",**pin(raw)}

        model._resume = resume
        with patch.object(runtime,"EVIDENCE",{"C":evidence}), fmt.BundleView(root) as view, ledger.invocation("restore"):
            runtime.selected_evidence(view,self.source_pins)
            found = runtime.restore_selected(view,work,model,step,comparison,cfg,ledger)
            self.assertEqual(found.read_bytes(),b"C compare")
            self.assertEqual(model.progress["best"],best)

    def test_duplicate_members_and_wrong_declared_seed_rejected(self):
        root,_,_,_ = self.write_variant("B")
        marker = json.loads((root/"variant.json").read_bytes())
        marker.pop("content_sha256")
        marker["seed"] = 103
        with self.assertRaisesRegex(ValueError,"variant_manifest_identity"):
            launch.variant_manifest(fmt.canonical(seal(marker)))
        archive = self.root/"bad.zip"
        with self.assertWarns(UserWarning), zipfile.ZipFile(archive,"w") as output:
            output.writestr("variant.json",b"one")
            output.writestr("variant.json",b"two")
        with self.assertRaisesRegex(ValueError,"variant_duplicate_member"), launch.reader(archive):
            pass
