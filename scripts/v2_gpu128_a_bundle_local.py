"""Open a standalone bundle for pinned model loading and bounded cache access."""

from collections import OrderedDict
from contextlib import contextmanager
import json
from pathlib import Path
import shutil
import tempfile
from types import SimpleNamespace

import torch

from rewind_archive.contracts import digest
from rewind_v1 import v2_cache, v2_contract as contract, v2_retention as retention
from rewind_v1.v2_budget import Ledger
from scripts.archvis_pilot_bootstrap import extract_pinned_zip
from scripts.run_v2_bulk_cache import Journal, PACKAGE_CAP, rows_for
from scripts.run_v2_gpu_cache_probe import load_catalog
from scripts.run_v2_gpu_training import load_inputs, training_config
from scripts.v2_gpu_training_pipeline import FullCache, FullData, Training

if __package__:
    from . import v2_gpu128_a_bundle_format as fmt
else:
    import v2_gpu128_a_bundle_format as fmt


CACHE_SCRATCH_CAP = 2*fmt.GIB


def pinned_journal(view, index, pins, doc, spec, ledger, workspace):
    members = index["members"]
    catalog_name = f"{fmt.LEGACY}/catalog.json"
    anchor = load_catalog(view.read(catalog_name, 64*1024, members[catalog_name]), pins)
    producer = {**v2_cache.binding(doc, spec, contract.config()), "source": pins["bulk_producer_source"],
                "torch": "2.14.1+cpu", "numpy": "2.4.6", "pillow": "11.3.0"}
    if digest(producer["source"]) != pins["bulk_producer_source_identity"]:
        raise ValueError("portable_cache_producer_pin")
    journal = Journal(workspace/"bulk", workspace/"legacy", doc, anchor, producer)
    entries = []
    previous = None
    for i in range(1063):
        ledger.check()
        name = f"{fmt.BULK}/boundary-{i:05d}.json"
        entry = json.loads(view.read(name, 16*1024, members[name]))
        expected = journal.genesis if i == 0 else journal.entry(i, previous, entry["package"], digest(journal.binding))
        if entry != expected:
            raise ValueError("portable_cache_boundary_chain")
        previous = entry["content_sha256"]
        package = entry["package"]
        base = fmt.LEGACY if i == 0 else fmt.BULK
        expected_name = f"shard-{i:05d}-{package['sha256']}.zip"
        if (package["name"] != expected_name or package["number"] != i or not 0 < package["bytes"] <= PACKAGE_CAP
                or members[f"{base}/{expected_name}"] != {k: package[k] for k in ("bytes", "sha256")}):
            raise ValueError("portable_cache_package_pin")
        entries.append(entry)
    if (index["cache_shards"] != len(entries) or index["frame_pairs"] != 136011 or entries[-1]["frames"] != 75
            or entries[-1]["content_sha256"] != fmt.FINAL_BOUNDARY or index["cache_boundary_sha256"] != fmt.FINAL_BOUNDARY
            or sum(e["frames"] for e in entries) != 136011
            or sum(e["package"]["bytes"] for e in entries) != index["cache_package_bytes"]):
        raise ValueError("portable_cache_final_boundary")
    return journal, anchor, entries


class BundleCache(FullCache):
    def __init__(self, root, journal, anchor, entries, spec, cfg, ledger, view, *, cap=CACHE_SCRATCH_CAP):
        super().__init__(root, journal, anchor, entries, spec, cfg, ledger)
        self.view, self.recent = view, OrderedDict()
        self.used = 0
        self.cap = cap
        self.metrics["bundle_read_bytes"] = 0

    def prepare_frontend(self):
        return  # Tools pass bounded frames directly from the pinned cache.

    def frame(self, key, split):
        number = self.index[tuple(key)] // v2_cache.CHUNK
        item = super().frame(key, split)
        self.recent.move_to_end(number)
        return item

    def _ensure(self, entries):
        for row in entries:
            index, package = row["index"], row["package"]
            while self.recent and self.used+package["bytes"] > self.cap:
                expired, size = self.recent.popitem(last=False)
                if expired in self.loaded:
                    images, _ = self.loaded.pop(expired)
                    images._mmap.close()
                shutil.rmtree(self.root/f"shard-{expired:05d}")
                self.verified.remove(expired)
                self.used -= size
            self.ledger.check()
            installed = self.root/f"shard-{index:05d}"
            binding = self.anchor["binding"] if index == 0 else self.journal.binding
            if installed.exists() or installed.is_symlink():
                recovered = v2_cache.receipt(self.root, index, binding, rows_for(self.doc, index), self.ledger)
                if recovered["content_sha256"] != package["receipt_sha256"]:
                    raise ValueError("portable_cache_reused_shard_receipt")
                self.metrics["reused_shards"] += 1
                self._remember(index, package["bytes"])
                continue
            if shutil.disk_usage(self.root).free < 2*PACKAGE_CAP+fmt.MIB:
                raise OSError("portable_cache_local_disk_low")
            base = fmt.LEGACY if not index else fmt.BULK
            with tempfile.TemporaryDirectory(dir=self.root, prefix=".incoming-") as directory:
                local = Path(directory)
                name = f"{base}/{package['name']}"
                archive_path = self.view.copy(name, local/"shard.zip", package, self.ledger.check)
                folder = local/f"shard-{index:05d}"
                extract_pinned_zip(archive_path, package["sha256"], folder, max_members=3,
                                   max_zip_bytes=PACKAGE_CAP, max_output_bytes=PACKAGE_CAP)
                receipt = v2_cache.receipt(local, index, binding, rows_for(self.doc, index), self.ledger)
                if receipt["content_sha256"] != package["receipt_sha256"]:
                    raise ValueError("portable_cache_local_shard_receipt")
                folder.rename(self.root/folder.name)
            self.metrics["bundle_read_bytes"] += package["bytes"]
            self._remember(index, package["bytes"])

    def _remember(self, index, size):
        self.used += size
        self.recent[index] = size
        self.verified.add(index)
        self.metrics["verified_shards"] = len(self.verified)


@contextmanager
def open_bundle(bundle: Path, workspace: Path, *, seconds: int, device: str, full_cache=None, restore_models=True,
                full_cache_headroom=6*fmt.GIB, editable_manifest=None):
    """Restore verified trained weights, then expose image data only when requested."""
    bundle, workspace = Path(bundle).resolve(strict=True), Path(workspace).absolute()
    if (workspace.exists() or workspace.is_symlink() or not workspace.parent.is_dir()
            or bundle == workspace or bundle.is_dir() and workspace.resolve().is_relative_to(bundle)
            or shutil.disk_usage(workspace.parent).free < 4*fmt.GIB):
        raise ValueError("portable_bundle_new_work_directory_with_4GiB_free")
    if device not in ("cpu", "cuda") or device == "cuda" and not torch.cuda.is_available():
        raise ValueError("portable_bundle_device_unavailable")
    if type(seconds) is not int or not 1 <= seconds <= 28800:
        raise ValueError("portable_bundle_seconds")
    torch.set_num_threads(1)
    workspace.mkdir()
    with fmt.BundleView(bundle) as view:
        index = view.index()
        members = index["members"]
        pins, diagnostic, step, comparison = fmt.metadata(view, members)
        if index["identity"] != step["identity"] or index["checkpoint_receipt_sha256"] != fmt.FINAL_RECEIPT:
            raise ValueError("portable_bundle_index_identity")
        groups = fmt.feature_groups(view, members)
        if index["features"] != {"identity": fmt.FEATURE_IDENTITY, "groups": len(groups), "clips": 3518,
                                 "package_bytes": sum(g[2]["bytes"] for g in groups.values())}:
            raise ValueError("portable_bundle_feature_manifest")
        consumer_pins = pins
        if editable_manifest is not None:
            if restore_models:
                raise ValueError("editable_source_requires_fresh_model")
            # The consumer core has changed. All original manifest, spec and cache producer pins stay in force.
            consumer_pins = {**pins, "training_core_sha256": digest(contract.source())}
        data, spec, plan = load_inputs(SimpleNamespace(invocation=workspace, workspace=workspace, slot="A"), consumer_pins)
        cfg = training_config(42)
        limits = {**cfg, "seconds_per_invocation": seconds, "external_timeout_seconds": seconds,
                  "terminal_reserve_seconds": 0, "copy_seconds": min(seconds, 600)}
        ledger = Ledger(workspace/"usage", step["identity"], limits)
        with ledger.invocation("portable_v2_128_a"):
            journal, anchor, entries = pinned_journal(view, index, pins, data, spec, ledger, workspace)
            cache_root = workspace/"cache" if full_cache is None else full_cache
            if full_cache is not None:
                if full_cache.is_symlink() or bundle.is_dir() and full_cache.resolve().is_relative_to(bundle):
                    raise ValueError("portable_full_cache_outside_bundle")
                missing = sum(e["package"]["bytes"] for e in entries if not (full_cache/f"shard-{e['index']:05d}").is_dir())
                if shutil.disk_usage(full_cache.parent).free < missing + full_cache_headroom:
                    raise OSError(f"portable_full_cache_space: need {missing + full_cache_headroom} free bytes")
            images = BundleCache(cache_root, journal, anchor, entries, spec, cfg, ledger, view,
                                cap=max(CACHE_SCRATCH_CAP,index["cache_package_bytes"]) if full_cache is not None else CACHE_SCRATCH_CAP)
            try:
                inputs = FullData(data, spec, cfg, plan, images, ledger, fmt.SOURCE_SHA)
                model = Training(cfg, inputs, workspace/"features", torch.device("cuda:0" if device == "cuda" else "cpu"),
                                 workspace/"run", workspace, ledger, lambda: None, step["identity"])
                if not restore_models:
                    yield SimpleNamespace(pipeline=model, ledger=ledger, diagnostic=diagnostic, comparison=comparison,
                                          bundle=bundle, root=workspace, comparison_path=None)
                    return
                # Weight loading is portable. Binding and best dependencies remain those of the authenticated run.
                model.binding["environment"] = {k: v for k, v in step["environment"].items()
                                               if k not in {"device_name", "device_memory_bytes"}}
                archive = step["archive"]
                local = view.copy(f"{fmt.A}/checkpoints/{archive['name']}", workspace/"trained-original.zip", archive, ledger.check)
                last_sha = retention.restore_run(local, archive, model.root, limits, ledger, model.binding)
                if last_sha != step["last"]["sha256"]:
                    raise ValueError("portable_bundle_last_sha")
                local.unlink()
                model._resume(last_sha)  # No optimizer restore or training is performed while opening the bundle.
                if model.last != step["last"] or any(model.progress[k] != v for k, v in step["cursor"].items()):
                    raise ValueError("portable_bundle_final_cursor")
                best = model.progress["best"]
                if ({k: v["sha256"] for k, v in best.items()} != comparison["binding"]["model"]
                        or {k: {f: v[f] for f in ("name", "bytes", "sha256")} for k, v in best.items()} != index["selected_models"]):
                    raise ValueError("portable_bundle_selected_model_pin")
                compare_pin = comparison["archive"]
                comparison_path = view.copy(f"{fmt.A}/{compare_pin['name']}", workspace/compare_pin["name"], compare_pin, ledger.check)
                yield SimpleNamespace(pipeline=model, ledger=ledger, diagnostic=diagnostic, comparison=comparison,
                                      bundle=bundle, root=workspace, comparison_path=comparison_path)
            finally:
                images.close()
