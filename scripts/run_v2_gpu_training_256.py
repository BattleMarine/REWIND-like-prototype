"""One manual 256px GPU training invocation, using the frozen 128 training policy."""

from types import SimpleNamespace

from rewind_archive.contracts import digest
from scripts import run_v2_gpu_training as common
from scripts.run_v2_cache256 import STORE, scan
from scripts.v2_gpu_training_cache256 import RangeCache256, SIDE
from scripts.v2_gpu_training_pipeline import FullData
from scripts.v2_gpu_training_storage import EVAL_SEED, FRAMES, GIB, SEEDS, SHARDS


RUN_FOLDER = "v2_gpu_training_256_v1"
RUN_ID = "V2-FULL-256-3X3-v1"
BOUNDARY = "2ae0b8523c89590e48d3363b26fcca9a18cf387308b6871b1a253b8b0849e33f"
PRODUCER_BINDING = "6a9c4a5e175dd29420337233a29063d015bb9e9fc8b2aa4e296f6eae07ebe077"
RESOURCE_LIMITS = {**common.RESOURCE_LIMITS, "read_bytes": 512*GIB, "cache_bytes": 24*GIB}


def training_config(seed):
    return {**common.training_config(seed), "run_id": RUN_ID, "resources": dict(RESOURCE_LIMITS)}


def identity_for(slot, pins):
    seed = SEEDS[slot]
    return {"run_id": RUN_ID, "seed": seed, "slot": slot, "source": pins["source_sha256"],
            "manifest": pins["manifest_content_sha256"], "cache_boundary": BOUNDARY, "eval_seed": EVAL_SEED,
            "input_size": [SIDE, SIDE], "config": digest(training_config(seed))}


class FullData256(FullData):
    image_size = (SIDE, SIDE)

    def frame(self, key, split, size):
        if tuple(size) != self.image_size:
            raise ValueError("v2_gpu256_frame_size")
        image, target = self.cache.frame(key, split)
        return (image, target["xy"], target["xy_valid"]), {}


def load_inputs(args, pins):
    if pins["size"] != SIDE or pins["final_boundary_sha256"] != BOUNDARY or pins["producer_256_binding"] != PRODUCER_BINDING:
        raise ValueError("v2_gpu256_release_pin")
    return common.load_inputs(args, pins, config=training_config, require_full_local_cache=False)


def open_cache(args, project, doc, spec, pins, cfg, ledger, drive, owner):
    store = project / "reports" / "cache" / STORE
    if not store.is_dir() or store.is_symlink() or not store.resolve().is_relative_to(project):
        raise ValueError("v2_gpu256_cache_path")
    binding = digest({"schema": "rewind_v2_cache256_binding_v1", "size": SIDE, "dtype": "float32",
                      "manifest": doc["content_sha256"], "spec": digest(spec), "source": pins["producer_256_overlay"],
                      "environment": pins["producer_256_runtime"], "resize": "bilinear_align_corners"})
    if binding != PRODUCER_BINDING:
        raise ValueError("v2_gpu256_producer_binding")
    entries = scan(store, doc, binding, ledger.check)
    if (len(entries) != SHARDS or entries[-1]["content_sha256"] != BOUNDARY or entries[-1]["frames"] != 75
            or sum(e["frames"] for e in entries) != FRAMES
            or len(list(store.glob("shard-*.zip"))) != SHARDS):
        raise ValueError("v2_gpu256_final_boundary")
    packages = sum(e["package"]["bytes"] for e in entries)
    if not pins["reported_cache_bytes"] - 64*1024**2 <= packages <= pins["reported_cache_bytes"]:
        raise ValueError("v2_gpu256_package_total")
    images = RangeCache256(args.workspace/"cache256", drive/"frame-index", store, entries, doc, spec, binding, cfg, ledger, owner)
    images.metrics["source_package_bytes"] = packages
    return images, len(entries)


PROFILE = SimpleNamespace(run_folder=RUN_FOLDER, config=training_config, identity=identity_for,
                          load_inputs=load_inputs, open_cache=open_cache, data=FullData256)


if __name__ == "__main__":
    common.main(profile=PROFILE)
