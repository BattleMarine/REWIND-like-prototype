"""Stream v2 128/A into a verified ZIP64 while leaving all evidence read-only."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import time
from types import SimpleNamespace
import uuid
import zipfile

from scripts.run_v2_gpu_training import load_inputs, read_journal
from scripts.v2_gpu_training_comparison import shared_clips
from rewind_v1 import v2_contract as contract

if __package__:
    from . import v2_gpu128_a_bundle_format as fmt
else:
    import v2_gpu128_a_bundle_format as fmt


OUTPUT = "v2_128_A_final_model_and_cache.zip"
TOOLS = ("v2_gpu128_a_bundle_format.py", "v2_gpu128_a_bundle_local.py",
         "v2_gpu128_a_bundle_predict.py", "v2_gpu128_a_bundle_tune.py",
         "v2_gpu128_a_bundle_train_full.py", "v2_gpu128_a_bundle_edit.py")


class Deadline:
    def __init__(self, seconds):
        self.started = time.monotonic()
        self.seconds = seconds

    def check(self):
        if self.elapsed() >= self.seconds:
            raise TimeoutError("v2_bundle_export_deadline")

    def elapsed(self):
        return time.monotonic()-self.started


class SequentialOutput:
    """No seek is exposed to zipfile, so each byte is sent to Drive once in order."""

    def __init__(self, stream):
        self.stream = stream
        self.hash = hashlib.sha256()
        self.size = 0

    def tell(self):
        return self.size

    def write(self, data):
        written = self.stream.write(data)
        if written != len(data):
            raise OSError("v2_bundle_short_outer_write")
        self.hash.update(data)
        self.size += written
        return written

    def flush(self):
        self.stream.flush()


def pin(path, deadline):
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"v2_bundle_source_missing: {path}")
    with path.open("rb") as source:
        hasher = hashlib.sha256()
        count = 0
        while block := source.read(2*fmt.MIB):
            deadline.check()
            count += len(block)
            if count > fmt.GIB:
                raise ValueError("v2_bundle_single_member_cap")
            hasher.update(block)
    return {"bytes": count, "sha256": hasher.hexdigest()}


def known(name, path, expected, deadline):
    item = pin(path, deadline)
    if item["sha256"] != expected:
        raise ValueError(f"v2_bundle_fixed_file_pin: {name}")
    return item


def native_package_pin(value):
    return {"bytes": value["bytes"], "sha256": value["sha256"]}


def model_roster(path, step, comparison):
    with zipfile.ZipFile(path) as archive:
        infos = {item.filename: item for item in archive.infolist()}
        manifest = infos["recovery_receipt.json"]
        if len(infos) != len(archive.infolist()) or not 0 < manifest.file_size <= 8*fmt.MIB:
            raise ValueError("v2_bundle_model_package_listing")
        recovery = json.loads(archive.read(manifest))
        fmt.check_json_pin(recovery, "rewind_v2_result_recovery_v1")
        parts = recovery["files"]
        if (set(infos) != set(parts) | {"recovery_receipt.json"}
                or parts["last.pt"] != native_package_pin(step["last"])):
            raise ValueError("v2_bundle_model_package_scope")
        for name, item in parts.items():
            if fmt.name_path(name).name != name or item["bytes"] != infos[name].file_size:
                raise ValueError("v2_bundle_model_package_member")
        chosen = {}
        for role, expected in comparison["binding"]["model"].items():
            matched = [n for n, p in parts.items() if n.startswith(f"best-{role}-") and p["sha256"] == expected]
            if len(matched) != 1:
                raise ValueError("v2_bundle_selected_model_pin")
            chosen[role] = {"name": matched[0], **parts[matched[0]]}
        if set(chosen) != {"frontend", "body", "hand"}:
            raise ValueError("v2_bundle_selected_roles")
        return chosen


def write_bundle(destination, files, body, clock):
    """Write a new ZIP, check whole-file readback and the central directory, then publish."""
    directory = destination.parent
    lock = directory/("."+destination.name+".lock")
    temporary = directory/("."+destination.name+".partial-"+uuid.uuid4().hex)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(destination)
    lock.mkdir()
    total = sum(p["bytes"] for _, p in files.values())
    try:
        if shutil.disk_usage(directory).free < total+2*fmt.MIB:
            raise OSError(f"v2_bundle_destination_space: at least {total+2*fmt.MIB} bytes")
        sizes = {name: p for name, (_, p) in files.items()}
        payload = {"schema": fmt.SCHEMA, **body, "members": sizes}
        index = fmt.canonical({**payload, "content_sha256": fmt.sha(fmt.canonical(payload))})
        with temporary.open("xb", buffering=2*fmt.MIB) as raw:
            sink = SequentialOutput(raw)
            with zipfile.ZipFile(sink, "w", compression=zipfile.ZIP_STORED, allowZip64=True) as archive:
                archive.writestr(fmt.INDEX, index)
                count = 0
                for name, (path, pinned) in files.items():
                    clock.check()
                    fmt.name_path(name)
                    if path.is_symlink() or path.stat().st_size != pinned["bytes"]:
                        raise ValueError(f"v2_bundle_source_size: {name}")
                    info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
                    info.external_attr = 0o100600 << 16
                    with path.open("rb") as source, archive.open(info, "w", force_zip64=True) as entry:
                        fmt.hash_stream(source, pinned, clock.check, entry)
                    count += pinned["bytes"]
                    if count//(2*fmt.GIB) != (count-pinned["bytes"])//(2*fmt.GIB):
                        print(f"ZIP source payload verified and written: {count}/{total} bytes", flush=True)
            sink.flush()
            os.fsync(raw.fileno())
            output_pin = {"bytes": sink.size, "sha256": sink.hash.hexdigest()}
        write_seconds = clock.elapsed()
        print(f"Full ZIP readback starts ({output_pin['bytes']} bytes)", flush=True)
        with temporary.open("rb") as stored:
            fmt.hash_stream(stored, output_pin, clock.check)
        with fmt.BundleView(temporary) as check:
            got = check.index()
            if got["members"] != sizes or any(check.members[name].file_size != p["bytes"] for name, p in sizes.items()):
                raise ValueError("v2_bundle_outer_central_directory")
        clock.check()
        if destination.exists() or destination.is_symlink():
            raise FileExistsError(destination)
        temporary.rename(destination)
        if destination.stat().st_size != output_pin["bytes"]:
            raise OSError("v2_bundle_published_size")
        return {"path": str(destination), **output_pin, "index_sha256": fmt.sha(index), "members": len(files)+1,
                "source_read_payload_bytes": total, "drive_zip_write_bytes": output_pin["bytes"],
                "zip_readback_bytes": output_pin["bytes"], "wall_seconds": clock.elapsed(),
                "preflight_and_write_seconds": write_seconds, "readback_and_publish_seconds": clock.elapsed()-write_seconds}
    finally:
        temporary.unlink(missing_ok=True)
        lock.rmdir()


def collect(project, source_zip, source_receipt, tools, work, clock):
    files = {}

    def add(name, path, *, expected=None, pinned=None):
        path = Path(path)
        if name in files:
            raise ValueError(f"v2_bundle_duplicate_member: {name}")
        got = pin(path, clock) if expected is None else known(name, path, expected, clock)
        if pinned is not None and native_package_pin(pinned) != got:
            raise ValueError(f"v2_bundle_source_member_pin: {name}")
        files[name] = path, got

    add(fmt.SOURCE, source_zip, expected=fmt.SOURCE_SHA)
    add(fmt.SOURCE_RECEIPT, source_receipt, expected=fmt.SOURCE_RECEIPT_SHA)
    pins = json.loads(source_receipt.read_bytes())["pins"]
    with fmt.BundleView(project) as view:
        add(fmt.DIAGNOSTIC, view.file(fmt.DIAGNOSTIC), expected=fmt.DIAGNOSTIC_SHA)
        add(fmt.STEP, view.file(fmt.STEP))
        add(fmt.COMPARISON, view.file(fmt.COMPARISON), expected=fmt.COMPARISON_SHA)
        final = json.loads(files[fmt.DIAGNOSTIC][0].read_bytes())
        model, demo = final["checkpoint"]["archive"], final["comparison"]["archive"]
        model_name, demo_name = f"{fmt.A}/checkpoints/{model['name']}", f"{fmt.A}/{demo['name']}"
        model_path, demo_path = view.file(model_name), view.file(demo_name)
        if model_path.stat().st_size != model["bytes"]:
            raise ValueError("v2_bundle_model_size")
        files[model_name] = model_path, native_package_pin(model)  # Full SHA is verified during the outer stream.
        add(demo_name, demo_path, pinned=demo)
        _, diagnostic, step, comparison = fmt.metadata(view_for_metadata(files), {n: p for n, (_, p) in files.items()})
        models = model_roster(model_path, step, comparison)
        groups = fmt.feature_groups(view)
        for marker, (raw, archive_name, archive_pin, folders) in groups.items():
            clock.check()
            expected = {f"{folder}/{part}" for folder in folders for part in ("clip.pt", "receipt.json")}
            feature = view.file(archive_name)
            if feature.stat().st_size != archive_pin["bytes"]:
                raise ValueError("v2_bundle_feature_package_size")
            with zipfile.ZipFile(feature) as package:
                if set(package.namelist()) != expected or len(package.infolist()) != len(expected):
                    raise ValueError("v2_bundle_feature_package_members")
            files[archive_name] = feature, archive_pin
            files[marker] = view.file(marker), {"bytes": len(raw), "sha256": fmt.sha(raw)}
        demo_members = {"comparison_metrics.json", *(f"{s}.{suffix}" for s in contract.SCENARIOS for suffix in ("gif", "npz"))}
        samples = work/"demo"
        samples.mkdir()
        with zipfile.ZipFile(demo_path) as archive:
            if set(archive.namelist()) != demo_members or len(archive.infolist()) != len(demo_members):
                raise ValueError("v2_bundle_demo_listing")
            for name in sorted(demo_members):
                info = archive.getinfo(name)
                if not 0 < info.file_size < 8*fmt.MIB:
                    raise ValueError("v2_bundle_demo_member_size")
                path = samples/name
                path.write_bytes(archive.read(info))
                add(f"demo/{name}", path)
        data, spec, plan = load_inputs(SimpleNamespace(invocation=work, workspace=work, slot="A"), pins)
        journal, anchor, entries = read_journal(project, data, spec, pins, clock)
        if len(entries) != 1063 or sum(e["frames"] for e in entries) != 136011 or shared_clips(plan) != {
                k: tuple(v) for k, v in diagnostic["common_clips"].items()}:
            raise ValueError("v2_bundle_fixed_dataset_scope")
        add(f"{fmt.LEGACY}/catalog.json", journal.legacy/"catalog.json", expected=pins["catalog_file_sha256"])
        for i, entry in enumerate(entries):
            clock.check()
            add(f"{fmt.BULK}/boundary-{i:05d}.json", journal.path(i))
            item = entry["package"]
            folder = fmt.LEGACY if i == 0 else fmt.BULK
            name = f"{folder}/{item['name']}"
            files[name] = view.file(name), native_package_pin(item)
        for reference in sorted((contract.ROOT/"reference").rglob("*")):
            if reference.is_file():
                name = reference.relative_to(contract.ROOT/"reference").as_posix()
                add(f"inputs/{name}", reference)
        if "inputs/spec.json" not in files or f"inputs/{Path(pins['embedded_manifest']).name}" not in files:
            raise ValueError("v2_bundle_reference_files")
        for name in TOOLS:
            add(f"tools/{name}", tools/name)
        for name in ("v2_bundle.py", "GUIDE_KO.md"):
            add(name, tools/name)
        provenance = work/"git.json"
        provenance.write_bytes(fmt.canonical({"git_url": fmt.GIT_URL, "training_revision": fmt.TRAINER_REVISION,
            "frozen_base_revision": pins["revision"], "training_core_sha256": pins["training_core_sha256"],
            "source_zip_sha256": fmt.SOURCE_SHA, "final_receipt": fmt.FINAL_RECEIPT, "cache_boundary": fmt.FINAL_BOUNDARY}))
        add("provenance/git.json", provenance)
    return files, {"identity": final["identity"], "checkpoint_receipt_sha256": step["content_sha256"],
                   "cache_boundary_sha256": fmt.FINAL_BOUNDARY, "cache_shards": len(entries), "frame_pairs": len(data["records"]),
                   "cache_package_bytes": sum(e["package"]["bytes"] for e in entries), "selected_models": models,
                   "features": {"identity": fmt.FEATURE_IDENTITY, "groups": len(groups), "clips": 3518,
                                 "package_bytes": sum(g[2]["bytes"] for g in groups.values())}}


def view_for_metadata(files):
    def read(name, cap, item):
        path = files[name][0]
        with path.open("rb") as handle:
            raw = handle.read(cap+1)
        if not 0 < len(raw) <= cap or {"bytes": len(raw), "sha256": fmt.sha(raw)} != item:
            raise ValueError(f"v2_bundle_metadata_pin: {name}")
        return raw
    return SimpleNamespace(read=read)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("project", "source-zip", "source-receipt", "tools", "work"):
        parser.add_argument("--"+name, required=True, type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--seconds", type=int, default=28700)
    args = parser.parse_args(argv)
    if not 1 <= args.seconds <= 28700:
        parser.error("invalid single-invocation time cap")
    project, work = args.project.resolve(strict=True), args.work.resolve(strict=True)
    if work.is_relative_to(project) or project.is_relative_to(work) or shutil.disk_usage(work).free < 2*fmt.GIB:
        raise ValueError("v2_bundle_requires_separate_local_scratch")
    if args.output:
        output = args.output.absolute()
        if output.parent.resolve(strict=True) != output.parent or any(
                output.is_relative_to(project/name) for name in (fmt.A, fmt.BULK, fmt.LEGACY)):
            raise ValueError("v2_bundle_export_directory")
    else:
        export = project/"reports"/"exports"
        if export.is_symlink() or export.parent.resolve(strict=True) != project/"reports":
            raise ValueError("v2_bundle_export_directory")
        export.mkdir(exist_ok=True)
        output = export/OUTPUT
    clock = Deadline(args.seconds)
    files, metadata = collect(project, args.source_zip, args.source_receipt, args.tools, work, clock)
    report = write_bundle(output, files, metadata, clock)
    report["status"] = "verified"
    (work/"verified-export.json").write_bytes(fmt.canonical(report))
    print(fmt.canonical(report).decode(), flush=True)


if __name__ == "__main__":
    main()
