"""Launch a v2 128 B/C bundle using the common cache and source from the final A ZIP."""

import argparse
from contextlib import contextmanager
import hashlib
import importlib.util
import json
from pathlib import Path, PurePosixPath
import re
import shutil
import subprocess
import sys
import tempfile
import zipfile


SCHEMA = "rewind_v2_gpu128_variant_v1"
SEEDS = {"B": 71, "C": 103}
MIB = 1024**2
BASE_INDEX_SHA256 = "4a4e1c119e08f0026ef4af9a87c8edbb757934fbcee7ace1b61c2e424e080f40"
SOURCE = "source/v2_gpu_training_128_v1.zip"
SOURCE_SHA256 = "9a68d8e45bcf3abf6fab3004740faa6805d4ce3fda5118c3a7102011937849d8"
RUNTIME = "tools/v2_variant_runtime.py"
BASE_TOOLS = ("v2_gpu128_a_bundle_format.py", "v2_gpu128_a_bundle_local.py", "v2_gpu128_a_bundle_predict.py",
              "v2_gpu128_a_bundle_tune.py", "v2_gpu128_a_bundle_train_full.py", "v2_gpu128_a_bundle_edit.py")
REQUIRED = frozenset({"checkpoint.zip", "comparison.zip", "comparison.json", "diagnostic.json", RUNTIME,
                      *("tools/"+name for name in BASE_TOOLS)})
CHILD = "import runpy,sys; sys.path[:0]=sys.argv[1:3]; del sys.argv[1:3]; runpy.run_path(sys.argv.pop(1),run_name='__main__')"


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def sha(value):
    return hashlib.sha256(value).hexdigest()


@contextmanager
def reader(path):
    if path.is_symlink():
        raise ValueError("variant_bundle_symlink")
    path = path.resolve(strict=True)
    archive = None if path.is_dir() else zipfile.ZipFile(path)
    try:
        members = {}
        if archive is not None:
            for info in archive.infolist():
                if info.is_dir():
                    continue
                if info.filename in members:
                    raise ValueError(f"variant_duplicate_member: {info.filename}")
                members[info.filename] = info

        def read(name, cap, pin=None):
            part = PurePosixPath(name)
            if not part.parts or part.is_absolute() or ".." in part.parts or part.as_posix() != name or "\\" in name or ":" in name:
                raise ValueError(f"variant_invalid_member_name: {name}")
            if name in members:
                info = members[name]
                size = info.file_size
                stream = lambda: archive.open(info)
            elif archive is None:
                item = path.joinpath(*part.parts)
                if item.is_symlink() or not item.is_file() or item.resolve() != item:
                    raise ValueError(f"variant_missing_member: {name}")
                size = item.stat().st_size
                stream = lambda: item.open("rb")
            else:
                raise ValueError(f"variant_missing_member: {name}")
            if not 0 < size <= cap or pin and size != pin["bytes"]:
                raise ValueError(f"variant_member_size: {name}")
            with stream() as handle:
                raw = handle.read(size+1)
            if len(raw) != size or pin and sha(raw) != pin["sha256"]:
                raise ValueError(f"variant_member_sha256: {name}")
            return raw

        yield read
    finally:
        if archive is not None:
            archive.close()


def variant_manifest(raw):
    data = json.loads(raw)
    unsigned = {k: v for k, v in data.items() if k != "content_sha256"}
    if (data.get("schema") != SCHEMA or data.get("slot") not in SEEDS
            or type(data.get("seed")) is not int or data.get("seed") != SEEDS[data["slot"]]
            or data.get("content_sha256") != sha(canonical(unsigned))):
        raise ValueError("variant_manifest_identity")
    members = data["members"]
    if not REQUIRED <= members.keys() or "variant.json" in members or len(members) > 64:
        raise ValueError("variant_manifest_members")
    for name, pin in members.items():
        if (set(pin) != {"bytes", "sha256"} or type(pin["bytes"]) is not int or not 0 < pin["bytes"] <= 1024*MIB
                or not re.fullmatch("[0-9a-f]{64}", pin["sha256"])):
            raise ValueError(f"variant_manifest_pin: {name}")
    return data


def run(command, variant, base, args, *, editable=None, output=None):
    if shutil.disk_usage(Path(tempfile.gettempdir())).free < 256*MIB:
        raise OSError("variant_launcher_requires_256MiB_temporary_space")
    if variant.resolve(strict=True) == base.resolve(strict=True):
        raise ValueError("variant_and_base_must_be_distinct")
    with tempfile.TemporaryDirectory(prefix="rewind-128-variant-") as directory:
        scratch = Path(directory).resolve()
        tools = scratch/"tools"
        tools.mkdir()
        with reader(base) as read:
            raw_index = read("BUNDLE_INDEX.json", 2*MIB)
            if sha(raw_index) != BASE_INDEX_SHA256:
                raise ValueError("variant_base_A_index_sha256")
            members = json.loads(raw_index)["members"]
            source_zip = scratch/"source.zip"
            source_zip.write_bytes(read(SOURCE, 24_477_377, {"bytes": 24_477_377, "sha256": SOURCE_SHA256}))
        with reader(variant) as read:
            config = variant_manifest(read("variant.json", 128*1024))
            if command != "fork-source":
                (tools/Path(RUNTIME).name).write_bytes(read(RUNTIME, 2*MIB, config["members"][RUNTIME]))
                for name in BASE_TOOLS:
                    key = "tools/"+name
                    if config["members"][key] != members[key]:
                        raise ValueError(f"variant_shared_tool_pin: {key}")
                    (tools/name).write_bytes(read(key, 2*MIB, members[key]))
        source = scratch/"source" if output is None else output
        if output is not None and (source.exists() or source.is_symlink() or not source.parent.is_dir()
                                   or source.parent.resolve() != source.parent
                                   or any(p.is_dir() and source.resolve().is_relative_to(p.resolve()) for p in (base, variant))):
            raise ValueError("variant_fork_source_requires_new_directory_outside_bundles")
        with zipfile.ZipFile(source_zip) as archive:
            archive.extractall(source)  # The exact original source ZIP was authenticated above.
        if command == "fork-source":
            print(source, flush=True)
            return
        if editable is not None:
            spec = importlib.util.spec_from_file_location("portable_source_edit", tools/"v2_gpu128_a_bundle_edit.py")
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            proof = scratch/"editable.json"
            proof.write_bytes(module.canonical(module.apply_snapshot(source, editable, SOURCE_SHA256)))
            args = [*args, "--source-identity-file", str(proof), "--editable-origin", str(editable)]
        if command in ("tune", "train-full") and not any(arg.split("=", 1)[0] == "--seed" for arg in args):
            args = [*args, "--seed", str(config["seed"])]
        subprocess.run([sys.executable, "-u", "-c", CHILD, str(source), str(tools), str(tools/Path(RUNTIME).name),
                        command, "--base", str(base), "--bundle", str(variant), *args], check=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("command", choices=("predict", "tune", "train-full", "fork-source"))
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--editable-source", type=Path)
    parsed, args = parser.parse_known_args(argv)
    if any(arg.split("=", 1)[0] in ("--source-identity-file", "--editable-origin") for arg in args):
        parser.error("use --editable-source")
    if parsed.editable_source is not None and parsed.command != "train-full":
        parser.error("--editable-source is supported only by train-full")
    output = None
    if parsed.command == "fork-source":
        fork = argparse.ArgumentParser(prog="v2_variant.py fork-source")
        fork.add_argument("--output", type=Path, required=True)
        output = fork.parse_args(args).output.absolute()
    run(parsed.command, parsed.bundle.absolute(), parsed.base.absolute(), args, output=output,
        editable=parsed.editable_source.absolute() if parsed.editable_source is not None else None)


if __name__ == "__main__":
    main()
