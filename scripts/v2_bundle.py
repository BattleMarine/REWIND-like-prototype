"""Standalone launcher shipped in v2_128_A_final_model_and_cache.zip."""

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import zipfile


SOURCE = "source/v2_gpu_training_128_v1.zip"
SOURCE_SHA = "9a68d8e45bcf3abf6fab3004740faa6805d4ce3fda5118c3a7102011937849d8"
SOURCE_SIZE = 24_477_377
TOOLS = ("v2_gpu128_a_bundle_format.py", "v2_gpu128_a_bundle_local.py",
         "v2_gpu128_a_bundle_predict.py", "v2_gpu128_a_bundle_tune.py",
         "v2_gpu128_a_bundle_train_full.py", "v2_gpu128_a_bundle_edit.py")
COMMAND = "import runpy,sys; sys.path[:0]=sys.argv[1:3]; del sys.argv[1:3]; runpy.run_path(sys.argv.pop(1),run_name='__main__')"


def sha(data):
    return hashlib.sha256(data).hexdigest()


def run(command, bundle, args, editable=None, output=None):
    if bundle.is_symlink():
        raise ValueError("bundle path may not be a link")
    if shutil.disk_usage(Path(tempfile.gettempdir())).free < 256*1024**2:
        raise OSError("The launcher requires 256 MiB of temporary local space")
    with tempfile.TemporaryDirectory(prefix="rewind-128-launch-") as directory:
        scratch = Path(directory).resolve()
        archive = None if bundle.is_dir() else zipfile.ZipFile(bundle)
        try:
            def read(name, cap):
                if archive:
                    matches = [entry for entry in archive.infolist() if entry.filename == name]
                    if len(matches) != 1 or matches[0].file_size > cap:
                        raise ValueError(f"Invalid bundle member: {name}")
                    source = archive.open(matches[0])
                else:
                    path = bundle.joinpath(*name.split("/"))
                    if not path.is_file() or path.resolve() != path or path.stat().st_size > cap:
                        raise ValueError(f"Invalid bundle member: {name}")
                    source = path.open("rb")
                with source:
                    raw = source.read(cap+1)
                if not 0 < len(raw) <= cap:
                    raise ValueError(f"Invalid bundle size: {name}")
                return raw

            marker = json.loads(read("BUNDLE_INDEX.json", 2*1024**2))
            stored = marker.pop("content_sha256")
            if (marker["schema"] != "rewind_v2_gpu128_a_portable_v1" or
                    sha(json.dumps(marker, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()) != stored):
                raise ValueError("Invalid bundle index")
            members = marker["members"]
            tools = scratch/"tools"
            tools.mkdir()
            for name in (() if command == "fork-source" else TOOLS):
                key = "tools/"+name
                raw = read(key, 2*1024**2)
                if members[key] != {"bytes": len(raw), "sha256": sha(raw)}:
                    raise ValueError(f"Tool SHA mismatch: {name}")
                (tools/name).write_bytes(raw)
            raw = read(SOURCE, SOURCE_SIZE)
            if len(raw) != SOURCE_SIZE or sha(raw) != SOURCE_SHA:
                raise ValueError("Frozen training source SHA mismatch")
            source_zip = scratch/"source.zip"
            source_zip.write_bytes(raw)
        finally:
            if archive is not None:
                archive.close()
        source = scratch/"source" if command != "fork-source" else output
        if command == "fork-source" and (source.exists() or source.is_symlink() or not source.parent.is_dir()
                                           or source.parent.resolve() != source.parent
                                           or bundle.is_dir() and source.resolve().is_relative_to(bundle)):
            raise ValueError("--output must be a new source directory outside the bundle")
        # The complete bytes of this known source ZIP were authenticated before extraction.
        with zipfile.ZipFile(source_zip) as original:
            original.extractall(source)
        if command == "fork-source":
            print(source, flush=True)
            return
        if editable is not None:
            spec = importlib.util.spec_from_file_location("portable_source_edit", tools/"v2_gpu128_a_bundle_edit.py")
            editor = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(editor)
            snapshot = editor.apply_snapshot(source, editable, SOURCE_SHA)
            proof = scratch/"source-identity.json"
            proof.write_bytes(editor.canonical(snapshot))
            args = [*args, "--source-identity-file", str(proof), "--editable-origin", str(editable)]
        tool = tools/f"v2_gpu128_a_bundle_{command.replace('-', '_')}.py"
        subprocess.run([sys.executable, "-u", "-c", COMMAND, str(source), str(tools), str(tool),
                        "--bundle", str(bundle), *args], check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("predict", "tune", "train-full", "fork-source"))
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--editable-source", type=Path)
    args, extra = parser.parse_known_args()
    if any(arg.split("=", 1)[0] in ("--source-identity-file", "--editable-origin") for arg in extra):
        parser.error("use --editable-source to supply model edits")
    if args.editable_source is not None and args.command != "train-full":
        parser.error("--editable-source is supported only by train-full")
    output = None
    if args.command == "fork-source":
        fork = argparse.ArgumentParser(prog="v2_bundle.py fork-source")
        fork.add_argument("--output", required=True, type=Path)
        output = fork.parse_args(extra).output.absolute()
    run(args.command, args.bundle.absolute(), extra,
        editable=args.editable_source.absolute() if args.editable_source is not None else None, output=output)


if __name__ == "__main__":
    main()
