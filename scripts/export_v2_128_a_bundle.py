"""Windows/POSIX launcher for the fixed v2 128/A archive exporter."""

import argparse
import hashlib
from pathlib import Path
import shutil
import subprocess
import sys
import time
import zipfile


SOURCE_SHA = "9a68d8e45bcf3abf6fab3004740faa6805d4ce3fda5118c3a7102011937849d8"
SOURCE_SIZE = 24_477_377
COMMAND = "import runpy,sys; sys.path[:0]=sys.argv[1:3]; del sys.argv[1:3]; runpy.run_path(sys.argv.pop(1),run_name='__main__')"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("project", "original-source", "work", "output"):
        parser.add_argument("--"+name, type=Path, required=name != "output")
    parser.add_argument("--seconds", type=int, default=28700)
    args = parser.parse_args(argv)
    started = time.monotonic()
    root, work = args.project.resolve(strict=True), args.work.absolute()
    upload = args.original_source.absolute()
    if (work.exists() or work.is_symlink() or not work.parent.is_dir() or work.resolve().is_relative_to(root)
            or not 1 <= args.seconds <= 28700 or shutil.disk_usage(work.parent).free < 2*1024**3):
        raise ValueError("Use a new local work directory outside the input root with 2 GiB free")
    if not upload.is_file() or upload.is_symlink() or upload.stat().st_size != SOURCE_SIZE:
        raise ValueError("The fixed 128 source ZIP is required")
    with upload.open("rb") as handle:
        if hashlib.file_digest(handle, "sha256").hexdigest() != SOURCE_SHA:
            raise ValueError("Fixed 128 source ZIP SHA mismatch")
    work.mkdir()
    source = work/"frozen-source"
    tools = Path(__file__).resolve().parent
    with zipfile.ZipFile(upload) as original:
        original.extractall(source)
    remaining = args.seconds-int(time.monotonic()-started)
    command = [sys.executable, "-u", "-c", COMMAND, str(source), str(tools), str(tools/"v2_gpu128_a_bundle_export.py"),
               "--project", str(root), "--source-zip", str(upload), "--source-receipt", str(tools/"source-receipt.json"),
               "--tools", str(tools), "--work", str(work), "--seconds", str(remaining)]
    if args.output is not None:
        command += ["--output", str(args.output.absolute())]
    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
