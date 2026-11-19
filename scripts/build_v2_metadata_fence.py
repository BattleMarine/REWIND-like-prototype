"""Build the fixed, default-off changes-only Colab notebook."""

import argparse
from pathlib import Path
import tempfile

from rewind_archive.contracts import canonical
from scripts.archvis_pilot_bootstrap import extract_pinned_zip
from scripts.build_archvis_pilot_handoff import _cell
from scripts.check_v2_metadata_fence import ZIP_SHA
from scripts.resume_v2_drive_metadata import sha_file


def notebook(pins):
    settings = '''RUN_CHANGES_AND_COMPARE = False
VERIFY_ONLY = False
SAVE_RECOVERY_TO_DRIVE = False
DRIVE_RECOVERY_DIRECTORY = ""
'''
    boot = Path(__file__).with_name("archvis_pilot_bootstrap.py").read_text(encoding="utf-8")
    boot += f"\nPINS={pins!r}\n"
    boot += '''
import datetime, shutil, signal, subprocess, sys, uuid

if RUN_CHANGES_AND_COMPARE and VERIFY_ONLY:
    raise ValueError("Choose one operation")
if RUN_CHANGES_AND_COMPARE or VERIFY_ONLY:
    incoming = Path("/content")
    base = incoming/"v2_metadata_changes_20261119"
    base.mkdir(exist_ok=True)
    source = base/("source-"+uuid.uuid4().hex)
    extract_pinned_zip(incoming/PINS["source_name"],PINS["source_sha256"],source,max_zip_bytes=64*1024**2,
        max_output_bytes=192*1024**2,max_members=3000,revision=PINS["revision"])
    archive = incoming/"results.zip"
    common = ["--archive",str(archive),"--reference",str(source/"reference")]
    call = [sys.executable,"-m","scripts.check_v2_metadata_fence"]
    subprocess.run(call+["verify",*common,"--work",str(base/("verify-"+uuid.uuid4().hex))],cwd=source,check=True,timeout=120)
    if RUN_CHANGES_AND_COMPARE:
        from google.colab import auth, files
        auth.authenticate_user()
        destination = None
        if SAVE_RECOVERY_TO_DRIVE:
            from google.colab import drive
            mount = Path("/content/drive")
            drive.mount(str(mount))
            requested = Path(DRIVE_RECOVERY_DIRECTORY)
            destination = requested.resolve()
            if (not DRIVE_RECOVERY_DIRECTORY or not mount.is_mount() or destination == mount.resolve()
                or not destination.is_relative_to(mount.resolve()) or requested.is_symlink()
                or not destination.is_dir() or not os.access(destination,os.W_OK)
                or shutil.disk_usage(destination).free < 8*1024**2):
                raise ValueError("Existing writable Drive recovery directory required")
        attempt = base/("attempt-"+uuid.uuid4().hex)
        attempt.mkdir()
        run = attempt/"results"
        stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        archive_out = base/("v2_metadata_changes_"+stamp+"_"+uuid.uuid4().hex[:12]+".zip")
        rc = None
        try:
            with (attempt/"worker.stdout.txt").open("wb") as stdout,(attempt/"worker.stderr.txt").open("wb") as stderr:
                with subprocess.Popen(call+["check",*common,"--work",str(run)],cwd=source,stdout=stdout,stderr=stderr) as child:
                    try:
                        rc = child.wait(timeout=900)
                    except BaseException:
                        if child.poll() is None:
                            child.send_signal(signal.SIGINT)
                        try:
                            child.wait(timeout=20)
                        except subprocess.TimeoutExpired:
                            child.kill()
                            child.wait(timeout=10)
                        raise
        finally:
            stage = archive_out.with_suffix(".partial")
            with zipfile.ZipFile(stage,"x",compression=zipfile.ZIP_STORED) as result:
                for name in ("worker.stdout.txt","worker.stderr.txt"):
                    result.write(attempt/name,name)
                for name in ("report.json","issues.jsonl"):
                    if (run/name).is_file():
                        result.write(run/name,name)
            stage.rename(archive_out)
            print("Local recovery ZIP:", archive_out, "worker exit:", rc)
            if destination is not None:
                try:
                    saved=subprocess.run([sys.executable,"-m","scripts.resume_v2_drive_metadata","save-recovery",
                        str(archive_out),str(destination),"180"],cwd=source,check=True,timeout=180,capture_output=True,text=True)
                    print("Verified Drive recovery:",saved.stdout.strip())
                except Exception as error:
                    print("Drive copy failed. Local ZIP kept.",type(error).__name__)
            files.download(str(archive_out))
'''
    return {"nbformat": 4, "nbformat_minor": 5, "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3"}},
            "cells": [_cell("markdown", "guide", "# v2 두 번째 DB 변경 기록 확인과 로컬 대조\n\n고정 source ZIP 및 04:18:37 UTC 원본 results.zip을 /content에 준비합니다. 조회는 기존 토큰의 changes.list만 사용합니다. 지정 폴더 제거 한 건을 확인한 뒤 좌우 RGB와 pose를 로컬에서 대조합니다.\n"),
                      _cell("code", "settings", settings), _cell("code", "check", boot)]}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--source-sha", required=True)
    p.add_argument("--revision", required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    with tempfile.TemporaryDirectory(dir=args.output.parent) as tmp:
        src = Path(tmp)/"source"
        extract_pinned_zip(args.source, args.source_sha, src, max_zip_bytes=64*1024**2, max_output_bytes=192*1024**2,
                           max_members=3000, revision=args.revision)
        for name in ("scripts/check_v2_metadata_fence.py", "scripts/resume_v2_drive_metadata.py", "reference/spec.json",
                     "reference/split/train.txt", "reference/split/validation.txt", "reference/split/test.txt"):
            if not (src/name).is_file():
                raise ValueError("v2_fence_source_missing:" + name)
    pins = {"source_name": args.source.name, "source_sha256": args.source_sha, "revision": args.revision, "parent_zip_sha256": ZIP_SHA}
    args.output.mkdir()
    ipynb = args.output/"v2_metadata_fence.ipynb"
    ipynb.write_bytes(canonical(notebook(pins)))
    (args.output/"receipt.json").write_bytes(canonical({"pins": pins, "notebook_sha256": sha_file(ipynb), "default_off": True}))


if __name__ == "__main__":
    main()
