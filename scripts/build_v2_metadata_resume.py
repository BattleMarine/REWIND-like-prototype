"""Build a source-pinned, default-off Colab notebook for the fixed 11/19 recovery."""

import argparse
import json
from pathlib import Path
import tempfile

from rewind_archive.contracts import canonical
from scripts.archvis_pilot_bootstrap import extract_pinned_zip
from scripts.build_archvis_pilot_handoff import _cell
from scripts.resume_v2_drive_metadata import ZIP_SHA, sha_file


def notebook(pins):
    settings = '''RUN_RESUME = False
VERIFY_ONLY = False
# Use only the upper bound and its evidence confirmed with the test session.
PRIOR_WALL_UPPER_SECONDS = None
PRIOR_WALL_BASIS = ""
SAVE_RECOVERY_TO_DRIVE = False
DRIVE_RECOVERY_DIRECTORY = ""
'''
    bootstrap = Path(__file__).with_name("archvis_pilot_bootstrap.py").read_text(encoding="utf-8")
    bootstrap += f"\nPINS={pins!r}\n"
    bootstrap += '''
import datetime, hashlib, json, math, shutil, signal, subprocess, sys, time, uuid, zipfile
from google.colab import auth, files

if RUN_RESUME and VERIFY_ONLY:
    raise ValueError("Choose one operation")
if RUN_RESUME or VERIFY_ONLY:
    started = time.monotonic()
    incoming = Path("/content")
    source_zip = incoming/PINS["source_name"]
    original = incoming/"results.zip"
    work = incoming/"v2_metadata_resume_20261119"
    work.mkdir(exist_ok=True)
    source = work/("source-"+uuid.uuid4().hex)
    extract_pinned_zip(source_zip,PINS["source_sha256"],source,max_zip_bytes=64*1024**2,
        max_output_bytes=192*1024**2,max_members=3000,revision=PINS["revision"])
    ref = source/"reference"
    spec = ref/"spec.json"
    spec_sha = hashlib.sha256(spec.read_bytes()).hexdigest()
    if RUN_RESUME:
        prior=PRIOR_WALL_UPPER_SECONDS
        basis=PRIOR_WALL_BASIS
        if type(prior) is not int or not 1<=prior<14100 or not isinstance(basis,str) or not basis.strip() or len(basis)>4096:
            raise ValueError("Pinned previous elapsed upper bound required before authentication")
        evidence={"schema":"v2_metadata_prior_wall_upper_v1","result_zip_sha256":PINS["parent_zip_sha256"],
            "prior_wall_upper_seconds":prior,"basis":basis}
        raw=json.dumps(evidence,sort_keys=True,ensure_ascii=False,separators=(",",":")).encode("utf-8")
        elapsed_sha=hashlib.sha256(raw).hexdigest()
    # A verified local copy is made before requesting authorization.
    subprocess.run([sys.executable,"-m","scripts.resume_v2_drive_metadata","verify",
        "--archive",str(original),"--work",str(work/("verify-"+uuid.uuid4().hex)),
        "--spec",str(spec),"--spec-sha",spec_sha,"--splits",str(ref/"split")],cwd=source,timeout=120,check=True)
    if RUN_RESUME:
        run = work/"resume-once"
        if run.exists():
            raise ValueError("Existing attempt present. Recover its files and use the new receipt to request a new pinned handoff.")
        auth.authenticate_user()
        drive_destination = None
        if SAVE_RECOVERY_TO_DRIVE:
            from google.colab import drive
            mount = Path("/content/drive")
            drive.mount(str(mount))
            requested = Path(DRIVE_RECOVERY_DIRECTORY)
            drive_destination = requested.resolve()
            if (not DRIVE_RECOVERY_DIRECTORY or not mount.is_mount() or not drive_destination.is_relative_to(mount.resolve())
                or drive_destination == mount.resolve() or requested.is_symlink() or not drive_destination.is_dir()
                or not os.access(drive_destination,os.W_OK) or shutil.disk_usage(drive_destination).free < 67805184):
                raise ValueError("Existing Drive recovery directory must have room for at least the original DB")
        overhead = math.ceil(time.monotonic()-started)
        timeout = 14400-prior-overhead-300
        if timeout <= 0:
            raise TimeoutError("Remaining wall budget exhausted")
        logs = work/("attempt-"+uuid.uuid4().hex)
        logs.mkdir()
        elapsed = logs/"resume_elapsed_evidence.json"
        elapsed.write_bytes(raw)
        stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        result = work/("metadata_resume_"+stamp+"_"+uuid.uuid4().hex[:12]+".zip")
        args = [sys.executable,"-m","scripts.resume_v2_drive_metadata","resume","--archive",str(original),
            "--work",str(run),"--spec",str(spec),"--spec-sha",spec_sha,"--splits",str(ref/"split"),
            "--elapsed-evidence",str(elapsed),"--elapsed-evidence-sha",elapsed_sha,
            "--invocation-overhead-seconds",str(overhead)]
        rc = None
        try:
            with (logs/"listing.stdout.txt").open("wb") as stdout,(logs/"listing.stderr.txt").open("wb") as stderr:
                with subprocess.Popen(args,cwd=source,stdout=stdout,stderr=stderr) as child:
                    try:
                        rc = child.wait(timeout=timeout)
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
            (logs/"notebook_receipt.json").write_text(json.dumps({"worker_exit_code":rc,
                "elapsed_this_notebook_seconds":time.monotonic()-started,"parent_zip_sha256":PINS["parent_zip_sha256"],
                "elapsed_evidence_sha256":elapsed_sha}),encoding="utf-8")
            temporary=result.with_suffix(".partial")
            with zipfile.ZipFile(temporary,"x",compression=zipfile.ZIP_STORED) as bundle:
                for path in sorted(logs.iterdir()):
                    bundle.write(path,path.name)
                if run.exists():
                    for path in sorted(run.rglob("*")):
                        if path.is_file():
                            bundle.write(path,path.relative_to(run).as_posix())
            temporary.rename(result)
            print("Local recovery ZIP:",result)
            if drive_destination is not None:
                try:
                    remaining=min(300,int(14400-prior-(time.monotonic()-started)))
                    if remaining < 1:
                        raise TimeoutError("Recovery reservation exhausted")
                    saved=subprocess.run([sys.executable,"-m","scripts.resume_v2_drive_metadata","save-recovery",
                        str(result),str(drive_destination),str(remaining)],cwd=source,check=True,timeout=remaining,
                        capture_output=True,text=True)
                    result.with_suffix(".drive-receipt.json").write_text(saved.stdout,encoding="utf-8")
                    print("Verified Drive copy:",saved.stdout.strip())
                except Exception as error:
                    print("Drive recovery failed. Local ZIP kept.",type(error).__name__,str(error)[:400])
            files.download(str(result))
'''
    return {"nbformat": 4, "nbformat_minor": 5, "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3"}},
            "cells": [_cell("markdown", "guide", "# v2 메타데이터 11/19 부분 DB 재개\n\n고정 source ZIP과 원본 results.zip을 /content에 준비합니다. 기본 스위치는 둘 다 False입니다. VERIFY_ONLY는 로컬에서만 pin을 검사합니다. RUN_RESUME 전 테스트 세션에서 확인한 이전 실행시간의 보수적 상계와 근거를 설정 셀에 입력합니다. 종료 뒤 회수 ZIP을 다운로드합니다.\n"),
                      _cell("code", "settings", settings), _cell("code", "resume", bootstrap)]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--source-sha", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(dir=args.output.parent) as temporary:
        root = Path(temporary) / "source"
        extract_pinned_zip(args.source, args.source_sha, root, max_zip_bytes=64*1024**2,
                           max_output_bytes=192*1024**2, max_members=3000, revision=args.revision)
        for path in ("scripts/resume_v2_drive_metadata.py", "reference/spec.json", "reference/split/train.txt",
                     "reference/split/validation.txt", "reference/split/test.txt"):
            if not (root/path).is_file():
                raise ValueError("v2_resume_source_member_missing:" + path)
    pins = {"source_name": args.source.name, "source_sha256": args.source_sha, "revision": args.revision,
            "parent_zip_sha256": ZIP_SHA}
    args.output.mkdir()
    path = args.output/"v2_metadata_resume.ipynb"
    path.write_bytes(canonical(notebook(pins)))
    (args.output/"receipt.json").write_bytes(canonical({"pins": pins, "notebook_sha256": sha_file(path),
                                                       "status": "default_off_runtime_elapsed_evidence_required"}))


if __name__ == "__main__":
    main()
