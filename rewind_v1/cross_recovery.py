"""Bounded reports/visualizations only; raw data, prepared tensors and model ZIP are excluded."""

import hashlib
import json
import os
from pathlib import Path
import tempfile
import time
import zipfile

from rewind_archive.contracts import canonical
from rewind_m1.local_rgb import _regular_read
from rewind_prepare.extracted import _local
from rewind_prepare.temporal_cache import _atomic
from .cross_data import MIB, RESULT_BYTES, RUN_ID, write_report
from scripts.v11_recovery import _measure


def result_files(run,*,extra_reports=()):
    run=_local(run,directory=True);files=[]
    for phase in ("selection","prepared","inference"):
        directory=run/phase
        if not directory.is_dir():continue
        files.extend(p for p in directory.iterdir() if p.name in ("selection.json","prepared.json","report.json","failure.json") and p.is_file())
        for number in range(4):
            clip=directory/f"clip-{number:02d}"
            if not clip.is_dir():continue
            if phase=="prepared":
                if (clip/"manifest.json").is_file():files.append(clip/"manifest.json")
            elif phase=="inference":
                files.extend(p for p in clip.iterdir() if p.is_file() and
                             (p.name in ("predictions.json","clip.gif") or (p.name.startswith("frame-") and p.suffix==".png")))
    files.extend(p for p in run.iterdir() if p.name in ("model_status.json","notebook_status.json") and p.is_file())
    for name in extra_reports:
        path=run/name
        if not path.is_relative_to(run) or ".." in Path(name).parts:raise ValueError("cross_extra_report_path")
        if path.is_file():files.append(path)
    for path in files:_local(path)
    if sum(p.stat().st_size for p in files)>RESULT_BYTES:raise ValueError("cross_result_file_sum")
    return files


def package(run,output,*,run_id=RUN_ID,extra_reports=()):
    run,output=_local(run,directory=True),Path(output).absolute()
    if output.exists() or output.is_symlink() or output.is_relative_to(run):raise ValueError("cross_recovery_new_output")
    files=result_files(run,extra_reports=extra_reports)
    if not files:raise ValueError("cross_recovery_no_reports")
    deadline=time.monotonic()+600
    pins={}
    for path in files:
        if time.monotonic()>deadline:raise TimeoutError("cross_package_timeout")
        size=path.stat().st_size
        with path.open("rb") as stream:pin=hashlib.file_digest(stream,"sha256").hexdigest()
        pins[path.relative_to(run).as_posix()]={"bytes":size,"sha256":pin}
    raw=canonical({"schema":"v12_cross_result_recovery_v1","run_id":run_id,"files":pins})+b"\n"
    if sum(p["bytes"] for p in pins.values())+len(raw)+16384>RESULT_BYTES:raise ValueError("cross_result_zip_upper")
    with tempfile.NamedTemporaryFile(dir=output.parent,prefix=".cross-package-",delete=False) as stream:temporary=Path(stream.name)
    try:
        with zipfile.ZipFile(temporary,"w") as archive:
            for path in files:
                if time.monotonic()>deadline:raise TimeoutError("cross_package_timeout")
                archive.write(path,path.relative_to(run).as_posix())
            archive.writestr("recovery_receipt.json",raw)
        if temporary.stat().st_size>RESULT_BYTES:raise ValueError("cross_result_zip_limit")
        temporary.rename(output)
    finally:temporary.unlink(missing_ok=True)
    with output.open("rb") as stream:pin=hashlib.file_digest(stream,"sha256").hexdigest()
    return {"archive":str(output),"bytes":output.stat().st_size,"sha256":pin}


def prepare_attempt(drive_root,run_id,phase,*,namespace=RUN_ID):
    if not run_id.startswith("run-") or not run_id[4:].isalnum() or phase not in ("selection","prepared","inference"):
        raise ValueError("cross_backup_identity")
    root=_local(drive_root,directory=True)
    for name in ("REWIND-like prototype","reports",namespace,run_id,phase):
        root=root/name;root.mkdir(exist_ok=True);_local(root,directory=True)
    attempt=Path(tempfile.mkdtemp(prefix="attempt-",dir=root))
    marker=canonical({"run_id":namespace,"local_run":run_id,"phase":phase})+b"\n"
    with (attempt/"preflight.json").open("xb") as stream:stream.write(marker);stream.flush();os.fsync(stream.fileno())
    if _regular_read(attempt/"preflight.json",MIB)!=marker:raise ValueError("cross_drive_preflight")
    return attempt


def publish(archive,attempt,*,run_id=RUN_ID):
    attempt=_local(attempt,directory=True);source=_local(archive["archive"])
    _regular_read(attempt/"preflight.json",MIB)
    deadline=time.monotonic()+600;partial=attempt/"results.zip.partial";final=attempt/"results.zip"
    try:
        with (attempt/"publish.lock").open("xb"):
            if final.exists() or final.is_symlink() or (attempt/"receipt.json").exists():raise FileExistsError("cross_attempt_used")
            with source.open("rb") as incoming,partial.open("xb") as target:
                if _measure(incoming,archive["bytes"],deadline,target)!=(archive["bytes"],archive["sha256"]):raise ValueError("cross_local_zip_hash")
                target.flush();os.fsync(target.fileno())
            with partial.open("rb") as stream:
                if _measure(stream,archive["bytes"],deadline)!=(archive["bytes"],archive["sha256"]):raise ValueError("cross_drive_zip_hash")
            partial.rename(final)
            receipt={"schema":"v12_cross_result_backup_v1","run_id":run_id,"state":"verified",**archive,
                     "drive_zip":str(final),"copy_readback_seconds":600-(deadline-time.monotonic())}
            write_report(attempt/"receipt.json",receipt);return receipt
    except BaseException as error:
        with __import__("contextlib").suppress(BaseException):
            write_report(attempt/"failure.json",{"state":"failed","local_archive":str(source),"type":type(error).__name__,"reason":str(error)[:512]})
        raise
