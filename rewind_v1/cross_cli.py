"""One explicit bounded cross-scenario phase per invocation, with no training operation."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
import time

import torch

from rewind_archive.contracts import canonical,digest
from rewind_prepare.extracted import _local
from .cross_data import (LIMITS,MODEL_SHA,POLICY,select,prepare,source_identity,write_report)
from .cross_inference import infer
from . import cross_recovery as recovery


def _linux_available_ram():
    # SC_AVPHYS_PAGES reports free pages, not Linux's reclaimable-aware MemAvailable.
    try:
        with Path("/proc/meminfo").open("r",encoding="ascii") as stream:
            text=stream.read(64*1024+1)
    except (OSError,UnicodeError) as error:
        raise RuntimeError("cross_available_ram_unavailable") from error
    if len(text)>64*1024:raise RuntimeError("cross_available_ram_invalid")
    rows=[line.split() for line in text.splitlines() if line.split()[:1]==["MemAvailable:"]]
    if not rows:raise RuntimeError("cross_available_ram_unavailable")
    if (len(rows)!=1 or len(rows[0])!=3 or rows[0][2]!="kB"
            or not rows[0][1].isascii() or not rows[0][1].isdecimal() or len(rows[0][1])>20):
        raise RuntimeError("cross_available_ram_invalid")
    return int(rows[0][1])*1024


def preflight(path):
    path=_local(path,directory=True)
    free=shutil.disk_usage(path).free
    if free<LIMITS["initial_free_disk_bytes"]:raise ValueError("cross_initial_free_disk_8GiB")
    available=_linux_available_ram()
    if available<LIMITS["initial_available_ram_bytes"]:raise MemoryError("cross_initial_available_ram_4GiB")
    return {"free_disk_bytes":free,"available_ram_bytes":available,
            "ram_measurement":"Linux /proc/meminfo MemAvailable",
            "ram_scope":"OS-visible availability; effective container/cgroup headroom not measured"}


def model_copy(source,output,status):
    source,output=_local(source),Path(output).absolute()
    if source.stat().st_size!=POLICY["model_bytes"]:raise ValueError("cross_model_zip_size")
    deadline=time.monotonic()+LIMITS["model_copy_seconds"]
    from scripts.v11_recovery import _measure
    if output.exists():
        with _local(output).open("rb") as stream:observed=_measure(stream,POLICY["model_bytes"],deadline)
        mode="reuse_verified_local"
    else:
        temporary=output.with_name(output.name+".partial")
        try:
            with source.open("rb") as incoming,temporary.open("xb") as target:
                observed=_measure(incoming,POLICY["model_bytes"],deadline,target)
                target.flush();os.fsync(target.fileno())
            if observed!=(POLICY["model_bytes"],MODEL_SHA):raise ValueError("cross_model_sha")
            temporary.rename(output);mode="copy_once_hash_on_read"
        finally:temporary.unlink(missing_ok=True)
    if observed!=(POLICY["model_bytes"],MODEL_SHA):raise ValueError("cross_model_sha")
    result={"state":"verified","sha256":MODEL_SHA,"bytes":observed[0],"mode":mode,
            "source":str(source),"output":str(output),"elapsed_seconds":LIMITS["model_copy_seconds"]-(deadline-time.monotonic())}
    write_report(status,result);return result


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    commands=parser.add_subparsers(dest="command",required=True)
    commands.add_parser("source")
    check=commands.add_parser("preflight");check.add_argument("--work",type=Path,required=True)
    model=commands.add_parser("model-copy")
    for name in ("source","output","status"):model.add_argument("--"+name,type=Path,required=True)
    for name in ("select","prepare","infer"):
        command=commands.add_parser(name)
        for option in ("inputs","output"):command.add_argument("--"+option,type=Path,required=True)
        command.add_argument("--source-sha256",required=True)
        if name=="select":
            command.add_argument("--rgb-root",type=Path,required=True);command.add_argument("--pose-root",type=Path,required=True)
        else:
            command.add_argument("--selection",type=Path,required=True);command.add_argument("--selection-sha256",required=True)
        if name=="infer":
            command.add_argument("--prepared",type=Path,required=True);command.add_argument("--prepared-sha256",required=True)
            command.add_argument("--model",type=Path,required=True);command.add_argument("--device",choices=("cpu","cuda"),required=True)
    pack=commands.add_parser("package")
    pack.add_argument("--run",type=Path,required=True);pack.add_argument("--output",type=Path,required=True)
    backup=commands.add_parser("backup-prepare")
    backup.add_argument("--drive-root",type=Path,required=True);backup.add_argument("--run-id",required=True);backup.add_argument("--phase",required=True)
    publish=commands.add_parser("publish")
    publish.add_argument("--archive",type=Path,required=True);publish.add_argument("--sha256",required=True)
    publish.add_argument("--bytes",type=int,required=True);publish.add_argument("--attempt",type=Path,required=True)
    args=parser.parse_args(argv);torch.set_num_threads(1)
    if args.command in ("select","prepare","infer") and args.source_sha256!=digest(source_identity()):
        raise ValueError("cross_source_pin")
    if args.command=="source":result={"source_sha256":digest(source_identity())}
    elif args.command=="preflight":result=preflight(args.work)
    elif args.command=="model-copy":result=model_copy(args.source,args.output,args.status)
    elif args.command=="select":result=select(args.inputs,args.rgb_root,args.pose_root,args.output)
    elif args.command=="prepare":result=prepare(args.inputs,args.selection,args.selection_sha256,args.output)
    elif args.command=="infer":
        device=torch.device(args.device)
        if device.type=="cuda":
            if not torch.cuda.is_available():raise RuntimeError("cross_requested_cuda_unavailable_no_fallback")
            torch.cuda.reset_peak_memory_stats(device)
        result=infer(args.inputs,args.selection,args.selection_sha256,args.prepared,args.prepared_sha256,args.model,args.output,device)
    elif args.command=="package":result=recovery.package(args.run,args.output)
    elif args.command=="backup-prepare":result={"attempt":str(recovery.prepare_attempt(args.drive_root,args.run_id,args.phase))}
    else:
        if not 0<args.bytes<=LIMITS["result_bytes"]:raise ValueError("cross_archive_limit")
        result=recovery.publish({"archive":str(args.archive),"bytes":args.bytes,"sha256":args.sha256},args.attempt)
    if args.command in ("select","prepare","infer"):
        result={k:result[k] for k in ("status","content_sha256")}
    print(canonical(result).decode())


if __name__=="__main__":main()
