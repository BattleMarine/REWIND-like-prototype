"""Independent v2 256px CPU pilot and manually authorized continuation."""

import argparse
import io
import json
from pathlib import Path
import re
import shutil
import tempfile
import time

import numpy as np
import torch

from rewind_archive.contracts import canonical, digest
from rewind_m1.local_rgb import _regular_read
from rewind_prepare.temporal_cache import _atomic
from rewind_prepare.temporal_selection import check_pin, seal
from rewind_v1 import v2_contract as contract, v2_retention as retention
from rewind_v1.v2_budget import Ledger
from rewind_v1.v2_space import available_ram
from scripts.archvis_pilot_bootstrap import extract_pinned_zip
from scripts.run_v2_bulk_cache import rows_for, writer
from scripts.run_v2_cpu_cache import ARCHIVE_CAP, MANIFEST_CAP, MANIFEST_NAME, read_manifest
from scripts.run_v2_resolution_ab import part, write_variant
from scripts.v2_bulk_quota import CLOSURE, DriveQuota, TimeQuota, publish


MIB, GIB = 1024**2, 1024**3
PILOT_SECONDS, LONG_SECONDS = 1800, 32400
PILOT_SHARDS, TARGET_SHARDS = 3, 1063
PACKAGE_CAP = 224*MIB
PILOT_BYTES = GIB
TIME_CEILING, DRIVE_CEILING = 365*86400, 1024*GIB
STORE = "v2_cache256_041837Z"
REPORTS = "v2_cache256_reports_041837Z"
SCHEMA = "rewind_v2_cache256_boundary_v1"
RECEIPT_SCHEMA = "rewind_v2_cache256_shard_v1"
PROOF_SCHEMA = "rewind_v2_cache256_pilot_proof_v1"
AUTH_SCHEMA = "rewind_v2_cache256_authorization_v1"


def sealed_file(path, schema, expected=None):
    value = json.loads(_regular_read(path, 64*MIB if schema == RECEIPT_SCHEMA else 64*1024))
    check_pin(value, schema, expected if expected is not None else value.get("content_sha256"))
    return value


def package_file(root, index, package):
    sha = package["sha256"]
    if (not re.fullmatch("[0-9a-f]{64}", sha) or package["name"] != f"shard-{index:05d}-{sha}.zip"
            or not 0 < package["bytes"] <= PACKAGE_CAP or not re.fullmatch("[0-9a-f]{64}", package["receipt_sha256"])):
        raise ValueError("cache256_package_pin")
    path = root/package["name"]
    if path.is_symlink() or not path.is_file() or path.stat().st_size != package["bytes"]:
        raise ValueError("cache256_package_missing_or_size")
    return path


def entry(index, rows, previous, package, binding):
    return seal({"schema": SCHEMA, "index": index, "previous": previous, "package": package,
                 "binding": binding, "records_sha256": digest(rows), "frames": len(rows)})


def scan(root, doc, binding, check):
    result = []
    for index, path in enumerate(sorted(root.glob("boundary-*.json"))):
        check()
        rows = rows_for(doc, index)
        if not rows or path.name != f"boundary-{index:05d}.json":
            raise ValueError("cache256_journal_gap")
        item = sealed_file(path, SCHEMA)
        previous = result[-1]["content_sha256"] if index else None
        if item != entry(index, rows, previous, item["package"], binding):
            raise ValueError("cache256_journal_binding")
        package_file(root, index, item["package"])
        result.append(item)
    names = {e["package"]["name"] for e in result}
    if any(p.name not in names and not p.name.startswith(f"shard-{len(result):05d}-") for p in root.glob("shard-*.zip")):
        raise ValueError("cache256_unexpected_package")
    return result


def verify_content(folder, index, rows, spec, binding, ledger):
    receipt = sealed_file(folder/"receipt.json", RECEIPT_SCHEMA)
    if (receipt["index"] != index or receipt["binding"] != binding or receipt["records_sha256"] != digest(rows)
            or receipt["size"] != 256 or {p.name for p in folder.iterdir()} - {"shard.zip"}
            != {"images.npy", "targets.pt", "receipt.json"}):
        raise ValueError("cache256_shard_receipt")
    for name in ("images.npy", "targets.pt"):
        if part(folder/name, ledger) != receipt["files"][name]:
            raise ValueError("cache256_shard_part_pin")
    images = np.load(folder/"images.npy", mmap_mode="r", allow_pickle=False)
    try:
        if images.shape != (len(rows),2,3,256,256) or images.dtype != np.float32:
            raise ValueError("cache256_image_contract")
        for image in images:
            ledger.check()
            if not np.isfinite(image).all() or image.min() < 0 or image.max() > 1:
                raise ValueError("cache256_image_range")
    finally:
        images._mmap.close()
    targets = torch.load(io.BytesIO(_regular_read(folder/"targets.pt", 4*MIB)), map_location="cpu", weights_only=True)
    n, j = len(rows), len(spec["joint_names"])
    shapes = {"xy": (n,2,j,2), "xy_valid": (n,2,j), "target_mm": (n,j,3), "target_valid": (n,j),
              "camera": (n,2,9), "rotation64": (n,2,3,3), "translation64": (n,2,3)}
    if set(targets) != set(shapes):
        raise ValueError("cache256_target_keys")
    for key, shape in shapes.items():
        value = targets[key]
        dtype = torch.bool if key.endswith("valid") else torch.float64 if key.endswith("64") else torch.float32
        if value.shape != shape or value.dtype != dtype or not torch.isfinite(value).all():
            raise ValueError("cache256_target_contract")
    return receipt["content_sha256"]


def hydrate(path, package, local, index, rows, spec, binding, ledger):
    archive = local/"payload.zip"
    retention.copy_verified(path, archive, package, ledger, ledger.clock()+min(120, ledger.remaining()))
    extract_pinned_zip(archive, package["sha256"], local/"unpacked",
                       max_zip_bytes=PACKAGE_CAP, max_output_bytes=PACKAGE_CAP, max_members=3)
    return verify_content(local/"unpacked", index, rows, spec, binding, ledger)


def authorize(args, root, binding, proof, owner):
    if not 0 < args.additional_seconds <= TIME_CEILING-PILOT_SECONDS:
        raise ValueError("cache256_long_time_not_confirmed")
    if not 0 < args.additional_gib <= (DRIVE_CEILING-PILOT_BYTES)//GIB:
        raise ValueError("cache256_long_drive_not_confirmed")
    auth = seal({"schema": AUTH_SCHEMA, "binding": binding, "pilot": proof["content_sha256"],
                 "additional_seconds": args.additional_seconds, "additional_drive_bytes": args.additional_gib*GIB})
    path = root/"long-authorization.json"
    if path.exists() or path.is_symlink():
        if sealed_file(path, AUTH_SCHEMA) != auth:
            raise ValueError("cache256_existing_authorization_pin")
    else:
        publish(path, auth, owner)
    return proof["charged_seconds"]+args.additional_seconds, proof["drive_bytes"]+args.additional_gib*GIB


def append(root, work, doc, spec, binding, current, count, args, ledger, space, owner, report):
    while len(current) < count:
        if ledger.remaining() <= 180:
            return "time_budget"
        index = len(current)
        rows = rows_for(doc, index)
        candidates = list(root.glob(f"shard-{index:05d}-*.zip"))
        if len(candidates) > 1:
            raise ValueError("cache256_multiple_unpublished_packages")
        if not space.admit(0 if candidates else PACKAGE_CAP):
            return "drive_budget"
        read_cap = 2*PACKAGE_CAP if candidates else 3*PACKAGE_CAP+sum(
            p["size_bytes"] for row in rows for p in row["files"].values())
        write_cap = PACKAGE_CAP if candidates else 3*PACKAGE_CAP
        usage = ledger.current
        if any(usage[k+"_bytes_reserved"]+cap > ledger.cfg["resources"][k+"_bytes"]
               for k,cap in (("read",read_cap),("write",write_cap))):
            return "invocation_io_budget"
        if shutil.disk_usage(work).free < 4*PACKAGE_CAP:
            return "local_disk_budget"
        with tempfile.TemporaryDirectory(dir=work, prefix="shard-") as temp:
            temp = Path(temp)
            if candidates:
                path = candidates[0]
                package = {"name": path.name, "bytes": path.stat().st_size, "sha256": path.stem.rsplit("-",1)[-1],
                           "receipt_sha256": "0"*64}
                package["receipt_sha256"] = hydrate(package_file(root,index,package), package, temp, index, rows, spec, binding, ledger)
            else:
                reserve = ledger.cfg["terminal_reserve_seconds"]
                ledger.cfg["terminal_reserve_seconds"] = reserve+180
                try:
                    variant, _ = write_variant(rows, spec, None, temp/"shard", 256, ledger,
                                               source_roots=(args.rgb,args.pose),
                                               shard_identity={"schema": RECEIPT_SCHEMA, "binding": binding, "index": index})
                    receipt = verify_content(temp/"shard", index, rows, spec, binding, ledger)
                finally:
                    ledger.cfg["terminal_reserve_seconds"] = reserve
                package = {**variant["zip"], "receipt_sha256": receipt}
                package["name"] = f"shard-{index:05d}-{package['sha256']}.zip"
                if package["bytes"] > PACKAGE_CAP or not space.admit(package["bytes"]):
                    raise ValueError("cache256_package_or_drive_cap")
                with tempfile.TemporaryDirectory(dir=root, prefix=".incoming-") as staging:
                    draft = Path(staging)/package["name"]
                    retention.copy_verified(temp/"shard/shard.zip", draft, package, ledger,
                                            ledger.clock()+min(120, ledger.remaining()))
                    owner()
                    if list(root.glob(f"shard-{index:05d}-*.zip")):
                        raise ValueError("cache256_concurrent_package")
                    draft.rename(root/package["name"])
                ledger.reserve("read", package["bytes"])
                final = part(root/package["name"], ledger)
                ledger.observed("read", package["bytes"])
                if final != variant["zip"]:
                    raise ValueError("cache256_drive_readback")
            ledger.check()
            previous = current[-1]["content_sha256"] if current else None
            if index and sealed_file(root/f"boundary-{index-1:05d}.json", SCHEMA)["content_sha256"] != previous:
                raise ValueError("cache256_predecessor_changed")
            item = entry(index, rows, previous, package, binding)
            publish(root/f"boundary-{index:05d}.json", item, owner)
            current.append(item)
            space.committed(item, orphan=bool(candidates))
            report.update(shards=len(current), frames=sum(e["frames"] for e in current), boundary_sha256=item["content_sha256"])
            print("256 cache retained shard", index, item["content_sha256"], flush=True)
    return "target_reached"


def run(args, pins):
    report = {"status": "failed", "scope": "independent_256_cache", "mode": args.mode}
    ledger = None
    try:
        torch.set_num_threads(1)
        if available_ram() < 4*GIB or shutil.disk_usage(args.work).free < 2*GIB:
            raise OSError("cache256_local_resources")
        if args.mode != pins["mode"] or digest(contract.source()) != pins["consumer_source_identity"]:
            raise ValueError("cache256_source_or_mode_pin")
        for name, expected in pins["consumer_overlay"].items():
            if contract.sha(_regular_read(contract.ROOT/name, MIB)) != expected:
                raise ValueError("cache256_overlay_pin")
        source = contract.ROOT
        extract_pinned_zip(source/pins["embedded_manifest"], pins["archive_sha256"], args.work/"manifest",
                           max_zip_bytes=ARCHIVE_CAP, max_output_bytes=MANIFEST_CAP, max_members=1)
        doc, spec = read_manifest(args.work/"manifest"/MANIFEST_NAME, pins, source/"reference")
        if not rows_for(doc, TARGET_SHARDS-1) or rows_for(doc, TARGET_SHARDS):
            raise ValueError("cache256_full_shard_count")
        import PIL
        env = {"torch": str(torch.__version__), "numpy": str(np.__version__), "pillow": PIL.__version__}
        if torch.version.cuda is not None or env != pins["cpu_runtime"]:
            raise ValueError("cache256_cpu_runtime_pin")
        binding = digest({"schema":"rewind_v2_cache256_binding_v1", "size":256, "dtype":"float32",
                          "manifest": doc["content_sha256"], "spec": digest(spec), "source":pins["consumer_overlay"],
                          "environment":env, "resize":"bilinear_align_corners"})
        args.store.mkdir(exist_ok=True)
        with writer(args.store, args.recover_writer) as owner:
            time_quota = TimeQuota(args.store, owner, cap=TIME_CEILING, baseline=binding, drive_cap=DRIVE_CEILING,
                                   minimum=360, session_limit=LONG_SECONDS)
            items = scan(args.store, doc, binding, lambda: None)
            report.update(binding=binding, shards=len(items), frames=sum(e["frames"] for e in items),
                          boundary_sha256=items[-1]["content_sha256"] if items else None)
            pilot = args.mode == "pilot"
            history = time_quota.history(initialize=pilot and not items, required_tip=args.budget_tip, enforce_tip=True)
            report["budget_tip_sha256"] = history["tip"]
            if args.budget_tip and args.budget_tip != history["tip"]:
                raise ValueError("cache256_latest_tip_required")
            if pilot:
                if (args.store/"pilot-proof.json").exists() or (args.store/"long-authorization.json").exists():
                    raise ValueError("cache256_pilot_already_sealed")
                budget, drive_cap, target = PILOT_SECONDS, PILOT_BYTES, PILOT_SHARDS
            else:
                proof = sealed_file(args.store/"pilot-proof.json", PROOF_SCHEMA, args.pilot_receipt)
                if (len(items) < PILOT_SHARDS or proof["binding"] != binding
                        or proof["boundary_sha256"] != items[PILOT_SHARDS-1]["content_sha256"]
                        or proof["packages"] != [item["package"] for item in items[:PILOT_SHARDS]]):
                    raise ValueError("cache256_pilot_proof_binding")
                time_quota.history(required_tip=proof["budget_tip_sha256"], enforce_tip=True)
                budget, drive_cap = authorize(args, args.store, binding, proof, owner)
                target = TARGET_SHARDS
            space = DriveQuota(args.store, [], cap=drive_cap)
            booking = time_quota.reserve(history, time.monotonic()-args.started,
                                         session_cap=PILOT_SECONDS if pilot else LONG_SECONDS, budget_cap=budget)
            if booking is None:
                report.update(status="paused", stop_reason="cumulative_time_budget")
                return report
            report["reserved_seconds"] = booking["seconds"]
            report["budget_tip_sha256"] = booking["content_sha256"]
            verified = False
            try:
                _atomic(args.work/"quota-grant.json", canonical({"deadline":args.started+booking["seconds"]}))
                cfg = contract.config()
                quota = {**cfg, "seconds_per_invocation": booking["seconds"]-CLOSURE, "terminal_reserve_seconds": 35,
                         "external_timeout_seconds": booking["seconds"], "resources": {
                             **cfg["resources"], "read_bytes":(8 if pilot else 1024)*GIB, "write_bytes":(4 if pilot else 512)*GIB}}
                ledger = Ledger(args.work/"usage", {"binding":binding}, quota)
                with ledger.invocation("v2_cache256_"+args.mode):
                    ledger.started = args.started
                    reason = append(args.store,args.work,doc,spec,binding,items,target,args,ledger,space,owner,report)
                    report["stop_reason"] = reason
                    if pilot and reason == "target_reached":
                        reread = scan(args.store,doc,binding,ledger.check)
                        if reread != items or len(reread) != PILOT_SHARDS:
                            raise ValueError("cache256_pilot_reopen_boundary")
                        for index, e in enumerate(reread):
                            package = e["package"]
                            with tempfile.TemporaryDirectory(dir=args.work) as tmp:
                                sha = hydrate(package_file(args.store,index,package),package,Path(tmp),index,
                                              rows_for(doc,index),spec,binding,ledger)
                                if sha != package["receipt_sha256"]:
                                    raise ValueError("cache256_pilot_reopen_receipt")
                        verified = True
                if not pilot and reason == "target_reached" and report["frames"] != pins["frames"]:
                    raise ValueError("cache256_full_frame_count")
                report["status"] = "passed" if reason == "target_reached" else "paused"
            finally:
                try:
                    used = space.measure()
                    report["drive_bytes"] = used
                finally:
                    settle = time_quota.settle(booking,time.monotonic()-args.started)
                    charged = history["charged_seconds"]+settle["seconds"]
                    report.update(budget_tip_sha256=settle["content_sha256"], charged_seconds=charged,
                                  time_remaining_seconds=budget-charged, drive_cap_bytes=drive_cap)
            if verified:
                proof = seal({"schema":PROOF_SCHEMA,"binding":binding,"boundary_sha256":items[-1]["content_sha256"],
                              "packages":[e["package"] for e in items],"charged_seconds":charged,"drive_bytes":used,
                              "budget_tip_sha256":settle["content_sha256"]})
                publish(args.store/"pilot-proof.json",proof,owner)
                report["pilot_receipt_sha256"] = proof["content_sha256"]
    except BaseException as error:
        report["status"] = "stopped" if isinstance(error, KeyboardInterrupt) else "failed"
        report["error"] = str(error)[:300]
        raise
    finally:
        if ledger is not None and ledger.path.is_file():
            report["usage"] = ledger.snapshot()["invocations"][-1]
        _atomic(args.work/"result.json",canonical(report))
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("pilot","long"))
    for name in ("pins","work","store","rgb","pose"):
        parser.add_argument("--"+name,type=Path,required=True)
    parser.add_argument("--started",type=float,required=True)
    parser.add_argument("--budget-tip",default="")
    parser.add_argument("--recover-writer",default="")
    parser.add_argument("--pilot-receipt",default="")
    parser.add_argument("--additional-seconds",type=int,default=0)
    parser.add_argument("--additional-gib",type=int,default=0)
    args = parser.parse_args(argv)
    roots = (args.work,args.store,args.rgb,args.pose)
    if (args.store.name != STORE or args.store.parent.name != "cache" or any(p.is_symlink() for p in roots)
            or any(a.resolve().is_relative_to(b.resolve()) or b.resolve().is_relative_to(a.resolve())
                   for i,a in enumerate(roots) for b in roots[:i]) or not 0 <= time.monotonic()-args.started <= 240):
        parser.error("independent cache256 paths and start required")
    for value in (args.budget_tip,args.pilot_receipt):
        if value and not re.fullmatch("[0-9a-f]{64}",value):
            parser.error("invalid receipt or budget tip SHA")
    if args.recover_writer and not re.fullmatch(r"empty|[0-9a-f]{32}",args.recover_writer):
        parser.error("invalid writer recovery token")
    if args.mode == "long" and not (args.pilot_receipt and args.budget_tip and args.additional_seconds>0 and args.additional_gib>0):
        parser.error("long mode requires explicit confirmed additional limits, pilot receipt and current budget tip")
    run(args,json.loads(_regular_read(args.pins,64*1024)))


if __name__ == "__main__":
    main()
