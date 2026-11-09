"""Build the pinned metadata bundle and, after commit, the ready Colab notebook."""

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import re
import zipfile

from rewind_archive.contracts import canonical
from rewind_data.audit import read_json
from rewind_m1.local_rgb import _regular_read
from rewind_prepare.converter import sha
from rewind_prepare.extracted import ExtractedLimits, _Budget, _local, _inventory, official_splits
from rewind_prepare.spec import validate_spec
from scripts.archvis_pilot import ASSETS, ENVIRONMENT, INVENTORY_SHA256, MIB


CONFIG = {"seed": 71, "batch_size": 2, "learning_rate": 0.0001, "scheduler_step": 51,
          "scheduler_gamma": 1.0, "amp": False, "training_image_size": [64, 64],
          "frontend_version": "archvis_direct_v0", "max_native_tensor_bytes": 64 * MIB,
          "max_heatmap_elements": 2 * MIB}


def build_inputs(metadata, inventory, spec_path, output):
    metadata, inventory, spec_path = (_local(metadata, directory=True), _local(inventory), _local(spec_path))
    spec_raw = _regular_read(spec_path, 256 * 1024)
    spec = validate_spec(read_json(spec_raw))
    budget = _Budget(ExtractedLimits(16 * MIB, 60))
    owners, _ = official_splits(metadata / "split", spec, budget)
    expected, duplicates, _ = _inventory(inventory, INVENTORY_SHA256, owners, ENVIRONMENT, budget)
    inventory_raw = _regular_read(inventory, MIB)
    if sha(inventory_raw) != INVENTORY_SHA256:
        raise ValueError("archvis_inventory_changed")
    data = {name: _regular_read(_local(metadata / name), 4 * MIB) for name in ASSETS[:4]}
    if (sha(data["scaramuzza.json"]) != spec["pinned_sources"]["scaramuzza_sha256"]
            or any(sha(data[f"split/{split}.txt"]) != pin for split, pin in spec["pinned_sources"]["split_sha256"].items())):
        raise ValueError("archvis_official_metadata_changed")
    sequences = sorted({seq for seq, _ in expected})
    counts = Counter(owners[seq] for seq in sequences)
    if counts != {"train": 286, "validation": 30} or len(duplicates) != 1:
        raise ValueError("archvis_inventory_scope")
    motions = {"schema_version": "explicit_motion_map_v1", "sequences": {}}
    groups = defaultdict(list)
    for seq in sequences:
        motions["sequences"][seq] = {"motion_id": "declared_sequence/" + sha(seq.encode()),
            "identity_status": "declared_unverified",
            "evidence": "Sequence path declared by the pinned metadata inventory. Original motion identity is unverified."}
        name = re.sub(r"__\d+_?$", "", seq.rsplit("/", 1)[1]).casefold()
        groups[name].append({"sequence": seq, "split": owners[seq]})
    review = {"schema": "archvis_motion_review_draft_v1", "independent_validation_verified": False,
              "matching_method": "casefolded display name with terminal __number removed; review candidates only",
              "sequence_counts": dict(counts), "cross_split_candidates": [
                  {"normalized_display_name": name, "sequences": rows}
                  for name, rows in sorted(groups.items()) if len({r["split"] for r in rows}) > 1],
              "known_file_duplicates": read_json(inventory_raw)["duplicates"]}
    data.update({"inventory.json": inventory_raw, "motion_map.json": canonical(motions) + b"\n",
                 "motion_review.json": canonical(review) + b"\n",
                 "mse.json": canonical({**CONFIG, "loss_kind": "mse"}) + b"\n",
                 "ce.json": canonical({**CONFIG, "loss_kind": "gaussian_cross_entropy"}) + b"\n"})
    receipt = {"schema": "archvis_pilot_inputs_v1", "spec_sha256": sha(spec_raw),
               "files": {name: {"sha256": sha(raw), "bytes": len(raw)} for name, raw in data.items()},
               "sequence_counts": dict(counts), "cross_split_name_candidate_groups": len(review["cross_split_candidates"]),
               "unresolved": ["mounted RGB and pose roots", "original motion identity",
                              "payload and visible RGB-pose alignment", "Colab device and resource limits",
                              "manifest and validation payload pins generated after runtime preparation"]}
    data["input_receipt.json"] = canonical(receipt) + b"\n"
    output = Path(output)
    output.mkdir(exist_ok=False)
    bundle = output / "archvis_pilot_inputs.zip"
    with zipfile.ZipFile(bundle, "x") as archive:
        for name, raw in sorted(data.items()):
            item = zipfile.ZipInfo(name)
            item.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(item, raw)
    for name in ("motion_map.json", "motion_review.json", "input_receipt.json"):
        (output / name).write_bytes(data[name])
    summary = {"bundle": bundle.name, "sha256": sha(bundle.read_bytes()), "bytes": bundle.stat().st_size,
               **{key: receipt[key] for key in ("sequence_counts", "cross_split_name_candidate_groups", "unresolved")}}
    (output / "handoff_pending.json").write_bytes(canonical(summary) + b"\n")
    return summary


def _cell(kind, ident, source):
    cell = {"cell_type": kind, "id": ident, "metadata": {}, "source": source.splitlines(keepends=True)}
    if kind == "code":
        cell.update(execution_count=None, outputs=[])
    return cell


def build_ready(source_zip, input_zip, revision, output):
    if not re.fullmatch("[0-9a-f]{40}", revision):
        raise ValueError("archvis_ready_revision")
    source_zip, input_zip = _local(source_zip), _local(input_zip)
    if source_zip.name == input_zip.name:
        raise ValueError("archvis_ready_same_zip_names")
    source_raw = _regular_read(source_zip, 64 * MIB)
    input_raw = _regular_read(input_zip, 8 * MIB)
    import io

    with zipfile.ZipFile(io.BytesIO(source_raw)) as archive:
        names = archive.namelist()
        if len({name.casefold() for name in names}) != len(names):
            raise ValueError("archvis_ready_source_duplicate_member")
        stamp = archive.getinfo("SOURCE_REVISION.txt")
        if not 40 <= stamp.file_size <= 42 or archive.read(stamp).decode("ascii").strip() != revision:
            raise ValueError("archvis_ready_source_stamp")
        if "scripts/archvis_pilot.py" not in names:
            raise ValueError("archvis_ready_source_missing_workflow")
        spec_info = archive.getinfo("configs/real_archvis62_v1.json")
        if not 1 <= spec_info.file_size <= 256 * 1024:
            raise ValueError("archvis_ready_source_spec_size")
        spec_raw = archive.read(spec_info)
        spec = validate_spec(read_json(spec_raw))
    with zipfile.ZipFile(io.BytesIO(input_raw)) as archive:
        entries = archive.infolist()
        if (len(entries) != len(ASSETS) + 1 or {entry.filename for entry in entries} != {*ASSETS, "input_receipt.json"}
                or sum(entry.file_size for entry in entries) > 16 * MIB):
            raise ValueError("archvis_ready_input_members")
        receipt_info = archive.getinfo("input_receipt.json")
        if not 1 <= receipt_info.file_size <= 64 * 1024:
            raise ValueError("archvis_ready_input_receipt_size")
        receipt = read_json(archive.read(receipt_info))
        if (receipt.get("schema") != "archvis_pilot_inputs_v1"
                or not isinstance(receipt.get("files"), dict) or set(receipt["files"]) != set(ASSETS)):
            raise ValueError("archvis_ready_input_receipt")
        if receipt.get("spec_sha256") != sha(spec_raw):
            raise ValueError("archvis_ready_source_spec_pin_mismatch")
        official_pins = {
            f"split/{split}.txt": pin for split, pin in spec["pinned_sources"]["split_sha256"].items()
        }
        official_pins["scaramuzza.json"] = spec["pinned_sources"]["scaramuzza_sha256"]
        for name in ASSETS:
            info = archive.getinfo(name)
            if not 1 <= info.file_size <= (MIB if name == "inventory.json" else 8 * MIB):
                raise ValueError("archvis_ready_input_member_size:" + name)
            raw = archive.read(info)
            pin = sha(raw)
            if receipt["files"][name] != {"bytes": len(raw), "sha256": pin}:
                raise ValueError("archvis_ready_input_receipt_mismatch:" + name)
            if name == "inventory.json" and pin != INVENTORY_SHA256:
                raise ValueError("archvis_ready_inventory")
            if name in official_pins and pin != official_pins[name]:
                raise ValueError("archvis_ready_official_pin_mismatch:" + name)
    pins = {"source": {"name": source_zip.name, "sha256": sha(source_raw), "revision": revision},
            "inputs": {"name": input_zip.name, "sha256": sha(input_raw)}}
    bootstrap = Path(__file__).with_name("archvis_pilot_bootstrap.py").read_text(encoding="utf-8")
    cells = [
        _cell("markdown", "guide", """# ArchVis 0버전 1,000페어 비교 실행
테스트 세션에서 받은 두 ZIP을 Colab 파일 영역에 올립니다. 아래 셀을 차례대로 사용합니다.
후보 경로와 stat 비용을 검토한 뒤 승인 SHA를 다음 셀에 옮깁니다. 경로와 자원 한도는 현재 런타임에서 확인합니다.
준비 단계와 비교 단계가 각각 고정된 뒤 실행됩니다. 마지막 셀에서 정상 또는 실패 기록을 회수합니다.
"""),
        _cell("code", "bootstrap", bootstrap + "\n\n" + f"PINS = {pins!r}\n" + '''import json
import sys

PACKAGES = Path("/content")
WORK = Path("/content/archvis_1000_pilot")
WORK.mkdir(exist_ok=False)
boot = {}
try:
    for name, cap, expanded in (("source", 64, 192), ("inputs", 8, 16)):
        pin = PINS[name]
        boot[name] = extract_pinned_zip(PACKAGES / pin["name"], pin["sha256"], WORK / name,
            max_zip_bytes=cap*1024**2, max_output_bytes=expanded*1024**2,
            max_members=3000 if name == "source" else 32, revision=pin.get("revision"))
    sys.path.insert(0, str(WORK / "source"))
    from scripts import archvis_pilot as pilot
    pilot.assets(WORK)
    boot["status"] = "verified"
except (Exception, KeyboardInterrupt) as exc:
    boot.update(status="failed", failure={"type": type(exc).__name__, "reason": str(exc)[:512]})
    raise
finally:
    with (WORK / "bootstrap_report.json").open("x", encoding="utf-8") as stream:
        json.dump(boot, stream, ensure_ascii=False, indent=2)

def stage(name, action):
    try:
        return action()
    except (Exception, KeyboardInterrupt) as exc:
        path = WORK / (name + "_failure.json")
        if not path.exists():
            pilot._write(path, {"status": "failed", "type": type(exc).__name__, "reason": str(exc)[:512]})
        raise
'''),
        _cell("code", "discover", '''MOUNT_ROOT = Path("/content/drive/MyDrive")
def mounted_candidates():
    if not MOUNT_ROOT.is_dir():
        from google.colab import drive
        drive.mount("/content/drive")
    return pilot.discover_roots(MOUNT_ROOT)
candidates = stage("discover", mounted_candidates)
pilot._write(WORK / "roots_candidates.json", candidates)
print(json.dumps(candidates, ensure_ascii=False, indent=2))
'''),
        _cell("code", "preflight", '''RGB_ROOT = None  # 후보가 여러 개면 정확한 경로를 입력합니다.
POSE_ROOT = None
CONFIRM_EXACT_ROOTS = False
METADATA_MAX_SECONDS = None
MAX_DIRECTORY_ENTRIES = 200_000
if CONFIRM_EXACT_ROOTS:
    def confirmed(role, value):
        options = candidates["candidates"][role]
        if value:
            return value
        if len(options) == 1 and candidates["stopped"] is None:
            return options[0]
        raise ValueError("경로를 명시하세요: " + role)
    plan = stage("preflight", lambda: pilot.preflight(
        WORK, MOUNT_ROOT, confirmed("rgb", RGB_ROOT), confirmed("pose", POSE_ROOT),
        max_seconds=METADATA_MAX_SECONDS, max_entries=MAX_DIRECTORY_ENTRIES))
    print(json.dumps({"status": plan["status"], "preflight_sha256": plan["content_sha256"],
                      "roots": plan["plan"]["roots"], "counts": plan["plan"]["counts"],
                      "duplicates": plan["plan"]["duplicates"], "cost": plan["cost"]},
                     ensure_ascii=False, indent=2))
else:
    print("경로와 metadata 시간 상한을 확인한 뒤 CONFIRM_EXACT_ROOTS를 켭니다.")
'''),
        _cell("code", "prepare", '''APPROVED_PREFLIGHT_SHA256 = None
PREPARE_MAX_READ_BYTES = None
PREPARE_MAX_SECONDS = None
if APPROVED_PREFLIGHT_SHA256:
    prepared = stage("prepare", lambda: pilot.prepare(
        WORK, APPROVED_PREFLIGHT_SHA256, max_read_bytes=PREPARE_MAX_READ_BYTES,
        max_seconds=PREPARE_MAX_SECONDS, max_entries=MAX_DIRECTORY_ENTRIES))
    print(json.dumps({k: prepared[k] for k in ("status", "manifest_sha256", "validation_keys_sha256",
                                              "elapsed_seconds", "read_headroom_pct_of_stat_minimum")}, indent=2))
else:
    print("preflight를 검토한 뒤 SHA와 준비 읽기·시간 상한을 입력합니다.")
'''),
        _cell("code", "budget", '''FREEZE_COMPARISON = False
DEVICE = "cuda"
LIMITS_EACH = {name: None for name in pilot.LIMIT_FIELDS}
print(json.dumps(stage("budget", lambda: pilot.runtime_resources(WORK)), ensure_ascii=False, indent=2))
if FREEZE_COMPARISON:
    comparison = stage("budget", lambda: pilot.comparison_plan(WORK, device=DEVICE, limits=LIMITS_EACH))
    print(json.dumps(comparison, ensure_ascii=False, indent=2))
else:
    print("실제 Colab 장치와 각 실행의 시간·시도·읽기·RAM·디스크·GPU 상한을 확인합니다.")
'''),
        _cell("code", "train", '''APPROVED_COMPARISON_SHA256 = None
if APPROVED_COMPARISON_SHA256:
    result = stage("train", lambda: pilot.compare(WORK, APPROVED_COMPARISON_SHA256))
    print(json.dumps(result, ensure_ascii=False, indent=2))
else:
    print("비교 계획의 SHA를 확인한 뒤 입력합니다.")
'''),
        _cell("code", "recover", '''from google.colab import files

if "pilot" in globals():
    recovery = pilot.recover(WORK)
else:
    recovery = WORK / "archvis_bootstrap_failure.zip"
    with zipfile.ZipFile(recovery, "x") as archive:
        archive.write(WORK / "bootstrap_report.json", "bootstrap_report.json")
print(recovery, hashlib.sha256(recovery.read_bytes()).hexdigest())
files.download(str(recovery))
'''),
    ]
    doc = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                        "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
    for cell in cells:
        if cell["cell_type"] == "code":
            compile("".join(cell["source"]), cell["id"], "exec")
    with Path(output).open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(doc, handle, ensure_ascii=False, indent=1)
        handle.write("\n")
    return pins


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    metadata = sub.add_parser("inputs")
    for key in ("metadata", "inventory", "spec-path", "output"):
        metadata.add_argument("--" + key, type=Path, required=True)
    ready = sub.add_parser("ready")
    for key in ("source-zip", "input-zip", "output"):
        ready.add_argument("--" + key, type=Path, required=True)
    ready.add_argument("--revision", required=True)
    args = vars(parser.parse_args(argv))
    command = args.pop("command")
    result = (build_inputs if command == "inputs" else build_ready)(**args)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
