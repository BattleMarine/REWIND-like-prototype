"""Build the pinned ArchVis v0 Colab notebook from the verified restore handoff."""

import argparse
import base64
import io
import json
from pathlib import Path
import zipfile

from rewind_archive.contracts import canonical, digest
from rewind_data.audit import read_json
from rewind_m1.archvis_v0 import (
    ARTIFACT_LIMIT, CACHE_LIMIT, CHECKPOINT_EVERY, CHECKPOINT_LIMIT, CONFIG, COUNTS, CUDA_ALLOCATED_LIMIT,
    CUDA_RESERVED_LIMIT, EVALUATE, MANIFEST, MAX_ATTEMPTS, MIB, RSS_LIMIT, RUN_SECONDS, VALIDATION, WALL_SECONDS,
)
from scripts.archvis_pilot_restore import _restore_read, _restore_sha
from scripts.build_archvis_pilot_handoff import _cell


RESTORE_NOTEBOOK = "6f6bb87791ac7270e1863ffbcc75cdc4625c2f19f540c7c96e0f63bc79de95b4"
PREPARED_ZIP = "e8a74121689c3910123122f9b88f025f98d0334bf3e290e36978da8f6677ac13"
BASE_REVISION = "ceb8591a3c2c45549d04f9fda6cab871ecd47ccf"
FILES = ("rewind_m1/archvis_v0.py", "rewind_m1/archvis_v0_model.py")


def build(restore_dir, output):
    restore_dir, output = Path(restore_dir), Path(output)
    original = _restore_read(restore_dir / "archvis_1000_pilot_restore_001.ipynb", MIB)
    prepared = _restore_read(restore_dir / "archvis_1000_prepared_001.zip", 8 * MIB)
    if (_restore_sha(original), _restore_sha(prepared)) != (RESTORE_NOTEBOOK, PREPARED_ZIP):
        raise ValueError("archvis_v0_original_package_pin")
    with zipfile.ZipFile(io.BytesIO(prepared)) as archive:
        manifest = read_json(archive.read("manifest/manifest.json"))
    if manifest["content_sha256"] != MANIFEST or manifest["selection"]["validation_keys_sha256"] != VALIDATION:
        raise ValueError("archvis_v0_builder_manifest_pin")
    source_root = Path(__file__).resolve().parent.parent
    pins = {}
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for name in FILES:
            raw = _restore_read(source_root / name, MIB)
            archive.writestr(zipfile.ZipInfo(name), raw, compress_type=zipfile.ZIP_DEFLATED)
            pins[name] = {"bytes": len(raw), "sha256": _restore_sha(raw)}
    overlay = buffer.getvalue()
    document = read_json(original)
    cells = {cell["id"]: "".join(cell["source"]) for cell in document["cells"]}
    bootstrap = cells["bootstrap"]
    marker = 'WORK = Path("/content/archvis_1000_pilot_restored")'
    if bootstrap.count(marker) != 1:
        raise ValueError("archvis_v0_unexpected_restore_bootstrap")
    bootstrap = bootstrap.replace(marker, 'WORK = Path("/content/archvis_1000_v0")')
    bootstrap = bootstrap.replace("original initial evaluation and per-batch size/SHA/PNG checks",
                                  "selected source size/SHA/PNG checked once at local cache build")
    plan = {
        "schema": "archvis_1000_v0_plan_v1", "device": "cuda", "config": CONFIG, "counts": COUNTS,
        "manifest_sha256": MANIFEST, "validation_keys_sha256": VALIDATION,
        "evaluation_epochs": EVALUATE, "max_attempts": MAX_ATTEMPTS, "checkpoint_every_updates": CHECKPOINT_EVERY,
        "hard_training_subprocess_wall_seconds": WALL_SECONDS, "cooperative_run_seconds": RUN_SECONDS,
        "cache_bytes": CACHE_LIMIT, "checkpoint_bytes": CHECKPOINT_LIMIT, "artifact_bytes": ARTIFACT_LIMIT,
        "rss_bytes": RSS_LIMIT, "cuda_allocated_bytes": CUDA_ALLOCATED_LIMIT, "cuda_reserved_bytes": CUDA_RESERVED_LIMIT,
        "source_payload_read_bytes": sum(item["size_bytes"] + 1 for row in manifest["records"]
                                         for item in row["files"].values()),
        "base_source_revision": BASE_REVISION,
        "source_overlay": {"revision_kind": "sha256_of_overlay_zip", "sha256": _restore_sha(overlay), "files": pins},
        "base_restore_notebook_sha256": RESTORE_NOTEBOOK, "prepared_zip_sha256": PREPARED_ZIP,
        "scope": "2D development, train_ready=false, original motion and visual alignment unverified",
    }
    plan["content_sha256"] = digest(plan)
    bootstrap += f'''
import base64
V0_PLAN = {json.loads(canonical(plan))!r}
V0_SOURCE = {base64.b64encode(overlay).decode()!r}

def install_v0_overlay():
    packed = WORK / "v0_overlay.zip"
    with packed.open("xb") as stream:
        stream.write(base64.b64decode(V0_SOURCE, validate=True))
    overlay = V0_PLAN["source_overlay"]
    extract_pinned_zip(packed, overlay["sha256"], WORK / "v0_overlay",
                       max_zip_bytes=2*1024**2, max_output_bytes=2*1024**2, max_members=2)
    for name, pin in overlay["files"].items():
        raw = _restore_read(WORK / "v0_overlay" / name, 1024**2)
        _restore_require(pin == {{"bytes": len(raw), "sha256": _restore_sha(raw)}}, "v0_source_pin")
        with (WORK / "source" / name).open("xb") as stream:
            stream.write(raw)
    _restore_require(digest({{k: v for k, v in V0_PLAN.items() if k != "content_sha256"}})
                     == V0_PLAN["content_sha256"], "v0_plan_pin")
    pilot._write(WORK / "v0_plan.json", V0_PLAN)

stage("v0_overlay", install_v0_overlay)
print("v0 plan SHA256:", V0_PLAN["content_sha256"])
'''
    launch = '''
from contextlib import suppress
import os
import tempfile

APPROVED_V0_PLAN_SHA256 = ""
RESUME_RECOVERY_SHA256 = ""

def write_v0_launch_result(path, launch):
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".v0-launch-", delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(canonical(launch) + b"\\n")
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    finally:
        if temporary is not None:
            with suppress(OSError):
                temporary.unlink(missing_ok=True)

def launch_v0():
    import subprocess

    plan = V0_PLAN
    _restore_require(APPROVED_V0_PLAN_SHA256 == plan["content_sha256"], "v0_plan_approval_required")
    _restore_seal(read_json(_restore_read(WORK / "v0_plan.json")), "archvis_1000_v0_plan_v1")
    _restore_require(read_json(_restore_read(WORK / "v0_plan.json")) == plan, "v0_plan_file_changed")
    _restore_require(not (WORK / "v0_launch.json").exists(), "v0_one_launch_per_workspace")
    for name, pin in plan["source_overlay"]["files"].items():
        _restore_require(_restore_sha(_restore_read(WORK / "source" / name)) == pin["sha256"],
                         "v0_installed_source_changed")
    if not Path(restored["mount_root"]).is_dir():
        from google.colab import drive
        drive.mount("/content/drive")
    confirmed = confirm_original_roots(WORK, expected_files=RESTORE_FILE_PINS, revision=PINS["source"]["revision"])
    resources = pilot.runtime_resources(WORK)
    _restore_require(resources["cuda_available"] and resources["gpu_total_bytes"] >= 8*1024**3
                     and resources["host_available_bytes"] >= 8*1024**3
                     and resources["disk_free_bytes"] >= 1024**3, "v0_runtime_resources")
    command = [sys.executable, "-m", "rewind_m1.archvis_v0", "--manifest", str(WORK / "manifest/manifest.json"),
               "--spec", str(WORK / "source/configs/real_archvis62_v1.json"), "--output", str(WORK / "v0"),
               "--rgb-root", confirmed["roots"]["rgb"], "--pose-root", confirmed["roots"]["pose"], "--device", "cuda"]
    if RESUME_RECOVERY_SHA256:
        from rewind_m1.archvis_v0 import restore_run
        extract_pinned_zip(PACKAGES / "archvis_1000_v0_recovery.zip", RESUME_RECOVERY_SHA256, WORK / "prior_v0",
                           max_zip_bytes=64*1024**2, max_output_bytes=96*1024**2, max_members=20)
        last = restore_run(WORK / "prior_v0", WORK / "v0", plan_sha256=plan["content_sha256"])
        command += ["--resume-sha256", last]
    launch = {"status": "started", "runtime": resources, "plan_sha256": plan["content_sha256"],
              "resumed_recovery_sha256": RESUME_RECOVERY_SHA256 or None,
              "time_scope": "training subprocess including verified cache build; Drive mount and result download excluded"}
    launch_path = WORK / "v0_launch.json"
    pilot._write(launch_path, launch)
    try:
        child = subprocess.run(command, cwd=WORK / "source", timeout=plan["hard_training_subprocess_wall_seconds"])
        launch.update(status="exited", returncode=child.returncode)
        child.check_returncode()
    except BaseException as exc:
        launch.update(status="stopped", failure_type=type(exc).__name__, reason=str(exc)[:512])
        # A receipt write failure must not replace the subprocess error.
        with suppress(BaseException):
            write_v0_launch_result(launch_path, launch)
        raise
    else:
        write_v0_launch_result(launch_path, launch)

if APPROVED_V0_PLAN_SHA256:
    try:
        launch_v0()
    except (Exception, KeyboardInterrupt) as exc:
        with suppress(BaseException):
            failure_path = WORK / "v0_failure.json"
            if not failure_path.exists():
                pilot._write(failure_path, {"status": "failed", "type": type(exc).__name__, "reason": str(exc)[:512]})
        raise
else:
    print("검토된 v0 계획 SHA를 입력하면 원래 Drive 경로 확인과 학습을 한 번에 시작합니다.")
'''
    recover = '''
from rewind_m1.archvis_v0 import recover_run
recovery, receipt = recover_run(WORK)
print(recovery, _restore_sha(recovery.read_bytes()), sorted(receipt["files"]))
from google.colab import files
files.download(str(recovery))
'''
    document["cells"] = [
        _cell("markdown", "guide", """# ArchVis 1,000페어 v0 2D 학습
원본 source·inputs와 회수 준비 ZIP을 /content에 올리고 복원 셀을 실행합니다.
검토된 계획 SHA를 실행 셀에 넣으면 Drive 경로를 확인하고 한 번의 GPU 실행을 시작합니다.
선택된 RGB·pose 파일만 원본 해시와 PNG decode를 검증해 기존 64px 전처리 결과를 임시 SSD에 보관합니다.
초기와 1·5·10·15·20 epoch에 공식 validation 전체를 평가합니다. 1 epoch는 기존 CE의 500 update 비교점입니다.
학습된 epoch 중 화면 안 whole native 1024px 오차가 가장 작은 checkpoint를 best로 저장합니다. 초기 평가 결과도 따로 비교합니다.
100 update마다 last를 저장합니다. 저장 사이에 중단하면 최근 최대 99 update 또는 마지막 미완료 평가를 다시 수행할 수 있습니다.
체크포인트를 쓰는 도중 실패하면 방금 완료한 저장 경계 update도 다시 수행할 수 있습니다.
중단·실패 시에도 회수 셀을 실행합니다. 새 런타임은 같은 계획과 실행 환경을 사용하고 회수 ZIP의 SHA를 입력해 저장 지점부터 재개합니다.
재개에서도 같은 선택 데이터로 로컬 캐시를 새로 검증하며, 실행당 같은 2시간 상한을 적용합니다.
원본 motion 독립성·RGB/pose 정합은 미확인 상태이며 이 실행의 범위는 2D 개발입니다.
"""),
        _cell("code", "bootstrap", bootstrap), _cell("code", "train", launch), _cell("code", "recover", recover),
    ]
    for cell in document["cells"][1:]:
        compile("".join(cell["source"]), cell["id"], "exec")
    output.mkdir(exist_ok=False)
    notebook = (json.dumps(document, ensure_ascii=False, indent=1) + "\n").encode("utf-8")
    (output / "archvis_1000_v0.ipynb").write_bytes(notebook)
    (output / "archvis_v0_source_overlay.zip").write_bytes(overlay)
    (output / "v0_plan.json").write_bytes(canonical(plan) + b"\n")
    receipt = {"plan_sha256": plan["content_sha256"], "base_source_revision": BASE_REVISION,
               "notebook": {"bytes": len(notebook), "sha256": _restore_sha(notebook)},
               "overlay": {"bytes": len(overlay), "sha256": _restore_sha(overlay)},
               "builder_sha256": _restore_sha(Path(__file__).read_bytes())}
    (output / "v0_release_receipt.json").write_bytes(canonical(receipt) + b"\n")
    return receipt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--restore-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    print(json.dumps(build(args.restore_dir, args.output), indent=2))


if __name__ == "__main__":
    main()
