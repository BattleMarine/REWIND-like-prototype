"""Freeze the Colab export notebook and Windows/POSIX tool kit."""

import argparse
from pathlib import Path
import tempfile
import zipfile

from scripts.build_archvis_pilot_handoff import _cell
from scripts import v2_gpu128_a_bundle_format as fmt


ROOT = Path(__file__).resolve().parents[1]
TOOLKIT = "v2_128_A_export_tools.zip"
NOTEBOOK = "v2_128_A_export.ipynb"
FILES = ("export_v2_128_a_bundle.py", "v2_bundle.py", "v2_gpu128_a_bundle_export.py", "v2_gpu128_a_bundle_format.py",
         "v2_gpu128_a_bundle_local.py", "v2_gpu128_a_bundle_predict.py", "v2_gpu128_a_bundle_tune.py",
         "v2_gpu128_a_bundle_train_full.py", "v2_gpu128_a_bundle_edit.py")
SETTINGS = '''RUN_EXPORT = False
DRIVE_PROJECT = "/content/drive/MyDrive/REWIND-like prototype"
'''
RUN = '''
import json
import signal
import subprocess
import sys
import tempfile
import time

def export_once():
    started = time.monotonic()
    upload = Path("/content")
    source_zip, kit_zip = upload/PINS["source_name"], upload/PINS["kit_name"]
    if not source_zip.is_file() or not kit_zip.is_file():
        raise FileNotFoundError(f"Upload {source_zip.name} and {kit_zip.name} to /content")
    mount, project = upload/"drive", Path(DRIVE_PROJECT)
    if project == mount or not project.is_relative_to(mount) or ".." in project.parts:
        raise ValueError("Set a project under Drive")
    from google.colab import drive
    drive.mount(str(mount), timeout_ms=120_000)
    project = project.resolve(strict=True)
    if not project.is_relative_to(mount.resolve(strict=True)):
        raise ValueError("Project resolved outside Drive")
    work = Path(tempfile.mkdtemp(prefix="v2-128-A-export-", dir=upload))
    print("Local export report:", work / "verified-export.json", flush=True)
    source, tools = work/"frozen", work/"tools"
    extract_pinned_zip(source_zip, PINS["source_sha256"], source, max_zip_bytes=64*1024**2,
                       max_output_bytes=192*1024**2, max_members=3000, revision=PINS["source_revision"])
    extract_pinned_zip(kit_zip, PINS["kit_sha256"], tools, max_zip_bytes=2*1024**2,
                       max_output_bytes=4*1024**2, max_members=32)
    remaining = min(28700, 28750-int(time.monotonic()-started))
    if remaining <= 0:
        raise TimeoutError("Export cell deadline")
    bootstrap = "import runpy,sys; sys.path[:0]=sys.argv[1:3]; del sys.argv[1:3]; runpy.run_path(sys.argv.pop(1),run_name='__main__')"
    cmd = [sys.executable, "-u", "-c", bootstrap, str(source), str(tools), str(tools/"v2_gpu128_a_bundle_export.py"),
           "--project", str(project), "--source-zip", str(source_zip), "--source-receipt", str(tools/"source-receipt.json"),
           "--tools", str(tools), "--work", str(work), "--seconds", str(remaining)]
    with subprocess.Popen(cmd) as worker:
        try:
            status = worker.wait(timeout=remaining)
        except BaseException:
            if worker.poll() is None:
                worker.send_signal(signal.SIGINT)
            try:
                worker.wait(timeout=20)
            except subprocess.TimeoutExpired:
                worker.kill()
                worker.wait(timeout=10)
            raise
    if status:
        raise RuntimeError(f"Export worker exited with status {status}")

if RUN_EXPORT:
    export_once()
else:
    print("내보내기 스위치가 꺼져 있습니다.")
'''


def build(original_source, source_receipt, output):
    import json
    if (output.exists() or not output.parent.is_dir() or fmt.sha(original_source.read_bytes()) != fmt.SOURCE_SHA
            or fmt.sha(source_receipt.read_bytes()) != fmt.SOURCE_RECEIPT_SHA):
        raise ValueError("v2_bundle_fixed_source_or_new_output")
    receipt = json.loads(source_receipt.read_bytes())["pins"]
    files = {name: (ROOT/"scripts"/name).read_text(encoding="utf-8").encode() for name in FILES}
    files["GUIDE_KO.md"] = (ROOT/"docs/V2_GPU128_A_PORTABLE_BUNDLE.md").read_text(encoding="utf-8").encode()
    files["source-receipt.json"] = source_receipt.read_bytes()
    for name in FILES:
        compile(files[name], name, "exec")
    with tempfile.TemporaryDirectory(dir=output.parent, prefix=".v2-a-export-") as directory:
        root = Path(directory)/"release"
        root.mkdir()
        kit = root/TOOLKIT
        with zipfile.ZipFile(kit, "x") as archive:
            for name, raw in sorted(files.items()):
                info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
                info.external_attr = 0o100600 << 16
                archive.writestr(info, raw, zipfile.ZIP_DEFLATED, compresslevel=6)
        pins = {"source_name": receipt["source_name"], "source_sha256": fmt.SOURCE_SHA, "source_revision": receipt["revision"],
                "kit_name": TOOLKIT, "kit_sha256": fmt.sha(kit.read_bytes()), "feature_identity": fmt.FEATURE_IDENTITY,
                "final_receipt": fmt.FINAL_RECEIPT}
        bootstrap = (ROOT/"scripts/archvis_pilot_bootstrap.py").read_text(encoding="utf-8")
        body = bootstrap+f"\nPINS = {pins!r}\n"+RUN
        compile(body, NOTEBOOK, "exec")
        guide = ("# v2 128 A 모델과 전체 캐시 ZIP64\n\n"
                 f"CPU Colab의 `/content`에 `{pins['source_name']}`과 `{TOOLKIT}`을 올립니다. "
                 "실행은 기본 OFF입니다. 전체 캐시·특징·모델의 새 ZIP을 Drive에 쓰고 전체 readback 후 게시합니다. "
                 "예상 결과는 약 51 GiB입니다. 결과를 모두 확인한 뒤 하나의 ZIP 경로와 SHA를 회수합니다.\n")
        notebook = fmt.canonical({"nbformat":4, "nbformat_minor":5, "metadata":{"kernelspec":{
            "name":"python3", "display_name":"Python 3", "language":"python"}}, "cells":[
                _cell("markdown", "guide", guide), _cell("code", "settings", SETTINGS), _cell("code", "export", body)]})
        (root/NOTEBOOK).write_bytes(notebook)
        (root/"receipt.json").write_bytes(fmt.canonical({**pins, "notebook": NOTEBOOK, "notebook_sha256": fmt.sha(notebook),
            "tool_files": {n: {"bytes":len(raw), "sha256":fmt.sha(raw)} for n, raw in files.items()}, "default_off": True}))
        root.rename(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("original-source", "source-receipt", "output"):
        parser.add_argument("--"+name, required=True, type=Path)
    build(**vars(parser.parse_args()))
