"""Build the default-off v2.1.0 Drive/CUDA notebook from pinned A evidence."""

import argparse
import hashlib
import json
from pathlib import Path
import tempfile
import zipfile

from rewind_archive.contracts import canonical
from rewind_prepare.temporal_selection import check_pin
from scripts.build_archvis_pilot_handoff import _cell
from scripts.v210_eval import A, COMPARE, DESTINATION, DIAGNOSTIC, FEATURE, FINAL, RECEIPT


ROOT = Path(__file__).resolve().parents[1]
BASE_INDEX_SHA = "4a4e1c119e08f0026ef4af9a87c8edbb757934fbcee7ace1b61c2e424e080f40"
SOURCE_SHA = "9a68d8e45bcf3abf6fab3004740faa6805d4ce3fda5118c3a7102011937849d8"
MODULES = ("v210_eval.py","v210_metrics.py")
NOTEBOOK = "v210_eval_128_A_cuda.ipynb"
TOOLS = "v210_eval_tools.zip"
CONFIG = '''RUN_V210 = False
DRIVE_PROJECT = "/content/drive/MyDrive/REWIND-like prototype"
RECOVER_WRITER_OWNER = ""
'''
RUN = '''
import json
import signal
import subprocess
import sys
import tempfile
import time

def once():
    started = time.monotonic()
    content = Path("/content")
    upload = content/PINS["base"]["source_name"]
    tool_zip = content/TOOL_NAME
    if not upload.is_file() or not tool_zip.is_file():
        raise FileNotFoundError(f"Upload {upload.name} and {tool_zip.name} to /content")
    mount = content/"drive"
    project = Path(DRIVE_PROJECT)
    if project == mount or not project.is_relative_to(mount) or ".." in project.parts:
        raise ValueError("Choose the project below mounted Drive")
    from google.colab import drive
    drive.mount(str(mount), timeout_ms=120_000)
    project = project.resolve(strict=True)
    if not project.is_relative_to(mount.resolve(strict=True)):
        raise ValueError("Project resolves outside Drive")
    work = Path(tempfile.mkdtemp(prefix="rewind-v210-",dir=content))
    source, overlay = work/"source", work/"overlay"
    extract_pinned_zip(upload, PINS["base"]["source_sha256"], source, max_zip_bytes=64*1024**2,
                       max_output_bytes=192*1024**2, max_members=3000, revision=PINS["base"]["revision"])
    extract_pinned_zip(tool_zip, TOOL_SHA, overlay, max_zip_bytes=1024**2, max_output_bytes=1024**2, max_members=2)
    for name, expected in PINS["modules"].items():
        raw = (overlay/name).read_bytes()
        if hashlib.sha256(raw).hexdigest()!=expected:
            raise ValueError("v2.1.0 overlay hash mismatch")
        with (source/"scripts"/name).open("xb") as active:
            active.write(raw)
    pins = work/"pins.json"
    pins.write_text(json.dumps(PINS),encoding="utf-8")
    cmd = [sys.executable,"-u","-m","scripts.v210_eval","--project",str(project),"--work",str(work),
           "--pins",str(pins),"--started",str(started),"--recover-writer",RECOVER_WRITER_OWNER]
    print("Local diagnostic:",work/"result.json",flush=True)
    left = started+28755-time.monotonic()
    if left<=0:
        raise TimeoutError("v2.1.0 deadline")
    with subprocess.Popen(cmd,cwd=source) as child:
        try:
            code = child.wait(timeout=left)
        except BaseException:
            if child.poll() is None:
                child.send_signal(signal.SIGINT)
            try:
                child.wait(timeout=25)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait(timeout=10)
            raise
    if code:
        raise RuntimeError(f"v2.1.0 worker exited with status {code}")

if RUN_V210:
    once()
else:
    print("v2.1.0 실행 스위치가 꺼져 있습니다.")
'''


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def build(source, a_bundle, output):
    if sha(source.read_bytes())!=SOURCE_SHA or output.exists() or not output.parent.is_dir():
        raise ValueError("v210_source_or_output")
    with zipfile.ZipFile(a_bundle) as archive:
        raw = archive.read("BUNDLE_INDEX.json")
        if sha(raw)!=BASE_INDEX_SHA:
            raise ValueError("v210_source_A_index")
        members = json.loads(raw)["members"]
        def original(name):
            value = archive.read(name)
            if members[name]!={"bytes":len(value),"sha256":sha(value)}:
                raise ValueError(f"v210_A_archive_member: {name}")
            return json.loads(value)

        base = original("source/receipt.json")["pins"]
        if base["source_sha256"]!=SOURCE_SHA or source.name!=base["source_name"]:
            raise ValueError("v210_fixed_zip_pin")
        step = original(RECEIPT)
        check_pin(step,"rewind_v2_gpu_training_checkpoint_v1",FINAL)
        diagnostic, comparison = original(DIAGNOSTIC), original(COMPARE)
        if diagnostic["checkpoint"]!=step or diagnostic["comparison"]!=comparison or diagnostic["status"]!="completed":
            raise ValueError("v210_A_evidence_scope")
        groups = {}
        for group in range(129,147):
            folder = f"{A}/features/{FEATURE}"
            key = f"{folder}/group-{group:05d}.json"
            marker = original(key)
            check_pin(marker,"rewind_v2_gpu_features_group_v1",marker["content_sha256"])
            package = marker["archive"]
            if marker["identity"]!=FEATURE or members[f"{folder}/{package['name']}"]!={k:package[k] for k in ("bytes","sha256")}:
                raise ValueError("v210_fixed_feature_index")
            groups[str(group)] = {"marker":members[key],"archive":package}
    overlays = {n:(ROOT/"scripts"/n).read_text(encoding="utf-8").encode() for n in MODULES}
    pins = {"base":base,"evidence":{name:members[name] for name in (DIAGNOSTIC,RECEIPT,COMPARE)},"groups":groups,
            "modules":{n:sha(raw) for n,raw in overlays.items()},"A_index_sha256":BASE_INDEX_SHA}
    with tempfile.TemporaryDirectory(dir=output.parent,prefix=".v210-build-") as temporary:
        root = Path(temporary)/"release"
        root.mkdir()
        tools = root/TOOLS
        with zipfile.ZipFile(tools,"x",compression=zipfile.ZIP_DEFLATED) as writer:
            for name,raw in overlays.items():
                entry = zipfile.ZipInfo(name,date_time=(1980,1,1,0,0,0))
                entry.external_attr = 0o100600<<16
                writer.writestr(entry,raw,compress_type=zipfile.ZIP_DEFLATED)
        tool_hash = sha(tools.read_bytes())
        boot = (ROOT/"scripts/archvis_pilot_bootstrap.py").read_text(encoding="utf-8")
        code = boot+f"\nPINS = {pins!r}\nTOOL_NAME = {TOOLS!r}\nTOOL_SHA = {tool_hash!r}\n"+RUN
        compile(CONFIG,"settings","exec")
        compile(code,NOTEBOOK,"exec")
        raw = canonical({"nbformat":4,"nbformat_minor":5,"metadata":{"kernelspec":{"name":"python3","display_name":"Python 3","language":"python"}},
            "cells":[_cell("markdown","guide",f"# v2.1.0 A 검증 421클립\n\n동결 128 source ZIP과 `{TOOLS}`를 /content에 업로드합니다. CUDA GPU 하나를 선택합니다. 결과는 `{DESTINATION}`에 보관됩니다. 실행은 기본 OFF입니다."),
            _cell("code","settings",CONFIG),_cell("code","run",code)]})
        (root/NOTEBOOK).write_bytes(raw)
        (root/"receipt.json").write_bytes(canonical({"pins":pins,"notebook":NOTEBOOK,"notebook_sha256":sha(raw),
            "tools":TOOLS,"tools_sha256":tool_hash,"default_off":True,"output_folder":DESTINATION}))
        root.rename(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source","a-bundle","output"):
        parser.add_argument("--"+name,type=Path,required=True)
    build(**vars(parser.parse_args()))
