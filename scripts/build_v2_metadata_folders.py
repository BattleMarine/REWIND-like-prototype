"""Build the independent default-off notebook for the nine pinned direct listings."""

import argparse
from pathlib import Path
import tempfile

from rewind_archive.contracts import canonical
from scripts.archvis_pilot_bootstrap import extract_pinned_zip
from scripts.build_archvis_pilot_handoff import _cell
from scripts.build_v2_metadata_fence import notebook as fence_notebook
from scripts.check_v2_metadata_fence import ZIP_SHA
from scripts.check_v2_metadata_folders import REPORT_SHA
from scripts.resume_v2_drive_metadata import sha_file


def notebook(pins):
    doc = fence_notebook(pins)
    doc["cells"][0] = _cell("markdown", "guide",
        "# v2 지정 폴더 9개 직계 대조\n\n"
        "04:18:37 UTC 원본 results.zip과 신규 source ZIP을 /content에 준비합니다. "
        "05:00 보고 ZIP은 source에 고정 포함됩니다. 기본 스위치는 꺼져 있습니다. "
        "원본을 검증한 뒤에만 지정 폴더 9개의 직계 목록과 관련 메타데이터를 읽습니다.\n")
    settings = "".join(doc["cells"][1]["source"]).replace("RUN_CHANGES_AND_COMPARE", "RUN_DIRECT_COMPARE")
    doc["cells"][1] = _cell("code", "settings", settings)
    code = "".join(doc["cells"][2]["source"])
    code = code.replace("RUN_CHANGES_AND_COMPARE", "RUN_DIRECT_COMPARE")
    code = code.replace("scripts.check_v2_metadata_fence", "scripts.check_v2_metadata_folders")
    code = code.replace("v2_metadata_changes_20261119", "v2_metadata_folders_20261119")
    code = code.replace('"v2_metadata_changes_"+stamp', '"v2_metadata_folders_"+stamp')
    anchor = '    call = [sys.executable,"-m","scripts.check_v2_metadata_folders"]'
    if code.count(anchor) != 1:
        raise ValueError("v2_folders_notebook_template")
    code = code.replace(anchor, '    common += ["--scope-report",str(source/"reference/v2_scope_050017Z_results.zip")]\n' + anchor)
    doc["cells"][2] = _cell("code", "check", code)
    return doc


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
        for path in ("scripts/check_v2_metadata_folders.py", "scripts/check_v2_metadata_fence.py",
                     "scripts/check_v2_metadata_scope.py", "scripts/resume_v2_drive_metadata.py",
                     "reference/spec.json", "reference/split/train.txt", "reference/split/validation.txt", "reference/split/test.txt"):
            if not (src/path).is_file():
                raise ValueError("v2_folders_source_member:" + path)
        if sha_file(src/"reference/v2_scope_050017Z_results.zip") != REPORT_SHA:
            raise ValueError("v2_folders_scope_report_source_pin")
    pins = {"source_name": args.source.name, "source_sha256": args.source_sha, "revision": args.revision,
            "parent_zip_sha256": ZIP_SHA, "scope_report_zip_sha256": REPORT_SHA}
    args.output.mkdir()
    ipynb = args.output/"v2_metadata_folders.ipynb"
    ipynb.write_bytes(canonical(notebook(pins)))
    (args.output/"receipt.json").write_bytes(canonical({"pins": pins, "notebook_sha256": sha_file(ipynb), "default_off": True}))


if __name__ == "__main__":
    main()
