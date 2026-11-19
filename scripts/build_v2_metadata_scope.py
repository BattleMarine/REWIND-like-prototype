"""Build the independent default-off v2 change scope notebook."""
import argparse
from pathlib import Path
import tempfile

from rewind_archive.contracts import canonical
from scripts.archvis_pilot_bootstrap import extract_pinned_zip
from scripts.build_archvis_pilot_handoff import _cell
from scripts.build_v2_metadata_fence import notebook as fence_notebook
from scripts.check_v2_metadata_fence import ZIP_SHA
from scripts.check_v2_metadata_scope import RECOVERY_ARTIFACT
from scripts.resume_v2_drive_metadata import sha_file


def notebook(pins):
    doc = fence_notebook(pins)
    doc["cells"][0] = _cell("markdown", "guide",
        "# v2 변경 범위 검증과 RGB·pose 대응 확인\n\n"
        "04:18:37 UTC 원본 results.zip과 이 배포본의 source ZIP을 /content에 준비합니다. "
        "저장된 변경 토큰의 모든 페이지를 대조한 뒤, 승인된 폴더 삭제 한 건과 외부임이 입증된 변경만 있으면 "
        "같은 실행에서 SQLite 대응 검사를 진행합니다. 결과 회수 폴더는 조회 종료 후에만 씁니다.\n")
    code = "".join(doc["cells"][2]["source"])
    code = code.replace("scripts.check_v2_metadata_fence", "scripts.check_v2_metadata_scope")
    code = code.replace("v2_metadata_changes_20261119", "v2_metadata_scope_20261119")
    code = code.replace('"v2_metadata_changes_"+stamp', '"v2_metadata_scope_"+stamp')
    anchor = '    call = [sys.executable,"-m","scripts.check_v2_metadata_scope"]'
    if code.count(anchor) != 1:
        raise ValueError("v2_scope_notebook_template")
    code = code.replace(anchor, '''    if PINS["external_evidence_sha256"]:
        common += ["--external-evidence",str(source/"reference/v2_external_scope.json"),
                   "--external-evidence-sha",PINS["external_evidence_sha256"]]
''' + anchor)
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
        for name in ("scripts/check_v2_metadata_scope.py", "scripts/check_v2_metadata_fence.py",
                     "scripts/resume_v2_drive_metadata.py", "reference/spec.json",
                     "reference/split/train.txt", "reference/split/validation.txt", "reference/split/test.txt"):
            if not (src/name).is_file():
                raise ValueError("v2_scope_source_missing:" + name)
        evidence = src/"reference/v2_external_scope.json"
        evidence_sha = sha_file(evidence) if evidence.is_file() else None
        recovery = src/"reference/v2_scope_recovery_043247Z.zip"
        if not recovery.is_file() or recovery.stat().st_size != int(RECOVERY_ARTIFACT["size"]) or sha_file(recovery) != RECOVERY_ARTIFACT["sha256"]:
            raise ValueError("v2_scope_recovery_archive_source_pin")
    pins = {"source_name": args.source.name, "source_sha256": args.source_sha, "revision": args.revision,
            "parent_zip_sha256": ZIP_SHA, "external_evidence_sha256": evidence_sha}
    args.output.mkdir()
    path = args.output/"v2_metadata_scope.ipynb"
    path.write_bytes(canonical(notebook(pins)))
    (args.output/"receipt.json").write_bytes(canonical({"pins": pins, "notebook_sha256": sha_file(path), "default_off": True}))


if __name__ == "__main__":
    main()
