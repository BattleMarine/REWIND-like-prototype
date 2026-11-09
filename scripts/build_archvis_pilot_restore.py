"""Build an exact completed-preparation ZIP and a separate Colab restore notebook."""

import argparse
import ast
import json
from pathlib import Path
import tempfile
import zipfile

from rewind_archive.contracts import canonical
from rewind_data.audit import read_json
from scripts.archvis_pilot_bootstrap import extract_pinned_zip
from scripts.archvis_pilot_restore import RESTORE_MEMBERS, _restore_read, _restore_sha, verify_prepared
from scripts.build_archvis_pilot_handoff import _cell


MIB = 1024**2


def _pinned(path, receipt, maximum):
    raw = _restore_read(path, maximum)
    if {"sha256": _restore_sha(raw), "bytes": len(raw)} != {k: receipt[k] for k in ("sha256", "bytes")}:
        raise ValueError("restore_original_release_pin")
    return raw


def build_restore(source_zip, input_zip, original_ready, release_receipt, recovered, output):
    source_zip, input_zip, output = map(Path, (source_zip, input_zip, output))
    release = read_json(_restore_read(release_receipt, 64 * 1024))
    revision, files = release["revision"], release["files"]
    _pinned(source_zip, files["source"], 64 * MIB)
    _pinned(input_zip, files["inputs"], 8 * MIB)
    original = read_json(_pinned(original_ready, files["notebook"], MIB))
    cells = {cell["id"]: cell for cell in original["cells"]}
    bootstrap = "".join(cells["bootstrap"]["source"])
    [assignment] = [node for node in ast.parse(bootstrap).body if isinstance(node, ast.Assign)
                    and any(isinstance(target, ast.Name) and target.id == "PINS" for target in node.targets)]
    expected = {"source": {"name": source_zip.name, "sha256": files["source"]["sha256"], "revision": revision},
                "inputs": {"name": input_zip.name, "sha256": files["inputs"]["sha256"]}}
    if ast.literal_eval(assignment.value) != expected:
        raise ValueError("restore_notebook_release_pin")

    recovered = Path(recovered)
    file_pins = {}
    for name in RESTORE_MEMBERS:
        raw = _restore_read(recovered / name)
        file_pins[name] = {"bytes": len(raw), "sha256": _restore_sha(raw)}
    with tempfile.TemporaryDirectory(dir=output.parent, prefix="archvis_restore_check_") as directory:
        work = Path(directory)
        extract_pinned_zip(source_zip, expected["source"]["sha256"], work / "source",
                           max_zip_bytes=64 * MIB, max_output_bytes=192 * MIB, max_members=3000, revision=revision)
        extract_pinned_zip(input_zip, expected["inputs"]["sha256"], work / "inputs",
                           max_zip_bytes=8 * MIB, max_output_bytes=16 * MIB, max_members=32)
        checked, original_raw = verify_prepared(work, recovered, expected_files=file_pins, revision=revision)

    output.mkdir(exist_ok=False)
    restore_zip = output / "archvis_1000_prepared_001.zip"
    with zipfile.ZipFile(restore_zip, "x") as archive:
        for name, raw in original_raw.items():
            entry = zipfile.ZipInfo(name)
            entry.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(entry, raw)
    bundle = {"name": restore_zip.name, "bytes": restore_zip.stat().st_size,
              "sha256": _restore_sha(restore_zip.read_bytes())}
    runtime = Path(__file__).with_name("archvis_pilot_restore.py").read_text(encoding="utf-8")
    wrapper = {"version": "archvis_prepared_restore_v1", "runtime_sha256": _restore_sha(runtime.encode("utf-8")),
               "builder_sha256": _restore_sha(Path(__file__).read_bytes())}
    original_work = 'WORK = Path("/content/archvis_1000_pilot")'
    if bootstrap.count(original_work) != 1:
        raise ValueError("restore_unexpected_original_bootstrap")
    bootstrap = bootstrap.replace(original_work, 'WORK = Path("/content/archvis_1000_pilot_restored")')
    bootstrap += ("\n" + runtime + f"\nRESTORE_ZIP = {bundle!r}\nRESTORE_FILE_PINS = {file_pins!r}\n"
                  + f"RESTORE_WRAPPER = {wrapper!r}\n")
    bootstrap += '''
RESTORE_ROOTS_CONFIRMED = False

def restore_completed_preparation():
    extract_pinned_zip(PACKAGES / RESTORE_ZIP["name"], RESTORE_ZIP["sha256"], WORK / "returned",
                       max_zip_bytes=8*1024**2, max_output_bytes=16*1024**2, max_members=3)
    return restore_prepared(WORK, WORK / "returned", expected_files=RESTORE_FILE_PINS,
                            revision=PINS["source"]["revision"])

try:
    restored = stage("preflight", restore_completed_preparation)
except (Exception, KeyboardInterrupt) as exc:
    pilot._write(WORK / "restore_report.json", {"status": "failed", "failure_type": type(exc).__name__,
        "reason": str(exc)[:512], "restore_wrapper": RESTORE_WRAPPER, "source_revision": PINS["source"]["revision"]})
    raise
pilot._write(WORK / "restore_report.json", {**restored, "restore_wrapper": RESTORE_WRAPPER,
             "prepared_archive_sha256": RESTORE_ZIP["sha256"]})
print(json.dumps(restored, ensure_ascii=False, indent=2))
'''
    mount = '''CONFIRM_ORIGINAL_ROOTS = False

def check_restored_roots():
    return confirm_original_roots(WORK, expected_files=RESTORE_FILE_PINS, revision=PINS["source"]["revision"])

if CONFIRM_ORIGINAL_ROOTS:
    if not Path(restored["mount_root"]).is_dir():
        from google.colab import drive
        drive.mount("/content/drive")
    roots = stage("discover", check_restored_roots)
    pilot._write(WORK / "roots_candidates.json", roots)
    RESTORE_ROOTS_CONFIRMED = True
    print(json.dumps(roots, ensure_ascii=False, indent=2))
else:
    print("표시된 원래 RGB·pose 경로를 확인한 후 이 셀의 경로 확인을 켭니다.")
'''
    budget = "".join(cells["budget"]["source"])
    train = "".join(cells["train"]["source"])
    for label, source, gate in (("budget", budget, "if FREEZE_COMPARISON:\n"),
                                ("train", train, "if APPROVED_COMPARISON_SHA256:\n")):
        if source.count(gate) != 1:
            raise ValueError("restore_unexpected_original_gate:" + label)
    budget = budget.replace("if FREEZE_COMPARISON:\n",
                            'if FREEZE_COMPARISON:\n    _restore_require(RESTORE_ROOTS_CONFIRMED, "restore_roots_confirmation_required")\n'
                            '    stage("budget", check_restored_roots)\n')
    train = train.replace("if APPROVED_COMPARISON_SHA256:\n",
                          'if APPROVED_COMPARISON_SHA256:\n    _restore_require(RESTORE_ROOTS_CONFIRMED, "restore_roots_confirmation_required")\n'
                          '    stage("train", check_restored_roots)\n')
    recover = "".join(cells["recover"]["source"])
    recover_marker = "    recovery = pilot.recover(WORK)\n"
    if recover.count(recover_marker) != 1:
        raise ValueError("restore_unexpected_original_recover")
    recover = recover.replace(recover_marker, recover_marker
                              + '    if (WORK / "restore_report.json").is_file():\n'
                              + '        include_restore_report(recovery, WORK / "restore_report.json")\n')
    original["cells"] = [
        _cell("markdown", "guide", """# ArchVis 1,000페어 준비 결과 복원
새 Colab 런타임에서 사용합니다. 고정 소스 ZIP, 입력 ZIP, 회수 3파일 ZIP을 Colab 파일 영역에 올립니다.
원본 준비 바이트를 검증해 복원한 뒤 표시된 Drive 원래 경로를 확인합니다. 경로 확인은 폴더 존재 검사입니다.
최초 평가와 이후 배치에서는 기존 학습기가 파일 크기, SHA, PNG를 검사합니다.
기존 manifest를 사용해 예산 계획을 고정하고 SHA를 승인하면 MSE·CE 비교를 실행합니다.
마지막 셀에서 정상 또는 실패 결과를 회수합니다. 준비 시간은 학습 처리량으로 환산하지 않습니다.
"""),
        _cell("code", "bootstrap", bootstrap), _cell("code", "original-roots", mount),
        _cell("code", "budget", budget), _cell("code", "train", train), _cell("code", "recover", recover),
    ]
    for cell in original["cells"]:
        if cell["cell_type"] == "code":
            compile("".join(cell["source"]), cell["id"], "exec")
    notebook = output / "archvis_1000_pilot_restore_001.ipynb"
    notebook_raw = (json.dumps(original, ensure_ascii=False, indent=1) + "\n").encode()
    notebook.write_bytes(notebook_raw)
    receipt = {**checked, "prepared_files": file_pins, "original_packages": expected, "restore_wrapper": wrapper,
               "restore_zip": bundle, "restore_notebook": {"name": notebook.name,
                   "sha256": _restore_sha(notebook_raw), "bytes": len(notebook_raw)}}
    (output / "restore_receipt.json").write_bytes(canonical(receipt) + b"\n")
    return receipt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source-zip", "input-zip", "original-ready", "release-receipt", "recovered", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    print(json.dumps(build_restore(**vars(parser.parse_args(argv))), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
