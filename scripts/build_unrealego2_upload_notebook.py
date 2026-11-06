"""Build/check the portable Colab notebook from the reviewed Python modules."""
import argparse
import base64
import hashlib
import json
import textwrap
import zlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks/unrealego2_drive_upload.ipynb"


def build():
    cells = []

    def cell(kind, name, src):
        entry = {"cell_type": kind, "id": name, "metadata": {},
                 "source": textwrap.dedent(src).strip().splitlines(keepends=True)}
        if kind == "code":
            entry.update(execution_count=None, outputs=[])
        cells.append(entry)

    def md(name, src):
        cell("markdown", name, src)

    def code(name, src):
        cell("code", name, src)

    md("intro", """
        # UnrealEgo2 · Drive 업로드 검사와 복구
        Colab CPU에서 위에서 아래로 실행합니다. 먼저 범위를 골라 읽기 전용 보고서를 확인하세요.
        **대상은 train_val/test × RGB/pose 네 폴더입니다.** 한 실행에서 환경 하나를 읽고 지정한 시퀀스만 임시 저장합니다.
        공식 split은 시퀀스를, 환경별 배포 스크립트는 분할 조각 수를 제공합니다. 개별 파일의 기준은 조각을 순서대로 풀어 확인한 완전한 tar 멤버입니다.
        [공식 배포 안내](https://unrealego.mpi-inf.mpg.de/#download)
    """)
    source = {}
    for name in ("unrealego2_upload_source", "unrealego2_upload_drive", "unrealego2_upload_survey"):
        source[name + ".py"] = (ROOT / "scripts" / (name + ".py")).read_text(encoding="utf-8")
    rows = json.loads((ROOT / "configs/fleet_inventory_observed.json").read_text(encoding="utf-8"))["archives"]
    registry = []
    for row in rows:
        if row["kind"] not in {"rgb", "pose"}:
            continue
        env, count = row["name"], row["official_part_count"]
        registry.append({"category": row["category"], "name": env, "official_part_count": count,
                         "official_count_source": row["official_count_source"],
                         "parts": [{"index": i, "name": f"{env}.tar.gz{i:03}"} for i in range(count)]})
    source["registry.json"] = json.dumps({"archives": registry}, sort_keys=True, separators=(",", ":"))
    payload = json.dumps(source, sort_keys=True, ensure_ascii=False).encode("utf-8")
    b85 = base64.b85encode(zlib.compress(payload, 9)).decode("ascii")
    bootstrap = """
        # 접힌 셀에 검토한 Python 모듈과 공개 조각 목록을 포함합니다.
        import base64, hashlib, json, sys, tempfile, zlib
        from pathlib import Path
        TOOLS = Path(tempfile.mkdtemp(prefix="unrealego2-tools-"))
        payload = zlib.decompress(base64.b85decode(EMBEDDED))
        if hashlib.sha256(payload).hexdigest() != EXPECTED_HASH:
            raise ValueError("내장 도구 해시 불일치")
        assets = json.loads(payload)
        for filename in ("unrealego2_upload_source.py", "unrealego2_upload_drive.py", "unrealego2_upload_survey.py", "registry.json"):
            (TOOLS / filename).write_text(assets[filename], encoding="utf-8")
        sys.path.insert(0, str(TOOLS))
        for module in ("unrealego2_upload_source", "unrealego2_upload_drive", "unrealego2_upload_survey"):
            sys.modules.pop(module, None)
        from unrealego2_upload_source import scan_source
        from unrealego2_upload_drive import audit, repair
        from unrealego2_upload_survey import folder_survey
        REGISTRY = json.loads(assets["registry.json"])["archives"]
        print(f"준비 완료 · {len(REGISTRY)}개 환경별 원본 · RGB/pose")
    """
    code("bootstrap", bootstrap.replace("EMBEDDED", repr(b85)).replace("EXPECTED_HASH", repr(hashlib.sha256(payload).hexdigest())))
    cells[-1]["metadata"]["cellView"] = "form"
    md("setup-help", """
        ## 1. Drive 연결과 입력 설정
        원본은 마운트 경로에서 읽고 업로드된 파일은 Drive API로 검사합니다. **ROOT_FOLDER**에는 네 대상 폴더를 직접 포함하는 상위 폴더의 URL이나 ID를 입력하세요.
        **ARCHIVE_ROOT**는 압축 조각을 담은 상위 폴더의 마운트 경로입니다. 공유 폴더는 내 드라이브에 바로가기를 추가하고 실제 경로를 지정하세요.
        기본 예시는 작은 test pose 환경에서 한 시퀀스를 검사합니다. 전체 환경을 검사하려면 **SEQUENCES = None**으로 설정하세요.
    """)
    code("settings", """
        ROOT_FOLDER = ""  # 예: https://drive.google.com/drive/folders/...
        ARCHIVE_ROOT = "/content/drive/MyDrive/UnrealEgo2"
        CATEGORY = "UnrealEgoData2_test_pose"  # @param ["UnrealEgoData2_train_val_pose","UnrealEgoData2_train_val_rgb","UnrealEgoData2_test_pose","UnrealEgoData2_test_rgb"]
        ENVIRONMENT = "ModularBuilding_Demo_Scene"
        SEQUENCES = ["ModularBuilding_Demo_Scene/Day/rp_carla_rigged_001_ue4/SKM_MenReadingGlasses_Shape_01/020/Drunk_Walk"]
        PARTS_DIR_OVERRIDE = ""  # 조각이 환경 폴더 바로 아래에 있지 않을 때 지정
        GiB = 1024 ** 3
        MAX_SOURCE_GIB = 2     # 마운트에서 읽을 압축 조각 총 크기
        MAX_DECODED_GIB = 16   # 선택 밖까지 포함한 gzip 해제량
        MAX_STAGE_GIB = 2      # 선택한 PNG/JSON을 Colab에 보관할 용량
        MAX_MEMBERS = 100000
        MAX_MANIFEST_MIB = 128
        MAX_SCAN_SECONDS = 7200
        MAX_API_REQUESTS = 50000  # 감사/복구 각 단계의 논리 API 요청
        MAX_API_SECONDS = 7200
        MAX_UPLOAD_GIB = 1
        MAX_WRITES = 5000        # 폴더 생성·업로드·이름 변경 포함
        SPLIT_ZIP_LOCAL = ""    # 비워두면 고정 공식 ZIP(약 122 KiB) 다운로드
    """)
    code("connect", """
        import re, shutil, signal, time, urllib.request
        from urllib.parse import urlparse
        from google.colab import auth, drive, files
        import google.auth, httplib2, google_auth_httplib2
        from googleapiclient.discovery import build
        connected_folder_input = ROOT_FOLDER.strip()
        root_id = connected_folder_input
        if root_id.startswith("https://"):
            url = urlparse(root_id)
            if url.hostname != "drive.google.com" or "/folders/" not in url.path:
                raise ValueError("Drive 상위 폴더 URL이 필요합니다")
            root_id = url.path.rstrip("/").split("/")[-1]
        if not re.fullmatch(r"[A-Za-z0-9_-]+", root_id):
            raise ValueError("1번 설정에서 ROOT_FOLDER를 입력하세요")
        auth.authenticate_user()
        credentials, _ = google.auth.default()
        transport = google_auth_httplib2.AuthorizedHttp(credentials, http=httplib2.Http(timeout=90))
        service = build("drive", "v3", http=transport, cache_discovery=False)
        root = service.files().get(fileId=root_id, fields="id,name,trashed,mimeType,webViewLink", supportsAllDrives=True).execute()
        if root.get("trashed") or root["mimeType"] != "application/vnd.google-apps.folder":
            raise ValueError("사용 가능한 대상 폴더를 선택하세요")
        print("검사 대상 Drive 폴더", root["name"], root.get("webViewLink", root_id))
    """)
    md("survey-help", """
        ## 공식 시퀀스 폴더 조사
        파일을 받기 전에 네 범주의 시퀀스 폴더 유무를 조사할 수 있습니다. 기본값은 선택한 환경이고, **all_four**를 고르면 네 범주 전체를 조사합니다.
        폴더가 있어도 내부 프레임은 일부만 업로드됐을 수 있습니다. 3~4번에서 파일 내용까지 확인하세요.
    """)
    code("survey", """
        SURVEY_SCOPE = "selected"  # @param ["selected","all_four"]
        DOWNLOAD_FOLDER_REPORT = False  # @param {type:"boolean"}
        entries = {(r["category"], r["name"]): r for r in REGISTRY}
        entry = entries[CATEGORY, ENVIRONMENT]
        if SPLIT_ZIP_LOCAL:
            split_zip = Path(SPLIT_ZIP_LOCAL)
        else:
            split_zip = TOOLS / "official_split.zip"
            split_url = "https://unrealego.mpi-inf.mpg.de/data/data_split/UnrealEgoData2.zip"
            with urllib.request.urlopen(split_url, timeout=30) as response:
                split_zip.write_bytes(response.read(1024**2 + 1))
        if split_zip.stat().st_size > 1024**2 or hashlib.sha256(split_zip.read_bytes()).hexdigest() != "40160ff2c2192b5c342a80a1f76b37ea651afc45d1fd99fdc778dd4f18b81171":
            raise ValueError("공식 split ZIP이 검토한 버전과 다릅니다")
        survey = folder_survey(service, root_id, split_zip,
            entries.keys() if SURVEY_SCOPE == "all_four" else [(CATEGORY, ENVIRONMENT)],
            max_requests=MAX_API_REQUESTS, max_seconds=MAX_API_SECONDS)
        survey_path = TOOLS / "sequence-folder-survey.json"
        survey_path.write_text(json.dumps(survey, ensure_ascii=False), encoding="utf-8")
        print("시퀀스 폴더 조사", survey["counts"], "요청", survey["requests"])
        for row in survey["sequences"][:3]:
            print(row)
        if DOWNLOAD_FOLDER_REPORT:
            files.download(str(survey_path))
    """)
    md("preflight-help", """
        ## 2. 조각 목록과 비용 확인
        아래 표는 공식 4개 배포 스크립트의 조각 수와 마운트 경로에서 확인한 파일 수·바이트입니다. 이름을 맞춰 본 결과이며 파일 내용 검사는 다음 셀에서 수행합니다.
        시퀀스 하나를 골라도 해당 환경의 gzip은 끝까지 읽습니다. RGB 환경이 크면 표를 보고 원본 읽기·해제량·임시 디스크 예산을 먼저 조정하세요.
    """)
    code("preflight", """
        import pandas as pd
        drive.mount("/content/drive")
        if not Path(ARCHIVE_ROOT).is_dir():
            raise ValueError("마운트된 ARCHIVE_ROOT 경로를 확인하세요")
        entries = {(r["category"], r["name"]): r for r in REGISTRY}
        entry = entries[CATEGORY, ENVIRONMENT]
        inventory = []
        for (cat, env), row in sorted(entries.items()):
            folder = Path(ARCHIVE_ROOT) / cat / env
            if (cat, env) == (CATEGORY, ENVIRONMENT) and PARTS_DIR_OVERRIDE:
                folder = Path(PARTS_DIR_OVERRIDE)
            wanted = {x["name"] for x in row["parts"]}
            seen = {p.name: p for p in folder.glob(env + ".tar.gz*") if p.is_file()}
            size = sum(p.stat().st_size for p in seen.values())
            inventory.append(dict(category=cat, environment=env, expected=len(wanted), found=len(seen),
                                  missing=sorted(wanted - seen.keys()), extra=sorted(seen.keys() - wanted),
                                  bytes=size, GiB=round(size / GiB, 3)))
            if (cat, env) == (CATEGORY, ENVIRONMENT):
                parts_dir, selected_inventory = folder, inventory[-1]
        display(pd.DataFrame(inventory)[["category","environment","expected","found","GiB"]])
        print("선택한 원본", selected_inventory)
        if selected_inventory["missing"] or selected_inventory["extra"]:
            raise ValueError("압축 조각 이름·수 불일치. 미완료 원본은 검사/복구할 수 없습니다")
        if selected_inventory["bytes"] > MAX_SOURCE_GIB * GiB:
            raise ValueError("선택한 원본 GiB와 읽기 예산을 확인하세요")
        print("임시 디스크 잔여 GiB", round(shutil.disk_usage("/content").free / GiB, 2))
        def scope_key():
            return (ROOT_FOLDER.strip(), CATEGORY, ENVIRONMENT,
                    None if SEQUENCES is None else tuple(SEQUENCES), ARCHIVE_ROOT, PARTS_DIR_OVERRIDE)
        planned_scope = scope_key()
    """)
    md("scan-help", """
        ## 3. 공식 목록과 압축 원본 검사
        시퀀스 목록 ZIP의 SHA256을 확인하고 모든 조각을 순서대로 읽습니다. 정상 tar 종료·gzip CRC와 공식 시퀀스 대응을 확인한 뒤 선택한 파일의 SHA256을 기록합니다.
        원본 조각 자체는 공급자가 개별 해시를 배포한 것으로 가정하지 않습니다. 실패하면 부분 결과와 오류 보고서를 보관하고 복구 셀을 차단합니다.
    """)
    code("scan", """
        source_report = drive_report = repair_result = None
        if planned_scope != scope_key() or connected_folder_input != ROOT_FOLDER.strip():
            raise ValueError("설정이 바뀌었습니다. 연결·목록 확인부터 다시 실행하세요")
        scanned_scope = planned_scope
        run = Path(tempfile.mkdtemp(prefix="ue2-upload-", dir="/content"))
        stage = run / "stage"
        report_dir = run / "reports"
        report_dir.mkdir()
        def save(name, value):
            tmp = report_dir / (name + ".tmp")
            tmp.write_text(json.dumps(value, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
            tmp.replace(report_dir / name)
        save("inventory.json", inventory)
        save("folder_survey.json", survey)
        def timeout_handler(*_):
            raise TimeoutError("source scan wall-time limit")
        def progress(event):
            print(f"원본 {event['compressed_bytes']/GiB:.2f}/{event['total_compressed_bytes']/GiB:.2f} GiB · {event['elapsed_seconds']:.0f}초", end="\\r", flush=True)
        old_handler = signal.signal(signal.SIGALRM, timeout_handler)
        try:
            signal.alarm(MAX_SCAN_SECONDS)
            if split_zip.stat().st_size > 1024**2 or hashlib.sha256(split_zip.read_bytes()).hexdigest() != "40160ff2c2192b5c342a80a1f76b37ea651afc45d1fd99fdc778dd4f18b81171":
                raise ValueError("공식 split ZIP이 검토한 버전과 다릅니다")
            source_report = scan_source(parts_dir, CATEGORY, ENVIRONMENT, split_zip, stage,
                registry_path=TOOLS/"registry.json", selected_sequences=SEQUENCES,
                max_output_bytes=MAX_STAGE_GIB*GiB, max_decompressed_bytes=MAX_DECODED_GIB*GiB,
                max_members=MAX_MEMBERS, max_manifest_bytes=MAX_MANIFEST_MIB*1024**2, reserve_free_bytes=GiB,
                max_compressed_bytes=MAX_SOURCE_GIB*GiB, progress_callback=progress)
            save("source.json", source_report)
            print("파일 수", len(source_report["files"]), "복구 입력 확인", source_report["upload_ready"],
                  "경과 초", round(source_report["elapsed_seconds"], 1))
        except (Exception, KeyboardInterrupt) as exc:
            save("source_error.json", getattr(exc, "report", None) or {"failure": str(exc)})
            source_report = None
            raise
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)
    """)
    md("audit-help", """
        ## 4. 읽기 전용 Drive 대조
        Drive의 파일 크기와 서버 MD5를 비교합니다. 정상·누락·내용 불일치·판정 보류·충돌·추가 파일의 상세 경로는 보고서에 저장합니다. 범위 밖 폴더와 함께 저장된 압축 조각은 그대로 둡니다.
    """)
    code("audit", """
        drive_report = repair_result = None
        def check_stage(source, plan):
            # 두 검사 사이에 로컬 파일이 바뀌거나 추가되면 쓰기 차단.
            remote_input = {ENVIRONMENT + "/" + i["path"]: i for i in plan["expected"]}
            if set(remote_input) != {i["relative_path"] for i in source["files"]}:
                raise ValueError("로컬 staging 파일 목록 변경")
            for i in source["files"]:
                item = remote_input[i["relative_path"]]
                sha, md = hashlib.sha256(), hashlib.md5()
                with Path(item["local_path"]).open("rb") as stream:
                    while piece := stream.read(1024**2):
                        sha.update(piece)
                        md.update(piece)
                if sha.hexdigest() != i["sha256"] or md.hexdigest() != item["md5"] or item["size"] != i["size"]:
                    raise ValueError("로컬 staging 파일 내용 변경: " + i["relative_path"])
        if not source_report or scanned_scope != scope_key() or source_report["status"] != "complete" or not source_report["files"]:
            raise ValueError("먼저 3번 원본 검사를 완료하세요")
        def run_audit():
            return audit(service, root_id, source_report["category"], source_report["environment"], stage,
                         max_files=MAX_MEMBERS, max_requests=MAX_API_REQUESTS, max_seconds=MAX_API_SECONDS)
        def counts(report):
            return {k: len(report[k]) for k in ("verified","missing","corrupt","pending","conflicts","extras")}
        try:
            plan = run_audit()
            check_stage(source_report, plan)
            save("drive_audit.json", plan)
            drive_report = plan
            print(counts(plan))
            for k in ("missing","corrupt","pending","conflicts"):
                print(k, plan[k][:3])
        except (Exception, KeyboardInterrupt) as exc:
            save("audit_error.json", {"failure":str(exc)})
            raise
    """)
    md("repair-help", """
        ## 5. 선택한 범위 복구 또는 묶음 업로드
        **감사만 할 때는 기본값을 유지하세요.** 보고서를 확인하고 해당 범위의 다른 업로드를 멈춘 뒤 쓰기 모드와 확인란을 선택합니다.
        - **loose**는 파일을 원래 경로에 복구합니다. 비어 있는 시퀀스는 임시 폴더에 원래 구조로 한 번씩 전송하고 전체를 검증해 폴더 이름을 한 번 바꿉니다. 일부가 있는 시퀀스는 누락·불일치만 전송하며 잘못된 기존 파일 이름에는 WRONG_UPLOAD 태그가 붙습니다.
        - **bundle**은 차이가 있는 시퀀스를 ZIP 한 개로 업로드합니다. RGB는 같은 motion의 좌우 뷰를 함께 묶습니다. ZIP은 별도 폴더에 저장되며 원래 파일 경로가 필요한 학습이나 복구에는 loose를 사용하세요.
        - 수동 업로드가 진행 중이거나 동명 항목·확인할 수 없는 체크섬이 있으면 검사를 다시 하세요. 중단 후에는 4번을 다시 실행해 남은 항목을 확인하세요.
    """)
    code("repair", """
        MODE = "audit"  # @param ["audit","loose","bundle"]
        OTHER_UPLOADS_PAUSED = False  # @param {type:"boolean"}
        if MODE == "audit":
            print("읽기 전용 검사를 마쳤습니다. 6번에서 보고서를 내려받으세요.")
        else:
            if not OTHER_UPLOADS_PAUSED or not drive_report or not source_report or scanned_scope != scope_key() or not source_report["upload_ready"]:
                raise ValueError("완전한 원본 검사, Drive 대조, 같은 범위 업로드 일시 중지가 필요합니다")
            if CATEGORY not in drive_report["folders"]:
                raise ValueError("선택한 상위 폴더에서 기존 category 폴더를 찾지 못했습니다. ROOT_FOLDER를 다시 확인하세요")
            if (root_id, CATEGORY, ENVIRONMENT) != (drive_report["root_id"], drive_report["category"], drive_report["environment"]):
                raise ValueError("대상 설정이 달라졌습니다. 처음부터 다시 검사하세요")
            try:
                repair_result = repair(service, drive_report, enabled=True, mode=MODE,
                    max_requests=MAX_API_REQUESTS, max_seconds=MAX_API_SECONDS,
                    max_writes=MAX_WRITES, max_upload_bytes=MAX_UPLOAD_GIB*GiB,
                    max_staging_bytes=MAX_STAGE_GIB*GiB, staging_dir=run)
                save("repair.json", repair_result)
                final_audit = run_audit()
                save("drive_after.json", final_audit)
                print("반영한 작업", len(repair_result["operations"]), "원래 경로 재검사", counts(final_audit))
            except (Exception, KeyboardInterrupt) as exc:
                save("repair_error.json", {"failure":str(exc),"mode":MODE,"instruction":"4번 재검사 후 남은 작업 확인"})
                raise
    """)
    md("report-help", """
        ## 6. 보고서 보관
        셀이 중단된 경우에도 실행할 수 있습니다. ZIP에는 실행 범위·원본/Drive 감사·실행 결과 또는 예외를 담습니다. 결과를 보관한 뒤 다른 환경이나 범위를 선택할 수 있습니다. 대용량 staging 공간은 런타임 삭제 시 회수됩니다.
    """)
    code("report", """
        if not report_dir.is_dir():
            raise ValueError("먼저 원본 검사 셀을 실행하세요")
        report_zip = Path(shutil.make_archive(str(run / ("ue2-upload-report-" + run.name)), "zip", report_dir))
        print(report_zip, report_zip.stat().st_size, "bytes")
        files.download(str(report_zip))
    """)
    return {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "name": "python3"},
             "colab": {"name": NOTEBOOK.name, "provenance": []}}, "nbformat": 4, "nbformat_minor": 5}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    rendered = json.dumps(build(), ensure_ascii=False, indent=1) + "\n"
    if args.check:
        if NOTEBOOK.read_text(encoding="utf-8") != rendered:
            parser.error("notebook needs rebuilding")
    else:
        NOTEBOOK.write_text(rendered, encoding="utf-8", newline="\n")
