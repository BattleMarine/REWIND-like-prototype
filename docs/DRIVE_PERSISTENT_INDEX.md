# Drive 원본의 지속형 인덱스와 닫힌 snapshot

2026-10-29 구현·모의 검사 후2026-10-30 실제 pose 소량 A→B 저장·복원 시험을 통과했다. 실제 결과와 남은 범위는 아래 결과 절을 따른다. 기존 API10/고정 배포물은 변경하지 않고 `rewind_archive_index`에 선택 adapter를 추가했다. 학습 epoch hook·대규모 row catalog는 이번 범위 밖이다.

## 고정 원본과 런타임 분리

`make_drive_source(archive_manifest)`는 기존 `archive_manifest_v1`의 완전한 Drive metadata 목록만 받는다. Archive/dataset/kind, 조각의 명시 index·file_id·size·version·modified_time과 고정 total bytes를 보존한다. 모든 조각 metadata가 있어야 하며, 다운로드 prefix 크기로 total을 대체하지 않는다. 경로·인증정보·session은 source나 SQLite에 저장하지 않는다.

`ArchiveIndex(path, runtime=DriveRuntime(session, limits))`에 세션을 런타임마다 주입한다. 기존 local source는 그대로 지원한다. Snapshot import/status는 원본에 접속하지 않으며 첫 실제 읽기 전에 metadata를 확인한다. 원본이 변경되면 기존 index와 혼합하지 않고 거부한다.

원격 읽기는 기존 `DriveSource.read_range`의 strict206/Content-Range/길이/encoding·전후 metadata 검사와 hidden retry guard를 재사용한다. 새 HTTP 스택을 만들지 않았다. 기본 재시도0, 256KiB Range·4MiB RAM LRU다. Cache hit도 operation 시작과 저장 전 전체 source metadata를 확인한다. 변경/불량 응답을 cache에 남기지 않는다.

기본 `DriveLimits`는 runtime 전체16MiB 응답 body/128HTTP/60초다. **같은 Runtime의 scan+extract는 하나의 전송 예산을 공유**한다. Operation 통계는 차분, `phase_*`는 누적이다. 마지막 metadata용 HTTP/body/시간 여유를 예약한다. 안전하게 저장할 예산이 없으면 이전 snapshot을 유지한다. Runtime을 임의로 새로 만들면 별도 예산이므로, 고정 pilot helper는 phase당 한 Runtime만 생성하고 run marker로 자동 재실행을 차단한다.

기존 IndexLimits의 read/decoded/RSS/catalog/DB 한도도 함께 적용된다. Operation의 `network_requested`는 실측과 일치하며, offline `status()`의 false는 원격 원본을 현재 재검증했다는 뜻이 아니다. 원격 전체 CRC/SHA 감사는 이번 경로에서 거부하며 catalog 끝을 보더라도 CRC pending/complete=false를 유지한다.

## 저장·복원 계약

`export_snapshot(index, directory)`는 SQLite backup API로 로컬의 일관된 사본을 만든 뒤 연결을 닫는다. Mount에는 닫힌 ZIP과 receipt만 복사한다. Live DB나 staging SQLite를 Drive mount에서 실행하지 않는다. Snapshot API 호출자는 로컬 working DB 경로를 제공해야 한다. Colab pilot helper는 `/content/drive/` 아래의 live DB workspace를 거부하고 notebook은 snapshot 경로의 실제 mount 여부도 확인한다.

ZIP에는 `index.sqlite`와 `manifest.json` 두 항목만 있다. Snapshot format/DB SHA/source fingerprint/generation/catalog·gzip blob hash/engine version/lineage/부모 snapshot ID를 보관한다. Receipt는 ZIP 전체 SHA/bytes와 정확한 파일명을 연결한다. 이름은 불변이며 기존 경로를 덮어쓰지 않는다. Live DB에는 마지막 receipt anchor를 별도 table로 기록하므로 B의 후속 export가 A를 부모로 유지한다. 이 local anchor 기록 때문에 복원된 working DB 전체 bytes는 ZIP 속 DB bytes와 달라질 수 있지만 checkpoint/source/catalog/blob 검증 결과는 유지한다.

Mount copy·readback은 원격 durability 입증이 아니다. Receipt가 먼저 보이거나 파일이 부분 동기화돼도 import는 전체 hash/길이가 맞기 전에는 거부한다. Mount rename/동기화의 원자성을 가정하지 않는다. **실제 새 runtime B의 복원 성공**이 지속 보관 시험의 기준이다.

`import_snapshot`은 사용자가 지정한 receipt만 읽는다. Expected source fingerprint·snapshot SHA·최소 generation을 필수로 대조하고, known anchor가 있으면 lineage·동일 세대의 다른 SHA·stale·직접 부모 관계도 검사한다. 최신 파일명 자동 선택·동시 writer 병합은 없다. Fresh runtime이 모르는 전역 최신/숨은 분기를 판정할 수는 없다. 신뢰할 anchor는 별도로 전달해야 하며 Drive 폴더에 같이 놓인 파일만으로 신뢰를 만들지 않는다.

ZIP central directory를 열기 전에 bounded EOCD/count를 검사하고, 정확한 두 파일·STORED 형식·개별/합산 크기를 확인한다. 임의 경로를 extract하지 않는다. DB/state/catalog/blob checksum·source·generation·schema·실제 engine version을 확인한 다음 새 로컬 경로에만 게시한다. 기존 대상 경로·부분 파일·receipt 혼합·파손·지원 밖 버전을 거부한다.

Snapshot 기본 예산은 DB64MiB/ZIP72MiB, I/O ledger512MiB/임시 disk256MiB/협력적60초다. Backup page 계수는 보수적 상한이고 source archive의 읽기는 포함하지 않는다. Source 원본을 재독하는 작업이 아니다. 로컬 filesystem의 SQLite 원자성과 Drive 복제 원자성은 구분한다.

## Python 진입점

```python
from rewind_archive_index import ArchiveIndex
from rewind_archive_index.drive import make_drive_source, DriveRuntime, DriveLimits
from rewind_archive_index.snapshots import export_snapshot, import_snapshot

spec = make_drive_source(manifest)  # 별도 JSON 입력, 개인 ID 하드코딩 없음
runtime = DriveRuntime(authorized_session, DriveLimits())
index = ArchiveIndex("/content/index.sqlite", runtime=runtime)
index.create(spec)
index.scan(16, verify_end=False)
saved = export_snapshot(index, mounted_snapshot_directory)

# 실제 다른 runtime에서 exact receipt/신뢰 anchor를 전달받는다.
restored = import_snapshot(exact_receipt_path, "/content/restored.sqlite",
    expected_source_sha256=trusted_receipt["source_sha256"],
    expected_snapshot_id=trusted_receipt["snapshot_id"],
    min_generation=trusted_receipt["generation"], known_anchor=trusted_receipt)
next_index = ArchiveIndex(restored["index_path"], runtime=new_runtime)
next_index.scan(16, verify_end=False)
```

이 코드는 구조 예시다. 실제 세션 생성·mount·개인 manifest와 anchor 입력은 아래 고정 notebook에서 명시적으로 실행한다. 기존 local API/CLI 계약과 source 방식은 유지하며 원격 auth를 CLI가 자동 시작하지 않는다.

## CPU A/B 수동 pilot

단일 진입점은 [drive_persistent_index.ipynb](../notebooks/drive_persistent_index.ipynb)다. Root 검토·커밋 후 source ZIP SHA/revision·manifest SHA를 고정한 ready notebook을 별도로 만든다. 현재 tracked notebook의 pin은 비어 있어 그대로 원격 실행하는 배포본이 아니다.

- A: 고정 전체 pose Drive metadata 입력 → 실제 mount 확인 → 명시 auth/session 주입 →16member 저장 → 선택 JSON 최대2개 추출/hash·JSON object 확인 → 닫힌 snapshot A/receipt → 보고서+anchor+선택 JSON ZIP 보존.
- A 보고서/anchor SHA를 별도로 보관하고 명시 gate에서 `drive.flush_and_unmount(timeout_ms=60000)`를 호출한다. 실패하면 현재 runtime과 보고서를 유지한다. Flush 성공만으로 지속성 통과를 주장하지 않는다. [공식 Colab 소스](https://github.com/googlecolab/colabtools/blob/main/google/colab/drive.py)
- 실제 Colab runtime을 삭제하고 새 CPU runtime B를 만든다. 같은 source ZIP/engine1.10.3/manifest를 사용하고, A에서 별도 보관한 anchor의 SHA를 대조한다. B는 exact receipt 복원 → 추가16member → A에서 선택한 JSON hash 대조 → 부모 A인 snapshot B를 보존한다.

Phase당 **합산 응답 body8MiB/120HTTP/300초**이고 두 phase 합최대16MiB/240HTTP/600초다. 인증·mount copy·flush·snapshot 비용은 별도다. IndexLimits도 pilot에서는300초·read32MiB·decoded32MiB로 고정한다. 16member에는 directory도 포함된다. 각 phase에서 new_members16개와 선택 JSON 검증·snapshot까지 완료해야 passed다. 상한 중단·부분 진행·실패는 보고서를 남기고 같은 phase를 자동 반복하지 않는다.

설치·mount·실행·다운로드·flush gate는 기본False다. Source ZIP 오업로드·manifest/anchor pin 불일치·미mount는 실제 auth/scan 전에 거부한다. Helper는 시작 marker를 남겨 완료·실패 뒤에도 같은 local run_id/phase를 소비한 것으로 처리한다. `RUNTIME_ID`는 process 구분이며 Colab VM 교체 증명은 사용자 절차다. Fresh runtime에서 전역 실행 이력을 모두 알 수 있다는 보장은 없다.

보고서에는 source revision/ZIP SHA/manifest SHA, Python/indexed_gzip/psutil 버전, before/after generation·scan·선택 hash·transport 누적·snapshot/receipt/부모와 실패 이유가 들어간다. Raw exception/token/URL은 저장하지 않는다. 작은 결과 ZIP과 별도 flush JSON을 내려받아 테스트 세션에 전달한다. 실제 user 실행/중간 안내는 테스트 세션이 맡는다.

## 개발 검증과 남은 한계

실제 Google/인증/Drive 업로드는 실행하지 않았다. 모의 Drive에서 source metadata 변경·cache warm 변경·Range200·전송 상한·final identity 실패·operation/phase 통계를 검사했다. Snapshot의 새 경로/새 프로세스 복원·기존 hash·부모 A→B·파손/버전/stale/분기·ZIP 경로/크기 거부를 검사했다. 고정 A/B helper의16+16·phase당 단일 Runtime·합산 상한·실패 marker·실제 child process B도 포함한다. Notebook은 code cell 구문과 False gate/미mount 거부를 로컬 모의 검사한다.

실제 Colab A flush/B 복원은 아래 소량 시험으로 확인했다. 대규모 Drive 처리량·재인증·mount 전파 지연의 일반적 보장·실제10GiB split·원본 전체 EOF/CRC·학습 연결은 미검증이다. Source 원본과 기존 API10 cache·배포물을 수정하지 않았으며, 이번 기능이 기존 `archive_extraction_v1` 출력과 자동 호환되거나 M0~M3 완료를 의미하지 않는다. 기존 local JSON catalog 한도도 유지한다.

개발 근거는 Git 제외 `outputs/persistent_drive_development_20261029/`에 보존한다. 실제 manifest/고정 배포물과 사용자 테스트 인계는 root 검토·커밋·별도 hash 고정 후 진행한다.

오케스트레이션 독립 검토에서도 기존13개와 신규14개, 총27개 검사를 실패·건너뜀 없이 통과했다. Notebook7개 code cell 구문, 기본False gate의 무실행, 미mount 상태의 인증 전 거부를 별도로 확인했다. 근거는 `outputs/persistent_drive_review_20261029/root_tests.json`, `root_tests.txt`, `root_notebook.json`에 보존한다. 실제 Drive 원본 읽기나 snapshot 게시는 이 개발 검사에 포함하지 않았다.

## 2026-10-30 실제 pose A/B 시험 결과

고정 revision `86d322c5913b77716f7a0335c70a5dc4c5b079cb`, run_id `pose_persistent_86d322c_001`의 실제 Colab CPU 시험을 통과했다. Python3.13.18/indexed_gzip1.10.3/psutil5.9.5. 기존 런타임 연결 해제·삭제 후 새 CPU 런타임에서 B를 실행했다는 사용자 확인을 테스트 세션이 전달했다. Anchor runtime ID도 다르지만 VM 교체 근거는 사용자 진술과 구분한다.

| 항목 | A | B |
| --- | --- | --- |
| 목록 / generation | 0→16 / 1 | 16→32 / 2 |
| 응답 body / HTTP | 1,049,656bytes / 16회 | 1,049,656bytes / 16회 |
| 원본 읽기 phase 시간 | 약5.924초 | 약3.890초 |
| flush | returned | returned |

원본 읽기 통계이며 인증·mount·snapshot·flush 비용은 별도다. B restore receipt는 A snapshot과 일치하고, B의 부모도 A이며 lineage가 같다. `frame_14.json`, `frame_6.json`의 A/B ZIP 실제 payload hash·크기·JSON 파싱과 anchor bytes hash, revision/source ZIP/manifest pin을 오케스트레이션이 독립 대조했다. 두 phase 모두 지정 예산 이내다.

이번 성공 범위는 A 저장본의 새 B 복원, 기존 파일 재추출, 물리 순서에 따른 목록 누적이다. A/B의 gzip `index_coverage=3791957`, `index_bytes=98411`은 동일하다. 기존 coverage 밖으로 restart point를 추가한 원격 검증은 아니며 B도 앞부분을 읽었으므로 다운로드 절감 효과를 주장하지 않는다. `complete=false`, `train_ready=false`를 유지한다. 원본 전체 EOF/CRC·RGB·실제10GiB 분할 경계·학습·B 저장본의 또 다른 런타임 복원은 미검증이다. Snapshot 원본 ZIP 자체는 로컬로 회수하지 않았으며 snapshot 복원 판정은 고정 코드의 B 실행 보고서에 근거한다.

근거는 Git 제외 `outputs/test_persistent_drive_86d322c_20261029/`의 A/B report ZIP·flush JSON과 `root_review.json`이다. A report ZIP SHA256은 `35a7435d796a21c1b6c405f4b8b1da4f9f7c799ffaf0c71e918fb12540f6171a`, B는 `3313345fff95429625638cb2868d8eafaecaf3307bb4a1816d923f9ad1012d44`다. 기존 고정 배포물과 Drive snapshot/receipt는 보존한다. 같은 시험의 반복은 필요하지 않으며 후속 구현·별도 검증 범위는 다음 작업에서 결정한다.
