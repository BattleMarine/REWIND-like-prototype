# 자동 인덱싱과 실제 데이터 준비 연결

2026-11-02 · 기존 API10·prefix cache·고정 A/B/C 시험과 분리된 명시적 opt-in 경로다. 로컬 개발 검증과 실제 Colab 통합 시험을 구분하며, 전체 archive 완료·실제 학습 readiness를 판정하지 않는다.

## 연결한 진입점

`rewind_archive.extraction.prepare(..., persistent_index=options)`와 `python -m rewind_archive extract --persistent-index options.json`에서 실제 target 요청을 받는다. 별도 helper를 사용자가 scan/export/import 순서로 수동 호출할 필요가 없다. 기본 `persistent_index=None`이면 기존 경로를 그대로 사용한다. API10 workflow·기존 notebook·학습 epoch에는 자동 실행 hook을 추가하지 않았다.

지원 범위는 완전한 ordered metadata가 있는 단일 RGB 또는 pose archive, 정확한 literal member 목록이다. Local split과 주입된 authorized session의 Drive split을 지원하며 HTTP source·불완전한 조각 목록·glob discovery·학습 shuffle은 이 경로에서 처리하지 않는다. RGB/pose 대응은 계속 sequence/frame 키로 검증해야 한다.

```python
from rewind_archive.contracts import Limits
from rewind_archive.extraction import prepare
from rewind_archive_index.preparation import PreparationContext

limits = Limits()
context = PreparationContext(limits, authorized_session=session)  # 로컬이면 session 생략
options = {
    "workspace": "/content/index-live",             # live SQLite는 로컬만
    "snapshot_directory": "/content/drive/MyDrive/new-index-store",
    "input_snapshot_directory": "/content/drive/MyDrive/pinned-B",
    "anchor": trusted_receipt,                     # 최초 생성이면 생략
    "batch_members": 16, "max_batches": 8, "spacing": 1048576,
    "request_id": "prepare_001",
}
report = prepare(manifest, "/content/selected", names, limits,
                 exact_targets=names, persistent_index=options,
                 persistent_context=context)
```

여러 준비 요청·새 workspace 복원까지 한 시험 예산을 공유하려면 같은 `PreparationContext`와 동일한 Limits를 계속 전달한다. Context는 archive identity를 바꾸거나 Runtime·전송량·read/decode ledger를 리셋하지 않는다. Context를 생략하면 단일 prepare 요청에 새 예산을 부여한다. 함수는 인증을 시작하거나 caller session을 닫지 않는다. Drive CLI 자동 로그인은 제공하지 않으며 Colab에서 session을 주입한다.

CLI는 manifest·limits·options JSON, 동일한 `--include`/`--exact-target`을 받는다. `scan`/`pilot` 명령에는 opt-in 옵션을 제공하지 않는다. 기존 profile/cache/request policy·외부 clock/sleeper와 혼합하면 명시적으로 거부한다. 실제 동작은 새 로컬 검사의 CLI 경유 사례로 검증한다.

## 복원·제한 탐색·저장·추출

Source의 명시적 조각 순서·원본 ID/version/수정시각/전체 길이를 fingerprint로 고정한다. 로컬 파일은 경로와 stat identity도 고정하므로 다른 기기로 옮긴 raw path를 동일 source라고 간주하지 않는다. Workspace 안의 fingerprint별 SQLite를 읽고, 없으면 해당 source의 정확한 HEAD receipt로 닫힌 snapshot을 복원한다. 초기 B/C receipt는 신뢰 입력으로 전달하며 입력 snapshot 폴더는 읽기만 한다.

먼저 target이 목록에 있는지 검사한다. 없으면16member 이하 batch를 최대8회까지 탐색하고, 전진한 checkpoint마다 자동으로 닫힌 snapshot을 저장한다. 디렉터리도 새 member 한도에 포함한다. Target이 모두 확인되면 즉시 탐색을 멈추고 저장된 gzip index를 `auto_build=False`로 사용해 선택 추출한다. 순서는 tar 물리 순서이며 요청 목록·frame 정렬·학습 shuffle과 무관하다.

출력은 기존 `extraction_manifest.json`/`files/...` 형식이다. `verify_extraction`, `match_extractions`, 기존 pose converter가 읽을 수 있고 성공은 `failure_class=bounded_stop`을 사용한다. Partial은 기존 verifier가 아는 benign stop으로 정규화하며 원래 원인은 `persistent_index.stop_reason`에 남긴다. Manifest에는 backend별 Python 파일 hash도 기록하고 verifier가 대조한다. 기존 `allow_code_change=True`의 명시적 선택 규칙은 유지하며 자동 재사용에서 켜지 않는다.

같은 출력에 같은 target 요청을 하면 manifest/source/code hash·파일 크기·내용 hash를 검증해 재사용한다. 원본 요청이나 DB 읽기는 하지 않는다. 반환의 `request_stats`는 이번 재사용의 read/decode/네트워크0을 표시하며 기존 `stats`는 원래 준비 이력이다. 이 결과는 고정 manifest 기준의 로컬 추출본 검증이며 현재 원격 원본의 live 재확인은 아니다. 변조·다른 target·불완전 출력은 덮어쓰거나 자동 보수하지 않고 거부한다.

## caller 예산과 중단

- 원격 body는 caller `max_transfer_bytes`와8MiB 중 작은 값, HTTP는 `max_ranges`와120 중 작은 값이다. 별도 source read는32MiB이며 body와 구분한다.
- 반환 decode는 caller `max_decompressed_bytes`와32MiB 중 작은 값, 전체 시간은 `max_elapsed_seconds`와300초 중 작은 값이다. Restore·snapshot·검증 준비 시간이 포함되고 인증은 Context 생성 전이다.
- 새 scan member는 caller `max_members`와128 중 작은 값이다. 기존 catalog의 전체 개수와 새 탐색 허용량을 구분한다. Caller `max_files`·`max_file_bytes`·`max_path_bytes`도 적용한다.
- 전체 local workspace·누적 새 snapshot·추출본과 보고서/임시 snapshot 예약은 caller `max_disk_bytes`와128MiB 중 작은 값에 제한한다. State/blob 상한도 그 디스크 한도의1/8 이하로 낮춘다. 입력으로 보존한 B/C 원본 저장소는 새 출력 누적량과 구분한다. RSS512MiB, 저장소 목록10,000항목의 제한도 있다.
- Core의 tar metadata 읽기와64KiB I/O chunk 계약 때문에 `max_metadata_bytes < 64KiB` 또는 다른 `io_chunk_bytes`는 지원한다고 가장하지 않고 실행 전에 거부한다.

이 한도는 Python I/O 경계의 협력적 검사이며 OS hard cap이나 native 내부 해제량의 완전한 계측이 아니다. Target 미발견·EOF·no-progress·batch/예산 도달은 partial로 보존한다. 원본 변경·checksum·CAS·lineage 오류는 failed다. BaseException은 종류·발생 단계만 기록하고 재전파하며 원시 message/token과 사용자 취소 의도는 기록하지 않는다. Cleanup 오류는 최초 중단을 대체하지 않는다. Hard kill/디스크 소실의 완전한 기록은 보장하지 않는다.

## HEAD·동시 writer·실패 보존의 한계

저장소의 `<source fingerprint>-head.json`은 최대64개 receipt의 부모·lineage·세대·source와 자체 hash를 검증한다. 디렉터리의 최신 ZIP을 추측하거나 서로 다른 gzip 인덱스를 결합하지 않는다. 신뢰 anchor가 제공되면 해당 receipt가 검증된 연결 안에 있어야 한다. Local generation CAS와 snapshot anchor를 검사하고 HEAD를 갱신하기 직전 다시 대조한다.

지원 운영은 **단일 writer와 신뢰한 저장소**다. `.lock`은 겹친 실행/미정리 중단을 거부하는 표식이며 Drive mount의 분산 atomic lock·전역 최신 writer·보지 못한 branch를 증명하지 않는다. 신뢰한 시작 anchor 이후의 HEAD를 자동 관리하지만 외부의 HEAD rollback을 새 VM에서 모두 검출하는 기능은 아니다. 임의로 lock을 지우고 동시 실행하지 않는다.

Snapshot 게시 후 HEAD 저장에 실패하면 이전 HEAD·게시 snapshot·local checkpoint·실패 기록을 모두 유지한다. 다음 요청의 local anchor/HEAD 불일치는 자동 병합하지 않고 거부한다. 검토자가 게시 receipt와 소스/부모를 확인한 뒤 새 workspace·별도 새 저장소·명시한 신뢰 receipt로 복구할 수 있다. 이 수동 복구 경계를 자동 최신 선택으로 대체하지 않는다. 실패/partial 출력에 이어 쓰지 않고, 새 출력/request ID의 같은 target 준비 요청으로 저장된 정상 HEAD부터 재개한다.

## 한 번의 Colab 통합 시험과 회수

[archive_prepare_indexed.ipynb](../notebooks/archive_prepare_indexed.ipynb)는 별도 opt-in 진입점이다. Root가 source ZIP/revision·manifest·B anchor SHA, 검증된 target3개 hash와 신규 run/output을 고정한다. 모든 gate는False이고 실제 mount를 인증/marker 전에 검사한다. 기존 업로드·pin 검증·실행1회·REPORT 없는 독립 회수·별도 다운로드/flush 패턴을 유지한다.

한 번의 실행은 기존 B를 입력으로 prepare 요청→자동 확장/저장/JSON3개 검증→동일 Context에서 cache만 비우고 새 local workspace의 prepare로 새 HEAD 복원/scan0/재추출→첫 추출본 검증 재사용 순서다. 입력 B generation·source, coverage/boundary 전진, 새 복원 receipt=첫 HEAD, 대상 hash3개와 reuse HTTP/read/decode 증가0을 함께 확인한다. 전체 preset은8MiB/120HTTP/300초/read·decode32MiB, disk32MiB/RSS512MiB다. Snapshot·복원이 ledger 안에 포함되며 여러 호출로 예산을 새로 만들지 않는다.

중단되면 실행 셀을 다시 돌리지 않고 회수 셀만 실행한다. 회수는 고정 run/provenance와 일치하는 marker/요청 보고서·기존 extraction manifest·검증된 JSON만8항목/총20MiB 이내 ZIP에 포함한다. 인증·원본·DB는 읽지 않고, 기존 ZIP은 목록·크기·hash 검증 후 재다운로드한다. 누락/손상 marker는 성공 대신 진단으로 표시한다. 실패에서 마지막 정상 저장본을 보존하며 자동 학습·예산 확대·추가 원격 시험은 하지 않는다.

개발 검증 당시 실제 Colab 통합 시험은 미실행이었고, 최초 실제 실행의 실패와 보완은 아래 후속 절에 기록한다. 로컬 합성 검증은 실제 source identity·Drive latency·새 VM 지속성 검증을 대체하지 않는다. 모든 결과는 `train_ready=False`, converter의 `training_release=not_implemented`를 유지한다.

## 2026-11-02 개발 검증과 남은 경계

신규 prepare12개·통합 preset/회수5개와 기존 C recovery10개, 총27개 검사가 실패·오류·건너뜀 없이 통과했다. 실제 prepare와 CLI→기존 verifier/matching/converter, 자동 저장과 새 workspace 복원, caller member/disk/file/metadata 한도, shared Runtime/read 예산, no-progress/EOF, source/HEAD 변조·writer 거부, 게시 뒤 HEAD 실패 보존, 원래 중단 예외 유지, partial/중단의 REPORT 없는 회수와 ZIP 재사용을 포함한다. 기존 `allow_code_change`와 학습 차단도 유지했다.

Notebook8개 code cell 구문·기본False gate·미mount 인증 전 거부·REPORT 없는 ZIP 생성/재다운로드·중단 상태 flush 거부를 별도로 확인했다. 개발 근거는 `outputs/automatic_index_development_20261102/{tests.json,tests.txt,notebook.json,code_hashes.json}`에 보존한다. 기존99개 회귀는 오케스트레이션이 수행했으며 sandbox loopback 제한으로 실패한 HTTP 사례는 localhost 허용 환경에서 별도 재검증했다. 이는 외부 Google 요청이 아니다.

오케스트레이션의 독립 재생은 보존 pose cache 앞8MiB의 chunk/hash를 대조한 뒤 유한 mock transport로 B32member를 재구성했다. 자동80member/3batch, 새 workspace scan0 복원과 reuse read0을 확인했고 mock52HTTP/응답body3,673,094bytes로 preset 이내였다. 근거는 `outputs/automatic_index_review_20261102/offline-automatic/review.json`이다. 실제 사용자 B snapshot 자체의 검증이나 Drive 지연·전체 archive·새 VM 내구성 검증을 대체하지 않는다. 추가 원본 접근·실제 Colab 통합 시험·학습·commit/push는 개발 세션에서 수행하지 않았다.

## 2026-11-02 첫 실제 자동 준비 실패와 mount HEAD 보완

고정 `0da5ac5`의 첫 Colab 자동 준비는 실패했다. 수령 ZIP은 `outputs/test_automatic_index_0da5ac5_20261102/pose_auto_0da5ac5_001-auto-report.zip`, 실제 SHA256은 `41e1c879310bdcc5eb5490489d4e4694e4fdedfc844b89ce52a0be775492022b`다. ZIP CRC와 내부 목록을 확인했다. B 복원 결과와 before/after generation2·32member·coverage3,791,957, 빈 scans만 남았고 HEAD나 새 snapshot 결과는 없다. 실패는 `failure_stage=restore`, `stop_reason=index_preparation_failed`이며 transport는 미기록/unknown이다. 원본 변경이나 실제 원본 요청0을 입증하지 않는다.

코드상 context/index 초기화와 before 조회 뒤 첫 `publish()`에서 초기 HEAD를 만들기 전/중의 형태와 일치한다. 기존 local `atomic_record(initial=True)`는 POSIX에서 `os.link`를 호출했고 HEAD temp와 target은 Drive mount에 있었다. Snapshot의 `_publish_file` hard link는 **로컬 DB 복원**에만 쓰이며 Drive snapshot export는 이미 exclusive write와 hash 대조를 사용한다. 기존 C 성공과 이번 HEAD 실패 가설은 양립한다.

고정 `0da5ac5` 소스를 로드한 합성 B(generation2·8member, 실제 수령 B가 아님)에서 HEAD hard link의 EPERM1/ENOTSUP95를 주입하니 같은 제어 흐름 기록을 재현했다. 임시 파일 생성 EACCES13도 같은 형태를 만들었다. 실제 보고서에 최초 예외 class/errno가 없어 **hard link 가설을 재현했지만 실제 원인은 확정하지 못했다.** 재현 중 원본 archive source 호출·실제 Google/auth는0이고 수령 ZIP SHA는 전후 동일했다. 근거는 `outputs/automatic_index_mount_diagnostics_20261102/{reproduce.py,reproduction.json}`이다.

보완은 mount의 HEAD 게시만 `head_storage.publish_head`로 분리한다. 로컬 marker의 `atomic_record`, snapshot core, 기존 실패 자료·B/C 저장본은 유지한다. 최초 HEAD는 exclusive create로 쓴 뒤 닫고 전체 JSON/hash와 읽은 내용을 확인한다. Hard link나 mount fsync 지원을 요구하지 않는다.

갱신 때는 검증된 기존 HEAD의 원시 bytes를 hash 이름의 별도 `*-previous-<sha>.json`에 exclusive 저장·재검증한다. 새 JSON도 별도 staging에 완전히 쓰고 닫은 뒤 CAS를 다시 대조하고 `replace`한다. 게시 후 전체 JSON·checksum·source/receipt chain을 다시 검사해야 성공이다. 이미 있는 backup은 bytes가 같을 때만 재사용하고 덮어쓰지 않는다. 초기 생성 충돌·stale HEAD·불완전 JSON·손상 backup·게시 hash 불일치는 거부한다.

이 방식은 단일 writer용이며 Drive rename의 atomicity·내구성을 주장하지 않는다. Rename 자체가 미지원이면 이전 HEAD/backup/닫힌 staging을 보존한 실패로 제한한다. 부분 쓰기/replace로 HEAD가 손상됐으면 이를 정상 HEAD로 읽지 않고 자동 rollback·branch 선택도 하지 않는다. 이전 bytes는 별도 검증한 backup에 남으며, mount 동기화나 런타임 상실 뒤 영구 보존까지 입증한 것은 아니다. 게시 결과도 `mount_atomicity_claimed=False`, `durability_proven=False`로 기록한다.

`head_publish_pre`, `head_check_pre`, `head_create_pre`, `head_backup_pre`, `head_stage_pre`, `head_cas_pre`, `head_replace_pre`, `head_verify_pre`, `head_publish_post`와 snapshot 전 단계를 고정해 기록한다. 실패에는 안전한 `failure.type`, 정수 또는null의 `failure.errno`, `failure.operation`을 남긴다. 원시 message·URL·token은 저장하지 않는다. 기존 실패/중단 기록과 예외 재전파는 유지한다.

신규 HEAD8개와 기존 준비12개·통합5개·C 회수10개, 총35개 로컬 검사가 실패·오류·건너뜀 없이 통과했다. EPERM/ENOTSUP형 hard link·fsync 미지원, 초기/갱신 충돌, 부분 HEAD 거부, 미지원 replace와 이전본/닫힌 staging 보존, 손상 backup 거부, safe errno/단계 기록과 기존 연결·회수를 확인했다. 근거는 `outputs/automatic_index_mount_diagnostics_20261102/{tests.json,tests.txt,code_hashes.json}`이다. 기존 예산·성공 기준을 확대하지 않았으며 실제 Drive에서 수정본의 지원 동작은 아직 미검증이다. 새 배포·사용자 재실행 안내·commit/push는 개발 세션에서 수행하지 않았다.
