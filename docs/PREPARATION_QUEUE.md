# 제한된 RGB·pose 준비 queue

2026-11-02 전체 학습 준비 3a 단계. 기존 `prepare_indexed`를 원본별로 호출하는 작은 영속 queue를 구현했다. **전체 데이터셋 자동 공급·학습 release는 아니다.** 기존 소량 backend 상한 안에서 준비 순서·부분 진행·원본 identity·누적 비용을 보존한다. 실제 Drive/Colab/GPU/학습 검증은 별도다.

## 지원 범위와 파일

`rewind_prepare_queue`는 기존 `rewind_archive_index`와 별도 패키지다. backend 파일과 hash 집합을 바꾸지 않아 과거 고정 archive 결과의 코드 identity를 유지한다. 핵심 경로는 `make_plan` → `initialize_queue` → `run_queue`이며, CLI도 동일 함수를 사용한다. 모델·화면·학습 코드를 import하거나 optimizer를 실행하지 않는다.

고정 plan(`bounded_preparation_queue_v1`)은 공통 dataset version, 순서 있는 최대16job, 전역 예산, 기존 backend Limits를 포함한다. 각 job은 다음을 명시한다.

- 고유 ID, 원본 archive manifest와 조각의 고정 순서·identity, 원본 source spec SHA
- 상대 `source_base`와 최대32개의 정확한 archive member target
- `batch_members`(1~16), `max_batches`(1~8), gzip spacing(64KiB~16MiB)

로컬은 모든 조각이 존재해야 하며 기존 source spec의 절대 경로·size·mtime·device/inode identity를 고정한다. source spec SHA는 **원본 바이트 전체 SHA가 아니다.** Drive는 전체 조각의 명시 metadata/version을 가진 manifest와 호출자가 주입한 `authorized_session`이 필요하다. CLI는 session을 만들거나 원격 인증·요청을 시작하지 않으므로 로컬 실행용이다. Python API의 Drive 경로는 상한 전달 mock만 검증했고 실제 원격 실행은 미검증이다.

plan 생성 시 각 target을 sequence/frame/view로 해석해 중복 view, 좌우·pose 누락, dataset version 혼합을 거부한다. 원본 RGB/pose archive의 순서가 같다고 가정하지 않는다. 최종 준비본도 파일 hash와 정확한 target 집합을 검증한 후 동일 키의 left/right/pose로 대조한다. 파일명 키가 맞는 것과 실제 시간·내용 정합성이 맞는 것은 다르다. 결과에 시간·RGB-pose 내용 정합성을 `unverified`, archive coverage를 `not_measured`, `train_ready=false`로 기록한다. split·원본 motion·좌표/단위·calibration의 학습 release를 발급하지 않는다.

queue root에는 다음을 보존한다.

| 경로 | 용도 |
| --- | --- |
| `plan.json` | 초기화 시 고정한 plan |
| `state.json` | cursor·job 상태·예산·진행 중 예약·최근64event·소비자 표시 |
| `writer.lock` | 단일 host의 OS 잠금 파일. 프로세스 종료 후 OS 잠금만 해제되고 파일은 유지 |
| `indexes/<source SHA>/` | 원본별 로컬 SQLite와 요청 기록 |
| `snapshots/` | 원본별 닫힌 snapshot·HEAD·이전 HEAD 보존본 |
| `attempts/<번호>_<job ID>/` | 시도별 추출본과 extraction manifest. 이전 partial 결과도 보존 |
| `matching.json` | 준비 완료 파일의 sequence/frame/view 및 payload hash 대조 결과 |

live queue/index root를 `/content/drive/`에 두지 않는다. 현재 queue의 snapshot도 root 안에 있으며 Drive 영구 저장 복사·새 VM 복원은 자동 연결하지 않았다. 같은 host의 새 프로세스 재개와 원격 VM 지속성을 구분한다. OS 잠금은 단일 host용이며 분산 writer를 보장하지 않는다.

## 예산 예약과 중단

전역 최대값은 원본 읽기32MiB, 반환 해제32MiB, 신규 catalog member128, 응답body8MiB, HTTP120회, 준비/검사 phase 누적300초, root 디스크128MiB다. 더 작은 plan 예산과 backend 상한이 있으면 작은 값을 사용한다. backend의 파일·RAM·상태·인덱스 상한도 그대로다. 기본 backend disk16MiB를 유지하면 root 전체를 검사하는 그 상한이128MiB보다 먼저 적용될 수 있다.

서로 다른 archive는 동일 `PreparationContext`에 합치지 않는다. source별 새 context를 만들고, queue 장부가 남은 전역 예산을 계산해 **실제 `Resources.index_limits`의 read/decode/time, effective `Limits`의 member/HTTP/body/time을 제한**한다. DriveRuntime도 그 작은 Limits를 전달받는다. 장부만 줄이고 backend가 기존 큰 한도를 쓰는 구조가 아니다.

실행 전 worst-case 비용을 `charged`에 영속 예약하고 `inflight`에 남긴다. backend가 정상 또는 계측 가능한 partial로 반환하면 실제 read/decode/member/전송/시간을 정산해 `observed`에 누적한다. 모든 phase가 정상 정산된 상태에서는 `charged == observed`다. 로컬 준비는 HTTP/body 비용0이다. 재개해도 장부를 초기화하지 않는다.

Python 예외, backend failed/interrupted 또는 cleanup 실패, 관측값 초과, 프로세스 강제 종료로 비용을 입증하지 못하면 예약을 환불하지 않는다. `unknown_cost_blocked`와 진행 중 예약을 보존하고 자동 재시도하지 않는다. 실패 시도에서 부분적으로 관측한 값은 job event에 남을 수 있지만 전역 `observed`는 정상 정산 phase의 합이며, 미해결 예약을 실측으로 표현하지 않는다. 근거 정산 없는 reset/reconcile API는 제공하지 않는다. 수동 복구가 필요하면 원본 기록·예약·snapshot을 보존한 상태에서 검토해야 한다.

완료본의 재사용 해시 검사, root 파일 inventory, 최종 matching도 별도 시간 예약을 거친다. 파일 inventory·완료 output·matching 입력/entry 경계에서 cooperative 시간을 검사한다. 개별 파일 hash, gzip 작업, filesystem 호출, 원자적 JSON 저장 중 강제 hard timeout은 제공하지 않는다. phase가 상한을 넘으면 시간 예약을 유지하고 차단한다. 초기 plan/schema·로컬 원본 stat 확인과 장부 게시 자체의 시간은 phase 계측 밖이다. 이 제한을 전체 프로세스 wall-clock hard limit로 해석하지 않는다.

partial의 명시적 retry를 거부하는 실행도 disk inventory를 검사하므로 시간은 증가할 수 있다. read/decode/member/HTTP/body는 준비를 새로 호출하지 않으면 증가하지 않는다. queue 성공·학습 epoch와 archive 전체 indexing/CRC 완료는 별개다.

## 상태 전이와 보존

`init`은 존재하지 않는 새 root만 만든다. `run`/`inspect`가 `state.json` 누락을 새 장부로 재생성하지 않는다. 잃어버린 state로 기존 시도의 예산을 초기화할 수 없으며, pin된 plan이 바뀌어도 거부한다. 다른 root의 새 `init`은 별도 예산의 새 작업이며 기존 queue 재개가 아니다. 이전 실행 기록을 지우고 같은 작업을 자동 반복하는 용도로 사용하지 않는다.

| 결과 | cursor와 다음 동작 |
| --- | --- |
| `bounded_job_stop` | 이번 `--max-jobs`까지 준비. 다음 실행이 완료본을 검증하고 다음 job부터 진행 |
| `partial` | 현 job에서 멈추고 기존 출력·index·snapshot 보존 |
| `explicit_retry_required` | partial 뒤 `--retry` 없는 재호출을 거부. 새 원본 읽기 없음 |
| 명시 `--retry` | 같은 원본 index에서 이어 탐색하고 새 attempt 출력 사용. 예산은 이전부터 누적 |
| `budget_exhausted` | 남은 읽기/해제/member/전송 예산이 없어 새 준비를 시작하지 않음 |
| `unknown_cost_blocked` | 미해결 예약 유지. crash/예외/검사 실패 기록을 보존하고 재시도 차단 |
| `prepared_not_training_release` | 모든 명시 job의 준비·키 대조만 완료. 실제 학습 허용 판정 아님 |

disk 상한을 넘으면 실패 reason을 장부에 남기며 아무 파일도 자동 삭제하지 않는다. 원본·이전 attempt·index·snapshot·활성 checkpoint를 삭제하는 기능은 없다. `mark_consumed(plan, root, job_id, consumer_artifact_sha256)`와 `consumed` CLI는 명시한 소비자 artifact ID만 기록하고, 실제 학습 완료나 artifact 존재를 독립 검증하지 않는다. 소비 표시가 있어도 삭제하지 않는다. 대규모 순환 저장·eviction·장시간 학습 공급은 후속 설계다.

## CLI

job list JSON은 plan과 같은 job 항목 중 `source_sha256`만 생략할 수 있다. `plan` 명령이 기존 `_source` 경로로 spec SHA를 고정한다. `--base`는 각 `source_base`의 기준 디렉터리다. 아래는 로컬 합성 자료로 실행 검증한 형태이며 새 경로를 사용한다.

```powershell
python -m rewind_prepare_queue plan --jobs outputs/queue_example/jobs.json --base outputs/queue_example --output outputs/queue_example/plan.json
python -m rewind_prepare_queue init --plan outputs/queue_example/plan.json --root outputs/queue_example/runtime
python -m rewind_prepare_queue run --plan outputs/queue_example/plan.json --base outputs/queue_example --root outputs/queue_example/runtime --max-jobs 1
python -m rewind_prepare_queue run --plan outputs/queue_example/plan.json --base outputs/queue_example --root outputs/queue_example/runtime --retry
python -m rewind_prepare_queue inspect --plan outputs/queue_example/plan.json --root outputs/queue_example/runtime
python -m rewind_prepare_queue consumed --plan outputs/queue_example/plan.json --root outputs/queue_example/runtime --job left --consumer-sha256 <64자리 실제 소비자 artifact SHA>
```

`inspect`는 장부·선언 표시이며 payload hash 검사나 원본 무결성 검증이 아니다. `run`이 이전 완료본 payload를 다시 검증한다. CLI 성공 exit code는 명령의 상태 기록 성공을 뜻하고 `prepared_not_training_release`, `partial`, `unknown_cost_blocked` 등 결과의 `status`를 함께 확인해야 한다. 파일 검사 예외는 0이 아닌 종료로 전달될 수 있으며 관련 상태와 이유는 보존한다.

## 개발 근거와 후속 작업

근거는 `outputs/full_training_readiness_20261102/developer/queue/`에 보존한다. 기존 `outputs/rgb_index_probe_20261029/deps`의 isolated indexed_gzip과 기존 psutil을 사용했고 설치하지 않았다. `verify.py`가 로컬 합성 분할gzip 세 원본을 만들고 partial→명시 retry→동일 index 재개→stereo/pose 준비→재사용→소비 표시까지 실행한다. 원본 조각의 전후 전체 hash 대조는 이 작은 합성 fixture에만 적용하며 실제 전체 archive 무결성으로 보고하지 않는다.

검사는 queue 순서, partial 새 출력·index 재개, 실제 backend 한도 축소, Python 실패 예약 유지, `os._exit(9)` 후 새 프로세스의 재시도 차단, 정상 새 프로세스 완료, payload/원본/plan 변경 거부, state 누락 초기화 거부, disk 보존 중단, 단일 writer, CLI를 포함한다. Drive 검사는 원격 요청 직전에 멈추는 주입 mock으로 실제 전송0이며 원격 지원 완료를 뜻하지 않는다.

고정13개가 failure/error/skip0으로 통과했다. 보존 실행은 read5856bytes·decode12756bytes·신규member12·phase7.345초·HTTP0/body0으로 원본3개/키1개의 준비를 마쳤다. 두 번째 완료 실행에서는 read/decode/member/HTTP/body 증가0이고 로컬 재검증 시간만 누적됐다. 실제 원본·학습·전체 archive 검사로 확대 해석하지 않는다.

남은 작업은 RGB+pose shard와 UI 독립2D trainer의 장시간 학습·재개, 근거 artifact 기반 실제 release 계약, 테스트 세션의 Colab Drive/GPU·새 VM 지속성·예산 pilot이다. 데이터 규모 확대나 전체 학습 가능 판정은 이 근거를 받은 뒤 검토한다. [전체 준비 단계](FULL_TRAINING_READINESS.md)를 따른다.
