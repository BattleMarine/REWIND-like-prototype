# v2 CPU 캐시 8시간 단일 호출

이 프리셋은 CPU Colab 한 개에서 기본 8 workers, Drive writer 한 개로 실행한다. 호출 전체는 최대 28,800초다. 누적 CPU 90시간·추가 Drive 60GiB, producer identity와 manifest 및 경계·예산 원장은 기존 값과 형식을 사용한다. 기존 [90분 프리셋](V2_BULK_CACHE_90MIN.md)과 [8GiB 프리셋](V2_BULK_CACHE_BUDGETED.md)의 배포본은 보존된다.

## 시작과 회수

1. CPU Colab에서 `v2_bulk_cache_8h_cpu.ipynb`를 열고 기존 `v2_cpu_cache_ed98124.zip`을 `/content`에 올린다. 다른 CPU writer가 완전히 종료됐는지 확인한다.
2. 직전 운영 호출의 최신 정산 또는 예약 SHA-256을 `EXPECTED_BUDGET_TIP_SHA256`에 입력한다. 245샤드 완료 경계 `ce372854fa8dfb7521a9ed4cbe48426cab3832fe6d318a7a044963e77de36bd1`는 이력에서 확인할 최소 기준이다. 최신 경계가 더 앞섰으면 그 다음 인덱스부터 처리한다.
3. 기본 OFF인 `RUN_APPROVED_CACHE`를 켜고 한 번 실행한다. Drive 마운트를 승인하고 `Writer lock`과 `Durable budget reservation SHA-256`이 출력됐는지 확인한다. 초기 마운트와 잠금·예약 확인 뒤 후속 작업은 한 호출에서 진행된다.
4. 진단의 완료 경계 SHA, `budget_tip_sha256`, `time_reserved_seconds`, `time_charged_seconds`, `stop_reason`을 회수한다. 전체 목표에 도달하면 조기에 완료한다. 자동 재시작은 없다. 이후 호출이 필요하면 최신 tip을 사용해 한 번씩 수동으로 재개한다.

콜랩 VM이 종료되면 Drive에서 이미 검증해 게시한 샤드와 경계는 재개에 사용한다. 다음 실행은 기존 경계 기록과 완성 ZIP 고아본을 확인한다. Drive에 남은 writer 잠금은 이전 VM과 다른 writer의 완전한 종료를 확인한 뒤에만 해당 owner 입력으로 복구한다. 정산이 없다면 원장은 해당 호출의 실제 예약 전체를 차감한다. 남은 누적 시간이 충분할 때 그 최대값은 28,800초이며, 남은 누적 예산이 적어 예약이 짧았으면 짧은 실제 예약이 남는다. Drive 연결과 Colab VM 가동 시간은 실행 환경에 따라 달라질 수 있다. [기존 복구 규칙](V2_BULK_CACHE_BUDGETED.md)

## 시간과 바이트

완전한 8시간을 예약하면 셀 시작 후 28,440초(7시간 54분)부터 새로운 원본 읽기·인코딩을 시작하지 않는다. 해당 시점까지 끝나지 않은 인코딩은 경계를 게시하지 않으며, 이미 인코딩한 샤드는 확인 후 게시할 수 있다. 남은 360초는 포장·복사 각 120초, worker 종료 30초, 상위 호출 정산·진단 90초로 배정한다. 정상 정산은 실제 경과의 초 올림과 90초 여유를 예약 이내에서 청구한다.

호출당 읽기 예약은 **640GiB**(687,194,767,360B), 쓰기는 **136GiB**(146,028,888,064B)다. 90분판의 84분 유효 작업에 사용한 112GiB·24GiB를 474분으로 비례 환산하면 약 632GiB·135.43GiB이며 올림했다. 최신 8-worker 실측은 17샤드에 append 409.373초, 전체 읽기 예약 8,245,650,854B와 쓰기 예약 1,770,651,762B다. 읽기에는 원본 외 ZIP 검증이, 쓰기에는 임시 샤드와 보관 복사가 포함된다. 이 수치는 호출별 논리 I/O 제한이다. RAM·로컬 디스크 상한과 Drive 추가 캐시 상한은 기존 값과 동일하며, 한 번에 여러 샤드를 로컬에 쌓지 않는다.

## 고정본 재현

```powershell
python -m scripts.build_v2_bulk_cache_budgeted --source <v2_cpu_cache_ed98124.zip> --catalog <catalog.json> --cpu-report <invocation-hlok774l.json> --resume-report <invocation-hk_nsqcd.json> --preset 8h --evidence-4 outputs/v2_plan_inputs/prefetch_90min_evidence/invocation-y9zf_pek.json --evidence-8 outputs/v2_plan_inputs/prefetch_90min_evidence/invocation-84w0mc8i.json --evidence-overnight outputs/v2_plan_inputs/prefetch_90min_evidence/invocation-jsdvuv09.json --output <new-output>
```
