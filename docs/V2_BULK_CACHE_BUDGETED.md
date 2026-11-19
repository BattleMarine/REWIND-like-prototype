# v2 확정 예산 CPU 캐시 재개

고정 경계 SHA-256 `9121e075033b9b922cb8fe93e4389f04914b70eb77cd0c496d138973a4098e55`의 기존 4샤드·512프레임 다음부터 재개한다. 첫 신규 샤드 인덱스는 4다. 목표는 인덱스 0~1,062의 1,063샤드·136,011프레임이다. 마지막 샤드는 인덱스 1,062의 75프레임이다. 새 GPU 시험과 학습은 별도 작업이다. [v2 계획](V2_PLAN.md) · [기존 다중 샤드 검증과 복구 규칙](V2_BULK_CACHE.md)

## 실행과 후속 호출

1. Colab에서 CPU 런타임으로 `v2_bulk_cache_budgeted_cpu.ipynb`를 연다. 기존 `v2_cpu_cache_ed98124.zip`을 이름 그대로 `/content`에 업로드한다. 기존 첫 샤드 및 예산 확정 전 4샤드는 동일 Drive 경로에 둔다. 예산 승인 이후의 추가 캐싱에는 이 고정본을 사용한다.
2. `DRIVE_PROJECT` 경로를 확인한다. 최초 호출은 `EXPECTED_BUDGET_TIP_SHA256`과 `RECOVER_WRITER_OWNER`를 비워 둔다. 기본 OFF인 `RUN_APPROVED_CACHE`를 켠 뒤 전체 실행하고 Drive 마운트를 허용한다.
3. `reports/cache/v2_bulk_cpu_budget_reports_041837Z/invocation-*.json`과 셀에 출력된 보고서 SHA를 회수한다. Drive 게시에 실패하면 `/content/rewind_v2_bulk_budgeted/invocation-*/diagnostic.json`과 `worker.log`를 회수한다.
4. 이전 호출의 `worker.budget_tip_sha256`을 `EXPECTED_BUDGET_TIP_SHA256`에 넣는다. 이후 호출에서는 이전 SHA가 필수다. 종료 보고서를 회수할 수 없으면 출력의 정산 또는 예약 SHA를 사용한다. 출력도 없으면 테스트 세션에서 남은 잠금과 Drive의 마지막 정산·예약을 회수한다. 이전에 저장한 pin이 이후 목록에서 사라졌으면 재초기화하지 않는다.
5. 잠금이 남으면 이전 VM과 모든 writer가 종료됐는지 확인한다. 오류에 표시된 정확한 토큰, 또는 owner가 아직 없던 경우 `empty`를 `RECOVER_WRITER_OWNER`에 입력한다. 다른 활성 writer의 잠금은 제거하지 않는다. 정산 여부에 따라 시간 예약은 그대로 복구되므로 잠금 해제와 예산 초기화는 별개의 작업이다.

`passed`는 1,063샤드·136,011프레임이 모두 확인된 상태다. 최종 경계 `boundary-01062.json`의 frame 수가 75인지 확인한다. `paused`는 추가 진척 전에 멈춘 상태다. 호출당 시간·I/O 또는 로컬 디스크 제한이면 완료 경계와 budget tip을 보존한 뒤 다음 호출을 판단한다. `cpu_cumulative_budget`나 `drive_cumulative_budget`이면 같은 정책으로 재호출하지 않는다. `failed`·`stopped`에서는 보고서와 영구 경계·예약을 확인해 재개 가능 여부를 판단한다. 노트북은 호출을 자동 반복하지 않는다.

## 예산 산정

- **시간**. 누적 324,000초와 호출당 28,800초 중 남은 작은 값을 Drive 호출 원장에 먼저 예약한다. 로컬 watchdog도 해당 예약으로 출력된 deadline을 사용한다. 다음 샤드를 시작할 최소 종료 여유가 남아야 추가 작업을 한다. 이전 미정산 호출은 예약 전액을 차감한다. 정산되면 셀 시작부터 캐시 작업과 최종 Drive 용량 점검까지의 실측 경과(초 올림)에 종료·진단 확보용 90초를 더해 예약 이내로 차감한다. 실패해도 예약 뒤 정산에 도달했으면 동일 규칙을 적용한다. 정상 호출의 마운트·부트스트랩 경과도 위 실측에 포함된다. 마운트·ZIP 준비가 실패해 Drive에 예약을 쓰지 못한 시도에는 캐싱을 시작하지 않으며 원장에 청구할 수 없다.
- **Drive**. 추가 캐시 상한은 60GiB(64,424,509,440바이트)다. 기존 4샤드 및 해당 완료 경계를 제외하고 새 ZIP, 추가 경계의 파일 바이트, 이전 중단으로 남은 임시 전송 파일까지 반영한다. 다음 ZIP과 경계가 여유 안에 있을 때만 생성·게시한다. Drive 내부의 버전, 휴지통, 원장·잠금·진단 파일과 물리 복제 비용은 추가 캐시 bytes와 별개다. Drive의 실제 여유 공간 확인도 유지한다.
- **호출**. 호출당 읽기와 쓰기 각각 8GiB, 기존 단일 writer와 로컬 스테이징을 유지한다. 이전 샤드의 해시 검증 결과와 고정 경계를 재사용하고, 이미 완료한 ZIP 전체를 새 VM에 내려받지 않는다. 예산 원장 `.cpu-quota-v1`과 원점·첫 예약 표시 파일을 경계 폴더에 보관한다. 원장은 연속된 예약·정산 SHA를 대조한다. 최초 원점 표시 전 빈 원장 생성만 자동 복구할 수 있다.

## 고정본 재현

기존 source ZIP, 최초 CPU 보고서·카탈로그 및 4샤드까지 재개한 회수 보고서가 필요하다. 빌더는 회수 보고서의 SHA와 4샤드·512프레임, SHA 경계 및 기존 생산 소스 identity를 검증한다. 런타임 오버레이 SHA와 notebook SHA는 `receipt.json`에 기록된다.

```powershell
python -m scripts.build_v2_bulk_cache_budgeted --source <v2_cpu_cache_ed98124.zip> --catalog <catalog.json> --cpu-report <invocation-hlok774l.json> --resume-report <invocation-hk_nsqcd.json> --output <new-output>
python -m unittest discover -s tests -p test_v2_bulk_quota.py -v
```
