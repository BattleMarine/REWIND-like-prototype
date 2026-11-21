# v2 CPU 캐시 90분 프리셋

이 프리셋은 기본 8 worker로 기존 CPU 캐싱을 한 번 재개한다. 호출 전체 최대 5,400초, 논리 읽기 예약 112GiB, 쓰기 예약 24GiB를 적용한다. 기존 8시간·8GiB 고정본은 보존한다. 누적 CPU 90시간, 추가 Drive 캐시 60GiB, 단일 writer, 기존 manifest·producer binding·카탈로그·예산 원장은 공통이다. [기존 재개와 보고서 규칙](V2_BULK_CACHE_BUDGETED.md)

## 실행

1. Colab CPU 런타임에서 `v2_bulk_cache_90min_cpu.ipynb`를 열고 기존 `v2_cpu_cache_ed98124.zip`을 `/content`에 올린다.
2. 직전 실제 호출의 최신 정산 SHA 또는 중단 시 예약 SHA를 `EXPECTED_BUDGET_TIP_SHA256`에 입력한다. 프리셋은 228샤드 완료 경계 `11b19dc1d82ecf58872b55c446da68e22eb84ccd5825dc800e91b9ea893dde00`가 검증된 이력에 있는지 확인한다. 더 최근 경계가 있으면 그 다음 인덱스부터 처리한다. 228샤드 당시의 예산 SHA를 고정값으로 사용하지 않는다.
3. 다른 writer가 종료됐는지 확인한 뒤 `RUN_APPROVED_CACHE`를 켜고 실행한다. 기존과 같은 Drive 폴더에서 처리하며 같은 진단 폴더로 보고한다. 종료 보고서의 새 예산 tip, 완료 샤드, `stop_reason`을 회수한다. 다음 호출은 수동으로 시작한다.

## 호출 시간과 I/O 상한

전체 한도가 예약되면 셀 시작 후 5,040초(84분)부터 신규 원본 읽기·인코딩을 시작할 수 없다. 원본·인코딩 중이던 샤드가 그때까지 완료되지 않으면 저장 경계를 게시하지 않는다. 이미 끝난 샤드는 기존 검증을 거쳐 게시할 수 있다. 5,400초 안에 포장·복사에 각각 120초, worker 종료에 30초, 상위 노트북 정산·진단 여유에 90초를 배정한다. 남은 누적 시간이 5,400초보다 짧으면 같은 여유를 그보다 이른 시점부터 적용한다. 처리 목표 완료, I/O 상한, Drive 상한 또는 파일 오류가 먼저 발생하면 호출은 더 일찍 끝날 수 있다.

미정산 VM 유실 시 원장에 추가되는 값은 해당 호출의 실제 사전 예약이다. 전체 예약은 5,400초이며 이전 8시간 예약·정산은 원래대로 합산한다. 정상 정산은 기존과 같이 실측 경과(초 올림)에 90초를 더해 예약 이내에서 차감한다.

| 관측 항목 | 4 worker 호출 | 8 worker 호출 |
| --- | ---: | ---: |
| 추가 샤드·프레임 | 17 / 2,176 | 17 / 2,176 |
| append 경과(초) | 676.555 | 421.385 |
| 원본 논리 읽기(B) | 5,544,694,752 | 5,834,222,466 |
| ZIP 검증 등을 포함한 전체 읽기 예약(B) | 8,126,692,662 | 8,416,220,376 |
| 전체 쓰기 실측 / 예약(B) | 1,721,111,059 / 1,770,651,762 | 1,721,111,059 / 1,770,651,762 |

8-worker 관측 속도로 84분 안쪽에서 약 202샤드를 처리한다고 가정하면 전체 읽기 약 100.0GB, 쓰기 예약 약 21.0GB다. 각각 약 20%, 22% 여유를 두어 읽기 112GiB(120,259,084,288B)와 쓰기 24GiB(25,769,803,776B)를 선택했다. 다음 샤드에 대한 보수적 예약 및 단일 파일·샤드 크기 상한은 유지한다. 호출별 원본 크기와 Drive 성능은 달라질 수 있다. 진단의 `invocation_preset`, `time_reserved_seconds`, `new_shard_cutoff_seconds_after_cell_start`, `read_cap_bytes`, `write_cap_bytes`, `usage`, `performance`에 적용값이 기록된다.

## 고정본 재현

기존 고정본 생성 명령에 프리셋과 실측 보고서 두 건을 지정하고 새 출력 경로를 사용한다. 두 보고서의 SHA와 기존 producer identity가 빌드 때 검증된다.

```powershell
python -m scripts.build_v2_bulk_cache_budgeted --source <v2_cpu_cache_ed98124.zip> --catalog <catalog.json> --cpu-report <invocation-hlok774l.json> --resume-report <invocation-hk_nsqcd.json> --preset 90min --evidence-4 outputs/v2_plan_inputs/prefetch_90min_evidence/invocation-y9zf_pek.json --evidence-8 outputs/v2_plan_inputs/prefetch_90min_evidence/invocation-84w0mc8i.json --output <new-output>
```
