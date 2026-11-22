# v2.4 metrics R2 연속 구간 노트북

기존 metrics R2의 T4 또는 L4 프로필에서 승인된 **한 arm과 한 seed의 연속 구간**을 한 셀로 실행한다. 기본 OFF다. 이미 실행 중인 셀과 기존 notebook은 그대로 사용한다. [본시험 설정과 지표](V24_DIFFUSION_MAIN_CANDIDATE.md)

## 입력과 승인 범위

- 승인된 `GPU_PROFILE`, `ARM`, `SEED`를 입력한다. T4와 L4는 각 프로필의 원래 프로토콜 SHA-256을 사용한다. 하나의 chain에서 GPU 프로필과 기존 checkpoint binding을 유지한다.
- 기존 `v2_gpu_training_128_v1.zip`과 `v24_main128_A_tools_metrics_r2.zip`을 `/content`에 둔다. 도구 ZIP은 T4·L4 공통이며 SHA-256은 `71819e873749c2178b7c820a3abd040adce967b0d1b71e6e8445540dfba70e6f`다. source SHA-256은 `9a68d8e45bcf3abf6fab3004740faa6805d4ce3fda5118c3a7102011937849d8`이다.
- `APPROVED_BOUNDARIES`에 승인받은 경계만 순서대로 명시한다. 예를 들어 `(0, 768, 1536)`은 두 구간, `(1536, 2304, 3097)`은 1,536부터 두 구간이다. 전체 순서는 `0 → 768 → 1536 → 2304 → 3097`이다. seed 137·181은 기존 별도 확장 승인 조건을 따른다.
- `0`부터면 `RESUME_RESULT`와 `RESUME_RESULT_SHA256`은 비워 둔다. 중도 시작이면 같은 GPU·arm·seed의 **직전 완료 결과 `result.json` 절대 경로와 그 파일 SHA-256**을 입력한다. 기존 metrics R2 단일 구간 영수증도 사용할 수 있다. `APPROVED_PROTOCOL_SHA256`에는 승인받은 프로필의 SHA를 입력한다.

| 프로필 | 프로토콜 SHA-256 |
| --- | --- |
| T4 | `0a2521c09088c48ed195e042210f2b1950efc6863dd6daa19764ca4e8ba392e8` |
| L4 | `16bf01e916f387f0fd56e925ca2b96c8f7a0be2766bc30e1c14238f1307d1a5f` |

## 진행과 예산

설정 셀을 실행한 뒤 승인된 경우에만 `RUN_V24_CHAIN=True`로 켜고 실행 셀을 한 번 호출한다. 구간 시작·완료·검증 경계·경과 시간과 새 Drive 결과 경로·SHA가 출력된다. 각 구간은 원래 metrics R2의 1,500초, worker 마무리 240초, Drive 게시 180초를 유지한다. 논리 읽기는 worker 2 GiB, 전체 논리 쓰기는 6 GiB, 로컬 여유는 4 GiB다. 구간마다 새 UUID를 만들고 별도 검증된 checkpoint를 저장한다.

연속 구간 N개의 계획 상한은 `N × 1,500초 + 180초`다. 추가 180초는 마운트·메타데이터 재조회와 chain 검증용이며 worker 예산에 더해 쓰지 않는다. 4구간은 6,180초(1시간 43분), worker 논리 읽기 합계 최대 8 GiB, 전체 논리 쓰기 합계 최대 24 GiB다. 남은 계획 시간을 확보할 수 없으면 다음 구간을 시작하지 않는다.

## 검증과 회수

1. 현재 호출이 반환한 `result.json` 경로·SHA만 사용한다. Drive 게시자가 모델 상태를 끝까지 읽어 SHA를 확인한 뒤 반환한다. chain은 지정된 경로에서 `result.json`, `worker-result.json`, `segment.json`을 재조회해 핀·경계·predecessor·checkpoint binding을 확인한다. 다음 worker는 재개 상태 전 바이트를 다시 검증한다.
2. `3097`까지 승인된 실행은 최종 공식 validation **421클립·16,425프레임**을 확인한다. 중간 6클립 결과와 구분한다. 최종 경계 이전에 끝나는 승인 계획은 해당 경계의 결과만 회수한다.
3. 각 Drive UUID에서 `result.json`의 절대 경로와 출력 SHA-256, `worker-result.json`, `segment.json`, `body-state.pt`를 회수한다. 실패·중단·검증 불일치가 발생하면 chain을 멈추고 마지막 검증 완료 경로·SHA를 출력한다. 자동 재시도는 없으며 새 승인 실행은 지정한 완료 영수증에서 재개한다. Colab이 작업 중 종료되면 이미 영구 보관된 구간까지 복구할 수 있다.
