# v2.4 x0/v 본시험 후보와 조건부 인계

상태는 오케스트레이션 검토 및 사용자 실행 승인 대기다. v2.2 추가 seed 작업과 별개다. [본시험 계획](V24_DIFFUSION_MAIN_PLAN.md) · [GPU 파일럿 계약](V24_DIFFUSION_PILOT.md)

## 파일럿 근거와 본시험 설계

R2 파일럿은 L4 FP32에서 65.70초였다. x0/v의 arm별 학습·중간 평가·저장 묶음은 5.89/5.72초였고 두 상태는 각각 180,268,441바이트였다. 6클립 pooled body 오차는 0→12→24 update에서 x0 662.35→245.84→363.51, v 493.23→415.15→440.40 mm였다. 두 군은 전역 손 오차도 12→24에서 다시 늘었다. 초기 출력의 의미가 달라 초기 MPJPE는 다를 수 있다. 24 update와 6클립만으로 순위 또는 수렴을 확정하기 어렵다. 원본 9파일과 검증 계획의 위치는 `outputs/v2_plan_inputs/v24_gpu_pilot_results_20261123/`다. `main-plan.json` SHA-256은 `d2976020b429550e2868baac65ef3b22db3ee3b97044cfef0e4d462d386b9681`이다.

- **후보 입력과 학습량.** 고정 v2 128 A와 A feature identity, 공식 train 3,097클립(119,586페어), batch 1, Adam 5e-5, seed별 1회 순회다. feature archive 그룹과 그룹 안 클립을 seed별로 정렬하고 마지막 부분 그룹은 끝에 둔다. 두 arm이 같은 순서 및 timestep/noise를 받는다. 두 arm은 같은 A backbone과 seed별 동일 재초기화 head에서 시작한다. 중간 head는 x0를 유지하며 frontend와 hand를 동결한다. v는 x0로 변환해 동일한 masked loss와 DDIM10을 쓴다.
- **고정 경계.** 0·768·1,536·2,304에서 파일럿의 6클립을 진단한다. 3,097에서 공식 validation 421클립·16,425 실제 프레임을 모두 평가한다. 같은 평가 noise seed 71을 사용한다. 최종 점수는 3시나리오의 몸·전역 손 MPJPE 6점을 동일 가중한 평균이다. 손목과 손 local, 유효 수를 함께 기록한다. Test split은 읽지 않는다.
- **선정 및 조건부 확장 제안.** seed 42·71·103에서 3,097 경계만 공식 비교에 사용한다. 중간 6클립의 마지막 두 구간에서 계속 개선되면 미수렴 가능성을 표시한다. 유망 판단의 후보는 v가 3쌍 중 최소 2쌍에서 낮고 평균 v−x0 종합 MPJPE가 −3 mm 이하이며 평균 전역 손 MPJPE가 증가하지 않는 경우다. 3 mm는 실행 전 검토할 실용 차이 기준이다. 통과하더라도 seed 137·181의 양쪽 실행은 별도 승인한다.

## 비용 산정과 중단

| 범위 | 제안 상한 |
| --- | ---: |
| 구간 1회 | 1,500초, worker 논리 읽기 2 GiB, 전체 논리 쓰기 6 GiB, 로컬 여유 4 GiB |
| 최초 6 arm × 4구간 | 상한 10 GPU시간, Drive 상태 최대 4.50 GiB 및 작은 보고서 |
| 조건부 4 arm 추가 | 상한 6시간 40분, Drive 상태 최대 3.00 GiB 및 작은 보고서 |

구간 예산에는 최다 793 update, 최대 53회 feature 그룹 입력(34 train + 18 공식 validation + 1 probe), A 복구, 이전 상태 가져오기와 현재 상태 게시·재읽기 검증을 반영했다. 파일럿의 arm 시간은 업데이트 단독 처리량으로 환산할 수 없다. 용량 예약은 그룹당 고정 ZIP 크기와 추출 상한 32 MiB, 부모 복구·resume·신규 저장을 합한다. 입출력 상한은 논리적 페이로드 기준이다. v2.3의 같은 L4 공식 validation 3개 DDIM 조건 전체가 181.69초였다는 점을 별도 실측 근거로 사용했다. 구간별 시간 참고치는 update당 0.8초까지 약 635초, 준비 및 IO 150초, 최종 평가 240초와 종료 여유 240초다. 합계 약 1,265초로 1,500초 안에 예약한다. 이 수치는 본시험 실측이 아니며 첫 승인 구간에서 학습·평가·게시 시간을 회수해 남은 예산을 재검토한다.

마지막 240초에는 새 학습·평가를 시작하지 않는다. 그 안의 마지막 180초는 controller 게시·재검증용이다. 실행 결과는 고유 UUID를 사용한다. 중단 시 마지막으로 완결된 구간의 Drive 결과를 기준으로 회수한다. worker가 완결됐고 Drive 게시가 실패하면 현재 VM에서 로컬 복구 폴더를 보존하고 경로를 출력한다. 중단된 구간의 일부 업데이트는 재개 지점으로 사용하지 않는다. 구간을 재시도하거나 상한을 바꾸려면 오케스트레이션 결정을 받는다.

## 조건부 Colab 실행 및 회수

1. 가이드와 발급 `receipt.json`에서 파일 SHA와 프로토콜 SHA를 검증한다. 승인된 경우 `v2_gpu_training_128_v1.zip`과 본시험 toolkit을 `/content`에 올리고 한 대의 L4에서 실행한다. 기본 `RUN_V24_MAIN=False`와 빈 `APPROVED_PROTOCOL_SHA256`으로 발급한다. 최초 예시는 x0·42·0→768이다.
2. 승인된 arm과 seed의 시작·종료 경계를 선택한다. 시작 0이면 resume 입력을 비운다. 이후 구간은 같은 arm·seed·환경의 직전 `result.json` Drive 절대 경로와 파일 SHA-256을 지정한다. 실제 실행 환경 전체와 초기화·데이터 순서·가중치·소스 식별이 이전 checkpoint와 일치해야 한다.
3. `reports/v24_main128_a_candidate/run-<UUID>/`에서 노트북이 출력한 `result.json` 경로와 SHA를 회수한다. `segment_complete`, 종료 코드 0, `worker-result.json`, `segment.json`, `body-state.pt`의 일치하는 크기·해시를 확인한다. 상태에는 body, Adam, scheduler, scaler, RNG와 update cursor가 포함된다. 부모 자료에는 읽기만 수행한다.
4. 두 arm의 각 seed에서 네 구간을 추적하고 같은 구간의 `common_sha256`, `paired_update_sha256`, validation `noise_and_roster_sha256`이 일치하는지 확인한다. 직전 결과 SHA가 다음 구간의 `resume_result_sha256`과 일치해야 한다. 부분 구간 또는 짝이 없는 seed는 공식 비교에서 대기 상태로 둔다.

### T4 첫 병렬 구간

2026-11-23 · 두 VM 모두 Tesla T4로 확인해 seed 42의 x0와 v 각각 0→768을 승인했다. 새 고정본은 `outputs/v2_plan_inputs/v24_main_t4_pair_20261123/`에 있다. 프로토콜 SHA는 `0a2521c09088c48ed195e042210f2b1950efc6863dd6daa19764ca4e8ba392e8`이며 기존 L4 프로토콜과 runtime.gpu만 다르다. 파일럿 시간은 L4 실측이며 T4의 구간 시간과 결과는 회수 대기다.
