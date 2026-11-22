# v2.1.4 · 손목 위치 손실 가중치 2

2026-11-23 · v2.1.4 L4 3회 완료(677초). 같은 seed 42·71·103에서 출력층 손목 위치 손실의 가중치 2−1 차이는 세 시나리오 동일 가중 평균에서 왼손목 −1.76, 오른손목 −1.18, 전역 손 −1.63, 손목 기준 손가락 −0.24, 손목 제외 몸 +0.67 mm였다. 다섯 지표 각각 세 seed에서 같은 방향이었다. 원본 요약·fit·head·비교·진단은 outputs/v2_plan_inputs/v214_l4_results_20261123/에 보존했고 그룹 NPZ는 로컬 미회수다. 독립 분기인 v2.2는 기존 A를 부모로 테스트 세션에 인계했다.

v2.1 최종 비교에서 head 또는 LoRA가 선정되면, 해당 방식으로 A의 원래 가중치에서 다시 시작한다. 선정된 trial의 학습된 가중치는 대조군 결과와 연결하는 데 사용한다. 새 학습은 seed 42·71·103, 각 3,097클립·1epoch·batch 1·Adam `5e-5`이며, 기존 weight 1과 같은 클립 순서와 teacher timestep·epsilon stream을 적용한다. 확장 seed는 결과 검토 후 별도로 결정한다. [튜닝 설계](V2_TUNING_DESIGN.md)

최종 body의 위치 손실은 `sum(w × normalized XYZ squared error) / (3 × sum(w))`로 계산한다. 유효한 좌우 손목엔 `w=2`, 그 밖의 유효한 관절엔 `w=1`을 준다. Padding 및 불완전 GT는 가중합·분모에서 제외한다. 보조 frame 손실, pose 단위·mask, 동결 A frontend와 hand 가중치, DDIM 및 optimizer는 기존 경로를 유지한다. Hand에는 해당 trial body의 예측 상체를 넘긴다.

## 부모 방식 판정과 실행 조건

고정본은 `outputs/v2_plan_inputs/v214_wrist_weight_128_A_verified_20261122`에 있고, 셀의 `RUN_V214`는 기본 OFF다. 최종 부모 판정 파일이 발급되기 전에는 학습을 보류한다.

1. 오케스트레이션이 초기 9개와 finalist 4개 요약을 회수한다. 기존 [최종 선별 도구](V2_TUNING_FINAL_SEEDS.md)로 검토된 `target-verdict.json`을 전달하고 `--output`에 최종 선택 결과 JSON을 저장한다. `validation_candidate`가 있는 head 또는 LoRA 결과로 다음 단계가 열린다. 부모 유지나 판정 대기는 실행 조건을 충족하지 않는다.
2. 아래 `v214_decision` 명령으로 그 결과를 다시 계산한다. 두 finalist receipt SHA, 동일한 초기 9회 근거, final summary 4개, 승인된 판정, winner와 동일 방식의 초기 3개 weight 1 요약을 대조한다. 출력 파일이 승인되면 파일 SHA-256을 기록한다.
3. `/content`에 기존 `v2_gpu_training_128_v1.zip`, `v214_wrist_weight_tools.zip`, 오케스트레이션이 검증한 `v214-selection-decision.json`을 올린다. `DRIVE_PROJECT`와 `V214_DECISION_SHA256`을 설정한 뒤 승인된 호출에서만 `RUN_V214 = True`로 바꾼다. Worker는 CUDA 시작 전에 파일 SHA와 재계산한 최종 선택을 확인한다. 새 weight 2 fit 전에 기존 weight 1의 세 요약과 각각의 18개 예측 그룹을 키·mask·SHA로 대조한다.

```powershell
$p = 'outputs/v2_plan_inputs'
$wrist = "$p/v214_wrist_weight_128_A_verified_20261122"
$head = "$p/v211_final_seeds_137_181_release_20261122"
$lora = "$p/v212_final_seeds_137_181_release_20261122"
$finalSummaries = @('v211-seed137-summary.json', 'v211-seed181-summary.json', 'v212-seed137-summary.json', 'v212-seed181-summary.json')
python -m scripts.v214_decision --trial-plan "$wrist/trial-plan.json" --releases $head $lora --final-summaries @finalSummaries --target-verdict target-verdict.json --selection-report final-selection.json --initial-results "$p/v211_r2_t4_results_20261122" "$p/v212_r2_t4_results_20261122" --output v214-selection-decision.json
```

`$finalSummaries`, `target-verdict.json`, `final-selection.json`에는 실제 회수·검토한 경로를 적용한다. 대조군 NPZ와 영수증은 원래 선정 방식의 `reports/v211_body_head128_a_v1/seed-*` 또는 `reports/v212_body_lora128_a_v1/seed-*`에서 읽는다.

## 예산과 복원

- 지원되는 CUDA GPU 1개, FP32·TF32 OFF, 스크래치 여유 4 GiB, 마운트를 포함한 호출당 8시간과 종료 여유 5분, 읽기 8 GiB·쓰기 2 GiB를 적용한다. 기존 T4의 세 seed 실행이 head 약 19분 21초, LoRA 약 15분 17초였고, 이 경로에는 weight 1 그룹 readback과 쌍별 집계가 추가된다. 계획상 T4 약 16~22분, 정상 새 보관 출력 약 0.12~0.16 GB다. 전체 CUDA 비용은 실측 전이다.
- `reports/v214_wrist_weight128_a_v1`에 선택 파일 SHA와 새 코드·입력·fit을 결합한다. 동일한 고정본과 동일한 판정 파일로 재호출하면 1·50회 경계의 마지막 확인된 fit, 혹은 평가 그룹부터 재개한다. 이전 VM 종료를 확인한 뒤 남은 writer에 한해 기존 owner를 사용한다. 초기·final weight 1 결과는 읽기 전용으로 재사용한다. [CUDA R2](V2_TUNING_CUDA_STARTUP_R2.md)

## 결과 회수

두 조건 모두 공식 validation 421클립·실제 16,425프레임, 평가 seed 71, DDIM 10으로 비교한다. 각 쌍에서 sequence·clip·frame·GT·mask 일치를 확인한다. 좌우 손목, 손목을 제외한 body, hand-global, hand-local MPJPE를 유효 관절 수와 함께 시나리오별·pooled·동일 가중 macro로 제공한다. Seed별 `comparison.json`은 weight 2−weight 1 차이를, 루트 `comparison.json`은 세 쌍의 평균과 표본 표준편차를 기록한다. `fit-03097.json`과 짝 가중치, seed별 `summary.json`과 `comparison.json`, 18개 평가 그룹과 NPZ, 루트 비교, Drive `diagnostics`·`usage`를 회수한다. [공식 지표 정의](V210_EVAL.md)

## 오프라인 재발급

| 파일 | SHA-256 |
| --- | --- |
| `v214_wrist_weight_128_A_cuda.ipynb` | `1a8c839e0c0461dad5532890c3b5af62070766275f0f4d397219465a157d2f9b` |
| `v214_wrist_weight_tools.zip` | `ed0077566ed79b377e59b49d1bcbe9be82fc40c78c6cc1a17f2a97d12643f06b` |
| `receipt.json` | `f1420fe71b8605e82c931675d5e42fcaab0af8686d56fc0681811b71f3665b52` |
| `trial-plan.json` | `03f8c3dea1d825e4870580d4f0e69850d7f1a814cdbb2ac6618549583973e3ef` |

로컬 회귀 30개와 기초 수치·gradient, 합성 head·LoRA fit 재개, 기본 OFF, CRC, 반복 발급 바이트 일치를 확인했다. 동결 소스에서 실제 오버레이를 사용해 판정 파일이 없을 때 CUDA·Drive 전에 중단하는 것을 확인했다. 기존 실제 9개 행과 모의 finalist 행으로 두 방식 모두 조건부 판정 도구를 확인했다. 실제 선정 입력은 로컬에서 head 추가 두 회차가 확인됐고 LoRA 추가 두 회차의 요약을 기다린다. 공식 판정 파일과 weight 2 실제 학습은 준비 후속 단계다.

```powershell
python -m scripts.v214_build --source outputs/v2_plan_inputs/gpu_training_128_20261122/v2_gpu_training_128_v1.zip --a-bundle outputs/v2_plan_inputs/gpu128_A_portable_release/v2_128_A_final_model_and_cache.zip --output outputs/v2_plan_inputs/v214_wrist_weight_128_A_verified_20261122
```
