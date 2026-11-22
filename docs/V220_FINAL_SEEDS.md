# v2.2 조건부 추가 비교 R2 · seed 137·181

2026-11-23 R2 고정본. 초기 쓰기 2 GiB 추가판은 실행 대상에서 제외했다. R2는 후속 실행 승인 전 준비 상태다. 각 seed에서 `gt_100_pred_0`과 `gt_50_pred_50`을 실행해 총 네 trial을 추가한다. 기존 seed 42·71·103의 세 쌍은 보존된 비교에서 읽는다. 완료 시 `comparison.json`에 다섯 쌍의 혼합−GT 차이와 평균·표본 표준편차를 기록한다. 결과 채택은 별도 검토 사항이다. [기존 세 쌍과 공통 계약](V220_HAND_UPPER.md)

## 실행 설정과 입력

- 각 trial은 A hand e2에서 독립 초기화한다. A frontend e3·body e3 고정, 공식 train 3,097클립, 1 epoch, batch 1, Adam `5e-5`, FP32·TF32 OFF다. 동일 seed 두 군은 같은 도착 순서와 학습 noise를 공유한다. 혼합 군은 seed별로 GT 1,549클립과 예측 상체 1,548클립을 쓴다.
- 공식 validation 421클립·16,425 실제 프레임에서 양쪽 모두 A 예측 상체를 사용한다. 평가 seed 71, DDIM 10-step, body noise offset 100, hand offset 200을 유지한다. 기존 비교의 세 쌍을 보존하고 새로운 두 쌍의 시나리오별 손목 기준·전역 손 MPJPE와 macro를 추가한다.
- `reports/v220_hand_upper128_a_v1/comparison.json`의 파일 SHA-256 `4b27038d48959026b7c99bdfed75207501c40b661ce8097002554573c2fbf2c9`와 `predicted-body/index.json`의 내용 SHA-256 `f64dfc14be817aabde48fb48ba98cfaed031200daddc6c401e01bf80806247b7`이 필요하다. 기존 index와 147개 그룹의 receipt·NPZ를 SHA-256, clip 키, mask, A·feature·설정 binding까지 검증한다. 공식 validation 예측 421개를 그대로 사용한다. 공식 train 키로 계산한 추가 두 seed의 합집합은 2,326클립이며 2,042개를 재사용하고 284개를 추가 계산한다. 원본 검증이 실패하면 중단한다.

## 승인 후 실행과 재개

1. `outputs/v2_plan_inputs/v220_final_seeds_137_181_r2_release_20261123`의 고정 notebook을 독립 CUDA Colab에서 연다. `v220_final_seeds_tools_r2.zip`과 기존 `v2_gpu_training_128_v1.zip`을 `/content`에 올린다. 128 source SHA-256은 `9a68d8e45bcf3abf6fab3004740faa6805d4ce3fda5118c3a7102011937849d8`이다. Colab 스크래치 4 GiB와 새 Drive 출력을 위한 여유를 확인한다. 정상 체크포인트의 기존 실측과 안전 여유를 고려해 Drive 여유 6 GiB를 기준으로 안내한다. A 반출 ZIP은 오프라인 발급용이다.
2. `DRIVE_PROJECT`를 확인하고 승인된 호출에서만 `RUN_V220_FINAL = True`를 설정한다. 기본값은 `False`다. 새 출력은 `reports/v220_hand_upper128_a_final_r2_v1`에 보관한다. 기존 결과, cache와 feature는 읽기 전용이다. CUDA 환경, 예산과 진단을 호출별로 기록한다.
3. 한 호출에서 네 trial을 순차 실행한다. 마운트 포함 8시간, 종료 여유 5분, 논리 읽기 16 GiB·쓰기 8 GiB다. 예산은 노트북 입력과 receipt에 고정하며 Drive 원장에도 상한과 실제 예약·관측 바이트를 기록한다. 실제 GPU 시간과 Colab 사용량은 추가 seed 실행 전 미측정이다.
4. 재호출은 같은 노트북과 ZIP을 사용한다. 검증된 예측 delta, hand·Adam·scheduler·scaler·RNG 체크포인트 및 평가 그룹을 재사용한다. 이전 VM 종료를 확인한 뒤 stale writer가 남아 있으면 기존 owner를 `RECOVER_WRITER_OWNER`에 입력한다. 8시간은 작업의 내부 상한이며 Colab이 그 전에 VM을 종료할 수 있다.

## 한 호출 I/O 산정

기존 마지막 호출의 실측은 논리 읽기 3,847,200,884 B, 쓰기 1,468,336,317 B였다. 원장상 완료한 trial의 체크포인트 재검증까지 포함된 값이다. Hand 상태는 이전 로컬 검증에서 약 187 MB였다. R2는 이 값을 참조하되 네 trial의 준비부터 보관까지 각 산출물의 상한으로 예약한다.

| 항목 | 읽기 (B) | 쓰기 (B) |
| --- | ---: | ---: |
| A ZIP 복구·readback, 추출 예약, A feature 147개 | 1,649,330,264 | 1,508,920,448 |
| 기존 상체 검증, 추가 상체 게시·검증 | 1,126,170,624 | 272,629,760 |
| 정규 hand 상태 12개, 각 게시 readback 포함 | 4,831,838,208 | 2,415,919,104 |
| 평가 72그룹 게시·readback·내용 재읽기 | 3,623,878,656 | 1,207,959,552 |
| trial당 체크포인트 1개 재검증·복원 및 평가 전체 재읽기 1회 | 2,013,265,920 | 0 |
| 경계 상태 최대 4개 및 종료·사전 예약 여유 | 1,879,048,192 | 1,073,741,824 |
| 합계 | 15,123,531,864 | 6,479,170,688 |
| 상한까지 남는 여유 | 2,056,337,320 | 2,110,763,904 |

## 회수와 고정본

루트 `comparison.json`, `diagnostics`, `usage`, `predicted-body-delta/index.json`과 그룹 receipt·NPZ를 회수한다. 네 trial의 `summary.json`, 평가 그룹 receipt·NPZ, `fit-03097.json`과 가리키는 hand 체크포인트를 확인한다. 전체 완료는 다섯 seed가 각 한 쌍이고, 기존 세 쌍의 기록이 원본 비교와 일치하며, 출력에서 작성된 파일의 SHA가 각 receipt와 일치할 때 판단한다.

| 파일 | SHA-256 |
| --- | --- |
| `v220_final_seeds_137_181_cuda_r2.ipynb` | `85de6ed99b308895d2c068e30a5a5a1d3b30d0ccb854e9107df6cda422ecab18` |
| `v220_final_seeds_tools_r2.zip` | `1b5b55f61c23292dfd660ae963b9a27331ca22b923a661b19dae093f441401f9` |
| `receipt.json` | `8c6a942fe8a200f421a9d1e4cc3a66d18788cdd818caf094fc5f4eab96636599` |
| `trial-plan.json` | `03f8c3dea1d825e4870580d4f0e69850d7f1a814cdbb2ac6618549583973e3ef` |

로컬에서 기존 세 쌍의 바인딩, 추가 seed 필터, 예측 delta의 게시·재개, 다섯 쌍 집계와 한 원장 호출의 전체 I/O 예약을 포함해 관련 회귀 8개를 확인했다. 동결 128 source와 ZIP을 풀어 공식 train 키에서 재사용 수를 계산하고, 고정본 반복 발급의 바이트 일치와 notebook 기본 OFF를 확인했다. Drive 원본 그룹의 readback과 추가 trial의 CUDA 실행은 승인 후 확인할 항목이다.

오프라인 발급은 저장소 루트에서 실행한다.

```powershell
python -m scripts.v220_final_build --source outputs/v2_plan_inputs/gpu_training_128_20261122/v2_gpu_training_128_v1.zip --a-bundle outputs/v2_plan_inputs/gpu128_A_portable_release/v2_128_A_final_model_and_cache.zip --initial-receipt outputs/v2_plan_inputs/v220_hand_upper_128_A_verified_20261122/receipt.json --comparison outputs/v2_plan_inputs/v220_validation_20261123/comparison.json --diagnostic outputs/v2_plan_inputs/v220_validation_20261123/invocation-57c7cc0e3d0244389decbd5a8addb6fb.json --output outputs/v2_plan_inputs/v220_final_seeds_137_181_r2_release_20261123
```
