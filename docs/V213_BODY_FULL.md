# v2.1.3 · 128 A body 전체 미세조정

v2-128-A의 선택 frontend e3·body e3·hand e2에서 각 seed를 독립적으로 시작한다. Body 파라미터 99개 텐서, 15,010,822개 FP32 값을 모두 학습한다. 예측 2D feature와 hand 가중치는 A를 사용하며, hand의 상체 입력은 해당 trial body에서 예측한다. Seed 42·71·103은 각각 공식 train 3,097클립·119,586페어를 1epoch, batch 1, Adam `5e-5`로 학습한다. 앞선 두 방식과 같은 clip 셔플·teacher timestep/epsilon·손실·평가 noise를 사용한다. [v2 튜닝 설계](V2_TUNING_DESIGN.md)

평가는 공식 validation 421클립·16,425 실제 프레임, seed 71, DDIM 10, FP32·TF32 OFF로 진행한다. 기존 분해 함수가 body·hand-global MPJPE와 A 대비 차이, 좌우 손목·상대 손가락·전역 손가락 및 교차항을 갱신한다. 세 시나리오의 body·hand-global MPJPE 6개를 동일 가중 평균한다. 예측·GT·mask·키를 평가 그룹 NPZ로 보관한다. `summary.json`의 `measurement`는 공통 CPU 선별기에 전달할 수 있다. [공식 분해와 mask](V210_EVAL.md)

## 8시간 호출과 I/O

| 항목 | 근거와 한도 |
| --- | --- |
| 체크포인트 1개 | 실제 A와 train feature로 Adam 상태를 생성한 뒤 로컬 측정 약 180.26 MB. 체크포인트 상한 192 MiB, 복사·읽기 검증에는 최대 210초를 배정한다. |
| 호출 | 마운트 포함 최대 28,800초, 새 작업을 시작하기 전 5분 여유. 논리 읽기 8 GiB, 쓰기 2 GiB. |
| 부모 복원 예약 | 매 호출 A ZIP 복사 435,178,624 B와 추출 상한 1 GiB 예약을 합산한다. 학습·평가 산출물용 쓰기 여유는 호출당 638,563,200 B(609 MiB)다. A feature ZIP 778,972,869 B를 호출당 한 번 읽는다. |
| 체크포인트 주기 | 업데이트 1·1550·3097과 시간 종료 시 안전한 완료 업데이트. 세 seed의 정규 체크포인트 9개는 약 1.62 GB이므로 현재 예약 계약에서 적어도 3번의 수동 호출이 필요하다. |
| 평가 및 다음 구간 | 3개 fit 파일이 약 540.8 MB이면 호출 안에 약 97.8 MB의 추가 쓰기 예약 여유가 있다. 기존 v2.1.1 T4의 fit·평가 출력 합계는 seed당 평균 약 39 MB였다. 이번 평가의 실제 크기·중단 지점에 따라 추가 호출이 필요할 수 있다. |

앞 구간의 fit을 검증 게시한 뒤 다음 fit 최대치(쓰기 192 MiB·읽기 검증 384 MiB)를 확보할 수 없으면 `v2_tuning_*_budget_pause_before_next_durable_unit`로 멈춘다. 새 평가 그룹도 상한을 확인한 뒤 시작한다. 기존 fit과 평가 그룹은 새 호출에서 재검증하며, 한 호출을 넘겨 이어갈 때 I/O 예산은 해당 호출에 다시 적용된다. VM 유실 시 마지막 유효 경계 이후 최대 1,549 update를 다시 계산할 수 있다. 미정산 호출은 원장에 최대 28,800초 예약으로 남는다. 회수 전에 필요한 누적 Colab 예산을 오케스트레이션이 확인한다.

비교용으로 기존 T4 v2.1.1은 부모·feature 준비 약 325초, seed당 평가 약 189초였다. 기존 A100 A의 body 전체 학습 step은 3epoch 합계 약 471초였다. 이번 full-body T4 학습·보관 시간과 VRAM은 첫 CUDA 관측에서 확인해야 한다. [v2.1.1 T4 실측](V211_BODY_HEAD.md#t4-전체-학습-결과)

## 고정 Colab과 복원

1. 발급 위치는 `outputs/v2_plan_inputs/v213_body_full_128_A_release_20261122`다. 오케스트레이션이 인계한 `v213_body_full_128_A_cuda.ipynb`를 열고 기존 `v2_gpu_training_128_v1.zip`, `v213_body_full_tools.zip`을 `/content`에 업로드한다. CUDA GPU 한 대를 선택하고 로컬 스크래치 여유 4 GiB 이상을 확인한다. `RUN_V213`은 기본 `False`다.
2. 승인된 실행에서 `DRIVE_PROJECT`를 확인하고 `RUN_V213 = True`로 한 번 호출한다. CUDA R2의 장치 확인·초기화·실제 할당 및 계측 오류 보존을 사용한다. 결과는 `reports/v213_body_full128_a_v1`에 별도 writer 및 원장과 함께 저장한다. 원래 A 체크포인트와 feature 및 cache 경계 메타데이터는 읽기 전용 입력이다. [CUDA 초기화와 진단](V2_TUNING_CUDA_STARTUP_R2.md)
3. 시간·I/O 정지 후 같은 고정본과 Drive 경로로 수동 호출한다. 새 런타임은 A를 다시 검증하고 최종 `fit-xxxxx.json`과 `body-*.pt`를 읽어 전체 FP32 body 파라미터, 99개 Adam 상태, scheduler·scaler·RNG와 update 위치를 복구한다. 모델의 고정 buffer SHA도 확인한다. 이미 완료한 trial과 게시 그룹은 검증 후 건너뛴다. 이전 VM 종료를 확인했고 writer가 남아 있으면 그 owner를 `RECOVER_WRITER_OWNER`에 입력한다.
4. seed별 `fit-03097.json`과 해당 `body-03097-*.pt`, `summary.json`, 18개 평가 그룹 영수증·NPZ, 최근 invocation·usage를 회수한다. 한 epoch이므로 같은 `fit-03097`이 trial best와 last다. 공통 `trial-plan.json` 내용 SHA는 `7753d66479f5603a63816159d64a57fe050a31fdd1e2dd2279fce449d2185968`이다.

## 발급 SHA와 로컬 검증

| 파일 | SHA-256 |
| --- | --- |
| `v213_body_full_128_A_cuda.ipynb` | `8f89945bb385bf3e52bbdadb0e2d0de0f8d47be849e76aa7326ec03a18d6c7a6` |
| `v213_body_full_tools.zip` | `9b4043be09cc948140a3b3e1d1036cc7727b9859ac4fbba0149fe4e468ea1211` |
| `receipt.json` | `9f2c86ef43578f436cb4cad37bb775f0e8eabc1b6cac9f923a3d1f654288b70f` |
| `trial-plan.json` | `03f8c3dea1d825e4870580d4f0e69850d7f1a814cdbb2ac6618549583973e3ef` |

동결 128 소스와 발급 overlay로 실제 A step 9, 실제 train 2클립·validation 1클립을 CPU에서 확인했다. Step 1에서 쓰기 예산 경계로 멈추고 새 teacher에서 step 2를 수행한 결과, 연속 실행과 99개 파라미터, Adam 누적 모멘트·카운터, scheduler·손실이 정확히 일치했다. Fit 파일은 step 1에서 180,264,273 B, step 2에서 180,264,529 B였다. A와 비교해 body 파라미터 99개 텐서가 모두 변했고 고정 buffer는 유지됐다. 평가 경로에서 `47 × 62 × 3` device-mm 예측과 동결 hand 연결을 확인했다. 전체 CUDA 학습·검증과 시간은 미측정이다.

오프라인 발급은 저장소 루트에서 아래 명령을 실행한다. `--output`에는 새 경로를 지정한다. A 반출 ZIP은 이 오프라인 단계에만 필요하다.

```powershell
python -m scripts.v213_build --source outputs/v2_plan_inputs/gpu_training_128_20261122/v2_gpu_training_128_v1.zip --a-bundle outputs/v2_plan_inputs/gpu128_A_portable_release/v2_128_A_final_model_and_cache.zip --output outputs/v2_plan_inputs/v213_body_full_128_A_release_20261122
```
