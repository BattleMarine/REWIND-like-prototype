# v2.1.1 · 128 A body 최종 head 미세조정

A의 frontend e3, body e3, hand e2를 공통 시작점으로 사용한다. 독립 seed 42·71·103을 한 CUDA GPU에서 순차 실행한다. 각 seed는 공식 train 119,586페어를 구성하는 3,097클립을 1회 순회한다. Batch 1, Adam 5e-5, 3,097회 업데이트며, 부모와 같은 body teacher 손실·noise schedule을 사용한다. `final_head.weight`와 `final_head.bias` 1,539개 값만 학습한다. 나머지 body 텐서는 시작과 끝의 SHA를 대조한다. Frontend는 A의 기존 예측 2D feature를 사용하며, hand는 A의 선택 가중치를 유지한 채 현재 trial body가 예측한 상체를 입력받는다. [베이스 튜닝 설계](V2_TUNING_DESIGN.md)

`scripts/v211_trial.py`의 `order`와 `update`가 동일 seed의 비교 경로다. `order`는 공식 clip 정렬에 seed별 CPU `randperm`을 적용한다. `update`는 `(seed, zero-based step)`으로 teacher timestep·epsilon을 뽑는 stream을 따로 시작한다. 모델 초기화는 별도 stream이다. 후속 body 방식도 비교하려면 같은 두 함수, 같은 clip과 장치별 RNG 구현을 사용한다. CUDA 종류나 Torch/CUDA 버전이 달라지면 FP32여도 결과의 비트 일치를 가정하지 않는다.

## 고정 Colab과 입력

CUDA 시작 수정 R2 발급 폴더는 `outputs/v2_plan_inputs/v211_body_head_128_A_cuda_startup_r2_20261122`다. `receipt.json`에 노트북·overlay·부모 입력 SHA가 있으며 `trial-plan.json`에 CPU 선별기와 공유하는 고정 계획이 있다. 부모 원본 source는 `outputs/v2_plan_inputs/gpu_training_128_20261122/v2_gpu_training_128_v1.zip`이며 SHA-256은 `9a68d8e45bcf3abf6fab3004740faa6805d4ce3fda5118c3a7102011937849d8`이다. A 최종 step 9 영수증의 내용 SHA는 `7d2cf12cdaf42916d98190cf2d57e11e93a6d94165d5e753db6fc1d11c3026d4`다. 기존 Drive의 A 체크포인트 ZIP과 feature 그룹 147개를 검증해 읽는다. 검증용 128 캐시 경계는 `dc79219153d69f44020e0c42f9f3ede6cd3add179211f6c300345b2416dab511`이며 해당 경계의 메타데이터만 읽는다. 51 GiB 로컬 A 반출 ZIP은 발급기에서 입력 핀을 확인할 때 사용한다. [CUDA 시작 수정·이전 보관본](V2_TUNING_CUDA_STARTUP_R2.md)

1. 오케스트레이션이 승인된 GPU 예산으로 인계하면 `v211_body_head_128_A_cuda_r2.ipynb`를 연다. A100·T4·L4 등 사용할 수 있는 CUDA GPU 하나를 선택한다. 비교·재개에는 같은 GPU 종류와 Torch/CUDA 환경을 우선한다.
2. Colab `/content`에 고정 `v211_body_head_tools_r2.zip`과 기존 `v2_gpu_training_128_v1.zip`을 올린다. 로컬 스크래치에 4 GiB 이상 여유를 둔다. `DRIVE_PROJECT`를 확인하고 `RUN_V211 = True`로 바꿔 한 번 실행한다. 설정의 기본값은 `False`다.
3. 호출 상한은 마운트 시도부터 8시간이다. 5분을 남기고 새 학습 업데이트나 평가 clip을 시작하지 않는다. 호출 논리 읽기 8 GiB, 쓰기 2 GiB를 적용한다. A feature ZIP은 총 778,972,869바이트로 호출 중 한 번만 읽어 스크래치에 복원한다. 같은 호출의 모든 seed가 이를 공유한다. Colab의 VM 종료·회수 정책에 따라 호출 상한 전에 중단될 수 있다. 미정산 호출은 다음 Drive 원장 실행 시 `interrupted_unknown`과 예약한 28,800초를 기록한다.

이 경로의 실제 CUDA 시간은 미측정이다. 기존 A100 A의 body 전체 3epoch 학습 step 구간은 약 471초, body와 hand DDIM 모델 계산은 각 약 50초/validation sweep이었다. 이를 단순 환산하면 한 seed의 학습·평가 기준은 약 4.3분, 세 seed 약 13분이다. 이 비교값에는 이번 Drive 입력 전송·체크포인트 게시·NPZ 저장·새 장치의 성능 차이가 들어 있지 않다. 실행의 각 시간은 별도로 기록된다.

## 저장·재개와 결과 회수

- `reports/v211_body_head128_a_v1`에 전용 단일 writer와 사용 원장을 둔다. `seed-42`, `seed-71`, `seed-103`에 첫 업데이트, 이후 50회 경계, 마지막 3,097회 step의 head 및 optimizer·scheduler·scaler·RNG를 기록한다. 예약 종료에 닿으면 안전한 완료 업데이트의 지점을 추가로 저장한다. 강제 중단이나 VM 유실 후에는 가장 최근의 검증된 영수증까지 복구한다. 완료 trial은 `summary.json`과 마지막 head 파일을 확인한 뒤 건너뛴다. 평가 중단은 기존 그룹 NPZ의 SHA·클립/프레임 키를 검증한 뒤 다음 그룹에서 재개한다. 이전 VM 종료를 확인한 뒤 잠금이 남아 있으면 출력된 기존 owner를 `RECOVER_WRITER_OWNER`에 입력하고 한 번 다시 실행한다. 영수증 커밋 전에 끊긴 파일은 재개 대상에서 제외한다.
- 시점별 `fit-xxxxx.json`이 head 파일의 이름·바이트·SHA를 가리킨다. 학습량이 1epoch이므로 `fit-03097.json`의 head가 trial의 last와 epoch best에 해당한다. 다른 seed에서 학습을 이어받지 않는다. 파일은 A body 부모에 결합되는 head와 학습 상태를 포함한다. 원래 A body와 최종 fit 영수증을 함께 보존한다.
- 세 seed의 `summary.json`, `fit-03097.json` 및 해당 head 파일, `group-00129.json`~`group-00146.json` 및 해당 NPZ, 최근 `diagnostics/invocation-*.json`과 `usage/` 최신 스냅샷을 회수한다. 진단에 GPU·VRAM·Torch·CUDA·FP32, 마운트 포함 경과시간, 입력 준비·학습·평가·검증·게시 시간이 들어 있다. 강제 유실에서 마지막 진단이 없으면 영수증과 원장을 기준으로 상태를 해석한다.
- 각 seed의 `summary.json` → `measurement` 세 행을 CPU `summarize(trial_plan, measurements, phase="screening")`에 전달한다. CPU plan의 내용 SHA는 `7753d66479f5603a63816159d64a57fe050a31fdd1e2dd2279fce449d2185968`이다. 이 세 행을 다른 방식의 같은 seed 결과와 합쳐 선별한다. 평가 완료 전에는 점수가 생성되지 않는다.

## 동결 평가 계약

공식 validation 421클립·실제 16,425프레임에서 평가 seed 71, body/hand noise offset 100/200, 10-step DDIM, FP32·TF32 OFF를 적용한다. NPZ에는 sequence·clip 시작·frame 키, 시작 후 명목 시간, 관절 순서와 device mm 예측·정답 XYZ, validity mask, 손목 상대 손가락 좌표가 들어간다. 평가에는 padding을 포함하지 않는다. 세 시나리오의 body·hand-global MPJPE 6개 값을 동일 가중 평균한다. A 공식 결과와의 차이 및 같은 쪽 손목·손목 상대 손가락·전역 손가락·교차항과 공통 mask 표본 수를 계산한다. [v2.1.0 분해 정의](V210_EVAL.md)

## 로컬 검증

동결 128 소스와 발급 overlay를 풀어, 보존 A step 9 및 실제 train·validation feature를 CPU에서 검증했다. Train 클립 1개로 step 1을 저장한 뒤 step 2의 중단 없는 실행과 재개 실행에서 head와 손실이 정확히 일치했다. 동결 body 경로의 SHA도 보존됐다. 실제 validation 클립 1개에서 튜닝 body → 동결 hand 경로의 device-mm 예측 shape은 `47 × 62 × 3`이었다. 전체 3회 학습과 전체 validation 수치 및 CUDA 장치의 소요시간은 아직 실측 전이다.

발급은 저장소 루트에서 아래 명령을 사용한다. `--output`에는 새 폴더를 지정한다.

```powershell
python -m scripts.v211_build --source outputs/v2_plan_inputs/gpu_training_128_20261122/v2_gpu_training_128_v1.zip --a-bundle outputs/v2_plan_inputs/gpu128_A_portable_release/v2_128_A_final_model_and_cache.zip --previous-receipt outputs/v2_plan_inputs/v211_body_head_128_A_release_20261122/receipt.json --output outputs/v2_plan_inputs/v211_body_head_128_A_cuda_startup_r2_20261122
```

## T4 전체 학습 결과

2026-11-22 · R2, FP32, 세 seed 각각 3,097회 업데이트 및 421클립·16,425프레임 validation 완료. 종합은 세 환경의 body·hand-global 동일 가중 평균, 나머지는 전체 유효 관절의 pooled MPJPE(mm)다.

| seed | 종합 | body | 왼손목 | 오른손목 | 전역 손가락 | 손목 기준 손가락 |
|---|---:|---:|---:|---:|---:|---:|
| 부모 A | 219.72 | 164.00 | 244.76 | 234.50 | 276.13 | 93.72 |
| 42 | 209.26 | 150.27 | 227.14 | 220.44 | 267.95 | 95.70 |
| 71 | 207.12 | 148.46 | 225.74 | 217.80 | 266.01 | 95.87 |
| 103 | 211.68 | 150.55 | 236.48 | 218.60 | 272.05 | 94.91 |

세 회차 종합 평균 209.35 mm, 표본 표준편차 2.28 mm. seed 71이 이 방식 내 validation 최저다. 다음 비교는 같은 A에서 시작하는 v2.1.2이며 세 방식 중 최종 선정은 대기 중이다. 총 1,161.06초, 부모 준비 41.49초, 147개 특징 그룹 복원 283.74초였다.

원본은 `outputs/v2_plan_inputs/v211_r2_t4_results_20261122/`에 보존했다. invocation 파일 SHA-256 `60729b6e8f8221fb5b01b0cf431e79268eee0a3b59124f029c4ef027f13e5fd4`. seed 순서 42/71/103의 summary 파일 SHA는 `32d9f1f61e140003dea811663ba5e6bb6e46226159be2f4097f83e50dd0b3352`, `e88911c078a1d6d0fec3db843114f08dc84c338592dfe81bfa3d437f46e0b4f3`, `bb46ceeb2a545b4bf26f7b0c3838ceb4732ea2d7711ce0425b001ee191504507`이다. 요약 봉인과 점수가 진단과 일치한다. 최종 head·fit과 예측 그룹의 별도 회수는 진행 중이다.
