# v2.4 · x0/v 수치 검사와 짧은 학습 파일럿

2026-11-23 · A → v2.4 GPU 파일럿 완료. L4 FP32에서 seed 42, x0/v 각 24회 업데이트와 여섯 validation 클립(300프레임) 평가까지 65.70초였다. 0·12·24회 몸 오차(mm)는 x0 662.35 → 245.84 → 363.51, v 493.23 → 415.15 → 440.40이다. 두 조건의 학습 noise 일치를 확인했다. 두 곡선 모두 12회 이후 반등해 본시험 우열은 미정이다. 원본 아홉 파일과 실측을 첨부한 계획 초안은 `outputs/v2_plan_inputs/v24_gpu_pilot_results_20261123/`에 보존했다.

2026-11-23 · v2.4 Colab CPU 수식 검사가 35.32초에 완료됐다. 합성 float64 입력에서 x0와 noise 복원 최대 절대 오차는 각각 4.44e-16, DDIM 10단계 출력 차이는 0이었다. 시간 mask와 중간 head의 x0 출력도 보존됐다. 결과 원본 3개의 SHA-256을 검사해 outputs/v2_plan_inputs/v24_numerical_results_20261123/에 보존했다. result.json SHA-256은 284dbd8abb5f821102e4d611be6f0c13a5bf653b6499a95f699540e5f8549289다. GPU 학습·검증 정확도는 순차 시험 대기다.

2026-11-23 · GPU 파일럿용 R2는 호출당 논리 쓰기 상한 4GiB와 Drive 게시 여유 360초를 적용한다. 기존 CPU 수식 결과의 증거는 위 결과를 따른다.

고정 v2-128-A를 부모로 사용한다. 첫 파일럿은 seed 42에서 x0와 v 두 arm을 각각 24회 업데이트한다. 이 값은 구현 경로·수렴 초기 신호·실제 비용을 확인하는 제한이다. 본 비교의 seed별 학습량과 예산은 파일럿을 검토한 뒤 따로 고정한다. [v2 튜닝 설계](V2_TUNING_DESIGN.md)

## 대조 조건

| 항목 | 두 arm의 고정 조건 |
| --- | --- |
| 초기화 | A frontend e3, body e3, hand e2를 복원한다. 두 body를 독립 strict load하고 final head weight/bias만 같은 CPU 난수 시드와 `U(−1/√512, 1/√512)`로 재초기화한다. 전체 body 초기 SHA와 final head SHA가 같아야 한다. 호환 backbone과 중간 frame head는 A에서 가져온다. |
| 중간 head | Spatial 처리 직후 나오는 중간 frame head는 두 arm 모두 x0를 예측한다. 같은 초기 가중치·입력에서 adapter 전후 frame 출력 일치, A frame 파라미터 유지, 두 head의 첫 유효 gradient를 확인한다. |
| 학습 | Body 전체를 FP32 Adam, batch 1, LR 5e-5로 각각 24회 업데이트한다. 순서·step별 k·epsilon을 공유하고 SHA를 대조한다. Frontend와 hand는 가중치를 고정한다. |
| 입력 | Train에서 시나리오별 8클립을 seed 기반 해시 정렬로 고정한다. 같은 순서로 24클립을 한 번 사용한다. Validation은 시나리오별 첫 두 full-50 클립, 총 6클립·300프레임이다. 시나리오별 첫 클립은 이전 A 비교의 고정 키와 일치해야 한다. |
| 평가 | 각 arm의 0·12·24 update에서 같은 validation 클립을 평가한다. 평가 noise seed 71, body/hand offset 100/200, 결정적 DDIM 10을 유지한다. Frozen hand는 각 arm body의 예측 상체를 받는다. Device-mm body·hand-global/hand-local MPJPE와 손목·표본 수를 기록한다. |

초기 raw head 텐서는 두 arm에서 같다. v 출력의 x0 변환에는 noisy 상태와 timestep이 들어가므로 초기 예측 점수는 각각 기록한다. Frontend·hand의 실제 tensor SHA와 선택 특징 텐서 SHA는 학습 전후 대조한다. 본 비교의 공식 validation 전체는 421클립·16,425프레임이다. 파일럿의 6클립은 별도 범위이며 test split은 사용하지 않는다.

## 변환·손실 계약

`a = sqrt(alpha_bar[k])`, `b = sqrt(1-alpha_bar[k])`, `x_t = a*x0 + b*epsilon`이다. v 표적은 `v = a*epsilon - b*x0`, 기존 sampler에 넣는 clean 좌표는 `x0_hat = a*x_t - b*v_hat`다. 기존 schedule·timestep·mask 및 DDIM 업데이트를 사용한다. Adapter는 중간 frame 출력에 변환을 적용하지 않는다.

두 arm의 최적화 목표는 기존 masked x0 위치 MSE + A의 frame-x0 MSE × 1.0이다. 기존 100mm scale로 정규화하고 `target_valid ∩ time_valid`의 유효 관절 및 xyz 성분을 평균한다. v arm의 최종 위치 항은 `mean((1-alpha_bar[k])*(v_hat-v)^2)`와 수치가 일치해야 한다. Raw v MSE는 진단에 남기고 별도 학습항의 가중치는 0으로 고정한다. 시간 미분 및 foot loss는 사용하지 않는다. Float64 합성 검사는 여러 timestep의 x0·epsilon 왕복, weighted loss와 공통 mask, DDIM10 oracle 출력, padding을 확인한다.

## 고정 입력·회수

동결 source ZIP `v2_gpu_training_128_v1.zip` SHA-256 `9a68d8e45bcf3abf6fab3004740faa6805d4ce3fda5118c3a7102011937849d8`. A 최종 step 9의 receipt content SHA `7d2cf12cdaf42916d98190cf2d57e11e93a6d94165d5e753db6fc1d11c3026d4`, 캐시 경계 `dc79219153d69f44020e0c42f9f3ede6cd3add179211f6c300345b2416dab511`. 기존 Drive의 A 결과·경계 메타데이터와 선택 특징 그룹을 읽는다. 이전 발급본의 helper는 코드 SHA를 고정해 사용한다. v2.3 GPU 산출물은 입력에 포함되지 않는다.

1. 검토 후 Colab에서 `v24_diffusion_pilot_128_A_r2_20261123.ipynb`를 열고 source ZIP과 `v24_diffusion_pilot_tools_r2.zip`을 `/content`에 올린다. 발급 `receipt.json`에서 크기와 SHA를 확인한다.
2. 기본 실행 스위치는 `RUN_V24 = False`다. R2의 기본 모드는 `MODE = "pilot"`이다. 승인된 호출에서 CUDA GPU 한 대와 `DRIVE_PROJECT`를 확인하고 스위치를 켠 뒤 두 코드 셀을 실행한다. Worker는 파일럿 시작 전에 수식 검사를 같은 호출에서 확인한다.
3. 결과는 `reports/v24_diffusion_pilot128_a_r2/run-<mode>-<UUID>/`의 새 경로에 보관한다. 마지막에 출력된 `result.json`의 경로와 SHA를 회수한다.

| 파일 | 내용 |
| --- | --- |
| `numerical.json` | 변환·loss·DDIM·mask 검사 |
| `pilot-plan.json` | A·입력·seed·train/validation 키·선택 그룹 pin |
| `x0-pilot.json`, `v-pilot.json`, `pilot-summary.json` | 학습·평가 곡선, 공통 noise, upstream 동결 증거 |
| `x0-pilot-final.pt`, `v-pilot-final.pt` | body, Adam·scheduler·scaler·RNG·step을 담은 마지막 파일럿 상태 |
| `worker-result.json`, `result.json` | 상태, 예산·시간, 결과물의 크기·검증 SHA |

완료 판정은 `completed_numerical` 또는 `completed_pilot`와 해당 파일 모두의 verified readback이다. 실패 시 이미 완성된 파일을 가능하면 같은 UUID에 보존한다. VM 종료 시 `started.json`만 남을 수 있다. 재실행은 새 UUID에서 시작한다. 체크포인트 재개와 본 비교는 별도 인계 대상이다.

## 비용과 예산 근거

- Pilot 호출은 이전과 같은 2,700초(45분)이며 마운트 최대 120초를 포함한다. Worker는 마지막 480초에 새 작업을 시작하지 않는다. Controller는 마지막 360초 전까지 기다리므로 worker의 전체 종료 처리를 위해 120초가 먼저 있다. 강제 종료 신호가 필요한 경우 최대 25초를 이 360초 안에서 사용한다. 나머지는 Drive 복사·SHA readback을 시도한다. Numerical 모드의 상한은 300초다.
- 총 48 body 업데이트와 `2 arm × 3평가점 × 6클립 × DDIM10 × body/hand = 720` denoiser forward를 계획한다. 기존 T4 공식 validation 1회는 421클립·8,420 forward·준비 포함 약 19.35분이었다. 파일럿 forward 수는 약 8.6%이며 부모 복구, 업데이트, 최종 상태 두 개 보관 비용은 같은 비율로 줄지 않는다.
- 주요 읽기 입력은 A 복구 ZIP과 선택 특징 그룹 최대 30개다. Worker 논리 읽기 상한 2GiB와 로컬 여유 4GiB를 적용한다. 로컬 특징 스크래치와 Drive의 물리 I/O는 논리 쓰기 예약과 별도로 본다.

### R2 쓰기 예약

| 예약에 포함하는 쓰기 | 상한 |
| --- | ---: |
| A 복구 ZIP 로컬 사본 | 435,178,624바이트 |
| 복구 ZIP 풀기 | 1GiB |
| x0·v 최종 상태 두 개를 로컬에 저장 | 384MiB |
| 로컬 JSON 최대 여섯 개 | 12MiB |
| Drive 최종 상태 사본 두 개와 JSON·영수증·오류 로그 | 415,399,936바이트 |
| 전체 계획 합계 | 2,339,556,480바이트 |
| 호출당 논리 쓰기 상한 | **4,294,967,296바이트(4GiB)** |

로컬 CPU에서 실제 직렬화한 상태는 arm당 180,257,563바이트였다. GPU 파일은 상태당 192MiB를 예약한다. Drive 사본 두 개의 SHA readback은 최대 384MiB의 **읽기**이며, 보고서 readback도 각 보고서 크기로 제한한다. ZIP 복구 시에도 사본 검증은 읽기로 계산한다. Worker ledger의 쓰기 상한은 로컬 보고서와 Drive 게시 예약을 뺀 3,866,984,448바이트다. 전체 계획 대비 여유는 1,955,410,816바이트다. 발급 영수증이 A의 체크포인트 핀과 기존 CPU 상태 크기 근거를 보존하고, worker가 실행 전에 이 산식을 다시 확인한다.

## 로컬 발급

```text
python -m scripts.v24_build --source <고정-128-source.zip> --a-bundle <로컬-A-묶음.zip> --parent-release <고정-v23-발급-폴더> --output <새-발급-폴더>
```

생성기는 A ZIP에서 고정 메타데이터를 읽는다. Colab 입력은 source와 toolkit이다. 본 비교는 파일럿의 수렴 초기 신호·동결 증거·시간·저장 비용을 검토한 뒤 동일 업데이트 예산과 독립 seed 계획부터 고정한다.
