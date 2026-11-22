# v2.2 · A hand의 GT·예측 상체 조건 비교

v2-128-A의 frontend e3와 body e3을 고정한다. 두 조건 각각에서 seed 42·71·103을 A의 hand e2 가중치로 새로 시작한다. Hand 전체 103개 파라미터 텐서, FP32 값 15,571,974개를 학습한다. 각 trial은 공식 train 3,097클립·119,586페어, 1epoch, batch 1, Adam `5e-5`다. 첫 세 seed만 발급했다. [튜닝 설계](V2_TUNING_DESIGN.md)

| 조건 | seed마다 적용되는 학습 상체 |
| --- | --- |
| `gt_100_pred_0` | GT 3,097클립 |
| `gt_50_pred_50` | GT 1,549클립과 동결 A의 예측 1,548클립 |

혼합 조건은 seed와 clip의 sequence·시작 frame을 해시해 순위를 매기고 앞 1,548클립을 예측으로 선택한다. 한 클립의 모든 프레임은 같은 상체 입력을 사용한다. 조건을 고르는 해시는 셔플·teacher timestep·epsilon과 별도이며, 같은 seed의 두 run은 같은 clip과 학습 noise를 사용한다. GT 상체는 기존 관절 validity를, 예측 상체는 고정 body SHA와 feature의 유효 시간 mask를 사용한다. 학습의 손목 기준 손가락 정답과 loss mask, 원래 스케일 및 loss는 두 조건에서 동일하다.

## 고정 Colab과 상체 캐시

1. `outputs/v2_plan_inputs/v220_hand_upper_128_A_verified_20261122`의 `v220_hand_upper_128_A_cuda.ipynb`를 연다. 기존 `v2_gpu_training_128_v1.zip`과 새 `v220_hand_upper_tools.zip`을 `/content`에 올린다. CUDA GPU 한 대와 스크래치 여유 4 GiB 이상을 확보한다. 오케스트레이션이 승인한 호출에서 `DRIVE_PROJECT`를 확인하고 `RUN_V220 = True`로 바꾼다. 기본값은 `False`다.
2. CUDA R2 초기화와 단계별 traceback을 사용한다. A checkpoint, 128 cache 경계와 기존 예측 2D feature 그룹 147개를 검증해 읽는다. A의 DDIM 10-step, noise seed 71·body offset 100으로 세 seed의 예측 train 합집합과 공식 validation 421클립의 body 좌표를 계산한다.
3. `reports/v220_hand_upper128_a_v1/predicted-body`에 clip 키, FP32 device-mm 좌표와 time mask를 그룹별로 보관한다. 공식 키에서 계산한 합집합은 train 2,718클립이며 validation 421클립을 더한 좌표 tensor는 41,434,800 B다. 부모·feature·assignment·seed·그룹 키·파일 SHA·shape·mask를 검증한 뒤 재사용한다. 학습에는 예측 body의 13개 상체 관절을 쓰고, 평가 때는 같은 캐시의 body 좌표를 손목 연결과 관절 비교에도 사용한다. 비교·재개에는 같은 GPU 및 Torch/CUDA 환경을 우선한다. 그룹을 생성한 환경은 각 영수증에 기록한다. [CUDA R2](V2_TUNING_CUDA_STARTUP_R2.md)
4. 같은 seed에서 `gt_100_pred_0`과 `gt_50_pred_50`을 순서대로 진행하고 다음 seed로 넘어간다. 동일한 고정본으로 수동 재호출하면 이미 검증한 상체 그룹, hand fit과 validation 그룹을 읽고 남은 작업을 이어간다. 중단된 작업의 이전 VM이 종료된 후 writer가 남아 있으면 출력에 표시된 owner를 `RECOVER_WRITER_OWNER`에 입력한다.

## 예산과 관측 범위

- 호출당 마운트 포함 8시간, 새 작업 시작 제한 5분, 논리 읽기 8 GiB·쓰기 2 GiB다. A checkpoint 복사 435,178,624 B 및 추출 예약 1 GiB를 빼면 호출당 새 산출물 쓰기 예약 여유는 638,563,200 B다. 기능 준비에서 A feature ZIP 778,972,869 B를 읽는다.
- Hand fit은 step 1·1550·3097과 시간 종료 시 안전한 완료 step에서 전체 hand·Adam·scheduler·scaler·RNG를 저장한다. 최대 192 MiB와 복사·읽기 검증 210초를 적용한다. 다음 fit 또는 평가 그룹을 게시할 용량이 없으면 그 단위 시작 전에 일시정지한다. 실제 상태 크기는 약 187 MB이며, 정규 hand fit 18개가 약 3.36 GB라 기존 I/O 계약에서는 **최소 6회 수동 호출**이 필요하다. 예측 캐시·평가·시간 경계 체크포인트에 따라 추가 호출이 발생할 수 있다. VM 유실 뒤 최대 1,549 update를 다시 계산할 수 있다.
- 보관량은 정상 완료 기준 약 3.6 GB로 추산한다. 기존 T4는 부모와 feature 준비에 약 58~325초, body 전체 3,097 update에 약 199초, body·hand 동시 validation에 약 189초였다. 상체 캐시가 validation 421클립에서 body를 공유한다는 점과 약 3,100클립의 첫 body 캐시 예측을 고려하면 계획용 GPU 벽시계 범위는 약 1~2시간이다. 이 값에는 각 호출의 준비·학습·평가·게시를 포함한 거친 추정이다. 새 hand 학습과 경우별 CUDA 장치의 시간·메모리 및 Colab 사용량은 실측 전이다. [T4 출력층](V211_BODY_HEAD.md#t4-전체-학습-결과) · [몸 전체 예산](V213_BODY_FULL.md)

## 평가와 회수

모든 trial의 공식 validation은 A의 예측 상체만 사용한다. 421클립·16,425 실제 프레임에 평가 seed 71, hand offset 200, DDIM 10, FP32·TF32 OFF를 적용한다. 좌표·GT·validity·프레임 키와 분해 항목을 18개 group NPZ에 저장한다. `comparison.json`에는 같은 seed별로 혼합−GT 차이, 세 시나리오의 hand-local·hand-global MPJPE, 동일 가중 macro, 세 paired 차이의 평균과 표본 표준편차를 기록한다. 같은 seed의 두 조건에서 frozen body 공식 지표가 일치해야 비교 파일을 게시한다. [지표와 분해](V210_EVAL.md)

회수 파일은 `predicted-body/index.json`과 그룹 영수증·NPZ, 각 조건·seed의 `fit-03097.json`과 해당 `hand-03097-*.pt`, `summary.json`, 평가 그룹 영수증·NPZ, 루트 `comparison.json`, 최근 `diagnostics`와 `usage`다. 조건별 전체 예측은 그대로 보관되고, 추가 seed나 기준 채택은 결과 검토 후 결정한다.

## 발급 SHA-256

| 파일 | SHA-256 |
| --- | --- |
| `v220_hand_upper_128_A_cuda.ipynb` | `61ecea63f0d1622fa7f24bfdcd096c49982a9a6190716c61f9e25d12ffebd05b` |
| `v220_hand_upper_tools.zip` | `dcc77a9f41730a4c53346f751f24bc51babecd3ae99639bfb3f86865885f9e0c` |
| `receipt.json` | `59945b7419a7272b748e72e6c8699c7f69ff9955361d6fcdac3972c9de41cd8e` |
| `trial-plan.json` | `03f8c3dea1d825e4870580d4f0e69850d7f1a814cdbb2ac6618549583973e3ef` |

동결 128 source와 위 overlay를 풀어 실제 A 최종 체크포인트와 feature로 CPU 검증했다. 실제 train 2클립·validation 1클립의 두 예측 그룹을 게시한 뒤, 재개에서 body 예측을 다시 계산하지 않고 SHA 검증으로 읽었다. 같은 클립 2개로 두 조건을 각각 step 1에서 쓰기 예산으로 멈춘 후 새로운 hand에서 step 2를 복원했다. 연속 실행과 모델·Adam 상태·손실이 두 조건 모두 정확히 일치했다. 두 조건의 checkpoint는 187,003,057 B와 187,003,377 B였다. 동일한 `47 × 62 × 3` 평가 경로에서 body 좌표는 일치했고 hand 좌표는 달랐다. 관련 회귀 24개와 기본 OFF, ZIP CRC 및 반복 발급 바이트 일치를 확인했다. 전체 CUDA 학습과 평가 점수는 미측정이다.

오프라인 발급은 저장소 루트에서 아래 명령을 실행한다. A 반출 ZIP은 이 단계에 사용한다.

```powershell
python -m scripts.v220_build --source outputs/v2_plan_inputs/gpu_training_128_20261122/v2_gpu_training_128_v1.zip --a-bundle outputs/v2_plan_inputs/gpu128_A_portable_release/v2_128_A_final_model_and_cache.zip --output outputs/v2_plan_inputs/v220_hand_upper_128_A_verified_20261122
```
