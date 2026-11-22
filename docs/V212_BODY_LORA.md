# v2.1.2 · 128 A body 시간축 QKV LoRA

A의 frontend e3, body e3, hand e2에서 각 seed를 독립적으로 시작한다. Body의 `temporal.0.qkv`, `temporal.1.qkv`, `temporal.2.qkv`에만 rank 4, alpha 4, dropout 0 LoRA를 연결한다. 각 fused QKV는 기존 `Equation4Block`의 linear projection에서 `W + (alpha/rank) × (up @ down)`을 사용한다. `down`은 학습과 분리된 seed에서 초기화하고 `up`은 0으로 초기화한다. 세 층의 어댑터 6개 텐서, 총 24,576개 값만 Adam이 업데이트한다. 원래 body 텐서와 bias는 그대로 보관하며 등록 전·후와 최종 체크포인트 저장 전 SHA-256을 검사한다. 학습된 head나 이전 trial의 결과를 로드하지 않고 A의 고정 body에서 시작한다. [베이스 튜닝 설계](V2_TUNING_DESIGN.md)

## 통제 조건

v2.1.1과 같은 `order`·`update` 함수를 호출한다. Seed 42·71·103 각각은 공식 train 119,586페어·3,097클립을 1epoch, batch 1, Adam 5e-5로 학습한다. 같은 seed에서 clip 순서와 teacher timestep·epsilon stream, motion loss, 원본 예측 2D가 일치한다. 재개와 방식 간 비교에는 동일 GPU 종류와 Torch/CUDA 버전을 우선한다. 장치와 소프트웨어가 바뀌면 기록에 남는다.

공식 validation 421클립·16,425실제 프레임에서 공통 평가 seed 71과 body/hand noise offset 100/200, FP32·TF32 OFF, DDIM 10을 적용한다. 동결 hand에는 각 trial body의 예측 상체를 넣는다. 패딩을 제외하고 sequence·clip 시작·frame으로 키를 붙인 device-mm 좌표와 정답·mask·상대 손가락 좌표를 저장한다. 공식 결과는 세 시나리오의 body·hand-global MPJPE 6개를 동일 가중 평균한다. A 대비 차이, 좌우 손목·손목 상대 손가락·전역 손가락·공통 mask 표본 수·교차항을 함께 산출한다. [분해 정의](V210_EVAL.md)

## 고정 Colab과 재개

1. CUDA 시작 수정 R2 발급 위치 `outputs/v2_plan_inputs/v212_body_lora_128_A_cuda_startup_r2_20261122`의 `v212_body_lora_128_A_cuda_r2.ipynb`와 `receipt.json`을 확인한다. 지원되는 CUDA GPU 1개를 선택하고 기존 `v2_gpu_training_128_v1.zip`과 새 `v212_body_lora_tools_r2.zip`을 Colab `/content`에 올린다. 오케스트레이션이 인계한 실행에서 `DRIVE_PROJECT`를 확인하고 `RUN_V212 = True`로 바꾼다. 기본값은 `False`다. [CUDA 시작 수정·발급 SHA](V2_TUNING_CUDA_STARTUP_R2.md)
2. Colab은 원래 Drive 프로젝트의 A step 9 체크포인트와 feature ZIP 147개를 검증하고 호출별 스크래치에 복원한다. 원본 cache는 최종 경계의 메타데이터를 검증한다. 로컬 51 GiB A 반출 ZIP은 오프라인 발급에 사용한다. 로컬 여유 4 GiB 이상, 1회 8시간, 종료 여유 5분, 호출 논리 읽기 8 GiB·쓰기 2 GiB를 적용한다. 각 호출의 GPU·VRAM·Torch/CUDA와 준비·학습·평가·검증·게시 시간을 기록한다. 실제 GPU 시간과 비용은 첫 실행에서 확인해야 하며 Colab VM은 상한보다 일찍 종료될 수 있다.
3. `reports/v212_body_lora128_a_v1`의 단일 writer로 seed별 `seed-42`, `seed-71`, `seed-103`에 저장한다. 첫 update·이후 50회 경계·마지막 3,097회, 또는 여유 시간에 도달할 때 확보한 완료 step에서 어댑터와 optimizer·scheduler·scaler·RNG를 검증 게시한다. 수동 재호출은 부모 A를 다시 불러온 뒤 최신 fit의 SHA와 원본 body fingerprint를 검증하고 다음 clip부터 재개한다. 이미 게시된 평가 그룹의 좌표 NPZ와 프레임 키도 다시 검증한다. 완료 seed는 검증된 최종 fit과 요약을 확인하고 건너뛴다. VM 유실 뒤 잠금이 남으면 이전 VM 종료를 확인한 뒤 표시된 기존 owner를 `RECOVER_WRITER_OWNER`에 넣는다. Drive 원장은 미정산 호출의 28,800초 예약을 남긴다.

## 회수할 파일

- 각 seed의 `fit-03097.json`, 해당 `lora-03097-*.pt` 및 `summary.json`. 1epoch 평가이므로 이 어댑터가 해당 seed의 epoch best이자 last다. 어댑터 파일은 원래 A body의 SHA에 결합되며 전체 body를 담지 않는다. 해당 A checkpoint를 함께 보관한다.
- 각 seed의 `group-00129.json`~`group-00146.json`과 대응하는 NPZ, `diagnostics/invocation-*.json`, `usage/` 최신 스냅샷. 그룹의 영수증이 NPZ의 정확한 이름과 SHA를 가리킨다.
- `summary.json`의 `measurement` 세 행과 고정 `trial-plan.json`. 계획의 내용 SHA는 `7753d66479f5603a63816159d64a57fe050a31fdd1e2dd2279fce449d2185968`이며 CPU `summarize(trial_plan, measurements, phase="screening")`의 v2.1.2 행 형식에 일치한다. 다른 방식의 같은 seed 결과를 회수한 뒤 선별한다.

## 최초 발급본·로컬 검증

최초 발급본은 `outputs/v2_plan_inputs/v212_body_lora_128_A_release_20261122`에 보존되어 있다.

| 파일 | SHA-256 |
| --- | --- |
| `v212_body_lora_128_A_cuda.ipynb` | `52e43a5ff5ca8898723725406fee5f84565f37ee827456cf50a36109df973754` |
| `v212_body_lora_tools.zip` | `199350e3a7d81a4344fda0633f8f92b3ff774ce89f8eba1a5a51349f20d32c47` |
| `receipt.json` | `9761c66695595e264ab1e6cf9474f5ea996973e966e33496f094dd0e3eb0d68e` |
| `trial-plan.json` | `03f8c3dea1d825e4870580d4f0e69850d7f1a814cdbb2ac6618549583973e3ef` |

발급 overlay와 동결 원본을 풀어 실제 A step 9, train clip 2개, validation clip 1개로 CPU에서 확인했다. LoRA 등록 직후 고정 noise로 예측한 body는 A와 비트 단위로 같았다. 첫 update의 세 `up` gradient L2는 `0.134793`, `0.030856`, `0.079925`였다. 업데이트 1에서 파일을 게시하고 새 teacher에 복원한 뒤, 중단 없는 업데이트 2와 손실·어댑터가 정확히 일치했다. 베이스 텐서 SHA `f281f09ccb5a1932780715b027058c8c29fda1a1b4751cba5501ee568bf4748d`도 동일했다. Validation 예측은 `47 × 62 × 3`이고, 두 업데이트 뒤 A 대비 최대 축별 차이는 body `0.377441 mm`, hand-global `0.428345 mm`였다. 이 표본은 학습 반영과 복원을 확인하는 증거다. 전체 CUDA 실행과 검증 정확도는 미측정이다.

R2를 다시 발급할 때는 저장소 루트에서 아래 명령을 사용한다. 최초 v2.1.2 발급본의 소스는 `590780b`다.

```powershell
python -m scripts.v212_build --source outputs/v2_plan_inputs/gpu_training_128_20261122/v2_gpu_training_128_v1.zip --a-bundle outputs/v2_plan_inputs/gpu128_A_portable_release/v2_128_A_final_model_and_cache.zip --previous-receipt outputs/v2_plan_inputs/v212_body_lora_128_A_release_20261122/receipt.json --output outputs/v2_plan_inputs/v212_body_lora_128_A_cuda_startup_r2_20261122
```

## T4 검증 결과

R2는 FP32에서 917.06초가 걸렸다. 각 seed에서 같은 부모·3,097 업데이트·공식 validation 421클립 16,425프레임을 사용했다. 종합 점수는 세 환경에 동일 가중을 주며, 몸·손 점수는 전체 유효 관절을 합쳐 구한 MPJPE(mm)다.

| seed | 출력층 종합 | LoRA 종합 | LoRA 몸 | LoRA 전역 손가락 | LoRA 손목 기준 손가락 |
|---|---:|---:|---:|---:|---:|
| 42 | 209.26 | 210.47 | 149.04 | 270.90 | 96.36 |
| 71 | 207.12 | 209.65 | 148.72 | 270.21 | 96.80 |
| 103 | 211.68 | 210.05 | 149.38 | 270.25 | 96.52 |

LoRA 평균 210.05 mm(표본 표준편차 0.41), 출력층 평균 209.35 mm(2.28), A 219.72 mm. 세 seed 모두 부모보다 종합 오차가 낮고 상대 손가락 오차는 높다. 두 방식의 종합 평균 차이는 0.70 mm이며, 세 번째 방식 평가 전 최종 선별은 보류한다.

진단과 세 summary 및 어댑터 실물은 `outputs/v2_plan_inputs/v212_r2_t4_results_20261122/`에 보존했다. 진단 파일 SHA-256은 `120b1f70d7f84ad592456abd13cca5f5716d931c977d66ed888d4913a1d4654e`다. summary의 봉인과 가중치 해시를 진단·실물과 대조했다. 세 최종 fit 영수증도 보존했고 각각 3,097번째 step과 어댑터 SHA가 summary 및 파일 실물에 일치했다. 그룹별 좌표·영수증은 별도 회수 대기다.
