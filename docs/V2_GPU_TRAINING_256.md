# v2 GPU 학습 · 256 전체 캐시

2026-11-23 · 이미 진행 중인 256 A/B 학습은 [병렬 입력 교체 노트북](V2_GPU_TRAINING_256_PARALLEL.md)에서 슬롯별 최신 Drive 체크포인트로 이어갈 수 있다. 아래 내용은 기존 순차 로더에 적용한다.

2026-11-22 기준. 128 고정본 `b7a2d00`과 같은 공식 manifest·split을 사용한다. 세 슬롯은 seed A(42)·B(71)·C(103)로 새로 초기화한다. 슬롯당 frontend·body·hand 각각 3 epoch, frontend batch 2와 body/hand batch 1, 합계 197,961 updates를 고정한다. 순서는 128과 같은 seed별 셔플이다. Train 119,586페어·3,097클립과 validation 16,425페어·421클립을 사용한다. 손 모델은 학습에서 GT 상체, 검증과 시각화에서 best body의 예측 상체를 입력받는다. 평가 clip·노이즈 seed 71·body/hand offset 100/200·순위 산식은 128과 같다. [128 기준 실행](V2_GPU_TRAINING_128.md) · [v2 계획](V2_PLAN.md)

## 실행 순서와 보관

1. 인계받은 `v2_gpu_training_256_v1.ipynb`, `v2_gpu_training_256_v1.zip`, `receipt.json`의 SHA를 확인한다. 각 GPU Colab에서 ZIP을 `/content`에 올리고 `DRIVE_PROJECT`를 확인한다. 노트북 실행 스위치는 기본 OFF다. 한 호출은 최대 8시간이다.
2. 먼저 A의 스위치를 켠다. A가 실제 본학습에서 첫 유효 업데이트를 만들면 같은 호출에서 체크포인트 ZIP을 Drive에 게시하고 readback한다. `reports/v2_gpu_training_256_v1/A-seed-42/checkpoints/first-update.json`이 보인 뒤 별도 GPU VM에서 B·C를 시작한다. B·C는 그 영수증의 256 실행 ID·체크포인트 SHA·optimizer 상태를 직접 확인한다. A의 영수증 게시 도중 VM이 끊겼다면 같은 A를 먼저 재개한다. 검증된 첫 체크포인트로 시작 조건을 복구한다.
3. CUDA가 동작하는 A100·T4·L4 등을 선택한다. 장치명, VRAM, PyTorch, CUDA runtime, FP32를 기록한다. 256의 batch·precision은 기기별로 자동 조정하지 않는다. T4/L4의 실제 메모리 여유와 처리량은 첫 운영 호출에서 확인한다. 다른 GPU를 사용하는 새 VM도 소프트웨어 binding이 일치하면 재개할 수 있다. 하드웨어에 따른 수치 차이는 seed 결과와 함께 표시한다.
4. 종료 시 슬롯의 `diagnostics/invocation-*.json`, `checkpoints/step-*.json` 최신 영수증과 완료 시 `comparison.json`·비교 ZIP을 회수한다. Drive 진단이 누락됐으면 슬롯 화면에 표시된 로컬 `result.json`·`worker.log`를 VM이 유지된 동안 회수한다.

입력은 `reports/cache/v2_cache256_041837Z`의 읽기 전용 샤드 1,063개·136,011페어다. 마지막 75페어 샤드와 체인 경계 `2ae0b8523c89590e48d3363b26fcca9a18cf387308b6871b1a253b8b0849e33f`를 확인한다. 완료 보고의 바이트는 214,229,436,279B(약 199.5 GiB)다. 새 쓰기는 슬롯별 `A-seed-42`, `B-seed-71`, `C-seed-103` 하위의 잠금·원장·체크포인트·특징·평가 진행·프레임 인덱스에 한정한다. 이미 발급한 128 파일과 운영 경로는 유지한다.

## 디스크·I/O와 재개

256 ZIP에 처음 접근할 때 전체 전송 SHA와 내부 이미지·pose 파일 SHA를 확인한다. 원본 ZIP에 저장된 NPY 데이터의 오프셋과 float32 프레임별 SHA를 슬롯의 `frame-index`에 게시하고 즉시 다시 읽어 확인한다. 인덱스가 있으면 새 VM에서 필요한 프레임 1.5 MiB와 필요한 pose 구간을 직접 읽어 각각 SHA를 확인한다. 로컬 ZIP·프레임을 재사용할 때도 대상 바이트를 해시한다. 이 방식은 슬롯별로 한 샤드씩 인덱스를 만들고 원래 배치·셋 순서를 보존한다.

| 항목 | 상한·검증 범위 |
| --- | --- |
| 로컬 이미지 캐시 | 최대 24 GiB, 디스크 여유 7 GiB를 보존하며 오래 사용하지 않은 로컬 사본부터 제거한다. 처음 시작할 때 가용 디스크 8 GiB 이상을 확보하면 작은 캐시로 시작할 수 있다. 가용 31 GiB 이상이면 24 GiB 창을 사용할 수 있다. |
| 읽기·쓰기 | 호출당 논리 예약 읽기 512 GiB, 쓰기 24 GiB. 입력 읽기는 최종 검증 ZIP용 8 GiB를 남기는 지점에서 중단한다. ZIP 전체 읽기와 프레임 구간 읽기, 로컬 캐시 사용량을 별도로 보고한다. |
| 결과·특징 | 로컬 특징 2 GiB, run 2 GiB, 복구 ZIP 1 GiB, checkpoint 256 MiB. 슬롯별 특징 그룹과 분할 validation 결과를 재사용한다. 최근 체크포인트 ZIP 두 개 및 별도 A 최초 업데이트 ZIP을 보관한다. |
| 시간 | 한 호출 최대 28,800초. 계산은 마지막 1,320초 전에 새 작업을 멈춘다. 읽기나 시간 상한이 먼저 오면 마지막 정상 cursor를 Drive에 저장한다. |

세 frontend epoch의 모든 train·validation 픽셀을 한 번씩 읽는 단순 합계는 슬롯당 약 598 GiB, 최종 frozen frontend 특징 생성은 약 199 GiB다. 각 슬롯의 전체 ZIP 최초 검사 약 199.5 GiB가 추가될 수 있다. 로컬 재사용 전 논리량의 규모이며, Drive 마운트가 내부적으로 가져오는 네트워크 블록과 VM 디스크 사용량은 Colab에서 달라질 수 있다. 세 GPU 동시 접근의 영향, 실제 GPU 누적 시간과 Colab 비용은 미측정이다. 호출은 매번 수동으로 시작한다.

같은 슬롯에서 재호출하면 최신 검증된 Drive 영수증을 사용해 모델·optimizer·scheduler·scaler·RNG·batch 위치를 복원한다. `EXPECTED_RESUME_TIP`을 채웠다면 최신 영수증 SHA여야 한다. VM 유실로 `.writer`가 남으면 이전 VM과 worker가 종료된 것을 확인하고 해당 owner 값을 `RECOVER_WRITER_OWNER`에 한 번 입력한다. 중단 원장에는 최대 28,800초 예약이 기록되며, Drive에 마지막으로 보관한 업데이트 이후의 연산은 다시 진행될 수 있다.

## 2D·3D 평가와 시각화

256 학습·평가 모두 실제 256×256 float32 스테레오를 사용한다. 2D 좌표는 생산 캐시와 같은 양 끝점 정렬을 사용한다. Teacher 입력 XY는 해상도에 맞춰 `2 × xy / (size − 1) − 1`로 정규화한다. Frontend 점수는 128·256 모두 `(1024 − 1) / (size − 1)` 비율로 원본 1024×1024 픽셀에 환산한다. 3D는 같은 `device_mm`와 camera origin·mask를 사용한다.

Heatmap loss는 양쪽 모두 grid sigma 1.5를 사용한다. 실제 grid는 128 입력에서 64×64, 256 입력에서 128×128이다. 따라서 원본 기준 sigma는 약 24.4px에서 12.1px로 줄어든다. Heatmap의 공간 분해능과 confidence 분포가 함께 변할 수 있으므로 결과는 해당 해상도 파이프라인 전체의 비교로 해석한다.

비교 ZIP은 시나리오별 같은 50프레임의 좌우 256px RGB/2D overlay와 카메라 XY 평면 3D 투영을 담은 768×276 PNG·GIF, 256 입력 픽셀·원본 픽셀·3D mm와 mask를 담은 NPZ를 포함한다. Seed별 `macro_score_mm`는 세 시나리오 body와 hand global MPJPE를 같은 비중으로 평균한다. 세 점수와 범위, hand local MPJPE와 뼈 길이를 함께 확인한다.
