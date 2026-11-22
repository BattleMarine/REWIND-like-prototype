# v2 GPU 학습 · 128 전체 캐시

2026-11-22 사용자 승인에 따라 A(seed 42), B(71), C(103)를 독립 CUDA Colab에서 실행한다. 매 슬롯은 모델을 새로 초기화하고 frontend·body·hand를 각각 3 epoch 학습한다. 학습에는 공식 train 119,586페어·3,097클립을, 검증에는 공식 validation 16,425페어·421클립을 사용한다. 계획 업데이트는 seed당 frontend 179,379회, body 9,291회, hand 9,291회다. 학습 순서와 모델 초기화에는 슬롯별 seed를 쓴다. 세 슬롯의 검증 샘플링에는 공통 seed 71과 body/hand offset 100/200을 쓴다. [v2 계획](V2_PLAN.md)

## 시작과 순서

1. 오케스트레이션이 인계한 동일 revision의 `v2_gpu_training_128_v1.ipynb`, `v2_gpu_training_128_v1.zip`, `receipt.json`을 사용한다. 노트북마다 ZIP을 `/content`에 업로드한다. 설정 셀의 `DRIVE_PROJECT`는 기존 128 캐시를 보관하는 프로젝트 경로다.
2. A100·T4·L4 등 지원되는 CUDA GPU 런타임을 한 개 선택한다. 장치 이름, VRAM, PyTorch, CUDA runtime, FP32를 각 호출에 기록한다. 새 VM에서 재개하려면 Python·PyTorch·CUDA 등 체크포인트의 소프트웨어 binding이 일치해야 한다. 물리 GPU 변경은 허용하고 실제 장치 메타데이터를 보고한다. 서로 다른 장치의 학습 결과는 수치 차이까지 분리하는 seed 통제 실험이 아니므로, 장치 정보와 함께 비교한다.
3. 첫 VM에서 `SLOT = "A"`로 두고 `RUN_V2_TRAINING`을 켠다. 본학습에서 처음으로 유효한 supervision과 양의 유한 gradient가 있는 optimizer 업데이트가 나타나면, 같은 호출에서 검증된 체크포인트를 Drive에 저장한 뒤 학습을 이어간다.
4. `reports/v2_gpu_training_128_v1/A-seed-42/checkpoints/first-update.json`이 보인 뒤 독립 GPU VM 두 개에서 각각 B, C의 스위치를 켠다. B/C 진입점은 A의 영수증·ZIP SHA·optimizer state를 직접 확인해야 시작한다. A의 순서 조건이 아직 충족되지 않으면 그 호출을 종료한다. B/C를 시작한 뒤에는 각 슬롯을 독립적으로 재개할 수 있다.
5. 종료·일시정지·실패 시 각 슬롯의 `diagnostics/invocation-*.json`, 최신 `checkpoints/step-*.json`을 회수한다. Drive 진단이 없으면 셀에 표시된 로컬 `result.json`, `worker.log`를 VM이 살아 있는 동안 회수한다. 체크포인트 게시와 진단 게시 상태는 따로 확인한다.

## 데이터와 예산

원본은 128×128 캐시 `reports/cache/v2_bulk_cache_041837Z` 및 최초 샤드의 `reports/cache/v2_cpu_cache_041837Z`다. 체인의 마지막 경계 `dc79219153d69f44020e0c42f9f3ede6cd3add179211f6c300345b2416dab511`, 1,063개·136,011페어, 마지막 75페어를 확인한다. 로컬 복원 시 ZIP SHA와 샤드 안의 이미지·pose 파일 영수증을 검증한다. 바뀐 파일, 끊긴 체인, 범위를 넘는 샤드는 오류 처리한다. 128 캐시는 읽기 전용이다. 슬롯별 결과 폴더와 VM 로컬 파일에만 쓴다.

| 항목 | 슬롯당 상한 또는 추정 |
| --- | --- |
| 전체 호출 | 시작부터 28,800초. 계산은 마지막 1,320초 전에 새 작업을 멈추고 저장에 사용한다. |
| 호출 I/O | 읽기 예약 96GiB, 쓰기 예약 24GiB. Drive 전송, 사본 검증과 결과 패키징에 쓰는 보수적 논리 한도다. 로컬 샤드 mmap 읽기와 Drive 사업자 측 실제 전송량은 별도다. |
| 캐시와 로컬 디스크 | 소스 ZIP 합계 상한 52GiB. 캐시를 처음 복원할 때 실제 미복원 바이트에 7GiB 여유를 더한 로컬 가용 공간을 검사한다. 실측 완료 캐시는 약 50GiB이므로 첫 복원에는 약 57GiB가 필요하다. |
| 특징·체크포인트 | 슬롯별 2D 특징 최대 2GiB, 로컬 run 폴더 최대 2GiB, 복구 ZIP 최대 1GiB, 단일 체크포인트 최대 256MiB. 처음에 가용 RAM 4GiB 이상을 확인하며 worker RSS 상한은 7GiB다. |
| Drive 보관 | 슬롯당 최근 모델 복구 ZIP 두 개를 유지한다. 검증된 A 최초 업데이트 ZIP, 24클립 단위 특징 ZIP, 검증 진행 JSON과 비교 ZIP은 별도 보관한다. 중단이 게시 사이에 발생하면 검증 영수증이 없는 파일이 남을 수 있다. |

Frontend 학습은 기존 전체 무작위 순서를 유지하기 위해 샤드를 seed별 VM 디스크에 복원한 뒤 읽는다. 같은 VM에서 다시 호출하면 이미 복원한 샤드의 해시를 확인해 재사용한다. 새로운 VM에서 frontend 도중 재개하면 복원 읽기가 다시 발생한다. Body/hand로 넘어간 뒤에는 Drive의 검증된 예측 2D 특징 그룹을 재사용할 수 있고, 새 특징이나 비교 영상에 필요한 이미지 샤드만 추가로 읽는다. 슬롯별 외부 I/O와 학습·검증 시간은 각 호출에서 측정한다. 세 GPU의 동시 Drive 읽기나 진행 중인 256 캐싱 때문에 각 호출의 처리량이 달라질 수 있다. GPU 누적 시간이나 크레딧 상한은 별도로 지정하지 않았으며, 호출은 매번 수동으로 시작한다.

## 새 VM에서 재개

기존 슬롯과 같은 ZIP/노트북/설정을 사용한다. Worker는 해당 슬롯의 최신 Drive 영수증 체인과 지정된 ZIP을 확인하고, 모델·optimizer·scheduler·scaler·Python/Torch RNG·batch 위치·best 모델을 복원한다. `EXPECTED_RESUME_TIP`을 설정했다면 최신 영수증 SHA와 같아야 한다. 정상 중단은 같은 슬롯에서 새 호출 한 번으로 재개한다. 확인된 최근 보존 지점 뒤의 작업은 다시 계산될 수 있다.

VM 유실로 `.writer`가 남았다면 종료한 VM 및 worker가 더 이상 쓸 수 없는지 먼저 확인하고, 오류에 표시되는 이전 owner 값을 `RECOVER_WRITER_OWNER`에 한 번 입력한다. 장시간 호출은 기다린 뒤 자동으로 재시작하지 않는다. 중단 호출은 원장에 최대 28,800초 예약으로 기록된다. A가 모델 영수증과 gate 사이에서 끊겼다면 같은 A를 먼저 재개한다. A가 검증된 모델 영수증으로 gate를 다시 생성한다.

## 결과와 seed 비교

Frontend의 best checkpoint를 동결해 clip 특징을 만들며, 손 모델을 학습할 때는 GT 상체를 사용한다. 검증과 시연의 손 모델에는 best body의 예측 상체를 입력한다. 세 슬롯에 대해 같은 규칙으로 고른 시나리오별 50프레임 연속 클립을 저장한다. ZIP에는 좌우 RGB의 예측·GT 2D 및 3D 카메라 XY 투영을 보이는 GIF와 원래 3D 좌표·2D·mask·시간을 담은 NPZ가 있다. 명목 25FPS를 사용하며 실제 timestamp·stereo synchronization은 별도 검증 사항이다.

세 슬롯의 `comparison.json`을 회수해 `macro_score_mm`가 작은 순서로 비교한다. 이 값은 세 시나리오와 body/hand global MPJPE를 각각 같은 비중으로 평균한다. 모든 슬롯의 점수와 범위, hand local MPJPE, 예측/GT 평균 뼈 길이와 영상을 함께 확인한다. ZIP/결과를 회수하기 전에는 GPU 성능, 재개 성공 또는 모델 수렴을 확정하지 않는다.
