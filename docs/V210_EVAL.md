# v2.1.0 · A 모델의 전체 validation 손목·손가락 오차

대상은 v2-128-A의 frontend e3·body e3·hand e2다. 체크포인트는 최종 197,961 업데이트의 step 9, 영수증 SHA `7d2cf12cdaf42916d98190cf2d57e11e93a6d94165d5e753db6fc1d11c3026d4`로 고정한다. 공식 validation 421클립의 실제 16,425프레임을 평가한다. 학습 업데이트는 0회다. [v2 베이스 튜닝 설계](V2_TUNING_DESIGN.md)

## CUDA 호출과 중단 복구

`v210_eval_128_A_cuda.ipynb`의 실행은 기본 OFF다. 오케스트레이션이 검토해 인계한 `v210_eval_tools.zip`과 기존 `v2_gpu_training_128_v1.zip`을 `/content`에 올리고 CUDA GPU 하나를 선택한다. 128 A 51 GiB 로컬 반출 ZIP은 빌드 시 입력 핀을 추출하는 데 사용했다. Colab에서는 원래 Drive 프로젝트의 A step 9 복구 ZIP과 feature 그룹, 기존 manifest 및 캐시 경계 메타데이터를 읽는다. 새 이미지 캐시를 생성하거나 기존 이미지 ZIP을 통째로 복원할 필요가 없다. Validation이 들어 있는 129–146번 feature 그룹 중 129번의 train 클립 1개는 평가에서 제외한다.

`DRIVE_PROJECT`를 확인한 뒤 승인된 호출에서만 `RUN_V210 = True`로 바꾼다. A의 동결 선택 가중치·FP32·10-step DDIM·평가 seed 71과 body/hand offset 100/200을 사용한다. Hand의 상체는 같은 선택 body가 해당 clip에서 예측한 유효 상체 좌표다. RTX·T4·A100 등 동작하는 CUDA 환경에서 장치·VRAM·Torch/CUDA·정밀도를 그룹별 영수증에 기록한다. FP32 실행에서 TF32를 끈다. 전체 호출 상한은 마운트·입력 검증·저장을 포함해 28,800초다. 호출 논리 읽기는 최대 4 GiB, 패키지 보관 등을 포함한 논리 쓰기는 최대 2 GiB다. 18그룹의 실행 시간과 GPU 비용은 아직 실측하지 않았다.

보관 폴더는 프로젝트의 `reports/v210_eval128_a_v1`이다. 그룹마다 NPZ를 Drive에 복사해 해시를 재검증한 뒤 `group-00129.json`부터 `group-00146.json`까지 봉인한다. 중단 뒤 같은 고정본을 한 번 더 실행하면 기존 영수증과 NPZ의 SHA·키·프레임을 확인하고 남은 그룹부터 진행한다. VM이 유실돼 잠금이 남으면 이전 worker가 종료된 것을 확인한 다음 오류의 이전 owner 값을 `RECOVER_WRITER_OWNER`에 입력한다. 영수증 없는 NPZ는 자동 재개에 사용하지 않는다. 전체 검증이 완료되면 `summary.json`과 각 그룹 영수증·NPZ를 회수한다. 로컬 실행 로그에는 `result.json`의 위치가 표시된다.

## 지표와 산출물

- 각 NPZ는 `sequence_id`, `clip_start`, `frame_index`, 명목 `seconds`, `joint_names`, 예측·정답 device mm XYZ, 관절 유효 mask, 손목 상대 예측·정답 XYZ 및 상대 유효 mask를 담는다. Padding은 쓰기 전에 제거한다.
- 전체 공식 body·hand-global MPJPE는 원래 각 관절의 유효 mask를 사용한다. 공식 hand-local은 같은 쪽 정답 손목과 손가락이 모두 유효한 항목을 사용한다. 좌우 손목 MPJPE는 각 손목이 유효한 프레임에서 계산하며, 좌우 local·global·공통 mask global의 표본 수를 함께 제공한다.
- 공통 mask에서 `g = w + r`로 놓는다. `w`는 손목 위치 오차 벡터, `r`은 손목 상대 손가락 오차 벡터, `g`는 전역 손가락 오차 벡터다. `E[‖g‖²]`, `E[‖w‖²]`, `E[‖r‖²]`, `E[2 w·r]`를 mm²로 보고한다. 교차항은 음수가 될 수 있다. FP32 전역 좌표의 반올림에 의한 폐합 잔차를 함께 기록한다. 이 분해는 MSE이며 MPJPE가 가법적이라는 뜻이 아니다.
- 각 시나리오에서 유효 항목을 합산한 뒤 3시나리오의 body·hand-global 6개 점수를 같은 비중으로 평균한다. 원래 A 공식 결과(219.72474295351844 mm) 대비 macro와 시나리오별 차이를 보고한다. 원래 선택 hand e2의 공식 hand-local은 전체 93.71527568730127 mm이며 같은 시나리오·전체 지표와 유효 관절 수를 대조한다. 결과는 3클립 손목 분석보다 넓은 범위를 덮으므로 두 범위의 집계를 구분한다.

## 2026-11-22 전체 validation 결과

T4 FP32에서 18그룹·421클립·16,425프레임 평가를 완료했다. 전체 호출은 준비 포함 1,160.924초였다. 준비 구간 시간은 별도 실측이 없다. 세 시나리오 몸·손 종합 점수는 219.724745 mm로 기존 A와의 차이는 +0.00000219 mm다.

| 전체 validation MPJPE (mm) | 왼손 | 오른손 |
|---|---:|---:|
| 손목 위치 | 244.76 | 234.50 |
| 손목 기준 손가락 | 84.14 | 103.29 |
| 전역 손가락 | 281.08 | 271.17 |

공통 mask에서 손목 MSE가 상대 자세 MSE보다 양손 모두 크고 교차항은 양수다. 다음 v2.1.1은 body 출력층 조정 후 실제 손목과 전역 손가락 오차가 함께 줄어드는지 비교한다. 현재 결과는 가중치를 고정한 기준 평가다.

로컬 원본은 `outputs/v2_plan_inputs/v210_t4_validation_20261122/`에 있다. summary.json 파일 SHA-256은 `796527609be8733fd44341e564641ff3b83314199fc55c7912418b81c2931da4`, result.json은 `642a7dcd75edaea4bd78eba1ef51f34b79b6f141c719fe9400dc381e67c03770`다. 두 결과의 summary 내용 SHA와 프레임 수를 대조했다. 그룹별 Drive NPZ 회수와 별도 CPU 재집계는 대기 중이다.
