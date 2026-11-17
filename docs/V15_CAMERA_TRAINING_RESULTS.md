# v1.5 카메라 표현 통제 학습 결과

2026-11-15 검토. **같은 초기화·입력·학습량의 두 군 비교에서 첫 프레임 stereo 중점 표현이 절대 world 원점 의존성과 도시 표본의 큰 위치 오류를 줄였다. 손 형태 축소와 자세 오류는 남았다.** 이번 결과는 단일 초기화 실험과 기존 6개 개발·회귀 클립에 대한 근거이며 독립 일반화 성공이나 REWIND 재현 완료가 아니다.

실행 계약은 [v1.5 설계](V15_CAMERA_TRAINING.md), 분기와 후속 판단은 [실험 계보](V1_EXPERIMENT_LINEAGE.md)를 따른다. v1.6은 흡수·폐기 상태이며 v1.3 혼합학습은 보류한다.

## 고정 실행과 회수 근거

- Run ID: `V15-CAMERA-TRAINING-001`, revision `4f24f3aa2265c2ac54c21600b57bdece4e540411`.
- Source identity: `10e25cd2cb6f5b8d5dd4242481bf1aca349869f7000985dba28e82214c9a4540`.
- 실행: `run-81091786895d41e48d03ce9421cc56fe`, T4 / VRAM15,637,086,208B / FP32 / Python3.13.16 / Torch2.14.1+cu130 / CUDA13.0.
- 같은 train1,000페어(20×50), 두 군 body·hand 각각1,000update, 합4,000optimizer update 완료. Pilot와 본 학습의 초기 모델·RNG 해시 일치.
- F9는 v1.2의20,888페어 사전학습 모델을 고정했다. 두 군 B/H는 새 동일 초기화이며 v1.2 B10/H9의 미세조정이 아니다. Hand는 GT 상체로 학습하고 평가에는 각 군의 예측 body·wrist를 사용했다.
- 결과 ZIP:1,553,751,536B, SHA256 `a46e6a65f3777469d9f70d9aa99bf0aadf5bb5972ad06bb488c5b96f0944a170`.
- 최종 last:734,618,089B, SHA256 `5399dfefecf09014c6e80c94546bfa8015fdd92a6182433000b8c99c7f552894`.
- 평가 report content SHA `e6f50fe90b7a234aabea6c44065f07ab4538fc8b53c040059f132309e5c73c27`.
- [프로젝트 보존 ZIP](../outputs/v15_camera_training_20261115/user_test/completed_001/results.zip), [독립 무결성 검토](../outputs/v15_camera_training_20261115/user_test/completed_001/root_integrity_review.json), [지표 재계산](../outputs/v15_camera_training_20261115/user_test/completed_001/root_metrics_review.json), [체크포인트 검토](../outputs/v15_camera_training_20261115/user_test/completed_001/root_checkpoint_review.json).

테스트 세션은 Drive `reports/V15-CAMERA-TRAINING-001/run-81091786895d41e48d03ce9421cc56fe/inference/attempt-qp5rq194/results.zip`의 실행 출력에 같은 크기·SHA와 `verified`가 표시됐다고 전달했다. 별도 Drive receipt 파일은 미첨부이며 root가 원격 Drive를 다시 읽어 확인한 것은 아니다. 로컬 ZIP 내부 recovery receipt 검증과 구분한다.

## 정확도 비교

공통 device 좌표계 mm, body22/hand40관절, GT 유효 mask를 사용한 MPJPE다. Hand local은 wrist-relative, global은 예측 wrist를 붙인 결과다. 각 시나리오2클립×50frame이며 표의 시나리오 평균은 각 관절의 유효 개수로 집계한다(이 표본에서는 두 클립의 분모가 같다). Test set은 사용하지 않았다.

| 시나리오 | Body 절대→중점 mm | Hand global 절대→중점 mm | Hand local 절대→중점 mm |
| --- | ---: | ---: | ---: |
| ArchVis | 312.28→207.27 | 410.72→311.24 | 108.74→73.43 |
| CityPark | 10,366.58→300.78 | 17,771.94→400.28 | 12,353.56→101.37 |
| Downtown | 2,461.21→385.08 | 3,658.74→432.74 | 3,121.13→67.75 |
| 전체 300frame | 4,380.02→297.71 | 7,280.47→381.42 | 5,194.48→80.85 |

Body는 6개 클립 모두 개선됐다. 그렇지만 Downtown Reloading은 중점에서도478.24mm이고 전체 body297.71mm/hand global381.42mm여서 정확한 자세 복원을 달성했다고 보기 어렵다. 전체 평균의 큰 개선폭은 도시 절대군의 매우 큰 실패값에도 영향을 받으므로 시나리오별 결과를 함께 봐야 한다.

## 형태와 원점 안정성

- Body 평균 뼈 길이: 절대55.52mm → 중점154.81mm, GT185.28mm. 도시 절대군의 약8mm 축소가 크게 완화됐다. 중점 Downtown Reloading은115.43mm로 축소가 남았다.
- Hand local 평균 뼈 길이: 절대3.08mm → 중점8.20mm, GT23.71mm. 중점도 정답의 약34.6%이며 손 형태는 여전히 심하게 작다. Hand local 오차 감소만으로 손 복원을 성공 판정하면 안 된다.
- Body 중심 위치 오차: 전체4,348.30mm →212.09mm. 위치 편향 감소와 뼈 길이 회복이 함께 관찰됐다.
- 같은 장면을 world x축+100m로 재표현하고 anchor를 다시 계산했을 때, 중점군은 6개 클립 모두 body/hand local/global/whole 예측이 원본과 정확히 같았다. 절대군 body 평균 변화는 클립별 약1,773~4,684mm였고 모든 클립에서 repeat-aware 허용오차를 초과했다.
- Δ0 반복은 양군 모든 예측에서 정확히 같았다. 로그의 full50/prefix25·camera streaming 검사는 통과 상태다. Root는 저장 예측의 반복/shift와 실제6클립의 float64 anchor를 재계산했다. +100m 뒤 정규화 translation 최대 차이는 약1.75e-10mm였다. GPU sampler를 다시 실행하지 않았다.

이 결과는 **일관된 중점 표현이 이번 비교의 불필요한 world translation 민감도를 제거했고 성능 개선에도 기여했다**는 근거다. 회전 원점·모든 카메라 조건·독립 시나리오까지 검증한 것은 아니다. 최초프레임 anchor 정규화는 새로운 네트워크 구조나 더 많은 데이터의 효과와 구분된다.

## 학습 표본의 복원 진단과 잔여 원인

고정 첫 train 클립 Standing Up50frame의 보고값은 body225.45→236.16mm, hand local66.80→64.76mm, hand global271.96→146.62mm다. Body는 이 학습 클립에서 오히려 약10.7mm 악화됐다. 중점 손 뼈 길이는 약8.12mm/GT23.71mm다. 이 진단은 회귀6클립과 합산하지 않는다. Report 값·봉인을 확인했으며 train 예측 배열은 저장돼 있지 않아 root가 train-fit 오차를 독립 재계산하지는 않았다.

따라서 camera 표현은 주요 실패 원인 하나를 해소했지만 작은 학습 세트 자체를 충분히 복원한 상태도 아니다. 예측2D의 잔여 오류, hand의 GT 상체 학습/예측 상체 평가 차이, 학습량·loss·sampling의 제한이 후속 가설이다. 이번 실행은 이를 분리하지 않았으므로 단일 원인으로 확정하지 않는다.

v1.2나 v1.4와의 단순 성능 순위 비교는 부적절하다. v1.2는 전체 데이터의 다른 초기화·학습량이고, v1.4는 단일클립 GT2D·GT상체 overfit이다. v1.5의 유효 비교는 같은 조건의 absolute 대 first_midpoint다.

## 시간·자원과 보관

학습 보고의 `elapsed_training_seconds`는779.57초(약13분)다. 구현은 초기화·복원 시간과 보수적인60초 종료 저장 allowance를 누적하므로 정확한 순수 GPU wall-clock이나 커널 지연이 아니다. 사용자는 학습 셀 약14분, 평가 셀4분 조금 넘음을 보고했다. 최초 준비의 전체 외부 경과시간은 미확보다. 고유1,000페어당 시간과 반복 노출/optimizer update당 시간을 혼동하거나 전체 pipeline 처리량으로 환산하지 않는다.

GPU peak allocated는1,874,572,288B(약1.75GiB)이며 CUDA 예약량·프로세스 총 메모리와 다르다. 최종 Drive copy/readback25.21초는 테스트 세션이 전달한 실행 출력 기준이다. 결과 ZIP에는 재개 상태와 중간 모델을 함께 넣으므로 작은 입력 데이터라도 산출물은 약1.55GB다.

새 Colab VM에서 이 실제 학습 상태를 복원해 재개하는 시험은 하지 않았다. 로컬 체크포인트 구조·재개 회귀 통과와 실제 원격 VM 재개를 구분한다.

## 독립 검토 범위

1. 다운로드 원본과 프로젝트 보존 ZIP의 SHA, 내부90파일 크기·SHA/CRC, JSON seal22개, source/config/input receipt·pilot/초기 RNG·최종 checkpoint 연결을 확인했다.
2. 6클립×2군×3variant=36개 예측의 body/hand local/global 오차, bone/spread, centroid/중심제거 지표를 NumPy로 재계산했다. MPJPE 평균의 최대 차이는 약2.20e-5mm로 float32 계산 차이 범위다. 저장 GT·feature hash, 반복 일치·shift 변화량·허용오차 초과 개수도 확인했다.
3. Checkpoint13개를 CPU `weights_only`로 로드하고 tensor 유한값, 양군 step0 모델 해시, final snapshot과 last 모델의 정확일치, optimizer/scheduler 각1,000step을 확인했다. 실제 GPU 학습을 재현한 것은 아니다.
4. Root는 ArchVis Standing과 CityPark Gunplay의 frame25 PNG를 직접 확인했다. 테스트 세션은6개 GIF의 frame0/24/49를 검토했다. 그림의 RGB 위 예측2D는 고정F9 출력이므로 양군 동일하며 새3D 모델을 투영한 점으로 해석하지 않는다.
5. 도시 그림은 absolute의 큰 위치 이탈을 포함한 공통 축 때문에 GT·중점 모델이 작게 보일 수 있다. 축 크기를 무시한 시각적 비교 대신 정량 shape·centroid 지표를 병행한다.

[ArchVis 대표 그림](../outputs/v15_camera_training_20261115/user_test/completed_001/extracted/evaluation/clip-00/frame-000000025.png) · [CityPark 대표 그림](../outputs/v15_camera_training_20261115/user_test/completed_001/extracted/evaluation/clip-02/frame-000000025.png).

## 후속 방향 — 당시 제안과 후속 결과

2026-11-17 후속: 아래 GT/예측 상체 교체 진단은 v1.5.1로 실행·검토했다. Local 평균10.8% 개선에도 손 축소가 남았다. [실제 결과](V151_UPPER_CONDITION_RESULTS.md). 아래 문단은 당시 제안 기록이다.

중점 표현을 후속 통제 진단의 기준 후보로 유지하되, 데이터 확대보다 **같은 checkpoint에서 hand에 GT 상체를 넣는 평가와 예측 상체 평가를 비교**하는 작은 진단이 우선 후보다. 예측2D·camera·noise는 유지하고 local 손 오차와 예측 wrist의 global 위치 오류를 분리한다. GT 조건 결과는 진단이며 배포 성능이 아니다. 그 뒤 필요할 때 고정 train 표본의 학습량 또는 GT/예측2D를 한 변수씩 바꾼다.

이는 v1.5의 승인된 방향 안에서 다음 범위를 선택하기 위한 제안이다. 이번 결과 기록으로 후속 학습·예산 확대·v1.3 혼합학습을 시작하지 않는다.
