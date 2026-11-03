# 미니 프로토타입 준비 계획

2026-11-03, 기준 코드 `af275f5`. 두 번째 big step인 제한된 시나리오의 REWIND-like 미니 모델·시연 준비다. 전체 인덱싱 완료를 기다리지 않고 진행하되, 기존371쌍을 최종 목표로 축소하지 않는다. 목표 데이터는1,000~10,000개의 고유한 좌우 RGB+pose 페어다. 이번 작업은 보존 자료·코드의 읽기 전용 감사와 계획 문서화이며 신규 학습·추론 실행·원격 추출·화면 구현 완료를 의미하지 않는다.

사용자는 인덱싱 중단 시 직접 알리기로 했다. 이 준비 작업은 실행 중인 인덱싱의 설정·예산·pending·store를 변경하거나 모니터링하지 않는다. 이후 인덱싱 결과가 전달되면 고정된 snapshot에서 필요한 key를 대조한다.

## 재사용할 수 있는 것과 부족한 것

| 항목 | 확인된 상태 | 미니 시연까지 남은 일 |
| --- | --- | --- |
| 실제 입력 | 기존102+신규269=371쌍,7개 sequence, 전부 train, validation0 | 중복 없는 통합 manifest/loader와 독립 validation 확보 |
| 실제 모델 |102쌍에서 RGB→예측2D→T1 3D 각51update 진단; weights와 재로드 검증 근거 보존 | 학습 없이 호출하는 추론 경로, 확대 데이터 학습·평가 |
| 학습 기반 | UI 독립 M1 shard/RGB trainer와 checkpoint/export 코드, 합성 GPU 재개 검증 | 실제 데이터 release, Colab 실측, 고정 split의 학습 |
| REWIND 구조 | M2 causal body·M3 body→hand 구현과 합성 검증 | 실제 연속 데이터 학습·정확도 검증; M1을 논문 전체 구현으로 표시하지 않음 |
| 영상·화면 | 공통 독립 frame 계약, GT 감사용 viewer | decoder·PTS·seek/reset·예측 결과 viewer 연결 |

근거: [102 진단](REAL_RGB102_DIAGNOSTIC.md), [269 추출과371 경계](RGB269_PAIRS.md), [전체 학습 준비](FULL_TRAINING_READINESS.md), [영상 방향](VIDEO_INPUT_PLAN.md).

보존 입력은 `outputs/test_rgb102_pairs_001_20261103/rgb102_pairs_001_result.zip`(274,628,517bytes)과 `outputs/test_rgb269_pairs_001_20261103/rgb269_pairs_001_result.zip`(688,954,705bytes)이다. 두 ZIP의 `inputs/config.json`, `inputs/input_manifest.json`, `inputs/pose/candidate.json`, pose/metadata pin 및 `run/report.json`을 재사용한다. 읽기 전용 감사에서 disjoint371 key를 확인했으며 실제 사용 시 기존 SHA·내용 검증을 유지한다. 원본 ZIP을 덮어쓰거나 대용량 파일을 Git에 추가하지 않는다.

Weights는 `outputs/rgb102_diagnostic_20261103/developer/run/`의 `frontend_weights.pt`, `regression_weights.pt`에 있다. 같은102 train-fit의 MPJPE634.07mm, 화면 안2D 평균330.28px로 오차가 크다. `outputs/rgb102_diagnostic_20261103/root/root_review.json`은 저장 모델의 재추론 일치 근거이며, 이번 감사에서 재학습하거나 checkpoint를 새로 실행한 것은 아니다. 이 weights는 연결 확인용이며 성능이 확보된 미니 모델이 아니다.

현재102 reader는 고정 ZIP/config SHA와102 key 수를 강제한다. 일반 weights loader와도 진단 schema가 다르고, 일반 `pack_window`는 GT3D target을 요구한다. 따라서 기존 trainer에269 ZIP을 바로 넣거나 일반 추론기에 weights 경로만 바꾸는 방식은 사용할 수 없다.

## 데이터 확대와 평가 분리

1. 기존371쌍은 개발 진단에 재사용한다. 각 candidate의 blockers와 `train_ready=false`를 보존한다. 일반 학습 gate를 우회하거나 진단 예외를 정식 release로 승격하지 않는다.
2. 목표1,000쌍까지 최소629쌍,10,000쌍까지9,629쌍이 더 필요하다. 이는 총량 차이이며 독립 validation이 자동 확보된다는 뜻은 아니다. Catalog 파일 수나 stereo 관측 수를 실제 RGB+pose 추출·검증 완료량으로 계산하지 않는다.
3. 새 고정 snapshot을 받을 때 sequence/frame 키로 좌우 RGB·pose를 join하고, 필요한 원본만 제한 추출해 hash·decode·관절·단위·좌표·mask를 검사한다. 이미 보존된371쌍은 다시 다운로드하지 않는다. 전송·요청·RAM·디스크 비용을 산출하고 실행 예산을 고정한 뒤 탐색 세션에 넘긴다.
4. Frame 무작위 분할 대신 sequence와 원본 motion 중복을 확인해 train/validation을 분리한다. 현재371쌍은 이미 train-fit 진단에 쓰였거나 train으로 정해졌으므로 임의 재분할을 독립 평가로 제시하지 않는다. Official test는 튜닝에 사용하지 않는다. 정확한 split 비율은 후보 확보 후 결정한다.
5. 동영상용으로 짧고 연속적인 구간도 확보한다. 기존102의5개 sequence는 frame 번호 gap0이고, 신규 BlowKiss122개는0..121, PullingRope147개는0..148 중51·80이 없다. 번호 연속성은 timestamp·stereo 동기화·실제 시간 간격의 검증을 대신하지 않는다.

## 개발 순서와 완료 기준 — 제안, 아직 미구현

### 첫 묶음: 보존 모델의 독립 추론·결과 내보내기

새 학습과 Drive 접근 없이 기존102 진단 weights용 replay CLI를 구현한다. 고정 입력·weights·전처리 identity를 검사하고 선택 stereo 이미지를 필요할 때 읽어, 기존64×64 resize→예측2D→native 좌표 복원→868개 특징→T1 62관절3D(mm)를 실행한다. GT가 없어도 특징을 구성하고 GT2D fallback은 금지한다. GT 비교는 별도 선택 경로로 둔다.

출력에는 sequence/frame, 원본·모델·전처리 hash, 관절/좌표/단위, 유효성, `diagnostic_only`, 시간 미확인 상태를 기록한다. CLI와 결과 artifact는 UI를 import하지 않고 optimizer·학습을 시작하지 않는다. 정식 weights 형식과 혼동하지 않게 별도 역할을 유지한다.

완료 기준은 기존 보존 예측과의 수치 일치, GT 없는 입력 실행, 잘못된 identity 차단, 출력 재읽기와 제한 메모리 사용이다. 로컬 검증으로 가능한 묶음이며 사용자 Colab 시험을 먼저 요구하지 않는다. 이는 시연 배선의 시작이고1,000~10,000쌍 학습 목표의 대체가 아니다.

### 두 번째 묶음: 통합 입력과 실제 미니 학습

102/269 전용 상수를 새 통합 입력 계약으로 분리하되 기존 진단 재현 경로는 보존한다. 중복·누락·split·출처·학습 허용 상태를 검사하는 manifest를 준비해1,000쌍 이상으로 확장 가능한 CLI/config를 연결한다. 필요하면371쌍은 제한 진단으로만 먼저 사용한다.

기존 Adam1e-4/B2/1pass는 진단 출발점이며 최적 설정이 아니다. 작은 learning rate와 제한 update의 pilot을 제안하되, 실제 데이터 release·GPU/VRAM·validation·자원 실측 후 학습 상한을 고정한다. Train loss와 validation 오차를 별도로 기록하고 baseline·checkpoint·optimizer/RNG/cursor를 보존한다. 낮은 점수도 정직하게 보고하며 반복 학습으로 일반화가 보장된다고 주장하지 않는다.

### 세 번째 묶음: 시연과 연속 입력

먼저 모델이 내보낸 예측을 좌우 이미지·3D skeleton으로 재생하는 독립 viewer를 연결한다. 모델·학습 실행과 UI를 분리하고 명시적인 weights/result 버전으로 결과를 교체한다. 학습 데이터 재생이면 화면과 보고서에 train-fit 진단임을 표시한다.

이후 검증된 stereo clip과 calibration/camera metadata를 대상으로 decoder·PTS·순서·좌우 동기화·seek/reset을 구현한다. 단일 프레임 예측을 나열한 영상과 causal M2/M3 추론을 구분한다. 실제 M2 body→M3 hand 학습은 별도 후속 묶음이며 M4 증류나 실시간30FPS를 이번 준비의 완료 조건으로 삼지 않는다. 임의 단안 영상·실시간 카메라 지원은 약속하지 않는다.

## 다음 검증 인계

첫 추론 묶음은 개발의 로컬 검증 후 root가 검토·커밋한다. 데이터 확대는 탐색 세션, 실제 GPU 학습·시연 사용자 시험은 기존 테스트 세션으로 구분한다. 실제 학습 입력과 실행 경로가 준비되면 고정 revision·소스/입력/weights SHA·예산·성공/중단 기준·회수 위치를 한 번에 전달한다. 준비되지 않은 notebook을 먼저 사용자에게 반복 실행하도록 요청하지 않는다.

첫 제한 시연의 완료는 재현 가능한 입력→예측→표시/내보내기, 독립 평가 여부 표시, 학습·추론 provenance와 제한사항 보고다. 일반화 성능이나 논문 재현 완료는 별도로 판정한다. 기존 M0~M6 완료 상태를 이번 문서화로 올리지 않는다.
