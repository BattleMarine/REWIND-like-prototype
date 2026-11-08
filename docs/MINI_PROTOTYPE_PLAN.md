# 미니 프로토타입 준비 계획

2026-11-09 구현 갱신: 직접 추출본 manifest·T1 lazy loader를 추가하고 관련40개 로컬 검사를 root가 확인했다. 약1,000 train 선택과 전체 적격 train 선택이 같은 입력 계약을 사용한다. 실제1,000페어·GPU 학습과 trainer 연결은 미완료이며, 아래 이전의 adapter 미완료 표현 중 입력 manifest/loader 부분만 이 기록으로 갱신한다. [지원 범위와 검증](EXTRACTED_STEREO_INPUT.md).

## 2026-11-09 확정: 약1,000페어 검증 후 ArchVis 전체 확대

사용자는 0버전 계획을 승인하면서 초기 약1,000 train 페어에서 검증한 뒤 ArchVis 전체 train으로 바로 확대하도록 지정했다. 중간1만페어 학습 단계는 두지 않는다. 아래2026-11-05의1,000~10,000페어 목표·총량 차이는 이전 계획이며 이 결정으로 대체한다.

- 첫 입력은 공식 train에서 고정 seed·sequence/frame 목록으로 약1,000개의 고유 페어를 선택한다. validation은 공식 validation에서 별도 구성하고, 원본 motion 중복 근거를 검토한다. test는 개발·튜닝에 넣지 않는다.
- 좌우 RGB·pose 대응, 중복·누락, 이미지 decode·크기·hash, pose 관절·단위·좌표·mask와 대표 overlay를 검증한다. 목록 존재만으로 학습 허용 상태로 승격하지 않는다.
- 제한된 Colab 학습에서 finite loss/gradient·실제 weight 갱신, 소량 overfit의 학습 신호, 저장·재개·재추론, 독립 validation 지표와 GPU 메모리·시간·디스크를 확인한다. 단순 실행 성공과 예측 정확도 개선은 구분한다. 기존 중앙 집중 실패가 그대로면 원인을 보완한 뒤 확대한다.
- 통과하면 동일 manifest/loader 계약으로 ArchVis 전체 train을 사용한다. 나머지 파일에도 동일 데이터 검사를 적용하며 실패 파일은 명시 보고한다. 전체를 RAM에 적재하지 않고 필요한 입력을 순차 공급한다. 실측으로 batch·epoch/update·보관 예산을 정한다.
- 이후 예측2D 기반3D baseline과 저장 모델의 overlay·3D skeleton·짧은 시퀀스 결과를 연결한다. 학습 코어는 UI와 분리한다. 실제 M2/M3 teacher·증류는 후속이다.

탐색 세션의2026-11-09 메타데이터 보고상 후보는 train20,889페어/286sequence, validation2,236/30, test2,897/34이며 왼쪽 RGB 중복1건이 있다. 이는 파일 내용 검증·학습 완료량이 아니다. 근거는 로컬 `outputs/archvis_drive_audit_20261109/sequence_summary.json`에 보존되어 있으며 Drive 변경 시 재확인한다.

## 이전 계획과 구현 근거

2026-11-05 갱신(초기 계획2026-11-03, 기준 코드 `af275f5`). 2026-11-04 인덱싱 포기 결정에 따라 사용자 Drive의 RGB·pose 추출본을 직접 사용하는 미니 모델 개발 계획이다. 목표는1,000~10,000개의 검증된 고유 페어와 독립 validation이며 기존371쌍을 최종 목표로 삼지 않는다. Depth는 초기 필수 입력이 아니다. 이번 문서 수정은 신규 학습·원격 추출·업로드 검증·화면 구현 완료를 의미하지 않는다.

자동 인덱싱·survey·실패 지점 재개는 현재 중단된 경로다. 기존 원본·snapshot·예산·pending·store와 고정 실행 근거는 보존하며 새 실행을 요청하지 않는다. 사용자 추출본의 업로드 완료·정확 경로·무결성과 split을 확인한 뒤 [데이터 확보 전략](DATA_ACQUISITION_STRATEGY.md)의 절차로 준비한다.

## 재사용할 수 있는 것과 부족한 것

| 항목 | 확인된 상태 | 미니 시연까지 남은 일 |
| --- | --- | --- |
| 실제 입력 | 기존102+269=371쌍,7sequence, 모두 train·validation0; 추가 로컬 Joyful57은 중복·split 미검토 | 추출본 통합 manifest/loader,371과57 중복 검토,1,000~10,000페어·독립 validation 확보 |
| 실제 모델 |102쌍 각51update 진단 weights; stereo→2D overlay CLI 검증, Joyful3frame 예측 중앙 집중 실패 | GT 없는3D 추론 연결,2D 학습 개선·확대 데이터 학습/평가 |
| 학습 기반 | UI 독립 M1 shard/RGB trainer와 checkpoint/export 코드, 합성 GPU 재개 검증 | 실제 데이터 release, Colab 실측, 고정 split의 학습 |
| REWIND 구조 | M2 causal body·M3 body→hand 구현과 합성 검증 | 실제 연속 데이터 학습·정확도 검증; M1을 논문 전체 구현으로 표시하지 않음 |
| 영상·화면 | 공통 독립 frame 계약, GT 감사용 viewer·Joyful57 GT overlay171PNG(정답 표시) | decoder·PTS·seek/reset·예측 결과 viewer 연결 |

근거: [102 진단](REAL_RGB102_DIAGNOSTIC.md), [269 추출과371 경계](RGB269_PAIRS.md), [전체 학습 준비](FULL_TRAINING_READINESS.md), [영상 방향](VIDEO_INPUT_PLAN.md).

보존 입력은 `outputs/test_rgb102_pairs_001_20261103/rgb102_pairs_001_result.zip`(274,628,517bytes)과 `outputs/test_rgb269_pairs_001_20261103/rgb269_pairs_001_result.zip`(688,954,705bytes)이다. 두 ZIP의 `inputs/config.json`, `inputs/input_manifest.json`, `inputs/pose/candidate.json`, pose/metadata pin 및 `run/report.json`을 재사용한다. 읽기 전용 감사에서 disjoint371 key를 확인했으며 실제 사용 시 기존 SHA·내용 검증을 유지한다. 원본 ZIP을 덮어쓰거나 대용량 파일을 Git에 추가하지 않는다.

Weights는 `outputs/rgb102_diagnostic_20261103/developer/run/`의 `frontend_weights.pt`, `regression_weights.pt`에 있다. 같은102 train-fit의 MPJPE634.07mm, 화면 안2D 평균330.28px로 오차가 크다. `outputs/rgb102_diagnostic_20261103/root/root_review.json`은 저장 모델의 재추론 일치 근거이며, 이번 감사에서 재학습하거나 checkpoint를 새로 실행한 것은 아니다. 이 weights는 연결 확인용이며 성능이 확보된 미니 모델이 아니다.

현재102 reader는 고정 ZIP/config SHA와102 key 수를 강제한다. 일반 weights loader와도 진단 schema가 다르고, 일반 `pack_window`는 GT3D target을 요구한다. 따라서 기존 trainer에269 ZIP을 바로 넣거나 일반 추론기에 weights 경로만 바꾸는 방식은 사용할 수 없다.

## 데이터 확대와 평가 분리

1. 기존371쌍은 개발 진단에 재사용한다. 각 candidate의 blockers와 `train_ready=false`를 보존한다. 일반 학습 gate를 우회하거나 진단 예외를 정식 release로 승격하지 않는다.
2. 기존371만 기준으로 한 총량 차이는 목표1,000까지629쌍,10,000까지9,629쌍이다. 추가 Joyful57은 중복·원본 motion·split 검토 전이므로 이 수치에서 바로 빼거나 총428쌍으로 확정하지 않는다. 이는 총량 차이이며 독립 validation이 자동 확보된다는 뜻은 아니다. Catalog 파일 수나 stereo 관측 수를 실제 RGB+pose 추출·검증 완료량으로 계산하지 않는다.
3. 사용자 Drive 추출본에서 정확한 sequence/frame 키로 좌우 RGB·pose를 join하고 크기·hash·decode·관절·단위·좌표·mask·영상 overlay를 검사한다. archive 인덱스나 새 Range 탐색은 전제하지 않는다. 기존371을 다시 다운로드하지 않고 업로드 완료·경로·선택 subset·IO/RAM/디스크 상한을 확인한 뒤 범용 manifest/adapter를 개발·검토한다.
4. Frame 무작위 분할 대신 sequence와 원본 motion 중복을 확인해 train/validation을 분리한다. 현재371쌍은 이미 train-fit 진단에 쓰였거나 train으로 정해졌으므로 임의 재분할을 독립 평가로 제시하지 않는다. Official test는 튜닝에 사용하지 않는다. 정확한 split 비율은 후보 확보 후 결정한다.
5. 동영상용으로 짧고 연속적인 구간도 확보한다. 기존102의5개 sequence는 frame 번호 gap0이고, 신규 BlowKiss122개는0..121, PullingRope147개는0..148 중51·80이 없다. 번호 연속성은 timestamp·stereo 동기화·실제 시간 간격의 검증을 대신하지 않는다.

## 개발 순서와 완료 기준 — 부분 구현과 남은 작업

### 첫 묶음: 보존 모델의 독립 추론·결과 내보내기 — 2D 구현,3D 후속

2026-11-04 `scripts/predict_stereo_overlay.py`로 기존102 진단 frontend의 stereo PNG→62관절2D JSON/overlay를 구현하고 저장 예측 재현·root 독립 실행을 확인했다. Joyful0·28·56은 실제 예측이지만 중앙에 몰리는 실패를 보였다. `scripts/render_pose_gt_overlay.py`의 전체57 GT overlay171PNG는 모델 추론과 구분한다. [실행·제한](PREDICTION_OVERLAY.md)을 따른다.

GT 없는868개 특징→T1 62관절3D(mm)→artifact 연결은 아직 후속이다. 고정 입력·weights·전처리 identity를 검사하고 기존64×64 resize·native 좌표 복원을 유지하며 GT2D fallback을 금지한다. 일반 weights schema와 진단 weights schema가 같다고 가정하지 않는다.

출력에는 sequence/frame, 원본·모델·전처리 hash, 관절/좌표/단위, 유효성, `diagnostic_only`, 시간 미확인 상태를 기록한다. CLI와 결과 artifact는 UI를 import하지 않고 optimizer·학습을 시작하지 않는다. 정식 weights 형식과 혼동하지 않게 별도 역할을 유지한다.

완료 기준은 기존 보존 예측과의 수치 일치, GT 없는 입력 실행, 잘못된 identity 차단, 출력 재읽기와 제한 메모리 사용이다. 로컬 검증으로 가능한 묶음이며 사용자 Colab 시험을 먼저 요구하지 않는다. 이는 시연 배선의 시작이고1,000~10,000쌍 학습 목표의 대체가 아니다.

### 두 번째 묶음: 통합 입력과 실제 미니 학습

102/269 전용 상수를 범용 RGB·pose 추출본 입력 계약으로 분리하되 기존 진단 재현 경로는 보존한다. 중복·누락·원본 motion·split·출처·학습 허용 상태를 검사하는 manifest와 adapter는 아직 미완료다. 이를 구현·검증해1,000쌍 이상으로 확장 가능한 CLI/config를 연결한다.371쌍의 제한 진단을 정식 release로 승격하지 않는다.

기존 Adam1e-4/B2/1pass는 진단 출발점이며 최적 설정이 아니다. 작은 learning rate와 제한 update의 pilot을 제안하되, 실제 데이터 release·GPU/VRAM·validation·자원 실측 후 학습 상한을 고정한다. Train loss와 validation 오차를 별도로 기록하고 baseline·checkpoint·optimizer/RNG/cursor를 보존한다. 낮은 점수도 정직하게 보고하며 반복 학습으로 일반화가 보장된다고 주장하지 않는다.

### 세 번째 묶음: 시연과 연속 입력

먼저 모델이 내보낸 예측을 좌우 이미지·3D skeleton으로 재생하는 독립 viewer를 연결한다. 모델·학습 실행과 UI를 분리하고 명시적인 weights/result 버전으로 결과를 교체한다. 학습 데이터 재생이면 화면과 보고서에 train-fit 진단임을 표시한다.

이후 검증된 stereo clip과 calibration/camera metadata를 대상으로 decoder·PTS·순서·좌우 동기화·seek/reset을 구현한다. 단일 프레임 예측을 나열한 영상과 causal M2/M3 추론을 구분한다. 실제 M2 body→M3 hand 학습은 별도 후속 묶음이며 M4 증류나 실시간30FPS를 이번 준비의 완료 조건으로 삼지 않는다. 임의 단안 영상·실시간 카메라 지원은 약속하지 않는다.

## 다음 검증 인계

남은 추론·추출본 adapter·학습 개선 묶음은 개발이 로컬 검증하고 root가 검토·커밋한다. 과거 탐색 세션의 인덱싱 작업은 중단됐으며 자동 재개하지 않는다. 사용자 Drive 업로드 완료·자료 확인과 실제 GPU 학습·시연 시험은 root가 범위를 고정한 뒤 해당 세션에 인계한다. revision·소스/입력/weights SHA·예산·성공/중단 기준·회수 위치를 함께 전달하며 준비되지 않은 notebook을 반복 실행하도록 요청하지 않는다.

첫 제한 시연의 완료는 재현 가능한 입력→예측→표시/내보내기, 독립 평가 여부 표시, 학습·추론 provenance와 제한사항 보고다. 일반화 성능이나 논문 재현 완료는 별도로 판정한다. 기존 M0~M6 완료 상태를 이번 문서화로 올리지 않는다.
