# REWIND-like prototype

스테레오 egocentric 영상에서 몸과 손의 3D 관절을 추정하는 연구용 프로토타입입니다. REWIND 논문의 구조를 참고해 비교 모델, causal diffusion teacher와 student를 단계적으로 개발합니다. 공식 구현이나 논문 성능 재현이 완료된 모델은 아닙니다.

최종 목표는 이미지 쌍과 동영상을 처리하는 미니 프로토타입입니다. 화면 UI는 학습·평가·추론 코어와 분리하며, 현재는 데이터 준비와 모델 학습 개선에 집중합니다.

## 현재 상태

2026-11-20 기준, **v2 첫 CPU 캐시 샤드를 Drive에 보관하고 해시를 검증했습니다.** 세 시나리오에서 고정한 136,011페어 가운데 공식 train 128페어(좌우 RGB·pose 384파일)를 처리했습니다. 이 샤드의 ArchVis 두 프레임을 A100으로 왕복 전송해 텐서 8종의 값과 dtype이 모두 보존됨을 확인했습니다. CPU·CUDA PyTorch 빌드 차이로 정식 학습 로더의 복원 조건은 충족하지 못했습니다. 다음 시험에서는 같은 샤드를 정식 복원 함수로 가져와 학습 로더에서 한 배치를 읽습니다. 전체 캐싱과 GPU 학습은 아직 시작 전입니다. [CPU 캐시 실행 가이드](docs/V2_CPU_CACHE_PILOT.md) · [GPU 읽기 시험](docs/V2_GPU_CACHE_PROBE.md) / [정식 복원 시험](docs/V2_GPU_CACHE_RESTORE.md)

2026-11-18: **v2 — ArchVis·CityPark·Downtown 전체 train 통합 학습** 준비를 시작합니다. 기존 v1.3 제안을 확대·이관합니다. 후속 승인으로 입력 준비·학습 각1회8시간을 확정해 v2 전용 보존·재개 인프라 구현·신규14검사와 root 검토를 완료하고 고정본 발급 대기 중이며 실제 학습은 아직 시작하지 않았습니다. [v2 계획](docs/V2_PLAN.md).

**실험 결과는 [계속 갱신하는 종합 보고서](docs/EXPERIMENT_RESULTS_REPORT.md)에서 확인할 수 있습니다.** 버전별 목적·수치·판정·남은 문제와 결과 파일 위치를 정리합니다.

2026-11-17 기준입니다. 학습은 A100에 제한하지 않고 지원되는 CUDA GPU를 사용합니다. 실제 GPU·VRAM·시간을 기록하며 A100 조건은 추론 레이턴시 비교 시험에서 별도로 적용합니다.

장시간 실행 결과는 Drive 자동 보관을 기본으로 준비합니다. v1.1은 종료 후 모델 ZIP을, 새 v1.2 notebook은 각 단계 종료 후 보고서를 보관합니다. [적용 범위·한계](docs/RESULT_RETENTION.md).

v1.2 ArchVis 전체 train20,888페어로 frontend/body/hand 각10epoch·115,720update를 완료했습니다. 참고 validation best는2D89.97px, body169.56mm, hand global318.56mm입니다. Body 형태 축소는 완화됐고 손 축소는 남았습니다. 회수 checkpoint18개 CPU 로드·유한값 검토를 완료했으며 로컬 train2프레임의 실제 예측 시각화에서 형태 복원과 잔여 자세 오류를 확인했습니다. 연속 validation 시각화와 독립 일반화 검증은 남았습니다. [결과·검증·한계](docs/V12_TRAINING_RESULTS.md).

v1.4 단일클립 GT조건 진단은 A100에서 각5,000update를 완료했습니다. Body13.71mm·hand local2.90mm로 형태 복원을 확인했으며, 실제 RGB 추론·일반화 성능과는 구분합니다. [결과와 해석](docs/V14_ORACLE_RESULTS.md) · [v1.1 비교·원인 가설·후속 테스트](docs/V11_V14_COMPARATIVE_REVIEW.md).

v1.1 T4 재실행은 동일1,000페어·각10epoch를 완료했고, Drive 자동 보관본의 모델13개를 회수·검토했습니다. 전체약97분24초, frontend best214.84px이며3D 형태 축소는 남았습니다. [재실행 결과](docs/V11_CUDA_RERUN_RESULTS.md).

이전 v1.1 동일1,000페어의 각10epoch 실행은 완료 로그만 확보했습니다. 2D 오차는205.27px로 줄었으나3D 형태 축소는 남았으며 모델 ZIP 다운로드 실패로 checkpoint는 미확보입니다. [결과와 복원 한계](docs/V11_EPOCH_EXPANSION_RESULTS.md).

v1 새 train1,000·validation2,236의 실제 파일·카메라 수치 감사가 완료됐습니다. 사용자가 GT overlay89장을 확인했으며, 실제 timestamp·motion 독립성 미확인을 명시한 별도 A100 진단 학습을 완료했습니다. Frontend/body/hand 각1epoch, 총540update·약14분11초이며 정확도 개선은 남아 있습니다. [진단 결과](docs/V1_DIAGNOSTIC_001_RESULTS.md). Raw cache는 사용자 Drive에 보관됐으며 후속 실행에서 복원해 재사용합니다. [감사 결과](docs/V1_ACTUAL_001_RESULTS.md).

| 항목 | 확인한 내용 | 남은 작업 |
| --- | --- | --- |
| M0 데이터 | ArchVis train1,000·validation2,236페어 준비 검사 완료, 추출본 로더 구현 | 실제 영상·원본 motion 정합과 독립 평가 검토 |
| M1 비교 모델 | ArchVis1,000페어 v0 T4 20epoch 완료 | 손·관절 정확도 개선, 독립 평가·전체 확대 준비 |
| M2·M3 teacher | Causal body diffusion·body–hand cascade 구현, 실제 A100 각1epoch 진단 | 정확도 개선·독립 평가·성능 비교 |
| M4~M6 | Student 증류·개선 실험·통합 계획 | 후속 구현·평가 |
| 이미지·동영상 | 저장된 v1 모델의 stereo 2D·카메라 조건부 3D 비교 시각화(T1) | 정확도 개선·영상 문맥 추론·decoder·시간 정합·UI |

v0 2D 모델의 1,000페어 학습을 완료했습니다. Validation 화면 안 평균 오차는335.29→195.31px로 줄었지만 손 오차와 예측 위치 오류가 남아 있습니다. 전체 약84분 중77분이 캐시 생성, 20epoch 학습은 약4분44초였습니다. 종료 기록 오류와 학습 완료를 구분해 검토했습니다. [결과·시간·한계](docs/ARCHVIS_V0_RESULTS.md). 이전102페어 CPU 진단도 정확도가 낮았습니다. Joyful Jump 3프레임의 실제 예측도 관절이 화면 중앙에 몰리는 실패를 보였습니다. 추가 로컬 Joyful Jump 57페어는 중복·split 검토 전이므로 기존 371개와 합산하지 않습니다. 해당 57프레임의 GT 관절점 이미지 171장은 **정답 시각화이며 모델 예측 결과가 아닙니다.**

v1.5는 T4에서 두 카메라 표현의 통제 학습 총4,000update를 완료했습니다. 중점 표현에서 개발6클립의 body 평균4,380→298mm, world x+100m 이동에 대한 예측 변화0을 확인했습니다. 손 형태 축소와 자세 오류는 남았고 독립 일반화 검증은 아닙니다. v1.6은 흡수·폐기했습니다. [결과·독립 검토](docs/V15_CAMERA_TRAINING_RESULTS.md) · [실행 계약](docs/V15_CAMERA_TRAINING.md).

v1.5.1은 같은 중점 모델에 GT/예측 상체 조건을 넣어 손 오류를 분리하는 **재학습 없는 파생 진단**입니다. T4 실제 결과를 검토했습니다. GT 상체로 손 local 오차는80.85→72.14mm로 줄었지만 손 축소는 남았습니다. [결과·한계](docs/V151_UPPER_CONDITION_RESULTS.md) · [실행 계약](docs/V151_UPPER_CONDITION.md) · [버전·범위](docs/EXPERIMENT_VERSIONS.md).

v1.5.2 [저장 모델 복원·생성 진단](docs/V152_SAMPLING_DIAGNOSTIC.md)과 v1.5.3 [예측2D 단일클립 학습](docs/V153_PREDICTED_CLIP.md)을 병행 준비했습니다. 같은 보존 입력을 재사용하며 관련 로컬30검사를 통과했습니다. v1.5.2 T4 결과를 회수해 저장 지표를 검토했습니다. DDIM10→50 확대의 이득은 작고 손 축소가 남았습니다. 환경 간 fingerprint 재현은 미확인입니다. v1.5.3 T4 단일클립 각5,000update에서 body10.78mm·hand local5.65mm로 형태 복원을 확인했습니다. GT 상체를 사용한 train 암기 진단이며 일반화 성능은 아닙니다. [v1.5.3 결과](docs/V153_PREDICTED_CLIP_RESULTS.md). [v1.5.2 결과](docs/V152_SAMPLING_RESULTS.md).

## 데이터 사용 방식

2026-11-18 사용자 결정으로 데이터셋 범위를 **ArchVis·CityPark·Downtown의 RGB·pose**로 고정했습니다. Factory와 depth는 현재 범위에서 제외합니다. 사용자가 중복 제거 완료를 확인했으며, 전체 파일 해시 검증 완료를 뜻하지 않습니다. [범위·검증 상태](docs/DATA_ACQUISITION_STRATEGY.md).

2026-11-04 결정에 따라 자동 인덱싱 운영을 중단하고, 사용자가 압축 해제해 Drive에 업로드하는 **RGB·pose 추출본**을 사용합니다. Depth는 초기 필수 입력이 아닙니다. 기존 압축 원본·인덱스·실험 근거는 보존합니다.

- 좌우 RGB와 pose를 sequence/frame 키로 대조하고 파일 무결성·관절·단위·좌표·시간 정합을 검증합니다.
- Sequence와 원본 motion 중복을 고려해 train/validation을 분리합니다. Test set으로 튜닝하지 않습니다.
- 필요한 subset을 Colab에서 준비·재사용하고 manifest·설정·결과·checkpoint는 영구 보관합니다.
- Drive 업로드 완료·정확한 경로·전체 내용은 확인 전이며, 추출본 manifest·T1 로더는 구현했으며 2D trainer의1,000페어 GPU 실행은 완료했으며 전체 데이터 내용 검증·학습은 미완료입니다. [입력 계약](docs/EXTRACTED_STEREO_INPUT.md)을 참조합니다.

자세한 준비 기준은 [데이터 확보 전략](docs/DATA_ACQUISITION_STRATEGY.md)을 따릅니다.

## 실행 안내

학습된 v1.2 모델을 전달하거나 수정하려면 [모델 공유 패키지 안내](docs/V12_MODEL_SHARING.md)를 참고하세요. 고정 소스·모델·실행 설명을 함께 제공하며, 새 파인튜닝 runner의 구현 완료를 뜻하지 않습니다.

생성 파일은 [산출물 위치 안내](docs/OUTPUTS.md)에서 현재 실행 자료·발표 이미지·과거 아카이브로 구분해 찾을 수 있습니다.

핵심 코드는 Python 모듈에, notebook은 환경 준비·실행 진입점에 둡니다. 주 학습 환경은 Colab이며 실제 실행의 GPU·디스크·학습 예산은 별도로 정합니다. 데이터·checkpoint·로컬 산출물은 Git에 포함되지 않으므로 각 가이드의 입력과 의존성을 먼저 확인합니다.

| 목적 | 진입점과 가이드 |
| --- | --- |
| Drive 추출본 검사·누락 복구·시퀀스 업로드 | [단독 Colab 노트북](notebooks/unrealego2_drive_upload.ipynb) · [실행 안내](docs/UNREALEGO2_DRIVE_UPLOAD.md) |
| Colab 환경과 결과 보관 | [Colab 준비](docs/COLAB_SETUP.md) |
| Pose JSON 감사·관절·좌표 검토 | [M0 도구](docs/M0_JSON_TOOLS.md), [좌표계·학습 관절 검토](docs/COORDINATE_AND_TRAINING_REVIEW.md) |
| 추출본 2D 학습·재개 준비 | [학습 계약](docs/ARCHVIS_DIRECT_RGB2D.md) · [Colab 인계](docs/ARCHVIS_COLAB_HANDOFF.md) · [준비 결과 복원](docs/ARCHVIS_PREPARED_RESTORE.md) — 복원 후1,000페어 GPU 진단 완료 |
| 1,000페어 v0 2D 학습 | [모델·설정·저장/재개](docs/ARCHVIS_V0_TRAINING.md) — 20epoch 결과 검토 완료 |
| v1 연속 구간·입력 묶음 준비 | [선택·병렬 준비·복원](docs/V1_INPUT_PREPARATION.md) — 실제 준비·감사 완료, Drive 보존 캐시 재사용 |
| v1 논문 기반 teacher 후보 | [구조·검증·남은 연결](docs/V1_TEACHER_IMPLEMENTATION.md) — 합성 CPU 검증, 실제 학습 전 |
| v1 frontend·teacher 전체 실행 | [학습·재개·캐시 가이드](docs/V1_TRAINING_RUNNER.md) · [Colab 진입점](notebooks/archvis_v1_pipeline.ipynb) — 실제 입력 근거와 고정 인계 필요 |
| v1 실제 입력·A100 시험 | [고정 테스트 인계](docs/V1_ACTUAL_TEST_HANDOFF.md) — 입력 감사 후 근거 충족 시 학습 |
| 명목25FPS v1 진단 학습 | [V1-DIAGNOSTIC-001](docs/V1_DIAGNOSTIC_001.md) — 사용자 승인 가정·참고 validation을 명시한 별도 시험 |
| 실제 데이터 학습 준비 | [전체 학습 준비](docs/FULL_TRAINING_READINESS.md) — 지원 기능과 미완료 adapter 구분 |
| 저장된 진단 모델의 예측·GT 표시 | [이미지 실행 안내](docs/PREDICTION_OVERLAY.md) — 두 결과를 별도로 생성 |
| 데이터 없는 합성 학습·개발 검증 | [M1](docs/M1_BASELINE.md), [M2](docs/M2_BODY_TEACHER.md), [M3](docs/M3_WHOLE_BODY_TEACHER.md) |

합성 검증 통과는 실제 데이터 정확도를 뜻하지 않습니다. 실제 학습은 작은 sample → overfit 진단 → 제한된 pilot 순서로 확인하며, 지표에 데이터 split·관절 구성·좌표계·측정 범위를 함께 기록합니다.

## 다음 작업

세 파생 시험은 **v1.2.1 도시 전이 → v1.2.2 ArchVis 대조 → v1.2.3 원점 민감도**로 명명합니다. 기존 실행 ID와 산출물은 보존합니다. 사용자 결정으로 v1.5를 카메라 표현 통제 학습 중심으로 재정의해 구현·로컬 검증했으며, 별도 v1.6은 폐기하고 필요한 내용을 흡수했습니다. T4 학습·개발 표본 평가를 완료했으며 잔여 손·자세 오류는 남았습니다. [번호 대응과 범위 비교](docs/EXPERIMENT_VERSIONS.md).

v1.2 본 학습과 도시 전이·ArchVis 대조·원점 민감도 세 파생 시험의 목적, 결과, 원인 판단과 후속 제안을 [종합 보고서](docs/V12_SYNTHESIS_REPORT.md)에 정리했습니다.

같은6클립300frame의 **카메라 원점 민감도 진단을 완료**했습니다. 원점만 정렬해 CityPark body6,721.67→276.27mm, Downtown2,277.83→306.05mm로 개선됐고 ArchVis의+100m 조건은 다시 축소됐습니다. 절대원점 의존성이 주요 기여 요인이라는 근거이며, ArchVis 손 악화도 있어 최종 해결책·일반화 완료는 아닙니다. 다음은 학습·추론에 일관된 camera 표현의 실험 설계 제안입니다. [결과·한계](docs/V12_CAMERA_GAUGE_RESULTS.md) · [실행 계약](docs/V12_CAMERA_GAUGE.md).

고정 v1.2 모델의 CityPark·Downtown 공식 validation200frame 무재학습 평가를 완료했습니다. 실행·회수는 성공했지만 body4,499.75mm와 심한3D 축소·위치 이탈을 확인했습니다. 원인은 미확정이며 같은 추론 경로의 ArchVis 대조와 입력 조건 검토를 제안합니다. v1.3 혼합학습은 계속 보류합니다. [결과·한계](docs/V12_CROSS_SCENARIO_RESULTS.md) · [실행 계약](docs/V12_CROSS_SCENARIO.md).

ArchVis validation100frame의 동일경로 대조와 camera300행 비교를 완료했습니다. Body197.34mm이며 도시와 달리 몸 크기를 유지했습니다. 카메라 조건의 큰 환경 차이도 확인했지만 실패 원인으로 입증한 것은 아닙니다. 모델 변경·재학습 없이 검토했습니다. [대조 결과·후속 가설](docs/V12_CAMERA_CONTROL_RESULTS.md) · [실행 계약](docs/V12_CAMERA_CONTROL.md).

v1·v1.1·v1.2·v1.4의 조건과 결과를 [종합 비교](docs/V1_COMPARISON_AND_V15_PLAN.md)했습니다. v1.5는 같은 클립에서 GT와 예측 입력의 효과를 분리하는 설계 제안이며 아직 구현·학습하지 않았습니다.

실험의 분기 트리·설계 목적·원인 가설·결과별 다음 판단은 [v1 실험 의사결정 기록](docs/V1_EXPERIMENT_LINEAGE.md)에 누적합니다. v1.1의 형태 축소를 분리하는 **v1.4: 한 train 클립의 GT 2D·GT 상체 조건 학습**을 구현·로컬 검증했습니다. [실행 계약](docs/V14_ORACLE_OVERFIT.md). 실제 단일클립 GT조건 학습은 완료됐으며 [결과](docs/V14_ORACLE_RESULTS.md)를 참고합니다.

v1.1은 같은1,000페어의epoch확대, v1.2는ArchVis 전체train확대를 병행 준비합니다. v1.3다른시나리오추가는보류합니다. [분기실험·비용](docs/V1_BRANCH_EXPERIMENTS.md). 아래기존순서보다이결정을우선합니다.

1. v1은 train1,000페어 규모를 유지하고 REWIND 논문 구조 재현을 우선합니다(2026-11-10 결정).
2. 연속 동작 구간 중심의 데이터 구성과 준비 병렬화·캐시 재사용을 설계합니다.
3. 논문 대비 구조 차이를 보완하고 실제 body/hand teacher 학습과 student 증류를 연결합니다.
4. ArchVis 전체 확대와 영상 시연은 이후 결과를 보고 진행합니다.

이 순서는 개발 계획이며 각 단계의 데이터·자원·완료 조건을 충족한 뒤 진행합니다. 최신 기준은 [v1 계획](docs/V1_REWIND_PLAN.md)이며, 기존 범위는 [미니 프로토타입 계획](docs/MINI_PROTOTYPE_PLAN.md)과 [영상 입력 계획](docs/VIDEO_INPUT_PLAN.md)을 따릅니다.

## 프로젝트 문서

| 문서 | 역할 |
| --- | --- |
| [ROADMAP](docs/ROADMAP.md) | M0~M6 단계, 완료 기준, 결정 사항 |
| [AGENTS](AGENTS.md) | 개발·검토·실험·커밋 작업 원칙 |
| [구현 이력](docs/IMPLEMENTATION_HISTORY.md) · [이전 README 기록](docs/README_HISTORY_20261105.md) | 과거 실험 결과·상세 명령·종료된 인덱싱 기록 |
| [로컬 보관·복원](docs/LOCAL_STORAGE.md) | 2026-11-05 정리와 아카이브 복원 방법 |

## 참고 자료

- Lee et al., *REWIND: Real-Time Egocentric Whole-Body Motion Diffusion with Exemplar-Based Identity Conditioning*, CVPR 2025. [공식 프로젝트](https://jyunlee.github.io/projects/rewind/)
- [UnrealEgo 공식 데이터 안내](https://unrealego.mpi-inf.mpg.de/)
- 세부 논문 근거와 프로젝트 구현 가정은 [로드맵](docs/ROADMAP.md#근거-자료)을 참조합니다.
