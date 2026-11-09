# REWIND-like prototype

스테레오 egocentric 영상에서 몸과 손의 3D 관절을 추정하는 연구용 프로토타입입니다. REWIND 논문의 구조를 참고해 비교 모델, causal diffusion teacher와 student를 단계적으로 개발합니다. 공식 구현이나 논문 성능 재현이 완료된 모델은 아닙니다.

최종 목표는 이미지 쌍과 동영상을 처리하는 미니 프로토타입입니다. 화면 UI는 학습·평가·추론 코어와 분리하며, 현재는 데이터 준비와 모델 학습 개선에 집중합니다.

## 현재 상태

2026-11-09 기준입니다.

| 항목 | 확인한 내용 | 남은 작업 |
| --- | --- | --- |
| M0 데이터 | 기존 371개 페어 확보·모두 train, 추출본 manifest·T1 로더 구현 | 실제 입력 검증과 독립 validation |
| M1 비교 모델 | 102페어 CPU 진단·합성 GPU 재개 검증, 추출본2D trainer·CE 옵션 구현 | 2D 정확도 개선, 독립 평가와 실제 GPU 학습 |
| M2·M3 teacher | Causal body diffusion·body–hand cascade 구현과 합성 검증 | 실제 데이터 학습과 성능 비교 |
| M4~M6 | Student 증류·개선 실험·통합 계획 | 후속 구현·평가 |
| 이미지·동영상 | 저장된 진단 모델의 stereo 이미지 2D 추론·표시 | GT 없는 3D 결과 연결, 동영상 decoder·시간 정합·UI |

실제 102페어의 train-fit 진단 지표는 측정했지만 정확도가 낮고 독립 평가는 미완료입니다. Joyful Jump 3프레임의 실제 예측도 관절이 화면 중앙에 몰리는 실패를 보였습니다. 추가 로컬 Joyful Jump 57페어는 중복·split 검토 전이므로 기존 371개와 합산하지 않습니다. 해당 57프레임의 GT 관절점 이미지 171장은 **정답 시각화이며 모델 예측 결과가 아닙니다.**

## 데이터 사용 방식

2026-11-04 결정에 따라 자동 인덱싱 운영을 중단하고, 사용자가 압축 해제해 Drive에 업로드하는 **RGB·pose 추출본**을 사용합니다. Depth는 초기 필수 입력이 아닙니다. 기존 압축 원본·인덱스·실험 근거는 보존합니다.

- 좌우 RGB와 pose를 sequence/frame 키로 대조하고 파일 무결성·관절·단위·좌표·시간 정합을 검증합니다.
- Sequence와 원본 motion 중복을 고려해 train/validation을 분리합니다. Test set으로 튜닝하지 않습니다.
- 필요한 subset을 Colab에서 준비·재사용하고 manifest·설정·결과·checkpoint는 영구 보관합니다.
- Drive 업로드 완료·정확한 경로·전체 내용은 확인 전이며, 추출본 manifest·T1 로더는 구현했으며 2D trainer 연결은 로컬 검증했으며 실제 검증·GPU 학습은 미완료입니다. [입력 계약](docs/EXTRACTED_STEREO_INPUT.md)을 참조합니다.

자세한 준비 기준은 [데이터 확보 전략](docs/DATA_ACQUISITION_STRATEGY.md)을 따릅니다.

## 실행 안내

핵심 코드는 Python 모듈에, notebook은 환경 준비·실행 진입점에 둡니다. 주 학습 환경은 Colab이며 실제 실행의 GPU·디스크·학습 예산은 별도로 정합니다. 데이터·checkpoint·로컬 산출물은 Git에 포함되지 않으므로 각 가이드의 입력과 의존성을 먼저 확인합니다.

| 목적 | 진입점과 가이드 |
| --- | --- |
| Drive 추출본 검사·누락 복구·시퀀스 업로드 | [단독 Colab 노트북](notebooks/unrealego2_drive_upload.ipynb) · [실행 안내](docs/UNREALEGO2_DRIVE_UPLOAD.md) |
| Colab 환경과 결과 보관 | [Colab 준비](docs/COLAB_SETUP.md) |
| Pose JSON 감사·관절·좌표 검토 | [M0 도구](docs/M0_JSON_TOOLS.md), [좌표계·학습 관절 검토](docs/COORDINATE_AND_TRAINING_REVIEW.md) |
| 추출본 2D 학습·재개 준비 | [학습 계약](docs/ARCHVIS_DIRECT_RGB2D.md) · [Colab 인계](docs/ARCHVIS_COLAB_HANDOFF.md) — 실제 실행은 테스트 세션에서 진행 |
| 실제 데이터 학습 준비 | [전체 학습 준비](docs/FULL_TRAINING_READINESS.md) — 지원 기능과 미완료 adapter 구분 |
| 저장된 진단 모델의 예측·GT 표시 | [이미지 실행 안내](docs/PREDICTION_OVERLAY.md) — 두 결과를 별도로 생성 |
| 데이터 없는 합성 학습·개발 검증 | [M1](docs/M1_BASELINE.md), [M2](docs/M2_BODY_TEACHER.md), [M3](docs/M3_WHOLE_BODY_TEACHER.md) |

합성 검증 통과는 실제 데이터 정확도를 뜻하지 않습니다. 실제 학습은 작은 sample → overfit 진단 → 제한된 pilot 순서로 확인하며, 지표에 데이터 split·관절 구성·좌표계·측정 범위를 함께 기록합니다.

## 다음 작업

1. 고정 Colab 준비 묶음으로 추출본 manifest·T1 로더·2D trainer를 실제 데이터에서 검증합니다.
2. 초기 약 1,000 train 페어와 독립 validation으로 데이터·학습 동작을 검증한 뒤 ArchVis 전체 train으로 바로 확대합니다(2026-11-09 사용자 결정). 중간 10,000페어 단계는 두지 않습니다.
3. 2D 학습을 개선하고 예측 2D 기반 3D baseline을 평가합니다.
4. 실제 M2·M3 학습과 영상 입력·결과 표시로 확장합니다.

이 순서는 개발 계획이며 각 단계의 데이터·자원·완료 조건을 충족한 뒤 진행합니다. 자세한 범위는 [미니 프로토타입 계획](docs/MINI_PROTOTYPE_PLAN.md)과 [영상 입력 계획](docs/VIDEO_INPUT_PLAN.md)을 따릅니다.

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
