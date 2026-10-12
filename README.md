# REWIND-like prototype

REWIND 논문의 핵심 구조를 공개 데이터에서 재현하고, 데이터와 diffusion 학습·추론 방식을 수정하며 정확도·강건성·실시간성을 개선하는 연구용 프로젝트입니다.

논문과 동일한 절대 성능을 보장하는 공식 구현이 아닙니다. 비교 가능한 기준 모델과 재현 가능한 실험 체계를 먼저 구축합니다.

## 현재 상태

- 프로젝트 로드맵 v0.1을 작성했으며 사용자 검토와 범위 조정 단계입니다.
- 데이터 준비, 모델 구현, 학습·평가는 아직 시작하지 않았습니다.
- 실행 코드와 의존성이 마련되면 검증한 설치·실행 명령을 이 문서에 추가합니다.

## 문서 안내

| 문서 | 역할 |
| --- | --- |
| [프로젝트 로드맵](docs/ROADMAP.md) | 단계별 목표, 완료 기준, 실험 계획과 검토할 결정 사항 |
| [에이전트 작업 지침](AGENTS.md) | 구현·실험·검증·문서 수정 시 지켜야 할 원칙 |

프로젝트 계획은 `docs/ROADMAP.md`를 기준으로 관리합니다. 이후 변경은 문서에 반영합니다. 로드맵의 미결정 항목 D1~D5와 잠정 목표는 확정된 요구사항과 구분합니다.

## 목표와 범위

입력은 스테레오 egocentric 영상과 카메라 자세이며, 출력은 몸과 손의 3D 관절 위치입니다. 다음 구성을 순차적으로 구현할 계획입니다.

1. 2D keypoint·confidence front-end와 regression 비교 모델
2. Graph Transformer와 causal relative-temporal Transformer를 포함한 body diffusion teacher
3. 상체 예측을 조건으로 사용하는 body–hand cascade
4. One-step student distillation과 streaming inference
5. 데이터 확장, diffusion 변경, 관측 오류에 대한 강건성 개선 실험

Identity conditioning과 avatar용 inverse kinematics는 후속 선택 과제입니다. Floor-alignment, 외부 static camera 융합, 완전한 physics/contact·self-penetration solver는 초기 범위에서 제외합니다.

## 개발 순서

| 단계 | 핵심 산출물 |
| --- | --- |
| M0 | 데이터 규격, 좌표 검증, 고정 split |
| M1 | 2D front-end, 특징 캐시, regression 기준 |
| M2 | Causal body diffusion teacher |
| M3 | Whole-body teacher와 cascade 평가 |
| M4 | One-step student와 정확도·지연 시간 비교 |
| M5 | 단일 변경별 개선 실험과 failure benchmark |
| M6 | 최종 모델, 평가 결과, 재현 가이드 |

각 단계의 진입 조건과 실험 E1~E8의 세부 내용은 [로드맵](docs/ROADMAP.md)에 기록합니다. 다음 착수 후보는 M0의 소규모 데이터 샘플 검증입니다.

## 개발과 실험 환경

- 주 학습 환경은 Google Colab GPU를 계획하고 있습니다. 실제 GPU, 저장 공간과 학습 예산은 미정입니다.
- 초기 데이터 후보는 UnrealEgo이며, annotation과 calibration을 샘플로 확인한 뒤 UnrealEgo2로 확장합니다.
- 데이터셋의 용량이 매우 크므로, 로컬 파일이 아닌 클라우드 위에 데이터셋을 올려서 학습합니다.
- 작은 데이터의 입출력 검증과 과적합 테스트를 통과한 뒤 학습 규모를 늘립니다.
- Notebook은 실행 진입점으로 사용하고 핵심 로직은 Python 모듈로 분리할 계획입니다.
- 설정, 데이터 분할, seed, checkpoint, 환경과 평가 결과를 함께 기록합니다.

## 평가 원칙

Body/Hand MPJPE와 PA-MPJPE, 구조·시간 오차, failure subset 성능을 함께 평가합니다. 속도는 3D 모델만의 지연과 front-end를 포함한 전체 지연을 구분합니다. 30 FPS는 지정 GPU에서 검증할 잠정 목표입니다.

개선 실험에서는 한 번에 하나의 요인을 변경하고 같은 조건의 기준 모델과 비교합니다. Test set은 최종 평가에 사용하며, 모델 선택과 튜닝은 validation에서 수행합니다. 아직 측정된 성능 수치는 없습니다.

## 참고 자료

- Lee et al., *REWIND: Real-Time Egocentric Whole-Body Motion Diffusion with Exemplar-Based Identity Conditioning*, CVPR 2025. [공식 프로젝트](https://jyunlee.github.io/projects/rewind/)
- [UnrealEgo 공식 데이터 안내](https://unrealego.mpi-inf.mpg.de/)
- 사용자가 제공한 REWIND 논문 및 프로젝트 요약서. 세부 근거는 [로드맵의 근거 자료](docs/ROADMAP.md#근거-자료)를 참조합니다.
