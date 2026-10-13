# REWIND-like prototype

REWIND 논문의 핵심 구조를 공개 데이터에서 재현하고, 데이터와 diffusion 학습·추론 방식을 수정하며 정확도·강건성·실시간성을 개선하는 연구용 프로젝트입니다.

논문과 동일한 절대 성능을 보장하는 공식 구현이 아닙니다. 비교 가능한 기준 모델과 재현 가능한 실험 체계를 먼저 구축합니다.

## 현재 상태

- 프로젝트 로드맵 v0.2의 첫 구현 묶음인 M0 사전 JSON 감사·공통 로더·로컬 HTML viewer를 구현했습니다. 전체 완료 범위와 예산은 계속 검토 중입니다.
- 지정 Drive 폴더의 UnrealEgo2 pose JSON `frame_0`~`frame_17` 18개에서 87관절의 필수 배열 수치 검사를 통과했습니다. 단위·축·회전·시간·calibration·분할·공식 skeleton 검증은 남아 있으며 M0 전체 완료가 아닙니다.
- 2026-10-13 사용자 안내에 따라 데이터셋을 처음부터 다시 다운로드 중이며 당분간 실제 데이터 확인은 보류합니다. 위 결과는 재다운로드 전 샘플의 검증 기록이며, 현재 파일의 존재나 다운로드 완료를 보장하지 않습니다.
- 모델 구현과 학습·평가는 아직 시작하지 않았습니다.
- Python 표준 라이브러리로 실행하며 설치할 추가 Python 패키지는 없습니다. 합성 테스트와 실제 sample smoke 결과는 [M0 도구 안내](docs/M0_JSON_TOOLS.md)에 기록합니다. 실제 브라우저 화면·조작 확인은 브라우저 URL 보안 정책으로 미실행입니다.

## 문서 안내

| 문서 | 역할 |
| --- | --- |
| [프로젝트 로드맵](docs/ROADMAP.md) | 단계별 목표, 완료 기준, 실험 계획과 검토할 결정 사항 |
| [에이전트 작업 지침](AGENTS.md) | 구현·실험·검증·문서 수정 시 지켜야 할 원칙 |
| [M0 JSON 도구 안내](docs/M0_JSON_TOOLS.md) | 실행법, 로더·mask·manifest 계약, sample 검증 근거와 제한 |

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

각 단계의 진입 조건과 실험 E1~E8의 세부 내용은 [로드맵](docs/ROADMAP.md)에 기록합니다. JSON 사전 도구 구현 이후에는 관절 규격·좌표·시간과 split을 검증하고 GT 2D 기반 진단 학습을 준비할 계획입니다. 대응 RGB 확보 후 예측 2D 기반 baseline으로 연결하며, 다음 구현은 별도 작업 지시 범위에 따릅니다. GT 입력 진단만으로 M1 완료를 선언하지 않습니다.

## 개발과 실험 환경

- 주 학습 환경은 Google Colab GPU를 계획하고 있습니다. 실제 GPU, 저장 공간과 학습 예산은 미정입니다.
- 기존 초기 데이터 후보는 UnrealEgo입니다. 현재 확보된 UnrealEgo2 JSON으로 사전 검증을 제안하며, 최종 baseline 데이터셋·버전·split은 검증 후 결정합니다.
- 데이터셋의 용량이 매우 크므로, 로컬 파일이 아닌 클라우드 위에 데이터셋을 올려서 학습합니다.
- 작은 데이터의 입출력 검증과 과적합 테스트를 통과한 뒤 학습 규모를 늘립니다.
- Notebook은 실행 진입점으로 사용하고 핵심 로직은 Python 모듈로 분리할 계획입니다.
- 설정, 데이터 분할, seed, checkpoint, 환경과 평가 결과를 함께 기록합니다.

## M0 사전 도구 실행

프로젝트 root에서 Python 3.10 이상으로 실행합니다. 아래 `python`은 실제 Python 실행파일을 가리켜야 합니다. 이 세션에서는 Codex 번들 Python 3.12.14로 명령을 검증했습니다. 원본 sample은 `data/`, 생성 산출물은 `outputs/`에 두며 두 경로 모두 Git에서 제외합니다.

```sh
python -m rewind_data --input data/raw/unrealego2_sample --skeleton configs/skeleton_unrealego2_dev.json --dataset-version UnrealEgo2_train_val_pose_unverified --provenance data/sample_source.json --output outputs/m0_sample
python -m unittest discover -s tests -v
python tests/smoke_sample.py --input data/raw/unrealego2_sample --skeleton configs/skeleton_unrealego2_dev.json --provenance data/sample_source.json --expected-frames 18 --expected-joints 87
```

위 데이터 의존 명령은 sample과 provenance가 다시 준비된 뒤 실행합니다. 기존 검증에 사용한 파일은 Git에 포함되지 않으며 현재 가용성을 가정하지 않습니다. 합성 fixture 단위 테스트는 데이터셋 없이 실행할 수 있습니다. 일반 데이터에는 `--input`을 바꾸고 `--provenance`를 생략할 수 있습니다. 감사 오류가 있으면 보고서를 생성하면서 exit code 1을 반환합니다. 산출물은 `manifest.json`, `audit_report.json`, `viewer.html`입니다. Viewer는 외부 서버/CDN 없이 로컬 브라우저에서 열도록 작성했으며 실제 브라우저 호환성 검증은 남아 있습니다. 자세한 데이터 준비·선택 규칙은 [도구 안내](docs/M0_JSON_TOOLS.md)를 확인합니다.

## 평가 원칙

Body/Hand MPJPE와 PA-MPJPE, 구조·시간 오차, failure subset 성능을 함께 평가합니다. 속도는 3D 모델만의 지연과 front-end를 포함한 전체 지연을 구분합니다. 30 FPS는 지정 GPU에서 검증할 잠정 목표입니다.

개선 실험에서는 한 번에 하나의 요인을 변경하고 같은 조건의 기준 모델과 비교합니다. Test set은 최종 평가에 사용하며, 모델 선택과 튜닝은 validation에서 수행합니다. 아직 측정된 성능 수치는 없습니다.

## 참고 자료

- Lee et al., *REWIND: Real-Time Egocentric Whole-Body Motion Diffusion with Exemplar-Based Identity Conditioning*, CVPR 2025. [공식 프로젝트](https://jyunlee.github.io/projects/rewind/)
- [UnrealEgo 공식 데이터 안내](https://unrealego.mpi-inf.mpg.de/)
- 사용자가 제공한 REWIND 논문 및 프로젝트 요약서. 세부 근거는 [로드맵의 근거 자료](docs/ROADMAP.md#근거-자료)를 참조합니다.
