# REWIND-like prototype

REWIND 논문의 핵심 구조를 공개 데이터에서 재현하고, 데이터와 diffusion 학습·추론 방식을 수정하며 정확도·강건성·실시간성을 개선하는 연구용 프로젝트입니다.

논문과 동일한 절대 성능을 보장하는 공식 구현이 아닙니다. 비교 가능한 기준 모델과 재현 가능한 실험 체계를 먼저 구축합니다.

## 현재 상태

2026-10-27 기준입니다. **사용자 Colab 테스트는 일시 보류**하고 독립적인 문서 정리를 진행합니다. 재개 전에는 테스트 세션이 기존 고정 인계의 revision·소스 ZIP·ready notebook·캐시 식별값과 실행 조건을 다시 확인합니다. 이번 문서 수정으로 배포본을 교체하거나 같은 테스트를 반복하지 않습니다.

| 단계 | 확인한 구현·검증 | 남은 완료 조건 |
| --- | --- | --- |
| M0 데이터 | JSON 감사·window·candidate 변환기, 실제 ArchVis 32 JSON 검토, API9 bounded archive 준비 구현 | 현재 Serena 목표 pose 미확보, 대응 RGB·좌표/시간·split/원본 motion 및 학습 입력 검증. M0 전체 미완료 |
| M1 비교 기준 | 합성 regression·예측 2D 캐시·학습 재개 검증 | 실제 RGB front-end·예측 입력 baseline·GPU 검증. M1 전체 미완료 |
| M2 Body teacher | 합성 causal diffusion·DDIM·streaming·학습 재개 검증 | 실제 데이터 body 학습·비교 성능·GPU 검증. M2 전체 미완료 |
| M3 Whole-body teacher | 합성 body–hand cascade·세 조건 평가·Colab CPU 재개 검증 | 실제 body/hand 학습·cascade 성능·GPU 검증. M3 전체 미완료 |
| M4 Student | 미착수 | One-step 증류와 정확도·지연 비교 |
| M5~M6 개선·통합 | 미착수 | Ablation·failure 평가·최종 재현 |

최신 실제 API8 warm128은 보관 캐시를 재사용해 **128MiB 확장에 성공했지만 목표 pose는 0개**였습니다. API9 warm192는 구현·로컬 검토 후 revision `0f6c5ca`로 인계했으며 실제 192MiB 탐색은 미검증입니다. 전체 248개 회귀·153개 notebook 사전검사는 이전 구현의 검증 이력이며 이번 문서 작업에서 재실행하지 않았습니다.

최신 진입점은 [실제 데이터 확보 상태·보류 및 재개 기준](docs/REAL_DATA_NEXT_RUN.md), [캐시 계약](docs/PREFIX_CACHE.md), [192MiB 비용·상한](docs/PREFIX_CACHE_192_PLAN.md)입니다. 날짜별 구현·검증 기록은 [구현 이력](docs/IMPLEMENTATION_HISTORY.md), 세부 계획과 M0~M6·E1~E8·D1~D5는 [로드맵](docs/ROADMAP.md)에서 확인합니다. 이전 자료와 현재 테스트 보호 범위는 [로컬 보관·복원 안내](docs/LOCAL_STORAGE.md)를 따릅니다.

## 문서 안내

이전 실행 자료는 Git에서 제외되는 `.local-archive/20261023/`에 정리했습니다. 현재 테스트 파일은 기존 `outputs/` 경로를 유지합니다. 과거 `outputs/` 링크의 자료가 보이지 않으면 [로컬 보관·복원 안내](docs/LOCAL_STORAGE.md)와 아카이브의 `INDEX.json`에서 원래 경로를 확인하세요.

| 문서 | 역할 |
| --- | --- |
| [프로젝트 로드맵](docs/ROADMAP.md) | 현재 결정, 단계별 목표·완료 기준과 날짜별 상세 기록 |
| [구현·검증 이력](docs/IMPLEMENTATION_HISTORY.md) | README에서 분리한 이전 구현·실행 기록 |
| [로컬 보관·복원 안내](docs/LOCAL_STORAGE.md) | 현재 테스트 보호 범위와 이전 산출물의 개별 복원 |
| [에이전트 작업 지침](AGENTS.md) | 구현·실험·검증·문서 수정 시 지켜야 할 원칙 |
| [M0 JSON 도구 안내](docs/M0_JSON_TOOLS.md) | 실행법, 로더·mask·manifest 계약, sample 검증 근거와 제한 |
| [실제 관절 대응표](docs/JOINT_MAPPING.md) | Raw 87개 전체 이름, 개발용 subset·공식 평가 순서 대조와 역할 후보·미확정 항목 |
| [단위·좌표계 및 학습 관절 검토](docs/COORDINATE_AND_TRAINING_REVIEW.md) | 실제 변환 수치 검증, 단위 근거의 강도, body22+finger40 권고와 학습 진입 조건 |
| [실제 pose 변환기](docs/REAL_DATA_PREPARATION.md) | 채택한 초기 규격, mm·mask·camera·손목 상대 특징 변환과 candidate 검증 |
| [다음 실제 데이터 준비 실행](docs/REAL_DATA_NEXT_RUN.md) | 현재 확보 상태, Colab 테스트 보류·재개 기준과 남은 학습 조건 |
| [데이터·window 계약](docs/DATA_CONTRACT.md) | 공식 자료 근거, 관절 집합 구분, causal window 정책과 합성 검증 |
| [Causal window 독립 검토](docs/CAUSAL_WINDOW_REVIEW.md) | 오케스트레이션의 코드·합성 실행·출처 교차 확인 기록 |
| [M3 whole-body teacher 안내](docs/M3_WHOLE_BODY_TEACHER.md) | 고정 body·손목 상대 hand·조건 비교·cascade 평가와 CPU 합성 실행 |
| [M2 body teacher 안내](docs/M2_BODY_TEACHER.md) | Diffusion·camera·loss·streaming 계약과 CPU 합성 실행·한계 |
| [M1 학습 기반 안내](docs/M1_BASELINE.md) | Tensor·mask·loss·예측 캐시·checkpoint 계약과 합성 실행법 |
| [Colab 준비 안내](docs/COLAB_SETUP.md) | 소스 ZIP 전달, 런타임·Drive 준비와 결과 보관 |
| [압축 원본 준비 도구](docs/ARCHIVE_PREPARATION.md) | Source/예산 계약, 제한된 순차 추출, 완전 파일·CRC·매칭·재사용 검증 범위 |

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

- 주 학습 환경은 Google Colab GPU를 계획하고 있습니다. 실제 GPU, 저장 용량과 학습 예산은 미정입니다.
- 기존 초기 데이터 후보는 UnrealEgo입니다. 이전 UnrealEgo2 JSON으로 사전 도구를 검증했으며, 재확보 데이터의 최종 baseline 데이터셋·버전·split은 검증 후 결정합니다.
- **2026-10-13 운영 결정:** Drive에는 압축 원본을 보존하고 필요한 시퀀스를 Colab/runtime 임시 디스크에 추출해 읽습니다. 같은 런타임에서는 검증된 추출본을 재사용하며 세션 종료·추출본 소실 후 다시 추출합니다. 실험 결과·checkpoint·manifest는 영구 저장소에 분리 보관합니다.
- 작은 데이터의 입출력 검증과 과적합 테스트를 통과한 뒤 학습 규모를 늘립니다.
- Notebook은 준비·실행 진입점이며 JSON/window는 `rewind_data`, regression은 `rewind_m1`, body diffusion은 `rewind_m2`, body–hand cascade는 `rewind_m3` 모듈로 분리했습니다.
- 설정, 데이터 분할, seed, checkpoint, 환경과 평가 결과를 함께 기록합니다.

### Drive와 Colab의 데이터 준비 — pose 소량 추출 확인

Drive의 현재 분할 압축 원본을 유지하고, 인증된 Drive API의 `Range` 요청으로 일정 바이트씩 읽어 Colab CPU에서 순차 해제하는 구조입니다. `rewind_archive`의 local/HTTP 소스와 추출 코어는 합성·기존 공식 prefix 로컬 재생으로 확인했습니다. Metadata HTTP200 chunked 실패는 사용자 Colab에서 관측했고 수정은 모의/실제 라이브러리 loopback으로 검사했습니다. 수정본의 실제 pose manifest/pilot 성공은 사용자 보고서로 확인했습니다. 좌우 RGB·pose의 정합성·split·GPU 학습 연결은 미완료입니다. 전체 압축본·결합본·전체 추출본을 런타임에 보관하지 않습니다. 결과·checkpoint·manifest는 별도 영구 보관합니다. [Drive 부분 다운로드](https://developers.google.com/workspace/drive/api/guides/manage-downloads#partial_download), [Colab 운영 제약](https://research.google.com/colaboratory/faq.html)

- 현재 `000`, `001` 등은 하나의 gzip 스트림을 나눈 조각입니다. 별도 인덱스 없이 중간 조각만 독립 해제하거나 원하는 학습 배치 위치로 바로 이동할 수 없습니다. 네트워크 읽기 단위와 학습 배치는 다릅니다.
- 불완전한 archive도 `000`부터 끊김 없이 확보한 앞부분에서는 온전히 읽힌 파일을 sample 후보로 사용할 수 있습니다. 잘린 마지막 파일은 제외하며, 일부 추출 성공을 전체 무결성·완전한 시퀀스 확보로 해석하지 않습니다.
- 저장할 sample이 작아도 앞부분을 읽고 해제하는 전송량·시간은 클 수 있습니다. 응답 body·시간·해제/IO buffer·파일/disk·metadata/range 수 상한을 구현했으며 [개발용 설정](configs/archive_limits_smoke.json)은 Colab 실측 예산이 아닙니다. 프로세스 RSS나 DNS/auth/OS 전체 hard deadline을 보장하지 않습니다. 실제 처리량·운영 상한은 Colab 소량 시험 후 정합니다.
- 같은 런타임에서는 검증된 추출본을 재사용합니다. 런타임 삭제·추출본 소실 후에는 재준비가 필요합니다. 직접 메모리에서 학습에 공급하는 방식, gzip 인덱스와 독립 압축 묶음 생성은 후속 검토 후보이며 현재 구현·채택된 기능이 아닙니다.

작은 sample → overfit → pilot 순서는 실행 가능한 개발 방향입니다. 다만 논문 크기 모델·대규모 학습·반복 실험을 현재 Colab 예산으로 수행할 수 있는지는 GPU·VRAM·입출력·step 시간 측정 후 판단합니다. 30 FPS와 성능 개선은 달성이 보장된 결과가 아닙니다.

## M0 사전 도구 실행

프로젝트 root에서 Python 3.10 이상으로 실행합니다. 아래 `python`은 실제 Python 실행파일을 가리켜야 합니다. 이 세션에서는 Codex 번들 Python 3.12.14로 명령을 검증했습니다. 원본 sample은 `data/`, 생성 산출물은 `outputs/`에 두며 두 경로 모두 Git에서 제외합니다.

```sh
python -m rewind_data --input data/raw/unrealego2_sample --skeleton configs/skeleton_unrealego2_dev.json --dataset-version UnrealEgo2_train_val_pose_unverified --provenance data/sample_source.json --output outputs/m0_sample
python -m unittest discover -s tests -v
python tests/smoke_sample.py --input data/raw/unrealego2_sample --skeleton configs/skeleton_unrealego2_dev.json --provenance data/sample_source.json --expected-frames 18 --expected-joints 87
```

위 데이터 의존 명령은 **추출된 JSON 디렉터리용**입니다. 압축 원본·분할 파일을 `--input`으로 직접 읽지 않습니다. 별도 `rewind_archive`가 manifest·상한을 받아 제한된 선택 추출과 이름 대응을 처리하며 실제 split·좌표·단위 검증은 별도로 남깁니다. gzip 인덱스 없이 원하는 멤버에 도달하려면 앞부분의 순차 해제가 필요합니다. 사용자 Drive의 pose 소량 추출·Colab verify와 후속 32 JSON annotation 감사는 확인했습니다. 실제 영상·시간·학습 준비의 남은 조건은 [감사 기록](docs/ARCHVIS_POSE_AUDIT.md)을 따릅니다.

기존 18프레임 sample과 provenance가 다시 준비된 뒤 해당 smoke 명령을 실행하며, 다른 subset에는 입력 경로와 기대 프레임·관절 수를 검증한 범위에 맞춥니다. 일반 데이터에는 `--provenance`를 생략할 수 있습니다. 기존 파일은 Git에 포함되지 않고 현재 가용성을 가정하지 않습니다. 합성 fixture 테스트는 데이터셋 없이 실행할 수 있습니다. 감사 오류가 있으면 보고서를 생성하면서 exit code 1을 반환합니다. 산출물은 `manifest.json`, `audit_report.json`, `viewer.html`입니다. Viewer는 외부 서버/CDN 없이 작성했으며 실제 브라우저 호환성 검증은 남아 있습니다. Archive와 loader의 경계는 [도구 안내](docs/M0_JSON_TOOLS.md#압축-원본-준비-계획과-구현-상태)를 확인합니다.

## 데이터 없이 분할 archive 준비 실행

Python 표준 라이브러리 환경에서 실행합니다. 명시한 sequence/member glob만 저장하며 새 output을 지정합니다.

```sh
python -m rewind_archive.fixture --output outputs/archive_fixture
python -m rewind_archive extract --manifest outputs/archive_fixture/rgb_manifest.json --limits configs/archive_limits_smoke.json --include "Fixture/Day/person/avatar/001/Motion__1_/*" --output outputs/archive_rgb
python -m rewind_archive extract --manifest outputs/archive_fixture/pose_manifest.json --limits configs/archive_limits_smoke.json --include "Fixture/Day/person/avatar/001/Motion__1_/*" --output outputs/archive_pose
python -m rewind_archive verify --extraction outputs/archive_pose --manifest outputs/archive_fixture/pose_manifest.json
python -m rewind_archive match --extraction outputs/archive_rgb --extraction outputs/archive_pose --output outputs/archive_matching.json
```

이 합성 입력은 triplet 4개를 맞추지만 split·원본 motion·물리/영상 정합성을 검증한 학습 데이터가 아니므로 train_ready=false입니다. Stop reason·CRC·완전 파일과 수신 범위를 확인하고, 손상·원본 변경으로 unusable인 추출은 재사용하지 않습니다. Source/limits schema·측정 범위와 실제 Drive adapter의 제한은 [압축 준비 안내](docs/ARCHIVE_PREPARATION.md)에 기록합니다.

Colab의 실제 pilot 입력은 version을 수동으로 채우지 않습니다. [Metadata 선택 규격과 실행 순서](docs/ARCHIVE_PREPARATION.md#drive-metadata에서-manifest와-pilot으로)는 한 archive의 ID·기대 파일명·total_parts와 사용자 glob만 요구합니다. Metadata 기본 128KiB/30초와 준비 기본 8MiB/60초는 별도이며, 실패하면 report만 남기고 이전 성공 결과로 진행하지 않습니다. 같은 준비 코어를 로컬에서 `python -m rewind_archive pilot --manifest ... --limits ... --include ... --output ...`로 실행하면 verify와 compact pilot_report.json도 남깁니다. 실제 Drive 연결은 notebook의 별도 활성화 단계입니다.

## 데이터 없이 causal window 계약 실행

현재·과거만 조립하고 gap·sequence 경계에서 history를 초기화합니다. 중복/역순과 구조 오류는 거부하며, 부족한 과거는 `None` + `time_valid`로 표시합니다. Field별 mask와 unknown visibility를 보존하고, FPS나 실제 시간을 만들지 않습니다. 실제 데이터나 기존 sample 산출물이 필요하지 않습니다.

```sh
python -m unittest discover -s tests -v
python -m rewind_data.window_fixture --window-size 3 --output outputs/synthetic_windows.json
```

Fixture는 `synthetic_fixture`로 구분된 6개 window를 출력합니다. 총 길이 정의, 명시적 container 경계의 streaming reset, 관절·단위·calibration 미확인 범위와 다음 진입 조건은 [데이터·window 계약](docs/DATA_CONTRACT.md)에 기록합니다. 기존 개발용 skeleton은 미검증 설정으로 유지합니다.

## 데이터 없이 M1 학습 기반 실행

PyTorch 환경에서 프로젝트 root를 기준으로 실행합니다. 이번 Windows 환경에서는 아래 `python`에 `.venv/Scripts/python.exe`를 사용했습니다. PyTorch가 없으면 M1 테스트는 건너뛰므로 [설치·검증 안내](docs/M1_BASELINE.md#환경과-의존성)를 먼저 확인합니다. 새 실행에는 비어 있는 output을 지정합니다.

```sh
python -m unittest discover -s tests -v
python -m rewind_m1 train --config configs/m1_single_synthetic.json --output outputs/m1_single --steps 80 --device cpu
python -m rewind_m1 train --config configs/m1_temporal_synthetic.json --output outputs/m1_temporal --steps 80 --device cpu
python -m rewind_m1 train --config configs/m1_single_synthetic.json --output outputs/m1_single --steps 100 --resume outputs/m1_single/last.pt --device cpu
python -m rewind_m1 evaluate --checkpoint outputs/m1_single/last.pt --output outputs/m1_single/evaluation.json --device cpu
python -m rewind_m1.predicted_fixture --config configs/m1_single_synthetic.json --output outputs/m1_predicted_chain --frontend-steps 20 --steps 30
```

GT 입력과 예측 입력 모두 `synthetic_code_check`입니다. 합성 loss 감소는 실제 데이터 정확도·실시간성의 근거가 아닙니다. `last.pt`에 전체 regression 학습 상태를 저장하며 code/config/data/environment가 같을 때 재개합니다. 예측 경로의 같은 output 재개, 특징 계약·mask·집계 방식과 미검증 항목은 [M1 안내](docs/M1_BASELINE.md)에 기록합니다. Colab 준비는 [노트북](notebooks/m1_baseline.ipynb)과 [환경 안내](docs/COLAB_SETUP.md)를 따릅니다. 실제 데이터 준비·검증 조건을 생략하지 않습니다.

## 데이터 없이 M2·M3 teacher 실행

M2 body diffusion과 M3 body–hand cascade의 CPU 합성 진입점입니다. M3 fixture는 이름이 명시된 body 6관절과 손가락 4관절로 별도 body checkpoint를 학습·고정한 뒤, GT 상체 조건 hand와 상체 조건 없는 hand를 각각 학습합니다. 조건 없음·GT 상체 진단·예측 상체 cascade를 나눠 평가합니다. 실제 UnrealEgo의 전체 body/hand 관절 규격은 아닙니다.

```sh
python -m rewind_m2 train --config configs/m2_body_synthetic.json --output outputs/m2_body_new --steps 100 --device cpu
python -m rewind_m3.fixture --output outputs/m3_cascade_new --body-steps 80 --hand-steps 100
```

새 실행에는 새 output을 사용합니다. M3 결과는 `comparison.json`, 단계별 `report.json`·`last.pt`, 세 조건의 평가 JSON에 기록합니다. 손목은 body가 담당하고 손가락 상대 좌표를 예측 body 손목에 연결합니다. GT 상체 진단도 연결 위치는 예측 body 손목을 사용합니다. 손목 상대 오차와 전체 좌표 오차를 구분하며, 결합 규칙상 손목 연결 차이가 0인 것은 예측 정확도 증거가 아닙니다.

재개·checkpoint 식별·지표의 정확한 범위는 [M2 안내](docs/M2_BODY_TEACHER.md)와 [M3 안내](docs/M3_WHOLE_BODY_TEACHER.md)를 따릅니다. 현재 M3는 CPU float32 경로만 지원합니다. 다른 개발 버전의 새 실행에는 필요한 모듈을 포함한 소스가 필요합니다. 현재 보류한 warm192 시험은 인계된 고정 소스 ZIP·ready notebook을 유지하며 문서 커밋으로 교체하지 않습니다. 실제 RGB/2D baseline과 GPU 학습·성능 검증은 남아 있습니다.

## 평가 원칙

Body/Hand MPJPE와 PA-MPJPE, 구조·시간 오차, failure subset 성능을 함께 평가합니다. 속도는 3D 모델만의 지연과 front-end를 포함한 전체 지연을 구분합니다. 30 FPS는 지정 GPU에서 검증할 잠정 목표입니다.

개선 실험에서는 한 번에 하나의 요인을 변경하고 같은 조건의 기준 모델과 비교합니다. Test set은 최종 평가에 사용하며, 모델 선택과 튜닝은 validation에서 수행합니다. 합성 코드 진단 수치만 있으며 실제 데이터 성능은 아직 측정하지 않았습니다.

## 참고 자료

- Lee et al., *REWIND: Real-Time Egocentric Whole-Body Motion Diffusion with Exemplar-Based Identity Conditioning*, CVPR 2025. [공식 프로젝트](https://jyunlee.github.io/projects/rewind/)
- [UnrealEgo 공식 데이터 안내](https://unrealego.mpi-inf.mpg.de/)
- 사용자가 제공한 REWIND 논문 및 프로젝트 요약서. 세부 근거는 [로드맵의 근거 자료](docs/ROADMAP.md#근거-자료)를 참조합니다.
