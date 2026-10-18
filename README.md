# REWIND-like prototype

REWIND 논문의 핵심 구조를 공개 데이터에서 재현하고, 데이터와 diffusion 학습·추론 방식을 수정하며 정확도·강건성·실시간성을 개선하는 연구용 프로젝트입니다.

논문과 동일한 절대 성능을 보장하는 공식 구현이 아닙니다. 비교 가능한 기준 모델과 재현 가능한 실험 체계를 먼저 구축합니다.

## 현재 상태

- 프로젝트 로드맵 v0.2의 첫 구현 묶음인 M0 사전 JSON 감사·공통 로더·로컬 HTML viewer를 구현했습니다. 전체 완료 범위와 예산은 계속 검토 중입니다.
- 이전 M0에서 지정 Drive 폴더의 UnrealEgo2 pose JSON `frame_0`~`frame_17` 18개, 각 87관절의 필수 배열 수치 검사를 통과했습니다. 이는 이전 검증 이력이며 단위·축·회전·시간·calibration·분할·공식 skeleton 검증은 남아 있습니다.
- 2026-10-14 Drive 목록을 재확인했고, 별도로 공식 서버 CityPark 원본에서 부분 다운로드·추출 가능성을 시험했습니다. RGB 앞 2MiB에서 PNG 2개, pose 앞 256KiB에서 JSON 16개를 완전히 읽었습니다. 이후 새 준비 도구로 저장된 prefix를 로컬 재생해 18개 hash 일치를 확인했습니다. 서로 다른 sequence여서 완전 RGB 좌우+pose triplet은 0개이며 당시 Drive 복사본 실행·Colab 연결은 미검증이었으며, 후속 pose 시험은 아래에 구분합니다. 전체 무결성은 미검증입니다. 세부 근거와 한계는 [로드맵](docs/ROADMAP.md#부분-다운로드-조사와-실행-계획)에 기록합니다.
- 데이터 없이 공식 규격 근거와 M1 준비용 causal window 조립·streaming API를 추가했습니다. 합성 테스트 40개(기존 17 + window 23)와 fixture 실행을 통과했으며 실제 데이터 검증·모델 학습은 포함하지 않습니다.
- 2026-10-14 M1 학습 기반을 추가했습니다. 명시적 tensor 계약, 작은 single-frame/causal temporal regression, loss·평가·checkpoint 재개와 train-only 합성 RGB 예측 캐시 연결을 구현했습니다. 당시 Python 테스트 59개와 합성 CPU 학습·재개·평가를 통과했습니다. 실제 데이터 baseline·EfficientNet front-end·Colab GPU는 미검증이며 M1 전체 완료가 아닙니다.
- [Colab notebook](notebooks/m1_baseline.ipynb)을 준비했습니다. 로컬 소스 ZIP 업로드를 기본으로 하고 임시 디스크에서 합성 실행 후 선택적으로 결과를 Drive에 export합니다. 초기 코드 셀의 로컬 CPU 실행과 로컬 복사 검증을 통과했습니다. 이후 사용자 Colab의 소스 업로드·CPU 실행은 아래 후속 결과로 확인했으며 GPU 검증은 남아 있습니다.
- `rewind_archive`에 분할 gzip/tar의 상한 있는 순차 읽기·선택 추출·이름 대응·hash 재사용 검사를 추가했습니다. 당시 Python 테스트 83개(기존 59+archive 24)를 통과했고, 실제 HTTP 검증은 로컬 loopback 서버입니다. M0/M1 전체 완료가 아닙니다.
- Drive의 명시한 part ID·기대 이름만 조회해 size/version을 자동 확인하는 metadata→manifest builder와 별도 pilot 보고서를 추가했습니다. 당시 전체 테스트 93개를 통과했습니다. [준비 notebook](notebooks/archive_prepare.ipynb)에서 단일 JOB 설정 → 네트워크 없는 preflight → 기본 비활성 1회 metadata/pilot → 결과 검토·ZIP 보관 순서로 실행합니다. 별도 선택 JSON 업로드는 필요하지 않습니다.
- 2026-10-15 사용자 Colab에서 metadata HTTP 200·chunked·Content-Length 없음 때문에 `metadata_protocol` 실패가 관측됐습니다. 양쪽 metadata 경로를 byte/시간 상한 내 EOF 읽기로 수정했고 전체 97개 회귀와 실제 라이브러리 loopback 오류 검사를 통과했습니다. Media의 strict 206 Range 조건은 유지합니다. 후속 사용자 Colab 보고서에서 metadata 생성과 pose 32개 추출·verify 성공을 확인했습니다. 원본 앞 512KiB를 읽고 파일 수 상한에서 정상 중단했으며, 당시 관절·좌표·RGB 대응과 실제 학습은 미검증이었으며, 후속 JSON 감사는 아래에 구분합니다.
- JSON·window 도구는 Python 표준 라이브러리로 실행합니다. M1은 별도 PyTorch 환경이 필요하며 이번 검증은 Python 3.12.14 / torch 2.7.1+cpu입니다. 실행법과 한계는 [M1 안내](docs/M1_BASELINE.md)에 기록합니다. 기존 실제 sample 이력은 [M0 도구 안내](docs/M0_JSON_TOOLS.md)에 있으며 실제 브라우저 화면·조작 확인은 브라우저 URL 보안 정책으로 미실행입니다.
- 2026-10-15 후속 [ArchVis sample 감사](docs/ARCHVIS_POSE_AUDIT.md)에서 Drive ZIP의 32 JSON hash·87관절 필수 배열을 대조했습니다. 연속 18프레임 smoke와 실제 causal window 검사를 통과했고, 산재된 Opening 14개는 gap 진단을 유지합니다. 좌표 사이의 수치 관계를 확인했으며 단위·시간·RGB/calibration·split과 학습은 미검증입니다.

- 후속 공식 자료 대조에서 두 sample의 train 경로 소속과 Scaramuzza 3D→2D annotation 정합성을 확인했습니다. 실제 RGB 대응·단위·시간은 남아 있고, 공식 RGB 부분 읽기는 상한 부근 오류로 재사용 불가 처리했습니다. [후속 근거](docs/ARCHVIS_POSE_AUDIT.md)를 따릅니다.

- M2 body diffusion teacher의 [합성 구현 묶음](docs/M2_BODY_TEACHER.md)을 추가했습니다. Graph convolution·spatial attention·RoPE causal denoiser, x0/DDIM, 명시적 camera 계약·checkpoint 재개·streaming을 구현했습니다. 전체 107개 테스트와 CPU 합성 100→110 update 재개를 통과했습니다. 실제 body baseline·regression 대비 효과·GPU 성능은 미검증이며 M2 전체 완료가 아닙니다.

- M3 body–hand cascade의 [합성 구현 묶음](docs/M3_WHOLE_BODY_TEACHER.md)을 추가했습니다. 학습된 body checkpoint 고정, GT 상체 조건 hand 학습, 예측 상체 평가, 독립 none 모델·GT 진단 비교와 손목 상대/전체 좌표 지표를 구현했습니다. 전체 119개 테스트, CPU body 80·hand 각 100 update와 조건부 hand 110까지 재개를 통과했습니다. 실제 UnrealEgo whole-body teacher·Colab GPU·cascade 성능 개선은 미검증이며 M3 전체 완료가 아닙니다.

- 2026-10-16 사용자 제공 Colab 결과 ZIP·실행 notebook을 검토했습니다. Python 3.13.16 / torch 2.11.0+cpu에서 전체 119개 테스트, named-body 80·upper/none hand 각 100 update, upper hand 110까지 재개·평가 성공을 확인했습니다. 실행 소스는 `5e52aa3`이고 31개 코드 hash와 checkpoint 참조가 일치합니다. 실제 데이터·GPU 검증과 구분하며 [M3 검토 기록](docs/M3_WHOLE_BODY_TEACHER.md#사용자-colab-cpu-검증-2026-10-16)을 따릅니다.

- 2026-10-16 실제 ArchVis 32개 JSON의 87개 관절을 대조한 [관절 대응표](docs/JOINT_MAPPING.md)를 작성했습니다. 기존 개발용 62개·평가 후보 16개의 포함 여부와 body/upper/손목/손가락 후보, 미선택 25개를 구분했습니다. 이름 존재 확인과 모델 역할·부모 연결의 확정은 구분하며 학습 설정은 변경하지 않았습니다.

- 2026-10-16 후속 [단위·좌표계 및 학습 관절 검토](docs/COORDINATE_AND_TRAINING_REVIEW.md)에서 실제 32개 JSON의 world↔camera/device 변환과 카메라 degree/device radian 관계를 수치 대조했습니다. 거리 단위 cm는 논문의 rig 간격과 실제 baseline으로 강하게 지지되지만 UE2 raw 필드 직접 선언과는 구분합니다. Body22+finger40, upper13을 초기 학습 구성으로 권고하며, 모델 연결에는 좌표계의 축 반전 처리와 별도 변환기가 필요합니다. 설정·학습 가드는 유지했고 실제 학습은 실행하지 않았습니다.

2026-10-16 후속 작업으로 [실제 pose 변환기](docs/REAL_DATA_PREPARATION.md)와 별도 `real_archvis62_v1` 규격을 추가했습니다. Body22+finger40·upper13·평가16 및 graph를 프로젝트 가정으로 채택하고, cm 근거를 명시해 출력은 mm로 통일합니다. 기존 개발용 설정은 유지합니다. 변환 산출물은 검토용 candidate이며 학습 입력으로 자동 승인하지 않습니다. 대응 RGB·독립 validation·원본 motion 검증이 남아 있어 실제 학습은 아직 실행하지 않았습니다. [다음 실제 데이터 준비 실행](docs/REAL_DATA_NEXT_RUN.md)에 Drive 선택 입력과 실행 순서를 준비했습니다. 이후 네 번의 RGB 실행 실패와 개선 흐름은 아래 기록을 따릅니다. 전체 133개 테스트와 실제 32파일 변환·독립 수치 대조를 통과했습니다.

- 2026-10-16 사용자 RGB pilot 4개 ZIP을 대조했습니다. 모두 완전 파일 0개이며, 3회는 64KiB media마다 전후 metadata를 요청해 약 2.69MiB에서 총128회 상한에 도달했습니다. 같은 prefix 반복이어서 이어받기가 아닙니다. 첫 실행의 HTTP 비206 오류는 당시 실제 상태 코드가 기록되지 않아 원인을 확정하지 않습니다. [실행 결과와 다음 준비 순서](docs/REAL_DATA_NEXT_RUN.md)를 따릅니다.

- 요청 효율 개선에서 HTTP Range는 기본256KiB, 내부 read1·해제 출력은64KiB로 분리했습니다. 총128요청·8MiB·60초 상한과 원본 전후 검사는 유지합니다. 로컬 실제 전송 라이브러리 시험은 총96요청에서 byte 상한으로 중단했고, 별도 저장된 공식 prefix HTTP 재생에서 PNG2·JSON16의 기존 hash가 모두 일치했습니다. 당시 전체 141개 테스트와 notebook 10개 코드 셀의 로컬 검증을 통과했습니다. 이후 실제 Drive/Colab에서 오른쪽 RGB2개 추출은 확인됐으나 대응 stereo sample은 아직 없습니다. 아래 후속 검토를 따릅니다.

- 2026-10-16 후속 사용자 결과에서 Serena `Getting_Up__1_` 오른쪽 frame10·1 확보를 확인했습니다. 왼쪽만 선택한 `5935e818`은 8,356,024bytes·96요청에서 0파일로 중단했습니다. 기존 실행과 같은 000 앞부분만 읽었고 후속 조각까지 도달하지 않아, 불완전 다운로드가 직접 원인이라는 근거는 없습니다. 필터는 gzip seek 기능이 아닙니다.
- `archive_prepare`를 단일 설정과 위에서 아래로 실행하는 흐름으로 재작성했습니다. 새 기본 scan은 기존8MiB·60초 상한에서 저장0파일로 regular member 순서·경로·완료 여부를 기록하는 진단입니다. 전체 archive 인덱스가 아닙니다. 전체155개 로컬 테스트와 저장된 실제 prefix scan 재생에서 RGB3·pose17개 경로 관측, 완료payload2·16개와 저장0파일을 확인했습니다. 후속 2026-10-17 사용자 Colab scan 결과는 아래에 구분합니다. [다음 실행 안내](docs/REAL_DATA_NEXT_RUN.md)를 따릅니다.

- 2026-10-17 테스트 세션이 취합한 `af6961a`의 실제 Colab scan 결과를 독립 검토했습니다. 기존과 동일한000 앞8,350,136bytes에서 오른쪽 RGB 경로7개(완료payload6·미완료1)를 관측하고 전송 상한으로 중단했습니다. Scan의 저장0파일은 정상이며 왼쪽 경로는 이 구간에서 관측되지 않았습니다. ZIP CRC·보고서 연결·소스11개 hash를 확인했으며, 전체 archive에서의 왼쪽 부재나 위치는 미확정입니다. 같은8MiB 재실행은 종료하고 다음 읽기 범위·비용 정책을 별도로 검토합니다.

2026-10-17 후속 준비로 `counterpart64MiB_v1` 고정 예산과 정확한 목표 파일을 모두 저장하면 멈추는 추출 경로를 추가했습니다. 기본 8MiB scan은 유지합니다. 이번 사용자 테스트의 대상은 기존 오른쪽 frame1·10에 대응하는 왼쪽 두 PNG이며, 준비 상한은 64MiB·300초·768요청입니다. 전체 165개 로컬 회귀 검사와 8MiB 뒤 목표를 둔 독립 모의 Drive 검증을 통과했습니다. 후속 사용자 Colab 결과에서 왼쪽 frame1·10 두 PNG 확보와 검증 성공을 확인했습니다. 압축27.5MiB를 읽고 준비150.50초에 목표 완료로 중단했습니다. 대응 pose·좌우 영상/시간 정합성과 실제 학습은 아직 미검증입니다. 실행 결과와 한계는 [다음 실행 안내](docs/REAL_DATA_NEXT_RUN.md)를 따릅니다.

후속 pose 테스트는 압축7.5MiB·약29.9초에서 tar header1000 상한에 도달했고 목표 JSON0개로 종료했습니다. 동일 실행 반복을 중단하고 pose의 header·metadata·해제량 제한을 함께 검토해야 합니다. 실제 pose 확보·좌표/영상 대조·학습은 아직 미완료이며 [실행 안내](docs/REAL_DATA_NEXT_RUN.md)를 따릅니다.

2026-10-19 후속 준비에서는 pose 전용 `poseCounterpart64MiB_v1`을 추가해 전송64MiB·300초는 유지하고 header10,000·누적metadata1MiB·해제512MiB를 적용합니다. 단일metadata64KiB와 저장/disk 상한도 유지합니다. API4 보고서에는 마지막 regular 관측과 총 관측 수를 추가합니다. 전체175개 회귀와 새 소스 ZIP의80개 사전검사·모의 노트북 실행을 통과했습니다. 실제 재실행·목표 pose 확보는 아직 미검증이며 [고정 1회 실행 안내](docs/REAL_DATA_NEXT_RUN.md)를 따릅니다.

## 문서 안내

| 문서 | 역할 |
| --- | --- |
| [프로젝트 로드맵](docs/ROADMAP.md) | 단계별 목표, 완료 기준, 실험 계획과 검토할 결정 사항 |
| [에이전트 작업 지침](AGENTS.md) | 구현·실험·검증·문서 수정 시 지켜야 할 원칙 |
| [M0 JSON 도구 안내](docs/M0_JSON_TOOLS.md) | 실행법, 로더·mask·manifest 계약, sample 검증 근거와 제한 |
| [실제 관절 대응표](docs/JOINT_MAPPING.md) | Raw 87개 전체 이름, 개발용 subset·공식 평가 순서 대조와 역할 후보·미확정 항목 |
| [단위·좌표계 및 학습 관절 검토](docs/COORDINATE_AND_TRAINING_REVIEW.md) | 실제 변환 수치 검증, 단위 근거의 강도, body22+finger40 권고와 학습 진입 조건 |
| [실제 pose 변환기](docs/REAL_DATA_PREPARATION.md) | 채택한 초기 규격, mm·mask·camera·손목 상대 특징 변환과 candidate 검증 |
| [다음 실제 데이터 준비 실행](docs/REAL_DATA_NEXT_RUN.md) | 현재 확보 상태, 다음 Colab의 bounded prefix scan 순서와 남은 학습 조건 |
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

위 데이터 의존 명령은 **추출된 JSON 디렉터리용**입니다. 압축 원본·분할 파일을 `--input`으로 직접 읽지 않습니다. 별도 `rewind_archive`가 manifest·상한을 받아 제한된 선택 추출과 이름 대응을 처리하며 실제 split·좌표·단위 검증은 별도로 남깁니다. gzip 인덱스 없이 원하는 멤버에 도달하려면 앞부분의 순차 해제가 필요합니다. 사용자 Drive의 pose 소량 추출·Colab verify는 보고서로 확인했으며 실제 annotation 내용 검증은 남아 있습니다.

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

재개·checkpoint 식별·지표의 정확한 범위는 [M2 안내](docs/M2_BODY_TEACHER.md)와 [M3 안내](docs/M3_WHOLE_BODY_TEACHER.md)를 따릅니다. 현재 M3는 CPU float32 경로만 지원합니다. 기존 Colab 소스 ZIP에는 새 모듈이 없을 수 있으므로 최신 커밋으로 ZIP을 다시 만들고 새 소스 디렉터리에서 실행합니다. 실제 RGB/2D baseline과 GPU 학습·성능 검증은 남아 있습니다.

## 평가 원칙

Body/Hand MPJPE와 PA-MPJPE, 구조·시간 오차, failure subset 성능을 함께 평가합니다. 속도는 3D 모델만의 지연과 front-end를 포함한 전체 지연을 구분합니다. 30 FPS는 지정 GPU에서 검증할 잠정 목표입니다.

개선 실험에서는 한 번에 하나의 요인을 변경하고 같은 조건의 기준 모델과 비교합니다. Test set은 최종 평가에 사용하며, 모델 선택과 튜닝은 validation에서 수행합니다. 합성 코드 진단 수치만 있으며 실제 데이터 성능은 아직 측정하지 않았습니다.

## 참고 자료

- Lee et al., *REWIND: Real-Time Egocentric Whole-Body Motion Diffusion with Exemplar-Based Identity Conditioning*, CVPR 2025. [공식 프로젝트](https://jyunlee.github.io/projects/rewind/)
- [UnrealEgo 공식 데이터 안내](https://unrealego.mpi-inf.mpg.de/)
- 사용자가 제공한 REWIND 논문 및 프로젝트 요약서. 세부 근거는 [로드맵의 근거 자료](docs/ROADMAP.md#근거-자료)를 참조합니다.
