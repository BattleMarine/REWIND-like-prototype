# REWIND-like prototype 프로젝트 로드맵

검토 초안 v0.2   |   2026년 10월 15일 운영·검증 상태 보완

REWIND의 핵심 구조를 공개 데이터에서 재현하고, 데이터와 diffusion 학습·추론 방식을 수정해 정확도·강건성·실시간성을 개선한다. 첫 목표는 비교 가능한 기준 모델과 재현 가능한 평가 체계를 확보하는 것이다.

### 프로젝트의 성공 기준

- 같은 데이터 분할·설정·checkpoint로 학습 재개와 평가를 재현할 수 있다.

- Body–hand cascade와 causal temporal modeling을 갖춘 teacher 및 one-step student를 확보한다.

- 기준 모델 대비 개선 효과를 동일 조건에서 검증하고, 정확도와 지연 시간의 교환 관계를 설명한다.

### 우선 범위

입력은 stereo egocentric images와 camera poses, 출력은 몸과 손의 3D keypoints로 둔다. Colab GPU에서 body-only 모델로 시작해 whole-body teacher와 one-step student로 확장한다. 논문 수치와의 절대적 일치보다 구조 재현과 내부 비교의 신뢰성을 우선한다.

Identity conditioning과 avatar용 inverse kinematics는 후속 선택 과제다. Floor-alignment, 외부 static camera 융합, 완전한 physics/contact·self-penetration solver는 초기 범위에서 제외한다.

| 단계 | 주요 산출물 | 다음 단계 진입 조건 |
| --- | --- | --- |
| M0 데이터 검증 | 데이터 규격과 고정 split | 좌표·관절·시퀀스 정합성 확인 |
| M1 입력과 비교 기준 | 2D front-end와 regression | 예측 2D로 학습·평가 완료 |
| M2 Body teacher | Causal body diffusion | 과적합·causality 검사 통과 |
| M3 Whole-body teacher | Body → hand cascade | 예측 상체로 손 평가 완료 |
| M4 Student | One-step 증류와 streaming | 정확도·latency 비교 완료 |
| M5 개선 실험 | 단일 변경별 ablation | 반복 실험으로 효과 확인 |
| M6 통합 | 최종 모델과 재현 가이드 | 새 실행 환경에서 평가 재현 |

## 1 데이터 준비와 입력 파이프라인

### 2026-10-15 현재 상태와 우선 실행 계획

**2026-10-15 공식 근거 확인 및 M2/M3 구현 진행 범위**

- 두 sample의 공식 train 경로 소속과 공식 Scaramuzza calibration의 3D→2D annotation 정합성을 독립 확인했다. 원본 motion 독립성·RGB overlay·단위·시간 등은 남아 있다. 공식 RGB 제한 읽기는 2,301,530 bytes/60.031초 후 `http_body_failed`로 재사용 불가 처리했으며 대상 pose와 대응하는 RGB는 확보하지 못했다. [상세 근거](ARCHVIS_POSE_AUDIT.md#후속-공식-근거rgb-준비-조사-2026-10-15)를 따른다.
- 사용자는 M3까지 진행하고 M2 후 커밋할 것을 지시했다. 현재 M0/M1 실제 데이터 완료 기준은 미충족임을 설명했으며, 우선 데이터 없이 가능한 M2 body teacher와 M3 body→hand cascade 구현·합성 검증을 진행한다. M2 검토 후 로컬 커밋하고 M3로 이어간다. 이 구현 범위 승인은 실제 학습 예산, M4, 원격 push 승인이나 M2/M3 성능 완료 선언이 아니다.

**2026-10-15 후속 실제 JSON 감사 — M0 부분 검증**

- [사용자 sample ZIP 감사](ARCHVIS_POSE_AUDIT.md)에서 32개 payload의 SHA-256·크기를 manifest와 대조했고 모두 동일했다. 각 87관절의 필수 배열 13,920개가 수치 검사를 통과했다. 기존 3개 보고서도 byte 단위로 동일하다.
- Brutal_Assassination 0~17번은 감사 오류 0과 실제 smoke 통과. Opening 14개는 산재 subset으로 `missing_frames`를 유지하므로 전체 감사는 exit 1이며 연속 학습에서 제외한다. 길이 3의 실제 causal windows 32개에서 offline/streaming 일치·현재/과거 제한·gap/sequence reset·미래 변경 격리를 확인했다.
- 독립 검사에서 좌우 camera 3D 평균과 device 3D가 최대 약 5.684e-14 raw units 잔차로 일치했다. 단위·좌표 의미·재투영·시간·split·RGB 대응을 검증한 것은 아니다. 개발용 skeleton과 train_ready=false를 유지하며 학습·GPU 실행은 하지 않았다. 다음 후보는 대응 RGB·calibration·공식 split 근거와 좌표/시간 규격 검증이다.

**2026-10-15 실제 Colab pose 소량 추출 — 사용자 보고서 검토 완료**

- 사용자가 제공한 `metadata_report.json`, `extraction_manifest.json`, `pilot_report.json`의 내용 hash와 상호 참조, manifest·통계·파일 수를 대조했다. 실행 revision은 `fbed7b13633b15ee95c35e9cb40c23517f75179f`, 환경은 Linux x86_64 / Python 3.13.16 / CPU다. 기록된 9개 소스 파일 hash는 배포 ZIP과 정확히 일치하며 Git 코드와의 차이는 텍스트 줄바꿈뿐이다. 원본 보고서·검토 요약은 Git 제외 `outputs/colab_archvis_pose_20261015/`에 보관한다. 보고서 무결성 검사는 실행 출처의 암호학적 인증을 뜻하지 않는다.
- `ArchVisInterior_ArchVis_RT.tar.gz000` pose 원본 368,145,316 bytes에서 앞 524,288 bytes(512 KiB)를 64 KiB Range 8회로 읽었다. 최초 metadata 290 bytes/0.282초, 준비 단계 525,728 bytes/8.427초이며 합계 응답 본문은 526,018 bytes다. 준비 단계에는 원본 전후 metadata 16회/1,440 bytes가 포함된다. 시간 합계 8.709초는 인증·업로드·별도 verify·전체 wire 전송 시간을 포함한 end-to-end 수치가 아니다.
- 완전 pose 파일 **32개**, 합계 **2,735,012 bytes**를 저장했고 Colab 보고서의 hash 재사용 검사는 **32개 모두 passed**다. `files_limit` / `bounded_stop`은 설정한 32개 상한 도달에 따른 정상 중단이다. 다음 member는 written_bytes=0으로 완료 목록에 들어가지 않았다. 전체 파일을 받은 것이 아니므로 gzip CRC·tar 종료·전체 archive 무결성은 false/미검증이고 `train_ready=false`를 유지한다.
- 이름 기준 `025/Brutal_Assassination`은 frame 0~17의 18개가 연속되고, `017/Opening`은 5,32,36,50,54,61,65,88,90,94,113,117,122,126의 14개가 흩어져 있다. Archive 순서를 시간 순서로 사용하지 않는다. 번호 연속성은 timestamp·시퀀스 전체 완전성의 증명이 아니다. 앞선 Lab_Scene 18프레임/공식 CityPark 16 JSON 이력과 별개다.
- 이번에 확인한 것은 **사용자 Drive → Colab metadata → 제한된 pose 부분 추출 → 해당 런타임 verify** 경로다. 당시 추출 JSON 본문은 미제공이었다. 이후 위 sample ZIP 감사에서 각 파일 hash·관절 배열을 재검사했다. 좌우 RGB 대응·단위·좌표/calibration·timestamp·split/원본 motion·GPU 학습은 남아 있고 M0/M1 전체 완료가 아니다. 보고서의 `live_drive_colab_validation`은 코드의 고정 미검증 목록이므로 이번 성공을 자동 반영하지 않는다. 원본 보고서는 수정하지 않는다.
- **후속 진행:** 추출본과 보고서 보관 및 JSON 구조·관절·좌표 단서 감사는 위 결과로 확인했다. 다음 후보는 같은 sequence/frame의 좌우 RGB와 calibration·split 확보 및 좌표/시간 규격 검증이다. 현재 추출본을 먼저 활용하며 전체 다운로드·상한 확대·GPU 학습은 자동 시작하지 않는다. 운영 예산은 이번 한 pose 시험만으로 확정하지 않는다.

**2026-10-15 Colab metadata 전송 오류 수정**

- 사용자 Colab의 HTTP200·Transfer-Encoding chunked·Content-Length 없음·JSON UTF-8 응답에서 기존 `metadata_protocol` 실패를 확인했다. Builder와 pilot 전후 Drive metadata에 공통 bounded EOF reader를 적용했다. Declared length 조건과 byte/deadline을 유지하며 무제한 body/JSON 읽기를 사용하지 않는다. 길이 없는 응답이 정확히 상한에 닿으면 EOF 확인용 1byte를 더 읽지 않고 미확정으로 거부한다. Media206/Content-Range/Length 검증은 그대로다.
- 전체97개(기존93+필수4)와 notebook 사전38개(archive24+metadata14), 실제 라이브러리 loopback15조건을 통과했다. 정상 chunked→manifest→pilot 전후metadata·verify, 잘린 chunk/JSON·종료 chunk 누락·상한·slow·인증상태·encoding·media조건 유지와 report reserve를 확인했다. 비밀 없는 status/헤더 분류·수신 byte를 기록한다. 수정 직후의 로컬 검증과 후속 사용자 Colab pose 소량 추출 성공을 구분해 기록한다. 전체 archive·학습 데이터·운영 예산 검증은 남아 있다. 재실행은 [Colab 가이드](COLAB_SETUP.md)의 새 runtime/소스 ZIP 절차를 따른다.

**2026-10-14 당시 데이터 상태(후속 10월 15일 검증은 위에 별도 기록):** Drive 목록을 재확인했고, 별도로 공식 서버 CityPark 원본에서 제한된 바이트 범위의 추출 가능성을 시험했다. 이후 `rewind_archive`로 저장된 공식 prefix를 로컬 재생해 PNG 2개·JSON 16개의 hash를 대조했다. 사용자 Drive 복사본의 내부 무결성·추출·프레임 검증과 Drive → Colab → 실제 데이터 학습 연결은 미검증이다. 아래 과거 18프레임 sample과 이번 prefix의 16 pose 파일은 다른 검증 이력이다. 원본 압축을 보존하고 필요한 파일만 런타임에 추출하는 운영 결정을 유지한다. 제한된 준비 기능과 별도 합성 notebook을 구현했으며 조사·합성/부분 실행으로 M0/M1 완료를 선언하지 않는다.

**2026-10-14 Colab pilot 진입 준비 후속 묶음 — 실제 실행 대기**

- 명시한 동일 archive의 연속 part ID·기대 이름·total_parts에서 metadata-only manifest를 생성한다. API의 실제 size/version/modifiedTime·파일 유형·trashed·canDownload를 검사하고 빈/누락/불일치 값·중복·혼합 archive를 거부한다. 검색·media·자동 재시도 없이 metadata 상한과 실패 report를 기록하며 실패 시 유효 manifest를 남기지 않는다.
- 준비 notebook을 선택 dict/JSON → 별도 기본 비활성 metadata/auth → 별도 기본 비활성 bounded pilot/verify/report로 연결했다. 선택 변경·metadata 재실행 실패는 이전 결과를 폐기하고 pilot을 차단한다. 한 archive만의 준비 진단이며 stereo pair·M1 학습 완료가 아니다. Metadata 기본 128KiB/30초와 준비 기본8MiB/60초는 별도 개발 상한이다.
- 전체 **93개** 테스트와 fresh ZIP notebook의 기본 auth0·합성 triplet4, mock metadata/pilot·선택 변경·실패 재실행·JSON 업로드를 확인했다. 실제 requests/urllib3/google-auth는 AnonymousCredentials+loopback에서 metadata3요청672bytes(media0) → 준비9요청1756bytes → pose4/verify/persist 일치를 확인했다. 실제 Google 요청·인증0이며 Colab/Drive 준비 비용·완전한 실제 clip·운영 예산과 D1~D5는 미확정이다. [사용자 입력과 보고서](ARCHIVE_PREPARATION.md#drive-metadata에서-manifest와-pilot으로)를 따른다.

**2026-10-14 M0/M1 실제 입력 준비 기반 묶음 — 부분 구현**

- `rewind_archive`에 명시적 source/part manifest와 상한 config, 연속 000-prefix의 local/HTTP Range 읽기, bounded gzip/tar 선택 추출을 구현했다. 정확한 206·범위·길이·원본 validator를 검사하고 200 본문 전체를 읽지 않는다. GNU LongName/PAX metadata도 제한하고 unsafe 경로·link/special·중복을 거부한다. 전송 body·협력적 시간·해제/IO buffer·파일/disk·metadata/range 수와 중단 사유를 기록한다. 프로세스 RSS·DNS/auth/OS 전체 hard limit을 보장하지 않는다.
- 완전 regular payload+padding만 남기고 잘린 파일은 제거한다. Gzip EOF/CRC와 tar 종료, 개별 완전 파일을 구분한다. CRC 오류·원본 변경·unsafe 실패는 추출 전체를 unusable로 기록하고 reuse/match에서 거부한다. 실제 운영 예산은 미정이며 `archive_limits_smoke.json`은 개발용 기본값이다.
- 이름 기반 좌우 RGB/pose의 sequence·숫자 frame 매칭, 관측 gap·누락·완전 triplet 수와 개별 hash 재사용 검사를 제공한다. Split은 unknown, 원본 motion은 unverified이며 triplet만으로 train_ready를 올리지 않는다.
- 당시 전체 Python 회귀 **83개**(기존 59+archive 24)를 CPU 환경에서 통과했다. 실제 HTTP는 로컬 loopback이고 Drive session은 초기 모의 계약 검사다. 기존 공식 prefix 로컬 재생은 PNG 2·JSON16의 과거 18개 SHA 일치, 마지막 payload 잘림·CRC false·partial 잔여0·verify 통과를 확인했다. 서로 다른 2 sequence라 triplet0/incomplete frame18이며 전체 archive/사용자 Drive 검증이 아니다.
- [준비 notebook](../notebooks/archive_prepare.ipynb)의 별도 ZIP/임시 checkout 로컬 실행을 확인했다. 합성 RGB8+pose4→triplet4/train_ready=false, source 추출·커널 import·기본 disabled Drive 분기를 확인했고 모든 code cell 문법·빈 출력 상태를 검사했다. 실제 Colab 업로드·Drive 인증/API·준비 비용·운영 상한과 실제 좌우+pose clip은 미검증이다.
- 자세한 계약·측정 범위·지원 포맷은 [압축 준비 안내](ARCHIVE_PREPARATION.md)를 따른다. M0/M1 전체 완료와 D1~D5 확정, 실제 RGB loader·좌표 변환·EfficientNet/M2 구현은 이 묶음에 포함하지 않는다.

**2026-10-14 M1 학습 기반 구현 묶음 결과 — 전체 완료 아님**

- `rewind_m1`에 명시적 관절·단위·영상 크기·split·출처 계약, 특징 tensor, 작은 single-frame MLP/고정 causal window GRU, masked normalized MSE·anchor MPJPE와 학습·평가·checkpoint CLI를 구현했다. 미검증 실제 데이터는 검증 상태·근거가 없으면 거부한다. Camera encoding·projection·물리 시간 loss는 포함하지 않는다.
- 작은 Conv heatmap 개발 모델로 train-only 합성 RGB 학습 → 고정 weights 예측 cache → sequence/frame 연결 → regression·validation과 동일 output 재개를 확인했다. GT 3D/mask와 unknown visibility를 보존한다. 이는 EfficientNet이나 실제 RGB baseline의 완료가 아니다.
- CPU PyTorch 2.7.1+cpu에서 Python 테스트 59개(기존 40 + M1 19)를 통과했다. 두 regression의 80 update 과적합, model/optimizer/scheduler/scaler/RNG 재개 일치, 주기 checkpoint/report 사이 중단의 누락 log 명시 복구, causality·streaming/reset을 검사했다. 실제 데이터/Colab/CUDA·AMP와 latency는 미검증이다.
- [Colab notebook](../notebooks/m1_baseline.ipynb)은 로컬 커밋의 소스 ZIP을 업로드하고 임시 디스크에서 합성 진단·재개 후 선택적으로 결과를 Drive에 복사하도록 준비했다. JSON/code cell 문법과 실행 셀의 로컬 CPU 실행, 로컬 결과 복사·파일별 hash를 확인했다. 실제 Colab 업로드·인증·Drive/GPU 실행을 대신하지 않는다.
- 실행법·계약·증거·미검증 범위는 [M1 안내](M1_BASELINE.md), 환경 준비는 [Colab 안내](COLAB_SETUP.md)에 기록한다. M1 진입/완료 기준과 D1~D5는 유지한다. 다음 실제 데이터 진입에는 관절·단위·좌표/calibration·split·좌우 RGB/pose·timestamp 검증과 small sample이 필요하다.

**2026-10-13 데이터 없는 후속 묶음 결과**

- 사용자 승인 범위로 [데이터 규격 근거](DATA_CONTRACT.md)와 M1 준비용 causal window 조립 모듈을 추가했다. 공식 안내의 16관절 평가/32+40 annotation, 과거 raw 87 이력, dev 62를 구분하고 저자 코드 revision·논문 페이지를 기록했다. 실제 archive 검증과 공식 자료 설명을 구분하며 skeleton·단위·회전·시간을 확정하지 않았다.
- `CausalWindowStream`과 `offline_windows`는 현재·과거 snapshot만 조립하고 부족한 과거는 None/time_valid로 구분한다. Gap·새 sequence·명시적 container 경계에서 history를 분리하며 중복/역순과 구조 오류는 거부한다. 선택 밖 미래-only 관절 union의 issue를 과거 window에 복사하지 않는다. Field mask/unknown visibility와 snapshot 격리를 유지한다.
- Python 합성 테스트 40개(기존 17 + window 23)와 6개 window fixture의 offline/streaming 일치 검증을 통과했다. 실제 데이터·Drive·이전 실제 sample 산출물 접근, smoke_sample 재실행, 모델·학습·projection 구현은 하지 않았다.
- M0/M1 전체 완료 및 D1~D5 확정은 보류한다. 다음 진입 조건은 재다운로드 subset의 manifest/hash·관절·split 재감사와 좌표·단위·FPS/timestamp·calibration 대조다. 이후 GT 2D 진단 학습은 별도 지시 범위로 진행한다.

전체 다운로드를 기다리지 않고, 확보된 pose JSON으로 데이터 감사와 학습 경로의 진단을 먼저 진행하는 계획이다. 아래 순서는 실행 제안이며 D1~D5의 범위·예산 확정을 대신하지 않는다. 2026-10-13 첫 구현 지시로 1~2번의 사전 도구를 구현했고 2026-10-14 합성 M1 학습 기반을 추가했다. 실제 데이터의 모델 학습·평가는 아직 시작하지 않았다.

**2026-10-13 첫 구현 묶음 결과**

- `rewind_data` 모듈과 CLI에 숫자 frame 정렬·부모 폴더별 sequence 분리·필수 배열/누락/중복 검사·관절 선택·manifest·감사 보고서·로컬 HTML viewer를 구현했다. 원본 좌표 변환과 시간 보간은 하지 않는다.
- 지정 `Jump__3_/json` 폴더의 0~17번 18개 파일(총 1,540,663 bytes)을 소량 sample로 사용했다. 각 파일의 원본 87관절 및 SHA-256을 보존했고 필수 배열 shape·유한값 검사 오류는 0건이다. 음수 2D 관측 109개를 수치 유효 annotation으로 유지했다. Drive에서 관측한 18개 목록과의 일치는 전체 원본 sequence의 완전성을 의미하지 않는다.
- 개발용 선택은 body 22개·왼손 21개·오른손 21개이며 양쪽 손목이 body와 겹쳐 고유 관절은 62개다. 관절 이름은 실제 sample에서 확인했으나 선택 및 61개 연결은 시각화용 미검증 가정으로 명시했다. 공식 skeleton 부모 관계나 REWIND 평가 관절 규격으로 확정하지 않았다.
- 합성 fixture 테스트 17개, 실제 sample smoke, 브라우저 없는 JavaScript DOM/canvas mock 검사 7개를 통과했다. Canvas mock의 그리기 명령을 정적 그림으로 확인했으나 실제 브라우저 화면·조작 검사는 `file://` URL 보안 정책 차단으로 미실행이다.
- 단위·축·회전·FPS/timestamp·calibration·split·RGB 정합성은 미검증이다. M0/M1 전체 완료와 D1~D5 확정을 선언하지 않는다. 실행법과 데이터 계약은 [M0 JSON 도구 안내](M0_JSON_TOOLS.md)에 기록한다.

**첫 구현 전 샘플 확인 이력과 당시 미확인 범위**

- 사용자는 데이터셋 다운로드를 진행 중이다. 앞선 Drive 읽기에서 확인한 폴더는 `UnrealEgoData2_train_val_pose`이며, 기존 UnrealEgo 우선 시작안과 데이터 버전이 다르다.
- `Lab_Scene/Day/rp_sophia_rigged_003_ue4/SKM_MenReadingGlasses_Shape_01/001/Jump__3_/json` 목록에 `frame_0.json`~`frame_17.json` 18개가 있었다. 이 중 0·1·17번 파일만 내용을 검사했다. 전체 시퀀스의 다운로드 완료 여부는 확인하지 않았다.
- 검사한 3개 파일에는 각각 관절 항목 87개와 이름 기준 손가락 항목 40개가 있었다. 각 관절에 좌우 `camera_*_pts2d`, `camera_*_pts3d`, `device_pts3d`가 있으며 해당 배열의 차원 오류·null·NaN·무한대는 발견하지 못했다.
- 카메라와 device의 위치·회전 필드는 존재하지만, 명시적인 timestamp·visibility·confidence 필드는 없었다. 단위·축·회전 표현·FPS·calibration은 미검증이다. `open3d_rot`도 camera와 device에서 같은 단위라고 가정하지 않는다.
- 이는 한 시퀀스의 제한적인 구조 확인 결과다. 관절 매핑, 재투영, split, 다른 시퀀스와의 일관성, RGB 확보 상태는 아직 검증하지 않았다. M0 완료나 성능 검증으로 간주하지 않는다.

샘플 근거: [Drive pose 폴더](https://drive.google.com/drive/folders/1hpABBbMVxiH5wFZx3_pTfhu3Xs2d2iYB), [검사한 frame_0.json](https://drive.google.com/file/d/1CAcEjI2YML4kQPuKRRLpWrvOwJzKoHID/view).

| 순서·연결 단계 | 작업과 산출물 | 완료·진입 조건 |
| --- | --- | --- |
| 1 · M0 사전 감사 | 소량 JSON 목록과 출처·hash를 담은 manifest, 필드·관절·누락 검사 보고서 | 데이터 버전과 sequence 식별, 파싱 가능 여부와 미확인 필드를 기록한다. 현재 샘플로 시작하고 가능한 경우 서로 다른 5~10개 시퀀스로 확인 범위를 넓힌다. |
| 2 · M0 로더와 시각화 | 공통 sample 로더, 이름 기반 skeleton 매핑, 3D 정답 재생 | 원본 87개 항목을 보존하면서 모델용 body/hand 관절을 명시적으로 선택한다. 부모 관계·단위·축을 근거와 대조하고, frame 순서·누락·짧은 clip 처리를 검증한다. |
| 3 · M1 사전 진단 | GT 2D → device-relative 3D의 작은 단일 프레임 regression, 평가·checkpoint 코드 | 1~4개 train clip에서 loss·gradient·과적합·중단/재개를 확인한다. 단위 확인 후 mm 지표를 보고한다. 이 결과는 GT 입력 진단이며 M1 완료가 아니다. |
| 4 · M0 완료와 M1 baseline | 대응 RGB·calibration·split 확인, 재투영, 2D front-end와 예측 특징 캐시, causal temporal regression | 영상·관절·프레임 정합성 및 분할 중복 검사를 통과한다. 예측 2D로 train/validation 평가를 실행하고 front-end 버전을 고정한다. 전체 RGB 다운로드 완료는 요구하지 않는다. |
| 5 · M2 | 작은 causal body diffusion teacher에서 시작해 baseline 비교 | 위치 loss의 과적합, 미래 입력 변경 시 과거 출력 불변성, sequence reset을 검증한다. 같은 예측 입력·split에서 regression과 비교한 뒤 규모를 늘린다. |
| 6 · M3 | Body → hand cascade와 손목 결합 | 조건 없음·GT 상체·예측 상체를 분리 평가하고, 실제 body 예측으로 whole-body 평가를 완료한다. |
| 7 · M4 | One-step student, 비증류 student 비교, streaming | 동일 noise의 offline causal/streaming 일치와 재시작을 검사한다. 정확도 및 지정 GPU의 모델 단독·전체 pipeline 지연을 측정한다. |
| 8 · M5~M6 | E1~E3 중 예산에 맞는 단일 변경 실험, 최종 통합 | Validation으로 후보를 선택한 후 고정 test 평가와 새 환경에서의 재현을 수행한다. E4~E8은 선행 결과와 예산을 보고 선택한다. |

**RGB 확보 전의 검증 경계**

- JSON만으로 파싱·관절 매핑·3D 재생·GT 2D 입력 학습 경로를 준비할 수 있다. 완전히 합성한 입력은 shape·mask·gradient 등의 코드 검사에만 사용하고 실제 데이터 결과와 분리한다.
- GT 입력에는 `input_source=gt_2d` 같은 출처를 기록한다. Confidence가 없는 경우 진단용 입력 처리 규칙을 명시하며, 임의 상수를 예측 confidence나 실제 visibility로 해석하지 않는다. 정답의 수치 유효성과 관측 visibility를 구분한다.
- 18프레임 목록을 근거로 FPS나 완전한 시퀀스를 가정하지 않는다. 실제 관측 길이에 맞는 짧은 window를 쓰고 padding이 필요하면 mask를 적용한다. 반복 복제로 50프레임짜리 실제 motion을 만들지 않으며 sequence 경계를 넘어 window를 구성하지 않는다.
- FPS·timestamp 확인 전에는 물리적 시간 단위의 velocity/acceleration loss와 지표를 보류한다. Calibration과 단위가 확인되면 저장된 2D 좌표와 수치 재투영을 비교할 수 있지만, 영상 위 정합성 검증에는 대응 RGB가 필요하다.
- RGB가 없으면 2D front-end 학습·실제 예측 입력 평가·전체 pipeline 성능 검증은 완료할 수 없다. GT 2D에 주입한 오류도 실제 front-end 오류를 대체하는 성능 근거로 삼지 않는다.
- 하나의 짧은 시퀀스를 프레임별로 나누어 일반화 성능을 보고하지 않는다. 공식 split 소속을 확인하고 train 진단용 sample을 선택한다. Test sample로 overfit이나 threshold 튜닝을 하지 않는다.

**첫 구현 묶음과 운영 제안**

첫 구현 묶음은 위 1~2번인 JSON 감사·공통 로더·정답 시각화 도구를 제공했다. 2026-10-14에는 합성 학습 기반과 이후 제한된 archive 준비 기능을 구현했다. 실제 small sample을 준비·검증하고 관절·좌표·단위와 split 진입 조건을 충족한 뒤 실제 3번 진단 학습으로 진행한다. JSON/window는 `rewind_data`, archive 준비는 `rewind_archive`, M1 tensor·모델·지표·학습은 `rewind_m1`에 분리하고 별도 notebook을 실행 진입점으로 둔다. 표준 라이브러리 M0/준비 도구와 PyTorch M1 환경을 구분한다.

데이터 보관·추출 방식은 아래의 2026-10-13 운영 결정에 따른다. Drive 압축 원본과 영구 저장할 manifest·설정·checkpoint·결과를 런타임 추출본과 분리한다. 문서 반영으로 현재 다운로드 작업을 중단하거나 변경하지 않는다.

장시간 학습 전에 D2의 실제 GPU·세션 길이·영구 저장 공간·총 예산을 정하고, pilot의 step 시간·VRAM·입출력 시간을 측정한다. 일정은 고정 날짜 대신 위 완료 조건으로 관리한다. D1·D3·D4·D5는 미결정으로 유지하며, JSON 개발에 UnrealEgo2를 활용하는 것과 최종 baseline 데이터셋을 확정하는 것을 구분한다.

### M0 데이터 확보와 공통 규격

기존 데이터 시작안은 UnrealEgo의 손 annotation이 포함된 시퀀스로 시작하고 UnrealEgo2로 확장하는 것이다. 이전 UnrealEgo2 JSON은 사전 도구 검증에 사용했으며, 재확보 데이터의 최종 baseline 데이터셋·버전·split은 검증 후 결정한다. 공식 안내의 전체 UnrealEgo2 저장 공간은 약 3TB이므로 RGB와 pose를 필요한 범위만 추출하고 depth는 초기 필수 입력에서 제외한다. [3]

### 부분 다운로드 조사와 실행 계획

**2026-10-14 확인한 사실 — 프로젝트 기능 구현과 구분**

- 공식 배포 스크립트는 환경별 `.tar.gz000`, `.tar.gz001` 등을 연결해 해제한다. 현재 분할은 독립된 시퀀스 압축 묶음이 아니다. Google Drive API는 blob 파일의 HTTP `Range` 다운로드를 지원하지만, 바이트 범위 접근만으로 gzip 중간 해제가 가능해지지는 않는다. [4][5]
- 공식 서버의 `UnrealEgoData2_train_val_rgb/CityPark_Showcase/CityPark_Showcase.tar.gz000`에서 `0–262143`, `262144–2097151` 범위를 받아 총 2MiB를 순서대로 연결했다. HTTP 206과 Content-Range를 확인했고, PNG 2개의 전체 payload와 PNG chunk CRC를 검사했다.
- 공식 서버의 `UnrealEgoData2_train_val_pose/CityPark_Showcase/CityPark_Showcase.tar.gz000`에서 `0–262143`의 256KiB만 받아 완전한 JSON 16개를 파싱했다. RGB `001`의 시작 64바이트에는 gzip 시작 헤더가 없었고 독립 gzip 해제 시 header 오류가 발생했다.
- 두 prefix 모두 gzip 끝에 도달하지 않았으며 마지막의 불완전한 파일은 제외했다. 전체 gzip CRC·archive 무결성은 미검증이다. RGB는 `Defender__1_`의 왼쪽 영상, pose는 `Stroke_Shrugging__1_`로 서로 다른 시퀀스였고 파일 순서도 frame 번호순이 아니었다. 대응 학습 sample이나 완전한 clip을 확보한 결과가 아니다.
- 이 시험은 공식 서버의 원본에 대한 조사다. Drive 복사본과의 바이트 동일성, Drive 인증·Range 실제 응답, Colab 실행·처리량, 좌우 RGB/pose 대응 및 학습은 검증하지 않았다. 초기 RGB 8MiB 요청은 본문 읽기 제한 시간 45초에 걸려 중단했고 더 작은 요청으로 확인했다. 안정적인 처리 속도의 근거로 사용하지 않는다.
- 조사용 스크립트·부분 바이트·로컬 검사 기록은 Git 제외 경로 `outputs/partial_archive_probe/`에 두었다. 다른 checkout에 존재한다고 가정하지 않으며, 현재 CLI의 정식 기능이나 재현 가능한 Colab 준비 모듈로 간주하지 않는다.

**준비 흐름과 현재 구현 경계**

1. 필요한 환경의 archive ID와 분할 파일 목록·순서·크기·변경 여부 및 검사 가능한 hash를 확인한다. 전체 archive 준비 여부와 `000`부터 연속된 prefix의 가용성을 구분한다. 중간 누락이나 잘린 파일을 건너뛰어 계속 해제하지 않는다.
2. Colab에서 인증된 Drive API로 제한된 바이트 범위를 읽고, HTTP 206·Content-Range·받은 길이를 확인한다. 범위가 무시되어 전체 응답이 오거나 원본이 바뀌면 중단한다. 단순 `drive.mount()` 읽기가 동일한 전송량 상한을 보장한다고 가정하지 않는다.
3. 번호순 바이트 스트림을 Colab CPU·메모리에서 순차 해제하며 필요한 파일만 `/content` 임시 디스크에 저장한다. 전체 압축본·결합본·전체 추출본을 먼저 만들지 않는다. 불완전한 prefix에서는 온전히 읽힌 파일만 후보로 남기고, 잘린 마지막 파일과 전체 무결성 미확인을 기록한다.
4. RGB·pose의 내부 순서에 의존하지 않고 sequence/frame 식별자로 좌우 영상·정답을 대조한다. Frame 정렬, 누락·연속성, split 및 원본 motion 중복을 확인한다. 매칭되지 않은 파일을 조용히 버려 편향된 평가 sample을 만들지 않는다. 한쪽 영상과 일부 JSON의 추출만으로 학습 준비 완료를 선언하지 않는다.
5. 검증된 추출 JSON 디렉터리를 기존 loader의 `--input`에 연결한다. 실제 데이터용 특징 계약과 RGB 로더 연결을 준비한 뒤 `rewind_m1` 학습 기반으로 배치를 구성해 GPU에 전달한다. 이 실제 데이터 연결은 아직 구현·검증하지 않았다. 같은 런타임에서는 추출본을 재사용하며 매 배치·epoch마다 원본을 처음부터 해제하지 않는다. 결과·checkpoint·manifest는 Drive 등 영구 저장소에 분리 보관하고 런타임 삭제·추출본 소실 후 재준비한다. [6]

네트워크 읽기 단위는 학습 배치와 다르다. 인덱스 없는 gzip에서 뒤쪽 sample에 도달하려면 앞부분을 읽고 해제해야 하므로 작은 추출본도 큰 전송량·준비 시간을 요구할 수 있다. 준비 코어의 byte/buffer/파일/disk 및 협력적 deadline을 구현했고, 실제 Colab 소량 시험에서 처리량·운영 예산을 정해야 한다. 프로세스 RAM과 DNS/auth/OS 전체 hard 상한은 보장하지 않는다. 불완전 파일 제거와 같은 런타임 hash 재사용은 로컬에서 검사했으며 실제 좌우 RGB/pose clip·새 Colab 런타임의 재준비·GPU 학습 연결은 남아 있다.

추출 manifest에는 archive ID, 분할 파일 이름·순서·크기·변경 식별 정보, 실제 읽은 바이트 범위, 확인한 hash의 알고리즘·값·검사 범위, 추출·누락 목록과 sequence/frame/split, 추출·전처리 버전을 남긴다. `rewind_archive`는 이 별도 source/추출 기록과 이름 기반 매칭을 제공하며 실제 split·좌표·시간 등은 미확인으로 유지한다. 부분 바이트의 hash와 전체 파일 hash를 구분하고 기존 JSON 감사와 파일 경로·SHA-256을 대조해 연결한다. 기존 디렉터리 loader와 준비 기능의 경계는 [M0 도구 안내](M0_JSON_TOOLS.md#압축-원본-준비-계획과-구현-상태)를 함께 참조한다.

**후속 검토 후보 — 미구현·미채택:** 직접 메모리에서 학습에 공급하는 경로, gzip 해제 재시작 인덱스와 tar 파일 위치 인덱스, RGB·pose를 맞춘 작은 독립 압축 묶음 생성. 인덱스·재포장에는 최초 순차 읽기 비용이 필요하고, 재포장에는 영구 저장 공간이 추가로 필요하다. 원본은 보존하며 반복 학습 비용을 측정한 뒤 별도로 선택한다. [7]

**Drive 목록 관측(2026-10-14):** train_val_pose 15개 약 18.02GB, test_pose 15개 약 1.97GB, ArchVis RGB `000`~`004` 5개 약 52.44GB는 전날과 같았다. CityPark RGB는 전날 `000`~`003` 4개 약 42.95GB에서 `000`~`012` 13개 약 139.59GB로 늘었다. 공식 17개 중 마지막 4개는 관측되지 않았다. 합계 48개 약 212.02GB이며 GB는 십진 단위다. 목록·크기 관측은 내부 무결성이나 최종 baseline 선택을 뜻하지 않는다. 모든 분할 이름이 보이는 ArchVis RGB와 pose 약 368MB는 첫 준비 후보이고, CityPark prefix는 제한된 추출 후보로 구분한다. 근거: [Drive 압축 원본 폴더](https://drive.google.com/drive/folders/1GIWTYCaHxz7YeBCabpZrUyioHwXVUo6O), [공식 RGB 분할 목록](https://unrealego.mpi-inf.mpg.de/data/download_unrealego2_train_val_rgb.sh).

### M0 검증과 규모 확대

| 데이터 단계 | 제안 규모 | 목적 |
| --- | --- | --- |
| Smoke test | 5~10개 시퀀스 | 로더·좌표 변환·재투영 확인 |
| Overfit test | 1~4개 클립 | 학습 경로와 loss 검증 |
| Pilot baseline | 수십~수백 개 시퀀스 | 일반화 추세와 학습 비용 측정 |
| 확장 실험 | 검증된 데이터부터 확대 | 규모·동작 다양성 효과 분석 |

공통 sample에는 좌우 영상 경로, timestamp, camera calibration/pose, 2D·3D joints, annotation mask, 관측 visibility, sequence·motion·identity ID를 저장한다. 정답 누락과 화면 밖 관절을 구분하고 관절 이름·순서·부모 관계를 별도 skeleton 규격으로 고정한다.

초기에는 device-relative 출력을 검토한다. 카메라 변환 방향, 회전 표현, 거리 단위, 어안 projection, 영상 resize 후 좌표를 검증한다. 속도 loss에서 camera motion이 섞이는 영향을 명시한다. World-space foot loss는 좌표와 접촉 정보가 검증되기 전에는 사용하지 않는다.

Train/validation/test는 sequence 단위로 고정하고 원본 motion의 중복 렌더링도 점검한다. 공식 split과 별도 generalization split을 혼동하지 않는다. UnrealEgo2 test split 공개 공지와 별개로 손 관절 포함 여부는 실제 파일로 확인한다. 최종 test를 튜닝에 사용하지 않는다. [3]

**완료 기준:** 좌우 재투영 시각화, 관절 연결, 누락 프레임·단위·분할 중복 검사와 manifest를 확보한다.

### M1 2D front end와 비교 기준

EfficientNet 계열 encoder와 heatmap decoder로 2D joints와 confidence를 예측한다. 해상도는 손 관절 정확도와 GPU 비용을 보고 선택한다. GT 2D 입력은 진단 전용으로 두고 실제 기준 성능에는 예측 2D를 사용한다.

Front-end를 학습·고정한 뒤 2D 특징을 캐시한다. 캐시에 모델과 전처리 버전을 기록하고, 단일 프레임 3D regression 및 temporal regression을 비교 기준으로 구현한다. GT 2D 사용 결과와 예측 2D 결과는 별도 표로 보고한다.

**완료 기준:** Body/hand별 2D 지표와 예측 2D 기반 3D 학습·평가를 끝까지 실행한다.

**2026-10-14 구현 상태:** 작은 regression·loss·checkpoint와 개발용 Conv front-end/cache의 합성 연결을 구현·CPU 검증했다. 설정 `m1_*_synthetic.json`과 모든 결과의 synthetic 출처를 보존한다. EfficientNet baseline, 실제 RGB 로더·데이터셋 학습/캐시, body/hand 지표, Colab GPU 비용 실측은 남아 있어 위 완료 기준을 충족하지 않는다. [M1 실행 안내](M1_BASELINE.md)를 기준으로 후속 범위를 검토한다.

## 2 기준 모델 구현과 학습

### M2 Body diffusion teacher

Noisy 3D joints에 stereo 2D joints·confidence, camera encoding, diffusion timestep을 결합한다. Graph Transformer로 공간 구조를, RoPE 기반 causal window attention으로 시간 정보를 처리하고 clean 3D joints를 예측한다. Graph와 temporal 모듈은 denoiser 내부에 둔다.

| 항목 | Teacher | Student |
| --- | --- | --- |
| 특징 차원 | 512 | 256 |
| Graph Transformer | 3블록 · 4 heads | 1블록 · 2 heads |
| Temporal attention | 3층 · 4 heads | 1층 · 2 heads |
| Window 설정 | 20 | 8 |
| 추론 | DDIM 10-step | 1-step |

위 값은 논문 부록의 시작 설정이다. 공통 설정은 50프레임 학습, diffusion timestep 1,000, x0 prediction, Adam과 학습률 5×10⁻⁵이다. 논문의 200만 step은 참고 규모이며, 실제 학습 예산은 pilot에서 측정한 비용과 validation 추세로 정한다. [1]

위치 복원 loss로 overfit을 확인한 뒤 velocity와 intermediate frame loss를 추가한다. Foot loss는 조건이 충족될 때만 사용한다. Loss 단위·reduction·mask를 고정하고 논문 가중치를 맹목적으로 이식하지 않는다. 작은 개발용 모델과 논문 크기 모델은 설정과 결과를 구분한다.

**완료 기준:** 소량 과적합, 미래 프레임 변경에 대한 과거 출력 불변성, 긴 시퀀스 추론을 검증한다. Diffusion과 temporal modeling의 효과를 regression 비교로 분리한다.

**2026-10-15 구현 상태 — 합성 검증 묶음:** `rewind_m2`에 명시적 camera R6D/translation·skeleton 입력, GCN/spatial attention·RoPE causal temporal denoiser와 intermediate head, x0 forward diffusion·DDIM eta=0, masked position/frame loss, anchor-window streaming·학습/평가/재개 CLI를 구현했다. Cosine beta endpoint 보간과 per-joint softmax temporal은 명시적 프로젝트 선택이다. Window 총 slot과 층별 과거 수를 분리하며 full-sequence denoise와 잘린 window의 동등성을 주장하지 않는다.

- 기존 97 + M2 10 = **107 tests**를 통과했다. CPU tiny 합성 100 update 후 동일 output에서 110까지 재개·평가, stochastic training의 5+7 vs 12 update 전체 상태 일치를 확인했다. 모델·sampler 미래 불변성, 58frame streaming/gap/reset 및 keyed noise를 검사했다.
- 오케스트레이션은 수학 oracle·3관절 future perturbation·RoPE offset·padding을 독립 검사하고, 저장된 110 update checkpoint의 validation batch/streaming MPJPE 차이가 약 4.77e-7 mm임을 확인했다. 이는 인위적 2관절 fixture의 코드 경로 검증이다.
- 실제 관절/입력 baseline·M0/M1 완료 기준, velocity/foot loss·논문 규모 학습·실제 과적합/pilot·regression 대비 효과·Colab CUDA/AMP/latency는 남아 있다. [M2 상세 안내](M2_BODY_TEACHER.md)를 따르며 위 M2 전체 완료 기준을 충족했다고 선언하지 않는다. 검토 후 로컬 커밋하고 M3 구현으로 이어간다.

### M3 Whole body teacher

Body teacher를 고정하고 Hand teacher를 학습한다. Hand 모델에는 상체 3D conditioning을 추가한다. 원본처럼 학습 시 GT 상체, 평가 시 예측 상체를 사용한다. 손의 출력 좌표와 손목 결합 규칙을 명시한다.

상체 조건 없음, GT 상체 조건, 예측 상체 조건을 비교해 cascade의 효과와 오류 전파를 측정한다. 손목 불연속·좌우 뒤바뀜을 검사하고 body 및 hand의 지표를 분리한다.

**완료 기준:** 실제 Body 예측으로 whole-body 평가를 완료하고 teacher checkpoint와 기준 설정을 고정한다.

### M4 One step student

Teacher를 고정하고 작은 student를 학습한다. Supervised loss에 논문 식 5의 증류 loss를 추가한다. k_small, noise schedule의 정확한 구현, teacher target의 stop-gradient 여부 등 불명확한 부분은 가정과 근거를 기록한다. 동일 크기의 증류 없는 student를 반드시 비교한다.

**완료 기준:** Teacher·비증류 student·증류 student의 정확도와 end-to-end latency를 비교한다.

## 3 평가와 실시간 검증

### 공통 평가 프로토콜

모든 실험은 고정 split과 같은 평가 코드를 사용한다. 관절 집합, 좌표계, PA alignment 적용 단위, 손목 상대 지표 여부, 유효 관절 mask, 프레임/시퀀스 평균 방식을 명시한다. 논문과 split·annotation·하드웨어가 다르면 동일 조건의 재현 수치로 주장하지 않는다.

| 평가 축 | 지표 | 해석과 조건 |
| --- | --- | --- |
| 정확도 | Body/Hand MPJPE<br>PA-MPJPE | mm 단위 보고 · 정렬 규칙 고정 |
| 구조 | Bone length error<br>손목 연결 오차 | 동일한 skeleton과 뼈 목록 |
| 시간 변화 | Velocity/acceleration error | Timestamp·좌표계 반영 |
| 떨림과 지연 | 정지 구간 jitter<br>빠른 동작 추종 지연 | 과도한 smoothing을 개선으로 오인하지 않음 |
| 강건성 | Failure subset별 오차 | 가림·화면 밖·spike·빠른 동작 |
| 복구 | Recovery time<br>복구 실패율 | Validation에서 문턱과 지속 조건 고정 |
| 실시간성 | p50/p95 ms/frame<br>처리율 · peak VRAM | Batch 1 · warm-up · GPU 동기화 |
| 발 안정성 | Foot skating | 유효한 world 좌표와 접촉 정보가 있을 때만 |

### Failure benchmark

자연 발생 failure와 인위적으로 주입한 corruption을 별도로 평가한다. Corruption은 한쪽 관측 누락, 짧고 긴 연속 가림, 좌표 spike, 잘못된 고신뢰 관측을 포함한다. 강도와 지속 시간은 validation에서 정하고, test에는 고정한 corruption seed를 적용한다.

GT visibility가 없으면 confidence 기반 subset을 실제 occlusion 정답으로 간주하지 않는다. Recovery time은 관측 복귀 후 오차가 고정 문턱 이하로 연속 유지되는 첫 시점까지로 정의하고, 미복구 사례도 집계한다.

### Streaming과 속도 측정

과거 window와 현재 프레임만 사용해 한 프레임씩 출력한다. 동일 noise에서 offline causal 실행과 streaming 실행의 일치, 초기 window, sequence reset, 장시간 실행의 메모리 안정성을 검사한다. Multi-step별 상태가 다른 attention cache를 무조건 재사용하지 않는다.

3D 모델만의 시간과 영상 전처리·front-end·body–hand 순차 추론을 포함한 시간을 분리한다. 지정 GPU에서 30 FPS를 잠정 목표로 삼되 p95 지연도 함께 판단한다. 렌더링 포함 여부와 hardware·precision·해상도를 기록한다.

**완료 기준:** 최종 후보는 제안상 학습 seed 3개로 반복하고 sampling seed 변동을 별도로 보고한다. 시퀀스 단위 분포와 clean 성능도 함께 확인한다.

## 4 성능 개선 실험 순서

기준 모델이 안정된 뒤 한 번에 한 요소를 변경한다. 먼저 비용이 낮고 원인 해석이 쉬운 실험을 수행하고, 효과가 확인된 변경만 조합한다. 데이터·모델·loss를 동시에 바꾼 결과는 단일 기법의 효과로 해석하지 않는다.

| 순서 | 실험과 변경 변수 | 판단할 결과 |
| --- | --- | --- |
| E1 | 학습 데이터 규모와 동작 다양성 | 동일 update 예산에서 일반화 개선 |
| E2 | 관측 오류 augmentation | 오류 subset 개선과 clean 성능 유지 |
| E3 | GT/예측 상체 conditioning 혼합 | 실제 cascade의 Hand 오차 감소 |
| E4 | x0 / ε / diffusion v prediction | 같은 모델·데이터·학습 예산의 정확도 |
| E5 | 지원하는 모델의 1/2/4/10-step 추론 | 정확도와 latency의 Pareto 개선 |
| E6 | Reliability-aware conditioning | False tracking 복구와 동작 detail 보존 |
| E7 | 대체 증류 또는 학습 방식 | 기존 증류 대비 비용과 품질 |
| E8 | Reliability 기반 adaptive inference | 평균·p95 지연과 최악 조건 안정성 |

### 먼저 실행할 실험

E1–E3을 우선한다. 데이터 규모 실험은 동일 학습 update 예산으로 비교하고 필요하면 수렴까지의 추가 비용을 별도로 보고한다. Augmentation은 프레임 독립 노이즈뿐 아니라 시간적으로 연속된 실패를 포함한다. 영상 변형과 keypoint 오류 주입은 구분한다.

상체 조건 혼합에는 일반화된 Body 오류가 필요하다. 학습 데이터에서 지나치게 정확한 Body 예측만 사용하는 것을 피하고, 필요하면 train 내부 분할의 out-of-fold 예측을 생성한다. 합성 noise 대체는 별도 ablation으로 남긴다.

### Diffusion 확장 시 주의할 경계

One-step 전용 student를 반복 호출하는 것을 유효한 few-step sampler로 간주하지 않는다. 해당 timestep·sampling 경로에 맞는 학습이 필요하다. Diffusion v prediction은 물리적 관절 velocity loss와 다른 항목이다.

Consistency, distribution matching, flow-based 방법은 서로 다른 학습 가정과 비용을 가진 후속 후보로 둔다. 동시에 구현하지 않고 한 방법의 목적함수·teacher 요구·예산을 검토한 뒤 별도 실험으로 진행한다. Adaptive inference는 reliability와 오류의 관계를 검증한 이후 착수한다.

### 채택과 통합

Validation에서 clean 정확도, failure 성능, latency를 함께 비교한다. 한 축의 개선을 위해 다른 축이 얼마나 나빠졌는지 기록하고 허용 기준은 본 실험 전에 결정한다. 유망한 변경 2~3개만 결합해 상호작용을 검사한다. 개선되지 않은 실험도 설정과 결과를 보존한다.

**완료 기준:** 최종적으로 정확도 우선 모델과 실시간 우선 모델을 구분해 선정하고, test 결과는 선택이 끝난 후보에 대해 보고한다.

## 5 실행 운영과 검토할 결정 사항

### 개발과 학습 운영

Colab notebook은 환경 준비·학습·평가의 진입점으로 두고, 핵심 코드는 data, models, diffusion, training, evaluation 모듈로 분리한다. 각 실험에는 config, 코드 revision, 데이터 manifest hash, front-end 버전, seed, checkpoint, 지표와 실패 예시를 함께 저장한다.

데이터는 필요한 시퀀스의 런타임 추출본에서 읽고, 실험 결과·checkpoint·manifest는 압축 원본과 분리해 영구 저장소에 보관한다. Model·optimizer·scheduler·AMP scaler·RNG·학습 위치를 저장하고 실제 중단/재개 테스트를 한다. 저장 주기는 pilot의 step 시간과 허용 손실량으로 정한다.

### 확정된 데이터 운영 결정

**2026-10-13 사용자 결정 — D2 중 보관·추출 운영 방식만 확정:** Drive 압축 원본을 보존하고 사용 시 Colab/runtime 임시 디스크에 필요한 시퀀스를 추출한다. 같은 런타임에서는 검증된 추출본을 재사용하고, 세션 종료 또는 추출본 소실 후 다시 추출한다. 실험 결과·checkpoint·manifest는 영구 저장소에 분리 보관한다. GPU 종류·세션 길이·저장 용량·총 예산과 D1/D3/D4/D5는 계속 미정이다. 압축 준비 기능의 구현·실행 완료를 의미하지 않는다.

**2026-10-14 계획 구체화와 후속 구현:** 현재 압축 형식을 유지한 Drive 부분 읽기 → Colab CPU 순차 해제·선택 추출 → 검증된 추출본에서 GPU 학습 배치 구성 경로를 따른다. 후속 `rewind_archive`의 제한된 준비 코어·모의 Drive adapter와 개발용 상한을 구현했다. 실제 Colab 운영 상한·처리량·인증/API·GPU 연결 실측은 남아 있으며 D2 자원·예산 확정을 뜻하지 않는다.

M0–M4는 제안하는 순차 개발 경로이며, M4까지를 필수 완료 범위로 할지는 D1에서 확정한다. M5는 GPU 예산에 맞춰 E1–E3부터 선택하고 M6에서 통합한다. 일정은 고정 날짜 대신 단계별 완료 조건으로 관리한다. Pilot 후 step당 시간 × 예정 update 수 × 실험 횟수로 GPU 시간을 추산하고 데이터 준비·평가·중단 후 재준비 여유를 별도로 잡는다.

**실현 가능성 판단:** small sample·overfit·pilot부터 확대하는 구조는 기술적으로 타당한 계획이다. 논문 크기 모델, 200만 step 규모나 여러 seed의 반복 실험을 현재 Colab 자원·예산 안에서 수행할 수 있는지는 미확정이다. 실제 GPU·VRAM, RGB front-end의 해상도·batch, 입출력 및 학습 시간을 측정한 뒤 모델 규모와 실험 수를 정한다. 30 FPS와 성능 개선은 측정할 목표이며 보장된 결과가 아니다. Colab 자원과 런타임 수명도 고정값으로 가정하지 않는다. [6]

### 주요 위험과 대응

| 위험 | 대응 |
| --- | --- |
| 손 annotation 또는 camera 정보 부족 | 샘플 감사로 먼저 확인하고 데이터 범위·주장을 조정 |
| 저장 공간과 Colab 세션 제한 | 부분 다운로드·2D 캐시·resume 검증 |
| 좌표 오류가 모델 성능으로 보임 | 재투영·단위·skeleton 검사부터 통과 |
| 정확도와 속도 개선이 충돌 | 지정 GPU의 Pareto 결과로 모델 선택 |
| 공개 정보로 세부 구현이 확정되지 않음 | 가정·원본과의 차이·ablation 결과를 기록 |

### 이번 검토에서 조정할 항목

- [ ] **D1** 최소 완료 범위: M4까지를 필수로 하고 개선 실험은 E1–E3부터 진행할지.
- [ ] **D2** 실행 자원: 실제 GPU 종류, 1회 세션 길이, 저장 공간과 총 학습 예산.
- [ ] **D3** 성능 우선순위: 정확도, 실시간성, failure recovery 중 가장 중요한 항목.
- [ ] **D4** 결과물 범위: 모델·평가 코드까지 또는 실시간 시각화 demo까지.
- [ ] **D5** 후속 범위: identity conditioning과 다른 데이터셋 확장을 언제 포함할지.

### 근거 자료

[1] Lee et al. *REWIND: Real-Time Egocentric Whole-Body Motion Diffusion with Exemplar-Based Identity Conditioning*. CVPR 2025. 첨부 논문 본문과 부록. [공식 프로젝트](https://jyunlee.github.io/projects/rewind/)

[2] 첨부 프로젝트 요약서. 프로젝트 목표, Colab 환경, 우선순위와 제외 범위.

[3] [UnrealEgo Benchmark 공식 데이터 안내](https://unrealego.mpi-inf.mpg.de/). 확인일 2026년 10월 12일. 공개 안내와 실제 archive의 annotation은 구현 착수 시 대조한다.

[4] UnrealEgo2 공식 분할·해제 스크립트: [train/val RGB](https://unrealego.mpi-inf.mpg.de/data/download_unrealego2_train_val_rgb.sh), [train/val pose](https://unrealego.mpi-inf.mpg.de/data/download_unrealego2_train_val_pose.sh). 2026-10-14 내용 확인 및 스크립트에 명시된 CityPark 원본 URL의 제한된 바이트 시험. 다운로드 스크립트 전체를 실행한 것은 아니다.

[5] [Google Drive API 부분 다운로드](https://developers.google.com/workspace/drive/api/guides/manage-downloads#partial_download). 2026-10-14 확인. API의 지원 사실과 사용자 Drive 복사본에서의 실제 연결 검증은 구분한다.

[6] [Google Colab FAQ](https://research.google.com/colaboratory/faq.html). 2026-10-14 확인. Drive I/O 비용·제한, 런타임 삭제와 가변 자원을 고려한다.

[7] [zlib의 압축 스트림 인덱스·임의 접근 예제](https://raw.githubusercontent.com/madler/zlib/master/examples/zran.c). 2026-10-14 확인. 해제 재시작에는 압축 위치뿐 아니라 bit 위치·사전 상태 등이 필요하다.
