# 구현·검증 이력

2026-10-27 README의 긴 상태 기록을 이 문서로 옮겼습니다. 아래는 각 작업 당시의 구현·검증·미검증 범위를 기록한 이력이며, 과거 수치와 순서를 유지했습니다. 상대 링크만 문서 위치에 맞췄습니다. 현재 상태와 실행 결정은 [README](../README.md) 및 [로드맵](ROADMAP.md#현재-상태와-실행-결정)을 따릅니다.

과거 `outputs/...` 근거 중 이동된 파일은 [로컬 보관·개별 복원 안내](LOCAL_STORAGE.md#개별-파일-복원)에 따라 인덱스를 확인한 후 복원합니다. 이력에 적힌 실행 명령이나 당시 후속 제안은 새 실행 지시가 아닙니다.

## 이전 README 상태 기록

- 프로젝트 로드맵 v0.2의 첫 구현 묶음인 M0 사전 JSON 감사·공통 로더·로컬 HTML viewer를 구현했습니다. 전체 완료 범위와 예산은 계속 검토 중입니다.
- 이전 M0에서 지정 Drive 폴더의 UnrealEgo2 pose JSON `frame_0`~`frame_17` 18개, 각 87관절의 필수 배열 수치 검사를 통과했습니다. 이는 이전 검증 이력이며 단위·축·회전·시간·calibration·분할·공식 skeleton 검증은 남아 있습니다.
- 2026-10-14 Drive 목록을 재확인했고, 별도로 공식 서버 CityPark 원본에서 부분 다운로드·추출 가능성을 시험했습니다. RGB 앞 2MiB에서 PNG 2개, pose 앞 256KiB에서 JSON 16개를 완전히 읽었습니다. 이후 새 준비 도구로 저장된 prefix를 로컬 재생해 18개 hash 일치를 확인했습니다. 서로 다른 sequence여서 완전 RGB 좌우+pose triplet은 0개이며 당시 Drive 복사본 실행·Colab 연결은 미검증이었으며, 후속 pose 시험은 아래에 구분합니다. 전체 무결성은 미검증입니다. 세부 근거와 한계는 [로드맵](ROADMAP.md#부분-다운로드-조사와-실행-계획)에 기록합니다.
- 데이터 없이 공식 규격 근거와 M1 준비용 causal window 조립·streaming API를 추가했습니다. 합성 테스트 40개(기존 17 + window 23)와 fixture 실행을 통과했으며 실제 데이터 검증·모델 학습은 포함하지 않습니다.
- 2026-10-14 M1 학습 기반을 추가했습니다. 명시적 tensor 계약, 작은 single-frame/causal temporal regression, loss·평가·checkpoint 재개와 train-only 합성 RGB 예측 캐시 연결을 구현했습니다. 당시 Python 테스트 59개와 합성 CPU 학습·재개·평가를 통과했습니다. 실제 데이터 baseline·EfficientNet front-end·Colab GPU는 미검증이며 M1 전체 완료가 아닙니다.
- [Colab notebook](../notebooks/m1_baseline.ipynb)을 준비했습니다. 로컬 소스 ZIP 업로드를 기본으로 하고 임시 디스크에서 합성 실행 후 선택적으로 결과를 Drive에 export합니다. 초기 코드 셀의 로컬 CPU 실행과 로컬 복사 검증을 통과했습니다. 이후 사용자 Colab의 소스 업로드·CPU 실행은 아래 후속 결과로 확인했으며 GPU 검증은 남아 있습니다.
- `rewind_archive`에 분할 gzip/tar의 상한 있는 순차 읽기·선택 추출·이름 대응·hash 재사용 검사를 추가했습니다. 당시 Python 테스트 83개(기존 59+archive 24)를 통과했고, 실제 HTTP 검증은 로컬 loopback 서버입니다. M0/M1 전체 완료가 아닙니다.
- Drive의 명시한 part ID·기대 이름만 조회해 size/version을 자동 확인하는 metadata→manifest builder와 별도 pilot 보고서를 추가했습니다. 당시 전체 테스트 93개를 통과했습니다. [준비 notebook](../notebooks/archive_prepare.ipynb)에서 단일 JOB 설정 → 네트워크 없는 preflight → 기본 비활성 1회 metadata/pilot → 결과 검토·ZIP 보관 순서로 실행합니다. 별도 선택 JSON 업로드는 필요하지 않습니다.
- 2026-10-15 사용자 Colab에서 metadata HTTP 200·chunked·Content-Length 없음 때문에 `metadata_protocol` 실패가 관측됐습니다. 양쪽 metadata 경로를 byte/시간 상한 내 EOF 읽기로 수정했고 전체 97개 회귀와 실제 라이브러리 loopback 오류 검사를 통과했습니다. Media의 strict 206 Range 조건은 유지합니다. 후속 사용자 Colab 보고서에서 metadata 생성과 pose 32개 추출·verify 성공을 확인했습니다. 원본 앞 512KiB를 읽고 파일 수 상한에서 정상 중단했으며, 당시 관절·좌표·RGB 대응과 실제 학습은 미검증이었으며, 후속 JSON 감사는 아래에 구분합니다.
- JSON·window 도구는 Python 표준 라이브러리로 실행합니다. M1은 별도 PyTorch 환경이 필요하며 이번 검증은 Python 3.12.14 / torch 2.7.1+cpu입니다. 실행법과 한계는 [M1 안내](M1_BASELINE.md)에 기록합니다. 기존 실제 sample 이력은 [M0 도구 안내](M0_JSON_TOOLS.md)에 있으며 실제 브라우저 화면·조작 확인은 브라우저 URL 보안 정책으로 미실행입니다.
- 2026-10-15 후속 [ArchVis sample 감사](ARCHVIS_POSE_AUDIT.md)에서 Drive ZIP의 32 JSON hash·87관절 필수 배열을 대조했습니다. 연속 18프레임 smoke와 실제 causal window 검사를 통과했고, 산재된 Opening 14개는 gap 진단을 유지합니다. 좌표 사이의 수치 관계를 확인했으며 단위·시간·RGB/calibration·split과 학습은 미검증입니다.

- 후속 공식 자료 대조에서 두 sample의 train 경로 소속과 Scaramuzza 3D→2D annotation 정합성을 확인했습니다. 실제 RGB 대응·단위·시간은 남아 있고, 공식 RGB 부분 읽기는 상한 부근 오류로 재사용 불가 처리했습니다. [후속 근거](ARCHVIS_POSE_AUDIT.md)를 따릅니다.

- M2 body diffusion teacher의 [합성 구현 묶음](M2_BODY_TEACHER.md)을 추가했습니다. Graph convolution·spatial attention·RoPE causal denoiser, x0/DDIM, 명시적 camera 계약·checkpoint 재개·streaming을 구현했습니다. 전체 107개 테스트와 CPU 합성 100→110 update 재개를 통과했습니다. 실제 body baseline·regression 대비 효과·GPU 성능은 미검증이며 M2 전체 완료가 아닙니다.

- M3 body–hand cascade의 [합성 구현 묶음](M3_WHOLE_BODY_TEACHER.md)을 추가했습니다. 학습된 body checkpoint 고정, GT 상체 조건 hand 학습, 예측 상체 평가, 독립 none 모델·GT 진단 비교와 손목 상대/전체 좌표 지표를 구현했습니다. 전체 119개 테스트, CPU body 80·hand 각 100 update와 조건부 hand 110까지 재개를 통과했습니다. 실제 UnrealEgo whole-body teacher·Colab GPU·cascade 성능 개선은 미검증이며 M3 전체 완료가 아닙니다.

- 2026-10-16 사용자 제공 Colab 결과 ZIP·실행 notebook을 검토했습니다. Python 3.13.16 / torch 2.11.0+cpu에서 전체 119개 테스트, named-body 80·upper/none hand 각 100 update, upper hand 110까지 재개·평가 성공을 확인했습니다. 실행 소스는 `5e52aa3`이고 31개 코드 hash와 checkpoint 참조가 일치합니다. 실제 데이터·GPU 검증과 구분하며 [M3 검토 기록](M3_WHOLE_BODY_TEACHER.md#사용자-colab-cpu-검증-2026-10-16)을 따릅니다.

- 2026-10-16 실제 ArchVis 32개 JSON의 87개 관절을 대조한 [관절 대응표](JOINT_MAPPING.md)를 작성했습니다. 기존 개발용 62개·평가 후보 16개의 포함 여부와 body/upper/손목/손가락 후보, 미선택 25개를 구분했습니다. 이름 존재 확인과 모델 역할·부모 연결의 확정은 구분하며 학습 설정은 변경하지 않았습니다.

- 2026-10-16 후속 [단위·좌표계 및 학습 관절 검토](COORDINATE_AND_TRAINING_REVIEW.md)에서 실제 32개 JSON의 world↔camera/device 변환과 카메라 degree/device radian 관계를 수치 대조했습니다. 거리 단위 cm는 논문의 rig 간격과 실제 baseline으로 강하게 지지되지만 UE2 raw 필드 직접 선언과는 구분합니다. Body22+finger40, upper13을 초기 학습 구성으로 권고하며, 모델 연결에는 좌표계의 축 반전 처리와 별도 변환기가 필요합니다. 설정·학습 가드는 유지했고 실제 학습은 실행하지 않았습니다.

2026-10-16 후속 작업으로 [실제 pose 변환기](REAL_DATA_PREPARATION.md)와 별도 `real_archvis62_v1` 규격을 추가했습니다. Body22+finger40·upper13·평가16 및 graph를 프로젝트 가정으로 채택하고, cm 근거를 명시해 출력은 mm로 통일합니다. 기존 개발용 설정은 유지합니다. 변환 산출물은 검토용 candidate이며 학습 입력으로 자동 승인하지 않습니다. 대응 RGB·독립 validation·원본 motion 검증이 남아 있어 실제 학습은 아직 실행하지 않았습니다. [다음 실제 데이터 준비 실행](REAL_DATA_NEXT_RUN.md)에 Drive 선택 입력과 실행 순서를 준비했습니다. 이후 네 번의 RGB 실행 실패와 개선 흐름은 아래 기록을 따릅니다. 전체 133개 테스트와 실제 32파일 변환·독립 수치 대조를 통과했습니다.

- 2026-10-16 사용자 RGB pilot 4개 ZIP을 대조했습니다. 모두 완전 파일 0개이며, 3회는 64KiB media마다 전후 metadata를 요청해 약 2.69MiB에서 총128회 상한에 도달했습니다. 같은 prefix 반복이어서 이어받기가 아닙니다. 첫 실행의 HTTP 비206 오류는 당시 실제 상태 코드가 기록되지 않아 원인을 확정하지 않습니다. [실행 결과와 다음 준비 순서](REAL_DATA_NEXT_RUN.md)를 따릅니다.

- 요청 효율 개선에서 HTTP Range는 기본256KiB, 내부 read1·해제 출력은64KiB로 분리했습니다. 총128요청·8MiB·60초 상한과 원본 전후 검사는 유지합니다. 로컬 실제 전송 라이브러리 시험은 총96요청에서 byte 상한으로 중단했고, 별도 저장된 공식 prefix HTTP 재생에서 PNG2·JSON16의 기존 hash가 모두 일치했습니다. 당시 전체 141개 테스트와 notebook 10개 코드 셀의 로컬 검증을 통과했습니다. 이후 실제 Drive/Colab에서 오른쪽 RGB2개 추출은 확인됐으나 대응 stereo sample은 아직 없습니다. 아래 후속 검토를 따릅니다.

- 2026-10-16 후속 사용자 결과에서 Serena `Getting_Up__1_` 오른쪽 frame10·1 확보를 확인했습니다. 왼쪽만 선택한 `5935e818`은 8,356,024bytes·96요청에서 0파일로 중단했습니다. 기존 실행과 같은 000 앞부분만 읽었고 후속 조각까지 도달하지 않아, 불완전 다운로드가 직접 원인이라는 근거는 없습니다. 필터는 gzip seek 기능이 아닙니다.
- `archive_prepare`를 단일 설정과 위에서 아래로 실행하는 흐름으로 재작성했습니다. 새 기본 scan은 기존8MiB·60초 상한에서 저장0파일로 regular member 순서·경로·완료 여부를 기록하는 진단입니다. 전체 archive 인덱스가 아닙니다. 전체155개 로컬 테스트와 저장된 실제 prefix scan 재생에서 RGB3·pose17개 경로 관측, 완료payload2·16개와 저장0파일을 확인했습니다. 후속 2026-10-17 사용자 Colab scan 결과는 아래에 구분합니다. [다음 실행 안내](REAL_DATA_NEXT_RUN.md)를 따릅니다.

- 2026-10-17 테스트 세션이 취합한 `af6961a`의 실제 Colab scan 결과를 독립 검토했습니다. 기존과 동일한000 앞8,350,136bytes에서 오른쪽 RGB 경로7개(완료payload6·미완료1)를 관측하고 전송 상한으로 중단했습니다. Scan의 저장0파일은 정상이며 왼쪽 경로는 이 구간에서 관측되지 않았습니다. ZIP CRC·보고서 연결·소스11개 hash를 확인했으며, 전체 archive에서의 왼쪽 부재나 위치는 미확정입니다. 같은8MiB 재실행은 종료하고 다음 읽기 범위·비용 정책을 별도로 검토합니다.

2026-10-17 후속 준비로 `counterpart64MiB_v1` 고정 예산과 정확한 목표 파일을 모두 저장하면 멈추는 추출 경로를 추가했습니다. 기본 8MiB scan은 유지합니다. 이번 사용자 테스트의 대상은 기존 오른쪽 frame1·10에 대응하는 왼쪽 두 PNG이며, 준비 상한은 64MiB·300초·768요청입니다. 전체 165개 로컬 회귀 검사와 8MiB 뒤 목표를 둔 독립 모의 Drive 검증을 통과했습니다. 후속 사용자 Colab 결과에서 왼쪽 frame1·10 두 PNG 확보와 검증 성공을 확인했습니다. 압축27.5MiB를 읽고 준비150.50초에 목표 완료로 중단했습니다. 대응 pose·좌우 영상/시간 정합성과 실제 학습은 아직 미검증입니다. 실행 결과와 한계는 [다음 실행 안내](REAL_DATA_NEXT_RUN.md)를 따릅니다.

후속 pose 테스트는 압축7.5MiB·약29.9초에서 tar header1000 상한에 도달했고 목표 JSON0개로 종료했습니다. 동일 실행 반복을 중단하고 pose의 header·metadata·해제량 제한을 함께 검토해야 합니다. 실제 pose 확보·좌표/영상 대조·학습은 아직 미완료이며 [실행 안내](REAL_DATA_NEXT_RUN.md)를 따릅니다.

2026-10-19 후속 준비에서는 pose 전용 `poseCounterpart64MiB_v1`을 추가해 전송64MiB·300초는 유지하고 header10,000·누적metadata1MiB·해제512MiB를 적용합니다. 단일metadata64KiB와 저장/disk 상한도 유지합니다. API4 보고서에는 마지막 regular 관측과 총 관측 수를 추가합니다. 전체175개 회귀와 새 소스 ZIP의80개 사전검사·모의 노트북 실행을 통과했습니다. 후속 실제 Colab 실행은 압축9.5MiB·41.53초·header1293에서 `authorized_request_failed`로 중단했고 목표 pose0개입니다. 새 예산 상한이 중단 원인은 아니며, 예외 종류가 기록되지 않아 인증·통신·timeout 원인은 미확정입니다. 원본 후검사도 미완료여서 관측은 신뢰 불가 진단으로 유지합니다. 동일 재실행을 중단하고 [결과 및 다음 진단 범위](REAL_DATA_NEXT_RUN.md)를 따릅니다.

2026-10-19 요청 실패 진단 보완으로 API5의 `request_failure`에 허용된 예외 분류와 발생 단계만 기록합니다. 최초 오류가 cleanup 오류에 가려지지 않도록 했고, 예산·timeout·자동 재시도 없음은 유지합니다. 전체189개 로컬 회귀, 실제 라이브러리 예외23조건, 새 소스 ZIP의94개 사전검사와 모의 notebook 검증을 통과했습니다. 이는 과거 Colab 실패 원인의 확인이나 해결 증명이 아닙니다. 같은 pose 목표의 새 진단 버전을 테스트 세션에서1회 실행하도록 준비했으며 [실행 및 판정 기준](REAL_DATA_NEXT_RUN.md)을 따릅니다.

2026-10-21 API5 실제 테스트에서 다음 media 요청 전 metadata 확인의 `requests.read_timeout` 진단을 회수했습니다. 압축14.5MiB·준비58.15초 뒤 중단했으며 예산 상한이 원인은 아닙니다. 목표 pose0개로 추출은 실패했고 근본 원인은 미확정입니다. 동일 실행은 종료했으며 [결과와 후속 검토 범위](REAL_DATA_NEXT_RUN.md)를 따릅니다.

2026-10-21 API6에 별도 `metadataPreTimeout10sRetry1Global3_v1` 정책을 추가했습니다. Pose targeted 준비의 사전 metadata 요청 개시만 최대10초, 허용된 무응답 timeout은 요청당 추가1회·전체 추가3회로 제한하며 대기도 기존300초 예산에 포함합니다. 기본 정책과 전체 상한은 유지하고 숨은 라이브러리 재시도를 거부합니다. 전체210개 회귀와 fresh ZIP의115개 사전검사·모의 실행을 통과했습니다. 로컬 검증과 실제 Drive 회복·pose 확보는 구분하며 [새1회 실행 안내](REAL_DATA_NEXT_RUN.md)를 따릅니다.

2026-10-21 API6 실제 실행은 오류·재시도 없이 준비768요청·약232.18초에서 종료했지만 목표 pose0개입니다. 응답body도64MiB에 근접해 요청 상한만 늘리는 것으로 해결을 기대하지 않습니다. 재시도 회복 효과는 실제로 시험되지 않았으며, 다음은 [탐색 범위와 대응 시퀀스 확보 전략 검토](REAL_DATA_NEXT_RUN.md)입니다. 동일 실행은 종료했습니다.

2026-10-21 후속 [데이터 확보 전략 검토](DATA_ACQUISITION_STRATEGY.md)에서 기존 sample 교집합과 순차 탐색 비용을 비교했습니다. 같은 64MiB 실행 반복은 종료하며, 다음 개발로 압축 prefix 보관·재사용과 관측 카탈로그를 권고합니다. 아직 미구현이며 탐색 예산 확대도 비교안입니다. 새 원본 요청이나 실제 학습은 실행하지 않았습니다.

2026-10-21 후속 API7에 opt-in `posePrefixCache96MiB_v1`을 추가했습니다. 검증된 연속 압축 prefix를 별도 캐시 ZIP으로 보관하고, 재사용 시 hash·원본 전후 검사 후 byte0부터 로컬 해제하며 미보관 후속만 Drive에서 읽습니다. 상한 있는 sequence/frame 관측 카탈로그를 결과 ZIP에 보존합니다. 준비 상한은 새 응답 body96MiB·논리 압축입력96MiB·420초·1,157요청이며 최초 metadata와 인증·캐시 보관 비용은 구분합니다. 전체232개 회귀·새 ZIP의137개 사전검사와 모의 노트북 흐름을 통과했습니다. 당시 실제 Drive 캐시·목표 pose 확보·학습은 미검증이었으며 후속 cold 결과는 아래와 같습니다. [최신 실행 안내](REAL_DATA_NEXT_RUN.md)를 따릅니다.

2026-10-21 실제 API7 cold 시험에서 압축100,561,588bytes의 캐시와92개 시퀀스·6,318개 관측을 보존하고 hash를 검증했습니다. 준비311.33초 뒤 전송량 상한으로 종료했으며 목표 pose는0개입니다. 실제 warm 재사용·전송 절감과 학습은 미검증이고 사용자 사전137개 검사 로그·실행 notebook도 미제출입니다. 동일 cold/warm 반복은 요청하지 않으며 [실제 결과와 다음 판단 범위](REAL_DATA_NEXT_RUN.md)를 따릅니다.

2026-10-21 실제 cold 결과를 바탕으로 [캐시 재사용과 후속 탐색 준비안](PREFIX_CACHE_NEXT_STEP.md)을 정리했습니다. 보존 캐시의 오프라인 import·byte0 해제를 확인했으며, 논리 압축입력128MiB까지의 제한된 warm 확장을 다음 개발 후보로 검토합니다. 기존 캐시 뒤 새 media는 약32.1MiB이며 원본 검사·재해제·보관 비용은 별도입니다. 현 API7 설정만 바꾸어 실행할 수 있는 기능은 아니며, 구현·로컬 검토 후 테스트 세션에 고정 버전을 인계합니다. 실제 warm 실행과 목표 pose 확보는 아직 미검증입니다.

2026-10-21 후속 API8에 `posePrefixWarm128MiB_v2`를 구현했습니다. 보존한 캐시를 필수 입력으로 검증하고, 논리 압축입력 128MiB 안에서 미보관 구간만 추가로 읽습니다. 새 원본 응답 body 40MiB·420초·392요청과 별도 초기 metadata 상한을 적용하며, v1 입력과 새 v2 출력·재입력을 지원합니다. 전체 240개 로컬 회귀·독립 합성 6조건·새 ZIP의 145개 사전검사와 노트북 모의 실행을 통과했고, 보존된 실제 v1 캐시의 새 정책 오프라인 import·해제도 확인했습니다. 당시 실제 Drive warm은 미검증이었으며 후속 결과는 아래에 구분합니다. 목표 pose 확보·학습은 아직 미완료입니다.

2026-10-23 테스트 세션의 실제 API8 warm128 결과를 검토했습니다. 기존 100,561,588bytes를 캐시에서 재사용하고 새 media 33,656,140bytes만 읽어 총 128MiB·513chunk의 v2 캐시를 보존했습니다. 준비 112.09초·389요청 뒤 논리입력 상한에서 정상 중단했고, 118시퀀스·8,434개 경로를 관측했으나 목표 pose는 여전히 0개입니다. 캐시 재사용·후속 탐색 성공과 sample 확보 실패를 구분합니다. 사용자 사전검사 개별 로그·실행 notebook과 Drive 영구보관은 미확인이며 실제 학습도 미실행입니다. 다음은 보존 v2 캐시 뒤 64MiB를 추가 탐색하는 제한된 192MiB 경로의 개발·검토이며 실제 실행은 테스트 세션으로 인계합니다.

2026-10-23 API9에 `posePrefixWarm192MiB_v3`를 추가했습니다. 검증된 128MiB 이상 캐시와 고정 입력 식별값을 요구하고, 논리192MiB·새 응답body80MiB·420초·773요청 안에서 후속만 읽습니다. 기존 v1/v2의 상한을 유지하고 새 v3 출력·재입력을 지원합니다. 전체248개 로컬 회귀(59.026초), 새 소스 ZIP의153개 사전검사·7개 코드 셀 모의 실행, 실제 보관 v2 캐시의 오프라인 import·재해제와 독립 합성6조건을 확인했습니다. 실제192MiB 원본 탐색과 목표 pose 확보는 아직 미검증입니다. [192MiB 비용·상한 검토](PREFIX_CACHE_192_PLAN.md)와 [테스트 실행 안내](REAL_DATA_NEXT_RUN.md)를 따릅니다.
