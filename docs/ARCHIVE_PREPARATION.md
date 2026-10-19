# 분할 압축 원본의 제한된 순차 준비

2026-10-19 후속: Pose 전용 API4의 실제 실행은39번째 media에서 `authorized_request_failed`로 중단했습니다. 해당 보고서에는 예외 타입/정확한 발생 단계가 없어 원인을 확정하지 않습니다. 이번 API5는 예산/목표/재시도를 바꾸지 않고 안전한 요청 실패 진단을 추가합니다. 고정 revision을 테스트 세션에 인계해 별도 확인하며 실제 결과는 [로드맵](ROADMAP.md)을 따릅니다. 목표 pose·좌표/영상 검증은 여전히 미완료입니다.

2026-10-16 · 제한된 준비 코어와 단방향 notebook을 유지합니다. 최신 기본 작업은 payload 저장 없이 archive 앞부분의 regular member 순서를 관측하는 `scan`입니다. 실제 Drive를 byte0부터 읽고 해제하므로 전송 비용이 발생합니다. 전체 목록·seek index·이어받기는 아닙니다. 기존 metadata→manifest/pilot·선택 추출 API는 유지합니다. 분할 조각을 000부터 순차 읽고 선택한 완전한 regular file을 남깁니다. 전체 원본 다운로드·결합 파일·전체 추출본을 만들지 않습니다. 사용자 Colab의 HTTP200 chunked 실패를 수정했고, 후속 보고서로 pose 32개 추출·런타임 verify 성공을 확인했습니다. 후속 [sample 감사](ARCHVIS_POSE_AUDIT.md)에서 32 JSON의 hash·구조·수치와 연속 18프레임을 확인했습니다. 좌표 의미·시간·split·RGB 대응·전체 archive 무결성은 미검증입니다.

## 현재 가능한 범위

Python 3.10 이상 표준 라이브러리로 local file·HTTP Range 소스를 읽습니다. `AuthorizedSession` 주입형 Drive API adapter와 metadata builder의 실제 전송 라이브러리는 익명 credential·로컬 loopback으로 검사했습니다. 사용자 실제 metadata 응답의 전송 조건과 로컬 수정 검증을 구분하며 실제 Drive/Colab의 pose 소량 pilot 성공은 후속 사용자 보고서로 확인했습니다. GPU는 준비 작업에 필요하지 않습니다. Archive 준비 notebook은 [archive_prepare.ipynb](../notebooks/archive_prepare.ipynb), M1 합성 학습 notebook은 별도 [m1_baseline.ipynb](../notebooks/m1_baseline.ipynb)입니다.

현재 기능은 일반 tar 도구 전체를 대체하지 않습니다. 단일 gzip stream의 regular file·directory, GNU LongName과 제한된 PAX metadata를 지원합니다. Concatenated gzip, sparse file, link·device·FIFO 등의 special file은 거부합니다. gzip restart/index와 재포장, 실제 RGB loader·좌표 변환·EfficientNet 학습은 포함하지 않습니다.

`inspection_candidates_usable=true`는 무결성/안전성 실패 없이 남은 검사 후보를 사용할 수 있다는 뜻이며 완전 파일이 있다는 보장은 아닙니다. Scan의 저장 파일은0개이므로 별도 `member_observations_usable`와 trust를 함께 읽습니다. 학습 준비 완료가 아닙니다. 추출·verify·matching 결과의 **train_ready는 항상 false**입니다. M0/M1의 관절·단위·좌표/calibration·timestamp·split·원본 motion·RGB/pose 내용 정합성 검증을 유지합니다.

## 데이터 없는 실행

프로젝트 root에서 실행합니다. 새 output은 비어 있는 경로를 사용합니다. 명령은 Python 3.12.14에서 검증했습니다. PyTorch는 이 도구만의 실행·테스트에 필요하지 않습니다.

```sh
python -m rewind_archive.fixture --output outputs/archive_fixture
python -m rewind_archive extract --manifest outputs/archive_fixture/rgb_manifest.json --limits configs/archive_limits_smoke.json --include "Fixture/Day/person/avatar/001/Motion__1_/*" --output outputs/archive_rgb
python -m rewind_archive extract --manifest outputs/archive_fixture/pose_manifest.json --limits configs/archive_limits_smoke.json --include "Fixture/Day/person/avatar/001/Motion__1_/*" --output outputs/archive_pose
python -m rewind_archive verify --extraction outputs/archive_rgb --manifest outputs/archive_fixture/rgb_manifest.json
python -m rewind_archive verify --extraction outputs/archive_pose --manifest outputs/archive_fixture/pose_manifest.json
python -m rewind_archive match --extraction outputs/archive_rgb --extraction outputs/archive_pose --output outputs/archive_matching.json
python -m unittest discover -s tests -p test_archive.py -v
```

Fixture는 실제 단위·카메라가 없는 4개 frame(0·1·2·10)의 합성 pose와 작은 PNG입니다. RGB/pose 내부 순서는 frame 순서와 다르게 만듭니다. gzip 중간에서 3개 조각으로 나뉘고 완전 triplet은 4개지만 train_ready는 false입니다. Gap 3~9를 별도로 기록합니다.

`extract`/`pilot`/`scan`은 report의 inspection 후보 또는 bounded 관측 사용 여부가 true이면 exit 0, 손상·원본 변경·안전하지 않은 입력 또는 pilot verify 실패이면 exit 1입니다. **Exit 0은 전체 완료가 아닙니다.** 반드시 stop_reason·failure_class·CRC·완전 파일 수를 읽습니다. `verify`는 manifest·개별 hash가 맞을 때 exit 0입니다. `match`는 이름 계약 오류·중복·dataset version 불일치가 없을 때 exit 0이며 triplet 개수는 별도입니다. 설정·입출력 오류는 argparse exit 2입니다.

현재 샌드박스는 loopback 연결도 차단할 수 있습니다. 검증에서 이 경우 HTTP 테스트를 skip하거나 프로토콜 검사를 끄지 않고, 허용된 로컬 loopback 실행 환경에서 테스트했습니다. 외부 데이터 전송은 없었습니다.

## Archive manifest

준비 입력은 `archive_manifest_v1` JSON입니다. 필요한 필드만 받으며 token/header/credential 필드와 signed URL query는 허용하지 않습니다. 다음은 규격 예시이며 실제 파일의 metadata를 대신하지 않습니다.

```json
{
  "schema_version": "archive_manifest_v1",
  "archive_id": "fixture_pose",
  "dataset_version": "synthetic_archive_v1",
  "kind": "pose",
  "source_scope": "synthetic_fixture",
  "total_parts": 3,
  "parts": [
    {
      "index": 0,
      "name": "pose.tar.gz000",
      "size_bytes": 128,
      "version": "fixture_v1",
      "source": {"type": "local", "path": "pose.tar.gz000"},
      "identity": {"mtime_ns": 0}
    }
  ]
}
```

`kind`는 rgb/pose, `source_scope`는 synthetic_fixture/official_server/official_prefix_local/user_drive/local_unverified입니다. `total_parts`는 원본의 명시적 기대 조각 수입니다. 목록은 index 0부터 증가하는 연속 prefix이고 조각 이름은 index의 3자리 숫자로 끝나야 합니다. Index/name 중복·누락·중간 조각 시작을 거부하며 자동 정렬하거나 gap을 건너뛰지 않습니다. Prefix에 존재하지 않는 뒤쪽 조각은 읽지 않습니다.

각 part의 `size_bytes`와 `version`, source 식별자는 반드시 명시합니다. Version은 출처/수집 버전 기록이며 자동으로 원본의 진실성을 증명하지 않습니다. local path는 manifest 디렉터리 기준의 상대 경로나 명시한 절대 경로입니다. 개인 경로·원본 manifest는 `outputs/`/`data/` 등에 두고 코드에 고정하지 않습니다. 파일 크기와 mtime_ns는 실제 stat에서 채웁니다. 열어 둔 fd의 fstat을 읽기 전후 확인하고 path의 dev/inode도 대조합니다. Metadata를 악의적으로 같은 값으로 위조한 교체까지 보장하는 인증 체계는 아닙니다.

HTTP source는 다음과 같습니다. URL의 username/password/query/fragment를 거부하며 redirect도 따라가지 않습니다. HTTPS 인증서 검증을 유지합니다. 환경 proxy·cookie는 사용하지 않습니다.

```json
{
  "source": {"type": "http", "url": "https://example.org/data/pose.tar.gz000"},
  "identity": {"etag": "\"immutable-version\"", "last_modified": "Wed, 14 Oct 2026 00:00:00 GMT"}
}
```

Strong ETag 또는 Last-Modified 중 최소 하나가 필요합니다. 제공한 모든 validator가 응답과 일치해야 하며 If-Match/If-Unmodified-Since도 보냅니다. Media는 HTTP **206**, 정확한 Content-Range(start/end/전체 size), Content-Length와 identity encoding을 검사한 후 요청한 구간만 읽습니다. Range를 무시한 200이나 잘못된 header는 본문을 읽지 않습니다. **Media의** 길이 없는 chunked 응답은 지원하지 않습니다. Metadata의 별도 bounded EOF 읽기는 아래 조건을 따릅니다. 자동 재시도는 구현하지 않습니다.

## Drive runtime adapter

Drive part source/identity는 아래 구조입니다. size/version/modifiedTime은 인증된 Drive `files.get`에서 확인하며 목록 도구가 반환하지 않은 값을 추측하지 않습니다. 원격 part의 size_bytes는 전체 blob 크기입니다. 로컬에 저장한 prefix 크기를 원격 전체 크기로 넣으면 Content-Range 검사가 실패합니다.

```json
{
  "source": {"type": "drive", "file_id": "USER_SELECTED_FILE_ID"},
  "identity": {"version": "ACTUAL_VERSION", "modified_time": "ACTUAL_MODIFIED_TIME"}
}
```

CLI는 새 인증을 시작하지 않습니다. 런타임에서는 `prepare(..., authorized_session=session)`에 사용자가 인증한 session을 주입합니다. Notebook은 단일 JOB 설정→네트워크 없는 preflight→기본 비활성 EXECUTE_ONCE 한 번으로 metadata와 scan/pilot을 연결합니다. 선택·예산·현재 설정의 동일성을 검사한 후에만 인증합니다. Metadata 성공 후 실제 size/version/modifiedTime을 표시하고 media를 읽습니다. 사용자 Colab에서 metadata 생성과 pose 소량 pilot·verify 성공을 확인했습니다. 이는 RGB 준비·실제 학습 성공을 뜻하지 않습니다. Mount와 Drive API 부분 읽기는 다릅니다.

Drive adapter는 media Range 요청 전후 metadata의 size/version/modifiedTime을 대조합니다. Metadata는 HTTP200·identity content encoding·유효 JSON object를 요구합니다. Content-Length가 있으면 기존 양의 길이/상한/short-body 검사를 유지하고, 없으면 chunked 또는 EOF 종료 응답을 read1로 제한 읽습니다. Declared length와 Transfer-Encoding 혼재, 미지원 encoding·HTTP 실패·잘린 JSON/chunk·종료 chunk 누락은 거부합니다. Media는 정확한206조건을 유지합니다. Null/list/string도 거부하며 auth 재요청/transport retry는 기존0설정을 유지합니다.

길이 없는 metadata는 response 상한(기본16KiB)과 단계의 남은 global body 상한을 넘겨 읽지 않습니다. **정확히 상한까지 읽으면 EOF를 확인하지 못한 상태로 실패**하며 1byte를 더 읽지 않습니다. Metadata report request_audit와 준비 stats.metadata_response_audit에는 정해진 status·길이·TE/CE/Content-Type 분류와 수신 byte만 남기며 원본 header·URL·본문을 출력하지 않습니다. Chunk framing은 실제 transport 라이브러리가 해석하고 오류는 `http_body_failed`로 유지합니다. [HTTP chunked framing](https://www.rfc-editor.org/rfc/rfc9112.html#section-7.1)

응답 body는 urllib3의 `raw.read1(size, decode_content=False)`를 요구하며 미지원 API에 `raw.read(size)`로 우회하지 않습니다. 이는 단일 read 후 deadline을 확인하기 위한 계약입니다. 접근 가능한 socket은 잔여 시간으로 timeout을 줄이고, 접근 불가하면 session의 요청 timeout과 read1 사이 협력적 검사를 사용합니다. 모의 session과 실제 라이브러리의 loopback 전송을 구분해 검사했습니다. [urllib3 read1 API](https://urllib3.readthedocs.io/en/stable/reference/urllib3.response.html#urllib3.response.HTTPResponse.read1), [google-auth AuthorizedSession](https://google-auth.readthedocs.io/en/latest/reference/google.auth.transport.requests.html#google.auth.transport.requests.AuthorizedSession), [Drive 부분 다운로드](https://developers.google.com/workspace/drive/api/guides/manage-downloads#partial_download)

최초 인증과 credential refresh는 builder/prepare 밖에서 수행하며 그 비용은 통계와 별도입니다. Credential은 session에만 존재하고 manifest/code/report/log에 저장·출력하지 않습니다. 준비 중 before_request refresh나 transport 자체 동작이 추가될 수 있어 실제 Colab에서 별도로 검증해야 합니다. 로컬 검증에서는 Google 인증·대용량 원본 읽기·프로젝트 의존성 변경 없이, Git 제외 경로에 격리 설치한 전송 검증용 라이브러리를 사용했습니다.

## Drive metadata에서 manifest와 pilot으로

Notebook은 단일 `JOB["selection"]`에 다음 `drive_part_selection_v1` dict를 입력합니다. 선택/plan JSON 업로드는 요구하지 않으며 숨은 plan 덮어쓰기 셀도 없습니다. 범용 metadata API/CLI는 기존 선택 dict/JSON 계약을 유지합니다. ID·기대 파일명은 사용자 Drive 목록에서, total_parts는 공식 목록에서 확인합니다. Size/version/modifiedTime을 입력하거나 이전 관측 값으로 채우지 않습니다. 개인 ID·경로·인증 정보는 커밋하지 않습니다.

```json
{
  "schema_version": "drive_part_selection_v1",
  "archive_id": "user_selected_pose",
  "dataset_version": "UnrealEgo2_unverified",
  "kind": "pose",
  "total_parts": 1,
  "parts": [{"index": 0, "name": "USER_ARCHIVE.tar.gz000", "file_id": "USER_FILE_ID"}]
}
```

1. 코드 source ZIP 한 개를 업로드하고 로컬 사전 검사94개(archive24+metadata14+planning8+scan7+workflow7+targets10+pose-profile10+diagnostics14)를 실행합니다. 배포 묶음을 올리면 내부 source ZIP 파일명을 안내하고 거부합니다. 중첩 ZIP을 자동 해제하지 않습니다. 새 source를 쓰려면 기존 결과를 먼저 보관하고 커널을 재시작합니다. Notebook/source API는5이며7개 코드 셀의 순서는 유지합니다.
2. 단일 `JOB` 설정 셀(id=`job-config`)만 편집합니다. Generic 기본값은 `profile="smoke8MiB_v1"`, `mode="scan"`, `purpose="prefix_inventory"`, `includes=[]`, `exact_targets=[]`입니다. 별도로 검토한 대응2파일 추출은 아래 `counterpart64MiB_v1` 계약을 사용합니다. 새 job_id와 목적 reason을 함께 기록합니다. 기존9필드 custom8MiB discovery 계약도 유지하며 이 경우 profile=None, limits.max_files2이하와 명시 glob을 사용합니다. Notebook이 목적이나 필터를 바꾸지 않습니다.
3. Preflight에서 mode/purpose·selection/archive·필터·Range256KiB/IO64KiB·예산·동일 설정/선택 이력을 확인합니다. 이 단계는 원본 요청과 인증0입니다. 실제 source size/version/modifiedTime은 아직 unknown이며 metadata 성공 뒤 따로 표시합니다. `includes`는 gzip 위치 점프가 아닌 저장/선택 필터입니다.
4. 다음 셀의 `EXECUTE_ONCE=True`를 명시해 한 번 실행합니다. 현재 JOB와 frozen preflight가 다르면 인증 전 거부합니다. 먼저 job을 기록하고 인증→metadata→scan/pilot을 실행합니다. Metadata는 [drive_metadata_limits_smoke.json](../configs/drive_metadata_limits_smoke.json)의 파일8·요청8, 응답16KiB·합128KiB,30초/요청5초입니다. 최초 metadata 단계와 선택한 준비 프로파일의 비용을 구분합니다. 실패도 job을 소비하고 자동 재시도하지 않습니다.
5. 아래 결과 검토→ZIP 셀로 진행합니다. 별도 아래쪽 새 job 셀이나 위쪽으로 돌아가 plan 업로드를 실행하는 절차는 없습니다. 같은 job ID는 재실행을 차단합니다. 같은 effective 설정 fingerprint를 새 ID로 반복하려면 기존과 **변경한 reason**도 단일 JOB에 명시한 뒤 preflight부터 실행해야 합니다. 이것은 새로운 byte0 재읽기이며 이전 압축 상태를 이어받지 않습니다. 현재 커널과 현재 source outputs의 작은 job 기록만 인식하므로, 과거 다른 런타임/ZIP도 별도로 검토합니다.
6. `mode="review"`, `purpose="review"`, `previous_result`는 기존 런타임의 결과 폴더만 검사합니다. 코드 hash가 바뀌거나 source 실패인 추출본은 현재 코드로 verified라고 표시하지 않습니다. 원본/인증 요청과 ZIP 자동 업로드는 하지 않습니다.

`build_drive_manifest(selection, session, MetadataLimits)`는 manifest 또는 None과 sanitized report를 반환합니다. `save_metadata_bundle`은 새 디렉터리만 사용하고 기존 파일을 덮어쓰지 않습니다. 성공이면 `metadata/archive_manifest.json`·`metadata_report.json`, 실패이면 report만 보존합니다. Code/revision/hash·실행 환경·metadata snapshot·요청 body 범위와 stats를 기록합니다. 실패한 부분 snapshot에서 유효 manifest를 만들지 않습니다.

`run_metadata_pilot`은 결과 hash/code 대조 → 기존 prepare → verify → compact `pilot_report.json`을 연결합니다. 저장 파일과 반환 report는 같고 metadata report hash와 **metadata_stage_stats/limits**를 포함합니다. Metadata128KiB/30초와 선택한 준비 프로파일은 별도이며 전후 metadata 비용은 후자의 stats에 포함됩니다. 준비 결과 파일·CRC·stop/failure·verify·미검증 항목과 train_ready=false를 확인합니다. Source/body/시간의 측정 한계는 아래와 같으며 인증·wire 전체나 실제 성능을 주장하지 않습니다. 로컬 manifest는 `python -m rewind_archive pilot --manifest ... --limits ... --include ... --output ...`로 동일 보고서를 생성할 수 있습니다. 작은 local bundle 외 Drive 쓰기·업로드는 추가하지 않았습니다.

## 안전한 요청 실패 진단

`request_failure`는 HTTP `response`/기존 `.diagnostic`과 분리된 optional 구조입니다. Metadata report와 request audit, extraction report/range audit/전후metadata audit, pilot, job, review와 결과 ZIP까지 전달합니다. 실패로 응답을 얻지 못하면 HTTP 상태를 만들어내지 않습니다. Pre/post metadata 요청 실패도 해당 audit에 수신 bytes와 단계만 기록합니다. 원시 예외 str/repr, URL/token/header/body, 임의 클래스명, traceback/exception chain·args는 직렬화하지 않습니다.

```json
{"schema_version":"request_failure_v1","category":"read_timeout","library":"requests","operation":"media","stage":"request_open"}
```

위는 예시이며 직전 실패를 이 유형으로 분류한 것이 아닙니다. Category/library는 이미 로드된 실제 `requests.exceptions`, `urllib3.exceptions`, `google.auth.exceptions`와 표준 라이브러리의 allowlist 타입에 `isinstance`로 대응합니다. 진단을 위해 추가 import·인증·네트워크·재시도를 시작하지 않습니다. 이름만 같은 사용자 클래스나 알 수 없는 wrapper는 unknown이며 내부 cause를 추측하지 않습니다.

| 필드 | 허용 값과 의미 |
| --- | --- |
| category | authentication, connect_timeout, read_timeout, timeout, tls, connection, transport, http, incomplete_body, request_other, os_error, unknown |
| library | requests, urllib3, google_auth, stdlib, unknown |
| operation | metadata_initial, metadata_pre, metadata_post, media, authentication, workflow, unknown |
| stage | request_open, response_wrap, body_read, response_validation, response_close, authentication_setup, unknown |

ConnectTimeout/ReadTimeout처럼 구체적인 타입만 구분합니다. 일반 TimeoutError는 timeout이며 connect/read를 추측하지 않습니다. Google TransportError는 authentication으로 통합하지 않고 transport로 구분합니다. `authentication`은 해당 라이브러리 타입군이라는 뜻이며 자격 증명 무효·권한 거부·quota의 확정 진단이 아닙니다. HTTPError 타입의 http 분류도 숫자 status를 대신하지 않습니다. Classifier는 원문이나 nested exception의 속성을 따라가지 않습니다. Body 예외의 기존 partial bytes는 누적량/hash 계산에만 사용하며 내용을 출력하지 않습니다.

Session.get 요청 호출과 SessionResponse wrapping을 분리하지만 두 실패의 기존 stop_reason은 `authorized_request_failed`로 유지합니다. 본문은 `http_body_failed`, 응답 protocol/원본 비교는 기존 중단 이유를 유지합니다. 먼저 발생한 body/validation 실패나 KeyboardInterrupt를 후속 close 오류가 덮지 않습니다. Cleanup만 실패했을 때 response_close를 기록하고 unusable 상태를 유지합니다. 새로운 진단은 source_validation_pending·partial 삭제·원본 전후 검증·usable/목표 완료 판정·budget/timeout/retry0를 우회하지 않습니다.

새 구조는 닫힌 enum과 고정 key만 허용해256bytes 이내로 제한됩니다. 임의 key/값은 버리거나 unknown으로 정규화합니다. Fail-stop이므로 실패 진단을 무한 축적하지 않으며 정상 단계별 metadata audit도 기존 max_ranges에 묶입니다. Media1+전후metadata2의 report 비용(두 report의 metadata audit 복사 포함)은 기존 요청당1024bytes 예약 안인지 검사하고, 상위 진단의 작은 고정 복사도 기존 base reserve를 사용합니다. Disk/ZIP 상한은 늘리지 않습니다.

정상/자체 중단 등 진단이 없으면 request_failure는 null입니다. 이전 보고서의 누락 필드는 review에서 null, request_failure_recorded=false로 표시하며 과거 원인을 새 코드로 소급 추정하지 않습니다. 새 optional 필드는 report 자체hash에 포함되고 report v1의 기존 입력은 읽을 수 있습니다. 코드 hash 검증을 임의로 해제하는 호환 기능은 아닙니다. API5 source와 notebook을 함께 사용하며 JOB11필드·14셀/7코드 셀·예산·정확한 목표는 그대로입니다.

직전 media39의 status 미기록/adapter 수신0bytes는 인증·연결·timeout·TLS나 wrapping 중 어느 것인지 구분할 근거가 아닙니다. Adapter의0bytes는 전체 network wire 전송0의 증거도 아닙니다. 동일 구버전 실행을 반복하지 않고 테스트 세션에 전달한 고정 진단 버전에서 수집한 결과만 검토합니다.

## 고정 예산 프로파일과 정확한 목표 완료 중단

| 프로파일 | 준비 body | 협력적 시간 | 총 source 동작 | 해제 상한 | 저장 파일 |
| --- | --- | --- | --- | --- | --- |
| smoke8MiB_v1 | 8MiB | 60초 | 128 | 32MiB | 최대32 |
| counterpart64MiB_v1 | 64MiB | 300초 | 768 | 128MiB | 정확한 목표 최대2 |
| poseCounterpart64MiB_v1 | 64MiB | 300초 | 768 | 512MiB | 정확한 pose 목표 최대2 |

기본 smoke 설정은 바꾸지 않습니다. [counterpart64MiB_v1 설정](../configs/archive_limits_counterpart64MiB_v1.json)은 별도 선택한 1회 targeted 추출 상한입니다. Range256KiB·IO64KiB·요청 timeout5초·파일4MiB·disk16MiB·경로1024bytes·tar metadata64KiB·header1000·관측목록128 등 나머지 상한은 같습니다. Python의 frozen Limits와 변경 불가 mapping으로 preset을 유지하고 JSON과 일치 검사합니다. 명시한 프로파일명과 실제 limits가 한 필드라도 다르면 거부하며, 수치만 임의로 올릴 수 없습니다.

Notebook의 유일한 JOB 셀에서 `profile`, `limits`, `mode="extract"`, `purpose="targeted"`, `exact_targets`의2개 경로와 동일 집합의 `includes`를 함께 설정합니다. BUDGET_PROFILE은 같은 셀에서 `profile_limits(...)`로 고정값을 가져오는 편의 변수일 뿐 별도 입력 경로가 아닙니다. Root가 배포하는 사용자용 사본도 이 셀만 미리 채웁니다. Generic notebook에는 개인 Drive ID를 넣지 않습니다.

`exact_targets`는 선택 glob과 별개인 literal 경로입니다. Canonical safe path, 경로 길이, 중복/대소문자 충돌, glob 문자(`*?[]`) 금지, 최대32개와 max_files 이내를 검사합니다. Includes와 정확히 같은 집합이어야 하므로 추가 저장 파일이나 모호한 wildcard 완료를 허용하지 않습니다. Counterpart 프로파일은 정확히2개와 max_files2를 요구합니다. 기존 API의 optional `profile=None`, `exact_targets=None` 기본값은 기존 extract/scan 동작을 유지합니다. 일반 CLI의 `--profile`도 정확한 preset 일치를 검사하며 `--exact-target`은 반복 지정합니다.

두 목표의 payload+padding을 소비하고 파일 게시까지 끝나면 `stop_reason="targets_complete"`, `failure_class="bounded_stop"`로 중단합니다. 다음 tar header나 source.read/Range를 더 호출하지 않습니다. 이미 받은 Range의 선수신 bytes는 transfer 통계에 포함되며 제거하지 않습니다. 현재까지의 원본 전후 metadata 검사도 유지합니다. 이후 gzip trailer/다른 member를 검사한 것이 아니므로 전체 archive 무결성과는 구분합니다. 관측목록128개를 넘겨도 extract는 목록만 자르고 목표 추출을 계속합니다.

Reports/job에는 profile·실제 limits·`exact_targets`, `stored_targets`, `completed_targets`, `missing_targets`, `targets_complete`, `targets_status`를 남깁니다. Stored는 디스크에 남긴 파일이며 usable/검증 완료를 보장하지 않습니다. Source/parser/후검사 실패이면 completed는 비우고 missing에 모든 목표를 표시하며 targets_status는unusable입니다. 정상 상한/불완전 prefix에서는 완전히 저장한 첫 목표만 completed에 남길 수 있습니다. Missing은 사용할 수 있는 완료 목표가 없다는 뜻이며 원본 전체에서 파일이 없다는 증거가 아닙니다.

**최종 성공은 stop_reason만으로 판정하지 않습니다.** Pilot의 verification.status=passed, targets_complete=true, targets_status=complete와 현재 파일 검증 통과를 함께 확인합니다. 파일 hash 재검사가 실패해도 저장 단계의 stop_reason은 targets_complete로 남을 수 있습니다. 이때 pilot 완료 목록은 비우고 missing 전체/targets_status=unusable로 보고합니다. 두 목표를 못 찾으면 해당 제한·오류와 미확보 목록을 보존하고 정지하며 자동 재시도·pose 추출·학습으로 이어가지 않습니다.

Counterpart 준비와 최초 metadata를 합친 명시 상한은 body64MiB+128KiB(64.125MiB), 협력적300+30초,768+8요청입니다. 인증/전체wire/OS hard deadline과 ZIP 비용은 별도입니다. 2026-10-17 실제 준비8,356,024bytes/29.116846초의 단순 비례는 약234초지만 더 넓은 원본의 처리량·성공·실행 시간을 보장하지 않습니다. Metadata32KiB 사전 예약 때문에 body 상한을 전부 채우지 않을 수 있고 동시 상한이면 ranges_limit도 가능합니다. Index/resume가 아니며 새 실행은 byte0부터 다시 읽습니다.

이전9필드 JOB는 profile=None/exact_targets=[]로 정규화하며 기존 smoke 이하 예산 정책과 이전 effective fingerprint를 유지합니다. 이 경로는 명명한 고정 preset으로 오표기하지 않고 `custom_limits`와 실제 limits를 보고합니다. 새11필드 JOB의 profile/targets/budget 변경도 frozen preflight와 다르면 인증 전에 거부합니다.

### Pose 전용 metadata 밀도 정책

`poseCounterpart64MiB_v1`은 [고정 JSON](../configs/archive_limits_poseCounterpart64MiB_v1.json)과 정확히 일치해야 합니다. 신규 실행은 kind=pose, mode=extract, purpose=targeted, 같은 literal exact_targets/includes2개를 강제합니다. Review는 원본/인증 요청 없이 가능합니다. Body64MiB/300초/768요청·Range256KiB·내부IO64KiB·개별파일4MiB·disk16MiB·저장2개는 그대로이며, **header10000·누적 tar metadata1MiB·해제512MiB**만 pose 전용으로 설정합니다. 기존 smoke/RGB-counterpart와 legacy 상한은 변경하지 않습니다.

직전 실제 pose 실행은 압축7.5MiB에서 header1000, tar metadata62,591bytes, 해제42,045,585bytes에 도달했습니다. 이를64MiB까지 단순 비례하면 대략 header8534·metadata534kB·해제342MiB입니다. 새 상한은 이 추정보다 여유가 있지만 분포/압축률/목표 위치를 모르므로 독립 상한 중 먼저 닿는 조건에서 종료하며 성공을 보장하지 않습니다. 이전 prefix도 byte0부터 다시 읽습니다. 최초 Drive metadata128KiB/30초는 별도이고 총 비용 제외 범위는 기존과 같습니다.

누적 tar metadata1MiB와 **단일 GNU/PAX payload64KiB**를 분리합니다. 새 pose profile에서 큰 선언은 body 수집 전에 `metadata_entry_bytes_limit`로 중단하며 report/preflight의 metadata_entry_limit_bytes에65536을 기록합니다. 기존 profile/custom은 종전 누적 상한과 stop_reason을 유지합니다. 개별 payload는 최대64KiB의 chunks/join과 PAX 문자열/필드로 일시 복사되며, IO/해제 출력은64KiB 단위입니다. 누적1MiB 전체를 한 metadata 버퍼로 보관하지 않습니다.

Case-insensitive 중복 검사용 seen set은 max_members 이내인 최대10000경로로 제한됩니다. UTF-8 경로 원문만으로 최대 약10MiB이며 Python 문자열·casefold·set·zlib·전송 buffer·allocator 비용은 추가입니다. 이는 disk16MiB와 다른 메모리 범위이며 **프로세스 RSS 하드 상한을 보장하지 않습니다.** 전체512MiB 해제 결과를 RAM/디스크에 모으지 않고 skip payload도 스트리밍으로 소비합니다.

관측목록128개는 유지하고 별도 `last_regular_observation` 한 행과 `regular_members_seen`만 갱신합니다. Last 행에는 마지막으로 안전하게 파싱한 regular header의 실제 ordinal/name/size·선택·payload 완료/저장 상태를 기록하며, 아직 읽지 않은 이후 경로나 정확한 파일 seek offset을 만들지 않습니다. Scan은128개 목록 cap에서 중단하지만 다음에 닿은 regular header 한 행은 payload 미완료로 표시할 수 있습니다. Source/CRC/parser/후검사 실패 시 **last 행에도 동일한 전체 member_observation_trust=untrusted_failed_source_or_parser**를 적용합니다. 개별 payload_complete=true만으로 사용할 수 없습니다.

Last 한 행의 JSON escape 최악과 extraction/pilot 양쪽 기록 공간을 사전 disk reserve에 더합니다. 목록을10000개로 늘리거나 본문/모든 path를 보고서에 복사하지 않습니다. Targets 완료 후 즉시 중단·partial 제거·원본 전후 검사·파일 hash/ZIP 판정은 기존 계약과 같습니다. 목표 pose가 아직 없고 기존 오른쪽 사용자 ZIP도 미회수여서 현재 관절·좌표·좌우 영상 정합성 검증은 할 수 없습니다. Frame1/10은 연속clip이 아니며 학습은 시작하지 않습니다.

## 작업 상한과 측정 범위

`--limits`는 [archive_limits_smoke.json](../configs/archive_limits_smoke.json)처럼 필드를 명시한 JSON입니다. 새 `media_range_bytes`만 생략 가능하며 생략 시256KiB를 사용하므로 기존 limits JSON도 읽습니다. 아래는 작은 개발용 기본값이며 Colab 실측으로 정한 운영 예산은 아닙니다.

| 필드 | 기본값 | 적용 범위 |
| --- | --- | --- |
| max_transfer_bytes | 8 MiB | 반환된 source 응답 body/로컬 읽기 + Drive metadata body 누적 |
| max_elapsed_seconds | 60 s | 준비코어의 요청·읽기·해제·쓰기 사이 협력적 deadline |
| request_timeout_seconds | 5 s | HTTP 요청 timeout; 현재 남은 시간보다 길게 설정하지 않음 |
| io_chunk_bytes | 64 KiB | 개별 read1/write·local 읽기와 zlib max_length |
| media_range_bytes | 256 KiB | HTTP/Drive 한 Range의 최대 반환 buffer. 마지막 Range는 잔여 예산에 맞게 축소 |
| max_decompressed_bytes | 32 MiB | 반환된 tar stream bytes 누적; skip 파일·padding·metadata 포함 |
| max_file_bytes | 4 MiB | 선택 여부와 관계없는 regular member 선언 크기 |
| max_disk_bytes | 16 MiB | 선택 파일(완료+현재 임시 파일)의 logical bytes와 report 예약량 |
| max_files | 32 | 저장한 완전 regular file 수 |
| max_members | 1000 | 읽은 nonzero tar header 수; metadata header 포함 |
| max_metadata_bytes | 64 KiB | GNU/PAX payload 누적; 읽기 전에 선언 크기 검사 |
| max_path_bytes | 1024 | Archive member UTF-8 경로 길이 |
| max_ranges | 128 | HTTP media + Drive metadata + local 읽기의 총 시도 수 |

Manifest 입력은 128 KiB, part 수는 최대 1000, selection은 최대 32개/각 1024 bytes입니다. Report 공간은 manifest·selection·최대 range/file record 수에 맞춰 사전 예약하고, 여유가 부족하면 추출 전에 거부합니다. 파일 크기·완료 수뿐 아니라 **현재 작성 중인 .partial도** disk 상한 안에 들어갑니다. Physical filesystem block·directory/ACL overhead와 다른 프로세스의 사용량은 logical bytes에 포함하지 않습니다.

Stats는 transfer/decompressed/payload_disk bytes, members/metadata/files/ranges, elapsed, 관측 최대 input/output chunk와 현재 파일을 포함한 peak payload bytes를 남깁니다. 새 `media_requests`, `metadata_requests`, `http_requests_total`(앞의 둘 합), `local_reads`를 함께 기록합니다. `ranges`는 이전 보고서와 같은 총 동작 카운터입니다. Source 호출을 시작한 시도도 실패 여부와 관계없이 셉니다. 준비 report는 예약량 이내의 `extraction_manifest.json`과 선택적 `pilot_report.json`이고 실제 출력 파일 bytes는 payload_disk_bytes와 이 report 크기를 합해 확인합니다. 최초 metadata bundle은 별도 단계입니다. 실패 후 partial을 제거하고 report를 남기는 정리 작업은 deadline이 지나도 수행합니다.

Drive media를 열기 전에 전후 metadata를 포함한 **3요청과 metadata 최대32KiB**를 사전 예약합니다. Metadata는 각각16KiB 이내이며 실제 수신량만 누적하므로 사용하지 않은 예약 bytes는 다음 Range에서 다시 고려합니다. 남은 body 예산에서32KiB를 뺀 크기까지만 media를 읽고, 여유가 없으면 `transfer_bytes_limit`로 중단합니다. 따라서 본문 상한을 정확히 모두 채우지 않을 수 있습니다. 원본 size/version/modifiedTime의 전후 비교는 Range마다 유지하며 후검사 중 시간·응답·원본 오류가 나면 `source_validation_pending=true`이고 후보 전체의 reuse/match를 거부합니다. 총 요청 여유가3회 미만이면 새 media를 시작하지 않습니다.

Transfer는 **이 adapter에 반환된 응답 본문** 기준입니다. HTTP/TLS header, 인증 교환, network transport 선수신·TCP 재전송은 측정하지 않습니다. 부분 body를 받은 후 오류가 나면 이미 반환된 bytes와 예외의 byte partial을 누적하며 retry로 초기화하지 않습니다. 숨은 transport/auth 교환까지 포함한 전체 wire 바이트의 하드 상한은 아닙니다.

Elapsed는 DNS/auth/OS syscall을 강제 중단하는 전체 프로세스 hard timeout이 아닙니다. URL open 및 read에는 남은 시간을 제한하고 slow/trickle 응답을 검사했지만 DNS나 접근 불가능한 socket·disk syscall은 deadline을 넘겨 반환할 수 있습니다. 준비 전 인증·notebook 업로드 비용도 별도입니다.

Input/output/metadata buffer와 path/record 개수에 상한을 두고 `zlib.decompress(..., max_length=...)`를 사용합니다. HTTP 반환/pending 입력은 최대256KiB이며 read1·해제 출력은64KiB입니다. Bytearray→bytes 변환 등의 일시 복사도 있어 전체 메모리가64KiB/256KiB로 제한되는 것은 아닙니다. zlib 내부 상태, Python object/allocator, HTTP/TLS buffer, 라이브러리·프로세스 RSS는 별도이므로 **임의 프로세스 RAM 상한을 보장하지 않습니다.** Gzip의 중간 조각은 독립 해제하지 않으며 restart state를 만들지 않습니다. [Python zlib streaming API](https://docs.python.org/3.12/library/zlib.html)

## 추출 파일과 신뢰 범위

저장 경로는 `output/files/<archive member>`입니다. Absolute/상위 경로/역슬래시/drive·ADS 경로, 제어 문자, Windows reserved name, link·special file과 case-insensitive 중복 member를 거부합니다. 만난 tar header/경로는 선택 밖이어도 검사합니다. GNU/PAX metadata도 크기와 경로를 검사하며 sparse·link metadata를 허용하지 않습니다.

선택 파일의 payload와 tar padding을 모두 읽은 뒤에만 완료 파일을 게시합니다. Windows는 기존 destination을 거부하는 rename, POSIX는 no-overwrite hard link를 사용합니다. 기존 output 전체를 재사용/덮어쓰기하며 추출하지 않습니다. 이 도구는 단일 writer로 쓰며 적대적인 로컬 프로세스의 동시 경로 교체를 막는 OS sandbox를 제공하지 않습니다. 잘린 payload/padding은 `.partial`을 삭제하고 완전 목록에 넣지 않습니다.

Report는 archive 원본 계약/hash, extractor version·파일별 code hash, selection, 읽은 part/range 및 수신 구간 SHA-256, 저장 파일의 SHA-256·크기, stop_reason/failure_class, 잘린 active member와 통계를 남깁니다. Source 구간 hash는 읽은 그 구간만의 hash이고 전체 원본 hash가 아닙니다. Source identity 검사는 출처 인증이나 알려진 공식 전체 SHA 대조를 대신하지 않습니다.

Gzip EOF/CRC와 tar end block은 별도입니다. Tar의 두 종료 block을 만나도 bounded로 나머지 zero padding과 gzip trailer를 읽기 전에는 전체 검증으로 보고하지 않습니다. 한 stream의 EOF/CRC, tar 종료, 모든 선언 source bytes를 확인하고 total_parts 전체가 있으며 official_prefix_local이 아닐 때만 `whole_archive_integrity_verified=true`입니다. 이는 **선언한 공급 archive의 gzip/tar 무결성**이며 공식 배포본의 인증이나 dataset 내용 검증이 아닙니다. Official prefix 로컬 재생은 어떤 경우에도 이 값이 false입니다.

연속 prefix의 정상 부족과 예산 중단은 완전 파일 후보를 남길 수 있습니다. CRC 오류·원본 변경·안전하지 않은 member·잘못된 parser/HTTP 응답 등은 전체 추출의 inspection 후보 사용을 거부합니다. 완전 gzip인데 tar가 자체적으로 잘린 경우도 invalid로 분류합니다. 이전에 남긴 파일이 있더라도 그 상태의 match/reuse는 실패합니다. 오류를 삭제하거나 usable 플래그를 임의로 바꿔 학습하지 않습니다.

## Payload 저장 없는 scan과 bounded 관측 목록

```sh
python -m rewind_archive scan --manifest outputs/archive_fixture/rgb_manifest.json --limits configs/archive_limits_smoke.json --output outputs/archive_scan
```

`scan`은 `run_pilot(..., mode="scan")`, `run_metadata_pilot(..., mode="scan")`, `prepare(..., mode="scan")`으로도 실행합니다. 기존 함수의 기본 mode는 `extract`이며 기존 CLI·선택 검사·원본 변경 검사·예산을 유지합니다. Scan은 `includes=[]`를 허용하고 payload 저장0입니다. 선택 glob이 있어도 `selected`만 기록하며 저장하지 않습니다. 선행 payload를 읽고 해제하는 전송/시간 비용은 그대로 발생합니다.

`extraction_manifest.json.member_observations`에는 도달해 검사한 regular header를 최대 `min(128,max_members)`개 기록합니다. `header_ordinal`은 metadata/directory를 포함한 nonzero tar header 순서, `regular_ordinal`은 regular 관측 순서입니다. `member`, `size_bytes`는 검증한 header의 선언이며 `selected`, `payload_complete`, `stored`를 구분합니다. `payload_complete`는 payload+padding을 소비한 상태이고 내용 검증이나 파일 저장의 증거가 아닙니다. `payload_bytes_returned`는 reader가 callback에 반환한 payload bytes이며 parser 내부 선수신량/정확한 seek offset이 아닙니다. Range offset이나 `decompressed_bytes`를 파일 위치 인덱스로 변환하지 않습니다.

저장하지 않는 member도 관측하고, payload/padding이 잘렸거나 파일 크기/예산/개수 제한에 걸렸으면 `payload_complete=false`로 남깁니다. Scan은 기록 cap 뒤 새 regular header에 닿으면 `observed_members_limit`로 중단하고 `member_observation_cap_reached=true`를 표시합니다. 기존 extract는 이 cap에서 목록만 자르고 원래 header/byte/시간 상한 안에서 선택 추출을 계속합니다. 이 목록은 source 전체 목록이나 gzip seek/restart state가 아니며 `full_source_listing_claimed=false`를 유지합니다. Tar/원본 내부 순서를 frame 번호나 다른 archive 순서로 가정하지 않습니다.

경로 UTF-8 bytes는 기존 max_path_bytes를 지키고, JSON escape 최악2배+record512bytes를 항목당 disk reserve에 추가합니다. 최대 record 수는 사전 고정하며 목록을 pilot report에 복사하지 않습니다. `pilot_report.json`은 관측 수·trust만 요약합니다. Payload+두 report 합은 기존16MiB 범위이며 workflow job/최초 metadata 및 ZIP은 별도 기록입니다.

정상 bounded 중단의 `member_observations_usable=true`, `member_observation_trust="bounded_observation_only"`는 관측 prefix만 진단 가능하다는 뜻입니다. Source 변경·후검사 미완료·unsafe 경로/link·CRC/parser 실패는 전체 목록을 `untrusted_failed_source_or_parser`로 표시합니다. 개별 행의 payload_complete가 true여도 이 전체 실패 상태를 우회하지 않습니다. Scan의 파일0개/`sample_acquired=false`는 의도한 동작이며 sample 확보 성공과 구분합니다.

## 매칭과 같은 런타임 재사용

UnrealEgo 이름 규칙은 관측한 공식 prefix에서 다음과 같습니다. 확장자가 다른 파일이나 다른 규칙은 무시하지 않고 matching issue로 기록합니다.

```text
<sequence>/fisheye_final_image/camera_left/final_<N>.png
<sequence>/fisheye_final_image/camera_right/final_<N>.png
<sequence>/json/frame_<N>.json
```

추출 archive 순서와 관계없이 sequence·숫자 frame·view로 매칭합니다. Report는 observed frame·연속 구간·관측 사이 gap·좌우/pose 누락과 완전 triplet 구간을 기록합니다. 앞뒤 미관측 프레임이나 sequence 전체의 완전성은 unknown입니다. frame 10을 2보다 앞에 lexical 정렬하지 않습니다. 중복 frame/view와 dataset version 불일치는 matching_valid=false입니다.

`matching_valid`는 이름 규격과 중복 검사 결과이며 **matched_triplet_count와 다릅니다**. 0 triplet이어도 이름 오류가 없으면 valid일 수 있으므로 CLI가 matched_triplet_count/incomplete_frame_count/observed_frame_count를 함께 출력합니다. 3종 이름이 모두 맞아도 PNG/JSON 내용·calibration·시간·원본 motion/분할을 증명하지 않습니다. Split unknown과 original_motion_verification=unverified를 유지합니다.

`verify`는 extraction manifest 내용 hash, 원본 manifest 동일성, extractor code hash, 개별 완료 파일의 길이·SHA-256·안전한 경로를 확인합니다. Hash 읽기는 기록된 크기만 bounded로 수행하고 fd 변화도 검사합니다. 실제 원격 원본을 다시 다운로드하거나 source가 이후 변하지 않았음을 증명하지 않습니다. 같은 런타임에서 검사된 추출본은 반복 학습 전에 재사용할 수 있지만 M1 입력 준비 조건은 별도입니다. 코드가 바뀌거나 임시 추출본이 소실되면 새 준비를 검토합니다. 원본 압축 조각은 유지합니다.

## 실행한 검증과 남은 일

### 2026-10-19 요청 실패 진단 검증

전체189개 회귀(기존175+진단14)를145.440초에 통과했습니다. 집중14개는 초기/pre/media/post 요청·wrapping/body/validation/close 단계, partial bytes/hash, 본문+close 이중 실패의 첫 실패 보존, KeyboardInterrupt 보존, 정상/누락 필드 legacy review와 ZIP을 포함합니다. 비밀 sentinel·str/repr 호출 금지·임의 enum/key 제거와 최대 길이 진단/기존 report reserve도 검사했습니다. 준비 예산·timeout·자동 retry0·목표 경로·pending/unusable 판정은 유지했습니다.

별도 검사에서 설치된 requests2.34.2·urllib3 2.8.0·google-auth2.62.0의 실제 예외 타입23조건을 skip 없이 통과했습니다. Connect/read timeout·TLS·인증/transport와 generic wrapper/위조된 동명 클래스/unknown을 구분했으며 호출은 mock1회뿐입니다. 기존 실제 전송 라이브러리의 chunked/후검사15조건도 익명 localhost에서 통과했습니다. 이 두 검사는 전체189개와 합산하지 않습니다.

Fresh source ZIP의 API5 notebook14셀/7코드 셀을 위에서 아래로 모의 실행했습니다. 사전94개, 기본 인증0·stale/중복 차단·일반 pose 목표 완료·인증 setup timeout·초기 metadata RefreshError·media ReadTimeout과 report→review→ZIP 전달을 확인했습니다. 원시 예외 원문을 결과에 남기지 않았고 과거 진단 없는 report도 원인을 만들어내지 않았습니다. 사용자/Google 실제 인증·Drive 원본 요청은0이며 새 버전의 실제 실패 유형은 아직 관측하지 않았습니다.

근거는 Git 제외 `outputs/request_diagnostics_20261019/developer/`의 unittest.log, diagnostics_final.log, actual_types.json, chunked/transport_report.json, notebook_review.json·notebook.log에 보존합니다. 고정 사용자 배포 ZIP은 커밋 후 오케스트레이션에서 별도 제작·검토합니다. 이번 수정은 pose 확보나 좌표/영상 검증·학습 완료가 아닙니다.

### 2026-10-19 Pose 밀도 정책과 bounded 진단 검증

전체 회귀175개를1회 실행해39.727초에 통과했고, 신규 pose-profile10개와 오케스트레이션 독립10개도 통과했습니다. 실제 PAX+작은 JSON600개 뒤 목표2개 fixture는 기존 정책에서 members_limit, 새 정책에서 targets_complete/파일 hash 검증을 재현했습니다. 별도 누적 metadata fixture는 기존64KiB 중단과 새1MiB 범위의 완료를 대조했습니다. 단일 metadata64KiB는 payload 할당 전에 거부하고, 누적1MiB·header10000은 각각 작은 압축 fixture에서 독립 중단을 확인했습니다. 해제512MiB 경계는 이미 소비한 prefix를 예산 상태로 표현하고 실제 zlib의 마지막64KiB를 처리하는 bounded mock으로 검증했으며 실제512MiB 파일 생성/다운로드·성능 시험은 하지 않았습니다.

추가 회귀는 잘못된 kind/profile/limits/목표 거부, 부분 목표 보존, 원본 변경 시 last 행까지 untrusted, 관측 cap 뒤 extract 지속·scan cap 유지, report reserve·ZIP 상한을 포함합니다. 기존 긴 경로/JSON escape reserve 검사도 전체 회귀에서 통과했습니다. Fresh source ZIP의 API4 notebook7코드 셀을 모의 Colab으로 위에서 아래 실행해80개 사전 검사, 새 pose profile/preflight/한 번 실행/목표 완료/last 행/review/ZIP을 확인했습니다. 기본 인증0·stale/중복 차단·실패 job-only ZIP도 유지했습니다.

오케스트레이션 독립 PAX700개 fixture는 기존 header1000·metadata60,500bytes에서0파일 중단, 새 정책은 header1402·metadata84,700bytes에서2파일 완료와 hash 일치를 확인했습니다. 모든 검증은 로컬 합성/모의/localhost 범위이며 실제 Google·Drive 원본 요청0입니다. 근거는 Git 제외 `outputs/pose_policy_20261019/developer/`와 `root/`에 보존합니다. 새 정책의 실제 pose 확보와 좌표/영상 정합성은 미검증이며 고정 revision 사용자 테스트와 구분합니다.

### 2026-10-17 고정 counterpart 프로파일·목표 완료 중단

오케스트레이션의 전체165개 회귀(38.495초)와 마지막 review 호환 보완 후 관련 target10개를 통과했습니다. 새10개는 프로파일 불변/JSON 일치·변조 거부, legacy9필드 정규화와 이전 fingerprint·stale 검사,8MiB 뒤 목표 도달, 첫 파일만 보존/둘째 잘림, 두 목표 완료 후 추가header/source.read0, 원본 변경의 usable 취소, 관측 cap 뒤 추출 지속, report reserve·ZIP, pilot hash 실패 시 완료 판정 거부를 포함합니다. 독립9MiB 선행 fixture에서도 smoke는0파일, counterpart는2파일과 targets_complete를 확인했습니다. 합성/모의 검증이며 실제 PNG/JSON 학습 준비를 증명하지 않습니다.

최종 fresh source ZIP의 API3 notebook14셀/7코드 셀을 모의 Colab으로 위에서 아래 실행했습니다. 사전70개·기본 인증0·profile/exact_targets 설정과 완료 중단·stale/중복 인증 전 차단·실패 job-only ZIP·기존 review·ZIP 재다운로드를 통과했습니다. 최초 개발용 임시 ZIP 검사에서 untracked profile JSON 누락을 발견해 ZIP 작성 도구를 정정했고 최종 source ZIP에서는70개 모두 통과했습니다. 실제 사용자 배포 ZIP은 커밋 후 별도 검토합니다. 근거는 Git 제외 `outputs/counterpart_prepare_20261017/developer/`와 오케스트레이션 `root/`에 보존합니다. 실제 Google 인증·원본 요청은0입니다.

### 2026-10-16 단방향 notebook·scan 검증

오케스트레이션의 전체155개 회귀와 디벨로퍼의 새 scan/workflow14개·기존 archive24개를 통과했습니다. Fresh source ZIP의7개 code cell을 위에서 아래로 실행하여 배포ZIP 내부 파일명 안내, 기본 인증0, 단일 설정 scan 관측3/저장0, stale 설정 인증 전 차단, 동일ID/fingerprint 재실행 차단, 새ID+변경reason, 인증 실패 job-only ZIP, 기존 결과 review와 ZIP 재다운로드를 모의 확인했습니다. 별도 선택/plan 업로드는0회입니다. 실제 Colab 인증·새 원본 읽기 결과가 아닙니다.

실제 requests/urllib3/AnonymousCredentials loopback scan6조건에서 기본8MiB 예산·96요청·payload저장0, 후검사 원본 변경·HTTP200/401/403/429 실패 시 관측 untrusted를 검사했습니다. 당시 새 notebook의 실제 Drive 실행은 남아 있었으며 이후10월17일 scan 결과를 로드맵에 별도로 확인했습니다. 근거는 Git 제외 `outputs/archive_notebook_rewrite_20261016/developer/`와 오케스트레이션 `root/`에 보존합니다.

### 2026-10-16 요청 상한 원인과 다음 sample 확보 절차

사용자 RGB 보고서3개는 모두 동일한 byte0 시작 입력에서 media43회×64KiB=2,818,048bytes를 수신했습니다. 당시 `ranges`는 media와 전후 metadata를 같은128회 상한에 더했고, metadata85회+media43회=128에서 마지막 후검사가 막혔습니다. 마지막64KiB는 이미 전송했지만 해제기로 반환되지 않았습니다. 별도1개는 `range_status_not_206`으로 실패했으며 당시 HTTP status는 기록되지 않아200/403/429 등을 확정할 수 없습니다. 이는 원인 대조이며 이 수정본의 실제 Drive 성공 검증이 아닙니다.

새 기본값은 요청 상한을 올리지 않고 HTTP Range를256KiB로 묶습니다. 로컬 실제 requests/urllib3/AuthorizedSession+AnonymousCredentials loopback에서 기본8MiB/60초/128회로 media32+metadata64=96회, 본문8,355,952bytes 후 `transfer_bytes_limit`로 중단했고 완전 합성 파일3개의 hash를 확인했습니다. 마지막 Range 축소·64KiB 해제·200/401/403/429 body 미읽기·후검사 원본 변경 거부·예약 여유 부족 시 네트워크0을 확인했습니다. 실제 Google 요청·인증·새 RGB sample 확보는 포함하지 않습니다.

이전 요청 효율 개선 묶음의 전체 unittest는141개(기존133+planning/예약8개)이며 기존 실제 라이브러리 chunked/오류15조건과 신규 요청 예산6조건을 별도로 통과했습니다. Fresh source ZIP의 notebook10개 code cell 문법·빈 출력, 기본 인증0,36glob 인증 전 차단, targeted4개/discovery2개 합성 파일, 같은 job metadata/media 재실행 차단, 명시적 새 job·실패 결과 폐기·JSON 입력·ZIP 재다운로드·원본 미접근 대응 제안을 모의 검증했습니다. 이 notebook 검증은 실제 Colab 인증이 아닙니다. 근거는 Git 제외 `outputs/archive_request_review_20261016/developer/`에 보관합니다.

이전10-code-cell notebook의 discovery/새 job 셀 흐름은 위 단일 JOB 방식으로 대체했습니다. 당시 다음 실제 실행은 동일8MiB에서 **저장하지 않았던 member 순서 증거를 얻는 scan 1회**였으며, 후속 결과와 더 넓은 고정 counterpart 예산 검토는 위10월17일 기록을 따릅니다. 관측 목록과 새 목적 없이 동일 prefix를 반복하지 않습니다.

결과 ZIP 셀은 현재 job/metadata/pilot·관측목록/완전 파일만 저장하며 `DOWNLOAD_RESULT_ZIP=True`로 내려받습니다. 이미 있으면 덮어쓰지 않고 다시 다운로드할 수 있습니다. ZIP에는 이후 파일을 자동 추가하지 않습니다. 최대17MiB 입력과 비슷한 별도 임시 ZIP 디스크가 필요하며 준비16MiB 통계와 구분합니다. 인증 실패로 `job.json`만 남아도 ZIP으로 보존할 수 있습니다.

```sh
python -m rewind_archive plan-counterparts --extraction outputs/discovery/pilot --max-frames 8 --output outputs/counterpart_plan.json
python tests/check_archive_request_transport.py --dependency-dir outputs/drive_pilot_review/transport_deps --output outputs/request_transport_check
```

`plan-counterparts`는 기존 code/manifest/완전 파일 hash를 검사하고 새 출력 파일만 생성합니다. `member_globs_by_kind`의 빈 목록은 해당 종류의 다음 선택이 없다는 뜻이며 빈 목록으로 pilot을 실행하지 않습니다. 서로 다른 dataset version·중복 view·변경/잘린 파일은 거부합니다. 관측 이름 대응은 split/content/시간 정합성이나 학습 준비 완료를 뜻하지 않습니다.

2026-10-14 로컬 검증:

- Archive 합성/loopback 테스트 **24개** 통과(최신 6.121 s). Split 경계, incomplete prefix·잘린 header/payload/padding, gzip CRC·종료 범위, GNU LongName/PAX, path/link/special/duplicate, 전송·해제·파일·metadata·시간·disk/range 상한, reuse hash와 숫자 매칭을 포함합니다.
- 실제 로컬 HTTP 서버의 206, 200 Range 무시, 잘못된 Content-Range/Length, 원본 ETag 변경, redirect, short body·slow/trickle/stall을 검사했습니다. 초기 Drive adapter 계약은 **모의** 검사였고 후속 실제 전송 라이브러리의 loopback 검증은 아래에 분리합니다.
- 오케스트레이션의 **기존 공식 서버 prefix 로컬 재생**: RGB 2 MiB → PNG 2개, pose 256 KiB → JSON 16개, 이전 검사와 완료 파일 18개 SHA-256 모두 일치. 둘 다 마지막 payload 잘림, gzip EOF/CRC false, 전체 무결성/train_ready false, partial 잔여 0. 이름은 서로 다른 2개 sequence라 완전 triplet **0개**입니다. 이번 새 도구 검증에서 공식 서버를 다시 다운로드한 결과가 아닙니다. 내부 실행 근거는 Git 제외 `outputs/archive_next_step/review_replay.json` 등에 보관합니다.
- 초기 준비 notebook은 5개 code cell의 별도 ZIP/새 임시 checkout 실행·문법·빈 출력을 검사했습니다(`outputs/archive_next_step/review_notebook.json`). 후속 notebook은 **7개** code cell로 metadata/pilot을 나눴고 fresh ZIP의 기본 auth0·합성 triplet4, mock metadata/pilot pose4·선택 변경 차단·실패 재실행 결과 폐기·선택 JSON 업로드를 확인했습니다. Notebook 사전 테스트는 archive24+metadata10, 총34개를 실행했습니다. 실제 Colab 업로드·Google 인증/API가 아니며 후속 근거는 `outputs/drive_pilot_review/notebook_review.json`입니다.

앞선 metadata 묶음은 **93개** 회귀와 실제 requests2.34.2/urllib3 2.8.0/google-auth2.62.0의 AnonymousCredentials+loopback을 검사했습니다(`outputs/drive_pilot_review/`). Chunked 수정 후 최신 전체는 **97개** 모두 통과했습니다(오케스트레이션17.470s). 사전 테스트는 archive24+metadata14=38개입니다. Fresh ZIP notebook의 기본 auth0·합성triplet4·mock metadata/pilot·선택변경·실패결과폐기·JSON업로드와7codecell을 확인했습니다(`outputs/chunked_metadata_review/notebook_review.json`).

실제 라이브러리의 별도 검사 **15조건**은 정상chunked→manifest→pilot 전후metadata, response/global cap·slow·잘린chunk/JSON·JSON완료뒤종료chunk 누락·HTTP인증상태·encoding·CL/TE혼재·media조건유지·post-media 오류를 통과했습니다. 두report의 byte합도 reserve안입니다. 별도 driver는 optional dependency를 skip하지 않고 명시적 환경에서 실행하며 unittest97개와 합산하지 않습니다. Google요청/인증0이고 근거는 `outputs/chunked_metadata_dev_final/transport_report.json`·오케스트레이션의 `outputs/chunked_metadata_review/pipeline_review.json`입니다.

```sh
python tests/check_chunked_transport.py --dependency-dir outputs/drive_pilot_review/transport_deps --output outputs/chunked_transport_check
```

사용자 Colab의 실제 관측은 status 200, Content-Length 없음, Transfer-Encoding chunked, Content-Encoding 없음, Content-Type JSON UTF-8과 기존 `metadata_protocol` 실패입니다. 이 최초 실패 관측과 별도로, 이후 사용자 보고서에서 수정본의 manifest 생성·pose 32개 추출·verify 성공을 확인했습니다. 수치와 검토 한계는 [로드맵](ROADMAP.md)에 기록합니다. 최신 소스 ZIP을 사용하고, 이미 import한 구모듈을 계속 쓰지 않도록 [Colab 재시작·재업로드 절차](COLAB_SETUP.md)를 따릅니다.

앞선 공식 prefix 로컬 재생은 PNG2·JSON16의 과거 SHA 일치·incomplete_contiguous_prefix·verify·triplet0/incomplete18 기록을 유지합니다(`outputs/archive_next_step/review_replay_final.json`·`review_matching_final.json`). 후속 ArchVis pose 시험에서는 실제 Drive metadata·부분 읽기·준비 비용을 한 차례 확인했습니다. 전체 blob 무결성·운영 상한 확정, 실제 좌우 RGB+pose clip과 공식 split·원본 motion은 미검증입니다. Colab·GPU 성능을 보장하지 않습니다. [로드맵](ROADMAP.md), [M0 도구](M0_JSON_TOOLS.md), [M1 입력 계약](M1_BASELINE.md), [Colab 안내](COLAB_SETUP.md)를 따릅니다.
