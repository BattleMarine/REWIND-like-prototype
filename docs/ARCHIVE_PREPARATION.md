# 분할 압축 원본의 제한된 순차 준비

2026-10-14 · `rewind_archive`의 검토 가능한 첫 구현 묶음입니다. 분할 조각을 000부터 읽어 하나의 gzip을 순차 해제하고 tar의 선택한 완전한 regular file을 남깁니다. 전체 원본 다운로드·결합 파일·전체 추출본을 만들지 않습니다. 실제 사용자 Drive와 Colab 실행은 미검증입니다.

## 현재 가능한 범위

Python 3.10 이상 표준 라이브러리로 local file·HTTP Range 소스를 읽습니다. Colab `AuthorizedSession`을 주입하는 Drive API adapter도 분리했으나, 실제 google-auth/requests/urllib3 stack의 접속·인증·응답은 실행하지 않았습니다. GPU는 준비 작업에 필요하지 않습니다. Archive 준비 notebook은 [archive_prepare.ipynb](../notebooks/archive_prepare.ipynb), M1 합성 학습 notebook은 별도 [m1_baseline.ipynb](../notebooks/m1_baseline.ipynb)입니다.

현재 기능은 일반 tar 도구 전체를 대체하지 않습니다. 단일 gzip stream의 regular file·directory, GNU LongName과 제한된 PAX metadata를 지원합니다. Concatenated gzip, sparse file, link·device·FIFO 등의 special file은 거부합니다. gzip restart/index와 재포장, 실제 RGB loader·좌표 변환·EfficientNet 학습은 포함하지 않습니다.

`inspection_candidates_usable=true`는 수치·영상 검사에 사용할 수 있는 완전 파일 후보를 뜻합니다. 학습 준비 완료가 아닙니다. 추출·verify·matching 결과의 **train_ready는 항상 false**입니다. M0/M1의 관절·단위·좌표/calibration·timestamp·split·원본 motion·RGB/pose 내용 정합성 검증을 유지합니다.

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

`extract`는 report의 inspection 후보 사용 여부가 true이면 exit 0, 손상·원본 변경·안전하지 않은 입력이면 exit 1입니다. **Exit 0은 전체 완료가 아닙니다.** 반드시 stop_reason·failure_class·CRC·완전 파일 수를 읽습니다. `verify`는 manifest·개별 hash가 맞을 때 exit 0입니다. `match`는 이름 계약 오류·중복·dataset version 불일치가 없을 때 exit 0이며 triplet 개수는 별도입니다. 설정·입출력 오류는 argparse exit 2입니다.

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

Strong ETag 또는 Last-Modified 중 최소 하나가 필요합니다. 제공한 모든 validator가 응답과 일치해야 하며 If-Match/If-Unmodified-Since도 보냅니다. HTTP **206**, 정확한 Content-Range(start/end/전체 size), Content-Length와 identity encoding을 검사한 후 요청한 구간만 읽습니다. Range를 무시한 200이나 잘못된 header는 본문을 읽지 않습니다. Content-Length가 없는 chunked 응답은 현재 지원하지 않습니다. 자동 재시도는 구현하지 않습니다.

## Drive runtime adapter

Drive part source/identity는 아래 구조입니다. size/version/modifiedTime은 인증된 Drive `files.get`에서 확인하며 목록 도구가 반환하지 않은 값을 추측하지 않습니다. 원격 part의 size_bytes는 전체 blob 크기입니다. 로컬에 저장한 prefix 크기를 원격 전체 크기로 넣으면 Content-Range 검사가 실패합니다.

```json
{
  "source": {"type": "drive", "file_id": "USER_SELECTED_FILE_ID"},
  "identity": {"version": "ACTUAL_VERSION", "modified_time": "ACTUAL_MODIFIED_TIME"}
}
```

CLI는 새 인증을 시작하지 않습니다. 런타임에서는 `prepare(..., authorized_session=session)`에 사용자가 인증한 session을 주입합니다. Notebook은 실제 경로를 기본 비활성으로 두고, manifest·선택 glob이 지정된 후에만 `google.colab.auth.authenticate_user()`와 `google.auth.default()`를 호출하도록 작성했습니다. 이 셀의 실제 인증·접속은 미검증입니다. Mount된 Drive 파일 읽기와 Drive API 부분 읽기는 다릅니다.

Drive adapter는 media Range 요청 전후 metadata의 size/version/modifiedTime을 대조합니다. Metadata는 Content-Length 1~16384가 있는 HTTP 200 JSON object만, media는 정확한 HTTP 206만 허용합니다. Null/list/string 등 잘못된 metadata는 report를 남기며 거부합니다. 자동 auth 재요청을 피하려고 notebook의 `AuthorizedSession(max_refresh_attempts=0)`와 `HTTPAdapter(max_retries=0)`를 사용합니다. Adapter도 알려진 session의 refresh retry 정책이 0이 아니면 거부합니다.

응답 body는 urllib3의 `raw.read1(size, decode_content=False)`를 요구하며 미지원 API에 `raw.read(size)`로 우회하지 않습니다. 이는 단일 read 후 deadline을 확인하기 위한 계약입니다. 접근 가능한 socket은 잔여 시간으로 timeout을 줄이고, 접근 불가하면 session의 요청 timeout과 read1 사이 협력적 검사를 사용합니다. 로컬에서 이 adapter는 모의 session으로만 검사했습니다. [urllib3 read1 API](https://urllib3.readthedocs.io/en/stable/reference/urllib3.response.html#urllib3.response.HTTPResponse.read1), [google-auth AuthorizedSession](https://google-auth.readthedocs.io/en/latest/reference/google.auth.transport.requests.html#google.auth.transport.requests.AuthorizedSession), [Drive 부분 다운로드](https://developers.google.com/workspace/drive/api/guides/manage-downloads#partial_download)

최초 인증과 credential refresh는 prepare 밖에서 수행하며 그 비용은 준비 통계와 별도입니다. Credential은 session에만 존재하고 manifest/code/report/log에 저장·출력하지 않습니다. 준비 중 credential의 before_request refresh나 transport 자체 동작이 추가될 수 있어 실제 Colab에서 지연·응답 조건·API를 별도로 검증해야 합니다. 신규 인증·추가 설치·대용량 읽기를 이번 로컬 검증에서 수행하지 않았습니다.

## 작업 상한과 측정 범위

`--limits`는 [archive_limits_smoke.json](../configs/archive_limits_smoke.json)처럼 **모든 필드가 명시된** JSON입니다. 아래는 작은 개발용 기본값이며 Colab 실측으로 정한 운영 예산은 아닙니다.

| 필드 | 기본값 | 적용 범위 |
| --- | --- | --- |
| max_transfer_bytes | 8 MiB | 반환된 source 응답 body/로컬 읽기 + Drive metadata body 누적 |
| max_elapsed_seconds | 60 s | 준비코어의 요청·읽기·해제·쓰기 사이 협력적 deadline |
| request_timeout_seconds | 5 s | HTTP 요청 timeout; 현재 남은 시간보다 길게 설정하지 않음 |
| io_chunk_bytes | 64 KiB | Range 단위, 개별 read/write와 zlib max_length |
| max_decompressed_bytes | 32 MiB | 반환된 tar stream bytes 누적; skip 파일·padding·metadata 포함 |
| max_file_bytes | 4 MiB | 선택 여부와 관계없는 regular member 선언 크기 |
| max_disk_bytes | 16 MiB | 선택 파일(완료+현재 임시 파일)의 logical bytes와 report 예약량 |
| max_files | 32 | 저장한 완전 regular file 수 |
| max_members | 1000 | 읽은 nonzero tar header 수; metadata header 포함 |
| max_metadata_bytes | 64 KiB | GNU/PAX payload 누적; 읽기 전에 선언 크기 검사 |
| max_path_bytes | 1024 | Archive member UTF-8 경로 길이 |
| max_ranges | 128 | Source Range 시도 + Drive metadata 요청 수 |

Manifest 입력은 128 KiB, part 수는 최대 1000, selection은 최대 32개/각 1024 bytes입니다. Report 공간은 manifest·selection·최대 range/file record 수에 맞춰 사전 예약하고, 여유가 부족하면 추출 전에 거부합니다. 파일 크기·완료 수뿐 아니라 **현재 작성 중인 .partial도** disk 상한 안에 들어갑니다. Physical filesystem block·directory/ACL overhead와 다른 프로세스의 사용량은 logical bytes에 포함하지 않습니다.

Stats는 transfer/decompressed/payload_disk bytes, members/metadata/files/ranges, elapsed, 관측 최대 input/output chunk와 현재 파일을 포함한 peak payload bytes를 남깁니다. Report는 예약량 이내의 별도 `extraction_manifest.json`이고 실제 전체 파일 bytes는 payload_disk_bytes와 report 파일 크기를 합해 확인할 수 있습니다. 실패 후 partial을 제거하고 report를 남기는 정리 작업은 deadline이 지나도 수행합니다.

Transfer는 **이 adapter에 반환된 응답 본문** 기준입니다. HTTP/TLS header, 인증 교환, network transport 선수신·TCP 재전송은 측정하지 않습니다. 부분 body를 받은 후 오류가 나면 이미 반환된 bytes와 예외의 byte partial을 누적하며 retry로 초기화하지 않습니다. 숨은 transport/auth 교환까지 포함한 전체 wire 바이트의 하드 상한은 아닙니다.

Elapsed는 DNS/auth/OS syscall을 강제 중단하는 전체 프로세스 hard timeout이 아닙니다. URL open 및 read에는 남은 시간을 제한하고 slow/trickle 응답을 검사했지만 DNS나 접근 불가능한 socket·disk syscall은 deadline을 넘겨 반환할 수 있습니다. 준비 전 인증·notebook 업로드 비용도 별도입니다.

Input/output/metadata buffer와 path/record 개수에 상한을 두고 `zlib.decompress(..., max_length=...)`를 사용합니다. zlib 내부 상태, Python object/allocator, HTTP/TLS buffer, 라이브러리·프로세스 RSS는 별도이므로 **임의 프로세스 RAM 상한을 보장하지 않습니다.** Gzip의 중간 조각은 독립 해제하지 않으며 restart state를 만들지 않습니다. [Python zlib streaming API](https://docs.python.org/3.12/library/zlib.html)

## 추출 파일과 신뢰 범위

저장 경로는 `output/files/<archive member>`입니다. Absolute/상위 경로/역슬래시/drive·ADS 경로, 제어 문자, Windows reserved name, link·special file과 case-insensitive 중복 member를 거부합니다. 만난 tar header/경로는 선택 밖이어도 검사합니다. GNU/PAX metadata도 크기와 경로를 검사하며 sparse·link metadata를 허용하지 않습니다.

선택 파일의 payload와 tar padding을 모두 읽은 뒤에만 완료 파일을 게시합니다. Windows는 기존 destination을 거부하는 rename, POSIX는 no-overwrite hard link를 사용합니다. 기존 output 전체를 재사용/덮어쓰기하며 추출하지 않습니다. 이 도구는 단일 writer로 쓰며 적대적인 로컬 프로세스의 동시 경로 교체를 막는 OS sandbox를 제공하지 않습니다. 잘린 payload/padding은 `.partial`을 삭제하고 완전 목록에 넣지 않습니다.

Report는 archive 원본 계약/hash, extractor version·파일별 code hash, selection, 읽은 part/range 및 수신 구간 SHA-256, 저장 파일의 SHA-256·크기, stop_reason/failure_class, 잘린 active member와 통계를 남깁니다. Source 구간 hash는 읽은 그 구간만의 hash이고 전체 원본 hash가 아닙니다. Source identity 검사는 출처 인증이나 알려진 공식 전체 SHA 대조를 대신하지 않습니다.

Gzip EOF/CRC와 tar end block은 별도입니다. Tar의 두 종료 block을 만나도 bounded로 나머지 zero padding과 gzip trailer를 읽기 전에는 전체 검증으로 보고하지 않습니다. 한 stream의 EOF/CRC, tar 종료, 모든 선언 source bytes를 확인하고 total_parts 전체가 있으며 official_prefix_local이 아닐 때만 `whole_archive_integrity_verified=true`입니다. 이는 **선언한 공급 archive의 gzip/tar 무결성**이며 공식 배포본의 인증이나 dataset 내용 검증이 아닙니다. Official prefix 로컬 재생은 어떤 경우에도 이 값이 false입니다.

연속 prefix의 정상 부족과 예산 중단은 완전 파일 후보를 남길 수 있습니다. CRC 오류·원본 변경·안전하지 않은 member·잘못된 parser/HTTP 응답 등은 전체 추출의 inspection 후보 사용을 거부합니다. 완전 gzip인데 tar가 자체적으로 잘린 경우도 invalid로 분류합니다. 이전에 남긴 파일이 있더라도 그 상태의 match/reuse는 실패합니다. 오류를 삭제하거나 usable 플래그를 임의로 바꿔 학습하지 않습니다.

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

2026-10-14 로컬 검증:

- Archive 합성/loopback 테스트 **24개** 통과(최신 6.121 s). Split 경계, incomplete prefix·잘린 header/payload/padding, gzip CRC·종료 범위, GNU LongName/PAX, path/link/special/duplicate, 전송·해제·파일·metadata·시간·disk/range 상한, reuse hash와 숫자 매칭을 포함합니다.
- 실제 로컬 HTTP 서버의 206, 200 Range 무시, 잘못된 Content-Range/Length, 원본 ETag 변경, redirect, short body·slow/trickle/stall을 검사했습니다. Drive adapter는 metadata root/type/version·read1·session 계약을 **모의** 검사했으며 실제 AuthorizedSession 실행이 아닙니다.
- 오케스트레이션의 **기존 공식 서버 prefix 로컬 재생**: RGB 2 MiB → PNG 2개, pose 256 KiB → JSON 16개, 이전 검사와 완료 파일 18개 SHA-256 모두 일치. 둘 다 마지막 payload 잘림, gzip EOF/CRC false, 전체 무결성/train_ready false, partial 잔여 0. 이름은 서로 다른 2개 sequence라 완전 triplet **0개**입니다. 이번 새 도구 검증에서 공식 서버를 다시 다운로드한 결과가 아닙니다. 내부 실행 근거는 Git 제외 `outputs/archive_next_step/review_replay.json` 등에 보관합니다.
- 준비 notebook은 별도 ZIP/새 임시 checkout에서 source 업로드를 로컬 mock하여 추출·커널 import를 확인했습니다. 환경·합성 실행·기본 disabled Drive 셀을 실행해 RGB 8+pose4→triplet4/train_ready=false를 확인했고 모든 5개 code cell 문법·빈 출력 상태를 검사했습니다. Test 셀은 앞선 전체 83개 회귀를 근거로 중복 실행하지 않았습니다. 실제 Colab 업로드·인증/API가 아니며 근거는 `outputs/archive_next_step/review_notebook.json`입니다.

최신 전체 회귀는 Python 3.12.14 / torch 2.7.1+cpu에서 **83개**(기존 59 + archive 24) 모두 통과했습니다(오케스트레이션 독립 실행 11.113 s, loopback 허용 환경). 최종 공식 prefix 재생도 PNG 2·JSON 16의 과거 SHA 일치, incomplete_contiguous_prefix, verify 통과, triplet 0/incomplete frame 18을 확인했습니다. 근거는 `outputs/archive_next_step/review_replay_final.json`·`review_matching_final.json`입니다. 아직 실제 사용자 Drive blob Range·원본 version 검사·Colab 인증/API/준비 비용·운영 상한, 실제 좌우 RGB+pose clip, 공식 split·원본 motion 검증은 남아 있습니다. Colab·GPU 성능을 보장하지 않습니다. [로드맵](ROADMAP.md), [M0 도구](M0_JSON_TOOLS.md), [M1 입력 계약](M1_BASELINE.md), [Colab 안내](COLAB_SETUP.md)를 따릅니다.
