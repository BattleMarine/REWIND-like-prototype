# ArchVis pose 한 장면의 파일명 선택 추출 시험

> **보존 기록 · 2026-11-05 정리:** 2026-11-04 사용자 결정으로 분할 압축 원본의 인덱싱·부분 읽기 운영을 중단했다. 이 문서 전체의 실행 명령, 다음 단계, 예산과 미완료 표시는 해당 실험 당시의 기록이며 현재 실행 지시가 아니다. 새 기본 경로는 [압축 해제된 RGB·pose 사용](DATA_ACQUISITION_STRATEGY.md)이다. 기존 코드·원본·인덱스·예약 기록은 보존하며 자동 재개하지 않는다. 이동된 로컬 근거는 [보관·복원 안내](LOCAL_STORAGE.md)를 따른다.

2026-10-29 · API10 · 개발·검토 단계. 실제 Drive/Colab 실행과 고정 배포물 생성은 아직 하지 않았다. Root 검토·로컬 커밋 뒤 revision/source ZIP/ready notebook의 정확한 경로·SHA256을 추가하고 테스트 세션에 인계한다. 기존 API9 source/ready notebook/cache는 변경하지 않는다.

## 목적과 고정 입력

RGB를 제외한 자료 중 ArchVis pose 단일 `ArchVisInterior_ArchVis_RT.tar.gz000` **368,145,316bytes**만 대상으로 한다. 파일명은 scene/kind/storage part를 식별할 뿐 tar 내부 위치를 제공하지 않는다. Byte0부터 기존 cache를 로컬 재해제하고 미보관 suffix만 원격으로 순차 읽어 정확한 내부 파일명 두 개를 선택한다. gzip 재개·seek index가 아니다.

- 새 JOB: `archvis_pose_1_10_tail352_001`, profile `posePrefixTail352MiB_v4`, policy `pose_prefix_tail352_v4`, 출력 cache `bounded_prefix_cache_v4`.
- 목표 sequence: `ArchVisInterior_ArchVis_RT/Day/rp_serena_rigged_004_ue4/SKM_MenReadingGlasses_Shape_01/003/Getting_Up__1_`. 정확한 목표는 이 경로 아래 `json/frame_1.json`, `json/frame_10.json`이다. Wildcard를 사용하지 않는다.
- 입력 ZIP: `outputs/test_archvis_pose_cache_warm192_20261023/prefix_cache.zip`, **201,414,370bytes**, SHA256 `7c538ca96670e5d8c8856659d5eab137d222d3ebab04321615eab3107baec157`.
- Cache manifest SHA256 `1c775f84085f1549e7fcb46f50e56d95a6de41ab1fa9b72a2e11e5dafb8c5234`, prefix SHA256 `a2ab3367ea55cf542f5b8a324439909142057a87a57d98be7d6710963bda96a9`, prefix **201,326,592bytes**/769chunk를 고정한다. 누락/불일치/원본 변경은 실패이며 cold 다운로드로 대체하지 않는다.

실제 warm192는 정상 bounded stop·목표0개였고 준비395.879초,그중 remote383.072초/신규64MiB였다. [검증 보고서 · 복원](LOCAL_STORAGE.md#개별-파일-복원) (`../outputs/test_archvis_pose_cache_warm192_20261023/verification.json`)와 [카탈로그 요약 · 복원](LOCAL_STORAGE.md#개별-파일-복원) (`../outputs/test_archvis_pose_cache_warm192_20261023/catalog_summary.json`)을 기준으로 한다. 180sequence/12,642entry는 부분 관측이고 목표 부재·사전순·원본 frame 정렬의 증거가 아니다. 마지막 관측 파일은 미완료이므로 추출본이 아니다.

## 예산과 중단

| 항목 | 새 시험의 고정 상한 |
| --- | --- |
| 논리 압축 입력 / 원본 잔여 media | 352MiB / 실제 원본 크기 기준166,818,724bytes(159.091MiB) |
| Range / 준비 HTTP | 256KiB 단위637회 / 정상1913회+pre retry3회=1916회 |
| 준비 응답 body / 시간 | 192MiB / 협력적1200초 |
| 초기 metadata 별도 | 128KiB /30초 /8HTTP |
| 두 단계 합계 | 192.125MiB /1230초 /1924HTTP |
| 누적 해제 / tar header / tar metadata | 2400MiB /60,000 /누적4MiB·단일64KiB |
| 저장 payload | 목표2파일·각4MiB 이하, 일반출력과 합산32MiB 이내 |
| Cache / chunk / local IO | 384MiB /1536 /708MiB |
| 별도 cache export | 60초 /local read1152MiB |
| Catalog / audit / 결과 ZIP | 6MiB·256sequence·16,000entry /1537행 /33MiB |

637회는 고정 입력과 원본 크기 기준이다. Profile은 원본 크기를 바꾸거나 더 작은 입력으로 재사용할 실행 허가가 아니다. 첫 신규 media offset은201,326,592여야 한다. 기존192MiB media를 다시 전송하지 않는다. 192MiB 응답 body는 신규media와 요청별 metadata·여유를 포함한다. 실제 원본 잔여량·요청 개수보다 어느 상한이든 먼저 닿으면 종료한다.

직전 remote 비용의 단순 비례는 약952초(16분)이며 속도·목표 위치·완주를 보장하지 않는다. 1200초는 요청/파싱 사이에서 확인하는 협력적 상한이다. 인증·업로드 UI/전송·별도 export·Drive 영구보관·전체 wire 비용은 위 준비 시간 밖이다. 업로드 후 로컬 처리의 기존60초 제한을 유지한다.

누적2400MiB 해제는 RAM에 한꺼번에 할당하지 않는다. 기존 입출력 chunk262,144/65,536bytes로 처리하되 cache 업로드가 최대384MiB Python bytes를 사용하고 JSON·catalog·중복경로 set·런타임 overhead도 있다. RSS hard limit을 구현한 것은 아니며 CPU 런타임의 가용 RAM **2GiB 이상**을 실행 전 확인한다. 일반 동시 파일 상한은 업로드384+생성cache384+exportZIP384+일반출력32+결과ZIP33=**1217MiB**이고 코드/환경/작은 보고서 여유를 더해 **임시 디스크2GiB 이상**을 확인한다. 기존 실제 입력은192MiB지만 일반 업로드 상한을 축소해 설명하지 않는다.

두 목표가 완전 저장되면 다음 header를 읽지 않고 종료한다. 이때 전체 gzip CRC 검증을 주장하지 않는다. 끝까지 읽은 경우에만 실제 tar end/gzip EOF·CRC 필드로 판정한다. 오류/시간초과/예산중단/정상미발견도 결과를 보존하고 같은 JOB를 자동 반복하거나 예산을 확대하지 않는다. 소스 변경·불완전 media·CRC/parser 오류의 unusable 상태를 유지한다.

Catalog는 기존 상한을 유지하므로256sequence 또는16,000entry 이후 `truncated=true`일 수 있다. Catalog 기록 중단과 목표 선택 추출은 독립적이다. 전체 목록이나 seek index를 주장하지 않는다. 입력v3는 기존 v3의 상한으로 읽고 출력만v4로 만든다. 이전 profile의 계약은 유지하고 이전 reader는v4를 거부한다.

## 테스트 세션 인계와 판정

Root 검토·커밋 후 고정 revision source ZIP와 API10 ready notebook의 정확한 경로/bytes/SHA256,입력ZIP/manifest/prefix 식별값,JOB/config/effective hash·원본 snapshot을 함께 기록한다. 그 전에는 사용자가 준비 중 notebook을 실행하도록 안내하지 않는다.

새 Colab CPU 런타임에서 가용 공간 확인 → source ZIP 한 개 업로드·revision/API10 대조 → 작은 offline preflight 3개 → 미리 채운 JOB 확인 → 고정 입력 cache ZIP 한 개 업로드·preflight 확인 → `EXECUTE_ONCE` 한 번만 켜서 Drive 읽기 전용 인증·실행 → review → cache/결과 ZIP 각각 export·영구보관 순서다. 개발용193MiB 합성 fixture 회귀는 사용자 notebook 기본 검사에서 제외한다. Test 세션은 실행 중 설정·코드·상한을 변경하지 않는다.

- **선택 추출 성공:** 목표 두 파일이 완전 저장되고 `targets_complete=true`, `file_verification=passed_current_code`, `selected_json_verification.selected_pair_passed=true`다. 파싱에 사용한 bytes도 entry SHA256과 대조한다. 이름 일치·JOB finished·빈 저장목록 검증만으로 성공 처리하지 않는다.
- **정상미발견:** usability가 유효한 정상 EOF 또는 bounded stop인데 목표가0/1개다. 확보분과 missing 목록,CRC/EOF 여부·실제 도달 범위를 기록하고 종료한다. bounded stop을 전체 부재로 해석하지 않는다.
- **실패:** unusable·원본 변경·입력 불일치·네트워크 요청 timeout 예외·CRC/JSON 오류 등이다. 준비 시간 예산의 `elapsed_limit`은 별도로 보고하며,후검증까지 usability가 유효한 경우 정상 bounded stop으로 분류한다. 요청/보고서/trace가 있는 범위만 보존하며 자동 대체/재실행하지 않는다. JSON object 구문 통과도 관절·좌표·RGB/시간 일치·학습 준비를 뜻하지 않는다.

회수 자료는 원본 결과 `archive_job_archvis_pose_1_10_tail352_001.zip`, 별도 `prefix_cache.zip`, 실행 notebook·preflight/review 출력,사용 runtime/Python·가용 RAM/disk·단계시간,Drive 영구보관 위치다. 로컬 보관 위치는 `outputs/test_archvis_pose_tail352_20261029/`로 예정하고 ZIP/hash·고정 revision을 이전 warm192와 분리한다. 실패/미발견도 동일 JOB로 종료한다.

## 다른 non-RGB와 RGB 후속 검토

보관 공식 split 세 파일은 `.local-archive/20261023/legacy_outputs.zip` 안 `outputs/m0_next_evidence_20261015/developer/official_split/UnrealEgoData2/{train,validation,test}.txt`에 있고 INDEX의 bytes/SHA256과 대조했다. 목표 sequence는 train에 정확히1행,validation/test0행이다. 이는 split 명시 근거이며 원본 motion 중복과 학습 release 검증을 대체하지 않는다. 확인된 calibration 단독 파일 목록은 없으므로 추측해 추가하지 않는다.

Drive 목록 조사는 [inventory · 복원](LOCAL_STORAGE.md#개별-파일-복원) (`../outputs/archive_lookup_20261028/drive_inventory.json`) 기준 train_val pose15scene/15개·18,020,771,152bytes와 RGB249개·2,597,969,076,181bytes의 metadata 확인이다. 내용·전체 무결성은 미검증이다. 별도 depth ArchVis는 metadata3,729,830,083bytes로 모든 non-RGB를 작다고 일반화할 수 없다. 이번 시험에서 다른 scene·depth·RGB를 받지 않는다.

RGB 후속 두 대안 모두 선택 scene의 최초 순차 읽기 비용이 든다. 필요한 sequence만 추출·영구보관하면 반복 학습이 단순하지만 추가 sequence에 새 탐색 비용이 든다. Tar 파일명→해제 offset와 gzip restart index를 함께 보관하면 추가 파일 접근에 유연하지만 Drive 분할/인증/원본 fingerprint 통합은 미검증이다. RGB000/001은10GiB 용량 분할이며 sequence·좌우 의미가 없다. 두 대안 모두 이번에 구현·원본 실행하지 않으며 pose 시험 성공을 RGB 해결이나 학습 준비 완료로 표시하지 않는다.

## 개발 검증과 실제 실행의 경계

신규 tail 회귀7개와 사용자용 소형 사전검사3개를 통과했으며 오케스트레이션도 각각 독립 실행했다(tail13.058초, 사전검사0.001초). Archive·모의 Drive 회귀163개 중 제한 환경의 localhost HTTP5개 검사에서10개 하위 조건이 연결 실패했고, 해당 HTTP 클래스6개를 연결 허용 환경에서 재검사해 모두 통과했다(5.450초). 나머지158개는 최초 실행에서 통과했으며 최초 실패 기록도 보존한다. 실제 Drive payload·Colab 실행·모델 학습 검증은 아니다.

개발 근거는 Git 제외 `outputs/pose_tail352_20261029/developer/`, 독립 검토는 `outputs/pose_tail352_20261029/root/review.json`에 보존한다. Notebook7개 code cell 구문과 빈 출력을 확인했으며, 고정 패키징 이후 별도 배포물 식별 대조를 수행한다.
