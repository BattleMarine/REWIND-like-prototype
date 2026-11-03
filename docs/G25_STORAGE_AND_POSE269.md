# g25 저장 구조 검토와 pose269 확보 준비

2026-11-03 실제 pose269 Colab 실행은257개 hash/JSON 검증 후 출력32MiB 한도에서 중단했습니다. 확장 cache를 원격0으로 조사해 완전한4개를 추가 확보하고 기존102개와 대조하여 g25 stereo371 중363개 pose 이름 대응을 확인했습니다. 신규 RGB 추출·학습은0이며 남은8개는 원본 결측이 아닌 현 prefix 미관측입니다. 기존269/67/일반 계약을 보존한 새 exact8 profile·40MiB 출력/41MiB 회수·239MiB cache pin·새job/Drive notebook을 준비했고 개발39개 검사가 통과했습니다. 신규8 실제 원격 실행과 Drive 정확 경로는 미확인입니다. [실제 결과·상한 계산·남은8개 실행 계약](POSE8_RECOVERY.md)을 따릅니다.

## 2026-11-03 업로드 준비 경로 보완

기존65a7244 pose269 notebook에서 큰 cache 업로드 중 연결 끊김이 보고됐다. 원인은 미확정이고 dataset 원본 prepare 실행 여부를 단정하지 않는다. 목표·cache pin·예산은 보존한 채 작은 두 ZIP만 업로드하고 확인된 Drive 파일에서 chunk staging하는 별도 notebook을 준비했다. Drive 경로는 확인 전 빈값이며 실제 Colab/OOM 해결은 미검증이다. [새 준비 순서·별도300초/256MiB 예산·부분 회수](POSE269_DRIVE_STAGING.md)를 따른다. 아래269 입력·목표·상한은 그대로다.

2026-11-03 · g25의 정상 보관본을 원격 없이 분석하고 bounded local compact helper와 pose269 실행 준비를 구현했다. Compact 복사본을 공식 resume head로 승격하거나 snapshot export에 자동 연결하지 않는다. 이번 사용자 실측 대상은 pose269 한 건이며 새 RGB 인덱싱·학습은 실행하지 않는다.

## 직전 실제 g25 결과

실제 `archvis_rgb_preindex_g16_002`는 g16/494→g25/782member의9개 정상 배치를 저장했다. Root가 ZIP CRC·DB/catalog hash·기존494개 prefix 보존·키/split 집계를 독립 확인했다. Stereo 관측371개 중 pose 관측 교집합은102개, validation RGB0·실제 추출0이다. 신규269개는 이름 대응 자료이며 아직 완전한 RGB+pose 입력269개를 확보한 결과가 아니다.

| 원본 pin | 값 |
| --- | --- |
| 실제 결과 ZIP SHA256 | `320343f3c65a780d1258877da56a539f777ea2acba115b210de65c78dfb9ce13` |
| g25 snapshot SHA256 | `24686a9c3e1ec63347198778005ea0eabc81753e687c69653aca356876b4250e` |
| Receipt 파일 SHA256 | `d17c09fe12912fe6e9ebf66d84154cb0c42793b44a5c0bd23eb1a8d0afd2f70d` |
| Snapshot 내부 DB SHA256 | `6272b03c276d8edbdbbb6377ae3f19fb28095d4beaf9f7d12f9984b031019fbc` |

결과는 `outputs/test_archvis_preindex_g16_002_20261103/`에 보존한다. Wall375.487초/process CPU33.667초·응답 body347149305bytes/1088HTTP였다. 다음 scan 전 전체 예약809984041bytes가768MiB를 초과해 `discovery_scan_snapshot_recovery_reserve`로 정상 중단했다. 직전 checkpoint 예산 오용이 다시 발생한 것이 아니다. 이번에도 전체 archive EOF/CRC·validation 발견·학습 준비는 완료하지 않았다.

## 실제 DB·gzip index·snapshot 비용

DB는55402496bytes, page size4096/page count13526/freelist6616이었다. 빈 페이지27099136bytes(약25.84MiB)는 동일 행의 큰 BLOB을 갱신하는 SQLite 파일에 남은 공간이다. JSON state267123bytes와 gzip index27999297bytes가 있으며, 파일 크기를 현재 논리 payload 크기와 혼동하지 않는다.

보존 `indexed_gzip==1.10.3`의 `zran.c` v1 export layout을 읽기 전용으로 대조했다. 인덱스855point 중854개가32768byte dictionary를 가져 dictionary만27983872bytes를 차지한다. Header·offset/bit/flag mapping을 더하면 실제27999297bytes와 일치한다. Spacing1MiB, 현재 decoded coverage899910484bytes다. Index32MiB까지 여유는5555135bytes뿐이다. DB compact는 이 dictionary payload를 줄이지 않는다.

9개 snapshot 전체 복사본의 합은412978442bytes, export wall 합17.306초였다. 변경되지 않은 기존 dictionary도 매 snapshot에 반복 저장되는 구조여서 누적 공간이 먼저 찰 수 있다. 이 값은 닫힌 ZIP들의 논리 크기·기존 export 호출 범위이며 Drive의 전체 wire/동기화 비용이나 순수 disk 시간은 아니다. 독립 index를 임의 합치거나 point를 지워 줄이는 최적화는 하지 않았다.

## 이번 구현: bounded 새 local compact 복사본

[compact.py](../rewind_archive_index/compact.py)는 닫힌 DB의 예상 SHA를 명시하고 새 출력 디렉터리에서 `VACUUM INTO`를 실행하는 별도 helper다. 실제 g25에서55402496→28303360bytes, freelist0으로 줄었고 SQLite integrity 및 checkpoint state/raw hash/index blob/hash/anchor byte가 모두 동일했다. 원본 SHA는 그대로이며 source 순서·identity·generation·coverage는 바뀌지 않았다. 개발 실제 helper는0.484초에 완료했다.

이미 실행한 명령은 다음과 같다. 출력은 소비된 경로이므로 반복 실행하지 않는다.

```text
python -m rewind_archive_index.compact --input-db outputs/g25_next_20261103/developer/g25_original_copy.sqlite --output outputs/g25_next_20261103/developer/verified_compact --expected-sha256 6272b03c276d8edbdbbb6377ae3f19fb28095d4beaf9f7d12f9984b031019fbc
```

입력64MiB·원본+출력 디렉터리256MiB·60초를 제한한다. 기존 경로/덮어쓰기·source 안의 output·활성 `-wal/-shm/-journal` source는 거부한다. Readonly 연결과 별도 SHARED read transaction을 검증/VACUUM 동안 유지하고 전후 main DB SHA를 대조한다. 이 계약은 닫힌 rollback-mode DB를 대상으로 하며 활성 writer/WAL DB의 일관된 백업을 제공하지 않는다.

SQLite temp store는 MEMORY, page cache는 연결당8MiB로 설정한다. 출력 DB와 journal 등은 새 출력 디렉터리에서 관측하며 Python/SQLite progress 기반 **협력** disk/time guard다. SQLite 내부 memory·OS IO·시간의 hard cap은 아니다. 입력 크기 자체도64MiB로 제한한다. 오류·중단 시 안전한 종류/원인과 report를 보존하고 생성된 partial을 자동 삭제하지 않는다. 입력이 없거나 sidecar가 있거나 시간 중단이어도 가능한 failure report를 남긴다.

결과 role은 `verified_local_copy_not_resume_head`, `official_resume_head=false`다. Physical DB SHA가 바뀌므로 기존 snapshot receipt 아래 파일을 이 복사본으로 바꾸면 안 된다. Snapshot export/runner에 compact를 연결하거나 원본/정상 receipt를 수정하지 않았다. 근거는 `outputs/g25_next_20261103/developer/verified_compact/report.json`이다. Root도 별도 helper 실행으로 동일 결과를 확인했다.

## 장기 구조 제안 — 미구현

작은 compact는 빈 페이지와 파일 복사량을 줄이는 수단이며 index32MiB·point4096·state/member/snapshot cap 또는 전체 archive 확장을 해결하지 않는다. 이후에는 다음 설계를 별도로 검토해야 한다.

1. Catalog를 member ordinal·tar offset·완전 파일 hash 단위의 append 행으로 관리하고, checkpoint는 닫힌 generation·연속 boundary·catalog prefix hash를 연결한다. 학습 shuffle과 물리 archive 순서는 분리한다.
2. 같은 고정 원본의 restart point mapping과 dictionary를 immutable chunk/CAS로 보존해 snapshot 간 동일 byte 중복을 줄인다. 원본 identity·조각 순서·engine/format/window/bit offset을 고정하며 독립 index를 섞지 않는다. 실제 dictionary 총량이 사라지는 것은 아니다.
3. Migration은 구 v1 DB/receipt를 그대로 보관하고 새 schema의 별도 manifest를 발급한다. Legacy 입력을 읽는 adapter와 기존 blob의 byte/hash 복원·member 재추출 동등성을 검사해야 한다. 현 engine이 전체 index import를 요구하는 문제와 선택적/lazy point 복원의 지원 여부도 확인해야 한다.
4. Point 간 최대 compressed/decoded read 거리와 RAM·시간 상한, incomplete source·EOF/CRC·경계 파일·source 변경·중단/rollback·세대 충돌·새 runtime 복원을 검증한다. Dictionary를 줄이려고 point를 희소화하면 read 거리가 늘 수 있으므로 별도 평가한다.
5. 새 manifest는 정상 parent/identity/prefix와 byte pin을 모두 확인한 뒤 게시한다. 실패하면 이전 정상 head가 기준이며 orphan/구 snapshot의 GC는 별도 명시 승인 없이 수행하지 않는다.

이번에는 이 저장 엔진·마이그레이션·선택적 point reader를 구현하거나 core cap을 늘리지 않는다. RGB 새 job도 준비하지 않는다.

## Pose269 정확 목표와 기존 예산

G25 stereo371−최신235MiB pose 관측102의 정확 차집합269개는 root의 독립 목록과 같다. 공통 prefix는 `ArchVisInterior_ArchVis_RT/Day/rp_serena_rigged_004_ue4/SKM_MenReadingGlasses_Shape_01`이고 경로는 각 sequence의 `json/frame_N.json`이다.

| Sequence | Frame | 개수 |
| --- | --- | --- |
| `004/Pulling_A_Rope__9_` | 0~148 중51·80 제외 | 147 |
| `017/Blow_A_Kiss__14_` | 0~121 | 122 |

둘 다 train이다. 51·80 제외는 현재 RGB stereo 대응 범위 때문이며 원본 pose 결측이라는 뜻이 아니다. Targets canonical SHA는 `fa0bdf8458a931d8adb8f5dfdf7dc2c2a2dcbc8e989e97680fd8c6720b65b41c`다.

새 `posePrefixTargets269_v1`만 정확269개와 고정235MiB cache를 허용한다. 기존67/일반32개 제한은 보존한다. Cache ZIP은 `outputs/test_pose_missing67_001_20261102/prefix_cache (3).zip`,246522402bytes/SHA `f902b25aadb1932c4c734795592e122061f8d30410bdeed5bb3866528ebff437`다. Cache manifest content SHA `9177444e8283f57b97f037f3ceeafffdaead90b210694a8b23b3f3ebf8e06f0f`, prefix SHA `822252bfc667956aa9791423d0e86da80fac33be21d3a26b2b150ff8ff7cb756`, prefix246415360bytes를 함께 고정한다. 누락/변조 시 cold fallback은 없다.

원본368145316bytes에서 prefix를 뺀 잔여121729956bytes(약116.09MiB)는 최대465개256KiB range다. Pre/media/post와 cached identity 검사를 포함해 약1397 prepare HTTP를 예상하며 최초 metadata는 별도다. 이론적으로 기존 전송 상한 안의 범위지만 실제 해제/시간/파일 위치·목표 완료를 보장하지 않는다. 관측 JSON85101~86031bytes를 적용한269개 payload 상단 추정은23142339bytes다. 실제 크기 상한은 따로 적용한다.

| 범위 | 유지하는 상한 |
| --- | --- |
| Prepare | body192MiB / HTTP1916 / 1200초 |
| 논리 압축 / 반환 해제 | 352MiB / 2400MiB |
| 파일·일반 출력 | 269개 / 개별4MiB / 보고서·catalog 포함32MiB |
| 초기 metadata | 128KiB / 8HTTP / 30초, prepare와 별도 |
| Cache | disk384MiB / import·replay local IO708MiB |
| Cache export | 별도60초 / local IO1152MiB / ZIP384MiB |
| 일반 결과 ZIP | 33MiB, 신규269 profile의 파일 inventory만512개로 허용 |

Metadata pre-open timeout만 제한적으로 재시도하는 기존 pose 정책을 유지한다. Hidden auth/media retry는0이며 RGB 인덱싱의 retry0와 혼동하지 않는다. Catalog16000entry가 truncate돼도 목표 추출은 계속한다. 모든269개 파일 hash와 JSON object가 통과해야 `selected_targets_passed=true`다. Partial/missing/invalid JSON·목표 조기 완료와 실제 EOF/CRC는 구분한다. 한도에 닿으면 부분 결과를 보존하고 자동 확대/재실행하지 않는다.

## 고정 입력·notebook·검증

Job은 `pose_missing269_001`이다. 입력 `outputs/g25_next_20261103/developer/pose_missing269_inputs.zip`은4345bytes/SHA `ef22cd2953a70a9c718b525402150cbc44a5a9ae514d096ae5de3fcb17789700`, config canonical SHA는 `b0cc29b9f979016d4a9a137c5bc6648f05ff20b2e7bd4ce4e59b2823b2801614`다. [pose_missing269.ipynb](../notebooks/pose_missing269.ipynb)는 검토된67 notebook의3ZIP 선검증·auth factory·새 CPU runtime·독립 회수를 재사용한다. Root가 commit 후 source/ready SHA와 revision을 고정해 탐색 세션에 인계한다. 기존 source/input/cache/notebook은 덮어쓰지 않는다.

가용 RAM/임시 disk 각각2GiB를 확인한다. Upload된 ZIP·staging cache·복원 prefix·확장 cache ZIP·일반 결과 ZIP은 별도 공간이며 준비/인증·로컬 replay·새 전송·export 비용을 구분한다. 실패·중단 후에는 회수 셀만 실행하고 결과 ZIP과 확장 cache ZIP을 각각 SHA와 함께 영구 보관한다. 원본 archive 전체 무결성·좌표/시간/RGB 내용 정합·학습 release는 이번 JSON 확보 성공만으로 승인하지 않는다.

개발 최종40개/15.103초 failure/error/skip0 통과다. 작은 warm fixture의269 목표·첫 신규 Range·hash가 맞는 invalid JSON 거부·기존32/67·cold 차단·partial ZIP·catalog truncation 비차단, helper failure/sidecar/시간·중단, 기존 snapshot·67 notebook/auth·targets 회귀를 포함한다. 실제235MiB cache의 pin/preflight와 신규 notebook 셀의3ZIP 준비·재사용·준비 실패 회수도 확인했다. Root는 핵심17개+legacy24개+신규 notebook3개, 총44개를 독립 통과하고 실제 g25 helper 및 입력 ZIP/원본 byte를 대조했다. 개발 로그는 `outputs/g25_next_20261103/developer/final_tests.txt`, root 근거는 같은 상위 폴더 `root/`다. 실제 pose269 원격 실행·신규 RGB 추출·학습은 아직 하지 않았다.
