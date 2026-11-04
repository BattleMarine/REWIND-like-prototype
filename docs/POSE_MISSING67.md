# Pose 미관측67 목표의 warm 추출 준비

> **보존 기록 · 2026-11-05 정리:** 2026-11-04 사용자 결정으로 분할 압축 원본의 인덱싱·부분 읽기 운영을 중단했다. 이 문서 전체의 실행 명령, 다음 단계, 예산과 미완료 표시는 해당 실험 당시의 기록이며 현재 실행 지시가 아니다. 새 기본 경로는 [압축 해제된 RGB·pose 사용](DATA_ACQUISITION_STRATEGY.md)이다. 기존 코드·원본·인덱스·예약 기록은 보존하며 자동 재개하지 않는다. 이동된 로컬 근거는 [보관·복원 안내](LOCAL_STORAGE.md)를 따른다.

## 2026-11-03 후속 pose269 준비

G25 stereo 관측371개 중 최신 pose 목록과 미대응인269개를 별도 `posePrefixTargets269_v1`로 고정했다. 이전67개 profile/목표/자료는 보존한다. 입력235MiB cache pin과 기존 전송·시간·출력 상한을 그대로 사용하며, 신규 pose 원격 실행·학습은 아직 하지 않았다. [정확 목록·원본 잔여·부분 회수·검증](G25_STORAGE_AND_POSE269.md)을 따른다. 아래67개 실제 결과와 초기 준비 기록을269개 확보 완료로 해석하지 않는다.

## 2026-11-02 실제 Colab 결과: 67개 전부 확보

고정5fe714a의 pose_missing67_001에서 정확목표67개·완전저장67개·미확보0, targets_complete 및 selected_targets_passed=true를 확인했다. Root도 각각의 파일SHA와JSON object파싱, 보고서selfhash, 두ZIP CRC, 확장cache의 모든chunk/prefix SHA를독립대조했다. 이전234MiB prefix SHA가그대로이고첫신규media시작은245366784다. 이번67개는원본결측이아니라기존조기종료목록의미관측파일이었다.

신규media1048576bytes(1MiB),prepare body1049476bytes/14요청/19.103초,초기metadata별도290bytes/1요청이다. cache는234→235MiB(prefix246415360bytes),선택JSON총5735165bytes다. 준비시간에는로컬기존prefix재해제를포함하며업로드/인증/별도export비용과구분한다. 관측목록15465entry/219sequence/truncated=false이나전체목록은아니다. 목표조기종료로gzip EOF/전체CRC는검증하지않았고train_ready=false다. RGB102페어의기존35관측+이번67JSON이름대응을확보했지만RGB실제추출·내용정합·학습은미완료다.

결과 ZIP SHA `7ebe1cea2b88e7945a380068e0ac7d3fc26e34794437900f896eae0482865211`,cache ZIP SHA `f902b25aadb1932c4c734795592e122061f8d30410bdeed5bb3866528ebff437`다. 원본과독립검토는 outputs/test_pose_missing67_001_20261102/에보존한다. 아래개발준비/미실행기록은실행전상태이며이결과로갱신한다.

2026-11-02 개발 준비 상태다. 기존 ArchVis pose catalog는 Serena frame1·10 목표에서 조기 종료한 관측 목록이다. 목록에 없었던 다음67개를 원본 결측으로 판정하지 않고, 고정 warm cache에서 다시 탐색·선택 추출한다. 실제 원격 추출·Colab 실행은 아직 하지 않았다. 학습 release와 `train_ready=false`는 유지한다.

## 대상과 지원 범위

공통 prefix는 `ArchVisInterior_ArchVis_RT/Day/rp_serena_rigged_004_ue4/SKM_MenReadingGlasses_Shape_01`이다. 파일은 각 sequence 아래 `json/frame_N.json`이다.

| Sequence | Frame | 개수 |
| --- | --- | --- |
| `003/Getting_Up__1_` | 5, 12 | 2 |
| `010/Football_Hike__4_` | 0~8 | 9 |
| `025/Leg_Sweep__2_` | 0~23 | 24 |
| `028/Dribble__4_` | 0~31 | 32 |

전용 profile `posePrefixTargets67_v1`은 위 정확한67개 집합과 고정234MiB cache만 허용한다. 기존 profile의32개 제한과2목표 retry 제한은 보존한다. 기존 `pose_prefix_tail352_v4` cache policy·workflow를 재사용하며, 별도 영속 gzip 인덱스를 새로 만들지 않는다. 기존 notebook도 변경하지 않는다.

## 고정 입력

기존 cache는 `outputs/test_archvis_pose_tail352_20261029/prefix_cache (2).zip`이다. 원본은 덮어쓰지 않는다.

| 식별값 | 값 |
| --- | --- |
| Cache ZIP 크기 | 245473378 bytes |
| Cache ZIP SHA256 | `47ec61cef1a9115cfc0b78d1192a9d0a84e5f0902d3a1f0fa1257449e0e842cd` |
| Manifest content SHA256 | `b7bcb31af4be74514615c48f551ef7f0a0449428c8260b79ce9abe0df88c05ab` |
| Prefix SHA256 | `dd3452a2b1a159419b8e0a247f38010752caeb1a29c92c4e24024cda5ab48306` |
| Prefix 크기 | 245366784 bytes = 234MiB |
| Archive manifest SHA256 | `d77fd670e66256bd7c4291bff9b5a100012ecbd046a32b3774afe8beb8f2572b` |
| 원본 크기 | 368145316 bytes |
| 원본 part | `ArchVisInterior_ArchVis_RT.tar.gz000` |
| Drive file ID | `183Fi9DoJwy32oYG5My9C0a13dtxfT5MX` |
| Drive version / modified | `1` / `2026-10-13T01:19:08.783Z` |

ZIP SHA와 manifest content SHA, prefix SHA는 서로 다른 식별값이다. 실행 전에 원본 metadata identity를 다시 검사한다. Cache 누락·변조·identity 불일치 시 cold fallback은 없다. 최초 신규 media Range는 byte245366784부터 시작한다. 기존234MiB는 로컬에서 byte0부터 다시 해제하므로 신규 전송 절감과 해제 비용 절감을 혼동하지 않는다. 원본 끝까지의 신규 media 잔여량은122778532 bytes(약117.091MiB)이며 HTTP·metadata 비용과 별개다.

고정 입력은 `outputs/pose_missing67_20261102/developer/pose_missing67_inputs.zip`이며 SHA256은 `888c6b4b2e5144ba8f645e16a79eb9e9454819e8131f3f63841098c9fc8854df`다. 내부 `config.json`과 `targets.json`을 사용한다. Canonical targets SHA256은 `c82b42face3924210d2b4f94bac01ac802311a077c6837858a9a5f089109e708`이다. Root가 검토·커밋한 새 source ZIP과 ready notebook의 revision·SHA를 별도로 고정한다. 이전 source ZIP은 신규 profile을 지원하지 않는다.

## 예산과 중단

| 범위 | 상한 |
| --- | --- |
| 최초 metadata | 응답128KiB / HTTP8 / 30초 |
| Prepare | 응답 body192MiB / HTTP1916 / 1200초 |
| 압축·해제 | 논리 압축352MiB / 반환 해제2400MiB / tar header60000 |
| 파일 | 목표67개 / 개별4MiB / 보고서 포함 일반 출력32MiB |
| Tar metadata | 총4MiB / 개별64KiB |
| Cache 준비 | disk384MiB / local IO708MiB / chunk1536 |
| 관측 목록 | 6MiB / sequence256 / entry16000 |
| Cache export | 별도60초 / local IO1152MiB / ZIP384MiB |
| 결과 ZIP | 33MiB, cache 본체는 별도 |
| 업로드 후 로컬 준비 | 60초, 사용자 업로드 대기 제외 |

`metadataPreTimeout10sRetry1Global3_v1`의 제한된 metadata pre-open timeout retry만 허용한다(회당1·전체3). 인증 내부·media hidden retry는0이다. 상한 도달·원본 변경·무결성 오류 시 자동 재시도하지 않고 결과를 보존한다. 관측 목록의 truncation은 목표 추출 중단이나 원본 결측 판정 근거가 아니다.

새 CPU 런타임에서 가용 RAM·임시 disk를 각각2GiB 이상 확인한다. 현재 경로는 peak RSS hard cap을 구현하지 않았으므로 실행 측정은 별도다. Colab upload가 작업 디렉터리에 남기는 ZIP, 명시 staging cache, 복원 prefix, 확장 prefix/cache ZIP은 별도 disk 비용이다. 인증·업로드/UI·최초 metadata·prepare·cache export 시간을 합쳐 하나의 prepare 시간으로 보고하지 않는다.

## 실행본과 회수

개발 notebook은 [pose_missing67.ipynb](../notebooks/pose_missing67.ipynb)다. Root가 source·입력 SHA와 revision을 채운 고정본을 데이터셋 탐색 세션에 인계한다. 사용자 실행 안내는 해당 세션이 담당한다.

1. 새 Colab CPU 런타임에서 고정 설정을 확인한다. 이전 `rewind_archive` 모듈이 다른 source 경로에서 로드돼 있으면 preflight가 인증 전에 차단한다. 자동 reload하지 않는다.
2. 고정 source·입력·기존 cache ZIP 세 개를 업로드한다. 모두 검사한 뒤 staging한다. 정확히 같은 기존 staging만 읽기 재사용하며 불일치는 덮어쓰지 않는다. Notebook은 의존성을 설치하지 않는다.
3. Offline preflight 후 인증·실행 셀을 한 번 실행한다. 작업 소비 후에는 같은 job을 다시 실행하지 않는다.
4. 성공·실패·중단 모두 회수 셀을 사용한다. 인증·원본 요청·실행을 반복하지 않는다. Job report가 없으면 source/입력 없이 준비 실패 ZIP을 회수할 수 있다. Job이 있으면 JSON 검토와 bounded cache export 후 결과 ZIP·확장 cache ZIP을 각각 회수한다. 각 ZIP 경로와 SHA를 출력한다.
5. 탐색 세션은 원본 cache, 회수 ZIP/SHA, missing 목록, 종료 이유, JSON/hash 검토, 전송·시간·disk 비용과 마지막 정상 결과를 root에 전달한다. 결과·cache는 런타임 밖에 영구 보관한다. 자동 후속 실행은 하지 않는다.

모든67개가 저장되고 현재 파일 hash 및 JSON object parsing이 통과해야 `selected_targets_passed=true`다. 기존2개용 `selected_pair_passed`와 구분한다. `targets_complete`, 실제 gzip EOF/CRC 확인, 예산 중단, invalid JSON, missing targets를 별도로 보고한다. 목표 조기 완료는 전체 archive 무결성·coverage 완료가 아니다. JSON 구문 검사는 관절·좌표·시간·RGB 정합성이나 학습 준비 검증을 대신하지 않는다.

## 로컬 검증

작은 합성 warm fixture로67개·첫 신규 Range·JSON 검토·EOF 결측·예산 중단·무warm 차단·기존32개 계약을 검사했다. 실제 notebook 셀의3ZIP 준비·변조 거부·동일 staging 재사용·이전 import 차단·준비 실패 회수·완료 작업 회수도 확인했다. 인증 factory는 검토된 RGB notebook과 AST가 같으며 보존된 실제 requests/google-auth 타입으로 hidden retry0·검사 실패 시 close를 검증했다.

개발 최종24개 검사 통과(skip0), 로그는 `outputs/pose_missing67_20261102/developer/local_tests.txt`다. Root의 별도 core14개와 실제 입력 preflight 검토는 실제 원격 추출을 뜻하지 않는다. 실제 Drive/Colab 추출,234MiB 재해제·자원 실측, 후속 RGB 결합·학습은 미실행이다.

Root 검토: core14개 및 notebook/auth10개를 독립 실행해 통과했다. 실제 고정 입력·234MiB cache ZIP으로 준비/preflight 셀을 원격 요청 없이 실행했고, 별도 합성 invalid JSON은67파일 저장·hash 일치에도 selected_targets_passed=false임을 확인했다. 근거는 outputs/pose_missing67_20261102/root/에 보존한다. 이 검사는 실제 pose gzip 재해제나 Drive 추출 검증을 대체하지 않는다.
