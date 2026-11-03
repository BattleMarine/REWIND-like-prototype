# Pose269 부분 결과와 남은8개 준비

2026-11-03 기록이다. 실제 Colab 실행은269개 중257개를 추출했고, 보존 확장 캐시를 원격 요청 없이 조사해4개를 더 확보했다. 현재269목표 중261개, 기존102개와 합쳐 g25 stereo371개 중363개가 pose 이름으로 대응한다. 이번 작업의 신규 RGB 추출·학습은0이다. 나머지8개는 현 prefix 미관측이며 원본 결측으로 판정하지 않는다.

## 실제 Colab 결과와 중단 원인

`outputs/test_pose_missing269_001_20261103/archive_job_pose_missing269_001.zip`의 SHA는 `61bd105189ca2c0165dbb6dbac3f7c002e4937a00888daf4f8a3229f9ed411e2`다. Root와 개발이257개의 실제 payload SHA·크기·JSON object를 검증했다. Payload21997729 + report 예약11533755 + 다음 파일85701 =33617185bytes가33554432bytes를 넘으면서 `payload_disk_bytes_limit`으로 정상 중단했다. 이는 초기 브라우저 업로드 연결 끊김의 원인을 설명하거나 OOM 해결을 입증하는 결과가 아니다.

확장 캐시는 같은 폴더의 `prefix_cache.zip`,250718584bytes/SHA `e8516bf6214bac6253551ec778dad6a57a3c6d6657af4200cab12994713f996f`다. 원본 파일은 수정하지 않았다.

## 로컬4개 복구와 검증 범위

처음부터 연속한 gzip prefix를 순차 해제하며 완전한 tar member만 선택했다. `004/Pulling_A_Rope__9_`의 frame46·77·20·127을342328bytes로 확보했다. Gzip prefix 끝에서 EOFError가 발생했고 불완전 파일은 사용하지 않았다. 전체 archive EOF/CRC는 검증하지 않았다.

처음 조사 스크립트는 전체 prefix SHA·길이 확인을 통과한 뒤, chunk가 항상256KiB라고 잘못 가정한 보조 비교에서 실패했다. 짧은 중간 chunk를 허용하는 실제 manifest와 맞지 않은 검사였다. 최초 스크립트·`recovery/report.json`·4파일을 보존했고 전체 scan을 반복하지 않았다. 별도 허용 예산256MiB/120초 안에서 manifest의 start/end대로957개 chunk와 prefix SHA를 검증했다. Root도 이를 독립 확인했다.

개발 후검증은4파일을 다시 읽어 SHA/JSON을 확인하고, 기존257개와 보존102개의 실제 파일도 hash/크기/JSON을 대조했다. 세 집합은 중복되지 않으며363개의 sequence/frame key가 고정 g25의 좌우 RGB371 key에 포함된다. 이는 이름 대응이다. 새 RGB 실제 추출·RGB 내용/시간 정합·원본 motion 독립성·validation 확보·학습 승인은 별도다.

근거는 `outputs/pose12_recovery_20261103/developer/recovery/`의 `report.json`, `audit.json`, `chunk_verification.json`이다. 최초 scan의 local read501437210bytes/decode1373962240bytes/4.828초와 후검증의 누적 local file read519457938bytes를 기록했다. SQLite state 반환267123bytes는 별도이며 OS 전체 disk IO를 뜻하지 않는다. Scan과 후검증의 실행 시간 합은5.969초로 중간 검토 대기 시간을 제외한다. 최초 준비의512MiB read·2400MiB decode·600초·32MiB pose 출력 상한과 별도 chunk 검증 예산을 섞어 보고하지 않는다. Cache 복사본250718584bytes는 별도 준비 파일이며 pose 출력에 포함하지 않는다.

Root의4파일 보관 ZIP은 `outputs/pose12_recovery_20261103/root/recovered_pose4.zip`,117298bytes/SHA `7d2db3c4abe4ec8ac6495c9d19560f37e393ea0e0273b3271e64ea2956a3781a`다. 이것은 runtime 회수 ZIP을 대체하지 않는다.

## 출력 상한 계산

실제261개 payload는22340057bytes다. 이번 검증 집합의 관측 최대 JSON85773bytes로 남은8개를 추정하면686184bytes,269개 합23026241bytes다. 기존 report 예약11533755bytes를 더한34559996bytes는32MiB를1005564bytes 초과한다. 40MiB에는7383044bytes 여유가 있다. 종전 관측 최대86031bytes로 계산해도40MiB 이하다. 이는 관측 크기 기반 계획이며 개별4MiB의 최악 경우 전체 완료를 보장하지 않는다.

기존 `posePrefixTargets269_v1`의32MiB와 옛 입력·notebook·캐시 pin은 재현을 위해 유지했다. 이번에 승인된 새 잔여8 profile에만 일반 출력40MiB·결과41MiB를 적용한다. 새8 자체의 report 예약은 `report_reserve_bytes`3935264 + catalog6291456 =10226720bytes이고 예상 payload686184bytes와 합10912904bytes다.40MiB는 이번에 명시 승인한 여유 상한이며269개를 다시 추출하지 않는다. 일반·67·269의 상한 확대는 구현하지 않았다.

## 후속8개 고정 계약 — 준비 완료, 원격 미실행

새 profile은 `posePrefixTargets8_v1`, job은 `pose_missing8_001`이다. 공통 sequence는 `ArchVisInterior_ArchVis_RT/Day/rp_serena_rigged_004_ue4/SKM_MenReadingGlasses_Shape_01/004/Pulling_A_Rope__9_`이며 `json/frame_N.json`의0·11·38·42·73·82·116·145만 허용한다. 기존 train sequence 소속을 유지하며 임의 목표·cold fallback·다른 캐시 pin을 거부한다.

| 고정 항목 | 값 |
| --- | --- |
| Targets canonical SHA | `7def9e3b9e15575df4bf7253b519f49814af94bfcd2fa20bee2096ada2c44c6c` |
| Cache manifest content SHA | `04cf05af85d707193e48445de896f05ccbf0d929262fcd3f0b03342def1e8b87` |
| Prefix SHA / bytes | `47390666a932805d9a156e69393a4b28a2b5ace433eb9af0432e3debb753a12d` /250609664 |
| Archive manifest SHA | `d77fd670e66256bd7c4291bff9b5a100012ecbd046a32b3774afe8beb8f2572b` |
| Input ZIP | `outputs/pose12_recovery_20261103/developer/pose_missing8_inputs.zip` /1597bytes |
| Input ZIP SHA | `e665b671846faee1e486c0369c1a9ff0d71eee0e3a5be7068c529a23c0f27762` |
| Config canonical SHA | `70abfaf58a29e3bb705c09d8742432588380ec1ffe89169d90003b41a2c27e1a` |

[pose_missing8_drive.ipynb](../notebooks/pose_missing8_drive.ipynb)는 작은 source/input 두 ZIP을 먼저 검증하고, auth/mount→명시 Drive cache readonly staging→기존 preflight→한 번 실행→독립 회수를 따른다. Source ZIP·revision·ready notebook SHA는 root가 검토·로컬 커밋 후 고정한다. 개발 template의 source/revision 및 `CACHE_DRIVE_PATH`는 빈값이다. 다른 job의 source/input/workspace를 재사용하지 않는다. 소비된 workspace와 실패 partial을 덮어쓰거나 자동 재실행하지 않는다. 준비 report가 바뀌면 별도 hash 이름으로 준비 ZIP을 회수한다.

| 범위 | 상한 |
| --- | --- |
| Cache staging | source250718584 + local hash250718584bytes,1MiB chunk,300초 협력,disk256MiB |
| Validator ZIP metadata | 별도2MiB |
| 원본 prepare | body192MiB,1916HTTP,1200초 |
| 논리 압축 read / decode | 352MiB /2400MiB |
| 선택 파일 / 개별 파일 | exact8 /4MiB |
| 일반 출력 / 결과 ZIP | 40MiB /41MiB |
| Cache disk / import·replay IO | 384MiB /708MiB |
| Cache export | 별도60초,IO1152MiB,ZIP384MiB |
| 초기 metadata | 별도128KiB,8HTTP,30초 |

원본368145316bytes에서 현 prefix를 뺀 잔여117535652bytes는 최대449개의256KiB range다. Pre/media/post 약1347HTTP와 별도 최초 metadata로 계산하면 현192MiB/1916HTTP 이내의 계획이지만 위치·시간·목표 완료는 보장하지 않는다. Pose metadata pre-open timeout의 기존 제한 retry만 유지하고 hidden auth/media retry는0이다.

기존235MiB `stage_pose269_cache`와 `stage_file` 기본 크기 상한을 보존하고 새 `stage_pose8_cache`만 고정250718584bytes 상한을 명시한다. 실제 local 복사본의 새 wrapper 검증은 source read0/local SHA250718584/metadata108922bytes/0.250초로 통과했다. Drive FUSE wire·mount·Colab 안정성 결과가 아니다.

확장 캐시는 사용자가 Drive의 `REWIND-like prototype/reports/cache/`에 업로드한다. 사용자가 `prefix_cache(4)`라는 이름을 예고했지만 완료 진술과 정확한 공백·확장자는 아직 미확인이다. 파일명을 추정하지 않고 에이전트는 업로드하지 않는다. 사용자가 완료했다고 알린 뒤 정확한 mounted filepath를 확인하고 ready에 고정하며 검색 지연만으로 재업로드를 요구하지 않는다. 실제 실행 안내와 결과 취합은 탐색 세션이 담당한다. 실패하면 원본·partial·report와 결과/확장cache ZIP을 각각 보관하고 자동 반복하지 않는다.

개발39개/17.297초 검사 failure/error/skip0이다. Warm 첫 신규 range·8개 SHA/JSON·변조/누락 pin·cold 차단·부분 결과 회수·새40/41MiB scope·기존269/67/일반 계약·staging 기본 cap·auth factory·notebook의 두 업로드와 mount 순서·소비 guard·stale 준비 ZIP 회귀를 포함한다. 보존 dependency 경로를 빠뜨리고 신규 테스트 파일 생성 전에 시작한 최초 검사 실패/skip 기록도 별도 보존했다. 최종 로그는 `outputs/pose12_recovery_20261103/developer/final_tests.txt`다. 신규8 실제 원격/Colab 실행·학습·commit/push는 개발 세션이 하지 않았다.

Root 독립36개 검사(14.627초, failure/error/skip0)와 실제 cache/input ZIP의 pin·CRC·정확8개 차집합·preflight를 확인했다. 근거는 `outputs/pose12_recovery_20261103/root/tests.txt`, `recovery_review.json`이다. 실제 후속8개 원격 실행은 아직 하지 않았다.
