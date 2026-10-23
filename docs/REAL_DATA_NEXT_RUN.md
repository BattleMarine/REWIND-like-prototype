# 다음 실제 데이터 준비 실행

## API9 bounded warm192 인계 준비

실제 API8 warm128은 112.094초에 cache 재사용/128MiB 확장을 완료했으나 목표 pose는 0개였습니다. 같은 128MiB 반복은 새 검색이 없으므로 `posePrefixWarm192MiB_v3`로 한 번의 후속 탐색을 준비합니다. 실제 192MiB 원본 실행은 아직 하지 않았습니다. [계획/실측 가정](PREFIX_CACHE_192_PLAN.md)과 [실행 계약](PREFIX_CACHE.md)을 따릅니다.

필수 입력은 128MiB v2 cache이며 manifest SHA `de57a189205a4ddeb7dc61379e09fca08b4d5cd5ec033d0b7a8d2b50abc775c1`, prefix SHA `2e05f6850bd5a019b232de3383efc03e91982bb38681523f0a2ac9c824029268`, 길이 134,217,728을 ready `cache_expectation`에 고정합니다. 원본 identity·입력 ZIP SHA도 인계에서 대조합니다. 기존 Serena frame1·10 목표를 유지하고 다른 sequence의 새 RGB 비용은 미확정으로 둡니다.

새 network body 80MiB·논리입력 192MiB·420초·773요청, 해제 1200MiB·header 30,000·tar metadata 2MiB, cache/각 ZIP 224MiB·chunk 896·local IO 388MiB·export 640MiB/60초입니다. 첫 새 Range 134,217,728·새 256범위·결과 769chunk를 확인합니다. 원본 후검사 미완료/실패·부분 본문·손상·import 검증 중단은 unusable로 유지합니다. 정상 상한에서 멈추고 후검사를 완료한 연속 prefix는 usable일 수 있습니다. Catalog 256/16,000/6MiB, audit 897/aggregate 출력 16MiB/결과 ZIP 17MiB를 지키며 기본 disk 합 705MiB 외 사전검사/환경 공간도 필요합니다.

고정 코드/ZIP/notebook 검토·커밋 뒤 테스트 세션에 한 JOB를 인계합니다. 목표0·부분·오류도 두 ZIP/보고서를 보존하고 종료하며 cold 재수신·자동 반복·192 이후 확대는 하지 않습니다. 아래 API8 절차와 JOB는 이력이며 새 고정 인계로 대체합니다.

## 종료된 API8 bounded warm128 인계 준비 기록

이 절은 API8 준비 당시 기록입니다. 이후 실제 warm128의 cache 재사용·확장 성공과 목표 pose0개를 확인했으며,이 절의 미실행/미배포 표현은 당시 상태를 가리킵니다. 현재 새 실행 안내는 위 API9 고정 인계를 따릅니다.

`posePrefixWarm128MiB_v2`는 실제 cold가 남긴100,561,588bytes/384chunk의 cache를 재사용하는 별도 고정 JOB입니다. 실제 원본 warm은 아직 실행하지 않았습니다. 새 network40MiB·논리128MiB·준비420초·392요청,해제800MiB·header20,000·tar metadata2MiB/단일64KiB,cache/ZIP160MiB·640chunk·local IO260MiB·export512MiB/60초를 적용합니다. 초기 metadata는 기존 별도 예산입니다. [최신 계약](PREFIX_CACHE.md)을 따릅니다.

고정 입력 manifest SHA는 `9b253b0f4f6c8153fa17daffa7f11f8cedd6e70b66252991fffdb2b37708db3c`,prefix SHA는 `6354437b0a27b8050e8111c81a71eec0cc35d915da0852d7978d4298016e857e`입니다. 실제 ready notebook에 이 두 값과 길이를 `cache_expectation`으로 고정합니다. ZIP SHA `4d63187ff49de40b5af1843aa4ded51caed255fda9996d78c028c47bf58c3319`는 upload 보고서·테스트 인계에서 별도로 대조합니다. 원본파일 ID는 검토된 개인 ready 입력에만 둡니다.

새 Range는100,561,588부터 시작하며128MiB까지 최대33,656,140bytes/129개를 추가하는 계산입니다. Cache 입력·출력과live 원본 검사·재시도·중단·카탈로그를 함께 판정하고 목표0개도 보관 후 종료합니다. 구형cache를 수정하지 않고 새v2 cache ZIP과 결과ZIP을 분리합니다. 실제 고정 source/ready 경로·hash와 회수 위치는 검토·커밋 뒤 테스트 세션에 인계하며 아래의 종료된 JOB를 재실행하지 않습니다.

다음 개발·검토 범위는 [캐시 재사용과 후속 탐색 준비안](PREFIX_CACHE_NEXT_STEP.md)을 따른다. 현재 문서의 cold 절차는 종료된 실행 기록이며 새 warm 실행용 고정 notebook은 아직 배포하지 않았다. 사용자 실행 안내는 구현·검토 후 테스트 세션에서 진행한다.

### 2026-10-21 API7 cold 실제 결과 — 캐시·관측 보존 성공, pose 미확보

- 고정 `696c234`/API7, `archvis_pose_1_10_cache_cold_001`은 `transfer_bytes_limit`/bounded_stop으로 종료했다. 준비311.326445초·응답body100,630,708bytes·1,152요청(media384+metadata768)이며 초기 metadata는 별도1요청·290bytes·0.145105초다. 시간/요청 상한이 아닌 body 예산에서 멈췄다. 목표 pose0개·missing2·targets_status=incomplete·sample_acquired=false이고 빈 파일목록의 verification passed는 추출 성공이 아니다.
- Cache는 `usable_prefix`, `ready_for_reuse=true`, cold/input_prefix0이다. byte0부터100,561,588bytes·384개 연속 범위가 모두206 완료했고 전후 metadata768개도200 완료했다. 개별 chunk·전체prefix hash, 원본 identity와 고정16모듈 producer hash를 독립 대조했다. 실제 warm import/재생/전송 절감은 이번에 실행하지 않았다.
- 관측 카탈로그는92sequence·6,318entry·1,437,945bytes, truncated=false/unclassified0이다. Payload 완료 관측6,317개와 마지막 Carla `024/Dodging` frame47 미완료1개를 포함하며 저장payload는0이다. 관측은 bounded_observation_only이고 전체목록·내용검증·seek index가 아니다. 기록된 sequence에 Serena는 없지만 전체 archive의 부재를 뜻하지 않는다.
- 해제551,914,001bytes·header13,061·tar metadata811,994bytes로 고정 상한을 준수했다. Local read/IO0, retry0, request_failure=null, source_validation_pending=false다. 실제 timeout 회복과 전체gzip/CRC 무결성·train_ready는 확인되지 않았다.
- 별도 cache export는1.125434초·local read301,818,194bytes로60초/384MiB 상한 이내다. 제출 결과ZIP137,598bytes와 cacheZIP100,606,124bytes의 CRC/hash·보고서 자체hash·참조, 고정JOB/config와16모듈hash를 오케스트레이션에서 대조했다. 결과ZIP SHA256은 `c4036a35fdd0e84782ba4f64b20b28cb64d02fa4570e324defa27c83fa2f926b`, cacheZIP은 `4d63187ff49de40b5af1843aa4ded51caed255fda9996d78c028c47bf58c3319`다.
- 원본2개와 검토 근거는 Git 제외 `outputs/test_archvis_pose_cache_cold_20261021/`에 보존한다. 사용자 사전검사 출력은 비어 있었다고 보고됐고 실행 notebook/PDF도 미제출이다. 따라서 사용자 런타임137개 개별검사 통과·전체7셀 상태·토글만 변경됐는지는 확인하지 못했다. Drive reports 영구보관·기존 오른쪽 성공ZIP은 미확인/미수령이며 좌표·영상·시간·split·학습 검증도 남아 있다.
- 이번 cold 시험은 종료한다. 같은96MiB 논리입력 상한에서 cache 뒤 여유는101,708bytes(약99.3KiB)뿐이므로 동일 설정 warm을 더 넓은 탐색으로 안내하지 않는다. 다음 판단은 보관 카탈로그와 cache를 근거로 재사용 검증·후속 범위의 비용을 검토하는 것이다. 원본 추가 읽기·warm 실행·예산 확대는 이번 결과 검토에서 시작하지 않았다.


## 종료된 API7 cold cache 테스트의 고정 절차

아래는 종료된 cold 시험 당시의 고정 절차입니다. `posePrefixCache96MiB_v1`과 API7의 구현·검토 후 고정 소스/ready notebook으로 테스트 세션에 인계했고, 위 결과를 회수했습니다. 절차·새 상한·산출물/별도ZIP·실패 시 보고서 보존은 [cache 실행 안내](PREFIX_CACHE.md)를 따릅니다. 아래 API6 기록의 source revision·JOB·설정은 새 테스트에 재사용하지 않습니다.

첫 시험은 기존 Serena frame1·10 pose 목표를 유지하는 **cold 한 번**입니다. `cache_input=None`,준비 network body/논리입력 각96MiB·420초·1,157요청,header16,000·해제640MiB·tar metadata2MiB·cache128MiB입니다. 초기 metadata는 별도 기존 예산이며 인증 비용은 제외합니다. 목표 확보,cache ready,카탈로그 보존,두 ZIP export를 각각 판정합니다. 미확보/실패도 결과를 보존한 뒤 종료합니다.

Warm 원본 시험은 이번 cold 셀에서 자동 실행하지 않습니다. 이후 별도 고정 JOB에서 검증된 기존 prefix의 재전송을 피하고 미보관 후속만 읽습니다. 동일96MiB profile도 실제 cold prefix가96MiB 미만이면 후속 요청을 할 수 있으므로 media0을 보장하지 않습니다. 입력ZIP과 생성cache·exportZIP·결과가 동시 존재하는 최대417MiB 정도의 disk 틀과 작은 보고서 여유,별도 export60초/384MiB IO·upload 후 local60초를 함께 안내합니다. 기존 오른쪽 성공 원본ZIP 회수·영상대조와 cache 독립 개발/검증을 구분합니다.

현재는 아래 API6 테스트가 종료된 상태입니다. 후속 판단은 [데이터 확보 전략 검토](DATA_ACQUISITION_STRATEGY.md)를 따릅니다. 아래 고정 설정과 실행 순서는 종료된 테스트의 기록이며, 새 실행 지시가 아닙니다. 새 테스트는 개발·검토 후 테스트 세션에 별도 고정 버전으로 인계합니다.

### 2026-10-21 API6 실제 결과 — 오류 없이 요청 상한 도달, pose 미확보

- `cb76a3c`/API6의 `archvis_pose_1_10_retry_001`은 `ranges_limit`/bounded_stop으로 종료했다. 준비768요청(media256+metadata512), body67,076,276bytes,232.180825초로 상한을 준수했다. 최초 metadata는 별도1요청·290bytes·0.158772초다. 전체 응답body64MiB에도 근접했으므로 요청 횟수만 늘려 해결될 것으로 보지 않는다.
- byte0부터 압축67,030,196bytes를 연속256범위로 읽었고 모두206 완료, 사전/후 metadata512개도 정상 완료했다. 목표 pose0개·missing2·targets_status=incomplete·sample_acquired=false다. verification passed/완료0개는 빈 저장목록 검증이며 추출 성공이 아니다. source_validation_pending=false, 관측은 bounded_observation_only로 유지한다.
- request_failure=null, retries_started0·events빈목록·wait0으로 실제 timeout/자동회복은 발생하지 않았다. 이전보다 길게 진행한 결과만으로 timeout 변경의 효과나 근본 원인 해결을 입증하지 않는다.
- 해제368,062,368bytes·header8652·regular4214·tar metadata541,603bytes를 관측했다. 첫128개 목록과 마지막 Kyle Dismissing_Gesture frame24(미완료)만 보존하며 전체 목록이나 목표 부재/위치를 증명하지 않는다. 전체gzip/CRC 무결성과 train_ready는false다.
- 오케스트레이션은 ZIP CRC·보고서4개 자체hash·배포14모듈hash·고정JOB/config·보고서/정책 연결·256완료Range를 독립 확인했다. 실행notebook7셀은 승인된 EXECUTE_ONCE 토글 외 동일하다. 사전검사 셀 성공 기록은 있지만 사용자 런타임115개 개별 로그는 미보존이다. 테스트 세션은 PDF5페이지도 확인했으며 일부 출력 누락 때문에 상세 수치는 ZIP/notebook을 근거로 삼았다.
- ZIP SHA256은 `4f72de49c111e2c17fe84df9f4e4e533e1350ad7c27a5f477672d38f4065bd33`이다. 원본3개와 검토 근거는 Git 제외 `outputs/test_archvis_pose_retry_20261021/`에 보존한다. Drive 영구보관·기존 오른쪽 원본ZIP은 미확인/미수령이다. Pose payload가 없어 좌표·영상 대조는 미실행이다.
- 이번 테스트는 종료한다. 다음은 단순 동일 실행 반복 대신 데이터 확보 전략을 검토한다. 더 넓은 순차 탐색의 전송/시간/요청/해제 비용과, 접근 가능한 시퀀스의 좌우 RGB·pose 교집합을 찾는 대안을 비교한다. 목표 변경이나 예산 확대·새 원본 요청은 아직 적용하지 않았다.

### 2026-10-21 API5 실제 진단 결과 — 사전 metadata 읽기 시간 초과

- `1ea5ff5`/API5, `archvis_pose_1_10_diagnostic_001`에서 `request_failure`는 `read_timeout / requests / metadata_pre / request_open`이다. 안전한 진단 회수는 확인했지만 pose 목표0개·미확보2개, `targets_status=unusable`, 검증 `not_usable`로 추출은 실패했다.
- Media58개 Range는206으로 완료했고 압축14.5MiB(15,204,352bytes)를 읽었다. 다음59번째 범위의 사전 metadata 요청에서 응답 없이 시간 초과가 발생했다. HTTP408/403/429나 서버·네트워크의 근본 원인, 이전 API4 실패 원인을 소급 확정하지 않는다. 새 media 전 실패라 `source_validation_pending=false`지만 전체 실행/관측은 사용 불가다.
- 준비58.151502초·body15,214,792bytes·175요청(media58+metadata117), 해제83,362,297bytes·header1961·tar metadata121,110bytes·저장0이며 예산 상한에 도달하지 않았다. 초기 metadata는 별도0.197897초·290bytes·1요청이다. 실행 셀236.460초는 인증 등을 포함한 더 넓은 기록으로 준비시간과 구분한다.
- ZIP CRC·보고서4개 자체hash·배포13모듈hash·ready JOB/설정hash·진단 및 보고서 연결을 오케스트레이션에서 독립 대조했다. 실행 notebook 코드7셀은 승인된 EXECUTE_ONCE 토글 외 동일하다. 사전검사 셀의 성공 종료 기록은 있지만94개 개별 로그는 보존되지 않았다. 테스트 세션은 실행PDF도 검토했으며 일부 출력이 잘려 상세 진단은 notebook/ZIP을 근거로 삼았다.
- ZIP SHA256은 `395cba63903060684fd6844f72aa0f83eaac7c771fa3e629c4720bd377fc09f0`이다. 원본 사본·검토 자료는 Git 제외 `outputs/test_archvis_pose_diagnostic_20261019/`에 보존하며 폴더명과 달리 검토일은10월21일이다. regular955개 중 첫128개와 마지막 Adanna Walk_Left frame6 관측도 untrusted 진단이며 seek index가 아니다.
- 이번 테스트는 종료한다. 다음 검토 후보는 전체 준비 예산 안에서 요청별 timeout과 제한된 재시도 정책을 설계·로컬 재현하는 것이다. 아직 적용하거나 새 실행을 요청하지 않았다. Drive 영구보관·기존 오른쪽 원본ZIP은 미확인/미수령이고, pose 내용·좌표/영상·실제 학습은 미완료다.

### 2026-10-19 pose 재검증 결과 — 요청 예외로 미확보

- `4571f30`/API4, `archvis_pose_1_10_density_001`의 실제 Colab 실행은 `authorized_request_failed`로 종료했다. 목표 JSON0개·미확보2개, `targets_status=unusable`, `source_validation_pending=true`다. 작업의 finished 상태는 추출 성공이 아니다.
- 압축9.5MiB(9,961,472bytes)를 읽고 준비41.529793초·116요청(media39+metadata77)·응답body9,968,402bytes에서 중단했다. Header1293·tar metadata80,643bytes·해제54,640,539bytes로 이전 제한은 넘었지만 새 제한에는 도달하지 않았다. 최초 metadata는 별도1요청·290bytes·0.127415초다.
- 39번째 media Range [9,961,472, 10,223,615]는 수신0bytes·응답 상태 코드 없이 실패했다. `session_opener`가 요청/응답 래핑 예외를 공통 오류로 바꾸므로 인증·통신·timeout 중 원인을 확정할 수 없다. `invalid_or_changed_source` 분류도 실제 원본 변경의 증거가 아니다.
- 처음128개 목록과 마지막regular 관측(총626개 중 Adanna Cards frame9, payload 미완료)은 `untrusted_failed_source_or_parser` 진단이다. 검증된 파일이나 seek index로 사용하지 않는다. 실제 pose 내용·좌표·영상 대조, 전체 archive 무결성·학습 준비는 미완료다.
- 오케스트레이션은 ZIP CRC·보고서4개 자체hash·배포 소스12개 hash·ready JOB/설정hash·보고서 연결을 독립 확인했다. 원본 ZIP SHA256은 `6e554d053340f9d074c3f486642bc68cc0455989bcd965432dfd1255e9f2fc99`다. 원본 사본·검토 근거는 Git 제외 `outputs/test_archvis_pose_density_20261019/`의 `verification.json`, `orchestrator_review.json`에 보존한다. 이번 실제 실행의80개 사전검사 전체로그·실행notebook·Drive 영구보관은 미확인이다.
- 동일 실행 반복은 종료한다. 다음 개발 검토는 비밀정보 없는 예외 종류·발생 단계 진단과 로컬 재현이다. 원시 예외문·token·URL을 기록하지 않으며 예산/timeout 확대나 자동 재시도는 이번 결과만으로 적용하지 않는다.

## 종료된 테스트의 절차: metadata 사전 요청 제한 재시도

2026-10-21 사용자 승인으로 새 API6의 opt-in `metadataPreTimeout10sRetry1Global3_v1`을 검증한다. 기존 API5 실행은 위 결과로 종료됐다. 새 정책은 metadata_pre의 요청 개시 timeout만 최대10초로 늘리고, 직접 확인된 connect/read timeout 무응답에 한해 pre당 추가1회·전체 추가3회와0.5초 대기를 허용한다. 남은 전체 준비시간으로 timeout/대기를 제한하며 초기 metadata·media·post·본문에는 적용하지 않는다.

대상은 그대로 Serena `Getting_Up__1_` frame1·10의 pose JSON이다. 전송64MiB·300초·768요청과 저장2파일·16MiB disk는 유지한다. 기존 smoke와 RGB용 counterpart 프로파일은 바꾸지 않는다. 실제 pose 위치·이번 예산 내 확보 여부는 미확정이며, 성공·부분·0개·오류 모두 한 번의 실행 결과를 보관한 뒤 중단한다.

기존 오른쪽 사용자 결과 ZIP과 새 pose ZIP을 회수하면 관절·좌표·GT2D 영상 대조는 로컬에서 진행한다. 사용자가 추가 진단 셀이나 변환기를 실행할 필요는 없다. 이번 테스트의 추출 성공과 이후 데이터 내용 정합성 검증 완료를 구분한다.

## 고정 설정

- `request_policy="metadataPreTimeout10sRetry1Global3_v1"`을 미리 채운다. 다른 profile로 옮기거나 숫자를 임의 변경하지 않는다.
- `job_id="archvis_pose_1_10_retry_001"`, `mode="extract"`, `purpose="targeted"`, `profile="poseCounterpart64MiB_v1"`.
- `selection.kind="pose"`, `total_parts=1`, index0의 `ArchVisInterior_ArchVis_RT.tar.gz000`을 사용한다. RGB archive와 파일명이 같아도 Drive ID와 원본이 다르다. 개인 Drive ID는 검토한 사용자용 notebook에만 채우고 저장소에는 기록하지 않는다.
- 과거 pose metadata의 크기368,145,316bytes/version1은 참고 이력이다. 실행 당시 metadata와 각 Range 전후 검사를 다시 거치며 과거 값을 현재 상태로 간주하지 않는다.
- 아래 두 exact 경로를 `exact_targets`와 `includes`에 동일하게 지정한다. 기존 JSON 규칙에서 파생한 후보 경로이며 아직 이번 sequence의 실제 pose payload를 확보한 것은 아니다.

```text
ArchVisInterior_ArchVis_RT/Day/rp_serena_rigged_004_ue4/SKM_MenReadingGlasses_Shape_01/003/Getting_Up__1_/json/frame_1.json
ArchVisInterior_ArchVis_RT/Day/rp_serena_rigged_004_ue4/SKM_MenReadingGlasses_Shape_01/003/Getting_Up__1_/json/frame_10.json
```

## 예산과 중단

| 항목 | 상한 |
| --- | --- |
| 준비 응답 body 합계 | 64MiB, media와 전후 metadata 포함 |
| 준비 시간 / 총 요청 | 협력적300초 / 768회 |
| 해제 출력 / tar header | 512MiB / 10,000 |
| Tar metadata 누적 / 단일 payload | 1MiB / 64KiB |
| 개별 파일 / 준비 disk | 4MiB / 16MiB |
| 저장 목표 | 정확한2개 |
| HTTP Range / 내부 IO | 256KiB / 64KiB |
| 최초 metadata builder | 별도128KiB·30초·8요청 |

두 단계 합계는64.125MiB·협력적330초·776요청이다. 인증·업로드·사전검사·ZIP 생성·전체wire 비용은 별도이며 OS hard deadline/RSS 상한이 아니다. 결과ZIP은 최대17MiB와 그에 필요한 별도 디스크 공간을 사용한다. 이전 pose 관측을64/7.5배로 단순 환산하면 header약8,534개·metadata약534,110bytes·해제약342MiB·약255초다. 이를 참고해 여유를 둔 설정이며 분포·목표 위치·실행 시간을 보장하지 않는다. 어느 상한이든 먼저 도달하면 종료한다. 512MiB는 누적 해제량이며 전체를 RAM이나 disk에 보관하는 크기가 아니다.

새 실행은 byte0부터 다시 읽는다. 중간 gzip 조각의 독립 해제·seek·resume가 아니다. 두 목표를 완전히 저장하면 즉시 종료하고, 미확보·부분·오류도 보고 후 중단한다. 정책 내 자동 재시도 외에 사용자나 테스트 세션이 예산·필터·코드를 바꾸거나 job을 반복 실행하지 않는다.

최대10,000개의 중복 검사 경로는 각UTF-8 1024bytes 이내이며 Python 문자열·casefold·set overhead는 별도다. 내부 IO/해제 chunk는64KiB이고 단일 tar metadata도64KiB로 제한한다. 프로세스 전체 RAM의 하드 상한이라고 해석하지 않는다.

API6에서도 보고서는 처음128개 관측과 별도로 마지막 regular 관측1개·관측한 regular 총수를 기록한다. 목록 cap 뒤 진행 확인용이며 source 실패 때는 신뢰하지 않는다. Seek offset이나 전체 목록으로 사용하지 않는다.

진단 필드 `request_failure`는 `category`, `library`, `operation`, `stage`와 schema 버전만 허용한다. 요청 열기·응답 wrapping·본문 읽기·응답 검사·정리 단계와 초기/전후 metadata·media를 구분한다. 원시 예외문이나 URL은 기록하지 않는다. 기존 보고서의 필드 부재와 새 정상 실행의 null은 구분하며, HTTP 상태만으로 추정한 인증 실패를 예외 타입 진단과 혼동하지 않는다.

Library의 숨은 재시도를 막기 위해 새 정책은 media 및 인증 session의 adapter retry0을 검사한다. 별도 인증 session도 retry0으로 만들어 주입한다. 기존 기본 정책은 유지한다. Timeout은 socket/협력적 상한이며 OS hard deadline은 아니다.

## 사용자 실행과 결과 회수

실행 안내와 중간 질문은 이 프로젝트의 **테스트 세션**이 담당한다. 검토한 고정 revision의 사용자용 `archive_prepare_pose_retry_ready_<revision>.ipynb`와 직접 소스 `rewind-archive-source-<revision>.zip`을 전달한다. 이전 왼쪽용 notebook을 재실행하지 않는다.

1. 기존 결과를 보관하고 새 CPU 런타임에서 전달된 pose notebook을 연다. GPU는 필요 없다.
2. 첫 셀에 직접 소스 ZIP 한 개를 업로드한다. 결과 ZIP이나 바깥 배포 묶음은 소스가 아니다.
3. 전달된 수정본의 사전115검사 전체 로그를 보관한 뒤 미리 채운 JOB와 preflight의 pose kind·두 경로·고정 예산을 확인한다.
4. `EXECUTE_ONCE=True`로 실행 셀을 한 번 실행한다. 인증→metadata→pose 추출 순서이며 별도 선택 JSON이나 새 셀이 없다.
5. 결과 검토 셀과 `DOWNLOAD_RESULT_ZIP=True` 셀을 실행해 `archive_job_archvis_pose_1_10_retry_001.zip`을 보관한다. 실패했어도 가능한 보고서를 보존한다.
6. 결과 ZIP·사전검사 로그·마지막 출력·실행 notebook을 테스트 세션에 전달한다. Drive 프로젝트 `reports/`의 영구보관 여부는 별도로 기록한다.

## 성공 판정과 후속 로컬 검토

- 재시도 발생 시 각 시도·대기·요청 계수와 최종 회복/소진 상태를 확인한다. 최종 request_failure가 null이어도 이전 실패 이력은 남아야 한다. 재시도가 없었던 성공은 해당 실행 성공이며 실제 timeout 회복 검증과 구분한다.

- 진단 검증과 추출 성공을 구분한다. 오류가 발생하면 안전한 예외 분류·발생 단계가 보고서와 ZIP에 남았는지 확인한다. unknown도 유효한 제한적 진단이며 원인을 추측하지 않는다. 오류가 재발하지 않았다는 사실만으로 과거 원인이 해결됐다고 판정하지 않는다.

- 추출 성공: pilot의 `verification.status=passed`, `targets_complete=true`, `targets_status=complete`, 정확한 두 완료 경로·미확보0 및 현재 파일 hash 검증 통과. `stop_reason=targets_complete`만으로 성공 판정하지 않는다.
- 하나만 완전하면 부분 확보,0개면 정한 범위에서 미확보다. 전체 archive에서 pose가 없다는 결론은 내리지 않는다. 원본 변경·후검사 미완료·parser/CRC/hash 오류면 unusable 상태를 우회하지 않는다.
- 회수 뒤 로컬에서 JSON schema·87관절·필수값·단위/좌표 관계, 기존62관절 변환 규격 적용 가능성, 좌우 RGB와 sequence/frame 대조 및 GT2D overlay를 검사한다. 기존 오른쪽 원본 보존본을 먼저 찾아 hash를 확인하고, 없다면 재다운로드보다 보관본 전달을 우선한다. 보관본이 없을 때 필요한 안내는 테스트 세션에서 취합한다.
- 로컬에서 찾은 이전 공식 서버 right10 추출본은 hash가 과거 사용자 결과와 같아도 해당 실행이 `http_body_failed`/unusable이므로 검증된 보존본으로 대체하지 않는다. 기존 사용자 오른쪽 ZIP(`rewind_pilot_2c11fc10.zip` 또는 동일 파일을 확보한 보관본)을 테스트 결과와 함께 회수하거나 누락으로 남긴다. 원본을 다시 읽는 작업은 이번 pose 테스트에 추가하지 않는다.
- 파일명 대조와 overlay만으로 timestamp·calibration·원본 motion·독립 split 검증을 모두 완료했다고 표시하지 않는다. frame1·10은 연속 clip이 아니며 학습을 시작하지 않는다. 해당 pose 확보·변환·영상 대조는 현재 미실행이다.

이전 왼쪽 확보 결과는 [로드맵](ROADMAP.md#2026-10-17-대응-왼쪽-rgb-실제-확보-결과), 고정 예산과 보고서 계약은 [압축 준비 도구](ARCHIVE_PREPARATION.md), 변환 규격은 [실제 pose 준비](REAL_DATA_PREPARATION.md)를 따른다.
