# ArchVis 한정 사전 인덱싱과 g16 후속 준비

> **보존 기록 · 2026-11-05 정리:** 2026-11-04 사용자 결정으로 분할 압축 원본의 인덱싱·부분 읽기 운영을 중단했다. 이 문서 전체의 실행 명령, 다음 단계, 예산과 미완료 표시는 해당 실험 당시의 기록이며 현재 실행 지시가 아니다. 새 기본 경로는 [압축 해제된 RGB·pose 사용](DATA_ACQUISITION_STRATEGY.md)이다. 기존 코드·원본·인덱스·예약 기록은 보존하며 자동 재개하지 않는다. 이동된 로컬 근거는 [보관·복원 안내](LOCAL_STORAGE.md)를 따른다.

## 2026-11-03 실제 g25 후속 상태

실제 job002는 g25/782member·stereo 관측371/pose 관측102까지 저장했다. 9개 snapshot 누적 후 다음 총 예약809984041bytes가768MiB를 넘어 정상 중단했으며 checkpoint 오류 재발이 아니다. 기존494개 prefix와 hash/CRC/catalog를 독립 대조했다. 별도 local compact helper는 원본·receipt를 바꾸지 않고 DB55.4→28.3MB를 확인했으나 index32MiB cap은 해결하지 않는다. 새 RGB job은 준비하지 않으며 정확 미관측 pose269 한 건의 warm 확보만 다음 실측으로 인계한다. 장기 append/CAS 구조는 미구현 설계다. [상세 분석·경계·검증과 예산](G25_STORAGE_AND_POSE269.md)을 따른다. 아래 g16 job002 준비 기록은 당시 상태다.

## 2026-11-03 실제 job001과 g16 후속 job002 준비

직전 `archvis_rgb_preindex_g9_001`은 고정 g9/270member부터 g16/494member까지 정상 저장했다. Root가 ZIP CRC·DB/전체 catalog hash·기존270개 prefix 보존과 split/key 재집계를 독립 확인했다. Stereo는102→224개(+122), pose 관측 교집합은102개로 유지됐으며 validation RGB0·실제 추출0이다. 전체 인덱싱·학습 준비 완료가 아니다.

실측은 body463560747bytes·1411HTTP·wall430.998초/process CPU18.562초였다. HTTP 호출~headers wall411.818초, 본문 read1 wall2.538초, snapshot export7회 wall5.487초다. 범위가 겹치므로 순수 network/gzip 시간을 빼서 계산하지 않는다. Media442회와 metadata969회는 uncached range마다 pre/post884회+최초5회+8scan의5source 전후80회로 설명된다. Identity 검사를 생략하는 최적화는 하지 않는다.

8번째 scan은 `checkpoint_disk_limit`으로 중단했다. 당시 runner가 약144MiB의 DB용 계산값을 core의 checkpoint peak 한도로 잘못 사용했다. Core 실제 수식은 `3*(기존 DB+새 state+새 index blob+16384)`이며 g16 동일 state조차153616008bytes로 당시151060480bytes 한도를 초과한다. 회수된 정상 g16은 보존했고 실패 scan을 새 coverage로 올리지 않는다.

후속 `archvis_rgb_preindex_g16_002`는 이 공간 예약을 고친 **실행 전 준비본**이다. 원격 요청0으로 g16 복사본의 기존 오류를 재현한 뒤 같은 state의 commit→snapshot→회수를 확인했다. 이는 generation만 증가시킨 오프라인 fixture 검사이며 새로운 원본 member 탐색이 아니다. Scan 후 DB가64MiB를 초과하면 snapshot 할당 전에 정상 중단하고 이전 receipt를 회수하는 검사도 통과했다. 768MiB 전체 상한과 기존 index32MiB/snapshot DB64MiB 상한은 유지한다.

| 후속 고정 값 | 값 |
| --- | --- |
| 실제 job001 결과 ZIP SHA256 | `ddc3762406e35ac6faa2353f3e7afff1bb9c4c2e58377b9e834bcabd7a049423` |
| g16 snapshot SHA256 | `2842dea84a61251340880455393d822d4cf0b0e3275a801cf37a72638160be4c` |
| g16 receipt 파일 SHA256 | `c8533610fa5927cf9909ba6e2dc9258a023468f15071ba084ec78780c51bb316` |
| Job002 input ZIP SHA256 | `9faa33c20328f9c5a3da49cc85a93d2c5710ddaa4b35d9e9bff556d86f8de0d4` |
| Job002 config canonical SHA256 | `ef9a45fd4074f9af0282b64e1940b5021a5632e44ce1e21ccb3840aa1ec424f4` |

새 입력은 `outputs/preindex_g16_20261103/developer/archvis_preindex_g16_inputs.zip`,17675368bytes다. 기존 split·최신235MiB pose catalog·metadata/source pin은 불변이고 g16 ZIP/receipt 원본 byte를 복사했다. 새 notebook은 [archvis_rgb_preindex_g16.ipynb](../notebooks/archvis_rgb_preindex_g16.ipynb)다. 구 입력·notebook·실제 결과는 보존하며, root가 새 source/ready SHA·revision·새 Drive 폴더를 고정한 뒤 탐색 세션에 인계한다.

Job002의 canonical config에만 `runtime_cache_bytes=16777216`을 명시한다. Run/preflight는 preindex에서4/16MiB만 허용하고 기존 기본과 구 입력은4MiB다. Range1MiB·retry0·uncached pre/media/post identity 검사는 유지한다. 동일 합성 source/read trace에서 cache4MiB의103HTTP(media31)→16MiB의58HTTP(media16), body31534480→15799780bytes로 줄었고32MiB는 추가 이득이 없었다. 같은160member/catalog SHA·read40110454bytes가 일치했다. 이 근거로 최소 유효16MiB를 선택했으며 실제 Drive 개선율을 입증한 것은 아니다. Root도 모의 결과를 독립 재현했다.

개발 최종35개 검사 failure/error/skip0 통과, root는 33개 검사와 추가된 notebook 검사 2개를 독립 실행해 총35개를 failure/error/skip0으로 통과했다. 로그·g16 회귀·cache 비교·입력 pin은 `outputs/preindex_g16_20261103/developer/`, root 검토는 같은 상위 폴더 `root/`에 보존한다. 다음 실행도 index32MiB·snapshot DB64MiB·전송/시간/누적 저장 공간에 도달할 수 있다. Cap에 도달하면 같은 job을 반복하지 않고 마지막 정상 receipt와 결과를 root에 인계한다. 새 coverage·성능 개선·validation 발견·전체 확장 해결은 아직 미검증이다.

## 초기 g9 준비 기록과 공통 계약

2026-11-03 초기 준비 당시 기록이다. 학습과 독립된 CPU Colab 작업을 준비했다. 기존 discovery runner로 고정 g9부터 순차 탐색을 재개하고 배치별 immutable snapshot·정상 receipt를 Drive에 저장한다. 이번 개발에서는 원격 요청·학습·다운로드·설치를 하지 않았다. 이후 실제 job001 결과와 최신 job002 준비는 위 기록을 따른다.

## 선정 근거와 고정 입력

ArchVis pose의 최신235MiB prefix 관측에는 완전 payload를 반환했던15465 key가 있다. 공식 split 대조 결과 train14134/validation1331이며 validation은20sequence다. 현재 g9의 stereo102개는 모두 최신 pose 관측 key와 대응한다. 이는 기존 조기종료 목록의35대응을 갱신한 결과다. **관측 payload_complete는 현재 저장된 pose 파일이나 seek index를 뜻하지 않는다.** RGB validation의 tar offset은 모르며 파일 순서를 추정해 점프하지 않는다. 이번 예산 안에서 validation 발견을 보장하지 않는다.

| Pin | 값 |
| --- | --- |
| Source fingerprint | `73b7007fc0586e72ddc0109290c79af93b7e6954c2f65a3e708bd6e8917b2c37` |
| g9 snapshot SHA256 | `ec213878a03784b22fa7e6d53f26f16a61409aa8854228e177fd97f4e2101f8b` |
| Receipt 파일 SHA256 | `3d4c6c11953318661b5cf905155b7e0384cb87430fe9f97192dff5df4f7351e9` |
| 최신 pose catalog 파일 SHA256 | `180aa3d7cff7a9df6fed9a5e04b3798ad259dc007d2d8dd03f35ae1091b635ec` |
| Catalog content SHA256 | `965d8562d82dc844a5cbfabb7f0b0115fe098ad61f72e20e8b52c968a84ac3f1` |

Pose 목록은 보존된 실제 pose67 결과 ZIP SHA `7ebe1cea2b88e7945a380068e0ac7d3fc26e34794437900f896eae0482865211`의 `archive_job_pose_missing67_001/pilot/observation_catalog.json`이다. 기존 원본/입력 pin·g9 snapshot/receipt·과거 실행 revision은 보존한다. 복원 API가 local DB에 신뢰 anchor를 기록하는 동작과 snapshot 내부 원본 DB를 구분하며 원본 DB를 새 원본으로 재라벨하지 않는다.

보존된 초기 g9 입력은 `outputs/preindex_20261103/developer/archvis_preindex_inputs.zip`,9382606bytes/SHA `e7e46e17921e54afc78c7b2e7cdcdb7cd9bbc511adfd547f47ccbececa4a24f1`이다. Config canonical SHA는 `1a94db146a28aeddac51fc7cf59e82661358ff15ed3158aeda422de5d0ea7d44`다. Config·고정 metadata·공식 split·최신 pose catalog·g9 ZIP/receipt·개별 파일 pin을 포함한다. 입력의 g9는270member이며 RGB stereo102/validation0이다. 준비 근거는 같은 폴더 `input_preparation.json`, root 선정 근거는 `outputs/preindex_20261103/root/source_priority.json`이다.

## 명시 profile와 한 번의 예산

기존 `DiscoveryLimits` 기본256MiB를 유지한다. `archvis_preindex512_v1`과 `PreindexLimits`를 함께 지정하는 별도 입구만 다음 예산을 허용한다. Source metadata·복원·scan·snapshot을 같은 job 예산에 포함하며, batch마다 body/HTTP/read/decode/time 잔여를 공유한다.

| 항목 | 상한 |
| --- | --- |
| 응답 body / HTTP / monotonic wall | 512MiB / 4096 / 1200초 |
| Source read / 반환 decode | 1GiB / 512MiB |
| 협력 RSS | 1GiB |
| Work+Drive snapshots+회수 ZIP | 논리 합768MiB |
| Batch / 횟수 | 32member / 최대32회 |
| Scan 전 시간 reserve | 30초 |
| 실행 전 가용 RAM / 임시 disk | 각각2GiB 이상 |

준비 upload·source/input 해제·기존 index 의존성 설치·인증/Drive mount는 phase 밖이다. 이 비용과 Drive 동기화의 보이지 않는 overhead를0으로 간주하지 않는다. Snapshot export는 local backup/해시/ZIP·Drive publish/재검증을 포함한다. Remote 원본 archive에는 읽기만 하고, 새 별도 MyDrive 폴더에 닫힌 snapshot/receipt를 쓴다. 저장 호출이 반환됐다는 사실과 새로운 runtime에서 영구 보관본이 실제 복원됐다는 검증을 구분한다.

공간 예약은 후속 수정본 기준으로 **scan 전**에 계산한다. 다른 영속 파일 크기(현재 DB 제외)와 회수 ZIP 최대72MiB·보고4MiB를 전체768MiB에서 뺀 잔여만 core checkpoint peak 예산으로 전달한다. 필요한 checkpoint 상한은 `3*(현재 DB+state cap16MiB+index cap32MiB+16384)`다. Snapshot peak는 export 가능한 DB64MiB+backup64MiB+temp/publish ZIP72MiB씩, 합272MiB다. 두 단계는 순차 실행하므로 큰 쪽의 peak와 영속 파일·회수 여유를 함께 검사한다. 이 값을 단순 live DB 크기 cap으로 사용하지 않는다.

Scan 후 실제 DB가64MiB를 초과하면 snapshot을 새로 할당하지 않고 마지막 정상 receipt를 회수한다. Checkpoint peak 예약은 이 경우의 작업 DB와 회수 공간도 감당한다. 부분 publish나 과거 snapshot을 자동 삭제하지 않는다. 예약이나 실제 capacity 때문에 body512MiB/32batch 전에 정상 중단할 수 있다. 초기 구현의 잘못된144MiB checkpoint cap은 위 실제 실패 기록으로 남기고 후속 수정본에서 사용하지 않는다.

현재 core의 index32MiB·state16MiB·member20000·point4096·snapshot DB64MiB 한도는 유지한다. 이번 도구는 g9 뒤의 bounded 구간을 측정·저장하는 준비이며 전체3TB를 무제한 끝까지 처리하는 기능이 아니다. 더 큰 규모는 실측과 저장 구조/상한 검토 후 별도 결정한다. GPU 전환·병렬화·압축 재포맷은 구현하지 않는다.

## 계측과 해석

`rewind_discovery.timing.Timing`은 monotonic wall과 `process_cpu_seconds`를 기록한다. 전체 job과 input validation/session factory/최초 metadata/snapshot restore/scan/snapshot export의 호출 수·wall·CPU를 제공한다. 실패·중단도 finally로 누적한다.

명시 profile에서만 session의 선택 hook을 붙인다. 기존 `session.get(..., stream=True, allow_redirects=False)`부터 headers 반환까지 `http_request_to_headers`, 기존 `response.raw.read1(size, decode_content=False)` 호출 동안 `http_body_read1`을 측정한다. Response/socket `_fp` 경로·timeout·retry 검사·close 수명·전송 장부는 바꾸지 않는다. Hook이 없으면 기존 직접 호출 경로다. URL·credential·원문 예외·raw response는 계측 보고서에 넣지 않는다.

각 `scan_attempt`에는 성공/실패 상태, read/decode·body/request/cache hit/miss, 전체 및 HTTP 호출별 wall/CPU delta가 남는다. Snapshot은 batch event에 export call wall/CPU와 기존 IO bytes를 기록한다. 누적 cache 통계도 별도 유지한다. 이 범위는 서로 **겹친다**. HTTP call wall에는 SDK/auth·Python·stream 소비가 들어가며 CPU는 다른 process 작업도 포함한다. 시간을 빼서 순수 network latency나 순수 gzip 해제 시간으로 보고하지 않는다. 이번 CPU 실측을 받은 뒤 병목을 검토하며 GPU의 이득을 미리 보장하지 않는다.

## 진행·중단·회수 계약

`working_discovered`는 현재 local scan의 key 관측 수다. `durable_discovered`/`durable_members`는 정상 receipt 저장이 확인된 시점에만 갱신한다. Export 실패로 local 관측이 앞서더라도 **다음 재개는 마지막 정상 receipt 기준**이다. 신규 train coverage와 validation 발견 여부, RGB stereo와 pose 관측 교집합을 split별로 구분한다. 실제 RGB/pose 추출 triplet은0이고 학습 입력 완료를 의미하지 않는다.

공간/시간/명시 capacity 한도만 예상된 `bounded_stop`으로 처리한다. Identity 변경·hash/transport/corruption 등은 실패로 남긴다. 실패 scan의 read/decode/HTTP 계측과 마지막 정상 receipt를 보존한다. 취소·실패 뒤 자동 재실행이나 budget reset은 없다.

[archvis_rgb_preindex.ipynb](../notebooks/archvis_rgb_preindex.ipynb)는6셀/5code cell이다. 새 CPU runtime에서 다음 순서로 사용하도록 root가 고정한다.

1. Source/input SHA·revision·후속 job `archvis_rgb_preindex_g16_002`과 새 MyDrive snapshot 폴더를 고정한다. 기존 snapshot/원본 폴더는 쓰지 않는다.
2. Source/input 두 ZIP 전체 검증 후 staging한다. 정확히 같은 기존 추출본만 읽기 재사용하며 불일치는 덮어쓰지 않는다. 인증 전에 입력 pin·g9 receipt/profile을 검사하고 이전 source의 로드된 모듈은 차단한다.
3. 인증/mount 후 한 번 실행한다. 각 정상 배치 snapshot/receipt를 새 Drive 폴더에 저장한다. Validation을 못 찾거나 한도에 도달해도 자동 후속 작업을 시작하지 않는다.
4. 실패·중단·완료 모두 독립 회수 셀만 실행한다. 인증/원본 read/scan 없이 보고서·최종 정상 snapshot/receipt를 다운로드하고 ZIP SHA를 출력한다. 후속 실행에서 새 snapshot이 없으면 입력 g16 receipt로 회수한다. 준비 자체가 실패해도 작은 준비 실패 ZIP을 받을 수 있다.
5. 탐색 세션은 결과 ZIP/SHA·마지막 정상 receipt·durable/working counts·종료 이유·누적/구간 timing·전송/RAM/disk를 root에 전달한다. 원본/회수본/모든 정상 snapshot을 런타임 밖에 보존한다. 다음 새 runtime은 root가 선택한 정확한 receipt/hash/anchor로 별도 job을 준비한다.

연속 archive coverage와 epoch 종료를 구분한다. Catalog 일부·개별 파일 SHA·부분 prefix는 전체 원본 무결성/EOF/CRC 검증을 대신하지 않는다. Validation 발견도 실제 입력 추출·좌표/시간/원본 motion·독립 split·학습 release 검증을 대신하지 않는다.

## 로컬 검증 상태

가짜 clock/session으로 headers/read1 wall·CPU 합계와 no-hook 호출/close 불변, 실패 span의 기록을 확인했다. 합성 g9 역할 anchor에서 재개·누적 계측·validation 집계·중단 scan IO·snapshot 실패의 working/durable 분리·작은 disk 한도 중단 후 회수·부분 publish 보존을 검사했다. 기존 discovery/snapshot/Drive metadata 및 실제 notebook 셀·인증 factory까지 개발38개/11.083초, failure/error/skip0으로 통과했다. 최종 결과는 `outputs/preindex_20261103/developer/final_tests.txt`에 보존한다. 실제 Drive 인덱싱·Colab 병목/저장 비용·새 runtime 복원은 이번 개발에서 실행하지 않았다.

Root는 관련24개 검사를 독립 실행해 skip0으로 통과했고, 입력 ZIP CRC·전 파일의 원본 byte 일치·실제 고정 입력 preflight도 확인했다. 근거는 `outputs/preindex_20261103/root/review.json` 및 `root/tests.txt`다. 검토 후 root가 로컬 커밋·source ZIP/ready notebook의 SHA와 revision·새 snapshot 폴더를 고정해 탐색 세션에 인계한다. 개발은 커밋·push나 실제 Drive 작업을 수행하지 않는다.
