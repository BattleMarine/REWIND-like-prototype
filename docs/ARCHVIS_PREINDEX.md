# ArchVis g9 이후 한정 사전 인덱싱 준비

2026-11-03 · 학습과 독립된 CPU Colab 작업의 개발 준비다. 기존 discovery runner로 고정 g9부터 순차 탐색을 재개하고 배치별 immutable snapshot·정상 receipt를 Drive에 저장한다. 이번 개발에서는 원격 요청·학습·다운로드·설치를 하지 않았다. 실제 첫 실측은 root가 고정본을 검토한 뒤 데이터셋 탐색 세션에서 한 번 수행한다.

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

새 입력은 `outputs/preindex_20261103/developer/archvis_preindex_inputs.zip`,9382606bytes/SHA `e7e46e17921e54afc78c7b2e7cdcdb7cd9bbc511adfd547f47ccbececa4a24f1`이다. Config canonical SHA는 `1a94db146a28aeddac51fc7cf59e82661358ff15ed3158aeda422de5d0ea7d44`다. Config·고정 metadata·공식 split·최신 pose catalog·g9 ZIP/receipt·개별 파일 pin을 포함한다. 입력의 g9는270member이며 RGB stereo102/validation0이다. 준비 근거는 같은 폴더 `input_preparation.json`, root 선정 근거는 `outputs/preindex_20261103/root/source_priority.json`이다.

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

공간을 소진한 뒤 회수하지 못하는 일을 막기 위해 **scan 전에** live DB 증가·snapshot backup/temp/publish·마지막 정상 ZIP 회수·보고서 여유를 보수적으로 예약한다. Local DB bound는 기존 state16MiB+index32MiB에 대한 `3*(state+index)+64KiB` 약144MiB이며, snapshot DB64MiB/ZIP72MiB·보고4MiB를 별도 예약한다. Export가 부분 ZIP을 남겨도 기존 receipt 회수가 가능하도록 고려했다. 저장 실패 파일이나 오래된 snapshot을 자동 삭제하지 않는다. 이 예약 때문에 body512MiB나32batch 전에 정상 `bounded_stop`할 수 있다.

현재 core의 index32MiB·state16MiB·member20000·point4096·snapshot DB64MiB 한도는 유지한다. 이번 도구는 g9 뒤의 bounded 구간을 측정·저장하는 준비이며 전체3TB를 무제한 끝까지 처리하는 기능이 아니다. 더 큰 규모는 실측과 저장 구조/상한 검토 후 별도 결정한다. GPU 전환·병렬화·압축 재포맷은 구현하지 않는다.

## 계측과 해석

`rewind_discovery.timing.Timing`은 monotonic wall과 `process_cpu_seconds`를 기록한다. 전체 job과 input validation/session factory/최초 metadata/snapshot restore/scan/snapshot export의 호출 수·wall·CPU를 제공한다. 실패·중단도 finally로 누적한다.

명시 profile에서만 session의 선택 hook을 붙인다. 기존 `session.get(..., stream=True, allow_redirects=False)`부터 headers 반환까지 `http_request_to_headers`, 기존 `response.raw.read1(size, decode_content=False)` 호출 동안 `http_body_read1`을 측정한다. Response/socket `_fp` 경로·timeout·retry 검사·close 수명·전송 장부는 바꾸지 않는다. Hook이 없으면 기존 직접 호출 경로다. URL·credential·원문 예외·raw response는 계측 보고서에 넣지 않는다.

각 `scan_attempt`에는 성공/실패 상태, read/decode·body/request/cache hit/miss, 전체 및 HTTP 호출별 wall/CPU delta가 남는다. Snapshot은 batch event에 export call wall/CPU와 기존 IO bytes를 기록한다. 누적 cache 통계도 별도 유지한다. 이 범위는 서로 **겹친다**. HTTP call wall에는 SDK/auth·Python·stream 소비가 들어가며 CPU는 다른 process 작업도 포함한다. 시간을 빼서 순수 network latency나 순수 gzip 해제 시간으로 보고하지 않는다. 이번 CPU 실측을 받은 뒤 병목을 검토하며 GPU의 이득을 미리 보장하지 않는다.

## 진행·중단·회수 계약

`working_discovered`는 현재 local scan의 key 관측 수다. `durable_discovered`/`durable_members`는 정상 receipt 저장이 확인된 시점에만 갱신한다. Export 실패로 local 관측이 앞서더라도 **다음 재개는 마지막 정상 receipt 기준**이다. 신규 train coverage와 validation 발견 여부, RGB stereo와 pose 관측 교집합을 split별로 구분한다. 실제 RGB/pose 추출 triplet은0이고 학습 입력 완료를 의미하지 않는다.

공간/시간/명시 capacity 한도만 예상된 `bounded_stop`으로 처리한다. Identity 변경·hash/transport/corruption 등은 실패로 남긴다. 실패 scan의 read/decode/HTTP 계측과 마지막 정상 receipt를 보존한다. 취소·실패 뒤 자동 재실행이나 budget reset은 없다.

[archvis_rgb_preindex.ipynb](../notebooks/archvis_rgb_preindex.ipynb)는6셀/5code cell이다. 새 CPU runtime에서 다음 순서로 사용하도록 root가 고정한다.

1. Source/input SHA·revision·job `archvis_rgb_preindex_g9_001`과 새 MyDrive snapshot 폴더를 고정한다. 기존 snapshot/원본 폴더는 쓰지 않는다.
2. Source/input 두 ZIP 전체 검증 후 staging한다. 정확히 같은 기존 추출본만 읽기 재사용하며 불일치는 덮어쓰지 않는다. 인증 전에 입력 pin·g9 receipt/profile을 검사하고 이전 source의 로드된 모듈은 차단한다.
3. 인증/mount 후 한 번 실행한다. 각 정상 배치 snapshot/receipt를 새 Drive 폴더에 저장한다. Validation을 못 찾거나 한도에 도달해도 자동 후속 작업을 시작하지 않는다.
4. 실패·중단·완료 모두 독립 회수 셀만 실행한다. 인증/원본 read/scan 없이 보고서·최종 정상 snapshot/receipt를 다운로드하고 ZIP SHA를 출력한다. 초기 실패라 새 snapshot이 없으면 입력 g9 receipt로 회수한다. 준비 자체가 실패해도 작은 준비 실패 ZIP을 받을 수 있다.
5. 탐색 세션은 결과 ZIP/SHA·마지막 정상 receipt·durable/working counts·종료 이유·누적/구간 timing·전송/RAM/disk를 root에 전달한다. 원본/회수본/모든 정상 snapshot을 런타임 밖에 보존한다. 다음 새 runtime은 root가 선택한 정확한 receipt/hash/anchor로 별도 job을 준비한다.

연속 archive coverage와 epoch 종료를 구분한다. Catalog 일부·개별 파일 SHA·부분 prefix는 전체 원본 무결성/EOF/CRC 검증을 대신하지 않는다. Validation 발견도 실제 입력 추출·좌표/시간/원본 motion·독립 split·학습 release 검증을 대신하지 않는다.

## 로컬 검증 상태

가짜 clock/session으로 headers/read1 wall·CPU 합계와 no-hook 호출/close 불변, 실패 span의 기록을 확인했다. 합성 g9 역할 anchor에서 재개·누적 계측·validation 집계·중단 scan IO·snapshot 실패의 working/durable 분리·작은 disk 한도 중단 후 회수·부분 publish 보존을 검사했다. 기존 discovery/snapshot/Drive metadata 및 실제 notebook 셀·인증 factory까지 개발38개/11.083초, failure/error/skip0으로 통과했다. 최종 결과는 `outputs/preindex_20261103/developer/final_tests.txt`에 보존한다. 실제 Drive 인덱싱·Colab 병목/저장 비용·새 runtime 복원은 이번 개발에서 실행하지 않았다.

Root는 관련24개 검사를 독립 실행해 skip0으로 통과했고, 입력 ZIP CRC·전 파일의 원본 byte 일치·실제 고정 입력 preflight도 확인했다. 근거는 `outputs/preindex_20261103/root/review.json` 및 `root/tests.txt`다. 검토 후 root가 로컬 커밋·source ZIP/ready notebook의 SHA와 revision·새 snapshot 폴더를 고정해 탐색 세션에 인계한다. 개발은 커밋·push나 실제 Drive 작업을 수행하지 않는다.
