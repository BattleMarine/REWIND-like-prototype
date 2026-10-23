# 보존128MiB cache에서192MiB까지의 bounded suffix

2026-10-23 · 기준 `2c5e334` · API9 개발 계약. 실제192MiB Drive 실행은 하지 않았다.

## 근거와160/192 비교

[실제 warm128 검증](../outputs/test_archvis_pose_cache_warm128_20261021/verification.json)과 [보존 독립 검토](../outputs/test_archvis_pose_cache_warm128_20261021/orchestrator_cache_retention_review.json)를 기준으로 한다. 폴더 날짜20261021은 고정 인계명이며 검토일은10월23일이다. 두 ZIP은 Git 제외 보관물이다.

- Cache는 논리128MiB(134,217,728bytes)·513chunk이며 기존 v1 prefix를 그대로 포함한다. 실제 warm 준비112.094초,그중 `remote_range_seconds=102.287`초다.
- 목표 pose0개,카탈로그118sequence·8,434entry,잘림/미분류0이다. 빈 저장목록 검증은 목표 성공이 아니다. 동일128MiB warm을 반복하면 새 탐색량은0이다.
- 원본 전체 부재나 다른 sequence의 대응 RGB 비용을 증명한 결과가 아니다. 기존 Serena frame1·10 목표를 유지하며 좌표·영상·시간·split·학습 검증은 남긴다.

| 논리 목표 | 새 압축량 | 새Range / 정상HTTP / retry3 포함 | 결과chunk | 실측 remote 비례 참고 |
| --- | --- | --- | --- | --- |
|160MiB |32MiB |128 /386 /389 |641 |약102초 |
|192MiB |64MiB |256 /770 /773 |769 |약204초 |

Remote 비례는 `102.287×새bytes/33,656,140` 계산이다. Cache import·기존128MiB 재생·약702MiB 해제·parser/저장비는 별도이며,420초 내 도달이나 목표 위치를 보장하지 않는다.160은 진행 폭32MiB,192는64MiB다. 이미 재사용 경로가 실측된 점과 제한된 추가 비용을 근거로192 한 번을 선택했다. 이후 자동 확대는 없다.

현재 밀도를 그대로 외삽한192의 전체 해제는 약1,053.7MiB,header26,117,tar metadata1,620,959bytes다. 이는 분포 가정이며 안전상한 보장이 아니다. 고정1200MiB/30,000/2MiB 중 어느 한계든 먼저 닿으면 보관 후 종료한다.

## 고정 구현 계약

| 항목 | 새 계약 |
| --- | --- |
| Profile / format / policy | `posePrefixWarm192MiB_v3` / `bounded_prefix_cache_v3` / `pose_prefix_warm192_v3` |
| 새 network body / 논리입력 |80MiB /192MiB |
| 준비시간 / HTTP |협력적420초 /773회 |
| 초기metadata |기존 별도128KiB·30초·8회; 합계80.125MiB·450초·781회 |
| 해제 / header / tar metadata |1200MiB /30,000 /누적2MiB·단일64KiB |
| Cache와각ZIP / chunk / local IO |224MiB /896개 /388MiB |
| 별도export |60초 /local read640MiB |
| Catalog / audit / 일반출력 |Sequence256·entry16,000·6MiB /897행 /16MiB |

80MiB는64MiB 새media와최악metadata8.03125MiB,예약 여유를 포함한다.773은 cache live pre/post2+256범위×3+retry3다. 입력은128MiB 이상과 `cache_expectation`의 manifestSHA·prefixSHA·정확한 길이가 필수다. 누락·불일치·변경·손상이면 실패하며 cold 재다운로드로 대체하지 않는다.

고정 시험 입력은 manifest SHA `de57a189205a4ddeb7dc61379e09fca08b4d5cd5ec033d0b7a8d2b50abc775c1`,prefix SHA `2e05f6850bd5a019b232de3383efc03e91982bb38681523f0a2ac9c824029268`,134,217,728bytes다. ZIP 자체SHA와 원본identity는 테스트 인계에서 추가 대조한다. Generic source에는 개인 파일ID를 넣지 않는다.

## 이전 상수를 바꾸는 방식과 구분

기존 v1/v2 profile·schema·상한을 유지한다. 새 reader는 각 구형 cache를 그 판의 상한으로 검사하고,신규 출력만 v3로 식별한다. 기존 v2 reader는 v3를 거부한다. Source는 byte0부터 다시 해제하며 미보관 후속 범위만 원본에서 읽는다. Gzip 상태 resume나 전체 archive index가 아니다.

- 기존v2 logical128/cache160/chunk640/local260/export512를 전역으로 올리지 않고 별도 고정정책을 추가했다.
- V3는 cache224/chunk896/local388/export640을 사용한다.192MiB 본체3pass만576MiB이므로 기존512MiB export 상한은 부족하다.
- 최소 입력128MiB가 HTTP773 전제다. 더 작은 입력으로 같은192 도달을 보장하지 않는다. 고정 시험의 첫 새 Range는 offset134,217,728이어야 한다.
- 원본 pre/post,strict206,부분chunk 배제,hash·identity 검사,중단/원본/CRC/parser 실패의 unusable,재시도와 단계 계측을 유지한다.

## Audit와16MiB 출력의 제약

Audit1024행을 유지하면 실제 고정 manifest/목표에서 보고서예약+catalog6MiB+payload8MiB가16MiB를112,220bytes 초과했다. 새정책에서는 **최대896개 cache행+실패1개=897행**으로 제한하며 HTTP와 분리해 예약한다. 같은 실제입력 계산은17,828bytes 여유다.

이는 모든 최장1024byte 경로/manifest에서 두 파일의4MiB 최대치를 동시에 보장한다는 뜻은 아니다.16MiB는 aggregate 상한이고 개별4MiB 상한보다 먼저 제한할 수 있다. 동적 예약과 `payload_disk_bytes_limit`으로 안전하게 멈추며 출력·결과 ZIP 상한을 늘리지 않는다. 이전 관측 pose의 약85KB 규모를 참고하면 여유가 있지만,미확보 Serena 목표의 실제 파일 크기는 미확정이다. 긴 경로와 aggregate 중단을 회귀에서 확인한다.

InputZIP224+생성cache224+exportZIP224+일반출력16+결과ZIP17=일반 단계상한 합705MiB에 작은 보고서·코드·환경 여유가 추가된다. 실제 입력은 이보다 작지만 일반 upload cap을 작은 것으로 주장하지 않는다. Upload/UI·export·영구보관과 전체wire 비용은 원본body 예산과 분리한다.

## 검증과 인계

합성 fixture는 임시 폴더에 스트리밍 생성하고 검사 후 제거한다. 큰 gzip/cache/ZIP을 evidence에 남기지 않으며 작은 script·log·hash만 보존한다. 실제 cache와 기존 파일은 삭제하지 않는다. 사전검사의 fixture CPU/RAM/임시disk·시간은 원본 준비420초/705MiB 틀 밖이다.

확인 범위는 v2→v3 import/export,192MiB/769chunk,첫후속offset,목표저장/미확보,신판 재import,구버전 거부,source변경/post실패/부분본문/import중단의 unusable,계측·audit·aggregate 한계와 기존 notebook 경로다. 근거는 `outputs/prefix_cache_warm192_20261023/`에 둔다. 원본 네트워크와 학습은 수행하지 않는다.

고정revision·소스ZIP/ready notebook SHA·입력ZIP/manifest/prefixhash·원본snapshot·모든 단계상한·임시공간·중단조건·두ZIP 보관위치를 오케스트레이션 검토 후 테스트 세션에 전달한다. Cache/관측/목표를 각각 판정하고0개여도 그 JOB를 종료한다.192 이후 범위·전체index·새sequence/RGB·학습은 별도 판단이다.
