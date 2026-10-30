# 저장된 coverage 밖 확장 C pilot

2026-10-30 · 구현·로컬 모의 검증 단계. 기존 A/B 고정 배포물은 그대로 보존한다. 실제 C의 Google 요청·사용자 시험은 아직 실행하지 않았다. 실행 안내는 root의 고정 revision/소스 ZIP/ready notebook 검토 후 테스트 세션이 담당한다.

## 목적과 범위

직전 실제 A/B는32member까지 누적했지만 gzip `index_coverage=3,791,957`, index98,411bytes로 동일했다. 이는 B 저장본 복원·기존 파일 재추출의 성공이며 기존 index 범위 밖 확장 증거는 아니다.

새 helper `rewind_archive_index.extension_pilot.run_extension`은 정확한 B receipt/신뢰 anchor를 복원하고 기존 coverage를 고정한 뒤, 동일 source의 저장 index를 실제로 확장한다. B의 source fingerprint·snapshot SHA·generation·engine은 기존 import 경로로 대조한다. 개인 Drive ID나 기존 snapshot ID는 코드에 고정하지 않는다. 전체 archive 스캔·CRC·학습·대규모 catalog는 이번 범위가 아니다.

## 실행과 성공 판정

1. 입력 B snapshot 폴더와 **아직 존재하지 않는 별도 출력 C 폴더**를 지정한다. 신뢰 B-anchor SHA, source ZIP SHA/revision, manifest SHA를 고정한다. 같은 run_id의 C 시작/실패/완료 marker나 기존 C 출력 폴더는 재실행하지 않는다.
2. B를 새 local DB로 복원해 oldcoverage·전체 oldcatalog·blob hash/size·generation을 기록한다. 한 DriveRuntime을 생성하고16member씩 최대8회, 추가128member까지만 scan한다.
3. 매 저장 후 기존 catalog의 exact prefix와 물리 offset/ordinal 연속성을 확인한다. 예상 generation도 지정해 동시 checkpoint 변경을 거부한다. `boundary>oldcoverage`와 `newcoverage>oldcoverage`를 모두 만족해야 한다.
4. 저장된 binary index를 **원본 read를 금지한 metadata-only handle**에 import해 실제 restart point `(해제offset u, 압축offset c)`를 확인한다. EOF dummy를 제외하고 oldcoverage를 넘는 새 point를 찾는다. 새 catalog의 완전 JSON 중 `offset>=새 point.u`, `end<=newcoverage`를 고른다. 따라서 단순히 oldcoverage 다음 파일이나 member 수 증가만으로 통과하지 않는다.
5. 기존 선택 JSON과 새 JSON의 SHA·JSON object 파싱을 확인하고 C snapshot을 게시한다. 부모가 B이고 lineage가 같은지 검사한다. 기존 B는 덮어쓰지 않는다.
6. C를 또 다른 local DB에 exact receipt로 import한다. 같은 Runtime의 **LRU만 비우고** 전송/시간 예산은 그대로 둔다. 새 ArchiveIndex/gzip handle에서 새 JSON부터 단독 `auto_build=False`로 추출한다. 실제 source read 시작이 old index의 최대 유효 압축 point보다 뒤이고 새 point의 압축 위치와±1byte 이내여야 한다. ±1은 native bit 정렬을 허용하는 것이며 index header를 수정하지 않는다. 실제 HTTP Range 시작은256KiB로 정렬되므로 source read trace와 구분한다.
7. 별도 새 handle로 기존 JSON도 대조한다. 위 조건과 저장본 검증을 모두 만족한 경우에만 `passed`다. C에서 새 VM을 다시 만들라는 절차는 요구하지 않는다.

Catalog boundary, index point coverage, 학습 순서는 서로 다르다. JSON 이름을 문자열 정렬해 frame 순서로 쓰거나 RGB/pose ordinal을 zip하지 않는다. Snapshot을 새 local DB로 읽는 검증은 동일 C 안에서 수행하며, 새 VM의 Drive 내구성까지 다시 증명한 것으로 표시하지 않는다.

## 누적 예산과 중단

| 항목 | C 전체 한도 |
| --- | --- |
| 원본 응답 body / HTTP / shared source phase 시간 | 8MiB /120회 /300초 |
| scan | 16member×최대8회, directory 포함 추가128member |
| source read / 반환 decoded 합계 | 32MiB /32MiB |
| 전체 local ledger 시간 / 관측 RSS | 300초 /512MiB |
| workspace+새 C 산출물 / 개별 파일 | 128MiB /8MiB |
| snapshot 한 작업 | 기존 DB64MiB·ZIP72MiB·IO ledger512MiB·disk256MiB·60초 제한도 적용 |

같은 Runtime의 scan·기존/새 파일 추출·C 재복원 후 cold 추출까지 합산한다. Helper는 잔여 read/decoded/time으로 각 IndexLimits를 만들고 예외가 발생해도 실제 operation 통계를 회수한다. 새 Runtime이나 budget을 만들어 상한을 리셋하지 않는다.

Auth와 최초 B 복원은 전체 ledger 시작 전이다. C snapshot 복사·재import 비용은 각각의 snapshot 통계로 분리하되, 더 엄격한 전체 ledger300초에도 들어간다. 공유 원본 phase clock도 중간에 리셋하지 않으므로 그 사이 local 작업이 길어지면 원본 읽기 여유도 줄어든다. 이 시간은 순수 네트워크/decoder benchmark가 아니다. RSS/disk/time은 Python I/O·operation 경계의 협력적 검사이고 OS/native hard cap은 아니다. Native 내부 재해제 총량은 미계측이다.

예산 부족·EOF·8회 도달·같은 coverage·새 point 뒤 JSON 미확보는 `partial`로 보고한다. C snapshot이 없을 수 있으며 마지막 local checkpoint만 남을 수 있다. 원본 변경·충돌·파싱/해시 오류는 `failed`다. **C 게시 뒤 검증 실패도 failed**이고 해당 snapshot/receipt/anchor를 보고서에 보존한다. 자동 반복·예산 확대·B 덮어쓰기는 없다. Snapshot 게시 자체를 passed로 취급하지 않는다.

## 별도 notebook과 회수 자료

[drive_index_extension.ipynb](../notebooks/drive_index_extension.ipynb)는 C 전용 진입점이다. 기존 A/B helper를 호출하지 않는다. Root가 ready에 정확한 입력 B 폴더와 신규 C output/run ID, 모든 pin을 채운다. Tracked notebook의 pin은 비어 있어 미검토 상태로 원격 실행되지 않는다.

소스 업로드/설치·manifest/anchor 입력·실제 mount 검사·명시 실행·보고서 ZIP·선택 flush 흐름을 재사용했다. Engine은1.10.3으로 고정하고 모든 실행 gate는 기본False다. B anchor는 Drive 폴더의 최신 파일명에서 자동으로 고르지 않고 외부 전달된 SHA로 대조한다.

회수 ZIP은 C marker 보고서, C anchor, 새 DB 검증으로 얻은 기존/새 JSON이다. 보고서에는 revision/source ZIP/manifest pin·환경, before/after coverage·boundary·blob hash/size·point 목록, 새 JSON의 exact member/offset/end/hash, 각 scan·추출 read trace, cold 새 point 대조, 전체 자원/전송 합계, C 부모/lineage를 남긴다. 실패 시 게시본 경로도 보존한다.

다운로드는 `DOWNLOAD_REPORT=True`, flush는False로 먼저 실행한다. 파일을 보관한 뒤 다운로드를False로 되돌리고 선택 flush를True로 설정한다. Flush timeout은60초이며 결과 JSON도 보관한다. 실패/partial에서 runtime을 삭제하거나 같은 작업을 반복하지 않는다. C 성공 뒤 추가 VM 교체를 요구하지 않는다.

## 로컬 검증과 실제 원본 prefix 모의 재생

신규 검사는64KiB/256KiB/1MiB 간격의 실제 partial gzip index를 확장해 C snapshot을 복원하고, cold 새 JSON/기존 JSON을 확인한다. 같은 coverage인데 member 수만 증가한 경우, 바뀐 원본, 작은 예산, C 게시 후 추출 실패, 읽기 시작이 옛 point로 남은 경우의 false pass를 거부한다. 기존27개 계약도 회귀 범위다.

추가 Google 요청 없이 보존 cache ZIP의 **앞8MiB만** 읽어 chunk SHA를 대조하고 유한 mock transport로 재생했다. Metadata와 total은 원본368,145,316bytes를 유지하며, mock이8MiB 이후를 요청하면 거부한다. 실제 사용자 B snapshot은 로컬에 없으므로 B 상태를 재구성한 시험이다. 실제 B snapshot 검토나 수동 C 결과를 대체하지 않는다.

| 항목 | B 재구성 | C offline 모의 |
| --- | --- | --- |
| member /generation | 32 /2 | 80 /5(3batch,48개 추가) |
| boundary | 1,927,680 | 6,109,696 |
| index coverage /bytes | 3,791,957 /98,411 | 6,324,814 /163,983 |
| blob SHA | `60949a375158509e7fc855dd0fad21250e1a67b3bf641a423504e88b38df5b02` | `8d951826f7d6c39f61c80112404caa3f5176898cbf679bd7c16d48a5f8bc3f7b` |

선택한 새 JSON은 `ArchVisInterior_ArchVis_RT/Day/rp_adanna_rigged_003_ue4/SKM_MenReadingGlasses_Shape_01/017/Opening/json/frame_70.json`, ordinal68, offset5,065,216/end5,150,720이다. SHA256은 `b09725d20ac9b27671a7e524ca0442b8bc0ba7adf2585bc54292f79ac8b3786a`다.

새 restart point는 `(5,056,401,920,448)`이고 cold read 시작은920,447로±1에 일치했다. 옛 최대 압축 point690,130보다 뒤다. Mock 합계 응답 body3,673,256bytes/54HTTP, read6,488,071bytes/decoded4,694,814bytes였다. 이는 지연을 모의하지 않은 유한 local 재생이며 실제 Drive 시간·요청 성공을 보장하지 않는다. 원본 전체 CRC/학습 준비는 계속 미검증이다.

개발 근거는 `outputs/persistent_extension_development_20261030/`의 검사·notebook 보고서와 `offline-pose/feasibility.json`에 보존한다. Root 독립 원본 payload/point 대조는 `outputs/persistent_extension_review_20261030/offline_evidence_review.json`이다. 실제 source ZIP/ready/hash 고정·로컬 커밋·테스트 세션 인계는 root가 담당한다. 개발자는 commit/push를 하지 않는다.


## 최종 독립 검토

오케스트레이션의 최종 검사에서 기존27개와 신규6개, 총33개가 실패·오류·건너뜀 없이 통과했다. 최종 코드 hash도 검사 시점과 동일함을 대조했다. 별도 C notebook7개 code cell 구문·기본False gate의 무실행·미mount 시 인증/marker 전 거부도 독립 확인했다. 근거는 `outputs/persistent_extension_review_20261030/final_tests.json`, `final_tests.txt`, `notebook_review.json`에 보존한다. 이 검사는 실제 Google/Colab C 실행을 포함하지 않는다. 기존 source/index/snapshot core와 A/B 고정 배포물은 수정하지 않았다.
