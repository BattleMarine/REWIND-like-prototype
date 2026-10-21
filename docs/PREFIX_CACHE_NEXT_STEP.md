# Cold cache 결과 이후의 최소 후속 작업안

2026-10-21 · 기준 `f0607fa` · **오프라인 검토와 계획 문서이며 새 profile·원본 실행을 적용하지 않았다.**

권고는 **기존 v1 cache에서 논리128MiB까지 한 번만 진행하는 bounded warm suffix 경로**다. 구형 cache 호환과 관련 상한을 함께 고친 뒤 검토·고정 버전으로 테스트 세션에 인계한다. 동일96MiB warm 반복과 전체 archive 인덱싱은 이번 후속 묶음에서 제외한다.

## 1. 실제 cold가 남긴 근거

근거는 [독립 검토](../outputs/test_archvis_pose_cache_cold_20261021/orchestrator_review.json),[테스트 검증](../outputs/test_archvis_pose_cache_cold_20261021/verification.json),[카탈로그 요약](../outputs/test_archvis_pose_cache_cold_20261021/catalog_summary.json)과 같은 폴더의 결과ZIP·`prefix_cache.zip`이다. 모두 Git 제외 보관물이며 새 checkout에 자동 포함되지 않는다.

| 관측 | 값 |
| --- | --- |
| Cache 압축 prefix / 범위 |100,561,588bytes=95.903004MiB /384개 |
| 준비 network body / HTTP / 시간 |100,630,708bytes /1,152회 /311.326445초 |
| 해제 / header / tar metadata |551,914,001bytes /13,061 /811,994bytes |
| 카탈로그 |92sequence·6,318관측,완료payload6,317·미완료1,잘림/미분류0 |
| 목표 pose / retry |0개 /0회 |

이96MiB profile의 논리입력 잔여는101,708bytes뿐이다. 같은 설정으로 끝까지 warm 재생해도 새로 진행할 압축량은 약0.097MiB이며,원본 pre/post·import·기존 약526.347MiB 해제를 다시 지불한다. 목표 위치 근거 없이 같은 warm을 반복하지 않는다.

92sequence의 인물은 Adanna26·Amit19·Kyle15·Joko14·Carla18이며 Serena는 없다. 이는 **이번 보존 카탈로그에 기존 Serena 목표가 없다는 근거**다. 전체 pose 원본의 부재나 다른 인물의 RGB 확보 비용이 작다는 뜻은 아니다. 기존 Adanna pose나 CityPark 쪽으로 옮겨도 대응 좌우RGB 위치·연속성·시간·split 비용은 미확정이다.

## 2. 실제 cache의 오프라인 import·byte0 재생

현 `PrefixCache.load/read`와 `GzipReader`에 작은 local read adapter를 연결해 실제 보관물을 한 번 검사했다. Socket 연결을 차단했으며 새 원본 요청0,모델/추가회귀 실행0이다. 임시 import 사본은 검사 뒤 삭제하고 원본ZIP을 보존했다.

- Import는 `verified`,384chunk·100,561,588bytes의 개별/전체hash를 통과했다.
- byte0부터 해제한 출력은551,914,001bytes로 cold의 누적 해제량과 정확히 같았다. Gzip EOF/전체CRC는false다. Tar parser·카탈로그를 새로 생성하거나 RGB/pose 내용을 검증한 결과는 아니다.
- Local IO는201,212,292bytes로 현196MiB 안이었다. Windows CPU/Python3.12.14에서 import(copy/hash)0.172초,replay+gzip+출력SHA1.391초였으며 **Colab 재생시간이나 warm 총시간으로 사용하지 않는다.**
- 결과는 [오프라인 기록](../outputs/prefix_cache_next_20261021/developer/offline_import_replay.json),재현 script는 같은 폴더 `offline_import_replay.py`다. 프로젝트 루트에서 `python outputs/prefix_cache_next_20261021/developer/offline_import_replay.py --output outputs/prefix_cache_next_20261021/developer/replay_rerun.json`으로 별도 새 결과를 만들 수 있다. 기존 결과는 덮어쓰지 않는다.

이는 module-level 오프라인 검증이다. 현재 workflow/notebook의 warm은 인증·초기metadata·live pre/post를 호출하며 offline-only 진입점을 제공하지 않는다. 따라서 이 결과를 실제 Drive warm 성공·현재 서버identity 확인·전송 절감 실측으로 올리지 않는다. 기존 cache를 import했다는 이유로 가짜 live 검사를 넣어 usable 실행을 만들지 않는다.

## 3.128/192MiB까지 진행하는 비용 시나리오

이하 MiB=1,048,576bytes이며 **목표 논리압축량**을 뜻한다. 기존 cache `C=100,561,588`에서 suffix는 `X-C`,새256KiB Range 수는 `ceil((X-C)/262144)`다. 기존 마지막 chunk가 짧으므로 전체chunk 수는 단순 `X/256KiB`보다1개 많다.

| 논리 목표 | 새 압축 suffix | 새Range / 결과cache chunk | 정상 warm HTTP / retry3 포함 |
| --- | --- | --- | --- |
|96MiB 반복 |101,708bytes(0.096996MiB) |1 /385 |5 /8 |
|128MiB |33,656,140bytes(32.096996MiB) |129 /513 |389 /392 |
|192MiB |100,765,004bytes(96.096996MiB) |385 /769 |1,157 /1,160 |

HTTP는 suffix의 각 pre/media/post와 cache 전체의 live pre/post2회를 포함하고 초기 metadata는 별도다. 실제90bytes/metadata 응답이 유지되면128안의 준비 network body는33,679,540bytes,192안은100,834,484bytes다. Metadata가 허용 최대16KiB이면 각각37,915,980bytes와113,413,452bytes까지 필요하다. 따라서 새network40/112MiB가 비교용 후보이며,192안은 기존network96MiB도 넘는다. 응답길이 예약과 전체retry 상한은 유지한다.

해제·header·tar metadata는 cold의 밀도가 뒤쪽에도 같다는 **가정**으로 `관측값×X/C`를 계산했다. 목표 위치·네트워크·분포를 보장하지 않는다.

| 논리 목표 | 전체 해제(MiB) | Header(올림) | Tar metadata(MiB) | Regular/sequence 밀도 참고 |
| --- | --- | --- | --- | --- |
|128MiB |702.505 |17,433 |1.034 |약8,433 /123 |
|192MiB |1,053.757 |26,149 |1.550 |약12,649 /185 |

기존640MiB 해제 제한은 약116.611MiB,header16,000은 약117.483MiB 논리입력에서 먼저 걸리는 계산이다.192안의 sequence 수도 현128 cap을 넘을 수 있다. Catalog cap은 잘림을 표시하며 parser를 멈추는 header cap과 다르다.

시간은 `warm import(C)+기존C 재해제/파싱+suffix 수신/해제+보고/저장`으로 나눠야 한다. Cache는 압축 본체의 재전송을 줄여도 약95.9MiB 입력·526.347MiB 해제와13,061header 처리를 없애지 않는다. Cold 단순 비례의128=415.5초·192=623.3초는 **cold 상당 비용 참고**이며 warm 예상시간이 아니다.420초 안에 도달할지는 Colab cache IO·CPU·새Range 응답을 분리 측정해야 한다. 로컬1.563초와 Colab cold 시간을 섞어 총시간을 보장하지 않는다.

## 4. Limits 숫자만 올려서는 안 되는 코드 경계

근거: [prefix_cache.py](../rewind_archive/prefix_cache.py),[sources.py](../rewind_archive/sources.py),[extraction.py](../rewind_archive/extraction.py),[catalog.py](../rewind_archive/catalog.py),[profile](../rewind_archive/profiles.py),[workflow](../rewind_archive/workflow.py).

| 현 경계 | 필요한 후속 검토 |
| --- | --- |
| `LOGICAL_BYTES=96MiB`,closed schema `bounded_prefix_cache_v1` | Import validator·append·SplitStream·보고서의 고정96을 함께 다뤄야 함. 새 이름의profile만 추가하면 동작하지 않음 |
| `MAX_CHUNKS=512` |128안513·192안769로 초과. 기존384개를 유지한 연결에서 경계를 검증 |
| Cache/ZIP128MiB,write의manifest1MiB 예약 | 본체128MiB는 저장/ZIP cap에 맞지 않음. Cache 디렉터리와input/export ZIP cap을 각각 조정 |
| Local IO196MiB | **이번 입력96의 import+재생은 충분함.** 향후 확장본의 재import+재생128×2/192×2에는 부족하므로260/388MiB 등 검토 |
| Export IO384MiB /60초 |128 본체3pass만384MiB여서metadata/ZIP overhead와 함께 초과.192 본체는576MiB.512/640MiB 등 별도 후보와 시간 측정 필요 |
| Catalog sequence128 /entry16,000 /6MiB,추출16MiB | Sequence 여유와 카탈로그 최악bytes·cap/truncation·파일2개/일반보고서 공간을 함께 증명 |
| HTTP 요청과local audit 수가 달라짐 |392HTTP보다cached+suffix513행이 많을 수 있음. `report_reserve_bytes`를HTTP수만으로 계산하지 말고 local replay/audit cap과 분리해 검증 |

전역 상수를 수정해 기존 v1 계약을 조용히 바꾸지 않는다. 새 cacheformat/policy 판을 명시하고 **신규 reader는 보관 v1을 그대로 읽으며 신규 출력은 신판으로 식별**하도록 한다. 구형 API7은 지원하지 않는 큰 cache를 명확히 거부해야 한다. Identity·범위·내용hash·post 검사·실패unusable·EOCD/경로/ZIP 보호는 유지한다. 입력manifest가 자의적인 상한 숫자를 선언하도록 허용하지 않는다.

## 5. 권고하는 최소 개발 묶음

**대상은 기존v1 실제cache를 사용한 논리128MiB 한회 suffix pilot 경로다.**192MiB는 새 전송이 약96.1MiB로3배이고 해제·header·disk도 커져 다음 최소 단계로 권고하지 않는다.128은 비용 절충이며 Serena 위치를 근거로 한 최적값은 아니다.

| 새 고정 계약의 검토 예시 |128 후보 |192 비교만 |
| --- | --- | --- |
| Network /논리 /준비 HTTP |40MiB /128MiB /392 |112MiB /192MiB /1,160 |
| 준비시간 /초기metadata |420초 /기존 별도128KiB·30초·8회 |같은420초이면 부족할 수 있어 별도 계측 필요 |
| 해제 /header /tar metadata |800MiB /20,000 /2MiB·단일64KiB |1,200MiB /30,000 /2MiB·단일64KiB |
| Cache와각ZIP /chunk cap |각160MiB /640개 |각224MiB /896개 |
| Local IO /export IO·시간 |260MiB /512MiB·60초 |388MiB /640MiB·60초 |
| Catalog /일반출력 |Sequence256·entry16,000·6MiB /16MiB를 검증 |같은cap의 잘림·용량을 별도 확인 |

392요청은 이번100,561,588bytes 입력 캐시를 기준으로 한 값이다. 더 작은 캐시로128MiB에 도달하는 계약으로 일반화하지 않으며, 고정 테스트는 입력ZIP/manifest hash로 이번 보관본을 지정한다.

새128 계약에 초기metadata 상한을 합하면network40.125MiB·450초·400요청이다(인증/업로드/export 제외). 범위 밖으로 자동 확대하지 않고420초에 멈추는 경로를 유지한다. 실제목표 미확보0개도 정상적인 pilot 결과로 보관한다.

이번v1 inputZIP 상한128+새cache160+export160+일반출력16+결과17=481MiB에 작은 보고서 여유가 필요하다. 신규v2 ZIP을다시입력하는일반warm에서는160×3+16+17=513MiB다.192 비교안의일반warm은224×3+16+17=705MiB다.실제보관ZIP은95.945477MiB로작지만 이를모든입력의상한으로일반화하지않는다. 업로드·영구보관의전송/시간은별도이며기존417MiB틀을확장안에그대로쓰지않는다.

### 완료 기준과 테스트 진입 조건

1. 현 실제 v1의manifest/hash·byte0 해제 기준을 보존하고 재현script/환경을 연결한다. 필요하면 네트워크가 구조적으로 차단된 offline-only 검사 진입점을 얇게 제공하되 live warm과 별도 결과로 표시한다.
2. v1 입력→신판 export/import 호환,513번째chunk,128MiB 본체+manifest/ZIP,3pass export,실패/부분chunk/원본변경/hash오류/시간중단을 완전한 합성gzip fixture에서 검증한다. 실제 prefix에 임의 합성suffix를 붙여 유효한 gzip으로 취급하지 않는다.
3. Network/local/logical/decompressed/header·audit/report/catalog·disk·시간계수를 분리하고 post 미완료를 usable로 바꾸지 않는다. 구형profile/API/실패결과ZIP 보존을 회귀 확인한다.
4. 고정 계약·코드·소스ZIP/notebook과 실cache 식별값을 리뷰한 뒤 **테스트 세션에한JOB만** 인계한다. 보관prefix는재전송하지않고첫새Range는offset100,561,588부터여야한다. 원본identity가다르면중단하며cold재수신을자동대안으로쓰지않는다.

필수 인계 정보: 프로젝트/세션ID,고정revision·소스/ready notebook SHA,입력ZIP SHA·manifest/prefixhash,새profile·모든단계상한·디스크,실행순서와중단조건,cache/catalog/목표각판정,새network와local재생시간분리,회수할두ZIP/보고서와영구보관위치다. 기존prefix SHA는 `6354437b0a27b8050e8111c81a71eec0cc35d915da0852d7978d4298016e857e`,입력ZIP SHA는 `4d63187ff49de40b5af1843aa4ded51caed255fda9996d78c028c47bf58c3319`다.

전체pose 약351MiB의 인덱싱·gzip state resume·모든sequence 재포장·RGB52GB대탐색은 별도 장기 작업이다. 최초 전구간 읽기와 더 큰cache/catalog/parser/보관비가 필요하고,새index를만들어도RGB와pose대응·시간·분할·내용검증이완료되는것은아니다.이번묶음에포함하지않는다.

이번에는 문서와 Git 제외 오프라인 evidence만 추가했다. 새profile 적용·원본네트워크·실warm·예산변경구현·학습·원격 push는 수행하지 않았다.
