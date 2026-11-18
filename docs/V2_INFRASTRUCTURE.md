# v2 시간·캐시 보존·재개 인프라

2026-11-18 사용자 승인 범위의 구현 안내다. 실제 세 시나리오 입력 생성·Drive 보관·새 VM·CUDA 시험은 미완료이며, 아래 CLI를 이 문서 작성으로 실행하지 않는다. 고정 배포 revision/source ZIP/notebook SHA는 root 검토·커밋 후 확정한다. [v2 계획](V2_PLAN.md)과 기존 실험 결과를 구분한다.

## 시간과 사용량

[config](../configs/rewind_v2.json)의 `seconds_per_invocation=28800`은 준비/학습 **각 호출 전체8시간**이다. 계산·입력 검증·중간 동기화는27,480초까지이며, 마지막1,320초는 package600·copy/readback600·checkpoint120초를 위해 안쪽에 예약한다. 중간 보관 전에 종료용 시간·byte 예약이 남는지 검사한다. Notebook 외부 timeout28,920초에 SIGINT60초 후 kill/wait30초를 적용한다. 이 유예는 새 계산시간 승인이 아니다. VM 수명이나 연속16시간을 보장하지 않는다.

원본/복사 read·산출물 write는 호출당 기존 보수적8GiB 상한을 유지한다. 현재cache12GiB·feature256MiB·run4GiB·archive2GiB·checkpoint256MiB·RSS4GiB도 자동 확대하지 않는다. 계획이 이를 넘으면 실제 payload 읽기 전에 `resource_budget_decision_required`로 막는다. Source 합계가8GiB를 넘는 것은 여러 명시 호출이 필요하다는 뜻이며 전체디스크 fit과 분리한다. 전체누적 상한은 미확정이다. `cumulative_limits=null`은 기록을 없애거나 자동반복을 허용하는 값이 아니며, 미지원 non-null 누적 정책은 거부한다.

[ledger](../rewind_v1/v2_budget.py)는 호출·종류별 active wall, 사전예약/관측 byte, 상태·이유와 복원된 lineage 사용량을 보존한다. 원본 읽기는 shard 전체를 먼저 예약하고 실제 읽은 값을 shard 경계에 관측 기록한다. 중간 강제 종료 시 관측값은 하한이며 사전예약은 남는다. 시간상한·byte상한·수동중단·실패를 구분한다. 미종료 호출은 `interrupted_unknown`과 관측 시간·외부timeout을 가정한 추정 상계로 구분한다.

Ledger는 새 이름에 완전 게시한 세대별 snapshot을 사용한다. 세대번호는8자리(1..99,999,999)이며 임의10만커밋제한은두지않는다. 최대8MiB/세대·최근4세대를 유지하며 이전 예약/호출 이력은 최신 snapshot 안에 그대로 있다. 게시마다 전체 디렉터리를 열거하지 않고 정해진 이전 세대만 정리한다. 시작 시 최대16개 게시본을 열거하고 이를 넘으면 명시 복구를 요구한다. `.publish-*` 미게시 임시본은 사용하지 않으며 최신 게시본 손상 시 이전 상태로 조용히 후퇴하지 않는다. 현재 검증된 snapshot만 retention에 포함한다.

Read/write 예약·관측은 외부 source/copy와 생성 payload 범위다. 로컬 shard 전체 해시·mmap frame 복사·target 역직렬화의 논리 읽기는 별도 `local_read_logical_bytes`로 기록한다. 메모리 page cache의 물리 IO·filesystem metadata·ledger/catalog 자체 쓰기는 이 byte 합계에 포함하지 않는다. 모든 처리의 시간은 같은 호출 deadline 안에 들어간다. 결과 ZIP에는 마지막 package·선택적 copy의 예약을 먼저 담고, 최종 실제 관측은 로컬ledger/보관상태에 남긴다.

## 입력 계약·비용

`rewind_v2_audited_manifest_v1`은 세 시나리오의 sequence/frame/공식split·left/right/pose 경로·size/SHA와 spec digest를 고정한다. 원본 경로의 sequence/frame/role를 파싱해 선언값과 비교한다. Spec에 pin한 공식 train/validation/test 텍스트3개를 읽되 test payload는 사용하지 않는다. 별도 `rewind_v2_frozen_inventory_v1`의 usable key 전체와 manifest가 정확히 일치하고, 공식 대상 sequence coverage와 누락·중복·제외 없음이 확인돼야 한다. 부분 manifest를 전체train으로 통과시키지 않는다.

이번 구현은 **입력 계약과 소비 경로**다. 실제 정리 후 inventory/감사 manifest를 아직 생산·회수하지 않았고 공식 전체 원본 무결성·실제 timestamp/stereo동기·motion 독립성을 새로 입증하지 않는다. 고정 manifest 입력을 확정하는 작업은 별도 인계가 필요하다. Schema seal만으로 외부 inventory 내용의 진실성이 증명되는 것은 아니다.

Plan은 scenario/split별 count, 연속run·short/tail·padding, 원본bytes, 이미지/GT/camera cache·feature·동시 작업disk와 현재free를 보고한다. 이미지128px float32 stereo는페어당393,216B다. Feature는clip당50slot의 실제 dtype에 근거해 추정한다. Cache/feature 추정에 run·archive·restore staging·실패보존본·checkpoint를 더해 사전 차단한다. 수치 추정은 실제 Colab 파일크기/VRAM/IO pilot을 대체하지 않는다.

## 캐시 스키마·복구

[v2 cache](../rewind_v1/v2_cache.py)는128페어마다 images.npy·targets.pt·receipt.json을 임시 디렉터리에서 완성하고 한 번에 게시한다. 원본 float64 R/t와 absolute frame camera를 보존하며, teacher clip에서는 첫 유효frame stereo midpoint anchor를 새로 설정한다. Padding은 camera/target/features0·time/target mask false다. 다른clip/sequence로 anchor를 넘기지 않는다. Schema·source·입력·camera표현·preprocessing·환경을 binding에 넣는다. 기존 v1.2 absolute float32 cache는 재명명 재사용할 수 없다.

빈 local catalog를 먼저 게시한다. Shard rename 직후 중단되면 다음 호출이 receipt를 검증하고 catalog에 반영한다. 기존 catalog의 전체 pin을 재검증하는 동안 진행량을 줄이지 않으며 새 shard commit 때만 늘린다. 누락/손상은 자동 덮어쓰지 않는다. `quarantine --cache <정확경로> --shard-name shard-00000 --receipt-file-sha256 <현재receipt바이트SHA>`로 해당 디렉터리를 cache 안 `.quarantine-*`로 이동하고, 검증된 보존본에서 복원한다. `.pending-*`도 정확한 basename으로 명시 격리할 수 있다. 자동 삭제·격리·재시도는 없다. 격리본도 disk 비용에 포함한다.

`copy-cache`는 **사용자가 명시 실행하는 경로**이며 notebook `USER_EXPLICIT_CACHE_COPY=False`가 기본이다. 에이전트의 원본/캐시 Drive 업로드나 준비 callback의 자동 캐시 복사를 도입하지 않는다. 완료 shard ZIP을 새 이름으로 복사하고 size/SHA readback 후 remote verified catalog를 늘린다. 이미 verified인 prefix는 기존 receipt·현재 파일 존재/크기를 확인해 재포장·전체재복사를 건너뛴다. 이때 기존 blob의 현재 전체내용을 재hash했다고 주장하지 않는다. 다른binding·짧은source가 기존remote catalog를 덮어쓰는 것은 거부한다.

`restore-cache`는 지정한 catalog 바이트SHA, seal/binding·shardZIP SHA·내부파일 SHA를 확인하고 완료 directory를 원자 게시한다. 기존local shard는 로컬 내용검증을 수행하고 새로 필요한 원격 shard만 읽는다. 기존prefix가1회 전송량을 넘더라도 이미 복사한 원격 prefix를 매번 읽어 새 shard 진행을 막지 않는다. 복원중 실패해도 이전local catalog는 줄지 않는다. 로컬 완료와 remote verified를 구분하며, remote catalog의 과거 readback 재사용과 복원시 새 내용검증의 범위를 구분한다.

## 학습과 모델 복원

학습 초기화·epoch·seed·GT/예측upper·평가 정책은 config에null로 남아 있으며 `training_decisions_pending`으로 차단한다. 인프라가 지원하는 기존 경로는 fresh frontend→body→hand, hand GT upper train/고정body 예측upper validation이다. 다른 초기화·혼합/예측upper 학습은 별도 설계 대상이다. Validation은 시나리오별 참고값과 pooled score를 분리한다. 명목25FPS 진단·velocity/foot0 가정은 유지한다.

승인 설정을 연결하면 기존모델구조·loss·학습단계를 재사용한다. Last는optimizer/scheduler/scaler/RNG/cursor·입력/source/config/environment를 보존한다. 모델checkpoint는25update 경계와 단계/종료 경계에, 원격 모델 결과는시간·byte예산이 허용하는60분 주기와종료에 보존한다. 입력캐시는 모델결과ZIP에서제외한다. 기존v1.2 hand last187,136,889B의 보존 근거상 현재단일stage optimizer256MiB 제한은 기존구조 범위에 해당하며, 실제v2 checkpoint 크기·저장working set은pilot에서확인해야한다. 여러teacher optimizer를한파일에추가하는범위가아니다.

`restore-run`은 고정ZIP 복사본SHA·내부파일·last·필수best/milestone·의존모델·source/config/input/environment를 검증한 뒤 새RUN에 게시한다. 다른GPU/Torch/CUDA환경을 수치동일재개로 통과시키지 않는다. 지원CUDA GPU의 모델명은 제한하지 않지만 기존실행 재개동일성은별도문제다. 복원시ledger lineage를합치며같은보존본 재사용으로누적사용량을중복합산하지않는다.

## 진입점과 검증 상태

CLI는 `python -m rewind_v1.v2_cli source`, 그 외 `plan/prepare/copy-cache/restore-cache/train/restore-run/quarantine`이다. 각명령은source/config/manifest/spec의명시pin과official split 디렉터리·WORK를 받으며자동재호출하지않는다. [빌더](../scripts/build_v2.py)는고정소스ZIP에서독립추출해기본OFF notebook과receipt를생성한다. 실제사용자안내는root가고정인계후테스트세션에서진행한다.

로컬 fixture 검사와 실제 Colab/Drive/GPU·새VM 시험을 구분한다. 개발중Windows에서 기존atomic replace가 ledger.json/last.pt에WinError5를 간헐적으로 반환했다. 정확한OS원인은미확정이며, 실패를통과로기록하지않는다. v2ledger를세대별저장으로바꿨고기존v1checkpoint/atomichelper는보존했다. 최종로컬검사수·로그는root인계시기록한다.

2026-11-18 최종 개발 로컬 검사: 신규v2 12개(59.140초), 기존v1.2 15개(32.521초) 모두 통과. 실제Windows CLI별도프로세스2호출·세대보존/손상거부, 세단계학습의model/Adam/scheduler/scaler/RNG exact재개를 포함한다. 동시검사실행으로측정한wall이며 실제학습/GPU성능값이 아니다. 로그는 `outputs/v2_infrastructure/developer_tests.txt`, `v12_training_regression.txt`다. 검증 중 기존atomic helper의WindowsWinError5로실패한이전실행도 있었으며 최종통과가OS원인규명을뜻하지않는다.

## 추가 검토 반영과 최종 로컬 상태

2026-11-18: v2_space는전action공통최초working-fit차단을제거하고준비/캐시복사/복원/학습/모델복원별추가bytes를실제출력volume별로합산한다. 이미있는캐시와특징파일은두번예약하지않는다. 캐시복사는새prefix분량과한shard임시ZIP만예약하며이미verified인prefix만재확인하는경우ZIP공간을예약하지않는다. 운영중package/copy도현재대상free를다시검사한다. Windows RAM은GlobalMemoryStatusEx의available physical memory를사용한다.

추가변경후신규14검사전체개발자100.562초/root103.394초통과. 100,000세대경계통과·8MiB제한단일snapshot읽기·증분공간판정과실제Windows CLI plan2회및128페어copy→12페어추가copy재개를포함한다. 최초copy CLI fixture실패는in-process2페어shard설정과실제CLI128페어binding차이로stderr에서확인해fixture를실제128+12로수정했다. 로그는 `developer_tests_final.txt`, `windows_cli_revision2.txt`다.

기존v1.2회귀는최초15개전체성공32.521초와재검증14개성공/1개WinError5실패48.559초를구분한다. 실패경로는기존 full_cache.prepare_frames의preparation.json atomic replace이며해당test_atomic_chunks_resume_exact_resize_and_source_corruption 단독재실행은7.213초통과했다(`v12_regression_final.txt`, `v12_atomic_isolated.txt`). 정확한Windows원인·실제운영데이터영향은입증하지않았고개발fixture의오류로기록한다. v1저장코드는수정하지않았다.
