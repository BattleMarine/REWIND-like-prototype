# 전체 데이터셋의 독립 사전 인덱싱

> **보존 기록 · 2026-11-05 정리:** 2026-11-04 사용자 결정으로 분할 압축 원본의 인덱싱·부분 읽기 운영을 중단했다. 이 문서 전체의 실행 명령, 다음 단계, 예산과 미완료 표시는 해당 실험 당시의 기록이며 현재 실행 지시가 아니다. 새 기본 경로는 [압축 해제된 RGB·pose 사용](DATA_ACQUISITION_STRATEGY.md)이다. 기존 코드·원본·인덱스·예약 기록은 보존하며 자동 재개하지 않는다. 이동된 로컬 근거는 [보관·복원 안내](LOCAL_STORAGE.md)를 따른다.

2026-11-03 당시 구현: 새 `bounded_auto_v1` profile의 자동 실행·중단·재개·coverage와 CPU worker notebook을 추가했다. 기존 calibration/legacy 계약은 유지하며, 정상 구간별 사용자 승인은 요구하지 않는다. [자동 프로그램의 검증 결과와 운영 경계](AUTOMATIC_FLEET_INDEXING.md)를 따른다. 아래 실험·실패 기록은 당시 조건의 근거로 보존한다.

## 버퍼·인덱스 간격의 격리 비교 (2026-11-03)

기준49e5a3c production을 변경하지 않고 한 번에 한 변수만 바꿨다. 압축 입력 readbuf_size64KiB, 첫 scan 논리 read soft4MiB/hard8MiB, decode32MiB, mock Drive cache4MiB와 고정 plan은 유지했다. buffer_size는 해제 출력 BufferedReader 용량이며 압축 Range cache와 다르다. 두 번째 scan은 같은 store/plan, 새 workspace를 사용한다. 누적 잔여 예산에 따라 예약과 유효 soft/hard 한도가 감소한다(예: Lab 기본 두 번째 soft3,741,404/hard7,482,807bytes). 후보별 첫 소비량 때문에 두 번째 유효 한도도 다르므로, 첫 scan은 고정 한도 비교이고 두 scan 결과는 실제 누적 계획의 운영 비교다. 두 번째 한도가 모두4/8MiB로 고정됐다고 해석하지 않는다. 과거 파일 추출은 별도의 논리8MiB/body16MiB/158HTTP 예산이다.

| 변형 | 합성 완전 파일 누적: 첫→둘째 | 실제 Lab PNG 누적: 첫→둘째 |
| --- | --- | --- |
| 기본 buffer64KiB / spacing1MiB |1→1|0→0|
| buffer256KiB만 변경 |2→3|1→1|
| buffer1MiB만 변경 |2→4|1→2|
| buffer2MiB만 변경 |5→5|3→3|
| spacing256KiB만 변경 |2→3|1→2|
| spacing512KiB만 변경 |1→2|0→0|
| buffer4MiB만 변경 |checkpoint hard 한도 실패·게시0|checkpoint hard 한도 실패·게시0|

개발14조건은 합계22.203초에 실행했다. Root는 기본/buffer1MiB/spacing256KiB의 두 입력, 총6조건을7.938초에 독립 재실행해 같은 파일수·cursor·중단 사유를 확인했다. Lab 두 PNG를 별도 표준 tar reader로 읽어 catalog SHA를 대조하고 실제 이미지 decode도 통과했다. 전체 prefix gzip CRC, 다른 scene·장기 전진·Colab 성능은 검증하지 않았다. 개발에서 파일을 게시한 조건의 과거 파일 SHA 재추출은 통과했다. 추출 reader는 scanner.extract의 기존 별도 경로(buffer=4×spacing, readbuf64KiB, auto_build=False, 전체 저장 pack import)다. Scan 후보 버퍼를 그대로 사용한 추출 성능으로 보고하지 않는다.

캐시 용량 가설은 관측과 맞지 않았다. 기본 Lab 첫 scan은 압축 cache resident2MiB로4MiB보다 작고 cache hit62회인 상태에서 논리 read4MiB에 도달했다. 서로 다른 압축 구간은1,114,112bytes뿐이었다. Cache는 압축 Range를 LRU로 교체하며 잘린 PNG를 누적 보관하는 곳이 아니다. PNG의 불완전 payload와 영구 catalog의 완전 member를 구분한다.

인덱스 간격과 버퍼의 비율은 영향을 주지만 단독 성공 조건이 아니다. 설치된 indexed_gzip1.10.3의 IndexedGzipFile 구현은 각 raw read가 이전 point에서 재시작하므로 기본 buffer를 spacing의4배로 정한다고 설명한다. 이 프로젝트에서는 buffer를 작게 고정했었다. 다만4MiB 후보는 읽기 선행과 lazy index 때문에 checkpoint에서 추가4MiB를 사용하고 hard8MiB까지 소진했다. spacing은 목표 간격이며 실제 deflate 재시작점의 간격·현재 cursor·checkpoint coverage도 함께 봐야 한다.

Root Lab 결과에서 buffer1MiB의 두 scan 논리 read는4,194,304/4,849,664bytes이며 두 번째에는 checkpoint 추가1,114,112bytes가 포함됐다. Pack은32,839→65,625bytes였다. spacing256KiB는4,128,773/3,735,557bytes, checkpoint 추가 읽기0, pack131,197→163,983bytes였다. 같은 PNG2개 확보 기준으로 이번 작은 구간에서는 촘촘한 인덱스가 읽기 비용은 적고 저장량은 컸다. 파일 크기·압축률·구간이 바뀌면 이 비교도 달라질 수 있다.

다음 검증 후보는 buffer1MiB와 spacing256KiB다. 제한된 추가 segment에서 지속 진전·첫/중간/마지막 파일 추출·원본 SHA·누적 예산·인덱스 크기를 비교한 후 기본 설정 채택을 결정한다. 두 변수를 동시에 바꾸거나 예산을 자동 증액하지 않는다. 이번은 시험 결과이며 production 수정·원격 재실행 승인·전체 인덱싱 완료가 아니다. 기존 Drive pending과 예약은 보존한다.

근거는 `outputs/fleet_read_cost_20261103/developer/compare.py`, `proof.json`, `proof-buffer4.json`, `FINDINGS.md`와 `root/compare.py`, `root/proof.json`, `root/png_check.json`이다. 실험 코드/산출물은 Git 제외 outputs에 보존하고 결과 요약만 버전 관리한다. 원격 요청0이며 합성 전체 gzip CRC와 실제 Lab prefix의 제한 읽기를 구분한다.

## 예외 복원 수정과 순차 평가 (2026-11-03)

Production scanner가 engine 호출마다 이전 source 사유를 지우고 해당 호출에서 감싸진 원인 예외를 복원한다. 알려진 예산 중단만 partial로 처리하고 완전한 member 경계와 유효한 checkpoint만 게시한다. 전진이 없으면 기존 archive head를 유지한다. 손상·미확인 오류는 추가 checkpoint 탐색 없이 실패 처리하며, checkpoint/cleanup의 후속 오류가 원래 실패를 지우지 않도록 기록한다. Runner는 실패 단계·제한된 type/reason chain·예약·관측량을 구분하고 관측 수집 실패도 보존한다. 미확인 소비를 자동 환급하지 않는다.

버퍼64KiB, spacing1MiB, source soft4MiB/hard8MiB와 기존 계획을 유지한 순차 평가 결과다.

| 평가 | 결과 |
| --- | --- |
| 합성4,802,934bytes, mock Drive 실제 runner 첫 scan | 완전한400,000bytes 파일1개 partial 저장. body1,701,555bytes/10HTTP 실측 정산 |
| 같은 store·계획, 새 workspace의 다음 scan | 기존 catalog/pack 복원. 추가 전진 없이 no_committed_progress, pending 없음. 누적 body3,403,110bytes/20HTTP가 두 실행 합과 일치 |
| 과거 파일 읽기 전용 재추출 |400,000bytes SHA 일치. 별도 예산 logical4MiB/body4,805,650bytes/19HTTP이며 위 indexing ledger에 포함하지 않음 |
| 보존된 실제 Lab prefix, root 독립 runner 두 번 | 첫 scan 디렉터리8개/PNG0개 보존, 다음 scan no_committed_progress. 두 실행 pending 없음, 예산 초기화 없음, 원격 요청0, 총0.969초 |
| 오류 경계 | 손상, stale source reason, checkpoint hard 한도, metadata finish, 관측 수집 및 cleanup 실패에서 오승격 없이 원인·예약·이전 정상본 보존 |

개발31개(15.410초), root 독립31개(12.417초) 검사가 통과했다. 범위는 기존 test_fleet/test_fleet_window_storage/test_fleet_frontend와 신규 test_fleet_exception이다. 후속 checkpoint 실패의 이전 head 보존은 runner 경계 주입으로 검사했고 실제 checkpoint 한도 실패는 별도로 검사했다. Windows Python3.12/indexed_gzip1.10.3 로컬·모의 transport 결과이며 Drive 내구성 검증은 아니다.

**평가: 읽기 한도 예외 복원 누락과 partial 손실은 개선됐지만 반복 읽기 비용은 해결되지 않았다.** Lab 첫 PNG는674,652bytes로 현재 한도 안에서 끝까지 읽지 못했다. 디렉터리 보존을 이미지 샘플 확보로 계산하지 않는다. 다음 작업은 읽기/재개 전략을 별도로 비교하여 동일 예산에서 실제 완전 파일의 연속 진전을 검증하고 checkpoint·과거 추출 비용까지 평가하는 것이다. 그 전에는 전체 인덱싱 준비 완료나 새 Colab 재실행 가능으로 판정하지 않는다. 실패한 기존 Drive store의 pending/예약은 그대로 보존한다.

근거: `outputs/fleet_exception_fix_20261103/developer/{FINDINGS.md,proof.json,tests.txt}`, root의 `root/evaluate_real_lab.py`와 `root/real_lab_evaluation/review.json`. Lab 입력 SHA는 아래 원인 대조 검증과 동일하다. 이 실제 prefix는 실패한 ArchVis와 다른 원본이므로 과거 Colab 사건의 직접 원인까지 확정하지 않는다.

## 첫 calibration 실패와 조사 (2026-11-03)

고정 `72a9ce1` 실행에서 metadata pin은 5HTTP/1483bytes/약0.87초로 성공했다. Runtime inventory와 calibration 계획의 hash·재계산·상한 일치를 root가 확인했다. 이후 한 segment는 보고서 경과4.183초에 `ZranError`로 실패했다. Archive 상태 목록은 비어 있고 pending 예약은 남았으며 lease와 settlement는 없다. 저장된 인덱스 진전이나 비용 측정 성공으로 해석하지 않는다.

회수 ZIP은 `outputs/test_fleet_calibration_001_20261103/fleet_recovery_20630a6d092c49fe82f56440b9d31d5d.zip`, SHA256 `8d46060d89e79cba9a03b2bdb1831f8841f9121b2200cb9d65d907ac2034e0b9`다. ZIP CRC도 통과했다. Plan SHA는 `63510a1b4e24b7a8931d1ed0d3bf2c42590f6e5e6891b34e6829996697f03515`, 마지막 HEAD는 `de1219c5dfcbef0de01cbac695a0b9df2dc2238c5ebdd6ac6917e6b855055a57`이다. 회수본에 run report는1개이며 출력이 두 번 보였다는 사실만으로 실제 이중 실행을 단정하지 않는다.

Source16MiB/158HTTP/120초/artifact 논리 I/O13,521,408bytes는 **실측이 아니라 소비량을 확인할 수 없어 전액 유지한 예약**이다. 이 정책에서 HTTP와 artifact 잔여 예산은0이다. Pending을 지우거나 새 store/job을 만들어 같은 실행을 반복하는 방식으로 예산을 초기화하지 않는다. Drive 상태와 기존 결과를 보존한다.

Root는 원격 요청 없이 보존된4.8MB/두 조각 합성 archive에 동일 calibration runner를 적용해 같은 `ZranError`를 재현했다. 이 재현의 원인 예외는 `IndexFailure(read_bytes_limit)`이고, 논리 source read4,194,304bytes에서 중단되었다. 반환 decoded bytes는532,480으로 초반 구간을 반복해서 읽는 trace가 확인됐다. 이는 논리 읽기이며 네트워크 재전송량이 아니다. 근거는 위 디렉터리의 `root_repro.py`와 `root_repro/proof.json`이다.

기존 `rewind_archive_index`는 `budget.last_source_failure`로 엔진 wrapper의 실제 중단 사유를 복원하지만, 수정 전 fleet scanner는 `EOFError/IndexFailure`만 처리하고 runner는 예외 타입명만 기록했다. 그 때문에 원래 사유·실패 단계·실제 관측량을 잃는 결함을 확인했다. 실제 Colab 회수본에는 원인 chain/read trace가 없어 동일한 읽기 한도가 직접 원인이었다고 확정하지 않는다. 압축 원본 손상, Drive 결측 또는 이중 실행으로도 단정하지 않는다.

아래는 production 수정 이전의 원인 조사 기록이다. 당시 코드 수정, Drive pending 복구, 원격 재실행, 상한 증액은 하지 않았다. 후속 보완은 순차 읽기 중 반복 비용, 원인 예외/실패 관측치 보존, 정상 예산 중단과 손상 구분, 완전한 member의 checkpoint 여부를 함께 검증해야 한다. 작은 합성 archive21개 검사 통과가 실제 source 한도 경계 동작을 보장하지 못했으므로 그 경계의 회귀 검사가 필요하다.

재개 후 개발 측 독립 조사에서도 같은 scanner가 local/mock Drive 양쪽에서 동일한 chain으로 실패했다.64회 논리 read 중 같은 구간 반복38회였으며, 이미 두 restart point가 있어도 현재 decoded 위치가 두 번째 point 앞이면 원점 point에서 반복 inflate/discard가 일어났다. 첫 point가 없는 경우에만 한정된 문제가 아니다. 모의 Drive의8호출/body1,701,185bytes/cache hit62회는 실제 Colab 사용량이 아니다.

같은4MiB 논리 읽기 한도로 직접 engine buffer를 비교하면64KiB와1MiB는 각각 반환 decoded589,824/1,048,576bytes에서 실패했고,4MiB buffer는1.5MiB 목표 읽기를 통과했다. 그러나 마지막 경우 export point가0인 lazy index 상태여서 checkpoint·재개 해결로 판정하지 않았다. 정확 scanner 재현에서는 finally export가 정상 완료하고 재seek하지 않아 원래 `gz.read` 오류가 가려진 경우는 아니었다. 다른 경계에서의 원예외 보존은 별도 검증이 필요하다. 이 조사에서는 코드 변경 없이6개 오프라인 probe를1.437초에 수행했다. 근거는 `outputs/test_fleet_calibration_001_20261103/developer_diagnosis/FINDINGS.md`, `proof.json`, `investigate.py`다.

### 실제 RGB bytes의 원인 대조 검증 (2026-11-03)

추가 원인 검증에서 root는 이미 보존한 공식 Lab_Scene gzip prefix20,636,340bytes(SHA256 `46bff8311c9e211c5dcda9d494b190849ce8d9e2aa0cd9643a1fabe16b168dcb`)를 동일 calibration runner에 입력했다. 논리 read4,194,304bytes/반환 decoded531,968bytes에서 `read_bytes_limit → ZranError`가 재현됐다. 이때 실제 원본의 최대 읽기 끝은1,114,112bytes로 prefix 끝에 도달하지 않았다. 표준 gzip은 동일 파일에서 decoded2,097,152bytes를 정상 반환했다. 따라서 합성 fixture에만 한정된 현상이 아니며, 이 재현에는 해당 구간의 원본 손상이나 prefix EOF가 필요하지 않다.

이 자료는 실패한 Drive ArchVis와 다른 공식 Lab_Scene 원본이다. 이번 검증은 원격 요청0이며, 전체 gzip CRC와 실제 Colab 사건의 직접 원인은 확정하지 않는다. 근거는 `outputs/test_fleet_calibration_001_20261103/real_lab_repro.py`, `real_lab_repro/proof.json`, `real_lab_repro/control.json`이다.

### 같은 입력·예산의 인과 비교

전체 gzip CRC가 정상인4.8MB 합성 원본을 사용했다. Source read soft4MiB/hard8MiB, decode32MiB, 모의 Drive body16MiB/158HTTP를 고정하고 production 파일을 바꾸지 않은 격리 overlay로 비교했다. 개발의4개 비교를 root도 별도 출력 경로에서 독립 재실행해 동일한 결과를 확인했다.

| 실험 | 바꾼 부분 | 결과 |
| --- | --- | --- |
| A | 원본 그대로 | 논리 read4MiB/decoded532,480bytes에서 `read_bytes_limit → ZranError`. Partial 결과를 반환하지 못함 |
| B | `gz.read` 예외에서 `budget.last_source_failure` 복원만 추가 | A와 같은 소비량에서 정상 partial 반환. 완전한400,000bytes 파일1개를 시험 store에 저장하고 새 reader로 SHA 동일 재추출. Scan+재추출 logical8MiB 이내 |
| C | B에 buffer2MiB 추가 | Partial 파일5개 반환. Checkpoint까지 logical5,308,416bytes 사용했으나 남은 예산으로 마지막 파일 재추출 실패. 버퍼 변경만으로 해결했다고 볼 수 없음 |
| D | B에 첫 deflate block 손상 입력 | 65,536bytes read 뒤 `ZranError` 유지, source budget 사유 없음. 실제 손상을 정상 budget 중단으로 오분류하지 않음 |

A의 보존 결과0은 오류 전에 완전한 파일을 하나도 파싱하지 않았다는 뜻이 아니다. B는 실패 분류만 바꿔 이미 완전한 파일을 반환·보존할 수 있음을 보여준다. 따라서 **읽기 한도 예외 복원 누락이 partial 진전을 잃게 만드는 원인**임을 이 대조 실험으로 확인했다. 작은 read의 반복 inflate가 예산을 빠르게 소진하는 문제는 별개이며 C만으로 해결이 입증되지 않았다.

이 검증은 직접 scanner와 시험용 Store.commit/extract 경로다. Full runner의 settlement, 다음 scan 재개, Drive 영구 보존까지 검증한 것이 아니며 production 수정본도 아니다. 실측·예약을 구분하는 실패 보고 보완은 여전히 필요하다. 개발 근거는 `causal_developer/final_trial/proof.json`, root 독립 근거는 `causal_developer/root_trial/proof.json`이며 모두 `outputs/test_fleet_calibration_001_20261103/` 아래에 있다. 기존 Drive pending과 예산은 변경하지 않았다.

## 현재 범위와 결정

2026-11-03 사용자는 학습과 별도로 전체 데이터셋 인덱싱 코드를 준비하고, **산출한 비용의 2배를 상한**으로 사용하기로 했다. `rewind_fleet`는 이 목적의 별도 구현이다. 기존 v1 인덱스·g25·RGB102/269 실행 계약과 학습 gate를 수정하지 않는다. 전체 원본 다운로드·결합·추출, 전체 학습을 자동 시작하지 않는다.

현재 구현은 원본별 순서와 identity 고정, 작은 gzip restart pack, 불변 catalog/commit 저장, 누적 예산 유지, 재개, 제한된 파일 추출 API, CLI와 CPU Colab 진입점을 제공한다. 실제 전체 원본 순회, Colab 처리량, 새 런타임의 Drive 복원은 아직 검증하지 않았다. 새 store를 기존 학습 준비 큐에서 직접 소비하는 어댑터도 별도 작업이다. M0/M1 또는 전체 데이터 학습 준비 완료로 해석하지 않는다.

## 확인한 원본 규모

Drive 메타데이터에서 90개 archive, 354개 분할 파일, 총 **3,094,462,001,717bytes**를 관측했다. RGB·pose·depth 각각 train_val/test 15개 scene이다.

| 종류 | 조각 수 | 압축 bytes |
| --- | ---: | ---: |
| train_val RGB | 249 | 2,597,969,076,181 |
| test RGB | 35 | 281,377,458,373 |
| train_val pose | 15 | 18,020,771,152 |
| test pose | 15 | 1,967,837,325 |
| train_val depth | 25 | 176,199,211,971 |
| test depth | 15 | 18,927,646,715 |

[UnrealEgo 공식 페이지](https://unrealego.mpi-inf.mpg.de/)의 UnrealEgo2 다운로드 스크립트 6개에 명시된 scene별 조각 수와 90개 모두 일치했다. 스크립트를 실행하지 않고 목록만 대조했다. 출처 URL·스크립트 SHA는 [관측 inventory](../configs/fleet_inventory_observed.json)에 보존한다. 조각 수 일치는 내용 무결성 증명이 아니다. Runtime에서 각 조각의 size/modifiedTime을 재대조하고 version을 고정해야 한다. 현재 관측 파일의 version은 미확인 상태다.

가장 큰 KyotoAlley train_val RGB는 569,460,442,516bytes/54조각이다. 전체 gzip 인덱스를 RAM에 한꺼번에 불러오는 기존 경로로 이 규모를 처리한다고 가정하지 않는다. Test 자료는 인덱싱 대상일 뿐 학습 대상으로 허용하지 않는다.

## 저장·재개 방식

- `indexed_gzip==1.10.3`의 동일 원본에서 생성한 restart point와 32KiB dictionary를 4MiB 이하 pack으로 저장한다. 절대 compressed/decoded offset과 bit 위치를 유지하고, 독립적으로 만든 gzip index 바이너리를 합치지 않는다.
- Archive commit은 부모 commit·catalog 구간·pack·원본 SHA·마지막 완전한 tar member 경계를 참조한다. 완전하지 않은 파일은 관측 목록에 올리지 않는다. 이름·ordinal은 archive 순서를 보존한다.
- 과거 pack은 과거 파일 추출에 재사용한다. 기존 인덱스 범위 밖으로 무제한 자동 seek/build하지 않는다. RGB/pose 학습 대응은 별도의 sequence/frame 대조가 필요하다.
- Catalog SQLite는 런타임 로컬 cache다. 정상 재개 시 추가 commit만 반영하고, 새 런타임에서는 제한된 양씩 복원한다. Live SQLite를 Drive에 두지 않는다.
- Drive에는 닫힌 불변 object·commit marker·HEAD를 기록하고 readback/hash를 확인한다. 임시 object와 이전 HEAD를 자동 삭제하지 않는다. FUSE rename/exclusive-create를 분산 원자성 또는 원격 영구 보장의 증거로 주장하지 않는다.
- 같은 store에는 한 writer만 사용한다. 강제 종료로 남은 lease/pending은 이전 owner 종료 확인과 정확한 token/HEAD를 사용한 명시 복구만 허용한다. 알 수 없는 소비량은 예약 전액을 사용한 것으로 유지한다.

## 비용 산출과 상한

`planning.plan`은 선택한 archive별 추정치와 근거를 받아 source bytes, HTTP 요청, 시간, artifact 논리 I/O, 저장 공간, RAM, 로컬 디스크를 계산한다. 각 항목의 **cap = 2 × base**이며 전체/원본별 누적 사용량을 함께 제한한다. 계획·선택 목록·inventory·원본 identity를 hash로 고정하고 실행 시 다시 계산해 대조한다. 세션이 바뀌었다고 예산을 초기화하지 않는다.

3.094TB는 전체 압축 원본을 한 번 순회할 때 읽어야 하는 입력량의 하한이다. 단순 2배인 6,188,924,003,434bytes를 최종 전송 상한으로 확정하지 않는다. 중간 재개 replay, metadata, 저장/읽기 검증과 실패 잔여 비용을 포함한 추정치를 먼저 산출한다. 시간에는 네트워크와 HTTP 지연, 해제/scan, 저장 I/O, segment 비용을 포함한다. 저장량에는 dictionary·catalog·commit·이전 정상본·새 쓰기·실패 잔여를 반영한다. `gzip ISIZE`는 2^32 나머지이므로 전체 해제 크기로 사용하지 않는다.

전체 계획의 해제 크기·member 수·처리량 등 미측정 값은 [추정치 템플릿](../configs/fleet_estimates_template.json)의 null로 남아 있으며, null을 0으로 대체하거나 전체 실행을 허용하지 않는다. Calibration의 가정값을 90개 전체 원본 실측으로 복사하지 않는다. 측정 후 확장비·파일 밀도·재개 비용의 편차와 근거를 검토하여 추정치를 작성하고 계획을 고정한다.

정상 segment는 실제 사용량에 명시된 정산 여유를 더해 기록하고, 소비량을 확인할 수 없는 중단은 사전 예약 전액을 유지한다. Artifact I/O는 프로그램이 읽고 쓴 논리 bytes이며 Drive FUSE의 실제 네트워크 bytes와 같다고 보장하지 않는다. 시간·RAM·디스크 검사는 협력적 검사이며 중단 불가능한 OS/원격 호출의 엄밀한 실시간 차단을 보장하지 않는다.

## 첫 실행: 한정 비용 측정

[CPU notebook](../notebooks/fleet_index_workbench.ipynb)은 배포 전 root가 source revision/SHA를 고정해야 하는 template이다. 사용자 실행 안내는 이 프로젝트의 데이터셋 탐색 세션에서 진행한다. RGB269와 다른 runtime·workspace·Drive store를 사용한다.

1. 고정 source ZIP·input ZIP·READY notebook을 준비한다. 기본 `ACTION="inventory"`는 원본 요청 없이 목록과 미측정 계획만 확인한다.
2. `pin`으로 선택 archive의 metadata를 64HTTP/300초/2MiB 이내에서 조회한다. 새로운 runtime inventory JSON 경로를 설정한다.
3. `calibration_plan`으로 고정된 소량 probe 정책의 계획을 계산한다. 이 정책의 source 목표 4MiB, decode allowance 32MiB와 처리량 가정은 첫 측정 예산을 정하기 위한 값이다. 전체 archive 비용 예측이 아니다.
4. 계획 hash를 확인하여 `EXPECTED_PLAN_SHA256`에 고정하고, `calibrate`로 **1 segment**만 실행한다. 기본 선택은 train_val RGB/ArchVisInterior다. 전체 원본 실행·실패 후 자동 반복은 하지 않는다.
5. 성공/실패 모두 마지막 회수 셀로 작은 보고서·계획·runtime inventory·store 상태를 보관한다. 전체 index store를 ZIP으로 복제하지 않는다. 최종 결과를 탐색 세션이 취합한다.

기본 ArchVis probe는 source bytes base 8,712,192/cap 17,424,384, HTTP 79/158, 누적 시간 138/276초, 저장 19,416,192/38,832,384bytes, RAM 574,619,648/1,149,239,296bytes, 로컬 디스크 25,296,896/50,593,792bytes, artifact 논리 I/O 6,760,704/13,521,408bytes다. 원본 metadata pin 비용은 별도로 보고한다. 한 segment 예약은 이 계획의 잔여량과 source 16MiB/1000HTTP/120초 중 작은 값이며, 일반 scan source soft limit은 4MiB다. Notebook 준비에는 별도로 가용 RAM 2GiB·디스크 4GiB를 확인한다.

중단 조건은 source identity/metadata 변경, 원본 불완전, 저장/경로/hash 불일치, 예산 소진, 완료 경계 진전 없음, 기존 writer/pending 감지다. 자동 상한 확대·임의 token 복구를 하지 않는다. 실제 비용 계획 및 새 runtime 보존 검증 후 전체 실행의 고정 입력을 따로 전달한다.

## CLI와 전체 실행의 후속 경계

저장소 루트에서 `python -m rewind_fleet --help`로 inventory/pin/calibration-plan/plan/run/status/recover-pending/release-stale-lease를 확인할 수 있다. `run`은 `--plan-sha256`, `--single-writer`, `--max-segments`, `--session-seconds`를 명시해야 한다. 기존 output 파일을 덮어쓰지 않는다. `status --store <path>`는 원본 요청 없이 ledger를 검사한다.

전체 실행은 검토된 별도 estimates JSON으로 `plan --scope full`을 작성하고 그 SHA를 고정한 `run`으로 수행한다. Notebook 기본 입력 estimates는 null이므로 `full_plan/index`를 선택하는 것만으로 전체 실행이 시작되지 않는다. 작은 calibration store와 전체 계획 store를 섞거나 이전 ledger의 계획 hash를 바꾸지 않는다. Calibration 결과를 full store로 이관하는 기능은 아직 없다.

Catalog EOF, rolling trailer 일치, 단일 gzip 전체 무결성, 선택 archive 완료, 90개 전체 완료는 다르다. 기본 `catalog_only`는 두 번째 전체 CRC pass를 자동 실행하지 않는다. `full_bounded` 검증은 32MiB 이하 작은 fixture 범위이며 큰 원본의 별도 전체 검증기는 아직 제공하지 않는다. 따라서 전체 원본 catalog를 채워도 현재 정책에서 전체 무결성 완료로 승격하지 않는다.

## 검증 근거

최종 개발 검사 21개와 root 독립 검사 21개가 모두 통과했다(failure/error/skip 0, root 9.204초). 범위는 `test_fleet`, `test_fleet_window_storage`, `test_fleet_frontend`이며, 54개 조각의 metadata-only pin, 실제 인증 transport의 숨은 retry 차단, notebook 계획/인증 gate와 실패 회수도 포함한다. Notebook 코드 셀 구문 검사, input ZIP CRC와 inventory/estimates의 원본 JSON 대조도 통과했다.

로컬 합성 archive에서 여러 원본의 누적 cost×2, 중단 예약 보존·명시 복구, 새 로컬 workspace 복원, 과거 파일 추출 SHA, 원본 identity 변경 차단을 확인했다. 비트 offset 4의 중간 restart에서 원점 재읽기 없이 300,000bytes를 동일하게 복원했다. 별도 4.8MB/두 조각 fixture는 5개 segment/12개 member를 저장하고 최초·중간·최종 파일 3개를 추출했다. 이 결과는 실제 3TB·Drive·Colab 검증을 대신하지 않는다.

Root 근거는 `outputs/full_index_prepare_20261103/root/`, 개발 근거·input ZIP은 `outputs/fleet_index_20261103/developer/`에 보관한다. 실제 archive media 원격 읽기는 이번 준비에서 하지 않았다. Colab calibration, Drive 새 runtime 복원, 전체 실행 비용 확정, 전체 무결성 검증, 학습 adapter 연결이 남아 있다.
