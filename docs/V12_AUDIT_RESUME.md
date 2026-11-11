# v1.2 파일 감사 저장·재개

2026-11-11 사용자 선택: v1.1을 기존 고정 조건으로 다시 진행하며, v1.2는 완료 구간 저장·재개 기능을 먼저 보완한다. 감사 시간만18시간으로 늘리는 방안은 선택하지 않았다.

## 목적과 고정 범위

이전6시간 실행의9,678페어 카운트는 개별 파일 검사 근거가 없어 재개점으로 쓸 수 없다. 새 버전에서 검증한 완료 페어의 기록부터 재사용한다. 전체 train20,888·validation2,236, 120분 사전검사·6시간 감사·65GiB 읽기·RAM3GiB·보고/회수512MiB 범위를 유지한다. 모든 파일의 내용 검사가 끝나고 전체 key/순서/중복/split/기존 고정 해시를 대조한 뒤에만 완성 manifest를 게시한다. `train_ready=false`, `temporal_ready=false`는 유지한다.

## 보존 계약

완료 기록에는 sequence/frame·split·motion·좌우RGB/pose의 경로/크기/SHA와 기하 검사 결과를 보존한다. 원본 source plan·snapshot·input 및 검사 코드 identity에 바인딩한다. 재개 시 저장본 해시·순서·완료 범위와 원본의 관측 identity를 검증하고 미완료 구간부터 내용 검사를 수행한다. 미완료 파일/페어는 다시 검사하며 카운트를 그대로 완료 근거로 승격하지 않는다.

파일 크기·mtime 등의 재대조는 변경 감지 수단이며 원본 전체 바이트를 다시 읽는 해시 검증과 다르다. 저장된 SHA는 완료 당시 읽은 내용의 근거다. 동일 메타데이터로 바뀐 내용까지 항상 검출하거나 원본 전체가 한 시점의 원자적 snapshot이었다고 주장하지 않는다. 원본 추출본을 검사 중/재개 사이에 수정하지 않는 운영 조건을 유지해야 한다.

## 종료와 복원

로컬 저장 기록을 기존 자동 종료 회수 ZIP과 Drive 보관에 포함하고, 새 런타임에서는 고정 ZIP/receipt를 검증해 새 작업 경로로 복원한다. 호출별 자원 한도와 전체 누적 진행량·읽기 비용을 구분한다. 시간 제한에 닿았다고 자동으로 예산을 초기화하고 무한 반복하지 않는다. 각 호출의 중단 상태와 보관 결과를 확인한 뒤 인계된 방식으로 재개한다.

실행 중 정기 Drive 동기화는 이번 범위에 포함하지 않는다. VM이 종료돼 마지막 Drive 보관을 못 했다면 로컬에만 있던 최신 기록은 잃을 수 있다. 마지막 검증된 Drive 회수본부터 복원한다. 재개 가능한 감사 기록은 RGB 학습 캐시가 아니므로 향후 학습 입력 준비는 별도다.

## 구현·실행 경계

`rewind_prepare/full_audit_journal.py`가 페어마다 JSONL append/fsync와 해시로 봉인한 head의 원자 교체를 수행한다. 저장 도중 잘린 마지막 행은 확정 prefix에 포함하지 않으며, 재개할 때 버리고 그 페어부터 다시 검사한다. 완료 페어의 geometry 집계와 split 간 동일 내용 검사를 재구성한다. Manifest 게시 직전 실패하더라도 완료 기록으로 다시 검증해 게시할 수 있다.

같은 런타임에서는 audit의 `--resume`, 새 런타임에서는 `restore-audit`로 고정 회수 ZIP을 새 WORK에 복원한다. Notebook의 복원 셀에 ZIP 경로·SHA와 source_plan SHA를 넣는다. 원본 root의 절대 경로도 plan의 일부이므로 같은 Drive 마운트 경로를 유지한다. 코드 identity가 달라진 기록이나 구버전 count-only ZIP은 거부한다. 새 호출에서는 목록/stat 확인은 다시 필요하나 완료 페어의 PNG·pose 본문을 다시 읽지 않는다.

Journal 상한128MiB, 완성 manifest64MiB, 회수 대상 합계 및 단일 ZIP256MiB, 로컬 회수본 예산512MiB를 유지한다. WORK의 기존 파일과 추가 예약 공간(journal128MiB+manifest64MiB+metadata4MiB)이512MiB를 넘으면 새 WORK 복원을 요구한다. 과거 실패 partial을 임의 삭제하지 않으며, 이미 게시한 manifest를 확인하는 재개에서는 새 pending을 만들지 않는다.

현재 고정 목록의 가장 긴 sequence와 파일명 자릿수를 적용한 보수적 크기 추산은 journal약44.4MB, manifest records약26.2MB, source plan selection약16.3MB다. snapshot/coverage와8MiB 여유를 합친 회수 대상 추산은약96.4MiB로256MiB 이하다. 실제 전체 산출물의 측정값이나 모든 가능한 입력의 수학적 상한은 아니며 코드의 실제 크기 검사를 유지한다.

초기 구현 시 실제 Drive 재개는 미실행이었다. 아래 후속 결과에서 실제 재개 전진을 확인했으며 전체 내용 감사·확대 학습은 아직 미완료다. 완료 기록의 SHA는 과거 내용 검사 증거이며 재개 시 새 전체 해시 검증을 한 것으로 보고하지 않는다. 최종 성공도 전체 내용 감사 완료까지이며 temporal 로더와 전체 학습 준비 완료를 의미하지 않는다.

Root가 저장·재개12개와 기존 전체 준비·추출본·회수 검사를 합쳐30개 로컬 검사를 통과했다. 완료 기록 손상·원본 변경·코드 변경·기존 해시 대조·split 중복 거부, 저장 중 head 교체 실패, 0페어/부분 ZIP 복원, manifest 게시 실패 뒤 본문 재읽기 없는 마무리를 확인했다. 실제 Drive에서의 mtime 안정성과 전체 실행 비용은 테스트 세션에서 확인한다.

## 2026-11-11 실제 중단 기록 검토

고정 소스 `e94c64d8689adc6186979b56906e1008b53fb76c` 실행은 notebook 기록상7,638.2초(2시간7분18초)에 KeyboardInterrupt로 중단됐다. 6시간 상한 도달이 아니다. 테스트 세션은 연결 단절 후 프로세스 종료와 수동 회수를 보고했으며, 브라우저 연결 단절 자체가 원인인지는 로그만으로 확정하지 않는다.

`outputs/restart_20261111/v12_resume_release/user_test/interrupted_001/reports.zip`은1,939,269bytes, SHA256 `9f514c4171e4ea8483628489f2e7814d686276afa555cba804ea292016f2df9e`다. Drive attempt `attempt-hwquttrq`의 verified receipt와 local receipt가 일치한다. Root가 ZIP CRC 및 내부11개 파일 크기/SHA를 독립 확인하고, 고정 source/input ZIP을 풀어 실제 `restore-audit` CLI를 새 로컬 WORK에 실행했다.

복원 검증은 성공했다. Journal의4,921개는 모두 train이며 전체23,124개 중약21.3%, 남은18,203개다. 고정 plan/snapshot/producer, 순서·hash chain·geometry 기록·motion·기존 파일 해시 및 split 충돌 검사에 통과했다. 확정 journal8,862,573bytes, 미확정 tail0bytes다. 저장된 완료 payload 크기 합계는11,571,683,696bytes이며 원본 본문을 로컬에서 다시 읽은 것은 아니다.

`audit_status`의4,750은250페어 간격으로 게시한 이전 상태이고, 매페어 확정 head의4,921이 재개 기준이다. Head invocation의status=running은 중단 전 마지막 저장 상태이며 현재 프로세스 생존을 뜻하지 않는다. 다음 audit 진입 시 이전 invocation을 interrupted로 기록한다. 누적 관측 read11,573,815,121bytes는 저장 경계의 하한이므로 정확한 최종 전송량으로 보고하지 않는다.

재개 source_plan SHA는 `df9f5fca3139de9e3f94d2ce9d86857d676a75f556b9d3b777c2ae4ed32a5401`이다. 기존9678 count-only 결과와 합치거나 그 plan을 사용하지 않는다. 검토 내역은 같은 interrupted_001 폴더의 `root_review.json`, 복원본은 `root_restore_review/restored/`다.

테스트 세션에 동일e94c64d 소스·기존 notebook과 위ZIP/pin으로 새 Colab WORK 복원을 인계한다. 재개 전 실제 Drive 목록/크기/mtime 검사는 아직이며, 변경이 감지되면 자동 우회하지 않는다. 재개된 실행이4,921개 기록을 재사용하고 이후 페어로 전진하는지 확인해야 실제 Drive 재개 검증 완료다. 기존6시간/65GiB/RAM3GiB 및 보관 상한은 유지한다. 이번에는 수동 회수 성공을 확인했으며 자동 종료 보관만으로 회수됐다고 기록하지 않는다.

## 2026-11-11 두 번째 중단·실제 재개 검토

동일 e94c64d 실행은 기존4,921페어를 재사용하고 신규7,498개를 검사하여12,419/23,124페어(약53.7%)까지 전진했다. 모두 train이며 남은10,705개는 train8,469와 validation2,236이다. 전체 감사·temporal 준비·학습 준비는 미완료이고 train_ready/temporal_ready=false다.

이번 호출11,565.397초(약3시간12분45초), read17,582,999,773bytes, peak RSS235,937,792bytes다. 6시간 상한 도달이 아니라 KeyboardInterrupt로 stopped됐다. 사용자 연결 단절 보고와 예외 기록은 확인했으나 단절 자체가 예외를 일으켰다고 확정하지 않는다. Head invocation의 failed는 예외 종료 기록이며 journal 손상 판정이 아니다.

자동 Drive 보관 receipt는 verified, attempt-9tk6zqu2다. `outputs/restart_20261111/v12_resume_release/user_test/interrupted_002/reports.zip`은3,825,010bytes, SHA256 `ebe17d13e1ced56dbb17b181021fec3e6b5e4024ec2a421a9d19cd1887ec742e`다. Root가 CRC와 내부12파일 크기/SHA·receipt binding을 검사했다. 이전4,921행 journal이 새 journal의 동일한 바이트 prefix임을 확인하고, 고정 source/input으로 실제 restore-audit CLI를 새 로컬 WORK에 실행해12,419페어의 chain/order/geometry/motion/split 검증을 통과했다. 미확정 tail은0bytes다.

Journal22,400,111bytes, binding `e485c8dc60a5b62f58cccdfa22c822f2dd92ff82ebfc2f1070de8cacb11c3afb`, last `134c56aa5f9d099215e2b61e3f178855ae9dda461a45107fef0938dcfa12f366`이다. Source plan과 snapshot은 이전과 동일하다. 누적 관측 read29,156,814,894bytes는 첫 호출의 하한을 포함하므로 전체 정확한 전송량이 아니다.

이번 실제 실행의 재사용·전진은 확인했다. Root의 검토는 회수 증거와 로컬 복원이며 현재 Drive 본문을 새로 읽거나 실시간 metadata를 검사한 것은 아니다. 완료 당시 hash와 재개 시 path/size/mtime 대조의 한계는 유지한다.

동일 폴더 `root_review.py/json`, `root_restore_stdout.txt`, `RESTART_HANDOFF.md`에 근거와 다음1회재개 인계를 저장했다. 새 WORK에 이번 ZIP을 복원하고 pairs_reused=12419에서 전진하는지 확인한다. 기존6시간/65GiB/RAM3GiB와 저장 상한을 유지하며 자동 반복·새 preflight·구9678카운트 병합은 하지 않는다.
