# 독립 자동 데이터셋 인덱싱

> **보존 기록 · 2026-11-05 정리:** 2026-11-04 사용자 결정으로 분할 압축 원본의 인덱싱·부분 읽기 운영을 중단했다. 이 문서 전체의 실행 명령, 다음 단계, 예산과 미완료 표시는 해당 실험 당시의 기록이며 현재 실행 지시가 아니다. 새 기본 경로는 [압축 해제된 RGB·pose 사용](DATA_ACQUISITION_STRATEGY.md)이다. 기존 코드·원본·인덱스·예약 기록은 보존하며 자동 재개하지 않는다. 이동된 로컬 근거는 [보관·복원 안내](LOCAL_STORAGE.md)를 따른다.

2026-11-03 당시 실행 정책: 새 Drive survey는 총 경과시간 상한 없이 타이머로 관측하고 사용자 stop을 지원하도록 준비한다. 아래 기존 full/legacy profile의 시간 예산 설명과 구분하며, 이미 고정한 plan·pending을 소급 변경하지 않는다. 시간 외 자원 cap은 유지한다. [새 조사 정책과 준비 상태](FLEET_DRIVE_SURVEY.md).

## 사용자 결정과 목표 (2026-11-03)

당시에는 전체 데이터셋의 자동 인덱싱 프로그램 구현을 우선했다. 정상 구간마다 사람이 실행·검토·승인하는 운영을 반복하지 않는다. 프로그램은 고정 원본과 누적 예산 안에서 순차 탐색, 파일 완전성 검사, 인덱스 저장, 사용량 정산, 다음 segment/archive 진행을 자동 수행한다. 미니 프로토타입의 데이터는 기존371쌍에 머무르지 않고1,000~10,000쌍 규모를 지향한다. 이 수치는 확보 완료량이나 전체 비용 추정치가 아니다.

당시에는 전체 인덱싱과 미니 프로토타입을 병행하고 인덱싱을 먼저 개발하도록 정했으며, 이 우선순위는2026-11-04에 철회됐다. 분할 압축 원본은 Drive에 유지하고 필요한 바이트만 읽는다. 전체 결합·추출 파일 생성은 기본 동작이 아니다. 실제 원격 대규모 실행은 원본 pin과 비용 산출·사용자 결정의2배 상한을 확정한 후 수행한다.

## 구현 범위

구현한 새 자동 profile `bounded_auto_v1`은 기존 legacy/calibration plan과 구분한다. 기존 저장소의 plan과 pending을 바꾸거나 새 job으로 예산을 초기화하지 않는다. 기존 runner의 반복 루프를 활용하고 해제 출력 버퍼1MiB, gzip spacing1MiB, 압축 입력 readbuf64KiB의 bounded 읽기 경로를 적용한다.

이 경로는 한 segment의 engine으로 chunk를 읽고, 구간 경계에서 checkpoint 저장·정산 후 다음 segment를 연다. 해제기의 내부 상태를 전체 실행 동안 계속 유지하는 새 native 엔진을 구현한 것은 아니다. 반복 읽기 비용은 없어졌다고 주장하지 않고 긴 자동 실행의 실제 진전·총비용으로 평가한다.

정상 동작에는 수동 승인을 끼워 넣지 않는다. 사용자는 실행과 상태 조회 및 정상 중단을 요청할 수 있다. 한도 소진, 원본 변경, 인증/요청 실패, 손상, 불확실한 pending, 진전 없음은 이유를 남기고 중단한다. 중단 원인을 무시하는 자동 반복·예산 증액은 하지 않는다. 프로세스 강제 종료와 정상 중단은 구분한다.

## 완료 판단과 검증 경계

- 한 호출에서10개 이상 segment를 처리하고 다음 archive로 자동 이동한다.
- 정상 중단 후 같은 plan/store와 새 workspace에서 기존 진전을 복원한다.
- 이전·중간·마지막 인덱스의 파일을 SHA 동일하게 추출한다.
- 예산, 원본 identity, 저장 실패, 손상, no-progress 경계에서 잘못된 완료나 무한 반복이 없다.
- 장기 실행의 이벤트 RAM을 제한하고 누적 상태와 최근 진전을 조회한다.
- 실제 보존 Lab prefix에서 완전 파일 진전을 확인하되 전체 archive CRC나 Drive 내구성을 대신하지 않는다.

Catalog 항목 수는 RGB/pose pair 수가 아니다. Pair 확보는 sequence/frame 키의 좌우RGB·pose 대조로 별도 계산한다. Catalog EOF, trailer 검사, 전체 무결성, 학습 사용 가능 상태도 구분한다. 현재 비용·기존 실패 내역은 [전체 인덱싱 상세](FLEET_INDEXING.md)를 따른다.

## 실행 진입점

저장소 루트에서 `python -m rewind_fleet auto-run --help`를 확인했다. 새 계획은 기존 `plan` 명령에 `--execution-profile bounded_auto_v1`을 명시해 만든다. 검토된 inventory와 estimates의 누락값을 임의로 채우지 않는다. 프로그램이 실행 가능한 것과 실제 전체 원본에 대한 입력·예산이 확정된 것은 다르다.

| 명령 | 동작 |
| --- | --- |
| `plan --execution-profile bounded_auto_v1` | profile·원본·설정·비용을 hash로 고정 |
| `auto-run` | 동일 plan 안에서 정상 segment와 다음 archive를 자동 처리 |
| `status` / `coverage` | 원본 요청 없이 저장된 진전·예약/정산·검증 수준 조회 |
| `stop` | 정확 plan SHA에 묶인 협력 중단 요청 |
| `resume` | 같은 plan/store에서 stop을 확인하고 잔여 예산으로 재개 |

`auto-run`과 `resume`에는 inventory, plan, plan SHA, store, local workspace, owner, single-writer, session-seconds, 새 output 경로를 전달한다. `max-segments` 기본100만은 예산이 아니며 실제 종료는 누적 cap·세션 시간·완료/오류 조건으로 제한된다. 원본 pin은 기존 `pin` 명령을 사용한다. legacy/calibration plan과 기존 실패 store를 새 profile로 바꾸지 않는다.

[CPU/Colab notebook](../notebooks/fleet_auto_index.ipynb)은 준비된 source·의존성·Drive mount·ADC 인증·확정 plan을 사용하는 얇은 진입점이다. 기본 ACTION=status는 원본을 읽지 않는다. auto-run/resume은 별도 CPU subprocess를 실행하고 셀은 즉시 반환하므로, 같은 notebook에서 ACTION=status/stop으로 마지막 셀을 실행할 수 있다. Worker log·최종 report 경로와 종료코드를 표시한다. Runtime 종료 시 작업은 멈추며 Colab runtime을 자동 재생성하는 기능은 없다. 정상 재개에는 같은 store/plan과 새 로컬 cache를 사용할 수 있다.

정상 실행은 segment마다 불변 catalog/pack/commit과 정산 receipt를 저장한다. 최근 이벤트64개만 보고서 RAM에 보존하고 누적 segment·파일·디렉터리·decoded 진전은 ledger에 남긴다. 마지막 HEAD에 묶인 RUN.json 요약은 예약된 정산 여유 안에서 best-effort로 남기며 저장 실패도 최종 보고서에 표시한다. 강제 종료·입력 검사 실패 등으로 최종 보고서가 없으면 worker log와 HEAD/pending/lease를 확인한다. 전체 index store를 회수 ZIP으로 매번 복제하지 않는다.

`stop`은 정상 segment 경계에서 반영되는 협력 중단이다. 진행 중 파일/네트워크 호출을 즉시 취소하는 보장은 없다. 새 segment 예약은 남은 세션 시간에 맞추고, member 경계에서도 checkpoint 여유를 남기도록 중단한다. OS/원격 호출의 강제 제한은 아니므로 hard kill이나 불확실한 실패의 pending은 명시 복구가 필요하다. 정상 stop과 새 runtime 재개가 누적 예산을 초기화하지 않는다.

## 로컬 검증 결과와 남은 경계

- 한 호출에서 두 합성 archive의800파일을 로컬16segment로 자동 처리했다. 최초·중간·마지막 파일 총6개 SHA를 확인했다. 이 파일 수는 실제 RGB/pose 페어 수가 아니다.
- 3segment 뒤 stop, stop 상태에서 추가 source 읽기 없이 유지, 같은 store/plan과 새 workspace의 resume을 검증했다. 원본 bytes/요청 누적 정산은 이전 사용량에 후속 실소비를 더한 값이다.
- Mock Drive에서도14segment/800파일과 body/HTTP 실제 합계 정산을 확인했다. 실제 Drive 요청이 아니다.
- 65개 성공 뒤66번째 실패를 주입해 최근 이벤트64개·누적진전65·실패 예약 보존을 확인했다. 이 항목은 fake scan의 운영 계약 검사로 실제 인덱싱 처리량 증거가 아니다.
- 실제 보존 Lab prefix는 자동2segment에서24PNG+8디렉터리/decoded16,161,280bytes를 저장했다. Root 독립 재실행5.172초,24개 표준 tar SHA·크기·PNG decode 일치, 최초·마지막2개 인덱스 추출을 확인했다. 추출은 별도 읽기 전용 예산이다.
- Lab 시험은 full profile soft28MiB/hard32MiB와 누적 잔여 예산으로 수행했다. 이전 calibration4MiB/8MiB와 동일 예산 비교가 아니다. 원격0이고 불완전 prefix이므로 전체 archive CRC나 실제 Drive 내구성 검증을 대신하지 않는다.
- 개발 최종 통합42개(121.288초)와 RUN 요약 표시 보완 후 관련5개 검사가 통과했다.
- Root 중간 회귀36개(113.763초), 작은 파일 checkpoint와 notebook worker 보완 후 영향 검사16개(12.487초)가 통과했다. 후속16개에는 기존 예외 검사10개가 중복 포함되며 별개의52개 시험으로 세지 않는다. CLI/노트북 기본 status의 인증 없음, worker 즉시 반환/활성 worker 중 stop/중복 실행 차단, 작은 파일의 lazy checkpoint를 검증했다.

근거는 `outputs/fleet_auto_20261103/developer/`와 `outputs/fleet_auto_20261103/root/`다. 산출물은 Git에서 제외하고 코드·검사·운영 문서만 커밋한다.

실제 전체 인덱싱은 아직 실행하지 않았다. 90개 원본의 runtime pin, 근거 있는 전체 비용 추정 및2배 상한 확정, 실제 Drive 저장/새 runtime 복원, 대규모 처리량·storage 증가 검증이 남았다. 자동 프로그램의 로컬 동작 검증을3TB 처리 시간 보장이나1,000~10,000페어 확보 완료로 해석하지 않는다. 대규모 full CRC 검증기와 학습 adapter/pair join도 별도다. 현재 cache 공간 검사는 workspace 파일 목록을 순회하므로 규모 증가 시 비용을 측정해야 한다.
