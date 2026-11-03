# 중단된 Drive survey의 고정 재개 준비

## 대상과 범위

2026-11-03 사용자 요청으로 b1960de의 실제 실패 지점에서 같은 조사 계획을 이어가는 실행본을 준비한다. 상태 집계 수정6863e74를 포함하며, 원본 목록·quota·누적 예산은 바꾸지 않는다. 원격 재개를 이미 수행했다는 뜻이 아니다.

- 기존 HEAD: `171db8be3f138b555d31b31c9c2d772e8ebee25e162fdfec50718f01575855ad`
- 기존 plan: `fcd4a6ee101d9a24d945d62ffd1a6d98ba0687f1bd30c185b3431eea695b6759`
- 기존 runtime inventory: `ce8b0c852a743e67cbbb4bd89ea1e770ee713706ef310fcbe46d9ed5349c7538`
- 실패 archive: `UnrealEgoData2_test_pose/ArchVisInterior_ArchVis_RT`
- pending token: `65d75884fada496a975924b4306b992f`
- 기존 source ZIP SHA: `6908c79e8c07e615ac61bf649b819763a41fef6a2ea01621ec7c81a418d85e4e`

46개 archive는 이미 각2scan quota를 마쳤다. 나머지44개를 같은 순서로 조사하며 정상 처리한46개를 처음부터 다시 스캔하지 않는다. 전체 데이터셋 인덱싱 완료가 아닌90개 archive의 제한 조사 완료를 목표로 한다.

## 예약과 시간 정책

실패 pending의64MiB/1000HTTP/16MiB artifact 예약을 전액 사용량에 남기고 pending 상태만 명시적으로 해제한다. 관측된 소량 응답을 근거로 불확실한 소비량을 자동 환불하지 않는다. 기존 사용량은 source676,544,538bytes/3903HTTP/artifact115,777,133bytes이며, 같은 전체 cap 내 남은 양은 source11,403,050,982bytes/176,097HTTP/artifact2,904,121,747bytes다. Archive별 cap도 유지하므로 전체 잔여량이 많아도 특정 archive의 예산 소진으로 중단될 수 있다.

총 실행시간 cap은 없다. 타이머는 현재 실행 경과를 표시하고 누적 정산 관측시간과 구분한다. 개별 요청 timeout5초·숨은 재시도 차단·기타 저장 공간/RAM 상한은 유지한다. 요청 실패의 상세 원인이나 재발 방지는 아직 확인되지 않았다.

## 기존 기록 보존과 새 코드 적용

원래 Drive의 `fleet_survey_001`과 `fleet_survey_001_prepared`를 사용한다. 기존 PREPARED의 source pin을 새 ZIP 값으로 덮어쓰거나 원본 metadata pin을 다시 만들지 않는다. 원본 PREPARED·inventory·plan의 고정값과 현재 HEAD·pending·writer 부재를 먼저 읽기 전용으로 대조한다.

새 코드 적용과 정확한 pending 한 건의 복구는 별도 기록으로 남긴다. 복구 전후 usage·archive_usage·완료 quota·archive 참조가 같아야 한다. 중간 실패나 예상하지 않은 HEAD 변경은 보존하고 중단하며, 자동 추측 복구·새 store 생성·예산 초기화를 하지 않는다. 재개 실행본은 이 실패 지점에 한정한다.

사용자 안내·실제 실행·결과 취합은 탐색 세션 `01a12dd6-3940-7b91-91dd-eae1bfaba270`이 담당한다. 소스 ZIP·입력 ZIP의 이름 변경은 허용하되 내용은 고정 SHA로 검증하며, 파일을 하나씩 선택해도 이미 검증한 파일을 보존한다. 캐시 업로드는 필요하지 않다.

## 검증과 인계 상태

고정 revision·패키지 manifest는 아래 인계 경로에 보관하고 로컬 검증 결과를 이 문서에 기록한다. 실제 Drive의 pending 복구·새 코드 적용·worker 재개는 아직 실행하지 않았다. 이전 회수 ZIP은 전체 index store의 대체 복원본이 아니다.


## 실행·회수 원칙

실제 실행본은 탐색 세션이 고정 manifest의 revision/SHA를 확인한 뒤 안내한다. 새로운 CPU runtime에서 source/input을 준비하고 Drive mount 후 기존 상태를 대조한다. 사전 점검이 일치할 때만 해당 pending 복구 및 새 source 적용 기록을 남기고 같은 plan의 worker를 시작한다. 기존 b1960de notebook을 수정하거나 반복 실행하지 않는다.

진행 중 상태·타이머를 확인할 수 있고 정상 stop과 작은 보고서 ZIP 회수를 지원한다. HEAD 불일치·writer 존재·원본/입력 hash 불일치·불완전한 복구 기록은 실행하지 않고 보고한다. 연결 실패가 다시 발생하면 그 실행의 결과와 새 pending을 보존하며 같은 복구 token으로 자동 재시도하지 않는다.

재개 준비 metadata 작업은 기존 준비 출력4MiB/논리IO8MiB 상한 안에서 제한한다. worker의 기존 누적 source·HTTP·artifact 예산을 새로 발급하지 않는다. Source ZIP과 작은 입력 ZIP 준비 비용은 worker 전송과 구분하며 대용량 cache/raw archive 업로드는 없다.


## 로컬 검증

Root는 고정 case의 HEAD·pending·usage·quota·progress·inventory/plan/source/원PREPARED pin을 실제 회수 ZIP과 독립 대조했다. Worker 종료는 앞서 사용자가 제공한 status의 exit0·lease없음에 근거하며, 회수 ZIP만으로 성공 종료를 추론한 것이 아니다.

핵심 독립 검사15개가20.046초에 통과했다(실패0, skip0). 읽기 전용 preflight의 파일 불변, 예약 전액 보존, 기존46 archive 참조와 quota 유지 후 같은 plan 재개, 다른 HEAD·writer·prepared/source pin 차단, 복구 중간 crash 차단, 복구기록 회수와 저장공간 상한, 상태 집계 및 기존 mock Drive 회수를 확인했다. 합성 원본/로컬 store 기반이며 실제 Drive 복구 성공을 대신하지 않는다. 근거: `outputs/fleet_resume_prepare_20261103/root/core_tests.json` 및 `case_check.json`.


노트북 독립 검사6개도0.266초에 통과했다. 하나씩 업로드·자동 변경/임의 이름의 두 ZIP·잘못된 SHA·취소·기존 파일 재사용, 고정 worker 인자/비차단 시작/stop 연결, 허용된 오류코드와 setup 실패 진단 회수를 확인했다. 실제 Colab 브라우저 렌더링과 Drive 재개 성공은 아직 미검증이다. 근거: `outputs/fleet_resume_prepare_20261103/root/notebook_tests.json`.

고정 실행본은 `outputs/fleet_resume_prepare_20261103/handoff/<revision>/` 아래 source ZIP, input ZIP, READY notebook, MANIFEST.json으로 보관한다. 추적 `notebooks/fleet_survey_resume_READY.ipynb`는 pin 치환 전 template이므로 직접 실행하지 않는다. 실제 실행 인계는 manifest의 정확한 SHA와 경로를 기준으로 한다.

최종 stop 경계 보완 후 root 관련2개 검사가4.828초에 통과했다. 복구 중 새 STOP 요청을 과거 중단처럼 자동 승인/해제하지 않고 첫 탐색 경계에서 멈추며, 일반 정상 재개도 유지한다. 앞선 핵심15개와 정상 재개1개가 중복된다. 근거: `outputs/fleet_resume_prepare_20261103/root/stop_tests.json`.
