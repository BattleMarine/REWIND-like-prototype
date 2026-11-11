# 장시간 실행 결과의 자동 보관

2026-11-11 사용자 요청: 브라우저/VS Code 연결이 끊겨도 런타임이 작업을 계속한다면 결과를 회수할 수 있도록 종료 처리에서 Drive에 보관한다. 자동 브라우저 다운로드는 클라이언트 연결에 의존하므로 기본 보관 경로로 삼지 않는다.

## 적용 범위

- v1.1 재실행: `bf56968`부터 학습 호출의 finally에서 모델 회수 ZIP을 Drive에 보관한다. 이번에는 학습 코드·모델·기존 고정 실행을 변경하지 않는다. [v1.1 보관 계약](V11_RERUN_PREPARATION.md).
- v1.2 준비: 새 notebook에서 preflight와 audit 각 호출의 종료 처리에 보고서 자동 회수·Drive 보관을 연결한다. 성공·실패·시간 초과 모두 해당 단계의 보고서를 보존하고 단계별 결과를 분리한다. 원본 파일이나 대용량 입력 캐시는 업로드하지 않는다.
- 과거 notebook 전체에 소급 적용하지 않는다. 실제 실행에는 새 소스 ZIP과 새 notebook의 고정 인계를 사용한다. 이미 실행 중인 런타임에 새 코드가 자동 적용되는 것은 아니다.

## 실행과 보관 판정

사용자가 실행을 켜면 실제 Drive mount와 새 보관 폴더 쓰기를 먼저 확인한다. 실행 종료 후 로컬 ZIP을 만들고 새 Drive 경로에 복사한 뒤 마운트 파일시스템에서 복사본을 다시 읽어 크기와 SHA256을 확인한다. 기존 파일은 덮어쓰지 않는다. 검증 receipt가 있어야 복사 검증 완료이며, 콘솔의 파일명만으로 모델/보고서를 확보했다고 판정하지 않는다. v1.1은 receipt가 training 종류이고 last_sha256과 실제 모델 ZIP이 있어야 모델 회수다. v1.2 보고서 보관은 전체 감사 완료나 train_ready를 의미하지 않는다.

보관 실패는 원래 실행 실패와 별도로 남기며 로컬 ZIP을 삭제하지 않는다. 런타임에 다시 접속할 수 있으면 수동 회수 셀에서 재시도한다. Drive 저장이 불가능하고 런타임에도 재접속하지 못하면 끝 처리만으로 자료 보존을 보장할 수 없다. VM 자체 강제 종료나 커널 사망 시에는 finally가 실행되지 않을 수 있다. 중간 checkpoint의 주기적 원격 동기화는 미구현이다. v1.2 부분 감사 재개는 후속 [저장·재개 계약](V12_AUDIT_RESUME.md)에 따라 새 버전에서 보완하며 완료 기록도 회수 ZIP에 포함한다.

## 현재 실행 제한

v1.1 학습2시간, v1.2 사전검사120분·내용검사6시간 및 기존 데이터/읽기/RAM/디스크 범위를 유지한다. 자동 회수·복사는 별도의 제한된 보관 작업이다. 캐시 업로드 규칙은 바꾸지 않는다. 이번 보완으로 v1.2의 이전6시간 검사에 이어읽기가 생기지 않으며, 단순 재실행을 자동 시작하지 않는다.

## 구현과 로컬 검증

v1.2는 `scripts/v12_recovery.py`와 notebook의 단계 종료 처리에 연결했다. Drive 결과는 `REWIND-like prototype/reports/v1_branch_experiments/v12_recovery/{preflight,audit}/attempt-*`의 reports.zip·receipt.json이다. Receipt에는 phase·original_work·source revision/ZIP hash·input hash·snapshot/coverage pin을 남긴다. 로컬 ZIP은 BASE/recovery/local-*에 보존하며 수동 회수 셀의 RECOVERY_LOCAL_ZIP으로 새 attempt에 재시도한다.

쓰기 시험60초, package/copy는 각각 내부600초·외부630초다. 기존 recover의 ZIP256MiB 상한과 로컬 회수본 예산512MiB를 적용한다. 원래 단계 예외를 보존하며 보관 실패 기록은 별도로 남긴다.

Root가 v1.2 회수 및 기존 v1.1/진단 관련15개 로컬 검사를 통과했다. 작은 실제 report ZIP의 생성·복사·context 보존, 손상/중복 방지, 실패 시 로컬 보존, 성공/실패/시간 초과/KeyboardInterrupt에서 finally 호출, 원래 오류 보존, 무마운트 및 기본 no-op를 확인했다. 실제 Drive·Colab 환경 검증은 아직이며 새 전체 데이터 감사나 GPU 학습을 실행하지 않았다. 검토 후 한국어 로컬 커밋과 고정 notebook 인계만 진행한다.
