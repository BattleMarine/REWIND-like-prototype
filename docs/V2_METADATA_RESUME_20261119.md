# v2 메타데이터 11월 19일 부분 DB 재개

부모는 `80f00d5`의 `outputs/v2_plan_inputs/user_test/metadata_result_20261119T031803Z/results.zip`이다. ZIP 바이트 SHA256은 `ffd3c77b03d1bff8aad209d3f9e6f07ee555398c863e644c1f861cc140e919bf`, SQLite 바이트 SHA256은 `b8f03ce299ac71ef5133d1a770d3274ee3a73f9c8c85272e7db57c52cab17027`이다. 직전 run은 동일 경로 폴더 두 개를 만나 해당 페이지 트랜잭션을 롤백했다. DB에는 메타데이터 79,345행과 원래 변경 토큰이 남았다.

- 이전 사용량은 2,417요청 및 33,671,553 응답 바이트다. 재개의 남은 요청은 17,583, 남은 응답 바이트는 503,199,359다.
- 이전 실행은 경과시간을 기록하지 않았다. ZIP의 멤버 시각으로는 시작 시각이나 이전 사용시간 상계를 산출할 수 없다. 이전 실행시간 근거가 고정되면 총 14,400초에서 이전 상계, 재개 준비 실측, 회수 예약 300초를 뺀 값을 worker 한도로 사용한다.
- 테스트 세션에서 이전 시간의 보수적 상계를 확인하면 노트북 설정 셀의 `PRIOR_WALL_UPPER_SECONDS`와 `PRIOR_WALL_BASIS`에 입력한다. 동일 노트북과 source ZIP을 계속 사용한다. 값이나 근거가 비면 인증 전에 중단한다. 노트북은 부모 ZIP SHA를 포함한 `resume_elapsed_evidence.json`을 만들고, 해당 파일과 그 SHA를 회수 ZIP에 보존한다.

노트북 빌더는 `python -m scripts.build_v2_metadata_resume --source <고정 ZIP> --source-sha <SHA> --revision <Git revision> --output <새 경로>`다. ZIP에 `SOURCE_REVISION.txt`, 재개 코드, 검증된 `reference/spec.json`, `reference/split/{train,validation,test}.txt`를 넣는다. 배포 receipt는 source와 노트북의 해시를 고정한다. 기본 실행 스위치는 꺼져 있다. `VERIFY_ONLY`로 source, 원본 ZIP 멤버, DB 해시·무결성·usage·공식 split/spec과 root·롤백된 커서의 연결을 로컬에서 확인한다.

`RUN_RESUME`은 쓰기 가능한 DB 사본에서 시작한다. 먼저 저장된 토큰의 변경 기록을 읽는다. 허용 항목은 원래 DB에서 관측되지 않은 폴더 ID `1it9nMyJ-TxZeE3hB_aWDAh2Xcux9aBU9`의 제거 한 건이다. API의 `removed=true` 또는 같은 ID의 폴더가 `removed=false`와 `file.trashed=true`로 응답한 경우를 확인한다. 부모가 제공되면 롤백된 원래 부모 `1atCfluTn9ehgJQiv8myr0LbcORNhIN_m`과 일치해야 한다. 다른 항목, 반복 제거 항목, 재개 후 변경은 차단한다. 완료 receipt의 `allowed_changes`에는 실제 API 관측값과 확인 단계를 기록한다.

SQLite는 각 성공 페이지의 상태를 보존한다. 일반 종료, 오류, 수동 중단에서 노트북의 종료 처리가 사본 DB·로그·완료 목록을 새 회수 ZIP에 넣고 다운로드를 제시한다. Drive 보관에는 설정 셀의 `SAVE_RECOVERY_TO_DRIVE`와 기존 마운트 대상 `DRIVE_RECOVERY_DIRECTORY`를 사용한다. 목록 조회 전에는 대상 폴더 존재·쓰기 권한·최소 원본 DB 크기의 여유 공간만 확인한다. 조회가 끝나면 실제 ZIP 크기와 남은 회수 시간 안에서 새 이름으로 복사하고, 완성본 크기와 SHA-256을 읽어 검증한다. Drive 복사가 실패해도 로컬 ZIP을 유지한다. VM 자체가 강제 종료되면 종료 중인 복사나 ZIP 생성은 보장되지 않는다. 중단된 재개 결과는 새로운 고정 인계에 사용한다. 원본으로 반복 실행하면 누적 예산을 보존할 수 없다.

이 작업은 메타데이터만 읽는다. 완료 목록을 회수한 뒤 로컬 producer에서 SHA·coverage·manifest 크기를 검토한다. 실제 Colab의 인증, 변경 응답, 재개와 회수는 사용자 테스트에서 확인할 항목이다. [v2 입력 목록 생산](V2_INPUT_PRODUCTION.md)
