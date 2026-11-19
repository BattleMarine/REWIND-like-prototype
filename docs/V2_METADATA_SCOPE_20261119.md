# v2 변경 범위 검사와 저장 DB 대조

이 진단은 04:18:37 UTC의 두 번째 결과 ZIP과 저장된 변경 토큰을 사용한다. 부모 ZIP과 SQLite pin·무결성 검사는 기존 검사와 같다. 04:32 진단과 기존 재개본은 그대로 보존한다. [두 번째 DB와 고정 pin](V2_METADATA_FENCE_20261119.md)

## 조회와 판정

저장된 토큰에서 변경 페이지를 끝까지 먼저 회수한다. 다음으로 수신한 각 변경의 파일 ID와 부모 ID를 기존 `entries`, `queue`, 루트에 대조한다. 이미 차단 사유가 있어도 남은 변경을 예산 안에서 조사한다. 공통 조상의 조회 결과를 재사용하며, 사전 판단에는 변경 응답의 파일 메타데이터를 사용한다. 통과 후보는 현재 파일을 한 번 더 조회해 변경 응답과 대조한다. 기존 토큰의 `changes.list`와 필요한 `files.get` 메타데이터만 호출한다. 외부 호출의 전체 한도는 16회·8MiB 응답·120초다. 파일 재목록, 새 시작 토큰 발급, 파일 본문 읽기는 사용 경로에 없다. Drive 변경 레코드는 파일의 갱신 상태를 제공하며, 버전은 파일의 서버 변경을 반영한다. [Drive 변경 레코드](https://developers.google.com/workspace/drive/api/reference/rest/v3/changes) · [Drive 파일 버전](https://developers.google.com/workspace/drive/api/reference/rest/v3/files)

다음 근거를 적용한다.

- 승인된 빈 폴더의 제거 한 건을 원래 예외 규칙으로 확인한다. 알려진 dataset 파일과 폴더, 원래 dataset 루트, 계정의 My Drive 루트 자체의 변경은 외부 항목으로 통과시키지 않는다.
- 이전에 승인·고정한 외부 체인 자료가 있으면 파일과 조상 각각의 부모·타입·서버 버전을 현재값과 비교한다. 외부에서 생성한 경우에는 신뢰할 수 있는 원래 토큰 생성 시각의 상계, 생성 시 부모·타입·버전의 독립 기록 및 현재와의 일치를 요구한다. `reference/v2_external_scope.json`은 `schema=v2_external_parent_evidence_v1`, `parent_zip_sha256`, `saved_cursor_sha256`, `my_drive_root_id`, `basis`, 선택적 `fence_start_upper_utc`, ID별 `items`를 담는 검토 자료다. 각 항목의 필드는 `parent_id`, `mime_type`, `version`, `source_sha256`, 그리고 `outside_before_saved_cursor` 또는 `outside_creation_receipt`를 지정하는 `kind`다. 생성 근거에는 `created_time`과 `observed_utc`도 담는다. 과거 근거를 검토할 때 현재 위치의 스냅샷을 소급해서 작성하지 않는다.
- 이미 회수된 04:32 진단 ZIP에는 별도 근거가 있다. 이전 실행이 외부 `reports` 폴더에 새 UUID 이름으로 복사한 기록과 로컬 결과본 2,136B의 SHA-256을 사용한다. 실제 Drive 응답의 이름·크기·SHA 및 외부 부모 체인이 모두 그 기록과 일치해야 한다. 서버 생성 시각과 변경 시각이 종료 기록 이후에 있어야 하고 동일 항목의 현재 버전과 메타데이터도 변경 응답과 일치해야 한다. 같은 식별 조건으로 여러 ID가 확인되거나 관측된 조상의 변경이 있으면 차단한다. 이 근거의 판정 범위는 회수 ZIP의 관측된 외부 생성과 04:18 DB에 대한 영향이다. 직전 리포트가 이미 기록한 첫 차단 항목은 이 ZIP이 생성되기 전의 다른 항목이다.
- 일반 생성 사건은 04:18:37 UTC보다 뒤인 `change.time`과 `file.createdTime`이 UTC로 정규화했을 때 소수점 이하까지 같은 경우에 한해 검사한다. 같은 ID의 관측 사건은 한 건이어야 하고, DB 미등재·현재 버전과 메타데이터의 재조회 일치·My Drive 루트까지 변경이 없는 외부 부모 체인을 확인한다. 제공된 수정 시각이 생성보다 늦으면 차단한다. 통과 이유는 `observed_external_creation_at_change_time_after_db`이며 관측된 생성 사건과 고정 DB 시점에 한정한다. 이름이나 SHA가 특정 회수 ZIP과 일치하면 해당 ZIP의 별도 규칙을 적용한다. [Drive 변경 및 생성 시각 필드](https://developers.google.com/workspace/drive/api/reference/rest/v3/changes) · [Drive 파일 시각 필드](https://developers.google.com/workspace/drive/api/reference/rest/v3/files)

DB에 ID가 없고 현재 외부에 있다는 정보만 있으면 과거 외부 판정은 미확정이다. 삭제된 외부 추정 항목, 이전 위치 불명의 이동, 접근 오류, 변경된 부모·버전, 공유 Drive·바로가기·복수 부모·불완전 체인도 차단한다. 승인 삭제가 정확히 한 건이고 그 외 전부 외부로 판정된 경우에만 통과한다. 부모 GET이 포함된 통과 경로는 처음 조회의 종료 토큰에서 변경을 한 번 더 확인한다. 새 변경이 있으면 같은 실행에서 비교하지 않는다.

`report.json`에는 초기와 재확인 변경의 ID 및 부모 ID, 각 ID의 DB 대조 결과, 부모 체인, 판정 이유와 비용을 남긴다. 원래 토큰과 자격 증명은 회수물에 기록하지 않는다. 조사 도중 예산이나 API에 막힌 경우에도 이미 수신한 항목과 판정 근거를 기록한다. 통과 시 기존 디스크 기반 RGB·pose 대응 검사를 같은 프로세스에서 실행하고 시나리오/split별 집계와 `frame_correspondence_passed`를 기록한다. 결과 회수는 manifest 크기와 독립적이다.

## 고정본 발급

검토할 Git revision의 `SOURCE_REVISION.txt`, 코드, `reference/spec.json`, 공식 split 세 파일을 source ZIP에 포함한다. 여기에 04:32 회수된 `outputs/v2_plan_inputs/user_test/v2_metadata_changes_20261119T043247Z_e81468933e0a/results.zip`을 `reference/v2_scope_recovery_043247Z.zip` 이름으로 추가한다. 빌더가 크기와 SHA를 검증한다. 검토한 과거 메타데이터 근거가 있으면 위 schema의 파일도 함께 고정한다.

```text
python -m scripts.build_v2_metadata_scope --source <ZIP> --source-sha <SHA256> --revision <40자리 revision> --output <새 폴더>
```

노트북 `v2_metadata_scope.ipynb`의 기본 스위치는 꺼져 있다. `VERIFY_ONLY`는 인증 없이 원본 검증만 실행한다. `RUN_CHANGES_AND_COMPARE`는 기존 토큰 확인과 조건부 대응 검사를 실행한다. 종료 시 보고서와 로그를 브라우저로 회수하고, 선택 시 기존 Drive 폴더로 새 이름의 검증된 ZIP을 복사한다. 강제 VM 종료 시 이 종료 처리는 보장되지 않는다. 현재 로컬 원본 검증과 모의 Drive 응답으로 개발 검사했으며 실제 Colab 인증과 변경 조회는 사용자 테스트가 남아 있다.
