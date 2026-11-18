# v2 공식 split 입력 목록과 manifest 생산

고정 v2 인프라 `6417fbc`와 연결하는 준비 단계다. 이 가이드의 검증은 로컬 fixture와 보존된 공식 split·spec을 사용했다. 실제 Drive 목록 수집·PNG decode·pose 검사·학습은 별도 인계 범위다.

수집기는 사용자 인증 환경에서 Google Drive v3 메타데이터를 읽는다. 바이너리 파일의 `sha256Checksum`은 API가 제공하는 경우에만 사용한다. 파일 본문 다운로드 경로는 없다. [Drive files 리소스](https://developers.google.com/workspace/drive/api/reference/rest/v3/files) · [Drive 페이지 조회](https://developers.google.com/workspace/drive/api/reference/rest/v3/files/list?hl=en)

1. root가 정한 RGB/pose 루트 ID, 공식 split 파일 디렉터리, spec 파일 경로·SHA, 작업 경로와 호출별 요청·시간·응답 bytes 예산을 입력한다. Colab에서는 사용자가 먼저 `google.colab.auth.authenticate_user()`를 호출한다. 같은 source checkout에서 `python -m scripts.list_v2_drive_metadata --help`로 인자를 확인한다. 아래 변수는 고정 인계값으로 채운다.

   ```bash
   python -m scripts.list_v2_drive_metadata \
     --rgb-root-id "$RGB_ROOT_ID" --pose-root-id "$POSE_ROOT_ID" \
     --spec "$SPEC" --spec-sha "$SPEC_SHA" --splits "$SPLITS" \
     --db "$METADATA_DB" --output "$LIST_OUTPUT" \
     --max-requests "$REQUEST_CAP" --max-seconds "$SECONDS_CAP" --max-bytes "$RESPONSE_BYTE_CAP"
   ```

2. SQLite는 페이지의 파일 메타데이터, 새 폴더, 다음 토큰을 하나의 트랜잭션으로 보존한다. 부분 완료 시 같은 DB·루트·공식 split pin으로 명시 재호출한다. 다른 프로세스가 같은 페이지를 먼저 게시했거나 API가 토큰을 거부하면 중단된다. 토큰 만료 복구에는 새 DB에서 별도 승인된 목록 조회가 필요하다. VM 교체가 예상되면 사용자 관리의 영구 위치에 DB를 보존한다. 한 DB는 한 프로세스에서 사용하고 복사 전에 연결을 닫는다.

3. 조회 전 각 루트가 속한 Drive 범위의 변경 토큰을 고정한다. 조회 후 변경 기록에 항목이 하나라도 있으면 `source_changed`로 중단한다. 보수적으로 같은 Drive 범위의 관련 없는 쓰기도 차단 조건에 포함한다. 수집 DB는 조회 중인 Drive 범위 밖에 보관한다. 이 검사는 API에 표시된 변경 기록 범위에서만 적용된다. [Drive 변경 토큰](https://developers.google.com/workspace/drive/api/reference/rest/v3/changes/getStartPageToken) · [Drive 변경 조회](https://developers.google.com/workspace/drive/api/reference/rest/v3/changes/list)

   완전 조회 상태에서만 exporter가 `rgb.jsonl`, `pose.jsonl`, `receipt.json`을 발급한다. 각 JSONL 행에는 `scenario, role, path, file_id, size_bytes, sha256, checksum_source, modified_time, version`이 들어간다. `path`는 RGB·pose 루트 각각에 상대적이다. 부분 페이지, 누락 공식 폴더, 동일 경로 중복은 완료 receipt 발급을 차단한다.

4. 목록을 로컬로 회수해 receipt 바이트 SHA와 함께 producer에 전달한다.

   ```bash
   python -m scripts.produce_v2_inputs \
     --receipt "$LIST_RECEIPT" --receipt-sha "$LIST_RECEIPT_SHA" \
     --spec "$SPEC" --spec-sha "$SPEC_SHA" --splits "$SPLITS" --output "$NEW_OUTPUT"
   ```

Producer는 공식 sequence 소속, 관측된 sequence/frame 각각의 좌·우 RGB와 pose, 파일 경로·버전·크기와 SHA를 대조한다. 공식 split 텍스트는 sequence만 고정하므로 세 파일이 전부 없는 프레임은 이 검사에서 발견할 수 없다. 6개 시나리오/split별 관측 페어 수와 비용, 미확보 SHA 건수와 bytes, 문제별 행은 `report.json`과 `issues.jsonl`에 남긴다. 목록·hash가 준비되고 크기 상한을 충족하면 현재 계약 이름으로 `frozen_inventory.json`과 `audited_manifest.json`을 발급한다. 결과의 provenance에는 `decode_audit=not_run`이 기록된다.

API가 SHA를 제공하지 않으면 producer가 미확보 파일과 총 바이트를 보고하고 발급을 차단한다. 별도 승인된 bounded payload 감사 결과나 과거 감사와 현재 Drive 객체 버전의 동일성까지 입증한 경우에만 `--evidence/--evidence-sha`로 검토된 JSONL을 전달할 수 있다. 보충 행은 `file_id, version, size_bytes, path, sha256, source, evidence_sha256`을 요구한다. `source` 값은 `bounded_payload_sha256` 또는 `prior_audit_same_object_version`이다. 날짜·이름·크기가 같다는 이유로 과거 SHA를 현재 행에 복사하지 않는다. 입력 evidence 파일의 SHA는 producer가 검증하며, 각 행의 원본 증거를 검토하고 고정하는 책임은 인계 단계에 있다.

공식 sequence 수는 ArchVis train 286·validation 30, CityPark 756·108, Downtown 371·48이다. 보존된 공식 경로로 가상 140,000페어를 canonical JSON으로 직렬화한 크기는 약 142.2MiB였다. 가장 긴 공식 sequence와 12자리 frame 번호를 적용한 보수적 크기는 약 167.6MiB였다. 기존 소비기 상한은 134,217,728B다. 실제 목록이 오면 producer가 정확한 후보 직렬화 크기와 SHA를 보고한다. 상한을 넘으면 검증된 목록·SQLite·보고서는 유지하고 고정 실행용 manifest와 그 pin 발급을 차단한다. v2 로더에만 192MiB 제한을 적용하는 안은 실제 입력의 로드·validator 최대 RSS와 직렬화 버퍼를 계측한 뒤 검토한다.
