# v2 두 번째 메타데이터 DB의 변경 기록 검사

부모는 `outputs/v2_plan_inputs/user_test/metadata_result_20261119T041837Z/results.zip`이다. ZIP SHA-256은 `0c0e685a079c68f7ac6c51874d45dc2e5feaad3120cc03cde0a0e3006e03800a`, SQLite SHA-256은 `d61aacaaecf38c427d6c62d31668edbb1be9d7bb2ce1fa5b1db0afd2a961dfc5`다. 부모는 메타데이터 408,033행을 저장했고, 미완료 폴더는 0개다. 변경 검사가 `source_changed`로 끝났으므로 완료 표시는 없다.

고정 ZIP에서 꺼낸 DB 사본을 읽기 전용으로 연다. DB 무결성과 원래 토큰, 공식 split/spec, 부모 사용량, 삭제 대상이 이전에 관측된 적 없는지 확인한다. 진단 worker의 외부 호출은 저장된 토큰의 Drive `changes.list`로 제한하며, 한도는 16요청·8MiB 응답·120초다. 별도 진단이므로 파일 재목록이나 토큰 재발급 경로를 호출하지 않는다. 마지막 페이지까지 확인해 지정 삭제 한 건만 있어야 한다. 허용 ID는 `1it9nMyJ-TxZeE3hB_aWDAh2Xcux9aBU9`다. 휴지통 표기라면 폴더 타입을 확인하고, 부모가 응답에 있으면 원래 부모 `1atCfluTn9ehgJQiv8myr0LbcORNhIN_m`와 대조한다. 토큰 오류, 추가 변경, 중복 삭제, 불완전 페이지, 예산 초과는 로컬 대조를 차단한다. [Drive changes.list](https://developers.google.com/workspace/drive/api/reference/rest/v3/changes/list)

통과한 같은 프로세스가 사본 DB를 순회하고, 디스크 기반 SQLite 인덱스로 sequence/frame/role을 비교한다. `report.json`에 시나리오와 공식 train/validation별 좌·우·pose 파일 수, 관측 프레임 수, 정확히 한 파일씩 대응한 프레임, 누락·중복 역할, SHA-256 미확보 파일 수와 바이트, 전체 합계를 기록한다. `comparison_complete`는 대조가 끝났다는 상태다. `frame_correspondence_passed`는 역할 누락·중복 및 미관측 공식 sequence가 모두 0일 때만 true다. 실패 기록은 첫 변경 불일치 종류, 지정 대상 일치 여부와 변경 타입을 토큰·경로 없이 남긴다. `issues.jsonl`은 로컬 대조에서 식별된 프레임·역할과 중복 경로를 기록한다. 파일 3종이 모두 없는 프레임은 DB만으로 발견할 수 없다. 실제 decode나 payload SHA 검증은 이 진단 범위 밖이며 manifest 생성·128MiB 상한과 독립적으로 회수한다.

## 고정 노트북 발급과 회수

소스 ZIP은 검토한 Git revision의 `SOURCE_REVISION.txt`, 실행 코드와 `reference/spec.json`, `reference/split/{train,validation,test}.txt`를 포함한다. 빌드 명령은 다음과 같다.

```text
python -m scripts.build_v2_metadata_fence --source <source ZIP> --source-sha <SHA256> --revision <40자리 revision> --output <새 폴더>
```

생성된 `v2_metadata_fence.ipynb`와 옆의 `receipt.json`을 고정한다. Colab `/content`에 정확한 source ZIP과 두 번째 부모 `results.zip`을 준비한다. 설정 셀 기본값은 모두 꺼져 있다. `VERIFY_ONLY`는 인증 없이 로컬 부모 검증을 수행한다. `RUN_CHANGES_AND_COMPARE`는 검증 후 변경 조회와 조건부 로컬 대조를 한 호출로 실행한다.

선택적 `SAVE_RECOVERY_TO_DRIVE`에는 기존 마운트 경로를 지정한다. 요청 전에 존재·권한·여유 공간만 확인한다. worker가 끝난 뒤에만 새 ZIP 이름으로 복사하고 크기·SHA-256을 재확인한다. 오류가 나면 로컬 ZIP으로 다운로드할 수 있다. 회수 ZIP에는 보고·이슈·로그를 넣으며 원본 DB와 결과 ZIP은 별도로 보존한다. VM 자체가 강제 종료되면 이 종료 처리가 실행되지 않을 수 있다.

실제 부모 검증과 로컬 fixture를 수행했다. 실제 Colab 인증 및 Drive 변경 응답은 사용자 테스트에서 확인한다.
