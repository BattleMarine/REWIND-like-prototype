# 프로젝트 정리 및 다운로드 보존 대조

2026-11-16 수행. 사용자가 확인한 대상은 `C:/Users/ktsvc/Downloads`다. 다운로드 파일은 삭제·이동·이름 변경하지 않았다. 프로젝트 소스·현재 v1.5.1 인계 자료·모델과 실험 결과의 고정 경로는 유지했다.

## 다운로드 대조 결과

- 전체 목록:104파일,4,860,424,568bytes.
- REWIND 관련 결과·캐시/감사 보고서·참고 자료:38파일,4,518,702,449bytes. 이 중34파일은 기존 프로젝트 파일과 크기·SHA256이 같았다.
- 기존 동일 파일을 찾지 못한4파일은 원형 복사해 보존했다. 새 자료의 보존은 실험 결과 검토 완료를 뜻하지 않는다. 출처 미확정 bootstrap 보고서는 이름에도 unattributed로 표시했다.
- 사용자의 이름 정리 요청에 따라38개 다운로드에 대응하는37개 고유 복사본을 `named_downloads/`에 만들고 크기·SHA256을 다시 확인했다. 보존량4,516,763,180bytes(약4.52GB). `reports.zip`과`reports (1).zip`은 같은 내용으로 한 파일에 대응한다.
- 검사 대상 ZIP은 CRC 검사를 통과했다. ZIP 무결성·복사 일치 확인이며, 내부 결과 수치를 새로 검증하거나 모든 실험이 성공했다고 판정한 것은 아니다.
- 나머지66파일,341,722,119bytes는 개인/타 프로젝트 자료 등 범위 밖이다. 보존을 보장하지 않으며 **Downloads 전체 삭제 가능 판정이 아니다.**

## 이름과 파일 위치

아카이브 루트는 [`.local-archive/20261116_download_audit/`](../.local-archive/20261116_download_audit/)다. `.gitignore`의 기존 규칙으로 전체 제외된다. 큰 원본은 Git에 올리지 않는다.

- [이름 정리된 복사본](../.local-archive/20261116_download_audit/named_downloads/)
- [삭제 전 확인용38파일 대조표](../.local-archive/20261116_download_audit/verified_downloads.csv): 다운로드 절대경로·크기·SHA256·보존 경로 포함.
- [보존 미확인66파일 목록](../.local-archive/20261116_download_audit/DO_NOT_DELETE_unverified.csv): 삭제 가능 목록과 분리.
- [전체 대조 근거](../.local-archive/20261116_download_audit/download_audit.json)
- [프로젝트 이동 전 해시 목록](../.local-archive/20261116_download_audit/cleanup_plan.json) · [이동 후 검증](../.local-archive/20261116_download_audit/cleanup_verified.json)

파일 이름은 실험명·자료 종류·SHA256 앞12자리를 조합했다. 원본과의 대응은 CSV/JSON으로 남겼고, 출처를 식별할 수 없는 경우 실험명을 추정하지 않았다. 기존 고정 ZIP·notebook 이름은 재현 계약을 위해 그대로 두었다. 따라서 이름 정리된 보존본은 기존 보존 자료와 별도 복사본이며 이번 작업이 디스크 사용량을 줄인 것은 아니다.

| 다운로드 이름 | 새 아카이브 이름 |
| --- | --- |
| results.zip | v1.2.1_citypark_downtown_inference_results__c302fede5d8f.zip |
| results (1).zip | v1.2.2_archvis_camera_control_results__48ca2d5cab15.zip |
| results (2).zip | v1.2.3_camera_origin_gauge_results__f870ea95b26d.zip |
| results (3).zip | v1.5_camera_training_results__a46e6a65f377.zip |
| training.zip | v1.1_t4_training_checkpoints__fcfbac1c26af.zip |
| training (1).zip | v1.4_gt_oracle_training_checkpoints__b78537b060a8.zip |
| training (2).zip | v1.2_archvis_full_training_checkpoints__48661e75e096.zip |

## 프로젝트 정리

총9개 항목의1,655파일·101,745,865bytes를 `project_cleanup/outputs/`로 이동하고 모든 파일의 크기·SHA256 일치를 확인했다.

- 이전 `outputs/v12_training_development/`, `outputs/v14_oracle_development/`의 개발 검증본.
- `outputs/test_tmp/`에 남은 임시 검증 자료. 빈 test_tmp는 기존 테스트 진입점 호환을 위해 다시 만들었다.
- outputs 최상위의 과거 커밋 메시지5파일과 `documentation_20261116/`의 커밋 메시지.

실제 학습 결과, 모델, 캐시, 현재 v1.5.1 입력·소스·노트북·검토 자료는 고정 위치에 남겼다. 소스 모듈을 쓰이지 않는다고 추정해 제거하지 않았다. v1.4 문서의 이동된 개발 검증 경로는 수정했다. 파일 삭제나 새 학습·GPU 테스트는 하지 않았다.

## 다운로드 삭제 전 주의

대조표의38개 파일은 점검 시점의 내용과 동일한 복사본을 확보했다. 이후 같은 이름으로 다시 다운로드되거나 내용이 바뀐 파일까지 포함하지 않는다. 이38개만 선별 정리할 수 있으며,66개 미확인 파일은 별도 보존 여부를 판단해야 한다. 프로젝트의 `.local-archive` 자체를 함께 삭제하면 이번 보존본도 사라진다. 로컬 보존이며 원격 백업/Drive 검증은 하지 않았다.
