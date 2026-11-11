# 산출물 위치 안내

2026-11-11 기준. `outputs/`는 실행에 필요한 현재 자료와 자주 확인하는 결과를, `.local-archive/`는 과거 실행 자료를 보관한다. 두 폴더 모두 Git 제외 대상이다. 파일 이동은 용량 확보용 삭제나 압축이 아니다.

## 현재 사용하는 경로

| 경로 (`outputs/` 기준) | 용도 |
| --- | --- |
| `v11_rerun_20261110/` | v1.1 재실행 소스·노트북·인계·회수 결과 |
| `auto_retention_20261111/` | v1.2 자동 결과 보관 보완본과 인계 |
| `v1_branch_experiments/` | v1.1/v1.2 이전 고정본·비용 추정·실제 테스트 결과 |
| `v12_preflight_120min/` | v1.2 120분 사전검사 고정본과 검토 |
| `v12_full_prepare/` | 전체 입력 snapshot·coverage와 로컬 검증 기록 |
| `archvis_v1_diagnostic_001/` | 현재 공통 input/evidence ZIP, 첫 진단 모델·결과·시각화 |
| `real_stereo_sample_20261102/` | 현재 시각화/진단에서 참조하는 실제 stereo 샘플 |
| `rgb102_diagnostic_20261103/` | 기존 pose 감사 스크립트가 참조하는 진단 근거 |
| `test_rgb102_pairs_001_20261103/` | 기존 pose 감사 스크립트가 참조하는 회수 ZIP |
| `presentation/` | Joyful Jump GT 이미지와 발표 자료 |

현재 고정 인계의 ZIP·notebook·receipt는 이름과 경로를 유지했다. 실제 실행 버전은 각 폴더의 TEST_HANDOFF/receipt를 따른다. 폴더 이름이나 최종 수정일만으로 버전을 선택하지 않는다. 회수 ZIP·모델·입력 캐시는 함부로 삭제하지 않는다.

## 발표 자료

- `outputs/presentation/joyful_jump_gt_20261104/`
- `outputs/presentation/joyful_jump_gt_all57_20261104/`
- `outputs/presentation/joyful_jump_presentation_20261104/`

GT 관절을 그린 이미지는 모델 예측 결과가 아니다. 이전 경로에서 presentation 아래로 이동했으며 내용은 보존했다.

## 과거 자료 찾기와 복원

이번에 옮긴 과거 자료는 `.local-archive/20261111_outputs/outputs/<기존 폴더명>/`에 있다. 상세 원래/새 경로와 파일별 크기·SHA256은 `.local-archive/20261111_outputs/move_manifest.json`에, 이동 후 확인 결과는 `verification.json`에 보관한다. 과거 문서의 아래 경로를 찾을 때는 아카이브 위치를 사용한다. 원위치 복원이 필요하면 원래 경로에 새 자료가 없는지 먼저 확인한다.

- archvis_drive_audit_20261109
- archvis_pilot_handoff_20261109
- archvis_v0_20261109
- archvis_v1_20261110
- archvis_v1_epoch_expansion
- archvis_v1_handoff_20261110
- rgb102_pair_prepare_20261102
- rgb269_prepare_20261103
- test_rgb269_pairs_001_20261103
- unrealego2_upload_20261106

2026-11-10 정리본은 별도 `.local-archive/20261110/`에 유지한다. [이전 정리 기록](WORKSPACE_CLEANUP_20261110.md).

## 이후 저장 원칙

새 임시 검증 파일은 해당 실험의 하위 폴더에 모은다. 산출물을 outputs 최상위에 낱개로 만들지 않는다. 진행 중인 인계의 고정 경로를 정리 목적으로 변경하지 않으며, 실험이 끝나고 의존성을 확인한 뒤 이동 목록과 해시를 남겨 아카이빙한다.

## 이번 정리 검증

최상위 폴더23개를10개로 줄였다. 총13개 폴더의3,772개 파일(1,997,894,010bytes)을 이동했고 전체 크기·SHA256 일치를 확인했다. 파일 삭제는 없으며 빈 test_tmp 폴더1개만 제거했다. 현재 테스트·입력·샘플 참조 경로는 유지했다.
