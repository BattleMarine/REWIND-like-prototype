# 2026-11-10 프로젝트 폴더 정리

현재 v1.1 회수 결과와 v1.2 테스트 자료, 고정 입력/소스 ZIP, 실제 샘플·checkpoint·시각화는 원래 위치에 보존했다. 소스 패키지·configs·notebooks·tests·가상환경·원본 데이터는 변경하지 않았다.

과거 시험·정리 기록과 로컬 bootstrap 검증 복사본 등40개 항목(4,642파일,176,664,991bytes)을 `.local-archive/20261110/` 아래에 원래 상대경로로 옮겼다. 이동 전후 전체 파일 SHA256·크기를 확인했다. 빈 시험 디렉터리16개만 삭제했다. 아카이빙은 위치 정리이며 디스크 공간을 확보하는 압축/삭제는 아니다.

아카이브는 기존 `.gitignore`의 `/.local-archive/`로 제외된다. 상세 파일 해시는 아카이브의 `move_manifest.json`, 확인 결과는 `verification.json`에 있다. 과거 문서에 아래 경로가 나오면 `.local-archive/20261110/`을 앞에 붙여 찾는다. 복원은 원래 경로가 비어 있는지 먼저 확인한 뒤 해당 항목을 되돌린다.

## 이동한 경로

- `outputs/cleanup_20261023`
- `outputs/cleanup_20261102`
- `outputs/cleanup_20261105`
- `outputs/documentation_refresh_20261105`
- `outputs/downloads_salvage_20261103`
- `outputs/drive_pilot_review`
- `outputs/frame_input_20261102`
- `outputs/full_training_readiness_20261102`
- `outputs/mini_prototype_prepare_20261103`
- `outputs/readme_cleanup_20261105`
- `outputs/real_gt2d_diagnostic_20261102`
- `outputs/real_rgb_diagnostic_20261102`
- `outputs/rgb_index_probe_20261029`
- `outputs/test_archvis_left_counterpart_20261017`
- `outputs/test_local_m1_cpu_65c4287_20261028`
- `outputs/test_local_rgb_cpu_20261028`
- `outputs/test_rgb_pose_manual_5340cbe_20261028`
- `outputs/test_synthetic_gpu_20b7aa9_20261102`
- `outputs/video_input_plan_20261102`
- `outputs/test_tmp`
- `outputs/archvis_v0_20261109/root_bootstrap`
- `outputs/archvis_v0_20261109/root_candidate`
- `outputs/archvis_v0_20261109/root_release_bootstrap`
- `outputs/archvis_v1_diagnostic_001/root_release_bootstrap`
- `outputs/archvis_v1_handoff_20261110/bootstrap_uploads`
- `outputs/archvis_v1_handoff_20261110/bootstrap_workspace`
- `outputs/archvis_pilot_handoff_20261109/root_release_bootstrap`
- `outputs/archvis_handoff_commit.txt`
- `outputs/archvis_plan_commit.txt`
- `outputs/archvis_restore_commit.txt`
- `outputs/archvis_results_commit.txt`
- `outputs/direct_rgb_training_commit.txt`
- `outputs/extracted_input_commit.txt`
- `outputs/v0_training_decision_commit.txt`
- `outputs/v1_actual_commit_message.txt`
- `outputs/v1_audit_result_commit.txt`
- `outputs/v1_cache_preserved_commit.txt`
- `outputs/v1_pipeline_commit_message.txt`
- `outputs/v1_plan_commit.txt`
- `outputs/v1_teacher_commit_message.txt`
