# v2 128 A · 업데이트 6,200 복구 인계

2026-11-22. 대상은 `reports/v2_gpu_training_128_v1/A-seed-42`의 `recovery-candidates/invocation-5ednwc4x-42c7bfe8f55d4fc09ab56a1e600da8e4/run`에 보관된 `last.pt`와 `run.json`이다. 복구 지점은 frontend epoch 0, batch 6,200, updates 6,200이다. 6,214는 중단 전 화면에 표시된 진행 위치다. 기존 체크포인트 체인의 첫 영수증은 `c15fe06f707ce6e14a22ea8b19b9759ff4ce42963b3f2279dd90395a22bd7aa5`다.

| 고정 입력 | 바이트 | SHA-256 |
| --- | ---: | --- |
| 후보 `last.pt` | 4,694,368 | `53b6287264d15eb5c5f9a33cf84f4d067ec1438b5c7bf8926d8cf817acd51746` |
| 후보 `run.json` | 5,143 | `45a00e519cc2e8670bcc6c59c87670e6d6df6b7864d51cacd68eb39806585c2d` |
| 기존 128 소스 ZIP | 24,477,377 | `9a68d8e45bcf3abf6fab3004740faa6805d4ce3fda5118c3a7102011937849d8` |

1. 기존 A 학습 worker가 종료됐는지 확인하고 그 VM은 유지한다. 복구용 `v2_gpu128_a_recovery_6200.ipynb`를 별도 CPU Colab에서 열거나, 기존 A VM에서 별도 노트북으로 실행한다. 사용하는 VM의 `/content`에 기존 `v2_gpu_training_128_v1.zip`을 올린다. 복구 worker는 GPU를 사용하지 않는다.
2. `RUN_RECOVERY = True`로 설정한다. 승인된 게시라면 `PUBLISH_VERIFIED_CANDIDATE = True`로 한 번 실행한다. 게시 호출은 현재 tip·기존 첫 체크포인트·후보를 확인하고, 로컬에서 정상 ZIP을 만들고 다시 복원한 뒤에만 고정본의 `Checkpoints.save()`를 호출한다. 사전 검증만 따로 할 때는 게시 스위치를 `False`로 두면 `verified_only`가 나온다. 이 별도 호출은 선택 사항이다. CPU에서 모델·optimizer·scheduler·scaler·Python/Torch CPU RNG 및 cursor를 확인한다. CUDA RNG는 형태를 검사하고 원본 바이트를 보존한다.
3. 게시된 `step-000002.json`의 ZIP을 정상 복원 경로로 다시 읽은 뒤 `published_verified`와 `normal_resume_tip`을 출력한다. 이미 게시된 동일 결과를 다시 검증하면 `already_published_verified`를 출력한다. 다른 tip, 남은 writer 또는 영수증 없는 step 2 ZIP을 만나면 게시를 중단한다. 표시된 로컬 `recovery-result.json`과 `normal_resume_tip`을 회수한다.
4. 검증된 `normal_resume_tip`을 확인한 뒤 기존 A VM의 원래 [128 실행 노트북](V2_GPU_TRAINING_128.md)으로 돌아간다. A 슬롯의 `EXPECTED_RESUME_TIP`에 새 영수증 SHA를 입력해 재개한다. 기존 VM을 유지하면 이미 복원한 로컬 128 캐시를 재사용한다.

복구 셀은 15분 안에 종료하며 128 MiB 읽기와 256 MiB 쓰기를 로컬 복구 원장으로 계상한다. 기존 A 학습 원장과 첫 gate, 후보 폴더, 공통 캐시를 보존한다. 복구 중에는 GPU 학습과 캐시 복원을 시작하지 않는다.
