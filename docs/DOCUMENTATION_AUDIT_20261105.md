# 문서 전수 감사와 로컬 정리 — 2026-11-05

## 현재 기준과 발견한 불일치

조사 기준은 시작 revision `0953dfc`의 Git 추적 Markdown 60개 전체다. README·AGENTS·ROADMAP을 포함해 현행 지시, 구현 계약, 실험 기록을 구분했다. 인덱싱 운영 중단 및 사용자 추출본 사용 결정은 2026-11-04, 이번 문서·보관 정리는 2026-11-05다.

- 분할 압축·Range 읽기·자동 인덱싱을 기본 또는 최우선으로 요구하던 지시를 **압축 해제된 RGB·pose 확보와 미니 프로토타입** 방향으로 바꿨다. depth는 현행 필수 입력이 아니다. 중앙 기준은 [데이터 확보 전략](DATA_ACQUISITION_STRATEGY.md)이다.
- 과거 인덱싱의 ‘다음 실행’, 시간·전송 예산과 pending은 과거 계약으로 보존한다. 자동 재개·예산 초기화·예약 해제를 승인하지 않는다. 마지막 재개 시험의 실제 연결 실패도 기록했다.
- 기존 371페어는 train이며 독립 validation은 없다. 새 로컬 Joyful Jump 57프레임은 중복·split·motion 검토 전이므로 428개 학습 입력으로 합산하지 않는다. 목표 1,000~10,000페어는 확보 실적과 구분한다.
- 실제 102페어 진단, 합성 GPU 성공, Joyful 3프레임의 실제 예측 실패, 57프레임 GT 시각화 171PNG를 분리했다. GT 점 표시는 모델 성능이 아니다.
- 추출 폴더를 범용 학습 manifest/release로 연결하는 adapter와 실제 평가가 미완료임을 명시했다. M2/M3의 합성 검증을 실제 학습 완료로 올리지 않았다. 영상 지원 방향과 화면·학습 코어 분리도 유지한다.
- 10-28 종료한 모델 비교용 오류 집계와 이전 테스트 지시를 역사 기록으로 구분했다. M0~M6·E1~E8·D1~D5 식별자와 고정 revision/hash는 유지한다.

## 조사 범위와 문서별 처리

원본 참고 PDF/DOCX, 의존성 문서, 아카이브 속 고정 소스·보고서는 수정 대상에서 제외했다. Notebook은 Markdown 셀의 인덱싱/압축 안내 유무를 확인했으며 고정 실행물의 내용·hash는 바꾸지 않았다. 종료된 notebook의 실행 여부는 현행 Colab 안내에서 구분한다. 코드 제거·새 adapter 구현·학습은 이번 범위가 아니다.

추적 notebook 21개의 Markdown 셀 53개를 목록화했다. 상세 근거는 `outputs/cleanup_20261105/notebook_inventory.json`에 있다.

| 문서 | 처리 |
| --- | --- |
| [AGENTS.md](../AGENTS.md) | 현행 정책·실행 본문 수정 |
| [README.md](../README.md) | 현행 정책·실행 본문 수정 |
| [docs/ARCHIVE_PREPARATION.md](ARCHIVE_PREPARATION.md) | 종료된 인덱싱 기록으로 명시 |
| [docs/ARCHVIS_POSE_AUDIT.md](ARCHVIS_POSE_AUDIT.md) | 실험 당시 범위·후속 상태 구분 |
| [docs/ARCHVIS_PREINDEX.md](ARCHVIS_PREINDEX.md) | 종료된 인덱싱 기록으로 명시 |
| [docs/AUTOMATIC_FLEET_INDEXING.md](AUTOMATIC_FLEET_INDEXING.md) | 종료된 인덱싱 기록으로 명시 |
| [docs/AUTOMATIC_INDEX_PREPARATION.md](AUTOMATIC_INDEX_PREPARATION.md) | 종료된 인덱싱 기록으로 명시 |
| [docs/BUNDLE_T1_DATA.md](BUNDLE_T1_DATA.md) | 현재 적용 범위·미완료 조건 또는 시각화 결과 갱신 |
| [docs/CAUSAL_WINDOW_REVIEW.md](CAUSAL_WINDOW_REVIEW.md) | 실험 당시 범위·후속 상태 구분 |
| [docs/COLAB_SETUP.md](COLAB_SETUP.md) | 현행 정책·실행 본문 수정 |
| [docs/COORDINATE_AND_TRAINING_REVIEW.md](COORDINATE_AND_TRAINING_REVIEW.md) | 실험 당시 범위·후속 상태 구분 |
| [docs/DATA_ACQUISITION_STRATEGY.md](DATA_ACQUISITION_STRATEGY.md) | 현행 정책·실행 본문 수정 |
| [docs/DATA_CONTRACT.md](DATA_CONTRACT.md) | 실험 당시 범위·후속 상태 구분 |
| [docs/DEVELOPMENT_REVIEW_LOG.md](DEVELOPMENT_REVIEW_LOG.md) | 실험 당시 범위·후속 상태 구분 |
| [docs/DRIVE_PERSISTENT_INDEX.md](DRIVE_PERSISTENT_INDEX.md) | 종료된 인덱싱 기록으로 명시 |
| [docs/FLEET_DRIVE_SURVEY.md](FLEET_DRIVE_SURVEY.md) | 종료된 인덱싱 기록으로 명시 |
| [docs/FLEET_INDEXING.md](FLEET_INDEXING.md) | 종료된 인덱싱 기록으로 명시 |
| [docs/FLEET_SURVEY_RESUME.md](FLEET_SURVEY_RESUME.md) | 종료된 인덱싱 기록으로 명시 |
| [docs/FRAME_INPUT.md](FRAME_INPUT.md) | 현재 적용 범위·미완료 조건 또는 시각화 결과 갱신 |
| [docs/FULL_TRAINING_READINESS.md](FULL_TRAINING_READINESS.md) | 현행 정책·실행 본문 수정 |
| [docs/G25_STORAGE_AND_POSE269.md](G25_STORAGE_AND_POSE269.md) | 종료된 인덱싱 기록으로 명시 |
| [docs/IMPLEMENTATION_HISTORY.md](IMPLEMENTATION_HISTORY.md) | 실험 당시 범위·후속 상태 구분 |
| [docs/JOINT_MAPPING.md](JOINT_MAPPING.md) | 실험 당시 범위·후속 상태 구분 |
| [docs/LOCAL_RGB_LOADER.md](LOCAL_RGB_LOADER.md) | 현재 적용 범위·미완료 조건 또는 시각화 결과 갱신 |
| [docs/LOCAL_STORAGE.md](LOCAL_STORAGE.md) | 보관·복원 결과 갱신 |
| [docs/M0_JSON_TOOLS.md](M0_JSON_TOOLS.md) | 현재 적용 범위·미완료 조건 또는 시각화 결과 갱신 |
| [docs/M0_REVIEW.md](M0_REVIEW.md) | 실험 당시 범위·후속 상태 구분 |
| [docs/M1_BASELINE.md](M1_BASELINE.md) | 현재 적용 범위·미완료 조건 또는 시각화 결과 갱신 |
| [docs/M2_BODY_TEACHER.md](M2_BODY_TEACHER.md) | 현재 적용 범위·미완료 조건 또는 시각화 결과 갱신 |
| [docs/M3_WHOLE_BODY_TEACHER.md](M3_WHOLE_BODY_TEACHER.md) | 현재 적용 범위·미완료 조건 또는 시각화 결과 갱신 |
| [docs/MINI_PROTOTYPE_PLAN.md](MINI_PROTOTYPE_PLAN.md) | 현행 정책·실행 본문 수정 |
| [docs/PERSISTENT_ARCHIVE_INDEX.md](PERSISTENT_ARCHIVE_INDEX.md) | 종료된 인덱싱 기록으로 명시 |
| [docs/PERSISTENT_INDEX_EXTENSION.md](PERSISTENT_INDEX_EXTENSION.md) | 종료된 인덱싱 기록으로 명시 |
| [docs/POSE269_DRIVE_STAGING.md](POSE269_DRIVE_STAGING.md) | 종료된 인덱싱 기록으로 명시 |
| [docs/POSE8_RECOVERY.md](POSE8_RECOVERY.md) | 종료된 인덱싱 기록으로 명시 |
| [docs/POSE_CACHE_BOUNDS.md](POSE_CACHE_BOUNDS.md) | 종료된 인덱싱 기록으로 명시 |
| [docs/POSE_MISSING67.md](POSE_MISSING67.md) | 종료된 인덱싱 기록으로 명시 |
| [docs/POSE_TAIL352_PLAN.md](POSE_TAIL352_PLAN.md) | 종료된 인덱싱 기록으로 명시 |
| [docs/PREDICTION_OVERLAY.md](PREDICTION_OVERLAY.md) | 현재 적용 범위·미완료 조건 또는 시각화 결과 갱신 |
| [docs/PREFIX_CACHE.md](PREFIX_CACHE.md) | 종료된 인덱싱 기록으로 명시 |
| [docs/PREFIX_CACHE_192_PLAN.md](PREFIX_CACHE_192_PLAN.md) | 종료된 인덱싱 기록으로 명시 |
| [docs/PREFIX_CACHE_NEXT_STEP.md](PREFIX_CACHE_NEXT_STEP.md) | 종료된 인덱싱 기록으로 명시 |
| [docs/PREPARATION_QUEUE.md](PREPARATION_QUEUE.md) | 현재 적용 범위·미완료 조건 또는 시각화 결과 갱신 |
| [docs/PREPARED_BUNDLE.md](PREPARED_BUNDLE.md) | 현재 적용 범위·미완료 조건 또는 시각화 결과 갱신 |
| [docs/REAL_DATA_NEXT_RUN.md](REAL_DATA_NEXT_RUN.md) | 현재 적용 범위·미완료 조건 또는 시각화 결과 갱신 |
| [docs/REAL_DATA_PREPARATION.md](REAL_DATA_PREPARATION.md) | 현재 적용 범위·미완료 조건 또는 시각화 결과 갱신 |
| [docs/REAL_GT2D_DIAGNOSTIC.md](REAL_GT2D_DIAGNOSTIC.md) | 실험 당시 범위·후속 상태 구분 |
| [docs/REAL_RGB102_DIAGNOSTIC.md](REAL_RGB102_DIAGNOSTIC.md) | 102페어 진단 범위가 일치해 내용 유지 |
| [docs/REAL_RGB2D_RELEASE.md](REAL_RGB2D_RELEASE.md) | 현재 적용 범위·미완료 조건 또는 시각화 결과 갱신 |
| [docs/REAL_RGB_DIAGNOSTIC.md](REAL_RGB_DIAGNOSTIC.md) | 실험 당시 범위·후속 상태 구분 |
| [docs/RGB102_PAIRS.md](RGB102_PAIRS.md) | 현재 적용 범위·미완료 조건 또는 시각화 결과 갱신 |
| [docs/RGB269_PAIRS.md](RGB269_PAIRS.md) | 현재 적용 범위·미완료 조건 또는 시각화 결과 갱신 |
| [docs/RGB_GT_SHARDS.md](RGB_GT_SHARDS.md) | 현재 적용 범위·미완료 조건 또는 시각화 결과 갱신 |
| [docs/RGB_INDEX_PROBE.md](RGB_INDEX_PROBE.md) | 종료된 인덱싱 기록으로 명시 |
| [docs/RGB_SHARD_TRAINING.md](RGB_SHARD_TRAINING.md) | 현재 적용 범위·미완료 조건 또는 시각화 결과 갱신 |
| [docs/ROADMAP.md](ROADMAP.md) | 현행 정책·실행 본문 수정 |
| [docs/SAMPLE_EXPANSION.md](SAMPLE_EXPANSION.md) | 현행 정책·실행 본문 수정 |
| [docs/SYNTHETIC_GPU_PILOT.md](SYNTHETIC_GPU_PILOT.md) | 합성 GPU 검증 범위가 일치해 내용 유지 |
| [docs/T1_FEATURE_CHUNKS.md](T1_FEATURE_CHUNKS.md) | 현재 적용 범위·미완료 조건 또는 시각화 결과 갱신 |
| [docs/VIDEO_INPUT_PLAN.md](VIDEO_INPUT_PLAN.md) | 현재 적용 범위·미완료 조건 또는 시각화 결과 갱신 |

## 아카이빙과 검증

종료된 실험 65개 항목, 2,587개 파일(3,450,380,309bytes)을 `.local-archive/20261105/indexing_retired/`로 이동했다. 이동 파일 전부와 기존 위치에 둔 보호 파일 3,948개의 크기·SHA256을 대조했다. 데이터 삭제·재압축·디스크 공간 회수는 없다. 실제 학습 입력·checkpoint·시연 자료는 보존했다. 복원 방법과 고정 경로를 쓰는 과거 스크립트의 제한은 [로컬 보관 안내](LOCAL_STORAGE.md)를 따른다.

Git에는 문서만 반영한다. 대용량 아카이브와 검증 목록은 기존 ignore 규칙 안에 둔다. 문서의 과거 로컬 산출물 경로는 보관 인덱스로 추적하며 다른 실행 결과로 대체하지 않는다. 로컬 검토는 UTF-8, diff 공백, 문서 링크 및 정책 정합성에 집중하고 모델/GPU 시험은 반복하지 않는다. 이번 문서 갱신으로 구현·실험 완료 상태를 올리지 않는다.

최종 링크 대조에서 이동된 로컬 산출물 참조 23곳을 확인했다. 17곳은 이번 보관 인덱스, 6곳(고유 파일5개)은 11-02 보관 ZIP에서 실재를 확인하고 원래 경로 표기와 복원 안내를 연결했다. 새 문서 링크의 대상 누락은 없다. 기존 CRLF를 유지한 문서는 `cr-at-eol`을 반영해 Git 공백 검사를 통과했다.
