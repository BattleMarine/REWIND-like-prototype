# V1-DIAGNOSTIC-001: 명목 시간 기반 A100 진단 학습

2026-11-10 사용자 명시 승인에 따른 **별도 진단 시험**이다. 실제 timestamp와 원본 motion 독립성이 검증됐다고 주장하지 않는다. 원래 V1-ACTUAL-001의 blocked 기록과 엄격 검증 경로는 보존한다.

## 채택한 근거와 가정

- 기존 실제 감사는 train1,000·validation2,236의 파일·좌표 수치 검사를 완료했다. 사용자가 GT overlay89장 전체를 확인하고 문제없다고 보고했다. 이는 모델 예측 정확도 또는 측정된 시간 동기 증거가 아니다.
- 공식 UnrealEgo2 논문의 명목25FPS 설명을 근거로 frame index/25의 시간을 유도한다. `time_basis=derived_from_dataset_fps`, `recorded_timestamps_verified=false`를 기록한다. 실제 timestamp를 발견하거나 복원한 것이 아니다.
- 좌우 대응은 공통 frame key와 사용자 시각 검토에 기반한 가정이다. `stereo_sync_measured=false`로 유지한다.
- 원본 motion 독립성은 미확인이다. `original_motion_verified=false`, `independent_validation_verified=false`, `validation_reference_only=true`를 유지한다. 공식 validation을 학습에 섞지 않되 그 지표로 독립 일반화 성능을 주장하지 않는다.
- 자세한 기존 근거와 공식 출처는 [감사 결과 및 사용자 결정](V1_ACTUAL_001_RESULTS.md#2026-11-10-후속-사용자-결정-별도-진단-학습)에 있다.

## 고정 입력과 실행

선택 content SHA는 `48e3c749c7d7710feb7fd61601b7406264837e7a67ceeb4a48787359825cafad`다. Train은20sequence×50frame=1,000 고유페어, validation은기존2,236페어다. Teacher는그중29clip/1,450페어에서 평가하고 나머지 제외 키를 보고한다. Test/depth는 사용하지 않는다.

보존 cache content SHA는 `70241b27f66565593f0c8de0aeb6fe1f94c3672f932d7609bc882fd24a21d7b0`, 사용자 Drive 위치는 `REWIND-like prototype/reports/v1_actual_001/raw_cache/`다. 65개 part 합7,554,249,675bytes와 metadata2개를 고정 해시로 로컬 복원한다. 동일한 로컬 캐시가 있으면 재사용한다. 원본 RGB/pose 경로의 재수집·재감사는 실행하지 않는다.

기존 감사·카메라 후보는 원본 바이트를 보존한다. 당시 audit producer의 코드 hash와 현재 진단 실행 코드 hash를 별도로 고정한다. 사용자 결정과89장 overlay 검토 기록도 content hash로 묶는다. 진단 가정과 입력 identity는 report와 checkpoint에 남으며 엄격 실행의 체크포인트와 혼용하지 않는다.

실제 A100 장치명·VRAM을 먼저 확인한다. Frontend/body/hand 각1epoch, 총500+20+20=540회 순회 update다. 입력128px·frontend batch2·FP32, body/hand hidden512·graph3·temporal3·4heads·DDIM10, velocity loss0·foot loss미사용이다. 이것은 학습 연결·비용·예측 품질의 초기 진단이며 모델 완성 시험이 아니다.

실행 호출은2시간, 출력1GiB, 로컬 cache복원12GiB, 보존 특징128MiB, checkpoint256MiB 상한을 유지한다. Wrapper hard timeout과 회수 ZIP 공간도 별도로 확보한다. 준비·학습·평가 시간은 분리하며, 새로운 런타임으로 복원하는 비용을 GPU 모델의 처리 시간으로 계산하지 않는다.

## 테스트 세션 안내와 판정

고정 소스·입력·진단 근거 ZIP과 notebook은 Git 제외 `outputs/archvis_v1_diagnostic_001/`에 보존하며, 배포 receipt의 정확한 revision·파일 SHA로 안내한다. 사용자 변수로 허용된 경로·실행 플래그 외의 코드·계약·예산은 시험 중 변경하지 않는다. 기존 v0 및 V1-ACTUAL-001 실행 폴더를 덮어쓰지 않는다.

성공 기준은 각 stage 종료, 유한 loss/평가 지표, 실제 A100 실행 기록,540회 순회 위치와 no-supervision batch 집계, 읽을 수 있는 last/best와 해시 영수증이다. 낮은 정확도와 실행 실패를 구분한다. NaN/Inf·OOM·hash 불일치·예산 초과 시 회수 후 중단하고 상한 확대·자동 재시도는 하지 않는다.

결과에는 nominal 시간·참고 validation 표시, 전체 및 teacher 평가 coverage, 학습·평가 지표, 환경·메모리·준비/계산/평가 wall-clock, checkpoint 식별값을 포함한다. 사용자 실행 안내·중간 질문·최종 회수는 테스트 세션이 담당하고 root가 결과를 검토한다. 이 문서는 준비 범위이며 실제 A100 실행 완료를 뜻하지 않는다.

## 로컬 검증 범위

Root 독립 검토에서 진단 계약·기존 엄격 감사·학습 재개·temporal 입력 관련25개 테스트가 통과했다. 실제 고정 근거 묶음의3,236개 카메라 행과89장 사용자 검토 기록도 확인했다. 개발 실행에서는 기존 Windows checkpoint 교체 경로의 일시적 권한 오류가 관측됐으나 root 전체 재검증에서는 재현되지 않았다. 실제 Drive 캐시 복원과 A100 학습은 사용자 시험으로 남아 있다.
