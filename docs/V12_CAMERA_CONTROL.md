# V12-CAMERA-CONTROL-001 실행 계약

2026-11-15 후속 상태: 고정 `fdab0f1` 실제 실행·회수와 root 독립 검토를 완료했다. 아래는 실행 전 고정 계약·개발 검증 기록이다. 최신 판정은 [결과 보고서](V12_CAMERA_CONTROL_RESULTS.md)를 따른다.

2026-11-15 사용자 승인에 따른 개발 준비입니다. 실제 Drive 파일 검사·Colab 추론·새 성능 결과는 아직 없습니다. 오케스트레이션 검토·로컬 커밋·고정 배포 후 테스트 세션에서 실행합니다. 새 학습, 도시 RGB 재추론, camera 정규화 변경과 oracle 실행은 포함하지 않습니다.

## 목적과 고정 입력

도시 평가의 큰 중심 이동과 형태 축소가 같은 추론 경로의 ArchVis validation에서도 관찰되는지 확인합니다. 카메라 조건과 오류의 관계는 관찰 비교이며 인과 증명이 아닙니다. RGB, 회전, motion, 2D 품질도 함께 달라집니다.

- ArchVisInterior_ArchVis_RT 공식 validation만 lexical availability 순서로 탐색합니다. 서로 다른 sequence 최대2개의 첫 공통 연속50frame을 예측 전에 seal합니다. 100pair가 목표이며1clip이면 `partial_availability`,0clip이면 실패입니다. Train/test fallback·오류 후 후보 교체는 없습니다.
- v1.2 best frontend9/body10/hand9 모델 ZIP은873,393,783B, SHA256 `48661e75e0969ca140ab9cf0e543d73d2bb3b9ec108d30652cf7fb4f4b76b506`입니다. 모델 로더의 checkpoint/dependency 검증을 유지합니다.
- 도시 reference는 [기존 결과](V12_CROSS_SCENARIO_RESULTS.md)의 `outputs/v12_cross_scenario_20261115/user_test/completed_001/results.zip`,47,144,765B, SHA256 `c302fede5d8f3f5d72548dedb627f57892e9d8420670381e4c49ebd049bc6910`입니다. 원래 revision은 `2de4262`, source identity는 `849b615f3129620aff585505d5ae435a5bf9650631ccb8511bdcd9a147c630f5`입니다.
- 기존 도시4clip200frame의 prediction/metrics를 재사용합니다. Reference ZIP 전체 pin, 모든 receipt member의 크기/SHA, JSON seal, source/model, selection→prepared→report→prediction의 순서/키 연결과 manifest를 검사합니다. 원본200pose만 선택 경로에서 읽고 기존 크기/SHA/frame/sequence·공식 validation 소속·geometry를 대조합니다. 합계17,179,658B이며 도시 RGB나 새 후보는 읽지 않습니다. 불일치시 중단합니다.
- 기존 `archvis_v1_inputs.zip` SHA256 `6256eb7bd148ec51bdb94627c68e54657be3808c607a12e13b68f7e479b15e97`을 재사용합니다. 설정은 [고정 config](../configs/v12_camera_control_001.json)를 따릅니다. Source ZIP/revision/notebook의 최종 pin은 root의 배포 receipt에서 확정합니다.

## 같은 추론 경로

기존 cross 함수에 명시적 `EvaluationContext`만 전달하며 생략시 기존 도시 계약이 유지됩니다. `predict`, `sample_features`, `metric_sums`, `aggregate`, `render`의 신경망·평가·시각화 로직을 재사용합니다. 전역 상수 monkeypatch는 없습니다.

CPU 기본값, 명시적 CUDA 선택, FP32/DDIM10/seed71/body offset100/hand offset200, full50/prefix25, clip reset을 유지합니다. CUDA는 지원되는 GPU를 허용하며 장치명·VRAM·시간을 기록하고 자동 CPU fallback은 없습니다. 모델 weight와 inference implementation pin을 이전 도시 report와 정확 대조합니다. Hand는 예측 body 상체와 wrist를 조건으로 받으며 GT는 geometry 감사·metrics·그림에만 사용합니다.

선정 pin은 `control_status.json`에 저장합니다. 다음 child가 이 기대 pin을 확인하고, 준비 뒤 prepared/cameras pin을 저장해 최종 추론·비교까지 연결합니다. 자체 seal만 같은 다른 자료를 대신 넣지 않습니다.

## 카메라 정의와 산출물

Frame마다 좌/우 순서로 전체18개 float32 조건값을 `cameras/cameras.json`에 저장합니다. 각view9개는 R의 첫 column xyz, 두 번째 column xyz, `t_xyz/100mm`입니다. 원래 `encode_camera(camera,100.)`을 그대로 사용합니다. ArchVis는 실제 `clip.pt` camera tensor와 정확 일치를 검사합니다. 도시는 고정 pose SHA와 동일 converter/encoder에서 복원한 값이며 과거 런타임 tensor를 직접 회수했다는 주장은 하지 않습니다.

`x_camera_mm = R @ x_canonical_world_mm + t_mm`에서 translation과 `-R.T @ t` camera center를 저장합니다. Center는 canonical world(mm)이며 native UE 좌표가 아닙니다. R 전체행렬, 각norm, 우-좌 translation/center/encoded 차이, baseline(center 차이 L2), 회전차이 Frobenius norm과 geometry residual도 저장합니다.

환경/clip별 각성분·norm의 min/max/mean/population std(ddof0)/p05/p50/p95를 frame축으로 계산합니다. Percentile은 NumPy linear interpolation입니다. `comparison/camera_distributions.png`는18성분과norm·baseline·회전차이를 공통 축의 환경별 min/p05/median/p95/max로 표시합니다. `comparison/comparison.md`에는 camera translation norm과 body 중심 오차, 중심제거 진단, MPJPE, bone/spread를 함께 표시합니다.

중심제거 진단은 frame마다 유효 body관절의 예측·GT 중심을 각각 제거한 joint-weighted L2 오차이며 sum/count를 저장합니다. 회전·scale 정렬을 하지 않고 공식 MPJPE를 대체하지 않습니다. GT 누락 mask는 유지합니다. ArchVis 평가 표본 min/max 밖 비율은 전체 train 분포나 OOD 증명이 아닙니다. 단일인물·이름 motion proxy·명목25FPS와 실제 timestamp/stereo/motion 독립성 미검증 한계를 유지합니다.

## 고정 실행과 보관

검토 후 배포 위치는 `outputs/v12_camera_control_20261115/release/`다. `handoff/v12_camera_control_001.ipynb`와 `handoff/handoff_receipt.json`에 source/input/reference ZIP 이름·SHA·revision을 고정한다. 실제 사용자 실행은 이 프로젝트 테스트 세션 `01a14391-c1b9-7a40-b537-af6b0b43709c`에서 안내한다.

빌더는 `python -m scripts.build_v12_camera_control --source-zip ... --source-sha256 ... --revision ... --input-zip ... --reference-zip ... --output ...`입니다. 기존 경로를 덮어쓰지 않습니다. 모든 notebook code cell compile과 기본 스위치 무실행을 로컬 검증합니다. 실제 배포 ZIP 생성·실행 인계는 root가 담당합니다.

Source/input/reference ZIP을 `/content`에 준비하고 bootstrap을 실행한 뒤 `RUN_CONTROL=False`를 명시적으로 켭니다. 사용자 실행 셀 하나에서 reference→선정→모델확인→준비/pose회수→추론/비교를 직렬 child로 실행합니다. 각child는 개별 timeout 뒤 kill/wait하며 자동 재실행은 없습니다. `SHOW_SAVED_RESULTS=False`는 이미 저장된 그림만 다시 표시합니다. 정상 종료 후 보관을 확인하고 비교표·분포 PNG·ArchVis2GIF를 표시합니다.

| 단계 | 상한 |
| --- | --- |
| Reference 검증 |600초, ZIP64MiB / unpack128MiB /100members; 실제ZIP size/SHA 고정 |
| Metadata 선정 |1800초 /100,000entries /32MiB |
| 모델 copy/hash |600초 /1GiB; 실제873,393,783B pin |
| ArchVis 준비+도시pose 회수 |3600초 / 원본read합2GiB |
| 추론+통계+그림 |3600초 |
| 결과 파일합 / ZIP |각256MiB |
| 각보관 packaging / copy+readback |각600초 /600초 |

외부child timeout에는 종료 처리 여유30초를 둡니다. Reference/model/metadata/prepare/infer 및 원본·local reread 비용은 구분합니다. 준비 시작시 free disk8GiB/Linux MemAvailable4GiB, 실행중 high-water RSS4GiB 관측 guard를 유지합니다. 이는 OS 메모리 hard limit이나 cgroup headroom 보장이 아닙니다.

Drive 마운트·쓰기/readback 가능성을 작업 전에 확인합니다. 선정 후, 준비 후, 최종 종료(실패 포함)에 누적 보고서를 각각 새 attempt로 자동 보관합니다. 정상시 최대3회이며 각 package600+copy/readback600초 비용이 별도 추가됩니다. Reference/model 도중 실패는 그 종료본을 회수하며 실패한 게시를 자동 재시도하지 않습니다. Raw·tensor·model·reference ZIP은 결과 ZIP에서 제외합니다. 보관 성공과 작업 성공을 구분하며 실패시 로컬 보고서·ZIP을 보존합니다. 브라우저 단절시 VM이 살아 있는 경우와 VM 자체 종료를 구분하고 VM 소실에는 finally를 보장하지 않습니다.

## 개발 검증과 남은 한계

집중 검증은 context 범위/validation·partial·zero, reference tamper/order/key/hash/model/source 거부, camera 단위·18성분·percentile·frame 정렬, 작은 positive fixture의 준비→camera회수→실제 tiny모델 추론→GIF/통계/표→package/가상Drive readback, 도시pose 변조 거부, 단계 pin 연결과 기존 cross 회귀입니다. 합성 fixture 결과는 실제 모델 정확도 근거가 아닙니다. 최종 통과 수·시간과 미실행 항목은 개발 인계 보고에 기록합니다.

Root는 새 비교6개·기존cross8개·메모리검사4개·모델시각화4개, 총22개를 독립 실행해87.364초에 통과했다. 중간 보관3회/실패회수/게시 재시도 금지와 단계 간 기대 pin도 검사했다. 실제 도시 reference ZIP을 새 로더로 검증했으며 알려진 수치의 평균·분위수와 합성 분포 PNG 표시를 별도로 확인했다. 기존 `predict`, `sample_features`, `metric_sums`, `aggregate`, `render`는 AST 비교로 동일함을 확인했다. 실제 ArchVis100frame·도시200pose·CUDA 실행 및 원격 Drive 보관은 아직이다.

이후 추가된 reference receipt의 단계 간 기대 SHA 검사도 root가 별도1개 테스트로0.668초에 통과했다. 따라서 로컬 검증은22개 통합 실행과 마지막 보완1개 집중 실행으로 구분한다.
