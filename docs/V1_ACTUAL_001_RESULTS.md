# V1-ACTUAL-001 입력 감사 결과

2026-11-10. 고정 revision `74ea55556ddddb44772a63ce48717573d835040a`의 사용자 Colab CPU 감사 결과다. **선택 자료의 수치 감사는 완료했고 실제 학습은 실행하지 않았다.** `status=blocked`, `train_ready=false`, `decision=external_review_required`를 유지한다.

## 회수 근거와 독립 검토

- 파일: Git 제외 `outputs/archvis_v1_handoff_20261110/user_test/archvis_v1_audit_recovery.zip`.
- 41,326,040 bytes, SHA256 `836e75cceb43819bed5d3400455a0208c55e7b09d2f3ac5886d3ae535b60efa3`.
- Root가 ZIP CRC와 receipt 96개 파일의 크기·SHA를 독립 확인했다. Audit·camera candidates·cache manifest 내부 content hash도 재계산해 확인했다.
- 준비 subprocess는 `exited`, returncode0. 사용자 보고 전체 벽시계 약40분이며, 감사 코드 계측은85.023초다. 전자가 원본 준비를 포함하므로 두 시간을 혼동하지 않는다.

## 확인한 데이터

| 항목 | 결과 |
| --- | --- |
| 선택 payload 감사 | Train1,000 + validation2,236페어 |
| 원본 stat 누락 | 0페어 |
| 프레임 번호 연속 구간 | Train20clip/1,000페어, validation29clip/1,450페어 |
| 카메라 변환 후보 | 3,236행, 시간 값은 모두 null |
| 시간 후보 필드 | 조사한 pose root·양쪽 camera·device에서0개 |
| GT overlay | 89PNG; 모델 예측이 아닌 수동 검토 자료 |
| Test·depth | 사용하지 않음 |

숫자 geometry 검사는 `numeric_geometry_passed=true`다. World→camera 최대 성분 잔차는1.3500311979441904e-13 raw, 역변환2.8421709430404007e-13 raw, camera/device 관계8.526512829121202e-14 raw, 재투영8.169600370379367e-05px다. 각 잔차 항목은563,064개 관측을 집계했다. 이는 저장된 좌표·보정값 사이의 수치 관계이며 **모델 정확도나 RGB에 대한 의미적 정합을 증명하지 않는다.** 채택된 단위·rig 가정의 범위도 유지한다.

## 캐시 보존

Raw pack은65개, 합7,554,249,675bytes이며 cache content SHA는 `70241b27f66565593f0c8de0aeb6fe1f94c3672f932d7609bc882fd24a21d7b0`다. 회수 ZIP에는 manifest·진행 기록만 있고 raw part 본문은 없다. 따라서 root의 manifest 검토를65개 raw pack의 로컬 독립 재검증으로 보고하지 않는다.

원본 읽기 예약 추정은7,550,764,961bytes, 감사 로컬 읽기 예약량은15,105,014,701bytes다. 후자는 pack hash와 member 재읽기를 포함하며 Drive 실제 네트워크 전송량이 아니다.

2026-11-10 후속 보관 보고: 사용자가 Drive의 `REWIND-like prototype/reports/v1_actual_001/raw_cache/`에 보관했다. 테스트 세션은 복사 후 읽기 검증 출력에서 `part-0000.zip`~`part-0064.zip` 65개와 `cache_manifest.json`, `progress.json` 검증 완료를 확인했고, 별도 Drive 목록에서도67개 파일과 part 합계7,554,249,675bytes를 확인했다고 보고했다. Cache content SHA는 위 고정값과 같다. 이 보관 확인의 근거는 테스트 세션 보고이며, root가 Drive의7.55GB 본문을 다시 읽어 독립 검증한 것은 아니다.

후속 실행은 이 폴더를 고정 cache SHA로 지정해 Colab 로컬 디스크로 복원한다. 복원 시 manifest·part hash를 다시 확인하고 동일한 로컬 보존본이 있으면 재사용한다. 원본 이미지·pose에서 pack을 다시 만들 필요는 없지만, 새 런타임의 묶음 전송·로컬 검증 비용은 발생한다. 시간·영상·원본 motion 검토와 학습 gate는 캐시 보존과 별개로 유지한다.

## 남은 근거와 다음 판단

1. **시간·stereo 동기:** 실제 timestamp가 없으며 명목 FPS는 적용하지 않았다. 동일 자료를 반복 감사해 timestamp가 생기지는 않는다. 공식 export 시간 규칙의 근거를 확보하거나, 명목 시간 기반 진단을 별도 가정으로 채택할지 판단해야 한다. 현재 recorded-timestamp 학습 gate는 바꾸지 않았다.
2. **영상/pose·좌우 정합:** 89개 GT overlay의 시각 검토가 남아 있다. Root가 이번에 확인한 것은 파일 무결성과 보고서이며 전체 overlay를 검토해 승인한 것이 아니다.
3. **원본 motion 독립성:** Train20·validation30 sequence 모두 `declared_unverified`. Boxing/Cross Punch 이름 기반 후보2그룹이 있으나 동일 motion이라는 결론은 아니다. 정확히 같은 stereo+pose 바이트 묶음의 cross-split 중복0개도 원본 motion 독립성을 증명하지 않는다.
4. **A100 학습:** 승인 묶음이 없어 미실행이다. 보존된 자료의 근거 검토 후, 검증된 실제 학습 또는 한계를 명시한 별도 진단 시험을 결정한다. 이번 결과를 이유로 1,000페어 규모·split·시간 가정·검증 gate를 자동 변경하지 않는다.

이 결과는 [실제 시험 인계](V1_ACTUAL_TEST_HANDOFF.md)의 감사 단계에 해당한다. [전체 실행 연결](V1_TRAINING_RUNNER.md)의 A100 성능·속도·정확도는 아직 확인하지 않았다.
