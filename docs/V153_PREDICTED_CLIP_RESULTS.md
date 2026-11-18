# v1.5.3 예측2D 단일클립 충분학습 결과

2026-11-18 · `V153-PREDICTED-CLIP-001` · T4 학습 완료, checkpoint·저장 예측 독립 검토 완료.

## 조건과 목적

[v1.5.3 고정 계약](V153_PREDICTED_CLIP.md)에 따라 같은 첫 train50프레임(Standing_Up__1_)을 예측 F9·첫 stereo 중점 카메라·GT 상체 조건으로 반복 학습했다. 부모 original u0 가중치에서 새 optimizer/RNG로 시작했으며 부모 u1000의 이어 학습이 아니다. Frontend는 고정했다. Body/hand 각5,000update, 총10,000optimizer update, FP32, seed71, DDIM10이다. 소스 `156475f`, identity `d5acc7ac4c04493aef71a15e3b3495931e2297a1f84fe93ca07658e6739001bb`, T4·Torch2.14.1+cu130·Python3.13.16이다.

## 결과

같은 train50, 정렬 없는 device 좌표의 body MPJPE와 wrist-relative hand local MPJPE(mm)다.

| 모델별 update | Body mm | Hand local mm |
| --- | ---: | ---: |
| 0 |412.24|200.23|
| 200 |89.87|52.23|
| 1,000 |29.23|14.75|
| 5,000 |10.78|5.65|

| 최종 형태 지표 | 예측 | GT | 예측/GT |
| --- | ---: | ---: | ---: |
| Body 뼈 길이 mm |183.334|185.279|0.9895|
| Hand 뼈 길이 mm |23.613|23.713|0.9958|
| Body spread mm |372.376|375.367|0.9920|
| Hand spread mm |65.841|66.116|0.9958|

최종 frame0/25/49의 body 오차는13.43/12.51/12.04mm, hand는8.93/3.89/8.07mm다. Body20mm·hand10mm 및 bone/spread비0.8~1.2의 잠정 공간 기준과 세 프레임 기준을 모두 통과했다. GT 클립 평균 상수126.42/23.02mm보다 낮지만 그 기준선은 미래 정답을 사용한 진단 참고값이다.

Root가 최종3프레임 whole/local 그림을 확인했다. 몸과 손의 크기와 자세가 GT에 가까워졌으며 손 축소가 크게 완화됐다. 손 그림은 두 wrist 원점을 겹쳐 표시하므로 실제 양손 간 거리를 의미하지 않는다.

## 독립 검토

- ZIP860,459,420bytes·SHA/CRC와 receipt16파일 크기·SHA, 평가 artifact8개 pin, snapshot4개와 last pin을 대조했다.
- Checkpoint5개를 CPU `weights_only=True`로 읽고 텐서 유한값을 확인했다. Snapshot4개의 body/hand를 strict load했고 original u0 fingerprint 및 마지막 snapshot/last 모델 가중치 일치를 확인했다. `last.pt`에는 optimizer·scheduler·scaler·RNG와 학습 위치가 있다. 실제 재개 실행을 검증한 것은 아니다.
- paired/durable5,000, 총optimizer10,000이며 last의 평가·snapshot 목록이 보고서와 일치한다.
- 원본 GT와 저장 GT가 정확히 일치했다. 좌표·mask로 MPJPE와 bone/spread를 별도로 재계산했다. 보고값과 최대 차이는 각각 `1.832e-7`mm, `4.883e-6`mm 미만이다.
- v1.5.2와 초기 body 예측은 정확히 같고 hand local 최대 차이는0.00020218mm다. 초기 가중치와 기록된 noise hash는 일치하지만 전체 출력을 비트 단위 동일로 표시하지 않는다. 앞서 발견한 로컬/Colab 간 noise·입력 fingerprint 재생성 불일치는 여전히 미해결이다.
- GPU forward·중단/재개·새 VM 복원·원격 Drive readback은 root가 재실행하지 않았다. 보관은 회수 receipt의 verified 기록과 로컬 SHA 대조 근거다.

## 해석과 다음 판단

**예측2D 특징을 사용해도 이 단일클립에서는 몸과 손 형태를 학습할 수 있다.** 따라서 예측2D를 넣으면 항상 붕괴한다거나 현재 모델이 형태를 전혀 표현할 수 없다는 설명은 이 결과와 맞지 않는다. 다만 한 동작의 암기만으로 입력 특징을 유용하게 활용했다는 인과나 일반화를 입증하지 못한다. GT 상체에는 wrist도 들어 있어 완전한 예측 입력 경로의 성공은 아니다.

v1.5.2에서 DDIM10→50의 개선은 작았고, v1.5.3에서는 같은 클립 반복으로 큰 개선이 나타났다. 이는 다중클립의 최적화·표본당 반복량·조건 활용을 후속 후보로 우선 검토할 근거다. 부모 v1.5는20클립, v1.5.3은1클립으로 학습 분포와 노출량이 함께 달라 **학습량 부족만이 원인이라고 확정하지 않는다.** v1.4의 GT2D 결과와도 입력·초기화·카메라 등 조건이 달라 한 변수 인과 비교가 아니다.

후속 후보는 고정된 소수 다중클립에서 학습 노출량과 클립별 수렴을 기록하는 통제 실험이다. 예측 상체를 사용하는 최종 경로와 독립 평가도 남는다. 이는 제안이며 새 버전·실행·예산 승인이 아니다. v1.3 혼합학습은 보류 상태를 유지한다.

## 시간과 원본

Operation run wall1245.656초(약20분46초), prepare7.953초, package9.246초·copy9.898초다. CLI compute1242.746초는 테스트 세션 전달값이다. Run에는 학습·평가·저장이 함께 포함되므로 pair당 순수 학습속도나 전체 데이터 학습시간으로 외삽하지 않는다. 실제 중단 재개 없이 완료했다.

프로젝트 상대 경로 `outputs/train_clip_diagnostics/user_test/v153/run-b461326a0e67445d9af902d5e771920a/`에 `results.zip`, `receipt.json`, `root_review.json`, 최종/초기 그림을 보존한다. ZIP SHA-256 `022fa1449ebb074701f17f1551de8403f6932f09c30c06f91ac26d3a067383ca`, last SHA `9b250c904a4c477a58c49447071c16ed4106dfb81bfee76859ab3b8ecfe5ea88`. ZIP 내부 `experiment/run.json`, `predictions-*.json`, `models-*.pt`, `last.pt`가 원본 근거다. Git에는 결과 문서만 포함한다.

[종합 보고서](EXPERIMENT_RESULTS_REPORT.md) · [실험 계보](V1_EXPERIMENT_LINEAGE.md) · [v1.5.2 결과](V152_SAMPLING_RESULTS.md)
