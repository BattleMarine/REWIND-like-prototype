# 카메라 원점 민감도 진단 결과

2026-11-15 `V12-CAMERA-GAUGE-001` 검토. **이번 6클립에서 절대 world 원점에 대한 큰 모델 민감도를 확인했다. 도시의 위치 편향·몸 축소에 camera translation 조건이 크게 기여한다는 직접 개입 근거다.** 모든 실패 원인의 규명이나 최종 수정·일반화 완료를 뜻하지 않는다.

## 근거와 독립 검토

- Revision `a0775ecebd3162ba4b0dcf64fa34059f60c6c92d`, source identity `3c8e7432ffdfb31ce35a7abef37396ed7570c94bb6e37380fb2f6227b1485595`.
- Run `fb16c34aa724435ba448690f9b70c67f`, 최종 attempt `9qi8w_d1`.
- 결과 ZIP71,859,703bytes, SHA256 `f870ea95b26d84f6c40b366f701bb784913083474a5af359dbaea4c63ee89c7a`.
- Report content SHA `aa680bf78a0f12a6a4d73e6ef33bcefa9d72718b1c1eda58ed1f8ca1b6d9c876`.
- 보존: `outputs/v12_camera_gauge_20261115/user_test/completed_001/`, 원본 ZIP·67파일·receipt·`root_review.json`.

Root는 ZIP/member 크기·SHA, JSON seal, selection/prepared/report/clip/variant 연결, 고정 모델·구현 pin, 6클립 키와 GT, 변형 간 동일 features hash를 확인했다. 저장된 예측·GT에서 body/hand global/local MPJPE, body centroid·뼈 길이, paired 변화와 tolerance 초과수를 독립 재계산했다. 원본 RGB·pose를 다시 읽거나 GPU 추론을 다시 실행하지는 않았다. 2D는 이전 저장 예측과 동일성을 확인했으며 원본 GT2D로 새 평가한 것은 아니다. Drive `verified`는 테스트 세션 보고이며 root가 원격 receipt를 다시 조회하지 않았다.

원본 baseline은 이전 결과와 정확히 같았고, 같은 실행의 zero repeat도 모든 저장3D좌표에서 최대차이0mm였다. 따라서 이번 큰 차이를 실행 간 GPU·난수 변화로 설명하기 어렵다. 모든 클립에서 두 변경 조건 모두 반복 오차를 고려한 tolerance를 초과했다. [고정 설계·판정](V12_CAMERA_GAUGE.md).

## 결과

각 환경은 같은50frame 클립2개이며 아래 body 값은 같은 관절 수의100frame MPJPE(mm)다. 원점 정렬은 **첫frame stereo 중점만** 사용했고 GT오차를 보고 선택하지 않았다. 회전·RGB·2D·GT·모델·noise는 유지했다.

| 환경 | 원본 | 첫frame 중점 원점 | 고정 x축+100m |
| --- | ---: | ---: | ---: |
| ArchVis |197.34|180.50|2,173.84|
| CityPark |6,721.67|276.27|8,023.31|
| Downtown |2,277.83|306.05|2,283.11|

| 환경 | Body 뼈 길이 원본→중점 (mm) | Hand global 원본→중점 (mm) | Hand local 원본→중점 (mm) |
| --- | ---: | ---: | ---: |
| ArchVis |184.87→172.13|293.29→319.78|82.15→165.44|
| CityPark |14.67→166.46|7,524.48→579.47|3,145.50→302.63|
| Downtown |18.73→162.04|2,912.33→377.02|1,188.49→139.71|

Body GT 평균 뼈 길이는185.28mm다. ArchVis의+100m 조건에서는 예측18.01mm로 다시 축소된다. 도시의 body centroid 오차는 CityPark6,715.69→181.33mm, Downtown2,239.34→199.43mm로 줄었다. **Body 개선과 손 개선은 동일하지 않다.** ArchVis 손은 원점 정렬 후 악화했고, 도시에도 자세·손 오차와 뼈 길이 축소가 남는다.

Root는 CityPark Gunplay와 ArchVis Standing의 frame25 비교 PNG를 직접 확인했다. 테스트 세션은6GIF의0/24/49번을 확인했다고 보고했다. 모든frame을 root가 시각 검토한 것은 아니다. 그림은 모든 변형에 같은3D축을 사용하므로 큰 위치 오차가 축을 넓혀 정상 크기 몸도 작게 보일 수 있다. 형태 판정은 위 뼈 길이 수치와 함께 한다.

## 좌표와 수치 오차

저장된 camera R/t로 공통Δ, `t′=t−RΔ`, float32 인코딩과 불변 잔차를 독립 검토했다. Camera-space 좌표·상대stereo·center 이동의 float64 오차는 약1.2e-10mm 이하다. 인코딩 translation roundtrip 최대0.01215mm, float32 camera-space 약0.02394mm, 상대stereo 약0.03021mm다.

입력 양자화는 완전히0이 아니며 신경망이 이를 증폭할 가능성 자체를 배제하지 않는다. 다만 관찰된 수천mm 변화·뼈 길이 붕괴와 반대 방향 개입 결과는 미세한 반복 오차 수준이 아니다. 이 시험은 물리적으로 같은 장면의 좌표 재표현에 대한 **현재 구현 모델의 큰 비불변성**을 보여준다. 절대원점 조건이 도시 실패의 주요 기여 요인이라는 판단은 이전의 단순 환경 간 상관관찰보다 강해졌다. 첫frame 중점 조건이 실제 전체 train 분포와 같아졌다고 주장하지 않는다.

## 비용·한계·다음 판단

A100-SXM4-40GB, CUDA FP32, torch2.14.1+cu130. Reference1.85초, selection5.55초, model5.90초, prepare47.62초, infer176.39초. 보고서 내부 추론 구간174.97초에는 반복·prefix·지표·렌더가 포함된다. 순수 모델 레이턴시나 전체 wall-clock이 아니다. 원본 논리 읽기711,188,043bytes이며 실제 네트워크 전송량은 측정하지 않았다. 모든 단계 complete 및 결과 보관 verified 보고를 받았다.

6클립300frame·한 인물 경로의 validation 진단이다. 명목25FPS, stereo 동기·timestamp·원본 motion 독립성 미확인을 유지한다. 재학습하지 않았으며 다른 인물/환경 일반화나 최종 수정 효과의 증거는 아니다.

다음 우선 제안은 **학습과 추론에서 같은 원점 기준을 사용하는 camera 표현을 별도 실험으로 설계**하는 것이다. 첫frame 기준과 causal 재개·sequence reset 계약을 고정하고, 원본 표현 대조와 body/hand 지표를 모두 유지한다. 추론에서만 중점을 빼는 방식을 바로 운영 기본값으로 교체하지 않는다. Rotation 조건과 영상·2D 오류의 영향도 남아 있다. v1.3 혼합학습을 먼저 시작하기보다 이 표현 문제를 우선 다루되, 새 구현·학습은 별도 승인 전이며 기존 v1.5 제안을 이 기록으로 자동 대체하지 않는다.
