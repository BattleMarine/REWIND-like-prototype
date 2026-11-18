# v1.5.2 저장 모델 복원·생성 진단 결과

2026-11-18 · `V152-SAMPLING-DIAGNOSTIC-001` · 실제 T4 실행 완료, 저장 결과 수치 검토 완료. 환경 간 noise·입력 fingerprint 재현은 미확인이다.

## 목적과 조건

[v1.5.2 실행 계약](V152_SAMPLING_DIAGNOSTIC.md)에 따라 v1.5 중점 모델 u0/u200/u1000을 재학습 없이 비교했다. 첫 train50프레임(Standing_Up__1_), 예측 F9·GT 상체 조건이며 독립 validation이나 배포 성능이 아니다. GT-noised 복원은 정답이 입력에 섞인 진단이고 pure-noise 생성과 구분한다. 소스 revision `156475f`, source identity `d5acc7ac4c04493aef71a15e3b3495931e2297a1f84fe93ca07658e6739001bb`. Tesla T4, FP32, Torch2.14.1+cu130, Python3.13.16, seed71이다.

## 결과

정렬 없는 device 좌표의 MPJPE(mm). Hand는 wrist-relative local이다.

| 모델·조건 | Body | Hand local |
| --- | ---: | ---: |
| u0 pure DDIM10 |412.24|200.23|
| u200 pure DDIM10 |500.83|64.34|
| u1000 pure DDIM1 |259.98|64.78|
| u1000 pure DDIM10 |236.16|67.22|
| u1000 pure DDIM50 |236.40|66.53|
| u1000 GT-noised k1 |133.85|54.76|
| u1000 GT-noised k100 |130.63|58.43|
| u1000 GT-noised k500 |169.70|64.01|
| u1000 GT-noised k1000 |258.39|64.73|

GT 클립 평균 상수의 body126.42/hand23.02mm는 미래 정답까지 사용한 참고값으로 실용 baseline이 아니다. u1000 DDIM10의 뼈 길이는 body155.25/GT185.28mm, hand7.88/GT23.71mm로 보고됐다. 손 축소는 남는다.

DDIM10→50에서 body는0.24mm 악화, hand는0.69mm 개선에 그쳐 이번 표본에서 횟수 증가만으로 실패를 해소하지 못했다. 약한 잡음의 GT 입력에서도 hand54.76mm가 남아 순수 생성의 시작 noise만으로 손 오류를 설명하기 어렵다. 다만 모델 학습량·조건 활용·구현 중 어느 것이 주원인인지는 확정할 수 없다. u0→u200에서 body 악화와 hand 개선이 함께 나타나 학습 진행을 단조 개선으로 해석하지 않는다.

## 독립 검토와 제한

- 결과 ZIP의 크기·SHA와 CRC, recovery receipt21파일의 크기·SHA, 9조건의 예측/그림18개 artifact pin을 대조했다. 이 산출물은 별도 content seal 대신 receipt/pin으로 검증한다.
- 원본 bundle 검증 경로로 부모 manifest·고정 모델을 읽고 u0/u200/u1000의 body/hand6개 fingerprint를 재계산했다. 실행 전후 기록과 모두 일치했다.
- 저장 좌표·mask로 body/hand MPJPE18개를 재계산했다. 보고값과 최대 차이는 `3.642e-7`mm 미만이다. 저장 GT와 원본 GT는 정확히 일치했다.
- 실행 전후 입력 fingerprint는 기록상 일치하고 모든 조건의 noise hash도 기록상 같다. 그러나 로컬 Windows Torch2.12.1+cpu에서 재생성한 noise와 입력 전체 fingerprint는 Colab 기록과 일치하지 않았다. 환경 차이는 후보 원인이며 아직 원인으로 입증하지 않았다. 원본 카메라 anchor는 일치한다. 중간 입력 텐서·noise 원본 배열이 결과에 없어 불일치 위치·수치 크기를 직접 대조하지 못했다. 환경 간 완전 재현 검증 완료로 표시하지 않는다.
- GPU forward·중간 head trace·원격 Drive readback은 root가 재실행하지 않았다. Drive 보관 성공은 회수 receipt의 `verified` 기록이며 로컬 ZIP을 그 SHA에 대조했다. 뼈 길이는 실행 보고값으로 구분한다.

내부 report 경과15.279초, CLI compute17.050초는 테스트 세션 전달값, operation run wall20.017초·prepare6.600초, package0.071초·copy0.084초다. 서로 측정 범위가 다르며 모델 단독 레이턴시가 아니다.

## 보관 위치와 후속

프로젝트 상대 경로 `outputs/train_clip_diagnostics/user_test/v152/run-0ef7db1cdffd48e495d84f1954b8c39a/`에 `results.zip`, `receipt.json`, root 독립 검토 `root_review.json`을 보존했다. ZIP6,489,604bytes, SHA-256 `43f3099886cf25debcfe442fd479d4f6d6be7523d6dab2f88f56164728e46eb1`. ZIP 내부 `experiment/run.json` 및 `experiment/predictions-*.json`이 원본 수치 근거다. 대용량 결과는 Git 제외다.

다음 판단 근거는 이미 준비된 v1.5.3 예측2D 단일클립 충분학습 결과다. 아직 결과가 없으므로 성공·실패를 예단하지 않는다. 환경 간 fingerprint 불일치를 별도 확인 대상으로 남기며, 이번 검토로 재실행·예산 변경·새 대규모 학습을 시작하지 않는다. [종합 보고서](EXPERIMENT_RESULTS_REPORT.md) · [판단 이력](V1_EXPERIMENT_LINEAGE.md).
