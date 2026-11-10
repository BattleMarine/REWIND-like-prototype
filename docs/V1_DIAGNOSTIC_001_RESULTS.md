# V1-DIAGNOSTIC-001 실제 A100 결과

2026-11-10. 실행 revision `04d60de4af5d253278a03bfe1001e49c7f12c846`. [시험 계약](V1_DIAGNOSTIC_001.md)의 명목25FPS 가정을 적용한 실제 데이터 진단이다. 실제 timestamp·좌우 시간 동기·원본 motion 독립성은 여전히 미검증이며 validation은 참고 지표다.

## 결과와 검토

A100-SXM4-40GB, PyTorch2.14.1+cu130, Python3.13.16, FP32, CPU threads1. Frontend/body/hand 각각1epoch를 완료했다. Cursor는500/520/540, frontend no-supervision batch는0, child exit0, status completed, failure null이다. Peak GPU allocated785,986,048bytes, reserved874,512,384bytes다. 낮은 할당량만으로 GPU 이용률을 단정하지 않는다.

Train1,000페어와 frontend validation2,236을 사용했다. Teacher validation은29clip/1,450frame이고786개는 연속 clip 평가에서 제외됐다. 손 평가는 고정 body의 예측 상체를 사용했다.

| 참고 validation 지표 | 결과 |
| --- | ---: |
| Frontend 화면 안 관절 오차, 원본 이미지 좌표 | 307.733px |
| Body MPJPE | 543.357mm |
| Hand global MPJPE | 551.690mm |
| Hand local MPJPE | 187.241mm |

학습·평가·저장 연결은 통과했지만 정확도는 낮다. 1epoch만으로 모델 수렴이나 논문 성능 재현을 주장하지 않는다. v0의20epoch 결과와 구조·학습량·train subset이 달라 단순 우열 비교도 불가하다. Global 손 오차와 local 손 오차의 차이에는 상체 위치 오차가 영향을 줄 수 있으나 단일 결과로 원인을 확정하지 않는다.

Root는 ZIP2개의 CRC, receipt7개 항목의 크기/SHA, 고정 실행 소스·config·decision pin을 독립 확인했다. Checkpoint4개를 CPU에서 `weights_only=True`로 읽고 모든 부동소수 tensor가 유한함을 확인했다. Last의540update와 진단 입력 표지도 확인했다. 저장 모델의 실제 추론 재실행·시각 품질·실제 GPU 재개 동등성은 이번 검토에서 실행하지 않았다.

## 시간과 pair당 비용

| 측정 범위 | 시간 | 해석 |
| --- | ---: | --- |
| Invocation 전체 | 850.846초 | 약14분11초, 복원·학습·평가·저장 포함 |
| 7.55GB 캐시 복원 | 52.032초 | Drive→로컬, 65parts |
| Frontend 학습 step | 165.749초 | 1,000 train페어 기준0.16575초/페어 |
| Frontend validation | 282.965초 | 2,236페어 기준0.12655초/페어 |
| Body 학습 step / 평가 | 1.295 / 3.379초 | 특징 준비를 포함한 전체3D pipeline 시간이 아님 |
| Hand 학습 step / 평가 | 0.969 / 7.027초 | 고정 body·입력 준비 비용과 측정 범위 구분 |
| CPU 전처리 누적 계측 | 631.866초 | 위 step/평가 시간과 중첩 가능 |
| 로컬 캐시 읽기 누적 | 51.229초 | Drive 복원과 구분 |
| Checkpoint 저장 | 10.819초 | 학습 연산과 구분 |

전체850.846초를 고유 train1,000으로 나누면 **0.85085초/페어**지만 validation과3단계·복원까지 포함한 이번 작업의 비용 배분이다. 순수 학습 처리율이나 데이터 전체 선형 예측값이 아니다. Frontend 학습 compute 계측만은42.996초로0.0430초/페어이며, 나머지 준비·평가를 제외한다. 이 계측은 wall-clock이고 GPU kernel 전용 시간으로 부르지 않는다.

전처리 누적시간과 작은 batch2가 병목 후보다. 다음 개선 후보는 PNG decode/pose 변환 결과의 재사용과 CPU 입력 준비 병렬화다. 같은 데이터·설정에서 전후 시간을 비교해야 개선을 입증할 수 있으며, GPU batch/precision 변경 효과와 섞지 않는다. 이번 시험만으로 전체 ArchVis 시간이나 최적 worker 수를 확정하지 않는다.

## 보존 근거와 다음 판단

원본 ZIP은 Git 제외 `outputs/archvis_v1_diagnostic_001/user_test/`에 보관한다.

- `v1_diagnostic_001_status-01.zip`: 2,998,587bytes, SHA256 `3dbdf22373872ba5a38e7dcac3e4326fd43ed976d735838d324eb6126b28ae13`.
- `v1_diagnostic_001_training-01.zip`: 311,613,615bytes, SHA256 `8bd6b99d1119f5f0255ba63143d931c69fde40bd7be93f9b5171fd3eb2d7257d`.
- Root 확인: `root_result_review.json`. 모델별 hash는 ZIP의 `recovery_receipt.json`과 `run.json`에 보존한다.

이번 시험의 반복 실행은 필요하지 않다. 다음 개발 후보는 저장 모델의 고정 clip 예측 시각화와 입력 준비 최적화다. 추가 epoch·데이터 확대·새 사용자 시험은 별도 범위와 고정 실행본을 준비한 뒤 진행한다. 엄격 검증 경로의 미해결 근거와 M2/M3 전체 완료 조건은 유지한다.
