# v1 teacher 후보 구현과 검증 범위

2026-11-10. [v1 계획](V1_REWIND_PLAN.md)의 구조 재현 구현 묶음이다. 실제 데이터 학습이나 논문 성능 재현 완료가 아니다. 이전 v0 모델과 체크포인트는 변경하지 않는다.

## 구현한 것

- `rewind_v1/frontend.py`: MBConv·HardSwish encoder와 Conv/BN/ReLU decoder 기반 stereo heatmap 후보. Stage 배치·stride·upsampling·학습 loss 등 논문에서 확정하지 못한 부분은 프로젝트 가정이다.
- `attention.py`, `teacher.py`: 식(4)의 회전된 분자와 회전하지 않은 분모를 사용하는 causal attention, body/hand teacher. ELU+1 feature map·FP32·분모 최소값 1e-6을 가정한다. 현 계산은 시간축 행렬을 만들며 선형 시간 attention 최적화 구현은 아니다.
- `data.py`: 검토 근거와 선택/팩/spec/frontend hash를 확인하는 50프레임 clip adapter. GT 2D 진단과 예측 2D를 구분하고, 손의 wrist-relative target·GT 상체 학습·예측 손목 결합을 지원한다. 시간·카메라·영상 정합·원본 motion 검토 근거가 없으면 실제 clip 변환을 차단한다.
- `training.py`: 한 update·DDIM 추론·동결 body 조건·optimizer/scheduler/scaler/RNG 저장 및 재개 인터페이스. 입력·소스·환경 identity를 고정한다. 실제 GPU 학습 진입 시 A100 이름과 VRAM을 보고하고 다른 GPU는 거부한다. CPU 개발 검증과 입력 준비는 별개다.
- 위치·중간 frame loss와 masked velocity/foot loss 함수. 기본 teacher 설정은 velocity 0이며 foot은 update 경로에 연결하지 않았다. World/contact 근거 없이 foot loss를 사용하지 않는다.

`configs/rewind_v1_teacher_candidate.json`은 hidden512·graph3·temporal3·4heads·과거20프레임·K1000·DDIM10 후보 설정이다. 논문의 2M update 실행 승인이 아니며, 학습 예산과 해상도는 실제 pilot 전에 고정한다. 논문과의 차이는 [구조 대조](V1_PAPER_ALIGNMENT.md)를 따른다.

## 검증

Root가 `test_rewind_v1.RewindV1Tests` 7개와 `test_temporal_prepare.TemporalPrepareTests` 3개를 합쳐 실행하여 10개 통과를 확인했다. 식(4) 직접 계산·mask·유한 gradient, frontend shape/gradient와 eval 프레임 독립성, 미래 변경 불변성·streaming 초기 구간과 gap/reset, CPU 저장/재개 후 동일 모델, 합성 raw clip 연결, 손목 상대 좌표, GPU gate를 포함한다.

A100/T4 비교는 mock 검사이며 실제 A100 실행이 아니다. Frontend의 BatchNorm은 학습 시 batch 통계를 사용한다. 프레임별 frontend 사전학습과 eval 고정 특징 추출을 구분하며, 시간축을 batch로 합친 end-to-end 학습의 causality를 검증한 것으로 해석하지 않는다. 본형 CPU shape 확인은 디벨로퍼 보고이고, 위 root 검증은 작은 합성 모델을 사용한다.

## 다음 구현과 실제 검증

1. 새 1,000페어 payload 준비와 시간·카메라·정합 검토 근거를 확보한다. 프레임 번호만으로 timestamp를 만들지 않는다.
2. Frontend 사전학습 runner·checkpoint 계약, teacher epoch 순회와 학습 위치 저장, validation 집계 및 결과 회수 CLI/notebook을 연결한다. 현재 한 update API를 전체 학습 runner로 부르지 않는다.
3. 실제 Drive worker 비교·팩 복원 비용과 A100 메모리/학습 처리량을 한 고정 시험으로 측정한다. Raw pack은 PNG/JSON 보관이며 해상도별 tensor 또는 예측 특징 캐시는 후속이다.
4. 검증된 teacher 후 student 증류를 연결한다. 현재 student·identity conditioning·IK는 구현 완료 범위가 아니다.

실제 실행은 준비한 고정 소스와 예산으로 테스트 세션에 인계한다. 아직 새 GPU 학습이나 대규모 Drive 읽기를 실행하지 않았다.
