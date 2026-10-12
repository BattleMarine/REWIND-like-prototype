# REWIND-like prototype 프로젝트 로드맵

검토 초안 v0.1   |   2026년 10월 12일

REWIND의 핵심 구조를 공개 데이터에서 재현하고, 데이터와 diffusion 학습·추론 방식을 수정해 정확도·강건성·실시간성을 개선한다. 첫 목표는 비교 가능한 기준 모델과 재현 가능한 평가 체계를 확보하는 것이다.

### 프로젝트의 성공 기준

- 같은 데이터 분할·설정·checkpoint로 학습 재개와 평가를 재현할 수 있다.

- Body–hand cascade와 causal temporal modeling을 갖춘 teacher 및 one-step student를 확보한다.

- 기준 모델 대비 개선 효과를 동일 조건에서 검증하고, 정확도와 지연 시간의 교환 관계를 설명한다.

### 우선 범위

입력은 stereo egocentric images와 camera poses, 출력은 몸과 손의 3D keypoints로 둔다. Colab GPU에서 body-only 모델로 시작해 whole-body teacher와 one-step student로 확장한다. 논문 수치와의 절대적 일치보다 구조 재현과 내부 비교의 신뢰성을 우선한다.

Identity conditioning과 avatar용 inverse kinematics는 후속 선택 과제다. Floor-alignment, 외부 static camera 융합, 완전한 physics/contact·self-penetration solver는 초기 범위에서 제외한다.

| 단계 | 주요 산출물 | 다음 단계 진입 조건 |
| --- | --- | --- |
| M0 데이터 검증 | 데이터 규격과 고정 split | 좌표·관절·시퀀스 정합성 확인 |
| M1 입력과 비교 기준 | 2D front-end와 regression | 예측 2D로 학습·평가 완료 |
| M2 Body teacher | Causal body diffusion | 과적합·causality 검사 통과 |
| M3 Whole-body teacher | Body → hand cascade | 예측 상체로 손 평가 완료 |
| M4 Student | One-step 증류와 streaming | 정확도·latency 비교 완료 |
| M5 개선 실험 | 단일 변경별 ablation | 반복 실험으로 효과 확인 |
| M6 통합 | 최종 모델과 재현 가이드 | 새 실행 환경에서 평가 재현 |

## 1 데이터 준비와 입력 파이프라인

### M0 데이터 확보와 공통 규격

UnrealEgo의 손 annotation이 포함된 시퀀스로 시작하고 UnrealEgo2로 확장한다. 전체 다운로드 전에 sample archive를 확인한다. 공식 안내의 UnrealEgo2 저장 공간은 약 3TB이므로 RGB와 pose를 필요한 범위만 확보하고 depth는 초기 필수 입력에서 제외한다. [3]

| 데이터 단계 | 제안 규모 | 목적 |
| --- | --- | --- |
| Smoke test | 5~10개 시퀀스 | 로더·좌표 변환·재투영 확인 |
| Overfit test | 1~4개 클립 | 학습 경로와 loss 검증 |
| Pilot baseline | 수십~수백 개 시퀀스 | 일반화 추세와 학습 비용 측정 |
| 확장 실험 | 검증된 데이터부터 확대 | 규모·동작 다양성 효과 분석 |

공통 sample에는 좌우 영상 경로, timestamp, camera calibration/pose, 2D·3D joints, annotation mask, 관측 visibility, sequence·motion·identity ID를 저장한다. 정답 누락과 화면 밖 관절을 구분하고 관절 이름·순서·부모 관계를 별도 skeleton 규격으로 고정한다.

초기에는 device-relative 출력을 검토한다. 카메라 변환 방향, 회전 표현, 거리 단위, 어안 projection, 영상 resize 후 좌표를 검증한다. 속도 loss에서 camera motion이 섞이는 영향을 명시한다. World-space foot loss는 좌표와 접촉 정보가 검증되기 전에는 사용하지 않는다.

Train/validation/test는 sequence 단위로 고정하고 원본 motion의 중복 렌더링도 점검한다. 공식 split과 별도 generalization split을 혼동하지 않는다. UnrealEgo2 test split 공개 공지와 별개로 손 관절 포함 여부는 실제 파일로 확인한다. 최종 test를 튜닝에 사용하지 않는다. [3]

**완료 기준:** 좌우 재투영 시각화, 관절 연결, 누락 프레임·단위·분할 중복 검사와 manifest를 확보한다.

### M1 2D front end와 비교 기준

EfficientNet 계열 encoder와 heatmap decoder로 2D joints와 confidence를 예측한다. 해상도는 손 관절 정확도와 GPU 비용을 보고 선택한다. GT 2D 입력은 진단 전용으로 두고 실제 기준 성능에는 예측 2D를 사용한다.

Front-end를 학습·고정한 뒤 2D 특징을 캐시한다. 캐시에 모델과 전처리 버전을 기록하고, 단일 프레임 3D regression 및 temporal regression을 비교 기준으로 구현한다. GT 2D 사용 결과와 예측 2D 결과는 별도 표로 보고한다.

**완료 기준:** Body/hand별 2D 지표와 예측 2D 기반 3D 학습·평가를 끝까지 실행한다.

## 2 기준 모델 구현과 학습

### M2 Body diffusion teacher

Noisy 3D joints에 stereo 2D joints·confidence, camera encoding, diffusion timestep을 결합한다. Graph Transformer로 공간 구조를, RoPE 기반 causal window attention으로 시간 정보를 처리하고 clean 3D joints를 예측한다. Graph와 temporal 모듈은 denoiser 내부에 둔다.

| 항목 | Teacher | Student |
| --- | --- | --- |
| 특징 차원 | 512 | 256 |
| Graph Transformer | 3블록 · 4 heads | 1블록 · 2 heads |
| Temporal attention | 3층 · 4 heads | 1층 · 2 heads |
| Window 설정 | 20 | 8 |
| 추론 | DDIM 10-step | 1-step |

위 값은 논문 부록의 시작 설정이다. 공통 설정은 50프레임 학습, diffusion timestep 1,000, x0 prediction, Adam과 학습률 5×10⁻⁵이다. 논문의 200만 step은 참고 규모이며, 실제 학습 예산은 pilot에서 측정한 비용과 validation 추세로 정한다. [1]

위치 복원 loss로 overfit을 확인한 뒤 velocity와 intermediate frame loss를 추가한다. Foot loss는 조건이 충족될 때만 사용한다. Loss 단위·reduction·mask를 고정하고 논문 가중치를 맹목적으로 이식하지 않는다. 작은 개발용 모델과 논문 크기 모델은 설정과 결과를 구분한다.

**완료 기준:** 소량 과적합, 미래 프레임 변경에 대한 과거 출력 불변성, 긴 시퀀스 추론을 검증한다. Diffusion과 temporal modeling의 효과를 regression 비교로 분리한다.

### M3 Whole body teacher

Body teacher를 고정하고 Hand teacher를 학습한다. Hand 모델에는 상체 3D conditioning을 추가한다. 원본처럼 학습 시 GT 상체, 평가 시 예측 상체를 사용한다. 손의 출력 좌표와 손목 결합 규칙을 명시한다.

상체 조건 없음, GT 상체 조건, 예측 상체 조건을 비교해 cascade의 효과와 오류 전파를 측정한다. 손목 불연속·좌우 뒤바뀜을 검사하고 body 및 hand의 지표를 분리한다.

**완료 기준:** 실제 Body 예측으로 whole-body 평가를 완료하고 teacher checkpoint와 기준 설정을 고정한다.

### M4 One step student

Teacher를 고정하고 작은 student를 학습한다. Supervised loss에 논문 식 5의 증류 loss를 추가한다. k_small, noise schedule의 정확한 구현, teacher target의 stop-gradient 여부 등 불명확한 부분은 가정과 근거를 기록한다. 동일 크기의 증류 없는 student를 반드시 비교한다.

**완료 기준:** Teacher·비증류 student·증류 student의 정확도와 end-to-end latency를 비교한다.

## 3 평가와 실시간 검증

### 공통 평가 프로토콜

모든 실험은 고정 split과 같은 평가 코드를 사용한다. 관절 집합, 좌표계, PA alignment 적용 단위, 손목 상대 지표 여부, 유효 관절 mask, 프레임/시퀀스 평균 방식을 명시한다. 논문과 split·annotation·하드웨어가 다르면 동일 조건의 재현 수치로 주장하지 않는다.

| 평가 축 | 지표 | 해석과 조건 |
| --- | --- | --- |
| 정확도 | Body/Hand MPJPE<br>PA-MPJPE | mm 단위 보고 · 정렬 규칙 고정 |
| 구조 | Bone length error<br>손목 연결 오차 | 동일한 skeleton과 뼈 목록 |
| 시간 변화 | Velocity/acceleration error | Timestamp·좌표계 반영 |
| 떨림과 지연 | 정지 구간 jitter<br>빠른 동작 추종 지연 | 과도한 smoothing을 개선으로 오인하지 않음 |
| 강건성 | Failure subset별 오차 | 가림·화면 밖·spike·빠른 동작 |
| 복구 | Recovery time<br>복구 실패율 | Validation에서 문턱과 지속 조건 고정 |
| 실시간성 | p50/p95 ms/frame<br>처리율 · peak VRAM | Batch 1 · warm-up · GPU 동기화 |
| 발 안정성 | Foot skating | 유효한 world 좌표와 접촉 정보가 있을 때만 |

### Failure benchmark

자연 발생 failure와 인위적으로 주입한 corruption을 별도로 평가한다. Corruption은 한쪽 관측 누락, 짧고 긴 연속 가림, 좌표 spike, 잘못된 고신뢰 관측을 포함한다. 강도와 지속 시간은 validation에서 정하고, test에는 고정한 corruption seed를 적용한다.

GT visibility가 없으면 confidence 기반 subset을 실제 occlusion 정답으로 간주하지 않는다. Recovery time은 관측 복귀 후 오차가 고정 문턱 이하로 연속 유지되는 첫 시점까지로 정의하고, 미복구 사례도 집계한다.

### Streaming과 속도 측정

과거 window와 현재 프레임만 사용해 한 프레임씩 출력한다. 동일 noise에서 offline causal 실행과 streaming 실행의 일치, 초기 window, sequence reset, 장시간 실행의 메모리 안정성을 검사한다. Multi-step별 상태가 다른 attention cache를 무조건 재사용하지 않는다.

3D 모델만의 시간과 영상 전처리·front-end·body–hand 순차 추론을 포함한 시간을 분리한다. 지정 GPU에서 30 FPS를 잠정 목표로 삼되 p95 지연도 함께 판단한다. 렌더링 포함 여부와 hardware·precision·해상도를 기록한다.

**완료 기준:** 최종 후보는 제안상 학습 seed 3개로 반복하고 sampling seed 변동을 별도로 보고한다. 시퀀스 단위 분포와 clean 성능도 함께 확인한다.

## 4 성능 개선 실험 순서

기준 모델이 안정된 뒤 한 번에 한 요소를 변경한다. 먼저 비용이 낮고 원인 해석이 쉬운 실험을 수행하고, 효과가 확인된 변경만 조합한다. 데이터·모델·loss를 동시에 바꾼 결과는 단일 기법의 효과로 해석하지 않는다.

| 순서 | 실험과 변경 변수 | 판단할 결과 |
| --- | --- | --- |
| E1 | 학습 데이터 규모와 동작 다양성 | 동일 update 예산에서 일반화 개선 |
| E2 | 관측 오류 augmentation | 오류 subset 개선과 clean 성능 유지 |
| E3 | GT/예측 상체 conditioning 혼합 | 실제 cascade의 Hand 오차 감소 |
| E4 | x0 / ε / diffusion v prediction | 같은 모델·데이터·학습 예산의 정확도 |
| E5 | 지원하는 모델의 1/2/4/10-step 추론 | 정확도와 latency의 Pareto 개선 |
| E6 | Reliability-aware conditioning | False tracking 복구와 동작 detail 보존 |
| E7 | 대체 증류 또는 학습 방식 | 기존 증류 대비 비용과 품질 |
| E8 | Reliability 기반 adaptive inference | 평균·p95 지연과 최악 조건 안정성 |

### 먼저 실행할 실험

E1–E3을 우선한다. 데이터 규모 실험은 동일 학습 update 예산으로 비교하고 필요하면 수렴까지의 추가 비용을 별도로 보고한다. Augmentation은 프레임 독립 노이즈뿐 아니라 시간적으로 연속된 실패를 포함한다. 영상 변형과 keypoint 오류 주입은 구분한다.

상체 조건 혼합에는 일반화된 Body 오류가 필요하다. 학습 데이터에서 지나치게 정확한 Body 예측만 사용하는 것을 피하고, 필요하면 train 내부 분할의 out-of-fold 예측을 생성한다. 합성 noise 대체는 별도 ablation으로 남긴다.

### Diffusion 확장 시 주의할 경계

One-step 전용 student를 반복 호출하는 것을 유효한 few-step sampler로 간주하지 않는다. 해당 timestep·sampling 경로에 맞는 학습이 필요하다. Diffusion v prediction은 물리적 관절 velocity loss와 다른 항목이다.

Consistency, distribution matching, flow-based 방법은 서로 다른 학습 가정과 비용을 가진 후속 후보로 둔다. 동시에 구현하지 않고 한 방법의 목적함수·teacher 요구·예산을 검토한 뒤 별도 실험으로 진행한다. Adaptive inference는 reliability와 오류의 관계를 검증한 이후 착수한다.

### 채택과 통합

Validation에서 clean 정확도, failure 성능, latency를 함께 비교한다. 한 축의 개선을 위해 다른 축이 얼마나 나빠졌는지 기록하고 허용 기준은 본 실험 전에 결정한다. 유망한 변경 2~3개만 결합해 상호작용을 검사한다. 개선되지 않은 실험도 설정과 결과를 보존한다.

**완료 기준:** 최종적으로 정확도 우선 모델과 실시간 우선 모델을 구분해 선정하고, test 결과는 선택이 끝난 후보에 대해 보고한다.

## 5 실행 운영과 검토할 결정 사항

### 개발과 학습 운영

Colab notebook은 환경 준비·학습·평가의 진입점으로 두고, 핵심 코드는 data, models, diffusion, training, evaluation 모듈로 분리한다. 각 실험에는 config, 코드 revision, 데이터 manifest hash, front-end 버전, seed, checkpoint, 지표와 실패 예시를 함께 저장한다.

데이터는 가능한 한 런타임 로컬 저장소에서 읽고 Drive에는 checkpoint와 결과를 주기적으로 동기화한다. Model·optimizer·scheduler·AMP scaler·RNG·학습 위치를 저장하고 실제 중단/재개 테스트를 한다. 저장 주기는 pilot의 step 시간과 허용 손실량으로 정한다.

M0–M4는 순차적인 필수 경로다. M5는 GPU 예산에 맞춰 E1–E3부터 선택하고 M6에서 통합한다. 일정은 고정 날짜 대신 단계별 완료 조건으로 관리한다. Pilot 후 step당 시간 × 예정 update 수 × 실험 횟수로 GPU 시간을 추산하고 평가·중단 여유를 별도로 잡는다.

### 주요 위험과 대응

| 위험 | 대응 |
| --- | --- |
| 손 annotation 또는 camera 정보 부족 | 샘플 감사로 먼저 확인하고 데이터 범위·주장을 조정 |
| 저장 공간과 Colab 세션 제한 | 부분 다운로드·2D 캐시·resume 검증 |
| 좌표 오류가 모델 성능으로 보임 | 재투영·단위·skeleton 검사부터 통과 |
| 정확도와 속도 개선이 충돌 | 지정 GPU의 Pareto 결과로 모델 선택 |
| 공개 정보로 세부 구현이 확정되지 않음 | 가정·원본과의 차이·ablation 결과를 기록 |

### 이번 검토에서 조정할 항목

- [ ] **D1** 최소 완료 범위: M4까지를 필수로 하고 개선 실험은 E1–E3부터 진행할지.
- [ ] **D2** 실행 자원: 실제 GPU 종류, 1회 세션 길이, 저장 공간과 총 학습 예산.
- [ ] **D3** 성능 우선순위: 정확도, 실시간성, failure recovery 중 가장 중요한 항목.
- [ ] **D4** 결과물 범위: 모델·평가 코드까지 또는 실시간 시각화 demo까지.
- [ ] **D5** 후속 범위: identity conditioning과 다른 데이터셋 확장을 언제 포함할지.

### 근거 자료

[1] Lee et al. *REWIND: Real-Time Egocentric Whole-Body Motion Diffusion with Exemplar-Based Identity Conditioning*. CVPR 2025. 첨부 논문 본문과 부록. [공식 프로젝트](https://jyunlee.github.io/projects/rewind/)

[2] 첨부 프로젝트 요약서. 프로젝트 목표, Colab 환경, 우선순위와 제외 범위.

[3] [UnrealEgo Benchmark 공식 데이터 안내](https://unrealego.mpi-inf.mpg.de/). 확인일 2026년 10월 12일. 공개 안내와 실제 archive의 annotation은 구현 착수 시 대조한다.
