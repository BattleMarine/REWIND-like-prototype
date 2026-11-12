# v1.1–v1.4 비교 평가와 후속 테스트 방향

작성일: 2026-11-12. 범위는 회수된 v1.1 T4 재실행(`70accf2`)과 v1.4 A100 재시험(`c05f548`)이다. 모델을 회수하지 못한 이전 v1.1 실행의 수치와 섞지 않는다. 이번 작업은 기존 결과·코드 검토와 실험 설계이며 새 학습·추론은 실행하지 않았다.

## 1. 핵심 판단

**v1.1은 파라미터 학습 자체가 멈춘 사례가 아니라, 예측 입력을 사용하는 다중 클립의 3D 복원·참고 validation에서 실패한 사례다. v1.4는 정답 입력을 준 한 클립을 충분히 반복하면 같은 계열의 teacher가 몸과 손의 형태를 복원할 수 있음을 보여준다.**

현재 가장 유력한 설명은 **teacher의 적은 업데이트와 클립별 반복 노출, 부정확한 2D 조건, hand의 학습·평가 상체 조건 차이가 겹쳤다**는 것이다. 각 요인의 인과적 기여는 아직 분리되지 않았다. “2D만 고치면 해결”, “epoch만 늘리면 해결”, “구현 오류가 전혀 없다” 중 어느 것도 현재 결과만으로 확정하지 않는다.

## 2. 실험 조건 비교

| 항목 | v1.1 T4 재실행 | v1.4 oracle 진단 | 해석 |
| --- | --- | --- | --- |
| 목적 | 1,000 train에서 epoch 확대 효과 확인 | 정답 조건으로 한 클립의 형태 복원 가능성 확인 | 성능 경쟁이 아닌 서로 다른 질문 |
| 고유 train | 1,000페어, 50frame × 20clip | 50페어, 1clip | v1.4는 훨씬 쉬운 암기 문제 |
| 평가 | Frontend 2,236frame, teacher 29clip/1,450frame의 참고 validation | 학습에 사용한 동일50frame | 일반화와 암기 점수를 직접 비교할 수 없음 |
| Body 입력 | 고정 frontend가 예측한 stereo 2D 특징 + 카메라 | GT stereo 2D 특징 + 카메라 | 좌표 외 confidence/presence도 달라짐 |
| Hand 학습 입력 | 예측2D + GT상체 | GT2D + GT상체 | v1.1도 hand 학습 때는 GT상체를 사용 |
| Hand 평가 입력 | 예측2D + 고정 body의 예측상체 | GT2D + GT상체 | v1.1에는 train/evaluation 조건 차이가 있음 |
| 손의 global 위치 | 예측 wrist에 부착 | GT wrist 부착과 예측 wrist 부착을 별도 기록 | local/global 지표를 구별해야 함 |
| Frontend 학습 | 10epoch, 5,000update, LR 1e-4 | 없음: GT2D 사용 | v1.4는 2D 모델 개선 시험이 아님 |
| Teacher 학습 | Body/hand 각각10epoch, 각각200update | Body/hand 각각5,000update | 모델당 업데이트25배 |
| 각 train clip 노출 | 각10회 | 동일clip5,000회 | 해당clip 반복 노출500배 |
| Teacher 설정 | hidden512, heads4, graph/temporal 각3층, LR5e-5, scale100mm | 동일 | 설정 JSON의 teacher 항목 일치 확인 |
| Diffusion/평가 | 학습1,000 diffusion step, 평가DDIM10 | 동일; 추가k1 GT-noised 진단 별도 | k1 결과를 실제 생성 성능과 섞지 않음 |
| Loss | position + frame, velocity/foot 미사용 | 동일 | 새 손실 때문에 성공한 실험이 아님 |
| 초기화 | Frontend71, body72, hand73 | Body71, hand72 | 공통 seed 항목만 같고 모델별 seed는 다름 |
| GPU/precision | T4, FP32 | A100, FP32 | 시간 비교와 동일 초기화 비교 모두 제한 |
| 기록 시간 | 5,844.470초(97분24초) | 593.880초(9분54초) | 작업 범위가 달라 GPU 속도비로 환산 불가 |

v1.1은 20clip × 10epoch = 200 teacher update다. v1.4의 5,000update를 v1.1의 “10epoch”와 같은 학습량처럼 읽으면 안 된다. Teacher 한 update는 50frame 클립 하나에 대한 갱신이며 frontend의 2페어 batch와도 다른 단위다. 두 실험 모두 명목25FPS·frame-key stereo 가정을 사용하고 실제 동기와 원본 motion 독립성은 미확인이다.

## 3. 결과 비교와 비교 가능한 범위

### 3.1 수치

| 관찰 | v1.1 | v1.4 |
| --- | ---: | ---: |
| 2D 화면 안 평균 오차 | Best214.84px, native1024px 기준 | 측정 대상 아님: GT 좌표 입력 |
| Body MPJPE | Best484.79mm(e9), 마지막549.25mm | 13.71mm(5,000update, 같은train) |
| Hand local MPJPE | Global best인e8에서124.67mm | 2.90mm(같은train, GT상체) |
| Hand global MPJPE | Best514.85mm(e8), 마지막565.97mm | 예측wrist 부착11.27mm, GTwrist 부착2.90mm; 둘 다 GT상체 조건 |
| Body bone / GT | 13.274 / 208.353mm ≈ 0.064 | 비율0.9975 |
| Body spread / GT | 39.303 / 516.717mm ≈ 0.076 | 비율0.9960 |
| Hand bone / GT | 2.329 / 25.782mm ≈ 0.090 | 비율0.9955 |
| Hand local spread / GT | 2.110 / 72.758mm ≈ 0.029 | 비율1.0017 |

v1.1의 best epoch는 역할마다 다르다(frontend9/body9/hand8). Hand local124.67mm는 global best에 대응하는 값이며 local 자체의 최저값이라고 주장하지 않는다. 위 표는 실패 양상과 복원 가능성을 비교하기 위한 것으로, 484.79→13.71mm를 동일 평가집합의 성능 개선율로 계산하지 않는다. 형상 비율도 서로 다른 표본 집계다.

v1.1의 별도 Getting_Up 두 이미지 시연은 T1 reset 추론이고, v1.4 그림은 학습된 50frame temporal 클립이다. 두 그림의 차이만으로 temporal 모델의 일반화 성능을 평가하지 않는다.

### 3.2 v1.4 안에서의 학습 변화

| 모델당 update | Body MPJPE(mm) | Hand local MPJPE(mm) | 관찰 |
| --- | ---: | ---: | --- |
| 0 | 495.39 | 143.02 | 몸과 손이 작게 뭉침 |
| 200 | 123.19 | 47.03 | 몸의 크기는 복원되기 시작, 손은 여전히 축소 |
| 1,000 | 19.19 | 10.82 | 형태가 크게 개선, 일부 frame 기준은 미달 |
| 5,000 | 13.71 | 2.90 | 잠정 공간 기준과 첫·중간·끝 오차 기준 통과 |

v1.4도 200 update 시점에는 손의 bone 비율이 0.133, spread 비율이 0.330으로 뭉침이 남았다. 이후 같은 입력·모델에서 개선됐다는 점은 **학습량이 해당 진단의 중요한 요인**임을 직접 지지한다. 다만 v1.1의 200 update는 20개 클립에 나뉘므로 v1.4의 단일 클립 200회 반복과도 동등하지 않다.

최종 body/hand 오차는 같은 클립의 평균 자세 baseline인 126.42/23.02mm보다 낮다. Frame 0→49 변위도 body 292.67mm(GT 291.55), hand local 38.96mm(GT 38.06)로 재현돼, 한 정지 자세만 출력한 성공은 아니다. 그렇더라도 한 클립의 시간·카메라 패턴을 암기했을 가능성은 남는다.

### 3.3 v1.1은 학습 자체가 되지 않았는가?

회수 `run.json`의 teacher train loss 합계를 다시 확인했다. 각 epoch는20update로 동일하며 position 항목의 epoch 합은 body279.06→159.79, hand49.21→6.98(e1→e10)로 줄었다. 첫 update gradient도0이 아니고 supervision skip0, 파라미터·결과 tensor는 유한하다.

이는 optimizer가 작동하고 훈련 목적이 개선됐다는 근거다. 그러나 임의 diffusion timestep의 GT-noised 복원 학습 loss와 pure-noise DDIM10의 mm 오차는 다른 지표다. 특히 hand train은 GT상체이고 validation은 예측상체다. Loss 감소를 곧바로 실제 생성 성공이나 일반화 성공으로 해석할 수 없다. Epoch10의 validation 악화만으로 과적합이 유일한 원인이라고도 판단할 수 없다.

## 4. v1.1 실패 원인 가설과 우선순위

| 가설 | 근거 | 현재 판단 | 아직 모르는 점 |
| --- | --- | --- | --- |
| H1. Teacher 업데이트·클립별 노출 부족 | 각200update/clip당10회. v1.4는 같은조건에서200→5,000 사이 특히손 형태 개선 | 우선 검증할 유력 원인 | v1.1 데이터·예측조건에서도 충분히 반복하면 수렴하는지 |
| H2. 2D 조건의 정보 품질 부족 | Best214.84px, 표본heatmap entropy0.8253. GT조건의v1.4는복원가능 | 유력한 기여 요인 | GT효과와 반복량·암기효과가분리되지 않음 |
| H3. Hand 상체 조건 차이와 body 오류 전파 | Train GT상체→validation 예측상체. Body 자체가축소됐고wrist도부정확 | 조건 차이는 확인된 사실, 영향 크기는 추정 | Hand local붕괴의얼마나많은부분이상체조건때문인지 |
| H4. 다양성·일반화 어려움 | v1.1은20clip 학습/29clip 평가, v1.4는1clip 암기 | 비교 차이를설명하는중요한요인 | v1.1의동일train DDIM10에서도붕괴하는지 |
| H5. 학습 복원과 실제 sampling 사이의 차이 | v1.1 train loss감소와생성붕괴가공존 | 후속조사대상 | timestep별수렴,조건의존도,노이즈 seed에대한안정성 |
| H6. 좌표·scale·mask 등 구현 결함 | v1.4 동일계열구조/scale에서형태복원성공 | 모든조건에서발생하는공통치명오류가능성은낮아짐 | 예측특징·cascade·여러sequence 경로의국소결함은배제못함 |

가장 그럴듯한 설명은 H1+H2가 body 복원을 어렵게 하고, 손에는 H3가 추가된다는 것이다. 조건이 불명확한 상황에서 출력이 평균에 가까워지거나 축소될 수 있다는 설명은 **현재 관측에 대한 가설**이며, 조건부 평균으로 수렴했음을 측정한 결과가 아니다.

추가로 GT 특징은 좌표만 바뀐 것이 아니다. v1.4는 유효한 GT의 confidence/presence를1로, 누락은0으로 기록하고 화면 밖 좌표도 유지한다. 예측 특징은 soft confidence와 항상 존재하는 measurement 표시를 사용한다. 따라서 전체 feature를 바꾸는 시험을 “2D 좌표 오차만의 효과”라고 부르면 안 된다.

T4여서 정확도가 낮았다고 볼 근거는 없다. 두 실행은 CUDA FP32였고 v1.1도 정상 종료했다. GPU 차이는 속도·수치 재현성의 통제 변수이며, A100으로 바꾸기만 하면 정확도가 해결된다는 결론은 지원하지 않는다. v1.4의 초기 metadata 비교 버그 역시 학습 시작 전 오류였으므로 v1.1의 3D 붕괴 원인과 구분한다.

## 5. 향후 테스트 방향 — 최소 비용부터 원인 분리

아래는 제안이며 새 실행 승인이나 확정된 버전명이 아니다. 테스트를 모두 한꺼번에 수행하지 않고 앞 결과가 다음 분기를 결정하게 한다. 고정 실행본을 준비한 뒤 사용자 실행은 기존 테스트 세션으로 인계한다.

```text
기존 v1.1 / v1.4 결과
├─ A. 저장 모델로 train 복원과 조건 민감도 평가(재학습 없음)
│  ├─ v1.1 train도 붕괴 → B. 동일 조건의 반복 학습량 진단
│  └─ train은 정상, validation만 실패 → D. 일반화·조건 분포 조사
├─ C. 같은 클립·같은 반복량으로 GT/예측 조건의 학습 비교
│  ├─ 예측2D 교체만 악화 → frontend 좌표·confidence 분리 조사
│  └─ 예측상체 교체만 악화 → body 품질·hand 조건 적응 조사
└─ v1.2 전체데이터 확대: 별도 진행, 위 원인분리 시험을 대신하지 않음
```

### A. 저장 checkpoint를 이용한 평가부터

먼저 v1.1 best 모델을 고정하고 **실제 학습에 사용한 같은 Standing_Up50frame**에서 DDIM10·train loss와 구분되는 MPJPE·형상 비율을 측정한다. 기존 validation 결과와 비교하면 훈련 표본 복원 실패인지 일반화 실패인지 좁힐 수 있다. Frontend는 기존best e9, body e9, hand e8의 의존 SHA를 그대로 사용한다. 초기에는 한 클립만 사용하고 확대 여부는 결과로 판단한다.

v1.4 모델에서는 GT2D/예측2D × GT상체/예측상체의 네 조건을 같은클립·카메라·노이즈로 **추론만** 비교한다. 예측 frontend와 body checkpoint를 명시하고 body는GT상체를입력받지않는다. Hand는local오차와GT/예측wrist 부착오차를함께보고한다. 현재 oracle API는 GT 조건을 강제하므로 예측 상체 시험에는 별도의 명시적 adapter가 필요하다. 기존 가드를 우회하지 않는다.

이 교체는 분포가 바뀐 입력에 대한 민감도 검사다. 성능 악화가 곧 해당 조건으로 학습한 모델도 실패한다는 뜻은 아니다. 수정된 조건은 반드시 별도 진단 태그로 기록한다.

### B. v1.1의 학습량 가설을 먼저 분리

A에서train도실패하면, 기존v1.1 frozen frontend·20clip·모델·LR·validation을유지하고teacher update를늘리는제한실험을제안한다. 실제확장범위는pilot으로확정하되200→1,000→5,000update 지점을후보로둔다. 20clip에서각각10→50→250회노출이며v1.4의단일클립5,000회와동일노출은아니다. 같은 초기 state에서 200/1,000/5,000update를 차례로 저장하는 실행을 우선한다. 현재 v1.1의 last.pt는 hand 단계이므로 body best 가중치만으로는 optimizer/RNG까지 동일한 200update 재개를 할 수 없다. 완전한 단계별 상태가 없는 경우 fresh initialization으로 대조하며, 가중치만 이어 쓰는 실험은 정확 재개로 부르지 않는다. Frontend를동시에재학습하지않아teacher학습량효과를분리한다.

1,000 update에서 train 형태와 오차가 개선되는지 먼저 판정한다. 개선이 없으면 무작정 확대하지 않고 GT 조건 비교로 넘어간다. 개선 중이라면 고정 예산 내 다음 지점의 진행 여부를 정한다. Validation 악화와 train 개선은 별도로 기록한다.

### C. GT/예측 조건을 바꾸는 통제 학습

한 클립에서 teacher 초기화·노이즈 일정·update·카메라·loss·평가 seed를 고정한다. 먼저 GT2D/GT상체 대 예측2D/GT상체를 비교하고, 그다음 2D 조건을 고정한 채 hand의 상체 조건만 교체한다. 모델별 seed 차이가 있는 현재 v1.4 결과를 그대로 동일 초기화 대조군으로 쓰지 않는다. 가능하면 동일한 초기 state를 복제한다.

예측 2D는 고정 frontend의 출력 캐시를 재사용한다. 전체 feature 교체로 악화되면 후속으로 좌표만 교체하고 confidence/presence는 고정하는 검사와, 반대로 confidence만 교체하는 검사를 분리한다. GT상체 학습→예측상체 평가의 민감도와 예측상체 학습→예측상체 평가의 적응 효과도 구분한다. 후자는 새로운 학습 조건이므로 별도 설계한다.

추가로 GT2D를 다른 frame 순서로 바꾸거나 고정 좌표로 대체하는 추론 대조를 후보로 둔다. 결과가 거의 변하지 않으면 카메라·시간·클립 암기를 이용했을 가능성을 조사한다. 이 인위적인 입력은 정상 성능 지표가 아닌 조건 의존도 진단으로만 표시한다.

### D. v1.2와 일반화 평가

v1.2는 고유 train 20,888개·teacher 564clip·각 10epoch로 데이터 폭을 넓힌다. 같은 epoch라도 teacher 5,640 update가 되어 v1.1보다 총 update가 늘지만 클립당 노출은 여전히 10회다. 따라서 v1.2가 좋아져도 데이터 다양성만의 효과라고 단정하지 않고 같은 update의 checkpoint와 같은 epoch를 함께 비교한다. 반대로 실패해도 “데이터를 늘려도 소용없다”가 아니라 클립별 노출과 조건 품질을 검토한다.

GT 조건의 소규모 다중 클립 학습은 한 클립 암기에서 벗어나는 후속 후보이다. Test set을 튜닝에 쓰지 않고 고정 train과 참고 validation을 분리한다. 원본 motion 독립성은 미확인으로 유지한다. v1.3의 다른 시나리오 혼합은 계속 보류한다.

## 6. 모든 후속 비교에서 기록할 항목

- 소스revision·입력/모델/특징캐시SHA, 고유frame/clip수, update와clip당노출 횟수, 초기화/노이즈seed.
- 2D좌표오차와confidence/entropy, body MPJPE, hand local/global, GT·예측상체/손목의명시, bone/spread비율,첫·중간·끝오차·시간변화.
- Train과참고validation,GT-noised 학습복원과pure-noise DDIM10 생성,GT oracle와예측입력 결과를분리.
- 실제GPU·FP32 등precision·준비/학습/평가/보관시간·VRAM·읽기량·디스크·중단/재개와모델보존성공.
- NaN/OOM/pin불일치/보관실패에는중단하고상한을자동확대하지않음. 초기예산은실제pilot으로고정하고이번보고서로추가장시간실행을시작하지않음.

기존 A100 oracle 약 10분을 T4나 20clip 실험 시간으로 단순 복사하지 않는다. 재학습 없는 평가와 캐시 재사용을 우선하고, 결과에 따라 추가 학습을 선택한다. 모델 교체·데이터 확대·epoch 확대를 한 실험에 동시에 적용하지 않는 것이 핵심이다.

## 7. 근거와 검토 한계

- [v1.1 실제 CUDA 재실행 결과](V11_CUDA_RERUN_RESULTS.md): T4·참고validation·시간·형태축소·의존checkpoint 검토.
- [v1.4 실제 결과](V14_ORACLE_RESULTS.md), [실행 계약](V14_ORACLE_OVERFIT.md): A100·동일클립oracle·학습경과·ZIP/모델검토.
- [실험 분기 계보](V1_EXPERIMENT_LINEAGE.md): 향후 설계·관찰·판정의 지속 기록.
- 고정 원본: `outputs/v11_rerun_20261110/user_test/cuda_70accf2/training.zip`, `outputs/v14_pinfix_release_20261112/user_test/retry_001/training.zip`. 각 결과문서의SHA와root_review.json을따른다.
- 코드 대조: `configs/rewind_v1_epoch_expansion.json`, `configs/rewind_v14_oracle_overfit.json`, `rewind_v1/run_stages.py`, `frontend.py`, `runner.py`, `oracle_data.py`, `oracle_overfit.py`.

이번에회수run.json의train loss와설정/조건/seed를재대조했다. 새모델추론·노이즈반복·통제학습은실행하지않았다. v1.4 성공은구조의제한된복원가능성을보여주지만논문전체재현이나범용성검증을완료하지않는다.
