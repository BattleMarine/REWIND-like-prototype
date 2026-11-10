# v1: 1,000페어에서 REWIND 구조 재현

2026-11-10 실제 감사 결과: Colab CPU에서 새 train1,000·validation2,236의 파일·카메라 수치 감사가 완료됐다. Timestamp는 발견되지 않았고 영상/원본 motion 검토가 남아 학습은 보류했다. Root가 회수96개 파일과 내부content pin을 확인했다. 후속 테스트 세션 보고로 raw cache65개와 metadata2개의 사용자 Drive 보관·복사 후 검증을 확인했다. 이후 고정 hash로 로컬 복원하며 학습 보류는 유지한다. [결과·한계·다음 판단](V1_ACTUAL_001_RESULTS.md).

2026-11-10 실제 시험 준비: 고정1,000 train·2,236 validation의 원본 사전검사·raw pack 재사용·카메라/시간/영상 검토 자료 생성과 결과 회수, 조건부 A100 학습 notebook을 준비했다. Root 관련20개 로컬 검사 통과. 실제 timestamp와 원본 motion 근거가 없으면 학습은 보류하며 수치 감사와 실제 학습 성공을 구분한다. [V1-ACTUAL-001 인계](V1_ACTUAL_TEST_HANDOFF.md).

2026-11-10 실행 연결 구현: frontend 사전학습→고정 예측 특징→body→hand teacher의 epoch 학습·평가·저장/재개와 CLI·Colab 진입점을 추가했다. 동일 로컬 raw pack 재사용과 실제 A100/입력 근거 확인을 분리한다. 실제 데이터/GPU 학습 결과는 아직 없다. [실행·캐시·검증 범위](V1_TRAINING_RUNNER.md).

2026-11-10 C 첫 구현: 논문 기반 frontend·식(4) causal body/hand teacher 후보와 근거 검증 clip adapter·한 update/저장·재개 API를 구현했다. Root 관련10개 검사 통과. 실제 입력 정합 검토, frontend/epoch 실행 연결과 GPU 학습은 남아 있다. [구현·검증 경계](V1_TEACHER_IMPLEMENTATION.md).

2026-11-10 B 첫 구현: 연속 구간1,000페어 목록 선택, 제한 병렬 raw pack 생성·중단 재개·복원/조회 도구를 로컬 검토했다. 실제 보존 목록의 계획을 독립 재현하고3개 검사를 통과했다. 원격 payload 확보·속도 측정·시간/카메라 검증·GPU 학습은 아직이다. [지원 범위와 근거](V1_INPUT_PREPARATION.md).

2026-11-10 사용자 결정: train 데이터 규모를1,000페어로 유지하며 REWIND 논문 구조를 최대한 재현한다. ArchVis 전체 확대보다 이 작업을 우선한다. v0 결과와 고정 manifest는 보존한다. 아래는 계획이며 구현·GPU 실행 완료가 아니다.

## 데이터 범위

1,000은 고유 train stereo RGB+pose 프레임 수다. 시간 window를 겹쳐 만들거나 epoch를 반복해도 고유 데이터가 늘어난 것으로 세지 않는다. 기존 공식 validation2,236은 별도 평가 집합으로 유지하고 test로 튜닝하지 않는다. 시간 context를 포함한 실제 사용 frame 수를 manifest에 기록하여 1,000 train 제한 밖의 context를 숨겨 추가하지 않는다.

기존 v0 manifest를 읽기 전용으로 조사한 결과 train1,000은230개 sequence에 분산돼 있고 프레임 번호 기준 최장 연속 구간은3프레임, 50프레임 연속 구간은0이다. 이는 timestamp 검증을 대신하지 않는다. 논문 부록의50연속프레임 입력을 고려해 **같은 규모의 연속 구간 중심 subset 재선정**을 계획한다. 정확한 sequence·프레임 구성은 아직 확정하지 않았으며, v0와 다른 manifest를 발급하고 데이터 선택 변경이 성능 비교에 미치는 영향을 기록한다. 카메라 자세·timestamp·RGB/pose 정합·원본 motion 중복을 검토한다.

## 구조와 진행 순서

1. 논문 본문·부록과 코드의 차이를 표로 고정한다. EfficientNet 기반2D encoder/CNN decoder, Graph Transformer, causal relative-temporal attention, diffusion schedule·좌표·loss를 대조한다. 원문에 없는 구현 세부는 가정으로 남기며 정확한 복제라고 주장하지 않는다.
2. 실제 연속 구간을 읽는 adapter와 영구 재사용 가능한 입력 묶음을 준비한다. 기존 단일프레임 선택·64px 캐시를 새 모델에 무조건 재사용하지 않는다. 해상도·전처리·모델/입력 버전과 hash를 고정한다.
3. 몸3D teacher를 실제 데이터로 학습하고 손 teacher를 연결한다. 손 학습의 GT 상체와 평가의 예측 상체를 분리한다. GT2D 진단과 실제 예측2D 평가도 분리한다.
4. 위치·중간 frame loss 외에 velocity·foot loss의 시간/좌표/접촉 근거를 확보하고 구현한다. 근거가 없으면 생략 사실을 논문과의 차이로 기록한다.
5. Teacher 결과를 바탕으로 one-step student와 증류를 구현한다. 실제 sequence의 MPJPE·시간적 안정성·causality·속도를 평가한다. Identity conditioning과 avatar IK는 선택 확장으로 구분한다.

구조 재현과 논문의 데이터 규모·2M 학습 step·성능 재현은 별개다. 작은 dataset 조건에서 과적합을 예상하며 논문의 학습 횟수를 무조건 복사하지 않는다. 구체적인 학습 예산은 새 모델의 실측 후 고정한다. M0~M6 식별자는 유지하며 v1은 이들 중 실제 데이터 연결과 구조 차이 보완을 진행하는 버전이다.

## v0 시간과 최적화 계획

회수 실행의 실제 GPU는 A100이 아닌 Tesla T4다. 보고 elapsed5,032.453초 중 원본 읽기/hash/decode/resize/캐시 생성4,629.253초, 반복 학습284.305초, 평가6회105.205초다. 준비 대상은 train1,000+validation2,236=3,236페어이고 학습 처리 수는1,000×20=20,000이다. 준비 약1.431초/고유페어, 학습 약0.014215초/처리페어이며 전체 wall-clock을 train1,000으로 나눈 값은 약5.03초다. 서로 다른 분모·측정 범위를 혼동하지 않는다. 자세한 근거는 [v0 결과](ARCHVIS_V0_RESULTS.md).

우선순위는 다음과 같다. 모두 신규 구현 전 계획이며 배속 향상을 보장하지 않는다.

- **캐시 영구 보존·묶음 전송:** 검증된 선택 데이터와 전처리 결과를 버전/hash가 있는 소수 묶음으로 만들어 새 Colab에서는 로컬 SSD로 복원한다. 최초 준비 비용과 재사용 실행 비용을 분리한다. 현재 v0 임시 캐시는 프로세스 종료 시 삭제되므로 이번 회수 ZIP에317MiB 캐시가 있다고 가정하지 않는다. 새 v1 해상도/선택 데이터에 맞는 캐시가 필요하며 Drive 업로드는 사용자에게 요청한다.
- **준비 단계 병렬화:** 제한된 worker2/4/8 등의 후보로 파일 읽기·hash·decode를 병렬 처리한다. 동일한 작은 입력에서 순차 방식과 비교하고 요청·메모리 상한, 결정적인 manifest 순서, 실패 회수를 유지한다. Drive 파일시스템의 동시 읽기 성능은 실측하며 worker 수만 늘리면 비례해서 빨라진다고 가정하지 않는다.
- **학습 입력과 GPU 연산 겹치기:** 로컬 묶음 기반 DataLoader worker·prefetch·pinned memory/비동기 전송을 적용하고 대기 시간을 계측한다. batch/AMP는 모델·VRAM·수치 검증 후 조정한다. 큰 batch로 변경할 경우 update 수와 학습 조건 변화도 기록한다.
- **실제 GPU 확인:** 실행 전에 GPU 이름과 VRAM을 출력하고 A100 기준 실행인지 확인한다. GPU 연산·전송·준비·평가 시간을 구분한다. v0 작은 모델의 속도를 v1 diffusion teacher에 그대로 적용하지 않는다.

캐시 생성92%가 모두 네트워크 시간이라는 근거는 없다. 구간별 계측을 추가하고 IO와 CPU 병목을 구분한다. 새로운 사용자 시험을 세분화해 반복하기보다 제한된 비교를 한 고정 실행으로 모아 테스트 세션에 인계한다.

참고: [Colab FAQ의 작은 IO/묶음 복사 권장](https://research.google.com/colaboratory/intl/en-GB/faq.html), [PyTorch 데이터 로딩·GPU 전송 최적화](https://docs.pytorch.org/tutorials/recipes/recipes/tuning_guide.html), [REWIND 원문](REWIND.pdf).
