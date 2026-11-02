# 전체 데이터 학습 준비

2026-11-02 사용자 결정: Colab 기준으로 전체 데이터셋 학습을 준비하며 화면 개발보다 우선한다. 화면용 프론트와 모델·학습·평가·추론 코어를 분리한다. RGB에서 2D 관절을 예측하는 신경망 frontend는 학습 모델에 포함한다. 전체 다운로드·실제 장시간 학습·의존성 설치·원격 접근은 이 개발 작업에서 실행하지 않는다.

## 감사 결과

| 경로 | 현재 제공 기능 | 전체 학습까지 남은 사항 |
| --- | --- | --- |
| `rewind_m1` | UI 없는 regression CLI, 계약·mask·sequence/motion split 검사, optimizer/scheduler/scaler/Python·Torch RNG·step 재개 | 기존 trainer는 전체 JSON과 모든 window tensor를 RAM에 올리고 replacement sampling을 사용. 새 shard 공급과 cursor 연결 필요 |
| `rewind_m1.shard_training`(2차 추가) | 별도 합성 M1 shard trainer, epoch cursor·실제 sample 수 재개·독립 validation·checkpoint 기반 추론 export | 실제 release·persistent 준비 공급·GPU/AMP pilot 미검증 |
| `rewind_m2`, `rewind_m3` | UI 없는 합성 causal diffusion·cascade 학습과 재개 | 실제 데이터 계약·shard 공급·독립 validation 연결 필요 |
| `rewind_m1.frontend_train` | 작은 CPU RGB→2D 진단, inference weights 저장 | 소량/100step 제한, 일반 재개·GPU·장시간 공급 미지원 |
| `rewind_m1.rgb_shards`(3b1 추가) | native RGB+GT 한-shard 공급·독립frame chunk·split/identity·PNG lazy/full 검사·epoch cursor | resize/2D trainer·기존3D featurecatalog로 chunk 자동 연결 미구현 |
| `rewind_archive_index.preparation` | 명시 target의 예산 제한 탐색·snapshot 저장/복원·선택 추출·로컬 재사용 | epoch와 archive coverage는 별개 |
| `rewind_prepare_queue`(3a 추가) | 기존 backend 소량 범위의 raw 준비 순서·영속 cursor·전역 예약/정산·partial 재개·준비 목록 | 학습 shard 변환/자동 공급·eviction·실제 Drive queue/새VM 지속성 미검증 |
| 실제 pose candidate | 원본 hash·좌표 수치 검사를 거친 진단용 candidate | `training_release=not_implemented`, 원본 motion identity·독립 validation 등 blockers 유지 |

화면 UI를 학습 코드가 import하지 않는 구조는 이미 존재한다. 화면을 분리한다는 이유로 신경망 2D frontend를 제거하거나 학습을 화면 실행에 종속시키지 않는다. 기존 소량 진단은 전체 데이터셋 baseline 성능을 입증하지 않는다.

## 1차 구현 범위

새 `rewind_m1.shards`는 **로컬의 준비된 `m1_features_v1` 파일**만 지원한다. GT 2D 진단 입력과 고정 버전 예측 2D 특징을 기존 `load_document`로 검증한다. RGB PNG·동영상·raw pose candidate를 직접 학습 데이터로 승격하지 않는다. M1 모델은 기존 `single_frame`과 `temporal` regression이며 논문 규모 모델 지원을 뜻하지 않는다. M2/M3/2D trainer는 이번 shard 경로에 연결되지 않았다.

`m1_feature_shards_v1` 카탈로그에는 다음을 고정한다.

- 공통 `dataset_identity.id`, `dataset_identity.dataset_version`, `contract`, `window_size`, 자원 상한
- 순서가 고정된 shard 상대 경로·파일 SHA-256·byte 수·record 수
- 각 sequence의 ID·motion ID·split·record 수·첫/마지막 frame
- 위 선언의 canonical content hash

각 shard의 `manifest.id`는 공통 데이터셋 ID다. shard마다 고유 ID를 넣는 필드가 아니며, shard 고유 식별은 경로와 payload SHA로 한다. `manifest.dataset_version`도 공통 버전과 같아야 한다. 다른 버전의 특징 파일을 한 카탈로그에 섞지 않는다.

같은 sequence가 여러 shard에 걸치면 거부한다. 이는 **제공한 특징 파일 집합 내에서 sequence 그룹이 한 shard 안에 완결**됨을 뜻하며 원본 sequence의 모든 frame을 확보했다는 증명이 아니다. 원본 일부 frame만 있어도 준비된 입력의 provenance와 gap reset을 보존한다. 원본 archive coverage, gzip CRC, frame 연속성 및 실제 시간 검증은 별도다. 단일 sequence 그룹이 shard 상한을 넘으면 현재 지원 범위 밖이며 임의 분할하지 않는다. 향후 causal context를 명시한 chunk 경계 계약이 필요하다.

카탈로그에서 sequence 중복과 motion의 split 교차를 검사하고, 전체 읽기에서는 실제 record 선언과 기존 tensor 계약을 대조한다. 숫자만 다른 shard descriptor나 검증 문자열을 근거로 실제 학습 release를 발급하지 않는다. 원본 motion ID의 진실성은 명시된 문자열의 중복 검사와 별도로 검증해야 한다.

## 메모리와 순회

[상한 설정](../configs/m1_shard_limits.json)의 기본값은 카탈로그4MiB·4096shard, shard32MiB·4096record·window tensor8,388,608요소다. 사용자 설정은 이 상한을 낮출 수 있다. `window_size`는1~256이다. 요소 수는 `records × window_size × (18 × joints + 1)`로 tensor화 전에 제한한다. byte 제한은 JSON 파싱 전, record·요소 제한은 window tensor화 전에 검사한다.

`ShardDataset.load(index)`는 이전 내부 cache를 해제하고 한 shard의 tensor만 보유한다. 기존 구현은 한 shard 안의 JSON, parsed 객체, window와 stack tensor를 일시적으로 함께 사용하므로 이 상한은 전체 RSS의 byte 상한이 아니다. 호출자가 반환 tensor를 계속 참조하면 cache 외부에 메모리가 남는다. 후속 trainer는 이전 shard/batch 참조를 해제한 다음 다음 shard를 로드해야 한다. GPU batch와 optimizer 메모리도 별도다. 모든 원본이나 전체 dataset tensor를 한번에 생성하는 기능은 없다.

`EpochSampler`는 seed와 epoch로 shard 순서 및 split 내 anchor 순서를 결정하며, 한 epoch에 각 해당 anchor를 한 번씩 방문한다. batch는 shard 경계를 넘지 않고 마지막 작은 batch도 유지한다. `peek(batch_size)`는 상태를 바꾸지 않으며 update 성공 뒤에만 `commit(batch_size)`한다. `state_dict`는 policy·catalog hash·split·seed·shuffle·epoch·shard 위치·offset을 포함한다. 복원 시 identity와 위치를 검사한다. 아래 2차 구현의 별도 trainer에 결합했으며 기존 step trainer의 checkpoint 형식은 유지한다. Python random 구현 버전과 실행 환경도 새 checkpoint의 재개 계약에서 고정한다.

## 실행 명령과 readiness 의미

아래 명령은 기존 환경에서 동작을 확인했다. 개발 검증 예시 경로는 소량 합성 특징이며 실제 학습 자료가 아니다. 새 입력에서는 준비된 파일의 상대 경로로 바꾼다.

```powershell
python -m rewind_m1.readiness build --base outputs/full_training_readiness_20261102/developer/fixture --shard train.json --shard validation.json --window-size 3 --output outputs/full_training_readiness_20261102/developer/fixture/catalog.json
python -m rewind_m1.readiness inspect --catalog outputs/full_training_readiness_20261102/developer/fixture/catalog.json
python -m rewind_m1.readiness inspect --catalog outputs/full_training_readiness_20261102/developer/fixture/catalog.json --full-read
python -m rewind_m1.readiness candidate --candidate outputs/real_stereo_sample_20261102/root/prepared/pose/candidate.json
```

`build`는 존재하지 않는 output만 만들고 원본 특징을 변경하지 않는다. `--limits configs/m1_shard_limits.json`으로 낮춘 상한을 적용할 수 있다. catalog는 shard base 바로 아래에 저장해야 한다. 이미 생성된 파일에는 `inspect`를 사용한다.

| 검사 | 실제 읽는 범위 | 의미 |
| --- | --- | --- |
| `build` | 각 로컬 shard의 hash·선언·window tensor 전체 검사 | 카탈로그 생성 시 고정된 로컬 payload 검증. 보고서의 `build_payloads_fully_read=true` |
| `inspect` | 카탈로그만 | `declarations_valid`, `full_payload_read=false`. 파일 존재·payload·원본은 검증하지 않음 |
| `inspect --full-read` | 모든 명시 shard를 하나씩 읽고 기존 tensor 계약 검사 | `local_payloads_valid`, `full_payload_read=true`. 원본 archive 전체 검증 아님 |
| `candidate` | 제한된 candidate JSON과 기존 candidate schema/hash 검사 | 기존 `training_eligibility`와 blockers 그대로 보고. raw·geometry 재검증 아님 |

모든 결과의 `full_training_ready`는 현재 **false**다. `dataset_sample`의 `verified=true`와 evidence 문자열이 있어도 별도 검증된 release나 전체 readiness로 간주하지 않는다. `model_support`는 지원 데이터 경로와 M1/M2/M3/2D 연결 상태를 구분한다. CLI 오류는 0이 아닌 exit code로 반환한다. 선언검사 성공과 학습 가능 승인을 혼동하지 않는다.

## UI와 가중치 경계

`rewind_m1.artifacts`는 trainer·CLI·화면 코드를 import하지 않는다. M1 regression 추론용 `m1_regression_weights_v1` 파일을 저장/읽는다. 기존 2D frontend용 `m1_frontend_weights_v1`와 다른 형식이다. 기존 checkpoint 형식도 바꾸지 않는다.

`export_weights`는 모델 state와 최소 model config(model/window/hidden/dropout/target scale), input contract, catalog hash, 원본 checkpoint SHA, source revision/code hashes, training step을 저장한다. 이는 학습 checkpoint의 optimizer/RNG를 대신하지 않는다. checkpoint SHA는 호출자가 실제 파일에서 계산해 넘겨야 하며, export 함수가 원본 checkpoint를 다시 열어 모델과 대조하지는 않는다. 후속 trainer 통합이 이 연결을 책임진다. 가중치 파일 SHA와 metadata/state digest는 provenance와 변경 탐지용이며 서명은 아니다.

`load_weights`는 소비자가 명시한 기대 contract·catalog·checkpoint·source·model config를 대조하고 strict state shape 검사를 거쳐 eval 모델과 metadata를 반환한다. 반환 모델 출력은 기존과 동일한 normalized 좌표이며 `model_config.target_scale_mm`을 곱해 mm로 바꾼다. 입력 feature/window 조립은 별도 계약이다. 신뢰하는 로컬 export만 읽으며, 임의 외부 파일을 안전하게 실행하는 보장을 하지 않는다.

export/load는 CPU Torch RNG를 보존하고 export는 원본 모델의 mode/device/state를 바꾸지 않는다. 파일은64MiB 제한이며 export도 직렬화 크기를 검사하고 기존 output을 덮어쓰지 않는다. 모델 상한은256관절·hidden1024이며 UI나 모델 학습 규모를 보장하는 값이 아니다. 재개 파일과 추론 파일을 분리해 UI는 고정 가중치를 로드하고 학습은 별도 프로세스에서 계속할 수 있다. 동시 파일 교체·UI hot reload는 이번 구현 범위 밖이다.

## 이후 구현과 완료 기준

1. **현재 묶음:** 로컬 특징 shard 계약·읽기·readiness·sampler·추론 가중치. 기존 dataset gate 유지.
2. **Shard trainer(로컬 합성 CPU 구현·검증):** 기존 M1 모델/loss 재사용, CPU/CUDA/AMP 실행 경로, 독립 validation 누적, cursor와 optimizer/scheduler/scaler/RNG·환경·catalog를 함께 저장. 합성 CPU uninterrupted/resume 동등성과 epoch coverage를 검증했다. CUDA/AMP 실제 실행은 미검증이며 기존 trainer API는 보존한다.
3. **준비 공급 연결(3a 로컬 구현·검증):** 기존 persistent-index의 명시 target·전역 예약/정산·재사용을 제한된 raw queue에 연결했다. RGB/pose 키 대조·임시 디스크 상한·중단/재개를 검사하고 자동 삭제 없이 소비 표시를 기록한다. 학습 shard 변환/자동 공급·eviction·실제 Drive/새VM 시험은 남아 있다. archive 탐색 순서와 학습 shuffle을 구분하고 epoch를 전체 indexing 완료로 사용하지 않는다. [queue 계약](PREPARATION_QUEUE.md)을 따른다.
4. **실제 release:** unit/좌표·calibration·RGB-pose·원본 motion·독립 validation의 근거 검토와 명시 release adapter. GT 입력 진단과 예측 입력 결과를 분리. 미검증 상태를 문자열로 해제하지 않음.
5. **Colab pilot:** 테스트 세션에서 실제 GPU/VRAM·디스크·세션 길이·준비 비용을 측정. 작은 sample→overfit→pilot 순서, 실행 전 전송량/시간/RAM/디스크/업데이트 예산과 중단 조건 고정. artifact/checkpoint/manifest를 runtime 추출본과 분리 보관. 결과를 보고 다음 규모 결정.
6. **M2/M3/2D 확장:** 지원한 M1 경로의 검증 후 모델별 input/loss/시간 정합성·재개 계약을 추가. 동영상 adapter/UI는 이 학습 준비 우선순위 뒤에서 진행.

Colab 자원은 아직 측정하지 않았으며 전체 원본 학습·논문 규모·실시간 목표의 달성은 미검증이다.

## 개발 검증 근거

`outputs/full_training_readiness_20261102/developer/verify.py`와 `tests.json`, `tests.txt`, `code_hashes.json`, `declarations.json`, `full_read.json`, `real_candidate.json`에 근거를 저장한다. 소량 합성 tensor 동일성, mask, identity/hash/상한/누수 거부, sampler epoch coverage와 cursor 복원, 추론 prediction 동일성·RNG·mode·파일 상한·UI 독립 import를 검사한다. 원본 Serena candidate는 학습 차단 상태로 보고하며 수정하지 않는다. 실제 학습 update와 원격 요청은 없다. 코드 검증과 GPU/데이터 readiness를 구분한다.

신규17개와 기존 공통 frame9개·M1 데이터 계약4개, 총30개가 failure/error/skip0으로 통과했다. Root도 신규17개를 별도로 확인했다. 실제 candidate SHA는 `44ec803069ddee642e45832f88a71979c5cf1966bbe2e8574d37bd9742bec0e2`이고 기존6개 blocker가 그대로 보고됐다. 전체 학습 완료나 실제 원본 검증을 새로 수행한 결과가 아니다.

## 2차 구현: headless M1 shard trainer

`rewind_m1.shard_training.ShardTrainer`와 `python -m rewind_m1.shard_cli`를 추가했다. 기존 `rewind_m1 train`의 API·sampler·checkpoint는 바꾸지 않는다. 새 진입점은 `m1_shard_checkpoint_v1`을 사용한다. 모델/loss/seed·RNG/원자적 checkpoint 저장은 기존 코드를 재사용한다. 화면 UI import나 실행이 필요하지 않다.

현재 새 trainer의 **학습 허용 데이터는 `synthetic_fixture`만**이다. `dataset_sample`은 검증 문자열이 있어도 `shard_dataset_sample_release_not_implemented`로 거부한다. 실제 candidate·기존 diagnostic-only checkpoint를 이 진입점으로 우회 학습할 수 없다. 실제 학습 release와 persistent 준비 공급은 후속 묶음이다.

학습 전에 config의 model/window/batch/seed/크기와 catalog 계약을 검사하고 모든 명시 payload를 한 번씩 검증한다. 별도 train/validation이 필요하며 sequence와 motion split 검사를 유지한다. `single_frame`/`temporal`, CPU/CUDA, CUDA AMP 실행 경로를 제공하되 이번 검증은 CPU FP32다. CUDA/AMP의 실제 속도·VRAM·수치/skip 동작은 Colab pilot에서 확인해야 한다. CPU mock의 scaler skip 제어 검사는 GPU 검증을 대신하지 않는다.

학습 중 같은 shard는 한 내부 cache로 재사용하며 다음 shard 이동 때 이전 cache를 먼저 해제한다. 현재 cache는 로드 시 hash를 확인한 고정 payload snapshot이며 매 batch마다 원본을 다시 읽지 않는다. 학습 중 원본 파일을 변경하지 않아야 하며 새 로드/새 실행 preflight에서는 변경 hash를 거부한다. 함수가 보유한 이전 batch 참조는 다음 shard를 읽기 전에 사라지고 평가도 이전 shard 지역 참조를 해제한다. 호출자가 tensor를 외부에 붙잡으면 RSS 상한은 여전히 보장하지 않는다.

성공 update 뒤에만 cursor·step·실제 `samples_seen`을 함께 증가시킨다. 마지막 작은 batch를 sample 수에 반영한다. coverage는 완료 epoch 수·현재 epoch 방문 sample 수·명시 catalog sample 수를 표시하며 archive 전체 coverage는 `not_measured`다. AMP gradient overflow로 scaler가 update를 건너뛰면 scaler/RNG/attempt는 현재 상태를 보존하되 optimizer 성공 step·scheduler·cursor·samples_seen은 진행하지 않는다. 실패한 loss/gradient/optimizer·중단 중 부분 상태는 checkpoint로 저장할 수 없다.

새 checkpoint는 model/optimizer/scheduler/scaler, Python·CPU/CUDA Torch RNG, cursor, 실제 sample/step/attempt/skip 수, 전체 catalog/config, code hashes, 실행 환경을 포함한다. resume는 catalog/config/code/environment와 cursor/step/sample 수, model dtype/shape/finite, Adam moment/step, scheduler 위치를 확인한다. 복원은 별도 객체에서 먼저 검증하며 오류가 나면 현재 모델·optimizer·RNG를 일부 바꾸지 않는다. 재개 임시 객체는 모델/optimizer 메모리를 추가 사용하므로 GPU pilot 예산에 포함한다. 같은 코드와 환경의 정확 재개를 지원하며 GPU checkpoint를 CPU에서 이어 학습하는 변경은 이 계약 밖이다.

CLI는 `--steps` 또는 `--epochs`로 재개 포함 총 목표를 명시한다. 기본 실행당 상한은1000attempt·300초이며 AMP skip도 attempt에 포함한다. 명시한 `--max-attempts`, `--max-seconds`로 제한하고 전자는 성공 update 목표와 구분한다. 시간 검사는 preflight shard, validation batch, update 전, 저장·export 직전의 cooperative 검사다. 단일 JSON 파싱·tensor 생성·GPU kernel·파일 저장 중 강제 시간 제한은 제공하지 않는다. 시간/시도 상한에 도달하면 `bounded_stop`이며 요청 목표 완료와 구분한다. 상한 중단 때 전체 최종 validation을 새로 시작하지 않는다.

기본 매 성공 update마다 checkpoint를 저장한다. `--checkpoint-every`로 변경할 수 있으며 저장 간격 중 중단되면 마지막 저장 이후 작업은 잃는다. 예외/KeyboardInterrupt 때 실패한 현재 모델로 `last.pt`를 덮지 않고 `report.json`에 실패 종류·원인과 마지막 durable 파일의 hash/위치를 기록한다. checkpoint는 atomic replace하며 report는 그 뒤에 저장한다. 파일 저장 중 디스크 자체 손상·전원 장애의 내구성까지 보장하지는 않는다. 기존 output을 재사용하려면 `--resume` 파일과 해당 output의 `last.pt`가 같은 SHA여야 한다. 다른 지점에서 분기하려면 새 output을 사용한다. identity 거부 시 기존 report/checkpoint도 보존한다.

validation은 train sample과 분리해 `distance_sum_mm`, `squared_sum_mm2`, `valid_joints`를 모든 batch/shard에 누적한 후 관절 수로 나눈다. 작은 batch 평균을 다시 평균하지 않는다. 모델 mode와 Python·Torch RNG를 평가 전후 보존한다. test는 `--split test --final-test`를 명시할 때만 평가하고 train CLI는 test를 자동 실행하지 않는다. 평가 output은 기존 파일을 덮지 않으므로 checkpoint/catalog/payload를 결과로 잘못 덮는 것도 거부한다. train/evaluate의 새 trainer는 각각 전체 로컬 payload preflight를 수행한다.

보고서 log는 기본 최근64event만 보관하며 `--log-limit`으로 제한한다. 각 batch의 sample ID는 앞8개까지만 저장한다. 전체 학습 step log를 메모리나 JSON list에 무제한 쌓지 않는다. 재개 실행의 report는 이번 실행을 기록하며 이전 실행의 무제한 history를 이어 붙이지 않는다. 원본 보고서를 영구 보존하려면 실행 전에 별도 경로에 보관한다.

`export`는 checkpoint만 읽고 model state를 복원하므로 원본 shard가 없어도 가능하다. 현재 모델 코드 hash, catalog/input contract/model shape를 대조한 뒤 원본 checkpoint 파일 SHA를 weights에 기록한다. training optimizer를 재개하거나 전체 dataset을 다시 읽지 않는다. train의 `--export-weights`도 저장 완료된 checkpoint를 다시 읽어 그 내용에서 export한다. UI는 이 고정 추론 파일을 사용하고 학습 재개 파일은 따로 유지한다. GPU 훈련 checkpoint를 CPU 추론용으로 export할 수 있는 경계이며 실제 GPU checkpoint export 시험은 남아 있다.

검토용 실행 예시는 보존된 합성 fixture를 사용한다. 이미 존재하는 output에는 새 이름을 쓰거나 명시한 resume를 사용한다.

```powershell
python -m rewind_m1.shard_cli train --config outputs/full_training_readiness_20261102/developer/trainer/config.json --catalog outputs/full_training_readiness_20261102/developer/fixture/catalog.json --output outputs/m1_shard_example --steps 2 --max-attempts 8 --max-seconds 120
python -m rewind_m1.shard_cli train --config outputs/full_training_readiness_20261102/developer/trainer/config.json --catalog outputs/full_training_readiness_20261102/developer/fixture/catalog.json --output outputs/m1_shard_example --steps 5 --resume outputs/m1_shard_example/last.pt --export-weights outputs/m1_shard_example/weights.pt --max-attempts 8 --max-seconds 120
python -m rewind_m1.shard_cli evaluate --checkpoint outputs/m1_shard_example/last.pt --catalog outputs/full_training_readiness_20261102/developer/fixture/catalog.json --output outputs/m1_shard_example/evaluation.json
python -m rewind_m1.shard_cli export --checkpoint outputs/m1_shard_example/last.pt --output outputs/m1_shard_example/separate_weights.pt
```

2차 개발 근거는 `outputs/full_training_readiness_20261102/developer/trainer/`에 별도로 보관한다. `verify.py`·`tests.json`·`tests.txt`·`code_hashes.json`과 `commands.json`·`cli_*.txt`·`run/last.pt`·추론weights·평가/report를 확인한다. 이번에는 합성 CPU optimizer update가 포함되며 1차의 update0 검증과 구분한다. 실제 데이터 학습/원격 요청/설치는 수행하지 않았다.

고정31개(신규trainer14개+기반17개)가 failure/error/skip0으로 통과했다. 최종 검사에서 합성 성공 update84개와 mock AMP skip1회를 확인했고 별도 보존 CLI 실행은2→5step 재개 총5update·실제sample22개·완료1epoch/현재10개를 기록했다. 두 모델 모두 shard 중간·shard 경계·epoch 경계 분할재개에서 model/optimizer/scheduler/scaler/RNG/cursor/counter 전체가 중단없는 실행과 동일했다. Root도 신규14개를 별도로 통과했으며 근거는 `outputs/full_training_readiness_20261102/root/stage2_tests.txt`, `stage2_review.json`이다.

## 3a 구현: raw 준비 queue

[제한된 준비 queue](PREPARATION_QUEUE.md)를 추가했다. 로컬 합성13개 검사와 보존 실행에서 서로 다른 left/right/pose 원본을 고정 순서로 준비하고, partial→새 attempt retry→같은 index 복원/계속→완료본 재사용→소비 표시를 확인했다. 별도 보존 결과는 source read5856bytes·returned decode12756bytes·신규member12·HTTP/body0·phase 누적7.345초이며 정확한 stereo/pose 키1개를 대조했다. 준비 결과는 `train_ready=false`이고 신경망 학습에 투입하지 않았다.

전역 예산은 실행 전에 예약하고 정상 계측만 정산한다. crash와 미계측 예외는 예약을 유지하며 자동 재시도를 차단한다. 기존 backend의 read/decode/member/HTTP/body/time 상한도 실제 남은 예산 이하로 제한한다. 별도 init 없이 state를 재생성하지 않고 기존 partial/완료 출력·원본·snapshot을 보존한다. 자동 eviction은 없으며 disk 상한에서 중단한다. 테스트의 Drive 항목은 상한 전달 mock이고 실제 원격 요청은 없다.

이는 기존 소량 상한 안의 raw 준비 기능이다. 전체 RGB/pose를 자동 순환하거나 학습 shard를 만들어 공급하지는 않는다. 다음3b는 pose 전체를 한번에 올리는 기존 `LocalRGBSupervision`의 한계를 피하도록 RGB+pose shard와 동일 epoch cursor를 활용하는 별도2D 학습/재개 경로를 설계한다. 실제 release에 필요한 독립 validation·원본 motion·단위/좌표/정합 근거와 Colab 자원 측정은 여전히 별도다.

## 3b1 구현: 독립 frame RGB+GT chunk 공급

[RGB·GT shard 계약](RGB_GT_SHARDS.md)을 추가했다. 같은 원본 sequence의 범위가 겹치지 않는 chunk를 허용하되 motion/split·dataset identity·native 크기/GT 계약을 고정한다. pose256record/4MiB와 RGB 선언 상한, catalog4MiB·최대4096chunk를 지키고 build는 각 descriptor의 byte 수를 누적해 append 전에 중단한다. 기존 `LocalRGBSupervision`의 전체 pose materialization은 현재 chunk 안으로 제한하고 기존 API·실데이터 gate는 바꾸지 않는다.

개발20개(신규11+frame9)와 root 독립11개가 failure/error/skip0으로 통과했다. train 동일seq 두chunk4frame과 별도validation2frame의 native RGB/GTxy/annotation mask가 기존 경로와 같고 전체PNG6sample을 확인했다. metadata 선언 검사와 full PNG hash/decode 결과를 별도로 저장하며 학습/원격/설치는 수행하지 않았다. 공급 유틸이므로 readiness는 false다.

다음3b2는 고정 학습해상도·명시 resize·native 좌표 복원과 작은2D CNN의 학습/정확재개를 연결한다. 이 2D chunk와 기존M1 3D featurecatalog의 완결sequence 제약은 다르므로 cache/join을 shard별로 만들더라도 3D catalog에 자동 결합 완료로 보고하지 않는다. T1 chunk 또는 temporal context/overlap 계약 확장과 실제 release/Colab pilot을 후속 검토한다.
