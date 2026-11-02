# 단일 프레임 특징 chunk 연결

2026-11-02 RGB chunk별 native 예측 cache→기존 join 결과를 같은 원본 sequence 이름으로 모아 M1 **단일 프레임** regression에 연결했다. 별도 T1 schema/policy를 사용하며 기존 완결 sequence·temporal 정책과 실제 데이터 학습 차단은 유지한다. 전체 데이터셋 학습 준비나 temporal 모델의 실제 데이터 검증 완료를 뜻하지 않는다.

## 명시 계약

| 항목 | 기존 경로 | 새 T1 경로 |
| --- | --- | --- |
| Catalog schema | `m1_feature_shards_v1` | `m1_t1_feature_shards_v1` |
| Sampler policy | `m1_shard_epoch_python_random_v1` | `m1_t1_feature_epoch_python_random_v1` |
| 같은 원본 sequence의 복수 shard | 거부 | motion/split 고정과 disjoint 범위 조건에서 허용 |
| Window | 기존1~256 계약 | literal1만 허용 |
| Trainer/export 모델 | 기존 설정 유지 | `single_frame`/`window_size=1`만 허용 |

기존 `build_catalog`와 기본 CLI는 새 모드로 자동 승격하지 않는다. `build_t1_catalog(paths, base, limits)` 또는 `readiness build-t1`을 명시해야 한다. 원본 sequence/motion 이름을 바꾸거나 자동으로 파일을 분할·정렬하지 않는다. inclusive frame 범위가 겹치면 실제 희소 frame 중복이 없더라도 보수적으로 거부한다. 같은 sequence의 motion/split 변경과 같은 motion의 split 교차도 거부한다.

파일은 기존 `m1_features_v1` 그대로이며, 공통 dataset ID/version·joint/단위/좌표/input contract·예측 frontend fingerprint/preprocessing version을 고정한다. shard별 파일 SHA/크기/record·sequence 범위와 기존 read/record/window tensor 상한을 재사용한다. 카탈로그 생성은 descriptor byte를 누적해 append 전에 크기 상한을 검사한다. 원래 join 문서의 RGB/cache/GT 출처와 모든 record는 수정하지 않고 파일 SHA로 보존한다.

GT2D 진단 특징도 기존 input contract에 따라 읽을 수 있으나 실제 검증한 통합 경로는 고정 RGB 예측2D 입력이다. synthetic과 실제 dataset 구분, 진단과 예측 입력 구분은 유지한다. `dataset_sample`의 검증 문자열만으로 학습 release를 내리지 않는다.

## 재사용한 실행 경로

`ShardDataset`은 한 shard씩 기존 `load_document(window=1)`로 읽는다. frame마다 현재 입력만 사용하고 target/mask·native prediction·confidence를 유지한다. 시간 history나 chunk 간 context를 만들지 않는다. 원본 sequence 전체 coverage·timestamp/temporal-ready는 별도이며 새 readiness도 `temporal_ready=false`, `full_training_ready=false`다.

기존 `ShardTrainer`, CLI, optimizer/scheduler/scaler/RNG·cursor·sample count·독립 validation·durable 실패 보존과 M1 추론 export를 재사용한다. T1 schema에서 temporal 모델 또는 큰 window를 요청하면 trainer와 checkpoint export 모두 거부한다. checkpoint 기본 형식/API는 유지하고 catalog schema/hash·config·distinct sampler policy로 재개 identity를 구분한다. 기존 M1 및 RGB sampler의 policy 문자열은 바꾸지 않는다.

```powershell
python -m rewind_m1.readiness build-t1 --base outputs/full_training_readiness_20261102/developer/t1_chunks/fixture/features --shard joined_0.json --shard joined_1.json --shard joined_2.json --output outputs/full_training_readiness_20261102/developer/t1_chunks/fixture/features/new_catalog.json
python -m rewind_m1.shard_cli train --config configs/m1_t1_synthetic.json --catalog outputs/full_training_readiness_20261102/developer/t1_chunks/fixture/features/catalog.json --output outputs/m1_t1_example --steps 1
python -m rewind_m1.shard_cli train --config configs/m1_t1_synthetic.json --catalog outputs/full_training_readiness_20261102/developer/t1_chunks/fixture/features/catalog.json --output outputs/m1_t1_example --resume outputs/m1_t1_example/last.pt --epochs 2 --export-weights outputs/m1_t1_example/weights.pt
python -m rewind_m1.shard_cli evaluate --checkpoint outputs/m1_t1_example/last.pt --catalog outputs/full_training_readiness_20261102/developer/t1_chunks/fixture/features/catalog.json --output outputs/m1_t1_example/evaluation.json
```

[Config](../configs/m1_t1_synthetic.json)는 작은 합성 검증용이며 [기존 shard 학습 예산/저장 규칙](FULL_TRAINING_READINESS.md)을 따른다. `build-t1`은 준비된 feature 파일만 읽고 PNG나 archive를 추출하지 않는다. `inspect --full-read`는 해당 특징 payload의 전체 검사이며 raw 데이터나 전체 archive 무결성 검증이 아니다.

## 통합 검증과 남은 조건

근거는 `outputs/full_training_readiness_20261102/developer/t1_chunks/`에 보존한다. 고정 `verify.py`는 **현재 코드로** 작은 합성 RGB를 학습해 checkpoint와 resize weights를 만든 후 native cache3개→기존 join3파일→T1 catalog→3D single-frame 분할재개/validation/export를 실행한다. 이전 checkpoint의 source version 검사를 우회하지 않고 이전 산출물을 변경하지 않는다.

`tests.json/txt`, `code_hashes.json`, `commands.json`, `verification_pipeline.json`, `fixture/`, `rgb_run/`, `m1_run/`, `m1_uninterrupted/`를 구분한다. 원본 이름·예측 입력/출처·GT3D/mask 보존, schema/policy 구분, 범위/motion/split/계약 거부, trainer/export의 T1 전용 제한, byte 상한, epoch/tail·정확 재개와 기존 temporal causality/streaming 회귀를 확인한다. 작은 합성 CPU update는 기능 검증이며 실제 데이터 성능 결과가 아니다.

고정52개(신규T1 8·기존M1 trainer14·특징 기반17·RGB 공급11·기존temporal2)가 failure/error/skip0으로 통과했다. 이 회귀는 합성 M1 성공update117개와 mock AMP skip1회를 포함한다. 별도 보존 통합은 RGB4update와 3D 분할재개4update/중단없는4update이며, 3D는2epoch·실제sample8을 기록했다. 두 실행의 전체 학습 state가 같고 export 모델의 예측도 동일했다. Checkpoint 파일 SHA와 직렬화 내부 바이트 동일성이 아니라 model/optimizer/scheduler/scaler/RNG/cursor/counter의 동일성을 대조했다.

Root도 신규T1 8개와 기존M1 trainer14개, 총22개를 독립 실행해 failure/error/skip0으로 확인했다. 근거는 `outputs/full_training_readiness_20261102/root/stage_t1_tests.txt`다.

실제 데이터 release 근거(원본 motion/단위·좌표·시간·RGB 대응·독립 validation)와 Colab Drive/새VM·GPU/AMP·VRAM/준비·학습 비용은 남아 있다. 이후 한 번의 제한된 통합 pilot으로 묶어 [테스트 세션](FULL_TRAINING_READINESS.md)에 인계할 수 있으나, 아직 실제 학습을 허용하거나 장시간 작업을 시작하지 않는다. Temporal 모델의 chunk/context 확장은 이번 범위 밖이다.
