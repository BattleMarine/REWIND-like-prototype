# UI 독립 RGB2D shard 학습·재개

2026-11-02 전체 학습 준비 3b2 단계. 작은 `StereoHeatmapNet`을 그대로 사용하고, [RGB·GT chunk](RGB_GT_SHARDS.md)를 한 shard씩 공급받는 별도 trainer/CLI를 추가했다. **합성 fixture만 학습하며 실제 `dataset_sample`은 release 미구현으로 거부한다.** 기존 native weights/cache API, 소량 diagnostic trainer, M1 regression 기본 경로는 유지한다. 논문 규모 모델이나 실제 GPU 성능을 검증한 결과가 아니다.

## Resize와 감독

config에 고정 `training_image_size=[width,height]`를 명시한다(각4~256). native 이미지는 **한 sample씩** 기존 로더로 hash/decode하고 CPU bilinear `align_corners=True`로 축소한 다음 작은 training batch를 만든다. native batch 전체를 먼저 만들지 않는다. GT2D는 각축 `(training_size-1)/(native_size-1)`을 곱하며 범위 밖 좌표를 clip하지 않는다. annotation 누락 mask·visibility·3D GT를 바꾸지 않는다.

모델 forward는 config에 따라 CPU FP32 또는 CUDA AMP 경로를 사용한다. softmax/target heatmap/reduction은 AMP context 밖에서 FP32로 계산한다. 기존 loss와 같이 유효한 화면 안2D만 heatmap을 supervise한다. batch에 유효 inside 감독이 하나도 없으면 **명시적 실패**로 중단하고 cursor를 진행하지 않는다. 화면 밖3D 정답을 삭제하거나 제공 frame을 몰래 건너뛰지 않는다.

할당 전에 native 한 sample의 RGB+GT tensor bytes와 training batch의 `B×2×J×ceil(H/2)×ceil(W/2)` heatmap 요소 수를 제한한다. 기본 config는 native32MiB·heatmap32Mi 요소이며 설정의 최대 허용값은128MiB·32Mi 요소다. batch는1~4다. 이 값은 전체 RSS/VRAM 상한이 아니다. decoder 임시 데이터·logits/probability/target·autograd·optimizer와 재개 검증용 임시 객체의 메모리는 별도로 필요하다. 실제1024 영상과64/256 훈련 해상도의 비용은 Colab pilot에서 측정해야 한다.

## 학습·재개·평가

`RGBTrainer`는 기존 shard trainer의 coverage·cursor/step/sample 수·checkpoint 검증과 원자적 저장을 재사용한다. 공유 코드에는 기본 format을 유지하는 class 상수/sampler factory와 CLI 함수 주입만 최소 확장했다. 기존 M1 trainer 회귀를 별도로 수행한다. 새 checkpoint 형식은 `m1_rgb_shard_checkpoint_v1`이다.

성공 update만 epoch cursor·step·실제 `samples_seen`을 증가시키고 작은 tail batch도 실제 크기로 센다. AMP overflow는 scaler/RNG·attempt를 보존하되 optimizer 성공 step·scheduler·cursor/sample 수를 진행하지 않는다. source/config/catalog·Python/PyTorch/Pillow·device/AMP·resize 크기/옵션·preprocessing 환경과 model/Adam/StepLR/scaler/Python·Torch RNG/cursor를 함께 고정한다. 복원은 별도 객체에서 검증한 후 적용하며 실패한 복원이 현재 상태를 일부 덮지 않는다.

기본 실행당1000attempt·300초·최근64log, 매 update checkpoint를 적용한다. config 외에 `--max-attempts`, `--max-seconds`, `--checkpoint-every`, `--evaluate-every`, `--log-limit`을 명시할 수 있다. 준비/평가 shard·sample 경계와 저장 직전에서 cooperative 시간을 확인하며 개별 decode/kernel/파일 저장을 강제 종료하는 hard deadline은 아니다. 기존 output 재개는 해당 `last.pt`와 같은 파일 SHA만 허용한다. 실패 시 오염될 수 있는 현재 state를 저장하지 않고 마지막 durable checkpoint와 실패 report를 보존한다. 요청한 `--steps`/`--epochs`는 재개 포함 총 목표다.

validation은 heatmap 오차의 **요소 합/개수**를 누적한 후 나누고 작은 batch 평균을 다시 평균하지 않는다. 예측2D는 native 좌표로 복원해 전체 유효 annotation·inside·outside의 거리 합과 개수를 구분한다. validation에 inside가 없는 batch는 그 사실과 개수를 기록하며 heatmap 분모에 넣지 않는다. 유효 native 평가와 모델 출력 finite 검사는 계속한다. 평가 전후 모델 mode와 Python·Torch RNG를 보존한다. test는 `--split test --final-test`를 명시할 때만 사용한다.

## 추론 가중치와 기존 cache/join

`rgb_frontend_artifacts`는 학습·CLI·UI 코드를 import하지 않는다. 새 `m1_rgb_resized_frontend_weights_v1`에는 GT/native input contract, 훈련 크기, frontend version, model fingerprint, 원본 checkpoint SHA, catalog/source identity와 metadata hash를 저장한다. 기존 `m1_frontend_weights_v1`는 변경하지 않으며 resize weights를 기존 native loader에 억지로 넣지 않는다.

전처리 버전은 `rgb_bilinear_ac1_<width>x<height>_native_v1`로 크기와 좌표 복원 방식을 구분한다. `load_weights` API는 기대 catalog/checkpoint/source/contract/preprocessing을 명시적으로 대조한 뒤 eval 모델을 반환한다. 실제 checkpoint 파일 없이도 소비자가 고정한 pins로 로드할 수 있다. export는 원본 데이터가 없어도 저장 checkpoint에서 모델을 복원해 만들며 optimizer를 재개하지 않는다. export/load는 RNG를 보존하고 export는 원래 모델 mode/state를 바꾸지 않는다. 신뢰하는 로컬 artifacts만 사용하며 파일은64MiB로 제한한다.

cache adapter는 한 native sample씩 동일 resize로 추론한 뒤 각축 `(native_size-1)/(training_size-1)`로 좌표를 복원한다. 출력은 기존 `predicted_2d_cache_v1`의 **native `image_sizes`/좌표**, joint 순서, RGB manifest hash, preprocessing version, model fingerprint를 유지한다. 추가 `rgb_shard_provenance`에 학습 checkpoint·catalog·weights 파일/metadata SHA와 source revision을 기록한다. batch당추론이 아니라 전체 chunk당 최대256record/4MiB이고, record를 append하기 전에 출력 byte를 누적 검사한다. mode/RNG·weights를 보존한다.

cache CLI도 pose를 읽지 않고 RGB manifest의 raw 파일 size/SHA/content hash·dataset/native 크기·record/split/frame 범위·PNG 선언 크기와 낮춘 catalog limits를 적용한다. CLI는 검증 pins를 얻기 위해 `--checkpoint`를 받지만 순수 추론 API는 checkpoint 파일을 열지 않는다. 각 chunk cache를 기존 `join_predictions`와 실제 결합해 GT3D·mask 보존을 확인한다.

**기존 M1 3D feature catalog는 같은 sequence의 여러 chunk를 아직 거부한다.** chunk별 cache/join 통과가 전체3D 학습 자동 연결 완료를 뜻하지 않는다. T=1 chunk 또는 temporal context/overlap 경계 확장은 후속이며, timestamp·원본 motion·독립 validation 등 실제 release 근거도 별도다.

## 실행 예시

[합성 config](../configs/rgb_shard_synthetic.json)는 검증용4×4 해상도다. 아래 fixture는 이전 공급 검증에서 보존한 합성 자료이며 새 output을 사용한다.

```powershell
python -m rewind_m1.rgb_training_cli train --config configs/rgb_shard_synthetic.json --catalog outputs/full_training_readiness_20261102/developer/rgb_shards/fixture/catalog.json --output outputs/rgb_training_example --steps 1
python -m rewind_m1.rgb_training_cli train --config configs/rgb_shard_synthetic.json --catalog outputs/full_training_readiness_20261102/developer/rgb_shards/fixture/catalog.json --output outputs/rgb_training_example --resume outputs/rgb_training_example/last.pt --epochs 2 --export-weights outputs/rgb_training_example/weights.pt
python -m rewind_m1.rgb_training_cli evaluate --checkpoint outputs/rgb_training_example/last.pt --catalog outputs/full_training_readiness_20261102/developer/rgb_shards/fixture/catalog.json --output outputs/rgb_training_example/evaluation.json
python -m rewind_m1.rgb_training_cli export --checkpoint outputs/rgb_training_example/last.pt --output outputs/rgb_training_example/offline_weights.pt
python -m rewind_m1.rgb_training_cli cache --checkpoint outputs/rgb_training_example/last.pt --weights outputs/rgb_training_example/weights.pt --catalog outputs/full_training_readiness_20261102/developer/rgb_shards/fixture/catalog.json --shard 0 --output outputs/rgb_training_example/cache_0.json
```

standalone evaluate/cache에도 기본300초 cooperative `--max-seconds`가 적용된다. 평가/cache output과 weights는 기존 파일을 덮지 않는다. report의 `full_training_ready=false`와 synthetic scope를 유지한다.

## 검증 근거

`outputs/full_training_readiness_20261102/developer/rgb_trainer/`에 고정 검사·실행 결과를 보존한다. `verify.py`, `tests.json/txt`, `code_hashes.json`, `commands.json`, `cli_*.txt`, `verification_pipeline.json`, `fixture/`, `run/`의 checkpoint·weights·native cache·joined feature·평가를 구분해 확인한다. 실제 학습/원격 접근/설치를 수행하지 않았으며 작은 합성 CPU update만 개발 회귀로 실행했다. GPU/AMP는 실행 경로와 mock skip 검사 범위이고 실제 GPU 수치·성능·VRAM 검증이 아니다. [전체 학습 준비 상태](FULL_TRAINING_READINESS.md)를 따른다.

고정38개(신규RGB13·기존M1 trainer14·RGB 공급11)가 failure/error/skip0으로 통과했다. 최종 회귀에서 RGB53/M1 84개의 합성 성공 update와 mock AMP skip2회를 실행했다. 별도 보존 CLI는1step에서2epoch까지 재개해 step4·실제sample8을 기록했고, 세 chunk의 native cache/join에서 GT3D·mask가 동일했다. 이 합성 검사는 실제 데이터 정확도나 대규모 학습 성능 결과가 아니다.
