# 실제 RGB→예측2D→3D 정적 sample 진단

> **2026-11-05 이력 구분:** 이 문서는2026-11-02의 고정2sample 진단이다. 후속 [102쌍 진단](REAL_RGB102_DIAGNOSTIC.md)과 [실제 예측·GT 표시](PREDICTION_OVERLAY.md)를 현재 상태와 함께 확인한다. 어떤 결과도 독립 validation이나 완성된 REWIND 성능을 뜻하지 않는다.

2026-11-02 · 보존된 Serena train frame1·10의 실제 stereo RGB2sample로 front-end→예측cache→단일 frame regression 연결을 확인했다. 같은 두 sample의 `train_fit_diagnostic`이며 validation·일반화·temporal·논문 성능 결과를 발급하지 않는다. 일반 candidate와 학습 gate는 유지한다.

## 고정 입력과 실행

[실제 준비 자료](REAL_DATA_NEXT_RUN.md)와 [GT2D 진단](REAL_GT2D_DIAGNOSTIC.md)의 명시적 진단 입구를 재사용한다. Candidate/RGB 파일 pin·공식 train split·spec·단위·raw pose/RGB hash·관절 순서·키·원래 차단조건을 먼저 대조한다. 둘은 같은 sequence의 비연속 정적 sample이며 독립 validation으로 나누지 않는다.

새 진입점은 `python -m rewind_m1.real_rgb_diagnostic`다. 기존 일반 Trainer/data loader·GT 진단 코드를 변경하지 않았다. 새 출력 경로와 `--diagnostic-only`, 두 raw 파일 SHA를 요구한다. 아래 인자는 같은 명령에 전달하며 의존성이 이미 준비된 Python 실행 파일을 사용한다.

```text
python -m rewind_m1.real_rgb_diagnostic
  --prepared outputs/real_stereo_sample_20261102/root/prepared
  --output outputs/real_rgb_diagnostic_20261102/developer/new_run
  --diagnostic-only
  --candidate-sha256 44ec803069ddee642e45832f88a71979c5cf1966bbe2e8574d37bd9742bec0e2
  --rgb-manifest-sha256 392ffe2184a71212acb551a834ccb8a6b51e3540d6b0648722c7ed11e842205a
```

CPU/Python3.12.10·torch2.12.1+cpu·Pillow12.2.0/thread1/float32, 추가 의존성이나 원격 요청 없이 실행했다. NumPy를 요구하지 않고 기존 tensor rawbytes hash를 사용한다. 실행 상한은 front-end100+regression100의 합200 update와300초다. 설정을 바꾸거나 결과를 보고 추가 학습을 반복하지 않았다.

## RGB·GT resize와 mask

기존 `LocalStereoRGB`로1024×1024 좌우 RGB의 실제 bytes/hash/CRC/해상도를 읽는다. RGB float `[0,1]`을 `torch.interpolate(mode="bilinear", align_corners=True)`로64×64에 축소한다. Antialias는False다. GT2D는 양 끝 pixel 중심이 대응하도록 `63/1023`을 곱하고, network 좌표 예측은 `1023/63`으로 원본1024좌표에 복원한다. 원본 RGB manifest를64해상도로 수정하지 않는다.

기존62관절 `StereoHeatmapNet`의 작은 CNN·32×32 heatmap·soft-argmax decoder·`heatmap_loss`를 재사용한다. Gaussian sigma는 기존 기본1.5 grid pixel이다. Seed71, Adam lr0.01, batch2 전체 sample, StepLR50/gamma0.8로100 update를 고정했다. Heatmap MSE는 annotation valid이면서 영상 안에 있는206개 joint-view만 supervise한다. 전체 유효 annotation은248개, 그중 화면 밖은42개다. 화면 밖 여부를 visibility나3D 정답 누락으로 바꾸지 않는다.

## 별도 cache와 예측 전용 regression

출력 `prediction_cache.json`의 최상위 schema는 `real_rgb_diagnostic_cache_v1`이다. `diagnostic_only=True`, `train_ready=False`로 일반 cache reader가 이 wrapper 자체를 거부한다. 내부에는 기존 builder/reader 계약의 `network_cache`(64좌표)와 `native_cache`(1024좌표)를 구분한다. 명시적 transform·원본 manifest 파일/content hash·frame/view별 source member/hash·입력 image tensor hash·weights 파일/state hash·관절 순서·전처리 버전을 포함한다.

재로드할 때 wrapper 자체hash, 고정 입력 manifest, resize, weights 실제 bytes/state/metadata, 두 nested cache 계약·정확한 key 집합·view별 좌표 변환을 대조한다. 저장 weights의 joint_names/data_kind/manifest hash/frontend version/fingerprint도 검사한다. `frontend_weights.pt`는 기존 형식의 추론용 weights-only export이며 optimizer/RNG 재개 파일은 별도로 둔다.

3D 특징은 디스크에서 다시 읽은 cache 예측만 기존 `pack_window`에 전달한다. Window에는 GT3D target field 하나만 있고 GT2D field/candidate feature cache는 없다. Prediction key 누락은 실패하며 GT로 보충하지 않는다. Joint 예측이 invalid이면 xy0은 수치 자리표시자이고 valid flag0을 유지한다. GT3D mask는 계속124개로 별도 유지한다. 고정 cache 아래 사용되지 않는 GT2D 특징을 바꿔도 regression input이 변하지 않음을 확인했다. 원본 입력 파일을 바꾸면 진단 입구 pin 검사가 먼저 거부한다.

예측2D의62×14 특징으로 `[2,1,868]` input, `[2,1,62,3]`mm target을 조립한다. 기존 `SingleFrameRegressor`/masked MSE·Adam/StepLR·진단 train step을 재사용한다. Hidden32/dropout0.1/batch2/seed37/lr0.01/scale1000mm/StepLR50/gamma0.8의100 update다. `target_scale_mm`는 정규화 loss와 출력 배율에만 쓰며 원래 mm 좌표를 다시 변환하지 않는다.

## 결과

실제 산출물은 `outputs/real_rgb_diagnostic_20261102/developer/run/`에 보존한다.

| 같은2sample train-fit | 초기 | 100 update |
| --- | --- | --- |
| Heatmap MSE, 안쪽206 joint-view | `3.38376085e-5` | `3.32889213e-5` |
| Regression normalized MSE | 0.1506957263 | 0.01584224589 |
| Regression MPJPE mm,124 joint | 623.06231099 | 187.98412298 |

원본 pixel 기준2D 오차는 화면 안206개 평균315.451996px, 전체 valid248개 평균345.417908px, 화면 밖42개 평균492.393890px였다. 화면 밖 GT에도 실제 예측 residual을 계산하며0으로 넣지 않는다. 평가 집합의 밖 관절이0개면 평균은null이다. 전체2D 오차가 크고 heatmap loss 감소도 작아서 front-end 정확도가 좋다는 결과가 아니다. 최종3D187.98mm 역시 좋은 정확도나 일반화 개선을 뜻하지 않는다. 실제 입력에서 두 단계의 loss/finite gradient와 저장cache 연결을 확인한 진단이다.

각 단계100/100 update 모두 gradient가 유한하고 nonzero였다. Front-end gradient L2는약`7.93e-8~1.50e-7`, regression은`0.00460~1.57398`이었다. Cache reload 뒤 특징 일치, GT2D-feature 교란 불변성, T=1 단독/순서변경/다른 record 교란 독립성이 통과했고 단독/batch 출력 차이는0mm였다. 같은 sequence가 통계적으로 독립인 원본 motion이라는 주장은 아니다.

총 실행은200 update/runner 내부6.656초였다. 입력 검증·resize·학습·checkpoint/cache 쓰기/읽기·상태 복원을 포함하며 Python/PyTorch 최초 import와 최종 report 쓰기는 별도다. 처리량 benchmark가 아니다. `_preserve_state`로 Python/CPU RNG·thread/deterministic 설정과 기존 환경을 돌려놓는다.

## Checkpoint 범위와 검증

`frontend_050.pt`, `frontend_100.pt`는 `real_rgb_frontend_diagnostic_checkpoint_v1`/`diagnostic_only_frontend`다. `regression_050.pt`, `regression_100.pt`는 `real_predicted2d_diagnostic_checkpoint_v1`/`diagnostic_only_regression`이다. 두 역할을 바꾸거나 일반 M1/GT2D 진단 checkpoint로 읽으면 거부한다. Model/Adam/StepLR/scaler/RNG/step/samples_seen/config·입력 identity·source/environment를 저장하고 regression에는 predicted_2d/cache/weights/resize identity를 명시한다. CPU AMP는False다.

각 단계50시점에서 저장한 파일을 새 trainer에 재로드해 전체 상태가 같은지 확인한 뒤 나머지50 update를 진행했다. 추가 replay update는 없다. **별도 uninterrupted100 실행과 끝 상태를 대조한 검사는 이번 범위에 포함하지 않았다.** 두 stage의50시점 loaded-state equality만 통과했다고 보고한다.

신규8개·이전 실제 GT진단10개·기존 M1 19개, 총37개 로컬 검사가 실패·오류·건너뜀 없이 통과했다. 새 검사는 학습 update를 수행하지 않고 resize/mask/cache hash·source·weights metadata·키·transform 변조 거부, GT fallback 금지, role 분리와 일반 candidate 차단을 확인한다. 실제 로컬 sample과 torch/Pillow가 없는 checkout에서는 해당 실제-input 검사가 skip될 수 있으며 이번에는 모두 실행했다.

`report.json`은 `real_rgb_predicted2d_diagnostic_v1`로 입력·설정·provenance·환경·각 단계 trace·pixel/3D train-fit 지표·resume/독립성·한도를 기록한다. `input_manifest.json`은 resize 입력 증거다. 개발 근거는 `outputs/real_rgb_diagnostic_20261102/developer/{tests.json,tests.txt,verification.json,code_hashes.json}`에 보존한다. 일반 candidate/loader gate, 기존 원본 RGB/pose/cache/index는 유지하며 `train_ready=False`다. 새 의존성/다운로드/원격/Colab 시험·commit/push는 개발 세션에서 수행하지 않았다.

오케스트레이션이 고정 코드를 `outputs/real_rgb_diagnostic_20261102/root/run`에 별도 실행해200 update/약6.0초와 모든 train-fit 지표 일치를 확인했다. 신규8개 회귀도 통과했다. 저장된 front-end weights로 원본 RGB를 같은 규칙으로 다시 축소·추론해 cache의 원해상도 xy/confidence를 전부 대조했고, cache만으로868특징을 별도 조립한 hash와 저장된 regression checkpoint 재추론의 MSE/MPJPE도 일치했다. 근거는 `outputs/real_rgb_diagnostic_20261102/root/root_review.json`이다. 이는 진단 경로의 독립 재현이며 새로운 성능 평가 집합을 추가한 결과가 아니다. 이번 작업은 로컬 검증으로 끝내며 추가 사용자 Colab 시험은 요청하지 않는다.
