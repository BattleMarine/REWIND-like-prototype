# M1 학습 기반과 합성 진단 실행

2026-10-14 · M1의 검토 가능한 구현 묶음입니다. 실제 UnrealEgo baseline, EfficientNet front-end와 Colab GPU 검증은 남아 있습니다. 아래 코드·학습 확인을 M0/M1 전체 완료나 정확도 개선으로 해석하지 않습니다.

## 구현 범위

### 독립 합성 fixture와 단일 CPU 통합 진단 준비

`rewind_m1.local_fixture`는 unittest나 테스트 파일 없이 4개 record와 stereo PNG 8개를 생성한다. Train/validation은 별도 sequence/motion이며 각각 연속 2프레임이다. 8×6 영상과 toy 관절 root/tip만 쓰고, 누락 GT2D 1개·화면 밖 GT2D 1개를 포함하되 각 split에는 유효 supervision을 유지한다. `toy_root/toy_tip`은 실제 body/hand 관절 집합이 아니다. Archive metadata는 합성 구조 검사용이며 대응 원본 archive나 실제 추출·좌표/시간 검증의 근거가 아니다. Fixture 전체128KiB, 파일 hash와 manifest 식별을 receipt로 남긴다. Pillow 버전이 다르면 생성 PNG bytes/hash가 달라질 수 있다.

`rewind_m1.diagnostic_smoke --output NEW_OUTPUT --expected-revision FULL_REVISION`은 패키징된 `SOURCE_REVISION.txt`를 기대값과 비교하고 CPU thread1 환경에서 fixture→기존 runner를 1회 실행한다. Frontend2회·regression2회 update와 정확한5단계 완료, validation2 record/2D pair8/3D valid4, finite metrics·coverage·원본 보존·파일 hash를 검사한다. 성공/실패 summary와 fixture·runner 근거를 보관한다. Source stamp 비교는 Git 이력의 진위를 인증하는 것이 아니며 ZIP bytes는 notebook receipt의 고정 SHA-256으로 별도 대조한다.

신규 notebook은 안내와 코드4셀(소스/환경/실행/회수)로 구성한다. Source ZIP64MiB·해제128MiB/5000entry, 단일 child180초·각 로그64KiB를 적용한다. CPU 환경변수로 GPU를 숨기고 OMP/MKL thread1을 지정하며 torch/Pillow 자동 설치·Drive 연결·재실행을 하지 않는다. 로그는 실행 중 제한하고 pump/cleanup 오류·살아 있는 thread는 성공으로 처리하지 않는다. 성공·실패·timeout 모두 마지막 회수 셀을 실행할 수 있으며 ZIP은 임시 파일을 닫고 크기 확인 후 최종 이름으로 교체한다. 결과 ZIP/uncompressed80MiB 초과 시 원본을 보존하고 제외 목록·부분/실패 export 사유를 남긴다. 기존 ZIP은 수정 없이 같은 파일만 다시 내려받는다. 전체 런타임 RSS·업로드/다운로드 시간은 이 제약에 포함되지 않는다.

현재는 로컬 개발검증과 고정 인계 준비 범위다. 실제 Colab에서의 새 통합 진단은 미실행이며 테스트 세션 안내가 필요한 지점에서 중단한다. 기존 고정 로더 notebook·warm192 자료는 변경하지 않는다.

로컬 신규 검사3개에서 테스트 파일을 포함하지 않은 소스 복사본의 worker CLI 전체 연결, source 실패·fixture 덮어쓰기 거부, notebook 코드 구문·모의 업로드/timeout/로그 오류·정리 오류·부분 export·압축 중단과 동일 ZIP 재다운로드를 확인했다. 기존 runner 회귀4개도 통과했다. Google 업로드·실제 Colab 런타임 실행·브라우저 다운로드는 검증하지 않았다.

### 로컬 M1 소량 end-to-end runner

`run_diagnostic` 및 `python -m rewind_m1.local_diagnostic`는 기존 모듈을 frontend 학습→고정 weights 읽기→예측 cache→validation 2D 평가→예측 특징 결합→CPU regression 학습·validation 순서로 호출한다. 새 모델이나 별도 학습 알고리즘은 없다. Train/validation을 사전 확인하며 test를 자동 평가하지 않는다. 입력·관절·출처·실데이터 검증 guard는 기존 API를 따른다. CLI 인자는 [README 실행 안내](../README.md#데이터-없이-m1-학습-기반-실행)에 있다.

새 출력 디렉터리에 `frontend/frontend_weights.pt`, `frontend/report.json`, `predictions.json`, `evaluation_2d.json`, `predicted_features.json`, `regression/last.pt`, `regression/report.json` 및 최상위 `report.json`을 저장한다. Frontend 파일은 추론용 weights이며 regression 파일은 기존 model/optimizer/scheduler/scaler/RNG·step·설정/코드/데이터를 담는 checkpoint다. Runner 자체 재개·부분 실행 재사용·자동 재시도는 지원하지 않는다. Regression checkpoint는 기존 Trainer의 동일 코드·환경 재개 계약을 유지한다.

사전검증 실패는 출력을 만들지 않을 수 있다. 실행이 시작되면 최상위 report는 running/failed/completed와 현재·실패 단계, 단계별 시간, 원본·설정·코드 식별값을 기록한다. `artifacts`는 완료 단계의 파일 SHA-256/크기이며 `partial_artifacts`는 실패 단계의 잔여 파일이다. 보고서 자신의 hash는 포함하지 않는다. 전역 상태 복원과 모든 단계가 성공한 뒤에만 completed를 기록한다. 실패·중단 시 이미 완료된 파일은 보존한다. 디스크 오류 등으로 실패 보고서를 쓸 수 없으면 report 부재나 running 상태와 파일만 남을 수 있으며 completed로 간주하지 않는다.

기존 JSON 각각4MiB/256 record, frontend100 update·1024²원해상도/256관절/32Mi heatmap 요소와 loader 상한을 유지한다. Regression은100 update, window20, hidden128, batch8 이하·AMP false다. Trainer에 넘기기 전에 계산하는 저장 tensor 추정치 `N*T*(69*J+1)`를64MiB 이하로 제한한다. 앞서 수행하는 LocalRGBSupervision 검증도 tensor를 만들며, 이 추정치는 transient tensor·모델/optimizer·전체 RSS 상한이 아니다. Checkpoint32MiB, 기타 파일 각각4MiB, 최상위 report를 제외한 산출물60MiB(최상위4MiB 예약)를 단계 완료 후 검사한다. 저장 중 OS 디스크 hard cap은 아니다.

전체600초는 사전검증 이후 stage/step 전후의 cooperative deadline이다. Frontend 내부120초 제약도 유지한다. 단일 호출 도중 강제 중단이나 전체 wall time/RSS 보장은 없다. 실행 후 성공·실패 모두 Python/CPU RNG, torch thread 수·deterministic/warn_only 및 CUBLAS 환경값을 복원한다. CUDA가 있는 환경에서는 기존 Trainer가 CUDA RNG도 사용하므로 이를 보존·복원하지만 모델은 CPU에서 실행한다. 복원 실패는 완료로 처리하지 않는다.

로컬 합성 신규 검사4개에서 frontend2회·regression2회 업데이트의 전체 연결, GT 불변·단계 provenance/파일 hash, 실패·중단의 완료/부분 산출물 분리, 전역 상태 복원 및 복원 실패 처리와 사전 guard를 확인했다. 관련 2D 평가6개·frontend 학습3개·M1 19개 회귀도 통과했다. 실제 데이터·Colab·GPU 실행은 이번 검증에 포함하지 않았다.

### 고정 예측 2D cache 평가

`evaluate_cache(pose_doc, cache_doc, expected_manifest_sha256, split=..., groups=...)`와 `python -m rewind_m1.evaluate_2d`는 고정 예측을 GT 2D와 비교한다. 기존 `join_predictions`의 전체 sequence/frame 일대일·출처·관절/영상/버전·실데이터 여섯 검증 guard를 재사용한다. Frame 자체가 cache에서 빠지면 계약 오류이며, frame 내부 관절의 false valid_mask/null 좌표는 예측 누락으로 집계한다. Torch는 필요하지만 평가 중 RGB 읽기와 모델 forward는 수행하지 않는다.

Split은 validation/test/train 중 명시하며 train은 `--train-diagnostic`도 필요하다. 그룹 파일은 `{"body": ["root"], "hand": ["tip"]}`처럼 실제 계약의 관절 이름을 명시한다. 해부학적 그룹을 추정하지 않으며 최대 32개 그룹의 중첩을 허용한다. 그룹 결과는 서로 합산하지 않는다. 예약된 `all`은 원래 전체 관절을 각각 한 번 집계한다.

각 그룹의 left/right/combined는 pixel 단위 Euclidean 거리 합과 유효 GT·예측 쌍 개수로 평균을 구한다. Frame/view별 평균을 재평균하지 않는다. GT 누락, 유효 GT에서 예측 누락, GT 화면 밖, 유효 쌍에서 예측 화면 밖을 따로 센다. 화면 밖 GT도 주지표에 포함한다. `coverage=paired_count/gt_valid_count`, `missing_prediction_rate=1-coverage`이며 GT 유효 개수 0이면 둘 다 null이다. Pair가 0이면 mean은 null, sum은 0이다. 평균에는 예측된 쌍만 들어가므로 coverage·count와 함께 해석한다. Visibility는 평가 mask로 사용하지 않는다. GT visibility unknown은 현재 M1 평가 계약에서 의미가 정의되지 않았다는 뜻이며 원본 파일에 다른 정보가 없다는 주장은 아니다. Confidence 임계값·PCK·정렬·스케일 보정은 적용하지 않는다.

```sh
python -m rewind_m1.evaluate_2d --features GT_FEATURES.json --cache PREDICTIONS.json --manifest-sha256 RGB_MANIFEST_SHA256 --split validation --groups GROUPS.json --output NEW_REPORT.json
```

입력과 출력은 기존 4MiB/256 record의 작은 문서 범위를 유지하며 strict JSON을 사용하고 기존 출력을 덮어쓰지 않는다. 결과에는 원본 pose/cache/RGB hash, frontend 식별값, 평가 코드 revision/hash, split, 관절·영상·그룹 및 집계를 기록한다. 실제 정확도 검증이나 M1 전체 완료를 의미하지 않는다.

합성 신규 검사 6개에서 손계산 픽셀 오차·불균등 개수 집계·그룹 중첩·coverage·누락·화면 밖·split·guard·출력 보호와 학습 weights→cache→평가 연결을 확인했다. 기존 결합 4개와 frontend 학습 3개 회귀도 통과했다. 실제 데이터·Colab·GPU 평가는 실행하지 않았다.

### 로컬 frontend CPU 진단 학습

`rewind_m1.frontend_train`은 `LocalRGBSupervision`과 작은 `StereoHeatmapNet`을 연결한다. 명시 train split만 Adam으로 업데이트하고 validation은 최종 고정 weights의 `eval/no_grad` 평가에 사용한다. Validation이 없으면 `not_available`, 유효한 in-frame 관절이 전체 0이면 `not_evaluable`과 null loss를 기록한다. Test는 학습·평가에서 제외한다. Loss는 기존 heatmap 격자의 MSE이며 validation 집계는 in-frame 관절 수로 batch loss를 가중한다. Train batch에 supervise할 관절이 없으면 중단한다.

설정은 seed, steps(1~100), batch_size(1~4), learning_rate(0 초과~0.1), frontend_version을 요구한다. Pose 순서의 순차 batch이며 마지막 짧은 batch 후 처음부터 반복한다. Shuffle·resize 없이 원해상도 최대 1024×1024, 관절 256개 및 stride2 heatmap 최대 32Mi 요소를 모델 실행 전에 검사한다. 기존 RGB·GT 합산 byte 상한도 적용한다. 120초 제한은 준비·코드 식별 이후 학습·평가·export의 검사 지점에서 확인하는 cooperative deadline이다. 단일 연산 강제 중단·준비 시간을 포함한 전체 실행 시간·성공 report 작성 후 시간이나 전체 RSS를 보장하지 않는다.

완료한 새 출력 디렉터리에는 `frontend_weights.pt`와 `report.json`이 생긴다. Weights는 기존 `m1_frontend_weights_v1` 추론 형식으로 예측 cache 생성에 사용할 수 있다. 학습 재개는 지원하지 않으며 중단한 실행은 처음부터 seed로 다시 시작한다. Scheduler/scaler는 사용하지 않아 report에 null로 기록한다. Report는 설정·학습 key/loss·validation 집계·원본 hash·코드 식별·환경·초기/최종 fingerprint를 남긴다. 전체 RGB manifest hash와 실제 train key 집합을 구분한다. CPU RNG는 실행 전후 복원하고 전역 thread/default 설정은 바꾸지 않는다. 실패 시 이번 실행의 산출물만 정리하며 기존 출력은 거부한다.

```sh
python -m rewind_m1.frontend_train --manifest RGB_MANIFEST.json --root RGB_ROOT --features GT_FEATURES.json --manifest-sha256 RGB_MANIFEST_SHA256 --joints root tip --config configs/frontend_local_smoke.json --output NEW_OUTPUT
```

위 관절 이름은 합성 예시이며 입력 문서의 명시 순서를 사용한다. 실제 데이터의 기존 여섯 검증 근거를 갖춘 GT 특징 문서가 필요하며 자동으로 검증 상태를 발급하지 않는다. 이 실행기는 진단용 학습 경로이며 실제 정확도나 장기 학습의 검증을 뜻하지 않는다.

로컬 합성 검사 3개에서 소량 train 업데이트, validation GT 변경에 대한 최종 weights 불변, 학습 모델과 export 일치, weights→cache reader 연결, CLI·진입 guard·실패 cleanup을 확인했다. 기존 supervision 3개·RGB 예측 5개·M1 19개 회귀가 통과했다. 실제 데이터·GPU·Colab 학습은 실행하지 않았다.

2026-10-28 `LocalRGBSupervision(loader, pose_doc, expected_manifest_sha256=..., joint_names=[...], split="train")`은 검증된 GT `m1_features_v1`과 `LocalStereoRGB`를 연결한다. 모든 record의 sequence/frame key와 split을 일대일 대조한 뒤 선택 split의 pose 순서를 유지한다. 생성 시 이미지를 읽지 않으며 `load_batch([0, 1])`에서 CPU `images[B,2,3,H,W]`, `gt_xy[B,2,J,2]`, bool `annotation_valid[B,2,J]`, 복사된 metadata를 반환한다. 이미지 전처리는 `rgb_0_1_v1`이다.

소형 pose 문서는 최대 256 record/4MiB이며 batch는 최대 4개와 기존 RGBLimits 중 더 작은 상한을 적용한다. RGB와 GT·mask의 합산 tensor bytes를 PNG 읽기 전에 검사한다. 원본 GT3D·시간·motion·split은 metadata에 보존하며 원본과 반환 metadata의 수정은 내부 복사본에 영향을 주지 않는다. 누락 GT2D는 false mask와 0 placeholder로 표현하고 화면 밖 GT2D는 annotation_valid를 유지한다. `heatmap_loss`가 별도 in-frame mask를 계산하며 visibility는 supervision에서 unknown으로 표시한다. 기존 실데이터 여섯 검증 guard를 유지하며 candidate·predicted 문서를 거부한다. 자동 shuffle·resize·학습·optimizer는 제공하지 않는다.

로컬 합성 검사 3개에서 lazy 읽기·키 순서·분할 선택·mask·복사본 격리·상한 거부와 heatmap forward/backward의 유한 gradient를 확인했다. 관련 결합 4개·RGB 예측 5개·M1 19개 회귀도 통과했다. 실제 데이터·GPU·Colab·성능 개선은 이번에 검증하지 않았다.

| 경로 | 현재 구현 |
| --- | --- |
| `rewind_m1/data.py` | 명시적 단위·관절·split 계약, stereo 2D 특징과 GT 3D tensor 조립 |
| `rewind_m1/models.py` | 작은 single-frame MLP와 고정 causal window GRU regression |
| `rewind_m1/metrics.py` | Masked normalized MSE, 현재 anchor MPJPE·MSE 집계 |
| `rewind_m1/training.py`, `__main__.py` | 학습·평가 CLI, 전체 학습 상태 checkpoint, 재개·주기 보고서 |
| `rewind_m1/frontend.py` | 작은 Conv heatmap 개발 모델, 버전·weights fingerprint가 있는 예측 cache |
| `rewind_m1/predicted_fixture.py` | Train-only 합성 RGB front-end → 고정 예측 → cache 검증 → regression·평가 |
| `rewind_m1/prediction_join.py` | GT 특징 문서와 예측 cache의 작은 일대일 결합 API·CLI |
| `rewind_m1/rgb_supervision.py` | 로컬 RGB와 GT 2D supervision을 키로 결합하는 소형 lazy API |
| `rewind_m1/local_rgb.py` | 별도 로컬 RGB manifest의 lazy sample/batch 로더. 2026-10-27 구현, 실행 미검증 |
| [Colab notebook](../notebooks/m1_baseline.ipynb) | 소스 ZIP 업로드, 환경·테스트, 합성 실행·재개, 선택적 결과 export |

Notebook은 실행 진입점이며 핵심 로직은 위 Python 모듈에 있습니다. 원본 archive를 다운로드·결합·추출하거나 실제 RGB 파일을 읽는 기능은 초기 합성 묶음에 포함하지 않았습니다. 후속 로컬 RGB 로더는 아래처럼 분리했으며 기존 [데이터·window 계약](DATA_CONTRACT.md)을 유지합니다.

## 2026-10-27 로컬 RGB 로더 — 구현, 실행 미검증

사용자의 테스트 없이 다음 구현을 진행하라는 지시에 따라 추출된 PNG의 `LocalStereoRGB`와 `RGBLimits`를 별도 모듈에 추가했습니다. 생성자는 manifest·경로·파일 metadata만 검사하고, `loader[index]` 또는 명시적인 작은 `load_batch`에서 해당 좌우 PNG만 읽어 CPU float32 RGB0~1의 `2,3,H,W` 또는 `B,2,3,H,W` tensor를 반환하도록 작성했습니다. 모든 view/record는 원래 크기가 같아야 하며 resize·augmentation·전역 이미지 캐시는 없습니다.

별도 `local_stereo_rgb_v1` manifest는 sequence/frame·좌우 view와 파일 크기/hash·출처·split 선언을 보존합니다. Pose candidate나 기존 학습 계약으로 취급하지 않으며 verified/train_ready를 발급하지 않습니다. Pillow를 선택적 [requirements-rgb.txt](../requirements-rgb.txt)에 선언했고 자동 설치는 하지 않았습니다. 상세 규격·상한·실행하지 않은 사용 예시는 [로컬 RGB 로더 안내](LOCAL_RGB_LOADER.md)를 따릅니다.

이번에는 정적 코드·계약·diff 검토만 수행합니다. 로컬 회귀·합성 검사·decoder import/실행·모델 forward·학습·Colab/Drive 요청은 하지 않았고 기능 동작·성능은 미검증입니다. 아래 초기 합성 실행 결과를 새 로더의 검증 결과로 사용하지 않습니다. 실제 front-end와 prediction cache/pose 학습 release 연결은 후속이며 M1 전체 완료가 아닙니다.

## 환경과 의존성

실행 검증 환경은 Windows, Python 3.12.14, PyTorch **2.7.1+cpu**, CPU float32, thread 1입니다. 프로젝트 `.venv`에 설치했고 Codex 공용 Python은 변경하지 않았습니다. PyTorch 외 직접 추가한 학습 패키지는 없습니다. NumPy가 없어 PyTorch 초기화 경고가 나오지만 이 코드는 NumPy 변환을 사용하지 않으며 아래 검증은 통과했습니다.

실행 당시 설치에는 `uv`를 사용했습니다. `--python`에는 사용할 Python 실행파일을 지정합니다. 다음 CPU 설치는 Colab GPU 설치용이 아닙니다.

```sh
uv --cache-dir outputs/uv_cache venv --python <Python-3.12-실행파일> .venv
uv --cache-dir outputs/uv_cache pip install --python .venv/Scripts/python.exe torch==2.7.1 --index-url https://download.pytorch.org/whl/cpu
```

[requirements-m1.txt](../requirements-m1.txt)는 `torch>=2.3`을 선언합니다. `torch.amp.GradScaler`와 `torch.load(weights_only=True)` 등 사용하는 API의 최소 기준이며 모든 해당 버전의 실행을 검증했다는 뜻이 아닙니다. 기존 Colab GPU PyTorch를 자동 교체하지 않고 환경 확인 후 테스트를 먼저 실행합니다. 새 버전/GPU에서 실패하면 해당 실행 근거로 호환성을 검토합니다. [PyTorch 2.3 AMP API](https://docs.pytorch.org/docs/2.3/amp.html), [PyTorch 2.3 load API](https://docs.pytorch.org/docs/2.3/generated/torch.load.html), [공식 설치 안내](https://pytorch.org/get-started/locally/)

아래 명령의 `python`은 PyTorch가 설치된 환경입니다. 이번 Windows 검증에서는 `.venv/Scripts/python.exe`를 사용했습니다. Linux/Colab에서는 해당 런타임 Python을 사용합니다. PyTorch가 없는 기존 stdlib 환경에서는 M1 테스트 19개를 건너뛰므로 그 실행만으로 M1 통과를 주장하지 않습니다.

## 실행 명령

프로젝트 root에서 실행합니다. 새 학습은 비어 있는 output을 지정하고 이미 `last.pt`가 있으면 `--resume`을 사용합니다. `--steps`는 추가 update 수가 아니라 재개를 포함한 **총 update 수**입니다. `--dataset` 생략은 인위적인 GT 2D fixture를 사용합니다.

```sh
python -m unittest discover -s tests -v
python -m rewind_m1 train --config configs/m1_single_synthetic.json --output outputs/m1_single --steps 80 --device cpu
python -m rewind_m1 train --config configs/m1_temporal_synthetic.json --output outputs/m1_temporal --steps 80 --device cpu
python -m rewind_m1 train --config configs/m1_single_synthetic.json --output outputs/m1_single --steps 100 --resume outputs/m1_single/last.pt --device cpu
python -m rewind_m1 evaluate --checkpoint outputs/m1_single/last.pt --split validation --output outputs/m1_single/evaluation.json --device cpu
```

합성 front-end 단독 gradient/cache 검사는 아래와 같습니다. 4개 RGB 입력의 20 update를 실행하며 `predictions.json`, `input_manifest.json`, `frontend_weights.pt`, `report.json`을 생성합니다.

```sh
python -m rewind_m1.frontend --output outputs/m1_frontend_fixture
```

예측 2D를 실제 regression 경로에 연결하는 **합성 코드 검사**는 아래와 같습니다. Train 12개와 validation 12개의 합성 stereo RGB 중 train만 front-end optimizer에 사용합니다. Validation GT로 모델·임계값을 선택하지 않습니다. 모델 고정 후 캐시를 sequence/frame으로 연결하며 GT 3D와 mask 불변성을 검사합니다.

```sh
python -m rewind_m1.predicted_fixture --config configs/m1_single_synthetic.json --output outputs/m1_predicted_chain --frontend-steps 20 --steps 30
python -m rewind_m1 train --config configs/m1_single_synthetic.json --output outputs/m1_predicted_chain --dataset outputs/m1_predicted_chain/features.json --resume outputs/m1_predicted_chain/last.pt --steps 35 --device cpu
python -m rewind_m1 evaluate --checkpoint outputs/m1_predicted_chain/last.pt --dataset outputs/m1_predicted_chain/features.json --output outputs/m1_predicted_chain/evaluation.json --device cpu
```

통합 경로는 `features.json`, `predictions.json`, `input_manifest.json`, `frontend_weights.pt`, `frontend_report.json`과 공통 regression의 `last.pt`, `report.json`을 보존합니다. Front-end 진단 보고서는 학습 재개 보고서와 분리합니다. `frontend_weights.pt`에는 모델 설정·관절 순서·영상 크기·전처리/모델 버전·weights fingerprint가 있으며 **weights-only**입니다. Front-end optimizer·scheduler·RNG 재개 기능은 구현하지 않았습니다. `frontend_checkpoint_sha256`는 정렬된 state tensor 이름·dtype·shape·bytes의 SHA-256이며 `.pt` 파일 자체의 hash와 다릅니다. Regression 재개는 고정된 특징 JSON을 사용하므로 front-end를 다시 학습하지 않습니다.

## 2026-10-28 로컬 RGB 예측 cache 연결

사용자 Colab CPU에서 로컬 RGB loader의 합성 검사 12/12가 통과했다(Python 3.13.18, torch 2.11.0+cpu, Pillow 11.3.0, subprocess 3.748초). 결과 ZIP SHA256 `0aef49e4e985212734c56f57c4d56bb65e5429128df6e93775fa0f2e80183697`, source `afe319796ad54d72bbe3042ac4f0ab40aae4736c`이며 실행 notebook은 미제출이다.

후속 `rewind_m1.local_rgb_predictions`는 로컬 RGB와 고정 frontend weights의 계약을 대조하고 split별 작은 batch로 추론해 기존 `predicted_2d_cache_v1`을 생성한다. 합성 연결 검사 4개가 한 차례 통과했지만 수정 전 재검사에서 기존 로더의 `rgb_file_changed`로 4개 모두 실패했다. Windows 파일 상태 비교를 보완한 후 합성 검사 5개와 기존 M1 회귀 19개가 통과했다. 파일 상태 비교의 관측 결과는 [로컬 RGB 안내](LOCAL_RGB_LOADER.md#2026-10-28-사용자-cpu-결과와-예측-cache-연결)에 구분한다. 실제 RGB/pose 검증, 학습 release 및 정확도는 여전히 미완료다.

### 2026-10-28 예측 cache와 GT 특징 문서 결합

`join_predictions(pose_doc, cache_doc, expected_manifest_sha256)` 및 아래 CLI는 GT 입력의 `m1_features_v1`을 복사해 예측 입력으로 바꾼다. Sequence/frame의 중복·누락·잉여를 거부하고 기존 record 순서, GT 3D·mask·timestamp·motion·split과 검증 근거를 보존한다. Cache의 `record_splits`가 있으면 키별 split을 대조하며 없으면 그 대조는 불가능하다. 기존 prediction 또는 결합 출처가 있는 문서는 재결합하지 않는다. `real_pose_candidate_v1`과 여섯 검증 근거가 없는 `dataset_sample`은 기존 guard로 거부하며 검증 상태를 생성하지 않는다.

호출자가 지정한 RGB manifest hash, 이미지 크기·관절 순서·data_kind를 대조하고 cache의 frontend 버전·weights fingerprint·전처리를 결과 입력 계약에 기록한다. 버전과 fingerprint는 cache 내부의 일관성을 확인할 뿐 별도 인증은 아니다. 결과 manifest의 `prediction_join_provenance`는 RGB manifest, cache 내용, 원본 문서 hash를 기록한다. API와 CLI 모두 각 JSON 4MiB, 최대 256 record를 제한하고 기존 출력을 덮어쓰지 않는다. 로컬 합성 RGB→cache→결합→기존 로더 등 신규 검사 4개와 기존 연결 검사 5개·M1 검사 19개가 통과했다. 실제 데이터 정합성이나 학습을 검증한 것은 아니다.

```sh
python -m rewind_m1.prediction_join --features FEATURES.json --cache CACHE.json --manifest-sha256 RGB_MANIFEST_SHA256 --output NEW_OUTPUT.json
```

## 데이터 입력 계약

CLI는 준비된 `m1_features_v1` JSON만 읽습니다. 원본 pose JSON 디렉터리나 압축 파일은 직접 학습 입력으로 받지 않습니다. 준비 문서는 다음 구조입니다. 아래는 **합성 예시**이며 실제 관절·단위·영상 규격을 확정하지 않습니다.

```json
{
  "schema_version": "m1_features_v1",
  "contract": {
    "joint_names": ["root", "tip"],
    "distance_unit": "mm",
    "coordinate_frame": "device_relative",
    "image_sizes": [[64, 48], [64, 48]],
    "input_source": "gt_2d",
    "data_kind": "synthetic_fixture",
    "frontend_version": null,
    "frontend_checkpoint_sha256": null,
    "preprocessing_version": "synthetic_pixel_v1"
  },
  "manifest": {"id": "synthetic_m1_v1", "dataset_version": "synthetic_fixture_v1"},
  "records": []
}
```

실제 입력에서는 `records`가 비어 있으면 거부합니다. Record마다 `sequence_id`, 원본 `motion_id`, `split`(train/validation/test), 비음수 정수 `frame_index`, 파싱된 `joints`가 필요합니다. `timestamp_raw`는 그대로 보존하며 FPS나 물리적 시간을 만들지 않습니다. 관절의 `camera_left_pts2d`, `camera_right_pts2d`, `camera_left_pts3d`, `camera_right_pts3d`, `device_pts3d`를 M0 규칙으로 검사하며 누락 필드는 mask로 표시합니다. 학습 target은 `device_pts3d`입니다. 관절 이름과 순서는 계약에 명시합니다.

`dataset_sample`에는 `verified` 및 `verification_evidence` 사전이 추가로 필요합니다. **joints, units, coordinates, calibration, split, rgb_pose_alignment** 각각 `true`와 실제 검증 근거의 비어 있지 않은 문자열이 있어야 합니다. 이 guard는 검증했다고 선언한 계약을 검사하며 좌표의 진실성을 자동으로 증명하지 않습니다. 현재 실제 데이터의 여섯 검증을 완료하지 않았으므로 임의로 값을 채워 학습하지 않습니다. 단위는 확인된 mm/cm/m만 허용하고 내부 GT를 mm로 변환합니다. 좌표 회전·카메라 변환·어안 projection과 camera encoding은 구현하지 않았습니다.

Sequence와 원본 motion의 split 중복을 거부합니다. Sequence record는 연속 묶음이고 frame이 증가해야 하며 자동 재정렬하지 않습니다. Gap과 sequence 경계에서는 기존 window 규칙에 따라 history를 초기화합니다. 개별 sequence가 가진 모든 frame/pose의 완전성이나 실제 원본 motion 중복은 준비 단계에서 별도로 감사해야 합니다.

예측 입력은 `input_source=predicted_2d`와 front-end 버전·weights fingerprint를 요구합니다. 각 record의 `prediction`에는 정확한 `joint_names`, 동일한 모델·전처리 버전, 좌우 `coordinates`, `valid_mask`, `confidence`, `visibility`가 있습니다. Missing prediction을 GT 2D로 대체하지 않습니다. `predicted_2d_cache_v1`은 전체 JSON 내용 hash, RGB manifest hash, 영상 크기, 관절 순서, 모델/전처리 버전과 record별 메타데이터를 검증한 뒤 연결합니다. 통합 fixture의 특징 manifest에도 RGB manifest와 prediction cache hash를 기록합니다.

기존 학습 경로는 준비 문서 전체를 메모리에 올려 tensor로 조립합니다. 후속 로컬 lazy RGB 로더는 별도로 구현했으나 실행·기존 학습 연결은 미검증입니다. Disk cache와 실제 RGB/pose 준비·학습 release CLI는 후속 구현입니다.

## 특징·mask·causality

Tensor shape은 `features[N,T,14J]`, `time_valid[N,T]`, `target_mm[N,T,J,3]`, `target_valid[N,T,J]`입니다. 각 관절마다 왼쪽 7개 다음 오른쪽 7개를 묶습니다.

| View별 7개 값 | 의미 |
| --- | --- |
| x, y | `2*x/(width-1)-1`, `2*y/(height-1)-1` 고정 pixel 정규화 |
| valid | 좌표의 수치적 유효성 |
| confidence, confidence_known | Confidence 값과 관측 여부 |
| visibility, visibility_known | Visibility 값과 정답 여부 |

GT confidence·visibility는 unknown이고 0/known=false로 표현합니다. 값 0은 unknown과 다릅니다. 예측 confidence는 heatmap peak probability mass이며 occlusion 정답이 아닙니다. Pad/누락 좌표의 tensor 0은 mask가 false인 자리표시자이고 원본 window의 None을 유효 원점으로 바꾸지 않습니다.

화면 밖 수치 유효 GT 2D는 clipping하지 않으며 3D 정답 mask를 제거하지 않습니다. Heatmap 학습만 영상 밖 관절을 격자 supervision에서 제외하며 경계에 가짜 정답을 만들지 않습니다. 원본 annotation mask와 3D 평가 mask는 유지합니다.

정규화는 명시적 영상 크기와 설정의 고정 target scale만 사용합니다. Sequence 전체·미래·test 통계는 사용하지 않습니다. Temporal 모델은 window마다 hidden을 0으로 시작하고 현재·과거 time_valid 입력에만 GRU 상태를 갱신합니다. Streaming은 고정 window를 재조립해 같은 forward를 호출하는 방식으로 검증했으며 무한 history hidden cache 최적화는 구현하지 않았습니다.

## Loss·평가와 결과의 해석

예측을 설정 `target_scale_mm`(fixture 100 mm)로 복원한 뒤 유효 GT 관절을 **먼저 선택**하여 normalized MSE를 계산합니다. `(prediction_mm-target_mm)/scale_mm`의 제곱을 유효 joint와 xyz 전체로 평균한 무차원 loss입니다. Mask 밖 NaN이 계산에 유입되지 않습니다. GT가 전부 없는 anchor는 Trainer 초기화에서 거부하고 전부 masked loss batch도 update하지 않고 오류를 냅니다.

학습과 평가는 window의 현재 마지막 anchor만 사용합니다. MPJPE는 mm Euclidean distance의 전체 유효 관절 합/관절 수이며 batch 평균의 평균이나 sequence별 동일 가중 평균이 아닙니다. MSE는 mm² 합/(3×유효 관절 수)입니다. 관절 집합은 명시된 계약 전체이고 body/hand별 분리·PA alignment·시간/접촉 loss·실시간 latency 측정은 아직 없습니다. Test 평가는 `--final-test` 명시가 필요하며 model 선택은 validation에서만 수행합니다.

실제 GT 입력이면 `gt_input_diagnostic`, 예측 입력이면 `predicted_input_baseline`으로 결과 종류를 구분합니다. **모든 합성 입력은 우선 `synthetic_code_check`**로 표시합니다. 2관절 fixture의 인위적인 2D–3D 선형 대응은 실제 카메라 projection이나 데이터셋 성능 근거가 아닙니다.

## Checkpoint·중단 재개와 기록

`last.pt`에는 model, Adam optimizer, StepLR scheduler, AMP scaler, Python/Torch CPU·CUDA RNG, 총 update 위치, samples_seen, sampler 정책, config, 입력 계약, 전체 특징 문서 hash, manifest, 환경과 source 정보가 있습니다. Sampler는 Torch RNG 기반 복원추출이며 RNG와 step이 다음 batch를 결정합니다. Checkpoint는 동일 디렉터리의 임시 파일로 저장 후 교체하고 `weights_only=True`로 읽습니다. 로컬에서 생성한 신뢰 가능한 파일만 사용합니다.

`source`는 Git HEAD와 dirty 상태 또는 ZIP의 `SOURCE_REVISION.txt` 40자리 SHA, `rewind_m1`·`rewind_data` Python 파일별 SHA-256을 기록합니다. Git/stamp가 없으면 `unversioned`와 코드 hash를 남깁니다. 실제 실행은 커밋 전 코드이므로 dirty 상태와 파일 hash로 식별하며 코드가 바뀐 중간 checkpoint는 최신 코드에서 정확한 resume/evaluate가 거부될 수 있습니다.

정확한 재개는 config·특징 문서 hash·두 패키지 code hash와 Python/PyTorch/device/CUDA/threads/AMP, CUDA인 경우 GPU 이름·VRAM·capability·cuBLAS 설정이 일치해야 합니다. 매번 변하는 GPU free memory 관측은 별도 metadata라 일치 비교에서 제외합니다. CUDA deterministic 설정은 연산 전 적용하지만 CUDA/AMP의 실제 재개는 미검증입니다. [NVIDIA cuBLAS 재현성 조건](https://docs.nvidia.com/cuda/cublas/index.html#results-reproducibility)

10 update마다 및 종료 시 checkpoint 다음 report를 저장합니다. `report.json`은 초기/현재 지표, 누적 log, 재개 구간 `segments`를 유지하고 원본 resume hash는 덮기 전에 고정합니다. 두 파일 교체는 하나의 transaction이 아니므로 사이에서 중단될 수 있습니다. Checkpoint보다 뒤처진 동일 실험 report는 `report_recovery`와 누락 log 범위를 기록한 뒤 재개합니다. 잃은 loss를 만들어내지 않습니다. Report가 checkpoint보다 앞서거나 다른 config/manifest/code면 거부하며 다른 분기에는 새 output을 사용합니다. 저장 이후 최대 9 update의 작업 손실이 가능하며 이미 저장된 checkpoint 위치에서 재개합니다.

## 실제 실행한 검증

2026-10-14 최신 CPU 검증:

- Python 테스트 **59개**(M0/window 40 + M1 19), 모두 통과. Tensor·mask·단위·split, gradient·작은 과적합, 미래 입력 불변성, window streaming/reset, 예측 cache 출처·변조·GT 대체 거부를 포함합니다.
- Disk checkpoint 6+8 update와 연속 14 update의 model/optimizer/scheduler/scaler/RNG/위치가 일치했습니다. CLI 중단을 checkpoint 저장 직후 주입한 stale report 복구도 연속 학습 model과 일치했습니다.
- GT 합성 single-frame/temporal 각각 80 update를 실행했습니다. Train normalized MSE는 각각 약 `0.120920 → 0.000093285`, `0.225422 → 0.000473713`입니다. 이는 fixture 코드 과적합 확인이며 temporal 성능 우위를 뜻하지 않습니다.
- 작은 frontend의 20 update와 train-only 예측 cache→regression 30 update, 동일 특징 JSON의 35 update까지 재개·validation CLI를 확인했습니다. Validation을 front-end optimizer에 사용하지 않고 cache 추론 중 weights와 GT 3D/mask 불변성을 검사했습니다.
- Notebook JSON/code cell AST·빈 출력 상태를 검사했고 Colab 의존 업로드/Drive 인증 셀을 제외한 실행 셀을 로컬 CPU에서 실행했습니다. 결과 export 함수는 로컬 디스크 복사·파일별 hash 확인과 기존 경로 덮어쓰기 거부를 확인했습니다. **실제 Colab 업로드·Drive 읽기/인증·GPU 실행 검증은 아닙니다.**

개발 산출물은 Git 제외 `outputs/`에 보존합니다. 최신 로컬 notebook 검사 요약은 `outputs/m1_latest_check.json`이고 각 실행은 고유 `m1_synthetic_*` 디렉터리를 사용합니다. 결과·checkpoint·manifest를 raw archive와 분리합니다.

## 다음 진입 조건과 미검증 항목

실제 데이터 small sample의 관절·단위·좌표/calibration·timestamp·split·좌우 RGB/pose 정합성을 근거와 함께 확보한 뒤 명시적 특징 계약을 준비해야 합니다. 이후 실제 GT 입력 진단, 실제 RGB front-end 학습/고정·캐시와 예측 입력 baseline을 순서대로 검증합니다.

EfficientNet 계열 encoder/decoder, body/hand 2D·3D 지표, 로컬 RGB 로더의 실행·실제 front-end 연결, augmentation·대형 cache, full front-end resume, CUDA/AMP·Colab VRAM/시간/준비 비용, 장시간 학습·failure recovery·전체 latency는 남아 있습니다. Camera encoding·Diffusion teacher/student·물리 loss는 이 단계에서 구현하지 않았습니다. [Colab 준비 안내](COLAB_SETUP.md)와 [로드맵](ROADMAP.md)의 완료 기준·D1~D5는 유지하며 M1 전체 완료를 선언하지 않습니다.
