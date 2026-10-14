# M1 학습 기반과 합성 진단 실행

2026-10-14 · M1의 검토 가능한 구현 묶음입니다. 실제 UnrealEgo baseline, EfficientNet front-end와 Colab GPU 검증은 남아 있습니다. 아래 코드·학습 확인을 M0/M1 전체 완료나 정확도 개선으로 해석하지 않습니다.

## 구현 범위

| 경로 | 현재 구현 |
| --- | --- |
| `rewind_m1/data.py` | 명시적 단위·관절·split 계약, stereo 2D 특징과 GT 3D tensor 조립 |
| `rewind_m1/models.py` | 작은 single-frame MLP와 고정 causal window GRU regression |
| `rewind_m1/metrics.py` | Masked normalized MSE, 현재 anchor MPJPE·MSE 집계 |
| `rewind_m1/training.py`, `__main__.py` | 학습·평가 CLI, 전체 학습 상태 checkpoint, 재개·주기 보고서 |
| `rewind_m1/frontend.py` | 작은 Conv heatmap 개발 모델, 버전·weights fingerprint가 있는 예측 cache |
| `rewind_m1/predicted_fixture.py` | Train-only 합성 RGB front-end → 고정 예측 → cache 검증 → regression·평가 |
| [Colab notebook](../notebooks/m1_baseline.ipynb) | 소스 ZIP 업로드, 환경·테스트, 합성 실행·재개, 선택적 결과 export |

Notebook은 실행 진입점이며 핵심 로직은 위 Python 모듈에 있습니다. 원본 archive를 다운로드·결합·추출하거나 RGB 파일을 실제 dataset에서 읽는 기능은 이 묶음에 포함하지 않습니다. 기존 [데이터·window 계약](DATA_CONTRACT.md)을 유지합니다.

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

현재 준비 문서 전체를 메모리에 올려 tensor로 조립합니다. 대형 dataset의 lazy RGB 로더·disk cache 및 실제 RGB/pose 준비 CLI는 후속 구현입니다.

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

EfficientNet 계열 encoder/decoder, body/hand 2D·3D 지표, 실제 RGB 로더·augmentation·대형 cache, full front-end resume, CUDA/AMP·Colab VRAM/시간/준비 비용, 장시간 학습·failure recovery·전체 latency는 남아 있습니다. Camera encoding·Diffusion teacher/student·물리 loss는 이 단계에서 구현하지 않았습니다. [Colab 준비 안내](COLAB_SETUP.md)와 [로드맵](ROADMAP.md)의 완료 기준·D1~D5는 유지하며 M1 전체 완료를 선언하지 않습니다.
