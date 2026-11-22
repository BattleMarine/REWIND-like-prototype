# v2 128 A 최종 모델과 전체 캐시

이 묶음은 seed 42의 128×128 GPU 학습을 197,961 업데이트까지 완료한 결과다. 선택 모델은 frontend epoch 3, body epoch 3, hand epoch 2다. 최종 모델 복구 ZIP에는 이 가중치와 마지막 학습 상태가 들어 있다. 기준 학습 소스는 `b7a2d00f3bc755808a451ed4ead873e6513d094b`, 동결 ZIP의 베이스는 `ed981245b57d46535abb096ba6a280c8c333ead3`다. 저장소 접근 없이 동봉한 소스를 실행에 사용할 수 있다. [프로젝트 저장소](https://github.com/BattleMarine/REWIND-like-prototype)

## 들어 있는 파일

| 경로 | 내용 |
| --- | --- |
| `reports/cache/v2_cpu_cache_041837Z/` | 첫 캐시 샤드 실물과 원래 카탈로그 |
| `reports/cache/v2_bulk_cache_041837Z/` | 나머지 샤드 ZIP 1,062개와 경계 JSON 1,063개. 두 폴더 합계 1,063샤드, 136,011페어 |
| `reports/v2_gpu_training_128_v1/A-seed-42/` | 최종 모델 복구 ZIP, step-000009 영수증, 최종 진단, 공식 비교 ZIP·영수증 및 선택된 frontend의 예측 특징 147개 그룹 ZIP과 147개 marker. 특징은 3,518개의 train·validation 클립을 덮는다. |
| `inputs/`, `source/`, `provenance/` | 고정 manifest ZIP·spec·공식 split, 동결 소스 ZIP·원본 영수증, Git URL·revision과 출처 해시 |
| `demo/` | 세 시나리오의 실제 예측 GIF, 좌표·mask NPZ, 공식 비교 점수 JSON |
| `v2_bundle.py`, `tools/`, `BUNDLE_INDEX.json` | 오프라인 실행기, 예측·튜닝 코드, 모든 파일의 크기와 SHA-256 |

## 프로젝트 없이 예측 재현

Python 3.11 이상과 해당 환경의 PyTorch, NumPy, Pillow가 설치돼 있어야 한다. 원래 결과의 런타임은 Python 3.13.16, PyTorch 2.14.1+cu130, Pillow 11.3.0, A100 FP32였다. 아래 명령은 CPU나 로컬 PyTorch에서 사용 가능한 CUDA로 실행된다. 작업 폴더에는 최소 4 GiB 여유를 확보한다. 작업·출력 경로는 새 폴더를 지정하며 묶음은 읽기 전용으로 사용한다.

ZIP 전체를 풀지 않고 실행기를 꺼내는 예시다. 먼저 전달된 외부 ZIP SHA-256을 확인한다.

```text
python -c "from zipfile import ZipFile; z=ZipFile('v2_128_A_final_model_and_cache.zip'); z.extract('v2_bundle.py'); z.extract('GUIDE_KO.md'); z.close()"
python v2_bundle.py predict --bundle v2_128_A_final_model_and_cache.zip --work predict-work --output predictions --device cpu
```

기본 실행은 공식 validation의 세 50프레임 클립을 고정 모델과 샘플링 seed 71로 다시 예측한다. `--scenario CityPark_Showcase` 한 개 또는 `--sequence <정확한-manifest-key> --start <프레임>`으로 공식 validation의 고정 50프레임 클립 하나를 고를 수 있다. 결과에는 예측 2D/3D·GT/mask NPZ, PNG·GIF, `prediction-report.json`이 들어 있다. 알려진 클립은 같은 파일의 공식 예측과 숫자를 대조한다. 장치·라이브러리 차이가 있으면 수치 차이를 기록한다. 압축을 이미 풀었다면 `--bundle`에 `BUNDLE_INDEX.json`이 들어 있는 폴더를 전달한다.

## 가중치에서 소규모 튜닝 시작

아래 명령은 선택된 hand 가중치와 새 optimizer에서 독립적으로 시작한다. 세 장면의 고정 train 표본으로 업데이트하고 같은 validation 표본에서 튜닝 전후를 평가한다.

```text
python v2_bundle.py tune --bundle v2_128_A_final_model_and_cache.zip --work tune-work --output tune-hand --stage hand --train-items 6 --validation-items 3 --updates 2 --device cpu
```

`--stage frontend|body|hand`, `--learning-rate`, `--seed`, `--updates`로 변경한다. `--train-items`와 `--validation-items`는 frontend에서는 프레임 수, body/hand에서는 클립 수다. `warmstart.pt`와 `warmstart.json`은 별도 실험 결과다. body/hand 튜닝은 선택된 상위 모델을 고정한다. Frontend 가중치를 바꿨으면 동봉한 예측 특징과 상위 모델의 결합을 새 조건으로 검증하고 2D 특징을 재생성한다. Body를 바꿨으면 그 body에 의존하는 hand의 검증도 다시 수행한다. 이 CLI가 원본의 최종 모델이나 공식 전체 학습 결과를 교체하지 않는다.

튜닝 출력 폴더를 `--warmstart`로 지정하면 해당 단계의 가중치 한 세트를 선택해 다시 추론한다. 보고서와 체크포인트의 부모 실행·stage·상위 모델 해시·설정·가중치 SHA가 모두 일치해야 한다.

```text
python v2_bundle.py predict --bundle v2_128_A_final_model_and_cache.zip --work tune-predict-work --output tuned-hand-predictions --warmstart tune-hand --scenario CityPark_Showcase --device cpu
```

Frontend 교체 시 선택 클립의 2D 특징을 새 모델로 다시 만들고, body 교체 시 hand의 예측 상체에는 새 body 출력과 그 해시를 사용한다. 이 두 경우 downstream 가중치는 기존 모델에서 가져오므로 새 조합의 실험 결과로 해석한다. `prediction-report.json`은 역할별 적용 해시와 원본 예측 대비 차이를 기록한다.

## 전체 공식 데이터로 재학습

`v2_bundle.py train-full`은 공식 train 119,586페어(3,097클립)와 validation 16,425페어(421클립)를 사용한다. 아래는 동봉된 최종 선택 가중치에서 단계별 새 optimizer로 시작하는 예다. 새 모델을 처음부터 만들려면 `--initialize fresh`를 지정한다. `--frontend-epochs`, `--body-epochs`, `--hand-epochs`는 각각 1–100이 허용된다. 모델 옵션과 입력 계약은 동결 128 소스 기준이다.

```text
python v2_bundle.py train-full --bundle v2_128_A_final_model_and_cache.zip --work full-work-01 --output full-fit --initialize best --frontend-epochs 4 --body-epochs 4 --hand-epochs 4 --seed 42 --device cuda --seconds 28800
```

순서는 frontend 학습·공식 validation·best 선택 → best frontend 동결 및 전체 2D 특징 생성 → body 학습·공식 validation·best 선택 → hand 학습·공식 validation이다. hand는 학습 시 GT 상체, 검증 시 이번 실행에서 선택한 body의 예측 상체를 쓴다. frontend batch 2, body/hand batch 1로 고정하며 계획 업데이트는 `59,793 × frontend epochs + 3,097 × (body epochs + hand epochs)`다. `--initialize best`는 역할별 시작 가중치를 동봉된 모델에서 검증해 가져온다. 위 명령은 튜닝 결과 폴더를 자동으로 선택하지 않는다.

첫 호출에는 입력 ZIP 공간 외에 압축 해제된 전체 이미지 캐시 약 50 GiB와 출력·특징 여유가 필요하다. 기본 `--run-gib 8` 기준으로 작업 시작 시 출력 볼륨의 약 61 GiB 가용 공간을 확인한다. 실제 요구량은 캐시 크기와 디스크 상태에 따라 계산된다. `full-fit/frame-cache`에 필요한 샤드를 순차적으로 검증·복원해 다음 호출에서 재사용한다. CPU도 선택할 수 있으며 전체 학습의 처리량과 GPU 비용은 환경별 실측이 필요하다.

`full-fit/run/run.json`에서 `last_checkpoint.sha256`을 확인한다. 한 호출의 시간 상한은 28,800초이며 개발 점검은 `--stop-after-updates N`으로 제한할 수 있다. 다음 호출은 같은 `--output`과 동일한 초기화·seed·epoch·저장 설정, 새 `--work`, 그리고 `--resume-sha256 <last_checkpoint.sha256>`을 사용한다. 정상 저장점에서 모델·optimizer·scheduler·scaler·RNG·cursor를 복원한다. epoch 목표 자체를 바꿀 때는 새 실험으로 실행한다. `run/`, `frame-cache/`, `features/`, `validation/`, `experiment.json`은 같은 재개 작업 폴더에 보존한다.

### Diffusion·loss·sampling 수정 지점

실행기로 동결 소스를 검증하고 편집용 복제본을 만든다. 아래의 경로는 `editable-diffusion/`을 기준으로 한다. 원본 manifest·spec·캐시 생산 binding과 기존 모델·영수증은 그대로 보존한다.

```text
python v2_bundle.py fork-source --bundle v2_128_A_final_model_and_cache.zip --output editable-diffusion
```

`editable-diffusion/rewind_m2/diffusion.py`의 `DiffusionSchedule.q_sample`이나 `sample`처럼 아래 표의 모델 파일을 수정한다. 복제본의 그 파일들과 `configs/rewind_v12_full_training.json`을 편집할 수 있다. 파일을 추가하거나 데이터·영수증·캐시 검증 코드를 바꾸면 이 진입점은 실행을 중단한다. Git 메타데이터와 Python bytecode는 제외한다.

```text
python v2_bundle.py train-full --bundle v2_128_A_final_model_and_cache.zip --editable-source editable-diffusion --work edit-work-01 --output edit-fit --initialize fresh --frontend-epochs 1 --body-epochs 1 --hand-epochs 1 --device cuda
```

각 호출은 동결 ZIP을 다시 검증하고 복제본의 변경 파일만 실행용 임시 트리에 복사한 뒤 그 트리에서 학습한다. `edit-fit/experiment.json`에는 동결 ZIP SHA, 변경 파일별 SHA와 별도의 소비자 소스 SHA가 기록된다. 재개할 때는 같은 복제본·epoch·초기화·seed를 사용한다. `edit-fit/run/run.json`의 `last_checkpoint.sha256`을 확인해 다음과 같이 실행한다.

```text
python v2_bundle.py train-full --bundle v2_128_A_final_model_and_cache.zip --editable-source editable-diffusion --work edit-work-02 --output edit-fit --initialize fresh --frontend-epochs 1 --body-epochs 1 --hand-epochs 1 --device cuda --resume-sha256 <last_checkpoint.sha256>
```

수정 파일의 한 바이트라도 바뀌면 `portable_full_run_source_changed`로 기존 실험을 재개하기 전에 중단한다. 수정 전 파일을 복원해 재개하거나 새 `--output`에서 처음부터 시작한다. 이 진입점에서 변경 소스는 `--initialize fresh`를 지원한다. 원래 `predict`와 `tune`은 계속 동결 소스에서 동작한다.

| 변경 | 파일과 실제 함수 |
| --- | --- |
| 폭·heads·graph/temporal 층 수·dropout·lookback | `configs/rewind_v12_full_training.json`의 `teacher`. `rewind_v1/v2_training.py::runtime_config`가 이 설정을 읽고 v2 실행 설정과 합친다. |
| Body/hand denoiser, 관절·조건 입력 | `rewind_v1/teacher.py::V1Body`, `V1Hand`, `_temporal`이 v2에서 사용하는 모델을 구성한다. 기본 projection·graph는 `rewind_m2/models.py::BodyDenoiser.__init__`, `BodyDenoiser.forward`, `GraphBlock.forward`에, hand 상체 결합은 `rewind_m3/models.py::HandDenoiser.forward`, `BoundHand.forward`에 있다. |
| Causal attention | `rewind_v1/attention.py::Equation4Block.forward`, `kernel_attention`. `rewind_m2/models.py::TemporalBlock.rotary`가 rotary 위치를 처리한다. |
| Noising과 학습 loss | `rewind_v1/training.py::TeacherRun._update`가 timestep·noise를 뽑고 유효 mask로 정규화한 표적을 구성한다. `rewind_m2/diffusion.py::DiffusionSchedule.q_sample`, `rewind_v1/teacher.py::motion_losses`, `rewind_m2/training.py::position_losses`가 각각 forward noising과 loss를 계산한다. Frontend heatmap은 `rewind_v1/run_stages.py::FrameStage._update`와 `rewind_m1/frontend.py::heatmap_loss`를 사용한다. |
| DDIM 반복·샘플링 seed | `rewind_m2/diffusion.py::DiffusionSchedule.timesteps`, `sample`과 `rewind_v1/training.py::TeacherRun.predict`. 평가 noise는 `scripts/v2_gpu_training_pipeline.py::CommonEvaluation._noise`의 공통 seed와 `rewind_v1/run_stages.py::CascadeStages.evaluate`의 body/hand offset에서 지정한다. |

원래 `--initialize best`와 `predict --warmstart`는 `load_state_dict(strict=True)`를 사용한다. hidden·관절·조건·feature 폭이나 파라미터 키를 바꿨다면 새 실험에서 fresh로 시작하는 것이 기본이다. 가중치 이관이 필요하면 각 키의 shape·dtype과 관절·좌표·조건 의미가 일치하는 항목만 명시적으로 매핑하고, 새로 초기화한 모형에 합쳐 최종 strict load를 수행한다. 복사·누락·재초기화된 키와 부모 SHA를 새 실행에 기록한다. `strict=False`만으로 shape 충돌이 해결되지는 않는다. Optimizer·scheduler·RNG·cursor의 원래 재개는 소스·설정·shape가 같은 체크포인트에만 적용한다. Frontend, 카메라, joint layout이 달라지면 관련 2D 특징과 downstream 모델을 별도로 재검증한다.

## 단일 ZIP64 생성과 비용

Colab은 기본 OFF인 내보내기 노트북과 발급 도구 ZIP, 기존 동결 소스 ZIP을 사용한다. CPU 런타임에서 실행하면 `reports/exports/v2_128_A_final_model_and_cache.zip`을 직접 쓴다. Windows·POSIX에서는 내보내기 도구 ZIP을 풀고 원래 `reports/cache/`·`reports/v2_gpu_training_128_v1/` 경로를 보존한 로컬 입력을 지정한다.

```text
python export_v2_128_a_bundle.py --project <원본-root> --original-source <동결-128-source.zip> --work <새-로컬-작업폴더> --output <새-결과-경로.zip>
```

생성기는 원본 샤드·모델·특징의 SHA를 읽으면서 확인하고 임시 ZIP64를 순차 기록한다. 임시 파일 전체를 다시 읽어 동일 SHA와 ZIP 목록을 확인한 뒤 최종 이름으로 게시한다. 원본 Drive 파일은 읽기 전용이다. 새 ZIP 예상 크기는 약 51 GiB이며 시작 전 실제 필요한 크기를 계산한다. 원본 읽기·ZIP 쓰기·전체 readback의 합은 약 153 GiB 논리 I/O다. 내보내기 CPU 호출의 상한은 8시간이다. Drive가 숨은 로컬 캐시를 사용하면 VM 임시 디스크가 더 필요할 수 있고, 실제 시간과 Colab/Drive 비용은 환경에 따라 달라진다. 최종 경로와 정확한 크기·SHA는 완료 출력의 `verified-export.json`에서 확인한다.
