# v2.1.1·v2.1.2 최종 비교 · seed 137·181

보존된 초기 9개 요약으로 기존 CPU 선별을 다시 실행해 v2.1.1과 v2.1.2를 확인했다. 두 방식이 각각 seed 137·181을 A의 동일한 선택 가중치에서 새로 학습한다. 각 방식은 기존 세 결과와 추가 두 결과를 합쳐 최종 5회로 비교한다. [선정 기준](V2_TUNING_DESIGN.md)

## 고정 실행과 재개

| 방식 | 고정본 위치 | Drive 결과 |
| --- | --- | --- |
| v2.1.1 몸 최종 head | `outputs/v2_plan_inputs/v211_final_seeds_137_181_release_20261122` | `reports/v211_body_head128_a_final_v1` |
| v2.1.2 몸 시간축 QKV LoRA | `outputs/v2_plan_inputs/v212_final_seeds_137_181_release_20261122` | `reports/v212_body_lora128_a_final_v1` |

- 승인된 실행에 노트북 하나를 열고 CUDA GPU 한 대를 선택한다. `/content`에 기존 `v2_gpu_training_128_v1.zip`과 해당 발급본의 `v2_tuning_final_tools.zip`을 올린다. `DRIVE_PROJECT`를 확인한 뒤 `RUN_V2_TUNING_FINAL = True`로 바꾼다. 두 노트북 모두 기본 OFF다. `.ipynb`가 방식·seed·결과 경로를 고정한다.
- 각 seed는 3,097회 업데이트, 1epoch, batch 1, Adam 5e-5, FP32·TF32 OFF다. 두 방식을 같은 seed로 비교할 때 clip 순서·teacher timestep·epsilon을 공유한다. 평가 seed 71, DDIM 10, A의 예측 2D와 hand 가중치, trial body의 예측 상체, 공식 validation 421클립·16,425프레임 및 좌우 분해를 유지한다. 기존 `seed-42`, `seed-71`, `seed-103`은 이 실행의 범위에 들어가지 않는다.
- CUDA R2의 초기화와 진단을 사용한다. 마운트 포함 호출당 최대 8시간, 종료 여유 5분, 논리 읽기 8 GiB·쓰기 2 GiB, 로컬 4 GiB 여유를 적용한다. 각 final 경로가 자체 writer·원장을 사용한다. 새 VM에서는 같은 발급본으로 수동 호출한다. 첫 update·50회 경계·마지막 fit과 평가 그룹별 NPZ를 검증하며, 완료 seed와 그룹은 재사용한다. VM 종료를 확인한 뒤 남은 잠금은 이전 owner로 복구한다. [CUDA R2](V2_TUNING_CUDA_STARTUP_R2.md)

기존 T4 실측의 준비 시간과 seed당 학습·평가·게시 평균을 적용하면 head 약 14분 46초, LoRA 약 10분 35초로 두 호출 합계 약 25분이다. 기존 호출의 논리 예약과 체크포인트·평가 크기에서 추산한 출력은 합계 약 184 MB이며, 부모 복원 예약을 포함한 이전 3-seed 호출도 각각 읽기 2.08 GB, 쓰기 1.67 GB 이하였다. 새 seed의 정확한 시간·I/O와 Colab 사용량은 실제 실행에서만 확인할 수 있다. [v2.1.1 T4](V211_BODY_HEAD.md#t4-전체-학습-결과) · [v2.1.2 T4](V212_BODY_LORA.md#t4-검증-결과)

## 결과 회수와 최종 선별

각 final 경로의 두 `summary.json`, `fit-03097.json`과 대응 가중치, `group-00129.json`~`group-00146.json`과 대응 NPZ, 최근 진단·원장을 회수한다. 비교기는 두 발급본의 동일한 `qualified-screening.json`에 들어 있는 초기 9행을 그대로 사용하고, final 발급의 모듈 SHA와 일치하는 추가 `summary.json`만 받는다. 13행을 처리한 뒤 통과 후보의 목표 지표 판정이 없으면 `awaiting_target_metric_verdict`를 반환한다. 실제 판정 결과를 승인한 뒤, 해당 방식 키에 JSON boolean을 지정한 파일을 `--target-verdict`에 전달한다.

```powershell
$p = 'outputs/v2_plan_inputs'
$head = "$p/v211_final_seeds_137_181_release_20261122"
$lora = "$p/v212_final_seeds_137_181_release_20261122"
$summaries = @('v211-seed137-summary.json', 'v211-seed181-summary.json', 'v212-seed137-summary.json', 'v212-seed181-summary.json')
python -m scripts.v2_tuning_final_selection --releases $head $lora --summaries @summaries
```

`$summaries`에는 회수한 파일의 실제 경로를 지정한다. 아직 회수하지 못한 요약을 생략하면 비교기에 빠진 방식·seed가 표시된다. 발급에는 기존 A 반출 ZIP과 원래 9회 GPU 요약·선별 보고서를 사용한다.

## 발급 SHA-256

| 파일 | SHA-256 |
| --- | --- |
| `v211_final_seeds_137_181_cuda.ipynb` | `f2177533912838f26a3701f13852372067990c6282db5a50f45d91751091afbe` |
| v2.1.1 `receipt.json` | `01fd02460f10ace4d6423cb829d234eb2c1e1b3e57bdcd22eb73e3787a9bdca1` |
| `v212_final_seeds_137_181_cuda.ipynb` | `29d6b67fd3fd24e814a24cb7087b40490df495573909d16916cd681bfefdf7d8` |
| v2.1.2 `receipt.json` | `fa8e0e2c85993d3a256fc383c5eb8a4a5af7456741c6317bd1ea49e6dc6fbd89` |
| 공통 `v2_tuning_final_tools.zip` | `bac5daca9597786b38c1a2ef88de6b34e905fa0a6ca4fd289d35e3030e774460` |
| 공통 `qualified-screening.json` | `344ad4f999afb21241ff6bd50daf99874bf46f54403677f254f0cfe25b11af65` |
| 공통 `trial-plan.json` | `03f8c3dea1d825e4870580d4f0e69850d7f1a814cdbb2ac6618549583973e3ef` |

로컬 회귀 23개와 두 신규 고정본의 반복 발급·ZIP CRC·기본 OFF를 확인했다. 동결 소스와 발급 overlay, 실제 A 체크포인트·feature로 각 방식·seed의 첫 업데이트와 실제 검증 클립의 `47 × 62 × 3` 예측을 CPU에서 확인했다. 두 방식에서 같은 seed가 같은 훈련 클립을 선택했다. 실제 초기 9행만 비교기에 넣으면 추가 4행을 기다린다. 출력층 CUDA 시험을 먼저 인계한다. 추가 seed 정확도는 결과 회수 후 기록한다.

두 고정본은 별도 로컬 경로에서 발급한다.

```powershell
$p = 'outputs/v2_plan_inputs'
$inputs = @('--source', "$p/gpu_training_128_20261122/v2_gpu_training_128_v1.zip",
            '--a-bundle', "$p/gpu128_A_portable_release/v2_128_A_final_model_and_cache.zip",
            '--screening-dirs', "$p/v211_r2_t4_results_20261122", "$p/v212_r2_t4_results_20261122", "$p/v213_t4_results_20261122",
            '--screening-report', "$p/v213_t4_results_20261122/screening.json")
python -m scripts.v2_tuning_final_build @inputs --method v2.1.1 --output "$p/v211_final_seeds_137_181_release_20261122"
python -m scripts.v2_tuning_final_build @inputs --method v2.1.2 --output "$p/v212_final_seeds_137_181_release_20261122"
```
