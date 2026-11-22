# v2.1.1·v2.1.2 CUDA 시작 수정 R2

v2.1.1 새 worker 두 개가 환경 기록 전에 `Invalid device argument`로 종료됐다는 테스트 보고를 받았다. 확인된 환경은 Tesla T4, CUDA 사용 가능, visible device 1개, Torch `2.14.1+cu130`이다. 최초 v2.1.1 발급 ZIP은 `e9918ba`, 최초 v2.1.2 ZIP은 `590780b`의 모듈과 일치했다. 두 발급본과 수정 직전 Git HEAD 모두 `reset_peak_memory_stats(cuda:0)`을 속성 조회 전에 호출했다. PyTorch 2.14.1의 해당 Python API는 장치 index를 CUDA allocator의 reset 함수로 넘기며, native allocator는 초기화된 장치 배열 범위를 검사할 때 보고된 문자열을 사용한다. `v211_single_cuda_and_start_time` guard의 예외 문자열은 다르다. 초기화 전 peak reset이 원인으로 추정되며, 원래 worker의 traceback은 회수되지 않았다. [PyTorch 2.14.1 peak API](https://github.com/pytorch/pytorch/blob/v2.14.1/torch/cuda/memory.py#L347-L362) · [PyTorch 2.14.1 allocator](https://github.com/pytorch/pytorch/blob/v2.14.1/c10/cuda/CUDACachingAllocator.cpp#L4693-L4715)

R2 worker는 CUDA 한 대와 index 0을 확인하고, 명시적 `cuda.init()` 후 현재 장치와 속성을 조회한다. 같은 장치에서 실제 FP32 텐서 1개를 할당한 뒤 peak reset을 시도한다. 성공하면 `CUDA worker ready on cuda:0`을 출력한다. Peak reset이 실패하면 진단에 `cuda_peak.reset_failure`를 저장하고 학습 경로를 진행한다. 종료 시 peak 조회가 실패하면 `peak_gpu_bytes`는 `null`이며 `cuda_peak.read_failure`가 원인을 보존한다. CUDA 초기화나 실제 할당에 실패하면 해당 단계에서 중단한다. 로컬 `result.json`에 `phase`, `failure.type`, `failure.message`, `failure.traceback`을 남기고, worker가 실패 코드로 끝나면 노트북 셀에도 표시한다. Drive writer 이전 실패는 노트북에 표시되는 로컬 경로에서 회수한다.

## 변경된 고정본

발급 경로는 각각 `outputs/v2_plan_inputs/v211_body_head_128_A_cuda_startup_r2_20261122`와 `outputs/v2_plan_inputs/v212_body_lora_128_A_cuda_startup_r2_20261122`다. 원본 128 source ZIP과 해당 R2 도구 ZIP을 Colab `/content`에 업로드한다. 두 노트북은 기본 OFF다. 기존 발급본은 각각 `v211_body_head_128_A_release_20261122`, `v212_body_lora_128_A_release_20261122`에 보존되어 있다. [v2.1.1 실행 안내](V211_BODY_HEAD.md) · [v2.1.2 실행 안내](V212_BODY_LORA.md)

| 파일 | SHA-256 |
| --- | --- |
| `v211_body_head_128_A_cuda_r2.ipynb` | `28c0d58f5803d2b1ab1c26bfdd29f0edb1f0ae6d572b18a27809e623019d1c1e` |
| `v211_body_head_tools_r2.zip` | `06e86ad1b0ec1831e642ba51ed4fcec169559f46e0e1660ed366c9df9a63c15a` |
| v2.1.1 `receipt.json` | `72601de9a12b549ea433bcc3c1f10ce60b9db13449b8adb1052b4f5b73d8412f` |
| `v212_body_lora_128_A_cuda_r2.ipynb` | `cd942441d0c22d9b117219ad3f221da73b87e47105403de1624eb6efdc900681` |
| `v212_body_lora_tools_r2.zip` | `4e24753db3a82b07d3a0536d87b559203670c2e27476dd1c8437ea8c3b87648b` |
| v2.1.2 `receipt.json` | `a384b4a1130e1274c2355acfa6defa1e837eb6ce1bf10d23eda898efdf384f9b` |
| 공통 `trial-plan.json` | `03f8c3dea1d825e4870580d4f0e69850d7f1a814cdbb2ac6618549583973e3ef` |

기존 source·부모·manifest·feature 핀, trial plan, seed와 clip 순서·noise, 8시간 호출 및 5분 게시 여유·읽기 8 GiB·쓰기 2 GiB, 평가와 Drive 출력 경로를 유지한다. R2는 실행 모듈의 SHA를 새로 검증하고 진단 `release.executed_modules`에 기록한다. 재개에는 방식별로 허용된 최초 발급본의 전체 입력 digest를 복원한 뒤 확인한다. 기존 원장·fit·validation binding을 그대로 읽으며 신규 원장도 동일한 방식별 재개 identity를 사용한다. GPU 초기화는 writer 획득과 예약보다 앞에 있다.

CPU 로컬 회귀 18개가 통과했다. Cold CUDA 모의 검증은 유효 index와 init → 할당 → reset 순서, reset·조회 실패 후 진행, 실제 할당 실패의 단계·traceback을 다뤘다. 두 R2 패키지를 다시 빌드해 바이트 일치와 ZIP CRC·모듈 SHA를 확인했고, 기존 입력으로 만든 로컬 원장을 읽어 동일 예약과 trial binding을 확인했다. 로컬 Torch는 CPU 전용이며 T4 cold-start 및 실제 Drive 재개는 후속 CUDA 테스트에서 확인한다.
