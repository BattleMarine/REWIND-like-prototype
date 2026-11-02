# 합성 RGB→T1 Colab GPU 통합 pilot

2026-11-02 **고정 `20b7aa9`의 실제 Colab 합성 GPU 시험이 통과했다. 아래 준비 절차는 당시 고정 시험의 기록이다.** 오케스트레이션이 최종 커밋·source ZIP·ready notebook의 revision/hash/job ID를 고정해 테스트 세션에 인계하고, 테스트 세션이 사용자 실행을 안내한다. 이 개발 세션은 작은 CPU smoke와 오류·회수 경로만 확인한다.

## 목적과 고정 범위

합성 RGB→작은2D CNN 학습/정확 재개→resize weights→native 예측 cache→기존 join/T1 catalog→3D single-frame 학습/정확 재개/export를 한 worker에서 확인한다. FP32/AMP는 별도 seed/config/output으로 분리하며, 각 stage는 **full4 성공 update 대 prefix1+resume3**의 전체 checkpoint state를 대조한다. loss 일치만으로 재개 성공을 판정하지 않는다.

| 항목 | Colab `gpu` | 로컬 `cpu_smoke` |
| --- | --- | --- |
| 데이터 | train4/validation2, 같은 train sequence의2chunk+별도val sequence/motion | 동일한 key 구조 |
| Native 크기·관절 | 1024×1024·합성62개 | 8×6·합성2개 |
| 훈련 해상도·batch | 256×256·2 | 4×4·2 |
| Precision | CUDA FP32, CUDA AMP 각각 실행 | CPU FP32만 |
| 성공 update 합계 | 32(RGB16+3D16) | 16(RGB8+3D8) |
| 전체 attempt/시간 | 최대96/600초 | 최대48/120초 |

62개 이름과 pose·카메라·motion은 **합성 계산량 profile**이며 실제62관절 정의·정확도·재현 성능을 의미하지 않는다. 작은CNN/MLP 구조를 유지한다. cache 생성과 export 예측 대조는 CPU FP32이며, 두 profile의 서로 다른 seed 결과를 정확도 ablation으로 비교하지 않는다. FP32는 CUDA matmul/cuDNN TF32를 끄고 benchmark=False·결정적 알고리즘을 설정해 기록한다. AMP는 FP16 autocast/GradScaler 경로이며 loss는 기존 FP32다.

GPU가 없거나 조건을 만족하지 않으면 CPU로 fallback하지 않는다. shape·예산을 자동 축소/확대하거나 설치·다운로드·Drive mount·실데이터 release를 수행하지 않는다.

## 자원 예산과 계측

- worker의 RSS 최대4GiB, Torch CUDA allocated/reserved 각각2GiB. GPU free2GiB·host available4GiB·출력 disk free256MiB 미만이면 사전 중단한다.
- worker 산출물120MiB, 외부 로그/보고 여유8MiB, 회수 ZIP128MiB를 합쳐 결과256MiB 안에 둔다. log는 stdout/stderr 각각64KiB, event는최근64개다.
- **source 업로드64MiB/해제128MiB 공간은 위 결과 예산 밖**이다. 기존 Python/Torch/Colab 환경의 공간도 별도다. source와 결과 상한만 합해도 최대448MiB이며 여유 공간이 필요하다.
- RSS는 worker 프로세스의 cooperative 관측 최대값이고 전체 notebook/kernel/OS 메모리나 정확한 순간 peak가 아니다. VRAM은 worker의 Torch allocator high-water이며 driver/context·다른 프로세스의 전체GPU 사용량이 아니다. 실제 GPU 이름/총VRAM/초기free, host RAM·disk, Python/Torch/Pillow/psutil/CUDA·TF32/결정성 설정을 함께 남긴다.
- phase 전후 CUDA synchronize 후 시간을 기록한다. 이는 fixture/학습/checkpoint/평가/cache/export 등 **IO 포함 phase 시간**이며 모델 단독 지연이나 실시간 성능이 아니다. runner wall time도 별도로 기록한다. 내부 자원 clock은 source/environment preflight 뒤 시작하고 notebook supervisor의600초는 worker import까지 포함한다.

sample·attempt·저장·phase 경계의 협력적 검사이므로 개별 kernel/decode/save가 순간 상한을 넘는 것을 선제 차단하지는 못한다. OOM/예산 초과는 실패로 보존하고 테스트 세션이 검토한다. `gpu_available`, `gpu_training_attempted`, `gpu_successful_updates`를 구분하며 GPU가 보인다는 사실만으로 실행 성공을 표시하지 않는다. 실패 report의 마지막 자원 관측은 실패/OOM 이전 값일 수 있다.

## Notebook 사용 순서

[일반 notebook](../notebooks/synthetic_gpu_pilot.ipynb)은 revision/source SHA/job ID가 비어 있다. **그대로 실행하지 않고** 테스트 세션의 고정 ready 사본을 사용한다. 필요한 기존 모듈은 PyTorch≥2.3·Pillow·psutil이다. 누락되면 중단하고 환경 정보를 돌려준다. 별도 환경 준비가 필요할 때만 기존 [M1](../requirements-m1.txt)·[RGB](../requirements-rgb.txt)·[index 환경](../requirements-index.txt)의 의존성을 참고해 테스트 세션에서 검토한다. 이 notebook은 pip나 환경 교체를 자동 실행하지 않는다.

4개의 code cell로 구성한다.

1. **고정 source ZIP 업로드:** byte SHA256·40자리 `SOURCE_REVISION.txt`, 단일 worker 경로, 경로/암호화/symlink/중복/제외파일·해제 상한을 검사한다. 샘플 ZIP은 거부한다. 결과는 `source_upload.json`에 남기며 source 확인 실패도 회수한다. ZIP bytes+source stamp 대조는 Git 이력 인증이 아니다.
2. **환경·자원 확인:** CUDA와 여유 자원을 확인하고 `environment.json`을 저장한다. import/CUDA probe 예외는 `ENV_READY=False`로 고정해 실행을 막는다. source 파일 hash도 source receipt에 기록한다.
3. **1회 worker 실행:** 고유 job 폴더와 `supervision_started.json`을 먼저 생성한다. 최대600초·RSS/디스크·log를 감시하고 사용자 중단/timeout/오류의 finally에서도 child를 kill/wait한다. 같은 cell/job 재실행은 새 학습을 시작하지 않는다. 실패면 예산·설정을 바꾸지 말고 회수한다.
4. **디스크 결과 회수:** 학습 worker나 모델을 import하지 않는다. `REPORT` 변수가 없어도 된다. 필요하면 셀 상단 `RECOVERY_ROOT`에 출력된 정확한 폴더를 입력하며, 명시값→현재 session root→단일 후보 순서로 선택한다. 살아 있는 child가 있으면 먼저 실행 중단/종료를 확인한다. 이 회수 셀만 재실행할 수 있다.

worker가 강제 종료돼 report가 `started/running`에 머물러도 이를 성공으로 추정하지 않는다. 시작 선언·configs/source/environment·마지막 phase·저장 완료된 checkpoint와 supervisor 기록을 회수한다. report 쓰기 오류가 원래 실패를 덮지 않도록 별도 오류로 남긴다. 초기화/자원 검사 실패는 학습 미시도로 구분한다.

## 산출물과 회수 기준

정확한 session 경로는 `/content/rewind_gpu_pilot_<JOB_ID>`다. 회수 ZIP은 그 아래 `synthetic_gpu_pilot_result.zip`, source 확인 실패만 있으면 `source_failure_result.zip`이다. cell이 **정확한 경로·SHA256**을 출력하고 내려받는다. ZIP에는 source 디렉터리를 넣지 않는다.

worker `run/`에는 다음을 보존한다.

- `started.json`, `configs.json`, `source.json`, `environment.json`, `report.json`
- `fixture/`의 합성 PNG·pose·RGB catalog
- `fp32/`, `amp/`별 RGB/M1 full·prefix·resumed checkpoint, validation, inference weights
- profile별 `features/`의 native cache·joined feature·T1 catalog
- source/environment/실패·lastdurable·phase 자원 계측과 supervisor/log

완성된 ZIP은 CRC/파일 hash를 확인해 재사용하며 학습을 다시 호출하지 않는다. 생성 중 실패한 ZIP이나 완료 receipt가 없는 ZIP을 자동으로 완성본으로 채택하지 않는다. 원본·checkpoint·실패 산출물을 삭제하지 않는다. 회수 공간이 부족하면 기존 결과를 보존하고 테스트 세션에 알린다.

성공 조건은 각 precision의 두 stage가 요청한 성공 update를 마치고 model/Adam/StepLR/scaler/Python·CPU/CUDA RNG/cursor/counters/config/catalog/source/environment 전체가 full/resume에서 같고, GT3D/mask·key·native cache/join·export 예측이 유지되며 예산을 만족하는 것이다. AMP overflow는 성공 update나 cursor로 세지 않고 전체 attempt 상한을 적용한다. 정확성 대조가 실패하면 tolerance 완화나 자동 재실행 없이 그 결과를 전달한다.

## 로컬 검증과 인계

로컬 명시 CPU smoke는 다음 형식이며 실제 GPU를 실행하지 않는다.

```powershell
$pilotRevision = (git rev-parse HEAD).Trim()
python -m rewind_m1.gpu_pilot --profile cpu_smoke --job-id local_cpu_smoke --expected-revision $pilotRevision --output outputs/local_gpu_pilot_cpu_smoke
```

개발 근거는 `outputs/full_training_readiness_20261102/developer/gpu_pilot/`의 `verify.py`, `tests.json/txt`, `code_hashes.json`, `verification_pipeline.json`, CPU 성공/예산중단 session과 회수 ZIP에 보존한다. GPU 없는 AMP metadata→CPU export 비교, CUDA 미제공 fallback 거부, 환경 probe 오류, 명시 회수 경로, 샘플 ZIP 거부, 가짜 child 중단/kill도 GPU 없이 검사한다.

최종12개가 failure/error/skip0으로 통과했고 root 독립12개도 통과했다(`root/gpu_pilot_tests.txt`). 별도 보존 CPU smoke는16attempt/16성공 update·자원 clock9.688초·관측 peak RSS287,330,304bytes이며 결과 ZIP140,982bytes를 회수했다. 주입한 예산 중단도 마지막 durable checkpoint와 ZIP37,554bytes를 보존했다. 이는 GPU/AMP 실제 실행 결과가 아니다.

개발 중 초기11검사의 CUDA mock 속성 누락과 초기12검사의 예상과 다른 실패 상태 기록은 `tests_initial_11.*`, `tests_initial_12.*`로 남겼다. 후자의 최초 임시 report가 정리돼 원예외를 확정할 수 없으며 원인을 해결했다고 주장하지 않는다. 단독 재현·최종12개·root 독립12개는 통과했다. 최종 `case_reports.json`은 각 case report를 임시 정리 전에 보관하며, 예산 중단의 실제 `PilotBudgetStop`과 소스/GPU 거부 이유를 구분한다.

최종 커밋 후 오케스트레이션이 source ZIP/ready notebook의 정확한 경로·revision·SHA·job ID를 고정해 테스트 세션 `01a14391-c1b9-7a40-b537-af6b0b43709c`에 인계한다. 디벨로퍼는 준비·로컬 검증까지만 수행하고 실제 Colab 실행을 시작하지 않는다.

**이 합성 GPU pilot으로 해결되지 않는 조건:** Drive queue의 새VM 지속성·전송/복원, 실제 독립 validation·원본 motion identity·단위/좌표/calibration/시간·RGB 대응 근거, 실제 데이터 학습 release, 전체 archive coverage/CRC·전체 데이터/논문 규모 정확도·실시간 성능. 실제 release 차단은 유지한다. [전체 학습 준비 상태](FULL_TRAINING_READINESS.md)를 따른다.

## 2026-11-02 실제 GPU 결과와 독립 검토

2026-11-02 고정 `20b7aa9`의 사용자 Colab 합성 GPU 시험 `synthetic_gpu_20b7aa9_001`이 통과했다. A100-SXM4-40GB에서 FP32/AMP 각각 RGB·M1의 전체 학습 상태 재개 일치, native cache/join의 GT3D·mask 보존, export 예측 일치를 확인했다. 32attempt/32성공 update·AMP skip0, supervisor24.000초·관측 worker RSS2,676,375,552bytes, Torch allocator peak allocated174,359,552/reserved203,423,744bytes로 고정 예산 이내였다. 시간은 IO 포함이며 전체 driver VRAM 계측이 아니다. Root도 결과 ZIP CRC와 checkpoint12개 SHA를 독립 대조했다. 실제 데이터 학습·정확도·Drive queue 새VM 지속성은 미검증이며 `full_training_ready=false`를 유지한다. Drive 영구보관은 미확인이다.

환경은 Python3.13.16, Torch2.11.0+cu130, CUDA13.0, Pillow11.3.0, psutil5.9.5다. source ZIP pin이 일치했고 supervisor/run 모두 `passed`, returncode0·child_alive=false였다. Runner wall18.113초와 supervisor24.000초를 구분한다. 합성 데이터 결과이므로 실제 정확도나 논문 규모 성능으로 해석하지 않는다.

결과는 `outputs/test_synthetic_gpu_20b7aa9_20261102/synthetic_gpu_pilot_result.zip`에 보관했다. 1,800,422bytes, SHA256 `93db55b5dc8293d33258679028c4af246d8ce7dd21d1be9e38ef4805a99c8882`. ZIP CRC와 마지막 durable checkpoint12개의 파일 SHA를 Root가 직접 확인했다. 앞선 CPU 런타임의 CUDA 미제공 검사는 worker 미실행 출력만 전달됐으며 ZIP은 미수령이다. 이후 GPU 시험 결과와 혼합하지 않는다. 같은 시험의 반복 실행을 요구하지 않는다.
