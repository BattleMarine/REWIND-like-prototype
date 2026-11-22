# v2 128 파생 시험 CPU 검사 인계

2026-11-22 기준 설계 `08cc82a`와 고정 A(seed 42, frontend e3·body e3·hand e2)를 사용한다. Colab CPU 검사 세 가지를 한 노트북에서 실행한다. 각 검사는 새 런타임에서 독립 실행할 수 있다. [튜닝 분기·반복 설계](V2_TUNING_DESIGN.md) · [A 원본 가이드](V2_GPU128_A_PORTABLE_BUNDLE.md)

## 지금 실행하는 검사

| JOB_NAMES 값 | 자료와 검사 | 통과 범위 |
| --- | --- | --- |
| `inputs` | 동결 128 소스·원본 영수증·manifest·공식 split과 A 완료 진단의 SHA/ID를 대조한다. 1,063개 캐시 경계와 실물 파일 크기, 147개 A 특징 그룹 marker와 파일 크기를 확인한다. 맨 처음과 마지막 캐시 샤드·특징 그룹의 실물 SHA·텐서·receipt를 검증한다. | train 119,586·validation 16,425페어, test 0·3,097/421클립, 128 입력 shape, proper rotation/mm와 masking·padding 계약. 샘플 밖의 캐시·특징 실물은 크기 대조 범위다. |
| `reference` | 원본 공식 비교 ZIP과 A의 세 50프레임 NPZ 실물 SHA, joint 순서·2D/3D shape·mask·25FPS 명목 시간과 좌우 손목 벡터 분해를 확인한다. 좌표 왕복·단위·어안 축은 독립 계산으로 확인한다. | 저장 validation 150프레임. screen 밖의 유효 2D 표점을 별도 집계한다. 같은 쪽 손목과 손가락이 모두 유효한 표본에서 잔차·내적을 계산한다. |
| `trials` | A 최종 원본 복구 ZIP과 내부 파일 SHA, 완료 cursor, 세 best strict load, frontend fingerprint로 원래 특징 ID 재계산, 실제 body 최종 head·시간축 QKV 후보 파라미터, 저장 optimizer·scheduler·scaler·CPU/CUDA RNG의 CPU 역직렬화를 확인한다. 합성 CPU 연속/재개 결과와 9/13회 trial plan·선정 계약도 확인한다. | 실제 A 아카이브 CPU 검사와 합성 CPU 재개·선정. 원래 CUDA RNG가 포함됐는지 확인하며 실제 GPU 재개는 테스트 세션의 별도 검증 대상이다. |

`reference`에는 v2.1.2 LoRA rank 4·초기 delta 0, v2.1.4 mask 포함 손목 가중 손실 감소 분모, v2.2 클립 단위 GT/예측 상체와 해당 mask 선택, v2.3 현행 DDIM 10/5/2 timestep 및 terminal/time mask, v2.4 forward noising·x0/v/epsilon 왕복을 포함한다. 합성 예제의 검사는 해당 분기의 구현·학습·지연 검증과 별개다.

원본 A 실행 계획 봉인에는 당시 로컬 디스크 여유가 포함돼 있다. `inputs`에는 당시 보고된 SHA, 이번 로컬 SHA, 여유 공간과 그 판정을 뺀 계획 SHA를 각각 남긴다. 분할·시나리오별 개수와 공통 클립은 원본 완료 진단과 직접 대조한다.

## 입력 위치와 고정 근거

작업자는 노트북과 kit만 먼저 받는다. kit에 약 24MiB의 동결 소스 ZIP, A 인덱스, 원본 소스 receipt가 들어 있다. 패키지 빌드 시 로컬 A ZIP에서 이 작은 파일들만 읽는다. Colab에는 A 51GiB ZIP 업로드가 필요 없다.

| 입력 | Drive 프로젝트 아래 상대 경로 |
| --- | --- |
| 원본 A 진단 | `reports/v2_gpu_training_128_v1/A-seed-42/diagnostics/invocation-roh3t3jf.json` |
| 원본 A 영수증·복구 ZIP | 같은 디렉터리의 `checkpoints/step-000009.json`, `checkpoints/result-000009-90fa6ef47b77416fa2a2022ab1c26fc6.zip` |
| 원본 A 비교 | 같은 디렉터리의 `comparison.json`, `comparison-5b9b141f0b724a00b7f00ed0b6ec4ce1.zip` |
| 원본 A 특징 | 같은 디렉터리의 `features/9588afef53e4802089baf7c51235f18068d110d02b1cac8ca6371b8d7aa5e276/` |
| 원본 128 RGB·pose 캐시 | `reports/cache/v2_cpu_cache_041837Z/`와 `reports/cache/v2_bulk_cache_041837Z/` |

소스 ZIP SHA-256 `9a68d8e45bcf3abf6fab3004740faa6805d4ce3fda5118c3a7102011937849d8`. 정확한 동결 ZIP revision과 겉 파일 SHA/크기는 발급 `receipt.json`에서 확인한다. 고정 manifest content SHA `51657741aedb25ddafaf6fbffef706f4465729f787a3fa0fa004d8dcaa65dd9c`, 마지막 cache boundary `dc79219153d69f44020e0c42f9f3ede6cd3add179211f6c300345b2416dab511`, A checkpoint receipt content SHA `7d2cf12cdaf42916d98190cf2d57e11e93a6d94165d5e753db6fc1d11c3026d4`, 복구 ZIP SHA `53209c40cb84ac250bc5078ba415efed8d59061d17724c5cc8e1849cde0fa2cc`. 원본 비교 ZIP과 내부 NPZ SHA는 고정 A 인덱스에 있다.

## Colab 실행과 회수

1. CPU 런타임에서 `v2_tuning_cpu_20261122.ipynb`를 연다. `v2_tuning_cpu_20261122_kit.zip` 1개를 `/content`에 올리고 발급 receipt의 SHA-256과 대조한다. Colab 기본 Python, CPU PyTorch, NumPy, Pillow를 사용한다. 런타임 재생성 시 kit를 다시 올린다.
2. 확인한 `DRIVE_PROJECT`를 설정하고 `RUN_CPU = True`로 변경한다. 한 번에 실행하려면 `JOB_NAMES` 기본값을 유지한다. 독립 실행이나 런타임별 동시 실행은 목록에 맡은 검사 이름만 지정한다. 한 런타임 안에서는 목록 순으로 실행한다.
3. 설정과 실행 셀을 순서대로 실행한다. Drive 마운트와 `MyDrive` 내부 경로, 새 출력 파일 쓰기·readback을 확인한다. 상위 실험 자료는 읽기 전용으로 연다. 새 실행마다 `reports/v2_tuning_cpu_20261122/run-<UUID>/`를 사용하므로 동시 실행끼리 결과를 덮어쓰지 않는다.
4. 노트북 마지막에 표시되는 정확한 Drive 경로와 `invocation-result.json`의 바이트·SHA-256을 회수한다. 각 JOB의 `process-result.json`, `report.json`, `usage.json`, `process.log` 및 `trials/trial-plan.json`의 pin은 상위 receipt를 따른다. `status=passed`와 각 child의 종료 코드 0, 보고서의 근거·검증 범위는 별도로 확인한다.

| 검사 | child 상한 | 프로젝트 읽기 상한 | 시작 시 로컬 scratch 여유 |
| --- | --- | --- | --- |
| inputs | 1,200초 | 512MiB | 2GiB |
| reference | 420초 | 32MiB | 1GiB |
| trials | 1,200초 | 512MiB | 3GiB |

읽기 상한은 프로젝트 파일에서 child가 직접 읽은 payload 기준이다. 경로·크기 조회, kit/동결 소스 검증과 로컬 재독해는 별도다. child는 30초 전부터 검사 중단과 보고서 작성을 시도한다. 전체 실행 시 2,820초 child 예약과 마운트 최대 120초, bootstrap·보고서 복사 시간을 고려한다. 각 보고서 파일은 최대 2MiB, 로그 회수는 끝 64KiB다. 해시·경로·split 불일치, 파일 누락, 공간 부족, 시간/읽기 상한 초과는 해당 검사를 중단하고 상태를 기록한다. 종료 보고서를 잃은 subprocess는 읽기 예약 상한으로 보수적 정산한다. VM 전체 종료 시 `invocation-start.json`만 남을 수 있으며 그 실행은 미회수 상태다. 다시 실행하면 새 UUID에 기록한다.

## GPU 실행 이후에 할 CPU 작업

| 분기 | 선행 조건과 CPU에서 할 작업 |
| --- | --- |
| v2.1.0 | A의 전체 공식 validation(421클립·16,425 실제 프레임) GPU 예측과 그룹별 marker·NPZ 회수가 필요하다. 전체 손목/손가락 마스크·내적·시나리오 재집계, 저장된 A 기준 점수와 대조, 시각화가 CPU 후처리 대상이다. 현재 저장 150프레임은 별도 검증 표본이다. |
| v2.1.1~.3 | 층 선택/LoRA적용, 독립 trial과 원점수·검증 체크포인트를 생성한 후 공통 seed 3회 선별과 5회 최종 비교를 수행한다. 같은 조건의 미완료/실패는 점수에서 제외하고 해당 seed를 대기 상태로 둔다. |
| v2.1.4 | v2.1.1~.3 선정 방식과 동일 seed의 가중치 1 대조가 필요하다. CPU에서 손목·손목 제외 body·hand-global을 비교한다. |
| v2.2 | GT/예측 상체 생산·혼합 학습과 공통 검증 결과가 필요하다. CPU에서 공통 seed, hand-local/global과 상체 생성 비용을 집계한다. |
| v2.3 | 동일 GPU·noise·클립·동기화로 10/5/2 품질 및 측정 5회를 확보한 후 CPU에서 품질/중앙값/범위를 표로 만든다. 허용 정확도 손실의 결정은 별도다. |
| v2.4 | x0/v 양측의 초기화·새 loss 가중치·update 예산, 학습·추론 구현과 결과가 필요하다. CPU에서 수렴 여부·같은 조건의 평가를 정리한다. |

GPU와 독립 test는 이 CPU 노트북의 범위 밖이다. 원본 motion 중복과 실제 timestamp/stereo 동기를 별도로 검증해야 한다. [종합 결과와 분기 상태](EXPERIMENT_RESULTS_REPORT.md)

### v2.1.0 결과 재집계

GPU 실행을 마치고 `reports/v210_eval128_a_v1/`의 원본 `summary.json`, 18개 marker, marker에 기록된 NPZ를 그대로 회수한다. GPU 발급본은 `dcd20c5`의 receipt 파일 SHA-256 `ac766ffff915349fbbc3528337dec3fb3ac0d86152611d8bd9011f92c5eee2dd`로 고정한다. `summary.json`의 파일 SHA-256도 회수한다. 아래 도구는 GPU 발급본에 기록된 버전의 `v210_eval.py`, `v210_metrics.py`와 원본 128 source ZIP을 사용한다. 읽기 전용 프로젝트·결과 디렉터리 밖에 새 JSON을 생성한다.

```text
python -m scripts.v2_tuning_cpu_v210 --project <원본-프로젝트> --results <v210_eval128_a_v1> --source-zip <동결-128-source.zip> --gpu-receipt <GPU-발급-receipt.json> --gpu-receipt-sha256 <파일-SHA> --gpu-summary-sha256 <summary.json-파일-SHA> --output <새-재집계.json>
```

그룹·프레임 순서와 manifest 포함 범위, 원본 모델·feature·code pin, 그룹 NPZ와 marker SHA를 확인하고 원래 CPU 집계 함수로 저장 좌표를 다시 계산한다. A의 원본 완료 진단과 비교 영수증에서 시나리오·전체 hand-local 기준을 함께 확인한다. 그룹 누락이나 원본 비교 조건 불일치 때는 통과 보고서를 작성하지 않는다. 지표는 수치 비교 허용오차 내에서 GPU 결과 요약과 일치해야 한다. 이 도구는 후처리 수치만 재계산한다.

`trials/trial-plan.json`은 이후 v2.1.1~.3 집계를 위한 제안 계약이다. `scripts.v2_tuning_cpu_selection.summarize()`는 같은 plan SHA·공통 조건·seed를 확인하고 표본 표준편차와 각 seed의 부모 대비 차이를 계산한다. 최종 방식 후보는 별도로 확정할 목표 지표 개선 판정을 입력받는다. GPU 실행기의 trial 출력 형식과 연결하는 작업은 해당 분기의 구현이 나온 뒤에 한다.

## 동일 발급본을 로컬에서 생성

```text
python -m scripts.v2_tuning_cpu_build --a-bundle <로컬-A-묶음.zip> --output <새-발급-폴더>
```

생성기는 A ZIP에서 고정 인덱스와 작은 소스·도구 멤버만 읽는다. 발급한 노트북·kit·가이드와 내부 멤버의 크기 및 SHA-256은 발급 폴더의 `receipt.json`에 저장한다.

## Colab reference 결과

2026-11-22 · 고정본 `b68a2d3`의 CPU reference 호출이 종료 코드 0으로 통과했다. 세 시나리오별 보존 클립 하나씩 총 150프레임의 NPZ와 손목 벡터 분해, 합성 입력의 x0/v/epsilon 왕복과 DDIM timestep을 확인했다. 화면 밖 유효 2D 표점 수는 ArchVis 147, CityPark 187, Downtown 1,715다. 이 표본 수로 시나리오 전체의 화면 밖 비율을 추정하지 않는다.

Drive의 `reports/v2_tuning_cpu_20261122/run-a44c960bc16f4938ab42d69abe3806d8`에서 회수했다. 로컬 보존 폴더는 `outputs/v2_plan_inputs/v2_tuning_cpu_reference_a44c960b`다. invocation-result SHA-256은 `eb5e193280e83a5629aeb03694b86ea57eed97867c1f14a191df517d02a1f25d`, process-result는 `64ac51df89d564dc81735df814fe90f347094371ac06433150fe2da7aa057e50`, report는 `212e28a591c3b21bbcdae2c3fc1e6ece74bfc5d76b1e5d973b06e87cbffc8445`다.
