# v2.3 · 고정 A DDIM 10·5·2 정확도와 추론 시간

2026-11-23 · v2 128 A → v2.3 완료. 동일 가중치와 초기 noise로 DDIM 10·5·2단계를 비교했다. 공식 validation 421클립·16,425프레임 종합 오차는 각각 219.72·214.82·202.20 mm였다. L4 FP32에서 GPU 적재 특징부터 몸·손 출력까지 150프레임 묶음의 중앙 시간은 프레임당 4.889·2.412·0.984 ms로 환산됐다. 세 시나리오 모두 2단계에서 몸·전역 손·손목 기준 손가락 오차가 감소했다. 원본 네 파일은 `outputs/v2_plan_inputs/v23_l4_results_20261123/`에 보관했다.

v2-128-A의 frontend e3·body e3·hand e2를 고정한다. Body와 hand의 결정적 DDIM 반복 수를 함께 10·5·2로 바꾸고 공식 validation 전체의 품질과 같은 GPU의 지연을 기록한다. 새 학습 update는 0회다. 모델을 선택하는 정확도 손실 허용값은 미확정이다. [v2 튜닝 설계](V2_TUNING_DESIGN.md) · [A 전체 validation 근거](V210_EVAL.md)

## 비교 조건

| 범위 | 고정 조건 |
| --- | --- |
| 정확도 | 공식 validation 421클립·16,425 실제 프레임. 각 설정에서 같은 순서로 한 번 계산한다. 본래 평가의 body·hand-global·hand-local MPJPE, 손목·좌우·시나리오별 지표와 유효 관측을 기록한다. 3개 시나리오의 body와 hand-global 6점 균등 평균이 기존 macro다. |
| 확률·조건 | 평가 seed 71, body/hand offset 100/200, sequence와 frame 키별 초기 noise. 클립마다 같은 두 초기 noise 텐서를 세 설정에 재사용하고 FP32 텐서 SHA를 남긴다. FP32, TF32 OFF, 결정적 DDIM eta=0. 동일 그룹에서 복원한 A frontend 예측 특징을 사용한다. |
| 상체 | 각 설정의 body 예측 상체를 같은 설정의 hand에 공급한다. 손목 상대 손가락은 같은 body의 손목에 붙여 전역 device-mm 좌표를 만든다. |
| 시간 | 원래 비교 영수증에 고정된 세 시나리오 첫 50프레임 클립 각 1개로 150프레임 묶음을 만든다. Batch 1로 세 클립을 순차 호출한다. arm마다 묶음 warm-up 1회 후, 실행 순서를 섞은 고정 5라운드에서 arm별 5회 측정한다. 시작 전과 종료 때 같은 GPU를 동기화한다. |
| 시간의 경계 | GPU에 적재된 예측 특징·카메라·시간 mask·초기 noise에서 시작해 body, 예측 상체 연결, hand, 전역 관절까지 측정한다. 전체 묶음의 동기화 wall clock과 CUDA stream event 시간, 각 5회 원시값·중앙값·최소/최대, 프레임당 환산 평균과 FPS를 보존한다. RGB 디코드·2D frontend·초기 H2D·출력 D2H·시각화·Drive I/O는 측정 경계 밖이다. |

10-step은 이번 GPU에서 다시 계산하며 5·2의 품질 차이와 속도배수의 기준으로 사용한다. 회수한 T4 v2.1.0의 10-step macro 219.72474514022193mm와 이번 결과의 차이도 기록한다. timing의 프레임당 값은 묶음 처리량을 환산한 수치다. 추론 레이턴시를 A100 조건으로 보고할 때는 같은 GPU 조건의 실제 A100 결과를 사용한다. 독립 test와 영상/실시간 end-to-end 지연은 별도 범위다.

## 고정 입력과 실행

동결 128 source ZIP `v2_gpu_training_128_v1.zip`의 SHA-256은 `9a68d8e45bcf3abf6fab3004740faa6805d4ce3fda5118c3a7102011937849d8`다. 새 발급본은 `outputs/v2_plan_inputs/v23_ddim_128_A_write4g_release_20261123`에 있다. `v23_ddim_128_A_tools_write4g.zip`에는 고정 v2.1.0 평가 모듈과 v2.3 전용 모듈이 있다. 최초 2 GiB 발급본은 `outputs/v2_plan_inputs/v23_ddim_128_A_cpu_verified_release_20261122`에 보존한다.

| Drive 원본 | 고정 기준 |
| --- | --- |
| `reports/v2_gpu_training_128_v1/A-seed-42/` | 원본 완료 진단·비교·step-000009.json, step 9의 복구 ZIP. 영수증 content SHA `7d2cf12cdaf42916d98190cf2d57e11e93a6d94165d5e753db6fc1d11c3026d4`. |
| A의 `features/9588afef53e4802089baf7c51235f18068d110d02b1cac8ca6371b8d7aa5e276/` | 129–146번 그룹 marker와 ZIP, 파일별 SHA. 129번에 포함된 train 클립은 로드한 파일 묶음에서 validation 입력과 분리한다. |
| `reports/cache/v2_cpu_cache_041837Z/`, `reports/cache/v2_bulk_cache_041837Z/` | 캐시 catalog와 boundary marker를 읽어 원본 입력 identity의 고정 경계 `dc79219153d69f44020e0c42f9f3ede6cd3add179211f6c300345b2416dab511`를 확인한다. |

1. 검토·승인 후 `v23_ddim_128_A_cuda_write4g_20261123.ipynb`를 Colab에서 열고 CUDA GPU 하나를 선택한다. source와 새 tools ZIP을 `/content`에 올려 발급 receipt의 SHA와 대조한다.
2. `DRIVE_PROJECT`를 확인한다. 기본 OFF인 `RUN_V23`을 켜고 설정 셀과 실행 셀을 순서대로 실행한다. 새로 지정하는 GPU에서 비교 세 설정을 같은 호출로 실행한다.
3. 출력은 `reports/v23_ddim128_a_write4g_v2/run-<UUID>/`에 새로 쓴다. 마지막에 표시되는 `result.json` 경로와 SHA-256을 회수한다. `status=completed`와 `worker-result.json`·`quality.json`·`latency.json`의 SHA와 크기를 함께 확인한다. 시간 또는 오류로 중단하면 이미 완성된 파일과 가능한 오류 기록을 별도 UUID에 보존한다. 다음 실행은 별도로 승인한다.

## 비용과 중단 조건

| 항목 | 계획·상한 |
| --- | --- |
| 시간 | 마운트 최대 120초. 전체 목표 상한 3,600초. worker는 마지막 150초를 새 평가·샘플링 시작 전에 예약한다. controller는 종료 신호와 보관을 위해 마지막 35초 전까지 기다린다. |
| 읽기·저장 | Worker 논리 읽기 2 GiB, 논리 쓰기 4 GiB. A 복구 ZIP 435,178,624 B와 validation feature ZIP 18개 합계 93,450,466 B를 읽는다. 시작 시 로컬 여유 4 GiB와 worker 결과 파일당 2 MiB 한도를 적용한다. |
| 연산량 | 품질 평가 `421 × (10+5+2) × 2 = 14,314`회와 시간 비교 warm-up·실측 `3 × (1+5) × (10+5+2) × 2 = 612`회 denoiser forward를 계획한다. |

기존 T4 v2.1.0의 10-step 품질 1회 호출(준비 포함)은 약 19.35분이었다. 같은 범위의 샘플링 1.7배를 단순 적용한 값은 약 33분이며 실제 v2.3 시간은 장치·I/O·결과 처리에 따라 달라진다. 예산·시간·해시·split·가중치·GPU 초기화 중 어느 하나라도 검증에 실패하면 성공 조건을 충족하지 않는다. VM 유실 시 `started.json`까지만 보존될 수 있다.

쓰기 산정에는 A ZIP 복사 435,178,624 B, A 복원 추출 상한 1 GiB, feature ZIP 93,450,466 B, 18개 feature 그룹의 추출 상한 576 MiB, 결과와 Drive 보관 여유 32 MiB를 포함했다. 합계는 2,239,905,122 B(약 2.09 GiB)이며 4 GiB 한도에서 2,055,062,174 B(약 1.91 GiB)가 남는다. 원장은 A 복사·추출 예약을 논리 쓰기에 반영한다. Feature 임시 저장과 결과 보관은 기존 경로의 구간별 파일 상한을 적용하며, 산정표에는 여유를 위해 합산했다. 150초 worker 여유와 35초 supervisor 종료·보관 여유를 유지한다. 계산은 상한 기준이며 실제 Drive 쓰기량은 실행 후 확인한다.

## 로컬 발급

```text
python -m scripts.v23_build --source <고정-128-source.zip> --reference-dir <최종-v210-eval-발급-폴더> --summary <회수-v210-summary.json> --output <새-발급-폴더>
```

발급기는 기존 v2.1.0 release receipt와 T4 요약의 고정 SHA를 확인한 뒤 새 노트북·도구·가이드·발급 영수증을 생성한다. 이번 GPU 품질과 지연의 실제 값은 사용자 실행 결과를 회수한 뒤 판단한다.
