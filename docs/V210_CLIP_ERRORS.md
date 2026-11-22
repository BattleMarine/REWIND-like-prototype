# v2.1.0 저장 예측의 클립별 오차

v2-128-A의 공식 validation 421클립·16,425 실제 프레임을 그대로 사용한다. 회수본의 저장 예측을 CPU에서 읽으며, 각 클립의 좌우 손목과 손가락 오차를 비교 가능한 형태로 기록한다. [원래 평가·회수 범위](V210_EVAL.md)

## 계산과 비교 지침

- 각 클립에서 좌·우 손목, 손목 상대 손가락, 전역 손가락의 MPJPE와 유효 관측 수를 기록한다. 좌표와 거리는 device 좌표계의 mm다. 각 측의 손목 분모는 유효 손목 프레임 수다. 상대 손가락은 같은 쪽 손목과 손가락이 모두 유효한 항목, 전역 손가락은 손목 유효 여부와 관계없이 유효한 손가락 항목을 사용한다.
- 양손 전역 MPJPE는 클립 내 좌우 유효 손가락 관측을 풀링한다. 좌우 유효 관측 수가 다르면 각 손의 비중도 달라진다. MPJPE는 저장 FP32 좌표를 FP64로 변환한 후 관절별 L2 거리의 평균으로 계산한다.
- 시나리오별 p50/p90은 유효 관측이 한 개 이상 있는 클립의 MPJPE를 클립당 같은 비중으로 계산한다. NumPy의 선형 보간을 사용한다. 14프레임 클립과 50프레임 클립도 같은 비중이다. `eligible_clips`와 `valid_observations`를 함께 제공한다. 유효 관측이 없는 지표는 JSON의 `null`, CSV의 빈 값으로 기록한다.
- 최악 클립은 결과를 보기 전에 정한 규칙에 따라 시나리오마다 하나를 뽑는다. 양손 전역 MPJPE가 최대인 클립이며, 동점이면 `sequence_id` 오름차순, 그다음 `clip_start` 정수 오름차순이다. 전체 421행을 같이 보존하고 후속 모델은 `(sequence_id, clip_start)`로 같은 클립을 맞춘다. 비교 시 유효 수와 실제 프레임 수까지 함께 대조한다.
- 모델 선정에는 기존 공식 validation macro를 사용한다. 이 지표는 시나리오 안에서 유효 관측을 풀링한 body 및 hand-global 6개 점수를 균등 평균한다. 클립 균등 p50/p90과 최악 클립은 오류 분석용이다. [본 학습의 선정 기준](V2_TUNING_DESIGN.md)

## 고정 노트북 실행

1. 오케스트레이션이 인계한 `v210_clip_errors_cpu_20261122.ipynb`를 Colab CPU 런타임에서 연다. 업로드 파일은 `/content/v210_groups.zip` 하나다. 크기는 37,004,795바이트, SHA-256은 `d47a24ffb032b3de299396827755d840d0c9c59b5fc7a9105dbf406d4caf1f0f`다.
2. 첫 코드 셀에서 `DRIVE_PROJECT`를 확인하고 `RUN_CLIP_ERRORS = True`로 바꾼다. 기본값은 OFF다. 두 코드 셀을 순서대로 실행한다. worker와 SHA는 노트북에 들어 있으며 CPU NumPy를 사용한다. 모델·캐시·원본 Drive 결과는 이 작업의 입력에 포함되지 않는다.
3. 실행기는 Drive를 마운트하고 `MyDrive` 아래 프로젝트의 기존 `reports/`를 확인한다. 새 UUID 폴더 `reports/v210_clip_errors_20261122/run-<UUID>/`에 쓰기·readback을 검사하고, worker 출력의 크기·SHA를 Drive 복사본과 대조한다. 마지막 출력의 `result.json` 경로와 SHA-256을 회수한다.

| 파일 | 범위 |
| --- | --- |
| `clip-errors.csv` | 전체 421행과 조인 키·실제 프레임·좌우 오차 및 유효 수 |
| `clip-errors.json` | 같은 421행, 입력·평가 조건, 시나리오별 각 지표 p50/p90과 최악 클립 |
| `result.json` | 상태, 입력 및 worker SHA, 결과별 크기·SHA와 경로 |

시간은 마운트 최대 120초, 분석 subprocess 최대 180초, 전체 목표 상한 360초다. 로컬 여유 공간 64MiB를 확인한다. 원본 ZIP의 크기와 SHA, 파일 수 37개·총 펼친 크기 64MiB 이내, 그룹 NPZ와 NPZ 내부 펼친 크기는 각각 최대 8MiB, 분석 결과는 파일당 최대 2MiB다. 18개 marker와 NPZ의 SHA·관절 순서·유효 mask·키·실제 프레임 수를 확인한다. 검증이나 보관에 실패하면 통과 상태를 쓰지 않는다. 런타임 자체가 종료되면 `started.json`만 남을 수 있다. 재실행은 별도 UUID에 기록한다.

## 실제 회수 ZIP 로컬 실행 결과

| 시나리오 | 클립 / 프레임 | 양손 전역 p50 / p90 (mm) | 최악 양손 전역 (mm) | 최악 클립 (시작 / 길이 / 유효 손가락 관측) |
| --- | ---: | ---: | ---: | --- |
| ArchVis | 59 / 2,236 | 254.22 / 466.79 | 674.54 | `Push__5_` (50 / 50 / 2,000) |
| CityPark | 248 / 9,620 | 246.39 / 416.80 | 875.49 | `Tut_Hip_Hop_Dance__11_` (100 / 14 / 560) |
| Downtown | 114 / 4,569 | 265.73 / 377.26 | 892.26 | `Walk_Backward_Arc__4_` (0 / 24 / 960) |

전체 `sequence_id`와 모든 지표의 백분위는 JSON을 따른다. 기존 GPU 요약에서 읽은 모델 선정 macro는 219.72474514022193mm다.

```text
python -m scripts.v210_tail --input <회수본-v210_groups.zip> --output <새-로컬-폴더>
python -m scripts.v210_tail_build --output <새-발급-폴더>
```

첫 명령은 CPU 검사를 수행하고 결과 크기와 SHA를 출력한다. 둘째 명령은 새 실행을 시작하지 않고 노트북과 발급 `receipt.json`을 만든다.

## Colab 재현 결과

18그룹·421클립·16,425프레임 분석이 종료 코드 0으로 완료됐다. 회수한 CSV·JSON은 로컬 검산본과 바이트 단위로 일치했다. 결과 영수증은 `outputs/v2_plan_inputs/v210_tail_root_verified_20261122/colab-result.json`에 보존했다(SHA-256 `0f75637fb5e512bb9e11c6f503c6efc6942134f5f7e28bf1ba541c280d4fda2d`).
