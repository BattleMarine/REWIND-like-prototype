# v2.1.0 A와 v2.1.1 세 seed의 동일 클립 비교

v2.1.0의 A 저장 예측과 v2.1.1 R2의 학습 seed 42·71·103 저장 예측을 같은 공식 validation에서 짝짓는다. 각 seed는 A와 421클립·16,425 실제 프레임을 비교한다. 공통 평가 noise seed는 71이다. 모델·GPU·RGB·학습 캐시는 필요 없다. [A 클립 꼬리 지표](V210_CLIP_ERRORS.md) · [v2.1.1 학습·보관](V211_BODY_HEAD.md)

## 입력과 고정 범위

| Colab 입력 모드 | 준비할 자료 |
| --- | --- |
| `drive` | 프로젝트 Drive의 `reports/v210_eval128_a_v1/`와 `reports/v211_body_head128_a_v1/seed-{42,71,103}/`를 읽는다. 각 경로의 `summary.json`, `group-00129.json`~`group-00146.json`, marker의 `artifact.name`으로 지정된 NPZ 18개가 필요하다. |
| `zip` | `/content/v210_groups.zip`에 A 원본 회수 ZIP을 올린다. `/content/v211_groups.zip`에는 `seed-42/`, `seed-71/`, `seed-103/` 각각의 위 37개 파일을 원래 이름으로 넣는다. ZIP 전체에는 일반 파일 111개가 들어가며 별도 디렉터리 엔트리 없이 패키징한다. |

A 기준은 기존 회수 ZIP SHA-256 `d47a24ffb032b3de299396827755d840d0c9c59b5fc7a9105dbf406d4caf1f0f`와 summary 파일 SHA-256 `796527609be8733fd44341e564641ff3b83314199fc55c7912418b81c2931da4`로 고정한다. v2.1.1은 R2의 실제 세 summary 파일 SHA를 발급 receipt에 기록했다. 각 summary가 원본의 seed·학습량·부모·공식 split·평가 조건을 고정하며, summary→group marker 봉인→NPZ SHA/바이트를 끝까지 검증한다. ZIP 모드는 추가로 외부 v2.1.1 ZIP SHA를 결과에 남긴다.

단독 summary, `fit-03097.json`과 head 파일만으로는 클립 비교를 수행할 수 없다. 분석에 필요한 추가 회수물은 v2.1.1의 seed별 18개 marker와 해당 NPZ, 합계 54쌍이다. 현재 로컬에는 A 원본 회수 ZIP과 R2의 세 실제 summary가 있다.

## 계산과 출력

- 각 그룹에서 같은 `(sequence_id, clip_start)` 집합을 확인한다. 그 안의 frame index·명목 시간·양쪽 mask·device/상대 GT 좌표가 정확히 일치해야 한다. 관절 순서는 기존 A 꼬리 분석의 고정 규칙을 재사용한다. 누락·중복·한 프레임 불일치·valid mask 불일치가 있으면 해당 실행 전체를 미통과로 보존한다.
- 각 seed와 클립마다 좌우 손목·손목 상대 손가락·전역 손가락, 양손 전역 및 양손 상대 손가락 MPJPE의 전후 값과 같은 유효 관측 수를 기록한다. 양손 지표는 좌우 유효 관측의 합으로 정규화한다. `delta = v2.1.1 - v2.1.0` mm이며 음수는 오차 감소다.
- 각 seed의 전체 공식 validation 및 시나리오별로 clip 균등 평균·p50/p90, 유효 관측 가중 평균 차이, 개선·악화·동률 클립 수를 따로 기록한다. 모든 seed에 같은 A를 사용하며, 각 seed의 원래 공식 validation macro를 함께 보존한다. paired 집계는 오류 분석용이다. 모델 선정에는 기존 공식 절차를 적용하며 독립 test 성능은 이 비교의 범위 밖이다.
- 예전 A 양손 전역 최악 규칙으로 고정한 ArchVis `Push__5_`(시작 50), CityPark `Tut_Hip_Hop_Dance__11_`(100), Downtown `Walk_Backward_Arc__4_`(0)를 그대로 따라간다. JSON에 전체 sequence key가 있다. 후속 seed 결과의 최악 클립으로 교체하지 않는다. 전체 1,263행과 이 세 기준 클립을 함께 보여준다.

| 출력 | 내용 |
| --- | --- |
| `paired-clips.csv` | 3 seed × 421클립, seed·전체 clip key·실제 프레임·각 지표의 전후·차이·유효 관측 수 |
| `paired-summary.json` | 검증 조건·SHA, seed별 전체/시나리오 요약, 이전 고정 어려운 클립 3개의 전후 값 |
| `result.json` | 실행 상태, 읽기량, 출력 크기·SHA, 실행 시간 |

## 검토 이후 Colab 실행 순서

1. 고정 `v211_paired_clips_cpu_20261122.ipynb`를 CPU 런타임에서 연다. NumPy 외에 모델 라이브러리 설치는 필요 없다. 기본 `INPUT_MODE = "drive"`에서 `DRIVE_PROJECT`를 확인한다. 회수 ZIP을 쓰려면 두 파일을 올리고 `"zip"`으로 바꾼다.
2. 승인된 실행에만 기본 OFF인 `RUN_PAIRED`를 켜고 설정 셀과 실행 셀을 차례로 실행한다. 상위 결과는 읽기 전용으로 열고, 출력은 `reports/v211_paired_clips_20261122/run-<UUID>/`에 새로 쓴다. 파일 복사 후 크기와 SHA를 다시 확인한다.
3. 마지막에 출력된 `result.json` 경로와 SHA-256을 회수한다. 성공은 `passed`, CSV 1,263행, 결과 2종 크기·SHA 검증으로 판정한다. 예외가 발생하면 상태와 가능한 오류 로그를 그 실행 UUID에 기록한다. 런타임 유실 시 `started.json`만 남을 수 있다.

마운트 최대 120초, worker 최대 240초, 전체 목표 상한 420초다. worker의 논리 입력 한도는 512MiB이며 ZIP mode는 외부 SHA 계산과 내부 멤버 읽기를 모두 포함한다. v2.1.1 회수 ZIP은 최대 192MiB, 결과는 파일당 최대 4MiB, 로컬 실행 스크래치 여유는 최소 128MiB다. 재실행마다 새 UUID를 발급한다.

```text
python -m scripts.v211_paired --base <A-회수-ZIP-또는-원본-폴더> --variants <v211-회수-ZIP-또는-시드-상위-폴더> --output <새-로컬-폴더>
python -m scripts.v211_paired_build --output <새-발급-폴더>
```

첫 명령은 로컬 CPU 분석, 둘째 명령은 노트북·발급 receipt 생성이다. 이 문서와 발급본 준비는 Colab 실행이나 v2.1.1 클립 분석 완료를 뜻하지 않는다.

## 2026-11-22 Colab CPU 결과

88.03초에 완료했다. 기존 validation 예측의 동일 클립을 짝지어 튜닝 후 오차에서 베이스 A의 오차를 뺐다. 단위는 mm이고 클립마다 같은 비중을 줬다.

| 학습 seed | 전역 손 오차 평균 변화 | 전역 개선 클립 / 421 | 손목 기준 손가락 오차 평균 변화 | 손목 기준 악화 클립 / 421 |
|---|---:|---:|---:|---:|
| 42 | −7.86 | 265 | +2.12 | 312 |
| 71 | −9.49 | 285 | +2.32 | 334 |
| 103 | −3.35 | 233 | +1.31 | 283 |

각 seed에서 같은 16,425프레임을 평가했다. 비교 CSV 1,263행에서 평균과 방향별 개수를 다시 계산했다. 공식 validation 종합 순위는 기존 집계 방식을 유지한다. 원본 3개는 `outputs/v2_plan_inputs/v211_paired_results_20261122/`에 보존했다. CSV SHA-256은 `d863d8d8a66967049b958baf96f8f5c4d3d1a2065cf8b912415aabfc7b9656ee`다.
