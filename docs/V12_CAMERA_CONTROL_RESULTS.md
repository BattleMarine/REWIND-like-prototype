# v1.2.2 — ArchVis 대조·카메라 조건 비교 결과

표시 버전: **v1.2.2** (2026-11-15 사용자 확정). 기존 실행 ID `V12-CAMERA-CONTROL-001` 및 파일·config·결과 해시는 보존한다. [번호 관리](EXPERIMENT_VERSIONS.md).

2026-11-15 `V12-CAMERA-CONTROL-001` 결과 검토. **동일 추론 경로에서도 ArchVis는 몸 크기를 유지했고, 도시에서는 큰 위치 편향·축소가 나타났다. 카메라 조건 분포 차이도 확인했지만 인과관계는 미확정이다.** 재학습·카메라 입력 보정은 하지 않았다.

## 근거와 검증 범위

- Revision `fdab0f1f499869a0d292a3d58944020b6a187f84`, run `94dda621d7a84e54a9e05c925bb2595e`, 최종 attempt `kiz9t_e7`.
- 회수 ZIP23,996,807bytes, SHA256 `48ca2d5cab15596b637bbb3db47a7736e32446cc516d6994fa6bdbf79ed7c63b`.
- 비교 보고서 content SHA `9de6bd8b79d64b9c7ac466ff11054effaa980abcfbcfd8af509abfd23589c55e`.
- 보존: `outputs/v12_camera_control_20261115/user_test/completed_001/`. 원본 ZIP·22개 구성파일·receipt·`root_review.json`을 유지한다.
- Root는22파일 크기/SHA, JSON seal, 고정 소스·모델·selection/prepared/cameras/inference/comparison 연결을 확인했다. Body·hand local/global MPJPE를 저장 예측과 GT에서 직접 재계산했다. 2D는 frame별 sum/count를 재집계했으며 원본 GT2D를 재회수한 것은 아니다.
- 카메라300행의 프레임 키, R/t→18차원 인코딩과 center 대수식, 저장된 모든 환경별 평균·표준편차·분위수·최솟값·최댓값을 독립 재계산했다. 원본 pose와 tensor를 다시 읽어 검사한 것은 아니다. 원격 Drive `verified`는 테스트 세션 보고이며 root가 원격 receipt를 재조회하지 않았다.

ArchVis는 공식 validation `012/Standing__1_`, `017/Petting_Animal__13_`의 각50frame이다. 이전 도시4클립200frame의 예측을 고정 reference에서 재사용했다. 여섯 클립 모두 같은 인물 경로 `rp_adanna_rigged_003_ue4/SKM_MenReadingGlasses_Shape_01`이며 다른 인물에 대한 일반화 평가는 아니다. 모델 best F9/B10/H9와 추론 implementation pin은 이전 도시 결과와 일치했다. [실행 계약](V12_CAMERA_CONTROL.md), [이전 도시 결과](V12_CROSS_SCENARIO_RESULTS.md).

## 성능 대조

| 지표 | ArchVis100frame | CityPark100frame | Downtown100frame |
| --- | ---: | ---: | ---: |
| 2D valid (native px) | 112.44 | 307.87 | 334.31 |
| Body MPJPE (mm) | 197.34 | 6,721.67 | 2,277.83 |
| Hand global MPJPE (mm) | 293.29 | 7,524.48 | 2,912.33 |
| Hand local MPJPE (mm) | 82.15 | 3,145.50 | 1,188.49 |
| Body 평균 뼈 길이 예측/GT (mm) | 184.87 / 185.28 | 14.67 / 185.28 | 18.73 / 185.28 |

ArchVis body 오차는 clip별243.86/150.82mm이며 중심 오차 평균130.83mm, 중심 제거 진단188.98mm다. Body spread는547.04/GT429.81mm로 크며, 손 뼈 길이는14.79/GT23.71mm로 축소가 남아 있다. **몸 크기를 유지한다는 것이 자세·손 정확도가 충분하다는 뜻은 아니다.** 중심 제거 진단은 회전·스케일 정렬이 없고 공식 MPJPE를 대체하지 않는다.

Root는 Standing frame25 PNG에서 몸 형태 유지와 자세·손 위치 오차를 직접 확인했다. 테스트 세션은 두 GIF 각각6개 정지프레임을 확인했다고 보고했다. 모든 프레임을 root가 시각 검토한 것은 아니다. 같은 full50 경로가 ArchVis에서도 항상 붕괴한다는 설명은 이번 대조와 맞지 않는다. 모든 공통 구현 오류를 배제한 것은 아니다.

## 카메라 조건 차이

아래는 **클립별 왼쪽 카메라 world→camera translation norm의 평균**이다. 카메라와 사람 사이의 거리나 depth가 아니며, 선택한 world 원점과 카메라 위치에 의존한다. 모델에는 이 translation의 각성분을100mm로 나눈 값이 입력된다.

| 환경·동작 | 평균 norm (m) | Body MPJPE (mm) |
| --- | ---: | ---: |
| ArchVis / Standing | 3.214 | 243.86 |
| ArchVis / Petting Animal | 1.684 | 150.82 |
| CityPark / Gunplay | 262.284 | 5,609.58 |
| CityPark / Kneeling Inspecting | 270.260 | 7,833.76 |
| Downtown / Reloading | 59.761 | 1,400.48 |
| Downtown / Quarterback Pass | 129.488 | 3,155.19 |

양쪽 카메라의 모든 frame을 포함한 norm 범위는 ArchVis약1.66~3.53m, CityPark262.14~270.40m, Downtown58.71~129.70m다. ArchVis100frame의 각성분 min/max 밖에 있는 도시 translation 성분 비율은 CityPark99~100%, Downtown92~100%였다. 이는 **이번 ArchVis 평가 표본**과의 차이이며 전체 train 분포의 범위나 공식 OOD 판정이 아니다.

코드상 camera 조건은 절대 world 원점에 의존하지만 정답은 device 기준이다. 공통 world 원점의 이동만으로도 조건이 달라질 수 있다는 구조적 우려가 실제 환경 간 큰 수치 차이와 함께 관찰됐다. 따라서 camera 조건의 민감도는 우선 검토할 가설이다. 다만 회전·영상·동작·2D 오차도 함께 달라졌으며, 크기와 오차의 표를 인과 증명으로 사용할 수 없다.

## 실행 비용과 후속 판단

A100-SXM4-40GB / CUDA FP32 / torch2.14.1+cu130, 모든 단계 complete. Reference0.76초, selection2.17초, model6.13초, prepare21.27초, infer+compare52.22초다. 추론 자체 보고 구간은50.41초이며 모델 로드·prefix 검증·metrics·렌더가 포함된다. 보관·프로세스 시작·업로드를 포함한 총 wall-clock이나 순수 모델 지연으로 해석하지 않는다. 원본 논리 읽기252,546,305bytes 중 도시pose17,179,658bytes이며 ArchVis pose의 로컬 재읽기8,551,520bytes는 별도다. 실제 네트워크 전송량을 측정한 것은 아니다.

Full50/prefix25 차이는0.000610/0.000366mm로 tolerance를 통과했다. 명목25FPS, 실제 timestamp·stereo 동기·원본 motion 독립성 미확인을 유지한다.

다음 후보는 **같은 RGB·같은 clip·같은 고정 모델에서 카메라 조건만 통제하는 재학습 없는 민감도 시험**이다. 예를 들어 모든 frame과 좌우 카메라에 같은 좌표 재표현 `x′_world=x_world+Δ`, `t′=t−RΔ`를 적용하면 회전과 물리적 stereo 관계를 유지하면서 world 원점 의존성을 조사할 수 있다. 이때 임의의 성분 clipping·scale 교체나 정답 오차를 보고 Δ를 튜닝하는 시험과 구분해야 한다. 구체적 Δ·예산·판정은 아직 정하지 않았으며 실행하지 않았다.

이번 근거만으로 혼합학습을 시작하거나 camera 수정이 문제를 해결했다고 판단하지 않는다. v1.3 보류와 v1.5 기존 제안은 유지한다.
