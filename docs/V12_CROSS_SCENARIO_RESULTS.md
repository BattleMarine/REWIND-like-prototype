# v1.2 CityPark·Downtown 무재학습 평가 결과

2026-11-15 검토. `V12-CROSS-SCENARIO-001`의 실행·회수는 완료됐지만 이번 4클립에서 정확도와 형태 복원은 실패했다. 새 시나리오 혼합학습의 효과나 실패 원인을 확정하는 시험은 아니다.

## 근거와 검토 범위

- 실행 revision: `2de4262494d54d3ff905350bbb0e85ff19cfd274`. 메모리 검사 수정 전 `68fd9b8`의 준비 전 실패와 구분한다.
- 실행: `run-56aad836c4134dbcbaa515288495017f`, inference 보관 attempt `hye69dbk`.
- 회수 ZIP: 47,144,765bytes, SHA256 `c302fede5d8f3f5d72548dedb627f57892e9d8420670381e4c49ebd049bc6910`.
- 추론 보고서 content SHA256: `7987ca3093a16b8f32b5ab454a0c16600d0ce404fbd01698813dbbe90bbe88e3`.
- Root 보존 위치: `outputs/v12_cross_scenario_20261115/user_test/completed_001/`. 원본 ZIP, 해제 보고서·그림과 `root_review.json`을 보존한다.
- Root는 receipt의29파일 크기·SHA, JSON content seal, 고정 배포 source/model 및 selection/prepared 연결을 확인했다. Body MPJPE를 저장 예측과 GT에서 재계산해 보고서와 일치함을 확인했다. 원본 RGB·pose·준비 tensor는 ZIP에 없으므로 입력 파일 전체를 재검증한 것은 아니다.
- Drive `verified`는 테스트 세션의 회수 보고다. Root가 원격 receipt 본문을 다시 읽은 것은 아니다. 선택·준비·추론 보고서는 모두 `complete`다.

모델은 frontend e9/body e10/hand e9로 고정했고 재학습하지 않았다. 예측2D·camera metadata·예측 body upper/wrist 조건을 사용했다. GT 관절은 평가·기하 검사·그림에만 사용한다. 명목25FPS이며 실제 timestamp·stereo 동기·원본 motion 독립성은 미확인이다. [고정 실행 계약](V12_CROSS_SCENARIO.md).

## 표본과 성능

공식 validation에서 시나리오별2sequence, 각 frame0~49를 선택했다. 모든 클립의 인물 경로는 `rp_adanna_rigged_003_ue4/SKM_MenReadingGlasses_Shape_01`로 같다. 따라서 여러 인물에 대한 일반화 평가가 아니다. 같은 인물명은 ArchVis train23sequence에 있고, `Quarterback_Pass` 동작명도1sequence에 있다. 이름은 중복 proxy이며 원본 motion 동일성·독립성을 증명하지 않는다.

| 환경·클립 | Body MPJPE (mm) | Hand global (mm) |
| --- | ---: | ---: |
| CityPark / 010 / Gunplay__13_ | 5,609.58 | 6,363.02 |
| CityPark / 010 / Kneeling_Inspecting__1_ | 7,833.76 | 8,685.94 |
| Downtown / 020 / Reloading | 1,400.48 | 1,113.11 |
| Downtown / 029 / Quarterback_Pass | 3,155.19 | 4,711.56 |

| 지표 | CityPark 100frame | Downtown 100frame | 전체200frame |
| --- | ---: | ---: | ---: |
| 2D valid (native px) | 307.87 | 334.31 | 321.09 |
| Body22 MPJPE (mm) | 6,721.67 | 2,277.83 | 4,499.75 |
| Hand40 global MPJPE (mm) | 7,524.48 | 2,912.33 | 5,218.41 |
| Hand40 local MPJPE (mm) | 3,145.50 | 1,188.49 | 2,166.99 |

2D 전체 유효 분모는24,800이고 화면 안 지표는287.56px/21,094관측이다. Body 분모4,400, hand local/global 각각8,000이며 joint별 합/분모 집계다. 이 작은 고정 표본의 결과를 시나리오 전체 평균이나 일반화 성능의 확정값으로 확장하지 않는다. 기존 ArchVis validation과 클립 구성·조건이 달라 수치 차이를 배경 하나의 효과로 해석하지 않는다.

## 형태 축소와 위치 이탈

Body 평균 뼈 길이는 예측16.70mm / GT185.28mm(약9.0%), spread는24.83 / 430.18mm(약5.8%)다. Hand finger 평균 뼈 길이는8.50 / 23.71mm(약35.9%)다. 실제 수치에서 축소가 확인되므로 단순히 시각화 autozoom 때문에 뭉쳐 보인다고 설명할 수 없다. 그림은 클립 전체의 GT+예측을 같은 축척으로 표시한다.

Root가 body22의 프레임별 중심을 비교한 평균 위치 오차는4,477.52mm다. 각 프레임에서 예측과 GT의 중심을 각각 빼고 다시 계산한 관절 오차도373.74mm로 크다. 이 보조 진단은 회전·스케일 정렬이 없으며 공식 MPJPE를 대체하지 않는다. **위치 이탈과 형태 축소가 동시에 있다.** 높은 MPJPE를 뼈 길이 축소만으로 설명하지 않는다.

Root는 CityPark Gunplay frame25 PNG에서 2D 관절의 GT 불일치와 3D의 크게 어긋난 위치·축소를 직접 확인했다. 테스트 세션은4GIF 각각6프레임을 시각 확인했다고 보고했다. 모든 프레임을 root가 시각 검토했다고 주장하지 않는다.

## 실행과 입력 검사

A100-SXM4-40GB, CUDA FP32, torch2.14.1+cu130, CPU threads1. 선택12.58초, 준비219.28초, 추론 보고86.62초다. 추론 보고에는 모델 로드·prefix 검사·지표·렌더가 포함되며 모델 복사와 보관은 별도다. 따라서86.62/200=약0.433초는 이 평가 단계의 frame당 상각 시간이지 순수 모델 지연이나 학습 속도가 아니다. 원본 논리 읽기는475,821,396bytes로 실제 네트워크 전송량과 구분한다.

Full50/prefix25 최대차이는 클립별0.005249/0.007324/0.001099/0.003662mm로 고정 tolerance를 통과했다. 준비 보고서의 최대 projection 잔차는약0.0000758px, world→camera 잔차는약6.66e-12 raw units다. 이는 검사한 GT·카메라 내부 일관성의 근거이며 예측 정확도나 다른 시나리오의 전체 좌표 계약을 증명하지 않는다.

## 원인 가설과 후속 판단

관찰로 확정한 것은 2D 오차 증가, 3D의 큰 위치 이탈과 축소다. 현재 근거만으로 학습량 부족·배경 변화·카메라 조건 분포·추론 구현 중 어느 하나를 단독 원인으로 확정하지 않는다.

디벨로퍼의 읽기 전용 대조에서는 학습과 교차 평가의 `_pose_targets`(raw→mm 배율10),128px RGB 전처리, camera 변환·100mm 정규화,14J 특징 배열, DDIM10·noise 및 예측상체/손목 연결이 공통이었다. 현재 확인한 코드에서 교차 평가만의 변환 불일치는 발견하지 못했다. 이는 원본·준비 tensor 직접 대조나 구현 오류의 완전한 배제는 아니다. 카메라의 world 원점 의존 가능성은 학습 때부터 공통인 설계 문제의 후보다.

1. 같은 고정 추론 경로로 ArchVis 연속 validation 클립을 평가해, 교차 시나리오에서만 발생하는지 먼저 구분하는 것이 유용하다. 기존 로컬 train2프레임 T=1 시각화는 이 대조군이 아니다.
2. 선택된 도시 클립과 ArchVis의 camera condition·단위·좌표·2D 특징 범위를 비교한다. 현재 `encode_camera`는 world→camera 회전과 translation/100mm를 입력으로 사용하며 교차 준비의100mm는 고정 v1.2 설정과 일치한다. World 원점·카메라 위치가 다른 환경에서는 device 좌표의 인체가 비슷해도 이 조건값의 분포가 달라질 수 있다. 내부 기하 검사가 통과해도 모델 입력이 학습 범위 밖일 수 있다는 가설이다. 현재 결과 ZIP에 원본 camera 수치나 준비 tensor가 없으므로 실제 분포 차이는 아직 직접 입증되지 않았다.
3. 같은 클립에서 예측2D/GT2D, 예측 상체/GT상체 조건을 분리하는 진단은 원인 구분에 도움이 될 수 있다. Oracle 결과는 실제 RGB 성능과 별도로 표시해야 한다.

이 항목들은 후속 제안이다. v1.3 혼합학습이나 v1.5·새 재학습을 시작하지 않았으며, 이번 실패만으로 데이터를 섞으면 해결된다고 결론 내리지 않는다.
