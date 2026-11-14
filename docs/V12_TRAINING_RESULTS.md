# v1.2 ArchVis 전체 학습 결과

검토일: 2026-11-15. 고정 소스 `7e1020159855e476c30dfde6eaf12783dba4dd80`. 실행 조건은 [전체 학습 연결](V12_TRAINING_CONNECTION.md), 분기 목적은 [실험 계보](V1_EXPERIMENT_LINEAGE.md)를 따른다.

## 완료와 검증 범위

Train 20,888페어, frontend validation 2,236페어를 사용해 frontend/body/hand 각10epoch, 총115,720update를 완료했다. Teacher는 train564clip, validation29clip/1,450frame이며 validation786frame은 제외됐다. `status=completed`, `failure=null`, history30개, frontend 무감독 batch0이다. A100-SXM4-40GB, FP32, torch2.14.1+cu130으로 실행했다.

Root는 회수 ZIP의 SHA256·CRC, 내부19파일의 크기·SHA, recovery/Drive receipt의 content seal을 확인했다. Checkpoint18개를 CPU에서 `weights_only=True`로 로드하고 부동소수 tensor의 유한값을 확인했다. 모델 forward, 실제 추론 시각화, 중단 없는 실행과 재개의 수치 동일성은 이번 검토에서 수행하지 않았다. CPU 로드를 실제 GPU 추론 검증으로 해석하지 않는다.

- 보존 폴더: `outputs/v12_training_release_20261112/user_test/completed_001/`
- `training.zip`: 873,393,783B, SHA256 `48661e75e0969ca140ab9cf0e543d73d2bb3b9ec108d30652cf7fb4f4b76b506`
- `last.pt`: SHA256 `b79a601f4f5563af4876dc1b1675e6677baabc68ade806a50dbf70176c34d92e`
- 검토 근거: `run.json`, `receipt.json`, `root_review.json`, `learning_curves.png`. 대용량 산출물은 Git 제외.

## 참고 validation 결과

| 지표 | v1.1 T4 best | v1.2 best | v1.2 epoch |
| --- | ---: | ---: | ---: |
| 2D 화면 안 평균 오차(native px) | 214.84 | 89.97 | 9 |
| Body MPJPE(mm) | 484.79 | 169.56 | 10 |
| Hand global MPJPE(mm) | 514.85 | 318.56 | 9 |
| Hand local MPJPE(mm), global best 시점 | 124.67 | 100.91 | 9 |

Hand local 행은 local 자체의 최저값이 아니다. v1.2 epoch1은 2D149.91px, body328.90mm, hand global337.18mm였다. 2D·body는 뚜렷하게 개선됐지만 hand global 개선은 상대적으로 작다.

Body best의 평균 뼈 길이는 예측184.76/GT208.35mm(약88.7%), 관절 퍼짐은558.76/516.72mm(108.1%)다. 이전의 극단적인 body 형태 축소에서 벗어났다는 수치 근거이며, 정확한 자세나 모든 프레임 정상 복원을 보장하지 않는다. Hand global best의 손가락 뼈 길이는14.54/25.78mm(56.4%), 퍼짐42.25/72.76mm(58.1%)여서 손 형태 축소가 남아 있다. Global과 local 오차 차이를 단순히 wrist 오차로 빼서 계산할 수 없다.

## 원인 해석과 한계

v1.1 대비 데이터와 업데이트가 함께 늘었다. Frontend5,000→104,440update, 각teacher200→5,640update이며 GPU와 short/tail 처리도 다르다. 따라서 개선을 데이터 확대만의 인과 효과로 주장하지 않는다. 충분한 학습 신호·노출과 개선된 2D 조건이 함께 작용했다는 가설을 지지한다. v1.4의 동일 train 클립 GT조건 body13.71mm/hand local2.90mm와 이번 예측 조건 validation을 성능 경쟁으로 비교하지 않는다.

명목25FPS와 frame-key stereo 가정을 유지한다. 실제 timestamp·stereo 동기·원본 motion 독립성은 미확인이므로 validation은 참고용이다. 손의 남은 오차에는 2D 품질, 상체/wrist 오차 전파, GT상체 학습과 예측상체 평가의 차이, 일반화 및 학습량 문제가 후보로 남는다. 어느 하나로 원인을 확정하지 않는다.

## 시간 해석

최종 호출 보고서의 실행시간은4,044.029초(67분24초)다. 테스트 세션이 보고한 이전 본학습 호출7,080.067초와 합하면11,124.096초(3시간5분24초), pilot37.578초는 별도다. 이전 호출 시간은 이번 최종 ZIP만으로 독립 재검증하지 않았다. 캐시 검사·특징 생성·학습·평가·checkpoint/주기 보관이 섞인 호출 시간이므로 순수 GPU 학습 시간이나 pair당 forward 시간으로 나누지 않는다. 원본 준비·캐시 복원·종료 보관까지 포함한 전체 wall-clock 비용도 아니다.

## 후속 방향 — 제안, 새 실행 미승인

1. 회수한 best 모델로 동일 고정 validation 클립의 RGB/GT/예측2D 및 body/hand3D를 시각화해 개선·실패 양상을 확인한다. 학습 곡선은 실제 포즈 시각화를 대신하지 않는다.
2. 같은 checkpoint·클립·noise 조건에서 hand의 GT상체/예측상체 입력을 교체해 조건 민감도를 확인한다. 분포 밖 입력 교체의 한계를 함께 기록한다.
3. 저장된200update 모델과 최종 모델을 같은 평가로 비교해 추가 업데이트 이후 변화를 확인한다. v1.1과 데이터가 달라 데이터 효과의 완전한 분리 시험은 아니다.
4. 위 결과를 보고 2D·손 학습 보완 또는 통제 실험을 선택한다. v1.3 다른 시나리오 추가는 계속 보류한다.


## 2026-11-15 실제 예측 시각화

사용자 요청으로 `scripts/visualize_archvis_v12.py`를 추가해 best frontend9/body10/hand9를 CPU FP32·DDIM10으로 추론했다. 기존 Getting_Up__1_ frame1/10을 입력 가용성으로 사전 선택했다. 두 프레임은 v1.2 manifest의 train이며 v1.1 고정1,000선택 밖이다. 로컬에 완전한 validation triplet은 없어 대표 성공/실패 표본이나 일반화 비교로 주장하지 않는다. 각각 T=1 reset이며 연속50frame 평가와 다르다.

입력은 RGB 예측2D와 pose JSON의 카메라 metadata다. Hand는 예측 body 상체와 wrist를 사용한다. GT 관절은 비교·기하 검사에만 사용했으며 GT조건 oracle 출력이 아니다. 카메라를 RGB에서 추정하는 완결된 RGB-only 추론도 아니다. 세 열은 v1.1/v1.2/GT, 두 RGB 행은 좌우 카메라, 마지막 행은 같은 축·축척의3D다. 주황은 몸, 청록은 손가락이다.

| 프레임 | 2D 화면 안 오차 v1.1→v1.2(px) | Body v1.1→v1.2(mm) | Hand global v1.1→v1.2(mm) |
| --- | ---: | ---: | ---: |
| 1 | 245.94→150.48 | 518.19→416.47 | 667.99→426.90 |
| 10 | 168.62→43.69 | 360.24→169.66 | 504.55→163.73 |

그림에서 v1.1의 점처럼 축소된 골격은 v1.2에서 크게 펴진다. 그러나 frame1의 팔·다리 방향과 손 위치는 GT와 많이 다르고, frame10도2D 개선에 비해3D 자세·사지 방향 오차가 남는다. '형태가 펴짐'과 '정확한 자세 복원'을 구별해야 한다. 두 표본에서만 확인한 차이이며 원인을2D 하나나학습량 하나로 확정하지 않는다. 특히 T=1의 문맥 부재와 학습 표본 포함 여부 차이가 있다.

산출물은 앞의 보존 폴더 아래 `visualization/`의 `frame-000001.png`, `frame-000010.png`, `comparison-frame-000001.png`, `comparison-frame-000010.png`, `predictions.json`이다. Root가 예측 JSON seal·모델 pin·PNG4개 hash/decode·공통축척을 검토했고, GT와v1.1 재추론 결과가 기존 결과와 정확히 일치함을 확인했다. 기존/신규 로더 관련4개 회귀 검사 통과. Root의 v1.2 추가 추론 재실행은 하지 않았다. 다음은 고정 validation 연속클립의 동일 시각화가 적합하며 이번 완료를 그 검증 완료로 보지 않는다.
