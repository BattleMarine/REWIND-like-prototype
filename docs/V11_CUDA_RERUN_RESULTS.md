# v1.1 CUDA 재실행 결과

2026-11-11. 학습 소스 `70accf2d6dc3ab7bb26836ffa7bb8f9d79a18473`. 이전 모델 미회수 실행과 분리한 새 결과다. 동일 train1,000·frontend/body/hand 각10epoch를 T4에서 완료하고 모델 본문을 회수했다. 명목25FPS·frame-key stereo 가정, 원본 motion 독립성 미확인 및 참고 validation 조건은 유지한다.

## 실행·보관 검토

- 환경: Tesla T4, VRAM15,637,086,208bytes(약14.56GiB), CUDA13.0, Torch2.14.1+cu130, Python3.13.16, FP32·batch2·입력128px·seed71·CPU threads1.
- `completed`, failure=null,5,400순회 update, 단계별10epoch·history30개·supervision skip0·지표 유한을 확인했다. 학습률·모델·입력/설정 pin은 고정 인계와 일치한다.
- 보관 위치: `outputs/v11_rerun_20261110/user_test/cuda_70accf2/training.zip`,684,818,352bytes, SHA256 `fcfbac1c26af1c3d40c40b171571a9292ee6e5dfe31ade10dc58b51ab7a1e7c1`. Drive attempt는 `attempt-6eix1ly3`, receipt는 `state=verified`, `kind=training`이다.
- Root가 ZIP의14개 내용 파일 CRC·크기·SHA를 확인했다. 체크포인트13개를 CPU에서 `weights_only=True`로 읽고 tensor 유한값·binding을 검사했다. Best/milestone 모델12개를 해당 구조에 strict load하고 frontend→body→hand 의존 해시를 확인했다.
- `last.pt` SHA256은 `c15e2d0623000bd0e51edfdc172012682fd2df15c4f19a10d56284e89dc304c7`. 모델·optimizer·scheduler·scaler·RNG 및5,400 update cursor가 존재한다. 실제 GPU 재개 시험을 한 것은 아니다. 독립 검토 내역은 같은 폴더 `root_review.json`에 보존한다.

실제 모델 본문을 확보했으므로 이전의 다운로드 실패는 이번 실행에서 반복되지 않았다. 로컬 본문과 보관 receipt 확인이며 VM 강제 종료 전에 주기적 원격 보존이 된다는 증거는 아니다.

## 참고 validation 결과

| 지표 | epoch1 | 최저 오차 | epoch10 |
| --- | ---: | ---: | ---: |
| Frontend 화면 안2D(px) | 307.509 | 214.841(e9) | 215.483 |
| Body MPJPE(mm) | 543.158 | 484.793(e9) | 549.247 |
| Hand global MPJPE(mm) | 563.272 | 514.854(e8) | 565.974 |

Hand global best와 같은 e8의 wrist-relative local 오차는124.671mm다. Frontend는validation2,236프레임, teacher는그중29clip/1,450프레임이다. 모델별 best epoch가 다르며 하나의 공통 best epoch로 보고하지 않는다.

Frontend e9 첫32 validation프레임의 normalized heatmap entropy는0.8253, 중앙과의 평균 거리는222.70px다(e1은0.9936·49.48px). 2D 중앙 집중은 완화됐지만 정확도가 충분하다는 뜻은 아니다.

Body best의 평균 bone13.274mm/GT208.353mm, 관절 spread39.303mm/GT516.717mm다. Hand best의 bone2.329mm/GT25.782mm, spread2.110mm/GT72.758mm다. 오차가 낮아진 best에서도 형태 축소가 크다. Epoch 증가만으로 해결되지 않았으며 과적합·조건 정보·loss/샘플링 중 단일 원인으로 확정할 수 없다. 이전1epoch 실행과는 frontend도 달라 teacher 학습량 효과만 분리할 수 없다.

## 실행 시간과 pair당 비용

`elapsed_this_invocation_seconds=5,844.470`(97분24초)로 사용자 관측약98분과 부합한다. 코드상 budget 시작 이후 raw cache 복원과 준비·학습·평가·checkpoint 저장을 포함하며 종료 후 결과 ZIP 생성/Drive 복사 시간은 별도다. Cache 복원290.798초를 전체에 다시 더하지 않는다.

- 현재 조건 전체 pipeline의 고유 train pair당 환산:5.844초/페어. 각 단계10epoch와 고정 validation을 포함한 값이다.
- 이를10epoch로 단순 나눈 비용은0.584초/(고유 train pair·epoch 묶음)이다. 세 단계와 validation·고정 준비 비용이 섞여 순수 학습 처리량이나 새 데이터 규모의 선형 예측치로 쓰지 않는다.
- Frontend 학습 step wall-clock은1,809.105초,10,000 pair 노출 기준약0.181초/페어다. 읽기·전처리·전송·계산을 포함한다. Frontend 계산 구간 wall-clock441.095초만 환산하면약0.044초/페어지만 모델 추론 레이턴시나 CUDA event 측정값이 아니다.
- Frontend validation3,110.615초, CPU 전처리4,095.348초, 로컬 캐시 읽기497.147초는 서로 중첩되는 계측이 있어 합산하지 않는다. 현재 시간 절감은 GPU 모델명보다 반복 전처리·평가 입력 경로를 우선 조사할 근거가 있다.
- GPU peak allocated약749.5MiB/reserved830MiB다. CUDA 할당기 관측이며 전체 장치 사용량이나 모든 설정의 VRAM 요구량을 뜻하지 않는다.

T4 실행 시간은 A100 추론 레이턴시 성능 근거가 아니다. 다음 최적화·학습 확대는 별도 변경이며 이번 결과 검토에서 자동 시작하지 않는다. v1.2 실행은 별도 테스트 세션 결과를 기다린다.

## 실제 예측 시각화

사용자 요청으로 동일 로컬 Getting_Up__1_ frame1·10의 RGB에서 이전1epoch 모델과 이번 best(frontend9/body9/hand8epoch)를 실제 추론했다. `outputs/v11_rerun_20261110/user_test/cuda_70accf2/visualization/`에 `comparison-frame-000001.png`, `comparison-frame-000010.png`, v1.1 단독 비교 PNG와 `predictions.json`을 보관한다. 학습 곡선은 상위 폴더 `learning_curves.png`다.

비교 그림은 열 순서로 이전1epoch/이번best/GT이며 행은 left RGB/right RGB/3D다. 주황은 몸, 하늘색은 손가락이다. 3D는 공통 투영·축척·위치를 사용하고 정답에 맞춰 재정렬하지 않았다. 이번2D는 이전보다 중앙에서 퍼졌지만 손·다리 위치 오류가 남고,3D는 GT보다 매우 작게 뭉친다.

두 표본은 고정train/validation 선택 밖이며 원본 motion 독립성은 미확인이다. 불연속 두 프레임을 각각T1로 reset하고 CPU/DDIM10으로 추론했다. 카메라는 로컬 pose 메타데이터를 사용하고 GT 관절은 geometry 감사와 비교 표시에만 사용한다. RGB만으로 카메라까지 추정한 결과나50frame temporal 평가 재현이 아니다.

기존 감사ZIP 전체를 이 실행에서 읽은 것은 아니다. 보존한 고정 `selection.json`의 content pin으로 membership를 확인하고 로컬 추출본의 파일 해시·카메라 geometry를 검증했다. 결과의 `audit_sha256=null`과 `selection_file_sha256`로 이를 구분한다. Root가 원본RGB/pose 해시·GT 좌표 일치·best 모델 해시·PNG 해시·결과seal 및 두 비교 이미지를 독립 확인했다. 상세는 `root_visualization_review.json`이다.

시각화 로더의 단계 의존성/receipt 불일치 거부, JSON과checkpoint coverage 표현 정합, milestone 선택 및2D 전용 비교·공통 축척의3개 로컬 검사를 root가 통과했다. 실제 출력은 개발 실행을 사용했고 root는 재학습 없이 모델·입력·GT·산출물을 독립 검토했다.
