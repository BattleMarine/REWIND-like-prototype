# 실제102페어 RGB→예측2D→3D 단일 pass 진단

2026-11-03 사용자 승인에 따라 실제102 train 페어를 각 단계 한 번씩 사용해 학습 연결을 확인했다. `train_fit_diagnostic`이며 과적합 가능성이 있는 동일 자료 평가다. Validation·일반화·temporal·정식 학습 release는 수행하지 않았고 `diagnostic_only=true`, `train_ready=false`와 원 candidate blockers를 보존했다.

## 입력·범위와 실행

[RGB102 추출 결과](RGB102_PAIRS.md)의 고정 `outputs/test_rgb102_pairs_001_20261103/rgb102_pairs_001_result.zip`을 사용했다. 274628517bytes/SHA256 `ec11b7eb5c4a69f76acb2e5f5045d8c7c560096a6dd33b2249e5d3a6a3e1dcae`다. 5sequence·102frame·204PNG·pose102 모두 train이며 validation은0이다.

신규 UI 독립 진입점은 [real_rgb102_diagnostic.py](../rewind_m1/real_rgb102_diagnostic.py)다. 고정 ZIP SHA와 사용 member CRC/경로/타입·입력 config pin·g9의204개 payload SHA/크기/key·native PNG 형식/해상도를 검사한다. Pose/metadata만 약14MiB의 별도 출력에 선택 추출해 기존 converter로 raw102의 관절·mm 단위·좌표·재투영·공식 split을 다시 감사하고 원 candidate records/계약/blockers와 대조한다. 기존32frame release/bundle 제한과 일반 synthetic/실제 loader gate는 바꾸지 않았다.

기존 [2sample 진단](REAL_RGB_DIAGNOSTIC.md)을 보존한다. 공유 `predicted_data`의 frontend version을 기본 구버전인 keyword 인자로 받도록 확장했으며, 새 adapter는102 진단용 version을 명시한다. 입력은 `StereoHeatmapNet`→저장된 예측2D cache→`SingleFrameRegressor`다. GT2D는 heatmap supervision·2D 진단 평가에만 사용한다. Regression window에는 GT3D target/mask만 제공하고 누락 예측을 GT로 보충하지 않는다.

아래는 이미 수행한 단일 실행 명령이다. 기존 출력에 재실행하지 않으며 결과를 보고 추가 학습·LR 조정·best checkpoint 선택을 하지 않았다.

```text
python -m rewind_m1.real_rgb102_diagnostic
  --source-zip outputs/test_rgb102_pairs_001_20261103/rgb102_pairs_001_result.zip
  --output outputs/rgb102_diagnostic_20261103/developer/run
  --diagnostic-only
```

## 고정 학습·좌표·coverage 계약

| 항목 | 값 |
| --- | --- |
| 환경 | Python3.12.10 / Torch2.12.1+cpu / Pillow12.2.0 / CPU FP32 / thread1 |
| Optimizer / learning rate | Adam / 1e-4, 두 단계 모두 고정 |
| Scheduler | StepLR step51 / gamma1.0, learning rate 불변 |
| Seed | Frontend71 / regression37 |
| Batch·학습량 | B2 / 각51update·1pass / 합102update |
| 순서 | Sequence/frame key의 고정 정렬, shuffle 없음, 중복 없는 순차 cursor |
| Frontend | 기존 StereoHeatmapNet62관절 / 32×32 heatmap / sigma1.5 |
| Regression | 기존 SingleFrameRegressor / T1 / 868특징 / hidden32 / dropout0.1 |
| 3D loss | 기존 masked MSE, 유효 joint·xyz 평균 / 고정 scale1000mm / dimensionless |

RGB는 ZIP에서 한 view씩 decode해 `[0,1]`로 바꾸고 bilinear/align_corners=True/antialias=False로1024→64 축소한다. GT2D는63/1023, 예측은1023/63으로 원좌표에 복원한다. 전체 native 영상 tensor를 RAM에 적재하지 않고102개의 작은64×64 stereo 입력 약9.56MiB만 보관한다. Heatmap은 영상 안 annotation12097 joint-view를 supervise한다. 전체 valid12648 중 화면 밖551개를 visibility나3D 누락 mask로 바꾸지 않는다. 3D target은mm·6324개 유효 관절이며 원래 mask를 유지한다.

Frontend/regression 각각 실제 `keys_seen`102개·unique102·cursor102·51update를 보고서에 기록했다. 각 단계의 초기·최종 평가는 같은102개를 eval mode로 사용한다. Heatmap loss는 유효 joint-view sum/count, 3D는 유효 joint·xyz sum/count다. 3D 초기·최종은 모두 최종 frontend가 만든 동일한 고정 예측 cache를 사용한다. 처음의 random frontend와 마지막 frontend를 섞어3D 학습 변화로 보고하지 않는다.

## 실제 결과

| 같은102개 train-fit | 초기 | 단일 pass 후 |
| --- | --- | --- |
| Heatmap MSE | 3.463602881e-5 | 3.463318794e-5 |
| 2D 화면안 평균 px,12097 joint-view | 330.328979 | 330.275146 |
| 2D 전체 valid 평균 px,12648 joint-view | 340.673615 | 340.615356 |
| 2D 화면밖 평균 px,551 joint-view | 567.786743 | 567.631714 |
| 3D normalized MSE | 0.182936177 | 0.156363443 |
| 3D MPJPE mm,6324 joint | 682.599067 | 634.074794 |

두 loss가 감소했으나2D 개선은 작고3D 오차도 크다. 좋은 정확도·일반화 향상·과적합 방지 성공을 주장하지 않는다. 기존2sample100update 시험과 자료·학습량이 다르므로 수치를 직접 비교해 학습률의 효과를 입증하지 않는다.

Frontend gradient L2는6.87008e-8~1.59893e-7, regression은0.0648803~0.761704였다. 두 단계 모두51개 update의 gradient가 finite/nonzero이고 초기 대비 weight delta L2는각0.0288810/0.189044다. Frontend 저장 weights 재로드 후102개 예측 xy/confidence와 cache가 완전히 일치했고 regression weights 재로드의102개3D 예측도 완전히 일치했다.

## 예산·실측과 읽기 범위

고정 상한은 원격0·102optimizer update·600초·협력 RSS2GiB·출력512MiB다. Source ZIP300MiB 이하, 선택 member8MiB 이하·합512MiB, source ZIP SHA 읽기와 선택 member의 논리 합1GiB를 별도로 제한한다. 사용 member를 끝까지 읽어 CRC를 검사하며, 별도 전체 ZIP CRC 재읽기는 하지 않는다. 원본은 root가 이미 전체 CRC를 검토한 고정본이다.

실제 runner 내부32.922초, 관측 peak RSS330444800bytes(약315.14MiB), 전체 산출물17255584bytes(약16.46MiB), 원격0이었다. Source 전체 SHA 읽기274628517bytes와 선택 member의 해제 bytes261972506bytes의 논리 합은536601023bytes다. 이 합은 ZIP header/central directory·compressed backend 중복 읽기·출력에 추출한 pose 재감사/checkpoint 재로드를 포함하는 전체 OS disk IO 측정이 아니다. 시간은 검증·resize·학습·평가·저장/재로드·상태 복원을 포함하며 Python/PyTorch 최초 import·마지막 보고서 쓰기는 별도다. 처리량 benchmark나 OS hard memory/time 제한은 아니다.

## Checkpoint·재개·검증의 한계

`frontend_025.pt`, `frontend_051.pt`, `regression_025.pt`, `regression_051.pt`는 `real_rgb102_diagnostic_checkpoint_v1` 및 단계별 `diagnostic_only_rgb102_*` role이다. Model/Adam/scheduler/disabled scaler/RNG·step/cursor/samples_seen·고정 순서가 들어 있는 입력 identity·config/source/environment를 저장한다. Stage·역할·입력·설정·환경·cursor 불일치는 거부한다.

각25update에서 파일을 새 Stage에 로드해 전체 상태가 같음을 확인한 뒤 남은26update를 진행했다. 이 과정까지 합쳐 각51update이며 별도 replay 학습은 없다. **중단 없는51update와 재개의 완전 동등성 시험 및 임의 중단/새 프로세스 재개 CLI는 지원 완료로 주장하지 않는다.**

`frontend_weights.pt`, `regression_weights.pt`는 추론용 가중치다. `prediction_cache.json`은102 전용 diagnostic wrapper로 network64/native1024 cache·resize·입력 manifest와 weights 파일/state hash를 보존한다. 재로드 시 wrapper hash·weights metadata/state·정확한102 key·좌표 transform·mask/confidence를 검사한다. 일반 학습 release를 발급하지 않는다.

신규7개·기존2sample8개, 총15개 로컬 검사를24.569초에 통과했고 failure/error/skip0이다. 이 검사들은 추가 optimizer update를 하지 않는다. Cache/weights 변조 거부·GT fallback0·GT feature 교란 불변·tensor 필드/단위/mask·진단 role/cursor·일반 gate 차단을 확인했다. Root도 같은15개를 독립 통과했다. 개발 로그·실제 console·report/입력 manifest/pose audit/checkpoint/cache는 `outputs/rgb102_diagnostic_20261103/developer/`에 보존한다. 코드 revision은40fda70 기반 dirty 작업본이며 실제 파일별 code hash를 checkpoint/report에 함께 고정했다. 이후 검토 커밋과 실제 실행 당시 code hash를 구분한다.

Timestamp/motion 독립성·validation0·전체 영상 정합의 사람 승인·GPU/AMP·성능 개선 근거·정식 M1 baseline은 여전히 별도다. 원본 ZIP과 기존2sample 자료·일반 gate는 보존하고 다운로드·의존성 설치·새 사용자 Colab 시험은 수행하지 않았다.

Root는 추가 학습0으로 source code SHA·입력 manifest 재생성·stage별102key/51trace·Adam step25/51·고정 learning rate/cursor·최종 checkpoint/weights 모델 일치를 독립 확인했다. 초기 seed 모델 대비 weight delta, raw candidate와 GT3D/mask, 저장 frontend의102개 xy/confidence cache, 최종 heatmap/2Dpx/3D MSE/MPJPE 재계산도 모두 일치했다. 근거는 `outputs/rgb102_diagnostic_20261103/root/root_review.json`이다. 실제 학습은 개발의 단일1회뿐이며 root 검토가 validation이나 별도 학습 재현을 추가한 것은 아니다. Loaded-state equality를 넘는 완전 재개 보장은 하지 않는다.
