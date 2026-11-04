# 실제 GT2D→3D 단일 frame 진단

> **2026-11-05 이력 구분:** Serena2sample GT 입력 진단의 고정 결과·checkpoint는 보존한다. 후속102쌍 RGB 예측 입력 진단과 Joyful Jump GT 표시를 이 결과에 합산하지 않는다. 아래 실행은 역사적 재현 경로이며 새 학습 지시가 아니다.

2026-11-02 · 실제 Serena 정적2sample의 학습 연결·손실·gradient·checkpoint 재개만 확인하는 명시적 진단이다. RGB 예측 성능, temporal 실험, validation/test 또는 일반화 결과가 아니다. 기존 candidate와 일반 학습 gate는 유지한다.

## 입력과 승인 범위

입력은 [확보 기록](REAL_DATA_NEXT_RUN.md)의 `outputs/real_stereo_sample_20261102/root/prepared`다. 동일 Serena Getting_Up__1_ sequence의 frame1·10이며 둘 다 공식 train에 속한다. 비연속 두 record를 각각 현재 frame 하나로 사용하고, 시계열 window나 독립 train/validation 분할로 만들지 않는다.

`python -m rewind_m1.real_gt2d_diagnostic`는 `--diagnostic-only`와 candidate/RGB manifest의 실제 파일 SHA256 두 개를 요구한다. 일반 `Trainer`/`load_document`를 수정하거나 candidate를 synthetic/verified로 바꾸지 않는다. 별도 `real_gt2d_diagnostic_data_v1` identity를 사용하며 기존 `pack_window`의 순수 tensor 조립만 재사용한다.

진단 입구에서 다음을 대조한다.

- Candidate 자체hash/schema·학습 차단·현재62관절 spec/계약·mm/device-relative/GT2D 명세.
- 공식 split3개·Scaramuzza·motion-map의 기존 source SHA와 spec pin. 두 record의 고유 sequence/frame·train 소속·motion 선언.
- 보존된 pose/RGB 추출 manifest와 전체 payload hash, RGB dataset version·view별 sequence/frame/side·해상도·원래 archive/추출 hash. 기존 과거 producer hash는 보존하고 준비 자료 읽기에만 명시적 code-change 허용을 사용한다.
- 원본 pose에서5개 field의 유효 mask와 좌표를 다시 구성해 candidate와 대조한다. 3D는 기존 spec의 raw→mm factor를 대조하고, 이미 mm인 candidate를 tensor로 만들 때 추가 배율을 적용하지 않는다.
- GT2D fields에서868특징을 다시 조립해 candidate feature cache와 float32 기준 정확히 대조한다. 학습 입력에는 조립한 GT2D·유효/known flag만 들어가며 GT3D는 target으로만 사용한다.

고정 관절은 body22+hand40의62개다. Tensor shape는 features `[2,1,868]`, target `[2,1,62,3]`, target mask `[2,1,62]`다. 각 view의14J 구성은 기존 M1 계약과 같고, 영상 범위 밖 좌표를 clipping/마스킹하지 않는다. Frame1의 left22/right20개 화면 밖 GT와 annotation validity를 구분한다. Visibility/confidence 의미는 unknown으로 보존한다.

## 고정 CPU 실행

의존성은 이미 설치된 Python3.12.10·torch2.12.1+cpu·Pillow 환경을 사용했다. 새 의존성/다운로드/GPU/Colab 실행은 없다. 아래 `python`은 이 환경의 Python 실행 파일을 뜻한다. 출력은 새 별도 디렉터리만 허용한다.

```text
python -m rewind_m1.real_gt2d_diagnostic
  --prepared outputs/real_stereo_sample_20261102/root/prepared
  --output outputs/real_gt2d_diagnostic_20261102/developer/new_run
  --diagnostic-only
  --candidate-sha256 44ec803069ddee642e45832f88a71979c5cf1966bbe2e8574d37bd9742bec0e2
  --rgb-manifest-sha256 392ffe2184a71212acb551a834ccb8a6b51e3540d6b0648722c7ed11e842205a
```

기존 `SingleFrameRegressor`, `masked_mse`, Adam·StepLR·GradScaler 상태·`Trainer.train_step`·checkpoint 저장/RNG 유틸을 재사용한다. 모델 hidden32/dropout0.1, batch2, seed37, 초기 lr0.01, StepLR50/gamma0.8, AMP=False, CPU thread1이다. Target scale1000mm는 loss 정규화·출력 배율이며 좌표 단위를 바꾸지 않는다. Loss는 유효 joint와xyz 전체 평균 `((prediction_mm-target_mm)/1000)^2`다.

중단 없는100 update를 수행하면서50시점 checkpoint를 저장하고, 새 진단 trainer가50부터100까지50 update를 재생한다. 합계150 update로200 한도 안이며 runner는300초를 협력적으로 검사한다. 설정을 바꿔 threshold를 맞추거나 더 긴 학습을 반복하지 않았다. `_preserve_state`로 CPU/Python RNG·thread·deterministic 설정·기존 CUDA 상태/환경을 복원한다.

## 결과와 의미

최종 코드의 결과는 `outputs/real_gt2d_diagnostic_20261102/developer/final_no_numpy/`이다. 앞선 초기 실행·`final`·root 독립 실행도 별도 경로에 보존하며 결과를 덮어쓰지 않는다. 입력 provenance 검사와 source identity 범위를 보완하고 tensor rawbytes hash를 PyTorch만으로 계산하도록 바꾼 뒤 같은 설정으로 재검증했다. 기존 tensor hash는 같고 새 의존성·성능 튜닝은 없다.

| 같은 두 sample의 train-fit 지표 | 초기 | 100 update |
| --- | --- | --- |
| 정규화 MSE | 0.1460779756 | 0.01588693075 |
| MPJPE mm | 618.1335685 | 188.4876985 |
| 유효 GT joint | 124 | 124 |

모든150 update의 loss/gradient가 유한하고 gradient L2는0.0047523~2.3700182, nonzero150/150이었다. 최종188.5mm는 좋은 정확도를 입증하지 않는다. 작은 실제 입력에서 optimizer까지 연결되고 같은 sample의 손실이 감소했음을 확인한 결과다. RGB 특징이나 예측 2D를 학습하지 않았고, held-out 오차·일반화·실시간 성능·전체 M1 완료로 해석하지 않는다.

T=1 batch와 각 record 단독 forward, 순서 변경, 다른 record 특징 교란을 대조했다. 세 비교가 모두 통과했고 실제 단독/batch 차이는0mm였다. 같은 sequence의 두 frame이 통계적으로 독립인 원본 motion이라는 의미는 아니다.

Runner 내부 경과는2.563초였다. 입력 대조·학습·checkpoint/재개 비교·상태 복원을 포함하고 Python/PyTorch 최초 import와 최종 report 쓰기는 별도다. 처리량 benchmark로 해석하지 않는다.

## Checkpoint와 근거

별도 format은 `real_gt2d_diagnostic_checkpoint_v1`, role은 `diagnostic_only_gt2d_single_frame`다. `diagnostic_only=True`, `train_ready=False`를 보존하고 일반 `m1_checkpoint_v1` reader는 거부한다. Model/Adam/StepLR/scaler/RNG/step/samples_seen/config·입력 identity·환경·source identity를 모두 저장한다. Source hash에는 `rewind_m1`, `rewind_data`, `rewind_prepare`, `rewind_archive`의 Python 파일을 포함한다.

`step_050.pt`, `uninterrupted_100.pt`, `resumed_100.pt`를 보존했다. 중단 없는100과50→100 재개 후 checkpoint의 전체 내용(tensor·optimizer·scheduler·scaler·RNG·학습 위치·data/code/environment)이 정확히 일치했다. 직렬화 컨테이너의 파일 SHA가 같다는 주장이 아니다. Resume은 동일 config/data/source identity/환경의 이 진단 경로로 제한하며 타 revision/환경의 자동 migration은 제공하지 않는다.

`report.json`은 `real_gt2d_single_frame_diagnostic_v1` schema와 `evaluation_kind=train_fit_diagnostic`를 사용한다. 두 source pin·원본 pose와 RGB 추출 provenance·관절 순서·spec·단위 상태·mask/tensor hash·seed/config·환경·각 update loss/gradient·재개/독립성 검사를 남긴다. `unit_status=adopted_metric_anchor_inference`와 원본 motion/validation/temporal/training-release blocker도 유지한다.

신규 진단 입구10개와 기존 M1 19개, 총29개 로컬 검사가 실패·오류·건너뜀 없이 통과했다. 최종 NumPy 비의존 hash 변경 후 관련10개도 모두 재검증했다. 명시 승인/pin, duplicate/nontrain, 특징 cache/GT target 변조, 관절 순서/source hash, 일반 loader의 candidate 거부, diagnostic checkpoint 역할, identity mismatch, T1 한도와 글로벌 상태 복원을 포함한다. 새 입구 검사는 torch/Pillow 실제 import와 검토된 로컬 sample이 필요하며, 이 실행에서는 모두 수행했다. 근거는 `outputs/real_gt2d_diagnostic_20261102/developer/{tests.json,tests.txt,no_numpy_tests.json,no_numpy_tests.txt,verification.json,code_hashes.json}`이다.

일반 학습 gate를 해제하지 않았고 입력 candidate/RGB·원본 cache/index도 변경하지 않았다. 새로운 사용자 Colab 시험은 요구하지 않는다. 후속 범위와 로컬 커밋은 오케스트레이션이 검토한다.

오케스트레이션도 최종 NumPy 비의존 코드로 `outputs/real_gt2d_diagnostic_20261102/root/reviewed/`에 독립 실행했다. 같은 지표와150update·full-state 재개·T1 독립성을 확인했고 신규10개 검사가 통과했다. 중단 후 재개 시 보존 보고서의 code hash와 현재 소스 및 두 checkpoint 전체 내용을 다시 대조했으며 학습을 반복하지 않았다. 최종 근거는 `outputs/real_gt2d_diagnostic_20261102/root/final_review.json`이다.
