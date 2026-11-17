# v1.5.2 복원·생성 과정 진단

2026-11-17 사용자 승인에 따른 `V152-SAMPLING-DIAGNOSTIC-001` 개발 계약이다. 구현·로컬 검증 단계이며 실제 CUDA 실행 결과는 없다. [v1.5 결과](V15_CAMERA_TRAINING_RESULTS.md), [v1.5.1 결과](V151_UPPER_CONDITION_RESULTS.md)를 부모 근거로 사용한다. 새 학습, 원점 이동·도시 전이·기존 6클립 회귀 반복은 포함하지 않는다.

## 고정 입력

[공통 입력 구현](../rewind_v1/train_clip_bundle.py)은 부모 ZIP 전체 SHA, 90파일의 크기·SHA/CRC와 JSON seal을 검증하고 첫 train50의 저장 예측 F9 특징·GT·카메라 R/t 및 중점 body/hand u0·u200·u1000을 원본 bytes로 복사한다. 기존 optimizer·raw RGB·raw cache·frontend 재추론은 사용하지 않는다. 첫 train 소속은 부모 train report keys와 feature manifest 및 evaluation.train_fit key로 대조한다. Validation/test로 대체하지 않는다.

| 식별값 | 값 |
| --- | --- |
| 부모 revision | `4f24f3aa2265c2ac54c21600b57bdece4e540411` |
| 부모 ZIP SHA | `a46e6a65f3777469d9f70d9aa99bf0aadf5bb5972ad06bb488c5b96f0944a170` |
| 부모 source SHA | `10e25cd2cb6f5b8d5dd4242481bf1aca349869f7000985dba28e82214c9a4540` |
| 첫 train tensor SHA | `3cf5a31747b6eb2e43fb661a97d67c3432eb4f7f3de3ae5d2712392580aba07e` |
| 공통 bundle | `outputs/train_clip_diagnostics/input_bundle.zip`, 368,252,243B |
| bundle ZIP SHA | `9b898777a61613da6f14537ec55a00973f4b77c2c3519e59643bb82308d187c2` |
| bundle manifest SHA | `a4d6c655f703250e5c2f3187de4f32e9098e78904d06f87793c4e379ac247353` |

클립은 `ArchVisInterior_ArchVis_RT/Day/rp_adanna_rigged_003_ue4/SKM_MenReadingGlasses_Shape_01/003/Standing_Up__1_`의 frame0..49다. 부모의 feature identity·spec·source·카메라 원본 값·snapshot binding을 검사한 뒤 첫 frame stereo midpoint 표현을 적용한다. 원본 timestamp·stereo 동기·motion 독립성의 미확인 상태를 새로 검증 완료로 바꾸지 않는다.

## 비교와 해석

[고정 config](../configs/v152_sampling_diagnostic.json), [진단 코어](../rewind_v1/train_clip_analysis.py), [runner](../rewind_v1/train_clip_runs.py)를 사용한다. 모델 표현은 normalized **x0**, 출력은100mm를 한 번 곱한 device mm다. FP32, seed71, body/hand noise offset100/200, sequence/frame keyed epsilon을 모든 checkpoint·sampling step에 재사용한다. 손은 학습 조건과 같은 GT upper를 고정한다. GT upper 안의 wrist 정보와 GT wrist 사후 부착은 진단 조건이며 배포 성능이 아니다.

- u1000: `k=1,100,500,1000`의 GT-noised 입력에 대한 direct x0 복원. `alpha_bar`·SNR과 frame head를 함께 저장한다. k1000도 clean 성분이 남으므로 pure-noise 생성과 동일하지 않다.
- u1000: pure-noise deterministic DDIM `1,10,50` step 생성. 기존 `DiffusionSchedule.sample`을 그대로 호출하고 wrapper로 매 단계 x0/frame head를 관측한다.
- u0/u200: 같은 epsilon의 pure-noise DDIM10만 수행한다. u1000 DDIM10 결과는 재사용한다.

모든 조건에 body·hand local MPJPE/count/sum·프레임별 분모, bone/spread, 첫·중간·끝 frame 오차, 첫-끝 변위를 저장한다. GT clip 평균의 constant pose는 정답을 사용하는 참고 기준선이다. Predicted wrist global은 common local mask와 native finger mask를 구분한다. GT wrist 부착 global은 finger&wrist 유효 공통 mask로만 보고한다. 누락 mask와 visibility를 바꾸지 않는다. 모델과 입력 tensor fingerprint를 전후 비교해 exact 불변 여부를 기록한다.

저장 예측 JSON과 첫·중간·끝 PNG를 출력한다. Whole body와 hand local 각각에서 선택 frame 전체 GT/예측 공통축을 사용하며 실제 bounds를 기록한다. Hand local 패널의 양 wrist는 각각의 원점을0에 겹쳐 표시한다. 이것은 두 손의 실제 global 배치가 아니다. 렌더용 wrist-root edge는 지표의 finger-only edge를 바꾸지 않는다.

## 예산·보관·실행

입력512MiB/64member·600초, 전체 추론/trace/지표/render3600초, 결과 파일합/ZIP128MiB. Disk free8GiB·Linux MemAvailable4GiB·관측 RSS4GiB를 확인한다. ZIP package600초와 Drive copy/readback600초는 별도 한도이며 실측 wall과 구분한다. 원본 ZIP 검증의 logical read 상한은4GiB, 600초다. 실제 생성 시3,498,244,961B·4.77초를 관측했으며 다른 장치 비용을 보장하지 않는다.

`python -m rewind_v1.train_clip_cli --experiment v152 source`로 source identity를 확인한다. `bundle`은 로컬 고정 부모에서 공통 입력을 만들고 `prepare`는 지정한 source/bundle/manifest SHA를 확인한다. 실제 사용자는 오케스트레이션이 고정 revision·source ZIP SHA·notebook SHA를 확정하여 테스트 세션에 인계한 배포본을 따른다. 현재 문서의 계획만으로 CUDA 실행을 시작하지 않는다.

[빌더](../scripts/build_train_clip_diagnostics.py)는 `--experiment v152`로 독립 notebook/receipt를 생성한다. RUN 기본False·CUDA 기본이며 CPU는 개발 fixture에만 사용한다. Drive mount/쓰기 preflight 후 실행하고 종료 finally에서 새 결과 경로에 ZIP을 복사·크기/SHA를 재검증한다. 실행 성공과 보관 성공은 별도로 기록하며 실패시 로컬 결과를 유지하고 자동 재시도하지 않는다. SIGINT 저장대기60초 후 미종료시 kill/wait30초다. 브라우저 단절과 VM 소실은 다르며 VM 종료 시 보존을 보장하지 않는다.

회수 대상은 experiment/run.json·predictions/PNG·operation/input status·recovery receipt다. 원본 부모 가중치와 입력 bundle은 결과 ZIP에서 제외한다. 로컬 fixture·원본 입력 검사와 실제 GPU 진단을 구분하며 결과 미회수 상태를 완료로 표시하지 않는다.

## 로컬 검토와 고정 배포 진입점

2026-11-17 구현·root 검토: 두 실험의 공통 입력 검증, 원본 모델 strict load, DDIM 추적 동등성, 중단/재개 상태 일치, 실패 결과 보관 및 기본 실행 OFF 노트북을 준비했다. 관련 CPU 회귀30개가 통과했다. 실제 GPU·Colab·원격 Drive 실행과 새 VM 복원은 미검증이다.

Root는 실제 u1000 모델과 첫2프레임의 DDIM10 wrapper/원본 출력을 비교해 body·hand 최대차이0을 확인했다. 공통 bundle12개 선택파일의 CRC/SHA와 개발 fixture의 whole/손local 그림을 확인했다. 검토 JSON·테스트 로그·최종 source/notebook 식별값·실행 순서는 [고정 테스트 인계](../outputs/train_clip_diagnostics/TEST_HANDOFF.md)에 보존한다. 이 로컬 검증을 정확도 개선이나 실제 학습 완료로 해석하지 않는다.
