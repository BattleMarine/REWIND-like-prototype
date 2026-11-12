# v1.4 단일 클립 oracle 암기 진단

2026-11-12 개발 계약. 사용자가 승인한 v1.4는 v1.1에서 관찰한 3D 형태 축소를 분리하는 실험이다. 같은 train 클립의 정답 2D와 hand용 정답 상체가 주어졌을 때 기존 teacher가 자세를 학습하는지 확인한다. 실제 GPU 학습 결과는 아직 없다. 부모 실험과 후속 판단은 [실험 계보](V1_EXPERIMENT_LINEAGE.md)를 따른다.

## 입력과 가정

기존 검증 train1,000페어에서 선택한 클립은 아래와 같다.

```text
ArchVisInterior_ArchVis_RT/Day/rp_adanna_rigged_003_ue4/
SKM_MenReadingGlasses_Shape_01/003/Standing_Up__1_
frame 0–49, train, 50 frames
```

`selection.json` content SHA는 `48e3c749c7d7710feb7fd61601b7406264837e7a67ceeb4a48787359825cafad`, raw cache manifest SHA는 `70241b27f66565593f0c8de0aeb6fe1f94c3672f932d7609bc882fd24a21d7b0`다. 기존 진단 결정·카메라 감사·overlay 검토의 고정 근거도 검증한다. v1.2 전체 데이터로 확대하지 않는다.

클립이 들어 있는 `part-0000.zip` 한 개는106,518,460bytes다. 원본 manifest의 크기·SHA로 검증하며 이 파트만 로컬에 복사한다. 50페어의 좌우 RGB·pose member 해시를 검사하고, 모든 pose의 좌표 변환·어안 재투영·관절·단위와 기존 카메라 수치를 대조한다. 학습에는 준비한 작은 텐서만 사용한다. RGB decode나 frontend 학습은 학습 update에 포함되지 않는다.

`oracle_input.json`은 150개 원본 파일의 path/size/SHA, 카메라 geometry 잔차, 고정 관절 spec, 선택 키, mask 집계, oracle 계약, 준비 텐서 SHA, 코드 hash를 보존한다. `oracle_clip.pt`와 `spec.json`은 이 기록에 연결된다. 원본 캐시의 과거 producer와 현재 코드의 검증을 구분하며 source pin을 무시해 이전 학습을 재개하지 않는다.

명목 시간은 frame index ÷25다. 실제 timestamp·stereo 동기·원본 motion 독립성은 미확인이다. 같은 train 클립을 반복 평가하므로 일반화·validation 지표로 사용하지 않는다.

## Oracle 입력 계약

좌우 뷰 각각의7개 특징을 관절별로 연결하여14개 값을 만든다.

```text
[x_normalized, y_normalized, valid, confidence, measurement_present, 0, 0]
x_normalized = 2*x/1023 - 1
y_normalized = 2*y/1023 - 1
```

유효한 annotation은 confidence와 measurement presence를1로 설정하고 누락은0으로 설정한다. 이 confidence는 정답 측정이 있다는 oracle 표시다. 화면 안 visibility를 뜻하지 않는다. 화면 밖의 유효 2D도 원래 좌표를 유지하며 clamp하지 않는다. 누락된 좌표만0으로 채운다. 2D mask와3D 정답 누락 mask는 각각 유지한다.

Body의 조건은 GT2D·카메라·diffusion timestep·time mask다. GT3D는 denoising 학습의 정답과 q-sample에 사용한다. Hand에는 GT2D와 같은 프레임의 GT 상체·유효 mask를 의도적으로 제공한다. 일반 teacher의 GT 거부 경로를 변경하거나 가짜 body checkpoint identity를 만들지 않는다.

## 학습·예산

설정은 `configs/rewind_v14_oracle_overfit.json`을 고정한다. Body/hand architecture, x0 및 frame 보조 MSE, target scale100mm, LR0.00005, FP32와 batch1의50프레임 조건을 유지한다. Velocity·foot loss는0이다.

Body는 seed71, hand는 seed72로 새로 초기화한다. 한 paired update에서 body와 hand를 각각 한 번 갱신한다. 5,000 paired update는 각 모델5,000회, 합계10,000 optimizer update다. 이전 모델을 이어 학습하지 않는다.

| 범위 | 상한·설정 |
| --- | --- |
| 준비 | 300초, 데이터 metadata·파트 복사·로컬 member 읽기 예약256MiB |
| 학습 호출 | 7,200초, 내부 종료 여유120초 |
| 마지막 전체 상태 저장 | 100 paired update마다 |
| 단계별 모델·평가 | paired0/200/1000/5000 |
| 실행 결과 디스크 | 2GiB, 단일 checkpoint512MiB |
| 회수 ZIP | 1GiB |
| 회수·복사 | 내부600초/외부630초 각각 |
| Colab 로컬 여유 공간 | 시작 전5GiB |

준비·학습·결과 보관은 각각 유한 호출이다. 학습의2시간에 준비·종료 후 ZIP 생성과 Drive 복사 시간을 합쳐 보장하지 않는다. 기존 T4 teacher 기록을 외삽하면 body+hand 계산은 약13분이지만, 새 실제 장치·준비·평가·checkpoint I/O 비용은 실행에서 확인한다. 각5,000 update가 수렴을 보장하지 않는다.

읽기 예약에는 원격 파트 전체 복사, 로컬 member payload, 고정 데이터 metadata와 ZIP 중앙목록/헤더의 반복 읽기를 포함한다. Drive의 실제 네트워크 전송량과 파일시스템 read-ahead는 별도 관측이며 논리 파일 읽기 예약값으로 동일시하지 않는다.

## 평가와 판정

모든 단계에서 동일 frame-key noise의 DDIM10 예측을 기록한다. GT target에 작은 노이즈를 넣은 k=1 직접 x0 복원은 별도의 정답 누출 진단으로 저장한다. 두 결과를 합산하지 않는다. 평가·시각화 전후에는 학습 RNG를 보존한다.

Body와 wrist-relative hand에 대해 MPJPE(mm), 평균 뼈 길이, 유효 관절 중심까지 Euclidean 거리 RMS spread, 처음/중간/끝(frame0/25/49) 오차와 첫·끝 변위를 저장한다. Gradient L2와 paired step wall-clock은 첫 update 및20회마다 기록한다. CUDA에서는 paired step 종료 시 동기화하고 실제 GPU·VRAM·precision·환경과 peak 할당을 보존한다. 시간은 모델 추론 latency 전용 측정이 아니다.

Hand global은 GT 손목에 부착한 oracle 지표와 예측 body 손목에 부착한 지표를 각각 기록한다. GT 손목이 누락된 프레임은 상대 좌표와 GT 손목 부착 지표에서 제외한다. 예측 손목 부착 지표는 유효한 전역 finger 정답을 사용한다.

전체 클립 GT의 관절별 평균 자세를 반복하는 상수 baseline도 기록한다. 이 baseline은 평가를 위한 전체 클립 정답 통계이며 모델 입력으로 사용하지 않는다. 평균 오차·형태가 개선돼도 상수 baseline, 세 프레임 오차, 실제 시간 변화까지 함께 검토한다.

Body20mm/hand-local10mm 이하, bone·spread의 GT 비율0.8–1.2는 프로젝트의 잠정 공간 복원 참고값이다. 자동 조기 종료나 논문 성능 판정 기준으로 사용하지 않는다. `spatial_reference_pass`, `better_than_constant_clip_mean`, `first_middle_last_below_error_limit`를 분리해 기록한다. 예산 내 미수렴만으로 구현 버그를 확정하지 않는다.

비교 PNG는 모델과 GT를 같은 투영·축척으로 표시한다. 합성 모델 전신 표시는 hand의 GT 상체 조건 및 예측 body 손목 부착을 명시한다.

## 실행·재개·자동 보관

CLI 진입점은 `python -m rewind_v1.oracle_cli`의 `source`, `prepare`, `train`, `recover`, `restore`다. `train`은 지원되는 CUDA GPU를 사용하며 이름을 A100으로 제한하지 않는다. 실제 GPU 실행은 고정본을 테스트 세션에 인계한 뒤 수행한다.

`last.pt`에는 두 모델·optimizer·scheduler·scaler·RNG와 완료 paired update, 평가·snapshot metadata를 저장한다. `run.json`의 관측 위치와 마지막 영구 저장 위치를 구분한다. 중간 한 모델의 update가 실패하면 마지막 완전 checkpoint에서 재개한다. 같은 source/config/input/environment가 필요하다.

회수 ZIP은 snapshot 모델, 작은 oracle 입력 텐서/spec, 마지막 전체 상태, 보고서, 예측 JSON과 PNG를 포함한다. raw RGB 파트·원본 캐시는 포함하지 않는다. 새 런타임의 `restore`는 ZIP SHA와 receipt의 파일별 크기/SHA, source와 input tensor를 검증한 뒤 새 폴더에 게시한다. 준비 텐서를 복원하므로 raw 파트를 다시 읽을 필요가 없다.

노트북은 실행 전 실제 Drive mount와 새 attempt 쓰기를 확인한다. 호출의 성공·실패·soft pause·일반 중단 후 finally에서 회수하고, `REWIND-like prototype/reports/v1_branch_experiments/v14_oracle/attempt-*`에 새 복사본을 검증한다. `state=verified`, `kind=training`, `last_sha256`과 원래 학습 상태를 함께 확인한다. 보관 실패는 별도로 표시하고 로컬 ZIP을 유지한다. 마지막 셀에서 새 attempt로 재시도할 수 있다.

브라우저 다운로드 기본값은False다. VM 자체가 종료되면 finally를 실행할 수 없으며, 실행 중 주기적 원격 보존을 보장하지 않는다.

## 고정 테스트 인계 초안

Root 검토·커밋 후 고정 source ZIP을 만들고 다음 생성기를 사용한다. 실제 파일 경로·revision·ZIP SHA와 notebook SHA는 생성한 `handoff_receipt.json`을 기준으로 인계한다.

```powershell
python -m scripts.build_archvis_v14_oracle --source-zip SOURCE_ZIP --source-sha256 SOURCE_SHA --revision REVISION --input-zip INPUT_ZIP --input-sha256 INPUT_SHA --evidence-zip EVIDENCE_ZIP --evidence-sha256 EVIDENCE_SHA --decision-sha256 DECISION_SHA --output NEW_HANDOFF_DIR
```

테스트 세션은 source/input/기존 진단 근거 ZIP을 `/content`에 올린 뒤 bootstrap을 실행한다. Drive raw cache 경로를 확인하고 RUN을 켠다. 재개 시 기존 준비 입력 SHA/last SHA 또는 새 런타임의 회수 ZIP/SHA를 사용한다. 두 방식을 섞지 않는다.

회수 대상은 Drive attempt의 receipt와 training ZIP이다. 학습 상태·완료 update·입력/환경·모든 단계 지표·예측/그림·모델/optimizer/RNG·보관 검증을 함께 취합한다. 시간/디스크/입력 pin 실패나 nonfinite loss·gradient에서 해당 실행을 중단하고 결과를 보존한다. 추가 update·LR 변경·새 클립 교체는 별도 실험으로 검토한다.

## 개발 검증 상태

Root는 신규11개 로컬 검사를 독립 재실행해 정상 종료로 통과했다. checkpoint SHA 보존과 nonfinite loss의 optimizer 갱신 차단 회귀를 포함한다. 의도적 KeyboardInterrupt 경로는 별도 child process의 비정상 종료와 finally 실행을 함께 확인한다.

확장 회귀29개는 최종 재확인에서21.086초, 종료 코드0으로 통과했다. 앞선 실행에서는 cold restore의 폴더 rename에 Windows `PermissionError(WinError 5)`가 한 번 발생했다. 해당 단독 검사와 전체29개 재실행에서는 재현되지 않았으며 원인은 미확정이다. 실패 로그와 재확인 로그는 `outputs/v14_oracle_development/regression_final.log`, `regression_recheck.log`에 각각 보존했다. 이 결과를 접근 거부 원인 해결로 해석하지 않는다.

미커밋 소스 검증용 ZIP으로 notebook bootstrap의 source/input/evidence ZIP 및 config/source pin, 기본 셀의 실행 비활성화를 확인했다. 검증용 ZIP은 정식 배포본이 아니다. `outputs/v14_oracle_development/final_builder_qa/validation.json`에 근거를 보존했다. 준비 논리 읽기 예약은228,690,928bytes로256MiB 이하이며, 파트 복사106,518,460bytes, 로컬 member payload 상한106,518,460bytes, ZIP metadata 상한1,445,770bytes와 입력 metadata14,208,238bytes를 포함한다. 이는 실제 Drive 전송량 실측이 아니다.

로컬 축소 모델·합성 텐서와 pose fixture로 입력 channel/mask, GT 상체 제한, 평가 RNG, 중단/재개, 회수·복원·복사 실패를 검증한다. 이는 실제50프레임 원본의 준비 완료나 CUDA 수렴 결과를 뜻하지 않는다. 실제 준비·CUDA 학습·Drive 보관은 고정본 테스트에서 확인한다.
