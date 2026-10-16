# M3 whole-body teacher — 합성 cascade 구현 검증

2026-10-15. M2 구현 커밋 `371589c` 이후 별도 승인한 M3 개발 묶음이다. 명명된 작은 합성 body를 학습·고정하고, 손 teacher의 GT 상체 학습과 예측 상체 평가를 연결했다. **실제 데이터 M3 완료나 정확도 개선을 입증한 결과가 아니다.** M0/M1 실제 데이터 gate와 D1~D5는 유지한다.

## 논문 근거와 구현 가정

[REWIND 논문](REWIND.pdf) 본문 p.4 식(3), 부록 p.12 S.3.1~2는 body/hand specialist를 나누고 손 모델에 상체 3D를 추가한다고 설명한다. Upper keypoints를 flatten하여 첫 층 Swish인 2층 MLP로 encoding하며, 손 학습에는 GT 상체, 평가에는 body가 예측한 상체를 사용한다. 앞선 M2에서 추출한 text·렌더로 해당 페이지를 다시 대조했다. PDF SHA-256은 `f9529af04e2efcf3f7ec1b197aca84dec82be65bf44caecbd777fd9ce3993aff`다.

이번 구현은 [M2의 schedule·attention·camera·anchor-window 가정](M2_BODY_TEACHER.md)을 재사용하며 다음을 추가로 선택했다.

- 손 좌표는 device 축과 평행한 **손목 상대 위치**다. Finger global device 위치는 `predicted body wrist + predicted finger offset`으로 복원한다. 손목 회전으로 local 축을 돌리지 않는다. 이는 논문에서 확정한 손 좌표 규격으로 주장하지 않는 프로젝트 선택이다.
- Body가 손목을 소유한다. Hand 모델은 손가락 관절만 예측하고 hand의 implicit wrist offset은 정확히 `[0,0,0]`이다. Body wrist를 hand loss나 whole-body 평균에 다시 더하지 않는다.
- 상체 MLP 입력에 좌표와 valid mask를 함께 제공한다. 누락 GT의 NaN을 condition encoding 전에 제거한다. Unknown visibility를 정답 누락으로 해석하지 않는다.
- **세 평가 모두 global 결합에는 예측 body 손목을 사용**한다. GT 상체 진단도 MLP cue만 GT이며 GT 손목을 global 결과에 붙이지 않는다. `condition_source`와 `attachment_source`를 각각 기록한다.
- `none`은 상체 MLP 자체가 없는 별도 hand 모델이다. 조건 모델에 0을 넣은 결과를 조건 없음의 학습 실험으로 간주하지 않는다. `none`도 global 결합에 예측 body wrist가 필요하므로 body 추론을 생략하는 실험은 아니다.

## 명명된 합성 규격

원본 문서의 `m1_features_v1.contract.coordinate_frame=device_relative`, 단위 mm를 유지한다. M1/M2 관측·단위·split·camera guard를 먼저 통과한 뒤 M3 내부 `HandSplit.coordinate_frame=wrist_relative_parallel_device` tensor를 만든다. 손목 상대 target을 device-relative라고 잘못 표기하여 기존 로더에 전달하지 않는다.

| 집합 | 이름/순서 |
| --- | --- |
| Body 6 | `root, head, shoulder_l, shoulder_r, wrist_l, wrist_r` |
| Upper 5 | `head, shoulder_l, shoulder_r, wrist_l, wrist_r` |
| Left hand | `wrist_l, index_l, tip_l` — wrist는 body 소유 |
| Right hand | `wrist_r, index_r, tip_r` — wrist는 body 소유 |
| Hand 모델 4 | `index_l, tip_l, index_r, tip_r` |
| Whole-body 고유 10 | Body 6 + finger 4 |

이 이름은 코드 검사용 합성 규격이며 실제 body/hand skeleton이 아니다. M2의 이전 `root/tip` checkpoint를 해부학적 body로 재해석하지 않았다. 새 body 6관절 문서를 생성해 M2 teacher를 따로 학습했다. 실제 손 40개·전체 72/87개 annotation을 학습하거나 해당 범위를 완료한 것은 아니다.

원본 whole 문서는 `m3_contract`에 `version=m3_roles_v1`, body/upper 이름 순서, 좌우 `wrists`, wrist를 첫 이름으로 포함한 좌우 `hands`, hand 좌표와 `wrist_owner=body`를 명시한다. Body와 finger coverage·소유 중복, 좌우 wrist 중복, upper가 body subset인지 검사한다. 전체 예시는 `rewind_m3.data.synthetic_document()`가 만든다. 영상은 64×48 규격의 인위적인 선형 2D–3D 대응이며 실제 RGB/어안 projection이 아니다.

Local target mask는 **finger GT valid AND 해당 wrist GT valid AND time_valid**다. GT wrist가 누락되면 그 손의 local target 전체가 무효다. 반면 finger의 global GT가 유효하면 global hand 평가는 유지하므로 원본 annotation mask를 임의로 줄이지 않는다. 손목 상대 변환의 0은 false mask의 자리표시자다.

## 모델·고정 body와 데이터 흐름

| 경로 | 역할 |
| --- | --- |
| `rewind_m3/data.py` | Whole device 문서 검증, 이름 기반 body 문서, 명시적 local target/mask 생성 |
| `rewind_m3/models.py` | M2 graph/spatial/RoPE block 재사용, upper 좌표+mask 2층 MLP, DDIM에 고정 조건 전달 |
| `rewind_m3/cascade.py` | Frozen body checkpoint 검증·추론, seq/frame cache 대조·출처 guard, 손목 결합 |
| `rewind_m3/training.py` | GT upper/none 학습, 예측/GT 진단 평가, hand 완전 checkpoint와 공통 identity 검증 |
| `rewind_m3/metrics.py` | Body, global/local hand, 좌우 wrist, 고유 whole-body 지표 |
| `rewind_m3/streaming.py` | Body current 추론 → 예측 history → hand current의 두 단계 streaming |
| `rewind_m3/__main__.py` | 개별 train/evaluate/resume CLI |
| `rewind_m3/fixture.py` | 명명된 body 학습·고정 + upper/none 학습 + 세 평가를 CPU에서 실행 |
| `configs/m3_*_synthetic.json` | 별도 body·upper hand·none hand tiny 설정 |
| `tests/test_m3.py` | 좌표·출처·누출·불변성·causality·streaming·완전 재개/CLI 검사 |

`FrozenBody`는 M2 checkpoint를 실제로 읽고, 같은 body 문서 hash·config·이름/순서·단위/scale·camera/skeleton 계약·코드·환경을 확인한다. Body는 eval/`requires_grad=False`로 고정한다. 모델 state hash와 checkpoint 파일 SHA-256을 별도로 기록하고 학습·평가 후 weights/buffers·grad·mode 불변을 검사한다. Hand optimizer에는 body parameter를 넣지 않는다.

Body의 각 원본 frame은 **자신의 body anchor window**로 추론한다. 그 예측을 `(sequence_id,frame_index)`에 연결하고 hand window의 모든 실제 slot을 정확히 대조한다. Cache의 frozen identity·payload hash·source를 검사하며 누락 frame을 GT로 대체하지 않는다. Prediction mask는 실제 관측 time slot의 모델 출력 여부이며 GT body mask나 visibility를 복사하지 않는다.

손 학습은 `gt_upper_train`, 조건 모델의 실제 cascade 평가는 `predicted_frozen_body`, 같은 조건 모델의 진단은 `gt_upper_diagnostic`다. Train API에 prediction-tagged condition을 넣거나 GT condition을 prediction 출처로 취급하면 거부한다. `none` 모델에 upper tensor를 넣거나 조건 평가를 요청해도 거부한다. 출처 태그는 암호학적으로 tensor의 의미를 증명하지 않으므로 외부 임의 cache를 가져와 학습하는 기능은 제공하지 않는다.

예측 경로는 3D target을 참조하지 않고 관측 2D/camera·고정 body·noise로 계산한다. 실제 검사에서 body/hand의 GT 3D target만 메모리에서 바꿔도 body cache hash와 body/local/global/whole 예측 tensor가 정확히 같았다. GT cue 진단의 출력은 GT upper 변경에 반응했다. 지표는 수정한 정답에 따라 달라질 수 있으므로 **예측값 불변과 지표 불변을 혼동하지 않는다.**

## Loss·평가·streaming

Hand position/보조 frame loss는 M2와 같은 normalized masked MSE다. mm local target을 고정 100mm scale로 나누고 유효 joint/xyz/window slot 전체 평균으로 reduction한다. Velocity/foot/contact loss는 적용하지 않는다. Teacher hidden32, spatial1/temporal2, heads2, DDIM4, K1000의 tiny 설정이며 논문 규모가 아니다.

평가 평균은 유효 관절별 Euclidean 거리 합/개수다. PA 정렬을 하지 않는다. Body는 6개, local/global hand는 손가락 4개, 전체는 중복 없는 10개를 집계한다. 각 손의 global/local 오류와 body wrist GT 오류도 나눈다. 따라서 body wrist 오차가 global finger 오류에 전파되며 local 오류와 구분된다. `body_wrist_attachment_error_mm=0`은 별도 예측 손목이 없는 구조에서 결합이 동일하다는 **구조적 값**이며 실제 손목 정확도 0의 의미가 아니다.

상체 조건 없음/GT 진단/예측 cascade에서 모두 `attachment_source=predicted_frozen_body_wrist`다. GT 진단을 실제 예측 입력 baseline으로 보고하지 않는다. 입력 2D도 현재 fixture에서는 GT 합성이므로 모든 결과를 최우선 `synthetic_code_check`로 표시한다.

Streaming은 body와 hand 각각 고정 anchor window를 재계산한다. 과거 body 예측을 hand 입력 history에 보관하므로 offline도 같은 원본 frame별 body 예측 cache를 사용한다. Body window 내부의 과거 출력 전체를 매 hand window에서 새롭게 정의하지 않는다. 두 단계의 raw input 영향 범위는 hand window 하나보다 넓을 수 있다. KV/DDIM state cache나 전체 sequence 단일 실행과의 동일성을 주장하지 않는다.

Noise는 M2의 frame-key SHA-256 seed 정책이며 body/hand seed offset을 나눈다. Gap·새 sequence·명시적 reset은 양쪽 history를 초기화한다. 각 history는 최대 4개이며 새 input 거부/hand 오류 시 이전 body state도 보존한다. 실제 코드 검사에서 전체 24 frame의 독립 offline 예측과 streaming을 대조했고 gap·새 container reset·추가 36개 연속 입력의 bounded history를 확인했다. 장시간 실제 스트리밍 성능 검증은 아니다.

## 로컬·Colab CPU 실행

추가 의존성 없이 기존 PyTorch 환경을 사용한다. 이번 지원·실행 범위는 Python 3.12.14/torch2.7.1+cpu, CPU float32/thread1이며 M3 CLI는 CUDA/AMP를 명시적으로 거부한다. GPU 지원을 검증했다고 표시하지 않는다. 아래 명령은 프로젝트 root에서 실행하며 이번 Windows의 `python`은 `.venv/Scripts/python.exe`다.

```sh
python -m unittest discover -s tests -v
python -m rewind_m3.fixture --output outputs/m3_named_new --body-steps 80 --hand-steps 100
```

Fixture는 새 output을 요구하고 `whole.json`·`body.json`, `body/`, `hand_upper/`, `hand_none/`, 세 평가 JSON과 `comparison.json`을 만든다. 실제 archive를 읽거나 외부 데이터를 다운로드하지 않는다. Colab에서도 같은 모듈/소스가 준비된 CPU 환경에서 실행할 수 있도록 명령을 제공했으나 **초기 개발 검증에서는 실제 Colab 실행을 하지 않았으며, 이후 사용자 CPU 실행은 문서 끝의 후속 검토에 기록한다.**

학습 위치를 110 updates까지 이어가는 예시는 다음과 같다. Body checkpoint는 바꾸지 않는다. `--steps`는 추가 수가 아니라 총 update 수다.

```sh
python -m rewind_m3 train --config configs/m3_hand_upper_synthetic.json --dataset outputs/m3_named_new/whole.json --body-checkpoint outputs/m3_named_new/body/last.pt --output outputs/m3_named_new/hand_upper --resume outputs/m3_named_new/hand_upper/last.pt --steps 110 --device cpu
python -m rewind_m3 evaluate --checkpoint outputs/m3_named_new/hand_upper/last.pt --body-checkpoint outputs/m3_named_new/body/last.pt --dataset outputs/m3_named_new/whole.json --condition predicted_frozen_body --output outputs/m3_named_new/resumed_prediction.json --device cpu
```

GT 진단은 같은 upper checkpoint에 `--condition gt_upper_diagnostic`, 조건 없음은 **hand_none checkpoint**에 `--condition none`을 사용한다. 개별 학습을 새로 시작하려면 새 output을 지정한다. Fixture에서 생성한 `comparison.json`은 당시 checkpoint hash/step을 기록한 정적 결과이며, 이후 개별 hand 재개 결과로 자동 갱신되지 않는다. 비교용 checkpoint를 보존하고 새 평가를 별도 파일로 기록한다.

## Checkpoint·검증 결과

`m3_checkpoint_v1`에는 hand model, optimizer/scheduler/scaler, Python/Torch CPU·CUDA RNG, step/samples_seen, config·manifest·원본 device 계약과 내부 hand 계약, 역할 매핑, schedule과 **frozen body checkpoint/weights/data/config/source identity**를 저장한다. Body weights는 원본 별도 checkpoint에 유지한다. 재개·평가 모두 같은 불변 identity 검사 함수를 거치며 다른 body weights·계약·schedule·data/code를 거부한다. Eval은 hand optimizer/RNG를 복원하지 않는다. Resume는 M1의 엄격한 environment/RNG/full-state guard와 주기 저장·report recovery 정책을 유지한다.

최종 **119 tests = 기존107 + M3 12** 통과. 다음을 검사했다.

- GT wrist 누락의 local mask, 원본 global mask 보존, 좌우 매핑·고유 관절 count와 원점/중복 wrist 정책.
- Upper/none 각각 100 updates의 작은 과적합·finite gradient와 upper MLP gradient, frozen body weights/hash/grad/mode 불변.
- 잘못된 조건 source·cache 변조·누락 frame/GT fallback 거부, GT target만 수정한 prediction 불변.
- 미래 2D/camera/noise/upper 변경에 대한 전체 hand DDIM 과거 출력 불변, 전체 cascade의 미래 관측 변경 불변·padding NaN 격리.
- Offline/streaming·sequence/gap/reset·bounded history와 dropout 포함 hand 5+7 vs 연속12 updates의 model/Adam/StepLR/scaler/RNG/위치 일치.
- 실제 CLI의 세 경로·재개와, 같은 config/data이지만 body weights만 바꾼 checkpoint를 지정한 평가의 거부/출력 미생성.

Git 제외 증거는 `outputs/m3_evidence_20261015/developer/`의 `execution.json`, `command_*_stdout/stderr.txt`, `pipeline/whole.json`·`body.json`·각 checkpoint/report·세 평가·`comparison.json`이다. 각 명령 60초 상한, 네트워크 다운로드 0이다. Body80·upper100·none100 실행과 upper110 재개·별도 평가가 모두 exit0이며, 비교용 upper100 checkpoint는 `hand_upper/at_100.pt`에 보존했다.

단일 seed·동일 인위적 생성 규칙의 validation 코드 검사값은 다음과 같다. 같은 frozen body를 사용하고 아래 손 모델은 모두 100 updates다.

| 평가 경로 | Body MPJPE mm | Hand wrist-relative mm | Hand global mm | 고유 whole-body mm |
| --- | ---: | ---: | ---: | ---: |
| 별도 학습 none | 6.45289 | 3.05592 | 8.17401 | 7.14134 |
| GT upper 진단 | 6.45289 | 2.96512 | 8.58717 | 7.30661 |
| 예측 upper cascade | 6.45289 | 2.97882 | 8.57019 | 7.29981 |

조건 모델이 모든 지표에서 우수하지 않다. 이 숫자로 상체 conditioning 효과·실제 데이터 일반화·정확도 개선을 입증하지 않는다. Body train normalized MSE는 `0.888439→0.00147705`(80 updates), upper의 예측-condition train local NMSE는 `0.200978→0.000342802`(110 updates), none은 `0.353100→0.000401655`(100 updates)다. 재개와 과적합 코드의 동작 근거이며 서로 다른 step의 성능 비교로 사용하지 않는다.

오케스트레이션 초기 검토에서 evaluate가 resume의 frozen identity 검사를 우회하던 문제를 발견해 공통 검사와 실제 wrong-body CLI 회귀를 추가했다. M1/M2/archive/raw 코드는 변경하지 않았다. 오케스트레이션 검토 후 이 합성 구현 묶음만 로컬 커밋하며 자동 push는 하지 않는다.

오케스트레이션의 독립 검토는 `outputs/m3_root_review/review.py`·`review.json`에 기록했다. 저장된 upper110 checkpoint를 불러와 GT 좌표와 valid mask 모두를 바꿔도 예측 cascade가 정확히 동일하고 GT 진단만 반응함을 확인했다. Validation 12 anchor의 batch/stream 최대 성분 차이는 `3.57628e-5 mm`, body/hand history는 각각 최대 4다. 왼쪽 손목 이동의 왼손가락 전파, 결측 GT 손목의 local mask 제외와 global GT 유지, frozen body 불변을 독립 대조했다. 이 수치는 실제 정답 대비 정확도가 아니라 같은 합성 추론 경로의 수치 일치 범위다. 최종 checkpoint에 기록된 31개 Python 파일의 hash를 현재 코드와 대조했다.

실제 skeleton·body/hand 대응 RGB/예측 2D·camera/단위/시간 gate, 실제 데이터의 GT 상체 기반 hand 학습·예측 body 기반 평가, 손 40개·전체 body-hand 범위·failure benchmark, 여러 seed/논문 규모·GPU/AMP/VRAM/latency·Colab GPU 실행은 남아 있다. 이번에는 합성 구현 묶음을 검토 가능하게 완료했으며 M3 마일스톤 전체 성능 완료를 선언하지 않는다.

## 사용자 Colab CPU 검증 (2026-10-16)

사용자가 실행한 notebook과 결과 ZIP을 2026-10-16에 재검토했다. 이전 절의 Windows 개발 실행과 별개이며 다음은 제공된 Colab 실행 산출물에 근거한다.

- `m3_check_0c3343b1.zip`: 1,797,066 bytes, SHA-256 `9c88fc5ec2facc2027bd6b6a72f183735a15d8507591d9e23eef7a587f5185ab`.
- 실행 `m1_baseline.ipynb` SHA-256: `94683cc85f327e5a924932185f1720493bda6c2f4183c413ece665f9b9416e17`.
- 환경: revision `5e52aa3b0b1e27e7d0cb2697629e8acfc63d38c3`, Python 3.13.16 / torch 2.11.0+cpu / CPU float32 / thread1 / AMP off.
- `tests.log`: 119 tests, 37.687초, OK, 실패·skip 없음. Notebook에 저장된 오류 출력도 없다. Notebook에는 별도 M1 합성 셀 실행이 있으나 M3 결과 경로와 분리되어 있다.
- Body80 → 고정 → upper/none hand 각각100 → 세 조건 평가를 확인했다. 비교용 upper100은 `at_100.pt`, 최종 upper110은 `last.pt`로 보존되어 있다. 재개 구간은 `(0,100)`·`(100,110)`이고 원 checkpoint SHA와 일치한다.
- 보고서의 소스 31개 파일 hash를 배포 소스 ZIP의 bytes와 대조했다. 평가가 참조하는 checkpoint 파일 SHA, frozen body identity와 실제 저장 파일이 일치한다. Checkpoint의 step·model 유한 값·optimizer/scheduler/scaler/RNG 저장도 확인했다. 로컬 환경에서 Colab checkpoint를 재학습하거나 환경 불일치 guard를 우회하지 않았다.
- ZIP CRC와 원본 보관 사본 hash를 검사했다. 결과 검사는 보고서의 상호 일치 확인이며 실행 출처의 암호학적 인증을 뜻하지 않는다.

100 update validation의 손목 상대/global hand MPJPE(mm)는 none `3.055925/8.173986`, GT upper 진단 `2.965118/8.587166`, 예측 upper `2.978817/8.570183`이다. Upper110 재개 후에는 `3.104896/9.107649`다. 추가 update에서 오차가 증가했지만 재개 실패가 아니며, 재개 검증과 성능 향상 검증을 구분한다. 모든 결과는 `synthetic_code_check`다.

사용자는 결과·notebook·소스를 Drive에 업로드했다고 확인했다. 안내한 보관 위치는 `REWIND-like prototype/reports/m3_colab_validation_20261015`이며 실제 Drive 경로·사본 bytes를 이번 검토에서 직접 조회하지는 않았다. 로컬 검토 원본과 요약은 Git 제외 `outputs/colab_m3_review_20261016/`에 보존한다. 해당 소스 ZIP을 다시 실행하거나 이 문서 커밋에 맞춰 기존 결과의 revision을 변경할 필요는 없다.

Colab CPU 합성 실행·학습 재개는 확인했다. 실제 데이터 규격·학습·일반화 성능, 예측 RGB front-end baseline, GPU/AMP/VRAM·latency 검증은 계속 남아 있다.
