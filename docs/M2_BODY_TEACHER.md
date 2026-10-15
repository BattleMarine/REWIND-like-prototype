# M2 causal body diffusion teacher — 합성 구현 검증

2026-10-15. M0·M1의 실제 데이터 진입 조건을 유지하며, 별도 승인한 데이터 없는 M2 구현 묶음이다. 실제 UnrealEgo body teacher의 학습·성능 완료를 뜻하지 않는다. M3는 오케스트레이션의 M2 검토·로컬 커밋 후 별도 진행한다.

## 논문 근거와 프로젝트 선택

[로컬 REWIND 논문](REWIND.pdf)의 본문 p.3~5, 부록 p.12~13 S.3을 읽고 p.4·5·12·13을 렌더링해 수식·표기를 대조했다. 파일 SHA-256은 `f9529af04e2efcf3f7ec1b197aca84dec82be65bf44caecbd777fd9ce3993aff`다. PDF의 사실과 코드의 선택을 구분한다.

| 항목 | 논문에 명시된 내용 | 이번 구현과 제한 |
| --- | --- | --- |
| 입력·출력 | Noisy 3D, stereo 2D·confidence, camera poses, diffusion time → clean 3D | 기존 M1의 유효성·confidence/visibility known bit도 보존. 실제 RGB front-end를 새로 학습하지 않음 |
| Camera | R의 6D 표현 + translation, 2층 MLP/첫 층 Swish | `x_camera=R@x_world+t` column-vector transform을 **프로젝트 입력 계약**으로 선택. R의 0·1열을 순서대로 연결하고 mm translation을 고정 scale로 나눔. Raw UE Euler를 자동 변환하지 않음 |
| Spatial | Graph convolution + spatial self-attention, temporal 이전 보조 pose head | 명시적 undirected edge+self-loop의 row-normalized GCN, pre-LayerNorm spatial MHA·FFN·보조 linear head. 정확한 저자 내부 block의 복제라는 주장은 하지 않음 |
| Temporal | RoPE causal relative attention, 식(4)의 과거 구간 `[j-w_s,j]` | Per-joint scaled-dot-product **softmax** attention과 FFN을 선택. 식(4)에 보이는 feature-map 비율 계산을 그대로 복사한 구현은 아님. RoPE·층별 local causal mask를 명시적으로 검사 |
| Diffusion | DDPM 학습, x0 motion loss, K=1000, cosine β endpoints 0.0001~0.02 | `β_k=β_start+(β_end-β_start)*(1-cos(π*(k-1)/(K-1)))/2`를 선택. 논문이 정확한 cosine 공식을 제시하지 않아 이 **beta interpolation은 가정**이다. Alpha-bar cosine schedule과 혼동하지 않음 |
| 추론 | Teacher DDIM 10 steps | Deterministic DDIM `eta=0`, descending k와 최종 clean endpoint를 명시. Tiny 검사는 4 steps. Teacher를 1회 호출한 결과는 증류 student가 아님 |
| Loss | Position + velocity + foot + intermediate frame loss | 우선 normalized masked position MSE + frame auxiliary MSE, weight=1. Timestamp·world/contact 미검증으로 velocity/foot은 구현·적용하지 않음 |
| 학습 규모 | 512차원, graph 3층, temporal 3층, heads 4, ws=20, 50프레임, Adam 5e-5, 2M steps | Tiny 32차원, graph 1층, temporal 2층, heads 2, anchor 4 slots, attention past 2, Adam .002, 100+10 updates. 논문 규모 실행은 하지 않음 |

## 코드·입력 계약

| 파일 | 역할 |
| --- | --- |
| `rewind_m2/data.py` | M1 guard·정규화·mask 재사용, camera proper rotation 검증과 skeleton 확장 |
| `rewind_m2/models.py` | Camera/time 2층 encoder, GCN/spatial attention, intermediate head, RoPE temporal denoiser |
| `rewind_m2/diffusion.py` | 명시적 schedule, forward diffusion과 DDIM x0 sampling |
| `rewind_m2/streaming.py` | Frame 키별 고정 noise, bounded anchor-window 재계산·gap/reset |
| `rewind_m2/training.py` | Train-only loss/gradient, DDIM anchor 평가, 전체 학습 상태·재개 guard |
| `rewind_m2/__main__.py` | Train/evaluate CLI와 M1 방식의 주기 checkpoint·보고서 구간/복구 기록 |
| `configs/m2_body_synthetic.json` | 실행 가능한 CPU tiny 설정 |
| `tests/test_m2.py` | 수학·mask·causality·streaming·재개·작은 합성 과적합 검사 |

Dataset 입력은 기존 `m1_features_v1`에 `m2_contract`와 각 record의 `camera_pose`를 추가한다. M1의 관절 순서·영상 크기·단위·출처·예측 cache provenance, sequence/원본 motion split guard를 우회하지 않는다. 예측 입력을 GT로 대체하지 않는다. 코드의 `synthetic_document()`가 완전한 합성 예시를 만든다.

```json
{
  "m2_contract": {
    "camera": {
      "transform": "world_to_camera_column_vector",
      "rotation_6d": "matrix_columns_0_then_1",
      "translation_unit": "mm",
      "translation_normalization": "fixed_target_scale_mm",
      "views": ["left", "right"]
    },
    "skeleton": {
      "version": "synthetic_two_joint_v1",
      "joint_names": ["root", "tip"],
      "edges": [["root", "tip"]]
    }
  }
}
```

각 record는 좌우 `camera_pose.left/right.rotation_matrix[3,3]`·`translation[3]`을 제공한다. Finite·shape·`RᵀR≈I`·det≈+1을 허용 오차 1e-6으로 검사하고 reflection·NaN·camera 누락을 거부한다. 실제 dataset은 M1의 여섯 verified/evidence guard에 더해 `m2_contract.verification_evidence.camera_transform`·`skeleton`의 비어 있지 않은 근거가 필요하다. 이 선언 검사는 실제 규격의 진실성을 자동 증명하지 않으며, 현재 실제 입력을 통과시키기 위해 값을 채우지 않았다.

추가 camera tensor는 `[N,T,2,9]`이고 padding은 0이다. 기존 2D tensor `[N,T,14J]`의 xy·valid·confidence/known·visibility/known을 유지한다. GT confidence와 visibility는 unknown이다. 화면 밖 2D를 clipping하거나 GT 3D mask에서 제외하지 않는다. 3D target은 `[N,T,J,3]` mm이며 모델 내부의 clean/noisy/output은 고정 `target_scale_mm=100`으로 나눈 좌표다. 전체 sequence·미래·test 통계로 정규화하지 않는다.

합성 2관절 `root/tip`은 해부학적 body skeleton이나 손목이 아니다. M3에는 body upper subset·좌우 wrist 이름·hand 매핑 및 frozen body checkpoint의 출처를 별도 명시해야 한다. 현재 checkpoint의 이름 순서·edge를 해부학적 규격으로 승격하지 않는다.

## Diffusion·loss·평가

`alpha_bar[0]=1`, `alpha_bar[k]=∏(1-beta[1:k])`이며 `q_sample(x0,eps,k)=sqrt(alpha_bar[k])*x0+sqrt(1-alpha_bar[k])*eps`다. 학습 k는 anchor batch마다 1~K에서 뽑아 해당 window 전체에 적용한다. GT mask 밖 NaN을 `torch.where`로 **q_sample 전에** 제거한다. Padding noisy token은 0이다. Mask 누락을 visibility 정답으로 해석하거나 missing GT 값을 model condition으로 제공하지 않는다.

Loss는 window의 `target_valid & time_valid`인 joint/xyz 원소 전체 평균이다. Final/auxiliary 각각 `(prediction_mm-target_mm)^2/scale_mm^2`를 평균하고 합한다. M1의 anchor-only loss와 달리 이 loss는 유효 과거 slot에도 적용하며 중첩 window의 과거 관측이 다시 가중되는 정책이다. 전부 누락된 anchor/batch는 거부한다.

DDIM k는 `round(linspace(K,1,sampling_steps))`를 내림차순 사용하고 마지막 다음 k=0으로 전이한다. 1 step이면 `[K]`다. X0에서 epsilon을 역산하고 `x_next=sqrt(a_next)*x0+sqrt(1-a_next)*eps`를 사용하며 eta=0이다. 최종 `a_next=1`이므로 clean head를 반환한다. Oracle clean denoiser의 1/2/4/10/32 step 복원과 q_sample k=0 identity를 별도로 검사했다.

평가는 마지막 anchor의 유효 관절 Euclidean 거리 합/관절 수 MPJPE(mm), squared xyz 합/(3×관절 수×scale²) normalized MSE다. PA/body-hand 분리/실시간 latency 지표는 없다. 모든 현 결과는 `synthetic_code_check`이며 fixture의 train/validation은 동일한 인위적 생성 규칙을 사용하므로 실제 일반화 근거가 아니다. Test는 `--final-test` 명시가 필요하다.

## Causality와 streaming 범위

`window_size`는 **현재 포함 총 anchor 입력 slot 수**이고 `attention_past_frames`는 층별 과거 수다. Tiny에서는 각각 4와 2다. 논문 ws=20은 식(4)를 따르면 현재 포함 21개이며 부록의 window 설정과 API 총 slot 수를 묵시적으로 동일시하지 않는다.

여러 temporal 층과 DDIM 단계를 거치면 raw-frame receptive field가 확대될 수 있다. 따라서 전체 sequence를 한 번에 denoise한 결과와 W개 raw frame만 잘라 재계산한 결과가 같다고 주장하지 않는다. 본 offline 정책은 **각 anchor에서 똑같은 유한 window를 조립해 모든 DDIM 단계를 다시 실행**하는 것이다. Streaming도 같은 정책이며 KV/step cache가 없다. 긴 입력 처리에서 유지하는 frame history는 최대 W개다. 직접 `BodyDenoiser`에 긴 T를 넣었을 때에도 미래 불변성을 검사했지만, 그것과 anchor별 재계산의 수치 일치를 완료 기준으로 섞지 않는다.

초기 noise는 고정 seed와 `(sequence_id,frame_index)`의 JSON key를 SHA-256으로 변환해 전용 Torch CPU generator를 초기화한다. 중첩 window에 같은 raw frame이 나타나면 같은 noise를 사용하고 미래 frame 추가·호출 순서·학습 RNG에 영향을 받지 않는다. Noise 정책/seed를 평가 결과에 기록한다. `AnchorSampler.push` 호출자는 같은 frame noise를 전달하며 입력 tensor는 복사해 보관한다.

Gap·sequence 변경은 자동 초기화하고 명시적 같은-ID 새 container에서는 `reset()`을 호출한다. 중복/역순·shape/finite 오류를 거부한 push는 기존 history를 보존한다. Left padding은 false time mask이며 미래 padding 복제·보간을 하지 않는다. RoPE absolute offset 변화, 미래 2D/camera/noise 동시 perturbation, 전체 DDIM sampler 출력, padding NaN 격리와 독립 anchor-window 대조를 검사했다.

## 실행·재개

기존 PyTorch 환경을 사용한다. 이번 검증은 Windows/Python 3.12.14/torch 2.7.1+cpu, CPU float32/thread1이다. 새 의존성은 추가하지 않았다. NumPy 미설치 경고는 기존 환경에서 발생하며 NumPy 변환은 사용하지 않는다.

```sh
python -m unittest discover -s tests -v
python -m rewind_m2 train --config configs/m2_body_synthetic.json --output outputs/m2_body_new --steps 100 --device cpu
python -m rewind_m2 train --config configs/m2_body_synthetic.json --output outputs/m2_body_new --steps 110 --resume outputs/m2_body_new/last.pt --device cpu
python -m rewind_m2 evaluate --checkpoint outputs/m2_body_new/last.pt --output outputs/m2_body_new/evaluation.json --device cpu
```

위 `python`은 이번 Windows에서 `.venv/Scripts/python.exe`다. 새 학습은 새 output, `--steps`는 재개 포함 총 update 수다. Dataset 생략 시 합성 fixture이며 raw pose/ZIP을 학습 입력으로 받지 않는다. M1과 같은 주기 저장·stale report recovery 흐름을 M2 CLI에 별도로 적용했고 M1 코드를 수정하지 않았다.

`m2_checkpoint_v1`은 model/Adam/StepLR/scaler, Python/Torch CPU·CUDA RNG, update·samples_seen(anchor windows), config·manifest/hash·input/m2 contract·source/environment에 더해 schedule alpha-bar·DDIM timestep/eta·anchor 재계산 정책을 저장한다. 정확한 resume는 코드·config·data·환경·schedule이 같아야 한다. Source는 M1/data/M2 Python 파일 hash를 포함하므로 커밋 전 결과도 식별 가능하다. CUDA/AMP 경로는 제공하지만 이번에 실행하지 않았고 성능·재개 일치를 보장하지 않는다.

10 update와 종료 시 atomic checkpoint 교체 다음 report를 저장한다. 두 파일 사이 중단은 누락 log 범위를 `report_recovery`로 기록하고 loss를 만들어내지 않는다. 구간·원 resume 파일 hash·코드/환경을 보존한다. 저장 간 최대 9 update의 작업이 유실될 수 있다.

## 검증 증거와 남은 일

최종 실행 기록은 Git 제외 `outputs/m2_evidence_20261015/developer_final/`의 `execution.json`, 명령 stdout/stderr, `features.json`, `run/report.json`, `last.pt`, `evaluation.json`이다. `developer/`는 평가 noise를 frame 키로 고정하기 전의 이전 검증이며 최신 결과와 혼합하지 않는다. 논문 추출 text/렌더는 상위 조사 폴더에 보관했다.

최종 **107 tests(기존 97 + M2 10)**를 모두 통과했다. 실행한 범위는 M2 수학·camera/mask·gradient·과적합·causality·streaming·disk resume·CLI recovery와 기존 전체 회귀다. Streaming 검사는 55개 연속 frame 뒤 gap와 sequence 변경을 포함한 58개 합성 frame에서 최대 4개 history만 보존한다. 이는 수시간 운영·GPU memory·실제 motion 길이 일반화의 검증이 아니다.

합성 CLI 100 updates 뒤 동일 output의 checkpoint로 110 updates까지 재개하고 별도 evaluation을 실행했으며 모두 exit 0이다. 고정 frame-key noise의 train normalized MSE는 `0.0795539 → 0.000215622`, 마지막 validation은 `0.000215556`이다. 2관절 인위적 대응의 코드 과적합 증거이며 실제 데이터 정확도나 diffusion의 regression 대비 우위가 아니다. Disk 재개 검사에서는 dropout·무작위 k/noise/batch를 포함한 5+7 updates와 연속 12 updates의 model/Adam/StepLR/scaler/RNG/학습 위치가 동일했다.

처음 전체 회귀를 sandbox에서 실행했을 때 기존 archive loopback 5개 test의 하위 조건 10개가 `http_open_failed`로 실패했다. 외부 다운로드 없는 로컬 HTTP 연결 권한으로 재실행해 구분한다. M2 첫 검사에서 camera NaN이 M1 manifest JSON 단계의 ValueError로 먼저 거부되는 경우를 M2 DataContractError로 정규화했으며 source 데이터는 변경하지 않았다.

실제 body 관절·단위·camera 변환·시간·대응 RGB/예측 2D baseline, 논문 크기 teacher·velocity/foot loss·실제 과적합/pilot, regression 대비 diffusion/temporal 효과·CUDA/AMP·속도·VRAM은 남아 있다. M0/M1 실제 gate와 D1~D5를 우회하지 않으며 M2 전체 완료나 논문 성능 재현을 선언하지 않는다.
