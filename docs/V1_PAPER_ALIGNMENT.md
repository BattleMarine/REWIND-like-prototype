# v1 REWIND 구조 대조

근거는 로컬 [REWIND 논문](REWIND.pdf) 본문 PDF 3–5쪽 §3과 12–13쪽 S.3이다. 구조 후보, 합성 검증 및 실제 데이터 실행의 상태를 구분한다.

| 요소 | 논문 근거 | 기존 코드 | v1 후보와 검증 경계 |
| --- | --- | --- | --- |
| RGB 2D | S.3.1 EfficientNet 계열 encoder, MBConv 폭 16/24/40·깊이 1/2/2와 HardSwish, Conv 2개·BN·ReLU로 구성된 decoder 3단 | v0 64px context CNN·GroupNorm | 별도 frontend. Stage 반복 배치, stride, resize, upsampling, BN 운영 및 heatmap 학습법은 구현 가정이다. 실제 RGB 예측의 좌표·confidence와 GT 2D 진단은 분리한다. |
| Frame graph | S.3.2 teacher 512차원·3 block·4 heads, student 256차원·1 block·2 heads. 보조 pose head | M2/M3 GCN·spatial MHA·보조 head와 명시적인 정규화 adjacency | 기존 block을 후보로 재사용한다. 정규화, joint embedding 및 입력 mask/known bit는 프로젝트 차이다. 관절 그래프와 역할 매핑은 데이터 규격에 따로 고정한다. |
| Causal temporal | §3.2 식(4), S.3.3. RoPE 내적을 분자, 회전하지 않은 내적을 분모에 사용한다. 과거 20개와 현재 1개, teacher 3층·4 heads | M2/M3의 per-joint RoPE는 causal softmax | 별도 식(4) 후보의 feature map은 ELU+1, 분모 최소값과 FP32 계산을 명시한다. 정확한 feature map, RoPE 기저와 관절축 처리는 논문에 확정돼 있지 않다. 직접 참조 계산, mask·미래 불변성·유한 gradient를 검사한다. |
| 입력 시간과 좌표 | §3 식(2), S.3.1/S.3.4. 양쪽 카메라 자세, R6D+이동과 2층 MLP, 학습은 50연속프레임 | 기존 T1 입력에는 검증된 timestamp가 없다. M2/M3의 world→camera, mm, R의 첫 두 열 연결은 프로젝트 계약이다. | Clip의 총 50 slot과 attention 과거 20을 분리한다. 번호 연속성만으로 시간 동기, camera 방향, pose와 영상의 정합 또는 원본 motion 독립을 선언하지 않는다. Gap과 sequence/clip 경계에서 history를 초기화한다. |
| Diffusion | S.3.4 DDPM x0, K=1,000, cosine β 시작·끝 0.0001/0.02, DDIM teacher 10 step | M2/M3는 β에 대한 half-cosine interpolation과 결정적 DDIM | 논문에 β의 정확한 중간식이 없어 현 보간은 후보로 유지한다. Alpha-bar cosine schedule과 구분한다. 논문의 2M update를 작은 데이터에 바로 적용하지 않는다. |
| Cascade | §3.1, S.3.1. 손의 상체 조건은 학습 시 GT, 추론 시 body 예측 | M3의 동결 body, GT 학습 및 예측 평가, wrist-relative 손 표현 | 손목 상대 좌표·관절 세부 매핑은 프로젝트 가정이다. GT 상체 평가를 별도 진단으로 보고하고, 실제 평가는 frozen body 예측과 RGB 기반 2D를 연결한다. |
| Loss | S.3.4 식(7) 위치·속도·발·중간 frame, 후반 velocity/foot 가중치 변경 | M2/M3의 masked normalized position/frame MSE | Velocity는 검증된 시간·좌표·인접 mask를 사용한다. Foot은 world 좌표와 발·접촉 근거가 있을 때만 허용한다. 시간/카메라 기반 학습 gate와 foot loss 적용 가능 상태는 분리한다. |
| Student | §3.3 식(5), S.3.4. k=K에서 단일 step, 작은 k에서 teacher 비교 | 현재 검증 범위는 teacher | Teacher 입력과 학습이 검증된 뒤 구현한다. 작은 k의 분포와 teacher target의 gradient 분리는 선택 근거를 남긴다. Identity conditioning과 IK는 별도 확장이다. |

연속 선택과 raw cache는 `python -m rewind_prepare.temporal_cli`의 `plan`, `pack`, `restore`로 분리한다. 팩은 원본 PNG/JSON 바이트의 버전과 해시를 보존한다. 이전 64px 캐시와 새 frontend의 해상도별 파생 입력은 별도 계약이다. 실제 Colab 준비와 GPU 학습은 별도 실행으로 검증한다.

병렬 팩의 작업 버퍼 예약은 처리 중인 각 페어의 원본 바이트 합계의 3배에 24MiB를 더한 추정치다. 프로세스 전체 RSS는 별도의 표본 측정값으로 기록한다. 실제 Drive 병렬 처리 속도는 아직 측정하지 않았다.
