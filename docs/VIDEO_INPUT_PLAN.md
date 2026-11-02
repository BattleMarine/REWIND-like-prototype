# 이미지·동영상 입력과 프론트 방향

2026-11-02 사용자 결정: 최종 프로토타입은 개별 이미지뿐 아니라 동영상도 처리한다. 아래는 설계 방향이며 동영상 decoder·UI·실제 temporal 학습의 구현 완료 기록이 아니다. 웹 framework, codec 지원 목록, 실시간 카메라 연결은 후속 결정이다.

현재 구현(2026-11-02)은 공통 프레임 선언과 기존 local RGB 이미지 경로 연결까지다. 원본·좌우·미확인 시간/geometry를 보존하며 실제 이미지 tensor/mask 동일성을 확인했다. [구현 계약과 검증](FRAME_INPUT.md)을 따르고, 아래 decoder·UI·실제 temporal 경로는 계속 후속 계획으로 유지한다.

## 논문 재검토

[REWIND 본문](https://jyunlee.github.io/projects/rewind/static/paper.pdf)의 §3, Fig.2, §3.1–3.3은 stereo egocentric 이미지 시퀀스와 카메라 자세를 조건으로 body를 먼저 추정하고, 예측 상체를 조건으로 hand를 추정한다. 현재·과거 window를 쓰는 relative-temporal attention과 one-step 증류가 시간 처리·속도의 핵심이다. 따라서 동영상 지원은 논문의 시간적 입력 구조에 맞는 방향이다. 임의의 외부 시점 단안 동영상을 같은 모델에 넣는 지원으로 확대 해석하지 않는다.

[보충자료](https://jyunlee.github.io/projects/rewind/static/supp.pdf) S.3.1–3.3은 EfficientNet 기반 encoder/CNN heatmap decoder, 2D keypoint·confidence와 camera 특징, temporal attention을 기술한다. 현재 작은 Conv 모델의 두 sample 진단은 이 frontend의 재현이나 동영상 성능 증거가 아니다. 영상 파일 해석·타임라인 UI·결과 내보내기는 프로젝트에서 추가할 기능이다.

## 공통 입력과 추론 구조 — 구현 계획

`이미지 쌍 / 순서 있는 이미지 시퀀스 / 동영상 → 공통 프레임 입력 → 2D frontend → causal body → 상체 조건 hand → 시간별 3D 결과`

- 독립 이미지 쌍은 T=1 진단으로 처리한다. 관계없는 이미지를 연속 동작으로 연결하지 않는다. 순서·시간·좌우 대응이 검증된 시퀀스와 동영상은 같은 temporal 경로를 사용한다.
- 첫 동영상 대상은 동기화된 좌우 egocentric 영상과 calibration·camera pose metadata다. 좌우 별도 파일 또는 명시적인 stereo layout을 입력 adapter에서 구분한다. 입력 누락 시 임의의 단위행렬 pose나 반대쪽 복제로 정상 결과를 만들지 않는다. Pose 추정/SLAM 내장은 별도 과제다.
- 공통 record에는 원본 identity, sequence/frame ID, 좌우 presentation timestamp(PTS)와 time base, 동기화 근거, calibration, 카메라 자세의 시점·변환 방향·단위, 유효성, 전처리 버전을 보존한다. Frame 번호만으로 timestamp를 추정하지 않는다.
- 가변 FPS·누락·좌우 시간차는 감지한다. 최초 adapter에서는 검증된 cadence와 동기화 허용 범위만 처리하고 다른 입력은 이유와 함께 중단한다. Resampling·보간·프레임 생략 정책은 별도 검증 후 설정으로 고정한다. Timestamp를 저장하는 것만으로 기존 모델이 가변 시간 간격을 이해하는 것은 아니다.
- Decoder는 presentation 순서로 제한된 buffer를 공급한다. Codec 내부 lookahead와 모델의 미래 관측 사용은 구분하고 decoder 지연도 측정한다. 전체 영상을 이미지 파일로 모두 풀어 저장하는 방식을 기본으로 삼지 않는다. Drive archive 인덱스와 영상 codec seek 인덱스는 별개다.
- Resize/crop 변환과 좌표 복원을 기록한다. 예측 cache는 원본·frame/PTS·decoder·전처리·모델 버전으로 식별하며 기존 계약을 검토한 뒤 확장한다. 장시간 실행도 buffer·cache·임시 디스크 상한을 둔다.
- 시퀀스 변경·역방향 seek 시 temporal 상태를 초기화한다. 목표 시점으로 이동하면 필요한 과거 문맥을 다시 공급하거나 검증된 상태를 복원한다. 초기 문맥 부족을 표시하며 이전 clip 상태를 섞지 않는다. 원본 noise/seed 조건을 고정해 재실행 결과를 비교한다.

## 사용자 화면 방향 — 제안

동영상 중심의 작업 화면에서 이미지 검사 모드도 유지한다.

1. 이미지/시퀀스/동영상 선택, 좌우 파일과 metadata 지정, 입력 준비 상태 확인.
2. 좌우 원본 영상과 2D overlay, 3D body/hand를 같은 timestamp로 표시.
3. 재생·일시정지·한 프레임 이동·타임라인 탐색. 처리 완료 구간과 아직 처리하지 않은 구간을 구분.
4. 누락·동기화 문제·초기 문맥 부족·추론 실패를 해당 구간에 표시. GT가 있을 때만 별도 비교를 제공.
5. timestamp·좌표계·관절 규격·mask·model revision이 포함된 결과와 실행 manifest 내보내기. 렌더링 영상 내보내기는 후속 후보.

우선 파일을 처리하고 결과를 재생하는 흐름을 목표로 한다. 재생 FPS, 모델 추론 FPS, decode·전처리·추론·렌더링을 포함한 처리 지연을 따로 표시한다. 실시간 카메라 입력과 30 FPS 달성은 별도 검증 대상이다.

## 로드맵 연결과 완료 기준

| 단계 | 다음 설계·구현 범위 | 확인할 조건 |
| --- | --- | --- |
| M0 | 공통 frame 계약, 좌우/pose 시간 정합성, 짧은 연속 clip 확보 | 시간·좌표·split 검증, 누락 처리, 원본 보존 |
| M1 | 이미지·동영상 adapter가 공유하는 2D frontend/cache | 정확도 평가, resize/좌표 복원, 출처·버전 보존 |
| M2~M3 | 실제 연속 clip의 causal body–hand 경로 | 동일 noise의 offline causal/streaming 일치, 미래 입력 불변성, reset·seek 재실행 |
| M4 | Student 연결과 비용 측정 | Teacher 대비 정확도, decoder 포함 지연, 장시간 메모리 상한 |
| M6 | 파일 동영상 UI·타임라인·결과 저장 통합 | 짧은 stereo clip의 입력부터 내보내기까지 재현, 신규 환경 검증 |

검증용 무손실 clip과 원본 이미지 시퀀스를 대조해 frame 순서·timestamp·좌표 및 추론 경로의 동등성을 확인한다. 손실 codec은 픽셀 완전 일치를 요구하지 않고 별도 오차 허용 기준을 정한다. 이후 누락·비동기·seek·긴 clip·중단 복구를 검사한다. GPU 및 독립 validation 확보 전 성능 완료를 선언하지 않는다.

현재 실제 확보본 Serena frame1·10은 비연속 두 장이다. RGB→예측2D→3D 단일 프레임 진단은 연결됐지만 영상 정확도·연속 motion·실시간성은 미검증이다. 다음 개발은 공통 입력 계약과 짧은 연속 clip 준비 범위를 정한 뒤 시작하며, 이번 문서화로 학습·대규모 다운로드를 자동 개시하지 않는다.
