# 공통 프레임 선언과 기존 이미지 경로

2026-11-02 · [이미지·동영상 입력 계획](VIDEO_INPUT_PLAN.md)의 공통 frame 규격과 기존 local RGB 이미지 연결을 구현했다. 영상 decoder·UI·네트워크·새 학습은 이번 범위에 포함하지 않는다.

## 실제 연결한 API

`rewind_m1.frame_input`은 `stereo_frame_input_v1` 선언과 validator, 기존 metadata를 옮기는 `from_image_metadata`를 제공한다. `LocalStereoRGB.metadata(index)`에 `frame_input`을 추가했다. 기존 `metadata['record']`, `sources`, manifest hash와 `[index]`/`load_batch`의 `images`·`metadata` 반환 구조는 유지한다. 원본 `local_stereo_rgb_v1` 파일을 다시 쓰거나 다른 schema로 바꾸지 않는다.

```python
from rewind_m1.local_rgb import LocalStereoRGB
from rewind_m1.frame_input import validate_frame_input

loader = LocalStereoRGB(manifest_path, image_root)
metadata = loader.metadata(0)                 # PNG decode 없이 선언만 읽음
frame = validate_frame_input(metadata['frame_input'])
sample = loader[0]                           # 기존 PNG hash/CRC/decode 검사
batch = loader.load_batch([0, 1])             # 기존 B,2,3,H,W tensor
```

실제 RGB 진단 `real_rgb_diagnostic.load_images`도 반환받은 공통 record를 `input_manifest.frame_inputs`에 넣는다. 따라서 후속 cache의 입력 identity가 공통 출처를 포함한다. 원본1024 RGB와64 resize tensor·GT·mask는 변하지 않지만, 추가 provenance가 포함되는 **새 진단 input manifest의 hash는 달라질 수 있다.** 과거 고정 cache/checkpoint를 몰래 수정하거나 재학습하지 않았다.

## Record 계약

모든 record는 `mode=independent_frame`, `temporal_ready=False`, `train_ready=False`로 제한한다. Sequence/frame ID·dataset version/data_kind·정확한 left/right 선언과 native image size·전처리를 보존한다. Record 자체hash를 검증하고 반환은 독립된 복사본이다.

| 필드 | 현재 이미지 adapter의 내용 | 검사 범위 |
| --- | --- | --- |
| `views.*.source_identity` | source ID, ordered archive manifest hash, 추출 manifest hash, 정확한 member, payload SHA/크기 | 원본 출처 선언·member의 sequence/frame/side·정규 경로와 hash 형식 |
| `views.*.timing` | timestamp/PTS/time_base/clock 모두null, unknown과 이유 | 누락을0/FPS로 보충하지 않음 |
| `stereo_sync` | unverified, 동일 sequence/frame 대조만 수행했다는 근거 | 파일명 대응을 시간 동기화 검증으로 표시하지 않음 |
| `geometry.calibration` | unknown, SHA/projection null과 이유 | RGB manifest에 없는 값을 추정하지 않음 |
| `geometry.camera_pose.*` | unknown, matrix/변환방향/이동단위/좌표계 null, 별도 timing과 이유 | 단위행렬·반대쪽 복사 등 대체값 없음 |
| `preprocessing` | rgb_0_1_v1/RGB/[0,1]/native_pixel/공간변환null | 기존 native RGB 해석 유지 |
| `validity` | view presence는 선언, annotation mask/visibility는null | 선언 유효성과 실제 bytes/decode 검증을 구분 |

Archive manifest hash는 원본의 명시적 조각 순서·ID·크기·version 등 metadata에 대한 hash다. 전체 archive bytes의 checksum/무결성을 새로 검증한 뜻은 아니다. Payload bytes/hash/PNG decode는 기존 이미지 로더가 읽을 때 확인한다. Metadata 조회만으로 이 검증을 완료했다고 표시하지 않는다. GT annotation mask나 visibility를 RGB source에 생성하지 않는다.

기존 loader가 허용하는 경로와 해상도 범위를 좁히지 않는다. Sequence/member는 기존 `safe_path`의 UTF-8 byte 한도를 따르고, member의 비정규 `./` 등 alias를 거부한다. Image size는 최소4, 기존 loader의 최대 pixel ceiling 범위를 따르며 변당65536 같은 새 제한은 두지 않는다. 실제 loader 인스턴스의 더 작은 pixel/tensor/file 예산은 그대로 적용한다.

## 제공된 시간·geometry와 미래 video 선언

다른 adapter가 시간을 제공할 경우 `status=provided`, 명시 clock을 요구한다. PTS는 정수이며 time base의 양의 정수 numerator/denominator와 쌍으로 있어야 한다. 명시 seconds와 PTS×timebase가 함께 있으면 상대허용오차0·절대1e-9초로 대조한다. 큰 절대시각 때문에0.5초 같은 오차를 상대오차로 허용하지 않는다. 비유한 값·bool PTS·0분모·불완전한 쌍·서로 모순된 값은 거부한다. PTS에서 검증용 seconds를 계산하더라도 record에 새 timestamp를 써 넣지 않는다. Frame 번호로 시간을 만들지 않는다.

Stereo sync를 provided로 선언하려면 근거·같은 clock·실제 비교 가능한 두 시각·명시한 max skew가 필요하다. Clock/허용 시간차가 맞지 않으면 거부한다. 이는 한 record의 선언 일관성 검사이며 cadence·단조 presentation 순서·VFR·누락 clip·실제 센서 동기화의 증명이 아니다. 올바른 두 timestamp가 있어도 temporal-ready는 발급하지 않는다.

Calibration이 provided라면 SHA/projection model 선언을, camera pose가 provided라면 finite4×4 행렬·homogeneous 마지막 행·변환방향·translation 단위·좌표계·시점 선언을 검사한다. Reflected-world convention을 보존하며 det=+1/SO(3)을 임의 강제하지 않는다. 이 검사는 geometry의 정확성·calibration 파일 내용·카메라와 이미지의 실제 시각 일치를 증명하지 않는다.

향후 video adapter용 `video_stream` source identity는 asset SHA/stream index/source frame index/decoder version의 형태만 정의했다. 이번 구현은 video 파일을 열거나 decode하지 않는다. 단일 파일의 stereo layout·seek/reset·buffer·resampling·cadence 검증은 후속 작업이다. 미래 video identity를 선언해도 현재 validator는 independent frame/temporal-ready false만 허용한다.

## Serena 보존본의 의미

현재 Serena frame1·10은 독립된 정적 입력으로만 연결한다. 기존 **RGB manifest에는 시간·calibration·camera pose가 없으므로** 공통 frame 선언은 null과 이유를 반환한다. 한편 별도 prepared candidate에는 camera pose와 calibration/spec hash 근거가 있다. 이를 삭제하거나 미존재라고 보고한 것이 아니다. 이번 이미지 로더는 candidate를 암묵 결합하지 않으며, 해당 geometry/시점을 frame에 연결하는 명시 adapter는 후속 범위다. 기존 candidate·학습 gate·단위/관절/정답 mask는 그대로 유지한다.

## 검증·재현 근거

개발 근거는 `outputs/frame_input_20261102/developer/`에 저장했다.

- `before.json`/`after.json`: 실제 PNG2쌍의 native sample/batch tensor와 GT feature/3D target/mask hash, 원본 candidate/RGB pin의 전후 동일. 변경 전 loader hash도 기록했다.
- `verify_real.py`: 학습·원격 없이 실제 `metadata/[index]/load_batch`와 진단 `load_images`를 읽고 공통 frame source 사용과 전후 동일성을 검사한 재현 스크립트. 진단 manifest에 실제 두 record가 포함되는지도 확인한다.
- `tests.json`/`tests.txt`: 신규9개와 관련 기존 이미지 manifest·예측 cache·검토/GT/RGB 진단 회귀, 총41개 실패·오류·건너뜀0. 학습 update/원격 요청0이다.
- `code_hashes.json`: 검증된 최종 코드·문서 identity.

실행은 의존성이 이미 있는 Python3.12로 `python outputs/frame_input_20261102/developer/verify_real.py`, 관련 검사는 tests 디렉터리를 import 경로에 두고 `test_frame_input`, `test_rgb_manifest`, `test_local_rgb_predictions`, `test_rgb_review`, `test_real_gt2d_diagnostic`, `test_real_rgb_diagnostic`을 실행했다. 시간 오류 회귀는 유한 입력의 hash를 다시 서명한 뒤 구체적인 시간 오류를 확인하므로 stale hash 때문에 잘못 통과하지 않는다. 큰 시각의0.5초 불일치도 거부한다.

오케스트레이션의 독립 실제 로딩은 `outputs/frame_input_20261102/root/review.json`에 보존했다. Native/resize RGB·GT/mask·기존 manifest 필드·blocker가 모두 동일하고 independent/unknown timing을 확인했다. 최종 시간 비교 수정은 provided-time 가지에만 적용되므로 이 실제 unknown-time 로딩 결과에는 영향이 없음을 diff와 신규검사로 확인했다.

과거 고정 배포 검사 `tests/check_local_rgb.py`의 loader/revision pin은 변경하지 않았다. 이번 변경에 맞춘 별도 호환 회귀로 검증하며 과거 배포본을 소급해서 새 코드로 바꾸지 않는다. 기존 원본·cache/index·진단 실행 산출물도 덮어쓰지 않았다. Decoder/UI/네트워크/학습이나 영상/temporal 성능 검증은 이번 완료 범위가 아니다.
