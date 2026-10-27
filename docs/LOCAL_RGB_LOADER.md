# 로컬 lazy stereo RGB 로더

2026-10-27 **M1 로컬 RGB 로더 구현, 실행 미검증**. 추출된 좌우 PNG를 필요한 sample 또는 작은 batch만 읽는 `rewind_m1.local_rgb`를 추가했다. 사용자 지시에 따라 로컬 회귀·합성 검사·이미지 디코딩·모델 실행·설치·Colab/Drive 요청은 하지 않았다. 아래 사용 예시도 실행 확인하지 않았다. 실제 데이터 baseline이나 M0~M3 완료를 뜻하지 않으며 M4는 미착수다.

## API와 입력 경계

`LocalStereoRGB(manifest_path, root, *, limits=None)`는 명시적 로컬 manifest를 읽는다. `root`는 추출 PNG가 있는 공통 디렉터리이며 코드에 개인 절대 경로를 고정하지 않는다. Archive 다운로드·해제·압축 prefix 읽기 기능은 없다.

| API | 동작 |
| --- | --- |
| `RGBLimits(...)` | 파일·픽셀·batch 등 로컬 상한을 별도로 지정 |
| `len(loader)` | Manifest record 수 |
| `loader.manifest_sha256` | RGB manifest의 canonical content hash |
| `loader.metadata(index)` | 해당 record·출처·split 선언의 복사본. 이미지 내용 읽기 없음 |
| `loader[index]` | 해당 좌우 PNG만 읽어 `images[2,3,H,W]`와 metadata 반환 |
| `loader.load_batch([indices])` | 같은 split 선언의 작은 batch를 `images[B,2,3,H,W]`와 metadata 목록으로 반환 |

Index는 bool·음수·slice가 아닌 범위 내 정수다. Batch index는 길이가 제한된 list/tuple만 받으며 generator를 무제한으로 펼치지 않는다. Manifest의 record 순서를 유지하고 자동 정렬·shuffle·시간 window·보간을 만들지 않는다. Sequence/frame의 원본 시간·연속성은 별도 검증이다.

생성자는 manifest 전체와 각 파일의 경로·stat를 검사한다. 이때 PNG 내용·hash·header를 읽거나 디코딩하지 않으며 torch/Pillow도 import하지 않는다. 파일 내용의 손상·hash·원래 해상도 검사는 해당 sample을 요청할 때 수행한다. 생성 성공을 모든 이미지의 무결성 확인으로 해석하지 않는다.

## Manifest 규격

별도 schema `local_stereo_rgb_v1`을 사용한다. `m1_features_v1`, `real_pose_candidate_v1`, archive matching report나 예측 cache와 서로 대체하지 않는다. 위에 없는 필드와 중복 JSON key를 거부한다.

| 위치 | 필드와 의미 |
| --- | --- |
| 최상위 | `schema_version`, `id`, `dataset_version`, `data_kind`, `image_size`, `sources`, `records`, `content_sha256` |
| `data_kind` | `dataset_sample` 또는 명시적인 `synthetic_fixture`. 원본 출처의 synthetic 여부와 일치해야 하며 자동 변환하지 않음 |
| `image_size` | 원래 `[width,height]`. 정수·각4 이상, 모든 record/view에서 동일 |
| `sources[source_id]` | 기존 `archive_manifest_v1` 전체인 `archive_manifest`와 추출 보고서의 `extraction_manifest_sha256` |
| Record | `sequence_id`, 비음수 정수 `frame_index`, `split`, `split_evidence`, `views` |
| `split` | `train/validation/test/unknown` 중 호출자의 선언. 근거 문자열 필수, 동일 sequence의 서로 다른 선언은 거부 |
| `views` | 정확히 `left`, `right` 각각 하나 |
| 각 view | `path`, `size_bytes`, `sha256`, 원래 `image_size`, `source_id`, 원래 `member`, `tar_complete=true` |

`sequence_id`는 끝의 motion 이름만 사용하지 않고 전체 상대 경로를 보존한다. `member`는 기존 `identify(member, "rgb")` 규칙의 `<전체 sequence>/fisheye_final_image/camera_<left/right>/final_<정수>.png`다. 각 view의 식별 결과가 해당 record의 sequence·frame·side와 정확히 같아야 한다. 좌우를 파일 정렬 순서로 짝짓지 않는다. 중복 sequence/frame과 대소문자만 다른 중복 파일 경로도 거부한다.

`path`는 로컬 `root` 기준 canonical POSIX 상대 경로이고 `member`는 원래 archive 내부 경로다. 저장 레이아웃이 다르면 두 경로는 달라도 되지만 같은 파일의 원래 bytes/hash를 가리켜야 한다. 절대 경로·상위 이동·backslash·ADS·glob·symlink·Windows junction/reparse point·root 이탈을 거부한다.

예를 들어 record 한 개의 형태는 다음과 같다. **경로·크기·hash는 형식 설명용 placeholder이며 실제 검증 입력이 아니다.** 원래 추출 보고서와 보존 파일에서 값을 대조한 후 사용해야 한다. 두 view의 hash를 서로 복사하지 않는다.

```json
{
  "sequence_id": "Environment/Day/person/avatar/001/Motion",
  "frame_index": 0,
  "split": "unknown",
  "split_evidence": "Local declaration only; official split not yet verified.",
  "views": {
    "left": {
      "path": "files/Environment/Day/person/avatar/001/Motion/fisheye_final_image/camera_left/final_0.png",
      "size_bytes": 1234,
      "sha256": "REPLACE_WITH_LEFT_FILE_SHA256",
      "image_size": [1024, 1024],
      "source_id": "rgb_source",
      "member": "Environment/Day/person/avatar/001/Motion/fisheye_final_image/camera_left/final_0.png",
      "tar_complete": true
    },
    "right": {
      "path": "files/Environment/Day/person/avatar/001/Motion/fisheye_final_image/camera_right/final_0.png",
      "size_bytes": 1234,
      "sha256": "REPLACE_WITH_RIGHT_FILE_SHA256",
      "image_size": [1024, 1024],
      "source_id": "rgb_source",
      "member": "Environment/Day/person/avatar/001/Motion/fisheye_final_image/camera_right/final_0.png",
      "tar_complete": true
    }
  }
}
```

이 record를 최상위 manifest의 `records`에 넣고, `sources.rgb_source.archive_manifest`에는 원래 RGB archive manifest 전체를 보존한다.

최상위 `id`는 명시적인 식별 문자열, `dataset_version`은 모든 source의 기존 선언과 같아야 한다. `sources`의 각 source에는 기존 RGB archive manifest 전체를 보존하며, 출처의 file ID·version·원본 위치를 임의로 새 값으로 바꾸지 않는다. `extraction_manifest_sha256`는 해당 추출 보고서의 canonical `content_sha256`이며 JSON 파일 bytes의 SHA256과 구분한다. `content_sha256`는 자기 필드를 제외한 RGB manifest에 기존 `rewind_archive.contracts.digest`를 적용한 소문자64자리 값이다.

로더는 기존 `read_json`, `safe_path`, `digest`, `validate_manifest`, `identify`를 재사용한다. `verify_extraction`은 전체 추출 payload를 읽으므로 생성자에서 자동 재실행하지 않는다. 이 manifest는 이미 검토한 완전 추출 파일과 보고서를 근거로 작성해야 한다. 로더가 원본 archive·추출 보고서를 다시 가져오거나 선언한 출처의 진위를 인증하는 기능은 없으며, 내용 hash와 출처 참조만으로 RGB/pose 정합성·공식 split을 증명하지 않는다.

## 디코딩·값·공간 상한

출력은 **CPU float32, RGB 채널 순서, 값0~1(`uint8/255`), 원래 H/W**다. 기존 `StereoHeatmapNet.forward`의 `B,2,3,H,W` 입력 형태에 맞춘다. Single sample은 batch 차원이 없으므로 별도 batch API를 쓰거나 호출자가 명시적으로 batch 차원을 붙여야 한다. 모델 호출·학습 연결은 이번 구현에서 하지 않았다.

8bit RGB 및 완전히 불투명한 RGBA PNG만 허용한다. RGBA의 alpha를 확인한 뒤 RGB로 변환하며 투명 픽셀·tRNS transparency·palette/grayscale/16bit·다중 frame을 거부한다. Resize·crop·flip·augmentation·EXIF 회전·ICC/gamma 보정을 하지 않는다. GT 2D 좌표나 mask를 읽거나 바꾸지 않는다.

요청 시 해당 파일만 상한 내 읽고 크기·SHA256·읽는 동안의 파일 변경 여부를 확인한다. 같은 bytes에서 IHDR의 원래 해상도·bit depth·channel을 먼저 확인한 뒤 Pillow의 PNG 검증과 실제 decode를 수행한다. 좌우 또는 record 간 다른 해상도를 맞추기 위해 자동 resize/padding하지 않는다. 실패한 sample/batch의 부분 tensor는 반환하지 않는다.

| 상한 | 기본값 | 설정 가능한 최대값 |
| --- | ---: | ---: |
| Manifest bytes | 4MiB | 16MiB |
| Records | 10,000 | 100,000 |
| Sources | 32 | 32 |
| 각 PNG 압축 bytes | 8MiB | 64MiB |
| 각 view 픽셀 수 | 4,194,304 | 16,777,216 |
| Batch sample 수 | 4 | 16 |
| 출력 float tensor bytes | 128MiB | 512MiB |

MiB는1,048,576bytes다. 상한은 별도 `RGBLimits`이며 기존 archive 전송·해제 예산을 바꾸지 않는다. Batch는 `B×2×3×H×W×4`bytes를 디코딩 전에 검사하고 최종 tensor에 한 sample씩 채운다. 기본 설정의1024×1024 batch4는 출력96MiB다. 이미지 전체 선로딩·전역 캐시·torchvision·NumPy 변환은 없다.

출력 tensor 상한은 프로세스 RSS 상한이 아니다. 제한된 manifest/metadata, 요청 view의 PNG bytes·decoder 버퍼·RGB bytes·uint8/float 변환 임시 공간이 추가로 필요하다. 호출자가 반환된 tensor를 계속 보관하거나 별도 DataLoader의 worker/prefetch·batch를 늘리는 비용은 이 API가 통제하지 않는다. 명시적 `load_batch`의 상한과 외부 병렬 로딩을 혼동하지 않는다.

## 의존성과 실행하지 않은 사용 예시

기존 [M1 PyTorch 요구사항](../requirements-m1.txt)을 유지하고 선택적 [requirements-rgb.txt](../requirements-rgb.txt)에 `Pillow>=10.4`를 별도로 선언했다. 선언한 모든 버전의 호환성을 검증한 것은 아니다. 이번에는 가상환경 파일 목록에서 torch2.7.1+cpu 설치 흔적과 PIL/Pillow 부재만 확인했으며 import·설치를 하지 않았다. 이미지 요청 때 torch/Pillow가 없으면 `RGBContractError`로 필요한 선택 의존성을 알린다. 로더가 자동 설치하거나 기존 Colab PyTorch를 교체하지 않는다.

아래 Python 예시는 **작성만 했고 실행하지 않았다.** Manifest와 원본 PNG는 사용자가 이미 보존한 검토 자료로 준비해야 하며 새 Drive 실행을 요청하지 않는다. 새로운 CLI 명령은 추가하지 않았다.

```python
from pathlib import Path
from rewind_m1.local_rgb import LocalStereoRGB, RGBLimits

loader = LocalStereoRGB(
    manifest_path=Path("outputs/local_rgb_input/manifest.json"),
    root=Path("data/extracted_rgb"),
    limits=RGBLimits(max_batch_size=2, max_batch_tensor_bytes=64 * 1024**2),
)
declaration = loader.metadata(0)  # 이미지 bytes를 읽지 않는 metadata 복사본
sample = loader[0]               # images: 2,3,H,W
batch = loader.load_batch([0])   # images: 1,2,3,H,W
images = batch["images"]         # StereoHeatmapNet의 입력 shape; 여기서는 모델 호출 안 함
```

`RGBContractError`는 schema/hash·식별·view·경로·상한·split·decode 오류를 구분하는 reason을 담는다. 누락 view나 잘못된 파일을 GT·다른 frame·검은 영상으로 대체하지 않는다. Metadata는 원본의 `data_kind`·split 선언·출처를 유지한다. 로더는 `verified`나 `train_ready`를 발급하지 않으며 candidate→학습 release를 구현하지 않는다. 실제 학습 조건은 [실제 pose 준비](REAL_DATA_PREPARATION.md#candidate는-학습-입력이-아니다)와 [M1 입력 계약](M1_BASELINE.md#데이터-입력-계약)을 유지한다.

새 모듈은 M1의 기존 Python 파일 hash 수집 범위에 포함된다. 기존 checkpoint의 정확한 resume/evaluate는 코드 hash 차이로 거부될 수 있으므로 과거 결과는 당시 소스·환경과 함께 보존한다. 현재 warm192 고정 배포본·notebook·캐시는 바꾸지 않는다.

이번 정적 검토는 manifest/파일 검사와 decode 순서, 상한 적용 위치, shape·채널·단위, 기존 학습 guard 분리를 읽어 확인하는 범위다. 실행·메모리·성능과 실제 front-end 연결은 미검증이다.
