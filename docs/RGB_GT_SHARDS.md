# 독립 frame용 RGB·GT shard 공급

2026-11-02 전체 학습 준비 3b1 단계. 로컬에 준비한 RGB manifest와 `m1_features_v1` GT 문서를 작은 chunk 카탈로그로 묶고, 한 shard/선택 split의 metadata만 유지한다. 픽셀은 요청한 batch에서 해시 검사·decode한다. UI·모델 학습·resize·예측 cache 생성은 이번 범위 밖이며 실제 학습 release를 발급하지 않는다.

## 계약과 경계

새 모듈은 `rewind_m1.rgb_shards`와 `rewind_m1.rgb_shards_cli`다. 기존 `LocalStereoRGB`, `LocalRGBSupervision`, `EpochSampler`의 검증·로딩·순회 연산을 재사용하며 기존 API·checkpoint·가중치·cache 형식을 수정하지 않는다.

`m1_rgb_gt_shards_v1`은 공통 dataset ID/version, GT input contract, 명시 joint 순서·native 좌우 크기와 다음 shard descriptor를 고정한다.

- chunk ID, pose JSON의 상대 경로·byte 수·파일 SHA
- RGB manifest의 상대 경로·byte 수·파일 SHA·content hash, payload root
- record 수·PNG 선언 byte 합, sequence별 motion/split·record 수·첫/마지막 frame
- 전체 카탈로그의 canonical content hash와 크기 상한

pose의 `manifest.id`/`dataset_version`은 공통 데이터셋 identity이며 chunk마다 다른 값을 붙이지 않는다. RGB manifest의 ID는 chunk별로 달라도 되지만 dataset version·data kind·native 크기·key/split은 pose와 일치해야 한다. 공통 GT 계약은 `gt_2d`, `device_relative`, 명시 단위·joint 순서와 기존 검증 요건을 유지한다. raw pose candidate나 predicted 특징 문서는 이 공급 경로에 넣지 않는다.

2D frontend는 독립 frame을 처리하므로 **같은 원본 sequence를 여러 disjoint frame chunk에 나눌 수 있다.** 원래 sequence/member 이름과 motion ID를 보존하며 sequence의 motion/split은 모든 chunk에서 같아야 한다. 같은 motion이 다른 split에 들어가면 거부한다. 같은 sequence의 inclusive frame 범위가 겹쳐도 거부한다. 희소 chunk 사이 실제 frame이 겹치지 않더라도 범위가 겹치면 현재 계약에서는 보수적으로 거부한다. chunk 안의 sequence 그룹과 frame 순서는 고정한다.

이는 원본 sequence 전체 coverage·timestamp 정합·연속 clip 검증이 아니다. `temporal_ready=false`를 유지한다. 원본 motion 문자열 중복을 검사한 것과 실제 원본 motion identity를 독립 확인한 것도 구분한다.

기존 **완결-sequence M1 3D feature shard 카탈로그는 동일 sequence가 여러 shard에 걸치면 계속 거부**한다. 이후 [별도 T=1 명시 계약](T1_FEATURE_CHUNKS.md)으로 독립 단일 프레임의 RGB cache/join chunk를 연결했다. 자동 승격하지 않으며 temporal context/overlap 경계를 보존하는 확장은 여전히 필요하다. 이를 전체 데이터/temporal 공급 완료로 표시하지 않는다.

## 자원 제한과 lazy 로딩

기본 상한은 catalog4MiB·4096chunk, chunk당 pose4MiB·RGB manifest4MiB·256record·선언 PNG payload256MiB, native batch tensor128MiB다. joint는 최대256, native 크기는 좌우 동일하며 기존 RGB 픽셀/파일 상한도 적용한다. 기존 supervision의 batch1~4·JSON/record 상한을 늘리지 않는다. `limits` 설정은 이 값을 낮출 수 있다.

파일 byte 상한은 JSON 파싱 전에 검사한다. catalog 입력도4MiB를 넘으면 파싱하지 않는다. build는 descriptor마다 직렬화 크기를 한 번씩 누적하고 append 전에 catalog 상한을 검사해 뒤의 입력을 읽지 않는다. 전체 catalog를 매번 다시 직렬화하는 O(n²) 방식은 쓰지 않는다. direct Python API가 이미 큰 객체를 넘긴 경우 그 입력 객체 자체의 메모리를 회수·제한하는 기능은 없다.

`RGBShardDataset.load(index, split)`는 해당 pose/RGB metadata와 원본 파일 stat을 검사하고 기존 supervision 객체를 반환한다. 같은 index/split은 재사용하며, 다른 shard 또는 split으로 바꾸기 전에 내부 cache를 해제한다. sampler의 split 내 offset은 **pose record 순서에서 해당 split만 선택한 순서**다. RGB manifest 순서와 독립적으로 sequence/frame key로 찾아 연결한다. 전체 corpus pose/window tensor를 한꺼번에 만들지 않는다. 기존 `load_document(window=1)`은 현재 chunk 안에서만 사용하고 그 임시 tensor를 이후 보관하지 않는다.

픽셀은 `supervision.load_batch(indices)`를 호출할 때만 읽고 SHA/PNG header/decode를 검사한다. 반환 `images`, `gt_xy`, `annotation_valid`는 기존 supervision과 동일한 native 좌표다. Resize·clipping·관절 mask 재해석을 하지 않는다. 화면 밖의 유효 GT도 좌표와 annotation mask를 보존하고 visibility를 임의로 만들지 않는다. 3D 정답이나 원본 pose를 바꾸지 않는다.

catalog의 payload byte는 **선언 상한**이며 build/metadata 검사는 PNG 전체 hash나 decode를 수행하지 않는다. 전체 PNG 검사에서는 작은 sample씩 읽고 즉시 반환 batch 참조를 버린다. 원본/준비 파일을 쓰거나 삭제하지 않으며, 전체 디스크 예산·eviction·archive 추출은 [준비 queue](PREPARATION_QUEUE.md)와 별도다. 한 shard cache는 전체 RSS 보장이 아니다. JSON 객체·기존 supervision의 metadata 복사·decode 임시 tensor가 함께 있을 수 있고, 호출자가 이전 객체/batch를 외부에 계속 참조하면 메모리가 남는다.

## 검사 범위와 CLI

build 입력 spec은 다음과 같은 명시 list다. 경로는 `--base` 기준이며 catalog output은 base 바로 아래에 둔다.

```json
[{"id":"train_chunk_0","pose":"train_chunk_0_pose.json","rgb_manifest":"train_chunk_0_rgb.json","rgb_root":"inputs"}]
```

보존한 합성 fixture의 실제 경로를 이용한 명령이다. build output이 이미 존재하면 다른 이름을 사용한다.

```powershell
python -m rewind_m1.rgb_shards_cli build --specs outputs/full_training_readiness_20261102/developer/rgb_shards/fixture/specs.json --base outputs/full_training_readiness_20261102/developer/rgb_shards/fixture --output outputs/full_training_readiness_20261102/developer/rgb_shards/fixture/new_catalog.json
python -m rewind_m1.rgb_shards_cli inspect --catalog outputs/full_training_readiness_20261102/developer/rgb_shards/fixture/catalog.json
python -m rewind_m1.rgb_shards_cli inspect --catalog outputs/full_training_readiness_20261102/developer/rgb_shards/fixture/catalog.json --metadata
python -m rewind_m1.rgb_shards_cli inspect --catalog outputs/full_training_readiness_20261102/developer/rgb_shards/fixture/catalog.json --full-png
```

| 호출 | 실제 확인 범위 |
| --- | --- |
| `build` | pose/RGB metadata 파일 hash·계약·키·split·stat·크기를 모든 chunk에 대해 순차 검증. PNG decode 없음 |
| `inspect` | catalog 선언·hash·chunk 범위/identity 검사만. payload 파일 존재/무결성 미확인 |
| `inspect --metadata` | 모든 chunk metadata와 파일 stat 검사. `metadata_verified=true`, `full_png_verified=false` |
| `inspect --full-png` | 위 검사와 모든 PNG sample의 SHA/decode 검사. `full_png_verified=true`와 sample 수 표시 |

보고서는 `train_ready=false`, `temporal_ready=false`를 유지하고 어느 명령도 전체 학습 가능 승인을 발급하지 않는다. 특히 실제 `dataset_sample`의 선언 검증 문자열만으로 학습 ready를 발급하지 않는다. full PNG가 통과해도 원본 archive 전체 CRC·RGB-pose 내용/시간·독립 validation·원본 motion·좌표/단위의 학습 release를 대신하지 않는다.

`RGBEpochSampler`는 기존 `EpochSampler`의 순회와 peek/commit/state 연산을 재사용하되 새 catalog hash와 별도 policy에 결박한다. 각 epoch에 split의 모든 제공 frame을 한 번 방문하며 tail을 버리지 않는다. shard와 그 안의 split offset을 저장해 복원한다. 현재는 공급 유틸이며 optimizer 성공/실패와 checkpoint를 결합하는 trainer는3b2 범위다.

## 검증 결과와 남은 작업

신규11개와 기존 공통 frame9개, 총20개가 failure/error/skip0으로 통과했다. 합성 train sequence의 chunk0(frame0~1)·chunk1(frame2~3), 별도 validation sequence/motion(frame0~1)을 사용했다. native batch는 `(2,2,3,6,8)`이며 기존 로더와 RGB·GT xy·annotation mask·metadata가 동일했다. 화면 밖 GT `[9,3]`와 유효 mask도 보존했다. train epoch의4key를 빠짐없이 방문하고 PNG6sample 전체 검사를 확인했다. 학습 update·원격 요청·설치는0이다.

Root도 고정 신규11개를 별도 실행해 failure/error/skip0으로 확인했다. 근거는 `outputs/full_training_readiness_20261102/root/stage3b1_tests.txt`에 보존한다.

근거는 `outputs/full_training_readiness_20261102/developer/rgb_shards/`의 `verify.py`, `tests.json/txt`, `code_hashes.json`, `verification_loading.json`, `commands.json`, `cli_0~3.txt`, `fixture/`에 보존한다. 선언/metadata/full PNG 범위, 기존/native hash 대조와 sampler key를 구분해 기록한다.

후속3b2에서 작은 `StereoHeatmapNet` 구조를 유지하고 명시 훈련 해상도·bilinear `align_corners=True`·GT 각축 좌표 변환·예측 원좌표 복원과 preprocessing version을 연결했다. full resolution heatmap 상한을 숫자만 늘리지 않는다. 별도 UI 독립 trainer의 합성 CPU 재개·validation·cache/join 검증은 [2D 학습 경계](RGB_SHARD_TRAINING.md)를 따른다. 실제 release와 GPU/AMP·Drive/새VM pilot은 남아 있다. [전체 학습 준비 상태](FULL_TRAINING_READINESS.md)를 따른다.
