# 완료 bundle의 단일 프레임 진단 데이터 계약

`rewind_bundle_data.PreparedBundleReader`는 [준비 bundle](PREPARED_BUNDLE.md)의 RGB와 pose candidate를 UI·모델·optimizer 없이 읽는다. `dataset_sample`, `input_source=gt_2d`, `diagnostic_only=true`, `train_ready=false`를 유지한다. generic `m1_features_v1` 문서나 학습 release를 발급하지 않으며 기존 trainer에 자동 연결하지 않는다.

## 명시 입력

`PreparedBundleReader(root, completion_sha256=..., split=..., diagnostic_only=True, limits=ReaderLimits())`를 사용한다. SHA는 `COMPLETE.json`의 **content_sha256**이며 파일 byte SHA와 구분한다. split은 `train`/`validation`/`test` 중 하나로 고정한다. 빈 split·명시 진단 동의 누락·완료 marker/파일/출처 불일치는 실패한다.

생성 시 완료 marker·정확한 파일 목록과 순서·전체 파일 byte/SHA를 검사한다. 이 단계는 **전체 bundle I/O**다. metadata pin·plan 전체 target·matching·RGB/pose의 sequence/frame/view/job·producer manifest·candidate source를 대조하고 전체 motion map의 split 누수를 확인한다. candidate를 한 job씩 감사하며 raw62관절의 유효 좌표/mask→mm 변환과 candidate의 fields/GT2D 특징을 대조한다. 전체 raw 감사는 첫 선택 split 밖의 candidate도 포함한다. marker는 변경 탐지용 hash이며 서명이나 원본 데이터의 신뢰 증명이 아니다.

공통 spec은 기존 `real_pose_spec_v1`의62관절·stereo1024²·mm/device_relative 가정을 사용한다. 단위·geometry 가정과 calibration/3종 split의 명시 pin을 보존하며 현재 원본 archive나 좌표 projection을 새로 검증하지 않는다. 원본 압축 조각·queue가 없는 보존 bundle도 읽을 수 있다. 코드 버전이 다른 producer를 현재 코드 hash로 덮어쓰지 않는다.

## 반환 API

`len(reader)`, `reader.metadata(index)`, `reader.identity`, `reader.load_batch(indices, include_rgb=True)`를 제공한다. index는 선택 split 안의 증가하는 `(sequence_id, frame_index)` 목록을 가리킨다. batch는 요청 index 순서를 보존하고 서로 다른 candidate job도 포함할 수 있다. `include_rgb=False`면 PNG를 decode하지 않는다.

| 필드 | shape / 의미 |
| --- | --- |
| `images` | RGB 요청 시 CPU FP32 `[B,2,3,1024,1024]`, 좌/우 native RGB `[0,1]` |
| `gt_xy` | CPU FP32 `[B,2,62,2]`, native pixel GT2D |
| `annotation_valid` | CPU bool `[B,2,62]`, GT 좌표의 수치/annotation 유효 mask |
| `target_mm` | CPU FP32 `[B,1,62,3]`, device_relative GT3D mm |
| `target_valid` | CPU bool `[B,1,62]`, GT3D 유효 mask |
| `features` | CPU FP32 `[B,1,868]`, GT2D의 관절별 좌/우 `xy, valid, confidence, known, visibility, known` |
| `time_valid` | CPU bool `[B,1]`, 현재 독립 frame만 유효함을 뜻함 |
| `metadata` / `identity` | key/split/motion·job/member/SHA·완료 pin·joint order·blockers·진단 계약 |

좌표가 없는 자리의 tensor 값은0이며 mask로 구분한다. 화면 밖의 유효 GT2D는 원래 값과 `annotation_valid=true`를 유지한다. annotation mask를 visibility로 바꾸지 않는다. confidence/visibility는 unknown이며 GT2D 특징의 known bit는0이다. 특징 정규화는 고정 native 크기에 대한 `2*x/(width-1)-1`이며 clipping·sequence 통계·미래 frame을 사용하지 않는다. 기존 `pack_window(T=1)`의 수치/shape와 같다. GT2D 특징을 실제 예측 입력이나 frontend 성능 결과로 해석하지 않는다.

`source_scope=synthetic_fixture`인 합성 통합 bundle도 candidate 자체의 `data_kind=dataset_sample`은 바꾸지 않는다. 두 필드를 함께 반환한다. split/motion은 전체 검증 후 선택 split만 반환하고 다른 split의 index를 섞는 API는 없다.

항상 `temporal_ready=false`와 `time_seconds=None`이다. gap이 있거나 job이 바뀌어도 temporal history를 만들지 않는다. metadata에는 원문 `timestamp_raw`를 노출하지 않는다. nominal FPS로 timestamp를 생성하거나 초 단위로 변환하지 않으며 `time_valid`는 시간 정합 증명이 아니다.

## lazy 범위와 상한

bundle은 기존128MiB/512파일/4096inventory 항목 안에서 검증한다. 최대32triplet, candidate 한 개4MiB/32record, batch2개/반환 tensor64MiB만 허용한다. `ReaderLimits`는 이 상한을 낮추는 옵션이다. 단일 native RGB가 들어갈 수 없는 tensor 상한은 기존 RGB loader의 생성 단계에서 먼저 거부될 수 있다.

생성 감사 후 candidate cache를 비운다. 요청 시 한 candidate만 parse/cache하며 job 전환 전에 이전 참조를 해제한다. cold job은 candidate와 해당 raw pose를 다시 parse·감사하고 warm job은 candidate와 **그 job 전체 raw pose 파일**의 pin을 다시 읽어 검사한다. batch마다 metadata/producer/report 등의 참조 파일과 완료 marker도 재검사한다. PNG hash/CRC/decode는 요청한 RGB에만 수행하며 decode tensor를 캐시하지 않는다. metadata-only 조회는 생성 때 검증한 snapshot을 반환한다. 다른 요청의 모든 candidate나 미요청 PNG를 매 batch 재감사하지 않는다.

단일 reader는 순차 호출용이며 thread/process 사이에서 cache를 공유하는 계약은 없다. 반환 batch는 호출자 소유다. 호출자가 여러 batch나 metadata를 보관하는 메모리는 cache 하나/반환 tensor 상한에 포함되지 않는다. 검사하는 sampled RSS는 CUDA/모델 등 호출 프로세스 전체를 포함한4GiB이며 reader 전용 메모리의 증명이 아니다. 기존 합성 GPU worker 관측RSS 약2.676GB를 고려한 값으로 bundle builder의1GiB 상한과 구분한다. 생성/각 batch의120초 및 RSS는 phase/청크 경계에서 협조적으로 검사하며 순간 peak나 강제 종료를 보장하는 watchdog은 아니다. 반환 tensor는 호출자의 Torch default device와 무관하게 CPU를 명시한다.

## CLI inspect

```powershell
python -m rewind_bundle_data inspect --root bundle --completion-sha256 CONTENT_SHA256 --split train --diagnostic-only --output new_inspection
python -m rewind_bundle_data inspect --root bundle --completion-sha256 CONTENT_SHA256 --split train --diagnostic-only --indices 0 1 --include-rgb --output new_batch_inspection
```

첫 명령은 전체 byte/raw 감사와 metadata 검사만 하며 Torch·모델·학습 모듈을 import하지 않는다. 두 번째는 bounded native RGB/GT batch까지 확인한다. 새 별도 output에 `report.json`만 저장하며 입력이나 이전 출력은 수정하지 않는다. 실패 이유를 보존하고 optimizer·네트워크·설치·GPU 실행을 하지 않는다.

## 검증과 남은 경계

회귀 fixture는 저장소 코드로 작은 분할 archive→queue→bundle을 새로 만든다. ignored `outputs/`의 특정 보존본에 의존하지 않으며 index 의존성이 없는 환경은 명시 skip한다. 기존 converter/raw·`pack_window(T=1)`의 수치/mask, job 사이 batch·정렬·split, 밖의 유효 좌표와 누락 mask, 한 job parse/cache·요청 PNG decode, cached raw 변조·marker 검증 parity, CPU 반환과 metadata CLI import 경계를 검사한다. 보존된3triplet/2pose job bundle 대조는 별도 ignored 실행 근거로 남긴다.

실제 Serena의 기존 작은 진단 API와 고정 GPU 소스/배포물을 변경하지 않는다. 합성 GPU `20b7aa9`의 A100 FP32/AMP 재개 시험은 이미 통과했으며, 이 reader 작업은 로컬 데이터 공급 검증이다. 실제 release·정식 batch 학습·실제 데이터 정확도·독립 validation·원본 motion/내용/시간/단위 근거·temporal window·Drive 새VM 지속성은 여전히 후속이다.

개발8개(22.499초) 및 root 독립8개(23.447초)가 failure/error/skip0으로 통과했다. 보존 bundle의3triplet/2pose job 중 train2개를 native RGB 포함 batch로 읽고 전체 입력 SHA 불변을 확인했다. metadata CLI subprocess의 Torch·모델·학습 import는0이며 sampledRSS28,426,240bytes, batch 관측 peak274,923,520bytes였다. `outputs/full_training_readiness_20261102/developer/bundle_data/`의 `final_tests.txt`, `preserved_inspection.json`, `metadata_imports.json`과 root `bundle_data_tests.txt`에 근거를 보존했다. 초기 상한 검사의 기대 지점을 기존 RGB 생성 단계보다 뒤로 잡은 fixture를 수정한 뒤의 최종 결과다.

후속 [검토형 실제 RGB2D release adapter](REAL_RGB2D_RELEASE.md)는 이 reader의 진단 계약을 바꾸지 않고 외부 policy로 명시 허용된 T1/2D 공급만 연결한다. 원래 dataset_sample/diagnostic_only/train_ready=false와 기존 trainer gate를 전역 해제하지 않는다. 현재 실제Serena는 train2/validation0·원본motion/검토근거 미충족으로 차단되어 실제 학습/release 발급은 없다.
