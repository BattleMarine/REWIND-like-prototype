# ArchVis 직접 추출본 입력

2026-11-09 구현·root 검토. `rewind_prepare.extracted`는 압축을 푼 RGB·pose 폴더에서 T=1 manifest를 작성하고 필요한 배치만 읽는다. 약1,000 train 페어와 ArchVis 전체 train이 동일 계약을 사용한다. **[2D trainer 연결](ARCHVIS_DIRECT_RGB2D.md)은 로컬 검증했으며 실제 Colab 입력 고정·실행은 후속**이며 결과는 `diagnostic_only=true`, `train_ready=false`, `temporal_ready=false`다.

## 입력과 선택

- RGB·pose root 바로 아래에 `ArchVisInterior_ArchVis_RT/Day/...` 구조가 있어야 한다. 로컬 파일 또는 Colab에서 접근 가능한 추출본 경로를 사용한다. 이 모듈은 Drive 다운로드·인증·압축 해제를 수행하지 않는다.
- 공식 `train.txt`, `validation.txt`, `test.txt`, `configs/real_archvis62_v1.json`, 해당 spec이 고정한 Scaramuzza calibration JSON이 필요하다. 공식 split·calibration SHA를 검사한다.
- motion map은 `explicit_motion_map_v1` 형식이며 로컬 적격 train/validation의 모든 sequence를 정확히 포함해야 한다. 각 항목은 `motion_id`, `identity_status`, `evidence`를 가진다. 원본 motion이 미확인인 경우 `declared_unverified`로 표시한다. 서로 다른 임의 ID를 부여한 것만으로 독립성을 입증하지 않는다.
- 선택적 `--inventory`와 `--inventory-sha256`은 탐색 세션의 고정 `sequence_summary.json`을 대조한다. 알려진 Drive 동명 파일은 마운트에서 하나로 보이더라도 해당 frame 키를 제외한다. ArchVis 실제 실행본에는 이 조사본을 포함해야 한다.
- 기본 `--train-limit 1000`은 고정 seed와 sequence/frame 키의 SHA 정렬로 선택한다. `--full-train`은 같은 로컬 입력의 적격 train 전체를 선택한다. 두 모드 모두 공식 validation의 적격 페어 전부를 유지하며 `validation_keys_sha256`으로 집합을 비교한다. 파일이 바뀌거나 누락되면 validation 집합도 달라질 수 있으므로 전환 전에 이 값을 확인한다.

## 개발 진입점

`python -m rewind_prepare.extracted --help`에서 실제 인자를 확인한다. 필수 경로 인자는 `--rgb-root`, `--pose-root`, `--split-dir`, `--motion-map`, `--spec-path`, `--calibration-path`, `--output-dir`이다. `--dataset-version`, `--max-read-bytes`, `--max-seconds`도 명시해야 한다. 출력은 원본 폴더 밖의 새로운 디렉터리로 지정한다.

읽기 상한은 전체 디렉터리 엔트리 수, 입력 바이트, 협력적 경과시간 검사다. 시간 제한은 진행 중인 파일시스템 호출을 강제 취소하는 timeout이 아니다. 출력 manifest는 최대64MiB, RGB 파일은 최대8MiB, pose 파일은 spec 상한을 따른다. 첫 실행도 train1,000 외 validation 전체의 파일을 읽으므로 준비 비용에 포함한다. 실제 Colab 예산은 별도 실행본에서 고정한다.

manifest에는 파일별 상대경로·크기·SHA, split/spec/calibration/motion/코드 식별값과 감사 결과를 담는다. 출력되는 `content_sha256`은 manifest의 canonical 내용 hash이며 JSON 파일 자체의 SHA와 구분한다. 선택 파일 검사 실패 시 성공 manifest를 발급하지 않는다. 누락·중복 키는 제외 목록으로 보고하며 `--full-train` 성공이 공식 전체 파일 확보를 뜻하지는 않는다. missing/incomplete/duplicate 보고를 함께 검토한다.

`ExtractedStereoT1(manifest_path, rgb_root, pose_root, spec_path, expected_sha256=content_sha256, split="train")` 또는 `split="validation"`로 로더를 연다. `load_batch([indices])`는 최대4페어를 CPU에서 읽는다. native RGB는 `[B,2,3,1024,1024]`, GT2D는 `[B,2,62,2]`, GT3D는 `[B,62,3]` mm이며 2D annotation mask와3D target mask를 분리한다. 매번 파일 hash를 재확인하며 timestamp는 추정하지 않는다. 2D trainer는 필요한 native 페어를 순차 resize하여 배치를 구성하고 학습 재개를 제공한다. 실제 GPU 처리량은 미측정이다.

## 검증과 한계

Root는 신규 입력 계약과 기존 pose converter/RGB manifest/frame input/prediction 관련40개 검사를 독립 실행해 통과를 확인했고 CLI help도 확인했다. Pilot 선택 재현, 동일 validation,32개 초과 입력, 누락·중복 제외, test payload 미접근, 손상 PNG·pose 수치 오류, motion 누출과 변경 파일 거부를 포함한다. 개발 검증에서는 기존 실제 triplet1개로 PNG decode·62관절·재투영도 확인했다.

실제 Drive1,000페어 준비·원본 motion 독립성·사람의 영상 정합 확인·GPU 학습은 아직 미실행이다. 수치 재투영은 RGB 이미지 속 사람과 pose가 실제로 대응함을 단독 입증하지 않는다. 과거 고정 revision용 `tests/check_local_rgb.py`는 loader source SHA를 고정하므로 원래 배포 revision에서 실행한다. 기존 archive 학습 gate를 이 입력으로 우회하지 않는다.
