# 완료 queue의 로컬 진단 bundle

`rewind_prepared_bundle`은 완료된 준비 queue의 RGB·pose 추출본과 명시 metadata를 새 디렉터리로 묶는다. RGB manifest와 pose job별 candidate/audit, 키별 연결 provenance를 발급한다. `train_ready=false`이며 학습·release·learning shard 생성 기능은 없다.

## 입력과 지원 경계

- 고정 plan과 `prepared_not_training_release` 상태의 완료 queue를 받는다. 미정산 inflight·부분 완료는 거부한다. plan/state/matching의 hash와 job 순서·producer extraction·원본 source SHA 선언·모든 target 파일 hash를 대조한다.
- selection은 queue의 **전체** sequence/frame triplet과 정확히 같아야 한다. 부분 선택이나 자동 sequence 선택은 지원하지 않는다. sequence별 split/evidence와 증가하는 frame 목록을 직접 제공한다. 출력은 sequence/frame 순으로 정렬한다.
- RGB는 기존 builder의 최대4개 extraction·32MiB payload 범위를 유지한다. pose는 기존 `real_pose_spec_v1`의62관절·1024² 좌우 영상·단위/좌표 가정과 calibration 및3종 split SHA를 요구한다. 이를 자동 추론하거나 새 관절 규격으로 바꾸지 않는다.
- 전체 sequence를 덮는 명시 motion map을 먼저 검증해 job 사이의 동일 motion/split 누수를 차단한다. pose job별 map은 입력 전체 map의 sequence 부분집합이며 부모 파일 SHA와 sequence 목록을 보고서에 남긴다. 서로 다른 extraction report를 합성하지 않는다.
- dataset version/source scope 혼합·중복 view·missing/partial payload·변조된 producer/code hash를 거부한다. 같은 dataset/scope의 여러 명시 archive는 허용한다. code 변경 허용 우회 옵션은 없다.
- 원본 압축 조각은 없어도 된다. 보존된 plan/producer의 source identity와 로컬 추출 파일을 검사하며 **현재 원본 archive를 다시 읽거나 원격 identity·전체 gzip 무결성을 확인하지 않는다**. UNC 및 명시 `/content/drive` 입력은 거부하고 로컬 경로만 사용한다. queue writer를 호출하거나 consumed를 표시하지 않는다.

selection 예시는 아래와 같다. split/evidence는 실제 제공된 근거에 맞춰 작성한다.

```json
[{"sequence_id":"Room/Day/person/glasses/001/Motion_A","frames":[0,10],"split":"train","split_evidence":"제공한 split 파일과 확인 근거"}]
```

## API와 CLI

기존 프로젝트의 Python·Torch·Pillow·psutil 환경을 사용한다. bundle build/verify 자체는 indexed_gzip이나 원본 archive를 열지 않지만 queue를 만드는 기능은 기존 index 환경을 요구한다. 신규 설치는 수행하지 않았다.

```powershell
python -m rewind_prepared_bundle build --plan plan.json --queue queue --selection selection.json --spec spec.json --calibration calibration.json --split-dir split --motion-map motion.json --output new_bundle --id bounded_local_bundle
python -m rewind_prepared_bundle verify --root new_bundle
```

`build_bundle(plan_path, queue_root, selection_path, spec_path, calibration_path, split_dir, motion_map_path, output, manifest_id=...)`와 `verify_bundle(root)`가 같은 동작을 제공한다. output의 부모는 존재해야 하며 output은 새 경로여야 한다. 입력 queue/metadata와 겹치는 출력은 거부한다.

성공 출력은 다음과 같다.

| 경로 | 내용 |
| --- | --- |
| `metadata/` | 원본 plan/state/matching/selection/spec/calibration/전체 motion map 및 split의 byte 그대로 복사 |
| `inputs/<job>/` | 원본 extraction manifest와 전체 target 파일. producer hash와 원본 identity 보존 |
| `rgb/manifest.json`, `rgb/report.json` | 기존 RGB builder 결과. 공통 이미지 root는 bundle의 `inputs/` |
| `pose/<job>/candidate.json`, `audit.json`, `motion_map.json` | job별62관절 candidate와 기존 blocker·명시 motion 부분집합 |
| `provenance.json` | RGB record index와 pose candidate/record index, 좌우/pose 원본 job·member·SHA 연결 |
| `report.json` | 입력 SHA·지원 범위·예산·후속 차단 사유·상태 |
| `COMPLETE.json` | 마지막에 쓰는 완료 marker. 파일 목록·SHA/byte와 입력 metadata SHA 포함 |

`verify_bundle`은 완료 marker와 정확한 파일 목록·모든 파일 hash/byte 및 명시 metadata SHA를 대조한다. 입력 queue나 압축 원본이 없어도 검사할 수 있다. marker는 변경 탐지를 위한 hash이며 서명이나 신뢰 증명서가 아니다. 이 검사는 원본 재감사·좌표/시간 근거 확립·학습 허용 판정이 아니다.

## 예산과 실패 보존

최대32triplet, RGB4job, RGB32MiB+pose32MiB payload, metadata8MiB, 새 출력128MiB/512파일 및 inventory4096항목을 제한한다. 입력 extraction report는 개별1MiB, spec256KiB, pose raw4MiB/RGB8MiB 이하만 지원한다. 기존 converter의 더 좁은 file/총 입력/output 상한도 적용하며 raw JSON을 두 번 읽는 비용을 preflight에 포함한다. 상한을 넘으면 더 작은 별도 queue가 필요하다.

120초와 RSS1GiB는 phase/파일/청크 경계에서 협조적으로 검사한다. 외부 watchdog이나 순간 peak의 완전한 계측은 아니며 기존 RGB builder·pose converter의 자체 deadline도 유지한다. Torch import 이전/다른 프로세스 메모리를 이 값으로 증명하지 않는다. RGB 호출 전 출력 metadata 여유8MiB를 확보하고 이후 파일은 누적 출력 예산 안에서 쓴다. 실패 보고서 저장 여유를 남기지만 디스크 고장/강제 종료에서 보고서 저장을 보장하지 않는다.

입력은 읽기만 하며 마지막에 pin된 plan/state/matching/metadata/추출 파일의 byte 동일성을 다시 확인한다. 중간 변경은 성공 발급을 막는다. 오류가 나면 부분 파일과 `report.json`의 이유를 보존하고 완료 marker를 제거한다. 실패 디렉터리의 재개/덮어쓰기/자동 삭제는 지원하지 않는다. 재시도는 새 출력 경로를 사용한다. 강제 종료로 남은 불완전 marker도 verify에서 실패한다.

## 검증과 남은 범위

로컬 합성 시험은1024² PNG와 수치적으로 생성한62관절 pose의 분할 gzip에서 실제 queue를 완료하고,2개 pose job을 포함한 bundle을 생성·검사한다. 명시 split/motion map과 source_scope를 합성으로 표시한다. 기존 converter는 candidate를 `dataset_sample` 진단 계약으로 발급하므로 이 필드를 합성 학습 계약으로 바꾸지 않는다. bundle report의 `synthetic_integration_only`가 근거 범위를 표시한다.

개발47개 회귀와 최종 POSIX inventory 정렬·정상 완료·변조3개가 failure/error/skip0으로 통과했다. Root 신규20개 및 verify3개도 통과했다. 보존 근거는 `outputs/full_training_readiness_20261102/developer/prepared_bundle/`의 `final_tests.txt`, `final_inventory_tests.txt`, `final_integration/integration_summary.json`이다. 최종 bundle은33파일/305,429bytes, 완료 content SHA는 `82fe2e7be9234b750c1cb0e343991e61f5e05ee0b9282135b98f4cb989157e8a`이며 원래 queue 제거 후에도 검사했다. 의도적 실패본은 같은 `final_integration/failed_bundle/`에 부분 파일과 사유를 보존했다. 초기 개발의 예상 오류명·fixture scope 오기와 보존 harness 초기화 누락을 수정한 뒤 얻은 최종 근거이며, 이전 출력은 이 최종 경로와 구분한다.

job별 train/validation 누락 blocker는 **각 candidate**의 범위다. bundle 전체에 양 split이 있어도 독립 validation·원본 motion 검증이 완료되었다고 해석하지 않는다. RGB/pose 내용 정합·timestamp·전체 archive coverage·단위 가정의 독립 검증, 실제 데이터 release와 candidate→학습 shard 변환은 여전히 미완료다.

이 bundle 구현 기록의 GPU 미실행은 준비 당시 상태다. 이후 고정 `20b7aa9`의 합성 A100 FP32/AMP 시험이 통과했으며 [최신 GPU 결과](SYNTHETIC_GPU_PILOT.md)를 따른다. 고정 배포물은 보존한다. 후속 [T1 진단 reader](BUNDLE_T1_DATA.md)는 candidate를 학습 문서로 바꾸지 않고 RGB/GT 데이터를 읽는다. 실제 데이터 release·정식 학습·temporal 지원은 여전히 별도다.
