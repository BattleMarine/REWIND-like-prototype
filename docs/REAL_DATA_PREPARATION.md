# 실제 pose 규격 고정과 candidate 준비

2026-10-16. 사용자가 승인한 규격 고정→변환기→RGB/validation 확보→작은 GT2D 학습 중 **첫 규격·변환 구현 묶음**이다. 실제32개 JSON을 검증해 새 candidate 문서로 변환했으며 학습은 실행하지 않았다. RGB·독립 validation·원본 motion 식별이 미충족이므로 기존 M1/M2/M3 guard를 유지한다. 후속 자료 준비는 [다음 실행 안내](REAL_DATA_NEXT_RUN.md)를 따른다.

## 채택한 새 규격

[real_archvis62_v1.json](../configs/real_archvis62_v1.json)은 기존 개발용/synthetic config를 보존하고 별도로 추가한 버전 있는 **프로젝트 학습 가정**이다. [관절 대응표](JOINT_MAPPING.md)와 [좌표 검토](COORDINATE_AND_TRAINING_REVIEW.md)의 후보를 다음처럼 고정했다.

| 항목 | 새 규격 |
| --- | --- |
| ID / preprocessing | `real_archvis62_v1` / `archvis_cm_mm_reflected_world_v1` |
| 관절·순서 | Body22 다음 left finger20, right finger20의 고유62. 이름으로 gather |
| Upper / wrist | Upper13, `hand_l/r`를 각 손 local origin으로 선택. Body가 wrist 소유 |
| Hand target | Fingers40의 `wrist_relative_parallel_device`, implicit wrist offset0이며 모델 target에서 손목2개 제외 |
| 평가16 | 기존 pinned 평가16의 별도 이름·순서를 보존. 전체62 평균과 구분 |
| Graph | 기존61edges를 명시적 프로젝트 prior로 채택. Native parent 검증은false |
| 단위 | Raw cm→출력mm×10, `adopted_metric_anchor_inference`를 근거 강도로 기록 |
| Camera | `p_G=F p_U`, `F=diag(1,-1,1)`, proper `R=BQᵀ`, `t_mm=−10 R F C_U` |
| 시간 | 명목25FPS를 출처 정보로만 보존. Timestamp/초 단위 시간을 만들지 않음 |

Cm는 공식 UE2의12cm rig와 raw baseline약12·좌표계간 거리 보존을 결합해 채택했다. Raw schema의 직접 `unit=cm` 선언이 확보됐다고 표시하지 않는다. `hand_l/r` 역시 rig key를 local origin으로 선택한 것이며 해부학적 손목 중심을 검증했다는 뜻이 아니다.

Graph는62nodes/61edges의 tree이며 body 부분22/21, wrist를 제외한 finger 부분40/30의10개 chain이다. M3 방식의 spatial attention은 전체 finger를 연결하지만 이 구조의 성능을 검증한 것은 아니다. 새 규격 채택으로 기존 개발용 config의 미검증 상태·원본 좌표·합성 checkpoint를 바꾸지 않았다.

## 코드와 입력

`rewind_prepare`는 Python 표준 라이브러리만 사용하는 새 모듈이다.

| 경로 | 역할 |
| --- | --- |
| `spec.py` | 62/22/40/13/16 역할·이름 순서·wrist/graph·단위/변환·상한 검증 |
| `coordinates.py` | 고정 camera 변환·inverse·proper rotation·Scaramuzza projection |
| `converter.py` | Raw hash·extraction manifest·공식 split·명시적 motion map 대조, mask/특징/candidate/report |
| `provenance.py` | SOURCE_REVISION.txt 우선, Git HEAD/dirty 또는 코드 hash 식별 |
| `__main__.py` | 새 output에만 candidate/report를 저장하는 CLI |

CLI의 입력은 추출본 `files` root, 원본 `extraction_manifest.json`, 공식 split3개 txt 폴더, 명시적 motion map, 새 spec, 공식 Scaramuzza JSON이다. Archive를 직접 해제·다운로드하지 않는다. Spec에는 현재 검토한 split train/validation/test 각각의 SHA-256과 calibration SHA-256을 pin했다. 다른 버전은 같은 공식 이름이라는 이유만으로 수용하지 않으며 별도 규격/출처 검토가 필요하다.

Extraction manifest는 자체 content hash·usable 상태·pose kind·완전 regular payload 기록을 검사한다. 파일의 member 경로·bytes·SHA-256이 일치해야 한다. 부분 archive에서 이미 완전히 읽은 payload는 사용할 수 있지만 전체 archive 검증으로 승격하지 않는다. 원본은 읽기만 하고 변환 후에도 hash가 같은지 확인한다.

Motion map은 자동 생성하는 원본 ID가 아니다. 각 sequence의 ID·원본 식별 상태·근거를 호출자가 명시한다.

```json
{
  "schema_version": "explicit_motion_map_v1",
  "sequences": {
    "Environment/Day/person/glasses/001/Motion": {
      "motion_id": "declared_local_example",
      "identity_status": "declared_unverified",
      "evidence": "Local sample label only; original mocap identity unknown."
    }
  }
}
```

실제32개 변환에서는 두 sequence에 명시적인 **local sample label**을 부여하고 `declared_unverified`로 기록했다. 파일명에서 원본 mocap ID를 알아낸 것으로 취급하지 않는다. 같은 ID가 다른 공식 split에 나타나면 거부한다. `original_verified`와 evidence를 선언받더라도 이 converter가 그 근거의 진실성을 외부에서 자동 증명하지 않으며 training release도 하지 않는다.

## 변환·mask·시간 계약

각 frame에서 raw camera의 world 위치·Euler field와 device midpoint를 확인한 뒤 proper R 및 mm translation을 만든다. Baseline 허용 구간은11.98~12.02raw다. 잘못된 side·배율·camera convention, world/camera/device 불일치 또는 annotation projection 오차가 기준을 넘으면 candidate를 만들지 않는다. World/inverse 잔차 기준은1e-6raw, projection은0.001px다. 확인 가능한 field 쌍에만 검사를 적용하고 검사 수·최대 잔차를 보고한다.

출력의 `fields`는 명시한62개 관절 순서대로 각 필수 field의 `coordinates`와 bool `valid_mask`를 가진다. Invalid/누락 vector는null·mask=false이며 다른 field를 함께 무효화하지 않는다. 실제 데이터에 없던 visibility/confidence는 좌우 null로 보존한다. 필드별 수치 유효성과 화면 안/가림 여부를 구분하고 2D를 clipping하지 않는다.

| 출력 | Shape/단위/정책 |
| --- | --- |
| `fields.camera_left/right_pts2d` | 각62×2, 원본 pixel 그대로 또는null |
| `fields.camera_left/right_pts3d`, `device_pts3d` | 각62×3, 원본 raw×10의mm 또는null |
| `observation_features` | **Record당868개의 값**(62×14). 868개의 독립 관측을 뜻하지 않음 |
| View별7값 | 고정 정규화x/y, valid, confidence/known, visibility/known |
| Pixel 정규화 | `2*x/(width−1)−1`, `2*y/(height−1)−1`; 범위 밖도 유지 |
| `camera_pose` | left/right 각각 proper R[3,3], world→camera t[3] mm |
| `hand_target` | Fingers40×3의 손목 상대mm, 별도40 bool mask |

3D는 **한 번만** mm로 바꿨고 candidate contract도mm라고 표시한다. 이후 적법한 학습 연결이 구현되면 mm target에 다시×10하지 않아야 한다. M2 camera translation 역시 이미mm다. Unit 환산과 향후 `target_scale_mm` 정규화는 다른 연산이며 준비 단계에서는 target scale 정규화를 적용하지 않는다.

Hand local mask는 finger/device3D와 해당 wrist/device3D가 모두 유효할 때만true다. Wrist GT가 없으면 해당 손 local target은null이지만 finger의 global GT·mask는 유지한다. Hand target은 device-relative라고 잘못 표기하지 않고 내부 계약을 `wrist_relative_parallel_device`로 구분한다. 준비 단계의 GT local target을 실제 평가 결과에 붙이는 inference 기능은 없다.

Sequence/frame별로 정렬하고 중복 frame은 거부한다. 연속 구간 시작/gap에 `history_reset`을 표시하되 window padding·보간을 만들지 않는다. `timestamp_raw`는 제공된 scalar/string만 보존하고 `time_seconds=null`, 의미는unverified다. 미래 데이터가 바뀌면 전체 dataset/manifest hash는 바뀌지만, 앞선 record의 모델 입력·좌표는 영향을 받지 않는다.

## Candidate는 학습 입력이 아니다

산출물 schema는 **`real_pose_candidate_v1`**이며 `m1_features_v1`이 아니다. `data_kind=dataset_sample`, `input_source=gt_2d`를 유지하고 `verified`를 채워 넣지 않는다. M1/M2/M3의 기존 로더가 schema 단계에서 거부하는 것을 검사했다. JSON header를 바꿔 우회하거나 실제 데이터를 synthetic으로 표기해서 학습하지 않는다.

`train_ready=false`와 `training_release=not_implemented`는 이 구현 단계의 고정 정책이다. RGB·validation이 나중에 준비되더라도 이 converter가 자동으로 학습 계약을 발급하지 않는다. 자료/근거를 검토하고 별도 training release 연결을 구현해야 한다.

실제32개 report의 차단 사유는 다음과 같다.

- `candidate_schema_not_training_release`
- `rgb_pose_alignment_not_checked`
- `independent_validation_not_verified`
- `required_train_validation_splits_missing` — 두 sequence 모두 공식train
- `original_motion_identity_unverified` — 현재 ID는 local declaration
- `gapped_sequences_require_contiguous_selection` — Opening은14개의 산재 관측

Brutal18의 연속 구간은 overfit 후보로 유지한다. Opening의14개를 이어 붙이거나 독립 validation이라고 바꾸지 않는다. 이번 단계에서 학습 guard를 완화하거나 Trainer를 실행하지 않았다.

## 로컬에서 재현한 명령

2026-10-23 정리 후 아래 명령의 `outputs/...` 입력은 아카이브에 보관한다. [개별 복원 안내](LOCAL_STORAGE.md#개별-파일-복원)에서 원래 경로와 SHA256을 확인해 다음 입력을 복원한 뒤 사용한다. 원본 `data/raw/...`와 현재 코드·spec은 기존 위치에 보존한다. 이번 문서 갱신은 변환·학습 실행을 요청하지 않는다.

- `outputs/m0_next_evidence_20261015/developer/official_split/UnrealEgoData2/`의 `train.txt`, `validation.txt`, `test.txt`
- `outputs/m0_next_evidence_20261015/developer/official_calibration/fisheye_calib_unrealego_scaramuzza.json`
- `outputs/real_data_prepare_20261016/developer/motion_map.json`

Colab 참조 bundle 예시를 재현할 때 필요한 `outputs/real_pose_references_20261016.zip`도 아카이브에서 개별 복원한다. 과거 실행 근거·manifest와 변환 입력을 구분하고 현재 고정 warm192 테스트 자료를 대신 사용하지 않는다.

아래 경로는 **당시 Windows 로컬 저장소에서 준비했던 Git 제외 파일**의 원래 경로다. 현재는 위 입력 복원이 선행되어야 한다. 새 checkout/Colab에 존재한다고 가정하지 않는다. 프로젝트 root에서 실행했으며 이번 `python`은 `.venv/Scripts/python.exe`다.

```sh
python -m rewind_prepare --input data/raw/colab_archvis_pose_20261015/bundle/pilot/files --extraction-manifest data/raw/colab_archvis_pose_20261015/bundle/pilot/extraction_manifest.json --split-dir outputs/m0_next_evidence_20261015/developer/official_split/UnrealEgoData2 --motion-map outputs/real_data_prepare_20261016/developer/motion_map.json --spec configs/real_archvis62_v1.json --calibration outputs/m0_next_evidence_20261015/developer/official_calibration/fisheye_calib_unrealego_scaramuzza.json --output outputs/real_pose_candidate_new
```

Output은 새 폴더여야 하며 원본 files root와 겹치거나 그 상위/하위, 저장소 `data/raw` 또는 입력 metadata를 덮는 경로는 거부한다. 성공 시 `candidate.json`·`report.json`을 저장하고 exit0이다. **Exit0은 변환 성공이며 학습 준비 통과가 아니다.** 입력/좌표/상한 오류는 exit2·`status=rejected` report를 남기고 candidate를 성공 산출물로 남기지 않는다. 기존 output을 덮지 않는다.

기본 상한은 파일256개·각4MiB·총읽기32MiB·출력64MiB·협력적 경과30초다. 총읽기에는 manifest·split/calibration·원본과 사후 hash 확인을 위한 재읽기가 포함된다. OS 전체 hard deadline/RSS 제한은 아니며 다른 조건의 운영 예산이 확정됐다는 뜻도 아니다. 이 CLI에는 네트워크·archive 다운로드가 없다.

## Colab 참조 bundle의 경로

Root가 만든 `outputs/real_pose_references_20261016.zip`은 split3개·calibration·현재 motion map·출처/hash manifest의6파일만 담는다. 원본 pose는 사용자가 보관한 기존 `rewind_pose_sample_2610151012.zip`을 별도로 사용한다. 최신 소스 ZIP에는 **rewind_prepare와새config**가 포함되어야 하며 준비 후 실제 경로를 확인한다.

예를 들어 원본 ZIP을 `/content/pose_sample/`에 풀어 `pilot/files`·`pilot/extraction_manifest.json`이 있고, 참조 ZIP을 `/content/`에 풀어 `real_pose_references/`가 생겼을 때의 명령은 다음과 같다.

```sh
python -m rewind_prepare --input /content/pose_sample/pilot/files --extraction-manifest /content/pose_sample/pilot/extraction_manifest.json --split-dir /content/real_pose_references/split --motion-map /content/real_pose_references/motion_map.json --spec configs/real_archvis62_v1.json --calibration /content/real_pose_references/fisheye_calib_unrealego_scaramuzza.json --output /content/real_pose_candidate_new
```

위는 reference ZIP의 실제 내부 경로에 맞춘 **후속 실행 예시**다. 이번에는 현재 Colab runtime이 없어 업로드·추출·이 converter 실행을 Colab에서 하지 않았다. 로컬의 긴 `outputs/m0_next_evidence_...` 경로를 Colab에 그대로 넣지 않는다. 새 source ZIP에서는 `SOURCE_REVISION.txt`의40자리 SHA가 우선되며 Git이 없으면 dirty는null이고 코드/config hash를 남긴다.

## 실행한 검증과 실제 결과

증거는 Git 제외 `outputs/real_data_prepare_20261016/developer/`에 있다. `execution.json`·`command_0/1_stdout/stderr.txt`는 전체 회귀와 실제 CLI의 원 실행 결과다. 추가한 음수2D 보존 회귀를 포함한 최종 테스트 기록은 `final_tests_execution.json`·`final_tests_stdout/stderr.txt`다.

실제32파일 변환은 Python3.12.14/표준 라이브러리로 exit0이었다. **최종 전체133개(기존119+준비14개) 테스트를 통과**했다. 기존 전체 테스트와 guard 검사에는 별도 torch2.7.1+cpu 환경을 사용했으며 모델 학습은 하지 않았다. 처음132개 전체 회귀 뒤 음수2D 보존 fixture1개를 추가했고133개를 다시 실행해 확인했다.

| 실제 변환 확인 | 결과 |
| --- | --- |
| Record·관절·특징 | 32 records, 각62관절, 각868값, local hand40 |
| 선택62의 필수field5 | 각1,984 valid, 합계9,920배열. 원본pixel/단일mm 환산 일치 |
| 원본 관절87 | `source.raw_joint_names`와 raw hash로 추적. 원본25개를 삭제하지 않음 |
| 시간·순서 | Brutal0~17 단일run, Opening14개·13gaps/14resets. 두 sequence공식train |
| GT·mask·unknown | 모든 선택 target 유효, visibility/confidence null/known0, timestamp/time_seconds미생성 |
| 좌표 대조 | World↔camera/device5,568쌍; 최대 성분 잔차world→camera7.461e-14raw, inverse1.137e-13raw |
| Projection | 5,568쌍 평균1.90656e-5px, 최대7.46509e-5px |
| 원본 불변 | 32개 원본의 실행 전후 SHA-256 동일 |
| 준비 비용 | 로컬 읽기7,069,358bytes, core 약0.437초, CLI 전체 약0.641초. 네트워크/Colab 전송 성능 아님 |
| 학습 | 미실행, `candidate_blocked`/train_ready=false |

Raw87의 음수2D192개는 이번 선택 밖25개에 있으므로 **선택62의 음수2D는0개**다. 이 결과를 clipping/visibility filtering으로 만든 것이 아니다. 음수2D·비양수 깊이를 가진 별도 고정 fixture에서 좌표·3D mask·범위 밖 정규화가 유지되는 검사를 추가했다. 실제 관측과 fixture 검사를 구분한다.

단위/shape·nontrivial rotation inverse·손목 누락 local/global mask 분리·invalid2D·gap/미래 변경 격리·raw/manifest/source pin 변조·공식 split/motion 중복·side/scale/baseline 오류·candidate hash/ready 승격 거부·읽기/파일 수 상한·output 원본 겹침/덮기 거부·실패 report·기존3개 학습 guard 거부·SOURCE_REVISION.txt를 검사했다. 초기 테스트 실패2개는 global manifest 변경까지 동일하다고 비교한 과한 조건과 Windows split 복원의 줄바꿈 차이였고, 모델 입력 불변 비교·원본 byte 복원으로 바로잡았다.

Root의 별도 실제 검토는 `outputs/real_data_20261016/root/candidate_review.json`이다. Raw32 hash/GTmm/868값 차이0, hand local+GTwrist 복원 약1.14e-13mm, 별도 Torch reflected-world→camera 약1.08e-12mm, M2 encode_camera32건과 candidate의 M1 guard 거부를 확인했다. 이는 실제 trainer 실행이나 모델 예측 성능 검증이 아니다.

실제 RGB·독립 validation·원본 motion 검증 및 적법한 training release가 남아 있다. Root의 별도 공식 validation probe는 기존8MiB/60초 상한에서2,498,141bytes·60.047초 후 `http_body_failed`, 선택 완전 파일0·재사용false였으며 변환기의 로컬 검사와 다른 작업이다. 사용자의 현재 Colab runtime은 없고 새 자료 확보 없이 실제 regression/body/hand 학습으로 넘어가지 않았다. 예산을 늘리거나 실패한 검사를 우회하지 않는다.
