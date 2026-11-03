# 새 pose269 감사와 g25 RGB269쌍 추출 준비

## 실제 추출 결과와 root 독립 검토 (2026-11-03)

사용자가 중단 후 재실행한 `rgb269_pairs_001`은 정상 완료했다. 최초 실행의 중단 원인과 실제 소비량은 알 수 없으며 아래 수치는 성공한 재실행 한 건의 값이다.

- 신규269쌍 / 좌우538PNG / 17묶음, 기존102와 중복 없이 총371쌍이다.
- 응답 body 683,808,609bytes, 2,131HTTP, 580.226초, 관측 peak RSS 324,087,808bytes다. 논리 source read 774,635,549bytes, 반환 PNG payload 647,429,967bytes다. 고정 상한 안에서 완료했다.
- 보고서는 `all_targets_content_audited=true`, `index_unchanged=true`, `scan_performed=false`다. 인덱스 변경 없음은 실행 보고서의 전후 SHA 근거이며 결과 ZIP에는 live DB가 포함되지 않는다. 새 인덱싱 없이 g25에서 원하는 좌우 파일을 추출한 결과다.
- 결과 ZIP은 `outputs/test_rgb269_pairs_001_20261103/rgb269_pairs_001_result.zip`, 688,954,705bytes/SHA256 `41f4e3ad306c9788c6d91d7db75422ae804f3d2d334cb0b0d8a51b834297d4db`다.

Root는 ZIP SHA·전체 member CRC, 고정 config SHA, 보존 입력 ZIP과 회수 입력285파일의 바이트 동일성, 정확한269 pose key, 고정 g25 선택538개 row와 모든 실제 PNG의 SHA/크기/1024×1024 RGB 디코딩을 독립 확인했다. 기존102 config pin과 신규 key의 disjoint union371도 확인했다. 원격 요청0·학습0이며 검토9.766초, 근거는 `outputs/test_rgb269_pairs_001_20261103/root_review/review.json` 및 상위 `review_result.py`다.

대표 GT2D overlay4장은 몸/사지의 대체적인 위치를 제한적으로 시각 확인했다. 손의 가림·영상 경계/화면 밖 관절이 있으며 정밀한 관절별 영상 정합, 좌우 timestamp, 전수 정합이나 사람의 승인을 입증하지 않는다. `validation_pairs=0`, `train_ready=false`와 기존 차단 사유를 유지한다. 전체 archive 무결성이나 모델 정확도 결과가 아니다.

다음 작업은 두 고정 결과 ZIP을 사용하는 별도371 진단 adapter 준비다. 신규 학습은 아직 실행하지 않았다. 아래 준비 시점 기록과 구분하며, 마지막 절의 설정·identity·예산 확정 조건을 따른다.

## 준비 당시 기록

2026-11-03 사용자 승인에 따라 pose269의 실제 내용·좌표를 감사하고, 기존102개에 추가할 RGB269쌍의 실행본을 준비했다. 실제 신규 RGB 원격 추출·371개 진단 학습은 아직 하지 않았다. 기존102개 입력·실험 설정·checkpoint·일반 학습 gate를 보존한다.

## 실제 pose 감사

다음 원본을 수정하지 않고 SHA·선택 payload CRC·크기·JSON object와 정확한269개 목표를 대조했다.

| 원본 | SHA256 |
| --- | --- |
| `outputs/test_pose_missing269_001_20261103/archive_job_pose_missing269_001.zip` /257개 | `61bd105189ca2c0165dbb6dbac3f7c002e4937a00888daf4f8a3229f9ed411e2` |
| `outputs/pose12_recovery_20261103/root/recovered_pose4.zip` /4개 | `7d2db3c4abe4ec8ac6495c9d19560f37e393ea0e0273b3271e64ea2956a3781a` |
| `outputs/test_pose_missing8_001_20261103/archive_job_pose_missing8_001.zip` /8개 | `fc31bd09f0c46e70a0ccabff98d71a25db9e7a78491e3ae34b48c3a8b8361e13` |

기존62관절 spec·공식 split·scaramuzza calibration·converter를 그대로 사용했다. Spec의256파일·입력32MiB 한도를 늘리지 않고 두 신규 sequence147개/122개를 각각 변환했다. 각 sequence 전체의 frame 정렬·gap/reset 의미를 유지한 서로 다른 두 결과를 합쳐269 candidate를 만들었으며 batch candidate SHA와 원본 ZIP별 provenance를 남겼다. 일반 release나 converter의 대규모 입력 지원으로 표시하지 않는다.

기존102개의 고정 RGB 결과 ZIP SHA `ec11b7eb5c4a69f76acb2e5f5045d8c7c560096a6dd33b2249e5d3a6a3e1dcae`와 config SHA `414279e6af3049dd88400e425162f898d940b4e00f714872f6225291e0f2b42c`를 확인하고, 그 input manifest에 기록된 기존 candidate·extraction manifest pin을 대조했다. 기존102개의 raw도 다시 감사해 이전 candidate의 records·contract·blockers와 같음을 확인했다. 추가 pin 대조 근거는 `outputs/rgb269_prepare_20261103/developer/old102_pin_review.json`이다. 전체371개는 train/validation0이며 g25 좌우 RGB371개와 sequence/frame key가 정확하게 일치한다. 3D 유효 관절은23002개(371×62), 2D 유효 joint-view는46004개이며 그중 화면 밖1074개를 정답 누락·visibility=false로 바꾸지 않는다.

| 전체371 raw annotation 내부 일관성 | 최대 오차 |
| --- | --- |
| World→camera | 1.19016e-13 raw unit |
| Inverse camera→world | 1.70530e-13 raw unit |
| Camera→device | 5.68434e-14 raw unit |
| 어안 재투영,64554 joint-view | 7.62003e-5px, 허용0.001px |

거리 변환은 기존 cm→mm×10과 `adopted_metric_anchor_inference` 가정을 유지한다. 위 검사는 annotation 사이의 수치 정합이다. 독립적인 실제 단위 측정·RGB 시각 정합·stereo 시간·원본 motion 독립성 검증이 아니며 `train_ready=false`와 차단 사유를 유지한다.

개발 감사는42.219초/산출물66070437bytes로 지정된600초·256MiB 이내였다. 근거는 `outputs/rgb269_prepare_20261103/developer/pose_audit/summary.json`, `groups/*/audit.json`, `old102_audit.json`이다. Root도 [audit_pose371.py](../scripts/audit_pose371.py)를 별도 실행해35.875초/66070406bytes로 동일한 수치·371 key·payload 합계를 확인했다. Root 결과는 `outputs/rgb269_prepare_20261103/root_pose_audit/summary.json`이다. 코드 revision·provenance 때문에 산출물 byte/hash가 다른 부분과 raw 수치/관절 계약의 일치를 구분한다. 원격 요청·학습은0이다.

## RGB 실제 목록과 비용 계획

Canonical g25 snapshot/receipt/DB/source identity는 [g25 기록](G25_STORAGE_AND_POSE269.md)을 따른다. DB SHA `6272b03c276d8edbdbbb6377ae3f19fb28095d4beaf9f7d12f9984b031019fbc`의 원본을 readonly로 읽었고 추출에서는 snapshot을 새 local DB로 복원한다. Scan·재인덱싱·원본 DB 수정은 하지 않는다.

선택은 기존 pose102와 겹치지 않는 정확269 key, 좌우538PNG다. Payload 합647429967bytes(617.44MiB), 최대 파일1302771bytes이며 decoded tar offset248017920~898913980에 있다. 물리 archive 순서로32파일×16묶음+26파일의17묶음을 사용한다. 문자열 frame 순서나 RGB/pose archive 순서를 같다고 가정하지 않는다. 기존102 RGB는 다시 다운로드하지 않는다.

G25 index는855 point/27999297bytes이고 dictionary 합27983872bytes다. Point 앞뒤와1MiB Range 정렬로 계산한 합685768704bytes(654MiB)는 계획 추정이다. Runtime의 실제 read trace·FUSE 전송량·엄밀한 upper bound·완료 보장이 아니다. Restart 재해제·engine read-ahead·chunk 간 cache 재사용 여부와 네트워크 지연이 비용에 영향을 준다. 기존102의 단일 Colab 실측271초를 비례하면 약715초이나 처리량 보장이 아니며1800초를 제한한다.

## 새 명시 profile과 기존 계약 보존

[pairs.py](../rewind_discovery/pairs.py)는 기존 `PairLimits`와 `indexed_stereo_pairs_v1`의 최대102쌍·예산을 보존한다. 새 `indexed_stereo_pairs269_v1`만 정확269 key·train·고정 g25 source/snapshot/receipt/generation25·job `rgb269_pairs_001`·아래 고정 예산을 함께 검사한다. 임의 pair 집합이나 일반 schema에 확대된 예산을 붙이면 인증 전 거부한다. Runtime에서 config의 모든 PNG row가 복원된 index row와 같은지도 확인한다.

| 범위 | 고정 상한 |
| --- | --- |
| 최초 metadata 포함 body / HTTP | 896MiB /4096 |
| 실행,metadata·restore·추출·내용 감사 포함 | 1800초 |
| 논리 compressed source read | 2GiB |
| 반환 decoded bytes | 768MiB |
| 관측 RSS | 1GiB |
| Workspace / 결과 ZIP | 896MiB /768MiB |
| Drive RAM cache / Range / extract 묶음 | 4MiB /1MiB /32파일 |
| 실행 전 가용 RAM / disk | 2GiB /4GiB |

`returned_decoded_bytes`는 `_read(gz,...)`가 반환한 선택 PNG payload 크기다. Seek 내부에서 버린 해제 byte까지 센 전체 decoder 작업량이 아니다. 내부 seek 비용은 source read·body·시간·RSS 제한과 함께 통제하며 총 해제량을 측정했다고 주장하지 않는다. 각 chunk의 read/반환 decode와 shared runtime body/HTTP를 누적하고 다음 chunk에서 초기화하지 않는다. Hidden auth/media retry0를 유지한다.

Workspace에는 약647MB payload+55.4MB DB·보고서·대표 overlay를 넣으며, 회수 ZIP은 payload+raw269 약23MB+candidate 약9MB+metadata/보고서/overlay를 포함한다. Snapshot/DB를 결과에 중복 포함하지 않는다. Work896MiB+회수768MiB 약1.625GiB와 source/input 압축·해제 및 의존성 공간은 별도이며 disk4GiB gate를 사용한다. Upload/source/input 준비·index 의존성 설치·인증은1800초 밖, 회수 ZIP 생성120초도 별도다. 모든 한도는 협력 검사이며 OS hard cap·장시간 blocked call 즉시 중단 보장은 아니다.

## 재현 가능한 로컬 준비와 인계

다음 명령은 실제 검증한 스크립트 진입점이다. 출력에는 기존에 없는 경로를 지정하고 보존 산출물을 덮어쓰지 않는다. 고정 원본과 기존102 raw/metadata가 프로젝트 outputs에 있어야 한다.

```text
python scripts/audit_pose371.py --output outputs/rgb269_review_new/pose_audit
python scripts/prepare_rgb269_inputs.py --audit-root outputs/rgb269_review_new/pose_audit --output outputs/rgb269_review_new/prepared
```

첫 단계는256MiB·600초 안의 실제 pose 감사와 인덱스 비용 계산이며, 두 번째 단계는256MiB·600초 안에서 별도 입력을 구성한다. 전체 archive를 다운로드하지 않는다. 입력에는 고정 g25 snapshot, raw269·합친 candidate·개별 sequence audit/manifest/motion map·split/calibration·ZIP별 provenance가 들어간다. 기존102 RGB는 포함하지 않는다. 입력 파일은 다시 SHA로 봉인하고 preflight를 통과시킨다.

현재 개발 입력은 `outputs/rgb269_prepare_20261103/developer/prepared/rgb269_inputs.zip`,35798440bytes/SHA `e4a30c47f240f333cff0e69f97d6db5e2a5903154eed806dcd3f5b02a2ea624d`다. 해제89701853bytes/287파일, config canonical SHA `c24400c1ffe38c282a45095f22642f9c327ab39e2ae57dc6538737ff688b04fc`다. 초기 metadata map 구성본은 별도 ZIP으로 보존하고 최종 입력에는 신규2sequence motion map을 넣었다. 파일 hash와 preparation report를 기준으로 구분한다.

[rgb269_pairs.ipynb](../notebooks/rgb269_pairs.ipynb)는 기존102 notebook의 검토된 인증 factory와 source/input 선검증·소비 workspace 차단·독립 회수를 재사용한다. 새 source/input/work/result 경로를 사용하며 작은 source ZIP64MiB, 입력 ZIP/해제128MiB를 각각 제한한다. Source ZIP SHA와40자리 revision·ready notebook SHA는 root가 검토·로컬 커밋 후 고정한다. 개발 template의 source/revision은 빈값이다. Root의 source 패키징에는 변경 모듈·필수 package·requirements를 넣고 `SOURCE_REVISION.txt`를 검증한다. Ready는 입력 ZIP SHA/config SHA/job/profile/예산과 factory AST를 재검사한 뒤 탐색 세션에 인계한다. 개발 세션은 source/ready를 최종 고정하거나 실제 Colab을 실행하지 않는다.

탐색 세션은 새 CPU runtime에서 고정 source/input만 준비하고 한 번 실행한다. 성공은538PNG의 index SHA/크기·native1024 RGB decode·269 pose key/62관절 annotation 연결이 전부 감사된 상태다. 대표 GT2D overlay는 신규2sequence×2view 검토 자료이며 사람 승인이나 전수 정합 검증으로 대체하지 않는다. 실패·부분 성공·identity 변경·CRC/hash/PNG 오류·예산 중단을 구분하고 자동 반복하지 않는다. 마지막 회수 셀만 실행해 현재 partial/완료 chunk·pose·metadata·report·overlay가 든 결과 ZIP/SHA를 runtime 밖에 보관한다. 원본 snapshot은 기존 보관본을 유지한다.

## 개발 검증과 독립 검토

신규 notebook8개/1.015초와 신규 HTTP 누적 cap 검사1개/20.436초는 failure/error/skip0으로 통과했다. 실제 notebook 셀의65MiB·269파일 입력 준비와 exact staging 재사용, 준비완료/준비실패 회수, stale module 차단, 검토된 factory AST와 실제 requests/google-auth 타입을 검사했다. 최초 준비완료 회수 fixture에서 `input_manifest.json` sentinel을 빠뜨려 정상적인 준비실패 branch로 간 기록은 보존했고 fixture를 고친 최종 검사와 구분한다.

전체17chunk 합성 완료·538PNG 감사/269쌍 집계·원본 index 불변·부분 회수·고정 key/identity/예산 격리의 신규3개도884.703초에 통과했다. 위 notebook8개와 별도 HTTP 누적 cap1개를 합쳐 신규12개 failure/error/skip0이다. 실행 시간은 병행 로컬 검사 환경의 각 검사 호출 값이며 실제 Colab 전송·처리량 결과가 아니다. 근거는 `outputs/rgb269_prepare_20261103/developer/`의 `new_runner_tests.txt`, `cumulative_tests.txt`, `notebook_tests_final.txt`다.

Root는 실제 입력 ZIP pin/preflight와 snapshot을 별도 DB로 복원해538개 row가 정확한 선택 목록과 같음을 확인했다. 근거는 `outputs/rgb269_prepare_20261103/root/review.json`이다. Root의 신규 notebook8개도 통과했다. Root의 기존102·신규269 runner·기존 notebook 독립 회귀19개도987.580초에 failure/error/skip0으로 통과했다. 신규 notebook 독립8개는0.892초에 통과했다. 추가한 HTTP 누적 cap1개는 개발의 별도 통과 결과를 검토했으며 root19개에 중복 산입하지 않는다.

## 371 진단 연결의 다음 경계

이번에는 추출 준비에서 종료한다. 실제 신규269쌍 RGB와 내용 감사가 완료되고 필요한 영상 정합 검토를 받은 뒤에 별도371 진단 adapter를 구현·검토한다. 기존 [102 진단](REAL_RGB102_DIAGNOSTIC.md)의 고정 설정·checkpoint identity·102 전용 reader는 수정하지 않았다.

후속 adapter는 기존102와 신규269의 두 고정 결과 ZIP을 각각 검증하고 disjoint union371 key를 보존해야 한다. 256파일 converter를 늘리지 않고 기존102와 신규2sequence의 감사를 재사용/재검증하며 native1024→64·GT2D supervision·예측2D cache→T1 3D 입력을 유지한다. GT2D fallback은 허용하지 않는다. Tensor 목표는 `(371,1,868)`, GT3D는 `(371,1,62,3)`이며 시간·sequence reset을 causal temporal 지원으로 해석하지 않는다.

승인된 방향은 LR1e-4·작은 B·최대1pass·diagnostic_only/train_ready=false/validation0다. B2이면 마지막 batch1을 포함해 각186update이며 현재102 runner의51step·짝수 cursor·102 key 상수와 다른 identity/role·예산이 필요하다. 초기화/기존 weights 사용 여부·최종 실행 예산은 후속 검토에서 확정하고 지금 학습하거나372update 완료로 보고하지 않는다. 같은 train 자료의 fit 결과를 validation·일반화·과적합 방지 성공으로 표시하지 않는다.
