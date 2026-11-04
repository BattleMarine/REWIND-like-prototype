# 검토된 실제 데이터의 bounded RGB2D 학습 연결

> **2026-11-05 현재 적용 범위:** 아래 Serena2frame 차단은 해당 고정 bundle의 결과다. 후속371 train 쌍과 Joyful Jump57 로컬 자료가 있어도 독립 validation·motion/split 검토와 명시 release는 별도다. 일반 사용자 추출 폴더를 이32frame bounded release에 바로 넣는 기능이나 전체 학습 승인은 없다. 최신 운영 기준은 [데이터 확보 전략](DATA_ACQUISITION_STRATEGY.md)을 따른다.

현재 구현은 완료 bundle→명시 release 검증→UI 독립 RGB2D train/validation/정확 재개/checkpoint/export를 연결한다. **현재 Serena 자료에 대한 실제 학습 승인은 발급하지 않았다.** 기존 진단 candidate·reader의 `dataset_sample/diagnostic_only/train_ready=false`와 원래 blockers는 보존한다. 별도 release는 한 bundle32frame 이내의 독립 T1/2D frontend train/validation만 허용한다. M1/M2/M3·temporal·전체 archive coverage 또는 일반 실제 학습 gate를 해제하지 않는다.

## 현재 실제 자료의 차단 이유

기존 Serena frame1·10은 공식 train의 한 sequence이며 원본 motion이 `declared_unverified`다. validation이 없고 원본 motion 독립성·RGB/pose 내용 정합을 독립 검토한 release 근거가 없다. cm→mm는 `adopted_metric_anchor_inference`라는 프로젝트 채택 가정이다. 두 frame의 gap은 T1/2D 범위에서 다룰 수 있지만 연속 clip으로 해석하지 않는다.

원 candidate의6개 blocker는 변경하지 않는다. `audit-candidate`는 기존 비-bundle 입력도 파일 SHA·키/split·원래 blockers/motion 선언/단위 상태와 최소 후속 자료를 새 보고서에 남기며 허용을 발급하지 않는다. 두 frame을 train/validation으로 나누거나 합성 fixture로 재라벨하지 않는다. 현재 legacy Serena 형식은 자동 bundle migration 대상도 아니다.

필요한 최소 후속 자료는 다음과 같다.

1. train과 원본 motion이 독립인 nonempty validation stereo+pose 및 검증된 완료 bundle.
2. 전체 sequence의 원본 motion ID와 원래 candidate 선언의 관계, split 간 중복 배제 근거.
3. 같은 bundle/key를 대상으로 좌우/RGB-pose 내용 정합·관절 정의·calibration/좌표/단위 채택과 한계를 검토한 근거 파일.
4. 고정 코드/config/release/근거를 검토한 운영자의 외부 policy 및 별도로 전달받은 예상 policy SHA.

이번 작업에서 새 실제 자료 확보·실제 release/policy 발급·새 실제 학습·GPU·네트워크·설치는 수행하지 않는다.

## 신뢰 모델과 승인 권한

로컬 단일 운영자의 `external_operator_pinned_allowlist` 모델이다. 사용자/오케스트레이션이 bundle 밖, release 파일 디렉터리 밖의 policy를 별도로 검토하고 예상 **파일 SHA256**을 명시한다. runtime은 policy의 SHA를 자동 채택하거나 policy를 생성하지 않는다. `--force`, `verified=true` 또는 release 자체의 선언만으로 허용하지 않는다.

policy에는 `schema_version=local_rgb2d_review_policy_v1`, 위 trust_model, `scope=native_rgb2d_t1_train_validation_v1`과 `allowed_releases`가 있다. 각 허용 항목은 release 파일 SHA/content SHA·review 파일 SHA·전체 code map의 canonical SHA·검토자 목록을 정확히 고정한다. 검토자가 수정한 policy의 새로운 예상 SHA는 운영자가 별도로 전달해야 한다.

이는 서명·원격 권한 서버·악의적인 로컬 파일/코드 소유자에 대한 보안 경계가 아니다. SHA는 byte 동일성을 증명하며 evidence 내용의 진실성을 증명하지 않는다. 운영자가 임의 policy를 믿도록 스스로 지정하면 그 선택까지 막는 모델도 아니다. 현재 범위에 서명 키/암호화 의존성을 추가하지 않았다. **구현 신뢰 모델의 승인은 실제 자료 승인과 다르다.**

## 자동 사실과 사람의 검토

자동 검사는 완료 content/file pin, 정확한 파일 byte/SHA, source scope, 계약·관절 순서·raw→mm/mask/GT2D 특징, 원본 producer/candidate/RGB 대응, 양 split 존재, sequence 고정 및 motion ID의 split 교차, code/config/evidence hash와 예산이다. 이는 motion ID나 내용 정합 근거가 사실임을 자동 판정하지 않는다.

인간 review JSON은 `rgb2d_human_review_v1`과 scope, completion SHA, 정렬된 `[sequence,frame,split]` 전체 목록의 canonical SHA, `evidence`, `reviewed_motion_map`, `geometry_statement`를 제공한다.

| evidence category | 검토 대상 |
| --- | --- |
| `source_provenance` | 원본 데이터 identity/출처·복사/추출 경로 |
| `motion_identity` | 원본 motion 식별 및 candidate 선언과의 차이 |
| `split_independence` | sequence·원본 motion의 train/validation 독립성 |
| `rgb_pose_alignment` | 좌우/RGB-pose 내용 정합·대상 key coverage |
| `joint_definition` | 62관절 정의/순서·annotation mask 의미 |
| `units_geometry` | 검증된 사실과 채택한 단위/좌표/calibration 가정 |

각 category는 `artifact`(review 디렉터리 안의 상대 파일), `artifact_sha256`, `reviewer`, `date`(`YYYY-MM-DD`), `conclusion`, `limitations` 목록, 동일 `key_coverage_sha256`을 요구한다. 파일당1MiB 이내다. 일반 conclusion은 `reviewed_for_rgb2d_t1`, units_geometry는 `adopted_assumptions_for_rgb2d_t1` 또는 `verified_facts_for_rgb2d_t1`이다. 후자의 문자열만으로 자동 사실 검증이 생기지 않는다.

`reviewed_motion_map[sequence]`는 `original_motion_id`, 원래 `candidate_motion_id`, `evidence_note`를 보존한다. 모든 sequence를 덮고 동일 원본 motion의 split 교차를 거부한다. 원래 candidate의 `declared_unverified`를 덮어쓰지 않는다. `geometry_statement`는 각각 비어 있지 않은 `verified_facts`와 `adopted_assumptions` 목록을 분리한다. 이 구분과 전체 review/provenance는 checkpoint·추론 weights에도 남긴다.

원 candidate의 진단 전용 schema, 원본 motion 미검증, 독립 validation/RGB 정합 미검증 등은 외부 검토 release가 **2D/T1 범위에서 무엇을 보완했는지** 별도로 기록한다. train/validation 누락은 실제 nonempty reader를 만들 수 없으면 거부되며 문자열로 면제하지 않는다. gap만으로 T1/2D를 막지는 않지만 temporal 허용은 없다. 새로운 미처리 blocker는 거부한다.

## 제안·실행 CLI

```powershell
python -m rewind_rgb_release audit-candidate --candidate legacy_prepared/pose/candidate.json --output new_blocked_audit
python -m rewind_rgb_release propose --bundle bundle --completion-sha256 BUNDLE_CONTENT_SHA --config config.json --review review/review.json --output new_proposal
```

`propose`의 `release.json`은 **제안 파일**이며 승인 권한이 없다. 자동 검사와 제공된 review를 고정하지만 별도 검토 policy가 이 정확한 파일을 허용해야 한다. CLI는 policy를 발급하지 않는다. 코드·config·bundle·review/근거가 바뀌면 제안과 외부 검토도 다시 고정해야 한다.

```powershell
python -m rewind_rgb_release train --bundle bundle --completion-sha256 BUNDLE_CONTENT_SHA --config config.json --release proposal/release.json --review review/review.json --policy external_trust/policy.json --policy-sha256 OPERATOR_PROVIDED_POLICY_FILE_SHA --steps 4 --output new_train
python -m rewind_rgb_release train --bundle bundle --completion-sha256 BUNDLE_CONTENT_SHA --config config.json --release proposal/release.json --review review/review.json --policy external_trust/policy.json --policy-sha256 OPERATOR_PROVIDED_POLICY_FILE_SHA --steps 8 --resume previous_train/last.pt --output new_resume
python -m rewind_rgb_release validate --bundle bundle --completion-sha256 BUNDLE_CONTENT_SHA --config config.json --release proposal/release.json --review review/review.json --policy external_trust/policy.json --policy-sha256 OPERATOR_PROVIDED_POLICY_FILE_SHA --checkpoint previous_train/last.pt --output new_validation
python -m rewind_rgb_release export --checkpoint previous_train/last.pt --output new_weights.pt
```

위 값은 형식 설명용 placeholder이며 현재 실제 승인본의 실행 지시가 아니다. 매 실행은 새 별도 output만 허용한다. `--steps`는 재개 전을 포함한 누적 목표다. train은 validation 평가를 포함하고 test 자동 평가/튜닝은 지원하지 않는다. validation CLI는 optimizer update 없이 같은 binding/checkpoint를 확인한다. 기존 `rewind_m1.rgb_training_cli`의 synthetic-only gate는 유지한다.

config는 기존 RGBTrainer 규격(seed/batch/Adam/StepLR/AMP/명시 training_image_size/native tensor·heatmap 상한)을 사용하며 release에 전체 canonical SHA를 고정한다. 첫 범위는 bundle32frame·B≤2, resize4~256, 요청누적64update 이하·이번128attempt 이하·협조적600초 이하이고 reader의4GiB sampled process RSS/native 반환64MiB 상한도 유지한다. 실제 자원·성능 보장은 별도다.

검증 가능한 config 예시는 [rgb2d_release_bounded_v1.json](../configs/rgb2d_release_bounded_v1.json)이다. B2/resize256²/FP32의 명시 설정이며 이것만으로 실제 실행 승인을 주지 않는다. 로컬 개발 검증은 같은 규격의 resize8²/B1 설정을 사용한다. AMP·seed·resize 등 하나라도 변경하면 config binding과 외부 검토를 새로 고정한다.

## 학습·재개·export 경계

새 `ReleasedRGBTrainer`는 기존 RGBTrainer의 resize·FP32 heatmap loss·성공 update/cursor·AMP skip과 validation sum/count를 재사용한다. ShardTrainer의 optimizer/scheduler/scaler/Python·Torch RNG/cursor 복원과 검증을 그대로 쓴다. 공유 코어에는 기본값이 synthetic 그대로인 데이터/모델 생성 hook과 `TRAINING_SCOPE`, CLI 보고 scope만 추가했다.

checkpoint는 `bounded_released_rgb2d_checkpoint_v1`, scope는 `reviewed_real_rgb2d_t1_only` 또는 **명시 개발 전용** `development_fixture_rgb2d_t1_only`다. 실제 state를 잠깐 synthetic scope로 바꾸어 재개시키지 않는다. mode/release/policy/bundle/config/code/env 차이는 재개를 거부한다. code map에는 새 release/reader뿐 아니라 재사용 모델/loss/resize/checkpoint/변환 모듈도 포함한다. 실제 모드는 update·재개·평가 전 policy/release/review/근거의 변경도 다시 확인한다.

공통 durable runner는 매 checkpoint의 SHA·step·sample/cursor와 bounded log를 보고한다. 실패하면 마지막 durable을 보존하며 실패한 update가 cursor를 전진시키지 않는다. AMP skip은 성공 update로 세지 않는다. decoder/검토 파일 읽기 등의 준비 단계도 외부 시간 budget 앞뒤 검사에 포함하지만 watchdog은 아니다.

weights는 `bounded_released_rgb2d_weights_v1`이다. `ReleasedInference(path, expected_file_sha256=...)`는 training/review 모듈의 승인 절차를 실행하거나 학습 권한을 발급하지 않는다. policy 파일 없이 오프라인 CPU 추론을 허용하되 metadata에 원래 scope·근거·가정·checkpoint/release binding을 보존한다. `predict(native_images)`는 한 sample CPU FP32 `[1,2,3,1024,1024]`를 받고 native pixel 예측2D·confidence를 반환한다.

기존 합성 weights/cache 형식은 계속 synthetic-only다. 새 weights를 기존 cache/T1 join에 자동 투입하거나 3D 학습을 열지 않는다. resize/native 복원 수학은 같으나 후속 cache adapter가 이 release provenance와 검증 범위를 명시적으로 이해해야 한다.

## 개발 검증과 실제 시험 인계

개발 CLI는 `development-fixture-train`/`development-fixture-validate`로 분리하며 CPU와 `source_scope=synthetic_fixture`만 허용한다. candidate의 `dataset_sample` 진단 계약은 그대로 유지한다. 실제 `train` admission은 이 source를 거부한다. 실제 조립 분기의 단위 검사는 **명시 mock facts**를 쓰며 archive 출처를 변경하거나 실제 승인을 발급하지 않는다.

현재 실제 수동 시험은 **인계 준비 미완료/학습 차단**이다. 테스트 세션은 지금 학습 명령을 실행하지 않는다. 먼저 위 최소 자료와 검토 policy를 확보한 뒤 root가 고정 revision·소스 ZIP/notebook·bundle/release/policy/evidence/config의 정확한 SHA, 대상GPU/precision·step/시간/전송/디스크 budget, 중단 조건과 결과 보관 위치를 확정해 별도로 인계한다. 기존20b7aa9 GPU ZIP/notebook/결과는 보존한다. 합성 A100 FP32/AMP 시험 통과를 실제 데이터 학습 승인으로 해석하지 않는다.

개발34개(신규7·기존RGB13·M1 shard14)가43.160초 failure/error/skip0으로 통과했고 root 신규7개27.710초·기존27개21.013초도 통과했다. source 위장 없이 development_fixture CPU의 full4 대 prefix2+resume2 모든checkpoint state 일치와 validation·export·offline import/CPU 모델·durable 실패 보존을 확인했다. 실제 admission의 policy/review/code 조립은 명시 mock facts 단위 시험이며 실제 자료 승인이 아니다. outputs/full_training_readiness_20261102/developer/rgb_release/에 validation_tests.txt, integration_summary.json, serena_blocked/report.json과 fixture checkpoint/weights를 보존했다. 실제 candidate SHA/내용은 변경하지 않았고 실제 update/release/policy/GPU/네트워크/설치0이다.
