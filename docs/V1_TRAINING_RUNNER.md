# v1 frontend·teacher 학습 실행 연결

2026-11-10. train 고유 1,000페어를 유지하는 [v1 계획](V1_REWIND_PLAN.md)의 실행 연결이다. 새 실제 데이터 학습 결과나 논문 정확도 재현 결과가 아니다. 실제 실행 전에 시간·카메라·영상 정합과 원본 motion 검토 근거를 준비한다.

## 실행 순서

1. 프레임 단위 stereo RGB로 MBConv frontend를 사전학습한다. Gaussian cross-entropy heatmap loss와 화면 안 관절의 2D 오차를 사용한다. 화면 밖 관절을 3D 평가에서 제거하는 규칙은 아니다.
2. Validation에서 선택한 frontend best를 eval/frozen 상태로 사용해 50프레임 clip의 예측 2D 특징을 만든다. 시간축 batch 통계로 미래 정보를 섞는 end-to-end 학습은 제공하지 않는다.
3. Body teacher를 학습하고 validation MPJPE로 best를 선택한다.
4. Hand teacher는 GT 상체로 학습한다. 평가는 고정 body best의 예측 상체와 예측 손목을 사용한다. 손목 상대 손 오차와 전체 좌표 손 오차를 구분한다.

Frontend는 공식 validation 2,236페어 전체를 평가한다. Teacher는 그 안의 검증된 비중첩 50프레임 clip만 평가하며, 평가한 키·개수·제외 키를 보고서에 기록한다. 현재 목록 기준 29개 clip 후보가 있다는 사실을 2,236페어 전체의 temporal 평가 완료로 해석하지 않는다. Test는 사용하지 않는다.

Frontend batch에 화면 안 유효 GT가 전혀 없으면 모델 gradient update 없이 순회 위치와 scheduler를 전진시키고 `no_supervision_batches`를 기록한다. 따라서 보고서의 전체 update/cursor 수만으로 모든 batch에서 학습 신호가 발생했다고 판단하지 않는다.

## 캐시와 체크포인트 구분

- 원본 PNG·pose raw pack: [입력 준비 도구](V1_INPUT_PREPARATION.md)가 만든 `cache_manifest.json`과 `part-*.zip`을 사용한다. 선택 목록·spec·파일 hash를 확인하며, `--cache-local`로 Colab 로컬 디스크에 묶음 복원할 수 있다. 이미 복원한 동일한 팩은 검증 후 재사용한다.
- 현재 runner는 원본 Drive 파일을 찾거나 다시 다운로드하지 않는다. `--cache`에 Drive 경로를 직접 지정하고 로컬 복원을 생략하면 그 캐시 자체를 Drive에서 읽으므로, 반복 학습에는 로컬 복원을 사용한다.
- Frontend는 로컬 PNG를 읽어 해상도를 변환한다. 매 epoch의 PNG decode는 남아 있으며 영구 tensor 캐시는 아직 제공하지 않는다. Teacher용 예측 특징은 메모리 상한 안에서 한 번 생성한다. 프로세스 재개 시 로컬 raw pack과 고정 frontend best로 다시 생성한다.
- `best_e1_*.pt`, `best_e20_*.pt` 같은 이름은 epoch별 모델 체크포인트로 보이지만 이름만으로 내용을 확정하지 않는다. 모델 체크포인트는 입력 이미지 캐시를 대신하지 않는다. 이전 v0 모델은 구조가 다른 v1에 자동으로 불러오지 않는다.
- 새 `last.pt`는 실행 위치·현재 stage의 모델/optimizer/scheduler/scaler/RNG를 보존한다. `best-frontend-*.pt`, `best-body-*.pt`, `best-hand-*.pt`는 단계 간 의존 모델이다. 기존 v0 파일은 별도 보존한다.

## 명령과 재개

프로젝트 루트에서 `python -m rewind_v1 --help`로 `source`, `synthetic`, `actual`, `recover` 명령을 확인한다. `synthetic`은 별도 CPU fixture이며 실제 1,000페어 실행이나 성능 근거가 아니다. `actual`은 실제 A100 이름·VRAM을 먼저 확인하고 source/config/selection/cache/evidence pin을 요구한다. CPU 입력 팩 준비·복원에는 GPU가 필요하지 않다.

실행 인자는 `python -m rewind_v1 actual --help`로 확인한다. 경로와 hash는 실제 보존 자료에서 고정해 전달하며, 임의 timestamp나 승인 근거를 만들어 채우지 않는다. `--resume-sha256`는 해당 실행의 `last.pt` 파일 SHA256이며 변경된 코드·환경·입력·설정으로 같은 실행을 재개하지 않는다. `--stop-after-updates`는 이번 호출에서 추가 진행할 update 수를 제한한다.

Epoch별 결정적 순서와 batch cursor를 저장하며, 마지막 정상 체크포인트 이후 저장되지 않은 update는 재개 시 다시 계산할 수 있다. 평가·특징 재구축은 학습 RNG와 분리한다. `run.json`의 진행 위치와 `last_checkpoint`의 저장 위치를 혼동하지 않는다.

`recover --run <실행폴더> --output <폴더밖의 새 ZIP> --max-bytes <상한>`은 보고서, 마지막 체크포인트와 그 체크포인트가 참조하는 best 파일을 해시 영수증과 함께 회수한다. 입력 raw pack은 결과 ZIP에 포함하지 않는다. 회수 ZIP도 별도 저장 공간이 필요하다.

## 비용과 남은 검증

Epoch·update·경과 시간·출력 디스크·체크포인트·보존 특징 tensor 상한을 config에 명시한다. 시간 상한은 호출별이며 원래 자동 인덱싱의 무제한 시간 정책과 다르다. 개선된 epoch별 best 파일이 누적되며 출력 디스크 상한에 포함된다. 특징 tensor 상한은 프로세스 전체 RAM이나 GPU peak 메모리의 강제 상한이 아니다.

`configs/rewind_v1_pipeline_pilot_candidate.json`은 frontend/body/hand 각 1epoch, 총 540회 순회 update의 **pilot 후보**다. 입력128px·frontend batch2, teacher hidden512·graph3·temporal3·4heads·DDIM10, FP32를 사용한다. 2시간/호출·출력1GiB·캐시 복원12GiB·보존 특징128MiB·체크포인트256MiB 상한은 후보 설정이며 실제 자원 적합성을 실측하지 않았다. 학습 완료나 본 학습 예산 승인으로 해석하지 않는다.

보고서에는 CPU 준비·로컬 읽기·전처리·학습·평가·저장 시간을 구분한다. 중첩된 타이밍은 더해서 전체 wall-clock으로 해석하지 않는다. 실제 A100 처리량·VRAM·Drive 복원 비용과 모델 정확도는 고정 Colab 시험에서 별도 측정한다. 이번 연결만으로 학습 속도 개선이나 실제 데이터 학습 성공을 주장하지 않는다.

Root는 최종 고정 코드에서 pipeline 6개·모델 7개·입력 준비 3개, 총 16개 검사를 독립 실행해 통과를 확인했다. 합성 CPU의 전체 stage 학습, 중간 batch/단계 전환/평가 전 중단 후 재개, 모델·optimizer·scheduler·scaler·RNG·cursor 상태 일치, 회수 ZIP·변조 거부·실제형 62관절 합성 raw adapter와 로컬 캐시 재사용을 포함한다. CLI 도움말과 notebook 코드 문법도 확인했다. 실제 Drive·A100 실행과 실제 자료의 시간/좌표 근거 검토는 수행하지 않았다.
