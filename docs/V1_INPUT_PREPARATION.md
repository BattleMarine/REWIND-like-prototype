# v1 연속 구간 선택과 입력 묶음

2026-11-10 첫 구현 묶음. CPU에서 동작하는 `rewind_prepare.temporal_cli`의 `plan`, `pack`, `restore`를 제공한다. 실제 Drive 준비 속도·GPU 학습은 아직 검증하지 않았다. 논문 구조 대조는 [별도 문서](V1_PAPER_ALIGNMENT.md)를 따른다.

## 선택과 지원 범위

보존된 공식 split·spec·파일 목록과 v0 manifest를 대조해 train의 50프레임 구간 20개를 선택한다. 모든 context를 포함한 고유 train 프레임 수는 1,000이다. 부족하면 부분 계획을 보고하며 validation/test에서 보충하지 않는다. 공식 validation 2,236개 키와 기존 hash를 유지한다.

Root는 실제 보존 목록으로 선택을 독립 재실행해 개발자 결과와 같음을 확인했다. 281개 비중첩 후보 구간 중 20개 sequence에서 50프레임씩 선택됐고 계획 SHA는 `48e3c749c7d7710feb7fd61601b7406264837e7a67ceeb4a48787359825cafad`다. 근거는 Git 제외 `outputs/archvis_v1_20261110/root_selection_plan.json`이다. 이는 목록 기준 선택이며 새 1,000페어의 payload를 확보·검증한 결과가 아니다.

Validation에는 번호 기준 비중첩 50프레임 구간이 29개 있다. 이를 전체 validation 평가 완료로 해석하지 않는다. 실제 시간 window의 적격성·초기 history 처리와 평가 coverage는 후속 연결에서 확정한다. `window_keys`는 sequence·clip 경계와 번호 누락에서 history를 초기화하는 인덱싱 함수이며, timestamp나 카메라 tensor를 만드는 실제 temporal adapter 완성을 뜻하지 않는다.

## 입력 보존과 병렬 처리

팩에는 원본 PNG·pose JSON 바이트를 `ZIP_STORED` 방식으로 50프레임 단위 보존한다. 학습용 resize tensor 캐시와 다르며 과거 대규모 분할 gzip 인덱싱을 재개하는 기능이 아니다. 이후 실행은 소수 묶음을 로컬로 복원해 member별로 읽을 수 있다.

- 준비는 worker 1/2/4/8을 선택하고 결과를 고정 순서로 기록한다. 입력 크기·hash·PNG decode·pose 수치 검사를 적용한다.
- 읽기 예약량·디스크·시간·작업 버퍼 상한을 명시한다. 작업 버퍼 상한은 전체 프로세스 RSS의 강제 상한이 아니며 RSS는 별도 관측값이다. 병렬 stage 시간 합은 wall-clock과 다르다.
- 완료된 part와 진행 기록을 보존하며 실패한 part를 최종 캐시로 공개하지 않는다. 재시도에서도 이미 예약한 읽기량은 되돌리지 않는다.
- 캐시 manifest·팩·member hash로 복원과 조회를 검증한다. 코드·입력·상한이 달라진 재개를 같은 실행으로 취급하지 않는다.

파일 읽기가 운영체제/Drive에서 막히면 협조적 시간 검사만으로 즉시 끊을 수 없다. 실제 네트워크 지연과 worker별 처리량은 Colab에서 별도 계측해야 한다. 캐시 Drive 업로드는 기존 규칙대로 사용자가 담당한다.

## 실행과 남은 검증

`python -m rewind_prepare.temporal_cli --help`로 명령을 확인한다. Plan에는 고정 공식 목록과 이전 manifest SHA, pack에는 새 계획 SHA·원본 경로·spec/calibration·예산, restore에는 완료 캐시 SHA와 디스크/시간 상한이 필요하다. 현재는 사용자 실행용 고정 Colab 묶음을 인계하지 않았다.

준비 완료도 `train_ready=false`, `temporal_ready=false`를 유지한다. 실제 timestamp·stereo 동기, camera 변환, 영상/pose 정합, 원본 motion 독립성의 근거가 필요하다. A100 학습 gate·모델 후보·실제 M2/M3 공급은 후속 묶음이며 이 CPU 준비 기능과 구분한다.

Root가 선택/부족/경계, 순차·병렬 동일성/복원/변조, 중단·읽기예약/상한을 다루는3개 검사를 재실행해 통과했다. 관련 검증 코드의 hash map을 재개 identity와 완료 manifest에 포함하며, 의존 코드가 달라진 재개가 거부되는 것도 확인했다. 테스트의 worker1/2 결과가 같다는 사실은 실제 Drive에서 worker4/8의 속도·메모리 검증을 뜻하지 않는다.
