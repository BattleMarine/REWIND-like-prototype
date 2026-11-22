# v2 256 A·B 병렬 로더 교체본

2026-11-23 기준. R2는 이미 시작한 256 A(seed 42)와 B(seed 71)의 학습을 각자 최신 Drive 체크포인트에서 이어가는 교체본이다. 로더 worker 4개와 최대 8프레임 lookahead를 사용하며 학습은 기존 순서의 프레임을 소비한다. [기존 256 입력·예산·평가 계약](V2_GPU_TRAINING_256.md)

기존 frame-index가 없는 샤드는 worker가 전체 ZIP 읽기·SHA 확인·인덱스 계산을 미리 준비할 수 있다. 인덱스가 있는 샤드는 frame·pose 범위를 읽고 SHA를 확인한 뒤 로컬 임시 파일까지 준비한다. 순차 소비 시 본 worker가 검증된 로컬 사본과 새 인덱스를 게시한다. 기존 디스크 상한과 읽기 예약 안에서 준비하며, 병렬 준비를 예약할 수 없으면 필요한 입력을 기존 경로에서 검증해 읽는다.

## 슬롯별 파일과 Drive 경로

| 노트북 | 기본·고정 슬롯 | 기존 Drive 결과 경로 |
| --- | --- | --- |
| `v2_gpu_training_256_parallel_r2-A.ipynb` | A / seed 42 | `reports/v2_gpu_training_256_v1/A-seed-42/` |
| `v2_gpu_training_256_parallel_r2-B.ipynb` | B / seed 71 | `reports/v2_gpu_training_256_v1/B-seed-71/` |

두 노트북은 새 `v2_gpu_training_256_parallel_r2.zip`을 같은 파일명으로 사용한다. 발급 `receipt.json`의 `source.sha256`가 업로드할 새 ZIP의 SHA-256이다. `runtime_pins.source_sha256`의 `2ace5591e9911b12b18dffa966e8f2da88cbefc40d54b48b4295af1130796d7f`는 기존 실행 식별값이다. 노트북은 새 ZIP SHA와 병렬 overlay 두 파일의 SHA·크기를 확인한 뒤 기존 worker pins를 그대로 전달한다. 원본 256 ZIP의 모든 멤버는 기존 바이트를 보존하고 두 overlay가 새로 포함된다. 원본 `SOURCE_REVISION.txt`는 기저 동결본의 식별값이며, R2 배송 소스와 overlay는 발급 receipt의 별도 핀으로 추적한다.

## 기존 체크포인트에서 재개하는 순서

1. 오케스트레이션이 해당 슬롯의 기존 worker가 정상 종료와 마지막 Drive 백업을 마쳤고 더 이상 Drive에 쓸 수 없음을 확인한다. 이 시점 이후 같은 슬롯의 `checkpoints/step-*.json`에서 최신 영수증과 대응 ZIP을 확인한다. A와 B를 서로 바꿔 재개하지 않는다.
2. 동일한 슬롯의 고정 R2 노트북을 열고 새 ZIP을 `/content/`에 올린다. 기존 `DRIVE_PROJECT` 값을 확인한다. 최신 step JSON **내용의 `content_sha256`**를 `EXPECTED_RESUME_TIP`에 입력한다. 예전 실행의 임시 경로나 B의 과거 초기 재개값을 그대로 붙여 넣지 않는다.
3. 일반 재개에서는 `RECOVER_WRITER_OWNER`를 비워 둔다. VM이 유실돼 기존 writer lease가 남으면 기존 worker/VM의 종료를 확인한 뒤 오류·잠금 기록의 정확한 owner 값을 한 번만 사용한다.
4. 승인된 슬롯에서만 기본 OFF인 `RUN_V2_TRAINING`을 켠다. 기존 슬롯의 최신 영수증 SHA가 입력값과 같아야 재개한다. ZIP이나 최신 영수증이 없거나 그사이 영수증이 바뀌면 새 학습을 시작하기 전에 오류를 보고한다. 소프트웨어 binding이 일치하는 CUDA 환경을 사용한다.
5. 로그의 `Verified source archive and overlay`와 `Restored verified Drive checkpoint`를 확인한다. 이후 새 진단의 슬롯·seed·이전/현재 cursor와 최신 step 영수증을 회수한다. 이미 완료된 최신 영수증이면 worker가 완료 상태를 돌려준다.

기존 run ID는 `V2-FULL-256-3X3-v1`이다. 256 캐시 경계·manifest·실행 설정·A 최초 갱신 영수증·슬롯별 Drive writer·체크포인트 체인을 기존 worker 경로로 확인한다. 기존 모델·optimizer·scheduler·scaler·RNG·batch 위치의 복원 계약과 호출 상한 28,800초를 사용한다. 현재 최신 영수증은 실제 전환 직전에 확인할 항목이다. 기존 작업의 중단, Drive 재개 실행과 GPU 성능 검증은 패키징 단계에서 수행하지 않았다.

이전 VM을 계속 사용할 수 있으면 원본 A/B와 같은 `/content/rewind_v2_gpu_training_256_v1/<SLOT>/`를 사용한다. 같은 경로의 `cache256/` 로컬 캐시와 `features/`가 남아 있으면 기존 바이트 검사 후 재사용할 수 있다. 새 VM에서는 필요한 로컬 파일을 다시 준비하고, Drive에서 같은 슬롯의 frame-index·특징·체크포인트를 검증해 사용한다.

## 로컬 발급

```text
python -m scripts.build_v2_gpu_training_256_parallel --original-source <원본-256-v1.zip> --original-receipt <256-v1-receipt.json> --output <새-R2-폴더>
```

발급기는 원본 256 receipt·source SHA를 확인하고, 원본 멤버의 바이트와 두 새 overlay의 크기·SHA를 대조한다. 결과로 슬롯별 노트북 2개, 공통 source ZIP, 한국어 가이드와 receipt를 만든다.
