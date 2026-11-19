# v2 연속 CPU 캐시와 GPU 경계 배치

세 시나리오의 고정 manifest 136,011페어를 128프레임 단위로 다룬다. 검증된 첫 샤드(인덱스 0)는 기존 `reports/cache/v2_cpu_cache_041837Z/`에서 참조한다. 이후 샤드와 완료 경계는 `reports/cache/v2_bulk_cache_041837Z/`에 저장한다. [첫 샤드 정식 복원 시험](V2_GPU_CACHE_RESTORE.md)

## CPU 캐싱과 재개

1. CPU Colab에서 `v2_bulk_cache_cpu.ipynb`를 열고 기존 `v2_cpu_cache_ed98124.zip` 하나를 이름 그대로 `/content`에 올린다. ZIP이 없으면 파일명과 경로를 표시하고 Drive 마운트 전에 중단한다.
2. `DRIVE_PROJECT`와 `TARGET_TOTAL_SHARDS`를 설정한다. 기본 3은 기존 첫 샤드에 두 샤드를 추가하는 범위다. 최대 목표는 1,063개다. 한 호출에 여러 샤드를 처리하며 8시간·I/O 제한에 따라 다음 샤드 전에 멈춘다. 누적 예산과 전체 실행은 별도 결정이다.
3. 첫 실행에서는 `EXPECTED_BOUNDARY_SHA256`과 `RECOVER_WRITER_OWNER`를 비워 둔다. `RUN_V2_BULK`를 켜고 전체 실행한다. `reports/cache/v2_bulk_cpu_reports_041837Z/invocation-*.json`과 화면에 출력된 SHA를 회수한다. 보고서의 `worker.boundary_sha256`, 완료 샤드·프레임 수, 추가 수, `stop_reason`을 확인한다. `passed`는 설정 목표 도달, `paused`는 완료 경계를 보존한 목표 미달이다.
4. 같은 고정본으로 재개하고 이전 보고서의 `EXPECTED_BOUNDARY_SHA256`과 새 누적 목표를 입력한다. 알려진 SHA 이후에 정상 검증·게시된 경계도 발견하므로 갑작스러운 종료로 보고서를 쓰지 못했을 때 해당 경계부터 이어간다. 번호 건너뜀, 고정 SHA 불일치, manifest 범위 불일치 및 보관 ZIP 누락은 중단 사유다.

CPU는 완료 샤드의 ZIP을 모두 복원하지 않는다. 카탈로그와 경계 기록, 보관 ZIP의 크기를 확인하고 다음 번호의 원본 파일만 읽는다. 새 샤드는 고정 크기·SHA와 RGB decode·pose를 확인해 기존 배열 형식으로 만든다. 게시된 ZIP을 다시 해시한 후에만 경계 JSON을 원자적으로 게시한다. 경계 게시 전에 중단돼 다음 번호 ZIP 하나만 남았으면 ZIP SHA와 내부 파일·receipt를 검증하고 해당 경계를 복구한다. 복수 후보나 범위를 건너뛴 파일은 중단 사유다. 이미 보관한 ZIP의 같은 크기 오염은 CPU 재개 시 검사하지 않으며 GPU가 선택해 읽는 ZIP은 실제 SHA를 검증한다.

샤드 한 개와 ZIP을 로컬 스테이징에 둔다. 호출당 원본·검증 읽기 예약과 쓰기 예약의 상한은 각각 8 GiB다. 개별 원본은 8 MiB, ZIP은 64 MiB, RSS 기준은 4 GiB다. 부트스트랩·마운트·진단 여유를 포함한 총 상한은 28,800초다. 원격 장기 보관 공간은 로컬 작업 공간과 별도다.

### 중단과 잠금

같은 Drive 경로에 CPU writer는 한 개만 실행한다. 여러 VM의 Drive 마운트 사이에서 완전한 분산 락은 확인하지 않았다. 정상 종료와 처리 가능한 중단은 writer 잠금을 해제한다. 이전 VM과 다른 CPU 실행이 모두 완전히 종료됐는지 확인한 후 일반 재개를 시도한다. `v2_bulk_writer_locked owner=…`가 나오면 표시된 정확한 값을 `RECOVER_WRITER_OWNER`에 넣어 재호출한다. owner 기록 전에 종료됐으면 `empty`가 표시된다. 복구는 고유한 `.abandoned-writer-*` 경로로 이전 잠금을 이동하며, 토큰 불일치나 잠금 소유권 상실 시 중단한다. 복구 동작 중 종료돼 `.writer-recovery`가 남으면 활성 실행이 없는지 확인하고 테스트 세션에서 상태를 회수한다. 잠금은 자동으로 제거하지 않는다.

## GPU 경계 배치

CPU 보고서가 나온 뒤 GPU Colab에서 `v2_bulk_cache_gpu.ipynb`와 동일한 source ZIP을 사용한다. CPU 보고서에서 확보한 SHA를 `EXPECTED_BOUNDARY_SHA256`에 넣고 오른쪽 인덱스 `RIGHT_SHARD_INDEX`를 설정한다. 기본값 1은 인덱스 0의 마지막 프레임과 1의 첫 프레임을 읽는다. 두 프레임은 모두 train이어야 하며 그 범위가 지정한 완료 경계 안에 있어야 한다. GPU 모델은 제한하지 않는다.

선택한 인접 ZIP 두 개만 내려받아 SHA·생산 binding·receipt·파일 SHA·manifest 범위를 검증한다. 기존 `Cache.frame`과 학습의 `frame_batch`를 사용해 이미지·2D 좌표·유효 마스크를 읽고 CUDA 왕복 전송한다. `status=passed`, `worker.cuda.all_exact=true`, 읽은 샤드 번호와 프레임 키를 확인한다. 허용한 Torch 생산→소비 빌드는 `2.14.1+cpu` → `2.14.1+cu130`이며 NumPy·Pillow와 다른 데이터 계약 조건은 같아야 한다.

GPU ZIP 입력은 최대 128 MiB다. 로컬 readback을 포함한 읽기 예약 상한은 256 MiB, 쓰기 예약 상한은 128 MiB다. 경계 메타데이터 읽기는 별도다. 호출 상한은 1,800초이고 진단은 `reports/cache/v2_bulk_gpu_reports_041837Z/`에 보관한다. 게시에 실패하면 표시된 로컬 `diagnostic.json`과 `worker.log`를 회수한다.

학습 엔트리는 기존대로 완전한 캐시와 strict binding을 요구한다. 이 시험은 혼합 생산 binding을 샤드별로 검증해 공통 학습 로더에서 두 프레임을 읽는 범위다. 전체 학습용 캐시 구성과 학습 시작은 별도 작업이다.

## 이번 Colab 검증 범위

첫 CPU 호출은 누적 목표 3으로 샤드 1·2를 연속 추가한다. 완료 시 384프레임이다. 새 CPU 런타임에 첫 보고서의 경계 SHA를 넘기고 누적 목표 4로 재개한다. 시작 샤드 3, 추가 샤드 1, 완료 512프레임을 확인한다. 이후 GPU는 두 번째 보고서의 경계 SHA와 오른쪽 인덱스 3을 사용한다. 이 경계의 train 프레임은 같은 `Guitar_Playing__15_` 시퀀스의 21·22다. 이 순서로 재개 직전·직후에 보관된 데이터를 한 배치로 확인한다.

## 고정본 생성과 검증

기존 `ed98124` source ZIP과 회수 CPU 보고서·카탈로그를 사용한다. 두 노트북은 고정 ZIP의 로컬 추출본에 LF 오버레이를 적용한다. 새 샤드는 신규 소스 identity를 기록하며 첫 샤드는 원래 생산 binding을 유지한다. 결과 `receipt.json`에 두 노트북과 source pin을 기록한다.

```powershell
python -m scripts.build_v2_bulk_cache --source <v2_cpu_cache_ed98124.zip> --catalog <catalog.json> --cpu-report <invocation-hlok774l.json> --output <new-output>
python -m unittest discover -s tests -p test_v2_infrastructure.py -k bulk -v
```
