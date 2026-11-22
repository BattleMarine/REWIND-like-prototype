# v2 256px 독립 CPU 캐시

2026-11-22 확정. 파일럿 이후 256 전용 추가 Drive 220GiB, 추가 CPU 누적 12시간, 회당 최대 9시간이다. 장시간판에서 `CONFIRMED_ADDITIONAL_DRIVE_GIB=220`, `CONFIRMED_ADDITIONAL_CPU_SECONDS=43200`을 사용한다. 파일럿의 3샤드·384페어 다음인 인덱스 3에서 첫 호출을 시작하도록 테스트 세션에 인계했다. 사용자 Colab 출력에서 봉인값을 대조했고 사용자가 확인한 Drive 표시 여유는 약 1.59TB다.

128 운영 캐시와 별도의 CPU Colab을 사용한다. 256 파일·경계·잠금·시간 원장은 `reports/cache/v2_cache256_041837Z`, 진단은 `reports/cache/v2_cache256_reports_041837Z`에 기록한다. 128 운영의 60GiB·시간 원장은 이 작업에 사용하지 않는다. 두 런타임이 공통 Drive를 읽을 때 성능과 실제 가용 공간에는 영향을 줄 수 있다. [128·256 사전 비교](V2_RESOLUTION_AB.md)

256 캐시는 고정 전체 manifest의 순서를 사용한다. 목표는 1,063샤드·136,011페어다. 첫 1,062샤드는 각각 128페어, 마지막은 75페어다. 기존 비율의 256×256 float32 양안 RGB, 원본에서 새 해상도로 변환한 2D 관절, 기존 pose·camera·mask 필드를 별도 256 binding과 함께 저장한다. 256 전체 이미지 배열만 약 199.2GiB이며, 텐서·ZIP 헤더·경계와 일시적인 복사 비용은 별도다. 이 구현은 전체 학습 로더 연결을 검증한 결과가 아니다.

## 첫 30분 파일럿

1. 별도 CPU Colab에서 `v2_cache256_pilot_cpu.ipynb`를 열고 고정 `v2_cpu_cache_ed98124.zip`을 Colab `/content`에 올린다. 원래 ZIP의 manifest를 사용한다. 256 작업에는 검증된 CPU Torch·NumPy·Pillow 버전이 필요하며, 다른 버전이면 작업을 시작하지 않는다.
2. 처음에는 tip과 복구 owner를 비워 둔다. 기본 OFF인 `RUN_CACHE256`를 켜고 한 번 실행한다. Drive 마운트·잠금·예약까지 확인한다. 최초 셀 시작부터 파일럿 예약·마감 상한은 30분이다. Drive의 파일럿 보관 한도는 1GiB다.
3. 성공하면 256 샤드 0~2·384페어가 게시된다. worker는 게시된 ZIP 세 개를 다시 로컬에 가져와 ZIP 및 내부 파일 SHA, RGB shape·dtype·범위, pose·2D·카메라 tensor와 기록된 경계를 검증한다. 진단의 `pilot_receipt_sha256`과 `budget_tip_sha256`, 완료 경계, 같은 256 폴더의 `pilot-proof.json`을 회수한다.

중단하면 기존 경계 SHA와 최신 정산 또는 미정산 예약 SHA를 확인한다. 남은 파일럿 시간이 허용하는 경우 같은 파일럿 노트북에 `EXPECTED_256_TIP_SHA256`을 넣고 다음 미완성 샤드부터 한 번 실행할 수 있다. 온전한 ZIP 하나가 게시됐지만 경계가 없다면 그 ZIP과 내부 receipt를 먼저 검증한다. 미정산 예약으로 파일럿 예산이 소진되면 작업을 멈춘다.

## 사용자 승인 뒤 최대 9시간 수동 호출

`v2_cache256_long_cpu.ipynb`는 파일럿을 재사용하며 기본 OFF로 배포한다. 사용자가 256 전용 추가 시간과 용량을 확정한 뒤에 다음 값을 입력한다.

- `VALIDATED_PILOT_RECEIPT_SHA256`에 성공한 `pilot-proof.json`의 봉인 SHA를 입력한다.
- `EXPECTED_256_TIP_SHA256`에 가장 최근 256 시간 원장의 정산 또는 예약 tip을 입력한다.
- `CONFIRMED_ADDITIONAL_CPU_SECONDS`와 `CONFIRMED_ADDITIONAL_DRIVE_GIB`에 승인받은 양의 정수 값을 입력한다. 초기값은 모두 0이다.

첫 장시간 호출은 파일럿 실차감과 추가 승인 시간을 합친 누적 한도, 파일럿 실사용량과 추가 승인 공간을 합친 Drive 한도를 256 작업에 적용한다. 최초 승인값은 파일럿 proof와 함께 별도 파일로 봉인한다. 같은 캐시를 이어갈 때 같은 승인값을 사용한다. 새 값이 필요하면 기존 기록을 보존하고 사용자 확인 후 별도 인계한다. 장시간 호출은 남은 누적 시간 안에서 회당 최대 32,400초이며 완전한 캐시가 되면 일찍 끝난다. 자동 반복은 없다.

각 호출은 한 번에 한 로컬 샤드를 스테이징하며 8-worker 입력 prefetch와 SHA 확인, 게시 ZIP의 SHA 재검사를 사용한다. 파일럿의 논리 I/O 한도는 읽기 8GiB·쓰기 4GiB이고 장시간판은 읽기 1TiB·쓰기 512GiB다. 실제 Drive 가용 공간, 승인 용량 및 로컬 작업 여유도 확인한다. ZIP 상한은 224MiB, 로컬 시작 여유는 2GiB, 가용 RAM은 4GiB다.

VM 유실 때 검증 후 게시된 경계와 ZIP을 다시 사용한다. 정산이 없으면 해당 호출의 실제 예약 전액이 차감된다. 잠금을 강제로 인계하는 조작은 이전 VM과 다른 writer가 모두 종료됐음을 확인한 뒤 정확한 owner를 사용한다. Colab 서비스가 30분 또는 9시간의 VM 가동을 보장하지 않는다. 외부 watchdog과 작업 예산은 설정한 시점 이후 작업 중단을 시도하며 시스템 호출이나 강제 VM 종료 시 정산 보고서까지는 보장할 수 없다.

## 고정본 재현

```powershell
python -m scripts.build_v2_cache256 --source <v2_cpu_cache_ed98124.zip> --catalog <catalog.json> --cpu-report <invocation-hlok774l.json> --output <new-output>
python -m unittest discover -s tests -p test_v2_cache256.py -v
```

## 파일럿 실측

2026-11-22 회수 진단 invocation-5yflyscc.json은 passed / target_reached, 3샤드·384페어, Drive 604,834,114바이트, 전체 약 101초, CPU 원장 정산 191초를 기록했다. 진단 SHA-256은 92c69a667ae23373be5e687c84bc09636d12817a54c268f2404580ea6a0f3e18이며 로컬 보관 위치는 outputs/v2_plan_inputs/cache256_pilot_5yflyscc다. 진단에 기록된 파일럿 봉인 SHA는 10b7ee0e9fdd83e45df30e095445fdaba1f0aa357eb1ff0c6c9c5960ec57eef1이다. 봉인 파일 자체와 계정 여유 공간은 별도 확인 대상이다.

프레임당 실사용량을 136,011페어로 환산하면 전체 199.52GiB, 남은 135,627페어에는 198.95GiB다. 추가 220GiB를 한도 후보로 제시하며 기존 128 작업의 잔여 공간도 별도로 확보한다. 같은 벽시계 처리량의 단순 환산은 잔여 약 9.9시간이다. 파일럿만의 최종 3샤드 재검증과 고정 준비 시간이 포함되며 실제 공유 Drive 처리량은 변동한다. 한 번의 9시간 호출에서 전체 완료는 미검증이다.

## 첫 장시간 호출

256 호출 invocation-7u4_1_h6은 23,921.656초 동안 3→907샤드를 게시했다. 116,096페어를 보관했고 다음 샤드의 쓰기 예약이 호출당 상한에 걸려 정상 중단됐다. 전체 256 예산에서 남은 시간은 19,190초, 남은 저장 공간은 53,966,499,504바이트다. 인덱스 907에서 같은 예산으로 재개한다. 진단 SHA-256은 2381a1345b66b183b5c9f9fbd540ad89a6544d0f51ec6f7261e07839551a9988이다. 병행한 128 호출 invocation-elp89e_2는 target_reached로 1,063샤드·136,011페어에 도달했다. 진단 SHA-256은 1e3aa2b19ffc9afe31b3f87c6991229c86c924a2de7b6db213a5ca67a49c3aed이다.
