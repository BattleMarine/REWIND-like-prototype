# ARCHVIS1000-001 준비 결과 복원

2026-11-09 사용자 시험에서 데이터 준비가 완료된 뒤 인터넷·런타임 연결 문제가 발생했다. 학습 전 budget 단계에서 `pilot` 변수를 사용할 수 없었으며, 학습은 미실행이다. 회수한 manifest·preflight·prepare report와 실행 노트북은 `outputs/archvis_pilot_handoff_20261109/user_test/`에 보존했다.

## 확인된 결과

train1,000·validation2,236페어의 준비가 완료됐다. 보고된 입력 읽기는7,567,367,096바이트, 준비 경과시간은3,517.166초다. 이는 준비 검사 시간이며 학습 처리량이 아니다. 원본 motion316개는 여전히 `declared_unverified`, 시간 정합은 미확인이다. 숫자 좌표 검사를 실제 영상 정합·독립 평가 승인과 구분한다. 현재 Drive 파일이 여전히 같은지는 새 실행에서 확인한다.

- 학습 source revision: `ceb8591a3c2c45549d04f9fda6cab871ecd47ccf`
- manifest content SHA: `ac9227722b7c780d7df7b1fb2b64e8e8be4d2e7076bab0b086668bdc4b962595`
- manifest 파일 SHA: `be1cfecc7972e391dafdda6f62dc0c1465ce05050b442df96191c57f5f7dc706`
- validation 키 SHA: `b4c52d654e772771e553ce887f81f502ea80082e825bab63d02f7c6088ba2dee`

## 고정 복원물

`outputs/archvis_pilot_handoff_20261109/restore_ARCHVIS1000_001/`에 다음을 보관한다. 영구 Drive 보관은 사용자가 수행하며 권장 위치는 `REWIND-like prototype/reports/archvis1000_001/prepared/`다. 데이터셋 원본을 이 ZIP에 포함하지 않는다.

| 파일 | 바이트 | SHA256 |
| --- | ---: | --- |
| archvis_1000_prepared_001.zip | 634174 | e8a74121689c3910123122f9b88f025f98d0334bf3e290e36978da8f6677ac13 |
| archvis_1000_pilot_restore_001.ipynb | 28419 | 6f6bb87791ac7270e1863ffbcc75cdc4625c2f19f540c7c96e0f63bc79de95b4 |

기존 `archvis_pilot_source_ceb8591_lf.zip`과 `release_inputs/archvis_pilot_inputs.zip`은 [원래 인계](ARCHVIS_COLAB_HANDOFF.md)의 pin을 그대로 사용한다. 복원 helper는 원래 학습 소스 ZIP 밖에 내장되며 버전·runtime/builder SHA를 `restore_receipt.json`에 기록한다. 원래 ready 파일에는 실행 결과가 저장되어 해시가 바뀌었으므로 이를 보존하고, 원본 ready를 별도 이름 `archvis_1000_pilot_ceb8591_original_rebuilt.ipynb`으로 재생성해 원래 배포 SHA와 일치를 확인했다.

## 새 런타임 절차

1. 테스트 세션의 고정 복원 노트북을 사용하고 `/content`에 기존 source ZIP, input ZIP, 새 prepared ZIP을 올린다.
2. Bootstrap은 새로운 `/content/archvis_1000_pilot_restored`에 패키지를 검증·추출한다. 회수3파일의 SHA·canonical hash·상호참조·공식 split·선택키·파일 경로/크기·source9개 코드 SHA를 검증한 뒤 원본 바이트 그대로 복원한다. 원래 preflight/prepare 셀은 실행하지 않는다.
3. 같은 계정의 Drive를 마운트하고 보고된 원래 RGB·pose 폴더 경로를 확인한다. 경로가 다르면 임의로 manifest/report를 고치지 않고 중단한다. 폴더 존재 확인은 파일 본문 확인이 아니다.
4. 기존 인계의 예산 정책으로 현재 GPU/RSS/디스크를 확인하고 새 comparison plan SHA를 고정한다. 별도 승인 SHA 전에는 학습하지 않는다. 동일 manifest의 MSE·CE 비교를 수행한다. 이는 optimizer checkpoint 재개가 아니라 준비 결과 복원 후 첫 학습이다.
5. 원래 trainer의 초기 train·validation 전체 평가가 파일별 size/SHA/PNG 검사를 수행한다. 변경 파일은 학습 update 전 차단하며 이후 변경도 배치별 검사로 거부한다. 준비 단계의 pose 재검사는 반복하지 않지만 초기 평가의 데이터 I/O는 필요하다.
6. 성공·실패 후 마지막 회수 셀을 실행한다. 기존 회수 ZIP에 `restore_report.json`과 해당 파일 SHA 영수증을 추가한다. 중단 후 임의 반복·예산 확대·전체 train 학습은 하지 않는다.

## 검증과 후속

Root는 실제 생성 노트북의 로컬 경로만 치환해 bootstrap·복원을 실행했고, 원본3파일 byte 일치, 경로 확인 전 train 차단, source 변조 거부, 회수 ZIP CRC·모든 receipt hash와 복원 보고서 포함을 독립 확인했다. 개발 검증에서는 준비 보고서 참조 변조 거부도 확인했다. 실제 새 Colab·Drive·GPU는 미실행이다.

사용자 요청에 따라 실제 학습 결과를 받은 후 학습 구간의 페어당 wall-clock과 준비·평가·저장을 포함한 총시간을 구분해 추정한다. MSE·CE별 실제 처리 페어 수, 재개·AMP skip, GPU·precision·batch·해상도·캐시 조건을 함께 기록하며 준비 시간으로 학습 속도를 대신하지 않는다.
