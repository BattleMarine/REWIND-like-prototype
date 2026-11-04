# Drive 자동 인덱싱의 첫 운영 준비

> **보존 기록 · 2026-11-05 정리:** 2026-11-04 사용자 결정으로 분할 압축 원본의 인덱싱·부분 읽기 운영을 중단했다. 이 문서 전체의 실행 명령, 다음 단계, 예산과 미완료 표시는 해당 실험 당시의 기록이며 현재 실행 지시가 아니다. 새 기본 경로는 [압축 해제된 RGB·pose 사용](DATA_ACQUISITION_STRATEGY.md)이다. 기존 코드·원본·인덱스·예약 기록은 보존하며 자동 재개하지 않는다. 이동된 로컬 근거는 [보관·복원 안내](LOCAL_STORAGE.md)를 따른다.

## 현재 원본 확인 (2026-11-03)

Google Drive connector의 files.get으로 기존 inventory의354파일을 모두 조회했다.90archive/354파일/3,094,462,001,717bytes이며 ID별 파일명·크기·수정 시각 차이는0이다. 이 조회는 원본 media를 읽지 않았고 archive 무결성을 확인한 것은 아니다. Connector는 version을 요청해도 normalized 응답에 제공하지 않았으므로 실행 전 Colab API에서 version을 고정해야 한다.

| 종류 | 현재 압축 bytes |
| --- | ---: |
| train_val RGB |2,597,969,076,181|
| train_val pose |18,020,771,152|
| train_val depth |176,199,211,971|
| test RGB |281,377,458,373|
| test pose |1,967,837,325|
| test depth |18,927,646,715|

[프로젝트 Drive 폴더](https://drive.google.com/drive/folders/1-_IOAgfCOh2OBSTRWb9XeocKIkCuFo-W)와 reports/indexes/fleet_index_v1을 확인했다. 기존 fleet_index_calibration_archvis_001의 pending/예약은 그대로 보존한다. 새 운영용 fleet_survey_001은 별도의 명시 조사 계획이며 이전 실패 비용을 초기화하는 재실행이 아니다.

근거는 `outputs/fleet_drive_prepare_20261103/root/drive_metadata.json`과 `inventory_comparison.json`이다. 선택 원본354개의 현재 metadata 대조 결과이며, Drive의 다른 파일 전체를 새로 수집한 목록은 아니다.

## 전체 비용과 첫 조사 비용의 구분

전체 비용을 아직 확정할 수 없다. RGB102/269 실전 전송·시간은 선택 파일 추출 결과이고, pose prefix에는 약5.48배 해제율과 일부 파일 밀도가 관측됐지만 전체 scene의 대표성은 미확인이다. Depth30archive의 해제율·파일 밀도 실측은 없다. 이 값을0으로 채우거나 다른 종류의 측정을 복사해 full plan을 통과시키지 않는다.

[공식 데이터 설명](https://unrealego.mpi-inf.mpg.de/)은 UnrealEgo2의 약1.25M stereo images, RGB/depth의8-bit PNG1024×1024 및 pose JSON을 설명한다. 이 전체 대략 수치는 archive별 실제 파일 수·해제 크기·현재 Drive의 완전 대응 pair 수를 대신하지 않는다.

첫 실행은90archive를 각 최대2개 정상 scan 구간까지 자동 조사한다. Train_val을 먼저, scene별 RGB→pose→depth, 이후 test의 같은 순서로 고정한다. 정상 구간은 자동 저장·정산하고 quota 또는 EOF에 도달하면 다음 원본으로 넘어간다. 예외·손상·원본 변경·진전 없음은 이유와 기존 정상본을 보존하고 중단하며, 실패 후 자동 retry로 예산을 새로 얻지 않는다.

조사 완료는 전체 archive catalog/CRC/학습 준비 완료가 아니다. 근거 수집과 인덱스 보존이 목적이며 자동 full 전환은 없다. 첫 구간의 관측치를 전체에 외삽할 때는 압축률·파일 크기·replay·storage·HTTP 편차를 함께 검토한다. Survey 저장소를 다른 plan으로 임의 변환하는 이관 기능도 이번 범위가 아니다.

## 조사 예산: 기준의2배 상한

아래 기준은 전체 데이터 비용 예측이 아닌 명시된 소량 측정 작업의 resource allocation이다. Archive당 두 segment, segment별 최대 source64MiB/1000HTTP/artifact16MiB의 기존 예약과 맞춘다. 전체 누적 cap과 archive별 cap을 함께 적용하며 세션 재개 때 초기화하지 않는다.

| 항목 | 기준 allocation | 실행 상한 |
| --- | ---: | ---: |
| 원본 응답 body |5.625GiB|11.25GiB|
| HTTP 요청 |90,000|180,000|
| 경과 시간 |관측값|상한 없음: 타이머·사용자 stop|
| Artifact 논리 I/O |1.40625GiB|2.8125GiB|
| 영구 index 저장 |1.40625GiB|2.8125GiB|
| 로컬 workspace |1.40625GiB|2.8125GiB|
| 프로세스 RAM |548MiB|1,096MiB|

RAM은 동시 최대값이고 시간 외 나머지는 archive별 합계다. Artifact 논리 I/O는 Drive FUSE의 실제 network bytes와 같지 않다. 2026-11-03 최신 사용자 결정으로 총 실행시간 상한은 두지 않는다. 경과 시간은 관측하며, 사용자 stop 요청은 안전한 저장 경계에서 처리한다. 개별 네트워크 요청 timeout은 유지한다. 실제 소비가 cap보다 작아도 scope의2segment quota에서 다음 archive로 넘어가며 임의로 더 읽지 않는다.

원본 version pin은 별도 기준354HTTP/1MiB, 상한708HTTP/2MiB다. 최초 pin·plan 생성 후 고정 입력을 보존하고 resume은 같은 입력을 사용한다. 반복 pin으로 inventory/plan hash를 바꾸지 않는다. 준비 JSON 영구 저장은 기준 출력2MiB/논리IO4MiB, 상한4MiB/8MiB다. 작은 결과 회수는 별도 allocation 기준 ZIP8MiB/논리IO16MiB, 상한 ZIP16MiB/논리IO32MiB다. 새 survey의 pin·준비·회수도 총 elapsed 상한 대신 관측값을 기록한다. 개별 요청 timeout5초는 유지한다. Runtime/source/의존성 준비 비용은 worker 소비와 별도로 표시한다.

## 실행본과 인계

인계 산출물은 `outputs/fleet_drive_prepare_20261103/handoff/` 아래 고정 revision 폴더에 source ZIP, input ZIP, READY notebook, SHA manifest로 보관한다. 추적 notebook은 pin 치환 전 template이므로 그대로 실행하지 않는다. 사용자 안내·실행·최종 회수는 이 프로젝트의 탐색 세션 `01a12dd6-3940-7b91-91dd-eae1bfaba270`이 담당한다.

CPU Colab에서 작은 source/input ZIP을 업로드하고 READY notebook을 실행한다. 인증·Drive mount 후354개 metadata의 version을 고정하며 원본명·크기·수정 시각을 재검증한다. source/input SHA와 승인 request를 확인하고 runtime inventory/plan/PREPARED를 저장한 뒤 별도 CPU worker를 시작한다. 준비 완료 후 정상 segment는 추가 승인 없이 자동 처리한다.

- 새 store: `/content/drive/MyDrive/REWIND-like prototype/reports/indexes/fleet_index_v1/fleet_survey_001`
- 준비 입력: 같은 상위 폴더의 `fleet_survey_001_prepared`에 저장한다.
- 상태 확인은 `survey_status()`, 정상 중단은 `survey_stop()`, 작은 보고서 ZIP 회수는 `make_recovery()`다.
- 같은 source/input과 Drive 경로로 `ACTION='resume'`을 사용한다. 실패 상태에서 새 store/repin/예산 reset으로 우회하지 않는다.
- 성공 판단은 `survey_allocations_finished`와90개 archive의 quota/EOF 결과다. 전체 catalog/CRC 완료나 pair 확보 완료로 해석하지 않는다.
- 오류·pending·진전 없음·원본 변경·기타 자원 cap 소진은 원인과 마지막 정상 상태를 보존하고 회수한다. 탐색 세션은 코드·예산을 임의 변경하거나 실패를 자동 반복하지 않는다.
- 회수 ZIP은 보고서·준비 manifest·coverage·제한 log만 포함한다. 실제 인덱스는 원래 Drive store에 보존하고 전체 pack store를 매번 다운로드하지 않는다.

아직 실제 Drive worker 실행 결과는 없다. Colab timer 표시·Drive 저장 후 새 runtime 복원은 탐색 세션의 실제 실행에서 확인한다.

## 시간 표시와 중단

새 survey plan은 `time_policy=observe_only`, `cap.seconds=null`로 고정한다. 임의로 큰 숫자를 시간 상한으로 넣지 않는다. 전송 timeout5초는 멈춘 개별 요청을 처리하는 제한으로, 프로그램의 전체 실행시간 제한과 다르다. Segment의120초 기준은 완전한 member 경계에서 저장·다음 구간으로 넘기기 위한 협력 yield이며 총 시간 초과 종료가 아니다.

실행 타이머는 현재 worker의 wall time을 표시하고, `observed_seconds`는 정산 receipt에 남은 처리 시간의 누적 관측값이다. 준비·세션 간 대기·실패 후 알 수 없는 시간을 모두 포함한 전체 wall time으로 해석하지 않는다. 강제 runtime 종료는 정상 stop과 다르며 pending/예약이 남을 수 있다. Colab 자체의 runtime 수명·연결 정책을 프로그램이 해제하는 것은 아니다.

## 로컬 검토 기록

Root 독립 검사에서 survey 정책·quota·준비 재사용·mock Drive·회수와 기존 Drive index/storage 관련20개 검사가15.766초에 통과했다(실패0, skip0). 가상 시간이 기존6시간을 넘겨도 조사 진행·정산·coverage가 유지되는 항목을 포함한다. 실제6시간 Drive 실행이나 실제 Colab UI 검증을 수행한 결과는 아니다. 근거: `outputs/fleet_drive_prepare_20261103/root/targeted_tests.json`과 `targeted_tests.txt`.

추가 독립 실행에서는 첫 RGB scan 뒤 stop 요청으로 추가 scan 없이 멈춘 후 새 workspace에서 resume했다. RGB2/pose2/depth1(작은 fixture EOF)의 조사 quota를 완료했고 누적 source 사용량이 초기화되지 않았으며 coverage 오류0이었다. 근거: `outputs/fleet_drive_prepare_20261103/root/stop_resume.json`. 합성 원본 기반 검사로 실제 Drive 재개를 대신하지 않는다.

Notebook 코드셀 문법, worker 비차단 시작·none 시간 인자·stop 연결, 준비 실패 시 stdlib 회수,354 metadata-only mock 요청, 준비 source pin 재사용 관련 후속5개 검사가0.531초에 통과했다. 앞선20개와1개 중복되므로 총25개 독립 검사라고 집계하지 않는다. 실제 브라우저의 타이머 렌더링은 미검증이다. 근거: `outputs/fleet_drive_prepare_20261103/root/notebook_tests.json`.

개발 측 선별 회귀50개도 통과했다(시험 본체35.782초, 원격 요청0). 기존 calibration/Drive/index/storage와 새 survey를 포함하며 root 검사와 중복된다. 근거: `outputs/fleet_drive_prepare_20261103/developer/tests.txt` 및 `test_result.json`.

최종 source 혼용 차단 보완 후 notebook 관련5개 검사를 root가 재실행해 통과했다(0.062초). 다른 경로에서 이미 불러온 프로젝트 모듈은 새 runtime을 요구하고 차단하며, 정상 source의 재사용은 허용한다.

## 2026-11-03 실제 첫 조사 결과와 상태 집계 수정

고정 b1960de의 실제 Colab worker는838.138초 동안46개 archive의 각2구간(총92segment), 파일3,752개와 디렉터리505개를 기록했다. 다음 test pose의 ArchVisInterior_ArchVis_RT에서 첫 media 요청을 열던 중 `requests`의 `connection` 분류 오류로 중단했다. 총 실행시간 cap이나 데이터 결측으로 인한 중단으로 판정하지 않는다. 인증 만료·개별 timeout 등 상세 원인은 미확정이다.

회수 ZIP `outputs/test_fleet_survey_001_20261103/fleet_survey_recovery_fd645e5fc0cb402cacb1bb65ad9b0e16.zip`의 SHA256은 `d83e6f794374e11c8627e4c23bb251a66d24250d5566b9b948c18a1b949bab02`이며 CRC를 확인했다. 회수본에는 전체 index pack/catalog가 없으므로 파일 내용·전체 무결성·새 runtime 인덱스 복원은 이 검사로 확인하지 않았다. 파일 수는 학습 가능한 RGB/pose 페어 수가 아니다.

실패 구간의 pending 예약은 source64MiB/1000HTTP/artifact16MiB로 보존한다. 표시 사용량에는 이 예약이 포함되어 있으며 정상 정산분과 구분해야 한다. 원격 store나 pending을 수정·초기화하지 않았다.

별도 `coverage` 오류는 미탐색44개 archive의 EOF 값이 없을 때 `None`을 합산하는 코드에서 발생했다. 미탐색을 false로 처리하고 빈 선택 목록을 완료로 판정하지 않도록 수정했다. 이는 조회 집계 수정이며 요청 실패 복구·학습 승인·원격 재개를 포함하지 않는다. 로컬 검증 결과는 아래에 기록한다.

Root 독립 회귀8개가7.438초에 통과했다(실패0, skip0). 실제 coverage 함수에서 미탐색·부분진행·46완료/44미탐색과 pending·EOF·전체 quota·빈 선택을 검증하고, 기존 survey 실행 완료·시간 관측 경로도 확인했다. 로컬 fixture store의 조회 전후 파일 hash·HEAD·pending·usage 불변을 확인했다. 실제 Drive store 재조회나 원격 재개는 하지 않았다. 근거: `outputs/fleet_coverage_fix_20261103/root/tests.json`.

후속 실패1건의 명시 재개 준비는 [고정 재개 계약](FLEET_SURVEY_RESUME.md)을 따른다. 원래 실행본으로 반복 재시도하거나 원 PREPARED의 source pin을 덮어쓰지 않는다.


## 종료 상태

고정 재개 af275f5에서도 같은 첫 media 요청 오류가 재발해 진전은92segment·3,752파일에 머물렀다. 복구·집계 수정과 네트워크 원인 해결을 구분한다. 이후 사용자가 인덱싱을 중단했으며 최종 근거는 [재개 결과](FLEET_SURVEY_RESUME.md)를 따른다.
