# 다음 실제 데이터 준비 실행

### 2026-10-19 pose 재검증 결과 — 요청 예외로 미확보

- `4571f30`/API4, `archvis_pose_1_10_density_001`의 실제 Colab 실행은 `authorized_request_failed`로 종료했다. 목표 JSON0개·미확보2개, `targets_status=unusable`, `source_validation_pending=true`다. 작업의 finished 상태는 추출 성공이 아니다.
- 압축9.5MiB(9,961,472bytes)를 읽고 준비41.529793초·116요청(media39+metadata77)·응답body9,968,402bytes에서 중단했다. Header1293·tar metadata80,643bytes·해제54,640,539bytes로 이전 제한은 넘었지만 새 제한에는 도달하지 않았다. 최초 metadata는 별도1요청·290bytes·0.127415초다.
- 39번째 media Range [9,961,472, 10,223,615]는 수신0bytes·응답 상태 코드 없이 실패했다. `session_opener`가 요청/응답 래핑 예외를 공통 오류로 바꾸므로 인증·통신·timeout 중 원인을 확정할 수 없다. `invalid_or_changed_source` 분류도 실제 원본 변경의 증거가 아니다.
- 처음128개 목록과 마지막regular 관측(총626개 중 Adanna Cards frame9, payload 미완료)은 `untrusted_failed_source_or_parser` 진단이다. 검증된 파일이나 seek index로 사용하지 않는다. 실제 pose 내용·좌표·영상 대조, 전체 archive 무결성·학습 준비는 미완료다.
- 오케스트레이션은 ZIP CRC·보고서4개 자체hash·배포 소스12개 hash·ready JOB/설정hash·보고서 연결을 독립 확인했다. 원본 ZIP SHA256은 `6e554d053340f9d074c3f486642bc68cc0455989bcd965432dfd1255e9f2fc99`다. 원본 사본·검토 근거는 Git 제외 `outputs/test_archvis_pose_density_20261019/`의 `verification.json`, `orchestrator_review.json`에 보존한다. 이번 실제 실행의80개 사전검사 전체로그·실행notebook·Drive 영구보관은 미확인이다.
- 동일 실행 반복은 종료한다. 다음 개발 검토는 비밀정보 없는 예외 종류·발생 단계 진단과 로컬 재현이다. 원시 예외문·token·URL을 기록하지 않으며 예산/timeout 확대나 자동 재시도는 이번 결과만으로 적용하지 않는다.

## 이번 테스트: 동일 목표 요청 실패 진단

2026-10-19 사용자 승인에 따라 요청 예외 종류와 발생 단계를 안전하게 기록하는 수정본을 준비한다. 이전 `4571f30` 실행의 실패 원인은 소급 확정할 수 없다. 새 실행은 같은 두 목표와 예산을 유지하며, 오류가 재발하면 진단을 회수하고 재발하지 않으면 그 실행 결과만 평가한다. 이번 요청은 위 종료된 실행의 반복 안내가 아니라 검토·커밋된 새 진단 버전의 1회 테스트다.

대상은 그대로 Serena `Getting_Up__1_` frame1·10의 pose JSON이다. 전송64MiB·300초·768요청과 저장2파일·16MiB disk는 유지한다. 기존 smoke와 RGB용 counterpart 프로파일은 바꾸지 않는다. 실제 pose 위치·이번 예산 내 확보 여부는 미확정이며, 성공·부분·0개·오류 모두 한 번의 실행 결과를 보관한 뒤 중단한다.

기존 오른쪽 사용자 결과 ZIP과 새 pose ZIP을 회수하면 관절·좌표·GT2D 영상 대조는 로컬에서 진행한다. 사용자가 추가 진단 셀이나 변환기를 실행할 필요는 없다. 이번 테스트의 추출 성공과 이후 데이터 내용 정합성 검증 완료를 구분한다.

## 고정 설정

- `job_id="archvis_pose_1_10_diagnostic_001"`, `mode="extract"`, `purpose="targeted"`, `profile="poseCounterpart64MiB_v1"`.
- `selection.kind="pose"`, `total_parts=1`, index0의 `ArchVisInterior_ArchVis_RT.tar.gz000`을 사용한다. RGB archive와 파일명이 같아도 Drive ID와 원본이 다르다. 개인 Drive ID는 검토한 사용자용 notebook에만 채우고 저장소에는 기록하지 않는다.
- 과거 pose metadata의 크기368,145,316bytes/version1은 참고 이력이다. 실행 당시 metadata와 각 Range 전후 검사를 다시 거치며 과거 값을 현재 상태로 간주하지 않는다.
- 아래 두 exact 경로를 `exact_targets`와 `includes`에 동일하게 지정한다. 기존 JSON 규칙에서 파생한 후보 경로이며 아직 이번 sequence의 실제 pose payload를 확보한 것은 아니다.

```text
ArchVisInterior_ArchVis_RT/Day/rp_serena_rigged_004_ue4/SKM_MenReadingGlasses_Shape_01/003/Getting_Up__1_/json/frame_1.json
ArchVisInterior_ArchVis_RT/Day/rp_serena_rigged_004_ue4/SKM_MenReadingGlasses_Shape_01/003/Getting_Up__1_/json/frame_10.json
```

## 예산과 중단

| 항목 | 상한 |
| --- | --- |
| 준비 응답 body 합계 | 64MiB, media와 전후 metadata 포함 |
| 준비 시간 / 총 요청 | 협력적300초 / 768회 |
| 해제 출력 / tar header | 512MiB / 10,000 |
| Tar metadata 누적 / 단일 payload | 1MiB / 64KiB |
| 개별 파일 / 준비 disk | 4MiB / 16MiB |
| 저장 목표 | 정확한2개 |
| HTTP Range / 내부 IO | 256KiB / 64KiB |
| 최초 metadata builder | 별도128KiB·30초·8요청 |

두 단계 합계는64.125MiB·협력적330초·776요청이다. 인증·업로드·사전검사·ZIP 생성·전체wire 비용은 별도이며 OS hard deadline/RSS 상한이 아니다. 결과ZIP은 최대17MiB와 그에 필요한 별도 디스크 공간을 사용한다. 이전 pose 관측을64/7.5배로 단순 환산하면 header약8,534개·metadata약534,110bytes·해제약342MiB·약255초다. 이를 참고해 여유를 둔 설정이며 분포·목표 위치·실행 시간을 보장하지 않는다. 어느 상한이든 먼저 도달하면 종료한다. 512MiB는 누적 해제량이며 전체를 RAM이나 disk에 보관하는 크기가 아니다.

새 실행은 byte0부터 다시 읽는다. 중간 gzip 조각의 독립 해제·seek·resume가 아니다. 두 목표를 완전히 저장하면 즉시 종료하고, 미확보·부분·오류도 보고 후 중단한다. 실패를 이유로 사용자나 테스트 세션이 예산·필터·코드를 바꾸거나 반복 실행하지 않는다.

최대10,000개의 중복 검사 경로는 각UTF-8 1024bytes 이내이며 Python 문자열·casefold·set overhead는 별도다. 내부 IO/해제 chunk는64KiB이고 단일 tar metadata도64KiB로 제한한다. 프로세스 전체 RAM의 하드 상한이라고 해석하지 않는다.

API5에서도 보고서는 처음128개 관측과 별도로 마지막 regular 관측1개·관측한 regular 총수를 기록한다. 목록 cap 뒤 진행 확인용이며 source 실패 때는 신뢰하지 않는다. Seek offset이나 전체 목록으로 사용하지 않는다.

진단 필드 `request_failure`는 `category`, `library`, `operation`, `stage`와 schema 버전만 허용한다. 요청 열기·응답 wrapping·본문 읽기·응답 검사·정리 단계와 초기/전후 metadata·media를 구분한다. 원시 예외문이나 URL은 기록하지 않는다. 기존 보고서의 필드 부재와 새 정상 실행의 null은 구분하며, HTTP 상태만으로 추정한 인증 실패를 예외 타입 진단과 혼동하지 않는다.

## 사용자 실행과 결과 회수

실행 안내와 중간 질문은 이 프로젝트의 **테스트 세션**이 담당한다. 검토한 고정 revision의 사용자용 `archive_prepare_pose_diagnostic_ready_<revision>.ipynb`와 직접 소스 `rewind-archive-source-<revision>.zip`을 전달한다. 이전 왼쪽용 notebook을 재실행하지 않는다.

1. 기존 결과를 보관하고 새 CPU 런타임에서 전달된 pose notebook을 연다. GPU는 필요 없다.
2. 첫 셀에 직접 소스 ZIP 한 개를 업로드한다. 결과 ZIP이나 바깥 배포 묶음은 소스가 아니다.
3. 전달된 수정본의 사전94검사 전체 로그를 보관한 뒤 미리 채운 JOB와 preflight의 pose kind·두 경로·고정 예산을 확인한다.
4. `EXECUTE_ONCE=True`로 실행 셀을 한 번 실행한다. 인증→metadata→pose 추출 순서이며 별도 선택 JSON이나 새 셀이 없다.
5. 결과 검토 셀과 `DOWNLOAD_RESULT_ZIP=True` 셀을 실행해 `archive_job_archvis_pose_1_10_diagnostic_001.zip`을 보관한다. 실패했어도 가능한 보고서를 보존한다.
6. 결과 ZIP·사전검사 로그·마지막 출력·실행 notebook을 테스트 세션에 전달한다. Drive 프로젝트 `reports/`의 영구보관 여부는 별도로 기록한다.

## 성공 판정과 후속 로컬 검토

- 진단 검증과 추출 성공을 구분한다. 오류가 발생하면 안전한 예외 분류·발생 단계가 보고서와 ZIP에 남았는지 확인한다. unknown도 유효한 제한적 진단이며 원인을 추측하지 않는다. 오류가 재발하지 않았다는 사실만으로 과거 원인이 해결됐다고 판정하지 않는다.

- 추출 성공: pilot의 `verification.status=passed`, `targets_complete=true`, `targets_status=complete`, 정확한 두 완료 경로·미확보0 및 현재 파일 hash 검증 통과. `stop_reason=targets_complete`만으로 성공 판정하지 않는다.
- 하나만 완전하면 부분 확보,0개면 정한 범위에서 미확보다. 전체 archive에서 pose가 없다는 결론은 내리지 않는다. 원본 변경·후검사 미완료·parser/CRC/hash 오류면 unusable 상태를 우회하지 않는다.
- 회수 뒤 로컬에서 JSON schema·87관절·필수값·단위/좌표 관계, 기존62관절 변환 규격 적용 가능성, 좌우 RGB와 sequence/frame 대조 및 GT2D overlay를 검사한다. 기존 오른쪽 원본 보존본을 먼저 찾아 hash를 확인하고, 없다면 재다운로드보다 보관본 전달을 우선한다. 보관본이 없을 때 필요한 안내는 테스트 세션에서 취합한다.
- 로컬에서 찾은 이전 공식 서버 right10 추출본은 hash가 과거 사용자 결과와 같아도 해당 실행이 `http_body_failed`/unusable이므로 검증된 보존본으로 대체하지 않는다. 기존 사용자 오른쪽 ZIP(`rewind_pilot_2c11fc10.zip` 또는 동일 파일을 확보한 보관본)을 테스트 결과와 함께 회수하거나 누락으로 남긴다. 원본을 다시 읽는 작업은 이번 pose 테스트에 추가하지 않는다.
- 파일명 대조와 overlay만으로 timestamp·calibration·원본 motion·독립 split 검증을 모두 완료했다고 표시하지 않는다. frame1·10은 연속 clip이 아니며 학습을 시작하지 않는다. 해당 pose 확보·변환·영상 대조는 현재 미실행이다.

이전 왼쪽 확보 결과는 [로드맵](ROADMAP.md#2026-10-17-대응-왼쪽-rgb-실제-확보-결과), 고정 예산과 보고서 계약은 [압축 준비 도구](ARCHIVE_PREPARATION.md), 변환 규격은 [실제 pose 준비](REAL_DATA_PREPARATION.md)를 따른다.
