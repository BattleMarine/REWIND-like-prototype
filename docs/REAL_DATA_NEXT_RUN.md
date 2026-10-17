# 다음 실제 데이터 준비 실행

## 이번 테스트: 확보한 RGB에 대응하는 pose 두 파일

2026-10-17 사용자 지시에 따라 Serena `Getting_Up__1_`의 frame1·10 pose를 확보하는 **한 번의 사용자 테스트**를 준비한다. 왼쪽 RGB 추출은 성공했고 같은 작업을 반복하지 않는다. 기존 오른쪽 결과와 새 왼쪽 결과는 보존한다. 이번에는 pose 결과 ZIP만 회수하고 관절·좌표 변환·영상 대조는 오케스트레이션과 디벨로퍼가 로컬에서 이어서 검토한다. 사용자가 추가 진단 셀을 작성할 필요는 없다.

기존 API3의 `counterpart64MiB_v1`과 정확한 두 목표 완료 중단을 재사용한다. 코어·예산·학습 guard는 변경하지 않는다. Pose의 archive 순서는 RGB와 다를 수 있으며 해당 파일을 이번 예산 안에서 만날지는 아직 모른다.

## 고정 설정

- `job_id="archvis_pose_1_10_64m_001"`, `mode="extract"`, `purpose="targeted"`, `profile="counterpart64MiB_v1"`.
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
| 해제 출력 / tar member | 128MiB / 1000 |
| 개별 파일 / 준비 disk | 4MiB / 16MiB |
| 저장 목표 | 정확한2개 |
| HTTP Range / 내부 IO | 256KiB / 64KiB |
| 최초 metadata builder | 별도128KiB·30초·8요청 |

두 단계 합계는64.125MiB·협력적330초·776요청이다. 인증·업로드·사전검사·ZIP 생성·전체wire 비용은 별도이며 OS hard deadline/RSS 상한이 아니다. 결과ZIP은 최대17MiB와 그에 필요한 별도 디스크 공간을 사용한다. Pose의 압축률에 따라 byte보다 해제량·member 상한이 먼저 닿을 수 있다. RGB의150.5초 실측을 pose 예상 시간으로 환산하지 않는다.

새 실행은 byte0부터 다시 읽는다. 중간 gzip 조각의 독립 해제·seek·resume가 아니다. 두 목표를 완전히 저장하면 즉시 종료하고, 미확보·부분·오류도 보고 후 중단한다. 실패를 이유로 사용자나 테스트 세션이 예산·필터·코드를 바꾸거나 반복 실행하지 않는다.

## 사용자 실행과 결과 회수

실행 안내와 중간 질문은 이 프로젝트의 **테스트 세션**이 담당한다. 검토한 고정 revision의 사용자용 `archive_prepare_pose_ready_<revision>.ipynb`와 직접 소스 `rewind-archive-source-<revision>.zip`을 전달한다. 이전 왼쪽용 notebook을 재실행하지 않는다.

1. 기존 결과를 보관하고 새 CPU 런타임에서 전달된 pose notebook을 연다. GPU는 필요 없다.
2. 첫 셀에 직접 소스 ZIP 한 개를 업로드한다. 결과 ZIP이나 바깥 배포 묶음은 소스가 아니다.
3. 사전70검사 로그를 보관한 뒤 미리 채운 JOB와 preflight의 pose kind·두 경로·고정 예산을 확인한다.
4. `EXECUTE_ONCE=True`로 실행 셀을 한 번 실행한다. 인증→metadata→pose 추출 순서이며 별도 선택 JSON이나 새 셀이 없다.
5. 결과 검토 셀과 `DOWNLOAD_RESULT_ZIP=True` 셀을 실행해 `archive_job_archvis_pose_1_10_64m_001.zip`을 보관한다. 실패했어도 가능한 보고서를 보존한다.
6. 결과 ZIP·사전검사 로그·마지막 출력·실행 notebook을 테스트 세션에 전달한다. Drive 프로젝트 `reports/`의 영구보관 여부는 별도로 기록한다.

## 성공 판정과 후속 로컬 검토

- 추출 성공: pilot의 `verification.status=passed`, `targets_complete=true`, `targets_status=complete`, 정확한 두 완료 경로·미확보0 및 현재 파일 hash 검증 통과. `stop_reason=targets_complete`만으로 성공 판정하지 않는다.
- 하나만 완전하면 부분 확보,0개면 정한 범위에서 미확보다. 전체 archive에서 pose가 없다는 결론은 내리지 않는다. 원본 변경·후검사 미완료·parser/CRC/hash 오류면 unusable 상태를 우회하지 않는다.
- 회수 뒤 로컬에서 JSON schema·87관절·필수값·단위/좌표 관계, 기존62관절 변환 규격 적용 가능성, 좌우 RGB와 sequence/frame 대조 및 GT2D overlay를 검사한다. 기존 오른쪽 원본 보존본을 먼저 찾아 hash를 확인하고, 없다면 재다운로드보다 보관본 전달을 우선한다. 보관본이 없을 때 필요한 안내는 테스트 세션에서 취합한다.
- 로컬에서 찾은 이전 공식 서버 right10 추출본은 hash가 과거 사용자 결과와 같아도 해당 실행이 `http_body_failed`/unusable이므로 검증된 보존본으로 대체하지 않는다. 기존 사용자 오른쪽 ZIP(`rewind_pilot_2c11fc10.zip` 또는 동일 파일을 확보한 보관본)을 테스트 결과와 함께 회수하거나 누락으로 남긴다. 원본을 다시 읽는 작업은 이번 pose 테스트에 추가하지 않는다.
- 파일명 대조와 overlay만으로 timestamp·calibration·원본 motion·독립 split 검증을 모두 완료했다고 표시하지 않는다. frame1·10은 연속 clip이 아니며 학습을 시작하지 않는다. 해당 pose 확보·변환·영상 대조는 현재 미실행이다.

이전 왼쪽 확보 결과는 [로드맵](ROADMAP.md#2026-10-17-대응-왼쪽-rgb-실제-확보-결과), 고정 예산과 보고서 계약은 [압축 준비 도구](ARCHIVE_PREPARATION.md), 변환 규격은 [실제 pose 준비](REAL_DATA_PREPARATION.md)를 따른다.
