# 다음 실제 데이터 준비 실행

## 이번 작업: 대응 왼쪽 RGB 두 파일의 제한된 추출

2026-10-17 사용자가 다음 단계 진행을 요청했다. 완료된 8MiB scan을 반복하지 않고, **최대64MiB 범위에서 기존 오른쪽 frame1·10에 대응하는 왼쪽 두 파일을 추출**하는 별도 작업을 준비한다. 두 목표가 완전히 저장되면 즉시 멈추며, 하나도 찾지 못해도 정해진 상한에서 종료한다. CPU 런타임을 사용하고 GPU는 필요 없다.

사용자 실행은 **테스트 세션**이 고정 revision·소스 ZIP·사용자용 notebook을 받아 안내한다. 이 문서의 계획만 보고 구버전 notebook의 숫자를 바꾸거나 새 셀을 추가하지 않는다. 코드와 실행 파일 검토 후 전달한 버전만 사용한다. 실제 새 Colab 실행과 왼쪽 확보는 아직 미검증이다.

## 근거와 목적

- 기존 오른쪽 PNG는 Serena `Getting_Up__1_`의 frame10·1이다. 기존 pose32개는 다른 sequence이므로 아직 RGB 좌우+pose triplet이 없다.
- `af6961a` scan은 같은000의 앞8,350,136bytes에서 오른쪽 경로10→1→6→8→11→9→7만 관측했다. 앞6개는 payload+padding을 소비했고 마지막7은 미완료다. 저장0은 scan의 정상 동작이다.
- 이전 왼쪽 선택 실행과 source parts 및32개 Range/수신hash가 같다. 선택이 무시된 것이 아니라 이 prefix에서 왼쪽 경로를 만나지 못했다. 전체 archive의 왼쪽 부재나 정확한 위치는 미확정이다.
- 다음 작업은 넓어진 범위에서 **찾은 목표를 바로 저장**한다. 목록 조사 후 같은 범위를 다시 읽는 별도 추출을 기본 절차로 두지 않는다. Gzip은 여전히 byte0부터 순차 해제하며 중간 점프·이어받기는 없다.

## 대상과 예산

두 목표의 공통 sequence는 다음과 같다.

```text
ArchVisInterior_ArchVis_RT/Day/rp_serena_rigged_004_ue4/SKM_MenReadingGlasses_Shape_01/003/Getting_Up__1_
```

`profile="counterpart64MiB_v1"`, `mode="extract"`, `purpose="targeted"`를 사용한다. `exact_targets`와 `includes`에는 동일한 두 경로를 지정한다. 이름 있는 프로파일은 실제 `limits`가 고정 preset과 정확히 같아야 한다.

해당 sequence 아래 `fisheye_final_image/camera_left/final_1.png`와 `final_10.png`를 정확한 파일명으로 지정한다. `camera_*` glob으로 다시 오른쪽을 선택하지 않는다. 기존 오른쪽 결과 ZIP은 그대로 보존한다.

| 항목 | 이번 작업 상한 |
| --- | --- |
| 준비 응답 본문 합계 | 64MiB, media와 전후 metadata 포함 |
| 준비 경과 시간 | 300초, 협력적 중단 |
| 준비 총 요청 | 768회 |
| 해제 출력 누적 | 128MiB |
| HTTP Range / 내부 IO·해제 chunk | 256KiB / 64KiB 유지 |
| 개별 파일 / 준비 디스크 | 4MiB / 16MiB 유지 |
| 저장 파일 | 정확한 목표2개 |
| 최초 metadata builder | 별도128KiB·30초·최대8요청 유지 |

64MiB를256KiB로 나누면256회이며 media마다 전후 metadata를 더한 총3요청 기준으로768회를 둔다. 실제 전송 상한에는 metadata body도 포함되므로 압축본문64MiB 수신을 보장하지 않는다. 이전 준비29.116846초·8,356,024bytes를 단순 환산하면 약234초지만 한 번의 관측에 따른 참고값이며 처리량·목표 발견·완료 시간을 보장하지 않는다.

첫8MiB도 다시 읽는 비용에 포함된다. 인증·소스 업로드·로컬 사전 검사·ZIP 생성은 준비 시간과 별도이며 DNS/auth/OS 전체 hard deadline이나 프로세스 RSS 상한은 아니다. Job·초기 metadata 기록과 결과 ZIP의 추가 디스크도 필요하다. 예산 확대는 이번 이름 있는 고정 프로파일에 한정하며 기본8MiB 설정은 유지한다.

## 사용자 실행 흐름

1. 테스트 세션이 전달한 **사용자용 notebook을 새 CPU 세션**에서 연다. 기존 `/content` 결과는 먼저 보관한다.
2. 첫 소스 업로드 셀에 같은 revision의 **직접 소스 ZIP** 한 개를 올린다. 바깥 배포 묶음·결과 ZIP을 올리지 않는다.
3. 로컬 사전 검사 후 단일 `JOB` 셀을 실행한다. 사용자용 사본에는 해당 Drive000, 이번 프로파일, 왼쪽 exact targets2개, 작업 ID와 사유가 미리 채워져 있다. 별도 plan JSON 업로드나 셀 추가는 없다.
4. Preflight에서 profile·mode·실제 예산·정확한 두 경로·byte0 재읽기를 확인한다. 설정을 바꿨다면 preflight를 다시 확인한다.
5. 실행 셀의 `EXECUTE_ONCE=True`로 바꾸어 **한 번만** 실행한다. 인증→metadata→targeted 추출 순서다. 같은 작업을 자동 반복하지 않는다.
6. 결과 셀에서 완료/미확보 targets, 저장 파일 수, 검증 상태, 실제 비용과 중단 이유를 확인한다.
7. 마지막 셀의 `DOWNLOAD_RESULT_ZIP=True`로 결과 ZIP을 받는다. 실패해도 가능한 보고서를 보관하고 테스트 세션에 전달한다. Drive 프로젝트 `reports/`에도 보관한다. Notebook 저장만으로 임시 결과는 보존되지 않는다.

## 결과별 판단

| 결과 | 다음 행동 |
| --- | --- |
| Pilot의 `targets_complete=true`, `targets_status=complete`, `verification.status=passed` 및 현재 파일 hash 검사 통과 | 왼쪽 파일 확보 후보로 보존하고 기존 오른쪽과 sequence/frame·PNG 내용·영상 정합성을 별도 검토한다. |
| 목표1개만 완전 저장 | 해당 파일과 미확보 목록을 보존한다. 두 목표 완료나 stereo clip 확보로 표시하지 않는다. |
| 목표0개·예산 상한 중단 | 도달 경로·비용을 보존하고 종료한다. 같은 설정 반복·자동 상한 확대를 하지 않는다. |
| 원본 변경·후검사 미완료·parser/CRC 실패 | 저장된 파일이 있어도 unusable 상태를 우회하지 않는다. |
| 인증·metadata 실패 | 결과 ZIP을 보관하고 중단한다. 이전 실행의 manifest로 진행하지 않는다. |

`stop_reason=targets_complete`만으로 최종 성공을 판단하지 않는다. 이는 목표 저장 후 추출을 멈춘 이유이며, 뒤이은 파일 hash 검사 실패는 별도로 보고한다.

원본 archive 전체 무결성, 대응 pose·timestamp·split·원본 motion 독립성, 독립 validation과 실제 학습은 별도 미완료 항목이다. 두 RGB 파일을 확보해도 M0/M1/M2/M3 전체 완료나 모델 성능 개선을 뜻하지 않는다. 다음 pose 준비나 학습은 결과 검토 후 별도 작업으로 인계한다.

이전 scan 결과의 근거는 [로드맵](ROADMAP.md#2026-10-17-사용자-colab-scan-결과-검토), 준비 계약과 제한은 [압축 준비 도구](ARCHIVE_PREPARATION.md), 환경·보관 방법은 [Colab 안내](COLAB_SETUP.md)를 따른다.
