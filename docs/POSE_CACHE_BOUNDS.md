# JSON 확보용 압축 캐시 상한 계산

2026-10-29 · 계산과 운영 제안이며 새 예산을 적용하거나 추가 원본 읽기를 실행한 것이 아니다.

## 결론과 적용 범위

한 장면의 모든 위치를 탐색할 수 있도록 압축 prefix를 보존하는 경우 본체 상한은 `S = 고정한 분할 원본 크기의 합`이다. 목표 JSON이 어디에 있는지 모르므로 작은 추출 파일 크기로 필요 압축량을 예측할 수 없다. 메모리 작업 버퍼, 디스크 압축 캐시, 추출 JSON, 학습 배치는 다른 항목이다. 캐시를 저장하지 않고 순차 처리할 수도 있지만 런타임 종료 후 원본 재읽기 비용이 발생한다.

2026-10-28 Drive **train_val pose 15장면** metadata 기준이며 test pose·다른 데이터셋·향후 변경 파일의 상한은 아니다. [목록](../outputs/archive_lookup_20261028/drive_inventory.json)과 [계산](../outputs/cache_bounds_20261029/bounds.json)을 보존한다. 실행 시 원본 크기와 identity를 다시 대조해야 한다.

| 범위 | 압축 본체 상한 | 권장 캐시/ZIP 예산 | 구현 상태 |
| --- | --- | --- | --- |
| 이번 Serena JSON 두 개 | 실제234MiB에서 확보 | 새 캐시ZIP245,473,378bytes | 실제 성공, 다른 JSON 위치 보장 아님 |
| ArchVis 한 장면 전체 | 368,145,316bytes =351.091MiB | 현재 각384MiB | v4 논리352MiB/캐시384MiB; EOF까지 실제 검증하지 않음 |
| 확인한 장면 중 최대 Kyoto | 3,369,982,352bytes =3.139GiB | 한 장면당4GiB 제안 | 기존 v4로 지원하지 않음 |
| 15장면 전체 보존 | 18,020,771,152bytes =16.783GiB | metadata·ZIP·여유 별도 | 전체 보관/다운로드 미실행·비권고 기본값 |

현재 prototype에서는 ArchVis의384MiB 정책을 유지하고 필요한 JSON 추출본을 재사용한다. 범용 운영을 구현한다면 한 번에 한 장면만 처리하고 최대4GiB 압축 캐시 예산으로 시작하는 설계를 제안한다. 이는 디스크 예산이며 RAM4GiB를 할당하라는 의미가 아니다. 다음 장면으로 전환할 때 검증된 추출본·manifest를 보존하고 활성 캐시를 교체하는 정책이 필요하다. 원본과 기존 시험 캐시는 자동 삭제하지 않는다.

## 임시 디스크와 재읽기 비용

입력 cache ZIP 크기를 Q, 이번 출력 압축 prefix를 C라고 하면 현재 구현은 업로드본 Q + 생성 cache C + export ZIP C를 동시에 둘 수 있다. 따라서 `D ≈ Q + 2C + manifest/보고서/선택 파일`이며 Q≈C≈S이면 압축 본체만3S다. 단순히 캐시384MiB만 비우면 충분한 구조가 아니다.

- ArchVis 전체의3S는 약1.029GiB다. 기존 v4의 보수적 상한 합384×3+32+33=1217MiB에 코드·환경 여유를 더해 임시 디스크2GiB 이상을 요구한다. 모든 라이브러리/RSS의 hard bound를 보장하지 않는다.
- 최대 Kyoto의3S는 약9.416GiB다. 각4GiB 예산을 그대로 쓰면12GiB와 출력/환경 공간이 필요하므로 **임시 디스크16GiB 이상**을 후속 설계의 출발점으로 제안한다. 실제 구현·처리량 검증 전 보장치가 아니다.
- Warm import와 replay는 대략2Q, export는 대략3C의 로컬 읽기를 요구한다. 현재 export는 복사·내용 검증·ZIP hash 확인을 수행한다. 모든 값을 압축 캐시 용량과 별도로 산정해야 한다.
- 현재 notebook의 `files.upload()`는 ZIP 전체를 bytes로 받는다. 수GiB 범용 지원에서는 디스크로 streaming하는 입력 방식으로 바꿔야 하며 캐시 상수만4GiB로 늘려서는 안 된다.

## 저장 용량만 늘리면 해결되지 않는 제약

현재 v4는 ArchVis용 고정 시험이다. 다른 장면에 그대로 적용할 실행 허가는 아니다.

- 논리352MiB, 캐시384MiB, 최대1536chunk, manifest1MiB, HTTP1916회, 준비1200초, local IO708MiB/export1152MiB가 독립적으로 적용된다.
- 최대 장면은256KiB 기준 최소12,856chunk/정상cold HTTP38,568회가 필요하다. 현재 일반 Limits의 HTTP 상한10,000회도 넘는다. 기존 비정렬 prefix 조각을 보존하면 chunk가 더 늘 수 있다.
- 현재 manifest 형식으로 최대 장면의 연속256KiB 행을 모형화하면 약1,459,384bytes로1MiB 제한을 넘는다. 이는 실제 전체 원본 검사 결과가 아닌 기존 고정 필드와 가상 chunk 행을 사용한 계산이다.
- Tar header·누적 해제량·metadata·파일당4MiB·출력 총량·시간과 요청 정책도 함께 검토해야 한다. 압축본 크기로 JSON 수나 해제량의 안전 상한을 확정할 수 없다. gzip 마지막 크기 필드를 대형 archive의 정확한 전체 해제 크기로 사용하지 않는다.
- 카탈로그의256sequence/16,000entry는 부분 관측 보존 한도다. 잘림은 추출 중단과 다르며 전체 파일 목록을 보장하지 않는다.

따라서 장면별 source manifest에서 S를 계산하고 실제 자원 예산과 대조해 시작 여부를 결정해야 한다. 상한에 도달하면 검증된 prefix와 결과를 보존한다. 앞부분 캐시를 지우는 rolling cache만으로 gzip 재시작이 가능해지는 것은 아니며, 그런 운영에는 별도 restart index 검증이 필요하다. 캐시가 충분하더라도 네트워크 오류·원본 변경·JSON 부재를 해결하지는 않는다.

## 이번 실제 성공 근거

고정33adef1/API10 `archvis_pose_1_10_tail352_001`에서 기존192MiB를 재사용하고 신규42MiB를 읽어234MiB prefix에서 두 JSON을 확보했다. 준비158.097초/506HTTP, 초기metadata 별도0.159초/1HTTP다. 저장170,985bytes, 두 JSON hash와 object 파싱을 확인했다. 목표 조기 종료이므로 전체 gzip CRC·archive 무결성은 미검증이고 `train_ready=false`다.

[제출 결과](../outputs/test_archvis_pose_tail352_20261029/verification.json)의 v4 cache는245,473,378bytes, SHA256 `47ec61cef1a9115cfc0b78d1192a9d0a84e5f0902d3a1f0fa1257449e0e842cd`다. 오른쪽 RGB 성공 원본 회수·영상/좌표/시간 정합성은 남아 있다. 실제 실행 notebook·사용자 사전검사3개 로그·RAM/디스크 숫자·Drive 영구보관은 미확인이다. 이번 계산 작업에서 다시 추출하거나 캐시를 확장하지 않았다.
