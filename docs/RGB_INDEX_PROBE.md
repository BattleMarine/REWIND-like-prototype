# RGB 부분 인덱싱 가능성 조사

2026-10-29 · Production 구현이나 전체 데이터 인덱싱이 아닌 제한된 조사다. 기준 코드는 `33adef1`, 독립 실험 스크립트·의존성·입출력은 Git 제외 경로 `outputs/rgb_index_probe_20261029/`에 보존한다. 기존 환경과 원본 archive는 변경하지 않았다.

## 목적과 구성

파일명→tar 해제 위치·크기를 보관하는 목록과, gzip 압축 위치→해제 위치·32KiB dictionary를 보관하는 restart 인덱스를 함께 시험한다. `000`, `001` 번호만으로 내부 영상 위치를 찾을 수는 없다. 최초 인덱싱은 대상 구간을 순차적으로 읽어야 하며, 이후에만 알려진 파일로 건너뛸 수 있다. [indexed_gzip 공식 설명](https://github.com/pauldmccarthy/indexed_gzip)

공식 UnrealEgo2 서버의 `Lab_Scene.tar.gz000`에서8MiB를206 Range로 받았다. 응답 범위·길이·encoding과 실제 수신량을 검사했다. 큰 장면 `KyotoAlley_Showcase.tar.gz000`의64MiB 요청은 전체300초 예산에서11,862,016bytes 수신 후 중단했고 불완전 다운로드를 폐기했다. Lab142.891초, Kyoto158.078초, 합300.969초/20,250,624bytes다. 시간 상한은 blocking read 사이에서 검사하는 협력적 제한으로 정확히300초인 hard deadline은 아니다. 추가 다운로드는 하지 않았다.

따라서 작은 장면과 큰 장면의 **완성된 전체 archive 두 개를 비교한 시험은 아니다**. 공식 원본의 로컬 prefix 시험이며 사용자 Drive 복사본과의 byte 일치·Colab·Drive API 처리량은 확인하지 않았다. 큰 실제 입력의 성능 경계는 네트워크 제한 때문에 미확정이다.

## 작은 입력·큰 입력 비교

추가 전송 없이 같은 실제 Lab 원본에서2MiB와8MiB prefix를 비교했다. 두 독립 장면이나 전체 archive 비교가 아니다. 최종 설정은 spacing1MiB/해제 버퍼4MiB이며, lazy index가 비어 있을 수 있어 알려진 구간을 명시적으로 인덱싱한 뒤 저장했다. 아래 시간은 로컬 생성·검사 시간으로 다운로드를 포함하지 않는다.

| 항목 | 작은 입력2MiB | 큰 입력8MiB |
| --- | --- | --- |
| 완전 PNG 수 | 3 | 12 |
| gzip 인덱스 크기 | 65,625bytes | 262,341bytes |
| 생성·검사 시간 | 0.047초 | 0.213초 |
| 로컬 누적 압축 읽기 | 4,194,304bytes | 18,800,692bytes |
| 관측 peak working set | 33,185,792bytes | 37,535,744bytes |
| 새 프로세스 앞·중간·끝 재추출 | 3/3 hash·PNG CRC 일치 | 3/3 hash·PNG CRC 일치 |
| 마지막 PNG의 압축 읽기 시작 위치 | 1,059,225 | 6,334,675 |

4MiB 버퍼에서2MiB 입력을 순차 읽기만 하면 seek point가0개인35bytes 인덱스가 만들어졌다. 명시적 인덱스 생성 후65,625bytes와 유효한 restart를 확인했다. 따라서 인덱스 파일의 존재만으로 생성 성공을 판정하면 안 된다. 최종 재추출에서는 PNG당 약0.99~4.81MiB를 읽었으며 큰 버퍼는 순차 생성의 중복 읽기를 줄이는 대신 한 파일 조회의 선행 읽기를 늘릴 수 있다.

[2MiB 측정](../outputs/rgb_index_probe_20261029/Lab_Scene.actual2MiB.prefix.catalog.json), [2MiB 재추출](../outputs/rgb_index_probe_20261029/Lab_Scene.actual2MiB.prefix.catalog.reopen.json), [8MiB 최종 측정](../outputs/rgb_index_probe_20261029/Lab_Scene.prefix.catalog.json), [8MiB 최종 재추출](../outputs/rgb_index_probe_20261029/Lab_Scene.prefix.catalog.reopen.json)을 보존한다. 아래는 원인 분석용 초기64KiB 버퍼 결과이며 최종 측정과 구분한다.

## 실제 Lab 8MiB 결과

- Python3.12/Windows, `indexed_gzip1.10.3`, spacing1MiB. 완전 PNG12개를 목록화하고 잘린 다음1개를 제외했다. 원본 prefix SHA256 `1a4465e6ea798a81e71a90eff947574fa7c2dc579445cb68fba7f635a13e98bd`.
- 초기 해제 버퍼64KiB에서 인덱스262,341bytes, 생성0.282초, 해제 위치8,421,407bytes, 관측 peak working set37,765,120bytes. 전체 archive/gzip CRC는 미검증이다.
- 인덱스를 export하고 새 프로세스에서 import하여 앞·중간·뒤 PNG3개의 size/hash/PNG 내부 CRC를 대조했다. 중간·뒤 추출은 각각 압축 위치3,169,161/5,279,351에서 읽기 시작했고 각 약1.69MiB를 읽었다. 인덱스 생성 때의 모든 압축 데이터를 처음부터 재해제하지 않았다.
- 오케스트레이션은 별도 표준 라이브러리 `gzip`+`tarfile` 순차 reader로12개 PNG 전체의 hash·크기·tar offset을 독립 대조했다.
- 같은 고정8MiB 입력에서 앞부분 인덱스를 import한 후 뒤쪽으로 확장해 seek point5→9개와 목표 hash 일치를 확인했다. 반면4MiB 소스로 만든 인덱스를8MiB 소스로 import하는 실제 API 시험은 `ZRAN_IMPORT_INCONSISTENT`로 거부됐다. 현재 방식으로 길이가 커진 prefix에 기존 인덱스를 그대로 붙일 수 없으며, 원격 중단·재개는 별도 설계가 필요하다.
- 한 물리 파일을 가상 조각으로 나눈 reader에서 경계 전후128bytes와 재추출을 확인했다. 실제10GiB 조각 사이 이동은 미검증이다.

## 운영 비용과 판단

작은 실제 데이터에서 **저장한 인덱스로 뒤쪽 PNG를 재추출할 수 있음**은 확인했다. 다만 단순히 파일 객체를 Drive API에 연결하는 제품화는 아직 이르다. 초기8MiB 인덱싱의 계측된 로컬 누적 읽기는88,569,754bytes로, 전송8MiB와 다르다. 라이브러리는 raw read마다 앞 seek point부터 해제하며 해제 버퍼를 spacing의4배로 권고한다. 해제 버퍼만4MiB로 늘린 로컬 재측정에서 생성0.149초/누적 읽기18,835,912bytes로 감소했다. 여전히 입력8MiB보다 많이 읽으므로 내부 재읽기를 그대로 네트워크 요청으로 만들면 비용이 커진다. 실제 적용에는 범위 캐시와 읽기 묶음이 필요하다.

인덱스도 무한히 작지 않다. spacing1MiB에서 dictionary만 대략 해제량의3.125%이며 PNG 중심 archive의 해제/압축 비율이1에 가깝다고 **가정**하면 ArchVis48.84GiB에 약1.53GiB가 된다. spacing16MiB라면 약98MiB지만 임의 조회 때 더 많이 해제해야 한다. 이는 공식32KiB dictionary 구조에 근거한 계산이며16MiB 설정의 실제 시험·RAM 상한이 아니다. 경로 목록·구조체·파서 비용도 별도다. 전체2.5TB에1MiB 간격을 적용하면 dictionary만 수십GB가 될 수 있으므로 장면별 분리·필요 시퀀스 추출 재사용이 필요하다.

다음 구현 후보는 장면별 파일 목록과 restart 인덱스를 원본 identity에 묶어 영구 저장하고, 제한된 압축 범위 캐시로 읽어 필요한 sequence만 추출하는 방식이다. 초기 인덱싱 비용, 원본 변경 감지, 실제 분할 경계, checkpoint 재개, 원격 요청 증폭을 확인하기 전 전체 RGB 인덱싱을 시작하지 않는다. 이번 조사로 M0~M3 완료·학습 가능 상태를 올리지 않는다.

## 근거 보존

[전송 기록](../outputs/rgb_index_probe_20261029/fetch_report.json), [실제 파일 목록·생성 측정](../outputs/rgb_index_probe_20261029/Lab_Scene.prefix.catalog.buffer64KiB.json), [새 프로세스 재추출](../outputs/rgb_index_probe_20261029/Lab_Scene.prefix.catalog.reopen.buffer64KiB.json), [고정 입력 부분 인덱스 확장](../outputs/rgb_index_probe_20261029/Lab_Scene.prefix.catalog.resume.buffer64KiB.json), [오케스트레이션 독립 대조](../outputs/rgb_index_probe_20261029/root_review.json)를 보존한다. 이 경로의 바이너리·실험 의존성은 Git에 올리지 않는다. 다른 checkout에서는 로컬 산출물이 없을 수 있다.
