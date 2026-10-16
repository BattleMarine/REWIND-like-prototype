# 다음 실제 데이터 준비 실행

현재 확보한 실제 데이터는 기존 pose 32개이며, 대응 좌우 RGB와 독립 validation은 아직 확보하지 못했다. [변환기 안내](REAL_DATA_PREPARATION.md)와 함께 사용한다. 이 준비 작업은 CPU 런타임으로 실행하며 GPU 학습을 시작하지 않는다.

## 네 번의 RGB 실행에서 확인한 문제

2026-10-16 제공된 `rewind_pilot_5dc9b9ba.zip`, `e2941e50`, `798b7bec`, `2b006838`의 ZIP CRC와 보고서를 대조했다. 모두 완전 선택 파일은 0개다.

| 실행 | 압축 본문 수신 | 준비 단계 요청 수 | 중단 이유 |
| --- | ---: | ---: | --- |
| 5dc9b9ba | 1,638,400 bytes | 77 | `range_status_not_206` |
| e2941e50 | 2,818,048 bytes | 128 | `ranges_limit` |
| 798b7bec | 2,818,048 bytes | 128 | `ranges_limit` |
| 2b006838 | 2,818,048 bytes | 128 | `ranges_limit` |

기존 코드에서는 media 64KiB마다 원본 변경 확인 metadata를 전후 두 번 요청했다. 후자의 세 실행은 media 43회와 metadata 85회로 128회에 도달했다. 43번째 media 본문은 수신했지만 후검사 요청을 시작하지 못해 해제기로 전달하지 않았다. 첫 실행의 실제 HTTP 상태 코드는 과거 보고서에 없으므로 인증·할당량·원본 손상 중 하나로 단정하지 않는다.

세 반복 실행은 동일한 앞부분을 다시 읽었다. 누적 다운로드나 이어받기가 아니며, 기존 18 pose의 `Brutal_Assassination` RGB 경로에 도달하지 못했다. 관측된 앞부분은 다른 sequence인 `rp_serena_rigged_004_ue4/.../003/Getting_Up__1_`의 오른쪽 RGB다. 이 sequence는 공식 train 목록에 있지만, 해당 파일을 실제로 확보하고 pose·반대쪽 영상과 대조해야 한다.

## 새 실행 순서

1. 새 배포 묶음의 `archive_prepare.ipynb`와 소스 ZIP을 사용한다. 이전 소스가 import된 런타임에서는 결과를 먼저 보관하고 새 세션에서 시작한다. 소스 업로드 후 표시되는 revision을 확인하고 테스트·합성 진단을 실행한다. [Colab 안내](COLAB_SETUP.md)를 따른다.
2. 첫 실제 작업은 **RGB discovery 한 번**이다. 기존 Brutal 프레임만 선택하지 않고, 지정한 ArchVis archive 앞부분에서 도달 가능한 완전 RGB를 찾는다. 아래 plan 입력을 metadata 셀 앞에 넣는다. 새 notebook의 사전 검사와 기본 상한을 유지한다.
3. Metadata를 활성화하고 `manifest_ready=True`를 확인한 뒤 pilot을 한 번 실행한다. 기본 준비 예산은 응답 본문 합계 8MiB·60초·최대32파일·총 요청128회다. Discovery는 빠른 샘플 확인을 위해 파일 수를 최대2개로 더 낮춘다. Metadata builder는 별도 128KiB·30초이며 비용을 합산한다. 인증·DNS·OS를 포함한 hard deadline이나 전체 RSS 상한은 아니다.
4. 파일 수, 검증된 sample 유무, 중단 사유, media/metadata 요청 수를 확인한다. 검증 통과라도 파일 0개면 샘플 확보 실패다. `train_ready=False`는 준비 sample을 학습 승인으로 취급하지 않는다는 뜻이다.
5. **결과 ZIP 보관 셀**에서 `SAVE_RESULT_ZIP=True`, `DOWNLOAD_RESULT_ZIP=True`로 설정해 실행하고 ZIP을 다운로드한다. `metadata/`, `pilot/`의 보고서·manifest·완전 파일을 함께 보존한다. Notebook 저장만으로 `/content` 파일은 보존되지 않는다. ZIP을 Drive의 프로젝트 `reports/`에도 보관한다.
6. 여기서 멈추고 보고서를 검토한다. 필요하면 네트워크를 읽지 않는 대응 계획 셀의 `MAKE_COUNTERPART_PLAN=True`로 관측 파일의 다음 선택 제안만 출력한다. 같은 작업을 다시 실행하지 않는다. 발견된 sequence/frame과 실제 저장 파일을 기준으로 다음 pose·반대쪽 RGB의 좁은 선택을 만든다. 각 후속 작업은 새 예산을 사용하는 별도 실행이며 자동으로 이어지지 않는다. 기존 검증 추출본을 우선 재사용한다.

## Plan 입력 셀

새 묶음의 `archive_sample_plan_20261016.json`을 사용한다. 과거 `real_data_colab_plan_20261016.json`의 36개 좌우 개별 glob은 입력 상한32를 초과하므로 사용하지 않는다. 기존 18프레임을 다시 선택해야 할 때는 `camera_*`를 사용한 18개 glob으로 표현하지만, 32파일 상한에서 좌우36파일 전체 확보는 보장되지 않는다.

아래 셀은 notebook의 선택 입력 셀 **직후**, metadata 셀 **직전**에 넣는다.

```python
from google.colab import files
from rewind_data.audit import read_json
from rewind_archive.drive_metadata import validate_selection
from rewind_archive.contracts import Limits

uploaded_plan = files.upload()
assert len(uploaded_plan) == 1
plan_bytes = next(iter(uploaded_plan.values()))
assert len(plan_bytes) <= 128 * 1024
plan = read_json(plan_bytes)
assert plan['schema_version'] == 'archive_sample_plan_v1'
JOB = 'rgb_discovery'
job = plan['jobs'][JOB]
validate_selection(job['selection'])
assert Limits.from_dict(plan['limits']) == Limits.from_dict(
    read_json(open('configs/archive_limits_smoke.json', 'rb').read()))
assert 1 <= len(job['includes']) <= 32
INPUT_MODE = 'dict'
DRIVE_SELECTION = job['selection']
SELECTED_MEMBER_GLOBS = job['includes']
PILOT_PURPOSE = 'discovery'
print({'job': JOB, 'selected_patterns': len(SELECTED_MEMBER_GLOBS)})
```

Plan은 관측한 Drive file ID와 기대 이름만 포함한다. Size/version과 다운로드 권한은 실행 시 다시 조회한다. 첫 실행이 끝나도 `JOB`을 바꿔 자동 반복하지 않는다. Notebook은 같은 `PILOT_JOB_ID`의 재실행을 차단한다. 후속 작업을 검토한 뒤에만 새 작업 셀의 `START_NEW_PILOT_JOB=True`와 `NEW_JOB_REASON`을 설정한다. 발견 결과에 따라 다음 선택과 예상 준비 비용을 검토한다. 중간 gzip 조각으로 점프하거나 이전 런타임의 해제 상태를 자동 복원하는 기능은 없다.

## 결과별 다음 행동

| 결과 | 다음 행동 |
| --- | --- |
| 완전 파일 0개 | 보고서를 보관하고 중단. 실제 HTTP 상태, 발견 경로, byte/time/request 제한을 검토한다. 같은 prefix 반복은 해결책이 아니다. |
| RGB만 일부 확보 | 실제 sequence/frame 목록과 hash를 보존하고 같은 프레임의 pose·반대쪽 RGB 후보를 만든다. Stereo/학습 완료로 취급하지 않는다. |
| 원본 변경·프로토콜 실패 | 추출본을 재사용하지 않는다. 원인을 확인하기 전에 자동 재시도하지 않는다. |
| 좌우 RGB+pose 확보 | 이름 대응, hash, 영상 크기·내용·재투영 overlay, timestamp·split을 검증한다. |

RGB와 pose의 archive 순서는 다르다. 첫 RGB discovery 성공도 대응 pose가 같은 예산 안에서 발견된다는 보장은 아니다. 필요한 파일이 먼 위치에 있다면 실제 측정 결과를 바탕으로 다음 준비 예산이나 인덱스·재사용 방식을 별도로 결정한다. 전체 다운로드·상한 확대는 기본 동작이 아니다.

독립 validation과 원본 motion 중복 검증은 별도로 남아 있다. 기존 train 18프레임을 나눠 validation처럼 사용하지 않는다. 현재 converter는 candidate만 만들며 학습 계약 연결, 실제 regression → body → hand 학습과 GPU 성능 검증은 아직 실행하지 않았다. M0/M1/M2/M3 전체 완료를 선언하지 않는다.
