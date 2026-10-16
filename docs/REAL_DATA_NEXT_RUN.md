# 다음 실제 데이터 준비 실행

2026-10-16 현재 Colab 런타임은 없다. 로컬 변환·검증과 Drive 목록 확인은 진행했지만 대응 RGB와 독립 validation 확보, 실제 regression → body → hand 학습은 아직 실행하지 않았다. [변환기 안내](REAL_DATA_PREPARATION.md)와 함께 사용한다.

## 현재 조사 결과

- Drive 프로젝트 reports 폴더에는 기존 32 pose ZIP이 있으며 새 RGB sample은 확인하지 못했다. ArchVis train_val RGB는 `000`~`004` 5조각, pose는 `000` 1조각의 목록을 확인했다. 이는 내부 파일·무결성 검증이 아니다.
- 두 pose sequence는 공식 train에 속한다. Brutal_Assassination은 0~17 연속, Opening은 산재된 14프레임이다. 이를 train/validation으로 나누지 않는다.
- 공식 서버 ArchVis pose에서 공식 validation 30개 경로만 선택하는 별도 시험을 실행했다. 2,498,141 bytes·60.047초·331 tar members 후 `http_body_failed`, 완전 선택 파일 0개, 재사용 불가로 끝났다. 이는 사용자 Drive 실행 결과가 아니다. 원본 파일 손상으로 단정할 근거도 아니다.
- 기존 상한을 유지했다. 기본 media 예산은 실행별 8MiB/60초/32파일이며 metadata는 별도 128KiB/30초다. 협력적 deadline이므로 실제 elapsed가 약간 초과할 수 있다. 자동 재시도·상한 확대·전체 다운로드는 하지 않았다.

## Colab에서 이어 실행

CPU 런타임이면 충분하다. [기존 준비 notebook](../notebooks/archive_prepare.ipynb)의 환경 → 최신 소스 ZIP → 테스트·합성 진단까지 실행한다. 소스 ZIP 생성·업로드 방법은 [Colab 안내](COLAB_SETUP.md)를 따른다.

로컬 산출물 `outputs/real_data_colab_plan_20261016.json`을 준비한다. 이 파일은 현재 확인한 Drive file ID와 선택 glob, 기본 상한만 담는다. 인증 정보나 고정된 size/version은 없다. `train_rgb`는 기존 연속 18 pose에 대응하는 좌우 36개 PNG 이름, `validation_pose`는 공식 validation에 속한 ArchVis 30개 sequence 경로를 선택한다. Archive 내부 위치·완전 clip 확보는 보장하지 않는다. 기본 파일 상한32로 좌우36개 전부를 얻을 수도 없으므로 보고서에서 실제 완전 pair를 확인해야 한다.

1. 기존 notebook의 **선택 입력 셀**을 실행한 직후, metadata 셀 **앞에** 새 코드 셀을 넣어 아래를 실행한다. 업로드 창에서 위 plan JSON 하나를 고른다.

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
assert plan['schema_version'] == 'real_data_next_run_plan_v1'
JOB = 'train_rgb'  # 별도 두 번째 실행에서는 'validation_pose'
job = plan['jobs'][JOB]
validate_selection(job['selection'])
assert Limits.from_dict(plan['limits']) == Limits.from_dict(
    read_json(open('configs/archive_limits_smoke.json', 'rb').read()))
INPUT_MODE = 'dict'
DRIVE_SELECTION = job['selection']
SELECTED_MEMBER_GLOBS = job['includes']
print({'job': JOB, 'selected_patterns': len(SELECTED_MEMBER_GLOBS)})
```

2. **metadata 셀**의 `RUN_DRIVE_METADATA = True`를 활성화하고 실행한다. Colab 계정 인증을 완료한다. metadata가 현재 size/version과 다운로드 가능 여부를 다시 확인한다. `manifest_ready=True`일 때만 진행한다.
3. **pilot 셀**의 `RUN_DRIVE_PILOT = True`를 활성화하고 실행한다. 기존 limits 셀은 변경하지 않는다. 성공 여부·완전 파일 수·중단 이유·전송량과 `inspection_candidates_usable`을 확인한다. 파일 0개 또는 실패라면 같은 작업을 자동 재실행하거나 상한을 올리지 않는다.
4. `PILOT_RUN_ROOT` 전체를 이름이 다른 결과 폴더에 보관한다. `metadata/`와 `pilot/` 아래 보고서·manifest·완전 추출 파일을 함께 보관해야 이후 hash 대조가 가능하다. notebook 저장만으로 `/content`의 추출본이 보관되지 않는다. 기존 [결과 보관 안내](COLAB_SETUP.md#4-drive-연결과-저장-위치)를 따른다.
5. 첫 결과를 보관한 뒤 validation 탐색을 할 경우, **선택 입력 셀부터 다시 실행**하여 이전 metadata 상태를 초기화한다. 새 셀의 `JOB='validation_pose'`로 바꾸고 1~4를 반복한다. 두 작업은 각각 별도 metadata/pilot 예산을 사용하며 총 비용은 합산한다. 원본은 매번 앞에서부터 다시 읽는다.

위 셀의 plan·selection·limits 연결과 코드 문법은 로컬에서 검사했다. Colab 인증·실제 Drive 읽기 재실행은 아직 하지 않았다. Drive 파일이 바뀌거나 권한이 없으면 검사 결과에 따라 중단한다.

## 결과를 확보한 뒤 할 검증

- RGB는 기존 pose와 sequence/frame 번호로 매칭하고 실제 좌우 영상의 크기·내용·재투영 overlay를 확인한다. 단순 이름 일치만으로 정합성 verified를 설정하지 않는다.
- Validation pose는 공식 split과 raw hash를 확인하고 원본 motion 식별 근거를 확보해 train 중복을 검사한다. 경로의 motion 이름만으로 원본 motion ID가 같거나 다르다고 확정하지 않는다.
- 연속 구간만 causal window로 묶고 gap에서는 history를 초기화한다. sample에 없는 timestamp를 25FPS로 만들어 넣지 않는다.
- 위 조건을 검토한 후 candidate를 학습 계약으로 내보내는 연결을 구현하고 작은 GT2D regression → body → hand를 실행한다. 현재 converter는 candidate만 만들며 training release는 제공하지 않는다. GT 입력 진단과 실제 예측 2D 입력 결과를 분리한다.

현재 M0/M1/M2/M3 실제 데이터 완료를 선언하지 않는다. 추가 준비 결과에 따라 다음 읽기 예산과 학습 범위를 결정한다.
