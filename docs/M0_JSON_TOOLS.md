# M0 JSON 사전 도구

2026-10-13 첫 개발 묶음의 실행법과 검증 근거다. 모델 학습·평가, 좌표 의미 검증, M0 전체 완료를 포함하지 않는다.

## 코드와 실행

Python 3.10 이상과 표준 라이브러리만 필요하다. `rewind_data`는 프로젝트 root에서 import하거나 `python -m rewind_data`로 실행한다. 추가 패키지 설치가 필요하지 않다. Notebook에서는 같은 모듈을 import해 사용할 수 있으며 이번 묶음은 CLI 진입점을 제공한다.

| 파일 | 역할 |
| --- | --- |
| `rewind_data/audit.py` | 원본 로더, field별 유효 mask, sequence 감사, manifest |
| `rewind_data/skeleton.py` | 설정 파일의 명시적 관절 선택·연결 검증 |
| `rewind_data/viewer.py`, `viewer_template.html` | 정적 HTML 생성과 canvas viewer |
| `rewind_data/__main__.py` | 경로 인자를 받는 실행 진입점 |
| `configs/skeleton_unrealego2_dev.json` | 개발용 관절 이름·순서·연결과 미검증 근거 |
| `tests/test_data.py` | 합성 fixture의 정상·오류 경로 검증 |
| `tests/smoke_sample.py` | 실제 dataset sample의 보존·선택·mask 검사 |
| `tests/test_viewer.cjs` | 브라우저 없는 JavaScript DOM/canvas mock 검사 |
| `rewind_data/windows.py` | Field별 snapshot과 causal offline/streaming window 조립 |
| `rewind_data/window_fixture.py`, `tests/test_windows.py` | 데이터 없는 합성 window 실행·검증 |

프로젝트 root에서 실행한다. `python`/`node`는 각각 실제 실행파일을 선택해야 한다. Windows의 Python 설치 안내용 실행 별칭과 혼동하지 않는다. 이 세션은 Codex 번들 Python 3.12.14·Node.js 24.19.0을 사용했다.

```sh
python -m rewind_data --input data/raw/unrealego2_sample --skeleton configs/skeleton_unrealego2_dev.json --dataset-version UnrealEgo2_train_val_pose_unverified --provenance data/sample_source.json --output outputs/m0_sample
python -m unittest discover -s tests -v
python tests/smoke_sample.py --input data/raw/unrealego2_sample --skeleton configs/skeleton_unrealego2_dev.json --provenance data/sample_source.json --expected-frames 18 --expected-joints 87
node tests/test_viewer.cjs outputs/m0_sample/viewer.html outputs/m0_sample/canvas_commands.json
```

`--dataset-version`은 선언 label이다. 파일 이름·폴더 이름만으로 공식 데이터 버전이나 split을 확정하지 않는다. `--provenance`는 선택이며 생략하면 외부 출처와 data kind가 unknown이다. 산출물 디렉터리는 입력 root 밖이어야 한다. 같은 산출물 디렉터리에서 재실행하면 생성 파일을 갱신한다. 원본을 수정하지 않는다.

Exit code는 0(필수 수치·sequence 검사 오류 없음, 의미 검증은 별도), 1(데이터 감사 오류, 산출물은 생성), 2(설정·입출력 오류)다. Python API에서는 `audit_dataset(...).require_clean()`으로 오류가 있는 결과의 후속 사용을 차단할 수 있다.

## 데이터 준비와 출처

첫 검증에서 사용한 로컬 sample 경로는 아래와 같다. 이후 2026-10-13 사용자가 데이터셋 재다운로드를 알렸으므로 현재 가용성은 가정하지 않는다. 데이터 의존 명령은 sample과 provenance가 다시 준비된 뒤 실행한다. 원본과 산출물은 `.gitignore`의 `data/`, `outputs/` 아래에 보관한다. 새 checkout에는 포함되지 않으므로 사용자가 확보한 subset을 별도로 준비해야 한다. 자동 다운로드·전체 데이터 확장은 제공하지 않는다.

```text
data/raw/unrealego2_sample/
  Lab_Scene/Day/rp_sophia_rigged_003_ue4/
    SKM_MenReadingGlasses_Shape_01/001/Jump__3_/json/frame_0.json ... frame_17.json
data/sample_source.json
outputs/m0_sample/manifest.json
outputs/m0_sample/audit_report.json
outputs/m0_sample/viewer.html
```

출처는 [지정 Drive JSON 폴더](https://drive.google.com/drive/folders/16ON8vzZDJ2yligsM_OmkwEA2MjJPnohv)다. Provenance 파일은 입력 root에 대한 상대 경로를 키로 하는 `files` 사전과 선택적인 폴더 inventory를 담는다. 인증 정보·임시 서명 URL을 쓰지 않으며 canonical Drive URL만 허용한다.

```json
{
  "data_kind": "dataset_sample",
  "source_root_uri": "https://drive.google.com/drive/folders/FOLDER_ID",
  "files": {
    "sequenceA/json/frame_0.json": "https://drive.google.com/file/d/FILE_ID/view"
  },
  "observed_folder_inventory": {"sequenceA/json": [0]}
}
```

`dataset_sample`은 실제로 가져온 dataset 파일이라는 뜻이며 실세계 촬영이라는 뜻이 아니다. UnrealEgo2는 합성 데이터다. `synthetic_fixture`는 코드 검사에서 만든 인위적 fixture를 의미한다. `observed_folder_inventory`는 그때 관측한 다운로드 폴더 목록이며 원본 motion 전체 길이를 보장하지 않는다.

## 로더·관절·mask 계약

- `frame_<정수>.json`을 숫자 번호로 정렬한다. 입력 root에 상대적인 각 부모 폴더가 하나의 sequence다. 중첩 폴더를 합치거나 경계를 넘어 window를 만들지 않는다. 감사 CLI는 window/padding/시간 보간을 하지 않는다. 후속 window 모듈의 None padding 계약은 아래에 별도로 설명한다.
- 원본 `joints` 전체와 camera/device metadata를 `FrameSample.raw_document`에 보존한다. JSON 객체 내부 중복 key, JSON 오류, 예상하지 않은 파일 이름, `frame` 값과 파일 번호 불일치도 보고한다.
- 필수 field는 좌우 `camera_*_pts2d`(2개), 좌우 `camera_*_pts3d`(3개), `device_pts3d`(3개)다. 숫자 배열의 길이·유한값을 검사하며 bool/null/string/중첩 배열을 유효 숫자로 취급하지 않는다.
- 같은 sequence의 관측 관절 합집합 및 선택 설정의 관절을 기준으로 누락을 찾는다. 숫자 frame 중복은 파일을 삭제하거나 한쪽을 선택하지 않고 두 경로를 보고한다. 내부 frame gap은 inclusive 범위로 기록하며 임의의 처음/마지막 번호를 가정하지 않는다. Inventory가 있으면 로컬 범위와 비교한다.
- `field_valid_mask[name][field]`는 해당 field의 수치 유효성이다. `annotation_valid_mask[name]`는 모든 필수 field가 유효한지 나타낸다. `select(names, field_name)`은 요청 순서와 field별 mask를 반환한다. 예를 들어 2D가 잘못되어도 유효한 device 3D 자체를 누락 정답으로 바꾸지 않는다. Frame/sequence 감사 오류는 별도로 확인한다.
- 유효하지 않은 좌표는 반환값 `null`/Python `None`과 false mask로 표현한다. 0 좌표를 대신 넣지 않는다. Visibility/confidence는 좌우 모두 unknown(`None`)이다. 음수·화면 밖 2D 좌표의 수치 유효성을 visibility와 혼동하지 않는다. RGB 크기·projection 미검증 상태에서 화면 밖 visibility를 계산하지 않는다.
- 카메라 회전과 device 회전은 원본 표현을 유지한다. 회전 변환·단위 변환·정규화·물리 시간 추론을 하지 않는다. 관측한 timestamp 값이 있다면 원본 값만 manifest에 기록하고 의미는 unknown으로 남긴다.

개발용 선택은 body 22개, 왼손/오른손 각각 21개다. 손목이 body와 각 hand에 중복되므로 고유 선택은 62개이며 원본의 나머지 관절은 삭제하지 않는다. 이름은 실제 sample에서 확인했지만 선택·61개 연결은 이름과 해부학적 순서를 바탕으로 만든 **시각화 가정**이다. 공식 rig 부모 관계, REWIND 관절 집합, 평가 bone 목록이 아니다. 이 설정을 모델·평가 규격으로 채택하려면 별도 검증이 필요하다.

## 산출물과 viewer

`manifest.json`에는 파일별 상대 경로·canonical 출처·SHA-256·크기·sequence/frame·원본 관절 목록·field별 mask·선택 설정·설정 hash·실행 코드 파일 hash·데이터 선언·전처리 버전을 저장한다. `audit_report.json`에는 요약과 파일/관절/field 위치별 오류, 누락 범위를 저장한다. `numerically_valid_file_count`는 파일 단위 검사이며 중복·gap 등 sequence 오류는 전체 `error_count`와 별도로 확인해야 한다. 오류가 0이어도 단위·좌표·시간·calibration·split 등이 검증된 것은 아니다.

Viewer는 좌표·설정을 HTML 안에 포함해 외부 라이브러리/서버/CDN 없이 작성했다. Sequence 선택, frame slider, 이전/다음, 미리보기 재생, 관절 그룹, XY/XZ/YZ 보기, 드래그 회전, 관절 hover를 제공한다. 축과 scale은 **sequence의 전체 유효 원본 좌표 범위**로 고정하고 frame마다 바꾸지 않는다. 표시를 위한 projection만 적용하며 원본 좌표를 변환해 저장하지 않는다.

Invalid/missing 관절과 그 관절에 연결된 뼈는 그리지 않는다. 연결은 미검증을 나타내는 점선이다. 재생은 관측 frame 순서만 따르고 gap을 메우지 않으며 sequence 끝에서 정지한다. 100/200/500ms는 미리보기 간격이며 데이터 FPS·실제 timestamp가 아니다. Sequence를 바꾸면 재생을 멈추고 첫 frame으로 돌아간다.

## 데이터 없는 후속 window 묶음

2026-10-13 재다운로드 중 진행한 후속 묶음은 공개 자료 조사와 causal 데이터 window 조립에 한정했다. [데이터·window 계약](DATA_CONTRACT.md)에 자료 URL/revision, 16·72·raw 87 이력·dev 62 구분, 단위·회전·시간·calibration 검증 경계와 전체 API 정책을 기록했다. 이 조사로 원본 감사 도구의 `unknown` 상태나 개발용 skeleton을 검증 완료로 올리지 않았다.

```sh
python -m unittest discover -s tests -v
python -m rewind_data.window_fixture --window-size 3 --output outputs/synthetic_windows.json
```

합성 테스트 40개(기존 17 + window 23)가 통과했고 fixture는 offline/streaming이 같은 6 windows를 생성했다. 현재를 포함한 총 길이로 왼쪽 None padding을 하고, 실제 관측 slot과 `time_valid`로 구분한다. Gap은 reset, 중복/역순은 rejection이다. 새 sequence ID는 자동 reset하며 같은 ID의 별도 container/빈 clip 경계는 streaming 호출자도 명시적으로 reset해야 한다. 선택 joint/field snapshot은 원본과 cache의 일치를 확인하고 nested tuple로 격리한다. Annotation 누락은 해당 field의 None/false mask로 유지하며 구조 오류는 상태 변경 전에 거부한다.

이번 검증은 Drive·기존 raw·이전 실제 sample 산출물에 접근하지 않았다. 기존 sample smoke·viewer JS 검증을 재실행하지 않았으며 아래 표는 이전 묶음의 이력이다. 합성 fixture는 annotation 경로의 코드 계약만 검증하며 모델 정확도·projection·실제 FPS·실시간성 결과가 아니다.

## 첫 묶음의 2026-10-13 검증 이력과 제한

| 검사 | 실제 실행 결과 |
| --- | --- |
| 합성 fixture 단위/통합 검사 | 17 tests 통과. 정렬·다중 sequence·gap/중복·오류 JSON·shape/finite/type·관절/mask·원본 hash·HTML escape·CLI exit 검사 |
| 실제 dataset sample 감사 | 지정 한 폴더의 0~17번 18개, 각 87관절, 총 1,540,663 bytes. 필수 vector 7,830개 검사, error 0 / warning 0 |
| 실제 sample smoke | 관절 선택 순서·좌표·mask·unknown visibility·원본 SHA-256 보존 통과. 음수 2D 관측 109개 및 해당 GT 3D 유효성 유지 |
| JavaScript DOM/canvas mock | 초기 draw, slider/고정 범위, 평면 보기, 그룹 선택, sequence 끝 재생 정지, 합성 두 번째 sequence reset, 회전 draw의 7개 검사 통과 |
| Canvas 그리기 결과 | 실제 첫 frame의 mock draw 명령을 정적 PNG로 렌더링해 점/점선/축을 확인. 로컬 경로 `outputs/m0_sample/viewer_canvas_mock.png`; 브라우저 screenshot이 아님 |
| 실제 브라우저 화면·조작 | **미실행**. Browser Use의 URL 보안 정책이 `file://` 접근을 차단했으며 다른 경로나 브라우저로 우회하지 않음. CSS/실제 DOM 이벤트/브라우저 호환성은 미확인 |

샘플 한 sequence에 한정한 검사다. 공식 부모 관계, 단위/축/회전, 카메라 변환 방향, FPS/timestamp, 어안 calibration/재투영, RGB 정합성, train/validation/test 소속·원본 motion 중복, 다른 sequence와의 일관성은 남아 있다. Metadata 객체 존재만 확인하며 내부 pose/calibration의 수치·물리 타당성은 검증하지 않는다. Annotation visibility/confidence가 추후 발견되면 별도 schema와 의미 검증 후 연결해야 한다.

M0/M1 전체 완료, 성능 개선, 실제 front-end 입력 평가, 실시간성에 대한 근거로 사용할 수 없다. 모델·학습 구현, D1~D5 결정, commit/push/PR은 이번 묶음에 포함하지 않았다.
