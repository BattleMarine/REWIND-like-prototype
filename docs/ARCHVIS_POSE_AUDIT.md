# ArchVis pose sample 감사 기록

2026-10-15 · 사용자 Colab 추출본을 Drive에서 받아 기존 M0 도구로 검사했다. 디벨로퍼 실행 결과와 오케스트레이션의 독립 hash·구조·좌표 수치 검토를 구분한다. M0/M1 전체 완료나 학습 성능 검증은 아니다.

## 출처와 무결성 범위

- 출처: [사용자 Drive sample ZIP](https://drive.google.com/file/d/1IU2BTyQMDfIL_wWJwgiNV01Itl_b4fXr/view), `REWIND-like prototype/reports/rewind_pose_sample_2610151012.zip`.
- ZIP 크기 533,925 bytes, SHA-256 `02244d3e7dfe716f0cd5ae4a6c4e771e4b5ffee7037b33673493b84fe54ba035`.
- 안전한 경로·파일 유형·개수·해제 크기 상한을 확인하고 ZIP CRC 검사를 통과했다. 49 entries 중 regular file 36개는 pose JSON 32개와 보고서/manifest 4개다.
- JSON 32개 모두 extraction manifest의 크기·SHA-256과 일치한다. 기존 제공된 metadata/pilot/extraction 보고서 3개도 byte 단위로 동일하다. 검사 전후 원본 bytes를 보존했다.
- ZIP 컨테이너의 canonical URI를 출처로 기록하고 member 경로와 개별 hash를 연결했다. 개별 JSON에 독립 Drive 파일 URI를 만들지 않았다.
- 이는 전달받은 sample의 일치 검사다. 원본 `ArchVisInterior_ArchVis_RT.tar.gz000` 전체 gzip CRC·tar 종료·완전성은 검증하지 않았다.

## 기존 도구의 실제 sample 실행

Python 3.12.14, 실행 revision `8bd5ab358acb6fb21d5c159e6bfcab87d7e09ab6`에서 기존 감사 CLI·`tests/smoke_sample.py`·window API를 실행했다. 코드와 skeleton 설정은 변경하지 않았다.

| 범위 | 결과 | 사용 제한 |
| --- | --- | --- |
| 전체 32 JSON / 2 sequence | 모두 파싱·수치 유효. 각 87관절, 필수 배열 13,920개 검사 | Opening의 `missing_frames` 1건으로 전체 CLI exit 1. 전체 clean 통과로 보고하지 않음 |
| `025/Brutal_Assassination` | frame 0~17, 감사 오류 0, CLI exit 0, 실제 smoke 통과 | 번호가 연속된 18프레임 subset. 원래 sequence 전체나 물리 시간 검증은 아님 |
| `017/Opening` | 5,32,36,50,54,61,65,88,90,94,113,117,122,126의 14개 | 의도된 소량 추출의 gap. 연속 motion 학습에서 제외하며 누락 진단을 유지 |
| Causal window | 길이 3 / dev 고유 62관절 / 32 windows, offline·streaming 전체 결과 일치 | Gap 13회와 sequence 변경 1회에서 history reset. 모델 학습·FPS 검증은 아님 |

Window는 현재·과거와 같은 sequence만 포함하고, 원본 좌표·mask를 보존했다. 미래 frame의 메모리 내 좌표 변경이 이전 windows에 영향을 주지 않았고 수동 reset 후 `[None, None, 0]`을 확인했다. Opening을 window API에 넣은 것은 gap 동작 검사 목적이며 clean 학습 gate를 통과시킨 것이 아니다.

## 독립 구조·수치 대조

- 32개 모두 동일한 87관절 이름을 가지며 내부 `frame`과 파일 이름이 일치한다. JSON 전체 수치에서 NaN/Inf를 발견하지 않았다.
- 좌표 2개 중 하나 이상 음수인 관절–카메라 2D 관측은 전체 192개, 연속 18프레임에서는 108개다. 수치 유효 annotation으로 유지하며 visibility나 정답 누락 mask로 바꾸지 않는다.
- 32 × 87 = 2,784개 관절에서 `device_pts3d = (camera_left_pts3d + camera_right_pts3d) / 2`의 성분별 최대 절대 잔차는 약 `5.684e-14` raw units다.
- 각 카메라의 `device_pts3d` offset을 해당 관절 camera 3D에 더한 값과 joint device 3D도 같은 최대 잔차 이내에서 일치한다. 이 sample의 수치 관계이며 임의 stereo rig의 일반 좌표 변환으로 적용하지 않는다.
- `camera_left.trans`와 `camera_right.trans` 거리 범위는 약 11.998898~12.000840 raw units다. 이 숫자만으로 cm·mm를 결정하지 않는다.
- 명시적 timestamp·FPS·visibility·confidence key는 관측되지 않았다. `device`에는 `open3d_trans`, `open3d_rot`가 있으며 camera/global `rot`와 같은 단위라고 가정하지 않는다.

## 산출물과 다음 단계

2026-10-23 정리 후 아래 `outputs/...` 증거·재현 스크립트는 로컬 아카이브로 이동했다. [개별 복원 안내](LOCAL_STORAGE.md#개별-파일-복원)에서 인덱스를 확인하고 필요한 스크립트와 입력 근거를 원래 경로로 복원한 후 재현한다. `data/raw/...` 원본은 보존하며 이 문서 갱신으로 스크립트를 실행하지 않는다.

Git 제외 로컬 보관 위치는 다음과 같다. 새 checkout에는 자동으로 포함되지 않으므로 재검사에는 위 Drive ZIP을 사용한다.

- `data/raw/colab_archvis_pose_20261015/`: ZIP과 `bundle/pilot/files`의 원본 JSON.
- `outputs/archvis_pose_audit_20261015/bundle_review.json`: ZIP·payload·기존 보고서 일치.
- 같은 폴더 `root_numeric_review.py`, `root_numeric_review.json`: 독립 수치 검사 방법과 결과. 프로젝트 root에서 `python -c "import runpy; runpy.run_path('outputs/archvis_pose_audit_20261015/root_numeric_review.py', run_name='__main__')"`로 실행했다.
- `developer/`: 실행 script, 명령·종료 코드, provenance, source linkage, 감사 분류, smoke·window 결과. `all32`, `brutal18`, `opening14` 각각 manifest·audit report·viewer HTML을 생성했다. 실제 브라우저 표시·조작은 이번에 검사하지 않았다.

관절 선택 62개·연결 61개는 기존 개발용 미검증 설정이다. 단위·축·회전 composition·변환 방향·calibration/재투영·FPS/timestamp·공식 split/원본 motion·RGB 대응·데이터 버전은 계속 미검증이다. `train_ready=false`와 M0/M1 미완료 상태를 유지한다.

다음 작업 후보는 이 18프레임과 대응하는 좌우 RGB·calibration·공식 split 근거를 제한된 예산으로 확보하고 관절/단위·좌표·시간 규격을 검증하는 것이다. 이후 train 소속과 입력 계약이 확인되면 GT 2D 진단 학습으로 연결한다. RGB 예측 입력 baseline과 GPU 검증은 별도이며 이번 감사에서 학습이나 데이터 규모 확대는 실행하지 않았다.

## 후속 공식 근거·RGB 준비 조사 (2026-10-15)

- [공식 UnrealEgo2 split ZIP](https://unrealego.mpi-inf.mpg.de/data/data_split/UnrealEgoData2.zip)의 SHA-256은 `40160ff2c2192b5c342a80a1f76b37ea651afc45d1fd99fdc778dd4f18b81171`이다. Opening과 Brutal_Assassination의 전체 sequence 경로가 train.txt 14행·23행에 각각 한 번 포함되고 validation/test에는 없다. 오케스트레이션도 독립 대조했다. 따라서 해당 경로의 공식 train 소속은 확인했지만 원본 motion 단위 독립성·전체 archive 버전 일치는 별도다. 기존 원본 report의 unknown은 소급 변경하지 않는다.
- [공식 calibration ZIP](https://unrealego.mpi-inf.mpg.de/data/fisheye_calib_unrealego.zip)의 SHA-256은 `0163a86866827dcd8e4fc5bba76f9812fe919cebed1941ce1fa95c8ba2102345`다. Scaramuzza JSON의 W2C polynomial·affine·center를 그대로 사용하고 `theta=atan2(-z,hypot(x,y))`로 camera 3D를 투영하면 5,568관측의 2D annotation과 평균 `1.90656e-5 px`, 최대 `7.46509e-5 px` 차이로 일치한다. 고정된 축/부호 후보 16개를 비교했으며 계수를 학습하거나 fitting하지 않았다. 독립 계산으로 동일 결과를 확인했다.
- 위 값은 **저장된 3D와 2D annotation 사이의 정합성**이다. 모델 예측 정확도, 실제 RGB overlay, 물리 단위·FPS·world extrinsics의 검증이 아니다. 음수 2D 192개와 z≤0 관측 704개도 유지했다. OpenCV 모델의 z>0 한정 비교는 평균 0.81146px/최대 7.77749px였으며 나머지 704개는 해당 경로에서 평가하지 않았다.
- 숫자 suffix를 제거한 motion 이름 후보가 다른 split에도 존재한다(Brutal train/validation/test 12/0/1, Opening 109/20/18). 이름만으로 같은 원본 mocap이라고 판정하거나 split 누수가 없다고 선언하지 않는다.
- Drive RGB 목록에 ArchVis `000`~`004` 5개가 있으며 공식 다운로드 스크립트의 part 수와 일치한다. 총 목록 크기는 52,441,303,081 bytes다. 내용·hash·전체 무결성 검사가 아니다. Drive media는 이번에 읽지 않았다.
- 별도의 공식 서버 RGB 부분 읽기는 기본 8MiB/60초/16MiB 저장 상한으로 실행했다. 사용자 중단 전 실행은 최종 보고서 없이 partial 983,040 bytes를 남겨 재사용하지 않는다(전송량 미확정). 재개 실행은 새 폴더에서 2,301,530 bytes/60.031초 뒤 `http_body_failed`로 종료했고 `inspection_candidates_usable=false`, verify `not_usable`이다. 완전 파일 기록 1개도 다른 인물의 Getting_Up__1_/camera_right/final_10.png여서 목표 pose와 대응하지 않는다. 실패 추출본은 학습에 쓰지 않고 예산을 자동 확대하지 않았다. 이 결과를 Drive/Colab 전송 성능이나 원본 손상 증거로 해석하지 않는다.
- Git 제외 `outputs/m0_next_evidence_20261015/`에 공식 파일·수신 log·split/재투영 결과·독립 검토·RGB 실패 보고서를 보존한다. `root/archvis_rgb_selection.json`은 당시 Colab notebook의 `INPUT_MODE=json_upload`용 개인 선택 입력이었다. 이 종료된 입력 방식은 현재 API9 단일 JOB notebook에 적용하지 않으며 최신 [실행 안내](REAL_DATA_NEXT_RUN.md)를 따른다. `root/colab_target.json`의 glob은 목표 sequence frame 0의 좌우 PNG만 선택한다. 형식과 이름 계약만 검사했으며 Drive RGB 실행은 미완료다. 상한 내 결과가 0이어도 원본에 해당 frame이 없다는 뜻은 아니다.

M0에서 남은 실제 RGB·좌표/단위·시간·skeleton·원본 motion 검증을 유지한다. 후속 M2/M3의 합성 구현 검증은 이 데이터의 학습 준비 완료와 구분한다.

## 후속 관절 이름 대응 (2026-10-16)

[관절 대응표](JOINT_MAPPING.md)에서 본 sample의 32개 JSON에 공통인 87개 이름을 전부 기록했다. 기존 개발용 고유 62개는 모두 존재하며 나머지 25개도 표에 보존한다. 부모 연결·해부학적 의미·최종 모델 subset의 검증과 이름 존재 대조를 구분한다. 원본 좌표·개발용 설정·학습 입력 계약은 변경하지 않았다.

## 후속 단위·좌표 변환 검토 (2026-10-16)

[별도 검토 문서](COORDINATE_AND_TRAINING_REVIEW.md)에 world↔camera/device 변환식과 회전 표현, 물리 단위 근거, 학습 관절 권고를 기록했다. 오케스트레이션은 32파일의 camera/device forward·inverse와 거리 보존을 독립 계산해 최대 약8.53e-14 raw units 이내의 수치 일치를 확인했다. Camera open3d Euler는 degree, device open3d Euler는 그 radian 변환과 일치한다. UE world→camera에는 반사가 포함되어 proper rotation만 허용하는 M2 계약에 그대로 넣을 수 없으며, right-handed world를 명시하는 변환이 필요하다.

이는 저장된 annotation의 내부 정합성 검증이다. cm는 공식 rig와 baseline의 강한 간접 근거이며 UE2 필드의 직접 단위 선언·RGB overlay·시간·전체 데이터 검증과 구분한다. 기존 원본/보고서/config와 `train_ready=false`는 유지했다.
