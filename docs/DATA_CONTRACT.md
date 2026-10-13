# 데이터 규격 근거와 causal window 계약

2026-10-13 · 공개 1차 자료 조사 및 합성 코드 검증. 이번 묶음에서는 Drive, 기존 raw 데이터, 실제 sample 산출물에 접근하지 않았다. **자료의 설명**, **과거 파일 관측 이력**, **프로젝트 구현 가정**을 분리하며 runtime의 `unknown` 상태를 자료 조사만으로 올리지 않는다.

## 관절 집합을 구분하는 기준

| 범위 | 근거와 확인 수준 | 프로젝트 적용 |
| --- | --- | --- |
| 공식 body 평가 16개 | [S1] Evaluation 및 [S2] `predict_pose.py` 15~32행의 이름·순서 | 평가 후보 규격. REWIND whole-body 관절 규격으로 확정하지 않음 |
| 안내의 body 32 + hand 40 = 72개 | [S1] Introduction의 UnrealEgo/UnrealEgo2 annotation 설명 | 전체 72개 이름·부모 관계와 재다운로드 archive의 대응은 미확인 |
| raw 87개 | 이전 M0에서 한 폴더 18파일을 확인한 [검증 이력](M0_JSON_TOOLS.md) | 현재 재다운로드 데이터의 관절 수·일관성을 보장하지 않음 |
| dev 고유 62개 | [개발 설정](../configs/skeleton_unrealego2_dev.json): body 22, 좌우 hand 각 21, body와 손목 2개 중복 | 이름 기반 시각화 가정. 공식 skeleton/평가 bone 목록 아님 |

[S2]의 body 16개 순서는 다음과 같다. [S1] 웹 목록에는 `thigt_l/r` 표기가 있지만 저자 코드에는 `thigh_l/r`이다. 이 차이를 기록하며 raw 파일 key를 자동 치환하지 않는다.

```text
head, neck_01, upperarm_l, upperarm_r, lowerarm_l, lowerarm_r,
hand_l, hand_r, thigh_l, thigh_r, calf_l, calf_r,
foot_l, foot_r, ball_l, ball_r
```

72·87·62·16은 포함 범위와 목적이 다르다. 이름 매핑 없이 인덱스로 맞추거나, 차이 15개를 특정 관절 집합으로 단정하지 않는다. REWIND 논문은 손 annotation 활용을 설명하지만 [S4]에서 최종 body/hand 이름·순서·부모 목록은 확인하지 못했다.

## 공개 자료에서 확인한 설명과 남은 검증

| 항목 | 공개 근거 | archive/구현에서 남은 일 |
| --- | --- | --- |
| 좌표 field | [S1] Metadata: `trans`는 UE4 global XYZ, `rot` 배열 표기는 P/Y/R. Camera 3D는 OpenCV camera-relative, 2D는 image XY. `device_pts3d`는 stereo camera 중점 상대 GT | 실제 축 방향·회전 composition·변환 방향과 역변환 검사. P/Y/R 배열 표기를 Euler 적용 순서로 해석하지 않음 |
| Device 좌표 | [S3] README는 UE1 camera-left/right 3D의 평균으로 device-relative pose를 얻는 절차를 안내 | UE2 원본 field와 수치 대조. 서로 다른 orientation의 임의 camera 좌표를 평균하는 일반 변환으로 사용하지 않음 |
| 단위 | [S5] p.8 Fig.4의 UE1 pelvis-relative 분포는 cm 표기. [S1]의 공식 평가 지표는 mm로 보고됨 | UE2 raw `device_pts3d`·global/camera field의 단위는 미확인. cm→mm 변환을 자동 적용하지 않음 |
| 회전 | [S1]/[S3]의 `rot`는 P/Y/R 표기 | 각도 단위·handedness·rotation composition, camera/device `open3d_rot` 정의는 미확인. 같은 단위라는 가정 금지 |
| 시간 | [S5] p.8 §3.2는 UE1 capture 25 FPS를 명시. 같은 페이지의 2 frames/s는 렌더 처리 속도 | UE2 및 재다운로드 subset의 FPS·timestamp 의미·누락 시간은 미확인. Frame index를 초로 변환하지 않음 |
| Calibration | [S1]은 OpenCV/Scaramuzza 모델, UE1/2 동일 camera, 좌우 동일 intrinsics, `[width,height]` 순서를 안내. [S3] calibration JSON에는 size/center/양방향 polynomial/affine field가 존재 | 공식 UE2 calibration archive 및 resize/crop·projection 방향·실제 RGB 재투영 대조. 이번에는 ZIP 내용을 받거나 projection을 구현하지 않음 |
| Split | [S1]은 train/validation/test txt를 제공. [S3]은 motion 단위 txt 사용. [S4] p.6은 UE1+2에서 짧은 sequence 제외 후 val 200/test 300 sequence를 별도 선택 | 논문 split과 공식 split을 구분. 실제 txt·sequence·원본 motion 중복 확인 필요. 폴더 `train_val`만으로 train 소속 확정 불가 |

Split 수치도 버전 근거를 함께 보관한다. [S5] p.11에는 UE1 train 3,821 / val 494 / test 526 motions, [S3]의 현재 README에는 3,820 / 493 / 526이 기재돼 있다. 수치 차이를 임의로 정정하지 않고 실제 split 파일을 재확보해 확인한다. [S4] p.6의 당시 공식 UE2 test 손 annotation 부재 주석과 [S1]의 후속 test 공개 공지는 시점·대상이 다르므로 손 포함 여부를 archive로 다시 검사한다.

## M1 준비용 window의 프로젝트 계약

`rewind_data.windows.CausalWindowStream(window_size, joint_names, field_names)`는 `FrameSample`을 받는다. `offline_windows(sequences, ...)`는 `SequenceSamples`를 받아 같은 push 경로로 조립한다. Field 기본값은 기존 필수 2D/3D 5개이며 subset을 명시할 수 있다. 이름·field 순서는 호출자가 고정하고 sequence 전체의 관절 union을 window 입력으로 쓰지 않는다.

| 상황 | 선택한 구현 정책 |
| --- | --- |
| 길이·초기 과거 부족 | `window_size`는 현재를 포함한 총 slot 수. 왼쪽은 `None`, `time_valid=False`. 관측 frame 복제 없음 |
| 정상 진행 | 현재와 연속된 과거만 사용. 최대 길이 이후 오래된 snapshot 제거 |
| Frame gap | History를 비우고 현재부터 시작. Gap 크기만큼 가짜 frame/보간/시간을 만들지 않음 |
| 새 sequence ID | 자동 초기화. 이전 sequence 관절 혼입 없음 |
| 명시적 같은 ID의 새 container 또는 빈 clip | Offline은 `reset()` 적용. 동일 결과를 얻는 streaming 호출자는 이 경계에서 직접 `reset()`해야 함 |
| 중복·역순 | 자동 정렬/중복 제거 없이 `WindowContractError`. 거부된 push는 history를 바꾸지 않음 |
| 파싱/schema/frame 식별 오류 | 거부. Validity cache의 shape·bool·finite 값 및 선택 raw annotation과의 일치도 확인 후 상태 변경 |
| 누락 관절·invalid 선택 field | 관측 slot의 `time_valid=True` 유지. 해당 좌표만 `None`, field별 annotation mask는 false. Visibility와 혼동하지 않음 |
| 선택 밖 joint/field annotation issue | Window 값·진단·수용 여부에 반영하지 않음. 구조 오류는 별도 거부 |

Window의 `fields[field].coordinates`는 `[time][joint][coordinate]`, `valid_mask`는 `[time][joint]`이며 padding slot 전체는 `None`이다. Visibility/confidence의 관측 slot은 좌우 `[None] * joint_count`이다. Timestamp는 원본 scalar만 복사하고 `timestamp_semantics/fps/units=unknown`을 유지한다. Mask된 관절 누락과 시간 padding은 서로 다른 상태다.

`FrameSnapshot`과 `CausalWindow`는 frozen dataclass 및 nested tuple로 저장한다. `to_dict()`는 새 nested list를 만들어 반환한다. 원본 `FrameSample`이나 이전 반환 dict를 수정해도 내부 history와 다른 window는 바뀌지 않는다. 선택 annotation 진단은 현재 값/mask에서 새로 만들므로 감사 도구가 미래-only 관절의 union을 보고 과거에 추가한 `missing_joint` issue도 유입되지 않는다.

이 총 slot 정의는 [S4] p.5 식(4)의 `[j-w_s, j]`와 구분한다. 그 식은 과거 `w_s`개와 현재를 포함한다. p.13의 teacher/student window 설정 20/8을 본 API 총 길이로 그대로 치환하지 않았다. Attention cache, 모델, 학습, tensor 변환, 물리 velocity loss는 이번 범위에 없다.

## 합성 검증과 다음 진입 조건

```sh
python -m unittest discover -s tests -v
python -m rewind_data.window_fixture --window-size 3 --output outputs/synthetic_windows.json
```

Python 3.12.14에서 **40 tests 통과**: 기존 합성 17개 + window 23개. 미래 변경·미래-only 선택 밖 관절의 union·offline/streaming 전체 출력 일치·padding·gap·sequence/명시적 container reset·중복/역순·invalid annotation·stale cache 거부·snapshot 격리를 확인했다. Fixture 실행은 2개 합성 sequence의 6 windows를 생성하고 `data_kind=synthetic_fixture`, `offline_streaming_equal=true`로 표시한다. 데이터 다운로드나 실제 sample smoke는 실행하지 않았다.

M0/M1 전체 완료나 실데이터 정확도·latency·projection 정합성의 근거가 아니다. 다음 진입 조건은 재다운로드 subset의 manifest/hash·split·관절 매핑을 재검사하고 좌표·단위·시간·calibration을 실제 파일과 대조하는 것이다. GT 2D 진단 학습은 별도 지시와 train 소속/단위 확인 후 진행한다.

## 근거 자료

- **[S1]** [UnrealEgo Benchmark 공식 안내](https://unrealego.mpi-inf.mpg.de/), Introduction/UnrealEgo2 Metadata/Evaluation, 확인 2026-10-13. Live 페이지이며 이번에 데이터 archive나 split 파일을 다운로드하지 않음.
- **[S2]** [CVPR2024 저자 inference 코드의 관절 목록](https://github.com/hiroyasuakada/3D-Human-Pose-Perception-from-Egocentric-Stereo-Videos/blob/5d79459da71d8382f8580d2e7c337dc30054b3fc/predict_pose.py#L15), revision `5d79459da71d8382f8580d2e7c337dc30054b3fc`. 같은 revision의 `dataloader/image_folder.py`는 첫 frame 복제 padding을 사용하지만 본 프로젝트 계약은 None padding을 선택함.
- **[S3]** [UE1 저자 README](https://github.com/hiroyasuakada/UnrealEgo/blob/9f1f57b5202a82f83d42dddab174b5a627793be5/README.md), [calibration metadata](https://github.com/hiroyasuakada/UnrealEgo/blob/9f1f57b5202a82f83d42dddab174b5a627793be5/utils/fisheye_calibration_UEP.json), revision `9f1f57b5202a82f83d42dddab174b5a627793be5`. 공개 텍스트만 조사했으며 코드를 실행·복사해 채택하지 않음.
- **[S4]** [로컬 REWIND 논문](REWIND.pdf), PDF p.5 식(4), p.6 §4.1·각주, p.12~13 §S.3. SHA-256 `f9529af04e2efcf3f7ec1b197aca84dec82be65bf44caecbd777fd9ce3993aff`; arXiv v2, 2025-04-08. [공식 프로젝트 페이지](https://jyunlee.github.io/projects/rewind/)도 확인했으나 공식 구현 코드 링크는 찾지 못함.
- **[S5]** [Akada et al., ECCV2022 UnrealEgo 논문](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136660001.pdf), PDF p.8 Fig.4/§3.2, p.10 §4.2, p.11 §5.1. 자료의 UE1 서술을 UE2 raw schema에 자동 적용하지 않음.
