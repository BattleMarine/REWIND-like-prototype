# 실제 ArchVis JSON 관절 대응표

2026-10-16. 확보한 **32개 실제 JSON의 raw 87개 이름을 전부** 대조했다. 기존 개발용 62개와 공식 body 평가 16개에 대한 이름의 포함 관계는 확인했으며, body/upper/wrist/finger 역할은 아래에 검토 후보로 구분한다. 이는 실제 데이터의 이름·배열 감사와 문서 대응표이며 **모델 설정 확정·변환기 구현·학습 준비 완료가 아니다.**

## 근거와 확인 범위

같은 날 후속 [단위·좌표계 및 학습 관절 검토](COORDINATE_AND_TRAINING_REVIEW.md)에서 이 이름 목록의 body22+finger40/upper13을 초기 학습 구성으로 권고했다. 기존 M3 역할 계약의 이름·소유 구조 검사와 실제 좌표 변환을 대조했으며, 해부학적 parent·공식72 목록을 확정한 것은 아니다. 아래 표의 raw/Dev/Eval 순서와 기존 설정은 유지한다.

| 구분 | 근거 | 이 문서에서의 취급 |
| --- | --- | --- |
| 실제 관측 | `data/raw/colab_archvis_pose_20261015/bundle/pilot/files/`의 32 JSON | 이름·직렬화 순서·필드·shape·finite·파일 hash를 이번에 직접 확인 |
| 개발용 선택 | [skeleton_unrealego2_dev.json](../configs/skeleton_unrealego2_dev.json) | 기존 미검증 body22 + 좌우 hand21씩. 선택·연결의 공식성은 확인하지 않음 |
| 공식 평가16 이름/순서 | [기존 DATA_CONTRACT의 S2](DATA_CONTRACT.md), 저자 [pinned predict_pose.py](https://github.com/hiroyasuakada/3D-Human-Pose-Perception-from-Egocentric-Stereo-Videos/blob/5d79459da71d8382f8580d2e7c337dc30054b3fc/predict_pose.py#L15) | 2026-10-13 revision 검토 기록의 목록을 재사용. 오늘 저자 코드를 다시 확인했다고 주장하지 않음 |
| 공개 개수 | [UnrealEgo Benchmark 공식 안내](https://unrealego.mpi-inf.mpg.de/) | 오케스트레이션이 2026-10-16 live 페이지의 body32+hand40=72 및 body 평가16 설명을 재확인. Raw87 목록과 근거를 분리 |
| 역할·해부학 해석 | 이름의 `spine`, `clavicle`, `hand`, `*_end`, `*_twist_*`, `l/r` 등 | 이름 기반 추정. 위치 중심·해부학적 landmark·rig parent·회전 의미를 검증한 것이 아님 |

Raw joint마다 관측한 field는 `trans`, `rot`, `camera_left_pts3d`, `camera_left_pts2d`, `camera_right_pts3d`, `camera_right_pts2d`, `device_pts3d`다. 검사한 joint 항목에 parent·계층 정의 field는 없었다. 따라서 이름 번호나 JSON에 나타난 순서를 bone parent 관계로 해석하지 않는다. 기존 설정의 61 edges도 시각화용 미검증 연결이다.

주대상은 같은 sequence의 `025/Brutal_Assassination` frame0~17, 보조는 `017/Opening`의 14개 산재 frame이다. 둘 모두 아래 공통 경로에 있고 같은 인물/메시 조합의 제한된 sample이다.

```text
ArchVisInterior_ArchVis_RT/Day/rp_adanna_rigged_003_ue4/
SKM_MenReadingGlasses_Shape_01/
```

| 검사 대상 | 프레임 | 관측 관절 수 | 사용 한계 |
| --- | --- | ---: | --- |
| Brutal_Assassination | 0~17, 18개 | 매 파일87 | 번호 연속성만 관측. 전체 sequence·물리 시간·실제 학습 검증 아님 |
| Opening | 5,32,36,50,54,61,65,88,90,94,113,117,122,126, 14개 | 매 파일87 | Gap을 유지하는 보조 이름/필드 대조. 연속 clip으로 취급하지 않음 |

32개 모두 **이름 집합과 직렬화 순서가 동일**하다. 각 파일의 크기·SHA-256도 기존 extraction manifest의 32개 완전 파일 기록과 일치했다. 합계 2,735,012bytes이며 전체 archive 무결성·다른 인물/sequence에 대한 일반화는 검증하지 않았다. 전달 ZIP·공식 split·calibration과 기존 검증의 범위는 [ArchVis 감사 기록](ARCHVIS_POSE_AUDIT.md)을 따른다.

## 표 읽기와 순서 정책

- `Raw #`: Brutal_Assassination `frame_0.json`의 `joints` key가 저장된 순서를 0부터 센 비교 번호다. 공식 joint ID나 모델 index가 아니다. 기준 파일 SHA-256은 `3fd55791ab0a06fd17cd09f3b599ccba34d710bed8f00802aaacb180c09afb41`이다.
- `Dev62 #`: 기존 config의 groups 순서 `body → left_hand → right_hand`를 연결하고 이름 첫 등장만 남긴 **고유 순서**의 0-based index다. `body,left_hand` 등의 표시는 두 그룹에 포함됨을 뜻하며 같은 joint를 두 번 저장한다는 뜻이 아니다.
- `Eval16 #`: 위 pinned 저자 코드 목록의 0-based 평가 순서다. `—`는 해당 16개 목록에 없음을 뜻하며 annotation 무효나 모든 평가에서의 제외를 뜻하지 않는다.
- 역할 후보 `B`는 기존 dev body22, `U`는 문서에서 제안하는 upper13, `WL/WR`는 `hand_l/r` 손목 기준 후보, `FL/FR`는 좌우 finger20 후보다. **모두 모델 역할의 검토 가정**이며 공식 REWIND/UnrealEgo skeleton 선택 확정이 아니다. `미선택`/`제외` 관절도 원본에는 유효하게 보존한다.

모델·평가는 이름으로 gather하고 각자 명시한 순서를 저장해야 한다. 예를 들어 raw의 `head`는5번이나 Eval16에서는0번이고, `hand_l`은 raw24/Dev9/Eval6이다. Raw array를 앞에서16개/62개 자르거나 현재 dict 순서를 모델 순서로 사용하는 방식은 대응표와 일치하지 않는다.

## Raw 87개 전체 대응표

아래 87행의 raw key·포함 index는 관측·기존 목록 대조 결과다. 한글 해석과 역할 후보는 그와 분리된 이름 기반 추정/선택 가정이다. 각 이름은 32개 모두에서 관측되었다.

| Raw # | Raw key (관측) | 이름 해석 (추정) | 역할 후보 (가정) | Dev62 # / 포함 그룹 | Eval16 # |
| ---: | --- | --- | --- | --- | ---: |
| 0 | `pelvis` | 골반 이름 | B | 0 / body | — |
| 1 | `spine_01` | 척추 01 이름 | B,U | 1 / body | — |
| 2 | `spine_02` | 척추 02 이름 | B,U | 2 / body | — |
| 3 | `spine_03` | 척추 03 이름 | B,U | 3 / body | — |
| 4 | `neck_01` | 목 01 이름 | B,U | 4 / body | 1 |
| 5 | `head` | 머리 이름 | B,U | 5 / body | 0 |
| 6 | `head_end` | 머리 끝 이름 | 미선택 | 제외 | — |
| 7 | `jaw` | 턱 이름 | 미선택 | 제외 | — |
| 8 | `jaw_end` | 턱 끝 이름 | 미선택 | 제외 | — |
| 9 | `eye_l` | 왼쪽 눈 이름 | 미선택 | 제외 | — |
| 10 | `eye_end_l` | 왼쪽 눈 끝 이름 | 미선택 | 제외 | — |
| 11 | `eye_r` | 오른쪽 눈 이름 | 미선택 | 제외 | — |
| 12 | `eye_end_r` | 오른쪽 눈 끝 이름 | 미선택 | 제외 | — |
| 13 | `mouth_l` | 왼쪽 입 이름 | 미선택 | 제외 | — |
| 14 | `mouth_r` | 오른쪽 입 이름 | 미선택 | 제외 | — |
| 15 | `eyelid_l` | 왼쪽 눈꺼풀 이름 | 미선택 | 제외 | — |
| 16 | `eyelid_end_l` | 왼쪽 눈꺼풀 끝 이름 | 미선택 | 제외 | — |
| 17 | `eyelid_r` | 오른쪽 눈꺼풀 이름 | 미선택 | 제외 | — |
| 18 | `eyelid_end_r` | 오른쪽 눈꺼풀 끝 이름 | 미선택 | 제외 | — |
| 19 | `eyebrow_l` | 왼쪽 눈썹 이름 | 미선택 | 제외 | — |
| 20 | `eyebrow_r` | 오른쪽 눈썹 이름 | 미선택 | 제외 | — |
| 21 | `clavicle_l` | 왼쪽 쇄골 이름 | B,U | 6 / body | — |
| 22 | `upperarm_l` | 왼쪽 위팔 이름 | B,U | 7 / body | 2 |
| 23 | `lowerarm_l` | 왼쪽 아래팔 이름 | B,U | 8 / body | 4 |
| 24 | `hand_l` | 왼쪽 손/손목 기준 후보 이름 | B,U,WL | 9 / body,left_hand | 6 |
| 25 | `thumb_01_l` | 왼쪽 엄지 01 이름 | FL | 22 / left_hand | — |
| 26 | `thumb_02_l` | 왼쪽 엄지 02 이름 | FL | 23 / left_hand | — |
| 27 | `thumb_03_l` | 왼쪽 엄지 03 이름 | FL | 24 / left_hand | — |
| 28 | `thumb_end_l` | 왼쪽 엄지 끝(end) 이름 | FL | 25 / left_hand | — |
| 29 | `index_01_l` | 왼쪽 검지 01 이름 | FL | 26 / left_hand | — |
| 30 | `index_02_l` | 왼쪽 검지 02 이름 | FL | 27 / left_hand | — |
| 31 | `index_03_l` | 왼쪽 검지 03 이름 | FL | 28 / left_hand | — |
| 32 | `index_end_l` | 왼쪽 검지 끝(end) 이름 | FL | 29 / left_hand | — |
| 33 | `middle_01_l` | 왼쪽 중지 01 이름 | FL | 30 / left_hand | — |
| 34 | `middle_02_l` | 왼쪽 중지 02 이름 | FL | 31 / left_hand | — |
| 35 | `middle_03_l` | 왼쪽 중지 03 이름 | FL | 32 / left_hand | — |
| 36 | `middle_end_l` | 왼쪽 중지 끝(end) 이름 | FL | 33 / left_hand | — |
| 37 | `ring_01_l` | 왼쪽 약지 01 이름 | FL | 34 / left_hand | — |
| 38 | `ring_02_l` | 왼쪽 약지 02 이름 | FL | 35 / left_hand | — |
| 39 | `ring_03_l` | 왼쪽 약지 03 이름 | FL | 36 / left_hand | — |
| 40 | `ring_end_l` | 왼쪽 약지 끝(end) 이름 | FL | 37 / left_hand | — |
| 41 | `pinky_01_l` | 왼쪽 소지 01 이름 | FL | 38 / left_hand | — |
| 42 | `pinky_02_l` | 왼쪽 소지 02 이름 | FL | 39 / left_hand | — |
| 43 | `pinky_03_l` | 왼쪽 소지 03 이름 | FL | 40 / left_hand | — |
| 44 | `pinky_end_l` | 왼쪽 소지 끝(end) 이름 | FL | 41 / left_hand | — |
| 45 | `lowerarm_twist_01_l` | 왼쪽 아래팔 twist 01 이름 | 미선택 | 제외 | — |
| 46 | `upperarm_twist_01_l` | 왼쪽 위팔 twist 01 이름 | 미선택 | 제외 | — |
| 47 | `clavicle_r` | 오른쪽 쇄골 이름 | B,U | 10 / body | — |
| 48 | `upperarm_r` | 오른쪽 위팔 이름 | B,U | 11 / body | 3 |
| 49 | `lowerarm_r` | 오른쪽 아래팔 이름 | B,U | 12 / body | 5 |
| 50 | `hand_r` | 오른쪽 손/손목 기준 후보 이름 | B,U,WR | 13 / body,right_hand | 7 |
| 51 | `thumb_01_r` | 오른쪽 엄지 01 이름 | FR | 42 / right_hand | — |
| 52 | `thumb_02_r` | 오른쪽 엄지 02 이름 | FR | 43 / right_hand | — |
| 53 | `thumb_03_r` | 오른쪽 엄지 03 이름 | FR | 44 / right_hand | — |
| 54 | `thumb_end_r` | 오른쪽 엄지 끝(end) 이름 | FR | 45 / right_hand | — |
| 55 | `index_01_r` | 오른쪽 검지 01 이름 | FR | 46 / right_hand | — |
| 56 | `index_02_r` | 오른쪽 검지 02 이름 | FR | 47 / right_hand | — |
| 57 | `index_03_r` | 오른쪽 검지 03 이름 | FR | 48 / right_hand | — |
| 58 | `index_end_r` | 오른쪽 검지 끝(end) 이름 | FR | 49 / right_hand | — |
| 59 | `middle_01_r` | 오른쪽 중지 01 이름 | FR | 50 / right_hand | — |
| 60 | `middle_02_r` | 오른쪽 중지 02 이름 | FR | 51 / right_hand | — |
| 61 | `middle_03_r` | 오른쪽 중지 03 이름 | FR | 52 / right_hand | — |
| 62 | `middle_end_r` | 오른쪽 중지 끝(end) 이름 | FR | 53 / right_hand | — |
| 63 | `ring_01_r` | 오른쪽 약지 01 이름 | FR | 54 / right_hand | — |
| 64 | `ring_02_r` | 오른쪽 약지 02 이름 | FR | 55 / right_hand | — |
| 65 | `ring_03_r` | 오른쪽 약지 03 이름 | FR | 56 / right_hand | — |
| 66 | `ring_end_r` | 오른쪽 약지 끝(end) 이름 | FR | 57 / right_hand | — |
| 67 | `pinky_01_r` | 오른쪽 소지 01 이름 | FR | 58 / right_hand | — |
| 68 | `pinky_02_r` | 오른쪽 소지 02 이름 | FR | 59 / right_hand | — |
| 69 | `pinky_03_r` | 오른쪽 소지 03 이름 | FR | 60 / right_hand | — |
| 70 | `pinky_end_r` | 오른쪽 소지 끝(end) 이름 | FR | 61 / right_hand | — |
| 71 | `lowerarm_twist_01_r` | 오른쪽 아래팔 twist 01 이름 | 미선택 | 제외 | — |
| 72 | `upperarm_twist_01_r` | 오른쪽 위팔 twist 01 이름 | 미선택 | 제외 | — |
| 73 | `thigh_l` | 왼쪽 허벅지 이름 | B | 14 / body | 8 |
| 74 | `calf_l` | 왼쪽 종아리 이름 | B | 15 / body | 10 |
| 75 | `foot_l` | 왼쪽 발 이름 | B | 16 / body | 12 |
| 76 | `ball_l` | 왼쪽 발 앞부분(ball) 이름 | B | 17 / body | 14 |
| 77 | `foot_end_l` | 왼쪽 발 끝 이름 | 미선택 | 제외 | — |
| 78 | `calf_twist_01_l` | 왼쪽 종아리 twist 01 이름 | 미선택 | 제외 | — |
| 79 | `thigh_twist_01_l` | 왼쪽 허벅지 twist 01 이름 | 미선택 | 제외 | — |
| 80 | `thigh_r` | 오른쪽 허벅지 이름 | B | 18 / body | 9 |
| 81 | `calf_r` | 오른쪽 종아리 이름 | B | 19 / body | 11 |
| 82 | `foot_r` | 오른쪽 발 이름 | B | 20 / body | 13 |
| 83 | `ball_r` | 오른쪽 발 앞부분(ball) 이름 | B | 21 / body | 15 |
| 84 | `foot_end_r` | 오른쪽 발 끝 이름 | 미선택 | 제외 | — |
| 85 | `calf_twist_01_r` | 오른쪽 종아리 twist 01 이름 | 미선택 | 제외 | — |
| 86 | `thigh_twist_01_r` | 오른쪽 허벅지 twist 01 이름 | 미선택 | 제외 | — |

## Body·upper·손의 후보와 미선택 관절

검토용 body22는 기존 dev body 그룹을 그대로 참조한다. `pelvis`, `spine_01/02/03`, `neck_01`, `head`, 좌우 `clavicle/upperarm/lowerarm/hand`, 좌우 `thigh/calf/foot/ball`이다. 이름이 모두 실제 파일에 존재한다는 사실과 이 22개가 최종 body 모델/평가 규격에 적합하다는 판단은 별개다.

Upper13은 이 문서의 **새 검토 제안**으로, body22 중 pelvis와 하체8개를 제외해 다음처럼 둔다. 중심/몸통 조건을 얼마나 포함할지 논의할 수 있게 실제 key를 열거한 것이며 config에는 반영하지 않았다. 합성 upper5를 검증된 실제 upper13으로 대체한 상태가 아니다.

```text
spine_01, spine_02, spine_03, neck_01, head,
clavicle_l, upperarm_l, lowerarm_l, hand_l,
clavicle_r, upperarm_r, lowerarm_r, hand_r
```

좌우 finger 후보는 `thumb/index/middle/ring/pinky` 각각 `01,02,03,end`인 20개씩이다. Raw에서 이 40개 key가 관측되고 dev hand 그룹에 포함되는 것은 확인했다. `01/02/03`를 특정 해부학 관절 중심에, `end`를 실제 피부 끝점에 바로 대응시키지는 않는다. 공개 안내의 hand40과 숫자가 같더라도 공식 key·parent 규격까지 확인된 것은 아니다.

개발용 62개 밖의 **25개도 전부 관측되고 수치 유효**하다. 이름 기반 묶음은 머리/얼굴 보조15(`head_end` 포함), twist8, `foot_end_l/r` 2개다. 이 표는 현재 모델 후보에서 선택하지 않은 이유를 보여주며, 해당 annotation이 불필요하거나 잘못됐다는 판단이 아니다.

| 미선택 이름 기반 묶음 | Raw # | 현재 후보에서의 취급 |
| --- | --- | --- |
| 머리/얼굴 보조15 | 6~20 | 현 dev body/hand 및 평가16에 없음. 얼굴/머리 보조 모델을 구현하지 않으므로 우선 미선택 |
| Arm twist4 | 45,46,71,72 | Rig 보조 twist라는 이름 해석. 위치/역할·parent 미검증으로 우선 미선택 |
| Leg twist4 | 78,79,85,86 | 위와 같은 제한. 일반 관절로 합치거나 제거한 원본을 만들지 않음 |
| Foot end2 | 77,84 | `ball`·`foot`과 구별되는 실제 key. 서로 같은 점으로 간주하지 않음 |

**87−72=15의 차이를 위 머리/얼굴15와 동일시하지 않는다.** 공식 안내는 body32+hand40 개수이고, 이번 raw에는87개 이름이 있다. 공식72의 완전한 key·parent·생성 버전 대조가 없어 어느15개가 빠지는지 확정할 수 없다. 현 개발용62는 또 다른 모델/시각화 선택이다. `raw87 → 공식72 → dev62`라는 검증된 계층이 존재한다고 표시하지 않는다.

## 공식 평가16과 기존 dev62

평가16의 순서를 명시하면 다음과 같다. 원문의 웹 표기 `thigt_l/r`와 저자 코드의 `thigh_l/r` 차이는 [데이터 계약](DATA_CONTRACT.md)에 기록돼 있으며, 실제 raw key는 `thigh_l/r`다. Raw key를 오탈자처럼 자동 수정하지 않는다.

```text
0 head, 1 neck_01, 2 upperarm_l, 3 upperarm_r,
4 lowerarm_l, 5 lowerarm_r, 6 hand_l, 7 hand_r,
8 thigh_l, 9 thigh_r, 10 calf_l, 11 calf_r,
12 foot_l, 13 foot_r, 14 ball_l, 15 ball_r
```

16개 모두 raw87와 기존 dev62에 존재하며 dev body22의 부분집합이다. Dev body에만 더 있는6개는 `pelvis`, `spine_01/02/03`, `clavicle_l/r`다. Hand finger40은 평가16에 없다. 따라서 dev62 전체 평균을 공식16 body MPJPE로 보고하거나 평가16에 없는 관절을 정답 누락으로 표시하면 안 된다.

Dev body22와 좌우 hand21을 단순 연결하면64 slot이지만 `hand_l/r` 각각이 body와 해당 hand에 중복된다. 기존 고유62 순서는 첫 등장을 보존하며 `22+21+21−2=62`다. 이름 기준으로 왼손은 `hand_l + left fingers20`, 오른손은 `hand_r + right fingers20`으로 조회할 수 있으나 전체 관절 평균에서는 손목을 다시 더하지 않는 정책이 필요하다.

M3의 `body owns wrist`를 실제 key에 적용하려면 우선 `hand_l/r`가 선택하려는 손목 위치인지 확인해야 한다. 해당 위치 정의가 검증되고 모델 역할로 선택된 경우에만 body에 두 이름을 두고, hand local wrist는0/손목 loss 제외, 손가락은40개를 예측하는 **후속 설계 후보**가 된다. Missing wrist GT이면 해당 손 local target mask는false이고, 원래 global finger GT 유효성은 보존한다. 예측 평가의 global 결합에는 예측 wrist를 사용해야 한다. 이번에는 이 변환이나 runtime mask 변경을 구현하지 않았다.

## 합성 M3 이름과 실제 key 사이의 미확정 부분

[M3 합성 규격](M3_WHOLE_BODY_TEACHER.md)의 10개 이름은 실제 rig의 공식10개 subset이 아니다. 단순 문자열 치환으로 checkpoint·좌표를 재사용할 수 없다.

| 합성 역할 | 실제 key 후보 | 현재 결론 |
| --- | --- | --- |
| `root` | `pelvis` | 골반 이름은 존재하지만 root landmark·rig root·좌표 원점은 구별해야 함. Device 원점은 stereo midpoint이므로 pelvis를0으로 만드는 변환을 승인한 것이 아님 |
| `head` | `head` | 문자열은 일치하나 합성 점의 의미와 실제 해부학적 중심/좌표 동등성까지 입증하지 않음 |
| `shoulder_l/r` | `clavicle_l/r` 또는 `upperarm_l/r` | 쇄골/상완 rig origin 중 어떤 점이 shoulder 정의인지 불명확. **1:1 확정하지 않음** |
| `wrist_l/r` | `hand_l/r` | 손목 기준의 유력한 이름 후보. 실제 손목/손바닥 rig origin의 위치 정의는 미검증 |
| `index_l/r` | `index_01/02/03_l/r` | 합성의 단일 index 점을 raw의 어느 세그먼트로 볼지 결정하지 않음 |
| `tip_l/r` | 검지 끝을 의도할 때 `index_end_l/r` 등 | 실제 raw에는 다섯 손가락의 end가 각각 존재. 합성 generic tip을 모든 end나 한 end로 자동 연결하지 않음 |

이번 대응표는 위 후보를 검토 가능하게 했지만 skeleton·bone 길이·부모 연결·landmark 정의를 검증하거나 실제 모델을 선정한 것은 아니다. 기존 합성 checkpoint를 87/62개 실제 입력에 연결하지 않는다.

## 필드 검사·추적성과 한계

각 field의 vector shape·수치 타입(bool 제외)·finite를 직접 검사했다. 관절당 joint field7개가 32파일 모두 동일했으며 누락·invalid vector는0이다.

| Field | Shape | Brutal18 유효 배열 | Opening14 유효 배열 | 전체 유효 배열 |
| --- | --- | ---: | ---: | ---: |
| `trans` | 3 | 1,566 | 1,218 | 2,784 |
| `rot` | 3 | 1,566 | 1,218 | 2,784 |
| `camera_left_pts3d` | 3 | 1,566 | 1,218 | 2,784 |
| `camera_left_pts2d` | 2 | 1,566 | 1,218 | 2,784 |
| `camera_right_pts3d` | 3 | 1,566 | 1,218 | 2,784 |
| `camera_right_pts2d` | 2 | 1,566 | 1,218 | 2,784 |
| `device_pts3d` | 3 | 1,566 | 1,218 | 2,784 |

필수 좌표5개는 합계13,920배열, trans/rot까지 합하면19,488배열이다. 회전 값이 유한하다는 검사만으로 각도 단위·rotation composition이 검증되는 것은 아니다. 음수 좌표가 하나 이상인 2D joint-view는 Brutal108·Opening84·합계192개로 보존했으며 visibility·정답 누락으로 바꾸지 않았다.

Git 제외 `outputs/joint_mapping_20261016/developer/`에 다음 증거를 저장했다.

- `audit.py`: 표 작성을 위한 읽기 전용 조사 script. 변환기나 학습 기능이 아니다.
- `sample_manifest.json`: 원본 경로·sequence/frame·32개 파일의 bytes/SHA-256·기존 manifest 일치·관측 순서.
- `joint_inventory.json`: 87개 이름별 raw/Dev/Eval index·후보·field별32회 유효성. Runtime 설정 파일이 아니다.
- `summary.json`: 두 sequence의 frame 목록·집합/순서·필드 통계, 원본 불변, revision·조사 script/config/manifest hash.
- `table.md`: raw inventory에서 생성한87행 표의 조사 사본. 최신 배포용 문서는 이 Markdown이다.

이번 config SHA-256은 `12e420401502847cde226a5f91d6c8c46db1d18ce5f86db80e605f3d4e376170`, extraction manifest SHA-256은 `4ed08a0c53c624710d4d8f4e6065c5dac305dbddc21bb2a3359b559186323878`이다. Script 재실행은 프로젝트 root에서 `.venv/Scripts/python.exe outputs/joint_mapping_20261016/developer/audit.py`다. Python 표준 라이브러리만 사용하며 새 파일 다운로드가 없다.

문서 87행의 누락·중복과 index/역할 대조, 로컬 링크·UTF-8 및 원본32개 hash 불변을 검사했다. 문서 작업이므로 모델/훈련 테스트는 추가 실행하지 않았다. Runtime·config·raw·훈련 guard·기존 report는 변경하지 않았고 학습은 실행하지 않았다. 문서 묶음은 오케스트레이션 검토 후 로컬 커밋하며 push하지 않는다.

오케스트레이션은 원본을 별도로 읽어 87행 이름·raw/Dev/Eval 번호·그룹/역할, 32개 manifest hash와 19,488개 유효 배열을 독립 대조했다. 근거는 Git 제외 `outputs/joint_mapping_20261016/root_audit.json`·`root_review.json`에 남겼다.

새 데이터셋 전체의 이름/rig 일관성, 공식72 key 목록, REWIND body/upper/hand 최종 규격·평가 관절 선택, 단위·좌표/회전·시간·대응 RGB overlay는 남아 있다. 공식 split 경로와 calibration annotation 대조의 기존 확인 범위는 [ArchVis 후속 감사](ARCHVIS_POSE_AUDIT.md#후속-공식-근거rgb-준비-조사-2026-10-15)를 따른다. 대응표 작성만으로 `train_ready`나 M0/M1/M2/M3 상태를 올리지 않는다.
