# 실제 좌표·단위 검증과 학습 관절 구성 검토

2026-10-16. `c4dbbde`에서 시작해 실제 ArchVis 32개 JSON을 읽고 고정된 변환식·역변환·좌우 관계·calibration을 대조했다. **World→camera/device의 수치 관계는 이 sample에서 확인했다. 물리 단위는 cm를 강하게 지지하는 공식 rig 근거와 실측을 확보했으나 raw field의 직접 unit 선언과 구분한다.** Body22+finger40/upper13은 후속 실제 모델 규격으로 검토할 수 있으며 이번에 채택·설정 변경·학습을 실행하지 않았다.

## 근거의 강도와 범위

| 항목 | 이번에 확인한 수준 | 남은 제한 |
| --- | --- | --- |
| 이름·필드 | [관절 대응표](JOINT_MAPPING.md)의 raw87과 기존 manifest를32개에서 재대조 | 같은 인물/메시·두 sequence뿐. 공식72의 완전한 key/rig 목록은 미확인 |
| 좌표 변환 | 저장된 camera pose로 고정식을 계산하면87관절×32frame×2view와 수치 정밀도에서 일치 | 다른 rig·인물·archive 버전으로의 적용은 추가 확인 필요 |
| 회전 | 이 sample의 camera `rot`/`open3d_rot`는 degree 식, device `open3d_rot`는 radian 식과 일치 | Joint `rot`의 orientation 의미·모든 Euler convention을 검증한 것이 아님 |
| 거리 단위 | UE2 공식12cm rig와 실제약12raw baseline·길이보존을 결합하면 cm를 강하게 지지 | UE2 raw field의 직접 단위 선언/생성 코드·export 설정은 미확보. Runtime 단위는 그대로 |
| Projection | 공식 Scaramuzza3D→2D annotation 관계를 독립 재현 | 실제 RGB overlay·영상 좌우 의미·가림은 미검증 |
| 시간 | UE2 공식 논문 관련 단락은 명목25FPS를 설명 | Sample에는 timestamp가 없고 frame index→실제 시간 대응·누락 여부는 별도 |

실제 입력은 [ArchVis 감사 기록](ARCHVIS_POSE_AUDIT.md)의32파일이다. 주대상 Brutal_Assassination0~17의18개는 번호가 연속되고, Opening14개는 산재 subset으로 gap을 유지했다. 파일별 SHA-256은 전달 extraction manifest와 일치하고 원본은 변경하지 않았다. 전체 archive 무결성이나 학습 준비 완료를 뜻하지 않는다.

## 단위: cm를 지지하는 metric anchor

다음의 서로 다른 근거를 연결했다.

1. [UnrealEgo ECCV2022 논문](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136660001.pdf) p.6 §3.1은 UE1 가상 안경 rig의 카메라 간격을12cm로 명시한다. p.8 Fig.4의 pelvis-relative 분포도cm 표기다. UE1 서술을 UE2 field 정의로 자동 복사하지 않는다.
2. 더 직접적인 [UnrealEgo2 CVPR2024 본논문](https://openaccess.thecvf.com/content/CVPR2024/papers/Akada_3D_Human_Pose_Perception_from_Egocentric_Stereo_Videos_CVPR_2024_paper.pdf) §3의 **UnrealEgo2 Dataset** 단락은 가상 rig 간격12cm와 명목25FPS를 설명한다. UE-RW의 별도 실물 장치 단락을 대신 사용하지 않았다. 이번에는 공식 검색 인덱스의 관련 본문을 확인했고 PDF 직접 열기는403이어서 전체 PDF의 새 열람/다운로드라고 기록하지 않는다.
3. [공식 metadata/calibration 안내](https://unrealego.mpi-inf.mpg.de/)는 UE1/UE2의 같은 camera·좌우 공통 intrinsics를 설명한다. CVPR2024 [보충자료 §B](https://openaccess.thecvf.com/content/CVPR2024/supplemental/Akada_3D_Human_Pose_CVPR_2024_supplemental.pdf)의 검색 본문도 UE1 setup 재사용을 뒷받침한다. 부록 직접 접속 실패와 검색 본문 확인을 구분했다.
4. 실제32frame의 world baseline 길이는 **11.9988978244~12.0008403456 raw units**, 평균12.0000256739다. 아래 변환은 배율을 추가하지 않으며 world/camera/device 사이 거리도 보존한다.

따라서 **이 sample에서 1 raw unit≈1cm, mm로 변환한다면×10이라는 해석은 강한 dataset-specific 추론**이다. 사람 키·보행 크기나 UE의 일반 기본 단위만 보고 정한 추정이 아니다. 다만 알려진12cm rig가 현 export에도 그대로 쓰였다는 연결과 실제 baseline에 기반한 간접 단위 검증이며, 모든 raw field에 명시된 `unit=cm` 문구나 생성 코드를 확보한 것은 아니다.

후속 변환기에는 이 근거 사슬을 보존해 cm→mm×10을 검토하는 방안을 권고한다. 채택 시 world translation·device/camera 3D·손목 상대 target 모두 같은 scale을 사용해야 한다. 이번에는 기존 `distance_unit/verified`·데이터·checkpoint를 수정하지 않았다. `ground_z_value` 및 joint rotation의 단위를 이 결과로 일괄 확정하지 않는다.

재투영만으로는 단위를 확인할 수 없다. Camera 3D를0.1/10/100배해도 같은 광선 방향이므로 Scaramuzza2D 변화 최대값은 약4.71e-12px에 그쳤다. 절대 scale의 근거는 이 projection 일치가 아니라 공식12cm와 실측 baseline의 연결이다.

## World→camera/device: 재현 가능한 고정식

모든 point는 column vector로 표기한다. `p_U`는 joint의 raw `trans`, `C_U`는 camera의 raw `trans`다. Raw camera `trans`는 **world에서의 camera 위치**이며 world→camera extrinsic translation `t`가 아니다.

먼저 world basis를 다음처럼 명시적으로 바꾼다.

```text
F = diag(1, -1, 1)
p_G = F p_U
C_G = F C_U = camera.open3d_trans

(ox, oy, oz) = radians(camera.open3d_rot)
Q = Rz(oz) Ry(oy) Rx(ox)

B = [[ 0, -1,  0],
     [ 0,  0, -1],
     [ 1,  0,  0]]

R = B Q^T
t = -R C_G
p_camera = R p_G + t = R F (p_U - C_U)
p_U = F R^T (p_camera - t)
```

여기서 `Rx/Ry/Rz`는 표준 right-handed active axis rotation matrix이고 곱셈 순서를 식으로 고정했다. `Q`를 곧바로 OpenCV optical camera→world matrix라고 부르지 않는다. Optical basis를 포함한 inverse는 `R^T=Q B^T`다. `R`은 reflected world G→camera, `R^T`는 camera→G다.

Raw camera `rot=[P,Y,R]`와 비교하면 `camera.open3d_rot=[R,-P,-Y]`가32frame의 좌우64camera에서 정확히 일치했다. 위 식의 결과는 raw P/Y/R로 별도 계산한 forward/right/up 식과 최대2.221e-16 matrix 성분 차이로 일치한다. 각 frame/joint를 least-squares로 fitting해 잔차를 줄인 결과가 아니다. 초기 후보 대조 뒤 같은 고정식으로 모든 실제 관절을 검사했다.

`device.open3d_rot`는 **camera.open3d_rot를 degree→radian으로 바꾼 값**과 최대4.441e-16rad 차이였다. 반면 camera의 동명 field는 degree로 해석해야 맞았다. `open3d_rot`라는 이름만 보고 camera/device에 같은 각도 단위를 적용하면 안 된다. 이 검사는 저장된 수치 관계이며 joint별 `rot`의 local/global rotation 의미를 별도로 증명하지 않는다.

Raw UE world에서 바로 camera로 가는 matrix는 `A=R F`다. 검사에서 `det(A)≈−1`, `det(R)≈+1`, `R R^T≈I`였다. 이는 basis 반전과 proper rotation을 분리해야 한다는 근거다. M2의 `encode_camera`는 det+1 rotation을 요구하므로 **A를 그대로 6D rotation에 넣을 수 없다.** Reflected world와 `R,t`를 명시적으로 사용하는 전처리가 후속 선택이다. Raw device/camera 좌표는 이미 변환된 target이므로 거기에 F를 다시 적용하지 않는다.

미래 실제 입력 계약에 이 방식을 채택한다면 preprocessing/version·manifest에 world basis와 변환 방향을 기록하고, scale 확정 후 `t_mm=scale_mm_per_raw * t_raw`를 제공해야 한다. `camera.open3d_trans`는 이미 F가 적용된 camera 위치이므로 다시 반전하지 않는다. Raw camera 위치를 t 자리에 복사하거나 회전의 transpose를 혼동하지 않는다. 이번에는 계약·변환기를 변경하지 않았다.

### Device midpoint와 좌우 관계

32개 각각의 frame 안에서 좌우 camera 회전 matrix가 같았다. `M_U=(C_left_U+C_right_U)/2`, `M_G=F M_U`로 두면 다음 식도 맞았다.

```text
device.open3d_trans = M_G
p_device = R (F p_U - M_G)
camera_side.device_pts3d = R (F C_side_U - M_G)
p_device = p_camera_side + camera_side.device_pts3d
p_device = (p_camera_left + p_camera_right) / 2
```

일반적으로 서로 다른 camera orientation의 좌표를 평균해도 된다는 뜻은 아니다. 이 실제 rig의 공통 회전·origin 관계를 확인한 범위다.

복원된 camera origin의 device X는 `camera_left`가+5.999449~+6.000420, `camera_right`가−6.000420~−5.999449 raw다. Y/Z는 각각 약0.00035raw 이내다. 따라서 이 convention에서 이름 `left/right`를 곧바로 X 부호로 추측하거나 통상적인 stereo 배치와 같다고 가정하면 안 된다. 본문은 **저장된 view 이름과 위치 수치**를 확인했으며, 실제 영상에서 어느 물리적/해부학적 좌우를 뜻하는지는 RGB overlay 전에는 단정하지 않는다. `hand_l/r`와 camera view 이름을 서로 교환하지 않는다.

좌우 3D와 2D field의 잘못된 짝을 만드는 대조에서는 projection 오차 평균228.001px, 최대1158.315px였다. 이는 side field를 섞으면 맞지 않는다는 확인이다. 좌우 전체를 일관되게 재명명한 경우까지 구분하는 물리적 좌우 증명은 아니다.

### 실제 수치 결과

아래 값은 double-precision Python 계산의 **raw units**다. 물리 단위 미확정 상태의 오차를 mm로 표시하지 않았다. Max component와 L2는 서로 다른 reduction이다.

| 검사 | 비교 수 | 최대 잔차 |
| --- | ---: | ---: |
| World→camera 성분별 절대 오차 | 5,568 joint-view | 7.461e-14 raw |
| World→device L2 | 5,568 joint-view 기준 반복 대조 | 6.917e-14 raw |
| Camera→world inverse L2 | 5,568 joint-view | 1.279e-13 raw |
| Stereo camera 평균→device 성분별 | 2,784 joint | 5.684e-14 raw |
| Camera offset을 더한 device 성분별 | 5,568 joint-view | 5.684e-14 raw |
| Camera origin→device offset 성분별 | 64 camera | 3.109e-15 raw |
| Left→right 상대 translation 식 성분별 | 2,784 joint | 5.684e-14 raw |
| `device.open3d_trans`와 reflected world midpoint | 32frame | 0 raw |
| 기존61edge의 world↔camera 길이 차이 | 1,952 edge-frame | 3.198e-14 raw |

World/camera edge 길이의 비는0.999999999999988~1.0000000000000095다. 같은 수치 단위로 저장됐음을 지지하며 cm라는 절대 단위는 위 metric anchor가 뒷받침한다. Edge 길이가 보존됐다고 그 edge가 실제 rig parent라는 결론을 내리지는 않는다.

잘못된 변환의 대조는 raw camera `trans`를 extrinsic t로 쓰면 평균284.358raw L2, F를 생략하면 평균84.360raw, R 대신Rᵀ를 쓰면 평균41.020raw였다. 정확식이 단순히 작은 residual을 내는지뿐 아니라 흔한 방향/translation 오해를 구별하는지도 확인했다.

오케스트레이션도 별도 Torch float64 구현으로32파일 변환·거리 보존을 대조했다. 해당 수치의 반올림·연산 순서에 따른 차이를 허용해 수치 정밀도 수준 일치로 해석한다. Root는 단위를 주장하지 않도록 t=0인 API 시험에서 properR32개 수용·반사matrix32개 거부와 실제 관절 역할의 구조 검사도 확인했다. 이 API 수용 자체가 데이터의 물리 의미 검증은 아니다.

## 3D→2D와 시간에서 확인한 범위

기존에 받은 공식 Scaramuzza calibration을 그대로 사용했다. `r=hypot(x,y)`, `theta=atan2(-z,r)`, `rho=polyW2C(theta)`에 공식 center·affine를 적용했다. 5,568 joint-view의2D annotation 오차는 평균1.90656e-5px, 최대7.46509e-5px로 [이전 검사](ARCHVIS_POSE_AUDIT.md#후속-공식-근거rgb-준비-조사-2026-10-15)를 재현했다. 새 fitting·모델 예측 정확도 평가가 아니다. Negative2D192개·z≤0 관측704개도 보존한다.

명목25FPS의 공개 명세는 이전 UE1만의 근거보다 강하다. 하지만 raw에 timestamp가 없으며, export/version 및 frame index가 그 sampling time을 그대로 나타내는지까지 이번 변환식으로 증명하지 못한다. 연속18frame을 임의로 다른 FPS로 시간 변환하지 않고, Opening gap을 보간하지 않는다. 추후 nominal time을 쓰면 `derived_from_dataset_fps`처럼 실제 timestamp GT와 구분해 기록해야 한다. 현재 velocity/acceleration·world/contact loss는 적용하지 않는다.

## 실제 학습 관절 후보와 기존 코드의 적합성

검토 권고는 **body22 + wrist를 제외한 fingers40 = 고유62**, upper13, 좌우 wrist 후보 `hand_l/r`, 별도 body 평가16이다. 이름·순서는 [전체 관절 대응표](JOINT_MAPPING.md)를 따른다. 이 선택은 raw87의 subset이며 공식72나 REWIND 공식 관절 규격과 동일하다고 주장하지 않는다.

| 후보 | 코드와의 관계 | 후속 선택·검증 |
| --- | --- | --- |
| Body22 | M2는 명시한 joint 수/edge로 모델을 생성하므로 새62문서에서 body22를 추출하는 구조와 맞음 | 실제 이름의 graph/version·단위/좌표 계약으로 새 모델/checkpoint 필요 |
| Upper13 | M3는 이름 기반 body subset을 허용하고 양쪽 wrist가 upper에 있는지 검사 | 몸통을 포함하는 문서 제안. 상체 조건 효과는 이후 별도 비교하며 합성 upper5와 동일시하지 않음 |
| Left/right hand 각21 | `hand_l/r + fingers20`으로 역할을 명시할 수 있음 | Body가 wrist를 소유하고 hand 모델 target에서는 wrist2개를 제외 |
| Fingers40 | M3 내부에서 device GT→손목 상대 target으로 분리하는 구조와 맞음 | 손목 위치 정의·관절 순서와 prediction attachment를 명시 |
| Eval16 | 기존 body22의 부분집합, 이름 순서가 따로 존재 | 모델 output22/62 평균과 분리해 gather·집계. 최종 지표 규칙은 별도 고정 |

Body22+양손21씩을 단순 합친64slot에서 wrist2개를 한 번씩 제거해야 고유62다. `hand_l/r`를 wrist 기준으로 선택한다면 body가 한 번 소유하고, hand의 implicit local wrist는0이며 해당 wrist loss를 중복 추가하지 않는다. Predicted body wrist+predicted local finger로 global을 만들고, GT upper 진단에서도 attachment source를 별도 표시한다. 실제 wrist landmark의 정의가 없어도 명시적인 rig joint를 local origin으로 쓰는 모델 설계는 가능하지만, 이를 해부학적 손목 중심의 검증으로 표시하면 안 된다.

기존 `rewind_m3.data.validate_roles`는 이 실제 이름의 body22/upper13/finger40 구성을 **구조적으로 수용**한다. 이는 Root의 검사 결과이며 부모·landmark·학습 품질 검증이 아니다. 합성 root/shoulder/index/tip 이름을 강제로 치환하거나 기존6/4관절 checkpoint의 embedding/adjacency를 새22/40관절용으로 간주하지 않는다.

### Parent 진실과 모델 graph 가정을 구분

기존 dev62는62nodes/61edges의 단일 tree, body 부분은22nodes/21edges의 단일 tree다. 원본 joint field에 parent 정의가 없으므로 이 구조만으로 native rig hierarchy가 확인되지는 않는다. 다만 **명시한 모델 graph prior를 가정으로 채택하고 구조·데이터 shape를 검증하는 것**과 native parent를 복원하는 것은 별개다. Point loss/GCN을 시작하기 위해 공개되지 않은 rig parent를 영구적으로 기다려야 한다고 권고하지 않는다. Native hierarchy 의존 물리·관절 회전 constraint는 별도로 보류한다.

현재 M3는 wrist를 body로 소유하게 하면서 hand graph에서 body↔finger edge10개를 제외한다. Fingers40만 남기면30edges/10개 독립 four-node chain이다. 이 사실은 기록할 graph 정책이며 손 모델이 동작할 수 없다는 뜻은 아니다. M2 `GraphBlock`의 spatial self-attention은 전체 finger token 사이를 연결하고 상체/wrist cue도 제공한다. 후속에는 이 prior를 버전 있는 모델 가정으로 채택할지, wrist conditioning node 등을 별도 설계할지 선택한다. 이번에는 edge를 바꾸거나 모델을 실행하지 않았다.

### Mask·loss·normalization 검토

- GT 유효성은 field별로 유지한다. 실제 sample의 finite2D·3D를 확인했지만 화면 밖/음수2D·z≤0를 임의 missing/visibility=false로 바꾸지 않는다. Confidence·visibility unknown은 M1의 known bit와 분리한다.
- Hand local target valid는 `finger GT valid & wrist GT valid & time_valid`다. Wrist GT가 없으면 그 손 local supervision을 제외하지만, 원래 finger global GT가 유효하면 global 평가 mask는 유지한다. GT wrist를 예측 결과에 붙여 global 오차를 숨기지 않는다.
- 거리 단위를 채택하면 body·hand·camera translation의 mm 환산을 일관되게 적용한다. M1은 선언한cm를×10해 내부mm로 바꾸지만, field를 제대로 검증하지 않은 채 `distance_unit=cm`만 넣으면 정당한 변환이 되지 않는다.
- 환산은 한 번만 한다. 후속 준비 문서에 raw device GT를 보존하고 `distance_unit=cm`를 명시한다면 M1이 mm 환산을 담당한다. 이미 GT를×10한 문서에는 `distance_unit=mm`가 필요하다. M2의 camera translation은 별도 계약상 처음부터 mm여야 하므로 이를 독립적으로 환산하고, 단위 환산과 `target_scale_mm` 정규화를 혼동하지 않는다.
- 손목 상대 target은 raw 문서에 device-relative라고 거짓 표기하지 않고 M3 내부 계약으로 만든다. Device target은 stereo midpoint 상대이며 pelvis-relative로 바꾸는 설계가 아니다.
- 위치/보조 frame loss는 유효joint·xyz·slot 평균인 normalized MSE다. Fixed target scale은 학습 설정으로 별도 보존하고 future/test 통계를 쓰지 않는다. 현재100mm scale을 실제 body/hand의 최적 설정으로 확정하지 않는다.
- 전체62 평균에는 finger40개가 들어간다. Body22·eval16·hand40 global/local·좌우 wrist 오류를 나누고, body/hand loss의 가중/단위를 명시해야 한다. Whole-body62와 공식body16을 같은 지표로 보고하지 않는다.
- 시간·world/contact 의미가 확인되지 않은 채 velocity/foot loss를 켜지 않는다. Raw joint rotation을 사용하지 않는 위치 추정에 해당 rotation의 해부학적 의미 검증을 불필요한 필수 조건으로 추가하지는 않는다.

가령62관절 문서를 현재 M1 feature 계약으로 만들면 관측 `[N,T,868]`, body22는308개/slot, hand40은560개/slot이다. Upper13은 pose39개+mask13개로 M3 upper MLP 입력52개다. 이는 코드 shape 검토 결과이며 실제 tensor 변환·모델 학습을 이번에 실행한 것이 아니다.

## GT2D 진단의 진입 조건과 최소 다음 단계

좌표 검증 진전만으로 현재 실제 학습을 시작할 수는 없다. 현 `dataset_sample`은 **joints/units/coordinates/calibration/split/rgb_pose_alignment** 여섯 verified/evidence를 요구하고, M2는 camera_transform/skeleton 근거도 요구한다. 실제 RGB 정합성은 아직 확인하지 못했다. Pose-only GT2D 진단이라고 `rgb_pose_alignment=true`로 꾸미거나 synthetic으로 표시해 통과시키지 않는다. 기존 gate는 이번에 유지했다.

다음 작업은 다음처럼 최소 단위로 분리할 수 있다.

1. **규격 선택 문서화:** 이 보고서의 고정 변환식·cm metric-anchor 추론을 검토하고, body22/finger40/upper13와 graph를 버전 있는 프로젝트 가정으로 채택할지 결정한다. Raw unit 직접 선언/생성 코드가 필요하면 해당 export 정의를 좁혀 확인한다. 사람 크기 추정이나 전체 다운로드로 대체하지 않는다.
2. **RGB 대응 또는 진단 전용 계약 검토:** 같은 sequence/frame의 좌우 RGB를 확보해 overlay·view 의미를 확인하는 경로와, GT2D 전용 진단에 불필요한 RGB 요건을 명시적으로 분리하는 새 계약을 검토하는 경로를 구분한다. 후자는 별도 구현·검토 사항이며 현재 guard를 우회하지 않는다. Sample/실험 종류와 미검증 항목을 보존해야 한다.
3. **읽기 전용 변환 검증:** 채택된 규격으로 작은 특징 문서를 만드는 후속 구현에서 raw hash·manifest·단위·basis·관절 순서·mask·preprocessing version을 보존하고 inverse/side/scale/shape를 검사한다. 87→62 선택에서 미선택25개 원본을 삭제하지 않는다. Body와 hand checkpoint는 새 관절 규격으로 별도 생성한다.
4. **작은 진단 순서:** 승인된 진단 gate가 충족된 뒤 regression→body→hand 작은 overfit을 순서대로 실행한다. 현재18frame은 train subset의 입출력/overfit 용도이며 겹치는 window를 나눠 일반화 성능으로 주장하지 않는다. Opening14는 gap 진단이고 독립 validation clip의 대체가 아니다. **현재 두 sequence는 모두 공식train 경로이고 기존 Trainer는 train+validation을 요구하므로 이 자료만으로 즉시 실행할 수 없다.** 공식 split과 원본 motion 중복을 고려한 별도 validation을 확보하며, train 일부를 임의로 validation이라고 바꾸거나 필수 조건을 끄지 않는다.

물리 단위·선택 graph를 구현 가정으로 명확히 채택할 수 있다는 권고와, 실제 RGB/평가 프로토콜·field 정의까지 모두 검증됐다는 선언을 구별한다. 준비되지 않은 항목을 무조건 verified로 올리지 않고, 실제 목적에 필요한 최소 계약을 검토하는 것이 다음 단계다.

## 실행 증거와 미실행 항목

Git 제외 `outputs/coordinate_audit_20261016/developer/`에 재현 script와 결과를 보존한다.

| 파일 | 내용 |
| --- | --- |
| `probe_rotation.py` | 초기 고정식·각도 부호 후보의 수치 탐색. Runtime 변환기 아님 |
| `audit_coordinates.py` | 표준 라이브러리의 실제32파일 고정식/역변환·negative control·projection·scale/graph 검사 |
| `coordinate_review.json` | 전체/sequence별 residual·determinant·Euler 비교·source/script hash와 한계 |
| `frame_transforms.json` | 원본 frame별 R/A/t·midpoint·baseline의 조사 기록. 학습 camera manifest 아님 |
| `sample_manifest.json` | 원본32개 파일의 sequence/frame·SHA-256 |
| `graph_review.json` | 기존 graph의 연결 성분·선택 이름·edge 길이 통계. Parent 증명 아님 |
| `source_review.json` | 공식 출처와 열람 방식/접속 실패·근거 강도 |

재실행은 프로젝트 root에서 다음과 같다.

```sh
.venv/Scripts/python.exe outputs/coordinate_audit_20261016/developer/audit_coordinates.py
```

Python3.12.14/표준 라이브러리로 실행했다. 수식·데이터·manifest·config·script hash를 보존하며 raw32개 hash 불변을 검사했다. 새로운 의존성·대량 다운로드·실제 특징 변환기·학습·기존 모델/config/ready gate 변경은 없다. 문서/수치 조사이므로 모델 회귀나 학습 성능 테스트는 실행하지 않았다. 실제RGB overlay, 다른 rig/인물·전체 archive, source-specific raw unit 직접 정의와 joint parent/rotation 의미, 실제 시간 대응·모델 성능은 미검증으로 유지한다.

오케스트레이션의 독립 증거는 같은 상위 폴더의 `root_review.py`·`root_review.json`, `root_projection.json`, `root_sources.json`, `document_review.json`에 보존한다. 별도 Torch float64 수식에서 world→camera 최대 성분 잔차4.974e-14raw, 역변환5.684e-14raw, 전체 관절 쌍 거리 보존8.527e-14raw를 확인했다. 별도 polynomial 계산의 재투영 최대7.46509e-5px와 scale10배 불변 최대8.065e-12px도 일치했다. 연산 순서 차이로 각 구현의 마지막 유효 숫자는 다를 수 있다. 문서 UTF-8·로컬 파일 링크, raw32개 hash와 skeleton config 불변을 검사했다. 재현 시 프로젝트 root에서 `.venv/Scripts/python.exe -c "import runpy; runpy.run_path('outputs/coordinate_audit_20261016/root_review.py', run_name='__main__')"`를 사용한다. 이 조사 산출물은 Git 제외이므로 새 checkout에는 자동으로 포함되지 않는다. 문서 묶음은 검토 후 한국어 메시지로 로컬 커밋하며 push하지 않는다.
