# v1.2.3 — V12-CAMERA-GAUGE-001 원점 민감도 진단

표시 버전: **v1.2.3** (2026-11-15 사용자 확정). 기존 실행 ID `V12-CAMERA-GAUGE-001` 및 파일·config·결과 해시는 보존한다. [번호 관리](EXPERIMENT_VERSIONS.md).

후속 상태: 2026-11-15 실제300frame 실행과 결과 독립검토를 완료했다. 원점 민감도와 도시의 큰 개선, ArchVis 손 악화를 함께 확인했다. 아래는 실행 전 고정 계약이며 최신 판정은 [결과 문서](V12_CAMERA_GAUGE_RESULTS.md)를 따른다.

2026-11-15 사용자 승인에 따른 무재학습 진단 준비입니다. 실제 Colab 결과는 아직 없으며 root 검토·커밋·고정 배포 후 테스트 세션이 사용자 실행을 안내합니다. [직전 대조 결과](V12_CAMERA_CONTROL_RESULTS.md)에서 관찰한 카메라 조건 차이가 단순 world 좌표 재표현에 대한 모델 민감도로 이어지는지 확인합니다. 새 학습·v1.3·v1.5·clipping·정규화 교체는 포함하지 않습니다.

## 고정 입력과 가설

기존 ArchVis2clip100frame과 CityPark·Downtown4clip200frame, 총6clip300frame만 사용합니다. 모두 공식 validation이며 재선정·다른 후보 탐색·오류 후 대체를 하지 않습니다. 같은 F9/B10/H9 모델, RGB, frame 순서, GT/device 좌표, 회전 R, 예측2D 특징, keyed noise/seed71, FP32/DDIM10, body offset100/hand offset200과 clip reset을 유지합니다. Hand는 예측 body 상체/wrist 조건을 유지합니다.

| 입력 | 크기·SHA256 |
| --- | --- |
| 도시 결과 ZIP |47,144,765B / `c302fede5d8f3f5d72548dedb627f57892e9d8420670381e4c49ebd049bc6910` |
| ArchVis 대조 결과 ZIP |23,996,807B / `48ca2d5cab15596b637bbb3db47a7736e32446cc516d6994fa6bdbf79ed7c63b` |
| 모델 ZIP |873,393,783B / `48661e75e0969ca140ab9cf0e543d73d2bb3b9ec108d30652cf7fb4f4b76b506` |
| 기존 input ZIP |SHA256 `6256eb7bd148ec51bdb94627c68e54657be3808c607a12e13b68f7e479b15e97` |

도시 source는 revision `2de4262`, identity `849b615f3129620aff585505d5ae435a5bf9650631ccb8511bdcd9a147c630f5`, 대조 source는 `fdab0f1f499869a0d292a3d58944020b6a187f84`, identity `f0742d9b8c137c50add64059c7404440add35c6ecce71f56a74d51646f9bc116`입니다. [고정 config](../configs/v12_camera_gauge_001.json)에 기록합니다. 새 소스 ZIP·revision/notebook pin은 root 배포 receipt가 기준입니다.

두 ZIP 전체 크기/SHA, receipt의 모든 member hash, JSON seal, source/model, selection→prepared→prediction, 대조 cameras→comparison 연결을 확인합니다. 6clip 순서와 모든 frame/pose/RGB 키·크기/SHA를 연결합니다. 이전 원격 mtime을 재사용하지 않고 고정900경로만 stat하여 현재 metadata를 고정합니다. Directory listing이나 새 선정은 하지 않습니다. 준비시 모든 RGB·pose의 기대 SHA를 확인하고 geometry를 다시 검사합니다. `cross_data.prepare`의 선택적 기대 SHA 검사는 기존 SHA 없는 경로에 영향을 주지 않습니다.

## 허용 변환

Canonical world(mm)의 `x′ = x + Δ`, `t′ = t − RΔ`만 사용합니다. `x_camera = R x + t`이며 각clip 모든 frame·양쪽 카메라에 동일 Δ를 적용합니다. Native UE 축과 canonical world 축을 섞지 않습니다. GT는 device 기준이므로 변경하지 않으며 원본 pose 문서도 수정하지 않습니다.

| Variant | 고정 Δ(mm) |
| --- | --- |
| baseline |(0,0,0) |
| zero_repeat |(0,0,0), 같은 실행에서 sampler를 새로 호출 |
| first_midpoint |첫 frame의 `C = −Rᵀt` 좌우 center midpoint의 음수 |
| world_x_plus_100m |(100000,0,0), canonical world x축 +100m |

Midpoint는 첫frame camera만 사용하므로 미래 frame이나 GT 성능으로 Δ를 정하지 않습니다. 첫frame 이후 camera 이동은 남습니다. 원점을 camera 근처로 옮긴 조건이 실제 train 분포로 돌아왔다는 주장은 하지 않습니다. 전체 train camera 범위는 감사하지 않았습니다.

각frame에 원본 R을 유지하고 `encode_camera(camera,100mm)`의 실제18개 float32값을 저장합니다. Baseline 인코딩은 준비 tensor와 정확 일치해야 합니다. 공통 Δ에 대해 임의의 고정 world probe의 camera-space 좌표, 상대 stereo translation, center의 공통 이동을 float64로 검사합니다(최대 허용1e-6mm). Camera-space가 같으면 같은 fisheye projection도 같습니다. 실제 GT camera 변환은 바꾸지 않습니다.

Float32 translation roundtrip 오차를 별도 기록하고0.1mm 상한을 둡니다. 인코딩된 rotation6D의 세 번째 column을 cross product로 복원해 실제 float32 표현의 camera-space/stereo 잔차도 별도로 기록합니다. 수학적 불변성과 float32 양자화 잔차를 동일한 주장으로 취급하지 않습니다.

## 비교와 판정

Frontend 특징은 clip당1회 계산하고 네 번의 sampler 호출에 그대로 재사용합니다. Baseline full50/prefix25 검증도 유지합니다. Noise는 동일 sequence/frame 키에서 유도하고 매 호출 reset합니다. 모델 weight/dependency와 원래 inference implementation pin을 고정 결과와 정확 대조합니다.

이전 결과 baseline 비교는 3D `atol0.01mm + rtol1e-5`,2D `atol0.001native_px + rtol1e-5`로 max 차이와 일치 여부를 기록합니다. 다른 GPU·Torch 버전에서 생긴 이전 실행 불일치는 `reference_mismatch`로 남기며 새 baseline을 몰래 정답으로 대체하지 않습니다. 이전·현재 장치와 Torch도 함께 기록합니다. 불일치만으로 새 paired 수치를 버리거나 원인 검증이 성공했다고 주장하지 않습니다.

같은 실행의 zero_repeat가 고정3D tolerance를 넘으면 변형 판정은 `inconclusive_repeat_unstable`입니다. 안정된 경우 각좌표에서

`abs(variant − baseline) > 0.01mm + 1e-5 × abs(baseline) + 2 × abs(zero_repeat − baseline)`

를 넘는 수/전체좌표수와 비율을 기록합니다. Joint L2 변화의 sum/count/mean/max도 별도로 기록합니다. 이 기준을 넘는지로 `exceeds_repeat_aware_tolerance` 또는 `within_repeat_aware_tolerance`를 표시하며 단순히0보다 크다는 판정을 하지 않습니다. 네 가지 출력 whole/body/hand local/global을 모두 비교합니다. 큰 수치 변화가 개선을 뜻하지는 않습니다.

Δ0 반복은 변형된 camera 입력의 float32 양자화 차이를 통제하지 못합니다. 따라서 tolerance 초과라도 미세한 출력 변화만으로 절대 world 원점에 대한 구조적 문제를 확정하지 않습니다. 입력의 양자화·camera-space/stereo 잔차와 출력 변화의 실제 크기를 함께 검토합니다.

Variant마다 MPJPE·body 중심 오차·중심제거 진단·bone/spread와 frame별sum/count를 기록합니다. 중심제거 진단은 valid body관절 중심만 제거하며 회전·scale 정렬이 없는 보조 지표입니다. GT오차로 변환을 선택하거나 결과가 좋은 Δ를 추가 탐색하지 않습니다. 명목25FPS·stereo 시간동기·원본 motion 독립성 미확인, 한 인물 경로라는 한계를 유지합니다.

## 출력과 실행

신규 CLI는 `python -m rewind_v1.camera_gauge_cli`, 빌더는 `python -m scripts.build_v12_camera_gauge --source-zip ... --source-sha256 ... --revision ... --input-zip ... --reference-zip 도시ZIP --control-reference-zip 대조ZIP --output ...`입니다. 서로 다른 ZIP 파일명을 사용합니다. 기존 경로를 덮어쓰지 않습니다.

Notebook은 `RUN_GAUGE=False`, `DEVICE='cpu'` 기본값입니다. 명시적으로 실행하면 reference→고정목록→모델확인→준비→모든variant추론/렌더를 독립child로 호출합니다. 지원 CUDA GPU를 선택할 수 있고 모델명 제한·자동fallback은 없습니다. 각단계 timeout+종료여유30초 뒤 kill/wait하며 자동 재시도하지 않습니다.

| 단계 | 상한 |
| --- | --- |
| 두 reference 검증 |합600초; 각ZIP64MiB/unpack128MiB/100members |
| 고정 metadata |1800초/100,000entries/32MiB;900경로 stat만 |
| 모델 copy/hash |600초/1GiB |
| 원본 준비 |3600초/모든 RGB·pose read합2GiB |
| 전체 추론/Δ0repeat/prefix/metrics/render |합3600초 |
| 결과 파일합·ZIP |각256MiB |
| 각회수 packaging·copy/readback |각600초·600초 |

초기 free disk8GiB/Linux MemAvailable4GiB, 실행중 high-water RSS4GiB 관측 guard를 유지합니다. OS hard limit·cgroup headroom 보장은 아닙니다. 비용은 단계별 기록하며 원본 논리 읽기와 실제 네트워크 전송량을 혼동하지 않습니다.

선정후·준비후·최종종료(실패포함) 누적 보고서를 각각 새 Drive attempt에 자동 보관합니다. 정상시 최대3회 package/copy 비용이 별도 추가됩니다. 마운트·쓰기/readback을 먼저 확인하며 게시 실패를 재시도하지 않습니다. 작업 성공·보관 성공을 구분하고 실패시 로컬 ZIP/보고서를 보존합니다. VM 종료에는 finally를 보장하지 않습니다.

회수에는 모든24개 variant JSON,6개clip report,6개 비교 GIF와 각clip first/mid/last PNG,6개 준비 manifest, 고정목록·준비·최종report와 상태를 포함합니다. 전용 명시 whitelist로 기존 `range(4)`의 뒤2clip 누락을 방지합니다. Raw·tensor·모델·reference ZIP은 제외합니다.

GIF는 GT/baseline/midpoint/+100m을 함께 표시합니다. Zero repeat는 수치 기준으로 저장합니다. 전체50frame·모든 변형·유효GT에 하나의 공통3D 축을 적용하고 실제 device xyz min/max(mm)를 report·화면에 표시합니다. +100m 변형의 큰 오차가 축을 지배해 다른 몸이 작게 보일 수 있으므로 수치표를 함께 봅니다. 이미지 좌우/2D는 변형 간 동일합니다. 종료후 보관을 확인하고 표와6GIF를 표시하며 `SHOW_SAVED_RESULTS=False`는 저장 결과만 다시 표시합니다.

## 검증 범위

2026-11-15 개발28검사가 통과했습니다. Root 독립24검사에서는23개 통과, 기존 control positive의 임시폴더 rename에서 Windows `WinError 5` 1건이 발생했습니다. 제품 수정 없이 해당검사를 단독 재실행해 통과했습니다(50.251초). 최초 실패 로그도 보존하며,24개가 한 번에 통과했다고 표현하지 않습니다. Root는 실제 두reference·300camera행·24조건 변환, 합성 비교 PNG, 생성 notebook 코드와 문서 링크를 추가 검토했습니다. 근거는 `outputs/v12_camera_gauge_20261115/`의 개발·root 로그와 보고서입니다.

로컬 검증은 두 실제고정reference의 키/pin 연결과 변조 거부, 비자명 회전·공통translation의 stereo/투영/center 불변성,100m 단위·첫frame만사용·float32, 반복 tolerance 및 이전실행불일치 분리, defaultoff·childtimeout·중간보관, 합성6clip의전체준비/24개 full50예측·6개 prefix25검증/6GIF/회수/가상Drive readback, wrong expected SHA 거부, 순서변조 거부와 기존 cross/control 회귀입니다. 합성 tiny 모델은 실제 성능 증거가 아닙니다. 실제 Colab/GPU/Drive와 민감도 결과는 실행 후 별도 검토합니다.
