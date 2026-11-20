# REWIND v1.2 오프라인 시연 데이터

로컬에서 원본이 확인된 ArchVis Serena `Blow_A_Kiss__14_`의 연속 프레임 0–49를 사용했다. 50페어 모두 공식 train에 속한다. 고정 v1.2 학습 manifest에 같은 프레임의 좌우 RGB와 pose가 있으며, 로컬 파일 150개의 SHA-256이 학습 manifest와 일치한다. 이 표본은 학습 데이터에서 뽑은 시연이며 독립 평가로 사용하지 않는다. 기존 Joyful Jump의 영상 원본은 지정된 Downloads 경로에서 확인되지 않아 보존된 다른 ArchVis 클립을 선택했다.

사용한 모델은 원래 완료 학습 ZIP에 있는 best frontend 9, body 10, hand 9다. 기존 연속 클립 추론 함수의 FP32·DDIM 10스텝·seed 71·body 및 hand noise offset 100·200을 사용한다. 모델에는 원본 RGB와 pose JSON의 절대 stereo 카메라 메타데이터가 들어가며 손 모델은 예측 상체에 조건화된다. GT 관절은 추론 뒤 비교 표시용으로만 변환한다. 파일 번호 0–49의 좌우·pose 대응과 PNG decode를 확인했다. 표시용 25 FPS는 명목값이다.

## 발표에서 열 파일

- [오프라인 재생 화면](../outputs/presentation/v12_offline_demo_20261120/playback.html)은 HTML 하나에 표시용 이미지와 관절 좌표를 담는다. 파일을 브라우저로 열면 재생·일시정지, 배속, 프레임 이동, GT 비교, 3D 회전을 사용할 수 있다. 민트색은 예측, 주황색은 GT다.
- [발표용 MP4](../outputs/presentation/v12_offline_demo_20261120/v12_archvis_demo.mp4)는 같은 시퀀스를 GT 없이 렌더링한 H.264 영상이다. 1280×674, 50프레임, 명목 25 FPS로 2초간 재생된다. 슬라이드에서는 반복 재생을 설정할 수 있다.
- [화면 미리보기](../outputs/presentation/v12_offline_demo_20261120/preview.png)는 12번 프레임이다. 표시용 RGB는 축소·JPEG 인코딩했다. 추론에는 원본 PNG를 사용했다.

재생에는 Python, GPU, 서버 연결이 필요 없다. 발표용 파일은 프로젝트의 로컬 `outputs/presentation/`에 보관한다. 화면 빌더는 모델을 로드하지 않으므로 v2에서도 동일한 프레임·좌표 형식의 결과를 공급할 수 있다. 기존 화면의 3D 좌표는 전체 클립에서 고정한 중심과 스케일로 투영하며 회전만 조작한다.

## 데이터와 재생 화면의 계약

`outputs/presentation/v12_offline_demo_20261120/demo.json`의 `frames`는 파일 인덱스 순서다. 각 `left`·`right`는 1024×1024 원본 PNG의 JSON 기준 상대경로다. `xy`와 `gt_xy`는 좌·우 순서의 원본 왜곡 영상에서 왼쪽 위가 원점인 픽셀 좌표다. `xyz`와 `gt_xyz`는 고정 모델 관절 순서의 device-relative mm다. GT 결측 관절은 `null`이다. 관절 62개와 edge 61개는 v1.2 고정 spec 기준이며 `edges`는 0부터 시작하는 인덱스 쌍이다. `confidence`는 좌우 순서의 모델 원본 신뢰도다.

기존 CPU 결과의 `demo.json` SHA-256은 `6dcfee425e2c56c12b30043d2c40ee08b01126e3b54b4440076c7df4fba24159`다. 좌우 100 PNG를 바이트 그대로 저장했다. 결과에 모델 및 입력 SHA가 포함된다.

## 재생성

프로젝트 루트에서 실행한다. `--device cpu`가 기본이다. CUDA를 지원하는 Colab 환경에서는 같은 소스와 로컬에서 접근 가능한 입력 경로를 사용하고 `--device cuda`로 실행한다. GPU 모델명 제한은 없다. CUDA 장치가 없으면 오류로 종료한다. 새 결과는 실제 사용 장치와 Torch 정보를 기록하며 CUDA의 경우 장치명, VRAM, compute capability와 CUDA 빌드도 기록한다. 기존 발표 파일과 별도의 새 출력 경로를 지정한다.

```powershell
$clip = "ArchVisInterior_ArchVis_RT/Day/rp_serena_rigged_004_ue4/SKM_MenReadingGlasses_Shape_01/017/Blow_A_Kiss__14_"
$rgb = ".local-archive/20261111_outputs/outputs/test_rgb269_pairs_001_20261103/rgb269_pairs_001_result.zip"
$pose = ".local-archive/20261111_outputs/outputs/rgb269_prepare_20261103/root_pose_audit/pose269/files/$clip/json"
python -m scripts.export_v12_offline_demo --training outputs/v12_training_release_20261112/user_test/completed_001/training.zip --spec outputs/v2_plan_inputs/reference/spec.json --split-dir outputs/v2_plan_inputs/reference/split --audit outputs/v12_training_release_20261112/root_bootstrap/uploads/reports.zip --rgb-zip $rgb --rgb-zip-sha256 41f4e3ad306c9788c6d91d7db75422ae804f3d2d334cb0b0d8a51b834297d4db --pose-dir $pose --sequence $clip --start 0 --frames 50 --threads 2 --device cpu --output outputs/presentation/v12_offline_demo_rerun
```

압축을 푼 원본을 사용하려면 `--rgb-zip`과 `--rgb-zip-sha256` 대신 `--rgb-dir <fisheye_final_image>`를 지정한다. `--pose-dir`에는 같은 sequence의 JSON 폴더를 준다. `--audit`이 있으면 고정 v1.2 학습 manifest에 대해 대상 프레임, split 및 파일 크기·SHA를 다시 확인한다. GT를 표시하지 않는 결과는 `--omit-gt`를 사용한다.

새 추론 결과로 단일 HTML을 만들려면 다음 명령을 실행한다. 새 결과의 좌우 이미지 해시를 대조하고 화면용 사본을 HTML에 삽입한다.

```powershell
python -m scripts.build_pose_playback outputs/presentation/v12_offline_demo_rerun/demo.json outputs/presentation/v12_offline_demo_rerun/playback.html
```
