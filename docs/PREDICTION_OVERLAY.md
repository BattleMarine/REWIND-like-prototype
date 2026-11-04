# 저장 모델의 RGB 관절 예측 표시

2026-11-04 발표 자료 준비용으로 `scripts/predict_stereo_overlay.py`를 추가했다. 기존102쌍 진단 `StereoHeatmapNet` weights를 고정 SHA와 계약으로 확인하고 stereo PNG 한 쌍에서62개2D관절·confidence를 추론한다. 정답 입력·추가 학습·GT fallback은 없다. M2/M3 또는 학습 완료된 REWIND 전체 모델의 시연이 아니다.

사용자가 요청한 `ArchVisInterior_ArchVis_RT/Day/rp_adanna_rigged_003_ue4/SKM_MenReadingGlasses_Shape_01/001/Joyful_Jump__16_`는 Drive에서 폴더 경로를 확인했으나 좌우 RGB 파일은0개로 조회됐고 사용자가 업로드 중임을 확인했다. 해당 시퀀스 다운로드·추론·발표 이미지 생성은 아직 수행하지 않았다. 다른 actor의 검증 이미지를 해당 동작의 결과로 사용하지 않는다.

## 실행 계약

Python 환경에는 기존 `requirements-m1.txt`, `requirements-rgb.txt`의 관련 의존성이 필요하다. Windows 검증에는 기존 Python3.12.10/Torch2.12.1+cpu/Pillow12.2.0 환경을 사용했다. 프로젝트 `.venv`는 현재 PIL이 없어 root의 첫 실행이 import 단계에서 실패했으며 시스템 Python의 기존 설치 환경으로 독립 실행했다. 새 패키지는 설치하지 않았다.

프로젝트 루트에서 다음 CLI 인자를 사용한다. 아래 좌우 파일은 실제 확보한 같은 frame의 PNG로 지정하고 출력은 새 폴더로 지정한다. 좌우 동기화는 CLI가 자동 판정하지 않는다.

```text
python scripts/predict_stereo_overlay.py --left LEFT.png --right RIGHT.png --weights outputs/rgb102_diagnostic_20261103/developer/run/frontend_weights.pt --training-manifest outputs/rgb102_diagnostic_20261103/developer/run/input_manifest.json --sequence SEQUENCE_ID --frame FRAME_NUMBER --output NEW_OUTPUT_DIRECTORY
```

출력은 `predictions.json`과 `overlay.png`다. 좌우 원본/예측4개 패널, 진단 모델명·미검증 표시·frame·입력/weights hash를 포함한다. JSON은62관절의 이름·native/network 좌표·confidence·출처·전처리·환경·스크립트 hash를 보존한다. Confidence는 공간 softmax 최댓값이며 보정된 정확도 확률이 아니다. Skeleton edge는 기존 프로젝트 가정이고 native rig parent 검증 완료를 의미하지 않는다.

64×64 bilinear resize(`align_corners=True`, `antialias=False`)와 `(native_size-1)/63` 좌표 복원을 기존 진단과 동일하게 사용한다. CPU `eval`/`no_grad`로 실행하며 weights 변경 여부를 검사한다. 입력 파일당16MiB/4메가픽셀 이하 PNG 한 쌍을 처리한다. 외부 네트워크·Drive 접근은 이 CLI에 포함하지 않는다.

## 검증과 한계

개발은 기존102쌍의 Burpee frame0·1을 재추론해 저장 cache와 native 좌표·confidence 차이가 모두0임을 확인했다. Root는 실제 CLI로 frame0을 독립 실행하고 개발 결과와 좌표·confidence 동일, GT 사용 없음, 학습 update0을 확인하고 패널을 시각 검토했다. 근거는 `outputs/joyful_jump_presentation_20261104/developer/validation.json`과 `root_validation/review.json`이다.

현재 두 검증 프레임에서 관절들이 화면 중앙에 집중되고 confidence도 약0.001 수준이다. 성공적인 관절 추정 사례로 소개하기 어렵다. 이번 검증은 기존 모델 출력의 충실한 재현과 시각화에 대한 것이며, 새로운 시퀀스의 정확도나 발표용 품질을 보장하지 않는다. 실제 Joyful Jump 결과는 업로드 완료 후 별도로 판단한다.
