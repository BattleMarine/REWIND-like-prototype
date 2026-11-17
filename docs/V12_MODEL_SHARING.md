# v1.2 모델 공유 패키지

2026-11-17 사용자 요청: 학습된 v1.2 모델과 수정 가능한 소스, 실행 방법, Git 저장소 주소를 하나의 ZIP으로 전달한다. 새 학습·파인튜닝 실행이나 공개 업로드는 포함하지 않는다.

## 공유 기준

- 저장소: https://github.com/BattleMarine/REWIND-like-prototype
- 주소는 프로젝트의 `origin` 설정에서 확인했다. 원격 접근 권한·최신 커밋 공개 여부를 확인하거나 push하지 않았다.
- 실행 기준은 ZIP에 동봉한 고정 소스다. 로컬 커밋이 원격에 없을 수 있으므로 원격의 최신 코드를 내려받아 동일 결과를 보장한다고 안내하지 않는다.
- 원본은 `outputs/v12_training_release_20261112/user_test/completed_001/training.zip`, SHA256 `48661e75e0969ca140ab9cf0e543d73d2bb3b9ec108d30652cf7fb4f4b76b506`이다.

| 모델 | Epoch | 원본 bytes | SHA256 |
| --- | ---: | ---: | --- |
| Frontend |9|1,739,127|`19363625849d7fc72232cf3af7e438e0c3c594eb52f17c7c15d5a44296153f38`|
| Body |10|60,196,015|`94a072551fd3da40621da5a589fbc3d6f919d3e7b52d29ea1c7f6099ef681305`|
| Hand |9|62,446,567|`d6c610ead412cc0495d4adf99ed3872e8c434ebfabecb2e9f551f5c1eb766f8f`|

원본 checkpoint bytes와 frontend→body→hand 의존 관계를 보존한다. 세 모델의 합은124,381,709bytes다. 중간 모델·optimizer 재개 상태·원본 RGB·pose·대용량 입력 캐시는 패키지 필수 구성에서 제외한다. 학습 결과와 알려진 한계는 [v1.2 결과](V12_TRAINING_RESULTS.md)를 따른다.

## 실행과 수정 범위

압축 안의 `README_KO.md`가 설치·검증·추론 API 사용의 진입점이다. 합성 입력 smoke는 원본 모델을 로드하고 전체 추론 경로가 유한한 출력을 내는지 확인하며 실제 포즈 정확도를 보여주는 예제가 아니다. 실제 입력에는 좌우 RGB뿐 아니라 올바른 world→camera metadata가 필요하다. GT 관절을 추론 조건에 넣지 않는다.

v1.2는 기존 absolute camera 표현을 유지한다. v1.5 중점 표현을 임의로 혼합하지 않는다. 다른 시나리오와 원점 변화에 민감하고 손 축소가 남아 있으며, 독립 일반화·RGB-only 카메라 추정·임의 단안 영상 지원 모델로 배포하지 않는다.

소스는 수정할 수 있지만 기존 runner의 resume는 동일 실행 복원용이다. 새 파인튜닝은 가중치 초기화·새 optimizer/scheduler·새 실험 ID를 관리하는 별도 warm-start 연결이 필요하며 이 패키지 제작으로 구현 완료되지는 않는다. 디퓨전의 예측 목표·noise schedule·network shape를 바꾸면 호환성 검토와 재학습이 필요하다. 가중치를 억지로 로드하거나 기존 binding 검사를 해제하는 방식으로 안내하지 않는다.

## 배포·검증 기록

최종 ZIP과 크기·SHA·고정 소스 revision·검증 기록은 Git 제외 `outputs/v12_model_share_20261117/`에 보관한다. 원본 결과 및 진행 중인 v1.5.1 자료는 변경하지 않는다. 실제 GPU·새 환경 설치·실데이터 파인튜닝 검증과 로컬 CPU smoke 검증을 구분한다.


실제 개발용 독립 추출본에서 `source/`를 작업 디렉터리로 두고 `python -m scripts.v12_share_smoke --package-root .. --device cpu --frames 2`를 실행해 세 원본 모델 strict load와 전체 추론 경로를 확인했다. Whole `[2,62,3]`, body `[2,22,3]`, hand local/global `[2,40,3]`가 모두 유한했다. 기존 로컬 Python3.12.10·Torch2.12.1+cpu 환경의 검사이며 새 환경 설치·CUDA·실데이터 정확도 검증은 아니다. 최종 Git 소스 배포본의 별도 확인 결과는 같은 산출물 폴더의 `root_release_review.json`을 따른다.
