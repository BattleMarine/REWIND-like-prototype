# ArchVis 1,000페어 v0 학습

2026-11-09 실제 결과: T4에서 동일1,000 train·2,236 validation으로20epoch/10,000update를 완료했다. 화면 안 whole 오차335.29→195.31px, 전체 약84분 중 캐시 생성77분·학습4분44초였다. 종료 기록 wrapper 오류는 학습 완료와 구분하며 재학습 없이 수정한다. 정확도·독립성·3D 연결은 남아 있다. [결과와 근거](ARCHVIS_V0_RESULTS.md).

2026-11-09 사용자 요청에 따라, 운영 진단에 사용한 동일한 train 1,000페어로 2D 모델의 개발 학습을 진행한다. 공식 validation 2,236페어는 평가와 best 모델 선택에만 사용한다. Test는 사용하지 않는다. 실제 GPU 실행은 완료됐으며 아래에는 재사용할 실행 계약을 유지한다. 성능과 종료 오류는 위 결과 문서를 기준으로 한다.

## 모델과 입력

기존 tiny 모델은 보존한다. 새 `context_heatmap_v0`는 64×64 입력을 단계적으로 축소하고 전역 문맥과 skip 연결을 이용해 32×32 관절 heatmap을 만든다. 몸·손 62관절, 좌우 영상 각각의 2D 예측이며, 이 단계만으로 3D REWIND-like 프로토타입이 완성되는 것은 아니다.

- Adam, learning rate `1e-4`, batch 2, FP32, seed 71.
- Gaussian cross-entropy와 기존 좌표 변환·visibility 처리 유지.
- 최대 20 epoch, 10,000 update. 초기와 1·5·10·15·20 epoch에 validation 평가.
- 학습된 epoch 중 화면 안 whole 관절 평균 오차가 가장 낮은 checkpoint를 best로 선택한다. 동률이면 먼저 평가한 모델을 유지한다. 초기 모델 대비 개선 여부와 몸·손별 결과도 별도로 확인한다.

검증된 manifest와 준비 보고서를 복원하므로 pose 준비 검사를 처음부터 다시 수행하지 않는다. 다만 실제 학습에 읽는 RGB·pose는 원본 파일 hash와 decode를 확인한다. 동일한 resize·target 결과를 Colab 로컬 임시 캐시에 저장하고, 이후 epoch에서는 이 캐시를 재사용한다. 캐시 파일도 읽을 때 hash를 검사하며 런타임을 재생성하면 다시 만든다.

## 실행과 결과 보존

고정 notebook·기존 source ZIP·inputs ZIP·prepared ZIP과 계획 SHA를 테스트 세션이 안내한다. 기존 source와 새 모델 overlay는 별도 식별하며 정확한 파일·revision·SHA는 해당 인계 receipt를 기준으로 한다. 실행 도중 코드나 데이터 선택을 바꾸지 않는다.

학습 subprocess는 캐시 생성부터 최대 2시간이며, 내부에서는 118분에 협조적 중단을 시도한다. 임시 캐시 384MiB, 결과 96MiB, RSS 6GiB, Torch CUDA allocated 4GiB·reserved 6GiB 상한을 적용한다. 원본 읽기 상한은 고정 manifest의 선택 파일 크기 합에서 산출한다. 이 값은 순수 네트워크 전송량과 다르다. Drive mount와 결과 다운로드 시간은 학습 subprocess 측정에서 제외한다.

100 update마다 `last.pt`를 저장한다. 정상 저장 사이의 중단은 최근 최대 99 update를 다시 수행하며, 경계 checkpoint 쓰기 자체가 실패하면 해당 경계 update까지 다시 수행할 수 있다. 완료되지 않은 평가도 같은 학습 지점에서 다시 수행한다. 회수 ZIP에는 마지막 상태, epoch/평가 기록, best 모델과 대표 예측 이미지, 설정·입력 식별값을 포함한다. 같은 계획과 환경을 유지한 재개는 회수 ZIP의 hash를 확인해 진행하며, 변경된 코드·설정으로 이어 붙이지 않는다.

## 결과 판단

20 epoch 완료 여부, epoch별 고유 train 1,000페어 순회, 유한한 loss·gradient와 저장/회수 상태를 먼저 확인한다. 정확도는 초기 모델·중앙 baseline 및 이전 CE 1 epoch와 비교한다. 새 모델의 처리 시간은 캐시 생성·학습·평가·저장으로 나누어 보고한다. 이전 tiny 모델의 약 0.143초/페어를 새 모델의 속도로 간주하지 않는다.

Validation의 원본 motion 독립성과 실제 영상·pose의 시각적 정합은 아직 미확인이다. 이 결과로 일반화 성능이나 전체 ArchVis 학습 준비 완료를 선언하지 않는다. 대표 예측 이미지는 64px 입력을 확대해서 표시한 모델 출력이며 정답 시각화와 구분한다. 실제 결과 검토 후 ArchVis 전체 확대와 예측 2D→3D 연결을 결정한다.

## 로컬 검토

Root가 새 경로의 4개 검사와 기존 RGB 학습 경로의 20개 검사를 확인했다. 캐시와 원본 전처리 동일성·변조 차단, 전역 문맥, 학습 중간/평가 경계 재개와 회수된 best 모델의 예측 재현을 검증했다. 기존 준비 3개 ZIP을 이용한 notebook 복원·overlay 설치·별도 프로세스 CLI import도 로컬에서 확인했다. 이는 Colab GPU 실행 검증과 다르다.

개발자가 보존된 실제 2페어로 CPU에서 100 update를 수행한 결과 CE 7.019→6.792, 같은 페어의 화면 안 평균 오차 341.39→330.41px였다. 학습 신호는 있으나 중앙 집중은 남아 있으며, 일반화 또는 본 1,000페어 결과가 아니다. 근거는 `outputs/archvis_v0_20261109/`에 보존한다.
