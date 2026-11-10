# v1.1 학습량 확대 결과 — 모델 파일 미회수

2026-11-10. 실행 소스 revision `18c4fb2d3900d9f65ad6bad733ad5e3bce9f5a8f`. 동일 train1,000·frontend/body/hand 각10epoch 진단. 명목25FPS·stereo frame-key 가정과 원본 motion 독립성 미확인을 유지하며 validation은 참고 지표다.

## 확인한 근거

- 상태 ZIP `outputs/v1_branch_experiments/user_test/v11/v1_epoch_expansion_status-01.zip`: 2,998,595bytes, SHA256 `aff74e15ad9b631b646839705b118c88366bc7f933507db7538077612a8041b4`. Root가 CRC와 receipt의2개 파일 크기/해시를 확인했다. Notebook 종료코드0이다. ZIP에는 모델이나 run.json 본문이 없다.
- 사용자 출력 원문을 같은 폴더의 `user_run_output.txt`에 보존했다. SHA256 `531bdd07bb84d86e38098d91f2c7539b052c00d4cab62af106351fedda65926b`. 파싱한 `user_run_report.json`은 출력 일부의 보존본이며 원본 run.json 전체나 checkpoint 대체물이 아니다.
- 출력 기록은 completed, failure=null, 5,400update, 단계별10epoch, 경과5,311.205초(88분31초)다. 학습 완료의 로그 근거이며 checkpoint 본문 검증 완료는 아니다.
- 테스트 세션 보고: training ZIP 683,078,890bytes는 생성됐으나 다운로드 실패 후 런타임 종료로 본문을 확보하지 못했다. 표시 SHA `e92782a5ace3c3865a97d293bb626b7c0aedae28c526fb6856c50093a1e813ba`는 검증되지 않았다. last.pt 표시 SHA `c7e31cf2bda481b710e67778180f63b56db10e5e6a503b98b51495169d629bcb` 역시 본문 검증이 아니다.

## 사용자 출력에 기록된 지표

| 모델·지표 | 이번 실행 epoch1 | 최저 오차 | epoch10 |
| --- | ---: | ---: | ---: |
| Frontend 화면 안2D(px) | 307.733 | 205.273(epoch10) | 205.273 |
| Body MPJPE(mm) | 543.038 | 481.442(epoch9) | 557.686 |
| Hand global MPJPE(mm) | 560.938 | 512.603(epoch8) | 562.917 |

Frontend의 첫32 validation frame에서 normalized heatmap entropy는0.99383→0.83547, 중앙으로부터 예측 거리 평균은50.31→216.11px다. 오차 감소와 함께 중앙 집중 완화의 로그 근거가 있으나 모델 파일을 이용한 새 시각 검증은 못 했다.

Body epoch10의 평균 bone 길이는21.06mm(GT208.35mm), 중심 기준 관절 거리 RMS는66.02mm(GT516.72mm)다. 손의 wrist-relative fingers 범위는 bone2.31mm(GT25.78mm), spread2.20mm(GT72.76mm)다. 형태 축소는 여전히 크며 최저점 이후 오차가 악화됐다. 이를 로그만으로 과적합이나 구현 오류 중 하나로 확정하지 않는다. 이번 teacher는 개선된 frontend 조건을 사용하므로 이전1epoch 실행과 비교해 teacher 자체 학습량의 효과만 분리할 수 없다.

## 복원과 다음 작업

1. 기존 다운로드/영구 보관본이 있다면 위 표시 해시와 실제 파일을 대조해 회수한다. 별도 복사본이 없고 런타임 저장소가 사라졌다면 해시나 로그만으로 가중치를 복원할 수 없다.
2. 현 상태에서는 이 모델의 추론·시각화·재개를 제공하지 않는다. 재학습을 자동 시작하지 않는다.
3. 다음 학습 전에 checkpoint 또는 회수 ZIP을 사용자 Drive에 영구 보존하고 복사본 크기/해시까지 확인하는 절차를 마련하는 것이 우선이다. 캐시 업로드는 기존대로 사용자 담당이며 새 자동 업로드 기능이나 학습은 이번 문서화에 포함하지 않았다.
4. v1.2 별도 런타임의 최종 결과는 아직 미확인이다. 이 결과로 v1.2 상태를 바꾸지 않는다.
