# v1 중앙 집중·3D 붕괴 원인 검토

2026-11-10. 대상은 `213b532` 코드와 V1-DIAGNOSTIC-001 저장 모델이다. [실제 학습 결과](V1_DIAGNOSTIC_001_RESULTS.md), [시각화](PREDICTION_OVERLAY.md)를 따른다. 학습 코드·checkpoint 변경이나 optimizer update 없이 진단했다.

## 판단

확인한 경로에서는 단위·좌표 복원·손목 연결·체크포인트 로드 오류를 발견하지 못했다. 가장 강한 근거는 **거의 균일한2D heatmap과3D 모델 자체의 미학습 상태**다. 학습량 부족이 유력하지만, 학습을 늘리기만 하면 해결된다는 인과 검증은 아직 없다. 카메라 조건에 치우친 반응 등 설계·최적화 문제도 추가 분리해야 한다. 모든 구현이 올바르다는 판정이나 REWIND 논문 재현 검증은 아니다.

이번 진단은 고정 선택 밖 Getting_Up__1_의 frame1·10, 각각T1 reset이다. 원본 motion 독립성 미확인, 명목25FPS, validation 참고용 조건을 유지한다. 단일 저장 시점과2장으로 전체 학습 원인을 확정할 수 없다.

## 2D: 중앙 좌표가 만들어지는 직접 경로

`StereoHeatmapNet.forward`는64×64 logits를 공간 softmax로 바꾸고 좌표의 기댓값을 출력한다. 확률이 균일하면 평균은 원본1024×1024의(511.5,511.5)다. 중심을 강제로 대입하는 코드는 없다.

| 두 로컬 프레임 실측 | 값 |
| --- | ---: |
| 학습 모델 heatmap 엔트로피 / 최대 엔트로피 | 99.267% |
| 평균 최대 확률 | 0.0005304 |
| 균일 분포의 확률 | 0.0002441 |
| 기댓값 좌표의 화면 안 GT 오차 | 315.43px |
| Argmax로만 읽었을 때 오차 | 347.38px |
| 모든 관절을 중앙에 둔 기준 오차 | 341.12px |
| BatchNorm을 현재 batch 통계로 계산한 진단 오차 | 313.29px |

훈련 중 평균 CE는8.2580이고 같은 격자 균일 분포의 CE는8.3178이다. 이는 epoch 평균이며 마지막 batch 손실과 혼동하지 않는다. 위치 분포가 아직 날카롭게 학습되지 않았다는 관찰과 일치한다. Argmax 치환은 이2장에서는 개선되지 않았고 BatchNorm 상태만 바꿔도 중앙 집중이 해소되지 않았다.

실제RGB를0으로 바꾸면 좌표 변화 평균24.25px에 그쳤다. 해당 입력 변화에 대한 반응은 있으나 충분한 관절 구분의 증거는 아니다. 저장 모델 복사본에서 실제2장 CE backward를 실행한 결과283개 파라미터 tensor 모두 유한·비영 gradient가 있었고 optimizer update는0이다. BN count500과 학습 기록은500회 frontend update를 뒷받침한다. 과거 모든 step의 gradient 크기를 복원한 것은 아니다.

## 3D: 출력 자체가 관절 구분을 잃음

| Body 실측 | frame1 | frame10 |
| --- | ---: | ---: |
| 예측 평균 뼈 길이 | 3.50mm | 3.36mm |
| 정답 평균 뼈 길이 | 199.31mm | 199.31mm |
| 중심 제거 후 예측 좌표 RMS spread | 1.70mm | 1.62mm |
| 정답 RMS spread | 216.85mm | 233.21mm |
| 원래 예측2D 입력의 body MPJPE | 487.20mm | 479.99mm |
| GT2D xy만 치환한 진단 body MPJPE | 487.42mm | 480.45mm |

GT2D 진단은 좌표만 바꾸고 confidence 등 나머지 특징은 유지했다. 완전한 oracle 입력의 성능 상한이 아니다. 그래도 뭉침이 회복되지 않아 frontend오류 단독 설명은 지지하지 않는다. DDIM1/10/25/50에서도 body오차는약467~487mm이며, GT에 매우 약한 노이즈를 더한k=1 입력에서 직접x0 head를 읽어도약470~475mm다. Sampler 반복 수만 늘려 해결되지 않았으며 denoiser 자체의 복원이 미흡하다.

학습은 frontend500회, body20회, hand20회로 합540회다. 1,000프레임을50frame clip20개로 처리했기 때문에3D 각 모델이1,000번 업데이트된 것이 아니다. Teacher timestep도 각 clip update에서 batch별 하나를 뽑는다. 마지막 hand의 Adam state103개 모두 step20·비영1차 모멘트를 가진다. Body/hand 파라미터도 초기화 대비 바뀌어 optimizer가 전혀 동작하지 않은 상태는 아니다. Step별 loss곡선·gradient norm, 이전body optimizer 상태는 현재 회수본에 없어 수렴 경과를 단정할 수 없다.

Raw cm→mm는spec의×10, 학습 target은/100, x0 예측은×100으로 일관된다. 손은 손목 상대 target으로 학습하고 예측body 손목을 더해 전역 위치를 복원하며 joint순서도 일치한다. DDIM은x0 학습과x0 복원 계약이 맞고 최종 출력은 모델head 값이다. 렌더러는 GT와 같은 축척을 사용하며 raw tensor에서도 뭉침이 확인된다.

## 추가 의심점과 다음 분리 실험

같은 카메라·noise에서 두 프레임의2D 특징을 서로 바꾸면 body는약0.12~0.14mm만 변했고 카메라만 교환하면약159mm 변했다. 카메라 입력은 세계→카메라 변환이고 target은장치 좌표다. 이런 조건 의존성이 현재 이미지·관절별 신호보다 강하게 나타나지만, 이2장만으로 카메라 부호 오류나 잘못된 좌표 설계를 확정할 수 없다. 조건별 분리 실험 대상이다.

다음 권장 순서는①고정 소수train 프레임에서2D overfit 및heatmap 집중도·CE·gradient 경과 확인, ②고정train clip에서3D overfit과관절spread·bone길이·loss곡선 확인, ③같은 조건에서예측2D/GT2D·카메라 conditioning을 한 변수씩 비교하는 것이다. 작은 표본도 외우지 못하면 epoch확대보다 구현·손실·정규화 검토를 우선한다. 작은 표본학습이 통과한 뒤에만 동일1,000페어 학습량을 늘리는 것이 적절하다. 이 권고는 후속 학습 실행 승인이 아니다.

## 검토 근거

Git 제외 `outputs/archvis_v1_diagnostic_001/collapse_audit/`의 `frontend_probe.py/json`, `frontend_gradient_probe.json`, `developer/probe.py/json`, `root_3d_probe.json`에 보존했다. Root는2D진단과gradient검사를 직접 실행하고3D개발probe를 독립 재실행해 JSON 결과가 정확히 일치함을 확인했다. 모델·원본데이터·학습코드는 변경하지 않았다.
