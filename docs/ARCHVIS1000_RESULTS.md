# ARCHVIS1000-001 실제 학습과 wall-clock 검토

2026-11-09 root 검토. 준비 결과를 새 Colab runtime에 복원한 뒤 MSE·Gaussian CE 비교가 각각 완료됐다. 학습 source는 `ceb8591a3c2c45549d04f9fda6cab871ecd47ccf`, 복원 wrapper는 `30aac86`이다. A10040GB, FP32, B2,64×64, Adam lr1e-4, seed71, 동일 train1,000·공식 validation2,236, loss만 다르다. 원본 motion 독립성과 실제 RGB-pose 시각·시간 정합은 미완료다.

## 근거와 실행 확인

회수 ZIP: `outputs/archvis_pilot_handoff_20261109/user_test/restore_run/archvis_pilot_recovery.zip`,951942bytes, SHA256 `d5aa05611e7ce6b77572bd7bdd6b3cdb72c811444be563c7f66a9ca929a7cb24`. Root는 ZIP CRC·receipt18파일 bytes/SHA, manifest canonical hash, checkpoint/weights 파일SHA와 tensor 유한성·두 artifact의 모델 상태 일치를 독립 확인했다. 두 실행 모두 completed, 실제500update/500attempt/고유train1,000·1epoch·nonzero gradient500, AMP skip0이며 초기·최종 공식 validation2,236 평가를 완료했다. 이는 운영 검증 통과이며 정식 성능 승인과 다르다.

계산 근거는 같은 폴더의 `root_wall_clock_review.json`에 보존했다. Drive 영구 보관은 사용자 확인 전이다. 체크포인트의 파일·상태 검사는 이번 GPU 실행의 무중단 대비 재개 동등성 시험을 뜻하지 않는다.

## 실측 시간

| 구간 | MSE | CE |
| --- | ---: | ---: |
| 초기 train+validation 평가 | 469.394초 | 465.885초 |
| train1,000페어 학습 | 143.115초 | 142.182초 |
| 최종 train+validation 평가 | 474.377초 | 470.467초 |
| 프로세스 보고 elapsed | 1094.875초 | 1081.828초 |
| 학습 구간 초/페어 | 0.143115 | 0.142182 |
| 학습 구간 페어/초 | 6.987 | 7.033 |

합산 elapsed는2176.702초(약36.3분)로 사용자의 약36분 wall-clock 관측과 부합한다. 각 프로세스 elapsed는 모델 초기화·데이터 읽기·학습·평가·저장 대부분을 포함하나 Colab 업로드·사용자 대기·Drive mount·최초 준비·결과 다운로드는 포함하지 않는다. 준비3517.166초는 별도다.

학습 구간은 순수 GPU kernel 시간이 아니라 입력 읽기·PNG decode·resize·forward/backward/update를 포함한다. checkpoint와 기타 비용은 구간 합과 elapsed의 차이로 MSE7.989초·CE3.294초이며 학습량과 함께 달라질 수 있다. 초기·최종 평가 합은 각각3236페어를 두 번 처리한 시간으로, 평가 처리량의 분모는6472페어다. 평가가 실행시간의 약86%를 차지했다. 이 사실만으로 Drive I/O가 병목이라고 확정하지 않는다. cache hit/cold 상태와 순수 전송량은 별도 계측되지 않았다.

RSS 관측 peak는약2.02GB다. Torch CUDA allocated peak는 MSE7.59MB·CE6.15MB, reserved29.36MB이며 전체 VRAM 사용량·GPU 이용률이 아니다. 현재 작은 CNN 결과를 큰 모델에 그대로 적용하지 않는다.

## ArchVis 전체 조건부 추정

목록상 중복1키를 제외한 train20,888·validation2,236를 가정한다. 같은 모델·batch·해상도·GPU·저장/캐시 조건에서 평균 학습0.142649초/페어, 평가0.145250초/페어를 선형 적용한다. 현재 코드는 전체 실행의 처음과 끝에만 train+validation 전량 평가한다.

`예상 초 = epoch × 20888 × 0.142649 + 2 × (20888+2236) × 0.145250`

| epoch | 학습 구간 | 초기·최종 평가 포함 |
| --- | ---: | ---: |
| 1 | 약49.7분 | 약2시간42분 |
| 10 | 약8시간17분 | 약10시간9분 |
| 30 | 약24시간50분 | 약26시간42분 |

이는 한 loss 실행의 조건부 추정이며 새 데이터 준비·저장 부대비용·재시작·추가 평가·네트워크 변동은 제외한다. MSE·CE를 둘 다 전체로 돌리면 두 실행을 합쳐야 하지만 이번 결과로 그 실행을 승인하지 않는다. 성능 최적화나 모델/해상도 변경 후에는 다시 측정한다. 확률적 오차 범위나 보장 상한이 아니다.

## 성능과 확대 판단

Validation 전체 유효 관절의 native1024 좌표 평균 오차는 초기360.593px, 최종 MSE360.327px·CE359.584px, 중앙 고정 baseline360.405px다. CE의 초기 대비 개선은약1.009px, 중앙 baseline 대비 개선은약0.821px에 그친다. 화면 안 whole 오차는 MSE335.262px·CE334.707px, 중앙335.258px다. 최종 대표2페어의 CE 예측 x는489~532px, y는458~583px에 집중한다. 전관절·전체frame의 시각 검토는 아니며 낮은 성능의 진단이다.

CE의 변화가 MSE보다 크지만 의미있는 관절 추적이 확보됐다고 판단할 근거는 부족하다. 운영 pilot은 통과했으나 전체 학습 승인 문서는 발급하지 않았다. 중앙 집중·모델의 공간 문맥 부족·학습량을 함께 검토하고, 사용자에게 성능 한계를 보고한 뒤 다음 실행을 결정한다. 자동 재시험이나 전체 학습을 시작하지 않았다. M2/M3 실제 학습 완료로 상태를 올리지 않는다.
