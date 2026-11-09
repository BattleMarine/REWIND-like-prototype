# ArchVis 0버전 RGB2D 진단 학습

[직접 추출본 입력 계약](EXTRACTED_STEREO_INPUT.md)에서 발행한 manifest를 고정해 UI 없이 학습합니다. 공식 train에서 고른 1,000페어와 공식 validation을 사용해 입력·학습 동작을 확인합니다. 검증 문서의 `diagnostic_only=true`, `train_ready=false`, 원본 motion 근거의 상태는 학습 보고서와 저장 파일에도 남습니다. 공식 split 소속과 원본 motion 독립성은 각각 확인합니다.

학습기는 기존 StereoHeatmapNet, native 영상 resize·좌표 복원, Adam, StepLR, AMP, epoch 순회와 재개 코드를 사용합니다. `loss_kind`를 생략하면 `mse`이며, `gaussian_cross_entropy`는 같은 정규화 Gaussian target에 FP32 `-sum(target * log_softmax(logits))`를 적용하고 영상 안의 유효 관절 뷰를 평균합니다. MSE와 교차 엔트로피 비교에서는 seed·입력·모델·해상도·배치·optimizer·평가 조건을 고정하고 loss만 바꿉니다. 손실 스케일이 다르므로 두 손실의 절대값보다 원본 영상 px 지표와 예측 분포를 함께 봅니다.

## 실행 입력

- 같은 공식 validation 키와 파일을 사용한 manifest와 content SHA를 지정합니다. 처음에는 직접 추출본 도구의 `--train-limit 1000`으로 준비합니다.
- RGB·pose에는 manifest의 상대 경로가 시작되는 디렉터리를 지정합니다. motion map의 근거를 검토하고 자료가 확정되기 전에는 `declared_unverified` 상태를 유지합니다. Manifest를 만든 62관절 spec 파일도 같은 바이트를 사용합니다.
- 소스 revision, source ZIP의 파일 SHA, config 파일의 SHA, 실제 입력 경로와 실행 자원 상한은 오케스트레이션 검토 후 테스트 세션이 고정합니다. Colab에는 기존 PyTorch 2.3 이상, Pillow, psutil이 필요합니다. [Colab 실행 템플릿](../notebooks/archvis_direct_rgb2d.ipynb)은 기본적으로 실행이 비활성화되어 있습니다.

비교 설정 예시입니다. 두 실행에 공통으로 같은 파일을 쓰고 CE 실행에서만 마지막 항목을 바꿉니다.

```json
{
  "seed": 71,
  "batch_size": 2,
  "learning_rate": 0.0001,
  "scheduler_step": 51,
  "scheduler_gamma": 1.0,
  "amp": false,
  "training_image_size": [64, 64],
  "frontend_version": "archvis_direct_v0",
  "max_native_tensor_bytes": 67108864,
  "max_heatmap_elements": 2097152,
  "loss_kind": "mse"
}
```

```text
python -m rewind_m1.extracted_rgb_cli
  --manifest MANIFEST --manifest-sha256 CONTENT_SHA256
  --rgb-root RGB_ROOT --pose-root POSE_ROOT --spec SPEC --config CONFIG
  --output NEW_OUTPUT --device cuda --epochs 1
  --max-seconds SECONDS --max-attempts ATTEMPTS --max-read-bytes READ_BYTES
  --max-rss-bytes RSS_BYTES --max-disk-bytes ARTIFACT_BYTES
  --max-cuda-allocated-bytes CUDA_ALLOCATED_BYTES --max-cuda-reserved-bytes CUDA_RESERVED_BYTES
```

위 인자는 한 명령으로 전달합니다. `--epochs` 또는 `--steps`는 재개 이전을 포함한 전체 목표입니다. `--max-attempts`는 이번 프로세스의 시도 횟수이며 AMP overflow 시도도 포함합니다. 처음과 마지막에 train·validation 전체를 평가하므로, 1 epoch 새 실행의 파일 읽기 예산은 train 파일량 세 번과 validation 파일량 두 번에 manifest·config·spec·증거 문서와 파일 길이 확인 여유를 더해 계산합니다. Report에는 이번 프로세스의 예약 입력 바이트와 단계별 시간, 관측 RSS, CUDA allocator 할당·예약 peak, 임시 파일을 포함한 산출물 peak를 저장합니다. RSS·시간은 협력적 검사이고 CUDA 수치는 PyTorch allocator 범위입니다.

## 결과, 재개와 확대 검토

`last.pt`에는 모델, optimizer, scheduler, scaler, RNG, epoch·step, sampler 위치, 방문 비트맵, manifest·loss·config, 최초 평가가 저장됩니다. 새 출력 디렉터리에서 `--resume LAST_PT --resume-sha256 FILE_SHA256`으로 같은 실행을 잇습니다. 성공한 optimizer update만 cursor와 고유 방문 수에 반영합니다. `weights.pt`는 원본 영상 좌표의 RGB2D 진단 추론에 쓰며 `DirectRGBInference`에서 파일 SHA를 대조해 읽습니다.

`report.json`에는 시작과 종료의 train·validation loss와 px, 전체·몸·손의 화면 안팎 관절 수와 오차, 화면 중앙 고정 예측의 px, softmax 최대 확률과 entropy, 각 split의 대표 좌표·confidence, 최근 gradient와 실제 방문 수가 들어갑니다. 같은 1,000개 고유 train 페어가 최소 한 epoch에서 실제 update에 쓰였고 두 split 평가와 저장이 끝났으며 유한한 gradient와 가중치 변화를 확인하면 운영 검사를 통과합니다. 중앙 집중과 구별되는 학습 신호는 오케스트레이션이 시작·종료 지표와 대표 예측을 검토해 별도로 판단합니다.

전체 적격 train은 별도의 전체 manifest를 준비합니다. 같은 평가 파일과 키, 같은 모델·loss·resize·optimizer 계약, pilot train 페어 파일 해시의 포함 여부를 대조합니다. 전체 모드는 성공한 pilot `report.json`과 별도의 확대 검토 문서를 다음 인자로 받습니다.

```text
--pilot-report REPORT --pilot-report-sha256 FILE_SHA256
--promotion-review REVIEW --promotion-review-sha256 FILE_SHA256
```

검토 문서에는 `schema=direct_rgb2d_promotion_review_v1`, `decision=approve_full_train`, `pilot_report_sha256`, `reviewer`, `learning_signal_evidence`, `representative_prediction_evidence`를 기록합니다. 전체 실행은 같은 seed 설정의 새 checkpoint 계보에서 시작합니다. 전체 규모에 맞춰 epoch 목표와 자원 상한을 정합니다.

## 로컬 검토와 실제 실행 경계

2026-11-09 root는 관련40개 CPU 검사를 독립 실행했고, 추가 multi-chunk 검사1개도 통과했다. 추가 검사는4+3개 입력 묶음의 순서 변경, 짧은 마지막 배치, 묶음/epoch 경계 재개, 정확 방문 bitmap 및 무중단 전체 상태 일치를 확인한다. 기존 MSE 기본값과 CE 계산, 모의 AMP skip, 추론 weights 재로딩도 검사했다. CLI help, notebook 코드 compile과 문서 링크를 확인했다.

CUDA AMP·실제1,000페어·전체 ArchVis 학습은 미실행이다. 노트북은 `EXECUTE=False`인 실행 템플릿이며 아직 배포용 완성본이 아니다. 공식 split/calibration, motion 근거, duplicate inventory, 약1,000페어 manifest 준비와 실제 Drive 경로·고정 소스 ZIP/config·자원 예산·결과 보관 위치를 갖춘 뒤 테스트 세션에 인계한다. 이 단계에서 사용자에게 빈 설정을 추측해 실행하도록 요구하지 않는다.

고정 소스·입력 ZIP을 검증하고 manifest 준비부터 결과 회수까지 연결하는 [Colab 인계 묶음](ARCHVIS_COLAB_HANDOFF.md)을 추가했다. 위 빈 설정 템플릿보다 테스트 세션이 전달하는 고정 ready 노트북을 우선 사용한다. 실제 경로·runtime manifest·자원 값은 Colab에서 확인한다.
