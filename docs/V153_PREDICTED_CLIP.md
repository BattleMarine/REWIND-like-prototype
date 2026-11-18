# v1.5.3 예측2D 단일클립 학습

2026-11-18 실제 T4 학습 및 checkpoint·저장 지표 검토 완료. Body10.78mm·hand local5.65mm, 단일클립 형태 기준 통과. [결과·범위·한계](V153_PREDICTED_CLIP_RESULTS.md). 아래 준비/미실행 표시는 실행 전 기록이다.

2026-11-17 사용자 승인에 따른 `V153-PREDICTED-CLIP-001` 개발 계약이다. 구현·로컬 검증 단계이며 실제 GPU 학습 결과는 없다. [v1.5.2 공통 입력·부모 pin](V152_SAMPLING_DIAGNOSTIC.md)과 같은 첫 train50 클립을 사용하지만 source/config/실행ID/notebook/result 경로는 분리한다. [v1.4 GT조건 결과](V14_ORACLE_RESULTS.md)는 기존 근거로 재사용하며 새 GT2D 대조군이나 다른 클립을 자동 실행하지 않는다.

## 학습 조건과 초기화

[고정 config](../configs/v153_predicted_clip.json)와 [paired runner](../rewind_v1/train_clip_runs.py)를 사용한다. 입력은 저장된 실제 예측 F9 특징, 첫 stereo midpoint camera와 GT upper train condition이다. Frontend는 학습하지 않는다. 손의 target은 wrist-relative40finger이며 유효 mask는 finger&wrist다. GT upper는 wrist도 포함하므로 실제 예측 입력만으로 완성된 배포 성능이 아니다.

부모 v1.5의 **original u0** body/hand 가중치를 strict load하고 원본 fingerprint를 확인한다. u1000 finetune이나 부모 optimizer 재개가 아니다. 새 Adam·StepLR·disabled scaler·cursor를 만들고 학습 RNG를seed71로 초기화한다. 부모 초기 weights는 bodyseed72/handseed73 결과다. Frame loss1·velocity0·FP32·x0 target scale100mm·기존 architecture와 학습률을 유지한다. 모델 하나당 최대5,000update, body/hand 총10,000optimizer update다.

`0/200/1000/5000` paired update에서 같은 keyed epsilon(body offset100/hand200)의 pure-noise DDIM10 평가와 모델 snapshot을 저장한다. 평가/RNG 보존 경로가 이후 학습에 난수 영향을 주지 않도록 한다. 지표·mask·GT clip 평균 기준선·3frame whole/local PNG는 [v1.5.2](V152_SAMPLING_DIAGNOSTIC.md)와 같다. Train memorization 진단이며 독립 validation/generalization이나 v1.4와 한 변수만 바꾼 인과 비교가 아니다.

잠정 기준은 body20mm·hand local10mm, bone/spread 비율0.8..1.2, GT clip 평균 constant baseline보다 낮은 오차와 첫·중간·끝 frame의 오차다. 첫-끝 motion도 기록한다. 예산 안에서 기준 미달이면 원인 검토 대상으로 남기며 구현 버그·학습량 확대 승인으로 해석하지 않는다.

## 저장·중단·재개

100paired update와 평가/종료/중단 경계마다 `last.pt`에 두 모델·Adam·scheduler·scaler·RNG·cursor·history·평가/snapshot pin·고정 source/config/input/environment를 원자적으로 저장한다. 한쪽 update 성공 후 다른쪽 실패하거나 history 기록 도중 중단하면 두 모델과 optimizer/RNG를 해당 pair 시작으로 복구한다. 평가 snapshot은 두 모델을 한 파일에 저장한다. 중단 당시 메모리 위치와 마지막 durable checkpoint 위치를 구분한다. 저장 실패시 이전 last를 보존한다.

동일 런타임에서 **기존 RUN 경로와 last SHA**를 명시하여 재개한다. 새 빈 RUN을 재개 대상으로 취급하지 않는다. 같은 notebook bootstrap에서 수행한 이전 RUN도 last가 있으면 재개할 수 있다. Source/config/input/environment가 바뀌거나 저장 artifact가 변조되면 거부한다. GPU 변경을 동일 재개로 통과시키지 않는다. 새 VM에서 결과 ZIP을 복원하는 운영 절차는 이번 지원/실증 범위에 포함하지 않는다.

실측 누적 시간과 보수적 checkpoint 저장 allowance를 별도 필드로 보존한다. 누적 실측은 checkpoint state capture까지의 active wall에 현재 invocation을 더한 값이며, 직전 last write 구간은 미포함이다. 재개 budget에는 이 미관측 구간을 위해60초 allowance를 별도로 적용하고 이를 실측에 합산하지 않는다. GPU·VRAM·Torch/CUDA/Python·FP32·실제 invocation wall·peak GPU memory를 결과에 기록한다.

## 예산·인계

입력512MiB/64member·600초, invocation7200초·누적14400초(종료 저장 reserve60초), snapshot0/200/1000/5000, checkpoint512MiB, 결과 파일합/ZIP1GiB다. Disk free8GiB·Linux MemAvailable4GiB·관측 RSS4GiB를 확인한다. 결과 ZIP과 임시 atomic save/copy 공간은 별도 disk 용량을 사용한다. Package600초·Drive copy/readback600초는 별도 한도이며 시간은 각 단계 실제 wall로 기록한다. 지원 CUDA GPU를 사용하며 A100 모델명 제한은 없다.

[빌더](../scripts/build_train_clip_diagnostics.py)의 `--experiment v153`로 독립 notebook/receipt를 생성한다. `python -m rewind_v1.train_clip_cli --experiment v153 source`는 source 확인 진입점이다. Notebook의 RUN 기본False, `RESUME_EXISTING_RUN=False`, CUDA 기본이다. 재개시 명시 경로·SHA를 입력하며 입력/모델을 자동 재선정하거나 학습을 자동 반복하지 않는다.

Drive 사전 mount/쓰기·readback을 확인한 뒤 호출을 시작한다. 실패·수동 중단에도 finally에서 로컬 보고서·새 학습 snapshot·last·예측/PNG를 ZIP으로 보관하며 입력 bundle/부모 모델은 제외한다. 보관 실패시 로컬 결과를 유지하고 자동 재시도하지 않는다. 실행 성공/paused/실패와 Drive 보관 성공을 구분한다. SIGINT후60초 저장대기, 미종료시kill/wait30초이며 VM 자체 종료 시 finally를 보장하지 않는다.

오케스트레이션 검토·고정 revision/source/notebook SHA 확정 후 테스트 세션이 실제 실행 안내·결과 회수를 담당한다. 개발자의 CPU fixture 재개 검사 통과를 실제 CUDA/Colab/Drive 재개 완료로 보고하지 않는다. 전용 실행 보고서와 [실험 계보](V1_EXPERIMENT_LINEAGE.md), [결과 종합](EXPERIMENT_RESULTS_REPORT.md)은 실제 회수 근거에 따라 갱신한다.

## 로컬 검토와 고정 배포 진입점

2026-11-17 구현·root 검토: 두 실험의 공통 입력 검증, 원본 모델 strict load, DDIM 추적 동등성, 중단/재개 상태 일치, 실패 결과 보관 및 기본 실행 OFF 노트북을 준비했다. 관련 CPU 회귀30개가 통과했다. 실제 GPU·Colab·원격 Drive 실행과 새 VM 복원은 미검증이다.

Root는 실제 u1000 모델과 첫2프레임의 DDIM10 wrapper/원본 출력을 비교해 body·hand 최대차이0을 확인했다. 공통 bundle12개 선택파일의 CRC/SHA와 개발 fixture의 whole/손local 그림을 확인했다. 검토 JSON·테스트 로그·최종 source/notebook 식별값·실행 순서는 [고정 테스트 인계](../outputs/train_clip_diagnostics/TEST_HANDOFF.md)에 보존한다. 이 로컬 검증을 정확도 개선이나 실제 학습 완료로 해석하지 않는다.
