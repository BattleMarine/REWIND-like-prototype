# v1.5 카메라 표현 통제 학습

**2026-11-15 후속 결과:** T4·총4,000update와 개발6클립 평가를 완료하고 원점 안정성·도시 오류 개선 및 잔여 손 축소를 확인했다. [실제 결과·독립 검토](V15_CAMERA_TRAINING_RESULTS.md). 아래는 변경하지 않은 실행 전 계약·로컬 검증 기록이며 당시 미실행 표현을 현재 상태로 해석하지 않는다.

2026-11-15 사용자 승인과 root 설계 선택에 따른 **구현·로컬 검증을 마친 실행 준비**다. 실제 Colab 학습·평가 결과는 아직 없다. 기존 v1.5의 GT/예측2D·상체 교체 진단은 조건부 후속으로 보존하며 이번 자동 실행에는 포함하지 않는다. v1.6 제안은 필요한 내용을 이 실험에 흡수한 뒤 폐기한다. 번호·계보는 root가 관리하는 [버전 문서](EXPERIMENT_VERSIONS.md)와 [실험 계보](V1_EXPERIMENT_LINEAGE.md)를 따른다.

## 질문과 통제 조건

[v1.2.3 원점 민감도 결과](V12_CAMERA_GAUGE_RESULTS.md)는 학습된 절대 world 원점 조건에 큰 민감도가 있음을 보였다. 이번에는 **학습과 추론에서 일관된 first-frame stereo midpoint 표현**을 사용했을 때를 절대 표현과 비교한다. 개선·수렴·독립 일반화를 미리 보장하지 않는다.

두 군은 `absolute`와 `first_midpoint`뿐이다. 같은 고정20clip×50frame=1,000 teacher train 페어, RGB, pretrained v1.2 frontend F9, 예측2D 특징, GT, 모델 구조, 초기 가중치, 학습 순서·노출·noise/timestep과 loss를 유지한다. 변경 변수는 camera translation 표현뿐이다. Test·v1.3 혼합학습·규모 확대·rotation 변경·clipping은 하지 않는다.

**F9는 v1.2 train20,888로 사전학습되었다.** 새 body/hand teacher가1,000페어를 사용한다는 뜻이며 전체 모델이1,000개만 학습했다는 뜻이 아니다. v1.1과 frontend가 같지 않다. 기존 v1.2 성적과 이번 새초기화·소규모 teacher의 성적을 개선 순위로 비교하지 않고, 이번 두 군 안에서만 원점 표현 효과를 해석한다.

## 초기화·학습량·모델 정책

Root가 첫 통제 실험으로 선택한 설정은 [고정 config](../configs/rewind_v15_camera_training.json)를 따른다.

| 항목 | 고정 설정 |
| --- | --- |
| Frontend |v1.2 best F9, frozen/eval; 학습 없음 |
| Body·hand 초기화 |새초기화 seed72/73, 각 군의 동일role state bit-identical 검증 |
| 각 군의 body / hand |각1,000update;20clip×50epoch 노출 |
| 총 optimizer update |2군×2teacher×1,000=4,000 |
| Optimizer / LR |Adam /5e-5 |
| Scheduler / precision |StepLR gamma1 /FP32, scaler 상태도 저장 |
| Teacher |hidden512, graph3, temporal3, heads4, 과거20frame |
| Diffusion |1,000step 학습, DDIM10 평가, seed71, body offset100/hand offset200 |
| Loss |기존단위·reduction·mask 유지, frame weight1, velocity0, foot 미사용 |
| Milestone |0/200/1,000 weights 저장;0/200은 저장만, 평가·선택하지 않음 |
| 최종 선택 |각teacher u1,000 고정; 도시 지표로 best 선택 없음 |

B10/H9 fine-tune은 적은 비용으로 적응할 수 있지만 이미 절대 표현에서 학습된 초기 가중치가 한 군에 유리할 수 있다. 이번에는 새초기화 복제로 이 편향을 줄인다. 동일 update에서 미수렴할 가능성은 남고, 두 군 모두 축소되거나 실패해도 camera 표현이 무효라고 단정하지 않는다. 결과가 좋지 않아도 자동 epoch/update 확대는 없다.

학습은 body 두 군을 paired update로 완료한 뒤 hand 두 군을 진행한다. Hand 학습은 기존 GT 상체 조건, 평가에서는 각 군 최종 frozen body의 예측 상체·wrist를 사용한다. 카메라와 동시에 손 조건 정책이나 논문 구조를 바꾸지 않는다. 각 pair는 같은 RNG에서 시작해 diffusion timestep/noise를 맞추며 두 update 모두 성공한 뒤에만 위치를 전진시킨다. 두 군 상태·optimizer의 transaction snapshot 복사 비용은 pilot/학습 시간에 포함하며 별도 wall time도 기록한다.

## 원점·causal streaming 계약

감사 bundle의 **full float64 R/t**를 사용한다. `x_camera_mm=R*x_canonical_world_mm+t_mm`, `C=-Rᵀt`이며 각clip 첫frame의 좌우 camera center midpoint A를 구한다. 정규화는 `t_norm=t+R*A`다. 회전 R, 장면, device GT는 바뀌지 않는다. 원본 raw pose를 수정하지 않는다. Encoder는 기존 column0/column1/t÷100mm의18차원 float32를 유지하고 translation roundtrip 오차0.1mm 상한과 실제 잔차를 기록한다.

`CameraOrigin`은 첫 chunk의 첫frame에서만 anchor를 계산한다. State에는 mode/sequence/clip start/first frame/next frame/anchor(mm)를 저장한다. 다음 chunk는 같은 clip·연속 위치만 허용하고 sequence 또는 clip이 바뀌면 명시적으로 reset한다. Checkpoint에는 고정clip anchor 상태도 저장하고 복원 때 다시 계산한 계약과 비교한다.

World 이동은 `x′=x+Δ`, `t′=t−RΔ`다. 이동된 camera로 **첫frame anchor를 다시 계산**한 후 정규화해야 하며 기존 anchor를 그대로 쓰지 않는다. Δ=(0,0,0), Δ0 repeat, Δ=(100000,0,0)mm를 비교한다. Root의 기존 실제 첫 train Standing Up50frame float64 감사 자료에서는 +100m 뒤 정규화 camera 최대차이 약7.11e-15를 확인했다. 이는 대수적 입력 검증이며 모델 성능 결과가 아니다.

최종 모델에서도 full50/prefix25 sampler 일치와 분할 camera state 저장·복원 일치를 확인한다. 전체 clip의 미래 camera로 anchor를 갱신하거나 미래 정규화 통계를 사용하지 않는다. 스트림 검증은 같은 과거 prefix를 재사용하는 causal sampler 경로이며 별도 KV-cache 속도 최적화 완료를 주장하지 않는다.

## 고정 입력·캐시 재사용

- Train selection content SHA `48e3c749c7d7710feb7fd61601b7406264837e7a67ceeb4a48787359825cafad`.
- 기존 raw cache content SHA `70241b27f66565593f0c8de0aeb6fe1f94c3672f932d7609bc882fd24a21d7b0`, 원본65part 합7,554,249,675B. Drive `REWIND-like prototype/reports/v1_actual_001/raw_cache/`. 현재 존재는 실행 때 확인한다.
- 기존 input ZIP SHA `6256eb7bd148ec51bdb94627c68e54657be3808c607a12e13b68f7e479b15e97`, evidence ZIP SHA `654af71d3864a6ccf9dcbed1610c6cae90651edcd47b30b9f4b2f386762d9c20`, decision SHA `f6b79bac09d168e48134cab58560eb26f07c7d6f8f4aae139b7aecfc662e789f`.
- v1.2 model ZIP873,393,783B, SHA `48661e75e0969ca140ab9cf0e543d73d2bb3b9ec108d30652cf7fb4f4b76b506`. **F9만 학습 입력으로 재사용**하고 B10/H9로 teacher를 초기화하지 않는다.
- 회귀6clip은 [v1.2.2](V12_CAMERA_CONTROL_RESULTS.md)·[v1.2.3](V12_CAMERA_GAUGE_RESULTS.md)의 기존 city/control 두 결과 ZIP과 동일 원본 키·크기·SHA를 사용한다. 새 후보 선정은 없다. 원래 실행 ID·핀은 보존한다.

RawCache 원본 manifest 전체 identity를 확인한 뒤 고정1,000keys가 참조하는 part만 복사·SHA 검증한다. Part 번호가 첫20개이거나 모든 part가 train이라고 가정하지 않는다. 이는 **선택 train part 복원이며65part 전체 복원이 아니다.** 원본 manifest는 그대로 보존하며 해당 keys의 원본RGB·pose member hash를 다시 확인한다. Validation2,236을 학습에 넣지 않는다. v12 full prepared catalog를 고정1,000 raw cache로 오인하지 않는다.

Root의 기존 고정 manifest 검토에서는 해당1,000keys가20parts, 합2,334,727,075B(약2.174GiB)를 참조하고 member payload 합은2,333,644,735B다. 이는 보존 metadata에서 확인한 예상 복원량이며 현재 Drive 파일 존재·실제 읽기 성공을 검증한 결과는 아니다. 복사 비용과 local member 재읽기 비용을 별도로 센다.

Pretrained frontend F9가 다르므로 v1.1 특징 cache는 재사용하지 않는다. F9 예측 특징을 한 번 생성해 두 군에서 공유한다. Camera는 그 뒤 군별로 바꾸며 feature를 다시 만들지 않는다. 작은 파생 특징·GT/R/t cache는 결과 보관에 포함하여 새 VM에서 동일 hash로 복원한다. 원본 RGB/raw cache는 자동 업로드하지 않는다. 회귀 GIF용 raw RGB는 결과 ZIP에서 제외하며 새 VM 평가 전에 동일 고정 경로·SHA로 재확보해야 한다.

## 평가와 해석

기존6클립300frame은 **회귀·개발 진단**이며 독립 일반화가 아니다. 한 인물 경로, 명목25FPS·실제 timestamp/stereo 동기·원본 motion 독립성 미확인 한계를 유지한다. 각군 최종 모델에서 body/hand local/global MPJPE, bone/spread, body centroid와 중심제거 보조 지표, world 이동 paired 변화·분모·max/mean, float32 camera 입력 잔차를 기록한다. GT 누락 mask와 visibility를 혼동하지 않는다.

추가 train-fit 진단은 고정 selection의 첫 train clip 하나를 사용한다. GT오차로 고르지 않고 최종 DDIM body/hand local/global 오류·형태를 별도로 기록하며6개 회귀 지표에 합산하거나 checkpoint 선택에 쓰지 않는다. Loss 감소만으로 형태 수렴을 선언하지 않는다.

Δ0 repeat는 같은 실행의 수치 기준이다. 기존 gauge의 절대·상대 tolerance와 repeat-aware paired sum/count/mean/max를 재사용한다. 변형 camera의 float32 양자화는 repeat가 통제하지 못하므로 미세 출력 변화만으로 구조적 문제를 확정하지 않는다. 학습·평가/재개 일치 여부와 성능 판단을 구분한다.

비교 GIF는 GT/absolute/first midpoint/first midpoint+100m을 공통 device3D축으로 표시한다. 실제 축 min/max(mm)를 함께 기록한다. 2D 특징은 군·world 이동과 무관하게 공유하며 GT 조건 교체 진단은 실행하지 않는다.

## 예산·pilot·재개·보관

| 작업 | 상한 |
| --- | --- |
| 선택 raw part 복사/검증 |4GiB/3,600초 |
| Train member 재읽기·특징 생성 |4GiB/3,600초; 특징 합64MiB |
| 회귀 raw 준비 |2GiB/3,600초 |
| 모델 copy/hash |1GiB/600초 |
| 실제 장치 short pilot |각teacher2paired update/군;600초 |
| 본 학습 |1회7,200초, 누적14,400초, optimizer 총4,000update; 종료 저장 여유60초 |
| 최종 평가·렌더 |3,600초 |
| Checkpoint / published 결과합·ZIP |1GiB /각2GiB |
| 각 package / copy+readback |600초 /600초, 단계 종료마다 별도 |

Free disk16GiB/MemAvailable4GiB와 학습중 RSS4GiB 관측 guard를 사용한다. OS hard limit이나 cgroup headroom 보장은 아니다. 원본 복사, local 재읽기, 특징, pilot, 학습, 평가, packaging·Drive readback 비용을 구분한다. Checkpoint 교체 중의 임시 공간은 별도 free disk에서 확보한다.

지원 CUDA GPU를 허용한다. 모델명 제한 없이 실행하되 **재개는 source/config/input/장치명·VRAM·Torch/CUDA/precision이 같은 고정환경**만 받는다. GPU가 바뀌면 동일조건 재개로 몰래 수용하지 않는다. CPU는 로컬 synthetic 검증용이며 본 학습 진입점은 CUDA를 요구한다.

Short pilot에서 양군 body/hand update·저장·유한값과 실제 비용을 확인한 뒤 본 모델을 seed부터 새로 만들고 초기 model/RNG hash 일치를 확인한다. Pilot optimizer나 학습 state를 본실험에 이어쓰지 않는다. 같은 notebook의 명시 RUN으로 처리하고 실패시 자동 재시도하지 않는다.

Checkpoint는100paired update마다, role 경계·수동중단·시간 상한·오류 종료 때 저장한다. Model/optimizer/scheduler/scaler/RNG, role/pair/전체 위치, anchor, 고정 input·source·environment, milestone pin을 포함한다. Half pair 실패는 두 군을 직전 정상 pair로 rollback한다. SIGINT로 안전한 저장 기회를 주고60초내 종료하지 않으면 kill/wait한다. VM 자체 소실에는 마지막100pair 경계 이후 진행을 보장하지 않는다.

각 단계 시작 전에 Drive 마운트·쓰기/readback을 확인하고 종료 finally에서 보고서·모델·파생 feature를 자동 보관한다. 원본/raw/reference/model ZIP 자동 재업로드는 제외한다. 성공과 보관 성공을 구분하고 게시 실패시 로컬 ZIP을 유지하며 자동 재시도하지 않는다. 새 VM에서는 고정 source와 result ZIP SHA를 검증해 복원하고 명시 last checkpoint SHA로 재개한다. 평가용 raw 재확보도 고정 pin 검증 경로로 수행한다.

Notebook의 모든 RUN은 기본false다. 준비→pilot/본학습→최종고정checkpoint 평가를 분리하고, pause·failure 뒤 원인 검토 없이 자동 다음 단계로 가지 않는다. 배포 builder·고정 source/input/evidence/reference ZIP 및 revision의 최종 pin은 root 검토·커밋 후 테스트 세션에 인계한다.

## 개발 검증과 남은 작업

로컬 테스트는 anchor world 이동 불변성·first chunk/sequence reset/state 복원, 동일 초기화·RNG와 half-pair rollback, body→hand/hand 중간/complete checkpoint 재개 동등성, 환경·anchor·milestone 변조 거부, raw 제외·feature/모델 fresh-runtime 회수/복원, 평가용 raw 재확보·frontend cache 재사용, defaultoff/중단/실패 보관과 기존 gauge/control 회귀를 대상으로 한다. 실제 실행한 검사와 시간은 개발 인계 보고에 기록한다. 이 문서만으로 구현 검증·학습·성능 성공을 완료 표시하지 않는다.


2026-11-15 root 독립 검토: 신규 학습·복원 검사와 기존 camera gauge/control 회귀를 합쳐 **25개 검사, 292.606초, 정상 종료 통과**했다. 같은 초기화·난수, 업데이트 도중 실패와 진행 횟수/history rollback, 두 군 중간 저장의 부분 실패, 학습 경계·완료 체크포인트 복원, 실제 합성 teacher의 평가·prefix, GIF 생성·결과 회수 경로를 포함한다. 추가로 기존 감사50frame의 +100m 정규화 입력 잔차 약7.11e-15와 고정1000keys의20part 예상 복원량을 확인했다. 문서7개의 내부 링크와 변경 형식도 확인했다.

실제 Drive 읽기·CUDA pilot·본 학습·실제 모델의 정확도·새 Colab VM 복원은 아직 수행하지 않았다. 로컬 fixture의 Drive 보관 성공을 원격 Drive 보관 성공으로 해석하지 않는다. 최종 배포 ZIP과 notebook은 커밋 후 생성하며 테스트 세션에 고정 hash로 전달한다.

추가된 선택 part 복원 검사1개도 root가 별도 실행해0.345초에 통과했다. 미선택 part를 읽지 않는 것, 원본 manifest 보존, 기존 복사본 hash 재검사·손상 거부를 확인했다. 전체25개를 다시 실행한 결과와 혼동하지 않는다.
