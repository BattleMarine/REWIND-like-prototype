# v1.5.1 상체 조건 진단

2026-11-17 v1.5.1 결과 검토 완료: T4에서 같은 중점 모델·6클립300프레임의 무재학습 진단을 회수했다. GT 상체로 hand local 80.85→72.14mm(10.8% 감소), 4클립 개선·2클립 악화이며 손 뼈 길이 8.20→7.99mm/GT23.71mm로 축소는 남았다. 34파일·7seal과 저장 지표 재계산, 부모 A·반복 일치를 확인했다. 상체 조건의 일부 기여는 확인됐지만 잔여 원인은 미확정이며 후속 번호·실행은 확정하지 않는다. [결과·검증 범위](V151_UPPER_CONDITION_RESULTS.md).

아래는 실행 전 고정 계약과 당시 준비 기록이며 변경하지 않는다.

2026-11-15 사용자 승인. 실행 ID는 `V151-UPPER-CONDITION-001`이다. [v1.5 결과](V15_CAMERA_TRAINING_RESULTS.md)의 중점 모델을 **재학습 없이** frozen 평가하여 손의 상체 조건 차이와 wrist 부착 오차를 구분한다. 구현·로컬 검증을 완료하고 실제 Colab 진단을 대기한다. 실제 진단 결과는 아직 없다. 기존 v1.5 release·결과·학습 정책과 GT 상체의 train-only 차단은 변경하지 않는다.

## 고정 부모와 작은 입력

| 항목 | 고정 식별값 |
| --- | --- |
| 부모 ZIP |1,553,751,536B / SHA256 `a46e6a65f3777469d9f70d9aa99bf0aadf5bb5972ad06bb488c5b96f0944a170` |
| 부모 revision |`4f24f3aa2265c2ac54c21600b57bdece4e540411` |
| 부모 source identity |`10e25cd2cb6f5b8d5dd4242481bf1aca349869f7000985dba28e82214c9a4540` |
| 부모 평가 report content SHA |`e6f50fe90b7a234aabea6c44065f07ab4538fc8b53c040059f132309e5c73c27` |
| 중점 body u1000 |`05a7ec1f6d7832ce25623ee065fe3848fc6361f64166df6427d0f14561f700f4` |
| 중점 hand u1000 |`baf7bf29055960b51c99b598e0d865d52fd0a7a507d93242b804b687be7f7489` |

부모 원본은 `outputs/v15_camera_training_20261115/user_test/completed_001/results.zip`이다. 새 source/config/bundle/notebook pin은 root 최종 배포 receipt에서 확정하며 **부모 source와 이번 실행 source를 구분**한다. [고정 config](../configs/v151_upper_condition_001.json)를 따른다.

로컬 bundle 생성기는 원본 ZIP 전체 SHA,90개 member의 크기/SHA·CRC와 JSON seal을 검증한 뒤 필요한 원본 파일만 복사한다. ZIP을 통째로 해제하거나734MB optimizer checkpoint를 역직렬화하지 않는다. 작은 bundle의25개 원본 member는 다음과 같다.

- 중점 body/hand u1000 snapshot2개: 원본 가중치 bytes와 hash를 유지한다.
- 부모 receipt·setup·train report·eval feature manifest·최종 eval report5개.
- 기존6clip의 feature tensor6개, clip report6개, 중점 예측 JSON6개.

합계는약135MB이며 최종 파일 크기·SHA·manifest pin은 bundle receipt를 따른다. 전체 부모 ZIP 검증과 작은 bundle의 선택 파일 검증을 혼동하지 않는다. Bundle에는 원본 전체 receipt와 부모 ZIP pin을 남기고 source builder가 작은 ZIP 전체 SHA와 manifest pin을 고정한다. Runtime은 모든 선택 파일을 부모 receipt에 대조하고 source/model/clip/feature/GT/anchor 연결을 검증한다. 원본1.55GB를 Colab에 다시 업로드할 필요가 없다.

Raw RGB,65part cache,train 특징,기존 optimizer 복원과 F9 재추론은 필요 없다. 기존 F9의 저장 예측 특징·GT·float64 camera R/t·spec을 그대로 사용한다. F9는 v1.2 train20,888로 사전학습되었고 이번 두 teacher는 v1.5의 고정1,000페어로 학습된 모델이다. 이번 작업에서 추가 학습은 없다. 첫 train clip 진단도 이번 필수 범위에 넣지 않는다.

## 조건과 계산 순서

기존6clip300frame의 순서를 그대로 유지하며 재선정·대체·GT오차 기반 선택을 하지 않는다. 첫frame stereo midpoint camera 표현,13개 상체 관절 순서,mm 단위,FP32,DDIM10,seed71,body offset100/hand offset200의 keyed noise를 유지한다. 지원 CUDA GPU를 사용하며 모델명 제한·자동 CPU fallback은 없다. CPU는 로컬 fixture와 명시적 선택에 사용할 수 있다.

1. Body full50을 한 번 추론하고 이후 두 손 조건에서 같은 예측 body를 고정한다.
2. **A predicted upper**: 그 body의 동일 상체13관절을 hand condition으로 넣는다. Valid mask는 기존 예측 상체 정책을 유지한다.
3. **B GT upper diagnostic**: 저장 GT의 같은13관절과 GT valid mask를 넣는다. 누락값은0으로 채우되 mask False를 전달한다. 이는 평가 전용 wrapper이며 `gt_upper_train`의 train-only 차단을 해제하지 않는다.
4. A와 B의 hand local을 각각 full50 한 번 sampling한다. 둘은 같은 keyed hand noise를 사용한다.
5. 같은 실행 수치 기준으로 A hand를 한 번 더 sampling한다. **A repeat는 같은 body/upper/features/camera를 고정한 hand 재샘플링이며 body 반복 추론이 아니다.**
6. Causality 검증만을 위해 body/A/B prefix25를 추가 sampling하여 full50의 처음25frame과 비교한다.

따라서 clip당 full body1회,full A/B local 각1회,A-repeat local1회와 prefix 전용 body/A/B 각1회다. 후처리2×2에는 추가 sampling을 하지 않는다. 모델 state fingerprint와 features/camera/GT/mask의 입력 fingerprint를 전후 비교하고 optimizer가 생성되지 않는 경로를 검증한다.

## Local 조건과 wrist 부착의2×2

| Hand local | Predicted wrist 부착 | GT wrist 부착 |
| --- | --- | --- |
| A predicted upper |A local + 고정 body wrist |A local + GT wrist |
| B GT upper diagnostic |B local + 같은 body wrist |B local + 같은 GT wrist |

주판정은 A/B의 hand local 오차와 local bone/spread다. GT 손가락과 대응 GT wrist가 모두 있는 공통 mask `finger_valid & wrist_valid`를 두 local 및 네 global 조합 모두에 사용한다. Count/sum/mean과 frame별 분모를 저장한다. 기존 부모의 predicted-global은 finger-only native mask이므로 별도 지표로 표시하고 공통 mask와 섞지 않는다. GT wrist가 누락된 곳에 GT wrist 부착 native 지표를 만들어 보고하지 않는다.

같은 공통 mask에서 `GT wrist 부착 global error = local error`를 확인한다. 실제 float32 덧셈·뺄셈에는 반올림이 있으므로 최대 norm 잔차0.001mm를 허용한다. GT 누락 mask를 visibility나 화면 안 여부로 대체하지 않는다. 고정 부모 자료의 upper는 각clip650/650 유효하다는 root metadata 확인이 있으나 코드·테스트는 누락 mask를 보존한다.

**GT upper13관절에는 wrist 입력도 포함된다.** B는 상체 condition 안의 wrist 정보도 바꾼다.2×2는 sampling 이후의 wrist 부착 효과를 분리할 뿐, 상체 condition에 들어간 wrist 정보의 효과까지 독립 분해한 실험이 아니다.

Body 예측·오차는 두 조건에서 같은 값이다. GT body를 다시 추론 조건에 넣거나 2D/camera/body 값을 B에 맞춰 수정하지 않는다. Local shape는 원래40finger 관절과 양끝 유효 edge를 사용하며 wrist-root edge를 새로 추가하지 않는다.

## 비교 기준과 해석

A의 body/local/global/whole을 부모 중점 baseline 저장 예측과 비교한다. 허용치는3D `atol0.01mm/rtol1e-5`이며 최대 차이와 일치 여부, 부모·현재 GPU/Torch 환경을 함께 기록한다. 불일치는 `cross_run_mismatch`로 남기며 GPU/환경 때문이라고 자동 단정하지 않는다. 같은 실행 A-repeat·입력/모델 pin·prefix 검증과 구분해 root가 원인을 검토한다. 실패를 숨기거나 seed/checkpoint를 바꿔 맞추지 않는다.

같은 실행 A-repeat가 고정 tolerance를 넘으면 A/B 조건 효과 판정은 `inconclusive_A_repeat_unstable`이다. 안정된 경우 기존 repeat-aware coordinate 기준 `abs(B-A) > .01mm + 1e-5*abs(A) + 2*abs(A_repeat-A)`와 초과 수/분모, joint L2 sum/count/mean/max를 기록한다. 이는 변화 여부 기준이며 개선을 자동 뜻하지 않는다. Local 오차와 형태를 함께 본다. 작은 차이를 입력 양자화나 수치 변화와 구분하지 않고 큰 구조적 결론으로 확대하지 않는다.

이번6clip은 개발·회귀 표본이고 한 인물 경로이며 독립 일반화 평가가 아니다. 명목25FPS,실제 timestamp/stereo 동기·원본 motion 독립성 미확인 한계도 유지한다. GT 조건 성적은 진단이며 배포 성능이나 모델 완성으로 부르지 않는다. 반복 실험·재학습·학습량·데이터 확대는 없다.

## 출력과 그림

Clip별 JSON은 body1개,A/B local 및A-repeat,네 global 조합,GT·valid/common mask,상체13관절 값·mask·순서,feature/camera/GT/mask의 입력 전후 fingerprint,조건별 sum/count/shape,부모 비교·repeat·prefix와 artifact pin을 저장한다. 모델 전후 fingerprint는 최종 report에 저장한다. 비교 Markdown 표도 생성한다.

그림은 raw RGB 없이 저장된3D에서 만든다. GT/A/B 열과 body/local hand/global hand 행을 사용하며 **각 행 안에서** 모든50frame과 세 열이 같은 축을 쓴다. 행별 source xyz min/max(mm),orthographic uv 범위,실제 pixel 범위·scale을 별도로 저장한다. Local/device/global 행이 같은 좌표 축인 것처럼 혼합하지 않는다. Local40관절은 양손을 각각 wrist 원점으로 옮긴 것이므로 **두 wrist 원점이 겹친다**는 점을 화면·문서에 표시한다. Wrist-root 점 자체는40finger 집합에 포함되지 않는다.

Global 그림은 같은 predicted wrist를 붙인 A/B를 보여주고,네 조합 전체의 정량값은 표·JSON에 남긴다. 각clip GIF1개와 first/mid/last PNG3개,총6GIF·18PNG다. 모든 GT 조건을 제목과 열에 명시한다. 저장된 그림 표시는 추가 추론·raw 재다운로드를 하지 않는다.

## 예산과 실행·보관

| 단계 | 상한 |
| --- | --- |
| 로컬 parent 검증·bundle 생성 |600초,부모ZIP2GiB/unpacked2GiB/128members,논리 읽기4GiB |
| 진단 bundle |ZIP/unpack 각192MiB/64members |
| Source ZIP |64MiB/unpack192MiB/3,000members |
| Colab 입력 검증 |600초 |
| 전체 frozen 추론·repeat·prefix·지표·렌더 |합3,600초 |
| 초기 disk / RAM |free disk2GiB / Linux MemAvailable4GiB |
| 실행 중 RAM |high-water RSS4GiB 관측 guard |
| 결과 파일합 / ZIP |각128MiB |
| 결과 packaging / copy+readback |각600초 /600초 |

이는 OS 메모리 hard limit이나 cgroup headroom 보장이 아니다. 각phase는 외부 child wall timeout+종료여유30초를 갖고,중단시 SIGINT 후30초내 미종료면 kill/wait한다. 준비·추론·packaging·copy/readback은 실제 `monotonic` wall time을 기록하고,timeout/copy allowance를 실측시간에 더하지 않는다. 사용자 업로드의 실제 네트워크 전송 시간·양을 측정했다고 주장하지 않는다. Parent archive read와 작은 bundle bytes도 분리한다.

Notebook은 CUDA 기본,`RUN_DIAGNOSTIC=False`,`SHOW_SAVED_RESULTS=False`다. Source ZIP+bundle만 `/content`에 준비한다. 명시 실행 한 번으로 prepare→infer를 수행한다. Drive 마운트·쓰기/readback을 먼저 확인하고 종료 finally에서 성공·실패 보고서와 그림을 새 attempt에 자동 보관한다. 입력 model/bundle/raw는 결과 ZIP에서 제외한다. 게시 실패시 로컬 보고서·ZIP을 보존하고 자동 재시도하지 않는다. VM 자체 종료에는 finally를 보장하지 않는다.

작업 성공과 보관 성공을 구분한다. Prepare/infer 실측은 결과ZIP의 operation에,완료 후 package/copy 실측은 별도 retention receipt와 실행 출력에 남긴다. ZIP을 만든 뒤에만 알 수 있는 원격 copy 시간을 그 ZIP 안에 이미 기록했다고 주장하지 않는다.

## 빌드와 검증 인계

로컬 bundle 생성: `python -m rewind_v1.upper_condition_cli bundle --parent .../results.zip --output .../v151_diagnostic_bundle.zip`.

고정 notebook 생성: `python -m scripts.build_v151_upper_condition --source-zip ... --source-sha256 ... --revision ... --bundle-zip ... --bundle-sha256 ... --bundle-manifest-sha256 ... --output ...`.

Source/bundle/notebook의 최종 크기·SHA와 source identity는 root commit 후 배포 receipt로 고정한다. Developer는 구현·로컬 검증,root는 검토·커밋·테스트 세션 인계를 담당한다. 이번 local 검증은 실제 부모 bundle chain·상대경로·두 모델 strict CPU load/no-optimizer,기존 sampler와A 동일성,GT 미래 변경의prefix불변,GT train-only gate 유지,누락mask·공통분모·GTwrist등가성,모델/입력 불변성,실제fixture GIF/회수·운영실패 finally/no-retry 및 defaultoff를 확인한다. 실제 Colab/GPU 진단 결과는 별도 회수 후 기록한다.
