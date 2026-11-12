# v1.4 GT 조건 단일 클립 과적합 결과

2026-11-12. 고정 소스 `c05f5482ea88478d5bd62cb7fbec320f73b109b4`의 실제 A100 재시험을 검토했다. 이전231e651은 준비 단계의 metadata 비교 오류로 학습 전에 중단됐으며 이번 결과와 구분한다.

## 판정과 실행 조건

**같은 train50프레임의 GT2D·hand GT상체 조건에서 몸과 손의3D 형태를 학습하는 데 성공했다.** Body에는GT3D상체를 조건으로 넣지 않는다. Hand는GT상체 조건을 받으며 실제RGB→예측2D 입력 시험이 아니다. 새로운 클립의 일반화나 독립validation 결과도 아니다.

A100-SXM4-40GB, FP32, torch2.14.1+cu130, 각모델5,000update(총10,000). status completed/failure null, durable paired update5,000. Runner 기록593.880초(약9분54초); 사용자 전체 소요 약11분과 구분한다. Runner시간에는 학습·평가·로컬저장이 포함되며 준비/최종원격보관까지의 전체시간은 아니다. Peak CUDA allocated977,797,632bytes(약0.91GiB)는 torch 할당량으로 GPU 전체 사용량과 다르다.

## 학습 경과

아래는 고정 pure-noise DDIM10의 동일train클립 MPJPE다. Body는 device 좌표, hand는 wrist-relative local 좌표이며 각각22/40관절,50frame의 유효관절 평균이다. 별도k1 GT-noised 직접복원과 구분한다.

| 모델별 update | Body(mm) | Hand local(mm) |
| --- | ---: | ---: |
| 0 | 495.39 | 143.02 |
| 200 | 123.19 | 47.03 |
| 1,000 | 19.19 | 10.82 |
| 5,000 | 13.71 | 2.90 |

최종 body/hand 오차13.71/2.90mm는 같은클립 평균자세 baseline126.42/23.02mm보다 낮다. Body 뼈길이·spread의GT비율은0.9975/0.9960, hand는0.9955/1.0017이다. 첫·중간·마지막 frame의 오차와 잠정 공간 기준을 양쪽 모두 통과했다. Body first-last 변위292.67mm(GT291.55), hand local38.96mm(GT38.06)로 정지 평균자세 출력만으로 설명되지 않는다. 이는50frame의 변화 재현이며 전체시간 추종지연/foot skating/streaming 검증은 아니다.

GT상체 조건으로 예측한 손을 **예측 wrist**에 붙인 global 오차는11.27mm다. GT wrist에 붙인2.90mm와 구분하며, 손 조건까지 모두 예측값인 실제cascade 성능으로 보고하지 않는다.

## 시각 검토와 해석

Root가comparison-00000/05000.png를 확인했다. 같은 축척의frame0·25·49에서 초기출력은소수점근처로축소되어있고5,000update출력은GT의몸통·사지·손형태와동작차이를근사한다. 이이미지는RGB overlay가아니라device좌표의골격시각화다.

이번 결과는 모델·loss·좌표·학습 경로가 **적어도 이 정답조건/한 클립에서는3D구조를 복원할 수 있음**을 보여준다. 모든조건에서 반드시출력이붕괴하는치명적오류가있다는가설을약화한다. 앞선준비버그수정이모델성능을개선한것은아니며그수정은학습진입을가능하게했다.

하지만v1.1과는입력(GT2D/GT상체),고유데이터수(1clip),동일클립반복노출이함께달라졌다. 따라서2Dfrontend오류가유일한원인이라고단정할수없고단순히epoch만늘리면해결된다고도결론낼수없다. 단일클립은시간/카메라패턴을외울수있으므로조건교체·제거실험없이는GT2D를실제로얼마나활용했는지도확정하지않는다. 깨끗한조건하의표현능력은확인됐지만일반화·입력품질·최적화의상대기여·미세구현문제는남는다.

후속 후보는동일클립·동일학습량에서①GT2D만예측2D로교체,②GT상체만예측상체로교체하여오차증가를분리하는것이다. 이는후속설계제안이며새실행승인이아니다. v1.2데이터확대는별도실험으로유지한다.

## 근거와 독립 검토

회수경로: `outputs/v14_pinfix_release_20261112/user_test/retry_001/`. Training ZIP860,003,688bytes, SHA `b78537b060a81c9f42105e0c5a8b59ff2f56baf105b1e607e66b5aba421548d4`. last.pt SHA `95e6e589bb7c28a7d32383a12e7686e0e766b7074c9e9d655563a6bb7794af5e`.

Root가ZIP전체SHA/CRC, recovery receipt seal 및17개본문의크기/SHA를독립확인했다. Last와0/200/1000/5000 모델5개를CPU에서weights_only=True로로드하고모든부동소수tensor유한값·단계번호를확인했다. Last에body/hand모델·optimizer·scheduler·scaler·RNG가보존됐다. CPU로드는실제학습재개나GPU재추론검증이아니다. 수치는회수run.json을검토한값이며별도추론재계산은하지않았다. `root_review.json`에검토근거가있다. 대용량ZIP·checkpoint는Git에포함하지않는다.
