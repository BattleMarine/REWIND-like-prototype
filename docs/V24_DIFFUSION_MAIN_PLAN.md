# v2.4 x0/v 본시험 계획 초안

파일럿 R2의 실제 GPU 결과 회수 대기다. 이 초안은 회차 목록과 채워야 할 결정을 고정한다. 실행 승인·예산·결과는 파일럿 검토 뒤에 별도로 기록한다. [파일럿 계약 및 고정본](V24_DIFFUSION_PILOT.md)

## 비교 단위와 추가 조건

| 구간 | x0 | v | 합계 |
| --- | --- | --- | ---: |
| 최초 비교 | 42, 71, 103 | 42, 71, 103 | 6회 |
| 조건부 추가 | 137, 181 | 137, 181 | 4회 |

각 seed 쌍은 같은 A backbone, seed별 동일한 final head 재초기화, 같은 clip 순서·학습 noise, Adam 초기화·batch·학습률·update 예산을 사용한다. 중간 head는 둘 다 x0를 유지하고 frontend와 hand 가중치를 고정한다. 공통 evaluation noise seed 71, DDIM 10 및 device-mm 기준을 적용한다. 본시험에서 여러 seed를 받는 worker와 resume·회차 보관 구현은 별도 검토 대상이다.

최초 3개 seed 쌍의 완료·조건 일치를 확인한 뒤, 실험 전에 고정한 개선 기준과 별도 예산·실행 승인을 모두 충족할 때 양쪽에 같은 두 seed를 추가한다. 설정이나 시드가 어긋난 결과는 비교에서 대기 상태로 유지한다. 시드별 차이와 평균·표본 편차를 보고하며, 표본 3~5개로 통계적 우월성을 단정하지 않는다.

## 회수 후 채우는 순서

1. R2의 `result.json`, 원본 `receipt.json`, `worker-result.json`, `numerical.json`, `pilot-plan.json`, 두 arm 보고서·최종 상태와 `pilot-summary.json`을 회수한다. 노트북 출력에 표시된 `result.json`의 파일 SHA-256을 기록한다. `completed_pilot`와 R2 고정 SHA가 맞는 경우 계획 초안에 첨부한다.
2. **입력과 초기화**. A 부모·세 모델 SHA, frozen frontend·hand·feature identity, 비교할 train 범위와 manifest SHA를 고정한다. 파일럿의 24개 train clip이나 6개 validation clip은 해당 파일럿의 범위다. 본 비교는 같은 공식 validation 421클립·16,425프레임과 평가 mask를 사용하도록 고정하며, 독립 test는 별도로 보존한다.
3. **수렴과 업데이트**. 같은 6개 validation 클립의 0→12→24 step 곡선을 함께 확인한다. 손·몸·손목의 변화와 유효 표본, finite loss·gradient를 확인한다. 짧은 곡선의 오르내림만으로 본시험 우열을 판정하지 않는다. 이 기록으로 양 arm의 동일 update 목표, 중간 validation 경계, 공통 중단 기준, 추가 2 seed 발동 기준을 별도로 사전 고정한다.
4. **비용**. 실제 전체 시간, worker 계측 I/O, 두 상태 파일 크기와 arm별 `fit_and_eval_seconds`를 같이 보고한다. 이 arm 계측값은 학습·중간 validation·저장을 합한 시간이다. 순수 업데이트 처리량과 전체 validation의 소요시간은 계측 범위 밖이므로 이 값만으로 확정하지 않는다. 전체 입력 로드·전체 validation·상태 저장과 SHA readback의 예약을 분리해 각 회차와 6회·최대 10회 비용을 산정한다. 근거가 부족한 항목은 계측 범위와 가정을 기록한다.
5. **본시험 발급과 판정**. 같은 GPU 조건·precision·중단/재개 규칙, 학습·검증 예산과 증거 보관을 고정한다. 공식 3시나리오의 body·hand-global 6점 균등 평균을 주 지표로 사용한다. 손 local·좌우 손목과 유효 표본도 보고한다. 본실행 승인과 예산을 별도로 기록한다.

## CPU 계획 템플릿

`main-plan-template.json`은 초기 6회와 조건부 4회 목록, 결정 대기 필드를 담는다. `v24_main_comparison_plan_draft_20261123.ipynb`에서 `ATTACH_VERIFIED_PILOT`의 기본값은 `False`다. R2 실제 GPU 회수물과 발급 receipt를 준비한 뒤에만 이를 켜서 별도 계획 초안 JSON에 증거를 붙인다.

```text
python -m scripts.v24_main_plan --pilot-directory <회수-R2-폴더> --pilot-release-receipt <R2-발급-receipt.json> --pilot-result-sha256 <회수-result.json-파일-SHA> --output <새-계획-초안.json>
```

검증 과정은 GPU 파일럿 완료 상태, 고정 R2 receipt, 전체 결과 파일의 크기·SHA, 두 arm의 24회 cursor 및 같은 noise·입력, 6클립 평가 곡선을 확인한다. 결과 부착 후 상태는 `pilot_verified_main_decisions_pending`이다.
