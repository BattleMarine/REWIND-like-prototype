# v1.5.1 상체 조건 교체 진단 결과

2026-11-17 검토. **GT 상체 입력으로 손 local 오차는 평균 10.8% 감소했지만 손 축소는 해소되지 않았다. 예측 상체의 오류 전파는 일부 기여하며, 손 모델 자체·2D 입력·학습량·loss·sampling의 잔여 영향은 미분리 상태다.**

## 실행과 고정 조건

- 실행 ID: `V151-UPPER-CONDITION-001`, 소스 revision `00f5c8b`, source identity `88ae929113fe12b9d05d8593cc07e4189db62573baf0d9341e3fdc9fa2ecc939`.
- Tesla T4 / Torch 2.14.1+cu130 / FP32 / seed 71 / DDIM 10. 재학습과 optimizer update는 없다.
- v1.5의 first_midpoint body·hand u1000 모델, 저장된 F9 특징·camera·keyed noise와 body 예측을 고정했다. 원본 checkpoint SHA와 입력 pin은 [실행 계약](V151_UPPER_CONDITION.md)을 따른다.
- 기존 ArchVis·CityPark·Downtown의 같은 인물 6개 개발·회귀 클립, 각 50프레임이다. 독립 일반화 표본이나 새로운 300프레임으로 합산하지 않는다. 명목 25FPS 가정과 시간·motion 독립성의 기존 제한을 유지한다.
- A는 예측 상체, B는 GT 상체를 hand에 입력한다. 각 local 손 출력에 예측 wrist 또는 GT wrist를 부착하는 2×2 후처리를 별도로 비교했다.

## 정확도와 형태

공통 device 좌표계 mm, 손 40관절의 동일 유효 mask를 사용한다. 각 클립의 분모는 2,000관절이며 전체 12,000관절 관측의 평균이다. Local은 wrist-relative, global은 wrist 부착 후 오차다.

| 클립 | A 예측 상체 local mm | B GT 상체 local mm |
| --- | ---: | ---: |
| ArchVisInterior_ArchVis_RT / Standing__1_ | 69.18 | 71.02 |
| ArchVisInterior_ArchVis_RT / Petting_Animal__13_ | 77.67 | 50.83 |
| CityPark_Showcase / Gunplay__13_ | 102.78 | 85.67 |
| CityPark_Showcase / Kneeling_Inspecting__1_ | 99.97 | 91.14 |
| DowntownWest_Demo_Environment / Reloading | 64.21 | 66.40 |
| DowntownWest_Demo_Environment / Quarterback_Pass | 71.30 | 67.76 |
| 전체 | 80.85 | 72.14 |

4개 클립에서 개선되고 2개에서 악화됐다. 평균 8.71mm(10.8%) 개선이며 모든 동작의 일관된 개선은 아니다.

| 지표 | A 예측 상체 | B GT 상체 |
| --- | ---: | ---: |
| 예측 wrist 부착 global MPJPE mm | 381.42 | 359.70 |
| GT wrist 부착 global MPJPE mm | 80.85 | 72.14 |
| 평균 손 뼈 길이 mm | 8.20 | 7.99 |
| GT 평균 손 뼈 길이 mm | 23.71 | 23.71 |
| 손 spread mm | 29.76 | 28.90 |
| GT 손 spread mm | 60.83 | 60.83 |

GT 상체를 넣어도 뼈 길이는 GT의 약 33.7%이며 A의 약 34.6%보다 회복되지 않았다. Body는 고정돼 평균 297.71mm다. GT wrist 부착 시 local과 global 오차가 같아지는 것은 좌표 부착의 성질이며 손 모델 개선 자체가 아니다. Global/local 평균을 단순히 빼서 wrist 오차나 기여율을 계산하지 않는다.

## 판단과 남은 가설

- 예측 상체 조건은 평균 손 오류에 일부 기여한다. 그러나 GT 상체로도 축소가 유지되므로 상체 오차만으로 손 실패를 설명할 수 없다.
- GT 상체 13관절에는 wrist도 들어 있다. 2×2 비교는 출력의 wrist 부착 효과를 분리하며 조건 입력 내부의 wrist 기여까지 독립 분해하지 않는다.
- GT 조건의 결과는 진단이며 실제 예측 성능이나 배포 성능이 아니다. A의 실제 예측 조건 결과와 구분한다.
- 후속 후보는 중점 표현과 GT 상체를 유지한 단일클립 GT/예측 2D 통제 학습 및 학습량별 형태 복원 확인이다. 정규화·loss·디퓨전 학습/샘플링 문제는 후보이며 구현 오류로 확정하지 않는다. v1.5.2 이후 번호·설정·예산·실행은 미확정이다. v1.3 보류와 별도 v1.6 폐기를 유지한다.

## 검증 범위와 시간

Root 보존 검토 기록에서 ZIP CRC·내부 34파일 SHA·JSON seal 7개를 확인했다. 저장 예측으로 local/global MPJPE·bone/spread를 독립 재계산했으며 평균 오차의 최대 차이는 약 1.14e-13mm였다. A body/local은 부모 v1.5와 정확히 같고 A 반복도 정확히 같다. 모델 전후 fingerprint·GT/입력 연결과 wrist 부착 관계를 확인했다.

보고된 full50/prefix25 최대 차이는 0.00042725mm로 허용 범위다. Prefix tensor는 저장되지 않아 이 항목은 보고값 검사이며 root GPU 재실행 검증이 아니다. 이번 문서화에서 새 학습·추론이나 GIF 전체 시각 검토를 수행하지 않았다.

내부 평가 report 경과 시간은 39.16초, operation의 infer 단계 wall-clock은 50.47초다. 준비·Drive 보관·전체 Colab 작업 시간이나 순수 모델 레이턴시와 혼동하지 않는다. GPU peak allocated는 214,657,536B이며 총 VRAM 사용량이 아니다.

## 원본과 보존 근거

- [결과 ZIP](../outputs/v151_upper_condition_20261115/user_test/completed_001/results.zip): 15,044,584B, SHA256 `7744f29ed980bc9946863d9aaa87c66cdd0726e34e77878e98f1b49f9c01608d`.
- [Root 검토·클립별 수치](../outputs/v151_upper_condition_20261115/user_test/completed_001/root_review.json).
- [실행 report](../outputs/v151_upper_condition_20261115/user_test/completed_001/extracted/inference/report.json), content SHA `f810c75f28ac7eb21c856a0fb4b613ebbef79a18f4cfeebed53e4f4b66cac56a`.
- [Drive receipt](../outputs/v151_upper_condition_20261115/user_test/completed_001/drive_receipt.json): 같은 크기·SHA의 `verified` 기록. 원격 Drive를 root가 재조회한 것은 아니다.
- Drive: `REWIND-like prototype/reports/V151-UPPER-CONDITION-001/run-60f3a25621894f1999f1401af0f32db3/inference/attempt-vorz94qh/results.zip`.
- GIF·PNG·예측 JSON은 결과 ZIP의 `inference/clip-00`~`clip-05`에 보존한다. 로컬 산출물은 Git 제외다. Downloads의 일반 이름 `results.zip`보다 이 보존 경로와 SHA로 식별한다.

[부모 v1.5 결과](V15_CAMERA_TRAINING_RESULTS.md) · [종합 보고서](EXPERIMENT_RESULTS_REPORT.md) · [실험 계보](V1_EXPERIMENT_LINEAGE.md).
