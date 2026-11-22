# 출력층·LoRA의 동일 클립 비교

CPU 런타임에서 같은 학습 seed의 출력층과 LoRA를 비교한다. 공식 validation의 세 시나리오, 421클립·16,425프레임을 seed 42·71·103·137·181마다 대조한다. CSV의 before는 출력층, after는 LoRA이고 양수 차이는 출력층의 오차가 더 작다는 뜻이다. 몸·양손 손목·전역 손·손목 기준 손가락 오차를 기록한다. 클립 동일 가중 통계는 공식 종합 점수와 구분한다. 공식 시나리오별 몸·전역 손 오차 60개(2방식 × 5시드 × 3시나리오 × 2지표)는 유효 관절 수로 가중해 0.01 mm 이내 일치를 요구한다.

1. CPU에서 outputs/v2_plan_inputs/v2_final_paired_cpu_verified_20261123/v2_final_paired_cpu.ipynb를 연다. SHA-256은 e519780427c10c7792785895677f0b740586f509207dc2fb094755c4b5c856cc다.
2. DRIVE_PROJECT를 확인하고 RUN_COMPARE를 True로 지정해 실행한다. 모듈과 10개 요약의 해시가 내장되어 있다. 네 기존 결과 폴더의 요약·그룹 영수증·NPZ만 읽는다. CPU worker 최대 780초, 마운트 포함 worker 종료 목표 900초, 입력 512 MiB, 출력 파일당 4 MiB, 필요 임시 여유 128 MiB다. Drive 결과 게시 대기는 별도다.
3. reports/v2_final_paired_cpu/run-UUID에서 result.json, paired-clips.csv, paired-summary.json을 회수한다. passed, 파일별 해시, 2,105행을 확인한다. 누락·해시·좌표·GT·mask 불일치나 시간 상한에 걸리면 해당 UUID 진단을 보존한다.

고정본 로컬 검사에서 기존 A의 421클립과 시나리오별 몸·전역 손 오차를 재현하고, 같은 예측을 양쪽으로 넣은 대조에서 2,105행 CSV 기록을 확인했다. 실제 두 방식 전체 대조는 Colab에서 실행한다.
