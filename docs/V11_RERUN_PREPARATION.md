# v1.1 재실행 준비

2026-11-10 사용자 요청으로 준비한다. 이전 실행은 완료 로그만 확보했고 모델 파일은 미회수다. [이전 결과](V11_EPOCH_EXPANSION_RESULTS.md)를 보존하며 새로운 실행 결과와 섞지 않는다.

## 고정 학습 조건

동일 train1,000페어, frontend/body/hand 각각10epoch, 총5,400update를 유지한다. 기존 config의 seed71·FP32·batch2·frontend LR1e-4·teacher LR5e-5·모델 구조를 변경하지 않는다. Frontend validation2,236 및 teacher validation29clip/1,450frame도 유지한다. 명목25FPS·frame-key stereo 대응과 원본 motion 독립성 미확인에 따른 참고 지표 조건은 같다. 기존1epoch/10epoch 실행과 결과를 분리한다.

2026-11-11 사용자 결정에 따라 학습의 A100 이름 제한을 제거한다. 지원되는 CUDA GPU(T4·RTX 등)를 확인하고 장치명·VRAM을 기록한 후 시작한다. A100은 추후 추론 레이턴시 비교 조건으로 분리한다. 다른 GPU에서는 메모리와 처리 시간이 달라질 수 있으며 기존 상한을 자동 확대하지 않는다. 학습 호출2시간, cooperative 종료 여유120초, run2GiB·회수 ZIP1GiB 등 기존 학습 상한을 유지한다. 이전 실제 경과88분31초는 참고치이며 이번 완료 시간을 보장하지 않는다. 입력 raw cache는 사용자 Drive의 기존 보존본을 검증해 재사용하고 캐시 업로드를 자동 수행하지 않는다.

## 보관 절차 변경

재실행 notebook에서 사용자가 실행을 켜면 Drive의 결과 보관 위치와 새 attempt 폴더의 쓰기 가능성을 학습 전에 확인한다. 학습 호출 종료 후 성공·실패·정상 pause 모두 로컬 회수 ZIP을 만들고, Drive의 새 attempt에 복사한 뒤 복사본을 다시 읽어 크기와 SHA256을 확인한다. 보관 확인 receipt를 남긴다. 브라우저 다운로드는 선택이며 기본으로 요구하지 않는다.

이전 결과 파일은 덮어쓰지 않는다. 보관 실패 시 로컬 ZIP을 보존하고 실패를 보고하며, 수동 recovery 셀에서 재시도할 수 있다. 학습 프로세스가 끝나기 전에 VM 자체가 종료되면 finally 실행이나 Drive 보관을 보장할 수 없다. 학습 도중 주기적 동기화는 이번 범위에 포함하지 않는다.

## 테스트 인계와 성공 기준

새 고정 소스 ZIP·notebook·receipt와 기존 입력/근거 ZIP의 해시를 함께 전달한다. 사용자 실행 안내와 최종 결과 취합은 기존 테스트 세션이 담당한다. 실제 학습은 이 준비 작업에서 시작하지 않는다. v1.2의 실행·소스·상한은 바꾸지 않는다.

학습 completed 및5,400update 확인과 모델 보관 완료를 구분한다. 결과 ZIP의 실제 본문, 복사 검증 receipt, run.json, best/milestone/last checkpoint를 확보해야 모델 회수 완료다. 콘솔에 표시된 해시만으로 보관 완료로 판정하지 않는다. 검증된 복사본이 확인되기 전 런타임을 종료하지 않는다. 이후 root가 모델 본문·dependency와 epoch별 결과를 검토한다.

## 구현·검토 범위

`scripts/v11_recovery.py`의 prepare/package/publish와 v1.1 notebook 분기를 보완했다. Drive mount 확인 후 쓰기 시험60초, ZIP생성 내부600초/외부630초, 복사·재읽기 내부600초/외부630초를 둔다. 학습 호출2시간과 별도 보관 시간이다. `/content/archvis_v11_rerun`을 사용하며 Drive 대상은 `REWIND-like prototype/reports/v1_branch_experiments/v11_rerun/attempt-*`다. 로컬 회수본은 WORK/recovery/local-*에 남는다. Drive에 결과 약1GiB 이상과 실패 partial의 추가 공간을 확보한다.

Drive receipt의 `state=verified`, `kind=training`, `last_sha256`과 실제 ZIP 본문을 확인한다. `kind=status`는 모델 확보가 아니다. 마운트 파일시스템을 통해 복사본을 재읽는 검증이며 독립 Drive 서버 조회나 강제 VM 종료에 대한 내구성 보장은 아니다.

Root가 기존 진단 및 새 회수 검사9개를 재실행해 통과했다. 실제 core recover의 작은 checkpoint ZIP 생성·복사, 덮어쓰기 방지, 손상/시간초과 후 로컬 보존, 새 attempt 재시도, mount 누락 및 기본 no-op, 학습 오류와 회수 오류 분리를 검사했다. 실제 Colab GPU·Drive 파일시스템 시험은 아직이며 재학습 성공으로 표시하지 않는다.

2026-11-11 후속 변경: 학습 gate는 T4·RTX·L4를 허용하며 실제 CUDA 사용 여부를 유지한다. 개발 검증31개에서 장치 gate와 기존 입력·재개 검사를 확인했다. 실제 해당 GPU에서의 학습 성공·VRAM 적합성은 미검증이며 이전 A100 시간과 구분해 실측한다.
