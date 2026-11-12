# v1.2 전체 train 학습 연결 설계

2026-11-12 구현·로컬 검토 완료, 실제 사용자 실행 전. 사용자는 완료된 감사를 이용한 학습 진행을 요청했고, root는 고정 학습 인계 전에 실행 연결을 보완하도록 지시했다. 이 문서는 설계와 개발 검증을 관리한다. 새 다운로드·Drive 쓰기·GPU 학습은 수행하지 않았으며, 아래 예산으로 제한된 준비·pilot·재개 실행을 테스트 세션에 인계한다. v1.4 고정본은 보존한다.

## 입력과 별도 진단 계약

- 입력은 `completed_003/reports.zip` SHA256 `30c46b8d3259d9a48eae7f4b94465a5563e3ad6bf4fb74b6b88ef546da8e4c2f`다. 내부 recovery receipt의 파일별 크기/SHA와 seal을 검증한다. 전체 파일 감사와 journal 재실행을 반복하지 않는다.
- `verified/manifest.json` content SHA는 `fc78e2a43e3b178f8b86abbe64c6221755ed4aa5bb309ea13e34d8cc08cb8877`, 바이트 SHA는 `180af995fdc56d474396d71225d6eb2d02b8a90fe83d1e0d022bcfaf28443302`다. 원본의 `train_ready=false`, `temporal_ready=false`를 유지한다.
- 실제 입력 준비는 고정 manifest의 경로·크기·SHA로 **실제 읽는 파일**을 확인한다. 이후 준비 캐시는 자체 source/spec/preprocessing/manifest identity와 파일별 SHA로 연결한다. 이 과정은 전체 감사의 재판정이 아니라 학습에 사용한 바이트의 확인이다.
- 별도 `V1.2-FULL-TRAIN-DIAGNOSTIC-001` 계약에 명목 `frame_index/25` 시간, frame-key 기반 stereo 대응 가정, 미검증 실제 timestamp·stereo 시간 동기·원본 motion 독립성, 참고 validation을 기록한다. 기존89장 GT overlay 검토를 전체23,124페어의 시각 검사로 확대하지 않는다. Velocity·foot loss는0이다.
- Frontend는 train20,888 전체, 참고 validation은 기존2,236 전체를 사용한다. Teacher train은 아래564clip, 비교 validation은 기존29개의 완전50frame clip/1,450frame만 유지한다. 나머지 validation786frame을 숨기지 않고 제외 키와 이유를 보고한다. Test는 사용하지 않는다.

## coverage·padding·학습량

고정 감사 metadata를 기존 `full_archvis.clip_coverage`로 다시 계산한 결과다. 원본 payload는 읽지 않았다.

| Train 구간 | clip | 실제 frame |
| --- | ---: | ---: |
| 완전50frame | 281 | 14,050 |
| 짧은 run | 156 | 4,244 |
| run 마지막 tail | 127 | 2,594 |
| 전체 | 564 | 20,888 |

287개 연속 run,286sequence, frame 번호 gap1개를 분리한다. Train의7,312padding slot은 post-padding하며 입력/target/camera를0으로 채우고 `time_valid=false`, `target_valid=false`를 적용한다. Seconds는 모델 계약에 맞게 증가시키되 padding 시간으로 실제 시간을 주장하지 않는다. 미래 입력·다른 run으로 padding을 채우지 않는다. 실제 train 키는 중복 없이 정확히20,888개여야 한다.

기존 `motion_losses`의 valid-joint 평균과 clip당1update를 유지한다. 따라서 짧은 clip과50frame clip은 각각1update이며 frame별 균등 가중 실험과 같지 않다. 별도 loss 재가중은 이번 범위에 추가하지 않는다. Padding target과 미래 slot을 바꾸어도 실제 frame의 loss/과거 예측이 변하지 않는지 검사한다.

각10epoch를 새 seed71/72/73으로 시작한다. Frontend104,440update(10,444/epoch), body5,640, hand5,640으로 총115,720update다. 기존 v1.1과 모델·128px·frontend batch2·FP32·LR·scheduler·diffusion/loss·hand GT upper train/예측 body validation 규칙을 유지한다. 이전 checkpoint를 pin을 우회해 이어 쓰지 않는다.

저장은 기존 best와 epoch1/5/10에 더해, **각 단계 내부** frontend500/5,000update와 teacher20/200update를 비교 지점으로 제안한다. Full-data 순회의 짧은 prefix이므로 동일 데이터 노출 비교가 아님을 명시한다. 각 teacher의 의존 frontend/body SHA를 보존하며 같은 epoch 번호를 임의로 결합하지 않는다.

## 재사용과 수정 범위

- 재사용: `full_archvis.clip_coverage`, 고정 spec·pose target/camera 변환, `frame_image`의 float32 resize, `predicted_features`, `FrameStage`, `TeacherRun`, `CascadeStages` 평가·loss·RNG, checkpoint의 optimizer/scheduler/scaler/RNG/cursor 저장, 기존 bounded ZIP/결과 보관 도구.
- 새 연결: 고정 감사 ZIP/manifest 검사와 진단 결정, 청크별 입력 준비·재개, 디스크 캐시 frame/clip adapter, frontend 의존 특징 캐시, 전체학습 CLI/config·checkpoint 비교 저장, 고정 notebook 생성기·결과 회수·새 런타임 복원, 전용 테스트.
- 기존 `DiagnosticData`/`ExpansionData`는1,000페어·완전50frame 계약을 유지한다. v1.4 oracle 경로는 수정하지 않는다. 일반 엄격 evidence 가드를 해제하지 않고 별도 v1.2 명령으로 진입한다.
- 기존 runner의 all-RAM 특징 적재는 전체 데이터 adapter에서 디스크 기반 mapping으로 대체한다. 전체clip Python metadata와 소수 clip tensor만 RAM에 둔다. 모델 구조를 바꾸는 일반 리팩터링은 피한다.

## 준비·재개·디스크 제안

Raw source54,422,334,222bytes를 모두 복사하는 대신, 읽은 파일의 SHA 확인 후 원래 변환과 동일한 **float32 128px stereo tensor**를 한 번 저장한다. uint8 재양자화로 baseline 입력을 바꾸지 않는다. 이미지 tensor는9,092,726,784bytes(약8.47GiB), 작은 pose/camera/label과 receipt를 추가한다.

128frame 이하 청크별 배열과 seal을 임시 경로에서 완성한 뒤 게시한다. 중간 실패 청크는 완료로 집계하지 않는다. 준비 cursor는 마지막 완성 청크의 경계다. 재개 시 완료 청크의 identity/해시/키 순서를 확인하고 원본 RGB decode를 반복하지 않는다. 무작위 학습 순서는 유지하며 memory-mapped 배열에서 필요한 frame만 복사한다. 큰 torch 청크 전체를 매 batch 역직렬화하는 방식은 피한다.

선택된 best frontend의 predicted feature는 manifest·frontend weight SHA·spec·preprocessing에 묶어 별도 청크로 저장한다. Body와 hand·학습 호출 재개가 이를 공유하며 frontend가 바뀌면 재사용을 거부한다. GPU 생성 feature를 CPU에 즉시 내리고 clip당 상한을 둔다. 준비 캐시를 checkpoint ZIP에 넣지 않는다.

| 항목 | 검토용 제안 |
| --- | --- |
| 준비1호출 | 7,200초, 종료여유120초, 최대8GiB 원본 논리읽기, 최대128frame/청크 |
| 준비 전체 | 고정23,124페어, source54.42GB를 최소1회, 누적 완료/재시도 비용 구분 |
| CPU working set | 입력128frame chunk 약48MiB, 최대 native stereo decode1페어 + metadata·모델 외4GiB RSS 목표; 실측 전 보장 아님 |
| 준비 로컬 디스크 | 12GiB 상한, 시작 여유16GiB |
| 학습1호출 | 7,200초, 종료여유120초, 완료한 checkpoint 경계에서 pause |
| checkpoint | 25update마다, 단일256MiB·run4GiB·회수2GiB 제안 |
| 호출별 학습량 | 남은115,720update 내에서 시간/지정 stop update에 도달하면 pause; 자동 무한 재호출 없음 |
| 결과 원격 보관 | 종료finally와 별도의 제한된 주기 checkpoint 보관 지점, 새 attempt·해시 readback·실패 시 로컬 보존 |

2026-11-12 테스트 세션이 확인한 사용자 선택은 **입력 캐시의 Drive 보존**이다. 제안 기본 경로는 `REWIND-like prototype/reports/cache/v12_prepared/input-<binding SHA>/`이며 계정의 실제10–12GiB 여유는 미확인이다. 에이전트가 원본/대용량 캐시를 직접 업로드하지 않는 원칙을 유지한다. 생성 notebook의 사용자 수행 셀만 완료 청크를 복사하며 기본 스위치는 꺼져 있다. 모델 결과의 자동 보관과 이 사용자 캐시 보존은 구분한다.

완료 청크 하나마다 `images.npy`, `targets.pt`, `receipt.json`을 deterministic ZIP으로 로컬에서 묶는다. 정확한 파일명은 `chunk-<5자리번호>-<ZIP SHA>.zip`이다. 사용자는 새 경로 복사·크기/해시 readback을 끝낸 뒤 `catalog-<완료청크수>-<catalog 바이트 SHA>.json`을 게시하고 마지막 출력의 정확한 경로·SHA를 기록한다. 기존 파일은 동일 해시를 확인해 재사용하며 덮어쓰지 않는다. 한 번에64MiB 이하 임시 ZIP 하나만 둔다. 캐시 복사는 사용자 호출당2시간(120초 여유), 신규 쓰기8GiB·Drive readback16GiB로 제한하고 무제한 재호출하지 않는다. 준비 호출이 끝날 때마다 완료 prefix를 보존하면 VM 소실의 재작업을 줄일 수 있다.

새 런타임은 같은 고정 소스/입력으로 `cache-restore`에 보존 폴더·catalog 경로·바이트 SHA를 넘긴다. 원본/코드/spec/preprocessing binding과 catalog seal, 각 ZIP/내부 receipt·파일별 SHA를 대조하고 새 청크 디렉터리에 원자 게시한다. 기존 청크는 검증하며 손상본을 덮어쓰거나 최신 로컬 prefix를 오래된 catalog로 낮추지 않는다. 부분 catalog는 실제 prefix만 복원하고 남은 준비를 원본 manifest로 이어간다. 전체 준비의 실제 원본 읽기54.42GB는8GiB 상한으로도 최소7호출이며 시간 한도·최초1청크 검사·실패 시 더 필요하다. 전체 캐시 약9.2GB를 새 런타임에서 읽는 복원도8GiB/2시간 상한에 따라 여러 호출이 필요하다.

브라우저 연결 단절과 VM 종료를 구분한다. VM 자체 종료에는 finally 실행을 보장하지 않는다. 특징 캐시는 재생성 가능한 약124MiB의 로컬 산출물이며 현재 사용자 Drive 보존 대상인 입력 프레임 캐시와 구분한다.

## 디스크·원격 보관 예산 보완

앞 표의16GiB는 준비 단계의 초기 여유 제안이다. 학습·cold restore는 **24GiB 여유**로 보완한다. 보수적 동시 예약은 입력12GiB + 특징0.25GiB + run4GiB + 회수ZIP2GiB + 복원 staging2GiB + 현재 임시 checkpoint0.25GiB + source/감사 metadata 약0.15GiB + 보존 실패ZIP2GiB = 약22.65GiB다. 성공한 자동 결과 ZIP은 원격 크기/해시 readback 뒤 로컬 임시본만 지우고 모델 원본은 유지한다. 보관 실패 ZIP은 남기고 종료 시 최신 paused/failed 상태를 반영한 ZIP을 새로 생성한다. 반복 실패가 누적되면 로컬 공간을 재확인하며 자동으로 실패본을 삭제하지 않는다. 대용량 입력 ZIP은64MiB 청크 하나씩 복원하고 원본 Drive 캐시는 수정하지 않는다. 실제 파일 크기·Colab 여유 공간·Drive 계정 quota는 pilot에서 별도로 확인한다.

로컬25update checkpoint와 원격 보관을 분리한다. 수정 제안은 **60분마다(2시간 호출 내 최대1회) + 호출 종료1회**다. 원격은 동일 run에 최대24attempt와 예약 누적12GiB를 독립 제한한다. 실제 ZIP bytes를 복사 전에 ledger에 예약하고 실패도 차감한다. 24×2GiB를 허용한다는 뜻이 아니다. 주기 예약은 항상 종료 보관용2GiB·1attempt를 남기며 부족하면 안전 checkpoint에서 pause하고 남은 종료 예약으로 보관한다. ZIP 생성·복사·readback 시간도 학습 호출 비용에 들어간다.

학습 budget7,200초는 데이터 캐시 검증·특징 생성·학습·평가·주기 보관을 포함한다. 종료 보관은 추가로 로컬 ZIP 생성600초·원격 복사와 readback 합계600초를 허용하며 notebook child의 hard timeout은 준비/학습 각각8,460초다. 캐시 복원은 원격 결과 게시가 없어7,330초다. 종료 보관을 포함한 호출 시간이2시간을 넘을 수 있음을 표시한다. 외부 hard timeout에도 로컬 fallback 회수를 시도하며, 중단된 원격 lease를 자동 탈취하지 않는다.

v1.1 실제 member 크기는 frontend1,738,999/body60,195,951/hand62,446,503bytes, hand last187,136,889bytes다. 각 role의 best1+epoch1/5/10모델3+비교update모델2를 모두 별개로 상한 추산하면 frontend 약10.4MB, body까지 모델 약371.6MB, hand까지 약746.3MB이며 마지막 상태를 더한 최종ZIP은 약933.4MB + 늘어난 binding/JSON이다. Frontend 단계는 이보다 작고 body 말기는 약0.55GB다. 이는 고정 구조의 기존 파일 크기 기반 추정으로 실제 v1.2 결과 크기는 미측정이다. 필요한 학습 의존 모델과 비교 보존 모델을 같은 ZIP에 두어 복원 절차를 단순하게 유지한다.

2.9시간/11시간 예시의 최대4회/12회는 준비·특징 생성·검증·ZIP/복사·재시작 비용을 더하면 늘어난다. 긴 frontend 단계의 작은 ZIP과 뒤 단계의 큰 ZIP을 나눠 비용을 기록하며 전체 횟수에 최종ZIP 크기를 일률 적용하지 않는다. 사용한 누적 bytes·횟수·실패 예약과 남은 종료 여유를 보고하고 자동 예산 초기화는 하지 않는다.

운영 계약은 **한 run에 한 active writer**다. 새 런타임 재개 전에 이전 호출이 종료됐는지 확인한다. Ledger seal·기존 attempt·lease readback 불일치, 관측 가능한 경합·stale lease에서는 fail closed하고 자동 lease 탈취를 하지 않는다. Drive mount 파일의 생성/readback만으로 서로 다른 Colab 간 원자적 분산 lock이나 전역 예산 보장을 입증했다고 주장하지 않는다. 실제 Drive consistency는 미검증이며 stale lease의 정리는 root 검토가 필요하다.

## 비용 근거와 미확정 항목

기존 A100 약11.2시간 추정은 완전clip281개 기준이며 최초 준비와 short/tail을 제외했다. 이번564개 teacher clip·반복 decode 제거 설계에 그대로 적용하지 않는다.

T4 v1.1 실측의 frontend10epoch compute441.095초를 train비율20.888로 단순 외삽하면 약9,214초(2.56시간), body/hand train step31.480초를 clip비율28.2로 외삽하면 약888초다. 고정 validation compute는 frontend180.751초·teacher약129초다. 합계 **약2.9시간은 계산 구간 중심의 구성 요소 외삽**이며 캐시 I/O·학습 오버헤드·특징 생성·저장·원격 보관·재시작을 빠뜨린 실행 예측치다. 실제GPU/VRAM·step당 비용을 작은 pilot에서 계측하고10epoch 비용을 갱신한다.

반복 decode가 남는 기존 T4 frontend train wall1,809.105초를 그대로 확대하면 약10.50시간이며 고정 validation3,110.615초·teacher 확대 등을 더해야 한다. 캐시로 이 차이를 줄일 가설은 있으나 속도 개선을 아직 입증하지 않았다. 최초 준비는 별도다. 원본 감사에서54.43GB 읽기에 누적약10시간이 관측됐으므로 유사 I/O의 장시간 비용 가능성을 유지하고, 추가 decode·9GB 기록은 별도 계측한다. 이 감사 관측을 새 준비 시간의 확정 상한/하한으로 쓰지 않는다.

실행 전 필수 근거: 현재 RGB/pose root의 고정 입력 실제 읽기 SHA, 캐시 전체 coverage·해시·변환 동등성, short/tail·gap·causality·mask·완전 재개 검사, Colab pilot의 GPU/메모리/IO/디스크 실측, 호출별/누적 예산과 입력 캐시 생존·복원 정책, 주기/종료 원격 결과 보관 및 실패 검증, 정식 revision/source ZIP/notebook pin 발급이다. 현재 문서로 이 항목이 완료된 것은 아니다.

## 계보·테스트 인계 초안

계보 추가안: v1.2 완료 감사 → 전체20,888train을 short/tail 포함해 학습에 연결하는 개발 단계. 가설은 동작·자세 coverage 확대 효과이며 v1.1 대비 update 수도 늘어 순수 규모 효과와 분리되지 않는다. v1.4 oracle 진단과 별도 branch를 유지한다. 실제 학습·성능 결과는 미확보다. 공유 계보 문서와 상단 분기 트리를 함께 갱신했다.

테스트 세션에는 root가 검토·고정한 revision/source ZIP/notebook·config/감사 ZIP/input ZIP SHA와 정확한 경로, 준비/복원/pilot/학습 실행 순서, 호출별 예산·중단 조건, 캐시 보존 절차, 결과 회수 경로를 함께 인계한다. 우선 기본 RUN=false, 파일 감사 반복 없음, 준비1청크 및 동일 변환 확인 → 제한 pilot → 고정10epoch 재개 호출 순서를 검토한다. 실행 중 설정·입력·예산을 임의 변경하지 않는다. 결과는 명목 시간 진단·참고 validation으로 표시하고 실제 timestamp/motion 독립성이나 정확도 개선을 미리 주장하지 않는다.


## 로컬 검토 결과

Root가 최종 신규15검사를22.064초에 통과했다. 청크의 정확한 변환·중단/재개·손상 거부, short/tail mask·gradient·미래 불변성·sequence reset, 모든 단계의 합성 학습 재개 일치, 캐시 보존/복원, 원격 예산/실패 보존, 최신 종료 상태와 준비 진행 회수를 확인했다. 실제 Drive 대신 로컬 fixture를 사용했다.

기존 경로를 포함한 최초 root51검사는50개 통과·기존 pause/resume 검사에서 Windows atomic replace WinError5가1회 발생했다. 해당검사 단독 재실행은12.510초 통과했지만 원인은 미확정이다. 개발자49검사 전체 통과 후 확장51검사에서는 다른 기존 평가실패 복구 검사에 같은 접근 오류가 있었고 단독 재검사는 통과했다. 이를 모든 확장검사의 일괄 정상종료로 보고하지 않는다. 기존 atomic 저장을 임의 변경하지 않았다.

실제 전체 입력 준비·CUDA 학습·Drive 읽기/쓰기·새 런타임 복원과 속도 개선은 아직 검증하지 않았다. 정식 Git 소스 ZIP과 생성 notebook의 bootstrap·기본no-op 결과 및 정확한 파일 식별값은 배포 폴더의 root_release_review.json·handoff_receipt.json·TEST_HANDOFF.md에 보존한다.
