# 실제 데이터 확보 전략 검토

2026-10-21 · 기준 revision `f6ef5c3e9a535d76360f0fbfd0a56dc50b7764eb` · M0 후속 계획 초안

**권고하는 다음 개발 묶음은 제한된 압축 prefix 보관·재사용과 sequence/frame 관측 카탈로그다.** 기존 보관본을 먼저 대조하고, 이 기능을 로컬에서 검증한 뒤에만 한 번의 확장 탐색을 별도로 결정한다. 같은64MiB 작업의 반복이나 근거 없는 sequence 전환을 다음 실행으로 권고하지 않는다.

이 문서는 저장된 보고서·ZIP·파일·현재 코드를 오프라인으로 대조한 결과다. 새 원본 요청, 코드·예산 변경, 학습, 사용자 재실행 요청을 하지 않았다. 아래 확대 수치는 비용 시나리오와 제안이며 승인된 preset·D2 확정 예산이 아니다. M0의 좌표/영상·시간·split 검증과 실제 baseline 완료 상태도 올리지 않는다.

> 후속 상태: 이 문서의 비용 비교와 제안은 `703c5f3`의 계획 기록이다. 이후 사용자 승인으로 API7 캐시·관측 보존을 구현했으며 실제 적용한 제한과 검증 상태는 [로드맵](ROADMAP.md) 및 [압축 준비 도구](ARCHIVE_PREPARATION.md)를 따른다. 아래 미구현·제안 표현과 1,155요청 예시는 계획 당시 상태이며 현재 실행 설정을 덮어쓰지 않는다.

## 1. 확인한 자료와 남은 미확정 사항

| 근거 | 이번에 확인한 내용 | 제한 |
| --- | --- | --- |
| [API6 verification](../outputs/test_archvis_pose_retry_20261021/verification.json), [독립 검토](../outputs/test_archvis_pose_retry_20261021/orchestrator_review.json), 같은 폴더의 `archive_job_archvis_pose_1_10_retry_001.zip` | ZIP 내 extraction/pilot/metadata/job 수치와 종료 상태 | 원본 전체 무결성·목표 위치는 미확정 |
| [API5 결과](../outputs/test_archvis_pose_diagnostic_20261019/verification.json), [API4 결과](../outputs/test_archvis_pose_density_20261019/verification.json), [이전 header 상한 결과](../outputs/test_archvis_pose_counterpart_20261017/verification.json) | 이전 prefix와 최신 실행의 겹친 완료 Range hash·원본 identity | 부분 hash 일치는 전체 archive 무결성 증명이 아님 |
| [왼쪽 추출 결과](../outputs/test_archvis_left_counterpart_20261017/verification.json), [RGB scan 결과](../outputs/test_archvis_prefix_scan_20261017/verification.json) | 왼쪽 완전 PNG2개와 오른쪽 경로 관측 | Scan은 payload를 저장하지 않음. 과거 오른쪽 성공 ZIP은 현재 미수령 |
| [ArchVis pose 감사](ARCHVIS_POSE_AUDIT.md), `data/raw/colab_archvis_pose_20261015/bundle/pilot/` | 기존32개 JSON의 파일 크기·SHA-256을 extraction manifest와 다시 대조 | 대응 RGB와 원본 motion 독립성·시간 검증은 별도 |
| `outputs/archive_efficiency_20261016/root/replay_allowed_rgb/`, `replay_allowed_pose/` | 저장된 공식 CityPark prefix의 기존 replay 보고서와 파일 이름 | 사용자 Drive/Colab 시험과 구분. 완전한 stereo clip이 아님 |
| [현재 Source 코드](../rewind_archive/sources.py), [추출 코드](../rewind_archive/extraction.py) | 예약·계수·byte0 시작·LocalSource·gzip 상태의 실제 구현 | 지속 cache·hybrid source·gzip resume는 미구현 |

`outputs/`와 `data/raw/`는 Git 제외 개인 근거다. 새 checkout에 자동으로 포함되지 않는다. 원본 ZIP·PNG·JSON과 과거 보고서를 수정하지 않았다. 계산 근거는 `outputs/data_acquisition_strategy_20261021/developer/offline_calculations.json`에 보존한다. 인터넷의 현재 상태나 새 Drive 목록은 조회하지 않았다.

## 2. API6에서 실제로 지불한 비용

대상은 ArchVis Serena `003/Getting_Up__1_` frame1·10 pose다. `cb76a3c`/API6에서 timeout 없이 다음과 같이 종료했다.

| 항목 | 실제 관측 | 당시 상한 |
| --- | --- | --- |
| 압축 media body | 67,030,196bytes = 63.924976MiB | 별도 media 전용 상한이 아닌 아래 body 예산을 공유 |
| 준비 응답 body 합계 | 67,076,276bytes = 63.968922MiB | 64MiB = 67,108,864bytes |
| 준비 요청 | 768회 = media256 + metadata512 | 768회 |
| 준비 경과 시간 | 232.180825초 | 협력적300초 |
| 누적 해제 출력 | 368,062,368bytes = 351.011627MiB | 512MiB |
| Tar header / regular | 8,652 / 4,214개 | Header10,000개 |
| Tar metadata payload 누적 | 541,603bytes | 1MiB, 단일 payload64KiB |
| 저장 / 목표 완료 | 0파일 / 0개 | 목표2개, 파일당4MiB, 출력16MiB |
| Retry / wait / 최종 진단 | 0회 / 0초 / `request_failure=null` | 추가시도 최대3회 정책 |

MiB는1,048,576bytes다. 최초 metadata는 위 준비 비용과 별도로1요청·290bytes·0.158772초다. 인증·업로드·사전 검사·ZIP 생성 및 전체 wire 비용은 준비시간/body에 포함되지 않는다. 타임아웃은 socket/협력적 제한이며 OS hard deadline이 아니다.

`ranges_limit`, `targets_status=incomplete`, `sample_acquired=false`다. 빈 파일 목록의 `verification=passed`는 pose 확보 성공이 아니다. 이번에는 retry가 발생하지 않았으므로 timeout 정책의 효과나 과거 실패 원인의 해결을 입증하지 않는다.

첫128개 regular 관측과 마지막 Kyle `017/Dismissing_Gesture__1_` frame24의 미완료 header만 남았다. Regular4,214개 전체 이름 목록은 저장되지 않았다. 이 목록은 도달한 header 진단이며 전체 archive의 목표 부재·정렬·seek offset을 증명하지 않는다. Serena pose 경로는 RGB 이름에서 파생한 후보이며 실제 pose payload는 아직 미확보다.

### 요청 횟수만 늘릴 수 없는 이유

현재 코드는 media를 열기 전에 pre+media+post3요청과 요청 media 길이+metadata32KiB를 예약한다. 마지막 Range는 `[66,846,720, 67,030,195]`의183,476bytes로 짧아졌다. 실제 전후 metadata는 이번에 응답당90bytes였지만, 개시 전에는 각각16KiB 여유를 잡는다.

종료 때 body 잔여량은 **32,588bytes**로 다음 metadata 예약32,768bytes보다 작다. 요청 상한만 올려도 다음 media를 열 수 없다. Body만 올리면768요청에서 다시 멈춘다. 두 값과 header·해제·시간·tar metadata 상한을 함께 검토해야 한다. 준비 예산을64MiB의 순수 압축 입력으로 해석하면 안 된다.

### 지금까지 반복한 prefix

| Pose 시험 | 압축 수신량 | 종료 이유 |
| --- | --- | --- |
| 2026-10-17 counterpart | 7.5MiB | Header1,000 상한 |
| 2026-10-19 API4 | 9.5MiB | 요청 예외, usable=false |
| 2026-10-19 API5 결과 | 14.5MiB | 사전 metadata read timeout, usable=false |
| 2026-10-21 API6 | 63.924976MiB | 768요청 상한 |

이 네 번의 압축 수신 합계는100,060,340bytes = 95.424976MiB다. 모두 byte0부터 시작했다. 원본 identity가 같고, 최신 실행과 겹친 이전 완료 Range30·38·58개의 hash가 각각 일치했다. 최대 prefix와 비교한 중복 압축 수신량은33,030,144bytes = **31.5MiB**다. 이 비용에는 초기 metadata·전후 검사·인증이 더해진다. 각 시험의 진단 가치는 있었지만, 저장된 보고서의 Range hash만으로 받은 압축 bytes를 복원하거나 이어 읽을 수는 없다.

## 3. 더 넓은 pose 순차 탐색의 비용 시나리오

아래는 **압축 입력 목표량**을 늘렸을 때의 계산이다. 현재 prefix의 평균 처리율과 header·해제·metadata 밀도가 뒤쪽에서도 같다는 가정이며 예측 보장·상한 증명이 아니다. 동작 종류, JSON 크기, 압축률, 네트워크 및 runtime에 따라 달라진다.

`C=67,030,196`, `T=232.180825`, `D=368,062,368`, `H=8,652`, `P=541,603`을 기준으로 압축 목표량`X`에 대해 `T·X/C`, `D·X/C`, `H·X/C`, `P·X/C`를 사용했다. 정상 요청 수는 `3·ceil(X/262144)`이며 초기 metadata·실패 시도는 별도다.

| 압축 입력 시나리오 | 정상 준비 요청 | 시간(초) | 해제 출력(MiB) | Header 추정(올림) | Tar metadata(MiB) |
| --- | --- | --- | --- | --- | --- |
| 64MiB | 768 | 232.5 | 351.4 | 8,663 | 0.517 |
| 80MiB | 960 | 290.6 | 439.3 | 10,828 | 0.646 |
| 96MiB | 1,152 | 348.7 | 527.1 | 12,994 | 0.776 |
| 128MiB | 1,536 | 464.9 | 702.8 | 17,325 | 1.034 |

Body 예산에는 압축 media와 전후 metadata를 더한다. 이번90bytes/metadata 응답이 유지된다면96MiB 압축의 정상 body는100,732,416bytes,128MiB는134,309,888bytes다. 마지막 Range 전32KiB 예약까지 만족하는 최소 body 예산은 각각100,765,004bytes와134,342,476bytes다. 이는 **관측한 metadata 길이를 고정한 계산**이며 안전 상한이 아니다. 모든 metadata가 허용 최대16KiB라면 각각108MiB·144MiB body가 필요하다. 무응답 retry도 요구 횟수와 시간을 소비하며, 전체 추가3회라면 정상 요청 수에 최대3회를 더한다.

현재 다른 상한에 도달할 압축량을 같은 밀도로 역산하면 header10,000은 약73.885MiB,300초는 약82.597MiB,해제512MiB는 약93.244MiB,tar metadata1MiB는 약123.763MiB다. Body/요청만96MiB 상당으로 늘리면 먼저 header에서 멈출 것으로 예상한다.128MiB에서는 tar metadata도 검토 대상이다. 모든 상한을 완화해도 목표 확보는 보장되지 않는다.

Pose 원본의 metadata상 크기는368,145,316bytes(약351.091MiB)다. 같은 밀도를 전체에 외삽하면 정상4,215요청·약1,275초·해제 약1,927.8MiB·header 약47,519다. 전체 읽기 권고도 최악 비용의 보장도 아니다. 전체 다운로드/추출을 기본 방침으로 삼지 않으며,Serena의 위치와 후보 경로의 실제 일치는 미확정이다.

## 4. 기존 pose·왼쪽·오른쪽의 교집합

키는 환경·시간대·인물·avatar·번호·motion을 포함한 **완전한 sequence 경로와 frame 번호**다. Frame 번호나 인물/motion 약칭만 일치한다고 대응시키지 않는다.

| 자료 | Sequence / frame | 보관·검증 상태 | 다른 모달리티와의 관계 |
| --- | --- | --- | --- |
| ArchVis pose32 | Adanna `025/Brutal_Assassination`:0~17, `017/Opening`:5,32,36,50,54,61,65,88,90,94,113,117,122,126 |32파일 개별 hash 일치. Brutal18은 연속 번호,Opening14는 gap 있음 | Serena PNG와 sequence 불일치 |
| ArchVis 왼쪽 PNG2 | Serena `003/Getting_Up__1_`:1,10 | `84e86ae` 결과ZIP과 보관PNG. 완전 저장·검증 성공 보고 | 같은 sequence pose 저장0. 오른쪽 성공 원본ZIP 부족 |
| ArchVis 오른쪽 RGB scan | 같은 Serena:10,1,6,8,11,9 payload 완료,7 미완료 | Scan 저장0. 완료 관측은 PNG 보유·내용 검증이 아님 | 왼쪽과 관측 키가 겹치는 frame은1,10. Pose32와는0 |
| 과거 오른쪽 성공 보고 | 같은 Serena:1,10 | 과거 성공 검토 기록은 있으나 `rewind_pilot_2c11fc10.zip` 등 성공 원본은 현재 로컬 미수령 | 원본 회수 후 개별hash·보고서 대조 필요 |
| 공식 서버 오른쪽10 보관물 | 같은 Serena:10 | 해당 실행은 `http_body_failed`/unusable. 과거 성공물과 hash가 같아도 채택하지 않음 | 성공한 오른쪽 원본의 대체물로 사용하지 않음 |
| CityPark 기존 replay | 왼쪽은 Serena `028/Defender__1_`:40,91,pose는 Adanna `008/Stroke_Shrugging__1_`:산재16개 | 저장된 공식 prefix의 기존 부분 검증. 오른쪽 없음 | Sequence 불일치,stereo+pose 교집합0 |

ArchVis 보관pose32와 왼쪽2의 sequence-frame 교집합은0,오른쪽의 완료 관측과도0이다. 성공 원본을 현재 보관·검증할 수 있는 범위의 완전 RGB좌우+pose triplet도0이다. API6의 미보존 regular4,214개 전체에 대해 교집합0을 입증한 것은 아니다.

저장된 공식 split에서 Serena의 위 완전 sequence는 train.txt240행에 있고 validation/test에는 없다. Adanna 기존2sequence의 train 소속도 [과거 독립 검토](ARCHVIS_POSE_AUDIT.md)에 기록됐다. 원본 motion 독립성·archive 간 내용·물리 시간은 별도다. Frame1·10 두 장이 확보돼도 temporal용 연속 clip이나 독립 validation이 완성되는 것은 아니다.

**Sequence 전환의 비용:** Adanna Brutal은 pose18개를 재사용할 수 있으나 대응 좌우RGB 위치가 미관측이다. ArchVis RGB 과거 목록5part·52,441,303,081bytes는 목표까지의 필요 전송량이 아니다. Pose 준비비를 아끼고 RGB 탐색비가 더 늘 수도 있다. Opening은 pose gap을 채우는 비용도 든다. CityPark도 현재 왼쪽/pose가 불일치하고 반대 카메라가 없다. 작은 prefix가 저렴한 대응 sample을 보장하지 않는다. 현재는 왼쪽2와 오른쪽의 알려진 경로를 활용하는 Serena를 유지하되, 위치 관측이 생기면 다시 비교하는 편을 권고한다.

## 5. 반복 prefix 비용을 줄이는 대안

| 대안 | 초기·반복 비용과 효용 | 현 상태·판단 |
| --- | --- | --- |
| 기존 PNG/JSON/ZIP 재사용 | 보관본 로컬 검증의 원본 읽기0. 확보 payload의 재추출을 줄임 | 먼저 가능한 작업. 오른쪽 성공ZIP과 Drive 영구보관은 미확인 |
| 동일64MiB 재실행·timeout/요청만 증가 | 매번 같은 prefix. 이번 retry0이고 body 상한도 남음 | 비권고 |
| Body 확대만 시행 | 매회 byte0부터96/128MiB 등을 다시 읽음. 실패하면 다음에도 중복 | 여러 상한을 함께 검토해야 하며 보관 없는 반복 확대는 피함 |
| **제한된 압축 prefix cache + 관측 카탈로그** | 최초 원본 읽기와 별도disk/보관비 필요. 다음에는 보관구간을 로컬 재생하고 후속만 새로 수신하는 설계 | **권고하는 후속 개발 한 묶음, 미구현.** 네트워크 중복을 줄이되 해제는 byte0부터 반복 |
| Media Range 확대 | 예:1MiB면96MiB 압축의 정상 요구1,152→288. Byte/해제량은 유지 | Body/header 한계는 남고 시간4분의1을 보장하지 않음. 별도preset/검증이 필요한 차순위 후보 |
| gzip 재개 index·해제상태 저장 | 올바른 state/window와 tar 위치 생성에 최초 순차 읽기 필요. 이후 해제 prefix 처리도 줄일 가능성 | 미구현. Range offset/header ordinal로 대체 불가. 첫 묶음에서 제외 |
| 대응 sequence의 작은 독립 묶음 | 최초 RGB/pose 도달 비용과 파생물 영구용량 필요. 이후 실험 준비비 절감 가능 | 대응 확보 후 후보. 기존 제공자의 묶음은 유용할 수 있으나 소재/호환 미확인. 전체52GB 추출을 전제하지 않음 |

**LocalSource는 cache 기능이 아니다.** 현재 `SplitStream`은 part0/offset0, `GzipReader`는 새 `zlib.decompressobj(31)`로 시작한다. `LocalSource`는 local file size/mtime을 검사할 뿐 Drive 수신물의 자동 보관·재검증·추가저장이나 같은 part 중간의 local/Drive 전환을 제공하지 않는다.

Local 읽기도64KiB마다 `local_reads`와 공유 `ranges`를 늘리고 `transfer_bytes`를 소비한다.768회면 최대48MiB여서 현재profile로64MiB cache 전체 재생도 보장되지 않는다. 새 설계는 네트워크 body·논리 압축입력·local IO·요청·해제·전체시간을 분리해 제한해야 한다. 단순히 local을 거쳐 예산을 우회하는 방식이 아니다.

현API6 ZIP에는63.925MiB 압축 본체가 없으므로 **지금 cache hit로 삼을 수 없다**. 보관돼 있었다면96MiB까지 새 압축 읽기는 약32.075MiB였겠지만, 실제로는 최초 cache 구축 읽기가 필요하다. 중간 gzip part나 마지막 Range만 독립 해제하는 방식이 아니다.

## 6. 권고 개발 한 묶음과 완료 기준

기존 물건의 보관·검증을 정리하면서 **상한 있는 prefix 영구재사용과 sequence/frame 관측 보존**을 개발한다. 오른쪽 성공ZIP 미수령은 이후 영상대조의 제한이며 독립적인 cache 개발 전체의 blocker로 삼지 않는다. 이번 문서는 구현·실행 지시가 아니며 후속 개발·검토 후 고정 버전을 테스트 세션에 인계한다.

| 완료 기준 | 검토할 근거 |
| --- | --- |
| 원본 identity와 bytes 연결 | 명시한 원본size/version/modifiedTime,연속범위·개별hash·생성code/manifest판을 기록. 원본 보존. 다음 사용 시 identity/hash 재검증 |
| 불완전·변경·실패 분리 | 요청 길이를 다 받지 못한 chunk,post 미완료,변경·gap·hash불일치는 재사용 제외. 원본/post/parser/CRC 실패 실행은 unusable,관측은 untrusted. 잘린 payload를 완전 파일로 올리지 않음 |
| Local/remote 비용 분리 | Cache replay와 후속수신의 모든실제요청·새body·local IO·논리입력·해제·전체시간 기록. Cache hit로 무제한 logical 입력을 허용하지 않음 |
| 관측 카탈로그에 상한 | 도달한 완전sequence명·frame·좌우/pose·payload완료·저장·trust를 보존. 건수/path/report disk cap과 잘림 기록. Header명은 내용검증/seek index가 아님 |
| 저장·보관·재사용 검증 | Cache 별도disk·영구보관 전송/용량/시간/중단조건,원본과결과의분리. 작은fixture에서 hit/miss/실패/원본변경/누락/출력hash/예산/post검사를 검증하고 보관본 재생 근거 제시 |

### 이후 한 번의 탐색에 대한 절충 예시

첫 원본 시험 후보로 Serena frame1·10을 유지하고 **body96MiB 정도를 cache/관측보존과 함께 한 번 탐색하는 안**을 권고한다.80MiB는 추가 진행이 약16MiB이고128MiB는 현재 평균으로 약465초·703MiB 해제가 필요하다.96은 중간 비용의 절충 예시이며 목표 위치에 근거한 최적값·최소 필요량이 아니다.80/96/128 어느 값에서도 성공 가능성은 미확정이다.

| 검토 후보 | 수치와 의미 |
| --- | --- |
| 새 network body / 논리 압축 prefix | 각각96MiB 상한 예시. Body는 metadata 포함이므로 압축96MiB 도달 보장 아님. Cache hit도 논리입력 상한 적용 |
| 준비시간 / 요구 | 협력적420초 / 최대1,155요청 예시.256KiB384범위의 정상1,152+추가3 계산이며 cache 검사 요청도 같은 단계 예산에 포함. 초기metadata는 별도 현행예산 유지 |
| Header / 해제 / tar metadata |16,000 /640MiB /누적2MiB 예시. 단일metadata64KiB 유지.96MiB 선형견적에 여유를 준 값이며 보장이 아님 |
| 파일·buffer | 목표2,각4MiB,IO64KiB,media256KiB 유지안. Range확대·새sequence·학습을 같은시험에 섞지 않음 |
| Cache·추출·ZIP 공간 | Cache별도128MiB 예시+기존출력16MiB+ZIP최대17MiB=임시공간161MiB 틀 검토. Python path집합/header16,000 overhead를 포함한 RAM hard상한은 아님 |
| 영구보관 | Cache128MiB 틀과 결과를 별도 보관. Drive등 복사는 추가전송/시간/용량이며 원본읽기body 밖이므로 별도 비용/중단조건 명시 |

수치는 승인된preset이 아니다. Cache가 기존16MiB 출력과 별도 공간을 요구함을 드러내고,쓰기/재생 시간·RAM·보고서 및 D2 비용을 검토해 고정한다. 원본이 같아 network를 절약해도 해제·header·논리입력·경과시간은 소비한다.

어느 상한/원본변경/protocol/parser/CRC 오류에서도 결과를 남기고 끝내며,목표가 완전히 저장·검증되면 조기 종료한다.0개라도 다음job·sequence·128MiB를 자동 시작하지 않는다. 고정revision/소스ZIP/notebook·설정·목적·예산·기대보고서를 검토 후 테스트 세션에 전달한다.

Pose를 얻으면 오른쪽 성공원본도 갖춰 로컬 sequence/frame·hash·GT2D/3D와 실제RGB 재투영을 검증한다. 얻지 못하면 보관 카탈로그/cache를 근거로 후속범위를 비교한다. 이후에도2frame 내용정합→연속clip·독립validation·원본motion분리→실제baseline 순서를 유지한다.
