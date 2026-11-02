# 저장 g9의 RGB102페어 추출·pose 내용 감사

2026-11-03 개발 준비 완료 기록이다. 목표는 저장 RGB 인덱스의102 stereo key에 대한204 PNG와102 pose의 실제 payload·내용을 대조하는 것이다. Pose102 확보·수치 감사는 로컬에서 완료했고, RGB204의 실제 Drive/Colab 추출은 아직 실행하지 않았다. 정식 학습 release·전체 인덱싱 완료·사람의 시각 정합 승인은 포함하지 않는다.

## 확보한 입력과 근거

RGB job003 g9는270개 저장 항목에서 stereo102개/5sequence를 제공한다. 선택204 PNG 합계247611434bytes(236.1407MiB), 단일 최대1294107bytes다. Gzip restart spacing1MiB, engine `indexed_gzip==1.10.3`이다. Burpee21·GettingUp16·Football9·LegSweep24·Dribble32개이며 모두 공식 train에 속한다. 원래 archive 순서와 sequence/frame 정렬을 구분한다.

235MiB pose cache에서 목표102 JSON을 새 출력으로 로컬 순차 해제했다. Raw8725589bytes, 반환 해제1350871040bytes였고 신규 원격 요청은0이다. 기존67개+이전2개의 SHA와 크기가 일치했으며 나머지33개도 완전 payload로 확보했다. Cache prefix의 전체 SHA와 ZIP CRC를 검사했으며, 조기 목표 완료이므로 gzip 전체 EOF/CRC를 주장하지 않는다. Cache·g9 원본은 수정하지 않았다.

기존 `real_archvis62_v1` spec와 고정 calibration·공식 split을 재사용해102개 모두 converter 수치 감사를 통과했다. 원본87관절 정보, 선택62관절·mm 단위·정답 누락 mask를 보존했다. World/camera·inverse·camera/device 검사의 최대 raw 오차는 약1.71e-13 이하, 17748개 재투영의 최대 오차는0.000074537px다. 이는 annotation 내부 수치 정합성이다. RGB 영상 내용·timestamp 의미·원본 motion의 독립성은 별도 검증이며, 화면 밖 관절을 누락 mask로 바꾸지 않는다.

| 고정 입력 | SHA256 |
| --- | --- |
| RGB job003 결과 ZIP | `d3e35128ee38278716515f3b111d306c81e5f3cb1df57ae70f31a1366a79cee1` |
| g9 snapshot | `ec213878a03784b22fa7e6d53f26f16a61409aa8854228e177fd97f4e2101f8b` |
| Receipt 파일 | `3d4c6c11953318661b5cf905155b7e0384cb87430fe9f97192dff5df4f7351e9` |
| 원본 source fingerprint | `73b7007fc0586e72ddc0109290c79af93b7e6954c2f65a3e708bd6e8917b2c37` |
| Snapshot 내부 DB | `1fbe4c8558715b5f8a7719eb287f9cf630ee7db9768af871bedfccd088b91d6c` |
| Pose67 실제 결과 ZIP | `7ebe1cea2b88e7945a380068e0ac7d3fc26e34794437900f896eae0482865211` |
| 235MiB cache ZIP | `f902b25aadb1932c4c734795592e122061f8d30410bdeed5bb3866528ebff437` |

새 입력은 `outputs/rgb102_pair_prepare_20261102/developer/rgb102_inputs.zip`,12009120bytes/SHA `ae1119b956c102202ccfd3a59022db1b5d762dc675d104a1bd733f466aae4f90`다. Config canonical SHA는 `414279e6af3049dd88400e425162f898d940b4e00f714872f6225291e0f2b42c`다. Snapshot/receipt·raw pose102·candidate/감사·metadata·파일별 pin을 포함하므로 Colab에235MiB cache를 다시 업로드할 필요가 없다. 실제 입력 구성의 검사 근거는 같은 폴더 `preparation_report.json`과 `inputs/input_manifest.json`이다.

## 추출 계약과 예산

[pairs.py](../rewind_discovery/pairs.py)는 g9 고정 snapshot만 복원한 뒤 기존 `ArchiveIndex.extract`를 호출한다. 물리 순서의32개 파일씩7묶음(32×6+12)을 한 번에 실행한다. Core32개 제한과 `auto_build=False`를 유지하며 scan·gzip 인덱스 생성·DB checkpoint 변경은 금지한다. 기존 `import_snapshot`은 복원 DB의 신뢰 anchor를 기록하므로 snapshot 내부 DB SHA와 복원 직후 DB SHA를 구분한다. 추출 전후의 복원 DB 전체 SHA가 같아야 하며, 원본 snapshot은 계속 별도 보존한다.

| 전체 실행 범위 | 고정 상한 |
| --- | --- |
| 응답 body / HTTP / 시간 | 512MiB / 3072 / 1200초 |
| Source read / 반환 decoded | 1GiB / 256MiB |
| RSS | 1GiB, 기존 read/작업 guard의 협력 검사 |
| Workspace | 512MiB |
| 결과 ZIP | 384MiB, 별도 회수120초 |
| 실행 전 가용 RAM / 임시 disk | 각각2GiB 이상 |

최초 metadata body·HTTP·시간도 전체 한도에 차감한다. 모든 묶음이 하나의 DriveRuntime/transfer budget과 누적 read/decode/time 잔여를 공유한다. 1MiB Range·4MiB RAM cache를 사용하며 묶음 경계에서 예산을 초기화하지 않는다. 숨은 auth/media 재시도는0이고 실행 실패 후 자동 후속 실행은 없다. 작은 선택 payload라도 restart 재해제·Range 정렬·묶음 간 반복 읽기로 전송이 더 클 수 있다. 512MiB 내102개 성공을 보장하지 않는다.

업로드·source/input 해제·필수 index 의존성 설치·인증은 위 실행 시간 밖의 준비다. 준비 ZIP/source/입력·복원 DB와 결과 ZIP을 포함하는 임시 disk가 별도로 필요하다. 최종 회수는 cache/원본 snapshot을 중복 포함하지 않고 payload·metadata·pose·감사·현재 partial을 넣는다. Runtime watchdog이나 전체 driver VRAM cap은 구현하지 않았다.

## 내용 감사와 부분 결과

추출 report의 SHA와 실제 PNG SHA·크기를 재대조하고 한 장씩 PNG RGB/native1024×1024 decode를 검사한다. Sequence/frame/view가 유일하게 대응해야 하며 pose62관절 좌표·정답 누락 mask를 연결한다. 각 sequence의 view별 첫 확보 이미지에 대표 GT2D overlay를 생성한다. 좌표는 원본1024 기준이며 사람의 영상·관절 정합 검토 자료다. 자동 감사 성공을 사람이 시각적으로 확인한 정합성으로 표시하지 않는다.

`complete_pairs`는 양쪽 PNG의 hash/decode/content 감사가 끝난 key만 센다. 추출 완료·PNG 감사 진행·누적 예산 중단·identity 변경·hash 실패·decode 실패·cleanup 실패를 보고서에 구분한다. 현재 실패 묶음의 잘린 파일이나 아직 감사하지 않은 파일도 회수하지만 완료 pair로 세지 않는다. Session close나 사후 DB SHA 읽기가 실패해도 안전한 단계/예외 종류와 최종 보고서 저장을 시도한다. 예외의 원문·자격증명은 기록하지 않는다.

`train_ready=false`, `human_visual_alignment_confirmed=false`, timestamp/motion 미검증·validation0를 유지한다. 학습을 위해서는 독립 validation·motion/시간/입력 계약과 별도 release 검토가 필요하다.

## 실행본·회수와 로컬 검증

[rgb102_pairs.ipynb](../notebooks/rgb102_pairs.ipynb)는6셀/5code cell이다. Root가 검토·로컬 커밋한 source ZIP SHA와 revision을 채워 ready본을 고정하고 데이터셋 탐색 세션에 인계한다. 새 Colab CPU 런타임에서 source/input 두 ZIP을 준비하며, 두 ZIP 전체 검증 후 staging한다. 정확히 같은 기존 staging만 읽기 재사용한다. 다른 source의 이전 모듈이 있으면 인증 전에 차단한다.

한 번의 실행 셀이102개 전체 목표를 수행한다. 실패·중단 후에는 독립 회수 셀만 실행한다. 회수는 원본 요청·인증·추출을 반복하지 않고 현재 partial/완료chunk·raw pose·감사·보고서를 ZIP으로 묶어 SHA를 출력한다. Source/input 준비 자체가 실패한 경우에도 작은 준비 실패 ZIP을 받을 수 있다. 탐색 세션은 결과 ZIP/SHA·완료/미완료 key·종료 이유·비용·대표 overlay를 root에 전달하고 원본·결과를 런타임 밖에 영구 보관한다.

합성102페어의7묶음 전체 완료, 무scan/원본 index 불변, decode·HTTP 누적 cap, identity/hash/PNG 실패, 중단·cleanup 실패 및 원격 없는 partial ZIP 회수를 검사했다. 실제 notebook 준비·누락·재사용·구모듈 차단·준비 실패 회수와 검토된 RGB 인증 factory의 AST 동등성·실제 requests/google-auth 타입도 검사했다. 개발16개/168.828초·root 독립16개가 모두 통과했고 skip0이다. 최종 로그는 `outputs/rgb102_pair_prepare_20261102/developer/final_tests.txt`, root 실제 입력·g9 독립 대조는 같은 상위 폴더 `root/input_review.json`이다. 실제 RGB Drive 추출·Colab 시간/RAM/disk·사람의 overlay 검토는 미실행이다.

## 2026-11-03 실제 Colab 결과와 독립 검토

고정14d94d42ab85487de03cd501c723d373a2fd7080/job rgb102_pairs_001은 completed, complete_pairs=target_pairs=102, files204, 7chunks content_audited로 성공했다. 선택 RGB payload247611434bytes를 확보했다. 실제 비용은 응답body266392949bytes/837HTTP/271.006914초, source read309657612bytes, 반환decoded247611434bytes, 관측RSS212058112bytes다. 고정 전송·요청·시간·read/decode/RSS 상한 이내였다. 이 한 번의 고정102개 실측이며 다른 archive·장거리 seek·전체 데이터 비용으로 일반화하지 않는다.

결과는 `outputs/test_rgb102_pairs_001_20261103/rgb102_pairs_001_result.zip`,274628517bytes/SHA `ec11b7eb5c4a69f76acb2e5f5045d8c7c560096a6dd33b2249e5d3a6a3e1dcae`다. Root는 ZIP 전체CRC,204PNG 각각의 g9 catalog SHA/크기·native1024 RGB decode, 회수된 pose/candidate/audit/config 등 모든 비snapshot 입력의 고정본 byte일치,7chunk 상태와read/decode 합계를 독립 확인했다. `root_review/verification.json`에 근거를 남겼다. 실행 보고서의 복원 전후 DB SHA는 동일하고 index_unchanged=true/scan_performed=false다. 회수 ZIP에는 DB를 넣지 않으므로 이 사후DB 불변은 실행 보고서 근거이며 root가 회수DB를 직접 검사한 것은 아니다.

대표 overlay10장(5sequence×2view)을 root가 시각 점검했다. 보이는 손·팔다리는 대체로 GT와 대응하며 뚜렷한 전체 좌표 밀림은 발견하지 못했다. 가림·영상 밖 GT는 유지한다. 각 view의 첫 확보 frame이 달라 Football 외의 좌우 대표는 같은 frame 비교가 아니며, timestamp/stereo 시간 검증으로 간주하지 않는다. 이는 AI의 제한 시각 점검이고204장 전수 시각 승인·사람 승인·모델 정확도 검증은 아니다. 원본 report의 human_visual_alignment_confirmed=false를 변경하지 않았다.

현재102개는 모두train/validation0이며 원motion·timestamp 미검증과train_ready=false를 유지한다. 다음은 보존 payload를 학습 준비 bundle로 연결하고 독립 validation·release 근거를 확보하는 단계다. 추가 원격 실행·학습·전체 인덱싱은 이번 검토에서 수행하지 않았다. Drive 영구 보관 여부는 이 로컬 검토로 확인하지 않았다.
