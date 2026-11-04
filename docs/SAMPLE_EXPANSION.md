# 실제 샘플 확대와 발견 인덱싱 보존 기록

## 현재 샘플 확대 방향 — 2026-11-05 갱신

2026-11-04 사용자 결정으로 발견·자동 인덱싱과 새 원격 실행은 중단했다. 사용자가 원본을 압축 해제해 Drive에 업로드하는 RGB·pose 추출본을 직접 사용하며 Depth는 초기 필수가 아니다. 업로드 완료·정확 경로·전체 확보량은 아직 확인하지 않았다. [현재 데이터 확보 기준](DATA_ACQUISITION_STRATEGY.md)을 따른다.

후속 추출에서 기존102+269의371 train 페어를 확보했으나 validation은0이다. 추가 로컬 Joyful57은 중복·원본 motion·split 검토 전이므로 총428로 계산하지 않는다. 1,000~10,000페어와 독립 validation을 목표로 파일 목록·SHA/byte·decode·pose 수치/좌표/단위·실제 overlay·sequence/frame·split/motion을 검토한다. 관측 catalog 건수는 검증된 페어 수가 아니다.

새 범용 추출본 manifest/학습 adapter·release는 미완료다. 검토 후2D 학습 개선→예측2D 기반3D baseline→M2/M3→영상 연결로 진행한다. 아래 discovery job의 예산·원본·고정 pin·receipt·snapshot은 보존하며 job003이나 실패 job을 다시 실행하지 않는다. 과거 outputs 경로의 이동·복원은 [보관 안내](LOCAL_STORAGE.md)를 따른다.

## 이전 발견 인덱싱 결과와 계획

아래 ‘현재·다음·미실행’은2026-11-02 당시 상태다. 새 실행 지시나 현재 확보량으로 해석하지 않는다. 당시 역할 분담도 새 인덱싱을 허용하지 않으며 후속 작업은 root의 별도 고정 인계를 따른다.

## 2026-11-02 job003 실제 재개 결과

Root가 결과 ZIP SHA d3e35128ee38278716515f3b111d306c81e5f3cb1df57ae70f31a1366a79cee1, 외부·내부CRC와 최종 snapshot/DB SHA 및 receipt 일치를 독립 확인했다. 근거는 outputs/test_archvis_rgb_discovery_677bc7f_003_20261102/archvis_rgb_discovery_677bc7f_003_result.zip이다. g6의 parent는 고정 g5와 일치하며 160항목/g5에서270항목/g9로 재개·확장했다. 최종 snapshot은ec213878a03784b22fa7e6d53f26f16a61409aa8854228e177fd97f4e2101f8b, receipt raw SHA는3d4c6c11953318661b5cf905155b7e0384cb87430fe9f97192dff5df4f7351e9다.

bounded_stop/transfer_bytes_limit으로 종료했다. API body267439414bytes·815요청·228.930초이며 disk 실측값은 없다. train stereo59→102(+43), stereo∩pose35→35(+0), validation0·실제추출0이다. RGB 재개는 성공했지만 추가100개의 RGB+pose 후보 목표는 진척하지 않았다. 앞선 약729MiB/637초는 첫 구간의 단순 비례 참조치이며 현재 구간에는 적용할 수 없다.

당시 후속 판단은 RGB 반복 실행보다 새 stereo 키와 기존 pose 목록의 누락 범위를 먼저 대조하는 것이다. pose 목록에 없다는 사실을 원본 pose 부재로 단정하지 않는다. 고정 g9와 기존 결과를 보존하며 job003 자동반복/예산확대는 하지 않는다. 관측35개에서 소량 추출을 검증하는 경로도 유지한다.

## job003 재개 준비 — 당시 추가100 관측 후보 목표, 위 실제 결과로 대체

현재 train stereo∩pose 관측 후보35개에서 추가100개(누적135개)를 목표로 두되, 기존 상한의 job003 한 번만 준비한다. 실제 추출0·validation0이며 관측 대응을 검증된 학습 pair로 취급하지 않는다. 달성을 보장하거나 자동 반복하지 않는다.

job002의 정상 g5 snapshot SHA9ae9c6581f0b382ca46512fd07a85bc2cae21a0a6e09285cef9f4690bf4e1a2f에서 이어 간다. receipt 파일 SHA는0f6e0f41f7a2573a8c2bb4ac13a93c7f4a1c7d9541f09c1b47acf51a60879300, source SHA는73b7007fc0586e72ddc0109290c79af93b7e6954c2f65a3e708bd6e8917b2c37이다. 외부/내부 ZIP CRC·SHA와 receipt·내부DB SHA를 읽기 대조했으며 DB import/scan·원격 요청은 하지 않았다.

실행 source677bc7f와 현재 runner/index 코드가 같아 기존 source ZIP을 유지하고, auth 보완 notebook1d776ff를 사용한다. job ID는 archvis_rgb_discovery_677bc7f_003이며 source/input/workspace/recovery와 새 snapshot 폴더를003으로 분리한다. INPUT_SNAPSHOT_DIRECTORY는002의 원 Drive 폴더, RESUME는 그 폴더의 정확한 g5 receipt 경로·위 파일SHA·anchor 전체다. 정확 설정과 원receipt는 outputs/discovery_job003_prep_20261102/developer/job003_settings.json 및 trusted_g5_receipt.json에 있다. source/input/selection pin·256MiB/2048HTTP/600초·read512MiB/decode256MiB/RSS1GiB/논리출력256MiB·32member×16batch 상한은 유지한다. 실제 재개에서는 원본 metadata/version과 anchor identity를 다시 대조하며 소비된job002를 재실행하지 않는다.

root가 고정한 ready는 outputs/discovery_job003_prep_20261102/handoff/archvis_rgb_discovery_677bc7f_003_ready.ipynb,16,086bytes·SHA75c1a4abdf80358aed61c00f608c2f8b0c499a1f8df1658817a34041dc583afb다. 아직 실행하지 않았고 root의 검토·탐색 세션 인계가 다음 단계다.

비용은 job002의 API body255.051MiB·phase222.865초/관측35개를 단순 비례하면 추가100개에 약728.717MiB·636.756초다. 이는 구간별 파일 순서/크기/pose 관측 coverage와 재개 비용에 따라 달라지는 참조치이며 요금 추정도 아니다. 준비·snapshot 동기화/회수 전송은 별도다. 단일256MiB job003에서100개 증가를 보장할 수 없고 이 계산으로 여러 job을 자동 시작하지 않는다.

현재는600초보다 전송 cap에서 먼저 멈췄다. [ArchiveIndex.scan/_commit](../rewind_archive_index/core.py)은 저장 boundary의 restart 상태를 복원하고 generation·기존 catalog prefix 불변을 검사한다. 동일 gzip의 미인덱싱 뒤쪽은 검증된 해제 상태·tar 경계가 없어 임의 offset으로 독립 scan할 수 없으며, 여러 worker가 같은 저장점에서 시작하면 재해제/전송이 중복된다. gzip 조각000/001을 독립 스트림처럼 해제하거나 별도 suffix 인덱스를 합치는 기능도 없다. 별개 archive별 DB나 알려진 restart 범위의 독립 읽기, read-prefetch는 원리상 병렬화할 수 있지만 [DriveRuntime.block](../rewind_archive_index/drive.py)의 현재 cache/누적 장부는 thread 동기화가 없다. 이번에는 병렬화 코드를 추가하지 않으며 같은 body 상한에서 병렬화가 더 많은 후보를 보장하지 않는다.

## 2026-11-02 job002 실제 부분 인덱싱 결과

수정 notebook1d776ff/실행 source677bc7f의 Colab job002 결과를 root가 독립 검토했다. 외부 ZIP SHA `251c8addd9e4d05ad0e5bcd09f5d65a2ac43d1cff39dc6ec015ff9c1c1a600a4`, 외부·내부 ZIP CRC와 receipt/report 일치, snapshot SHA `9ae9c6581f0b382ca46512fd07a85bc2cae21a0a6e09285cef9f4690bf4e1a2f`, SQLite SHA `3092b18f3f8cbaeec448ab797570b43ccfcf4a3bf8f46399ede97263ea454902`를 확인했다. 근거는 outputs/test_archvis_rgb_discovery_677bc7f_002_20261102/archvis_rgb_discovery_677bc7f_002_result.zip이다.

status=bounded_stop, 6번째 batch에서 transfer_bytes_limit으로 중단하고 generation5/160항목의 마지막 정상 snapshot을 회수했다. tar boundary167135232·gzip index coverage167598184는 서로 다른 지표이며 압축 원본52441303081bytes의 완료 비율로 단순 나누지 않는다. train left59/right78/stereo59, stereo와 기존 pose 관측 교집합35이며 validation/test/unknown은0이다. 실제 추출triplet0·PNG decode 미검증·train_ready=false이고 전체 archive CRC/완료를 뜻하지 않는다.

보고된 API body267440332bytes(약255.05MiB)/825requests, source read432874462bytes, decoded200673280bytes, elapsed222.865초, sampled RSS182218752bytes다. disk 실측값은 보고서에 없어 전체 디스크 상한 준수를 독립 확정하지 않는다. recovery의 archive_network_requested=false는 회수 작업의 무네트워크 의미이며 실제 인덱싱 전송이 없었다는 뜻이 아니다. 이번 결과는 부분 인덱싱·snapshot 회수를 검증했으며 새 런타임 복원·확장 및 실제 후보 추출은 별도 확인한다.

당시 다음 우선 작업은 저장한 generation5와 원본identity를 고정해 관측35개 중 소량의 좌우RGB+pose를 추출·파싱·대조하는 것이다. validation 후보 탐색과 원본motion/split 검증도 계속 필요하며 train 후보만으로 정식 baseline을 승인하지 않는다. 소비된 job002를 재실행하거나 예산을 자동 확대하지 않는다.

## 2026-11-02 session retry 설정 실패 보완

job archvis_rgb_discovery_677bc7f_001의 회수 ZIP SHA b3191fbcd96ce756014683093a30dcfa938db5040d0e10d0866e188ade3c246c에서 CRC와 보고서를 대조했다. 초기 metadata5건/1,483bytes 이후 scan 진입의 session_retry_policy로 중단됐고 read/decode0·정상 receipt 없음이다. 외부 AuthorizedSession adapter만 retry0으로 설정하고 기본 내부 인증 session의 retry를 남긴 것이 원인이다.

notebook factory만 requests.Session의 인증 transport를 retry0으로 고정해 Request(session=auth_http)로 주입하고, 외부 session도 retry0/max_refresh_attempts0으로 설정하도록 보완했다. 초기 metadata 전 validate_no_hidden_retries를 적용하며 정상/실패 시 외부·인증 session을 정리한다. 보존 transport_deps의 실제 requests2.34.2/google-auth2.62.0/urllib3 2.8.0으로 구형 factory 거부·수정 factory 통과·close 수명4개와 현재 업로드7개, 총11개 검사를 네트워크0으로 통과했다. root 독립11개도1.219초 failure/error/skip0으로 통과했다. 새 의존성 설치와 실제 Google 호출은 없으며 실제 Colab 연결 성공을 입증한 것은 아니다.

runner·계약·예산 변경이 없어 실행 source677bc7f와 고정 ZIP은 유지할 수 있다. job001은 이미 소비됐으므로 재실행하지 않는다. root가 수정 notebook revision을 별도로 기록하고 job002의 새 경로와 동일 예산을 고정해 인계한다. 기존 결과/고정 자료는 보존한다. 개발 근거는 outputs/discovery_retry_fix_20261102/developer/final_tests.txt와 summary.json이다.

## 2026-11-02 준비 업로드 실패 보완

고정677bc7f 실행에서 source ZIP만 선택해 source 추출 후 input SHA 검사에 실패했다. auth/원본 읽기/scan 전에 중단됐으므로 인덱싱 실행 결과로 취급하지 않는다. source/input ZIP을 한 개씩 받되 양쪽 SHA·안전 경로·CRC·크기와 기존 추출본의 정확한 파일/폴더 집합·byte를 모두 검사한 뒤에만 추출하도록 template을 보완했다. 동일한 기존 추출본만 읽기 재사용하며 불완전/변조·추가 파일은 거부한다. 삭제·덮어쓰기 안내는 하지 않는다. WORKSPACE가 있으면 소비된 job이므로 준비 재실행은 계속 금지한다.

고정 SELECTION_SHA256을 설정 셀에 두어 input config가 없어도 준비 실패를 회수한다. WORKSPACE가 없는 준비 실패는 별도 *_preparation_result.zip에 stdlib로 상태만 보존한다. 이후 정상 job의 결과 ZIP과 충돌하지 않으며, WORKSPACE가 있는 기존 job은677bc7f의 recover API와 호환한다. 이번 사용자의 실행 source ZIP/revision/hash·input ZIP·원job/경로·예산은 유지하고, 수정 notebook revision만 별도 고정한다. 기존 배포물은 보존하며 root가 검토 후 새 버전의 준비 notebook만 인계한다. 실제 notebook 셀의 로컬 mock 검사7개가 통과했고 Google·설치·scan은 실행하지 않았다. 근거는 outputs/discovery_upload_fix_20261102/developer/에 보존한다.

2026-11-02 · 실제 자료의 저장 목록을 늘리는 작업이다. 인덱싱 증가·관측 key 대응·완전 추출 triplet을 구분하며 학습·release·전체 archive 완료를 선언하지 않는다. 이번 개발에서는 원격 archive 전송0이며, 실제 Colab 실행은 고정본 인계 뒤에 진행한다.

## 보존 자료 조사

공식 train/validation/test 목록은 real_archvis62_v1의 pinned SHA와 일치한다. 다음 숫자는 sequence/frame 키를 중복 제거해 집계했다. 관측 catalog의 payload_complete는 읽기 관측이며 저장된 payload나 seek index가 아니다.

| 자료 | train / validation key | 저장·재개 위치와 한계 |
| --- | --- | --- |
| Lab RGB 불변19.68MiB prefix | left4/0, right27/0, stereo4/0 | 40member/gen2, tar boundary20,683,264, gzip coverage20,125,645. 다음 PNG 불완전 |
| ArchVis pose warm192 관측 | 11,377 / 1,264 | seek_index=false, 대부분 미추출 |
| ArchVis pose tail234 관측 | 14,067 / 1,331 | 총15,398key, validation20sequence, seek_index=false |
| 보관 pose B/과거 사용자 C 미완료 DB | 22 / 0 | 32member/gen2, boundary1,927,680, coverage3,791,957 |
| 보관 offline C DB | 70 / 0 | 80member/gen5, boundary6,109,696, coverage6,324,814. 과거 cached 공식자료 진단이며 현재 Drive byte 동일성 증명 아님 |
| 실제 Serena 추출본 | 완전 triplet2 / 0 | frame1·10, 공식 train·비연속·학습 차단 유지 |

Lab의 stereo4key는 Claudia/Pull_Heavy_Object frame2·5·12·15이며 모두 train이다. 보존된 대규모 pose 관측은 ArchVis 장면이어서 Lab RGB와의 교집합은0이다. 알려진 로컬 관측/추출 key 합집합에서도 완전 triplet은 Serena train2/validation0뿐이다. 이 합집합은 source DB나 gzip 바이너리를 결합했다는 뜻이 아니다.

과거 개발 DB는 .local-archive/20261102/INDEX.json의 정확한 member/크기/SHA로 새 출력에만 복원했다. RGB DB의 원래 로컬 source는 dev/ino/mtime/size가 일치하며20,636,340bytes·SHA46bff8311c9e211c5dcda9d494b190849ce8d9e2aa0cd9643a1fabe16b168dcb다. 공식 서버 prefix를 현재 사용자 Drive 복사본의 인덱스로 재사용하지 않는다. 보존 Drive 명세/현재 metadata가 맞아도 archive byte 동일성은 별도이며 최신 Drive g5 snapshot은 root metadata 조회만으로 내부 member 수를 추정하지 않는다.

## 로컬 실제 scan1회 결과

복원 RGB DB의 또 다른 복사본에서 한 번만 scan했다. read64MiB/decode16MiB·16신규member·120초·RSS512MiB·DBdisk128MiB로 기존 core 상한 이하를 적용했다. 결과는 incomplete_tar, members40→40/PNG31동일/새후보0이다. generation2→3 저장 후 새 프로세스에서 상태가 같았지만 세대 증가는 샘플 증가가 아니다.

실제 scan0.109초, source583,160bytes·반환 해제27,408bytes·관측RSS31,711,232bytes, scan+재개0.359초였다. 원prefix 길이/SHA와 원DB SHA는 불변이며 원격0이다. 추가로 한 전후 원prefix hash 읽기 비용은 scan 계측과 별도다. 근거는 outputs/sample_expansion_20261102/developer/local_scan/report.json이다. 같은 잘린 prefix에서 반복 scan하지 않는다.

## 당시 다음 실행 계획: ArchVis RGB 신규 source — 현재 종료

이미 pose 관측이 있는 ArchVis RGB의5조각을 명시 순서로 고정해 **새 source DB**를 만든다. 오래된 Lab prefix나 pose DB를 재라벨하지 않는다. Drive file ID/name/index/size/modifiedTime은 root metadata 조사와 대조하며, 도구가 반환하지 않은 version은 추측하지 않는다. authorized metadata API가 얻은 version을 manifest에 고정한 뒤 runtime이 다시 검사한다. 원본 조각 전체를 다운로드하거나 결합하지 않는다.

최대1 job의 명시 profile은 아래와 같다. 기존 라이브러리 default와 pose notebook은 변경하지 않았다.

| 항목 | 상한 |
| --- | --- |
| metadata+archive Range HTTP 응답 body | 256MiB 누적 |
| HTTP 요청 | 2,048 누적, 숨은 retry 금지 |
| source 읽기 / 반환 해제 | 512MiB / 256MiB 누적 |
| phase 시간 / 저장 여유 | 600초 / scan 시작 전30초 예약 |
| sampled RSS | 1GiB, 호출 프로세스 전체 |
| local 작업+snapshot+회수 논리 파일 byte | 256MiB |
| batch / job / 누적 목록 | 최대32member ×16batch /512신규member 이하 /20,000member 이하 |
| Range / cache / spacing | 1MiB /4MiB /1MiB |

초기 metadata 요청/응답도 합산하고 동일 DriveRuntime의 전송 장부를 batch마다 재생성하지 않는다. read/decode/time/disk도 job 전체 장부로 남은 값을 전달한다. phase에는 입력 감사·metadata·scan·snapshot이 포함된다. 업로드/설치/auth/mount·source/input 해제는 phase 전 준비이며 그 시간/공간은 별도다. HTTP 숫자는 Python Drive API metadata/Range 응답이며 Drive mount의 snapshot 업로드·백그라운드 동기화 트래픽은 계측하지 않는다. snapshot 논리 파일은 disk 상한에 포함한다.

상한은 확보량 목표가 아니다. PNG의 압축률/크기·HTTP 지연·목록 순서·체크포인트 예약 때문에 더 일찍 멈출 수 있다. source/index/전송/메모리 validator는 이 profile의 생성 검사를 통과했지만 실제 Colab 자원과 처리량은 아직 미검증이다. 시간/RSS/디스크 검사는 협조적이며 native 순간 peak나 timeout을 hard cap으로 보장하지 않는다.

## 도구와 결과

새 rewind_discovery.runner.run은 기존 ArchiveIndex/DriveRuntime/snapshot을 재사용한다. 한 job은 새 workspace와 새 snapshot 폴더만 허용하며 같은 경로 재실행은 인증/media 전에 거부한다. 다음 작업은 root가 새 job ID/경로/예산과 정확한 receipt 파일 SHA+anchor를 명시해야 한다. HEAD 자동 추종·기존 저장소 수정·budget 자동 리셋 반복은 없다.

매 batch는 완전 파일 경계·generation/source identity와 기존 catalog prefix 불변을 확인한 뒤 정상 snapshot/receipt를 게시한다. 다음 job은 해당 정확한 snapshot만 복원한다. import에도 남은 시간/디스크 한도를 전달한다. 실패/KeyboardInterrupt에서는 마지막 정상 receipt를 보고서에 보존한다. 부분 작업 DB보다 마지막 정상 receipt가 재개의 기준이다.

report에는 train/validation/test/unknown별 left/right RGB key, stereo key, 각각 pose 관측 교집합과 예시를 남긴다. 실제 추출은 이번 범위에 없으며 actual_extracted_triplets=0·PNG decode 미검증·train_ready=false다. unknown split은 별도 집계한다. 검색 목표는 특정 validation 파일을 끝까지 추적하는 것이 아니라 작은 연속 목록 확장과 저장이다.

recovery.recover는 인증·source read·scan 없이 report/metadata와 마지막 정상 snapshot/receipt의 byte SHA를 확인해 ZIP으로 회수한다. 보고서가 없으면 report_missing_execution_state_unknown으로 회수하고 성공/미실행을 추정하지 않는다. 이전 job의 정상 receipt를 복원한 직후 실패했다면 고정 input_snapshot_directory를 함께 지정한다. 회수 ZIP과 flush가 성공해도 Drive 영구 내구성을 보장하지 않는다.

## 고정 입력과 notebook

[CPU notebook template](../notebooks/archvis_rgb_discovery.ipynb)은 설정·source/input 업로드 검증/설치·인증 후 단1회 실행·독립 회수의4개 code cell이다. 빈 SOURCE/INPUT SHA·40자리revision·snapshot 폴더로는 실행되지 않는다. 실제 mount/ MyDrive 존재와 새 snapshot 폴더가 그 하위인지 확인한 뒤 media를 시작한다. 회수 셀은 run/scan을 재호출하지 않으며 이미 회수된 ZIP은 별도 SHA/CRC/job pin으로 재확인한다.

고정 input ZIP은 outputs/sample_expansion_20261102/developer/archvis_rgb_discovery_inputs.zip,362,409bytes·SHA d42ba7a1a831d6f109306991c8c6b75261a1e02eb09a328cd9698729cf37ba37이다. 해제5,109,061bytes이며 config.json·split3종·pose_catalog.json만 포함한다. 개별 SHA는 같은 폴더 input_pins.json, pose catalog 파일 SHA는4250fffbb672117c0524466c1df865c854b8c79addfca60ae929d163191458b9다. 토큰·자격증명·RGB/pose 원본 body는 포함하지 않는다.

source ZIP과 ready notebook은 아직 만들지 않았다. root가 검토/로컬커밋 후 source/input SHA·revision·새 경로·job ID·명시 예산을 고정한 본으로 인계한다. 이 문서는 지금 사용자에게 Colab 실행을 요구하지 않는다. 기존 합성 GPU 시험이나 학습을 반복하지 않는다.

## 운영 인계와 역할

2026-11-05 현재 인덱싱 실행 역할은 중단 상태다. 아래는 당시 고정 job의 책임·회수 규칙이며 새 실행을 승인하지 않는다. 추출본 목록·검증 또는 모델 시험은 root가 새 범위와 고정본을 전달한 경우에 진행한다.

동일 프로젝트 데이터셋 탐색 세션 01a12dd6-3940-7b91-91dd-eae1bfaba270가 root의 정확한 고정본을 받아 목록 검색·key/split 대조·정해진 budget 안의 인덱싱 실행과 결과 취합을 맡는다. 기존 테스트 세션01a14391-c1b9-7a40-b537-af6b0b43709c는 모델/GPU 등 검증 역할을 유지한다. 새 세션은 만들지 않는다.

탐색 세션은 코드·notebook·원본 pin·예산·anchor를 임의 변경하지 않는다. 실패하면 정상/부분 결과·마지막 receipt·회수 ZIP SHA를 보존해 root에 전달한다. root가 수정/다음 작업을 결정하고 디벨로퍼01a13299-7dcb-7080-8e8f-6af5ce67303d가 필요한 코드 보완을 맡는다. 각 인계는 프로젝트 경로 C:/Users/ktsvc/Projecets/REWIND-like prototype와 세션 ID를 함께 확인한다.

## 로컬 검증

mock Drive4개 검사는2batch→정상 snapshot→새job2batch 재개·누적 전송 장부·split/pose 교집합, interrupt/보고서 없는 회수, metadata/split/anchor 변경 거부, 낮춘 전송 cap을 확인했다. 최종4개2.334초 failure/error/skip0, root 독립4개도 통과했다. notebook code cell Python AST와 빈 pin/mount gate는 로컬 검사 대상이며 실제 Colab 실행 결과가 아니다. 근거는 outputs/sample_expansion_20261102/developer/final_tests.txt와 root/discovery_tests.txt에 있다.
