# 지속형 로컬 archive 인덱스

후속 [실제 샘플 확대 discovery](SAMPLE_EXPANSION.md)는 별도 명시 profile과 신규 ArchVis RGB source로 batch별 snapshot/receipt를 누적한다. 아래 기본 한도·기존 pose notebook/DB는 바꾸지 않으며 metadata+Range body256MiB/600초 등의 확대 profile은 그 고정 job에만 적용한다.

2026-10-29 첫 구현. `rewind_archive_index`는 RGB·pose의 로컬 분할 gzip/tar를 대상으로 인덱스를 저장·복원·확장하는 선택 기능이다. 기존 API10, archive_prepare notebook, 원본과 캐시는 변경하지 않는다.

## 지원 범위와 순서

- `make_source`에 조각을 명시적 순서로 전달한다. 조각 순번·경로·크기·mtime·파일 identity와 총 길이를 고정한다. 기본 복원은 원본 전체를 hash 검사하지 않는다. 원본 metadata를 유지한 변경까지 검출한다고 보장하지 않으며, 기대 SHA가 있는 경우 `audit-source`로 별도 전체 검사를 한다.
- 하나의 archive마다 SQLite 파일 하나에 원본 명세·파일 목록·마지막 완전 파일 경계·gzip restart 인덱스를 함께 저장한다. RGB와 pose는 각각의 원본을 인덱싱하며 대응은 향후 sequence/frame 키로 확인해야 한다. 두 archive의 파일 순번을 서로 대응시키지 않는다.
- 목록은 tar 내부 물리 offset 순서로 누적한다. 요청 순서·frame 문자열 정렬·학습 shuffle로 재배열하지 않는다. 파일명 중복, 순번/경계 불연속, 원본 변경, 세대 충돌과 진행률 후퇴를 거부한다. `.gzidx` 바이너리끼리 결합하는 기능은 없다.
- checkpoint는 SQLite 트랜잭션과 generation 비교로 갱신한다. 갱신 실패 시 이전 저장본을 유지한다. 읽기 예산 일부를 checkpoint용으로 예약하며, 저장에 필요한 예산까지 부족하면 `checkpoint_deferred=true`로 이전 저장본을 유지한다. 아직 저장되지 않은 이번 작업은 재실행 비용이 생긴다.
- GNU longname/PAX 처리 중 또는 파일 payload 중간에서 중단되면 마지막 완전 항목 경계부터 다시 읽는다. 저장 인덱스를 사용한 추출은 자동 인덱스 생성을 끈다. gzip 인덱스 엔진 버전도 저장·대조한다.

## 사용

선택 의존성은 `requirements-index.txt`에 분리했다. 아래 경로는 사용자가 준비한 로컬 파일 예시다. `--parts`는 반드시 원본 순서로 전달하며 wildcard 정렬에 의존하지 않는다. GPU는 필요하지 않다.

```sh
python -m pip install -r requirements-index.txt
python -m rewind_archive_index create outputs/scene.sqlite --archive-id scene_rgb --kind rgb --parts data/scene.tar.gz000 data/scene.tar.gz001
python -m rewind_archive_index scan outputs/scene.sqlite --max-new-members 128 --defer-crc
python -m rewind_archive_index status outputs/scene.sqlite
# 다음 실행은 같은 DB를 읽어 이미 저장한 경계 뒤로 진행한다.
python -m rewind_archive_index scan outputs/scene.sqlite --max-new-members 128 --defer-crc
python -m rewind_archive_index lookup outputs/scene.sqlite scene/path/final_1.png
python -m rewind_archive_index extract outputs/scene.sqlite --members scene/path/final_1.png --output outputs/selected_rgb
# 목록 끝까지 확인한 후 전체 CRC 검사. 추가 전체 읽기 비용이 든다.
python -m rewind_archive_index verify outputs/scene.sqlite
```

Pose는 `--kind pose`로 별도 DB를 만들고 JSON의 정확한 member 경로로 조회·추출한다. Python API는 `ArchiveIndex`, `make_source`, `IndexLimits`를 제공한다. `scan(..., expected_generation=N)`으로 예상 세대를 지정할 수 있다. CLI의 `--limits limits.json`은 하위 명령 앞에 놓는다. 재시작/scan/extract CLI는 합성 fixture와 보유 실제 구간에서 확인했으며 설치 명령은 이번 작업에서 재실행하지 않았다.

새 추출 출력은 `persistent_index_extraction_v1`이다. `files/`에 선택 payload와 `extraction.json`을 남기며 size/hash를 확인한다. 기존 `rewind_archive verify`와 호환되는 출력은 아니며 학습용 release도 아니다. `train_ready=false`를 유지한다. 실패한 추출 디렉터리를 정상 결과로 재사용하지 않는다.

## 완료와 자원 한도

`boundary`는 tar의 마지막 완전 항목 끝이다. `index_coverage`는 저장된 내부 restart point의 최대 해제 offset으로, 데이터셋의 완료율이 아니다. Directory도 member 수에 포함된다.

`catalog_complete`는 연속 탐색으로 tar 종료의 두 zero block을 확인했다는 뜻이다. `gzip_crc_valid`는 별도 처음부터 끝까지의 bounded zlib 검사로 trailer/CRC와 tar 뒤의 zero padding을 확인했다는 뜻이다. 두 조건을 모두 만족해야 `complete=true`다. CRC 검사 예산이 부족하면 pending 상태로 남으며, 후속 verify는 처음부터 다시 읽는다. 잘린 prefix는 전체 원본 완료로 표시하지 않는다.

학습1epoch는 사용한 split·sampler·추출본의 범위일 수 있다. 모든 원본 바이트를 읽고 완료 검증을 끝낸 경우에만 해당 원본의 인덱싱 완료이며, epoch 횟수로 이를 대체하지 않는다.

기본 한도는 호출당 source 읽기256MiB·반환 해제량128MiB·파일8MiB·120초·관측 RSS512MiB, 누적20,000member·state16MiB·gzip 인덱스32MiB·DB 저장 예산128MiB다. 읽기 trace/인덱스 점 수에도 제한을 둔다. Python I/O 시점의 협력적 검사로 OS/native hard cap이나 내부 재해제 총량 상한을 보장하지 않는다. 목록을 전체 JSON snapshot으로 저장하므로 수TB 전체 장면의 대규모 운영 구현은 아니다.

## 검증과 다음 연결

- 신규13개 로컬 검사 통과: 실제 여러 파일로 나눈 합성 RGB/JSON, 새 프로세스 복원, 물리 순서, 저장 인덱스 추출, 파일/PAX 중간 예산, stale writer, SQLite UPDATE 도중 rollback, 원본/저장본 변경, tar/gzip 완료 판정. Root 독립 재검사13개도 통과했다.
- 기존 준비 경로34개 회귀 검사 통과. 모델·GPU·원격 테스트는 실행하지 않았다.
- 기존 실제19.68MiB 구간을 불변 로컬 진단 입력으로 등록해20→40member를 새 프로세스에서 누적했다. 선택 PNG hash·offset은 앞선 독립 조사와 일치하며 root가 실제 추출 payload hash를 재대조했다. 전체 Lab 원본 명세나 커지는 cache reader를 검증한 것이 아니며 complete/CRC는 false다.
- Windows DB 핸들 누수를 수정했고, Windows 생성은 기존 파일을 덮어쓰지 않는 rename, POSIX는 같은 filesystem hard link를 사용한다. 최종 검사는 일반 로컬 실행에서 통과했다. Python3.12.10과 이전 격리 indexed_gzip1.10.3·설치된 psutil을 사용했으며 기존 .venv는 바꾸지 않았다.

근거는 [개발 검사](../outputs/persistent_index_development_20261029/tests.json), [기존 회귀](../outputs/persistent_index_development_20261029/existing_core_regression.json), [실제 구간](../outputs/persistent_index_development_20261029/real_prefix/result.json), [독립 검사](../outputs/persistent_index_review_20261029/root_final_tests.txt)에 보존한다. 최종 방어 조건과 EOF 빈 줄 정리 후 root가13개 검사를 다시 통과했으며 최종 코드 hash를 함께 보존했다.

기존 local source DB에는 경로와 파일 identity가 있으므로 다른 런타임에 복사만 해서 바로 재사용할 수 없다. 후속 [Drive 고정 source·닫힌 snapshot 연결](DRIVE_PERSISTENT_INDEX.md)은 runtime 경로 없는 원본 metadata와 명시 receipt로 저장·복원하는 adapter를 추가했으며 모의 검사와2026-10-30 실제 pose A→B 복원·목록16→32개 누적을 확인했다. 2026-10-30 후속 C 시험에서는 실제 pose의 기존 gzip coverage 밖 확장·저장본 재추출도 확인했다([C 결과](PERSISTENT_INDEX_EXTENSION.md#2026-10-30-수정본-실제-c-시험-결과)). RGB 원격·대규모 처리량은 미검증이고 학습 hook·대규모 행 catalog도 미구현이다. Live DB는 로컬에서만 실행하며 Drive 동기화 원자성을 가정하지 않는다. 전체 데이터 인덱싱·실제 사용자 pilot은 root의 별도 고정 인계 후 진행한다.


2026-11-02 [자동 준비 연결](AUTOMATIC_INDEX_PREPARATION.md)을 추가했다. Core 직접 extract의 독립 출력은 유지하며, 기존 prepare/CLI의 opt-in 경로가 자동 복원·확장·저장 후 기존 extraction manifest로 변환해 verify/matching/candidate 변환에 연결한다. 실제 Drive 통합·학습 epoch hook·전체 규모 처리는 별도이며 자동으로 전체 인덱싱 완료를 선언하지 않는다.
