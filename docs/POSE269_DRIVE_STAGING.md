# Pose269의 Drive cache 준비 경로

2026-11-03 · 기존65a7244 notebook의 약247MB cache `files.upload` 중40% 부근에서 연결 중·위젯 비활성·반복 연결 끊김이 보고됐다. Traceback이나 OOM 근거가 없어 원인을 확정하지 않는다. 이번 변경은 큰 cache의 브라우저 업로드를 피하는 별도 준비 경로이며 실제 연결 끊김/OOM 해결을 입증한 것은 아니다.

기존 [pose269 notebook](../notebooks/pose_missing269.ipynb)과 입력·자료는 보존한다. 새 [pose_missing269_drive.ipynb](../notebooks/pose_missing269_drive.ipynb)는 source/input의 작은 두 ZIP 업로드만 유지하고 cache는 명시한 Drive 파일에서 local staging한다. **실제 Drive 경로는 아직 확인되지 않았으며 `CACHE_DRIVE_PATH`는 빈값이다. 확인된 파일 경로와 새 source/ready pin을 root가 고정하기 전에는 실행하지 않는다.** 자동 검색·추측·다른 cache·cold fallback은 없다.

## 유지하는 고정 값

| 항목 | 값 |
| --- | --- |
| 목표 / profile | 기존 정확269개 / `posePrefixTargets269_v1` |
| Job / workspace | `pose_missing269_001` / `/content/pose_missing269_001-work` |
| Cache ZIP 크기 / SHA256 | 246522402bytes / `f902b25aadb1932c4c734795592e122061f8d30410bdeed5bb3866528ebff437` |
| Prefix 크기 | 246415360bytes = 235MiB |
| Cache manifest SHA256 | `9177444e8283f57b97f037f3ceeafffdaead90b210694a8b23b3f3ebf8e06f0f` |
| 기존 작은 input ZIP SHA256 | `ef22cd2953a70a9c718b525402150cbc44a5a9ae514d096ae5de3fcb17789700` |

목표 목록·source archive identity·metadata retry 정책과 dataset prepare body192MiB/1916HTTP/1200초·일반출력32MiB·결과ZIP33MiB·cache export 상한은 그대로다. 정확한 나머지 pin과 partial/EOF 계약은 [pose269 준비 문서](G25_STORAGE_AND_POSE269.md)를 따른다. 새 notebook에서도 소비된 workspace가 있으면 준비/실행을 거부한다. 이전 runtime의 실행 여부를 추측해 guard를 우회하지 않으며 새 CPU runtime을 사용한다.

## 순서와 모듈 경계

1. Root가 고정한 작은 source/input ZIP 두 개를 업로드하고 SHA·경로·목표를 검사한 뒤 staging한다. 큰 cache ZIP을 이 위젯으로 선택하지 않는다. 정확히 같은 source/input만 재사용하며 불일치는 덮어쓰지 않는다.
2. 확인된 절대 `CACHE_DRIVE_PATH`가 있어야 다음 셀로 간다. 이전 source의 모듈이 로드돼 있으면 차단한다.
3. **인증과 Drive mount를 cache 복사 전에 수행한다.** 기존 notebook의 전체 workflow preflight보다 이 인증이 먼저 일어나는 순서 변경이다. Dataset 원본 archive 요청은 아직 하지 않는다.
4. [cache_staging.py](../rewind_archive/cache_staging.py)의 `stage_pose269_cache`가 지정 파일 한 개를 readonly로 읽어 `.partial`에 chunk copy하면서 SHA를 계산한다. Local 전체 SHA를 재확인하고 기존 `input_spec`으로 ZIP/manifest를 검증한 뒤에만 최종 cache 파일로 게시한다.
5. 최신 준비 report가 검증 완료 상태일 때 기존 workflow preflight→한 번의 dataset 실행으로 간다. 검토된 auth factory·재시도 정책·실행 경로는 보존한다. 인증 UI 호출을 run 셀에서 반복하지 않는다.
6. 실패·중단·완료 뒤에는 회수 셀만 사용한다. Dataset 실행 보고서가 없으면 인증이나 source 모듈 없이 작은 준비 ZIP을 회수한다. 실제 job이 있으면 기존 결과·확장 cache 회수 경로에 준비 report도 함께 넣는다.

## 별도 준비 예산과 계측

| Cache staging 범위 | 상한 |
| --- | --- |
| Readonly source file 논리 읽기 | 정확246522402bytes, 새 복사 시 한 번 |
| Local 전체 SHA 검증 | 최대246522402bytes, 한 번 |
| Chunk | read/write 단위1MiB, 전체 cache를 Python bytes로 적재하지 않음 |
| Validator ZIP metadata 읽기 | 별도2MiB 이하 |
| Copy+hash+validator 시간 | 300초 협력 한도 |
| Target/partial와 report 예약 | 논리256MiB |

새 복사의 본문 논리 읽기 합은493044804bytes다. 이는 실제 Drive FUSE wire 전송량·prefetch·재접속 또는 OS 전체 disk IO를 계측한 값이 아니다. `source_read_bytes`, `local_verify_bytes`, `written_bytes`, `validator_metadata_read_bytes`, 경과 시간을 구분한다. Validator를 위한 `input_spec`의 선택 `read_callback`은 기본None에서 기존 동작을 유지하고 새 준비 경로에서만 metadata를 따로 센다. 후속 preflight의 metadata 재검사와 prepare의 prefix import/replay·export는 각자 별도 범위다.

Mount/auth·작은 ZIP 업로드/해제·원본 dataset prepare192MiB·최종 cache export는 위300초와 별개다. 가용 RAM/임시 disk는 기존대로 각각2GiB 이상 확인한다. Timeout은 Python read 경계와 validator 전후의 협력 검사다. 한 번의 FUSE read가 막히면300초에 즉시 중단된다고 보장하지 않는다. Runtime 전체가 사라지면 Python finally/report 저장도 보장하지 않는다.

## 파일 보호와 실패 회수

원본 Drive cache는 쓰거나 지우지 않는다. Source/target 동일 파일·hardlink 동일성·symlink/junction·source/input 코드 트리 안의 경로·Drive 아래의 출력·workspace 안의 staging/report를 거부한다. 새 partial은 `xb`로 만들고 최종 게시도 기존 target을 덮어쓰지 않는다. 정확한 기존 local target은 전체 SHA와 validator를 통과하면 읽기 재사용하며 이때 Drive 본문은 다시 읽지 않는다.

불일치 target/기존 partial을 덮어쓰거나 자동 이어붙이지 않는다. 실패 partial은 보존하고 report에 현재 byte 수와 안전한 예외 종류/원인을 남긴다. Root 검토 없이 삭제·새 경로 fallback·dataset job 재실행을 하지 않는다. 같은 pin의 report에는 최대8개의 준비 시도 기록을 보존한다. 원문 transport 예외나 자격증명을 report에 넣지 않는다.

준비 ZIP에는 report와 partial의 상태만 넣고 약247MB partial 자체를 중복 압축하지 않는다. Partial은 local 경로에 남는다. Report가 바뀌면 report SHA별 새 `*_stage_<hash>_preparation_result.zip`을 만들고 기존 ZIP을 보존한다. 기존 ZIP을 재사용할 때도 내부 report byte가 현재 report와 같아야 한다. 따라서 후속 준비 시도 뒤 오래된 준비 ZIP을 다시 내려받는 문제를 막는다. 실제 pose job의 결과/확장 cache ZIP은 기존 회수 계약을 따른다.

## 로컬 확인과 남은 한계

보존 local cache로 실제 chunk copy246522402bytes+local hash246522402bytes와 `input_spec`을 통과했다. Local 복사 경과0.797초였으며 Drive/Colab 성능 결과가 아니다. Exact staging 재사용은 source 본문0/local 검증246522402/validator metadata107044bytes로 통과했다. 근거는 `outputs/pose269_drive_stage_20261103/developer/local_validation/stage.json`이다. 원본 cache pin과 dataset 목표/상한은 유지했다.

개발 최종21개/8.860초 failure/error/skip0 통과다. Chunk copy·변조·partial 실패·timeout 후 read byte 집계·validator timeout·job/경로/링크 거부·작은 두 ZIP만 업로드·auth/mount 순서·stale 준비 ZIP 방지·기존269 profile/notebook·실제 auth factory 계약을 검사했다. 링크 거부는 Windows 권한에 의존하지 않는 predicate 모의 검사다. 로그는 `outputs/pose269_drive_stage_20261103/developer/final_tests.txt`에 보존한다.

실제 Drive 파일 존재/경로·mount 복사·Colab 연결 안정성·OOM 원인과 해결은 미검증이다. Cache의 Drive 보관과 정확 경로 확인은 root/탐색 세션에서 별도로 진행한다. 개발은 원격/실제 Colab·dataset 원본 요청·학습·다운로드·의존성 설치·커밋/push를 하지 않았다.

Root 독립 검사27개가 failure/error/skip0으로 통과했다. 실제 local cache 복사·재사용과 metadata107044bytes 계측도 확인했으며 근거는 `outputs/g25_next_20261103/root/drive_stage_tests.txt`와 `stage_review/stage.json`이다. 실제 Drive 성능 측정은 아니다.

사용자는 같은 cache를 기존 `REWIND-like prototype/reports`에 새 `pose_prefix235_f902b25a.zip`으로 보관하는 것을 승인했다. 연결 도구의100MiB 제한 및 Chrome 확장 프로그램의 파일 URL 접근 미설정으로 업로드는 완료되지 않았다. 파일 존재를 확인하기 전에는 경로를 고정하거나 실행 준비 완료로 표시하지 않는다.

## 사용자 업로드 확인 (2026-11-03)

사용자가 `reports/cache/pose_prefix235_f902b25a.zip`을 업로드했다. 폴더 화면과 파일 ID `1YbDRfa576EssySfyHJiZrE1DBWbT57MU` metadata 조회로 부모 cache 폴더 `1_34O03XngrcP_8JugV4ItbX6GEAIPY0L` 및246522402bytes를 확인했다. 검색 목록에는 아직 표시되지 않았으므로 빈 검색 결과를 파일 부재로 단정하지 않는다. 전체 원격 SHA는 이번 조회에서 제공되지 않았으며 Colab staging의 고정 SHA 검증은 유지한다. 앞의 경로 미확인 기록은 업로드 전 상태다.

실행본의 명시 경로는 `/content/drive/MyDrive/REWIND-like prototype/reports/cache/pose_prefix235_f902b25a.zip`이다. 검토된 코드1212266의 source/input을 그대로 사용하고 경로만 채운 별도 ready notebook을 발급한다. 실제 mount·복사·pose269 실행은 아직 미검증이다. 향후 캐시 업로드는 사용자에게 정확한 파일과 목적지를 안내해 요청하며 에이전트가 직접 업로드하지 않는다.
