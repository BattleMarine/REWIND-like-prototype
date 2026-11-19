# v2 CPU 캐시 첫 샤드 실행

04:18:37 UTC Drive 메타데이터 스냅샷에서 고정한 136,011개 train·validation 프레임 중 정렬 순서의 첫 128개를 처리한다. 첫 샤드의 RGB 좌우·pose 384개 원본 파일은 307,418,238바이트다. 처음 실행이 성공하면 같은 접두 구간을 이어서 한 샤드 더 만드는 방법을 아래에 설명한다. 실제 Drive·Colab 실행 결과와 비용은 회수 후 판단한다. [전체 v2 계획](V2_PLAN.md)

스냅샷의 메타데이터 탐색 중 Drive 변경이 관측됐다. 이 실행은 해당 시점에 정리된 입력에서 공식 train·validation 소속과 SHA가 결합된 명단을 고정한다. 런타임은 전체 manifest와 공식 split을 읽고, 대상 프레임의 파일만 명단의 경로로 직접 연다. 파일별 크기와 SHA-256, 좌우 RGB 디코드, pose frame 번호·관절 supervision mask·카메라 변환을 확인한다. 이 검사는 대상 프레임의 payload에 적용된다. 전체 Drive 동기화·무결성이나 timestamp·stereo 동기·motion 독립성 검증은 별도로 남아 있다. 공식 test 파일은 읽지 않는다.

## 고정본 준비

오케스트레이션에서 제공하는 고정 source ZIP 하나와 `v2_cpu_cache.ipynb`를 사용한다. Source ZIP에는 `SOURCE_REVISION.txt`, v2 소스와 기본 config, 공식 spec/split, `reference/v2_snapshot_manifest.zip`이 들어간다. Source ZIP 이름·revision·SHA와 portable source identity, config·spec·manifest의 크기 및 SHA, notebook SHA는 릴리스 `receipt.json`에 기록한다. Manifest ZIP 21,189,691바이트의 SHA-256은 `1557ab67f765ab1f5b545ea2dfe75bae7878672cf357a6ce4cc68091e58690c5`이고, 단일 JSON 파일은 141,390,812바이트다. 일반 v2 CLI의 128 MiB 제한은 기존대로 적용한다. 이 CPU 경로만 정확한 바이트 크기와 ZIP·JSON SHA를 고정하고 256 MiB 상한으로 같은 manifest 스키마를 읽는다.

오케스트레이션은 새 revision 소스 ZIP을 구성할 때 위 reference 파일들과 현재 CPU 실행 스크립트를 포함한다. 그런 다음 source ZIP과 스냅샷 manifest receipt를 다음 명령으로 검증해 전달용 노트북을 생성한다. 출력 디렉터리의 부모는 미리 준비한다.

```powershell
python -m scripts.build_v2_cpu_cache --source <fixed-source.zip> --source-sha256 <sha256> --revision <40-char-revision> --manifest-receipt <snapshot-receipt.json> --output <new-handoff-folder>
```

## Colab 실행

1. Colab에서 CPU 런타임으로 노트북을 열고 source ZIP을 이름 그대로 `/content`에 올린다. ZIP 내부 manifest는 자동으로 검증하고 풀기 때문에 별도 업로드가 필요 없다.
2. `DRIVE_PROJECT`에 데이터가 이미 있는 프로젝트 폴더를 지정한다. 기본값은 `/content/drive/MyDrive/REWIND-like prototype`이다. 그 아래 `datasets/UnrealEgo2/`의 `UnrealEgoData2_train_val_rgb`, `UnrealEgoData2_train_val_pose`와 기존 `reports/cache/` 폴더를 사용한다.
3. 첫 실행에서는 `RETAINED_CATALOG_FILE_SHA256`을 비운다. `RUN_ONE_SHARD`를 켠 뒤 전체 실행하고 Drive 마운트를 허용한다. 이 스위치는 생성한 캐시 샤드와 카탈로그 및 작은 진단 JSON의 Drive 게시도 요청한다.
4. 성공은 진단 JSON의 `remote_verified`, `worker.local_completed_frames=128`, `worker.retention.verified_shards=1`과 카탈로그 파일 SHA-256으로 확인한다. `reports/cache/v2_cpu_cache_041837Z/`의 `catalog.json`, `shard-00000-<sha256>.zip`, `invocation-*.json`을 회수한다. 샤드 ZIP에는 `images.npy`, `targets.pt`, `receipt.json`이 있다.

보고서에는 대상 프레임 범위·split, 고정 입력 바이트, 실제 원본·복사 관찰량, 준비/복원/보관 및 전체 시간, 준비 처리량과 package pin이 들어간다. OS와 Drive 내부의 물리적 읽기 또는 청구량을 직접 측정하는 계수는 아니다. 성공·실패·중단 경로에서 작은 진단 JSON을 먼저 로컬에 남긴다. 마운트와 보고서 폴더에 접근할 수 있으면 Drive에도 올리고 게시본을 다시 읽는다. Drive 연결이나 게시가 실패하면 출력에 표시된 `/content/rewind_v2_cpu_cache/invocation-*/diagnostic.json`과 `worker.log`를 회수한다. 런타임 VM이 강제 종료되면 종료 핸들러가 실행되지 않을 수 있다.

## 접두 구간 복구와 자원 제한

다음 실행에서 같은 노트북·source ZIP을 사용하고 최근 출력의 **카탈로그 파일 SHA-256**을 입력한다. 같은 VM에서는 로컬 receipt와 파일을 재검사한다. 새 VM에서는 Drive ZIP의 size/SHA 및 샤드 receipt를 확인해 로컬 접두 구간을 복원하고, 다음 샤드의 원본 파일만 읽는다. 이전 로컬 접두 구간은 그대로 보존한다. 로컬 생성까지 성공하고 카탈로그 게시 전에 실패했다면 같은 VM과 직전 카탈로그 pin으로 다시 실행해 보관을 재시도할 수 있다. 첫 샤드 보관에 실패해 아직 카탈로그가 없으면 pin을 비운다. 카탈로그 게시 후 진단 JSON만 실패했다면 출력 또는 로컬 `result.json`에서 이미 게시된 카탈로그의 핀을 확인한다. 실패한 실행의 진단과 기존 접두 구간을 보존하고 수동 실행 전에 실제 상태를 확인한다.

샤드는 기존 v2 캐시 계약을 사용한다. manifest 콘텐츠 pin, spec, cache/preprocessing 소스와 camera 정책, **PyTorch 버전 문자열(빌드 접미사 포함), NumPy, Pillow**가 모두 맞아야 재사용할 수 있다. 보고서에 이 값과 Python 버전, Torch 빌드 CUDA 버전을 기록한다. CPU에서 처리했더라도 설치된 Torch가 CUDA 빌드일 수 있다. 후속 GPU 런타임에서 같은 binding으로 읽을 수 있는지는 보고서와 해당 런타임을 대조해야 한다.

한 호출의 상한은 노트북 실행 셀 기준 28,800초다. 부트스트랩·마운트 경과를 worker 시간에서 차감하고 외부 종료·진단 시간을 먼저 남긴다. Worker는 기존 1,320초 종료 예비 시간과 준비 단계의 별도 패키지·복사 여유를 내부에 둔다. 추가 로컬 디스크, RAM과 4 GiB RSS 기준, 호출당 read/write 각각 8 GiB, 캐시 12 GiB, 개별 원본 8 MiB 및 개별 샤드 ZIP 64 MiB를 확인한다. 대기 중인 전체 누적 예산·학습 정책을 가정하지 않고, 고정 한 샤드 작업만 실행하며 자동 반복은 없다.

이 경로의 첫 호출에서 나오는 결과는 전체 136,011프레임 중 128프레임의 접두 구간이다. 기존 전체 v2 cache 구성을 기준으로 RGB 배열만 약 49.8 GiB이며 pose·카탈로그·복사 임시 공간은 별도다. 전체 실행/학습에 필요한 대용량 저장, 복원 비용과 batch 크기, 일반 v2 CLI 로더 변경, 후속 GPU 호환은 별도 검토한다.
