# 제한된 압축 prefix cache와 관측 카탈로그

## API9:128MiB cache에서192MiB까지

`posePrefixWarm192MiB_v3`를 추가했습니다. 기존 128MiB 이상의 cache와 고정 `cache_expectation`이 필수이며 cold fallback은 없습니다. 새 network body 80MiB·논리입력 192MiB·420초·773HTTP, 해제 1200MiB·header 30,000·tar metadata 2MiB/단일 64KiB입니다. 초기 metadata는 기존 별도 예산이므로 합계 80.125MiB·450초·781회이며 인증·upload/export 비용은 별도입니다.

V3 format/policy는 `bounded_prefix_cache_v3` / `pose_prefix_warm192_v3`입니다. Cache와 각 ZIP 224MiB·896chunk·local IO 388MiB·export 640MiB/60초, sequence 256/entry 16,000/catalog 6MiB를 사용합니다. HTTP와 독립된 audit 897행을 예약합니다. 기존 v1/v2 계약은 그대로 검사하고 v3 출력은 구버전 reader가 거부합니다. 호출은 `input_spec(path,policy=WARM192_POLICY)`, `prepare(...,profile=WARM192_PROFILE,cache_input=spec,cache_expectation=expected)`이며 export는 판에 맞게 자동 선택합니다.

실제 고정 입력은 134,217,728bytes/513chunk입니다. 첫 새 Range는 같은 offset에서 시작하고 64MiB/256범위를 추가하면 192MiB/769chunk입니다. 최소 입력 128MiB와 세 semantic 핀을 preflight/실제 import에서 검사합니다. 단일 JOB/기본 비활성·실패 결과ZIP·원본 전후 검사·재시도·시간 계측은 API8과 동일합니다. 실제 192MiB 원본 시험은 로컬 개발 검증과 구분합니다.

일반 동시 disk 합은 224×3+16+17=705MiB와 작은 보고서/코드/환경 여유입니다. 일반 출력 16MiB는 aggregate 한계여서 최장 경로에서 두 4MiB 파일의 최대 동시 저장을 보장하지 않습니다. 예약에 걸리면 안전 중단하며 현재 pose 목표에 맞춰 출력 상한을 임의 확대하지 않습니다. 사전 153개 합성 검사는 큰 fixture를 임시 생성·제거하고 그 시간/공간은 준비 예산 밖입니다. [160/192 비교·예약 근거](PREFIX_CACHE_192_PLAN.md)를 따릅니다.

## 기존 API8 / v2 계약

## API8:128MiB bounded warm 확장

`posePrefixWarm128MiB_v2`는 기존 cache를 필수 입력으로 받습니다. 새 network body40MiB, 논리입력128MiB, 준비420초·HTTP392회, 해제800MiB·header20,000·tar metadata2MiB/단일64KiB입니다. 초기 metadata는 기존 별도128KiB·30초·8회이며 전체는 최대40.125MiB·450초·400회입니다. 기존 네 가지 profile과 v1 계약은 변경하지 않습니다. 후속 실제 warm128에서 cache 재사용·확장은 확인했으나 목표 pose는 미확보입니다. 이 절은 구형 v2 계약을 설명합니다.

노트북 사전 검사145개에는 큰 합성 gzip과128MiB 경계 cache를 생성하는 검사가 있습니다. 이 단계의 CPU·RAM·임시 disk·시간은 원본 준비420초 및 단계별 disk 합513MiB에 포함되지 않으며,여러 fixture/cache를 동시에 만들기 때문에 추가 임시 공간이 필요합니다.

| 새 고정 cache 정책 | 값 |
| --- | --- |
| Format / policy_id | `bounded_prefix_cache_v2` / `pose_prefix_warm128_v2` |
| Cache 디렉터리 / input·export ZIP | 각각160MiB |
| 최대chunk / local import·read IO |640개 /260MiB |
| Export local read / 별도 시간 |512MiB /60초 |
| Catalog / range audit | Sequence256·entry16,000·6MiB /1,024행 |
| 일반 추출·보고서 |16MiB,카탈로그와 local audit를 별도 예약 |

JOB은 `cache_expectation`을 추가한14필드입니다. 이 객체의 `manifest_sha256`, `prefix_sha256`, `prefix_bytes`를 검토된 입력으로 고정합니다. `cache_input="upload"`는 그 cache 한 개를 명시적으로 올린다는 뜻입니다. Upload 뒤 manifest가 다르면 실행을 진행하지 않고, 실제 import에서 길이·prefix hash와 bytes를 다시 대조합니다. 입력 누락/손상/변경 시 cold 재다운로드로 전환하지 않습니다. 원본파일 ID는 generic source에 넣지 않습니다.

392회는 cache pre/post2+새 Range129×3+추가retry3에서 정했습니다. 최소 입력은 `128MiB−129×256KiB=100,401,152bytes`(95.75MiB)이며 이를 preflight부터 강제합니다. 실제 고정 시험은100,561,588bytes·384chunk와 그 manifest/prefix hash를 사용합니다. 이 입력에서 새 media는33,656,140bytes·129범위,결과는513chunk입니다. 더 작은 cache를 같은 예산으로128MiB까지 읽는 계약으로 일반화하지 않습니다.

새 reader는 구형 `bounded_prefix_cache_v1`도 구형96MiB/512chunk 규칙으로 읽고,확장 출력은 v2로 표시합니다. v2 재import에는 `input_spec(path,policy=WARM128_POLICY)`,새 `prepare(...,profile=WARM128_PROFILE,cache_input=spec,cache_expectation=expected)`를 사용합니다. `export_cache(directory)`는 출력판에 맞는 고정 정책을 고릅니다. 구형 reader는 v2를 거부하며 v1 생성·기존 API 입력의 동작은 유지합니다. 입력 manifest에서 임의의 큰 상한을 선언할 수 없습니다.

Cache160+input ZIP160+export ZIP160+일반출력16+결과ZIP17=최대513MiB와 작은 보고서 여유를 계산합니다.481MiB는 입력ZIP을 별도128MiB로 강제하는 경우의 계산이며 현재 API8 upload cap은160MiB이므로 일반 상한으로 쓰지 않습니다. Upload 후 local 단계는 별도60초,인증·업로드UI·전체wire·영구보관은 준비 예산과 구분합니다. 보관cache를 다시 업로드하는 비용도 있으므로 원본API 전송 절감을 전체wire 절감으로 해석하지 않습니다.

추가 시간 계수는 `cache_import_seconds`(ZIP 검증·hash·local 복사),`cache_read_seconds`(cached seek/read/hash),`cache_live_check_seconds`(cache window live pre/post),`remote_range_seconds`(새 media와 전후metadata·retry/wait)입니다. 원본 Range 시간에서 cache append는 제외합니다. Cache read 시간은 전체 gzip/parser 재생 시간이 아닙니다. Gzip decode·tar parsing·append·catalog/보고서 처리는 별도 분해하지 않았습니다. 이 값들은 전체 준비 `elapsed_seconds` 안의 부분 계측이므로 전체와 더하지 않으며 `timing_scope`가 범위를 설명합니다. 짧은 모의 요청은 clock 해상도 때문에0일 수 있습니다.

첫 새 Range는 고정 prefix 끝100,561,588부터 시작해야 합니다. 목표가 있어도 준비420초 안의 도달을 보장하지 않습니다.192MiB·전체 인덱싱·다음JOB 자동 실행은 하지 않습니다. [준비안](PREFIX_CACHE_NEXT_STEP.md)의 비용 가정과 [실행 기록](REAL_DATA_NEXT_RUN.md)을 함께 확인합니다.

## 기존 API7 / v1 계약

2026-10-21 · API7 개발 안내. 후속 실제 cold 시험에서 캐시와 관측 카탈로그 보존·hash 정합성을 확인했으나 목표 pose는 미확보다. 실제 warm 재사용과 전송 절감 효과는 미검증이며 [검토 결과](REAL_DATA_NEXT_RUN.md)를 따른다.

## 실행 계약

`posePrefixCache96MiB_v1`은 단일 Drive pose part, `extract/targeted`, 정확한 경로2개에만 적용한다. 기존 세 profile·9/11/12필드 JOB과 기본 scan을 유지한다. 새 JOB은 `cache_input`을 더한13필드를 사용하며, 기존 `metadataPreTimeout10sRetry1Global3_v1`을 함께 선택할 수 있다. 실제 파일 ID는 사용자용 고정 notebook에서만 지정한다.

| 항목 | 고정 상한 |
| --- | --- |
| 준비 network 응답 body |96MiB. 새 media+원본 전후 metadata 포함 |
| 논리 압축 입력 |96MiB. Cache와 새 media를 합쳐 gzip reader에 공급하는 양 |
| 준비 시간 / HTTP 요청 | 협력적420초 /1,157요청 |
| 초기 metadata | 기존 별도128KiB/30초/8요청. 실제 단일part는 정상1요청 |
| 해제 / header / tar metadata |640MiB /16,000 /누적2MiB·단일64KiB |
| Cache local read |196MiB. Import 검증96+재생96+ZIP/manifest 여유4MiB |
| Cache 디렉터리 |128MiB. `prefix.bin`과 manifest 합계 |
| 추출·일반 보고서·카탈로그 | 기존16MiB. 파일2개·각4MiB,카탈로그6MiB를 별도 예약 |
| 관측 카탈로그 | Sequence128개,entry16,000개,파일6MiB |
| Import할 cache ZIP |128MiB,고정2member·STORED |
| Cache export | 별도 협력적60초 /local data read384MiB /ZIP128MiB |
| Upload 후 local 준비 | 파일 선택 UI 이후 별도60초. Upload bytes≤128MiB,metadata read 상계2MiB |

1,157은 정상256KiB Range384개×3요청+추가retry3+warm pre/post2에서 정했다. 초기 metadata까지 최대1,165요청·협력적450초·96.125MiB이고 인증·전체wire는 제외다. 인증UI/토큰교환·HTTP/TLS헤더·OS/RSS hard상한은 기존처럼 별도다. Cache import·검증·local 재생·live 검사·새 media는 모두 준비420초에 포함한다. Export/upload/영구보관·결과 ZIP 생성은 별도 단계다.

Export는 복사·내용검증·최종ZIP hash로 압축 본체를 세 번 읽어96MiB 근처에서 약288MiB local data read가 필요하다.384MiB는 그 여유를 포함한다. IO 계수는 도구가 읽고 쓴 데이터 기준이며 OS/cache/파일시스템의 물리 IO 전체가 아니다. Cache용 disk는 기존 추출16MiB와 분리한다.

Warm에서 input ZIP128+생성cache128+export ZIP128+일반출력16+결과ZIP17을 동시에 보관하면 각 상한 합은417MiB다. 작은 export보고서 공간도 추가로 필요하다. 최초161MiB 제안에는 input ZIP/export ZIP 동시 보관이 빠져 있었으므로 현재 실행의 필요공간으로 사용하지 않는다. 영구보관 업로드는 별도 전송·용량·시간을 소비한다.

## 단일 JOB에서 cold와 warm 구분

1. **Cold:** 고정 source와 notebook을 올리고 사전 검사를 실행한다. 단일 JOB의 profile을 새 고정 profile로,selection을 검토한 pose part로,includes/exact_targets를 같은2경로로 설정한다. `cache_input=None`이므로 cache upload는 없다. 검토용 notebook은 이 값을 미리 채울 수 있다.
2. Preflight에서 실제 상한·설정hash·기존JOB 중복을 확인한다. `EXECUTE_ONCE=True`는 해당 JOB 한 번만 실행한다. 실패·미확보도 그 JOB을 소비하며 새 원본 작업을 자동 반복하지 않는다.
3. 결과 검토 후 cache export를 먼저 시도하고 결과 ZIP을 만든다. `DOWNLOAD_RESULT_ZIP`, `DOWNLOAD_CACHE_ZIP`은 기본False이며 실제 안내는 테스트 세션의 고정 절차를 따른다. 두 ZIP의 경로·SHA-256을 별도로 표시한다.
4. **Warm:** 이후 별도 고정 JOB에서 새job_id/reason과 `cache_input="upload"`를 지정한다. Preflight에서 cache ZIP 한 개만 추가 업로드하고 manifest 식별값으로 바꾼다. 별도 selection JSON·계획 JSON 업로드는 없다. Cache가 같은 원본인지 실제 초기 metadata와 대조한 뒤 검증·재생한다.
5. Warm 전체 재생 시험과 후속 범위 확대 시험은 별도 JOB/누적비용으로 취급한다. 한 셀에서 cold→warm 원본 작업을 연속 자동 실행하지 않는다. 현재 테스트 인계의 우선 대상은 cold 한 번과 두 ZIP의 보존이다.

Cold의 실제 압축 prefix는 metadata와 예약 비용 때문에96MiB보다 작다. 따라서 같은96MiB profile의 warm에서도 목표가 미완료라면 남은 논리입력 여유에 후속 media를 요청할 수 있다. 보장하는 것은 검증된 기존 prefix의 media 재전송을 피한다는 것이며,실제 warm의 media0을 보장하지 않는다. Media0은 목표가 cache 안에 있거나 같은 논리prefix 상한을 둔 로컬fixture의 결과로 구분한다.

Cache upload는 선택 UI가 데이터를 반환한 뒤 크기를 검사하므로 UI/브라우저의 전체 메모리·전송 하드 상한은 아니다. 이후 local 단계는60초,고정이름의 atomic `.partial`→ZIP 저장이다. `cache_upload_report`에 upload bytes·local write·metadata read 상계·시간을 남기며 준비420초와 구분한다.

## 원본 검사와 실패 처리

- Cold 새 media는 기존처럼 pre→정확206 Range→post 검사 후에만 cache에 추가한다. 요청한 길이를 받지 못한 chunk·post 실패는 저장하지 않는다. byte0부터 연속한 범위만 허용한다.
- Warm import는 manifest 자체hash,원본 전체식별자,size/version/modifiedTime,범위 연속성,개별chunk와전체prefix hash를 검증한다. Atomic import가 끝나기 전 시간 초과·중단·오류이면 준비/카탈로그/cache 모두 usable로 올리지 않는다.
- Local 재생 전 live pre 검사,cache 끝 또는 목표 조기 종료 후 live post 검사를 수행한다. 입력은 gzip byte0부터 다시 해제한다. Cache 뒤쪽만 새 Drive Range로 수신하며 중간 gzip의 독립 해제·상태 resume는 아니다.
- 원본변경·hash불일치·parser/CRC·post 실패 실행은 관측 untrusted,cache unusable이다. 실패 cache ZIP도 진단용으로 보관할 수 있으나 warm import는 거부한다. 이전 input ZIP 자체를 수정하지 않는다. 정상 상한 중단에서 post 검증된 연속 prefix는 재사용 가능하다.
- `prefix_cache.ready_for_reuse=true`와 `targets_complete=true`는 별개의 성공 조건이다. Cache가 남아도 pose 저장0개일 수 있고,빈 파일목록의 verification passed는 목표 확보가 아니다.0byte cache는 ready로 세지 않는다.

## 파일과 보고서

```text
archive_job_<job_id>/
  job.json
  metadata/...
  pilot/extraction_manifest.json
  pilot/pilot_report.json
  pilot/observation_catalog.json
  pilot/cache/manifest.json
  pilot/cache/prefix.bin
  pilot/prefix_cache.zip
  pilot/cache_export_report.json
```

`archive_job_<job_id>.zip`은 기존 최대17MiB 결과 bundle이다. 카탈로그와 요약 보고서를 넣고 raw cache 디렉터리/별도cache ZIP은 제외한다. Cache export가 실패해도 결과ZIP 경로는 유지한다. 결과ZIP은 덮어쓰지 않으며 재다운로드 시 기존 snapshot을 반환한다. 최초 export 실패 뒤 나중에 export가 성공한 경우 기존 결과ZIP에 새 export보고서가 소급 포함되지는 않는다. 이후 export의 경로/hash/보고서는 notebook 출력과 별도cache ZIP을 함께 보존한다.

Cache ZIP은 `manifest.json`, `prefix.bin` 두 member의 STORED 형식이다. ZIP 파싱 전 EOCD의2entry·central directory4KiB·무comment/무다중disk를 검사한다. 중복·경로변형·링크·junction·압축bomb·과대manifest/ZIP을 거부하고,새 파일을 기존 파일 위에 덮어쓰지 않는다. 실패한 import/export `.partial`은 제거한다. 기존 export를 다시 받으면 ZIP SHA-256을 검증하고 재사용 검사 IO/시간을 표시한다.

Cache manifest의 `producer_code_sha256`은 생성 코드 출처다. 재사용은 지원하는cacheformat v1과 원본identity/bytes/hash로 검사하며 현재코드hash 일치를 강요하지 않는다. 이것은 원본 전체hash·서명·신뢰할 수 없는 제공자의 진위를 증명하지 않는다. 개인이 보존한 검증된 cache를 입력으로 사용한다. 추출 파일의 기존 현재코드hash 검증 정책은 유지한다.

`observation_catalog.json`은 단일 파일이며 자체hash와크기·건수·trust·cap/truncation 요약만 extraction/pilot/job/review에 연결한다. Entry의sequence_ref는같은파일sequences표의완전경로를가리키고frame/kind/side/header/regular ordinal/payload완료/반환bytes/stored/trust를보존한다. 명명규칙미인식건수와마지막경로도표시한다. Header관측은payload내용검증·전체목록·seekindex가아니다. 기존first128/last1진단도유지한다.

준비 `stats`의 `transfer_bytes`는network 응답body,`logical_compressed_bytes`는reader에공급한cache+새media,`local_io_bytes`는cacheimport/replay의읽기,`cache_write_bytes`는 압축 본체의 실제 write를 나타낸다. Manifest bytes는 `prefix_cache.cache_disk_bytes`에 함께 포함한다. `local_reads`는cache재생횟수이며HTTP요청에넣지않는다. `range_audit.input_source`로cache/drive를구분한다. 단순 LocalSource를cache로바꾼것이아니며,legacy LocalSource의공유ranges/body계수는그대로다.

관절·좌표·좌우영상·시간·분할과실제학습은후속검증이다. [데이터 확보 전략](DATA_ACQUISITION_STRATEGY.md)과[실제 실행 기록](REAL_DATA_NEXT_RUN.md)을함께확인한다.
