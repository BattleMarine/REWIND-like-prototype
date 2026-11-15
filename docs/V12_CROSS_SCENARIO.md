# V12-CROSS-SCENARIO-001 연속클립 평가

2026-11-15 사용자 승인 범위는 저장된 v1.2 모델로 `CityPark_Showcase`, `DowntownWest_Demo_Environment`를 재학습 없이 평가·시각화하는 것이다. v1.3 혼합학습이나 v1.5 학습을 시작하지 않는다. 개발·합성 fixture 검증과 실제 시나리오 결과는 구분하며 정식 revision/source ZIP/notebook 고정·테스트 세션 인계는 root가 담당한다.

## 고정 모델과 입력

- 모델은 v1.2 `training.zip` 873,393,783bytes, SHA256 `48661e75e0969ca140ab9cf0e543d73d2bb3b9ec108d30652cf7fb4f4b76b506`이다. Best frontend9/body10/hand9의 receipt·binding·upstream SHA와 실제 구현 source pin을 기존 시각화 로더로 확인한다. Optimizer 생성·gradient 학습은 없다.
- `configs/v12_cross_scenario_001.json`이 경로·범위·상한을 관리한다. 기존 모델 Drive 경로는 테스트 세션 기록에서 확인했으며 root가 이번에 원격 본문을 다시 읽은 것은 아니다. 런타임에서 크기/SHA를 확인해 최대1GiB를 한 번 로컬 복사하거나 검증한 로컬본을 재사용한다. 새 모델 업로드를 자동 시도하지 않는다.
- 고정 입력 ZIP은 기존 `archvis_v1_inputs.zip`, SHA256 `6256eb7bd148ec51bdb94627c68e54657be3808c607a12e13b68f7e479b15e97`다. 기존 receipt,62관절 spec·단위·calibration, 공식 train/validation/test 인덱스를 검증한다. Test 인덱스의 hash 확인은 test payload 사용을 의미하지 않는다.
- 실제 데이터는 사용자 Drive의 `REWIND-like prototype/datasets/UnrealEgo2/UnrealEgoData2_train_val_rgb`, `UnrealEgoData2_train_val_pose` 아래 RGB 좌우 PNG와 pose JSON 추출본이다. Root는 두 시나리오의 Day 폴더 metadata 존재를 확인했지만 전체 내용·완전성을 검증한 것은 아니다. Archive 다운로드·인덱싱·재개를 하지 않는다.

## 결과와 무관한 사전 선정

공식 validation 후보 CityPark108sequence·Downtown48sequence를 문자열 순으로 탐색한다. 각 시나리오에서 서로 다른 sequence 최대2개, 각 sequence의 좌우 RGB·pose 공통 frame 번호 중 첫 연속50frame을 선택한다. 최대4clip/200pair다. 같은 sequence에서 여러 clip을 골라 quota를 채우지 않는다.

부재 디렉터리, 좌우/pose 누락, 숫자키 중복, frame 번호 gap과50frame 미만 구간의 근거를 기록한다. Quota 뒤의 후보는 검사하지 않았다고 표시하며 누락으로 가장하지 않는다. 한 시나리오에1개만 가용하면 `partial_availability`,0개면 실패다. Train fallback·공식 test 사용은 없다. 모든 최종 clip을 `selection.json`에 seal한 뒤에만 준비와 추론을 시작한다. 성능을 보고 좋은 표본을 고르거나 바꾸지 않는다.

`prepare`는 고정 split 원본에서 각 sequence의 validation 소속을 다시 확인하고, 시나리오/role/path/frame/model/source/input binding을 대조한다. 선택 후 payload hash/decode·frame 식별자·관절/단위·camera 변환·projection·mask 검증이 실패하면 중단하며 다음 후보로 교체하지 않는다. 관측 파일의 새 SHA는 그 파일의 로컬 identity이며 공식 전체 archive 무결성 증거가 아니다.

학습 ArchVis sequence의 인물명·motion명과 일치하는 목록을 `overlap_proxy`에 보존한다. 이름 일치는 실제 원본 motion/인물 중복 가능성의 proxy이며 동일성 확정이 아니다. 이름 불일치도 독립성 증명이 아니다. 원본 motion 독립성·실제 timestamp·stereo 시간 동기는 계속 미확인이다.

## 추론·평가·그림

CPU 또는 지원 CUDA GPU를 명시한다. A100 이름 제한이나 OOM 자동 fallback은 없다. FP32·DDIM10·seed71, body/hand keyed-noise offset100/200을 유지한다. Frontend의 예측2D와 camera만 body 입력으로 사용하며 hand는 예측 body upper와 wrist를 사용한다. GT 관절은 기하 감사·metric·그림 비교에만 사용한다. Pose JSON camera metadata를 쓰므로 camera까지 RGB에서 추정하는 RGB-only 시스템으로 보고하지 않는다.

각 클립은 연속50frame을 한 번에 처리하고 다음 클립에 history를 넘기지 않는다. 같은 noise의 full50 prefix25와 독립25 결과를 `atol=.01mm, rtol=1e-5`로 확인한다. 이는 미래 condition 누출을 검토하는 수치 검사이며 속도/실제 시간 동기 증명이 아니다. 명목 시간은 `frame_index/25`다.

- Frame별2D valid/화면 안 오차(native px), body·hand local/global MPJPE(mm)와 유효 분모를 기록한다. 화면 밖 GT는 valid 전체 지표에서 유지한다.
- Hand local은 finger와 GT wrist가 모두 유효할 때만 사용하고, global metric은 GT finger mask를 유지한다. 같은 값에서 손목 오차를 단순 차감하지 않는다.
- Body·wrist-relative fingers의 bone length와 관절 중심까지 거리 RMS spread를 예측/GT 각각 보고한다. 손가락만의 shape에는 wrist→finger 뼈가 포함되지 않는 기존 정의를 따른다.
- 클립·시나리오·전체 결과는 오차 합/유효 joint 수로 집계한다. Frame별 평균의 단순 평균으로 mask 차이를 숨기지 않는다.
- First/middle/last(clip index0/25/49) PNG3장과 전체50frame GIF, `predictions.json`을 만든다. 3D 투영 축척은 **클립 전체 GT+prediction을 함께** 기준으로 고정하며 frame마다 autozoom하지 않는다. GIF40ms/frame은 명목25FPS일 뿐 실제 녹화 시간을 검증한 것이 아니다.

## 고정 중단 상한과 비용 해석

아래는 예상 실행시간이 아니라 이번 실행의 중단 상한이다.

| 단계/항목 | 상한 |
| --- | --- |
| Metadata 목록·선정 | 1,800초,100,000entry, directory-entry 인코딩/metadata 보수 계량32MiB |
| 선택 payload 준비 | 3,600초,원본 논리 읽기2GiB,개별 파일8MiB,최대200pair |
| 추론·모델 로드·metric·그림 | 3,600초 |
| 모델 copy/hash | 별도600초,최대1GiB; 고정873MB ZIP |
| 결과 일반 파일 합/회수 ZIP | 각각256MiB |
| 시작 여유 | 로컬 disk8GiB,Linux `/proc/meminfo`의 `MemAvailable`4GiB |
| CPU RSS | 4GiB 관측guard; OS 강제 보장 아님 |
| 보관 | 단계마다 package600초 + copy/readback 합계600초 |

Source ZIP64MiB/unpack192MiB·input ZIP16MiB/unpack32MiB는 별도의 bootstrap 범위다. 고정 metadata/소스 pin 재확인 비용과 모델 ZIP 읽기는 payload2GiB와 구분한다. Directory-entry 계량이나 논리 파일 읽기를 실제 Drive/FUSE 네트워크 전송량으로 부르지 않는다.

`MemAvailable`은 회수 가능한 캐시 등을 고려한 OS의 추정치다. `MemFree` 또는 `SC_AVPHYS_PAGES`를 같은 지표로 대신 쓰지 않는다. 메모리 지표 누락·형식 오류는 검증 실패로 처리하며4GiB 조건을 생략하지 않는다. 이 검사는 컨테이너 cgroup 제한·미래 여유·VRAM을 보장하지 않으며 실행 중 RSS 관측guard는 별도로 유지한다.

모델1GiB + raw2GiB + 준비 tensor 약0.08GiB + source/input unpack0.25GiB + 결과0.25GiB + 회수0.25GiB + 렌더 임시 약0.25GiB가 주된 예약이다. 이미 업로드한 모델을 복사하면 로컬 원본0.9GiB가 추가될 수 있어8GiB 여유를 요구한다. Clip별 이미지/특징만 메모리에 두고 GIF는320px tile·클립별로 생성한다. 실제 GPU명/VRAM·threads·단계별 elapsed와 읽기 카운터를 보고한다.

세 단계가 각각 한 번 성공해도 보관 상한은 합계3×(600+600)초가 추가된다. 작업 상한은 metadata30분+prepare60분+infer60분+모델10분이고 bootstrap·보관은 별도다. 이 합계를 실제 소요시간으로 예측하지 않는다. Mount glob/stat 대기는 Python timer로 즉시 중단되지 않을 수 있어 notebook에서 phase subprocess에 작업상한+30초를 적용하고 timeout/중단 시 child를 kill→wait하여 종료를 확인한다. 자동 예산 초기화·무한 재호출은 없다.

## Notebook·CLI와 테스트 세션 순서

메모리 지표 수정본의 배포 위치는 `outputs/v12_cross_scenario_20261115/release_memory_fix/`다. `handoff/v12_cross_scenario_001.ipynb`와 `handoff/handoff_receipt.json`을 사용하며 source ZIP의 정확한 이름·SHA·revision은 receipt에 고정한다. 이전 `release/`는 첫 실행 근거로 보존한다. 실제 실행은 이 프로젝트 테스트 세션 `01a14391-c1b9-7a40-b537-af6b0b43709c`에서 안내한다.

생성기:

```powershell
python -m scripts.build_v12_cross_scenario --source-zip SOURCE_ZIP --source-sha256 SOURCE_SHA --revision REVISION --input-zip INPUT_ZIP --output NEW_HANDOFF_DIRECTORY
```

Root가 검토·커밋한 source ZIP과 기존 input ZIP을 `/content`에 둔다. 모델은 이미 있는 고정 Drive 경로를 기본값으로 사용하며 경로 조정이 필요하면 테스트 세션이 같은 SHA의 파일을 안내한다. 환경은 `requirements-v12.txt`를 확인한다. 개발 세션이 환경 설치·모델 업로드를 실행한 것은 아니다.

1. Bootstrap에서 source/input pin·revision·평가 source identity를 검증하고 새 `run-<UUID>`를 만든다. 기존 결과를 덮어쓰지 않는다.
2. `RUN_SELECTION`만 켜서 metadata를 조사한다. 선정 목록·skip 근거·partial 여부·selection SHA를 기록한다.0clip 실패를 승인 없이 데이터 확대로 해결하지 않는다.
3. `SELECTION_SHA256`에 실제 출력을 넣고 `RUN_PREPARE`를 켠다. 파일/기하 오류가 있으면 해당 실패를 회수한다. 성공한 prepared SHA를 기록한다.
4. `PREPARED_SHA256`과 `DEVICE=cpu` 또는 `cuda`를 명시하고 `RUN_INFERENCE`를 켠다. 모델 copy/hash와 실제 의존성 검사 후 추론한다. 실제 예측 JSON·metric·PNG/GIF를 회수한다.
5. `SHOW_EXISTING_RESULTS`는 저장된 최대4GIF만 표시한다. `SHOW_KEYFRAMES`를 켜면 각3PNG도 표시한다. 새 추론·모델 로드·업로드는 하지 않는다.

모든 실행·표시 스위치는 기본False다. CLI는 `python -m rewind_v1.cross_cli`의 `select`, `prepare`, `infer`, `model-copy`, `package`와 보관 명령으로 분리한다. Notebook의 실제 command는 frozen source·input 경로와 phase pin을 사용한다. 실패/시간 초과 후 임의 rerun보다 결과와 마지막 정상 상태를 테스트 세션이 확인한다.

## 결과 보관과 회수

각 phase의 Drive mount·쓰기 preflight 후 호출을 실행하고 성공·실패·중단·timeout 모두 종료 처리에서 결과 ZIP을 새 attempt에 보관한다. 저장 위치는 `REWIND-like prototype/reports/V12-CROSS-SCENARIO-001/run-<UUID>/{selection,prepared,inference}/attempt-*`다. 크기/SHA readback 및 receipt `state=verified`가 있어야 보관 성공이다. 원래 작업 예외와 보관 실패는 별도로 기록하며 실패 시 로컬 결과·ZIP을 보존한다.

회수 whitelist는 selection/준비 manifest·보고서·예측 JSON·PNG/GIF·notebook/model 상태다. **원본 RGB/pose·준비 tensor·모델 ZIP은 업로드하지 않는다.** 이 자동 결과 보관은 사용자 입력 캐시 업로드 정책 변경이 아니다. VM 강제 종료·커널 사망에서는 finally를 보장하지 못하며, 브라우저 연결 단절과 VM 종료를 구분한다. Model/result publication은 새 경로만 쓰고 기존 파일을 덮어쓰지 않는다.

## 로컬 검증 상태와 한계

첫 사용자 실행 `68fd9b8`은 bootstrap 후 preflight가 `SC_AVPHYS_PAGES`의 free2.5GiB를 available로 오인해 중단됐다. 직후 테스트 세션 보고상 available은80.55GiB였다. Selection·payload·추론 전이며 모델 성능 결과는 없다. 후속 버전은 위 `MemAvailable` 지표로 수정하되4GiB 기준과 나머지 실행 계약을 유지한다. 구버전 노트북의 가드를 직접 수정하지 않고 새 receipt에 고정된 배포본을 사용한다.

기본6검사는 개발20.270초·root15.876초에 통과했다. Validation-only 선정/partial·0개 실패, self-sealed official test/role/model/environment 변이 거부, payload 실패 후 후보 교체 금지, 미래 feature 변조/prefix·sequence reset, optimizer 미생성, missing wrist local/global mask·합/분모와 raw/model 제외 회수를 확인했다.

추가 positive fixture는 유효한62관절·stereo rig·projection을 갖는 합성 pose/PNG의 두 validation50frame clip으로 prepare→tensor load→축소모델 infer→GIF50frame/PNG3장 decode→ZIP→fake 로컬 Drive copy/readback·중복 거부를 확인했다. Notebook 모든 cell compile과 기본no-op도 확인했다. 개발 추가2검사18.349초, root 전체8검사33.711초 통과다. 합성 fixture의 이미지/관절은 실제 시나리오 정확도 근거가 아니다.

Root는 별도로 실제 고정 best 모델+합성50frame 입력에서 새 helper와 기존 `infer_3d`의 whole-body 출력이 정확히 일치(maxdelta0mm)함을21.265초에 확인했다. 기존 helper의 full50/prefix25 관측 최대차이0.006256mm는 위 수치 tolerance 근거다. `outputs/v12_cross_scenario_20261115/root_new_inference_review.json`, `root_actual_weights_temporal_smoke.json`을 따른다. 실제 CityPark/Downtown 파일 본문·추론 성능·CUDA 경로·실제 Drive 보관은 아직 검증하지 않았다.
