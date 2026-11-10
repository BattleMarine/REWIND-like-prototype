# v1.1·v1.2 병행 실험

2026-11-10 사용자 결정. v1.1은 동일train1,000페어에서epoch를 늘리고 v1.2는ArchVis 전체train으로 확대한다. 규모·예상비용을 비교한 뒤 사용자가전체를 선택했다. v1.3의 다른 시나리오(CityPack 등 사용자 명칭)는 계획만 기록하고 구현·수집·학습을 시작하지 않는다. 이전의v1 전체1,000제한은v1.1에 유지하며v1.2확대는 별도실험으로 허용한다.

## 비교 설계

| 항목 | v1.1 | v1.2 |
| --- | --- | --- |
| 질문 | 같은 데이터에서 학습량을 늘리면 형태가 학습되는가 | ArchVis 범위를 늘리면 예측이 개선되는가 |
| Train | 기존 고정1,000페어·20개50frame clip | 보존 공식split/Drive목록 기준20,888페어, 새파일검증 필요 |
| Validation | 기존2,236, teacher현재29clip/1,450 | 같은키·평가coverage 유지 |
| 초기 학습 설계 | frontend/body/hand 각10epoch | 데이터검증·tail처리 후 고정학습설정 발급 |
| 현재 준비 | epoch1/5/10 저장·진단·회수 | 전체키 대조·파일감사·완전coverage 계획 |

모델구조·128px·frontend batch2·FP32·LR·seed는 기존v1을 기준으로 고정한다. 데이터와학습량을 동시에 바꾼 결과를순수데이터효과로 해석하지 않는다. v1.2에는동일epoch뿐 아니라처리페어·optimizer update·GPU시간을 함께 보고하며, 동일update비교를 위한저장시점은학습인계에서 고정한다. v1.1은소수샘플overfit검사가 아니라전체1,000페어 학습량확대다.

v1.1은같은seed의새run으로 시작하여epoch1/5/10을 비교한다. 기존1epoch checkpoint의config/source pin을 무시하고resume하지 않는다. body는선택된고정frontend로, hand는선택된고정body로학습한다. 따라서각단계epoch5파일을임의로결합한것을end-to-end5epoch결과로 부르지 않는다. 저장된의존checkpoint SHA를따른다.

## 유지하는 데이터 계약

RGB·pose 추출본을 사용하며depth/test는 사용하지 않는다. 명목25FPS와frame-key stereo대응 가정, 실제timestamp·motion독립성미확인,참고validation조건을명시한다. 기존엄격가드와V1-DIAGNOSTIC-001을완화하거나덮어쓰지않는다. 새데이터의카메라/좌표/정합감사는별도로통과해야하며,89장기존GT검토를전체추가자료의시각검토로확장하지않는다.

보존inventory는2026-11-09snapshot이다. 공식ArchVis train286sequence에서중복키1개를제외한20,888페어와validation2,236을계획한다. 최신Drive의추가업로드를검증없이자동합산하지않는다. 현재50frame완전구간은281개/14,050train frames뿐이다. 전체학습에서남은short/tail을숨겨제외하지않도록valid-mask를포함한별도coverage계획과로더검증이필요하다. 준비manifest가생성된것을전체teacher학습연결완료로표시하지않는다.

## 비용 추정과 제약

기준은실제A100-SXM4-40GB,FP32,128px,batch2,1epoch/540update,복원포함850.846초다. 같은코드의계측을선형외삽했으며 확정실행시간이아니다. 반쪽데이터 구성은미선정이므로절반추정은근사다.

| train범위 | 각단계1epoch | 각단계10epoch | validation포함원본캐시추정 |
| --- | ---: | ---: | ---: |
| 1,000 | 실측14분11초 | 약85분 | 7.55GB |
| 절반 약10,444 | 약56분 | 약6.1시간 | 29.6GB |
| 전체20,888 | 약1시간44분 | 약11.2시간 | 54.0GB |

이는완전50frame clip기준 외삽이다. Short/tail처리·저장시점추가·재개시특징재구축·Drive변동은추가비용이며최초전수검사/캐시생성은표에서제외했다. CPU전처리계측은다른step/평가와중첩되어직접더하지않고,전체에서분리계측단계를제외한잔여비용을특징준비추정에사용했다. 산식은Git제외 `outputs/v1_branch_experiments/cost_estimate.json`에보존한다.

모델파라미터는frontend352,918/body15,010,822/hand15,571,974,합30,935,714개다. FP32파라미터123,742,856bytes(약124MB/118MiB)로데이터크기에따라증가하지않는다. Optimizer·중간activation·데이터/특징캐시·여러checkpoint는별도다. 실제기존peakGPU할당은약0.786GB였으나확대실행의VRAM이나RAM보장은아니다.

v1.1은호출별2시간과저장/메모리상한을유지하며중단시동일고정config의checkpoint로재개한다. v1.2전체약11시간을한번의2시간런타임에넣지않는다. 새입력감사와청크/재개·디스크공간확인이먼저다. 현재v1.2준비는원본을직접읽어metadata를검증하는단계이며54GB복사캐시를자동생성하거나Drive로업로드하는실행은아니다.

## 완료와 인계

실험별source revision·config·selection·checkpoint·출력경로를분리하고rawcache는검증된동일부분만재사용한다. 로컬개발검증후고정노트북을테스트세션에인계한다. v1.1은실제학습곡선·2D확률집중도·3D관절spread·gradient·예측비교·시간을회수한다. v1.2는전수파일검증과누락/중복/short-tail/예산결과를먼저회수하며실제데이터확인없이학습가드를해제하지않는다. 준비와실행결과는별도로보고한다.

## 2026-11-10 개발 검토

v1.1은별도 `epoch-expansion` CLI/config와고정notebook생성기를구현했다. 단계별1/5/10epoch 모델·dependency SHA, epoch첫update gradient L2, validation첫32frame heatmap entropy·중앙거리와3D형태지표를회수한다. 3D spread는관절중심까지의Euclidean거리RMS이며원인진단문서의성분별RMS와수치정의가다르다. v1.1 run저장2GiB·회수1GiB·호출2시간·종료여유120초, rawcache12GiB·feature128MiB를유지한다. 새런타임복원시에도고정source/config/environment계약을확인한다.

Root는관련28개검사를실행했다. 27개통과, 기존checkpoint재개검사1개는Windows sandbox의원자적파일교체권한오류후동일검사를격리제한밖에서재실행해통과했다. 검사를생략하거나코드가드를우회하지않았다. 실제A100의5,400update는아직실행하지않았다.

v1.2 metadata계산은payload읽기없이완료했다. 전체train20,888 real frames는564개post-padded clip으로계획되며padding7,312자리,실제프레임중복0이다. 완전clip281개외에short156개/4,244frames와tail127개/2,594frames가있다. Train번호gap1개가있고gap을가로지르지않는다. 이는학습로더구현완료가아닌coverage계획이다. 기존validation평가조건은별도로유지한다.

## v1.2 전체 파일 검사 구현

`rewind_prepare.full_archvis`와 `scripts/build_archvis_v12_prepare.py`를 추가했다. 고정 보존 목록의 train 20,888·validation 2,236페어(파일 69,372개)를 대상으로 목록/stat 검사 후 source plan SHA를 고정해 내용 검사를 수행한다. RGB·pose만 읽으며 원본을 수정하지 않는다. 기존 3,236페어의 파일 해시도 보존한다. 목록 밖 업로드나 누락이 발견되면 임의로 범위를 변경하지 않고 보고한다.

Colab CPU·Pillow 환경이며 목록 검사 120분/16MiB, 내용 검사 6시간/65GiB, Linux 주소공간 3GiB를 상한으로 둔다. 외부 실행 제한은 각각 7,230초/21,630초다. 로컬 보고서·회수 예산 512MiB와 여유 1GiB가 필요하다. 실제 전송·속도·메모리는 Colab에서 확인해야 한다. 최초 검사 약54.26GB는 학습 시간 추정에서 제외한 별도 비용이다.

진행 상태는 첫 페어와 250페어마다 저장한다. 일반 실패에서는 마지막 파일 경로와 원인을 보존하고, 강제 종료에서는 마지막 게시 상태까지만 남는다. 마지막 셀은 상태·목록·coverage·완성 manifest와 파일별 해시가 있는 회수 ZIP을 생성한다. 원본 이미지나 캐시는 포함하지 않는다. 내용 검사 이어읽기는 아직 지원하지 않으며 실패 후 자동 재시도하지 않는다.

Root가 새 검사 5개와 기존 추출본 검사 7개, 총12개를 재실행해 통과했다. 개발자는 추가 temporal/audit 회귀 7개도 통과했다. 실제 Drive 전수 내용 검사·확대 GPU 학습은 미실행이다. 검사 성공 시에도 `train_ready=false`, `temporal_ready=false`를 유지하며, 짧은 구간의 mask/padding 학습 로더와 전체 캐시·학습 연결은 후속 작업이다.

### 2026-11-10 사전검사 시간 상한 변경

첫 실제 사전검사는 목록 대조에 성공했으나 901.12초에 `extracted_elapsed_limit`로 중단됐다. 누락·예상 밖 키·새 중복·불완전 페어는 없었으며 파일 내용 검사는 미실행이다. 회수 ZIP SHA256은 `f610811aea8b249193dc30850d2065f9167c416d22535819f470e7641de48f28`이며 root가 CRC와 내부5개 파일 크기/해시를 확인했다.

사용자 요청에 따라 사전검사만 120분(내부7,200초/외부7,230초)으로 변경한다. 읽기량16MiB, 내용 검사6시간/65GiB, RAM·저장·데이터·검사 로직은 유지한다. 기존 실패 결과를 보존하고 새 고정 소스/노트북의 새 attempt에서 처음부터 재시도한다. 진행 중인 v1.1은 기존 고정본을 유지한다. 이 변경은 시간 부족 원인을 확정하거나 실제 재시도 성공을 의미하지 않는다.
