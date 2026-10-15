# Colab 환경 준비·부분 추출·합성 모델 실행 가이드

2026-10-15 · 개인 Google 계정의 Colab 관리형 런타임을 사용하는 안내입니다. 로컬 검증과 실제 Colab 검증은 구분합니다. 사용자의 실제 Colab 진단에서 Drive metadata HTTP 200 응답을 확인했습니다. 후속 사용자 보고서에서 manifest 생성과 pose 32개 추출·런타임 verify 성공을 확인했습니다. 후속 [sample 감사](ARCHVIS_POSE_AUDIT.md)에서 32 JSON의 구조·수치와 연속 18프레임을 검사했습니다. 공식 train 경로 소속과 annotation 재투영의 수치 정합성도 후속 확인했지만, 단위·좌표 의미·시간·원본 motion 독립성·실제 RGB 대응·GPU 실행은 남아 있습니다. 수치와 검토 범위는 [로드맵](ROADMAP.md)에 기록합니다.

오케스트레이션 검토에서는 노트북의 환경 확인·테스트·합성 학습·재개·평가·캐시 셀을 Windows CPU에서 실행했습니다. 59개 테스트가 통과했고, 결과 17개 파일의 로컬 복사·SHA-256 일치를 확인했습니다. 업로드·Drive 인증 셀은 실행하지 않았으며 로컬 복사 검증을 실제 Drive 전송 성공으로 해석하지 않습니다.

Chunked metadata 수정까지 포함한 최신 Python 회귀검사는 97개가 통과했습니다(기존 59개 + 압축 준비 24개 + metadata/pilot 14개). HTTP 응답·느린 전송 검사는 로컬 loopback 서버를 사용했습니다. 실제 Drive·Colab 추출 성공을 검증한 수치는 아닙니다.

압축 준비 노트북은 별도 로컬 ZIP·임시 폴더에서 코드 추출·커널 import와 합성 추출/재사용/매칭을 실행해 RGB 8개·pose 4개, 대응 triplet 4개를 확인했습니다. 업로드 UI는 모의 호출했고 실제 Drive 분기는 비활성으로 검사했습니다. 노트북 구문 검증과 로컬 실행 결과를 실제 Colab 인증 성공으로 해석하지 않습니다.

새 metadata·pilot 활성 분기도 모의 인증/session으로 검사해 pose 4개 추출, 보고서 일치, 선택 변경 거부와 실패 시 이전 결과 폐기를 확인했습니다. 별도로 requests 2.34.2·urllib3 2.8.0·google-auth 2.62.0의 실제 전송 코드를 익명 자격과 로컬 서버로 대조했습니다. 이 로컬 검증에서는 Google API·계정 인증을 호출하지 않았습니다. 후속 사용자 Colab 실행 결과와 구분합니다.

## 준비할 파일과 실행 범위

- 프로젝트의 `notebooks/m1_baseline.ipynb`: Colab에 업로드할 실행 노트북입니다.
- 프로젝트의 `notebooks/archive_prepare.ipynb`: 분할 압축의 작은 구간을 읽어 완전한 파일을 준비하고 이름 대응을 검사하는 별도 노트북입니다. 이 준비 단계에는 CPU 런타임을 사용합니다.
- 같은 코드 revision으로 만든 `rewind-colab-source.zip`: 모델·학습·설정·테스트 코드를 담은 소스 묶음입니다. 원본 데이터·checkpoint·인증 정보는 포함하지 않습니다.
- Google Drive: 원본 압축 데이터 보관과 실행 결과의 영구 저장에 사용합니다. 합성 진단은 데이터셋 없이 실행할 수 있습니다.

현재 자동 push를 하지 않으므로 GitHub의 `main`과 로컬 최신 구현은 다를 수 있습니다. 이 가이드는 로컬 커밋의 소스 ZIP 업로드를 기본으로 합니다. 사용자가 별도로 원격에 올린 뒤에는 동일한 revision을 지정해 GitHub에서 가져오는 경로를 선택할 수 있습니다.

현재 가능한 합성 진단과 실제 데이터 학습의 조건은 [M1 실행 안내](M1_BASELINE.md)를 따릅니다. 합성 loss 감소나 재개 검증은 실제 UnrealEgo 정확도·M1 전체 완료의 근거가 아닙니다.

## 1. Colab 노트북 열기

1. [Google Colab](https://colab.research.google.com/)에 로그인합니다.
2. **파일 → 노트북 업로드**에서 `m1_baseline.ipynb`를 선택합니다. 노트북은 Drive에 저장하고 알아보기 쉬운 이름으로 바꿉니다.
3. **런타임 → 런타임 유형 변경**에서 Python 3와 사용 가능한 GPU를 선택합니다. 작은 합성 진단은 CPU로 먼저 확인해도 됩니다.
4. 연결 후 노트북의 환경 확인 셀에서 Python·PyTorch 버전, CUDA 사용 가능 여부, GPU 이름과 여유 디스크를 확인합니다. GPU를 선택했더라도 `torch.cuda.is_available()`이 false이면 GPU 학습을 시작하지 않습니다.

GPU 종류·메모리·사용 시간은 계정과 시점에 따라 달라집니다. 특정 GPU나 고정 세션 길이를 전제하지 않습니다. 초기 합성 진단을 위해 유료 요금제를 먼저 구매할 필요는 없습니다. [Colab FAQ](https://research.google.com/colaboratory/faq.html)

## 2. 로컬 최신 코드 가져오기

전달받은 소스 ZIP이 있다면 노트북의 업로드 셀에서 선택합니다. 노트북 파일만 업로드하는 것으로 Python 모듈까지 전송되지는 않습니다.

소스 ZIP을 직접 만들 때는 로컬 프로젝트 root에서 커밋을 확인하고 다음 명령을 실행합니다. ZIP에는 커밋된 파일만 들어가며 미커밋 수정은 포함되지 않습니다. 생성된 ZIP은 `outputs/`에 있어 Git에서 제외됩니다.

```powershell
New-Item -ItemType Directory -Force outputs | Out-Null
$sourceRevision = git rev-parse HEAD
git archive --format=zip --output=outputs/rewind-colab-source.zip "--add-virtual-file=SOURCE_REVISION.txt:$sourceRevision" HEAD
```

위 명령은 같은 HEAD의 전체 SHA를 ZIP의 `SOURCE_REVISION.txt`에 함께 기록합니다. 전달받은 배포용 ZIP에도 이 파일이 포함됩니다. ZIP 생성 후에는 다음 커밋을 포함하지 않으므로, 코드가 바뀌면 다시 생성합니다.

노트북에서는 ZIP을 Colab 임시 디스크에 풀고 프로젝트 root로 이동합니다. 데이터셋은 이 ZIP에 포함되지 않습니다. 환경 준비 셀에서 코드·설정 파일을 찾지 못하면 ZIP 버전과 추출 위치부터 확인합니다.

## 3. 의존성 확인과 합성 진단

Colab의 기존 PyTorch와 CUDA 조합을 먼저 확인합니다. 로컬 Windows CPU 검증용 설치 명령을 Colab GPU 환경에 그대로 적용하지 않습니다. 추가 설치가 필요한 경우 노트북과 M1 안내의 의존성 범위를 따르고, GPU용 설치가 필요하면 [PyTorch 공식 설치 안내](https://pytorch.org/get-started/locally/)에서 런타임에 맞는 명령을 선택합니다.

노트북 순서에 따라 테스트와 짧은 합성 진단을 실행합니다. 실행 기록에서 다음을 확인합니다.

- 데이터 출처가 합성임을 명시하고 GT 2D 진단과 예측 2D 입력을 구분하는지
- Loss와 gradient가 유한하며 작은 fixture에 대한 학습이 동작하는지
- 중단 checkpoint에서 학습을 이어갈 수 있는지
- 실행 설정·코드 revision·데이터 식별 정보와 결과가 기록되는지

현재 single-frame 진단의 CLI 예시는 다음과 같습니다. 프로젝트 root에서 실행하며 `--dataset`을 생략하면 합성 fixture를 사용합니다. 새 실행에는 비어 있는 output 디렉터리를 지정합니다. 아래 예시는 로컬 CPU에서 동작을 확인한 명령이며 GPU 실행 결과를 뜻하지 않습니다.

```sh
python -m rewind_m1 train --config configs/m1_single_synthetic.json --output outputs/m1_single --steps 80 --device cpu
```

학습 결과의 `data_kind=synthetic_fixture` 표시를 확인합니다. 실제 단위의 성능으로 해석하지 않고, loss 감소·gradient·저장 경로 검사용으로 사용합니다. Temporal 모델, 재개·평가·예측 캐시 명령은 M1 안내와 노트북에 따른 순서로 실행합니다.

실제 Colab에서 실행한 뒤에는 GPU·precision·버전과 결과를 보관해야 합니다. 로컬 CPU 통과 기록을 Colab GPU 통과로 바꾸지 않습니다.

### M2·M3 합성 teacher 실행

M2/M3가 포함된 최신 소스 ZIP을 새 디렉터리에 풀고 그 프로젝트 root로 이동한 뒤, 노트북에 코드 셀을 추가해 아래 CLI를 호출할 수 있습니다. 기존 M1 노트북이 자동으로 M3를 실행하지는 않습니다. M3는 현재 CPU float32만 지원하므로 GPU 런타임을 요구하지 않습니다. 다음 명령의 모델 경로는 로컬 CPU에서 검증했으며 실제 Colab에서의 M2/M3 실행은 아직 검증하지 않았습니다.

```python
import subprocess
import sys

subprocess.run([
    sys.executable, "-m", "rewind_m3.fixture",
    "--output", "outputs/m3_cascade_new",
    "--body-steps", "80", "--hand-steps", "100",
], check=True)
```

`outputs/m3_cascade_new`가 이미 있으면 새 이름을 지정합니다. `comparison.json`은 독립 학습한 none 모델, GT 상체 진단, 고정 body 예측 상체의 결과를 구분합니다. 손목 상대 오차와 global 오차의 의미, 개별 학습 재개 명령은 [M3 안내](M3_WHOLE_BODY_TEACHER.md)를 따릅니다. 별도의 작은 M2 body 실행은 [M2 안내](M2_BODY_TEACHER.md)를 사용합니다.

M3 결과는 기존 M1 노트북의 자동 export 대상에 추가되어 있지 않습니다. `whole.json`, `body.json`, body·hand의 checkpoint/보고서와 조건별 평가를 포함한 **M3 output 폴더 전체**를 별도 Drive 실험 폴더에 보관해야 합니다. 이때 사용한 소스 ZIP도 함께 보관합니다. 기존 로컬 CPU checkpoint를 다른 Python/Torch·코드 환경에서 그대로 재개하는 것은 보장하지 않으며, 현재 정확한 재개 검사는 동일 환경을 요구합니다.

## 4. Drive 연결과 저장 위치

결과 저장이 필요한 경우 노트북에서 Drive를 연결하고 화면에 표시되는 계정 인증 절차를 수행합니다. 인증 토큰을 복사해 코드·설정·출력에 넣지 않습니다. Colab용 연결 예시는 다음과 같으며 로컬 Python에서는 실행하지 않습니다.

```python
from google.colab import drive
drive.mount('/content/drive')
```

| 위치 | 저장할 내용 | 수명 |
| --- | --- | --- |
| Colab `/content` 아래 프로젝트·작업 디렉터리 | 코드, 합성 fixture, 필요한 시퀀스의 추출본, 학습 중 임시 파일 | 런타임 삭제 시 소실될 수 있음 |
| Drive의 사용자 지정 실험 디렉터리 | 실행 설정, 코드 revision, 데이터 manifest, checkpoint, 지표·로그 | 영구 보관 |
| 기존 Drive 데이터셋 폴더 | 다운로드한 원본 분할 압축파일 | 원본 보존 |

Drive 마운트와 Drive API를 통한 부분 읽기는 별개입니다. `drive.mount()`만 실행한다고 범위 다운로드·순차 추출 기능이 생기지는 않습니다. 원본 압축 데이터의 준비 경로는 [로드맵](ROADMAP.md#부분-다운로드-조사와-실행-계획)을 따릅니다.

노트북의 결과 경로 셀에서 `PERSIST_TO_DRIVE = True`로 바꾸면 Drive 인증과 영구 저장 경로를 준비합니다. 학습은 계속 로컬 임시 디스크에서 수행합니다. 학습이 끝나면 마지막 **완료 결과 영구 보관** 셀을 실행해 checkpoint와 관련 metadata를 Drive에 복사하고, 파일을 다시 읽어 SHA-256 일치를 확인합니다. 기존 export 경로는 덮어쓰지 않습니다.

기본값 `False`에서는 임시 디스크에만 결과가 있습니다. 학습 중 자동 Drive 동기화는 구현하지 않았으므로 갑작스러운 런타임 종료 시 마지막 export 이후 결과를 잃을 수 있습니다. 노트북 저장만으로 모델 checkpoint가 보관되지는 않습니다.

## 5. 실제 데이터로 넘어가기 전

먼저 `archive_prepare.ipynb`를 별도로 업로드하고 같은 revision의 소스 ZIP을 가져옵니다. CPU에서 합성 archive 진단을 실행한 뒤, [부분 압축 준비 안내](ARCHIVE_PREPARATION.md)에 따라 읽을 Drive 파일을 선택합니다. 아래 metadata 단계가 인증된 API에서 크기·버전을 확인해 manifest를 생성하므로 이 값을 직접 채우지 않습니다. 실제 Drive 읽기에는 노트북의 Colab 계정 인증과 `AuthorizedSession`이 필요합니다. 결과 저장용 `drive.mount()`와 원본의 HTTP Range 읽기는 서로 다른 연결입니다.

구현은 `000`부터 이어지는 조각을 작은 바이트 범위로 읽고 gzip/tar를 순차 해제합니다. 중간 조각만 독립 해제하거나 전체 조각을 합친 파일을 만들지 않습니다. 필요한 파일의 위치가 뒤쪽이면 선택하지 않은 앞부분도 읽어야 하므로, 작은 추출본이 작은 전송량을 보장하지 않습니다. 기본 상한·중단 이유·지원하는 응답 조건은 부분 압축 준비 안내를 따릅니다. 상한에 도달하면 보고서를 확인하고 다음 실행의 비용을 판단합니다.

추출 보고서에는 완전한 파일의 hash, 읽은 범위와 바이트 수, 중단 이유를 남깁니다. 잘린 파일은 보관하지 않습니다. 추출 manifest와 파일을 함께 보관하면 재사용 검사로 크기·hash·코드·원본 manifest의 일치를 확인할 수 있습니다. 런타임 재생성으로 추출본을 잃으면 앞부분 읽기 비용이 다시 발생합니다. 진행 중인 gzip 내부 상태를 checkpoint에서 재개하는 기능은 없습니다.

Drive의 압축파일이나 추출 결과를 그대로 학습 CLI에 넣지 않습니다. 좌우 RGB·pose의 sequence/frame 식별자 대응, 관절·단위·좌표·split을 검증해야 합니다. 이름 대응이 성공해도 실제 내용 정합성이나 학습 준비 완료를 뜻하지 않으며, 현재 준비 보고서는 항상 `train_ready=false`입니다. 실제 입력의 검증 조건은 README 및 M1 안내를 확인합니다.

로컬 검토에서는 이전에 확보한 공식 서버 RGB 2 MiB·pose 256 KiB 구간을 새 추출기로 재생해 PNG 2개·JSON 16개의 기존 SHA-256 일치를 확인했습니다. 이 구간에서는 좌우 RGB·pose가 모두 대응하는 프레임이 0개입니다. 이는 사용자 Drive 복사본, 실제 Colab 인증·전송, 전체 archive 무결성 또는 학습 sample 확보를 검증한 결과가 아닙니다.

특히 `train_val` 폴더명만으로 train 소속을 확정하거나, unknown 단위·좌표를 임의로 검증 완료 표시하지 않습니다. 준비된 small sample과 명시적 계약을 얻은 뒤 GT 입력 진단, 예측 입력 baseline 순서로 진행합니다. 한 시퀀스를 프레임별로 나누어 일반화 성능을 보고하지 않습니다.

### Drive 파일 선택부터 소량 읽기까지

1. **CPU 런타임**에서 준비 노트북의 환경·소스 ZIP·사전 테스트 38개·합성 진단을 실행합니다.
2. 선택 입력에 `archive_id`, `dataset_version`, `kind`(`rgb` 또는 `pose`), 원본의 `total_parts`, 읽을 조각의 `index`·`name`·`file_id`를 넣습니다. `drive_part_selection_v1` JSON을 업로드하거나 노트북에서 직접 입력할 수 있습니다. 파일 공유 URL의 `/d/` 뒤 식별자가 file ID이며, 폴더 ID와 혼동하지 않습니다. 목록은 `000`부터 연속해야 합니다. `total_parts`는 현재 다운로드된 개수가 아니라 공식 배포 목록 등으로 확인한 원본 조각 수입니다.
3. `INPUT_MODE`를 직접 입력이면 `"dict"`, 선택 JSON 업로드이면 `"json_upload"`로 정합니다. **`RUN_DRIVE_METADATA = True`만 활성화**해 계정을 인증합니다. 파일 ID·이름·유형·다운로드 가능 여부·크기·version·modifiedTime을 검사하고, 모두 통과한 경우에만 `archive_manifest.json`과 `metadata_report.json`을 만듭니다. 이 단계는 압축 본문을 읽지 않습니다. 실패하면 보고서의 중단 이유를 확인합니다.
4. manifest와 읽기 상한을 확인하고, `SELECTED_MEMBER_GLOBS`에 저장할 member glob을 입력한 뒤 **`RUN_DRIVE_PILOT = True`를 별도로 활성화**합니다. 원본 내부 시퀀스 경로를 아직 모르면 작은 조사 목적의 glob을 명시적으로 선택할 수 있지만, 기존 전송·시간·파일 수 상한을 유지합니다. 실패·0개 추출 시 자동으로 범위를 늘리거나 재시도하지 않습니다. Glob만 바꿀 때는 별도 셀에서 해당 변수만 설정합니다. 입력 셀 전체를 다시 실행하면 metadata 결과가 초기화되므로 metadata부터 다시 실행합니다.
5. `pilot_report.json`에서 완전 파일 수, 재사용 검사, 전송량·시간, 중단 이유와 gzip CRC 상태를 확인합니다. 추출 manifest·metadata 보고서·pilot 보고서를 함께 보관합니다. `train_ready=false`는 유지되며 한 archive의 읽기 성공만으로 좌우 RGB·pose 대응이나 학습 준비가 완료되지 않습니다.

Metadata 기본값은 최대 8개 파일·8회 요청·응답당 16 KiB·누적 본문 128 KiB·30초입니다. 이후 압축 준비 기본값은 별도로 본문 8 MiB·60초이며 두 단계 비용은 합쳐서 판단합니다. 이 값은 개발용 시작점이며 실제 Colab에서 측정한 처리량이나 운영 예산이 아닙니다. 최초 인증, transport 내부 처리와 원본을 앞에서부터 다시 읽는 비용도 고려합니다.

Metadata 조회와 media 읽기는 모두 기본 비활성입니다. 같은 커널에서 이전 실행 결과가 남아 있더라도 선택 입력을 바꿨다면 metadata부터 다시 생성합니다. 셀 실패 이후의 오래된 manifest를 그대로 사용하지 않습니다.

## 6. 세션 종료와 재시작

1. 종료 전에 checkpoint와 설정·manifest·지표가 Drive에 저장됐는지 확인합니다.
2. 새 런타임에서는 GPU 연결, 코드 ZIP 업로드, 의존성 확인, Drive 인증을 다시 수행합니다.
3. 동일 코드 revision·모델 설정·데이터 계약으로 checkpoint를 불러옵니다. 현재 정확한 학습 재개는 Python·PyTorch·device·CUDA와 GPU 환경 정보도 일치해야 합니다. Colab에서 다른 GPU나 버전이 할당되면 재개가 거부될 수 있으므로, 기존 환경을 확인하고 맞춘 뒤 실행합니다. 추출본이 사라졌다면 동일 manifest의 데이터를 다시 준비합니다.
4. 가중치만 불러오는 평가와 optimizer·scheduler·RNG·학습 위치를 복원하는 학습 재개를 구분합니다. 코드·데이터가 달라졌다면 같은 실험의 정확한 재개로 기록하지 않습니다.

## 자주 막히는 지점

### Metadata 단계에서 `metadata_protocol`이 나온 경우

2026-10-15 사용자 Colab에서 `status=200`, `Content-Length` 없음, `Transfer-Encoding=chunked`, JSON 응답을 확인했습니다. 이전 코드는 metadata 응답에 길이 헤더가 반드시 있다고 가정해 본문을 읽기 전에 거부했습니다. 정상 chunked 응답을 처리하되 응답별·누적 읽기 상한과 시간 제한을 유지하도록 수정했습니다. 최초 manifest 생성과 pilot 중 원본 변경 확인에 같은 metadata 읽기를 적용합니다. 실제 압축 본문의 206·Range 검증은 유지합니다.

수정된 코드로 다시 실행하는 순서는 다음과 같습니다.

1. 프로젝트에서 새로 제공한 `outputs/rewind-colab-source.zip`을 준비합니다. 선택 JSON은 그대로 사용할 수 있습니다.
2. Colab **런타임 → 세션 다시 시작**(표시 언어·UI에 따라 런타임 다시 시작)을 실행합니다. 이미 import한 이전 Python 모듈을 없애기 위한 절차입니다. 필요한 임시 결과가 있다면 먼저 내려받습니다.
3. 기존 준비 노트북을 맨 위 환경 셀부터 다시 실행합니다. 소스 업로드 셀에서는 **새 ZIP**을 선택합니다. 출력된 revision이 새 배포 커밋인지 확인합니다. ZIP을 바꾸기만 하고 이전 커널의 모듈을 계속 사용하지 않습니다.
4. 사전 테스트·합성 진단 후 입력 셀을 `INPUT_MODE = "json_upload"`로 실행합니다. 첫 pose 조사에는 `SELECTED_MEMBER_GLOBS = ["*/json/*.json"]`을 사용할 수 있습니다. 이는 읽기 상한 내 JSON 후보 선택이며 특정 sequence 확보를 보장하지 않습니다.
5. `RUN_DRIVE_METADATA = True`로 실행하고 기존 선택 JSON을 업로드·인증합니다. `manifest_ready=True`인지 확인한 뒤에만 `RUN_DRIVE_PILOT = True`로 진행합니다. 실패하면 새 `stop_reason`과 metadata 보고서의 `request_audit`를 확인합니다. 인증 토큰은 공유하지 않습니다.

이후 사용자가 제공한 세 보고서로 수정본의 실제 pose 소량 추출 성공을 확인했습니다. `files_limit`은 저장 파일 수 상한에 따른 정상 중단이며, gzip CRC false는 끝까지 읽지 않아 미검증이라는 뜻입니다. 성공한 기존 추출본과 보고서를 보관하고 내용 감사에 활용합니다. 이번 문서 갱신 때문에 런타임을 재시작하거나 다시 추출할 필요는 없습니다. 새 오류가 있으면 상한이나 검사를 끄지 말고 보고서를 확인합니다.

| 증상 | 확인할 사항 |
| --- | --- |
| 모듈을 찾을 수 없음 | 노트북과 같은 revision의 소스 ZIP 업로드·추출·작업 디렉터리 |
| GPU가 보이지 않음 | 런타임 GPU 선택, 실제 할당 상태, PyTorch CUDA 지원 |
| Drive 파일이 보이지 않음 | 인증한 계정, 접근 권한, 직접 선택한 폴더 경로 |
| Drive 읽기가 느리거나 실패 | 많은 작은 I/O를 피하고 필요한 묶음을 임시 디스크에서 재사용; 용량·할당량 상태 확인 |
| 런타임 재시작 후 데이터·모델 소실 | `/content`와 Drive 저장 위치 구분, 마지막 영구 checkpoint·manifest 확인 |
| 실제 입력 검증 오류 | 필수 metadata와 실제 검증 근거 확인; 검사를 끄거나 임의의 값으로 채우지 않음 |

Colab 실행 전후의 실제 확인 범위는 실행 보고서에 남깁니다. GPU 메모리 부족 시에는 작은 모델·해상도·batch로 pilot부터 다시 측정하고, 장시간 학습 규모는 자원·예산을 정한 뒤 확대합니다.
