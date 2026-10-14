# Colab 환경 준비와 M1 실행 가이드

2026-10-14 · 개인 Google 계정의 Colab 관리형 런타임을 사용하는 안내입니다. 로컬 검증과 실제 Colab 검증은 구분합니다. Colab 접속·Drive 인증·GPU 실행은 아직 이 프로젝트에서 검증하지 않았습니다.

오케스트레이션 검토에서는 노트북의 환경 확인·테스트·합성 학습·재개·평가·캐시 셀을 Windows CPU에서 실행했습니다. 59개 테스트가 통과했고, 결과 17개 파일의 로컬 복사·SHA-256 일치를 확인했습니다. 업로드·Drive 인증 셀은 실행하지 않았으며 로컬 복사 검증을 실제 Drive 전송 성공으로 해석하지 않습니다.

## 준비할 파일과 실행 범위

- 프로젝트의 `notebooks/m1_baseline.ipynb`: Colab에 업로드할 실행 노트북입니다.
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

Drive의 압축파일을 그대로 학습 CLI에 넣지 않습니다. 부분 읽기·순차 해제로 얻은 파일을 좌우 RGB·pose의 sequence/frame 식별자로 맞추고 관절·단위·좌표·split을 검증해야 합니다. 현재 압축 준비 기능의 미구현 상태와 실제 입력의 검증 조건은 README 및 M1 안내를 확인합니다.

특히 `train_val` 폴더명만으로 train 소속을 확정하거나, unknown 단위·좌표를 임의로 검증 완료 표시하지 않습니다. 준비된 small sample과 명시적 계약을 얻은 뒤 GT 입력 진단, 예측 입력 baseline 순서로 진행합니다. 한 시퀀스를 프레임별로 나누어 일반화 성능을 보고하지 않습니다.

## 6. 세션 종료와 재시작

1. 종료 전에 checkpoint와 설정·manifest·지표가 Drive에 저장됐는지 확인합니다.
2. 새 런타임에서는 GPU 연결, 코드 ZIP 업로드, 의존성 확인, Drive 인증을 다시 수행합니다.
3. 동일 코드 revision·모델 설정·데이터 계약으로 checkpoint를 불러옵니다. 현재 정확한 학습 재개는 Python·PyTorch·device·CUDA와 GPU 환경 정보도 일치해야 합니다. Colab에서 다른 GPU나 버전이 할당되면 재개가 거부될 수 있으므로, 기존 환경을 확인하고 맞춘 뒤 실행합니다. 추출본이 사라졌다면 동일 manifest의 데이터를 다시 준비합니다.
4. 가중치만 불러오는 평가와 optimizer·scheduler·RNG·학습 위치를 복원하는 학습 재개를 구분합니다. 코드·데이터가 달라졌다면 같은 실험의 정확한 재개로 기록하지 않습니다.

## 자주 막히는 지점

| 증상 | 확인할 사항 |
| --- | --- |
| 모듈을 찾을 수 없음 | 노트북과 같은 revision의 소스 ZIP 업로드·추출·작업 디렉터리 |
| GPU가 보이지 않음 | 런타임 GPU 선택, 실제 할당 상태, PyTorch CUDA 지원 |
| Drive 파일이 보이지 않음 | 인증한 계정, 접근 권한, 직접 선택한 폴더 경로 |
| Drive 읽기가 느리거나 실패 | 많은 작은 I/O를 피하고 필요한 묶음을 임시 디스크에서 재사용; 용량·할당량 상태 확인 |
| 런타임 재시작 후 데이터·모델 소실 | `/content`와 Drive 저장 위치 구분, 마지막 영구 checkpoint·manifest 확인 |
| 실제 입력 검증 오류 | 필수 metadata와 실제 검증 근거 확인; 검사를 끄거나 임의의 값으로 채우지 않음 |

Colab 실행 전후의 실제 확인 범위는 실행 보고서에 남깁니다. GPU 메모리 부족 시에는 작은 모델·해상도·batch로 pilot부터 다시 측정하고, 장시간 학습 규모는 자원·예산을 정한 뒤 확대합니다.
