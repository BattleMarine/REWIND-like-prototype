# v2 128 B·C 저장 모델 공유본

B(seed 71)와 C(seed 103)는 각각 frontend·body·hand 3epoch, 총 197,961 업데이트를 완료했다. 각 ZIP에는 해당 실행의 선택 가중치와 마지막 상태, 공통 validation 비교 자료가 들어 있다. 선택 epoch는 B가 frontend 3·body 1·hand 3, C가 frontend 3·body 3·hand 2다. 두 실행의 평가 샘플링 seed는 71이다. [공식 비교 결과](EXPERIMENT_RESULTS_REPORT.md) · [프로젝트 Git 저장소](https://github.com/BattleMarine/REWIND-like-prototype)

## 파일과 입력

| 파일 | 내용 |
| --- | --- |
| `variant.json` | 슬롯·학습 seed와 각 파일의 크기·SHA-256을 담은 봉인 메타데이터 |
| `checkpoint.zip`, `diagnostic.json` | 해당 슬롯의 원본 복구 ZIP과 완료 진단 |
| `comparison.zip`, `comparison.json` | 공식 validation 비교 GIF·예측/GT 좌표 및 mask NPZ·지표와 원본 비교 영수증 |
| `v2_variant.py`, `tools/`, `GUIDE_KO.md` | 실행기, A 인덱스와 SHA가 같은 공통 도구 6개, 한국어 안내 |

비교 자료는 내부 `comparison.zip`을 열어 볼 수 있다. 코드를 실행하기 전에 전달받은 B 또는 C 외부 ZIP의 SHA-256을 확인한다. 예측·튜닝·전체 학습은 별도의 `v2_128_A_final_model_and_cache.zip`을 `--base`로 받는다. A ZIP의 고정 SHA-256은 `4591393d3aceb4bb4af9b387c4fda038bdcdd5fd5d613fdcc95f7cb50cdc736a`다. A는 입력 캐시 1,063샤드·136,011페어, manifest·spec·공식 split과 동결 소스의 공급원이다. 캐시 경계는 `dc79219153d69f44020e0c42f9f3ede6cd3add179211f6c300345b2416dab511`이며 기준 학습 소스 revision은 `b7a2d00f3bc755808a451ed4ead873e6513d094b`다. [A 데이터 묶음 안내](V2_GPU128_A_PORTABLE_BUNDLE.md)

예측과 튜닝은 선택한 B 또는 C의 가중치를 사용한다. 2D 특징은 사용 중인 frontend의 가중치와 입력 계약에 묶인다. 선택 클립이나 전체 데이터에 필요한 특징은 해당 frontend로 다시 생성한다. Frontend를 튜닝해 교체한 경우에도 해당 가중치의 특징을 새로 만든다. Body를 교체한 경우 hand에 넘기는 예측 상체도 그 body의 출력으로 다시 만들며, 나머지 단계의 원래 가중치와 조합한 별도 결과로 해석한다.

## 예측·튜닝·재추론

Python 3.11 이상과 PyTorch, NumPy, Pillow가 필요하다. CPU에서 소규모 작업을 실행할 수 있고 CUDA는 설치된 PyTorch가 지원하는 장치를 사용한다. 아래는 B 기준이다. C는 C ZIP과 별도의 작업·출력 폴더를 지정한다. `--work`와 새 실험의 `--output`에는 존재하지 않는 경로를 지정한다.

```text
python -c "from zipfile import ZipFile; z=ZipFile('v2_128_B_portable.zip'); z.extract('v2_variant.py'); z.extract('GUIDE_KO.md'); z.close()"
python v2_variant.py predict --bundle v2_128_B_portable.zip --base v2_128_A_final_model_and_cache.zip --work B-predict-work --output B-predictions --device cpu
python v2_variant.py tune --bundle v2_128_B_portable.zip --base v2_128_A_final_model_and_cache.zip --work B-tune-work --output B-tune-hand --stage hand --train-items 6 --validation-items 3 --updates 2 --device cpu
python v2_variant.py predict --bundle v2_128_B_portable.zip --base v2_128_A_final_model_and_cache.zip --work B-retuned-work --output B-retuned-predictions --warmstart B-tune-hand --device cpu
```

기본 예측은 세 시나리오의 고정 validation 클립을 재생성한다. `--scenario CityPark_Showcase`로 한 시나리오를 고를 수 있다. 저장된 공식 비교 자료와 재생성 결과는 별도 파일로 보존한다. 라이브러리와 장치가 달라지면 수치가 달라질 수 있다. 소규모 튜닝은 선택 가중치와 새 optimizer에서 별도 실험을 시작한다. Frontend의 표본 수 단위는 프레임, body·hand는 클립이다. `--warmstart`에는 같은 슬롯에서 튜닝한 출력 폴더를 지정한다.

## 전체 학습과 diffusion 수정

전체 학습은 공식 train 119,586페어와 validation 16,425페어를 사용한다. `--initialize best`는 선택 슬롯의 세 단계 가중치에서 시작한다. 새로 초기화하려면 `--initialize fresh`를 사용한다. 학습을 위해서는 압축 해제한 128 이미지 캐시 약 50 GiB와 특징·체크포인트 저장 공간을 확보한다. 다음 예의 seed와 epoch는 새 실험 설정이다.

```text
python v2_variant.py train-full --bundle v2_128_B_portable.zip --base v2_128_A_final_model_and_cache.zip --work B-full-work --output B-full --initialize best --frontend-epochs 1 --body-epochs 1 --hand-epochs 1 --seed 71 --device cuda
```

`fork-source`로 만든 복제본에서는 `rewind_m2/diffusion.py`의 noising·DDIM, `rewind_v1/teacher.py`의 body·hand 구성과 loss, `configs/rewind_v12_full_training.json`의 teacher 설정처럼 허용된 모델 파일을 수정할 수 있다. 변경한 소스는 `train-full --editable-source`에서 fresh 초기화로 학습한다. `predict`와 `tune`은 동결 소스와 공유본의 저장 가중치 계약을 사용한다. 모델 폭·관절·조건 입력을 변경한 새 가중치를 원래 공유본의 재추론에 적용하려면 별도의 호환성 검증과 실행 연결이 필요하다. [편집 가능한 파일 및 함수](V2_GPU128_A_PORTABLE_BUNDLE.md#diffusionlosssampling-수정-지점)

```text
python v2_variant.py fork-source --bundle v2_128_B_portable.zip --base v2_128_A_final_model_and_cache.zip --output B-editable
python v2_variant.py train-full --bundle v2_128_B_portable.zip --base v2_128_A_final_model_and_cache.zip --editable-source B-editable --work B-edit-work --output B-edit-fit --initialize fresh --frontend-epochs 1 --body-epochs 1 --hand-epochs 1 --seed 71 --device cuda
```

## 저장소에서 공유 ZIP 생성

해당 슬롯의 완료 진단, 복구 ZIP, 비교 영수증, 비교 ZIP을 준비한다. 복구 ZIP의 원래 이름과 SHA-256은 완료 진단의 `checkpoint.archive`에 기록돼 있다. 아래 인자에는 실제 파일 경로를 넣는다. B·C 각각 한 번 호출하고 기존 로컬 폴더 안에 다른 출력 경로를 지정한다. POSIX에서는 새 파일을 게시할 때 출력 파일시스템의 하드 링크를 사용한다.

```text
python -m scripts.build_v2_variant_bundle --slot B --base v2_128_A_final_model_and_cache.zip --diagnostic invocation-r_wv09gl.json --checkpoint result-000008-837a014d87914503880dc36277b2c7ce.zip --comparison comparison.json --comparison-zip comparison-82bc369d9bb04e3792a78f4c77561d60.zip --output v2_128_B_portable.zip
python -m scripts.build_v2_variant_bundle --slot C --base v2_128_A_final_model_and_cache.zip --diagnostic invocation-25b3kwky.json --checkpoint result-000008-78b379551f444bccb4d557e11c4e4efc.zip --comparison comparison.json --comparison-zip comparison-4c718428551c4e54a6700fa9f9d92617.zip --output v2_128_C_portable.zip
```

생성기는 A ZIP 전체 SHA와 양쪽의 공통 데이터·평가 계약을 확인한다. 슬롯별 완료 진단의 바이트, 선택 모델·특징 식별값과 복구·비교 파일을 대조하고 원본 네 파일을 그대로 복사한다. 생성된 ZIP을 다시 읽어 모든 멤버를 확인한 뒤 최종 경로에 게시한다. 출력의 경로·슬롯·seed·크기·SHA-256은 표준 출력에 JSON으로 표시한다.
