"""Package git-archived source and the original three v1.2 best checkpoints; no data or optimizer."""

import argparse
import hashlib
import io
import json
from pathlib import Path,PurePosixPath
import subprocess
import zipfile

from rewind_archive.contracts import canonical
from rewind_prepare.converter import sha
from scripts import visualize_archvis_v1 as view
from scripts.v12_share_smoke import PARENT_SHA,BINDING,EXPECTED,IMPLEMENTATIONS


ROOT=Path(__file__).resolve().parents[1]
GIT_URL='https://github.com/BattleMarine/REWIND-like-prototype'


def readme(revision):
    qa='**개발 QA용 합성 revision이며 정식 배포본이 아닙니다.**\n\n' if revision=='0'*40 else ''
    return qa+f'''# v1.2 모델 공유 패키지

스테레오 egocentric RGB와 카메라 조건에서 body/hand 3D 관절을 추정하는 연구용 프로토타입입니다. 공식 REWIND 구현이나 논문 성능 재현 완료 모델은 아닙니다.

- Git: {GIT_URL}
- 패키지 소스 revision: `{revision}`
- 로컬 commit이 원격에 push되지 않았을 수 있습니다. **실행 기준은 이 ZIP의 source/입니다.** Git URL의 최신 파일과 섞지 마세요.
- 원본 training ZIP SHA256: `{PARENT_SHA}`. 이 큰 ZIP을 따로 받을 필요는 없습니다.
- 모델: frontend best epoch9 / body best epoch10 / hand best epoch9. 원본 checkpoint bytes와 binding을 그대로 보존했습니다.
- Raw RGB·pose·대용량 cache·optimizer/last checkpoint는 포함하지 않습니다. metadata/run.json은 원본 report 중 binding 검증 필드만 보존한 최소 문서입니다.

## 실행 환경과 합성 확인

ZIP을 새 폴더에 해제하세요. 확인한 로컬 환경은 **Python3.12.10 / Torch2.12.1+cpu / Pillow12.2.0 / NumPy2.5.0**입니다. 이 환경에서 실제 모델3개 strict load와 합성2frame CPU 추론을 확인했습니다. **새 환경의 의존성 설치 자체와 CUDA 실행은 검증하지 않았습니다.** 합성 검증은 실제 RGB 정확도 검증이 아닙니다.

새 로컬 환경에서는 선택적으로 가상환경을 만들고 다음을 실행합니다. 아래 경로는 압축 해제한 패키지 루트 기준입니다.

```text
python -m venv .venv
# Windows PowerShell: .venv/Scripts/Activate.ps1
# Linux/macOS: source .venv/bin/activate
python -m pip install -r source/requirements-v12.txt
cd source
python -m scripts.v12_share_smoke --package-root .. --device cpu --frames 2
```

기존 Colab에서는 설치된 CUDA torch를 우선 사용하세요. 위 requirements 전체를 무조건 다시 설치해 torch를 교체하지 말고 `import torch`로 버전·CUDA 사용 가능 여부를 먼저 확인하세요. Pillow/NumPy가 없을 때만 해당 의존성을 설치합니다. CUDA를 사용하려면 마지막 명령에 `--device cuda`를 명시합니다. 지원 CUDA GPU 모델명 제한은 없고, 사용할 수 없으면 오류를 냅니다.

이 명령은 세 모델 strict load → 합성 RGB/카메라 → 고정 frontend → body/hand DDIM10 추론을 확인합니다. `passed`, 유한값과 whole `[2,62,3]` / body `[2,22,3]` / hand local/global `[2,40,3]`가 예상 출력입니다. **합성 결과는 실제 자세 정확도 시연·성능 측정이 아닙니다.**

## 실제 입력을 연결하는 얇은 API

`source/`에서 Python을 실행하면 됩니다. GT 관절은 입력하지 않습니다.

```python
from scripts.v12_share_smoke import load_models, predict
from rewind_m2.data import encode_camera

models = load_models("..", device="cpu")  # 또는 명시적 cuda
# images: torch.float32 [T,2,3,128,128], RGB 0..1, 순서 left/right
# camera: torch.float32 [T,2,9], 아래 정의대로 encode_camera(...,100.)를 frame별 stack
# keys: [(sequence_id, frame_index), ...], 같은 sequence의 연속 프레임
prediction_mm = predict(models, images, camera, keys)
whole = prediction_mm["whole"]
```

T는1~50입니다. 새 clip/sequence는 별도 호출하며 호출 사이 모델 history/cache를 넘기지 않습니다. 출력 whole62/body22/hand40 관절 순서는 `metadata/spec.json`을 따릅니다. 3D 출력은 **device 기준 mm**, hand_local은 대응 wrist-relative mm, hand_global은 예측 body wrist를 붙인 device mm입니다.

Camera는 원래 v1.2의 **절대 world→camera** 표현입니다. `x_camera_mm = R @ x_world_mm + t_mm`. 각 view9개는 R column0 xyz, column1 xyz, translation xyz ÷100mm이며 left/right 순서입니다. 예를 들어 각 frame의 `{{"left":{{"rotation_matrix":R_L,"translation":t_L_mm}},"right":{{"rotation_matrix":R_R,"translation":t_R_mm}}}}`를 `encode_camera(poses,100.)`에 넘깁니다. 잘못된 rotation6D, 단위·축·world/camera 방향을 임의로 추정하지 마세요.

실제 RGB에는 좌우 영상과 그에 맞는 camera metadata·좌표계·전처리가 필요합니다. 기존 입력은 원본1024×1024 RGB를 float32 0..1로 만든 뒤 bilinear `align_corners=True`로128×128에 맞춥니다. 이 공유본은 이미지/동영상 파일을 자유롭게 읽는 신규 decoder/CLI를 제공하지 않습니다. 기존 데이터 전처리는 `rewind_v1/data.py::frame_image`와 `rewind_prepare/converter.py`를 참고하세요. 임의 단안 영상·웹캠을 바로 지원한다는 뜻이 아닙니다.

## 한계와 코드 수정 위치

v1.2는 절대 원점 조건에 큰 민감도가 있고 도시 전이에서 위치 편향·형태 축소가 관찰됐습니다. first-frame midpoint를 추론 때만 임의 적용하면 원본 v1.2와 다른 진단 조건입니다. 이 모델을 v1.5 중점 모델로 부르지 마세요. source/docs/V12_TRAINING_RESULTS.md와 source/docs/EXPERIMENT_RESULTS_REPORT.md의 관찰·조건·제한을 함께 확인하세요. 손 축소와 자세 오류도 남습니다.

- Frontend: `source/rewind_v1/frontend.py`
- Body/hand 구조·loss: `source/rewind_v1/teacher.py`, `source/rewind_m3/models.py`
- 추론: `source/rewind_v1/cross_inference.py`, `source/rewind_m2/diffusion.py`
- 학습 핵심/순서: `source/rewind_v1/training.py`, `runner.py`, `run_stages.py`
- v1.2 전체학습 연결·설정: `source/rewind_v1/full_training.py`, `source/configs/rewind_v12_full_training.json`, `source/scripts/build_archvis_v12_training.py`

이 모델3개는 학습 **resume checkpoint가 아닙니다**. Resume는 같은 source/config/input/environment와 optimizer/scheduler/scaler/RNG/학습 위치가 필요합니다. Warm start/fine-tune은 학습된 model weight로 새 실행을 시작하고 optimizer·seed·학습 위치 등의 정책을 새로 정하는 별도 실험입니다. 공유 ZIP만으로 기존 학습을 이어갈 수 있다고 가정하지 마세요. 새로운 학습 runner나 fine-tune 실행은 이 패키지 범위가 아닙니다.

Loader는 고정 source·원본 binding을 검증합니다. **source를 수정하면 manifest/source pin 검사로 원본 baseline loader 실행이 차단됩니다.** 먼저 원본 패키지를 검증하고 보존한 뒤, 별도 실험 복사본에서 변경 기록과 새로운 loader/warm-start 정책을 명시적으로 설계하세요. 기존 pin 검사를 무력화하는 방식으로 변경본을 원본처럼 실행하지 마세요. 이 공유본은 그 별도 학습 API를 구현하지 않습니다. SHA256_MANIFEST.json은 manifest 자신을 제외한 전체 패키지 파일의 크기·SHA를 기록합니다. 모델들은 원본 bytes이므로 직접 수정하지 마세요.
'''


def git_archive(revision):
    resolved=subprocess.check_output(['git','rev-parse','--verify',revision+'^{commit}'],cwd=ROOT,text=True).strip()
    return resolved,subprocess.check_output(['git','archive','--format=zip',resolved],cwd=ROOT)


def assemble(source_raw,revision,training_zip,output):
    """Internal assembly; all-zero revision is reserved for an explicitly labelled developer QA overlay."""
    output=Path(output)
    if output.exists() or output.is_symlink():raise ValueError('v12_share_new_output')
    if len(source_raw)>64*1024**2:raise ValueError('v12_share_source_zip_cap')
    source={}
    with zipfile.ZipFile(io.BytesIO(source_raw)) as z:
        if len(z.infolist())>3000 or sum(i.file_size for i in z.infolist())>192*1024**2:raise ValueError('v12_share_source_unpacked_cap')
        for info in z.infolist():
            if info.is_dir():continue
            name=info.filename;p=PurePosixPath(name)
            if p.is_absolute() or '..' in p.parts or '\\' in name or name in source:raise ValueError('v12_share_source_path')
            source[name]=z.read(info)
    if 'scripts/v12_share_smoke.py' not in source:raise ValueError('v12_share_commit_smoke_before_final_build')
    spec_raw=source['configs/real_archvis62_v1.json']
    candidates=(spec_raw,spec_raw.replace(b'\r\n',b'\n'),spec_raw.replace(b'\r\n',b'\n').replace(b'\n',b'\r\n'))
    spec_raw=next((x for x in candidates if sha(x)==view.SPEC_SHA),None)
    if spec_raw is None:raise ValueError('v12_share_original_spec')
    spec=json.loads(spec_raw)
    # Existing strict loader verifies original archive SHA, report, dependencies and weights.
    models,pins,report,_=view._models(training_zip,spec,True,training_sha256=PARENT_SHA,run_id=view.FULL_RUN_ID)
    del models
    slim={k:report[k] for k in BINDING};slim.update(status=report['status'],cursor={'updates':report['cursor']['updates'],'best':report['cursor']['best']})
    for name in IMPLEMENTATIONS:
        raw=source[name];lf=raw.replace(b'\r\n',b'\n')
        if report['source_code_sha256'][name] not in (sha(raw),sha(lf),sha(lf.replace(b'\n',b'\r\n'))):raise ValueError('v12_share_archived_implementation:'+name)
    files={'source/'+n:b for n,b in source.items()}
    files['source/SOURCE_REVISION.txt']=(revision+'\n').encode()
    with zipfile.ZipFile(training_zip) as z:
        receipt_raw=z.read('recovery_receipt.json');receipt=json.loads(receipt_raw)
        for role,(name,expected) in EXPECTED.items():
            raw=z.read(name)
            if sha(raw)!=expected:raise ValueError('v12_share_original_model_bytes')
            files['models/'+name]=raw
    files.update({'metadata/run.json':canonical(slim),'metadata/spec.json':spec_raw,
        'metadata/original_checkpoint_receipt.json':receipt_raw,'metadata/checkpoint_pins.json':canonical({
            'parent_training_zip_sha256':PARENT_SHA,'original_run_sha256':receipt['files']['run.json']['sha256'],'models':pins}),
        'README_KO.md':readme(revision).encode('utf-8')})
    manifest={'schema':'v12_share_manifest_v1','source_revision':revision,'git_url':GIT_URL,'parent_training_zip_sha256':PARENT_SHA,
        'source_origin':'developer QA overlay, not a commit' if revision=='0'*40 else 'git archive',
        'files':{n:{'bytes':len(b),'sha256':sha(b)} for n,b in sorted(files.items())}}
    files['SHA256_MANIFEST.json']=canonical(manifest)
    with zipfile.ZipFile(output,'x') as z:
        for name,raw in sorted(files.items()):
            z.writestr(name,raw,compress_type=zipfile.ZIP_STORED if name.startswith('models/') else zipfile.ZIP_DEFLATED,compresslevel=1)
    with output.open('rb') as stream:pin=hashlib.file_digest(stream,'sha256').hexdigest()
    (output.with_suffix(output.suffix+'.sha256')).write_text(pin+'  '+output.name+'\n',encoding='utf-8')
    return {'zip':str(output),'bytes':output.stat().st_size,'sha256':pin,'revision':revision,'files':len(files),'model_pins':pins,'source_origin':manifest['source_origin']}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--revision',default='HEAD');parser.add_argument('--training-zip',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();revision,source=git_archive(args.revision)
    print(json.dumps(assemble(source,revision,args.training_zip,args.output),ensure_ascii=False))


if __name__=='__main__':main()
