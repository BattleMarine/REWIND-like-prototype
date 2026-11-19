# v2 첫 CPU 샤드 정식 복원과 CUDA 배치 읽기

CPU Colab에서 보관한 첫 샤드 128프레임을 기존 `rewind_v1.v2_retention.restore_cache`로 로컬 복원한다. 이어 학습 프런트엔드와 같은 `Cache.frame`과 `rewind_v1.run_stages.frame_batch`에서 공식 train의 첫 두 프레임을 읽고 이미지, 2D 관절 좌표, 2D 유효 mask를 CUDA로 왕복 전송한다. 고정 샤드 시험의 이전 결과와 달리, 이번 시험의 CUDA 배치는 표준 프런트엔드 입력 3종이다. 기존 샤드 로더는 3D·카메라를 포함한 일곱 target 배열의 shape·dtype·유한값도 검사한다. [이전 직접 읽기 시험](V2_GPU_CACHE_PROBE.md)

## 고정 입력과 복원 정책

- Drive 프로젝트의 `reports/cache/v2_cpu_cache_041837Z/`에서 기존 `catalog.json`과 첫 샤드 ZIP만 읽는다. 카탈로그 파일은 13,700바이트, SHA-256 `fe606a869c80599b742ed192fd99960faa1cf2c6242208c3bc1173eb0888ac26`이다. 샤드는 50,627,408바이트, SHA-256 `3b8220c7b9abec2bf7929d5281bd38587471e049729b8e0de79bbaf3d525f61e`다. 샤드 receipt SHA, 원래 binding, manifest·spec·공식 split pin과 원래 usage lineage를 검증한다.
- 명시적인 첫 샤드 읽기 전용 경로는 관측된 Torch `2.14.1+cpu` 생산본 → `2.14.1+cu130` 소비본만 허용한다. NumPy·Pillow, camera, preprocessing, manifest/spec 및 스키마를 포함한 나머지 binding은 일치해야 한다. 생산 소스 전체는 기존 pin에 묶인다. 소비 소스 전체와 이번 복원·캐시·학습 로더의 변경 파일은 별도로 고정한다. 복원되는 카탈로그와 샤드에는 생산 당시 binding이 보존된다.
- 학습 진입점은 기존의 완전한 캐시 및 strict binding 검증을 그대로 적용한다. 캐시 생성·재개, 학습 재개, v1 실험과 기존 배포 ZIP의 규칙도 기존 strict 경로를 따른다. 이번 실행에는 학습, 모델 생성, 원본 RGB·pose 읽기, 캐시 확장 및 Drive 캐시 갱신이 없다.

## 실행과 회수

1. Colab GPU 런타임에서 `v2_gpu_cache_restore.ipynb`를 연다. CPU 실행 당시와 같은 `v2_cpu_cache_ed98124.zip`을 이름 그대로 `/content`에 올린다. 누락 시 필요한 이름과 경로를 표시하고 Drive 마운트 전에 중단한다.
2. `DRIVE_PROJECT`를 확인한다. 기본값은 `/content/drive/MyDrive/REWIND-like prototype`이다. 기본 OFF인 `RUN_GPU_CACHE_RESTORE`를 켜고 전체 실행한 다음 Drive 마운트를 허용한다.
3. `reports/cache/v2_gpu_restore_041837Z/invocation-*.json`과 화면에 출력된 보고서 SHA-256을 회수한다. 게시 실패 시 로컬 `/content/rewind_v2_gpu_cache_restore/invocation-*/diagnostic.json`과 `worker.log`를 회수한다.

성공 기준은 최상위 `status=passed`, `worker.restore.status=local_partial`, `worker.restore.restored_frames=128`, `worker.cuda.all_exact=true`와 텐서 3종의 dtype·값 일치다. `worker.strict_binding_match`는 기존 strict binding 결과이고, `worker.allowed_binding`에 통과한 읽기 전용 CPU→CUDA 조건이 별도로 기록된다. GPU 모델은 제한하지 않는다. 런타임과 동기화 경과를 기록하며, 시간은 단일 배치 시험의 관찰값이다.

노트북 상한은 마운트·부트스트랩·복원·전송·보고서 종료 여유를 포함해 1,800초다. 기존 복원 함수가 카탈로그를 다시 검증하므로 성공한 원격 논리 입력은 카탈로그 2회와 샤드 1회, 총 50,654,808바이트다. Ledger의 읽기 계수에는 ZIP 로컬 검증도 포함된다. 시작 자유 공간은 512 MiB이고 가용 RAM 기준은 4 GiB다. Drive에는 192 KiB 이하 진단 JSON만 게시하고 게시본을 읽어 바이트를 확인한다. VM 자체가 종료되면 종료 처리가 실행되지 않을 수 있다.

## 빌드와 개발 검증

기존 production ZIP의 revision `ed981245b57d46535abb096ba6a280c8c333ead3`, SHA-256 `c73016fbdcc2daef64ae2ffeba418209f433a96e52c843c481b4b53f627ae493`를 그대로 사용한다. 노트북은 ZIP을 검증한 뒤 로컬 추출본에 LF 파일 5개를 적용한다. Notebook receipt는 CPU·GPU 선행 보고서, 기존 source ZIP, 각 overlay 파일, 계산된 소비 소스 identity 및 노트북의 SHA-256을 고정한다.

```powershell
python -m scripts.build_v2_gpu_cache_restore --source <v2_cpu_cache_ed98124.zip> --catalog <catalog.json> --cpu-report <invocation-hlok774l.json> --gpu-report <invocation-1xyc6uuv.json> --output <new-handoff-directory>
python -m unittest discover -s tests -p test_v2_gpu_cache_probe.py -v
```

로컬에서는 합성 샤드의 정식 복원, 학습과 공통 배치 경로의 호출, strict 복원 및 완전 캐시 gate, 허용 범위 밖의 Torch·NumPy·manifest 값 차이 거부, 고정 노트북 생성을 검증한다. 실제 GPU·Drive 확인은 고정 노트북의 사용 결과로 판단한다.
