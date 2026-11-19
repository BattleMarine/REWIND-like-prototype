# v2 첫 CPU 캐시의 CUDA 읽기 시험

CPU Colab에서 생성해 Drive에 보관한 첫 128프레임 샤드 중 공식 train 두 프레임을 기존 v2 샤드 로더로 읽고, CUDA로 왕복 전송해 shape·dtype·값 일치를 확인한다. 원본 RGB·pose는 읽지 않는다. GPU 모델은 제한하지 않으며 학습, 모델 생성, checkpoint와 캐시 변경은 실행하지 않는다. [CPU 샤드 실행 가이드](V2_CPU_CACHE_PILOT.md)

## 실행

1. Colab에서 GPU 런타임으로 `v2_gpu_cache_probe.ipynb`를 연다. CPU 실행에 사용한 그대로의 `v2_cpu_cache_ed98124.zip`을 Colab 왼쪽 파일 탭의 업로드로 `/content`에 올린다. ZIP이 없으면 노트북이 필요한 이름과 경로를 표시하고 Drive 마운트 전에 중단한다.
2. 설정 셀의 `DRIVE_PROJECT`를 확인한다. 기본값은 `/content/drive/MyDrive/REWIND-like prototype`이다. `RUN_GPU_CACHE_PROBE`를 켠 뒤 전체 실행하고 Drive 마운트를 허용한다.
3. 종료 후 `reports/cache/v2_gpu_cache_041837Z/invocation-*.json`과 셀에 출력된 보고서 SHA-256을 테스트 세션에 전달한다. 보고서 게시에 실패하면 로컬 `/content/rewind_v2_gpu_cache_probe/invocation-*/diagnostic.json`과 `worker.log`를 회수한다.

읽는 원격 입력은 `reports/cache/v2_cpu_cache_041837Z/`의 아래 두 파일이다. 두 파일에 대한 접근 권한이 필요하다.

| 파일 | 바이트 | SHA-256 |
| --- | ---: | --- |
| `catalog.json` | 13,700 | `fe606a869c80599b742ed192fd99960faa1cf2c6242208c3bc1173eb0888ac26` |
| `shard-00000-3b8220c7b9abec2bf7929d5281bd38587471e049729b8e0de79bbaf3d525f61e.zip` | 50,627,408 | `3b8220c7b9abec2bf7929d5281bd38587471e049729b8e0de79bbaf3d525f61e` |

## 판정

- `worker.strict_restore_binding`은 원래 CPU binding과 현재 런타임의 전체 binding을 비교한다. `match`는 이 비교의 일치다. `runtime_mismatch`는 Torch·NumPy·Pillow 버전 차이에 한해 기록하며, Torch 빌드 접미사 차이도 포함한다. 원본 SHA와 binding으로 고정된 이 샤드의 읽기 전용 진단은 계속한다. 그 외 binding 차이는 샤드 복사 전에 중단한다. 원래 정식 복원 함수와 전체 캐시 검증은 기존 기준을 그대로 사용한다. 이번 시험에서는 정식 복원 함수를 호출하지 않는다.
- `worker.cuda.all_exact=true`와 최상위 `status=passed`는 이미지, 2D·3D 좌표, 유효 mask, 카메라 인코딩과 float64 원래 카메라 행렬·이동 벡터가 모두 실제 CUDA 장치에 올라갔고, 동기화 후 CPU로 가져온 값과 dtype이 일치했다는 뜻이다. 원래 binding과 샤드 receipt, 배열·텐서 모양과 타입, 원본에서 기록된 supervision 유효 mask를 먼저 확인한다.
- 결과에는 Python, Torch와 빌드 CUDA, NumPy/Pillow, GPU 이름·VRAM·compute capability, 논리 입력 바이트, CUDA 전송 바이트와 동기화 경과를 기록한다. 단일 배치 시간은 학습 처리량이나 추론 레이턴시 지표로 사용하지 않는다. 두 프레임 결과의 적용 범위는 이 샤드와 해당 런타임에 한정한다.

노트북 상한은 마운트·부트스트랩·실행·보고서 종료 여유를 포함해 1,800초다. Drive 입력은 위 두 파일 합계 50,641,108바이트와 샤드 초과 읽기 확인 1바이트 이내다. 시작 시 로컬 자유 공간 512 MiB와 실행 전 가용 RAM 4 GiB를 확인하며, 새 Drive 산출물은 192 KiB 이하의 진단 JSON 한 개다. 실패·중단도 로컬에 먼저 기록한 뒤, 마운트가 가능하면 보고서 전용 폴더에 게시하고 게시본을 다시 읽는다. VM 자체가 종료되면 종료 처리가 실행되지 않을 수 있다.

## 고정 노트북 빌드

포터블 캐시 소스는 revision `ed981245b57d46535abb096ba6a280c8c333ead3`의 기존 ZIP이다. SHA-256은 `c73016fbdcc2daef64ae2ffeba418209f433a96e52c843c481b4b53f627ae493`이다. 노트북은 해당 ZIP의 해시와 revision을 검사하고, 별도로 해시가 기록된 진단 모듈을 로컬 추출본에 추가한다. 고정소스 원본 및 v2 cache 모듈은 바뀌지 않는다. 출력 receipt에 두 소스의 pin과 노트북 SHA를 남긴다.

```powershell
python -m scripts.build_v2_gpu_cache_probe --source <v2_cpu_cache_ed98124.zip> --catalog <catalog.json> --cpu-report <invocation-hlok774l.json> --output <new-handoff-folder>
python -m unittest discover -s tests -p test_v2_gpu_cache_probe.py -v
```

로컬 테스트는 고정 카탈로그·샤드 검증, 런타임 차이 분류, 기존 로더의 배열·tensor 읽기, 잘못된 ZIP 차단, CUDA 미지원 차단, 노트북 기본 OFF와 source 누락 분기를 확인한다. CUDA 전송은 실제 GPU Colab에서 위 기준으로 판정한다.
