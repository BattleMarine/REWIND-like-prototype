# v2 스테레오 캐시 128·256 비교

독립 CPU Colab에서 v2 입력 해상도 128과 256을 비교한다. 같은 입력을 한 번만 Drive에서 로컬에 준비한 뒤 양쪽을 같은 VM·소스·8-worker 읽기·Torch 단일 스레드·ZIP 저장 형식으로 변환한다. 현재 운영 캐시, writer 잠금, Drive 출력 경계와 시간·용량 원장은 별도의 경로를 계속 사용한다.

## 표본과 절차

고정 manifest의 기존 행 순서에서 시나리오별 공식 train 첫 128페어를 사전에 고정한다. 같은 샤드에 들어갈 수 있는 여러 sequence를 포함하며, 모델 결과와 영상 품질로 표본을 다시 선정하지 않는다. 표본 레코드 SHA는 노트북 receipt에 기록한다.

| 시나리오 | 페어 | 원본 바이트 |
| --- | ---: | ---: |
| ArchVisInterior_ArchVis_RT | 128 | 307,418,238 |
| CityPark_Showcase | 128 | 306,797,410 |
| DowntownWest_Demo_Environment | 128 | 352,260,473 |
| 합계 | 384 | 966,476,121 |

핀된 좌우 PNG·pose 1,152파일의 크기와 SHA를 확인하며 한 번에 로컬로 복사한다. 이후 입력은 여섯 변환에서 모두 그 로컬본을 사용하며 SHA를 확인한다. ArchVis와 Downtown의 실행 순서는 128→256, CityPark는 256→128이다. RGB float32·bilinear align_corners, 카메라 표현과 정답 mask는 공통이며 2D 좌표의 배율도 선택한 해상도에 맞춘다. 각 시나리오에서 독립 기존 v2 128 인코더를 실행해 `images.npy`와 `targets.pt`를 SHA로 대조한다. 이 별도 확인 시간은 A/B 변환 시간에서 제외된다. ArchVis 표본은 기존 운영 첫 샤드의 레코드 SHA와 일치하므로 회수 receipt에 있는 이미지·텐서 SHA와의 직접 일치 여부도 별도 보고한다.

## 실행과 자료 회수

1. 독립 CPU 런타임에서 `v2_resolution_ab_cpu.ipynb`를 열고 기존 `v2_cpu_cache_ed98124.zip`을 Colab `/content`에 올린다. 원래 source ZIP 안의 고정 manifest를 사용한다.
2. `DRIVE_PROJECT`를 확인하고 기본 OFF인 `RUN_RESOLUTION_AB`를 켠 뒤 한 번 실행한다. Drive 마운트를 승인한다. 이 시험의 외부 상한은 60분이며 작업 디스크 여유 6GiB, 가용 RAM 4GiB를 확인한다. 원본 stage는 4GiB 이하이고 시험 자체의 논리 읽기·쓰기는 각각 16GiB 이하로 제한한다.
3. `reports/cache/v2_resolution_ab_041837Z`의 결과 ZIP과 진단 JSON을 회수한다. 결과 ZIP에는 비교 CSV·JSON과 시나리오별 PNG 3장을 넣고 게시 후 SHA를 재검증한다. 진단 JSON의 `worker.status=passed`, 6개 비교행, 3개 128 참조 SHA 검증과 `worker.retention`을 확인한다. 실제 실험용 샤드 ZIP 6개는 VM의 `/content/rewind_v2_resolution_ab/invocation-*/scenario-*/`에 두고, 원본과 샤드 ZIP 본문은 보고서 ZIP에 포함하지 않는다.

`stage_wall_seconds`는 최초 Drive 읽기·파일별 SHA·로컬 저장까지 포함한다. 변환별 `read_wait_seconds`는 다음 순서의 입력을 기다린 벽시계이고, 겹쳐 실행된 worker 읽기 시간 합계는 별도다. `transform_seconds`, `serialize_and_hash_seconds`, `zip_seconds`, `variant_wall_seconds`와 실제 파일 bytes·dtype·shape를 함께 기록한다. JSON에서 시나리오별 256/128의 시간·이미지 bytes·ZIP bytes 비율도 확인할 수 있다. PNG는 각 표본의 같은 가운데 프레임과 같은 좌·우 시야를 사용하며 중심 50% 영역을 같은 크기로 확대한다.

두 Colab이 동시에 같은 Drive를 읽으면 초기 stage 속도에 경쟁이 반영될 수 있다. 로컬 A/B는 이러한 원격 읽기 비용을 해상도의 효과에 넣지 않는다. OS 페이지 캐시, 실행 순서, 작은 표본의 영향을 완전히 제거하지 못하며, 이 시험으로 전체 운영 캐싱 속도나 pose 정확도를 판정하지 않는다.

## 고정본 생성

```powershell
python -m scripts.build_v2_resolution_ab --source <v2_cpu_cache_ed98124.zip> --catalog <catalog.json> --cpu-report <invocation-hlok774l.json> --output <new-output>
python -m unittest discover -s tests -p test_v2_resolution_ab.py -v
```

## 실측 결과

2026-11-21 독립 Colab CPU 호출에서 384페어를 검증했다. 원본 966,476,121바이트를 로컬에 준비하는 데 66.39초, 참조 검사와 결과 보관을 포함한 전체 호출에 226.26초가 걸렸다.

| 시나리오 | 128 생성·포장 | 256 생성·포장 | 256/128 |
| --- | ---: | ---: | ---: |
| ArchVis | 11.86초 | 13.26초 | 1.118 |
| CityPark | 11.25초 | 12.39초 | 1.101 |
| Downtown | 11.47초 | 12.86초 | 1.121 |

각 샤드 ZIP은 50,615,696바이트에서 201,610,641바이트로 늘었다. ArchVis에서 128 이미지와 정답은 운영 첫 샤드와 일치했다. 표본 PNG를 육안으로 대조한 결과 256에서 ArchVis 바닥 무늬와 손목 주변, CityPark 팔과 상의 주름, Downtown 소매와 손의 경계가 더 선명했다. 관절 오차를 평가하려면 예측과 정답을 대조하는 후속 시험이 필요하다. 이 측정은 시나리오별 한 샤드의 로컬 생성·포장 비용이며 대용량 샤드의 Drive 게시·재검증 비용은 별도다.

원본 진단은 invocation-n1zws275.json (SHA-256 dddce2115a08b6fd6a15e72b3bcf0d094dfb6ed8121a9ec7c163f53534ba4d80), 결과 ZIP의 SHA-256은 7fb9e5c899fb6f6680e67dfd155649395b261224bbdf7aa8224cb3126b3d34ba다. Drive의 reports/cache/v2_resolution_ab_041837Z 및 로컬 outputs/v2_plan_inputs/resolution_ab_n1zws275에 보존했다.
