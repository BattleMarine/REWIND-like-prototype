# UnrealEgo2 추출본의 Drive 무결성 검사와 복구

[Colab 노트북](../notebooks/unrealego2_drive_upload.ipynb) 한 파일에 필요한 모듈과 공개 조각 목록을 담았다. Google Colab에서 **파일 → 노트북 업로드**로 열고 CPU 런타임에서 사용한다. 대상은 `UnrealEgoData2_train_val_rgb`, `UnrealEgoData2_train_val_pose`, `UnrealEgoData2_test_rgb`, `UnrealEgoData2_test_pose`다.

## 처음 실행

1. **설정**에서 `ROOT_FOLDER`에 네 범주를 직접 포함하는 Drive 상위 폴더의 URL을 넣는다. 같은 계정으로 인증하고 표시되는 대상 폴더를 확인한다.
2. **공식 시퀀스 폴더 조사**에서 `selected`는 선택한 환경, `all_four`는 네 범주 전체의 폴더를 대조한다. 폴더 존재·누락·중복 또는 타입 충돌을 시퀀스별로 기록한다. `DOWNLOAD_FOLDER_REPORT`로 결과만 먼저 받을 수 있다. 파일의 실제 내용은 4번에서 확인한다.
3. **조각 목록과 비용**에서 원본 마운트 경로를 확인한다. 기본은 `ARCHIVE_ROOT/<범주>/<환경>/<환경>.tar.gz000` 구조다. 공유 폴더 바로가기나 다른 배치를 쓰면 마운트 경로 또는 `PARTS_DIR_OVERRIDE`를 지정한다. 표에서 필요한 조각 수, 누락 번호와 읽을 용량을 확인한다.
4. **원본 검사 → Drive 대조**를 실행한다. 기본 예시는 `test_pose / ModularBuilding_Demo_Scene / Drunk_Walk` 한 시퀀스다. 다른 시퀀스는 `CATEGORY`, `ENVIRONMENT`, `SEQUENCES`를 함께 바꾼다. 폴더 조사 결과의 `sequence` 값을 복사할 수 있다. 한 환경 전체를 검사하려면 `SEQUENCES = None`으로 설정한다.
5. 먼저 기본 `MODE = "audit"`로 결과를 검토한다. 같은 시퀀스에 대한 수동 업로드를 잠시 멈출 수 있을 때만 쓰기 모드를 고르고 `OTHER_UPLOADS_PAUSED`를 켠다. 다른 사용자의 이동·이름 변경과 동시에 실행하면 충돌할 수 있다.
6. 마지막 셀에서 보고서 ZIP을 보관한다. 같은 런타임에서 쓰기가 중단됐다면 Drive 대조 셀을 다시 실행해 남은 작업을 확인한다. 새 런타임에서는 같은 범위의 원본 검사를 다시 수행한다.

## 검사 기준

| 단계 | 판정 근거 |
| --- | --- |
| 공식 범위 | 환경별 조각 수·이름은 공식 다운로드 스크립트, 시퀀스와 split은 공식 ZIP의 train/validation/test 목록과 대조 |
| 압축 원본 | 환경별 조각 순서와 완전성, gzip 종료와 CRC, tar 종료·안전한 경로와 중복 검사. 선택한 완전 파일의 경로·크기·SHA256 기록 |
| Drive 파일 | 같은 공식 경로의 일반 파일에 대해 원본과 Drive가 제공한 크기·MD5 대조 |
| 보류 항목 | 체크섬·크기 미제공, 동명 파일·폴더, 바로가기, 불완전 목록, 검사 이후 변경을 구분해 보고 |
| 추가 파일 | 조회 범위 안의 예상 밖 파일 경로를 보고. 원본 압축 조각·별도 파일을 보존 |

공식 목록의 단위는 시퀀스다. 각 PNG·JSON의 파일명과 기준 바이트는 선택한 분할 원본에서 얻는다. 따라서 결과가 보증하는 범위는 **검사한 로컬 원본과 Drive 추출본의 일치**다. 공식 배포본에서 프레임 하나가 빠졌는지, pose 좌표가 정확한지, 학습 split이 독립적인지는 별도의 검증 대상이다. [공식 데이터 구조·배포 스크립트](https://unrealego.mpi-inf.mpg.de/#download) · [공식 split ZIP](https://unrealego.mpi-inf.mpg.de/data/data_split/UnrealEgoData2.zip)

## 쓰기 방식과 비용

| 모드 | Drive에 남는 결과 | 전송 단위 |
| --- | --- | --- |
| `audit` | 읽기 전용 대조 보고서 | 폴더별 메타데이터 조회 |
| `loose` | 원래 경로의 개별 PNG·JSON | 없는 motion 폴더는 임시 트리에 파일을 한 번씩 올리고 목록·체크섬 검증 후 폴더 이름을 한 번 변경. 일부만 있는 폴더는 누락·불일치 파일만 복구 |
| `bundle` | 환경 아래 `__verified_sequence_bundles__`에 시퀀스별 ZIP | 같은 motion의 좌우 RGB 또는 pose를 한 ZIP으로 저장. 여러 전송 청크를 쓰더라도 Drive 파일 생성 단위는 ZIP 하나 |

압축 해제된 폴더를 학습에 쓸 때는 **loose**를 선택한다. RGB에서 motion 하나는 `motion/fisheye_final_image/camera_left·camera_right/*.png`를 포함한다. 개별 파일을 유지하면 파일당 미디어 전송이 필요하며, 시퀀스 단위 검증·게시로 반복 조회와 이름 변경을 줄인다. **bundle**은 묶음 보관용이다. ZIP을 올린 뒤에도 원래 경로가 비어 있으면 파일 대조 보고서에는 누락으로 표시된다.

불일치 파일은 정상본을 임시 업로드하고 크기·MD5를 확인한 뒤 기존 파일을 `원래이름.WRONG_UPLOAD-Drive파일ID`로 보존한다. 검증된 정상본을 원래 이름으로 배치하고 다시 읽어 확인한다. 다른 업로드나 중복 항목을 감지하면 보존본·임시본을 유지한 채 중단한다. 같은 실행 설정으로 재검사하면 검증된 임시 파일·폴더를 재사용한다.

분할 gzip은 시퀀스 하나만 선택해도 환경 하나의 압축 스트림을 끝까지 읽는다. 예를 들어 수십 GiB 환경에서 여러 시퀀스를 가져올 때는 같은 실행의 `SEQUENCES`에 모아 한 번만 읽는다. `MAX_SOURCE_GIB`은 압축 원본 읽기, `MAX_DECODED_GIB`는 전체 해제량, `MAX_STAGE_GIB`는 선택한 파일의 임시 공간이다. 묶음 업로드 중에는 시퀀스 ZIP 공간이 추가로 필요하다. 요청·쓰기·전송량과 실행 시간도 별도로 제한한다. 원본 준비는 Colab 런타임에 쌓이므로 다음 환경으로 넘어가기 전 남은 임시 공간을 확인한다.

## 검증과 재현

2026-11-06 기준 네 공식 스크립트의 60개 환경별 조각 수와 스크립트 SHA256을 대조했다. 공식 test pose의 작은 환경 원본 하나(20,320,491바이트)에서 19개 시퀀스·JSON 1,272개를 확인했다. 노트북 기본 시퀀스의 74개 파일은 별도 tar 판독 결과와 SHA256이 일치했고, 모의 Drive에서 복구 후 74개 모두 일치했다. 실제 사용자 Drive의 조사 범위는 상위 폴더와 일부 환경의 읽기 전용 조회까지다. 실제 Colab 인증·마운트와 서버 쓰기·복구 성능은 사용자 런타임에서 확인한다.

검토용 원본은 `scripts/unrealego2_upload_{source,drive,survey}.py`에 있다. 모듈 변경 후 `python scripts/build_unrealego2_upload_notebook.py`로 노트북을 다시 생성하고 `--check`로 내장본의 일치를 확인한다. 이 도구는 추출본 업로드 작업용이며 학습 시 gzip 인덱싱을 요구하지 않는다.
