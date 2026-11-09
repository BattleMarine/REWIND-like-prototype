# ArchVis 1,000페어 Colab 인계

인계물은 고정 소스 ZIP, 입력 ZIP, 두 ZIP의 파일 SHA와 소스 revision이 들어간 실행 노트북입니다. 소스 ZIP과 노트북은 최종 검토·커밋 후 생성합니다. 노트북은 소스 ZIP 밖에 두어 자기 자신의 해시를 포함하는 구조를 피합니다.

2026-11-09 로컬 보존본에서 공식 split과 Scaramuzza calibration의 바이트를 프로젝트 spec의 SHA와 대조했습니다. Drive 조사본은 기존 파일 SHA로 고정합니다. 조사에 포함된 train 286개와 validation 30개 sequence의 motion ID는 경로에 따른 선언입니다. 원본 motion 상태는 모두 `declared_unverified`입니다. 이름 정규화로 얻은 cross-split 검토 후보 13그룹은 `motion_review.json`에 기록했습니다. 이름 일치나 불일치만으로 원본 motion의 동일성이나 독립성을 판정할 수 없습니다. 기존 Dodging sequence의 frame 41 왼쪽 RGB 중복은 원본 조사본과 검토 목록에 보존하고 manifest 선택에서 제외합니다.

입력 ZIP은 로컬 보존 자료에서 아래 명령으로 만들 수 있습니다. 출력 디렉터리는 새 경로여야 합니다.

```text
python -m scripts.build_archvis_pilot_handoff inputs
  --metadata METADATA_DIRECTORY --inventory SEQUENCE_SUMMARY
  --spec-path configs/real_archvis62_v1.json --output NEW_INPUT_BUNDLE_DIRECTORY
```

2026-11-09 생성한 입력 ZIP은 161,064 bytes이며 SHA-256은 `9ccf95e91e785733418df088dddd87ffebb23833ebd140500d6aeb7e1f538e9b`입니다. 보존 metadata는 로컬 `outputs/rgb269_prepare_20261103/root_pose_audit/metadata/`, 조사본은 `outputs/archvis_drive_audit_20261109/sequence_summary.json`입니다. 인계용 입력 ZIP과 미확정 항목은 `outputs/archvis_pilot_handoff_20261109/release_inputs/`에 있습니다. 초기 working-tree spec 기반 ZIP은 사용하지 않습니다. Windows 줄바꿈 변환에 따른 SHA 차이를 피하기 위해 입력 receipt는 배포할 Git blob의 spec 바이트를 기준으로 생성했습니다. `inputs --spec-path`에는 실제 소스 ZIP과 같은 바이트의 spec을 전달해야 합니다.

소스 ZIP은 최종 커밋의 소스와 40자리 revision을 담은 `SOURCE_REVISION.txt`를 archive root에 넣습니다. 이후 아래 명령의 인자를 한 명령으로 전달해 ready 노트북을 생성합니다.

```text
python -m scripts.build_archvis_pilot_handoff ready
  --source-zip FIXED_SOURCE_ZIP --input-zip FIXED_INPUT_ZIP
  --revision FULL_COMMIT_SHA --output NEW_READY_NOTEBOOK
```

## Colab에서 확인할 값과 실행 순서

1. 테스트 세션의 파일 SHA를 확인한 두 ZIP을 Colab 파일 영역에 올립니다. 기본 위치는 `/content`입니다. Bootstrap은 ZIP SHA와 소스 revision을 확인하고, 위험 경로·중복·링크·파일 수·압축 전후 크기를 검사합니다. 작업 폴더가 새 경로일 때만 풉니다.
2. 사용자 Drive를 마운트하고 제한된 깊이·시간·entry 수로 RGB·pose category 폴더 후보를 찾습니다. 후보가 하나씩이면 표시된 경로를 확인하고, 여러 개거나 이름이 다른 경우 실제 category 상위 경로를 지정합니다. 각 경로 아래에 ArchVis 환경 폴더가 있어야 합니다.
3. Metadata 시간 상한을 정하고 경로를 확인합니다. Preflight는 inventory와 공식 split을 대조해 중복 키를 격리하고 seed 71로 train 1,000페어와 해당 공식 validation을 고정합니다. PNG·pose 본문은 이 단계에서 읽지 않습니다. 결과에서 경로·누락·중복·키 수와 로컬 파일 크기 기반 예상 읽기량을 확인합니다. 조사본과 적격 키에 차이가 표시되면 원인을 검토합니다.
4. 승인한 preflight SHA와 실제 준비 읽기·시간 상한을 입력합니다. 경로·파일 크기·선택·metadata pin을 다시 대조한 뒤 선택 파일의 SHA, PNG 디코딩, pose 수치와 좌표 검사를 수행합니다. 생성된 manifest 파일과 content SHA, validation 키 SHA를 `prepare_report.json`에 기록합니다. 실제 파일 내용과 대표 영상·정답 정합, 원본 motion은 이 근거와 별도로 검토합니다.
5. Colab 장치와 실행당 시간·시도·읽기·RSS·산출물 디스크·CUDA 상한을 설정합니다. 비교 계획은 생성된 manifest에 대해 두 loss의 최소 논리 읽기량과 설정 상한의 여유율을 표시합니다. 승인된 비교 계획 SHA를 입력하면 같은 manifest를 사용해 MSE와 CE를 순차 실행합니다. 두 config는 loss만 다릅니다. 별도 프로세스와 출력 폴더를 사용하고, 앞 실행이 중단되면 결과를 보존합니다. 협력적 시간 상한 외에 노트북 감독은 설정한 상한에 15초를 더한 시점에서 자식 프로세스를 종료할 수 있습니다.
6. 마지막 회수 셀은 manifest·준비/비교 계획·보고서·checkpoint·weights와 파일 SHA 영수증을 ZIP으로 내려받습니다. 실패한 단계의 보고서도 회수합니다. 데이터셋 파일은 회수 범위에서 제외합니다. 로컬 ZIP에 필요한 여유 공간을 별도로 확인합니다.

## 예상 읽기량의 기준

준비 예상치는 선택 파일의 stat 크기, 파일 길이 확인용 여유, 공식 split·spec·calibration·motion map·inventory의 입력 바이트를 사용합니다. 디렉터리 entry와 selected stat 수를 따로 표시합니다. 실제 준비에서 선택한 PNG 두 장을 디코딩하고 각 pose를 검사합니다. 입력 ZIP 검증, preflight 재대조, 회수 압축에는 별도의 로컬 읽기와 처리 시간이 듭니다. 마운트 캐시를 포함한 Drive의 실제 네트워크 전송량과 처리 시간은 실행 환경에서 측정합니다.

1 epoch 새 실행마다 train을 초기 평가·학습·최종 평가로 세 번 읽고, validation을 초기·최종 평가로 두 번 읽습니다. 이 파일량에 manifest·spec·config 읽기를 더합니다. 두 비교 실행의 합계를 계획서에 표시합니다. 같은 epoch 목표로 재개할 때는 저장된 초기 평가를 사용하므로 남은 train 학습과 최종 train·validation 읽기에 checkpoint·metadata 읽기를 더합니다. 준비본은 남은 train 전체를 다시 읽는 경우를 상한 예시로 표시하며, 재개 시 실제 남은 cursor와 예산을 테스트 세션에서 별도로 고정합니다.

정확한 마운트 경로, 원본 motion 검토 근거, 실제 선택 manifest와 validation 키 SHA, GPU·시간·RSS·디스크 상한은 런타임에서 확정합니다. 이 인계 묶음의 개발 검증은 로컬 fixture에 한정합니다. 학습·재개 및 전체 확대 계약은 [ArchVis RGB2D 실행 안내](ARCHVIS_DIRECT_RGB2D.md)를 따릅니다.

## 개발·배포 검증

Root는 관련21개 로컬 검사를 독립 실행했고, 배포 보완 후 handoff9개(신규2개 포함)를 다시 실행했다. 총23개 고유 검사 범위다. 실제 배포 과정에서 Windows working-tree와 Git blob의 spec 줄바꿈 차이를 발견해 source ZIP spec와 input receipt·공식 자산을 ready 생성 전에 대조하도록 보완했다. 수정한 입력 ZIP의 CRC와9개 자산 SHA를 독립 확인했다. 실제 Colab 인증·Drive payload·GPU 학습은 미실행이며 비교 subprocess는 로컬 모의 실행으로 검사했다.
