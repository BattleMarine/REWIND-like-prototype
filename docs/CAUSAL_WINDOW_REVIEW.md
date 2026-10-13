# 데이터 없는 causal window 단계 검토

검토일: 2026-10-13. 기준 revision: `9b80a53`. 구현은 이 프로젝트의 디벨로퍼 세션, 검토와 커밋·푸시는 오케스트레이션 세션이 담당했다.

## 검토 범위와 수정

- `windows.py`의 현재·과거 선택, field별 mask, padding, gap·sequence 경계, 오류 후 상태 보존과 snapshot 격리를 검토했다.
- 중간 검토에서 여러 sequence의 offline/streaming `boundary_reason` 불일치를 발견해 수정했다. 같은 ID의 별도 container와 빈 container는 streaming 호출자도 명시적으로 `reset()`해야 한다.
- 빈 입력에서도 잘못된 window 설정을 거부하도록 보완했다. 두 수정 사항은 회귀 테스트에 포함했다.
- 전체 sequence의 관절 union으로 발생하는 선택 밖 annotation issue가 과거 window를 바꾸지 않는지 확인했다. 구조·식별 오류와 annotation 누락을 구분하며, 원본과 검사 cache가 불일치하는 프레임은 history를 바꾸기 전에 거부한다.
- [데이터 계약](DATA_CONTRACT.md)의 공식 웹 안내와 저자 코드의 16관절 이름, UE1 논문의 25 FPS·cm 표기·분할 수치를 교차 확인했다. 로컬 REWIND PDF의 SHA-256도 문서와 일치한다. 이 확인으로 UE2 raw의 미확인 단위·시간을 확정하지 않았다.

## 오케스트레이션의 독립 실행

Python 3.12.14에서 다음 명령을 실행했다. 시스템 `python` 대신 설치된 Python 실행 파일을 사용했다.

```powershell
python -m unittest discover -s tests -p "test_*.py"
python -m rewind_data.window_fixture --window-size 3 --output outputs/orchestrator_window_review/window_fixture.json
```

- 기존 데이터 계약 17개와 신규 window 계약 23개, 총 40개 테스트 통과.
- CLI가 `synthetic_fixture` window 6개를 생성했으며 offline/streaming 결과가 일치했다.
- 별도 메모리 내 기준 계산으로 seed `20261013`, window 길이 1·3·20에서 총 750개를 비교했다. 무작위 gap·수동 reset·sequence 전환 뒤 frame 목록과 선택 순서별 좌표가 일치했고 중복 frame을 거부했다.
- 별도 검사에서 입력 좌표 변경이 이미 반환한 snapshot에 전파되지 않고, 잘못된 현재 프레임을 거부한 뒤 정상 프레임을 다시 받을 수 있음을 확인했다.

## 한계

이번 단계는 실제 데이터·Drive·이전 sample 산출물에 접근하지 않았다. 합성 출력은 Git에서 제외한다. 이 검토는 window 조립 계약의 검증이며 모델의 causality·정확도·속도 검증이 아니다. 실데이터 단위·회전·FPS·calibration·split·RGB 정합성과 M0/M1 전체 완료는 남아 있다. Viewer 코드를 변경하지 않았으며 실제 브라우저 검증을 추가로 수행하지 않았다.
