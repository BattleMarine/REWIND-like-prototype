# 첫 M0 구현 묶음 검토

검토일: 2026-10-13. 범위는 JSON 감사·공통 로더·개발용 관절 선택·HTML viewer이며 모델 구현과 학습은 포함하지 않는다.

검토 후 사용자가 데이터셋 재다운로드를 알렸다. 아래 실행 결과는 당시 샘플의 이력이며 현재 데이터 가용성을 보장하지 않는다. 새 데이터 확인은 재확보 이후에 진행한다.

## 담당과 판정

- 작업 경로: `REWIND-like prototype` 프로젝트 root.
- 개발 담당: `디벨로퍼`, 세션 ID `01a13299-7dcb-7080-8e8f-6af5ce67303d`. 세션 이름뿐 아니라 작업 디렉터리를 대조해 다른 프로젝트의 동명 세션과 구분했다.
- 검토 담당: `오케스트레이션`, 세션 ID `01a12df3-bbe2-73a1-8964-0b32aec9d391`.
- 판정: 코드와 실행 가능한 검증 범위에서 첫 사전 도구 구현을 수용한다. 검토한 범위에서 즉시 수정이 필요한 결함은 발견하지 못했다. 실제 브라우저 화면·조작 검증은 보류이며 M0 전체 완료가 아니다.

## 독립 확인 근거

| 확인 | 결과 |
| --- | --- |
| Python 단위·통합 테스트 | `python -m unittest discover -s tests -v`: 17개 통과 |
| 실제 sample smoke | 도구 안내의 명령 재실행: 18프레임·각 87관절, error 0 / warning 0, 원본 hash 불변 |
| 음수 2D와 정답 유효성 | 음수 좌표가 있는 좌우 2D 관측 109개 보존, 해당 GT 3D 유효성 보존 |
| 별도 변형 입력 검사 | 숫자 frame 2→10 정렬, 두 sequence 분리, gap 감지, 잘못된 2D와 유효한 3D의 mask 독립성 통과 |
| Viewer 로직 | `node tests/test_viewer.cjs outputs/m0_sample/viewer.html`: DOM/canvas mock 7개 검사 통과 |
| CLI 독립 실행 | 입력·설정·provenance를 동일하게 지정하고 출력만 `outputs/orchestration_review`로 분리: 보고서와 HTML 생성, error 0 / warning 0 |
| 정적 그림 | 개발 담당이 만든 `outputs/m0_sample/viewer_canvas_mock.png`를 읽어 관절·점선·축 표시 확인. 실제 브라우저 screenshot이 아님 |
| 문서·산출물 관리 | 상대 파일 링크 정상. data와 outputs의 Git 제외 확인. 기존 CRLF를 고려한 diff 공백 검사 통과 |

검토 실행에는 번들 Python과 Node 실행파일을 사용했다. 재현 명령과 옵션은 [M0 도구 안내](M0_JSON_TOOLS.md)에 있다. 검토 CLI가 기록한 `outputs/orchestration_review/manifest.json`의 코드 파일 hash를 완료 보고 후 현재 파일과 대조했으며 변경은 없었다. 이 출력 경로는 Git에서 제외되므로 새 checkout에서 재생성해야 한다.

## 남은 검증과 다음 작업 후보

- 브라우저 도구의 URL 정책이 `file://` 접근을 차단했다. 다른 브라우저·주소·실행 경로로 우회하지 않았다. 실제 CSS·DOM 이벤트·브라우저 호환성은 검증하지 못했으며 mock 테스트 통과로 대체해 주장하지 않는다.
- Body 22개·손 각각 21개와 연결은 개발용 가정이다. 공식 skeleton, 단위·축·회전, FPS/timestamp, calibration/재투영, split과 원본 motion 중복, 다른 sequence의 일관성 검증이 남아 있다.
- 다음 작업 후보는 공식 관절·좌표·시간·split 근거 확인과 로더 규격 확정이다. 단위와 train 소속을 확인한 뒤 GT 2D regression 진단으로 넘어가는 것이 적절하다.
- 이번 검토로 다음 구현·학습이나 데이터 확장을 지시하지 않았다. Commit·push·PR도 수행하지 않았다.
