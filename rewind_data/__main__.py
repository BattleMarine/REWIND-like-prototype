"""python -m rewind_data로 소량 JSON 감사와 viewer를 생성한다."""

import argparse
import json
import sys
from pathlib import Path

from .audit import audit_dataset, load_provenance
from .skeleton import load_skeleton
from .viewer import write_outputs


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="M0 pose JSON 사전 감사 (단위/좌표/calibration 검증은 별도)")
    parser.add_argument("--input", type=Path, required=True, help="sample root; frame 파일의 부모 디렉터리가 sequence")
    parser.add_argument("--skeleton", type=Path, required=True, help="명시적 관절 선택/연결 JSON")
    parser.add_argument("--output", type=Path, required=True, help="원본 root 밖의 생성 산출물 디렉터리")
    parser.add_argument("--dataset-version", default="unknown", help="사용자 선언 label; 데이터 버전 검증을 의미하지 않음")
    parser.add_argument("--provenance", type=Path, help="Drive 출처 및 관측한 폴더 inventory JSON (선택)")
    args = parser.parse_args(argv)
    source, output = args.input.resolve(), args.output.resolve()
    if output == source or output.is_relative_to(source):
        parser.error("산출물은 입력 root 밖에 저장해야 합니다.")
    try:
        result = audit_dataset(source, load_skeleton(args.skeleton), args.dataset_version, load_provenance(args.provenance))
        write_outputs(result, output)
    except (OSError, ValueError) as exc:
        print(f"설정/입출력 오류: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result.report["summary"], ensure_ascii=True))
    print(f"manifest.json / audit_report.json / viewer.html: {args.output}")
    return 1 if result.report["summary"]["error_count"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
