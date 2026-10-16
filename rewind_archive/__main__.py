"""상한이 명시된 추출·재사용 검사·이름 대응 CLI."""

import argparse
import json
from pathlib import Path

from rewind_data.audit import read_json
from .contracts import Limits, PreparationError, canonical
from .extraction import prepare
from .matching import match_extractions, verify_extraction
from .pilot import run_pilot
from .planning import counterpart_plan


def read_bounded_json(path, maximum=128*1024):
    if path.stat().st_size > maximum:
        raise PreparationError("input_json_bytes_limit")
    with path.open("rb") as f:
        raw = f.read(maximum+1)
    if len(raw)>maximum:
        raise PreparationError("input_json_bytes_limit")
    return read_json(raw)


def main(argv=None):
    parser = argparse.ArgumentParser(description="분할 archive의 bounded 준비; 실제 학습 준비 판정 없음")
    sub = parser.add_subparsers(dest="command",required=True)
    for command in ("extract","pilot"):
        extract = sub.add_parser(command)
        extract.add_argument("--manifest",type=Path,required=True)
        extract.add_argument("--limits",type=Path,required=True)
        extract.add_argument("--include",action="append",required=True,help="완전한 archive member path glob, 반복 가능")
        extract.add_argument("--output",type=Path,required=True)
    match = sub.add_parser("match")
    match.add_argument("--extraction",type=Path,action="append",required=True)
    match.add_argument("--output",type=Path,required=True)
    plan = sub.add_parser("plan-counterparts")
    plan.add_argument("--extraction",type=Path,action="append",required=True)
    plan.add_argument("--max-frames",type=int,default=8)
    plan.add_argument("--output",type=Path,required=True)
    verify = sub.add_parser("verify")
    verify.add_argument("--extraction",type=Path,required=True)
    verify.add_argument("--manifest",type=Path,required=True,help="원본 manifest 동일성 검사; 네트워크는 읽지 않음")
    args = parser.parse_args(argv)
    try:
        if args.command in ("extract","pilot"):
            operation=prepare if args.command=="extract" else run_pilot
            report = operation(read_bounded_json(args.manifest),args.output,args.include,
                Limits.from_dict(read_bounded_json(args.limits)),base=args.manifest.parent)
            print(json.dumps({"stop_reason":report["stop_reason"],"complete_files":len(report["files"]) if args.command=="extract" else report["complete_files"],
                "inspection_candidates_usable":report["inspection_candidates_usable"],
                "whole_archive_integrity_verified":report["whole_archive_integrity_verified"],"train_ready":False,"stats":report["stats"]},indent=2))
            return 0 if report["inspection_candidates_usable"] else 1
        if args.command=="verify":
            report = verify_extraction(args.extraction,read_bounded_json(args.manifest))
            print(json.dumps({"verified_complete_file_hashes":len(report["files"]),"train_ready":False,"stop_reason":report["stop_reason"]},indent=2))
            return 0
        report = counterpart_plan(args.extraction,args.max_frames) if args.command=="plan-counterparts" else match_extractions(args.extraction)
        args.output.parent.mkdir(parents=True,exist_ok=True)
        with args.output.open("xb") as f:
            f.write(canonical(report)+b"\n")
        if args.command=="plan-counterparts":
            print(json.dumps(report,ensure_ascii=False,indent=2))
            return 0 if report["frames"] else 1
        print(json.dumps({"sequences":len(report["sequences"]),"matching_valid":report["matching_valid"],
            "matched_triplet_count":report["matched_triplet_count"],"incomplete_frame_count":report["incomplete_frame_count"],
            "observed_frame_count":report["observed_frame_count"],"issues":report["issues"],"train_ready":False},indent=2))
        return 0 if report["matching_valid"] else 1
    except (ValueError,OSError,KeyError) as exc:
        # 사용자 입력과 외부 exception의 credential을 출력하지 않는다.
        parser.error(exc.reason if isinstance(exc,PreparationError) else "invalid_input_or_output")


if __name__=="__main__":
    raise SystemExit(main())
