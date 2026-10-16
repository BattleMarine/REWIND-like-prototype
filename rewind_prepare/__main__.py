"""원본을 수정하지 않는 real_pose_candidate_v1 준비 CLI."""
import argparse
import json
from pathlib import Path

from .converter import ROOT,convert,sha


def output_path_allowed(output,input_root,protected):
    output=output.resolve();input_root=input_root.resolve()
    if output.is_relative_to(input_root) or input_root.is_relative_to(output) or output.is_relative_to(ROOT/'data/raw'):
        raise ValueError('output_overlaps_raw_source')
    if any(p.resolve()==output or p.resolve().is_relative_to(output) for p in protected):
        raise ValueError('output_overlaps_metadata_source')
    if output.exists():raise ValueError('output_must_be_new')
    return output


def main(argv=None):
    p=argparse.ArgumentParser(description='읽기 전용 실제 pose 검사→학습차단candidate/report. 학습 release 기능 없음')
    p.add_argument('--input',type=Path,required=True,help='archive member 경로를 유지한 pose files root')
    p.add_argument('--extraction-manifest',type=Path,required=True)
    p.add_argument('--split-dir',type=Path,required=True)
    p.add_argument('--motion-map',type=Path,required=True,help='원본motion ID/미확정 선언과 근거를 명시한 JSON')
    p.add_argument('--spec',type=Path,required=True)
    p.add_argument('--calibration',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args(argv)
    output=None
    try:
        protected=[args.extraction_manifest,args.motion_map,args.spec,args.calibration,*[args.split_dir/(s+'.txt') for s in ('train','validation','test')]]
        output=output_path_allowed(args.output,args.input,protected)
        output.mkdir(parents=True)
        candidate,report=convert(args.input,args.extraction_manifest,args.split_dir,args.motion_map,args.spec,args.calibration)
        raw=(json.dumps(candidate,ensure_ascii=False,indent=2,allow_nan=False)+'\n').encode('utf-8')
        if len(raw)>candidate['preparation_spec']['limits']['max_output_bytes']:raise ValueError('prepare_output_byte_limit')
        report['candidate_file_sha256']=sha(raw)
        # 새 output에만 저장. 오류가 나면 candidate를 성공 산출물로 남기지 않는다.
        candidate_path=output/'candidate.json'
        candidate_path.write_bytes(raw)
        try:(output/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
        except Exception:
            candidate_path.unlink(missing_ok=True)
            raise
        print(json.dumps({'status':report['status'],'records':report['records'],'train_ready':False,'blockers':report['blockers'],'output':str(output)},ensure_ascii=True))
        return 0
    except (ValueError,OSError,KeyError,TypeError,OverflowError,RecursionError) as exc:
        if output is not None and output.exists():
            (output/'candidate.json').unlink(missing_ok=True)
            (output/'report.json').write_text(json.dumps({'status':'rejected','train_ready':False,'error_type':type(exc).__name__,'reason':str(exc)},ensure_ascii=True,indent=2)+'\n',encoding='utf-8')
        p.error(str(exc))


if __name__=='__main__':raise SystemExit(main())
