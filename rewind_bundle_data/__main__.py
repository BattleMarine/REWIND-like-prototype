"""학습 없이 bundle T=1 입력 metadata와 선택한 bounded batch를 검사한다."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path

from .integrity import local
from .reader import PreparedBundleReader


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['inspect'])
    parser.add_argument('--root',required=True)
    parser.add_argument('--completion-sha256',required=True)
    parser.add_argument('--split',choices=['train','validation','test'],required=True)
    parser.add_argument('--diagnostic-only',action='store_true',required=True)
    parser.add_argument('--indices',type=int,nargs='+',help='생략하면 metadata만 검사. 최대2개')
    parser.add_argument('--include-rgb',action='store_true',help='indices를 지정한 경우 native PNG decode')
    parser.add_argument('--output',required=True,help='새 별도 디렉터리에 report.json 저장')
    args = parser.parse_args(argv)
    root = local(args.root)
    output = Path(args.output).absolute()
    # 없는 output 자체 대신 부모의 링크/존재를 확인한다.
    parent = local(output.parent); output = parent/output.name
    if output.is_symlink() or output.exists() or output.is_relative_to(root) or root.is_relative_to(output):
        raise ValueError('reader_new_separate_output_required')
    output.mkdir(exist_ok=False)
    report = {'schema_version':'prepared_bundle_t1_inspection_v1','status':'failed',
              'diagnostic_only':True,'train_ready':False,'training_run':False}
    try:
        if args.include_rgb and args.indices is None:
            raise ValueError('reader_rgb_requires_indices')
        reader = PreparedBundleReader(root,completion_sha256=args.completion_sha256,split=args.split,
                                      diagnostic_only=args.diagnostic_only)
        report.update(identity=reader.identity,limits=asdict(reader.limits),
                      verification_scope='whole bundle byte hashes and raw pose numeric audit; PNG decode only for requested RGB batch')
        if args.indices is not None:
            batch = reader.load_batch(args.indices,include_rgb=args.include_rgb)
            report['tensor_shapes'] = {k:list(v.shape) for k,v in batch.items() if hasattr(v,'shape')}
            report['requested_indices'] = args.indices
        report.update(status='passed',observed_peak_rss_bytes=reader._integrity.peak_rss)
    except Exception as exc:
        report.update(error_type=type(exc).__name__,reason=str(exc)[:256])
        raise
    finally:
        (output/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps({'status':report['status'],'train_ready':False,'records':len(reader)},ensure_ascii=False))
    return report


if __name__ == '__main__':
    main()
