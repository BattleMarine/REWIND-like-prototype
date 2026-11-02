"""로컬 또는 명시 API session으로만 준비. CLI는 원격 접근을 제공하지 않는다."""
import argparse
from pathlib import Path
import json

from rewind_archive.contracts import digest
from rewind_archive_index.preparation import _source
from .core import initialize_queue, make_plan, mark_consumed, read, run_queue, validate_plan


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command',required=True)
    create = sub.add_parser('plan')
    create.add_argument('--jobs',type=Path,required=True,help='manifest/source_base/targets/index_options/id를 포함한 명시 job list')
    create.add_argument('--base',type=Path,required=True)
    create.add_argument('--output',type=Path,required=True)
    init = sub.add_parser('init')
    init.add_argument('--plan',type=Path,required=True)
    init.add_argument('--root',type=Path,required=True)
    run = sub.add_parser('run')
    run.add_argument('--plan',type=Path,required=True)
    run.add_argument('--base',type=Path,required=True)
    run.add_argument('--root',type=Path,required=True)
    run.add_argument('--max-jobs',type=int,default=16)
    run.add_argument('--retry',action='store_true')
    inspect = sub.add_parser('inspect')
    inspect.add_argument('--plan',type=Path,required=True)
    inspect.add_argument('--root',type=Path,required=True)
    consumed = sub.add_parser('consumed')
    consumed.add_argument('--plan',type=Path,required=True)
    consumed.add_argument('--root',type=Path,required=True)
    consumed.add_argument('--job',required=True)
    consumed.add_argument('--consumer-sha256',required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == 'plan':
            jobs = read(args.jobs)
            for job in jobs:
                job['source_sha256'] = digest(_source(job['manifest'],args.base/job['source_base']))
            plan = make_plan(jobs)
            with args.output.open('x',encoding='utf8') as stream:
                stream.write(json.dumps(plan,ensure_ascii=False,indent=2)+'\n')
            result = {'plan_sha256':plan['content_sha256'],'jobs':len(jobs),'train_ready':False}
        else:
            plan = validate_plan(read(args.plan))
            if args.command == 'init':
                result = initialize_queue(plan,args.root)
            elif args.command == 'run':
                result = run_queue(plan,args.root,base=args.base,max_jobs=args.max_jobs,retry=args.retry)
            elif args.command == 'consumed':
                result = mark_consumed(plan,args.root,args.job,args.consumer_sha256)
            else:
                # 선언/장부만 표시. payload/전체 원본 무결성 검증 명령이 아니다.
                from .core import _state, writer
                with writer(args.root): result = _state(args.root,plan)
        print(json.dumps(result,ensure_ascii=False,indent=2))
    except (ValueError,OSError,KeyError,TypeError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
