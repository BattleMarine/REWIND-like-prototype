"""로컬 persistent archive index의 bounded CLI. 원시 외부 예외는 출력하지 않는다."""
import argparse
import json
from pathlib import Path
import sqlite3

from rewind_archive.contracts import PreparationError
from . import ArchiveIndex, IndexLimits, IndexFailure, make_source


def main(argv=None):
    p=argparse.ArgumentParser(description='Local ordered split gzip/tar persistent index')
    p.add_argument('--limits',type=Path,help='IndexLimits 필드 일부를 담은64KiB 이하 JSON')
    sub=p.add_subparsers(dest='command',required=True)
    create=sub.add_parser('create');create.add_argument('index',type=Path)
    create.add_argument('--archive-id',required=True);create.add_argument('--kind',choices=['rgb','pose','mixed'],required=True)
    create.add_argument('--dataset-version',default='local_unverified');create.add_argument('--parts',nargs='+',type=Path,required=True)
    create.add_argument('--spacing-mib',type=int,default=1)
    scan=sub.add_parser('scan');scan.add_argument('index',type=Path);scan.add_argument('--max-new-members',type=int,default=128)
    scan.add_argument('--expected-generation',type=int);scan.add_argument('--defer-crc',action='store_true')
    for name in ('status','verify','audit-source'):
        q=sub.add_parser(name);q.add_argument('index',type=Path)
    q=sub.add_parser('lookup');q.add_argument('index',type=Path);q.add_argument('member')
    q=sub.add_parser('extract');q.add_argument('index',type=Path);q.add_argument('--members',nargs='+',required=True);q.add_argument('--output',type=Path,required=True)
    args=p.parse_args(argv)
    try:
        limits=IndexLimits()
        if args.limits:
            with args.limits.open('rb') as f:raw=f.read(65537)
            if len(raw)>65536:raise IndexFailure('limits_file_size')
            settings=json.loads(raw)
            if not isinstance(settings,dict):raise IndexFailure('limits_fields')
            limits=IndexLimits(**settings)
        index=ArchiveIndex(args.index,limits)
        if args.command=='create':
            result=index.create(make_source(args.parts,args.archive_id,args.kind,args.dataset_version),args.spacing_mib*1024**2)
        elif args.command=='scan':result=index.scan(args.max_new_members,args.expected_generation,not args.defer_crc)
        elif args.command=='status':result=index.status()
        elif args.command=='verify':result=index.verify()
        elif args.command=='audit-source':result=index.audit_source_sha()
        elif args.command=='lookup':result=index.lookup(args.member)
        else:result=index.extract(args.members,args.output)
        # source manifest/path 전체와 raw exception은 stdout에 노출하지 않는다.
        if 'stats' in result:result['stats'].pop('source_read_trace',None)
        print(json.dumps(result,ensure_ascii=False))
        return 0
    except (IndexFailure,PreparationError) as exc:
        print(json.dumps({'error':exc.reason}));return 2
    except (OSError,ValueError,TypeError,KeyError,sqlite3.Error):
        print(json.dumps({'error':'invalid_local_index_input_or_storage'}));return 2
    except Exception:
        print(json.dumps({'error':'local_index_operation_failed'}));return 2


if __name__=='__main__':raise SystemExit(main())
