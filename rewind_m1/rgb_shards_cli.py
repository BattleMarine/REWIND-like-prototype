"""로컬 RGB+GT chunk 카탈로그 생성/검사. UI·학습·원격 준비를 실행하지 않는다."""
import argparse
import json
from pathlib import Path

from .rgb_shards import DEFAULT_LIMITS, RGBShardDataset, build_catalog, read_catalog
from .shards import _read, _require


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command',required=True)
    build = sub.add_parser('build')
    build.add_argument('--specs',type=Path,required=True,help='id/pose/rgb_manifest/rgb_root의 명시 list')
    build.add_argument('--base',type=Path,required=True)
    build.add_argument('--limits',type=Path)
    build.add_argument('--output',type=Path,required=True,help='base 바로 아래의 새 파일')
    inspect = sub.add_parser('inspect')
    inspect.add_argument('--catalog',type=Path,required=True)
    inspect.add_argument('--metadata',action='store_true',help='모든 pose/RGB metadata와 파일 stat 검사')
    inspect.add_argument('--full-png',action='store_true',help='모든 PNG hash/decode까지 검사')
    args = parser.parse_args(argv)
    try:
        if args.command == 'build':
            _require(args.output.resolve().parent == args.base.resolve(), 'rgb_shard_catalog_output_base')
            specs = json.loads(_read(args.specs,DEFAULT_LIMITS['max_catalog_bytes']))
            limits = json.loads(_read(args.limits,4096)) if args.limits else None
            catalog = build_catalog(specs,args.base,limits)
            raw = (json.dumps(catalog,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n').encode('utf8')
            _require(len(raw) <= catalog['limits']['max_catalog_bytes'], 'rgb_shard_catalog_byte_limit')
            with args.output.open('xb') as stream: stream.write(raw)
            result = {'catalog_sha256':catalog['content_sha256'],'shards':len(catalog['shards']),
                      'metadata_verified':True,'full_png_verified':False,'train_ready':False,'temporal_ready':False}
        else:
            catalog = read_catalog(args.catalog)
            if args.metadata or args.full_png:
                result = RGBShardDataset(catalog,args.catalog.parent).verify_all(full_png=args.full_png)
            else:
                result = {'catalog_sha256':catalog['content_sha256'],'shards':len(catalog['shards']),
                    'scope':'catalog_declarations_only','metadata_verified':False,'full_png_verified':False,
                    'train_ready':False,'temporal_ready':False,'full_training_ready':False}
        print(json.dumps(result,ensure_ascii=False,indent=2))
    except (ValueError,OSError,KeyError,TypeError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == '__main__': raise SystemExit(main())
