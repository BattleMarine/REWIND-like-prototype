"""명시 metadata로 새 bundle 생성 또는 완료본 로컬 무결성 확인."""
import argparse
import json
from .core import build_bundle, verify_bundle


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    build = sub.add_parser('build')
    for name in ('plan','queue','selection','spec','calibration','split-dir','motion-map','output','id'):
        build.add_argument('--'+name, required=True)
    sub.add_parser('verify').add_argument('--root', required=True)
    args = parser.parse_args(argv)
    if args.command == 'verify':
        result = verify_bundle(args.root)
    else:
        result = build_bundle(args.plan,args.queue,args.selection,args.spec,args.calibration,args.split_dir,
                              args.motion_map,args.output,manifest_id=args.id)
    print(json.dumps({'status':result['status'], 'train_ready':False, 'content_sha256':result['content_sha256']}))
    return result


if __name__ == '__main__':
    main()
