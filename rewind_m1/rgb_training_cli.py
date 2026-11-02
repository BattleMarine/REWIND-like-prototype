"""합성 RGB2D headless train/evaluate/export/cache. 기존 native inference API와 분리."""
import argparse
import hashlib
import io
import json
import time
from pathlib import Path

import torch

from .__main__ import save_report
from .data import InputContract
from .frontend import StereoHeatmapNet
from .rgb_frontend_artifacts import build_cache, export_weights, load_weights, preprocessing_version
from .rgb_shards import open_rgb, read_catalog, validate_catalog
from .rgb_training import FORMAT, RGBTrainer, validate_config
from .shard_cli import MAX_CHECKPOINT_BYTES, train
from .shards import _read, _require
from .training import source_identity


def read_checkpoint(path):
    raw=_read(path,MAX_CHECKPOINT_BYTES)
    state=torch.load(io.BytesIO(raw),map_location='cpu',weights_only=True)
    _require(isinstance(state,dict) and state.get('format_version')==FORMAT,'rgb_checkpoint_format')
    return state,hashlib.sha256(raw).hexdigest()


def export_checkpoint(checkpoint, identity, output):
    _require(checkpoint.get('format_version')==FORMAT and checkpoint.get('training_scope')=='synthetic_fixture_only'
             and checkpoint.get('full_training_ready') is False,'rgb_checkpoint_format')
    config=validate_config(checkpoint['config']); catalog=validate_catalog(checkpoint['catalog'])
    contract=InputContract.from_dict(catalog['contract'])
    _require(contract.data_kind=='synthetic_fixture','rgb_dataset_sample_release_not_implemented')
    _require(checkpoint['source']['code_sha256']==source_identity()['code_sha256'],'rgb_export_code_mismatch')
    with torch.random.fork_rng(devices=[]): model=StereoHeatmapNet(len(contract.joint_names))
    model.load_state_dict(checkpoint['model'],strict=True)
    return export_weights(model,config,catalog['contract'],catalog['content_sha256'],checkpoint['source'],
                          checkpoint['step'],identity,output)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='command',required=True)
    run=sub.add_parser('train')
    for option in ('config','catalog','output'): run.add_argument('--'+option,type=Path,required=True)
    target=run.add_mutually_exclusive_group(required=True)
    target.add_argument('--steps',type=int); target.add_argument('--epochs',type=int)
    run.add_argument('--resume',type=Path); run.add_argument('--device',default='cpu')
    run.add_argument('--max-attempts',type=int,default=1000); run.add_argument('--max-seconds',type=float,default=300.)
    run.add_argument('--checkpoint-every',type=int,default=1); run.add_argument('--evaluate-every',type=int,default=10)
    run.add_argument('--log-limit',type=int,default=64); run.add_argument('--export-weights',type=Path)
    evaluate=sub.add_parser('evaluate')
    for option in ('checkpoint','catalog','output'): evaluate.add_argument('--'+option,type=Path,required=True)
    evaluate.add_argument('--split',choices=('train','validation','test'),default='validation')
    evaluate.add_argument('--final-test',action='store_true'); evaluate.add_argument('--device',default='cpu')
    evaluate.add_argument('--max-seconds',type=float,default=300.)
    export=sub.add_parser('export')
    export.add_argument('--checkpoint',type=Path,required=True); export.add_argument('--output',type=Path,required=True)
    cache=sub.add_parser('cache')
    for option in ('checkpoint','weights','catalog','output'): cache.add_argument('--'+option,type=Path,required=True)
    cache.add_argument('--shard',type=int,required=True)
    cache.add_argument('--max-seconds',type=float,default=300.)
    args=parser.parse_args(argv)
    try:
        if args.command=='train':
            result=train(args,trainer_class=RGBTrainer,catalog_reader=read_catalog,checkpoint_reader=read_checkpoint,
                         exporter=export_checkpoint,run_schema='m1_rgb_shard_run_v1')
        elif args.command=='evaluate':
            _require(0 < args.max_seconds <= 86400,'rgb_cli_max_seconds')
            started=time.monotonic()
            def check():
                if time.monotonic()-started >= args.max_seconds: raise TimeoutError('rgb_evaluation_time_limit')
            _require(not args.output.exists(),'rgb_evaluation_output_exists')
            state,identity=read_checkpoint(args.checkpoint)
            trainer=RGBTrainer(state['config'],read_catalog(args.catalog),args.catalog.parent,args.device,check_budget=check)
            trainer.resume(state)
            result={'checkpoint_sha256':identity,'metrics':trainer.evaluate(args.split,args.final_test),
                    'catalog_sha256':trainer.catalog['content_sha256'],'source':trainer.source,'environment':trainer.environment}
            save_report(result,args.output)
        elif args.command=='export':
            state,identity=read_checkpoint(args.checkpoint)
            result={'checkpoint_sha256':identity,'state_sha256':export_checkpoint(state,identity,args.output)}
        else:
            _require(0 < args.max_seconds <= 86400,'rgb_cli_max_seconds')
            started=time.monotonic()
            def check():
                if time.monotonic()-started >= args.max_seconds: raise TimeoutError('rgb_cache_time_limit')
            _require(not args.output.exists(),'rgb_cache_output_exists')
            state,identity=read_checkpoint(args.checkpoint); catalog=read_catalog(args.catalog)
            _require(state['catalog']==catalog and type(args.shard) is int and 0 <= args.shard < len(catalog['shards']),
                     'rgb_cache_catalog_shard')
            model,artifact=load_weights(args.weights,expected_catalog_sha256=catalog['content_sha256'],
                expected_checkpoint_sha256=identity,expected_source={k:state['source'][k] for k in ('revision','code_sha256')},
                expected_contract=catalog['contract'],expected_preprocessing_version=preprocessing_version(state['config']['training_image_size']))
            loader=open_rgb(catalog,args.catalog.parent,args.shard)
            result=build_cache(loader,model,artifact,check_budget=check)
            raw=(json.dumps(result,separators=(',',':'),allow_nan=False)+'\n').encode('utf8')
            _require(len(raw) <= 4*1024**2,'rgb_cache_byte_limit')
            with args.output.open('xb') as stream: stream.write(raw)
        print(json.dumps(result,ensure_ascii=False,indent=2))
    except (ValueError,OSError,KeyError,TypeError,RuntimeError) as exc: parser.error(str(exc))
    return 0


if __name__=='__main__': raise SystemExit(main())
