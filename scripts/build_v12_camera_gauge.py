"""Pinned six-clip gauge notebook, reusing the tested staged control/retention wrapper."""

import argparse
import json
from pathlib import Path
import subprocess
import sys
import tempfile

from rewind_archive.contracts import canonical
from rewind_prepare.converter import sha
from rewind_prepare.full_archvis import INPUT_ZIP,frozen_snapshot
from rewind_v1 import camera_gauge as gauge
from scripts.archvis_pilot_bootstrap import extract_pinned_zip
from scripts import build_v12_camera_control as base


NOTEBOOK='v12_camera_gauge_001.ipynb';MIB=1024**2


def notebook_document(pins):
    doc=base.notebook_document(pins)
    code=''.join(doc['cells'][1]['source'])
    expected=f'POLICY={base.POLICY!r}\nLIMITS={base.LIMITS!r}'
    if expected not in code:raise ValueError('gauge_shared_notebook_contract_changed')
    code=code.replace(expected,f'POLICY={gauge.POLICY!r}\nLIMITS={gauge.LIMITS!r}')
    code=code.replace('v12_camera_control_001','v12_camera_gauge_001').replace('rewind_v1.camera_control_cli','rewind_v1.camera_gauge_cli')
    code=code.replace("REFERENCE=UPLOADS/PINS['reference']['name']","REFERENCE=UPLOADS/PINS['reference']['name']\nCONTROL_REFERENCE=UPLOADS/PINS['control_reference']['name']")
    code=code.replace("'--reference',str(REFERENCE),'--model-source'","'--reference',str(REFERENCE),'--control-reference',str(CONTROL_REFERENCE),'--model-source'")
    code=code.replace('def run_control(device):','def run_gauge(device):')
    code=code[:code.index('\ndef display_saved():')]+'''
def display_saved():
    from IPython.display import display,Image as InlineImage,Markdown
    report=RUN/'inference/report.json'
    if not report.is_file():return
    if report.stat().st_size>32*1024**2:raise ValueError('gauge_display_report_limit')
    doc=json.loads(report.read_bytes())
    print('reference baseline match',doc['reference_match_all'],'same-run repeat stable',doc['repeat_stable_all'])
    table=RUN/'inference/comparison.md'
    if table.stat().st_size>1024**2:raise ValueError('gauge_display_table_limit')
    display(Markdown(table.read_text(encoding='utf-8')))
    total=0
    for number in range(6):
        folder=RUN/'inference'/f'clip-{number:02d}';info=folder/'report.json'
        if not info.is_file():continue
        if info.stat().st_size>8*1024**2:raise ValueError('gauge_display_clip_limit')
        clip=json.loads(info.read_bytes());print(clip['sequence'],'validation / nominal25FPS')
        print('shared device-coordinate bounds mm',clip['render_bounds_device_mm'])
        item=next(a for a in clip['artifacts'] if a['name']=='comparison.gif')
        path=folder/'comparison.gif'
        if path.is_symlink() or path.stat().st_size!=item['bytes']:raise ValueError('gauge_display_path_size')
        total+=item['bytes']
        if total>LIMITS['result_bytes']:raise ValueError('gauge_display_total_limit')
        raw=path.read_bytes()
        if hashlib.sha256(raw).hexdigest()!=item['sha256']:raise ValueError('gauge_display_hash')
        display(InlineImage(data=raw,format='gif'))
'''
    doc['cells'][1]['source']=code.splitlines(keepends=True)
    doc['cells'][0]['source']=['# V12-CAMERA-GAUGE-001\n','고정6clip300frame의무재학습world원점민감도진단입니다. RGB/2D/GT/R/noise를유지하며 baseline/Δ0repeat/first-frame midpoint/+100m canonical x만비교합니다. GT오차로Δ선택금지. source/input/도시reference/control reference ZIP을 /content에 준비합니다. 기존 CUDA torch/Pillow/NumPy 환경과 requirements-v12.txt를 사용합니다.\n']
    doc['cells'][2]['source']=['RUN_GAUGE 한 번으로 고정reference검증→exact목록→모델확인→SHA검증준비→전체variant추론→자동보관→저장표/6GIF표시를실행합니다. 재선정/학습/입력clipping없음. reference600초(각ZIP64MiB/unpack128MiB/100members),고정metadata1800초/100k/32MiB,모델600초/1GiB,준비3600초/원본합2GiB,모든variant+repeat+render합3600초. free disk8GiB/MemAvailable4GiB/RSS4GiB관측guard. 결과합/ZIP각256MiB. 정상시목록후/준비후/최종3회각새attempt에package600초+copy/readback600초추가. 각childtimeout+30초후kill/wait,게시실패자동재시도없음. VM소실에는finally보장없음. 이전GPU결과불일치와같은실행paired판정을구분하며repeat불안정시inconclusive입니다.\n']
    doc['cells'][3]['source']=["RUN_GAUGE=False\nDEVICE='cpu'  # Explicit cpu/cuda; any supported CUDA GPU.\nif RUN_GAUGE:\n    run_gauge(DEVICE)\n    display_saved()\n"]
    return doc


def build(args):
    paths=(args.source_zip,args.input_zip,args.reference_zip,args.control_reference_zip)
    if args.output.exists() or len({p.name for p in paths})!=4:raise ValueError('gauge_builder_new_output_names')
    with tempfile.TemporaryDirectory(dir=args.output.parent) as directory:
        root=Path(directory);source=root/'source';inputs=root/'inputs'
        source_pin=extract_pinned_zip(args.source_zip,args.source_sha256,source,max_zip_bytes=64*MIB,max_output_bytes=192*MIB,max_members=3000,revision=args.revision)
        input_pin=extract_pinned_zip(args.input_zip,INPUT_ZIP,inputs,max_zip_bytes=16*MIB,max_output_bytes=32*MIB,max_members=64)
        if json.loads((source/'configs/v12_camera_gauge_001.json').read_bytes())!=gauge.POLICY:raise ValueError('gauge_builder_policy_mismatch')
        frozen_snapshot(inputs);gauge.load_references(args.reference_zip,args.control_reference_zip,root/'reference')
        child=subprocess.run([sys.executable,'-m','rewind_v1.camera_gauge_cli','source'],cwd=source,timeout=60,capture_output=True,text=True,check=True)
        pins={'source':{**source_pin,'name':args.source_zip.name,'revision':args.revision},'inputs':{**input_pin,'name':args.input_zip.name},
            'reference':{'name':args.reference_zip.name,**gauge.POLICY['references']['city']},
            'control_reference':{'name':args.control_reference_zip.name,**gauge.POLICY['references']['control']},
            'source_code_sha256':json.loads(child.stdout)['source_sha256'],'model_sha256':gauge.POLICY['model_sha256'],'model_bytes':gauge.POLICY['model_bytes']}
    raw=(json.dumps(notebook_document(pins),ensure_ascii=False,indent=1)+'\n').encode();args.output.mkdir()
    (args.output/NOTEBOOK).write_bytes(raw)
    result={'schema':'v12_camera_gauge_handoff_v1','pins':pins,'policy':gauge.POLICY,'notebook':{'name':NOTEBOOK,'bytes':len(raw),'sha256':sha(raw)}}
    (args.output/'handoff_receipt.json').write_bytes(canonical(result)+b'\n');return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('source-zip','input-zip','reference-zip','control-reference-zip','output'):parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--source-sha256',required=True);parser.add_argument('--revision',required=True)
    print(canonical(build(parser.parse_args())).decode())


if __name__=='__main__':main()
