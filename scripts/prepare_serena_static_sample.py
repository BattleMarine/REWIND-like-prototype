"""검토된 원본 ZIP3개에서 Serena 정적2sample을 오프라인 준비한다. 학습/원격 요청 없음."""
import argparse
import hashlib
import json
from pathlib import Path,PurePosixPath
import stat
import sys
import zipfile

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from rewind_archive.contracts import canonical,digest,safe_path
from rewind_archive.matching import verify_extraction,match_extractions,identify
from rewind_m1.rgb_manifest import build_manifest
from rewind_prepare.converter import convert
from rewind_prepare.rgb_review import create_review

SEQUENCE='ArchVisInterior_ArchVis_RT/Day/rp_serena_rigged_004_ue4/SKM_MenReadingGlasses_Shape_01/003/Getting_Up__1_'
PINS={'right':'3264743f26e254b87c86031cdd45071fa595f1173127baee59e3724d2fdadac2',
      'left':'61a64a6ef635dec46f8582708c425ef4af4956f43b078839ea635c2edf22697a',
      'pose':'e072992e64c907f3ff3e9bcfeb02b42bc18cbc41171821fecc50341011115544'}
REFERENCE_PREFIX='outputs/m0_next_evidence_20261015/developer/'


def sha(raw):return hashlib.sha256(raw).hexdigest()
def write(path,value):path.write_bytes(canonical(value)+b'\n')


def restore_zip(path,output,label,allow):
    if path.is_symlink() or path.stat().st_size>16*1024**2:raise ValueError('source_zip_limit_or_link')
    raw=path.read_bytes()
    if sha(raw)!=PINS[label]:raise ValueError('fixed_source_zip_sha256_mismatch')
    with zipfile.ZipFile(path) as archive:
        infos=archive.infolist()
        if len(infos)>128 or sum(i.file_size for i in infos)>16*1024**2:raise ValueError('source_zip_expansion_limit')
        names=[i.filename for i in infos]
        if len(names)!=len(set(names)):raise ValueError('duplicate_zip_members')
        if any(stat.S_ISLNK(i.external_attr>>16) for i in infos):raise ValueError('source_zip_symlink')
        if archive.testzip() is not None:raise ValueError('source_zip_crc')
        reports=[n for n in names if n.endswith('/extraction_manifest.json')]
        if len(reports)!=1:raise ValueError('single_extraction_report_required')
        prefix=PurePosixPath(reports[0]).parent.as_posix()
        report_raw=archive.read(reports[0]);report=json.loads(report_raw)
        if report.get('content_sha256')!=digest({k:v for k,v in report.items() if k!='content_sha256'}):raise ValueError('extraction_selfhash')
        if len(report['files'])!=2:raise ValueError('fixed_two_payloads_required')
        side='pose' if label=='pose' else label
        frames=set()
        destination=output/label;destination.mkdir()
        for row in report['files']:
            member=safe_path(row['member']);relative=safe_path(row['relative_path'])
            if relative!='files/'+member:raise ValueError('payload_relative_path')
            sequence,frame,view=identify(member,'pose' if label=='pose' else 'rgb')
            if sequence!=SEQUENCE or frame not in (1,10) or view!=side:raise ValueError('fixed_key_view_mismatch')
            frames.add(frame)
            if not 0<row['size_bytes']<=8*1024**2:raise ValueError('payload_size_limit')
            payload=archive.read(prefix+'/'+relative)
            if len(payload)!=row['size_bytes'] or sha(payload)!=row['sha256']:raise ValueError('payload_hash')
            target=destination/relative;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(payload)
        if frames!={1,10}:raise ValueError('fixed_frame_pair_required')
        (destination/'extraction_manifest.json').write_bytes(report_raw)
    checked=verify_extraction(destination,allow_code_change=allow)
    return {'path':str(path),'bytes':len(raw),'sha256':sha(raw),'report_file_sha256':sha(report_raw),
            'extraction_content_sha256':checked['content_sha256'],'producer_code_sha256':checked['code_sha256']}


def preview(review,output):
    from PIL import Image,ImageDraw
    canvas=Image.new('RGB',(2048,1120),'#20252b');draw=ImageDraw.Draw(canvas)
    for position,row in enumerate(review['frames']):
        for side_index,side in enumerate(('left','right')):
            view=row['views'][side]
            image=Image.open(output/'review'/view['image']).convert('RGB')
            image.load();marked=image.copy();points=ImageDraw.Draw(marked)
            for number,(point,valid) in enumerate(zip(view['coordinates'],view['valid_mask'])):
                if not valid:continue
                x,y=point
                if not (0<=x<image.width and 0<=y<image.height):continue
                color='#00ff88' if number<22 else '#ffcc33'
                points.ellipse((x-5,y-5,x+5,y+5),outline=color,width=3)
            for mode,panel in enumerate((image,marked)):
                x=(side_index*2+mode)*512;y=position*560
                draw.text((x+12,y+10),f'frame {row["frame_index"]} / {side} / '+('raw' if mode==0 else 'GT2D; no resizing of coordinates'),fill='white')
                canvas.paste(panel.resize((512,512)),(x,y+36))
    canvas.save(output/'representative_overlay.png')


def run(args):
    output=args.output.absolute();sources={k:getattr(args,k).resolve() for k in PINS}
    references=args.reference_zip.resolve()
    if output.exists() or any(p.is_relative_to(output.resolve()) for p in (*sources.values(),references)):
        raise ValueError('new_output_separate_from_sources_required')
    output.parent.mkdir(parents=True,exist_ok=True);output.mkdir()
    report={'schema':'serena_static_sample_v1','status':'started','train_ready':False,'training_run':False,
            'remote_requests':0,'new_network_bytes':0,'frames':[1,10],'contiguous_clip':False,
            'allow_code_change':args.allow_code_change,'sources':{}}
    try:
        inputs=output/'inputs';inputs.mkdir()
        for label,path in sources.items():report['sources'][label]=restore_zip(path,inputs,label,args.allow_code_change)
        matching=match_extractions([inputs/k for k in PINS],allow_code_change=args.allow_code_change)
        if matching['matched_triplet_count']!=2 or not matching['matching_valid']:raise ValueError('two_static_triplets_required')
        write(output/'matching.json',matching)
        spec_path=ROOT/'configs/real_archvis62_v1.json';spec=json.loads(spec_path.read_bytes())
        metadata=output/'metadata';metadata.mkdir();split_dir=metadata/'split';split_dir.mkdir()
        reference_rows={}
        with zipfile.ZipFile(references) as archive:
            for split in ('train','validation','test'):
                name=REFERENCE_PREFIX+'official_split/UnrealEgoData2/'+split+'.txt';data=archive.read(name)
                if len(data)>2*1024**2 or sha(data)!=spec['pinned_sources']['split_sha256'][split]:raise ValueError('pinned_split_mismatch')
                (split_dir/(split+'.txt')).write_bytes(data);reference_rows[name]=sha(data)
            name=REFERENCE_PREFIX+'official_calibration/fisheye_calib_unrealego_scaramuzza.json';data=archive.read(name)
            if len(data)>65536 or sha(data)!=spec['pinned_sources']['scaramuzza_sha256']:raise ValueError('pinned_calibration_mismatch')
            calibration=metadata/'scaramuzza.json';calibration.write_bytes(data);reference_rows[name]=sha(data)
        report['reference_members']=reference_rows
        split=[s for s in ('train','validation','test') if SEQUENCE in (split_dir/(s+'.txt')).read_text(encoding='utf-8').splitlines()]
        if split!=['train']:raise ValueError('serena_expected_official_train_only')
        selection=[{'sequence_id':SEQUENCE,'frames':[1,10],'split':'train','split_evidence':'SHA 대조한 공식 UnrealEgo2 train 목록. 두 frame은 한 sequence이며 validation으로 분리하지 않음.'}]
        motion={'schema_version':'explicit_motion_map_v1','sequences':{SEQUENCE:{'motion_id':'declared:Serena/003/Getting_Up__1_',
            'identity_status':'declared_unverified','evidence':'같은 archive sequence 경로로 묶은 선언. 원본 motion 식별과 다른 sequence와의 독립성은 미검증.'}}}
        write(metadata/'motion_map.json',motion);write(metadata/'rgb_selection.json',selection)
        rgb,rgb_report=build_manifest(inputs,[inputs/'left',inputs/'right'],selection,[1024,1024],output/'rgb',
            manifest_id='serena_static_1_10_v1',allow_code_change=args.allow_code_change)
        candidate,audit=convert(inputs/'pose/files',inputs/'pose/extraction_manifest.json',split_dir,
                              metadata/'motion_map.json',spec_path,calibration)
        candidate_dir=output/'pose';candidate_dir.mkdir();write(candidate_dir/'candidate.json',candidate);write(candidate_dir/'audit.json',audit)
        review=create_review(output/'rgb/manifest.json',inputs,candidate_dir/'candidate.json',
            [{'sequence_id':SEQUENCE,'frame_index':f} for f in (1,10)],output/'review')
        preview(review,output)
        report.update(status='prepared_static_samples',matched_triplet_count=2,official_split='train',
            numeric_validation=audit['numeric_validation'],blockers=audit['blockers'],
            visibility_semantics=review['visibility_semantics'],unit_status=audit['unit_status'],
            human_rgb_pose_alignment='pending',source_archives_unchanged=all(sha(p.read_bytes())==PINS[k] for k,p in sources.items()))
    except BaseException as exc:
        report.update(status='rejected',error_type=type(exc).__name__,error=str(exc))
        raise
    finally:write(output/'preparation_report.json',report)
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for key in (*PINS,'reference-zip','output'):parser.add_argument('--'+key,type=Path,required=True)
    parser.add_argument('--allow-code-change',action='store_true',help='검토된 고정 ZIP의 과거 producer hash만 허용; 내용 hash 검증은 유지')
    result=run(parser.parse_args());print(json.dumps(result,ensure_ascii=False,indent=2))
