"""Pose JSON의 native GT2D 점을 stereo RGB에 표시한다. 모델/학습/좌표 보정 없음."""
import argparse,hashlib,io,json,math,sys
from pathlib import Path
from PIL import Image,ImageDraw,ImageFont,__version__ as pillow_version

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from rewind_archive.contracts import canonical

BODY=(255,177,42);HAND=(31,223,240);BACKGROUND=(19,24,33)

def require(ok,reason):
    if not ok:raise ValueError(reason)

def raw_file(path,limit=16*1024**2):
    path=Path(path);require(path.is_file() and not path.is_symlink() and path.stat().st_size<=limit,'source_file_limit')
    raw=path.read_bytes()
    return raw,{'path':str(path.absolute()),'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}

def render_frame(pose_dir,rgb_dir,spec,sequence,frame,output):
    raw,pose_source=raw_file(Path(pose_dir)/f'frame_{frame}.json');doc=json.loads(raw)
    require(str(doc['frame'])==str(frame),'pose_frame_mismatch')
    names=spec['joint_names'];require(len(names)==62 and set(names)<=set(doc['joints']),'pose_spec_joints')
    body=set(spec['body_joint_names']);views={};images=[]
    for index,side in enumerate(('left','right')):
        raw,source=raw_file(Path(rgb_dir)/('camera_'+side)/f'final_{frame}.png')
        with Image.open(io.BytesIO(raw)) as image:
            require(image.format=='PNG' and image.mode=='RGB' and list(image.size)==spec['coordinates']['image_sizes'][index],'native_rgb_contract')
            image.load();canvas=image.copy()
        width,height=canvas.size;draw=ImageDraw.Draw(canvas);points=[]
        for name in names:
            xy=doc['joints'][name]['camera_'+side+'_pts2d']
            require(isinstance(xy,list) and len(xy)==2 and all(type(v) in (int,float) and math.isfinite(v) for v in xy),'gt_coordinate_finite')
            inside=0<=xy[0]<=width-1 and 0<=xy[1]<=height-1
            group='body' if name in body else 'hand'
            points.append({'joint':name,'native_xy':list(xy),'group':group,'drawn':inside,
                'hidden_reason':None if inside else 'offscreen','visibility':'unknown'})
            if inside:
                x,y=xy;radius=5
                draw.ellipse((x-radius,y-radius,x+radius,y+radius),fill=BODY if group=='body' else HAND,outline='black',width=2)
        views[side]={'source':source,'image_size':[width,height],'points':points,
            'drawn':sum(p['drawn'] for p in points),'offscreen':sum(not p['drawn'] for p in points)}
        images.append(canvas)
    width,height=images[0].size;require(images[1].size==(width,height),'stereo_size')
    font=ImageFont.load_default(size=27);small=ImageFont.load_default(size=20)
    def header(canvas,title,subtitle):
        d=ImageDraw.Draw(canvas);d.text((24,14),title,fill='white',font=font)
        d.text((24,52),'not model predictions | body: amber / hand: cyan | black point outlines',fill=(210,219,231),font=small)
        d.text((24,81),subtitle,fill=(180,193,210),font=small)
    panel=Image.new('RGB',(2*width,height+156),BACKGROUND)
    header(panel,'GT 2D ANNOTATIONS  |  '+sequence.split('/')[-1]+f'  |  frame {frame}',
        f'Native {width}x{height}; no clipping, coordinate adjustment or visibility inference')
    for index,side in enumerate(('left','right')):
        panel.paste(images[index],(index*width,124))
        d=ImageDraw.Draw(panel);v=views[side]
        d.text((index*width+16,height+129),f'{side.upper()} | drawn {v["drawn"]}/62 | offscreen {v["offscreen"]} | SHA {v["source"]["sha256"][:16]}',fill='white',font=small)
        view=Image.new('RGB',(width,height+156),BACKGROUND)
        header(view,'GT 2D ANNOTATIONS  |  '+side.upper()+f'  |  frame {frame}',sequence.split('/')[-1])
        view.paste(images[index],(0,124));ImageDraw.Draw(view).text((16,height+129),f'drawn {v["drawn"]}/62 | offscreen {v["offscreen"]} | SHA {v["source"]["sha256"][:16]}',fill='white',font=small)
        view.save(output/f'frame_{frame:03}_{side}.png')
    panel.save(output/f'frame_{frame:03}_panel.png')
    return {'frame_index':frame,'sequence_id':sequence,'sequence_provenance':'user supplied; no embedded sequence identity in pose JSON',
        'pose_source':pose_source,'raw_joints':len(doc['joints']),'selected_joints':62,'views':views,
        'coordinate_source':'joints.<joint>.camera_<side>_pts2d; direct native pixels',
        'coordinates_clamped':False,'occlusion_inferred':False,'model_predictions':False,
        'image_pixel_offset_in_output':[0,124],'native_image_resized':False}

def run(pose_dir,rgb_dir,output,*,sequence,frames):
    output=Path(output);require(not output.exists(),'new_output_required')
    require(frames and len(frames)<=10 and len(set(frames))==len(frames) and all(f>=0 for f in frames),'small_unique_frames')
    spec_path=ROOT/'configs/real_archvis62_v1.json';raw=spec_path.read_bytes();spec=json.loads(raw)
    output.mkdir(parents=True)
    records=[render_frame(pose_dir,rgb_dir,spec,sequence,frame,output) for frame in frames]
    for record in records:
        for source in [record['pose_source']]+[v['source'] for v in record['views'].values()]:
            require(hashlib.sha256(Path(source['path']).read_bytes()).hexdigest()==source['sha256'],'source_changed')
    report={'schema':'native_gt2d_overlay_v1','label':'GT 2D ANNOTATIONS / not model predictions','frames':records,
        'spec_id':spec['id'],'spec_sha256':hashlib.sha256(raw).hexdigest(),'joint_names':spec['joint_names'],
        'pillow':pillow_version,'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'source_files_unchanged':True,'model_used':False,'training_updates':0,'coordinate_scaling':None,
        'point_style':{'radius_native_px':5,'outline':'black','outline_width_px':2,'body_rgb':BODY,'hand_rgb':HAND}}
    (output/'report.json').write_bytes(canonical(report)+b'\n')
    return report

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('pose-dir','rgb-dir','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--sequence',required=True);p.add_argument('--frames',nargs='+',type=int,required=True);a=p.parse_args()
    r=run(a.pose_dir,a.rgb_dir,a.output,sequence=a.sequence,frames=a.frames)
    print(json.dumps({'output':str(a.output),'model_used':False,'frames':[{'frame':f['frame_index'],'drawn':{s:v['drawn'] for s,v in f['views'].items()}} for f in r['frames']]}))

if __name__=='__main__':main()
