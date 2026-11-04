"""고정102 진단 frontend로 stereo PNG1쌍 추론·62관절 JSON/비교패널. 학습/GT 없음."""
import argparse,hashlib,io,json,platform,sys
from pathlib import Path
import torch
from torch.nn import functional as F
from PIL import Image,ImageDraw,ImageFont,__version__ as pillow_version

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from rewind_archive.contracts import canonical,digest
from rewind_m1.frontend import StereoHeatmapNet,weights_hash

WEIGHTS_SHA='00bf8a95e1ec2a30405f0016e63e60e97c321a7ea4faead514edcfb17efa2c15'
STATE_SHA='2a09070fb19485ccff7e43cc671ee274da1209c7b9170fc478b5b383a5de46f6'
MANIFEST_SHA='a3ea989256ba8a90d0785ca13048184c940c1cb14617cebba914c13cb98efc16'
VERSION='small_cnn_real_rgb102_64_diagnostic_v1'
PREPROCESS='rgb_0_1_resize64_bilinear_align_corners_v1'

def require(ok,reason):
    if not ok:raise ValueError(reason)

def read(path,maximum):
    path=Path(path);require(path.is_file() and not path.is_symlink() and path.stat().st_size<=maximum,'input_size_or_path')
    return path.read_bytes()

def load_model(weights,training_manifest):
    raw=read(weights,1024**2);require(hashlib.sha256(raw).hexdigest()==WEIGHTS_SHA,'fixed_rgb102_weights_sha')
    manifest=json.loads(read(training_manifest,1024**2));require(digest(manifest)==MANIFEST_SHA,'fixed_rgb102_manifest_sha')
    state=torch.load(io.BytesIO(raw),map_location='cpu',weights_only=True)
    expected={'format_version':'m1_frontend_weights_v1','weights_only':True,
        'model_config':{'architecture':'small_cnn_v1','joints':62,'hidden_channels':16,'stride':2},
        'joint_names':manifest['joint_names'],'image_sizes':[[64,64]]*2,'frontend_version':VERSION,
        'frontend_checkpoint_sha256':STATE_SHA,'preprocessing_version':PREPROCESS,
        'data_kind':'dataset_sample','manifest_hash':MANIFEST_SHA}
    require(all(state.get(k)==v for k,v in expected.items()),'rgb102_weights_contract')
    with torch.random.fork_rng(devices=[]):model=StereoHeatmapNet(62)
    model.load_state_dict(state['model'],strict=True);require(weights_hash(model)==STATE_SHA,'rgb102_weights_state')
    model.eval();return model,list(manifest['joint_names'])

def load_png(path):
    raw=read(path,16*1024**2)
    with Image.open(io.BytesIO(raw)) as image:
        require(image.format=='PNG' and image.mode=='RGB' and min(image.size)>=4 and image.width*image.height<=4*1024**2,'rgb_png_contract')
        image.load();rgb=image.copy()
    native=torch.frombuffer(bytearray(rgb.tobytes()),dtype=torch.uint8).reshape(rgb.height,rgb.width,3).permute(2,0,1).float().div_(255)
    small=F.interpolate(native[None],size=(64,64),mode='bilinear',align_corners=True,antialias=False)[0]
    return rgb,small,{'path':str(Path(path).absolute()),'sha256':hashlib.sha256(raw).hexdigest(),'bytes':len(raw),'image_size':[rgb.width,rgb.height]}

@torch.no_grad()
def predict(model,images,sizes):
    require(not model.training,'eval_required')
    _,xy,confidence=model(torch.stack(images)[None])
    require(torch.isfinite(xy).all() and torch.isfinite(confidence).all(),'finite_prediction_required')
    result={}
    for i,side in enumerate(('left','right')):
        network=xy[0,i].tolist();width,height=sizes[i]
        result[side]={'network_xy':network,'native_xy':[[p[0]*(width-1)/63,p[1]*(height-1)/63] for p in network],
            'confidence':confidence[0,i].tolist(),'valid_mask':[True]*62,'visibility':[None]*62}
    return result

def panel(images,predictions,names,spec,sequence,frame,sources,output):
    # 원본 native 좌표에 그린 뒤 표시용으로만 축소한다. 예측 이동·threshold·보정 없음.
    tiles=[];font=ImageFont.load_default(size=21);smallfont=ImageFont.load_default(size=17)
    for rgb,side in zip(images,('left','right')):
        overlay=rgb.copy();draw=ImageDraw.Draw(overlay);points=predictions[side]['native_xy']
        for a,b in spec['edges']:
            draw.line([tuple(points[names.index(a)]),tuple(points[names.index(b)])],fill=(255,190,20),width=2)
        for j,(x,y) in enumerate(points):
            color=(0,230,255) if j>=22 else (255,80,60)
            draw.ellipse((x-3,y-3,x+3,y+3),fill=color)
        tiles.append((rgb,overlay))
    canvas=Image.new('RGB',(2048,2304),(20,24,32));draw=ImageDraw.Draw(canvas)
    draw.text((24,15),'StereoHeatmapNet / '+VERSION,fill='white',font=font)
    draw.text((24,48),'DIAGNOSTIC - UNVALIDATED | no GT / no retraining / uncorrected model predictions',fill=(255,198,90),font=font)
    draw.text((24,80),f'{sequence} | frame {frame}',fill='white',font=smallfont)
    draw.text((24,105),'weights SHA '+WEIGHTS_SHA[:16]+' | L '+sources['left']['sha256'][:16]+' | R '+sources['right']['sha256'][:16],fill=(180,190,205),font=smallfont)
    for view,side in enumerate(('left','right')):
        for row,image in enumerate(tiles[view]):
            shown=image.copy();shown.thumbnail((1024,1024),Image.Resampling.LANCZOS)
            x=view*1024;y=156+row*1054;canvas.paste(shown,(x+(1024-shown.width)//2,y))
            draw.text((x+16,y-27),side.upper()+(' ORIGINAL' if row==0 else ' PREDICTED 2D (62 joints)'),fill='white',font=font)
        conf=predictions[side]['confidence']
        draw.text((view*1024+16,2264),f'Heatmap max softmax: min {min(conf):.6f} / mean {sum(conf)/62:.6f} / max {max(conf):.6f}',fill='white',font=smallfont)
    canvas.save(output)

def run(left,right,weights,training_manifest,output,*,sequence,frame):
    output=Path(output);require(not output.exists(),'new_output_required');require(frame>=0 and 0<len(sequence)<=512,'frame_identity')
    model,names=load_model(weights,training_manifest)
    spec_path=ROOT/'configs/real_archvis62_v1.json';spec=json.loads(read(spec_path,1024**2));require(spec['joint_names']==names,'joint_order')
    items=[load_png(p) for p in (left,right)]
    predictions=predict(model,[i[1] for i in items],[i[0].size for i in items])
    require(weights_hash(model)==STATE_SHA,'weights_changed_during_inference')
    result={'schema':'diagnostic_stereo_png_predictions_v1','diagnostic_only':True,'train_ready':False,'validated_accuracy':False,
        'model':'StereoHeatmapNet small_cnn_v1','frontend_version':VERSION,'weights_file_sha256':WEIGHTS_SHA,'weights_state_sha256':STATE_SHA,
        'training_manifest_sha256':MANIFEST_SHA,'sequence_id':sequence,'frame_index':frame,'joint_names':names,
        'sources':{s:i[2] for s,i in zip(('left','right'),items)},'predictions':predictions,
        'preprocessing':{'version':PREPROCESS,'network_size':[64,64],'resize_mode':'bilinear','align_corners':True,'antialias':False,
            'native_restore':'x*(native_width-1)/63, y*(native_height-1)/63'},
        'confidence_semantics':'maximum spatial softmax probability; not calibrated detection certainty',
        'timing':'unknown; stereo sync not verified','gt_used':False,'optimizer_updates':0,
        'skeleton_edges':spec['edges'],'skeleton_edges_status':'project assumption; native rig parent unverified',
        'skeleton_spec_sha256':hashlib.sha256(spec_path.read_bytes()).hexdigest(),
        'environment':{'python':platform.python_version(),'torch':torch.__version__,'pillow':pillow_version,'device':'cpu','precision':'float32'},
        'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    result['content_sha256']=digest(result)
    output.mkdir(parents=True)
    panel([i[0] for i in items],predictions,names,spec,sequence,frame,result['sources'],output/'overlay.png')
    (output/'predictions.json').write_bytes(canonical(result)+b'\n')
    return result

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('left','right','weights','training-manifest','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--sequence',required=True);p.add_argument('--frame',type=int,required=True);a=p.parse_args()
    torch.set_num_threads(1)
    result=run(a.left,a.right,a.weights,a.training_manifest,a.output,sequence=a.sequence,frame=a.frame)
    print(json.dumps({'output':str(a.output),'model':result['frontend_version'],'diagnostic_only':True,'optimizer_updates':0}))

if __name__=='__main__':main()
