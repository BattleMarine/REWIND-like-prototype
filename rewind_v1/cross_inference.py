"""Fixed v1.2 continuous-clip inference and metrics. No optimizer or GT pose conditioning."""

from collections import defaultdict
import io
from pathlib import Path
from types import SimpleNamespace
import time

from PIL import Image, ImageDraw, ImageFont
import torch

from rewind_archive.contracts import digest
from rewind_m1.local_rgb import _regular_read
from rewind_m2.diffusion import DiffusionSchedule
from rewind_m2.streaming import keyed_frame_noise
from rewind_m3.models import BoundHand
from rewind_prepare.converter import sha, official_splits
from rewind_prepare.extracted import _local, _Budget, ExtractedLimits
from rewind_prepare.temporal_cache import _check_part
from rewind_prepare.temporal_selection import seal
from .cross_data import (Budget, MODEL_SHA, MIB, RESULT_BYTES, RUN_ID, LIMITS, assets, checked_selection,
                         read_sealed, source_identity, write_report, evaluation_context)
from .data import attach_predicted_hands, _hand_indices
from .frontend import predicted_features
from .run_stages import CascadeStages
from scripts import visualize_archvis_v1 as view


@torch.inference_mode()
def predict(models,images,camera,keys,config,device):
    features = torch.cat([predicted_features(models["frontend"],v.to(device)).cpu() for v in images.split(2)])
    return sample_features(models,features,camera,keys,config,device),features


@torch.inference_mode()
def sample_features(models,features,camera,keys,config,device):
    cfg = config["teacher"];scale=cfg["target_scale_mm"]
    spec = models["spec"]
    schedule = DiffusionSchedule(cfg["diffusion_steps"],cfg["beta_start"],cfg["beta_end"])
    features = features.reshape(1,len(keys),len(spec["joint_names"]),14).to(device)
    camera = camera[None].to(device);valid=torch.ones((1,len(keys)),dtype=torch.bool,device=device)
    def sample(role,model,offset):
        names = view._group_names(spec,role);indices=[spec["joint_names"].index(n) for n in names]
        noise = torch.stack([keyed_frame_noise(seq,frame,len(names),config["seed"]+offset) for seq,frame in keys])[None].to(device)
        return schedule.sample(model,features[...,indices,:].flatten(-2),camera,valid,noise,cfg["sampling_steps"])*scale
    body = sample("body",models["body"],100)
    upper = body[...,[spec["body_joint_names"].index(n) for n in spec["upper_joint_names"]],:]
    conditioned = BoundHand(models["hand"],upper/scale,valid[...,None].expand(upper.shape[:-1])).eval()
    local = sample("hand",conditioned,200)
    global_hand = attach_predicted_hands(body,local,spec)
    names = view._group_names(spec,"body")+view._group_names(spec,"hand")
    points=dict(zip(names,torch.cat((body,global_hand),dim=-2).unbind(-2)))
    result={"whole":torch.stack([points[n] for n in spec["joint_names"]],dim=-2)[0].cpu(),
            "body":body[0].cpu(),"hand_local":local[0].cpu(),"hand_global":global_hand[0].cpu()}
    if not all(torch.isfinite(v).all() for v in result.values()):raise ValueError("cross_nonfinite_prediction")
    return result


def metric_sums(prediction,features,data,spec):
    views=features.reshape(len(features),len(spec["joint_names"]),2,7).transpose(1,2)
    xy=(views[...,:2]+1)*1023/2
    target_xy=torch.stack((data["left_xy"],data["right_xy"]),1)
    xy_valid=torch.stack((data["left_xy_valid"],data["right_xy_valid"]),1)
    inside=xy_valid & (target_xy>=0).all(-1) & (target_xy<=1023).all(-1)
    body_idx=[spec["joint_names"].index(n) for n in spec["body_joint_names"]]
    fingers,wrists=_hand_indices(spec["joint_names"],spec)
    target,valid=data["target"],data["target_valid"]
    local_target=target[:,fingers]-target[:,wrists]
    masks={"body":valid[:,body_idx],"hand_global":valid[:,fingers],"hand_local":valid[:,fingers]&valid[:,wrists]}
    targets={"body":target[:,body_idx],"hand_global":target[:,fingers],"hand_local":local_target}
    errors={k:(prediction[k]-targets[k]).norm(dim=-1) for k in masks}
    errors.update(two_d_valid=(xy-target_xy).norm(dim=-1),two_d_inside=(xy-target_xy).norm(dim=-1))
    masks.update(two_d_valid=xy_valid,two_d_inside=inside)
    per_frame=[]
    for t in range(len(features)):
        row={}
        for name,error in errors.items():
            values=error[t][masks[name][t]]
            row[name]={"sum":float(values.double().sum()),"count":int(values.numel()),
                       "mean":float(values.double().mean()) if values.numel() else None,
                       "unit":"native_px" if name.startswith("two_d") else "mm"}
        for role,names,pred,truth,mask in (("body",spec["body_joint_names"],prediction["body"],targets["body"],masks["body"]),
                ("hand_local",view._group_names(spec,"hand"),prediction["hand_local"],local_target,masks["hand_local"])):
            edges=[(names.index(a),names.index(b)) for a,b in spec["edges"] if a in names and b in names]
            clip=SimpleNamespace(target_mm=truth[t:t+1][None],target_valid=mask[t:t+1][None],time_valid=torch.ones((1,1),dtype=torch.bool))
            row[role+"_shape_sums"]=CascadeStages._shape(pred[t:t+1][None],clip,edges)
        per_frame.append(row)
    return per_frame,xy,views[...,3]


def aggregate(rows):
    output={}
    for name in ("body","hand_local","hand_global","two_d_valid","two_d_inside"):
        total=sum(r[name]["sum"] for r in rows);count=sum(r[name]["count"] for r in rows)
        output[name]={"sum":total,"count":count,"mean":total/count if count else None,"unit":rows[0][name]["unit"]}
    for name in ("body_shape_sums","hand_local_shape_sums"):
        sums={k:sum(row[name][k] for row in rows) for k in rows[0][name]}
        output[name]={**sums,**{f"{label}_mean_{metric}_mm":sums[f"{label}_{metric}"]/sums[denom] if sums[denom] else None
                      for label in ("predicted","target") for metric,denom in (("bone","bones"),("spread","frames"))}}
    return output


def _canvas(raw,xy,gt2d,projected,spec,sequence,frame):
    size=320
    image=Image.new("RGB",(size*2+36,85+3*(size+29)+12),view.DARK)
    draw=ImageDraw.Draw(image);font=ImageFont.load_default(size=15)
    draw.text((12,7),"v1.2 continuous50 | PREDICTION / GT",font=ImageFont.load_default(size=20),fill="white")
    draw.text((12,34),f"{sequence.split('/')[-1][:55]} / frame {frame}",font=font,fill="white")
    draw.text((12,57),"Nominal25FPS | shared3D scale over entire clip",font=font,fill="white")
    for r,side in enumerate(("left","right")):
        for c,points in enumerate((xy[r],gt2d[r])):
            x,y=12+c*(size+12),86+r*(size+29)
            draw.text((x,y),f"{side} / {'PRED' if c==0 else 'GT'}",font=font,fill="white")
            image.paste(view._rgb_tile(raw[side],points,spec,size),(x,y+23))
    for c,points in enumerate(projected):
        x,y=12+c*(size+12),86+2*(size+29)
        draw.text((x,y),"device mm / "+("PRED" if c==0 else "GT"),font=font,fill="white")
        tile=Image.new("RGB",(size,size),"#202a38");view._draw_joints(tile,points,spec);image.paste(tile,(x,y+23))
    return image


class CappedOutput:
    def __init__(self,stream,limit):self.stream,self.remaining=stream,limit
    def write(self,value):
        if len(value)>self.remaining:raise ValueError("cross_render_result_limit")
        self.remaining-=len(value);return self.stream.write(value)
    def __getattr__(self,name):return getattr(self.stream,name)


def render(output,clip_root,entry,prediction,xy,data,spec,budget,available):
    target=data["target"].tolist();pred=prediction["whole"].tolist();valid=data["target_valid"]
    gt=[[(point if mask else None) for point,mask in zip(points,masks)] for points,masks in zip(target,valid.tolist())]
    # One projection transform for all50 frames and both GT/pred, never per-frame autozoom.
    projected=view._shared_3d_groups(([p for row in pred for p in row],[p for row in gt for p in row]),320)
    projected=[ [points[t*62:(t+1)*62] for t in range(50)] for points in projected]
    pictures=[];files=[]
    for t,row in enumerate(entry["frames"]):
        budget.check()
        raw={}
        for role,pin in row["files"].items():
            path=clip_root/pin["path"];payload=_regular_read(_local(path),8*MIB,pin["size_bytes"])
            if sha(payload)!=pin["sha256"]:raise ValueError("cross_render_source_changed")
            raw[role]=payload
        gt2d=[data["left_xy"][t].tolist(),data["right_xy"][t].tolist()]
        for side,name in enumerate(("left_xy_valid","right_xy_valid")):
            gt2d[side]=[p if ok else None for p,ok in zip(gt2d[side],data[name][t].tolist())]
        image=_canvas(raw,xy[t].tolist(),gt2d,(projected[0][t],projected[1][t]),spec,entry["sequence"],row["frame"])
        if t in (0,25,49):
            name=f"frame-{row['frame']:09d}.png"
            with (output/name).open("xb") as stream:image.save(CappedOutput(stream,available),format="PNG")
            size=(output/name).stat().st_size;available-=size;files.append(name)
        pictures.append(image.convert("P",palette=Image.Palette.ADAPTIVE,colors=128))
    path=output/"clip.gif"
    with path.open("xb") as stream:
        pictures[0].save(CappedOutput(stream,available),format="GIF",save_all=True,append_images=pictures[1:],duration=40,loop=0,optimize=False)
    files.append(path.name)
    for picture in pictures:picture.close()
    return [{"name":name,"bytes":(output/name).stat().st_size,"sha256":sha((output/name).read_bytes())} for name in files]


@torch.inference_mode()
def infer(inputs,selection,selection_sha,prepared,prepared_sha,model,output,device,*,budget=None,context=None):
    context=evaluation_context(context)
    budget=budget or Budget(LIMITS["infer_seconds"])
    plan=checked_selection(selection,selection_sha,context=context)
    prepared=Path(prepared);doc=read_sealed(prepared/"prepared.json","v12_cross_prepared_v1",prepared_sha,context=context)
    snapshot,spec,_,binding=assets(inputs)
    owners,_=official_splits(Path(inputs)/"split",spec,_Budget(ExtractedLimits(8*MIB,60)))
    if (plan["binding"]!=binding or any(owners.get(c["sequence"])!="validation" for c in plan["clips"])
            or doc["selection_sha256"]!=selection_sha or doc["binding"]!=binding or doc["model_zip_sha256"]!=MODEL_SHA
            or len(doc["clips"])!=len(plan["clips"])):raise ValueError("cross_prepared_binding")
    output=Path(output);output.mkdir()
    completed=[]
    try:
        models,pins,report,implementation=view._models(model,spec,True,training_sha256=MODEL_SHA,run_id=view.FULL_RUN_ID)
        models={k:v.to(device) for k,v in models.items()};models["spec"]=spec
        for number,entry in enumerate(doc["clips"]):
            budget.check();root=prepared/f"clip-{number:02d}";pin=entry["tensor"]
            if (entry["number"]!=number or entry["sequence"]!=plan["clips"][number]["sequence"]
                    or entry["environment"]!=plan["clips"][number]["environment"] or entry["split"]!="validation"
                    or [r["frame"] for r in entry["frames"]]!=[r["frame"] for r in plan["clips"][number]["frames"]]):
                raise ValueError("cross_prepared_clip_membership")
            _check_part(root/"clip.pt",pin,budget.check)
            data=torch.load(io.BytesIO(_regular_read(root/"clip.pt",32*MIB)),weights_only=True,map_location="cpu")
            if (data["images"].shape!=(50,2,3,128,128) or data["target"].shape!=(50,62,3)
                    or not all(torch.isfinite(v).all() for v in data.values())):raise ValueError("cross_prepared_tensor_contract")
            keys=[(entry["sequence"],r["frame"]) for r in entry["frames"]]
            prediction,features=predict(models,data["images"],data["camera"],keys,report["config"],device)
            budget.check()
            prefix=sample_features(models,features[:25],data["camera"][:25],keys[:25],report["config"],device)
            causal=max(float((prediction[k][:25]-prefix[k]).abs().max()) for k in prediction)
            if any(not torch.allclose(prediction[k][:25],prefix[k],atol=.01,rtol=1e-5) for k in prediction):
                raise ValueError("cross_causal_prefix_mismatch")
            frame_metrics,xy,confidence=metric_sums(prediction,features,data,spec)
            directory=output/f"clip-{number:02d}";directory.mkdir()
            used=sum(p.stat().st_size for p in output.rglob("*") if p.is_file())
            artifacts=render(directory,root,entry,prediction,xy,data,spec,budget,RESULT_BYTES-2*MIB-used)
            result=seal({"schema":"v12_cross_clip_predictions_v1","run_id":context.run_id,"sequence":entry["sequence"],
                "environment":entry["environment"],"split":"validation","overlap_proxy":plan["clips"][number]["overlap_proxy"],
                "keys":keys,"nominal_seconds":[k[1]/25 for k in keys],"predicted_2d_native":xy.tolist(),"confidence":confidence.tolist(),
                "prediction":{k:v.tolist() for k,v in prediction.items()},"gt_target_device_mm":data["target"].tolist(),
                "gt_target_valid":data["target_valid"].tolist(),"frame_metrics":frame_metrics,"aggregate":aggregate(frame_metrics),
                "causal_prefix_max_abs_mm":causal,"causal_tolerance":{"atol_mm":.01,"rtol":1e-5},
                "sequence_reset":"fresh per-clip inputs; no history/cache carried between clips", "artifacts":artifacts,
                "projection_scope":"single shared transform over all50frame GT+pred points"})
            write_report(directory/"predictions.json",result)
            if sum(p.stat().st_size for p in output.rglob("*") if p.is_file())>RESULT_BYTES-2*MIB:
                raise ValueError("cross_total_result_files_limit")
            completed.append(result)
        all_rows=[r for clip in completed for r in clip["frame_metrics"]]
        result=seal({"schema":"v12_cross_report_v1","run_id":context.run_id,"source_sha256":digest(context.source()),
            "selection_sha256":selection_sha,"prepared_sha256":prepared_sha,"model_zip_sha256":MODEL_SHA,
            "model_weights":pins,"inference_implementation_sha256":implementation,"status":plan["status"],
            "clips":[{"sequence":c["sequence"],"content_sha256":c["content_sha256"],"aggregate":c["aggregate"]} for c in completed],
            "aggregate":aggregate(all_rows),"by_environment":{env:aggregate([r for c in completed if c["environment"]==env for r in c["frame_metrics"]]) for env in set(c["environment"] for c in completed)},
            "device":str(device),"torch":str(torch.__version__),"precision":"fp32","ddim_steps":10,"seed":71,
            "device_name":torch.cuda.get_device_name(device) if device.type=="cuda" else "CPU",
            "device_vram_bytes":torch.cuda.get_device_properties(device).total_memory if device.type=="cuda" else None,
            "threads":torch.get_num_threads(),"prepared_logical_source_bytes":doc["source_bytes_read"],
            "noise_offsets":{"body":100,"hand":200},"hand_condition":"predicted_body_upper_and_wrist",
            "gt_usage":"metrics and geometry audit only; no GT pose conditioning","nominal_fps":25,
            "recorded_timestamps_verified":False,"stereo_sync_measured":False,"original_motion_independence_verified":False,
            "elapsed_seconds":time.monotonic()-budget.start,
            "gpu_peak_allocated_bytes":torch.cuda.max_memory_allocated(device) if device.type=="cuda" else None})
        write_report(output/"report.json",result)
        if sum(p.stat().st_size for p in output.rglob("*") if p.is_file())>RESULT_BYTES:
            raise ValueError("cross_total_result_files_limit")
        return result
    except BaseException as error:
        write_report(output/"failure.json",{"status":"failed","type":type(error).__name__,"reason":str(error)[:512],
                                           "completed_clip_sequences":[c["sequence"] for c in completed]})
        raise
