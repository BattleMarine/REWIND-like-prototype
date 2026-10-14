"""소형 2D heatmap 개발 모델과 예측 특징 캐시. 논문 EfficientNet 재현 아님."""

import argparse
import hashlib
import json
import re
from pathlib import Path

import torch
from torch import nn

from .data import content_hash, DataContractError, _observation
from rewind_data.audit import finite_number


class StereoHeatmapNet(nn.Module):
    def __init__(self, joints):
        super().__init__()
        self.joints = joints
        self.encoder = nn.Sequential(nn.Conv2d(3,16,3,stride=2,padding=1),nn.ReLU(),nn.Conv2d(16,joints,1))

    def forward(self, images):
        if images.ndim != 5 or images.shape[1:3] != (2,3) or min(images.shape[-2:]) < 4 or not torch.isfinite(images).all():
            raise DataContractError("Images: 유한한 B,2,3,H,W tensor 필요")
        b, _, _, h, w = images.shape
        logits = self.encoder(images.flatten(0,1))
        hh, ww = logits.shape[-2:]
        logits = logits.reshape(b,2,self.joints,hh,ww)
        probability = logits.flatten(-2).softmax(-1).reshape_as(logits)
        gx = torch.linspace(0,w-1,ww,device=images.device)
        gy = torch.linspace(0,h-1,hh,device=images.device)
        xy = torch.stack(((probability.sum(-2)*gx).sum(-1),(probability.sum(-1)*gy).sum(-1)),dim=-1)
        confidence = probability.flatten(-2).amax(-1)
        return logits, xy, confidence


def heatmap_loss(logits, gt_xy, valid, image_size, sigma_grid=1.5):
    if gt_xy.shape != logits.shape[:3]+(2,) or valid.shape != logits.shape[:3] or valid.dtype != torch.bool:
        raise DataContractError("2D supervision shape/bool mask 불일치")
    if not torch.isfinite(gt_xy[valid]).all(): raise DataContractError("유효 GT 2D에 비유한 값")
    width,height = image_size
    if any(type(v) is not int or v<2 for v in (width,height)) or not finite_number(sigma_grid) or sigma_grid<=0:
        raise DataContractError("영상 크기/양의 유한 sigma 오류")
    inside = valid & (gt_xy[...,0]>=0) & (gt_xy[...,0]<=width-1) & (gt_xy[...,1]>=0) & (gt_xy[...,1]<=height-1)
    if not inside.any(): raise DataContractError("Heatmap 격자에서 supervise할 유효 2D 관절 없음; 3D mask는 변경하지 않음")
    safe = torch.where(inside[...,None],gt_xy,torch.zeros_like(gt_xy))
    hh,ww = logits.shape[-2:]
    x = safe[...,0]*(ww-1)/(width-1)
    y = safe[...,1]*(hh-1)/(height-1)
    gx = torch.arange(ww,device=logits.device)
    gy = torch.arange(hh,device=logits.device)
    target = torch.exp(-((gx-x[...,None])[...,None,:].square()+(gy-y[...,None])[..., :,None].square())/(2*sigma_grid**2))
    target = target/target.sum((-2,-1),keepdim=True)
    probability = logits.flatten(-2).softmax(-1).reshape_as(logits)
    return (probability[inside]-target[inside]).square().mean(), inside


def weights_hash(model):
    digest=hashlib.sha256()
    for name,value in sorted(model.state_dict().items()):
        tensor=value.detach().cpu().contiguous()
        digest.update(name.encode());digest.update(str(tensor.dtype).encode());digest.update(str(tuple(tensor.shape)).encode())
        digest.update(bytes(tensor.view(torch.uint8).flatten().tolist()))
    return digest.hexdigest()


def save_frontend_weights(model, cache, path):
    """재사용할 weights와 입력 계약 저장. Optimizer/RNG 재개 파일이 아니다."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"format_version":"m1_frontend_weights_v1","weights_only":True,"model":model.state_dict(),
        "model_config":{"architecture":"small_cnn_v1","joints":model.joints,"hidden_channels":16,"stride":2},
        **{k:cache[k] for k in ("joint_names","image_sizes","frontend_version","frontend_checkpoint_sha256",
            "preprocessing_version","data_kind","manifest_hash")}}, path)


def prediction_cache(model, images, identities, joint_names, frontend_version, preprocessing_version, manifest_hash, data_kind):
    if data_kind not in ("synthetic_fixture","dataset_sample") or len(identities)!=len(images) or len(joint_names)!=model.joints:
        raise DataContractError("Cache provenance/식별/관절 불일치")
    if not frontend_version or not preprocessing_version or not isinstance(manifest_hash,str) or not re.fullmatch(r"[0-9a-f]{64}",manifest_hash) or len(set(joint_names))!=len(joint_names):
        raise DataContractError("Cache 버전/manifest/명시적 관절 순서 필요")
    records=[]; seen=set(); previous_mode=model.training
    model.eval()
    with torch.no_grad(): _,xy,conf=model(images)
    model.train(previous_mode)
    fingerprint=weights_hash(model)
    for i,identity in enumerate(identities):
        if not isinstance(identity,(tuple,list)) or len(identity)!=2: raise DataContractError("Cache identity=[sequence,frame] 필요")
        seq,frame=identity
        if not isinstance(seq,str) or not seq or type(frame) is not int or frame<0 or (seq,frame) in seen:
            raise DataContractError("Cache 중복/잘못된 sequence/frame")
        seen.add((seq,frame))
        prediction={"joint_names":list(joint_names),"frontend_version":frontend_version,"frontend_checkpoint_sha256":fingerprint,
            "preprocessing_version":preprocessing_version}
        for view,side in enumerate(("left","right")):
            prediction[side]={"coordinates":xy[i,view].tolist(),"valid_mask":[True]*len(joint_names),
                "confidence":conf[i,view].tolist(),"visibility":[None]*len(joint_names)}
        _observation(prediction,joint_names)
        records.append({"sequence_id":seq,"frame_index":frame,"prediction":prediction})
    body={"schema_version":"predicted_2d_cache_v1","data_kind":data_kind,"input_source":"predicted_2d",
        "manifest_hash":manifest_hash,"image_sizes":[[images.shape[-1],images.shape[-2]]]*2,"joint_names":list(joint_names),
        "frontend_version":frontend_version,"frontend_checkpoint_sha256":fingerprint,"preprocessing_version":preprocessing_version,"records":records}
    return {**body,"content_sha256":content_hash(body)}


def read_prediction_cache(doc, contract, manifest_hash):
    body={k:v for k,v in doc.items() if k!="content_sha256"}
    if doc.get("schema_version")!="predicted_2d_cache_v1" or content_hash(body)!=doc.get("content_sha256"):
        raise DataContractError("Cache schema/content hash 불일치")
    checks={"data_kind":contract.data_kind,"manifest_hash":manifest_hash,"joint_names":list(contract.joint_names),
        "image_sizes":[list(s) for s in contract.image_sizes],"frontend_version":contract.frontend_version,
        "frontend_checkpoint_sha256":contract.frontend_checkpoint_sha256,"preprocessing_version":contract.preprocessing_version,"input_source":"predicted_2d"}
    if contract.input_source!="predicted_2d" or any(doc.get(k)!=v for k,v in checks.items()):
        raise DataContractError("Cache와 입력 계약/manifest 불일치")
    result={}
    for record in doc["records"]:
        if not isinstance(record,dict) or not isinstance(record.get("sequence_id"),str) or not record["sequence_id"] or type(record.get("frame_index")) is not int or record["frame_index"]<0:
            raise DataContractError("Cache record sequence/frame type 오류")
        key=(record["sequence_id"],record["frame_index"])
        if key in result: raise DataContractError("Cache 중복 frame")
        for name in ("frontend_version","frontend_checkpoint_sha256","preprocessing_version"):
            if record["prediction"].get(name)!=checks[name]: raise DataContractError("Cache record 버전 불일치")
        result[key]=_observation(record["prediction"],contract.joint_names)
    return result


def main(argv=None):
    parser=argparse.ArgumentParser(description="합성 RGB의 소형 2D front-end gradient/cache 진단")
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args(argv)
    torch.manual_seed(71);torch.set_num_threads(1)
    images=torch.rand(4,2,3,32,32)
    xy=torch.tensor([[[[8.,9.],[20.,18.]]]*2]*4)
    valid=torch.ones(4,2,2,dtype=torch.bool)
    for b in range(4):
        for v in range(2):
            for p in xy[b,v]: images[b,v,:,int(p[1])-1:int(p[1])+2,int(p[0])-1:int(p[0])+2]=1.
    model=StereoHeatmapNet(2)
    optimizer=torch.optim.Adam(model.parameters(),lr=.02)
    initial=float(heatmap_loss(model(images)[0],xy,valid,(32,32))[0].detach())
    for _ in range(20):
        optimizer.zero_grad();loss,_=heatmap_loss(model(images)[0],xy,valid,(32,32));loss.backward()
        if any(not torch.isfinite(p.grad).all() for p in model.parameters()): raise ValueError("Frontend gradient 비유한 값")
        optimizer.step()
    identities=[("synthetic_rgb",i) for i in range(4)]
    manifest={"data_kind":"synthetic_fixture","fixture_seed":71,"identities":[list(v) for v in identities],
        "image_sizes":[[32,32]]*2,"image_tensor_sha256":hashlib.sha256(bytes(images.contiguous().view(torch.uint8).flatten().tolist())).hexdigest(),
        "preprocessing_version":"rgb_0_1_v1"}
    cache=prediction_cache(model,images,identities,["root","tip"],
        "small_cnn_fixture_v1","rgb_0_1_v1",content_hash(manifest),"synthetic_fixture")
    args.output.mkdir(parents=True,exist_ok=True)
    (args.output/"predictions.json").write_text(json.dumps(cache,indent=2)+"\n",encoding="utf-8")
    (args.output/"input_manifest.json").write_text(json.dumps(manifest,indent=2)+"\n",encoding="utf-8")
    save_frontend_weights(model,cache,args.output/"frontend_weights.pt")
    report={"data_kind":"synthetic_fixture","evaluation_kind":"synthetic_code_check","steps":20,
        "initial_heatmap_mse":initial,"final_heatmap_mse":float(heatmap_loss(model(images)[0],xy,valid,(32,32))[0].detach()),
        "cache_records":len(cache["records"]),"frontend_checkpoint_sha256":cache["frontend_checkpoint_sha256"]}
    (args.output/"report.json").write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(report,indent=2))


if __name__=="__main__": main()
