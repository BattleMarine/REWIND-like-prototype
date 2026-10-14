"""Train-only 합성 front-end → 고정 예측 cache → regression 실행 연결."""

import argparse
import copy
import hashlib
import json
from pathlib import Path

import torch

from .data import InputContract, content_hash, load_document, synthetic_document
from .frontend import StereoHeatmapNet, heatmap_loss, prediction_cache, read_prediction_cache, weights_hash, save_frontend_weights
from .__main__ import main as regression_main


def prepare_fixture(frontend_steps=20, weights_output=None):
    if type(frontend_steps) is not int or frontend_steps<1: raise ValueError("양의 frontend_steps 필요")
    document=synthetic_document()
    records=document["records"]
    generator=torch.Generator().manual_seed(29)
    images=torch.rand(len(records),2,3,48,64,generator=generator)
    gt=torch.zeros(len(records),2,2,2)
    valid=torch.zeros(len(records),2,2,dtype=torch.bool)
    for i,r in enumerate(records):
        for v,side in enumerate(("left","right")):
            for j,name in enumerate(document["contract"]["joint_names"]):
                p=r["joints"][name][f"camera_{side}_pts2d"]
                gt[i,v,j]=torch.tensor(p);valid[i,v,j]=True
                x,y=round(p[0]),round(p[1])
                if 1<=x<63 and 1<=y<47: images[i,v,:,y-1:y+2,x-1:x+2]=1.
    train=torch.tensor([i for i,r in enumerate(records) if r["split"]=="train"])
    # Validation 이미지/label은 optimizer update에 사용하지 않는다.
    torch.manual_seed(71);torch.set_num_threads(1)
    model=StereoHeatmapNet(2)
    optimizer=torch.optim.Adam(model.parameters(),lr=.02)
    for _ in range(frontend_steps):
        optimizer.zero_grad(set_to_none=True)
        loss,_=heatmap_loss(model(images[train])[0],gt[train],valid[train],(64,48))
        loss.backward()
        if any(not torch.isfinite(p.grad).all() for p in model.parameters()): raise ValueError("Frontend gradient 비유한 값")
        optimizer.step()
    identities=[(r["sequence_id"],r["frame_index"]) for r in records]
    manifest={"data_kind":"synthetic_fixture","identities":[list(v) for v in identities],
        "image_sizes":[[64,48]]*2,"image_tensor_sha256":hashlib.sha256(bytes(images.contiguous().view(torch.uint8).flatten().tolist())).hexdigest(),
        "preprocessing_version":"rgb_0_1_v1","split_by_sequence":{r["sequence_id"]:r["split"] for r in records}}
    manifest_hash=content_hash(manifest)
    before=weights_hash(model)
    cache=prediction_cache(model,images,identities,document["contract"]["joint_names"],
        "small_cnn_train_only_fixture_v1","rgb_0_1_v1",manifest_hash,"synthetic_fixture")
    if before!=weights_hash(model): raise AssertionError("Cache 추론 중 frontend weights 변경")
    if weights_output is not None:
        save_frontend_weights(model,cache,weights_output)
    predicted=copy.deepcopy(document)
    predicted["contract"].update(input_source="predicted_2d",frontend_version=cache["frontend_version"],
        frontend_checkpoint_sha256=before,preprocessing_version=cache["preprocessing_version"])
    predicted["manifest"].update(id="synthetic_predicted_m1_v1",rgb_manifest_sha256=manifest_hash,
        prediction_cache_content_sha256=cache["content_sha256"])
    observations=read_prediction_cache(cache,InputContract.from_dict(predicted["contract"]),manifest_hash)
    for r in predicted["records"]:
        r["prediction"]=copy.deepcopy(observations[(r["sequence_id"],r["frame_index"])])
    original=load_document(document,3)[1];converted=load_document(predicted,3)[1]
    for split in original:
        if not torch.equal(original[split].target_valid,converted[split].target_valid) or not torch.equal(original[split].target_mm,converted[split].target_mm):
            raise AssertionError("캐시 연결이 GT 3D 또는 annotation mask를 변경")
    report={"data_kind":"synthetic_fixture","evaluation_kind":"synthetic_code_check","frontend_steps":frontend_steps,
        "frontend_training_splits":["train"],"train_images":len(train),"validation_images":sum(r["split"]=="validation" for r in records),
        "frontend_weights_fixed":True,"gt3d_and_mask_preserved":True,"frontend_checkpoint_sha256":before,"input_manifest_hash":manifest_hash}
    return predicted,cache,manifest,report


def main(argv=None):
    parser=argparse.ArgumentParser(description="합성 RGB 예측 cache를 regression에 연결하는 작은 CPU 진단")
    parser.add_argument("--config",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--frontend-steps",type=int,default=20)
    parser.add_argument("--steps",type=int,default=30)
    args=parser.parse_args(argv)
    if args.steps<1: parser.error("양의 steps 필요")
    if (args.output/"last.pt").exists(): parser.error("기존 checkpoint가 있습니다. 새 output 사용")
    document,cache,manifest,report=prepare_fixture(args.frontend_steps,args.output/"frontend_weights.pt")
    args.output.mkdir(parents=True,exist_ok=True)
    for name,doc in (("features.json",document),("predictions.json",cache),("input_manifest.json",manifest)):
        (args.output/name).write_text(json.dumps(doc,indent=2)+"\n",encoding="utf-8")
    report.update(frontend_weights_file="frontend_weights.pt",frontend_weights_only=True)
    (args.output/"frontend_report.json").write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")
    regression_main(["train","--config",str(args.config),"--dataset",str(args.output/"features.json"),
        "--output",str(args.output),"--steps",str(args.steps),"--device","cpu"])
    regression=json.loads((args.output/"report.json").read_text(encoding="utf-8"))
    print(json.dumps({"data_kind":report["data_kind"],"input_source":regression["validation"]["input_source"],
        "frontend_training_splits":report["frontend_training_splits"],"frontend_weights_fixed":True,"gt3d_and_mask_preserved":True,
        "step":regression["step"],"validation":regression["validation"]},indent=2))


if __name__=="__main__": main()
