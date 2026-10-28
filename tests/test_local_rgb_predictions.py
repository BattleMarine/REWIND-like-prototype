"""작은 합성 PNG의 lazy RGB→고정 모델→예측 cache 연결 검증."""
import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

try:
    import torch
    from PIL import Image
except ImportError:
    torch = None

if torch is not None:
    from rewind_archive.contracts import digest
    from rewind_m1.data import InputContract, DataContractError
    from rewind_m1.frontend import StereoHeatmapNet, read_prediction_cache, save_frontend_weights, weights_hash
    from rewind_m1.local_rgb import LocalStereoRGB
    import rewind_m1.local_rgb as local_rgb
    from rewind_m1.local_rgb_predictions import build_cache, load_weights, main


@unittest.skipUnless(torch is not None, "torch와 Pillow가 필요합니다")
class LocalPredictionTests(unittest.TestCase):
    def test_platform_file_signature(self):
        baseline=dict(st_dev=1,st_ino=2,st_size=3,st_mtime_ns=4,st_ctime_ns=5)
        def signature(platform, **changes):
            with patch.object(local_rgb.os,"name",platform):
                return local_rgb._signature(SimpleNamespace(**{**baseline,**changes}))
        self.assertEqual(signature("nt"),signature("nt",st_ctime_ns=6))
        self.assertNotEqual(signature("posix"),signature("posix",st_ctime_ns=6))
        for field in ("st_dev","st_ino","st_size","st_mtime_ns"):
            self.assertNotEqual(signature("nt"),signature("nt",**{field:baseline[field]+1}))

    def setUp(self):
        temp_root = Path(__file__).resolve().parents[1] / "outputs" / "test_tmp"
        temp_root.mkdir(parents=True, exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=temp_root)
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        rows=[]
        for i,split in enumerate(("train", "train", "validation")):
            seq=f"Fixture/Day/person/avatar/{i:03d}/Motion"
            views={}
            for side in ("left", "right"):
                member=f"{seq}/fisheye_final_image/camera_{side}/final_0.png"
                path=self.root/member
                path.parent.mkdir(parents=True,exist_ok=True)
                with Image.new("RGB",(8,6),(i*40,100,200)) as image: image.save(path)
                raw=path.read_bytes()
                views[side]={"path":member,"member":member,"size_bytes":len(raw),"sha256":hashlib.sha256(raw).hexdigest(),"image_size":[8,6],"source_id":"one","tar_complete":True}
            rows.append({"sequence_id":seq,"frame_index":0,"split":split,"split_evidence":"fixture","views":views})
        archive={"schema_version":"archive_manifest_v1","archive_id":"fixture","dataset_version":"fixture","kind":"rgb","source_scope":"synthetic_fixture","total_parts":1,"parts":[{"index":0,"name":"test000","size_bytes":1,"version":"1","source":{"type":"local","path":"fixture"},"identity":{"mtime_ns":0}}]}
        doc={"schema_version":"local_stereo_rgb_v1","id":"fixture","dataset_version":"fixture","data_kind":"synthetic_fixture","image_size":[8,6],"sources":{"one":{"archive_manifest":archive,"extraction_manifest_sha256":"0"*64}},"records":rows}
        doc["content_sha256"]=digest(doc)
        self.manifest=self.root/"manifest.json"
        self.manifest.write_text(json.dumps(doc),encoding="utf-8")
        self.loader=LocalStereoRGB(self.manifest,self.root)
        torch.manual_seed(7)
        self.model=StereoHeatmapNet(2)
        self.weights=self.root/"weights.pt"
        meta={"joint_names":["root","tip"],"image_sizes":[[8,6],[8,6]],"frontend_version":"fixture_v1","frontend_checkpoint_sha256":weights_hash(self.model),"preprocessing_version":"rgb_0_1_v1","data_kind":"synthetic_fixture","manifest_hash":"0"*64}
        save_frontend_weights(self.model,meta,self.weights)

    def test_prediction_reader_order_and_batch(self):
        model, checkpoint=load_weights(self.weights,self.loader,["root","tip"])
        model.train()
        before=weights_hash(model)
        a=build_cache(self.loader,model,checkpoint,["root","tip"],1)
        b=build_cache(self.loader,model,checkpoint,["root","tip"],4)
        self.assertTrue(model.training)
        self.assertEqual(before,weights_hash(model))
        self.assertEqual(a["record_splits"],["train","train","validation"])
        self.assertEqual(a["record_sources"][0],self.loader.metadata(0)["record"]["views"])
        self.assertEqual([x["sequence_id"] for x in a["records"]],[self.loader.metadata(i)["record"]["sequence_id"] for i in range(3)])
        model.eval()
        with torch.no_grad(): _,xy,conf=model(self.loader.load_batch([0])["images"])
        for side,index in (("left",0),("right",1)):
            self.assertTrue(torch.allclose(torch.tensor(a["records"][0]["prediction"][side]["coordinates"]),xy[0,index],atol=1e-6))
            self.assertTrue(torch.allclose(torch.tensor(a["records"][0]["prediction"][side]["confidence"]),conf[0,index],atol=1e-6))
        for ar,br in zip(a["records"],b["records"]):
            for side in ("left","right"):
                self.assertTrue(torch.allclose(torch.tensor(ar["prediction"][side]["coordinates"]),torch.tensor(br["prediction"][side]["coordinates"]),atol=1e-5))
        contract=InputContract.from_dict({"joint_names":["root","tip"],"distance_unit":"mm","coordinate_frame":"device_relative","image_sizes":[[8,6],[8,6]],"input_source":"predicted_2d","data_kind":"synthetic_fixture","frontend_version":"fixture_v1","frontend_checkpoint_sha256":before,"preprocessing_version":"rgb_0_1_v1"})
        self.assertEqual(len(read_prediction_cache(a,contract,self.loader.manifest_sha256)),3)

    def test_contract_limits_and_failure_mode(self):
        with self.assertRaises(DataContractError): load_weights(self.weights,self.loader,["tip","root"])
        checkpoint=torch.load(self.weights,weights_only=True)
        for key,value in (("data_kind","dataset_sample"),("preprocessing_version","wrong"),("image_sizes",[[9,6],[9,6]]),("frontend_checkpoint_sha256","1"*64)):
            changed=copy.deepcopy(checkpoint);changed[key]=value
            path=self.root/(key+".pt");torch.save(changed,path)
            with self.assertRaises(DataContractError): load_weights(path,self.loader,["root","tip"])
        model,checkpoint=load_weights(self.weights,self.loader,["root","tip"])
        for key,value in (("data_kind","dataset_sample"),("preprocessing_version","wrong"),("image_sizes",[[9,6],[9,6]]),("joint_names",["tip","root"])):
            altered=copy.deepcopy(checkpoint);altered[key]=value
            with self.assertRaises(DataContractError): build_cache(self.loader,model,altered,["root","tip"])
        model.train()
        with self.assertRaises(DataContractError): build_cache(self.loader,model,checkpoint,["root","tip"],max_records=2)
        with self.assertRaises(DataContractError): build_cache(self.loader,model,checkpoint,["root","tip"],max_bytes=100)
        with patch.object(self.loader,"load_batch",side_effect=RuntimeError("injected")):
            with self.assertRaises(RuntimeError): build_cache(self.loader,model,checkpoint,["root","tip"])
        self.assertTrue(model.training)
        self.assertEqual(weights_hash(model),checkpoint["frontend_checkpoint_sha256"])

    def test_cli_output_and_existing_file(self):
        output=self.root/"cache.json"
        argv=["--manifest",str(self.manifest),"--root",str(self.root),"--weights",str(self.weights),"--joints","root","tip","--output",str(output)]
        main(argv)
        before=output.read_bytes()
        with self.assertRaises(FileExistsError): main(argv)
        self.assertEqual(output.read_bytes(),before)

    def test_interrupted_output_is_removed(self):
        output=self.root/"interrupted.json"
        original=Path.open
        class Interrupted:
            def __init__(self, handle): self.handle=handle
            def __enter__(self): return self
            def __exit__(self,*args): self.handle.close()
            def write(self, raw):
                self.handle.write(raw[:10])
                raise KeyboardInterrupt()
        def open_interrupted(path,*args,**kwargs):
            handle=original(path,*args,**kwargs)
            return Interrupted(handle) if path==output and args and args[0]=="xb" else handle
        argv=["--manifest",str(self.manifest),"--root",str(self.root),"--weights",str(self.weights),"--joints","root","tip","--output",str(output)]
        with patch.object(Path,"open",open_interrupted):
            with self.assertRaises(KeyboardInterrupt): main(argv)
        self.assertFalse(output.exists())
