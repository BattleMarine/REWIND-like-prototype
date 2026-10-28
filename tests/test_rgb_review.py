"""합성 생성 candidate/RGB 계약의 검토 도구 검사. 실제 자료 정합성 증거 아님."""
import copy
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import unittest
from unittest.mock import patch
from types import SimpleNamespace

try:
    import torch
    from PIL import Image
except ImportError:
    torch = None

if torch is not None:
    import test_prepare as fixtures
    import rewind_prepare.rgb_review as review
    from rewind_archive.contracts import digest


@unittest.skipUnless(torch is not None,"torch와 Pillow가 필요합니다")
class RGBReviewTests(unittest.TestCase):
    def setUp(self):
        fixture = fixtures.PreparationTests("test_adopted_spec_and_single_mm_conversion")
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        fixture.add_frame(fixture.seq,1); fixture.reseal()
        candidate,_ = fixture.prepare()
        self.candidate = json.loads(json.dumps(candidate))
        self.root,self.seq = fixture.base,fixture.seq
        self.rgb = self.root/"rgb"; self.rgb.mkdir()
        rows = []
        for frame in range(2):
            views = {}
            for side in ("left","right"):
                member=f"{self.seq}/fisheye_final_image/camera_{side}/final_{frame}.png"
                path=self.rgb/member; path.parent.mkdir(parents=True,exist_ok=True)
                with Image.new("RGB",(1024,1024),(40+frame,80,100 if side == "left" else 160)) as image:
                    image.save(path)
                raw=path.read_bytes()
                views[side]={"path":member,"member":member,"size_bytes":len(raw),"sha256":hashlib.sha256(raw).hexdigest(),
                    "image_size":[1024,1024],"source_id":"rgb","tar_complete":True}
            rows.append({"sequence_id":self.seq,"frame_index":frame,"split":"unknown","split_evidence":"generated test declaration only","views":views})
        archive={"schema_version":"archive_manifest_v1","archive_id":"different_rgb_archive","dataset_version":"test_v1","kind":"rgb",
            "source_scope":"local_unverified","total_parts":1,"parts":[{"index":0,"name":"rgb000","size_bytes":1,"version":"fixture_v1",
                "source":{"type":"local","path":"synthetic_placeholder"},"identity":{"mtime_ns":0}}]}
        self.manifest={"schema_version":"local_stereo_rgb_v1","id":"review_fixture","dataset_version":"test_v1","data_kind":"dataset_sample",
            "image_size":[1024,1024],"sources":{"rgb":{"archive_manifest":archive,"extraction_manifest_sha256":"b"*64}},"records":rows}
        self.manifest_path,self.candidate_path=self.root/"rgb_manifest.json",self.root/"candidate.json"
        self.selection=[{"sequence_id":self.seq,"frame_index":0}]
        self.save()

    def save(self):
        for path, doc in ((self.manifest_path,self.manifest),(self.candidate_path,self.candidate)):
            doc["content_sha256"]=digest({k:v for k,v in doc.items() if k != "content_sha256"})
            path.write_text(json.dumps(doc),encoding="utf-8")

    def run_review(self,name,selection=None):
        return review.create_review(self.manifest_path,self.rgb,self.candidate_path,self.selection if selection is None else selection,self.root/name)

    def test_ready_raw_pixels_masks_unknown_and_originals(self):
        field=self.candidate["records"][0]["fields"]["camera_left_pts2d"]
        field["coordinates"][0]=[-5.,2000.]
        field["coordinates"][1]=None; field["valid_mask"][1]=False
        self.save()
        originals={p:hashlib.sha256(p.read_bytes()).hexdigest() for p in [self.candidate_path,self.manifest_path,*self.rgb.rglob("*.png")]}
        result=self.run_review("ready")
        self.assertEqual(result["status"],"review_ready")
        self.assertFalse(result["train_ready"])
        self.assertFalse(result["rgb_pose_alignment_confirmed"])
        self.assertEqual(result["candidate_training_eligibility"],self.candidate["training_eligibility"])
        html=(self.root/"ready/review.html").read_text(encoding="utf-8")
        self.assertEqual(json.loads(re.search(r'id="reviewData">(.*?)</script>',html,re.S).group(1)),result)
        frame=result["frames"][0]; view=frame["views"]["left"]
        self.assertEqual(frame["split_status"],"unknown")
        self.assertEqual(view["coordinates"],field["coordinates"])
        self.assertEqual(view["valid_mask"],field["valid_mask"])
        self.assertTrue(view["outside_image"][0]); self.assertIsNone(view["coordinates"][1])
        self.assertEqual(result["unselected"]["rgb"],[[self.seq,1]])
        self.assertEqual(result["unselected"]["candidate"],[[self.seq,1]])
        for side in ("left","right"):
            copied=(self.root/"ready"/frame["views"][side]["image"]).read_bytes()
            self.assertEqual(hashlib.sha256(copied).hexdigest(),frame["views"][side]["source"]["sha256"])
        self.assertEqual(originals,{p:hashlib.sha256(p.read_bytes()).hexdigest() for p in originals})
        self.assertIn("rewind_prepare/rgb_review_template.html",result["source"]["code_sha256"])
        self.assertEqual(set(result["source"]["code_hash_scope"]),set(result["source"]["code_sha256"]))
        before=(self.root/"ready/report.json").read_bytes()
        with self.assertRaises(FileExistsError): self.run_review("ready")
        self.assertEqual((self.root/"ready/report.json").read_bytes(),before)
        selection_path=self.root/"selection.json"
        selection_path.write_text(json.dumps(self.selection),encoding="utf-8")
        review.main(["--rgb-manifest",str(self.manifest_path),"--rgb-root",str(self.rgb),"--candidate",str(self.candidate_path),
            "--selection",str(selection_path),"--output",str(self.root/"cli")])
        self.assertEqual(json.loads((self.root/"cli/report.json").read_text(encoding="utf-8"))["status"],"review_ready")

    def test_provenance_split_missing_and_dimensions_rejected(self):
        original_candidate,original_manifest=copy.deepcopy(self.candidate),copy.deepcopy(self.manifest)
        mutations=[("kind",lambda:self.candidate["manifest"].update(data_kind="synthetic_fixture"),"data_kind"),
            ("raw_ref",lambda:self.candidate["records"][0]["source"].update(sha256="0"*64),"provenance"),
            ("extraction_ref",lambda:self.candidate["manifest"]["source_hashes"].update(extraction_manifest="0"*64),"provenance"),
            ("missing_extraction_hash",lambda:self.candidate["manifest"]["source_hashes"].pop("extraction_manifest"),"extraction provenance hash"),
            ("split_conflict",lambda:[r.update(split="test") for r in self.manifest["records"]],"split 충돌"),
            ("version",lambda:self.candidate["manifest"].update(dataset_version="another"),"dataset_version"),
            ("pose_key",lambda:self.candidate["records"][0]["source"].update(member=f"{self.seq}/json/frame_8.json"),"provenance")]
        for name,mutate,reason in mutations:
            self.candidate,self.manifest=copy.deepcopy(original_candidate),copy.deepcopy(original_manifest)
            mutate(); self.save()
            with self.subTest(name=name),self.assertRaisesRegex(ValueError,reason): self.run_review(name)
            rejected=json.loads((self.root/name/"report.json").read_text(encoding="utf-8"))
            self.assertEqual(rejected["status"],"rejected"); self.assertTrue(rejected["reason"])
            self.assertFalse((self.root/name/"review.html").exists())
        self.candidate,self.manifest=original_candidate,original_manifest;self.save()
        with self.assertRaisesRegex(ValueError,"누락"):
            self.run_review("missing",[{"sequence_id":self.seq,"frame_index":9}])
        with self.assertRaisesRegex(ValueError,"중복"):
            self.run_review("duplicate",self.selection*2)
        with self.assertRaisesRegex(ValueError,"최대8"):
            self.run_review("oversize_selection",self.selection*9)
        for row in self.manifest["records"]:
            for view in row["views"].values(): view["image_size"]=[512,512]
        self.manifest["image_size"]=[512,512];self.save()
        with self.assertRaisesRegex(ValueError,"원해상도"):
            self.run_review("size")

    def test_escaping_and_browser_script(self):
        hostile='pelvis</script><script>alert("x")</script>&'
        def replace(value):
            if isinstance(value,list): return [replace(v) for v in value]
            if isinstance(value,dict): return {k:replace(v) for k,v in value.items()}
            return hostile if value == "pelvis" else value
        self.candidate=replace(self.candidate);self.save()
        result=self.run_review("escaped")
        html=(self.root/"escaped/review.html").read_text(encoding="utf-8")
        self.assertNotIn(hostile,html)
        raw=re.search(r'id="reviewData">(.*?)</script>',html,re.S).group(1)
        payload=json.loads(raw)
        self.assertEqual(payload["joint_names"][0],hostile)
        node=shutil.which("node")
        self.assertIsNotNone(node,"기존 viewer JS 검증에 쓰는 Node가 필요합니다")
        script=re.search(r'<script>\s*(.*?)</script>',html,re.S).group(1)
        harness=r'''
const fs=require('fs'),vm=require('vm');const input=JSON.parse(fs.readFileSync(0,'utf8'));const arcs=[];
const elements={};function element(id){if(!elements[id])elements[id]={value:'',checked:false,textContent:'',append(o){if(this.value==='')this.value=o.value},getContext(){return {clearRect(){},drawImage(){},beginPath(){},arc(...a){arcs.push(a)},fill(){},fillText(){}}}};return elements[id]}
element('reviewData').textContent=input.payload;
class Image{set src(v){this.onload()}}
vm.runInNewContext(input.script,{document:{getElementById:element,createElement:()=>({})},Image,Math,JSON,Object});
if(!elements.coordinates.textContent||arcs.length===0)throw Error('overlay render missing');
console.log(JSON.stringify({arcs:arcs.length,metadata:elements.metadata.textContent.includes('train_ready')}));
'''
        completed=subprocess.run([node,"-e",harness],input=json.dumps({"payload":raw,"script":script}),text=True,capture_output=True,check=True)
        self.assertTrue(json.loads(completed.stdout)["metadata"])

    def test_file_change_alias_path_budgets_and_compact_rejection(self):
        with self.assertRaises(ValueError):
            review.create_review(self.manifest_path,self.rgb,self.candidate_path,self.selection,self.rgb/"unsafe")
        original=review._regular_read
        target=self.rgb/self.manifest["records"][0]["views"]["left"]["path"]
        before=target.read_bytes()
        def changed(path,*args,**kwargs):
            if path==target:
                raw=bytearray(before);raw[-1]^=1;target.write_bytes(raw)
            return original(path,*args,**kwargs)
        with patch.object(review,"_regular_read",side_effect=changed):
            with self.assertRaisesRegex(ValueError,"PNG 변경"): self.run_review("changed")
        target.write_bytes(before)
        right=self.rgb/self.manifest["records"][0]["views"]["right"]["path"]
        original_stat=Path.stat
        def alias_stat(path,*args,**kwargs):
            info=original_stat(path,*args,**kwargs)
            if path != right:
                return info
            physical=original_stat(target)
            fields={name:getattr(info,name) for name in dir(info) if name.startswith("st_")}
            fields.update(st_dev=physical.st_dev,st_ino=physical.st_ino)
            return SimpleNamespace(**fields)
        # Windows sandbox의 hardlink 생성이 막혀 동일 dev/ino stat를 모의한다.
        with patch.object(Path,"stat",alias_stat):
            with self.assertRaisesRegex(ValueError,"동일 실제 파일"):self.run_review("alias")
        with patch.object(review,"MAX_IMAGE_BYTES",1):
            with self.assertRaisesRegex(ValueError,"16MiB"):self.run_review("budget")
        original_json=review._json_bytes
        def too_large(value):
            if isinstance(value,dict) and value.get("frames"):
                raise ValueError("forced report byte limit")
            return original_json(value)
        with patch.object(review,"_json_bytes",side_effect=too_large):
            with self.assertRaisesRegex(ValueError,"report byte limit"):self.run_review("compact")
        compact=json.loads((self.root/"compact/report.json").read_text(encoding="utf-8"))
        self.assertTrue(compact["report_compacted"])
        self.assertEqual(compact["status"],"rejected")
        self.assertFalse((self.root/"compact/review.html").exists())
