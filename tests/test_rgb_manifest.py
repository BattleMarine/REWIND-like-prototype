"""소형 실제 PNG archive 추출본의 manifest builder 검사."""
import copy
import gzip
import hashlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

try:
    import torch
    from PIL import Image
except ImportError:
    torch = None

if torch is not None:
    import rewind_m1.rgb_manifest as builder
    from rewind_archive.contracts import Limits, digest
    from rewind_archive.extraction import prepare
    from rewind_archive.fixture import SEQUENCE, local_manifest, tar_bytes
    from rewind_m1.local_rgb import LocalStereoRGB


@unittest.skipUnless(torch is not None, "torch와 Pillow가 필요합니다")
class ManifestBuilderTests(unittest.TestCase):
    def setUp(self):
        temp_root = Path(__file__).resolve().parents[1]/"outputs"/"test_tmp"
        temp_root.mkdir(parents=True,exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=temp_root)
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.inputs = self.root/"inputs"; self.inputs.mkdir()
        self.selection = [{"sequence_id":SEQUENCE,"frames":[0,1],"split":"unknown","split_evidence":"fixture declaration only"}]
        self.left = self.extract("left",(2,1,0))
        self.right = self.extract("right",(1,0))

    def extract(self, side, frames, name=None):
        name = name or side
        entries = []
        for frame in frames:
            buffer = io.BytesIO()
            with Image.new("RGB",(8,6),(30+frame*30,90,180 if side == "right" else 20)) as image:
                image.save(buffer,format="PNG")
            entries.append((f"{SEQUENCE}/fisheye_final_image/camera_{side}/final_{frame}.png",buffer.getvalue()))
        source = self.root/(name+"_archive")
        manifest = local_manifest(source,gzip.compress(tar_bytes(entries),mtime=0),"rgb")
        output = self.inputs/name
        report = prepare(manifest,output,["**"],Limits(io_chunk_bytes=512,media_range_bytes=512,max_ranges=128),base=source)
        self.assertTrue(report["inspection_candidates_usable"])
        return output

    def build(self, name, **changes):
        args = dict(root=self.inputs,extractions=[self.left,self.right],selection=self.selection,
            image_size=[8,6],output=self.root/name,manifest_id="fixture_builder")
        args.update(changes)
        return builder.build_manifest(**args)

    def alter_report(self,path,change):
        path = path/"extraction_manifest.json"
        doc = json.loads(path.read_text(encoding="utf-8"))
        change(doc)
        doc["content_sha256"] = digest({k:v for k,v in doc.items() if k != "content_sha256"})
        path.write_text(json.dumps(doc),encoding="utf-8")
        return doc

    def test_multi_source_loader_and_cli(self):
        before = {p:hashlib.sha256(p.read_bytes()).hexdigest() for p in self.inputs.rglob("*") if p.is_file()}
        selection = copy.deepcopy(self.selection)
        manifest, report = self.build("created")
        self.assertEqual(report["status"],"created")
        self.assertFalse(report["train_ready"])
        self.assertEqual(report["rgb_pose_alignment"],"not_checked")
        self.assertEqual(report["unselected"],[{"sequence_id":SEQUENCE,"frame_index":2,"missing_views":["right"]}])
        self.assertEqual([r["frame_index"] for r in manifest["records"]],[0,1])
        for index,path in enumerate((self.left,self.right)):
            doc = json.loads((path/"extraction_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["sources"][f"source_{index}"]["extraction_manifest_sha256"],doc["content_sha256"])
        self.assertTrue(manifest["records"][0]["views"]["left"]["path"].startswith("left/files/"))
        loader = LocalStereoRGB(self.root/"created/manifest.json",self.inputs)
        self.assertEqual(loader.load_batch([0,1])["images"].shape,(2,2,3,6,8))
        self.assertEqual(selection,self.selection)
        self.assertEqual(before,{p:hashlib.sha256(p.read_bytes()).hexdigest() for p in before})
        selection_path = self.root/"selection.json"
        selection_path.write_text(json.dumps(self.selection),encoding="utf-8")
        builder.main(["--root",str(self.inputs),"--extraction",str(self.left),"--extraction",str(self.right),
            "--selection",str(selection_path),"--image-size","8","6","--id","cli","--output",str(self.root/"cli")])
        with self.assertRaises(ValueError):
            builder.main(["--root",str(self.inputs),"--extraction",str(self.left),
                "--selection",str(selection_path),"--image-size","8","6","--id","cli_fail","--output",str(self.root/"cli_fail")])
        self.assertFalse((self.root/"cli_fail/manifest.json").exists())
        self.assertEqual(json.loads((self.root/"cli_fail/report.json").read_text(encoding="utf-8"))["status"],"rejected")
        original = (self.root/"created/manifest.json").read_bytes()
        with self.assertRaises(FileExistsError): self.build("created")
        self.assertEqual((self.root/"created/manifest.json").read_bytes(),original)

    def test_missing_duplicate_and_decode_rejected(self):
        duplicate = self.extract("left",(0,),"duplicate")
        cases = (("missing",{"extractions":[self.left]}),
                 ("missing_frame",{"selection":[{**self.selection[0],"frames":[9]}]}),
                 ("duplicate",{"extractions":[self.left,self.right,duplicate]}),
                 ("decode",{"image_size":[9,6]}))
        for name,changes in cases:
            with self.subTest(name=name), self.assertRaises(ValueError): self.build(name,**changes)
            self.assertFalse((self.root/name/"manifest.json").exists())
            self.assertFalse((self.root/name/"manifest.pending.json").exists())
            report = json.loads((self.root/name/"report.json").read_text(encoding="utf-8"))
            self.assertEqual(report["status"],"rejected")
        self.alter_report(self.right,lambda d:d["files"][0].update(tar_complete=False))
        with self.assertRaises(ValueError): self.build("incomplete")
        self.assertFalse((self.root/"incomplete/manifest.json").exists())

    def test_explicit_code_change_preserves_file_hash_verification(self):
        self.alter_report(self.left,lambda d:d.update(code_sha256={"old.py":"0"*64}))
        with self.assertRaisesRegex(ValueError,"extraction_code_changed"): self.build("old_default")
        _, report = self.build("old_allowed",allow_code_change=True)
        self.assertFalse(report["sources"][0]["code_matches"])
        self.assertTrue(report["allow_code_change"])
        path = next((self.left/"files").rglob("*.png"))
        raw = bytearray(path.read_bytes()); raw[-1] ^= 1; path.write_bytes(raw)
        with self.assertRaisesRegex(ValueError,"extracted_file_hash"):
            self.build("corrupted",allow_code_change=True)

    def test_global_budget_selection_and_path_guards(self):
        with patch.object(builder,"MAX_PAYLOAD_BYTES",1),patch.object(builder,"verify_extraction") as verify:
            with self.assertRaisesRegex(ValueError,"누적32MiB"): self.build("budget")
            verify.assert_not_called()
        with patch.object(builder,"MAX_FILES",1),patch.object(builder,"verify_extraction") as verify:
            with self.assertRaisesRegex(ValueError,"파일 수/개별 byte"): self.build("file_count")
            verify.assert_not_called()
        with patch.object(builder,"verify_extraction") as verify:
            with self.assertRaisesRegex(ValueError,"pixel/단일 tensor"):
                self.build("pixels",image_size=[100000,100000])
            verify.assert_not_called()
        with self.assertRaises(ValueError): self.build("inside",output=self.inputs/"invalid")
        self.assertFalse((self.inputs/"invalid").exists())
        invalid = [{**self.selection[0],"frames":[1,0]}]
        with self.assertRaisesRegex(ValueError,"증가 정수"): self.build("reverse",selection=invalid)
        original_right = (self.right/"extraction_manifest.json").read_bytes()
        self.alter_report(self.right,lambda d:d["archive_manifest"].update(dataset_version="different"))
        self.alter_report(self.right,lambda d:d.update(archive_manifest_sha256=digest(d["archive_manifest"])))
        with self.assertRaisesRegex(ValueError,"dataset_version/data_kind"): self.build("version")
        (self.right/"extraction_manifest.json").write_bytes(original_right)
        self.alter_report(self.right,lambda d:d["archive_manifest"].update(source_scope="local_unverified"))
        self.alter_report(self.right,lambda d:d.update(archive_manifest_sha256=digest(d["archive_manifest"])))
        with self.assertRaisesRegex(ValueError,"dataset_version/data_kind"): self.build("data_kind")

    def test_individual_byte_and_metadata_hash_preflight(self):
        original = (self.left/"extraction_manifest.json").read_bytes()
        self.alter_report(self.left,lambda d:d["files"][0].update(size_bytes=8*1024**2+1))
        with patch.object(builder,"verify_extraction") as verify:
            with self.assertRaisesRegex(ValueError,"파일 수/개별 byte"): self.build("individual")
            verify.assert_not_called()
        (self.left/"extraction_manifest.json").write_bytes(original)
        doc = json.loads(original)
        doc["content_sha256"] = "0"*64
        (self.left/"extraction_manifest.json").write_text(json.dumps(doc),encoding="utf-8")
        with patch.object(builder,"verify_extraction") as verify:
            with self.assertRaisesRegex(ValueError,"content hash"): self.build("metadata_hash")
            verify.assert_not_called()
