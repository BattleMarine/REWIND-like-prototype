import io
import json
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import PIL
from PIL import Image

from rewind_archive.contracts import canonical
from scripts import build_v2_cache256 as builder, run_v2_cache256 as app, run_v2_bulk_cache as bulk
from scripts.run_v2_resolution_ab import decode


class Cache256Tests(unittest.TestCase):
    def test_interrupted_pilot_resumes_and_proven_shards_feed_tail_only(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            for name in ("rgb", "pose", "cache", "first", "second", "third", "blocked", "stale"):
                (root/name).mkdir()
            legacy = root/"cache"/"v2_bulk_cache_041837Z"
            legacy.mkdir()
            (legacy/"sentinel").write_bytes(b"unchanged")
            store = root/"cache"/app.STORE
            stream = io.BytesIO()
            Image.new("RGB", (1024,1024), (32,70,80)).save(stream, format="PNG")
            image = stream.getvalue()
            spec = {"joint_names": ["j"], "coordinates": {"mm_per_raw_unit":10,"raw_consistency_tolerance":.01,
                     "image_sizes":[[1024,1024],[1024,1024]]}}
            cam = {"trans":[1,2,3],"rot":[0,0,0],"open3d_trans":[1,-2,3],
                   "open3d_rot":[0,0,0],"device_pts3d":[0,0,0]}
            rows = []
            for index in range(7):
                pose = canonical({"frame":index,"camera_left":cam,"camera_right":cam,"joints":{
                    "j":{"device_pts3d":[1,2,3],"camera_left_pts2d":[10,20],"camera_right_pts2d":[30,40]}}})
                files = {}
                for role in ("left","right","pose"):
                    raw = pose if role == "pose" else image
                    name = f"{index}-{role}"
                    ((root/("pose" if role == "pose" else "rgb"))/name).write_bytes(raw)
                    files[role] = {"path":name,"size_bytes":len(raw),"sha256":app.contract.sha(raw)}
                rows.append({"sequence_id":"toy","frame_index":index,"split":"train","files":files})
            manifest = {"records":rows,"content_sha256":"test"}
            raw = {role: ((root/("pose" if role == "pose" else "rgb"))/p["path"]).read_bytes()
                   for role,p in rows[0]["files"].items()}
            _, at128 = decode(raw,rows[0],spec,128)
            _, at256 = decode(raw,rows[0],spec,256)
            self.assertTrue(app.torch.allclose(at128["xy"]/127,at256["xy"]/255))
            for key in at128.keys()-{"xy"}:
                self.assertTrue(app.torch.equal(at128[key],at256[key]),key)
            pins = {"consumer_source_identity":app.digest(app.contract.source()),"consumer_overlay":{},
                    "embedded_manifest":"fixture","archive_sha256":"fixture","frames":7,
                    "cpu_runtime":{"torch":str(app.torch.__version__),"numpy":str(app.np.__version__),"pillow":PIL.__version__}}

            def args(label, mode, tip="", proof="", time_budget=0, drive=0):
                return SimpleNamespace(work=root/label,store=store,mode=mode,budget_tip=tip,pilot_receipt=proof,
                                       additional_seconds=time_budget,additional_gib=drive,recover_writer="",
                                       rgb=root/"rgb",pose=root/"pose",started=time.monotonic())

            verify = app.verify_content
            extract = app.extract_pinned_zip
            def extract_payload(path, *args, **kwargs):
                if path.name == "payload.zip":
                    return extract(path,*args,**kwargs)

            def fail_second(folder, index, *other):
                if index == 1:
                    raise OSError("fixture interruption before second publish")
                return verify(folder,index,*other)

            with (patch.object(app,"extract_pinned_zip",side_effect=extract_payload),
                  patch.object(app,"read_manifest",return_value=(manifest,spec)),
                  patch.object(app,"available_ram", return_value=8*app.GIB),
                  patch.object(app,"TARGET_SHARDS",4), patch.object(bulk.cache,"CHUNK",2)):
                with patch.object(app,"verify_content",side_effect=fail_second), self.assertRaises(OSError):
                    app.run(args("first","pilot"),{**pins,"mode":"pilot"})
                interrupted = json.loads((root/"first/result.json").read_bytes())
                self.assertEqual(interrupted["shards"],1)
                original = (store/"boundary-00000.json").read_bytes()
                self.assertFalse((store/"pilot-proof.json").exists())
                # A new work directory starts from the persisted prefix without needing its original raw files.
                for i in (0,1):
                    for role in ("left","right","pose"):
                        ((root/("pose" if role=="pose" else "rgb"))/f"{i}-{role}").unlink()
                pilot = app.run(args("second","pilot",interrupted["budget_tip_sha256"]),{**pins,"mode":"pilot"})
                self.assertEqual(pilot["shards"],3)
                self.assertEqual(pilot["status"],"passed")
                self.assertEqual(original,(store/"boundary-00000.json").read_bytes())
                with self.assertRaisesRegex(ValueError,"latest_tip_required"):
                    app.run(args("stale","long",interrupted["budget_tip_sha256"],pilot["pilot_receipt_sha256"],1500,1),
                            {**pins,"mode":"long"})
                self.assertFalse((store/"long-authorization.json").exists())
                with self.assertRaisesRegex(ValueError,"time_not_confirmed"):
                    app.run(args("blocked","long",pilot["budget_tip_sha256"],pilot["pilot_receipt_sha256"]),
                            {**pins,"mode":"long"})
                self.assertFalse((store/"long-authorization.json").exists())
                long = app.run(args("third","long",pilot["budget_tip_sha256"],pilot["pilot_receipt_sha256"],1500,1),
                               {**pins,"mode":"long"})
            self.assertEqual(long["status"],"passed")
            self.assertEqual(long["shards"],4)
            self.assertEqual(long["frames"],7)
            self.assertEqual(long["drive_cap_bytes"],pilot["drive_bytes"]+app.GIB)
            self.assertLessEqual(long["reserved_seconds"],1500)
            self.assertEqual((legacy/"sentinel").read_bytes(),b"unchanged")
            tail = app.sealed_file(store/"boundary-00003.json",app.SCHEMA)
            self.assertEqual(tail["frames"],1)

    def test_256_notebook_separation_caps_and_blank_authorization(self):
        for mode, cap in (("pilot",1800),("long",32400)):
            book = builder.notebook({"source_name":"fixture.zip","mode":mode},{})
            scope = {}
            with patch("builtins.print"):
                for cell in book["cells"]:
                    if cell["cell_type"] == "code":
                        exec(compile("".join(cell["source"]),"<frozen-cache256>","exec"),scope)
            self.assertFalse(scope["RUN_CACHE256"])
            self.assertEqual(scope["LIMIT"],cap)
            self.assertEqual(scope["STORE"],"v2_cache256_041837Z")
            self.assertEqual(scope["EXPECTED_256_TIP_SHA256"],"")
            if mode == "long":
                self.assertEqual(scope["CONFIRMED_ADDITIONAL_DRIVE_GIB"],0)
                self.assertEqual(scope["CONFIRMED_ADDITIONAL_CPU_SECONDS"],0)
                self.assertEqual(scope["VALIDATED_PILOT_RECEIPT_SHA256"],"")
