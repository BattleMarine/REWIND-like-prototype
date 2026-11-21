import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import zipfile

import numpy as np
from PIL import Image

from rewind_archive.contracts import canonical
from scripts import build_v2_resolution_ab as builder, run_v2_resolution_ab as ab


class ResolutionTests(unittest.TestCase):
    def test_manifest_selection_is_train_only_and_keeps_first_order(self):
        records = [{"sequence_id": scene+"/clip", "frame_index": index, "split": split}
                   for scene in ab.SCENARIOS for index, split in ((0,"test"), (1,"train"), (2,"train"))]
        self.assertEqual([rows[0]["frame_index"] for rows in ab.selection(records, 1).values()], [1]*3)

    def test_all_scenarios_run_and_retain_a_readback_verified_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name in ("rgb", "pose", "work", "cache"):
                (root/name).mkdir()
            args = SimpleNamespace(rgb=root/"rgb", pose=root/"pose", work=root/"work", seconds=500,
                                   drive_results=root/"cache/v2_resolution_ab_041837Z")
            spec = {"joint_names": ["j"], "coordinates": {"mm_per_raw_unit": 10, "raw_consistency_tolerance": .01,
                     "image_sizes": [[1024,1024],[1024,1024]]}}
            gradient = np.indices((1024,1024), dtype=np.int16).sum(axis=0).astype("uint8")
            stream = io.BytesIO()
            Image.fromarray(np.stack((gradient, gradient, gradient), -1)).save(stream, format="PNG")
            rgb = stream.getvalue()
            cam = {"trans": [1,2,3], "rot": [0,0,0], "open3d_trans": [1,-2,3],
                   "open3d_rot": [0,0,0], "device_pts3d": [0,0,0]}
            records = []
            for scene in ab.SCENARIOS:
                files = {}
                pose = canonical({"frame": 0, "camera_left": cam, "camera_right": cam, "joints": {
                    "j": {"device_pts3d": [1,2,3], "camera_left_pts2d": [10,20], "camera_right_pts2d": [30,40]}}})
                for role in ("left", "right", "pose"):
                    raw = pose if role == "pose" else rgb
                    name = f"{scene}/{role}"
                    target = (args.pose if role == "pose" else args.rgb)/name
                    target.parent.mkdir(exist_ok=True)
                    target.write_bytes(raw)
                    files[role] = {"path": name, "size_bytes": len(raw), "sha256": ab.contract.sha(raw)}
                records.append({"sequence_id": scene+"/clip", "frame_index": 0, "split": "train", "files": files})
            groups = ab.selection(records, 1)
            pins = {"manifest_content_sha256": "fixture", "consumer_source_identity": ab.digest(ab.contract.source()),
                    "embedded_manifest": "fixture.zip", "archive_sha256": "fixture", "sample": ab.selection_pin(groups),
                    "consumer_overlay": {"scripts/"+n: ab.contract.sha(Path(ab.__file__).with_name(n).read_bytes())
                                         for n in ("run_v2_resolution_ab.py", "v2_ordered_prefetch.py")}}
            with (patch.object(ab, "extract_pinned_zip"), patch.object(ab, "read_manifest",
                    return_value=({"records": records, "content_sha256": "fixture"}, spec)),
                  patch.object(ab, "selection", return_value=groups), patch.object(ab, "available_ram", return_value=8*ab.GIB)):
                result = ab.run(args, pins)
            self.assertEqual(result["status"], "passed")
            self.assertEqual(len(result["reference_128"]), 3)
            self.assertEqual(len(result["comparisons"]), 6)
            self.assertEqual([r["receipt"]["size"] for r in result["comparisons"]], [128,256,256,128,128,256])
            self.assertEqual(result["stage_io"]["source_bytes"], sum(e["source_bytes"] for e in pins["sample"].values()))
            archive = args.drive_results/result["retention"]["name"]
            self.assertEqual(result["retention"]["sha256"], ab.contract.sha(archive.read_bytes()))
            with zipfile.ZipFile(archive) as z:
                self.assertEqual(len(z.namelist()), 5)
                self.assertEqual(len(json.loads(z.read("comparison.json"))["comparisons"]), 6)

    def test_notebook_default_off_and_separate_results(self):
        nb = builder.notebook({"source_name": "test.zip"}, {})
        scope = {}
        with patch("builtins.print"):
            for cell in nb["cells"]:
                if cell["cell_type"] == "code":
                    exec(compile("".join(cell["source"]), "<notebook>", "exec"), scope)
        self.assertFalse(scope["RUN_RESOLUTION_AB"])
        self.assertEqual(scope["LIMIT"], 3600)
        self.assertIn('"v2_resolution_ab_041837Z"', "".join(nb["cells"][-1]["source"]))
