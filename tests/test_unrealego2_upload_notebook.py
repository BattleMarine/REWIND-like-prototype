"""Verify the folder survey and that the delivered Colab contains the reviewed modules."""
import json
from pathlib import Path
import re
import sys
import tempfile
import unittest
from unittest.mock import patch
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from build_unrealego2_upload_notebook import NOTEBOOK, build
from unrealego2_upload_drive import DriveSafetyError, FOLDER
from unrealego2_upload_survey import folder_survey


class Listing:
    def __init__(self):
        self.rows = []
        self.requests = []

    def files(self):
        return self

    def list(self, **kw):
        self.requests.append(kw)
        ids = set(re.findall(r"'([^']+)' in parents", kw["q"]))
        names = set(re.findall(r"name = '([^']+)'", kw["q"]))
        matching = [f for f in self.rows if ids.intersection(f["parents"]) and (not names or f["name"] in names)]
        start = int(kw.get("pageToken") or 0)
        page = {"files": matching[start:start+1]}
        if len(matching) > start+1:
            page["nextPageToken"] = str(start+1)
        class Request:
            def execute(self):
                return page
        return Request()

    def add(self, parent, name, mime=FOLDER):
        entry = {"name": name, "parents": [parent], "id": str(len(self.rows)+1), "mimeType": mime}
        self.rows.append(entry)
        return entry["id"]


class UploadNotebookTest(unittest.TestCase):
    def test_portable_modules_match_and_cells_compile(self):
        notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
        self.assertEqual(notebook, build())
        cells = {c["id"]: "".join(c["source"]) for c in notebook["cells"] if c["cell_type"] == "code"}
        for name, code in cells.items():
            compile(code, name, "exec")
        with tempfile.TemporaryDirectory() as temp, patch("tempfile.mkdtemp", return_value=temp):
            original_path = sys.path.copy()
            try:
                scope = {}
                exec(cells["bootstrap"], scope)
                for module in ("source", "drive", "survey"):
                    name = f"unrealego2_upload_{module}.py"
                    self.assertEqual((Path(temp)/name).read_text(encoding="utf-8"), (ROOT/"scripts"/name).read_text(encoding="utf-8"))
                self.assertEqual(len(scope["REGISTRY"]), 60)
                self.assertFalse(any("depth" in r["category"] for r in scope["REGISTRY"]))
                self.assertNotIn("file_id", scope["assets"]["registry.json"])
            finally:
                sys.path[:] = original_path

    def test_presence_missing_duplicate_and_paginated_parent_groups(self):
        cat = "UnrealEgoData2_test_pose"
        d = Listing()
        parent = d.add("root", cat)
        parent = d.add(parent, "E")
        day = d.add(parent, "Day")
        for human in ("A","B"):
            person = d.add(day, human)
            glasses = d.add(person, "G")
            take = d.add(glasses, "01")
            d.add(take, "M")
            if human == "B":
                d.add(take, "M")
        with tempfile.TemporaryDirectory() as temp:
            z = Path(temp) / "split.zip"
            with ZipFile(z, "w") as f:
                for split in ("train","validation","test"):
                    data = "\n".join("E/Day/"+h+"/G/01/M" for h in "ABC") if split == "test" else ""
                    f.writestr("UnrealEgoData2/"+split+".txt",data)
            r = folder_survey(d,"root",z,[(cat,"E")],max_requests=40,max_seconds=10)
            self.assertEqual(r["counts"], {"folder_present":1,"ambiguous_or_wrong_type":1,"folder_absent":1})
            self.assertTrue(any(" or " in x["q"] for x in d.requests))
            with self.assertRaises(DriveSafetyError):
                folder_survey(d,"root",z,[(cat,"E")],max_requests=1,max_seconds=10)
            with self.assertRaises(ValueError):
                folder_survey(d,"root",z,[("UnrealEgoData2_test_depth","E")],max_requests=40,max_seconds=10)
