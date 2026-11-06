"""원격 호출 없는 Drive 감사·복구 계약 검사."""
import copy
import hashlib
import importlib.util
from pathlib import Path
import re
import tempfile
import unittest
import zipfile

spec = importlib.util.spec_from_file_location("upload", Path(__file__).resolve().parents[1] / "scripts/unrealego2_upload_drive.py")
upload = importlib.util.module_from_spec(spec)
spec.loader.exec_module(upload)


class Request:
    def __init__(self, call):
        self.call = call
        self.headers, self.callbacks = {}, []

    def add_response_callback(self, callback):
        self.callbacks.append(callback)

    def execute(self):
        result = copy.deepcopy(self.call())
        for callback in self.callbacks:
            callback({"etag": "test-etag"})
        return result


class FakeDrive:
    def __init__(self):
        self.items, self.payloads, self.calls = {}, {}, []
        self.hook = None
        self.rename_headers = []
        self.page_size = 2
        self.incomplete = False
        self.add("root", upload.FOLDER, None, file_id="root")

    def files(self):
        return self

    def add(self, name, mime, parent, data=b"", file_id=None, **extra):
        file_id = file_id or str(len(self.items) + 1)
        item = dict(id=file_id, name=name, mimeType=mime, parents=[parent] if parent else [], version="1", modifiedTime="fixed", trashed=False)
        if mime != upload.FOLDER:
            item.update(size=str(len(data)), md5Checksum=hashlib.md5(data).hexdigest())
        item.update(extra)
        self.items[file_id] = item
        self.payloads[file_id] = data
        return item

    def list(self, **kw):
        def run():
            parents = re.findall(r"'([^']+)' in parents", kw["q"])
            self.calls.append(("list", parents[0] if len(parents) == 1 else tuple(parents)))
            children = [i for i in self.items.values() if any(parent in i["parents"] for parent in parents) and not i.get("trashed")]
            names = re.findall(r"name = '([^']+)'", kw["q"])
            if names:
                children = [i for i in children if i["name"] in names]
            # Force pagination even for small fixtures.
            offset = int(kw.get("pageToken") or 0)
            result = {"files": children[offset:offset + self.page_size], "incompleteSearch": self.incomplete}
            if offset + self.page_size < len(children):
                result["nextPageToken"] = str(offset + self.page_size)
            return result
        return Request(run)

    def get(self, fileId, **kw):
        def run():
            self.calls.append(("get", fileId))
            return self.items[fileId]
        return Request(run)

    def create(self, body, media_body=None, **kw):
        def run():
            self.calls.append(("create", body["name"]))
            data = Path(media_body).read_bytes() if media_body else b""
            item = self.add(body["name"], body.get("mimeType", "application/octet-stream"), body["parents"][0], data,
                            appProperties=body.get("appProperties", {}))
            if self.hook:
                self.hook(self, item)
            return item
        return Request(run)

    def update(self, fileId, body, **kw):
        def run():
            self.calls.append(("update", fileId))
            self.rename_headers.append(dict(request.headers))
            self.items[fileId].update(body)
            self.items[fileId]["version"] = str(int(self.items[fileId]["version"]) + 1)
            return self.items[fileId]
        request = Request(run)
        return request


class Contract(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.stage = Path(self.temp.name)
        self.env = self.stage / "Lab" / "Day" / "sequence"
        self.env.mkdir(parents=True)
        self.drive = FakeDrive()
        self.category = "UnrealEgoData2_train_val_rgb"
        self.cat = self.drive.add(self.category, upload.FOLDER, "root")["id"]
        self.remote_env = self.drive.add("Lab", upload.FOLDER, self.cat)["id"]
        self.day = self.drive.add("Day", upload.FOLDER, self.remote_env)["id"]
        self.seq = self.drive.add("sequence", upload.FOLDER, self.day)["id"]

    def local(self, name, content=b"good"):
        (self.env / name).write_bytes(content)

    def audit(self):
        return upload.audit(self.drive, "root", self.category, "Lab", self.stage)

    def repair(self, report, **kw):
        return upload.repair(self.drive, report, enabled=True, media_factory=str, **kw)

    def official_sequence(self, count=20, existing=False):
        relative = "Day/human/glasses/001/motion/fisheye_final_image"
        for side in ("camera_left", "camera_right"):
            directory = self.stage / "Lab" / relative / side
            directory.mkdir(parents=True, exist_ok=True)
            for index in range(count // 2):
                (directory / f"final_{index}.png").write_bytes(f"{side}-{index}".encode())
        self.drive = FakeDrive()
        self.drive.page_size = 1000
        parent = "root"
        for name in (self.category, "Lab", "Day", "human", "glasses", "001"):
            parent = self.drive.add(name, upload.FOLDER, parent)["id"]
        if existing:
            self.drive.add("motion", upload.FOLDER, parent)
        return parent

    def test_pagination_classification_and_scope(self):
        for name in ("a", "b", "c", "d"):
            self.local(name)
        self.drive.add("a", "image/png", self.seq, b"good")
        self.drive.add("b", "image/png", self.seq, b"bad")
        self.drive.add("d", "image/png", self.seq, b"good", md5Checksum=None)
        self.drive.add("source.tar.gz000", "application/octet-stream", self.remote_env)
        excluded = self.drive.add("other-sequence", upload.FOLDER, self.day)["id"]
        self.drive.add("not-read", "image/png", excluded)
        report = self.audit()
        self.assertEqual(report["verified"], ["Day/sequence/a"])
        self.assertEqual(report["missing"], ["Day/sequence/c"])
        self.assertEqual(report["corrupt"][0]["path"], "Day/sequence/b")
        self.assertEqual(report["pending"][0]["path"], "Day/sequence/d")
        self.assertEqual(len(report["extras"]), 1)
        self.assertNotIn(("list", excluded), self.drive.calls)
        with self.assertRaises(upload.DriveSafetyError):
            self.repair(report)

    def test_preserve_corrupt_and_missing(self):
        self.local("a")
        self.local("b")
        old = self.drive.add("a", "image/png", self.seq, b"bad")
        result = self.repair(self.audit())
        self.assertEqual(len(result["operations"]), 2)
        self.assertEqual(self.drive.items[old["id"]]["name"], "a.WRONG_UPLOAD-" + old["id"])
        self.assertEqual(len(self.audit()["verified"]), 2)
        self.assertEqual(self.drive.payloads[old["id"]], b"bad")
        self.assertFalse(any(i.get("trashed") for i in self.drive.items.values()))
        self.assertTrue(result["if_match_applied_to_all_renames"])
        self.assertTrue(all(h.get("If-Match") == "test-etag" for h in self.drive.rename_headers))

    def test_duplicate_and_shortcut_block(self):
        self.local("a")
        self.drive.add("a", upload.SHORTCUT, self.seq)
        report = self.audit()
        self.assertFalse(report["missing"])
        self.assertEqual([r["path"] for r in report["pending"]], ["Day/sequence/a"])
        self.assertTrue(report["conflicts"])
        with self.assertRaises(upload.DriveSafetyError):
            self.repair(report)
        self.drive.add("a", "image/png", self.seq)
        self.assertTrue(self.audit()["conflicts"])

    def test_changed_after_audit_blocks_writes(self):
        self.local("a")
        report = self.audit()
        self.drive.add("concurrent", "image/png", self.seq)
        with self.assertRaises(upload.DriveSafetyError):
            self.repair(report)
        self.assertFalse(any(c[0] == "create" for c in self.drive.calls))

    def test_concurrent_during_upload_retains_pending(self):
        self.local("a")
        self.drive.hook = lambda d, item: d.add("a", "image/png", self.seq, b"other")
        with self.assertRaises(upload.DriveSafetyError):
            self.repair(self.audit())
        self.assertTrue(any(i["name"].startswith(".upload-") for i in self.drive.items.values()))
        self.assertFalse(any(c[0] == "update" for c in self.drive.calls))

    def test_interrupted_promotion_recovers_without_reupload(self):
        self.local("a")
        old = self.drive.add("a", "image/png", self.seq, b"bad")
        # one create and one rename are allowed; promotion exceeds limit.
        with self.assertRaises(upload.DriveSafetyError):
            self.repair(self.audit(), max_writes=2)
        creates = sum(c[0] == "create" for c in self.drive.calls)
        self.repair(self.audit())
        self.assertEqual(sum(c[0] == "create" for c in self.drive.calls), creates)
        self.assertEqual(self.audit()["verified"], ["Day/sequence/a"])
        self.assertIn("WRONG_UPLOAD", self.drive.items[old["id"]]["name"])

    def test_byte_budget_and_opt_in(self):
        self.local("a")
        report = self.audit()
        with self.assertRaises(upload.DriveSafetyError):
            upload.repair(self.drive, report)
        with self.assertRaises(upload.DriveSafetyError):
            self.repair(report, max_upload_bytes=1)
        self.assertFalse(any(c[0] == "create" for c in self.drive.calls))

    def test_bundle_is_one_media_and_externally_readable(self):
        relative = "human/001/motion/fisheye_final_image"
        left = self.env / relative / "camera_left"
        right = self.env / relative / "camera_right"
        left.mkdir(parents=True)
        right.mkdir(parents=True)
        (left / "a.png").write_bytes(b"good")
        (right / "b.png").write_bytes(b"second")
        report = self.audit()
        result = self.repair(report, mode="bundle")
        self.assertFalse(result["loose_originals_repaired"])
        self.assertEqual(len(result["operations"]), 1)
        archives = [i for i in self.drive.items.values() if i["name"].endswith(".zip")]
        self.assertEqual(len(archives), 1)
        import io
        with zipfile.ZipFile(io.BytesIO(self.drive.payloads[archives[0]["id"]])) as zf:
            self.assertEqual(zf.read("Day/sequence/" + relative + "/camera_left/a.png"), b"good")
            self.assertEqual(zf.read("Day/sequence/" + relative + "/camera_right/b.png"), b"second")
        self.assertEqual(len(self.audit()["missing"]), 2)
        # Re-extraction timestamps do not change content-addressed bundle identity.
        import os
        os.utime(left / "a.png", (1700000000, 1700000000))
        self.repair(self.audit(), mode="bundle")
        self.assertEqual(len([i for i in self.drive.items.values() if i["name"].endswith(".zip")]), 1)

    def test_depth_and_local_mutation_rejected(self):
        self.local("a")
        with self.assertRaises(upload.DriveSafetyError):
            upload.audit(self.drive, "root", "UnrealEgoData2_test_depth", "Lab", self.stage)
        report = self.audit()
        self.local("a", b"changed")
        with self.assertRaises(upload.DriveSafetyError):
            self.repair(report)

    def test_invalid_budgets_and_request_cap(self):
        self.local("a")
        for value in (float("nan"), float("inf"), -1):
            with self.assertRaises(upload.DriveSafetyError):
                upload.audit(self.drive, "root", self.category, "Lab", self.stage, max_seconds=value)
        with self.assertRaises(upload.DriveSafetyError):
            upload.audit(self.drive, "root", self.category, "Lab", self.stage, max_requests=1)

    def test_resumable_chunks_reserve_each_request(self):
        class Chunked:
            resumable = True
            calls = 0

            def next_chunk(self, num_retries):
                self.calls += 1
                return None, {"id": "done"} if self.calls == 3 else None

        request = Chunked()
        budget = upload._Budget(4, 60, writes=1, upload_bytes=100)
        self.assertEqual(budget.upload(request, 100), {"id": "done"})
        self.assertEqual(budget.requests, 4)
        self.assertEqual(budget.writes, 1)
        request = Chunked()
        with self.assertRaises(upload.DriveSafetyError):
            upload._Budget(2, 60, writes=1, upload_bytes=100).upload(request, 100)
        self.assertEqual(request.calls, 1)

    def test_missing_sequence_fast_path_reduces_requests_and_preserves_layout(self):
        self.official_sequence(40)
        fast = self.repair(self.audit())
        self.assertEqual(fast["operations"][0]["status"], "sequence_repaired")
        self.assertEqual(len(self.audit()["verified"]), 40)
        self.assertEqual(sum(c[0] == "update" for c in self.drive.calls), 1)
        self.assertFalse(any(i["name"].endswith(".zip") for i in self.drive.items.values()))
        self.official_sequence(40, existing=True)
        slow = self.repair(self.audit())
        self.assertLess(fast["stats"]["requests"], slow["stats"]["requests"] / 2)
        self.assertEqual(len(slow["operations"]), 40)

    def test_missing_sequence_interrupted_resume_reuses_verified_files(self):
        self.official_sequence(10)
        with self.assertRaises(upload.DriveSafetyError):
            self.repair(self.audit(), max_writes=7)
        staged_files = [i for i in self.drive.items.values() if i["name"].endswith(".png")]
        self.assertEqual(len(staged_files), 3)
        self.assertFalse(any(i["name"] == "motion" for i in self.drive.items.values()))
        original_ids = {i["id"] for i in staged_files}
        result = self.repair(self.audit())
        self.assertEqual(result["operations"][0]["reused_staged_files"], 3)
        self.assertEqual(len(self.audit()["verified"]), 10)
        self.assertTrue(original_ids.issubset(self.drive.items))
        self.assertEqual(len([i for i in self.drive.items.values() if i["name"].endswith(".png")]), 10)

    def test_sequence_extra_or_bad_file_never_promotes(self):
        for invalid in ("extra", "corrupt", "duplicate", "shortcut", "folder"):
            with self.subTest(invalid=invalid):
                self.official_sequence(10)
                with self.assertRaises(upload.DriveSafetyError):
                    self.repair(self.audit(), max_writes=7)
                staged_file = next(i for i in self.drive.items.values() if i["name"].endswith(".png"))
                parent = staged_file["parents"][0]
                if invalid == "extra":
                    self.drive.add("extra.png", "image/png", parent)
                elif invalid == "corrupt":
                    staged_file["md5Checksum"] = "bad"
                elif invalid == "duplicate":
                    self.drive.add(staged_file["name"], "image/png", parent)
                elif invalid == "shortcut":
                    self.drive.add("unexpected", upload.SHORTCUT, parent)
                else:
                    self.drive.add("unexpected", upload.FOLDER, parent)
                with self.assertRaises(upload.DriveSafetyError):
                    self.repair(self.audit())
                self.assertFalse(any(i["name"] == "motion" for i in self.drive.items.values()))

    def test_sequence_canonical_appearing_during_upload_blocks_promotion(self):
        parent = self.official_sequence(10)
        def concurrent(drive, item):
            if item["name"].endswith(".png") and not any(i["name"] == "motion" for i in drive.items.values()):
                drive.add("motion", upload.FOLDER, parent)
        self.drive.hook = concurrent
        with self.assertRaises(upload.DriveSafetyError):
            self.repair(self.audit())
        self.assertTrue(any(i["name"].startswith(".upload-sequence-") for i in self.drive.items.values()))
        self.assertEqual(sum(c[0] == "update" for c in self.drive.calls), 0)

    def test_sequence_upload_response_and_final_inventory_are_verified(self):
        for mutation in ("bad_response", "extra_after_upload"):
            with self.subTest(mutation=mutation):
                self.official_sequence(10)
                changed = []
                def change(drive, item):
                    if item["name"].endswith(".png") and not changed:
                        changed.append(True)
                        if mutation == "bad_response":
                            item["md5Checksum"] = "wrong"
                        else:
                            drive.add("unexpected.json", "application/json", item["parents"][0])
                self.drive.hook = change
                with self.assertRaises(upload.DriveSafetyError):
                    self.repair(self.audit())
                self.assertFalse(any(i["name"] == "motion" for i in self.drive.items.values()))

    def test_small_media_multipart_and_large_media_resumable(self):
        import sys
        import types
        from unittest.mock import patch
        module = types.ModuleType("googleapiclient.http")
        module.MediaFileUpload = lambda path, **kw: kw
        small = self.stage / "small.png"
        small.write_bytes(b"x")
        large = self.stage / "large.zip"
        with large.open("wb") as stream:
            stream.truncate(8 * 1024**2 + 1)
        with patch.dict(sys.modules, {"googleapiclient.http": module}):
            self.assertFalse(upload._default_media(small)["resumable"])
            self.assertTrue(upload._default_media(large)["resumable"])

    def test_incomplete_search_blocks_audit_and_repair(self):
        self.local("a")
        report = self.audit()
        self.drive.incomplete = True
        with self.assertRaises(upload.DriveSafetyError):
            self.audit()
        with self.assertRaises(upload.DriveSafetyError):
            self.repair(report)


if __name__ == "__main__":
    unittest.main()
