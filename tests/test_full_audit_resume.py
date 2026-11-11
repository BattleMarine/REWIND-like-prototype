from argparse import Namespace
from contextlib import redirect_stdout
import io
import json
import os
from pathlib import Path
from unittest import TestCase
from unittest.mock import patch
import zipfile

from rewind_archive.contracts import canonical, digest
from rewind_prepare import extracted, full_archvis as full, full_audit_journal as journal_module
from rewind_prepare.converter import sha
from rewind_prepare.temporal_selection import seal
import test_full_archvis


class FullAuditResumeTests(TestCase):
    def setUp(self):
        fixture = test_full_archvis.FullArchVisTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        self.fixture, self.work, self.root = fixture, fixture.work, fixture.root
        self.plan_pin = fixture.preflight()
        self.inputs = fixture.args["inputs"]

    def audit(self, resume=False):
        return self.fixture.run_phase("audit", source_plan_sha256=self.plan_pin, resume=resume)

    def pause(self, pairs=2):
        observe = full._observe

        def observer(*args):
            original = observe(*args)

            def callback(event, **detail):
                original(event, **detail)
                if event == "pair" and detail["pairs_verified"] == pairs:
                    raise KeyboardInterrupt("fixture pause")

            return callback

        with patch.object(full, "_observe", side_effect=observer):
            with self.assertRaisesRegex(KeyboardInterrupt, "fixture pause"):
                self.audit()

    def track_payload_reads(self, destination):
        read = extracted._Budget.read

        def tracked(budget, path, *args, **kwargs):
            if Path(path).suffix == ".png" or Path(path).name.startswith("frame_"):
                destination.append(str(path))
            return read(budget, path, *args, **kwargs)

        return patch.object(extracted._Budget, "read", tracked)

    def restore(self, archive, pin, output=None):
        output = output or self.root / "restored"
        args = Namespace(work=output, inputs=self.inputs, archive=archive, archive_sha256=pin,
                         snapshot_sha256=self.fixture.snapshot["content_sha256"], source_plan_sha256=self.plan_pin)
        with redirect_stdout(io.StringIO()):
            full.restore_audit(args)
        return output

    def recovery(self, name="recovery.zip"):
        archive = self.root / name
        with redirect_stdout(io.StringIO()):
            full.recover(self.work, archive)
        return archive, sha(archive.read_bytes())

    def rewrite_entries(self, transform):
        path = self.work / "audit_resume/pairs.jsonl"
        entries = [json.loads(line) for line in path.read_bytes().splitlines()]
        head = json.loads((path.parent / "head.json").read_bytes())
        transform(entries)
        previous = digest(head["binding"])
        raw = b""
        for ordinal, entry in enumerate(entries):
            entry = seal({**{k: v for k, v in entry.items() if k != "content_sha256"},
                          "ordinal": ordinal, "previous_sha256": previous})
            previous = entry["content_sha256"]
            raw += canonical(entry) + b"\n"
        path.write_bytes(raw)
        head = seal({**{k: v for k, v in head.items() if k != "content_sha256"},
                     "count": len(entries), "bytes": len(raw), "last_sha256": previous})
        (path.parent / "head.json").write_bytes(canonical(head) + b"\n")

    def test_resume_reuses_complete_pairs_and_combines_geometry(self):
        self.pause()
        self.assertFalse((self.work / "verified").exists())
        head = json.loads((self.work / "audit_resume/head.json").read_bytes())
        self.assertEqual(head["count"], 2)
        reads = []
        with self.track_payload_reads(reads):
            self.assertEqual(self.audit(resume=True), 0)
        self.assertEqual(len(reads), 6)
        self.assertTrue(all("final_0.png" not in path or self.fixture.val in path.replace("\\", "/") for path in reads))
        manifest = json.loads((self.work / "verified/manifest.json").read_bytes())
        self.assertEqual(manifest["audit"]["pairs_reused"], 2)
        self.assertEqual(manifest["audit"]["pairs_content_read_this_invocation"], 2)
        self.assertEqual(manifest["audit"]["resume"]["committed_by_split"], {"train": 3, "validation": 1})
        baseline_work = self.root / "baseline"
        original_work = self.fixture.args["work"]
        self.fixture.args["work"] = baseline_work
        self.fixture.preflight()
        self.audit()
        self.fixture.args["work"] = original_work
        baseline = json.loads((baseline_work / "verified/manifest.json").read_bytes())
        self.assertEqual(manifest["records"], baseline["records"])
        for field in ("geometry_checks", "max_geometry_residuals"):
            self.assertEqual(manifest["audit"][field], baseline["audit"][field])
        self.assertLess(manifest["audit"]["read_bytes"], baseline["audit"]["read_bytes"])

    def test_cold_restore_tail_and_continue_without_rereading(self):
        self.pause(pairs=1)
        journal = self.work / "audit_resume/pairs.jsonl"
        with journal.open("ab") as stream:
            stream.write(b'{"partial_last_record":')
        archive, pin = self.recovery()
        restored = self.restore(archive, pin)
        status = json.loads((restored / "restore_status.json").read_bytes())
        self.assertGreater(status["journal"]["discarded_uncommitted_tail_bytes"], 0)
        self.fixture.args["work"] = restored
        reads = []
        with self.track_payload_reads(reads):
            self.audit(resume=True)
        self.assertEqual(len(reads), 9)
        self.assertTrue((restored / "verified/manifest.json").is_file())
        self.assertEqual(sha(archive.read_bytes()), pin)

    def test_committed_damage_and_producer_change_are_rejected(self):
        self.pause(pairs=1)
        archive, pin = self.recovery()
        with patch.object(journal_module, "producer_identity", return_value={"changed": "f"*64}):
            with self.assertRaisesRegex(ValueError, "producer_changed"):
                self.restore(archive, pin)
        journal = self.work / "audit_resume/pairs.jsonl"
        raw = journal.read_bytes()
        journal.write_bytes(raw.replace(b'"ordinal":0', b'"ordinal":1'))
        reads = []
        with self.track_payload_reads(reads):
            with self.assertRaisesRegex(ValueError, "document_pin"):
                self.audit(resume=True)
        self.assertEqual(reads, [])

    def test_resealed_duplicate_and_prior_sha_substitution_are_rejected(self):
        self.pause(pairs=4)
        original = (self.work / "audit_resume/pairs.jsonl").read_bytes()
        head = (self.work / "audit_resume/head.json").read_bytes()
        self.rewrite_entries(lambda entries: entries.__setitem__(1, entries[0].copy()))
        with self.assertRaisesRegex(ValueError, "record_scope"):
            self.audit(resume=True)
        (self.work / "audit_resume/pairs.jsonl").write_bytes(original)
        (self.work / "audit_resume/head.json").write_bytes(head)
        self.rewrite_entries(lambda entries: entries[3]["record"]["files"]["left"].update(sha256="f"*64))
        with self.assertRaisesRegex(ValueError, "prior_payload_identity_changed"):
            self.audit(resume=True)

    def test_completed_source_metadata_change_blocks_reuse(self):
        self.pause(pairs=1)
        path = self.fixture.fixture.rgb / self.fixture.train / "fisheye_final_image/camera_left/final_0.png"
        info = path.stat()
        os.utime(path, ns=(info.st_atime_ns, info.st_mtime_ns + 1_000_000_000))
        reads = []
        with self.track_payload_reads(reads):
            with self.assertRaisesRegex(ValueError, "completed_source_metadata_changed"):
                self.audit(resume=True)
        self.assertEqual(reads, [])

    def test_all_records_survive_manifest_partial_and_finalize_without_payload_reads(self):
        replace = Path.replace

        def fail_publish(path, target):
            if path.name == "manifest.json.partial":
                raise OSError("fixture manifest publish failure")
            return replace(path, target)

        with patch.object(Path, "replace", fail_publish):
            with self.assertRaisesRegex(OSError, "manifest publish failure"):
                self.audit()
        self.assertEqual(json.loads((self.work / "audit_resume/head.json").read_bytes())["count"], 4)
        self.assertTrue(list(self.work.glob(".pending_audit-*/manifest.json.partial")))
        reads = []
        with self.track_payload_reads(reads):
            self.audit(resume=True)
        self.assertEqual(reads, [])
        self.assertTrue((self.work / "verified/manifest.json").is_file())

    def test_legacy_count_only_zip_cannot_resume_and_outer_hash_is_required(self):
        archive, pin = self.recovery()
        with self.assertRaisesRegex(ValueError, "legacy_counts_not_supported"):
            self.restore(archive, pin)
        with self.assertRaisesRegex(ValueError, "sha256_mismatch"):
            self.restore(archive, "f"*64)

    def test_cross_split_triplet_check_replays_completed_records(self):
        self.audit()

        def collide(entries):
            prior = full.frozen_snapshot(self.inputs)[2]
            index = next(i for i, entry in enumerate(entries[:3])
                         if (entry["record"]["sequence_id"], entry["record"]["frame_index"]) not in prior)
            for role in extracted.ROLES:
                entries[index]["record"]["files"][role]["sha256"] = entries[3]["record"]["files"][role]["sha256"]

        self.rewrite_entries(collide)
        with self.assertRaisesRegex(ValueError, "triplet_content_split_overlap"):
            self.audit(resume=True)

    def test_head_publish_failure_preserves_last_committed_prefix(self):
        write_head = journal_module.PairJournal._write_head

        def fail_second(journal, head):
            if head["count"] == 2:
                raise OSError("fixture head failure")
            return write_head(journal, head)

        with patch.object(journal_module.PairJournal, "_write_head", fail_second):
            with self.assertRaisesRegex(OSError, "head failure"):
                self.audit()
        head = json.loads((self.work / "audit_resume/head.json").read_bytes())
        self.assertEqual(head["count"], 1)
        self.assertGreater((self.work / "audit_resume/pairs.jsonl").stat().st_size, head["bytes"])
        reads = []
        with self.track_payload_reads(reads):
            self.audit(resume=True)
        self.assertEqual(len(reads), 9)

    def test_source_change_during_read_rejects_pair_and_zero_prefix_can_be_recovered(self):
        decode = extracted.decode_rgb_payload
        path = self.fixture.fixture.rgb / self.fixture.train / "fisheye_final_image/camera_left/final_0.png"

        def mutate_after_read(*args):
            value = decode(*args)
            info = path.stat()
            os.utime(path, ns=(info.st_atime_ns, info.st_mtime_ns + 1_000_000_000))
            return value

        with patch.object(extracted, "decode_rgb_payload", side_effect=mutate_after_read):
            with self.assertRaisesRegex(ValueError, "source_changed_during_audit"):
                self.audit()
        archive, pin = self.recovery()
        restored = self.restore(archive, pin)
        self.assertEqual(json.loads((restored / "audit_resume/head.json").read_bytes())["count"], 0)

    def test_existing_published_manifest_resume_does_not_accumulate_pending_outputs(self):
        self.audit()
        pending_before = set(self.work.glob(".pending_audit-*"))
        original = (self.work / "verified/manifest.json").read_bytes()
        reads = []
        with self.track_payload_reads(reads):
            self.audit(resume=True)
        self.assertEqual(reads, [])
        self.assertEqual(set(self.work.glob(".pending_audit-*")), pending_before)
        self.assertEqual((self.work / "verified/manifest.json").read_bytes(), original)

    def test_restore_checks_inner_receipt_after_outer_hash(self):
        self.pause(pairs=1)
        original, _ = self.recovery()
        damaged = self.root / "damaged.zip"
        with zipfile.ZipFile(original) as source, zipfile.ZipFile(damaged, "x") as target:
            for name in source.namelist():
                raw = source.read(name)
                if name == "audit_resume/pairs.jsonl":
                    raw = raw.replace(b'"ordinal":0', b'"ordinal":1')
                target.writestr(name, raw)
        with self.assertRaisesRegex(ValueError, "part_pin"):
            self.restore(damaged, sha(damaged.read_bytes()))
