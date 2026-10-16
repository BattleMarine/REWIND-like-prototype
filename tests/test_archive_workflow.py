"""단일 설정 preflight·중복/실패 job·scan 결과 보관. 원본/인증은 모의한다."""

import copy
import gzip
import json
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path

from rewind_archive.contracts import Limits, PreparationError
from rewind_archive.drive_metadata import MetadataLimits
from rewind_archive.fixture import SEQUENCE, tar_bytes
from rewind_archive.workflow import (make_preflight, run_workflow_job, load_history,
    review_run, save_result_zip, read_report)
from test_drive_metadata import FakeDrive


class WorkflowTests(unittest.TestCase):
    def setUp(self):
        base=Path(__file__).resolve().parents[1]/"outputs/workflow_test_tmp"
        base.mkdir(parents=True,exist_ok=True)
        self.tmp=tempfile.TemporaryDirectory(dir=base);self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name)
        self.selection={"schema_version":"drive_part_selection_v1","archive_id":"fake_pose",
            "dataset_version":"synthetic_v1","kind":"pose","total_parts":1,
            "parts":[{"index":0,"name":"fake.tar.gz000","file_id":"fake_id"}]}
        self.data={"fake_id":gzip.compress(tar_bytes([(f"{SEQUENCE}/json/frame_{i}.json",b"abc") for i in (5,1,10)]),mtime=0)}
        self.config={"job_id":"scan_001","reason":"prefix inventory first observation","mode":"scan","purpose":"prefix_inventory",
            "selection":self.selection,"includes":[],"limits":asdict(Limits()),
            "metadata_limits":asdict(MetadataLimits()),"previous_result":""}
        self.history={};self.auth_calls=0;self.sessions=[]

    def session_factory(self):
        self.auth_calls+=1
        session=FakeDrive(self.selection,self.data)
        session.close=lambda:None
        self.sessions.append(session)
        return session

    def run_job(self, config=None):
        config=config or self.config
        return run_workflow_job(config,make_preflight(config,self.history),self.history,self.root,self.session_factory)

    def test_preflight_network_zero_effective_mode_identity_and_budgets(self):
        snapshot=make_preflight(self.config,self.history)
        self.assertEqual(self.auth_calls,0)
        summary=snapshot["summary"]
        self.assertEqual(summary["mode"],"scan")
        self.assertFalse(summary["payload_storage_enabled"])
        self.assertTrue(summary["reads_from_byte_zero"])
        self.assertFalse(summary["gzip_resume_or_seek"])
        self.assertEqual(summary["limits"]["media_range_bytes"],262144)
        self.assertEqual(summary["limits"]["io_chunk_bytes"],65536)
        self.assertEqual(snapshot["config"]["selection"],self.selection)

    def test_stale_config_any_field_blocks_before_authentication(self):
        for field,value in (("includes",["*/json/*"]),("purpose","targeted"),("job_id","new_id")):
            with self.subTest(field=field):
                config=copy.deepcopy(self.config);snapshot=make_preflight(config,self.history)
                config[field]=value
                with self.assertRaises(PreparationError):
                    run_workflow_job(config,snapshot,self.history,self.root,self.session_factory)
        self.assertEqual(self.auth_calls,0)
        self.assertFalse(list(self.root.glob("archive_job_*")))

    def test_scan_one_execution_same_id_and_fingerprint_repeat_block(self):
        root,record=self.run_job()
        self.assertEqual(record["status"],"finished")
        self.assertEqual(record["observed_regular_members"],3)
        self.assertFalse(record["sample_acquired"])
        self.assertFalse((root/"pilot/files").exists())
        count=len(self.sessions[-1].calls)
        with self.assertRaisesRegex(PreparationError,"already_started"):self.run_job()
        config=copy.deepcopy(self.config);config["job_id"]="scan_002"
        summary=make_preflight(config,self.history)["summary"]
        self.assertEqual(summary["same_config_previous_jobs"],["scan_001"])
        with self.assertRaisesRegex(PreparationError,"changed_reason"):self.run_job(config)
        self.assertEqual(self.auth_calls,1)
        self.assertEqual(len(self.sessions[-1].calls),count)
        config["reason"]="explicit new measurement after reviewing first inventory"
        _,second=self.run_job(config)
        self.assertEqual(second["repeat_of"],["scan_001"])
        self.assertEqual(self.auth_calls,2)

    def test_auth_failure_sanitized_job_only_zip_and_no_automatic_retry(self):
        def fail():
            self.auth_calls+=1
            raise RuntimeError("token=do_not_record_secret")
        root,record=run_workflow_job(self.config,make_preflight(self.config,self.history),self.history,self.root,fail)
        self.assertEqual(record["status"],"failed")
        self.assertEqual(record["stage"],"authentication")
        self.assertNotIn("secret",(root/"job.json").read_text())
        self.assertEqual(self.auth_calls,1)
        self.assertFalse((root/"metadata").exists())
        review=review_run(root)
        self.assertEqual(review["job"]["status"],"failed")
        archive=save_result_zip(root)
        self.assertTrue(archive.is_file())
        first=archive.read_bytes()
        self.assertEqual(save_result_zip(root).read_bytes(),first)
        with self.assertRaisesRegex(PreparationError,"already_started"):
            run_workflow_job(self.config,make_preflight(self.config,self.history),self.history,self.root,fail)
        self.assertEqual(self.auth_calls,1)

    def test_metadata_failure_job_reports_and_extract_compatibility(self):
        def invalid_session():
            session=self.session_factory();session.docs["fake_id"]["version"]="0";return session
        config=copy.deepcopy(self.config)
        root,record=run_workflow_job(config,make_preflight(config,self.history),self.history,self.root,invalid_session)
        self.assertEqual(record["status"],"metadata_failed")
        self.assertTrue((root/"metadata/metadata_report.json").is_file())
        self.assertFalse((root/"pilot").exists())
        self.assertTrue(save_result_zip(root).is_file())
        config.update(job_id="extract_001",mode="extract",purpose="targeted",includes=["*/frame_1.json"])
        root,record=self.run_job(config)
        self.assertTrue(record["sample_acquired"])
        report=read_report(root/"pilot/extraction_manifest.json")
        self.assertEqual(len(report["files"]),1)

    def test_history_reload_and_review_no_network(self):
        root,_=self.run_job()
        history=load_history(self.root)
        self.assertIn("scan_001",history)
        self.assertTrue(make_preflight(self.config,history)["summary"]["job_id_already_used"])
        before=self.auth_calls
        review=review_run(root)
        self.assertFalse(review["network_requested"])
        self.assertEqual(review["file_verification"],"passed_current_code")
        self.assertEqual(self.auth_calls,before)
        config=copy.deepcopy(self.config);config.update(mode="review",purpose="review",previous_result=str(root))
        self.assertEqual(make_preflight(config,history)["summary"]["source_size_version_modified_time"],"no_network")

    def test_invalid_config_bounds_and_globs_rejected_before_auth(self):
        for change in ({"includes":["x"+str(i) for i in range(36)]},{"reason":""},
            {"limits":{**self.config["limits"],"max_transfer_bytes":8388609}},
            {"mode":"extract","purpose":"discovery","includes":["**"]}):
            with self.subTest(change=change):
                config=copy.deepcopy(self.config);config.update(change)
                with self.assertRaises(PreparationError):make_preflight(config,self.history)
        self.assertEqual(self.auth_calls,0)


if __name__=="__main__":unittest.main()
