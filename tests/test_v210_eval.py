"""v2.1.0 inference conditioning, masks, decomposition and restart receipts."""

import io
from pathlib import Path
import tempfile
from types import SimpleNamespace
from unittest import TestCase

import numpy as np
import torch

from rewind_prepare.temporal_selection import seal
from rewind_archive.contracts import canonical
from rewind_v1.data import Clip
from rewind_v1.run_data import FixtureData
from scripts.v210_eval import GROUP_SCHEMA, read_group, segments
from scripts.v210_metrics import aggregate, rows, sums


class Budget:
    def check(self):
        pass

    def reserve(self, *_):
        pass

    def observed(self, *_, **__):
        pass


class EvaluationTests(TestCase):
    def fixture(self):
        spec = FixtureData().spec
        time = torch.arange(50)[None]<2
        mask = time[...,None].expand(1,50,4).clone()
        mask[0,1,0] = False  # Left wrist missing; its finger is still valid in official global MPJPE.
        clip = Clip(torch.zeros(1,50,56),torch.zeros(1,50,2,9),torch.zeros(1,50,4,3),mask,time,
                    torch.arange(50,dtype=torch.float64)[None]/25,"predicted_2d",split="validation",input_sha256="fixture")
        offsets = []

        def noise(key, joints, offset):
            offsets.append(offset)
            return torch.zeros(1,50,joints,3)

        def body_predict(_, __):
            body = torch.zeros(1,50,2,3)
            body[...,0,0],body[...,1,0] = 3,4
            return body

        def hand_predict(_, __, *, upper):
            upper.require("predicted_frozen_body", {"checkpoint_sha256":"fixture-body"})
            self.assertTrue(torch.equal(upper.valid,time[...,None].expand(1,50,2)))
            hand = torch.zeros(1,50,2,3)
            hand[...,0,0],hand[...,1,0] = -1,2
            return hand

        stages = SimpleNamespace(spec=spec,body_names=spec["body_joint_names"],body_idx=[0,1],upper_idx=[0,1],
                                 fingers=["left_tip","right_tip"],finger_idx=[2,3],_noise=noise)
        key = ("ArchVisInterior_ArchVis_RT/fixture",7)
        inferred = rows(stages,SimpleNamespace(predict=body_predict,checkpoint_sha256="fixture-body"),
                        SimpleNamespace(predict=hand_predict),clip,key,Budget())
        self.assertEqual(offsets,[100,200])
        return inferred,spec,key

    def test_real_slots_and_negative_cross_term(self):
        payload,spec,key = self.fixture()
        self.assertEqual(payload["frame_index"].tolist(),[7,8])
        self.assertEqual(payload["pred_device_mm"].shape,(2,4,3))
        part = sums(payload,spec)
        self.assertEqual(part["official"]["hand_global"],(16.,4))
        self.assertEqual(part["official"]["hand_local"],(5.,3))
        self.assertEqual(part["sides"]["left"]["common_count"],1)
        self.assertEqual(part["sides"]["left"]["terms"],{"wrist_mse_mm2":9.,"relative_mse_mm2":1.,
                         "global_mse_mm2":4.,"cross_mm2":-6.,"closure_mean_mm2":0.})
        reference = {"macro_score_mm":4.,"scores":{"ArchVisInterior_ArchVis_RT":{"body_mpjpe_mm":3.,"hand_global_mpjpe_mm":4.}}}
        previous = {"mpjpe_mm":5/3,"valid_joints":3}
        reference["hand_local_baseline"] = {"pooled":previous,"per_scenario":{"ArchVisInterior_ArchVis_RT":previous}}
        score = aggregate([(key,payload)],spec,reference)
        self.assertAlmostEqual(score["macro_score_mm"],(11/3+4)/2)
        self.assertEqual(score["prior_comparison"]["per_scenario"]["ArchVisInterior_ArchVis_RT"]["hand_global"]["delta_mm"],0.)
        self.assertEqual(score["prior_comparison"]["pooled_hand_local"]["valid_joint_delta"],0)

    def test_reload_checks_key_and_payload_sha(self):
        payload,spec,key = self.fixture()
        names = spec["joint_names"]
        stream = io.BytesIO()
        np.savez_compressed(stream,**payload,joint_names=np.asarray(names))
        raw = stream.getvalue()
        data = SimpleNamespace(real_frames=lambda *args:2)
        self.assertEqual(len(list(segments(raw,[key],data,names))),1)
        with self.assertRaisesRegex(ValueError,"prediction_frame_order"):
            list(segments(raw,[(key[0],2)],data,names))
        import hashlib
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory).resolve()
            path = folder/"coordinates.npz"
            path.write_bytes(raw)
            marker = folder/"marker.json"
            marker.write_bytes(canonical(seal({"schema":GROUP_SCHEMA,"binding":{"trial":"fixed"},"keys":[list(key)],
                "artifact":{"name":path.name,"bytes":len(raw),"sha256":hashlib.sha256(raw).hexdigest()},"real_frames":2})))
            self.assertEqual(read_group(folder,marker,{"trial":"fixed"},[key],data,names,Budget())[1]["real_frames"],2)
            path.write_bytes(raw[:-1]+bytes([raw[-1]^1]))
            with self.assertRaisesRegex(ValueError,"prediction_group_sha256"):
                read_group(folder,marker,{"trial":"fixed"},[key],data,names,Budget())
