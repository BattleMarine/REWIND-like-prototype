"""Export common validation predictions and a compact stereo/3D visual review."""

from pathlib import Path
import tempfile
import uuid
import zipfile

import numpy as np
from PIL import Image, ImageDraw
import torch

from rewind_archive.contracts import canonical
from rewind_m1.rgb_frontend_artifacts import scale_xy
from rewind_prepare.extracted import IMAGE_SIZE
from rewind_prepare.temporal_cache import _hash_file
from rewind_m3.cascade import UpperCondition
from rewind_v1.data import attach_predicted_hands, hand_clip
from rewind_v1.run_stages import preserve_rng
from rewind_v1.v2_contract import SCENARIOS
from rewind_v1.v2_retention import copy_verified
from scripts.v2_gpu_training_storage import EVAL_SEED, MIB, publish_json, read_sealed


def shared_clips(plan):
    return {scenario: min((r["sequence"], r["start_frame"]) for r in plan["clips"]
                          if r["split"] == "validation" and r["real_frames"] == 50
                          and r["sequence"].startswith(scenario+"/")) for scenario in SCENARIOS}


def draw_clip(path, images, xy, xy_gt, xy_mask, points, gt, valid, names, spec):
    joints = {name: i for i, name in enumerate(names)}
    edges = [(joints[a], joints[b]) for a, b in spec["edges"] if a in joints and b in joints]
    projection = gt[..., :2][valid]
    if not len(projection):
        raise ValueError("v2_gpu_comparison_reference_scope")
    low, high = np.quantile(projection, [0.02, 0.98], axis=0)
    scale = 220 / max(float((high-low).max()), 1.)
    center = (low+high)/2
    side = images.shape[-1]
    if side not in (128, 256) or images.shape[-2] != side:
        raise ValueError("v2_gpu_comparison_image_size")
    frames = []
    for t in range(len(images)):
        frame = Image.new("RGB", (2*side+256, side+20), "#161b23")
        draw = ImageDraw.Draw(frame)
        draw.text((5, 4), "L | prediction / GT", fill="white")
        draw.text((side+5, 4), "R | prediction / GT", fill="white")
        draw.text((2*side+6, 4), "3D camera XY projection | prediction / GT", fill="white")
        for view in range(2):
            rgb = (images[t, view].transpose(1, 2, 0)*255).round().astype(np.uint8)
            frame.paste(Image.fromarray(rgb), (side*view, 20))
            for values, mask, color in ((xy_gt[t, view], xy_mask[t, view], "#54e596"),
                                         (xy[t, view], np.ones(xy.shape[2], dtype=bool), "#ff9b55")):
                for x, y in values[mask]:
                    if 0 <= x < side and 0 <= y < side:
                        draw.ellipse((side*view+x-1, 19+y, side*view+x+1, 21+y), fill=color)
        for values, mask, color in ((gt[t], valid[t], "#54e596"), (points[t], np.ones(len(names), dtype=bool), "#ff9b55")):
            q = (values[:, :2]-center)*scale*0.51 + np.array([2*side+128, side//2+19])
            for a, b in edges:
                if mask[a] and mask[b]:
                    draw.line((*q[a], *q[b]), fill=color)
        frames.append(frame)
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=40, loop=0)
    if side == 256:
        frames[0].save(path.with_suffix(".png"))


@torch.no_grad()
def export_comparison(pipeline, drive, scratch, ledger, owner):
    marker = drive / "comparison.json"
    pins = {role: pipeline.progress["best"][role]["sha256"] for role in ("frontend", "body", "hand")}
    data, tree = pipeline.data, pipeline.cascade
    sample = shared_clips(data.plan)
    binding = {"experiment": pipeline.identity, "model": pins,
               "eval_seed": EVAL_SEED, "clips": {k: list(v) for k, v in sample.items()}}
    size = pipeline.frontend_size
    if size == [256, 256]:
        binding["input_size"] = size
        binding["native_image_size"] = list(IMAGE_SIZE)
    schema = "rewind_v2_gpu_comparison_v1"
    if marker.is_file():
        doc = read_sealed(marker, schema)
        pin = doc["archive"]
        if doc["binding"] != binding or Path(pin["name"]).name != pin["name"]:
            raise ValueError("v2_gpu_existing_comparison_binding")
        ledger.reserve("read", pin["bytes"])
        if _hash_file(drive / pin["name"], expected_size=pin["bytes"], check_time=ledger.check) != pin["sha256"]:
            raise ValueError("v2_gpu_existing_comparison_sha256")
        ledger.observed("read", pin["bytes"])
        return doc
    with preserve_rng(), tempfile.TemporaryDirectory(dir=scratch, prefix=".comparison-") as temporary:
        root = Path(temporary)
        frontend = pipeline._make_frontend()
        pipeline._load_best("frontend", frontend)
        body, hand = tree.make("body"), tree.make("hand", pins["body"])
        pipeline._load_best("body", body)
        pipeline._load_best("hand", hand)
        names = tree.body_names + tree.fingers
        index = tree.body_idx + tree.finger_idx
        for scenario, key in sample.items():
            ledger.check()
            clip = tree.clips["validation"][key]
            b = body.predict(clip.select_joints(tree.body_idx), tree._noise(key, len(tree.body_names), 100)).cpu()
            values = b[..., tree.upper_idx, :]
            upper = UpperCondition(values, clip.time_valid[..., None].expand(values.shape[:-1]), "predicted_frozen_body", pins["body"])
            ledger.check()
            local = hand.predict(hand_clip(clip, data.spec), tree._noise(key, len(tree.fingers), 200), upper=upper).cpu()
            points = torch.cat((b, attach_predicted_hands(b, local, data.spec)), dim=2)[0].numpy()
            target = clip.target_mm[0, :, index, :].numpy()
            valid = clip.target_valid[0, :, index].numpy()
            images, xy, xy_gt, xy_mask = [], [], [], []
            for offset in range(0, data.width, 2):
                ledger.check()
                pair = [data.cache.frame((key[0], key[1]+i), "validation") for i in range(offset, offset+2)]
                inputs = torch.stack([image for image, _ in pair])
                predicted = frontend.model(inputs.to(pipeline.device))[1].cpu()
                images.extend(inputs.numpy()); xy.extend(predicted.numpy())
                xy_gt.extend(meta["xy"].numpy() for _, meta in pair)
                xy_mask.extend(meta["xy_valid"].numpy() for _, meta in pair)
            xy, xy_gt, xy_mask = map(np.asarray, (xy, xy_gt, xy_mask))
            native = {}
            if size[0] == 256:
                native = {name: scale_xy(torch.from_numpy(value), size, IMAGE_SIZE).numpy()
                          for name, value in (("predicted_2d_native_px", xy), ("ground_truth_2d_native_px", xy_gt))}
            np.savez_compressed(root / f"{scenario}.npz", predicted_3d_mm=points, ground_truth_3d_mm=target,
                                valid_3d=valid, joint_names=names, predicted_2d_px=xy, ground_truth_2d_px=xy_gt,
                                valid_2d=xy_mask, seconds=clip.seconds.numpy(), **native)
            draw_clip(root / f"{scenario}.gif", np.asarray(images), xy, xy_gt, xy_mask, points, target, valid, names, data.spec)
        all_metrics = {r["stage"]: r["validation"] for r in pipeline.progress["history"]
                       if r["epoch"] == pipeline.progress["best"][r["stage"]]["epoch"]}
        combined = {scenario: {"body_mpjpe_mm": all_metrics["body"]["per_scenario_reference"][scenario]["score"],
                               "hand_global_mpjpe_mm": all_metrics["hand"]["per_scenario_reference"][scenario]["score"]}
                    for scenario in SCENARIOS}
        metrics = {"binding": binding, "scenario_scores": combined, "models": pipeline.progress["best"], "metrics": all_metrics,
                   "macro_score_mm": sum(sum(m.values()) for m in combined.values())/(2*len(combined)),
                   "comparison_rule": "equal weight to each scenario and to body/hand global MPJPE"}
        (root / "comparison_metrics.json").write_bytes(canonical(metrics))
        name = f"comparison-{uuid.uuid4().hex}.zip"
        path = root / name
        with zipfile.ZipFile(path, "x") as archive:
            for file in sorted(root.iterdir()):
                if file != path:
                    archive.write(file, file.name)
        pin = {"name": name, "bytes": path.stat().st_size, "sha256": _hash_file(path, check_time=ledger.check)}
        if pin["bytes"] > (64 if size[0] == 256 else 32)*MIB:
            raise ValueError("v2_gpu_comparison_output_cap")
        owner()
        copy_verified(path, drive / name, pin, ledger, ledger.clock()+min(120, ledger.remaining()))
        return publish_json(marker, schema, {"binding": binding, "archive": pin, "scores": metrics["scenario_scores"],
                                             "macro_score_mm": metrics["macro_score_mm"]}, owner)
