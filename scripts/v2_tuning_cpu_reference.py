"""Stored A comparison checks and isolated mathematical contracts for v2.1–v2.4."""

import io
import json

import numpy as np
import torch

from rewind_m2.diffusion import DiffusionSchedule
from rewind_prepare.coordinates import camera_to_world, camera_transform, project as fisheye, world_to_camera
from scripts.v2_gpu128_a_bundle_format import A
from scripts.v2_tuning_cpu_common import BundleView, MIB, native_evidence, require, sha, staged_archive


NPZ_ROSTER = {
    "predicted_3d_mm": ((50, 62, 3), np.float32), "ground_truth_3d_mm": ((50, 62, 3), np.float32),
    "valid_3d": ((50, 62), np.bool_), "joint_names": ((62,), np.str_),
    "predicted_2d_px": ((50, 2, 62, 2), np.float32), "ground_truth_2d_px": ((50, 2, 62, 2), np.float32),
    "valid_2d": ((50, 2, 62), np.bool_), "seconds": ((1, 50), np.float64),
}


def wrist_metrics(prediction, truth, valid, body, fingers, wrist):
    """Pair each finger with the wrist on its own side, using the common validity mask."""
    error = prediction - truth
    hand_error = error[:, fingers]
    wrist_error = error[:, wrist]
    local = ((prediction[:, fingers] - prediction[:, wrist, None])
             - (truth[:, fingers] - truth[:, wrist, None]))
    paired = valid[:, fingers] & valid[:, wrist, None]
    w, local, total = np.broadcast_to(wrist_error[:, None], hand_error.shape)[paired], local[paired], hand_error[paired]
    require(len(w) > 0, "comparison has no valid wrist/finger pairs")
    dot = np.einsum("ij,ij->i", w, local)
    squared = lambda value: np.einsum("ij,ij->i", value, value)
    residual = np.max(np.abs(total - w - local))
    squared_residual = np.max(np.abs(squared(total) - squared(w) - squared(local) - 2 * dot))
    require(residual < 1e-8 and squared_residual < 1e-6, "wrist residual decomposition")
    return {"wrist_observations": int(valid[:, wrist].sum()),
            "wrist_mpjpe_mm": float(np.linalg.norm(wrist_error[valid[:, wrist]], axis=-1).mean()),
            "paired_observations": int(len(w)),
            "paired_wrist_mpjpe_mm": float(np.linalg.norm(w, axis=-1).mean()),
            "paired_hand_local_mpjpe_mm": float(np.linalg.norm(local, axis=-1).mean()),
            "paired_hand_global_mpjpe_mm": float(np.linalg.norm(total, axis=-1).mean()),
            "mean_inner_product_mm2": float(dot.mean()),
            "body_mpjpe_mm": float(np.linalg.norm(error[:, body][valid[:, body]], axis=-1).mean()),
            "maximum_vector_residual_mm": float(residual), "maximum_squared_residual_mm2": float(squared_residual)}


def inspect_npz(raw, spec, start):
    with np.load(io.BytesIO(raw), allow_pickle=False) as contents:
        require(len(contents.files) == len(NPZ_ROSTER) and set(contents.files) == NPZ_ROSTER.keys(),
                "comparison NPZ roster")
        values = {name: contents[name] for name in NPZ_ROSTER}
    for name, (shape, dtype) in NPZ_ROSTER.items():
        value = values[name]
        require(value.shape == shape and (value.dtype.type is dtype)
                and (dtype is np.str_ or bool(np.isfinite(value).all())), f"comparison NPZ shape/dtype/finite: {name}")
    names = values["joint_names"].tolist()
    fingers = [name for name in spec["joint_names"] if name not in spec["body_joint_names"]]
    require(names == spec["body_joint_names"] + fingers and len(set(names)) == 62,
            "comparison joint roster/order")
    require(np.array_equal(values["seconds"][0], np.arange(start, start + 50, dtype=np.float64) / 25),
            "comparison nominal time axis")
    pred, gt = (values[name].astype(np.float64) for name in ("predicted_3d_mm", "ground_truth_3d_mm"))
    mask = values["valid_3d"]
    gt2, mask2 = values["ground_truth_2d_px"], values["valid_2d"]
    inside = ((gt2 >= 0) & (gt2 <= 127)).all(axis=-1)
    both_sides = {}
    for side in ("l", "r"):
        same_side = [names.index(name) for name in fingers if name.endswith(f"_{side}")]
        require(len(same_side) == 20, "expected twenty fingers/joints per side")
        both_sides[side] = wrist_metrics(pred, gt, mask, range(22), same_side, names.index(f"hand_{side}"))
    return {"bytes": len(raw), "sha256": sha(raw), "frames": 50,
            "valid_3d": int(mask.sum()), "valid_2d": int(mask2.sum()),
            "valid_2d_outside_128": int((mask2 & ~inside).sum()), "sides": both_sides}


def math_checks(teacher):
    gen = torch.Generator().manual_seed(71)
    rand = lambda *shape: torch.randn(*shape, generator=gen, dtype=torch.float64)

    # v2.1.2: independent low-rank algebra. No production adapter is constructed here.
    rank = 4
    a, b, base, x = rand(rank, 12), torch.zeros((36, rank), dtype=torch.float64), rand(36, 12), rand(3, 12)
    require(torch.equal(x @ base.T, x @ (base + b @ a).T), "LoRA zero delta")
    b = rand(36, rank)
    require(int(torch.linalg.matrix_rank(b @ a)) == rank
            and torch.allclose(x @ (base + b @ a).T, x @ base.T + (x @ a.T) @ b.T),
            "LoRA low-rank contribution")

    # v2.1.4: doubled weight only for valid wrist elements. Empty masks stay explicit.
    valid = torch.tensor([True, True, False, True])
    wrist = torch.tensor([False, True, True, False])
    loss = torch.tensor([2., 6., 99., 4.])
    weighted = torch.where(valid, 1. + wrist, 0.)
    empty = torch.where(torch.zeros_like(valid), 1. + wrist, 0.)
    require(float(weighted.sum()) == 4 and float((weighted * loss).sum() / weighted.sum()) == 4.5
            and float(loss[valid].mean()) == 4.0 and float(empty.sum()) == 0, "valid-only wrist loss weighting")

    # v2.2: clip-level selection and mask pairing. The four-clip mask is an example, not the trial RNG.
    gt, pred = rand(4, 5, 2, 3), rand(4, 5, 2, 3)
    gt_mask = torch.ones((4, 5, 2), dtype=torch.bool)
    pred_mask = gt_mask.clone()
    gt_mask[2, 1, 0] = False
    pred_mask[1, 0, 1] = False
    choice = torch.tensor([False, True, False, True])
    mixed = torch.where(choice[:, None, None, None], pred, gt)
    mask = torch.where(choice[:, None, None], pred_mask, gt_mask)
    require(torch.equal(mixed[choice], pred[choice]) and torch.equal(mixed[~choice], gt[~choice])
            and torch.equal(mask[choice], pred_mask[choice]) and torch.equal(mask[~choice], gt_mask[~choice])
            and torch.equal(torch.where(torch.zeros_like(choice)[:, None, None, None], pred, gt), gt),
            "clip-level upper source and mask")

    schedule = DiffusionSchedule(teacher["diffusion_steps"], teacher["beta_start"], teacher["beta_end"])
    clean, eps = rand(3, 5, 2, 3), rand(3, 5, 2, 3)
    k = torch.tensor([0, schedule.steps // 2, schedule.steps])
    alpha = schedule.alpha(k, clean)
    sa, so = alpha.sqrt(), (1 - alpha).sqrt()
    xt = schedule.q_sample(clean, eps, k)
    v = sa * eps - so * clean
    recovered = sa * xt - so * v
    recovered_eps = so * xt + sa * v
    require(torch.allclose(recovered, clean, atol=1e-12, rtol=0)
            and torch.allclose(recovered_eps, eps, atol=1e-12, rtol=0)
            and torch.equal(xt[0], clean[0]), "x0/v/epsilon forward and inverse")

    class ConstantClean(torch.nn.Module):
        def forward(self, noisy, features, camera, timestep, time_valid):
            return clean, ()

    class EquivalentVelocity(torch.nn.Module):
        def forward(self, noisy, features, camera, timestep, time_valid):
            a = schedule.alpha(timestep, noisy)
            # An analytically equivalent v-head wrapped as an x0 output for the fixed sampler.
            v_from_state = (a.sqrt() * noisy - clean) / (1 - a).sqrt()
            return a.sqrt() * noisy - (1 - a).sqrt() * v_from_state, ()

    time_valid = torch.tensor([[True, True, True, False, False]] * len(clean))
    model = ConstantClean().eval()
    timesteps = {}
    for steps in (10, 5, 2):
        times = schedule.timesteps(steps)
        require(len(times) == steps and times[0] == schedule.steps and times[-1] == 1
                and all(a > b for a, b in zip(times, times[1:])), "DDIM descending schedules")
        result = schedule.sample(model, None, None, time_valid, eps, steps)
        require(torch.equal(result, torch.where(time_valid[..., None, None], clean, 0.)), "DDIM terminal x0/mask")
        from_v = schedule.sample(EquivalentVelocity().eval(), None, None, time_valid, eps, steps)
        require(torch.allclose(from_v, result, rtol=0, atol=1e-12), "equivalent v-derived sampler output")
        timesteps[str(steps)] = times

    camera = {"trans": [100., 25., 5.], "rot": [10., 15., -5.],
              "open3d_trans": [100., -25., 5.], "open3d_rot": [-5., -10., -15.], "device_pts3d": [0., 0., 0.]}
    transform, tr = camera_transform(camera, 10, 1e-8)
    r = transform["rotation_matrix"]
    point = [107., 32., 201.]
    require(np.allclose(camera_to_world(world_to_camera(point, r, tr), r, tr), point, rtol=0, atol=1e-10)
            and np.allclose(world_to_camera(camera["trans"], r, tr), 0, rtol=0, atol=1e-10)
            and transform["translation"] == [10 * n for n in tr], "camera direction, unit and round trip")
    calib = {"image_center": [64., 64.], "polynomialW2C": [10.], "affine": [1., 0., 0.]}
    require(fisheye([0., 0., 2.], calib) == [64., 64.]
            and fisheye([1., 0., 2.], calib) == [74., 64.]
            and fisheye([0., 1., 2.], calib) == [64., 74.], "fisheye axis reference")
    return {"scope": "isolated_contract_and_current_x0_ddim",
            "lora_rank": rank, "wrist_weighted_example": 4.5,
            "upper_predicted_example_count": int(choice.sum()), "ddim_timesteps": timesteps,
            "x0_v_max_roundtrip_error": float((recovered - clean).abs().max()),
            "epsilon_max_roundtrip_error": float((recovered_eps - eps).abs().max())}


def run(kit, project, index, workspace):
    project.meter.mark("reference/evidence")
    _, diagnostic, _, comparison = native_evidence(kit, project, index)
    name = f"{A}/{comparison['archive']['name']}"
    copy = staged_archive(project, name, comparison["archive"], workspace)
    spec = json.loads((kit / "frozen/reference/spec.json").read_bytes())
    inp = index["members"]["inputs/spec.json"]
    require(sha((kit / "frozen/reference/spec.json").read_bytes()) == inp["sha256"], "reference spec SHA")
    clips = comparison["binding"]["clips"]
    members = index["members"]
    checked = {}
    with BundleView(copy) as archive:
        require(archive.members.keys() == {f"{key}.{ext}" for key in clips for ext in ("gif", "npz")}
                | {"comparison_metrics.json"}, "reference ZIP member roster")
        official = json.loads(archive.read("comparison_metrics.json", MIB, members["demo/comparison_metrics.json"]))
        require(official["binding"] == comparison["binding"]
                and official["scenario_scores"] == comparison["scores"]
                and official["models"] == diagnostic["checkpoint_cursor"]["best"]
                and official["macro_score_mm"] == comparison["macro_score_mm"], "official reference binding")
        for key, (sequence, start) in clips.items():
            project.meter.mark(f"reference/npz/{key}")
            raw = archive.read(f"{key}.npz", MIB, members[f"demo/{key}.npz"])
            checked[key] = {"sequence": sequence, "start": start, **inspect_npz(raw, spec, start)}
    project.meter.mark("reference/math")
    teacher = json.loads((kit / "frozen/configs/rewind_v12_full_training.json").read_bytes())["teacher"]
    return {"comparison_zip": {"path": name, **comparison["archive"]},
            "scope": "three_saved_validation_clips_150_frames", "eval_seed": 71, "clips": checked,
            "official_full_validation_macro_mm": official["macro_score_mm"],
            "math": math_checks(teacher)}
