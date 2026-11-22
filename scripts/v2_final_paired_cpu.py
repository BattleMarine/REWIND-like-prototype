"""Compare frozen head and LoRA predictions on matched validation clips (CPU only)."""
import json
import sys
from pathlib import Path

import numpy as np
if __package__:
    from . import v211_paired as paired, v210_tail as tail
else:
    import v211_paired as paired
    import v210_tail as tail


def score(key, arrays, names):
    row = paired.clip_metrics(key, arrays, names)
    valid = arrays["target_valid"][:, :22]
    errors = tail.distances(arrays["pred_device_mm"][:, :22], arrays["target_device_mm"][:, :22], valid)
    row.update(body_valid_count=len(errors), body_mpjpe_mm=float(errors.mean()) if len(errors) else None)
    return row


def compare(reports, plan, output):
    budget = paired.Budget()
    # The dedicated worker reads ten model/seed runs from Drive.
    paired.WORKER_SECONDS = 780
    rows, by_seed = [], {}
    for seed, arms in plan.items():
        sources, summaries = {}, {}
        for arm, spec in arms.items():
            source = paired.Evidence((reports / spec["folder"]).resolve(strict=True), budget)
            raw = source.read(f"seed-{seed}/summary.json", 65536)
            tail.require(tail.sha(raw) == spec["sha256"], "summary pin")
            doc = tail.sealed(raw, spec["schema"])
            tail.require(doc["measurement"]["seed"] == int(seed) and doc["counts"] == paired.COUNTS, "trial scope")
            sources[arm], summaries[arm] = source, doc
        trial, seen, frames = [], set(), set()
        for index, group in enumerate(tail.GROUPS):
            head, names = paired.read_group(sources["head"], summaries["head"], index, group, int(seed))
            lora, other = paired.read_group(sources["lora"], summaries["lora"], index, group, int(seed))
            tail.require(names == other and head.keys() == lora.keys() and seen.isdisjoint(head), "clip pairing")
            seen.update(head)
            for key, left in head.items():
                right = lora[key]
                tail.require(all(np.array_equal(left[k], right[k]) for k in paired.SAME_ARRAYS), "GT or mask mismatch")
                ids = {(key[0], int(frame)) for frame in left["frame_index"]}
                tail.require(frames.isdisjoint(ids), "duplicate frames")
                frames.update(ids)
                a, b = score(key, left, names), score(key, right, names)
                row = paired.paired_row(int(seed), a, b)
                row.update(body_valid_count=a["body_valid_count"], body_before_mm=a["body_mpjpe_mm"],
                           body_after_mm=b["body_mpjpe_mm"], body_delta_mm=b["body_mpjpe_mm"]-a["body_mpjpe_mm"]
                           if a["body_valid_count"] else None)
                trial.append(row)
        tail.require(len(seen) == 421 and len(frames) == 16425, "validation coverage")
        scenes = {}
        for scene, (nclips, nframes) in tail.EXPECTED.items():
            group_rows = [r for r in trial if r["scenario"] == scene]
            tail.require(len(group_rows) == nclips and sum(r["real_frames"] for r in group_rows) == nframes, "scene coverage")
            scenes[scene] = paired.describe(group_rows)
            for arm, field in (("head", "before"), ("lora", "after")):
                for metric in ("body", "hand_global"):
                    count = f"{metric}_valid_count"
                    pooled = sum(r[count]*r[f"{metric}_{field}_mm"] for r in group_rows if r[count])/sum(r[count] for r in group_rows)
                    official = summaries[arm]["measurement"]["scenario_scores"][scene][metric+"_mpjpe_mm"]
                    tail.require(abs(pooled-official) < 0.01, "official MPJPE reconciliation")
        by_seed[seed] = scenes
        rows.extend(trial)
    return paired.write_outputs(output, {"delta": "LoRA minus head (mm); positive favors head",
                                         "rows": len(rows), "by_seed_scenario": by_seed, "input": plan}, rows)


if __name__ == "__main__":
    compare(Path(sys.argv[1]), json.loads(Path(sys.argv[2]).read_text()), Path(sys.argv[3]))
