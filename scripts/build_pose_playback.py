"""Package synchronized stereo images and poses as one offline HTML player."""

import argparse
import base64
import hashlib
import io
import json
import math
from pathlib import Path

from PIL import Image


def image_payload(base, relative, pin):
    path = (base / relative).resolve(strict=True)
    if not path.is_relative_to(base):
        raise ValueError(f"Image outside the demo directory: {relative}")
    raw = path.read_bytes()
    if pin and (len(raw) != pin["bytes"] or hashlib.sha256(raw).hexdigest() != pin["sha256"]):
        raise ValueError(f"Image differs from the inference input: {relative}")
    with Image.open(io.BytesIO(raw)) as image:
        width, height = image.size
        rgb = image.convert("RGB")
        rgb.thumbnail((720, 720), Image.Resampling.LANCZOS)
        output = io.BytesIO()
        rgb.save(output, format="JPEG", quality=88)
    return {"width": width, "height": height,
            "src": "data:image/jpeg;base64," + base64.b64encode(output.getvalue()).decode("ascii")}


def check_points(points, joints, dimensions):
    if len(points) != joints:
        raise ValueError("Joint count mismatch")
    for point in points:
        if point is not None and (len(point) != dimensions or not all(math.isfinite(x) for x in point)):
            raise ValueError("Invalid joint coordinates")


def build(source, destination):
    base = source.resolve().parent
    data = json.loads(source.read_bytes())
    joints = len(data["joint_names"])
    if (not data["frames"] or not joints or not 0 < data["fps"] <= 240
            or any(not (0 <= a < joints and 0 <= b < joints) for a, b in data["edges"])):
        raise ValueError("Invalid clip metadata")
    for i, frame in enumerate(data["frames"]):
        if i and frame["index"] != data["frames"][i-1]["index"] + 1:
            raise ValueError("Expected consecutive synchronized frames")
        for field in ("xy", "gt_xy"):
            if field.startswith("gt_") and frame.get(field) is None:
                continue
            if len(frame[field]) != 2:
                raise ValueError("Expected two views")
            for view in frame[field]:
                check_points(view, joints, 2)
        check_points(frame["xyz"], joints, 3)
        if frame.get("gt_xyz") is not None:
            check_points(frame["gt_xyz"], joints, 3)
        for side in ("left", "right"):
            frame[side] = image_payload(base, frame[side], frame.get("sources", {}).get(side))
    payload = json.dumps(data, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
    payload = payload.replace("&", "\\u0026").replace("<", "\\u003c").replace(">", "\\u003e")
    template = Path(__file__).with_name("pose_playback.html").read_text("utf-8")
    destination.write_text(template.replace("__POSE_DATA__", payload), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    build(args.source, args.output)
