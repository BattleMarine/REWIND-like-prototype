"""실제 데이터와 무관한 작은 PNG/GT M1 입력을 독립 생성한다."""
import argparse
import copy
import hashlib
from pathlib import Path

from PIL import Image

from rewind_archive.contracts import digest
from .data import synthetic_document
from .prediction_join import _serialized


MAX_FIXTURE_BYTES = 128 * 1024


def create_fixture(output):
    output = Path(output)
    output.mkdir(exist_ok=False)
    pose = synthetic_document()
    template = copy.deepcopy(pose["records"][0])
    pose["records"] = []
    pose["contract"]["image_sizes"] = [[8, 6], [8, 6]]
    pose["manifest"] = {"id":"local_m1_fixture_v1", "dataset_version":"synthetic_fixture_v1"}
    rows = []
    for clip, split in enumerate(("train", "validation")):
        seq = f"Fixture/Day/person/avatar/{clip:03d}/Motion"
        for frame in range(2):
            views = {}
            for v, side in enumerate(("left", "right")):
                member = f"{seq}/fisheye_final_image/camera_{side}/final_{frame}.png"
                path = output/member
                path.parent.mkdir(parents=True, exist_ok=True)
                with Image.new("RGB", (8, 6), (40+clip*60, 60+frame*40, 100+v*70)) as image:
                    image.save(path)
                raw = path.read_bytes()
                views[side] = {"path":member, "member":member, "size_bytes":len(raw), "sha256":hashlib.sha256(raw).hexdigest(),
                    "image_size":[8,6], "source_id":"fixture", "tar_complete":True}
            rows.append({"sequence_id":seq, "frame_index":frame, "split":split,
                "split_evidence":"synthetic fixture with independent sequence/motion per split", "views":views})
            row = copy.deepcopy(template)
            row.update(sequence_id=seq, motion_id=f"fixture_{split}_motion", split=split, frame_index=frame)
            for j, name in enumerate(("root", "tip")):
                row["joints"][name]["camera_left_pts2d"] = [2.+j*2+frame, 2.+j]
                row["joints"][name]["camera_right_pts2d"] = [3.+j*2+frame, 2.+j]
            if split == "train" and frame == 1:
                row["joints"]["tip"].pop("camera_left_pts2d")
            if split == "validation" and frame == 1:
                row["joints"]["tip"]["camera_right_pts2d"] = [9.,3.]
            pose["records"].append(row)
    archive = {"schema_version":"archive_manifest_v1", "archive_id":"generated_fixture", "dataset_version":"synthetic_fixture_v1",
        "kind":"rgb", "source_scope":"synthetic_fixture", "total_parts":1,
        "parts":[{"index":0, "name":"fixture000", "size_bytes":1, "version":"generated_v1",
            "source":{"type":"local", "path":"synthetic_only"}, "identity":{"mtime_ns":0}}]}
    manifest = {"schema_version":"local_stereo_rgb_v1", "id":"local_m1_fixture_v1", "dataset_version":"synthetic_fixture_v1",
        "data_kind":"synthetic_fixture", "image_size":[8,6], "sources":{"fixture":{"archive_manifest":archive,
            "extraction_manifest_sha256":"0"*64}}, "records":rows}
    manifest["content_sha256"] = digest(manifest)
    frontend = {"seed":17, "steps":2, "batch_size":2, "learning_rate":.001, "frontend_version":"small_cnn_local_fixture_v1"}
    regression = {"model":"single_frame", "window_size":2, "hidden":8, "dropout":.1, "batch_size":2, "learning_rate":.01,
        "seed":37, "target_scale_mm":100., "amp":False, "scheduler_step":20, "scheduler_gamma":.95}
    documents = {"manifest.json":manifest, "features.json":pose, "frontend_config.json":frontend,
        "regression_config.json":regression, "groups.json":{"toy_root":["root"], "toy_tip":["tip"]}}
    for name, document in documents.items():
        with (output/name).open("xb") as handle:
            handle.write(_serialized(document))
    files = {p.relative_to(output).as_posix(): {"size_bytes":p.stat().st_size, "sha256":hashlib.sha256(p.read_bytes()).hexdigest()}
             for p in sorted(output.rglob("*")) if p.is_file()}
    receipt = {"schema_version":"local_m1_fixture_v1", "data_kind":"synthetic_fixture", "records":4,
        "joint_names":["root","tip"], "rgb_manifest_sha256":manifest["content_sha256"], "files":files,
        "limitations":"synthetic structural provenance only; no real archive extraction/calibration/time verification"}
    raw = _serialized(receipt)
    if sum(info["size_bytes"] for info in files.values()) + len(raw) > MAX_FIXTURE_BYTES:
        raise ValueError("fixture 128KiB 상한 초과; 부분 생성물 보존")
    with (output/"fixture_receipt.json").open("xb") as handle:
        handle.write(raw)
    return receipt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    create_fixture(args.output)


if __name__ == "__main__":
    main()
