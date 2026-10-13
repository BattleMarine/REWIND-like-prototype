"""CDN 없이 원본 좌표를 확인하는 로컬 HTML viewer."""

import json
from pathlib import Path

from .audit import AuditResult, sha256


def viewer_payload(result: AuditResult) -> dict:
    sequences = []
    for seq in result.sequences:
        names = list(dict.fromkeys(seq.joint_names + [n for g in result.manifest["skeleton"]["groups"].values() for n in g]))
        frames = []
        for f in seq.frames:
            selected = f.select(names)
            frames.append({"frame_index": f.frame_index, "file": f.source_rel_path, "source_uri": f.source_uri,
                           "points": selected["coordinates"], "valid_mask": selected["valid_mask"],
                           "issues": f.issues, "parse_ok": f.raw_document is not None})
        sequences.append({"id": seq.sequence_id, "joint_names": names, "frames": frames,
                          "issues": seq.issues, "missing_frame_ranges": seq.missing_frame_ranges})
    return {"dataset_version_declared": result.manifest["dataset_version_declared"],
            "data_kind": result.manifest["data_kind"], "source_root_uri": result.manifest["source_root_uri"],
            "verification_state": result.manifest["verification_state"], "skeleton": result.manifest["skeleton"],
            "audit_summary": result.report["summary"], "coordinate_field": "device_pts3d",
            "sequences": sequences}


def write_outputs(result: AuditResult, directory: str | Path):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    payload = viewer_payload(result)
    # Annotation 문자열이 script 태그를 닫지 못하도록 escape한다.
    encoded = json.dumps(payload, ensure_ascii=False, allow_nan=False).replace("&", "\\u0026").replace("<", "\\u003c").replace(">", "\\u003e")
    template = Path(__file__).with_name("viewer_template.html").read_text(encoding="utf-8")
    html = template.replace("__POSE_JSON__", encoded)
    (directory / "viewer.html").write_text(html, encoding="utf-8")
    result.manifest["viewer_payload_sha256"] = sha256(encoded.encode("utf-8"))
    for name, doc in (("manifest.json", result.manifest), ("audit_report.json", result.report)):
        (directory / name).write_text(json.dumps(doc, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
