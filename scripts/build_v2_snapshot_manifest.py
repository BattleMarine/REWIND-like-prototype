"""Package the pinned 04:18 listing for file-by-file verified CPU cache reads."""

import argparse
from contextlib import closing
import hashlib
import json
from pathlib import Path
import tempfile
import zipfile

from rewind_archive.contracts import canonical, digest
from rewind_prepare.temporal_selection import seal
from rewind_v1.v2_contract import SCENARIOS, validate
from scripts import check_v2_metadata_fence as snapshot
from scripts.produce_v2_inputs import reconcile
from scripts.resume_v2_drive_metadata import sha_file


def build(database, reference, output):
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    if sha_file(database) != snapshot.DB_SHA:
        raise ValueError("unexpected snapshot database")
    with closing(snapshot.read_only(database)) as db:
        owners, _, _ = snapshot.db_contract(db, reference)
        pins = json.loads(db.execute("SELECT value FROM binding WHERE slot=1").fetchone()[0])["pins"]
        rows = (json.loads(row) for (row,) in db.execute("SELECT row FROM entries ORDER BY kind,path"))
        records, sequences, problems, origins = reconcile(rows, {}, owners)
    if problems or len(records) != 136011 or origins != {"drive_sha256Checksum": 408033}:
        raise ValueError("snapshot triplets, coverage or file hashes differ")
    spec = json.loads((reference / "spec.json").read_bytes())
    inventory = seal({
        "schema": "rewind_v2_frozen_inventory_v1",
        "usable_keys": [[r["sequence_id"], r["frame_index"]] for r in records],
        "sequences": sequences, "failed": False, "capped": False,
        "missing": [], "duplicates": [], "excluded": [],
        "snapshot_db_sha256": snapshot.DB_SHA,
    })
    provenance = {
        "snapshot_zip_sha256": snapshot.ZIP_SHA, "snapshot_db_sha256": snapshot.DB_SHA,
        "listing_status": "source_changed", "global_drive_consistency": "not_verified",
        "scope": "complete triplets among observed official train and validation frame keys",
        "checksum_file_counts": origins, "payload_decode_audit": "performed_per_frame_during_cache",
    }
    manifest = seal({
        "schema": "rewind_v2_audited_manifest_v1", "scenarios": list(SCENARIOS),
        "coordinates": "device_mm", "spec_sha256": digest(spec), "official_splits": pins["official_splits"],
        "records": records, "inventory": inventory, "provenance": provenance,
    })
    validate(manifest, spec, owners, pins["official_splits"])
    raw = canonical(manifest)
    name = "audited_manifest.json"
    archive = "v2_snapshot_manifest.zip"
    with tempfile.TemporaryDirectory(dir=output.parent) as tmp:
        stage = Path(tmp) / "ready"
        stage.mkdir()
        info = zipfile.ZipInfo(name, (1980, 1, 1, 0, 0, 0))
        info.create_system = 3
        info.external_attr = 0o100644 << 16
        with zipfile.ZipFile(stage / archive, "x") as z:
            z.writestr(info, raw, compress_type=zipfile.ZIP_DEFLATED, compresslevel=6)
        report = {
            "archive": archive, "archive_sha256": sha_file(stage / archive),
            "archive_bytes": (stage / archive).stat().st_size, "member": name,
            "manifest_bytes": len(raw), "manifest_file_sha256": hashlib.sha256(raw).hexdigest(),
            "manifest_content_sha256": manifest["content_sha256"], "frames": len(records),
            "first_128_source_bytes": sum(p["size_bytes"] for r in records[:128] for p in r["files"].values()),
            "all_source_bytes": sum(p["size_bytes"] for r in records for p in r["files"].values()),
            "provenance": provenance,
        }
        (stage / "receipt.json").write_bytes(canonical(report))
        stage.rename(output)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ("database", "reference", "output"):
        parser.add_argument("--" + arg, type=Path, required=True)
    print(json.dumps(build(**vars(parser.parse_args())), indent=2))
