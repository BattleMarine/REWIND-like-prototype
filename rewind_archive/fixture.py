"""아주 작은 합성 RGB/pose split archive와 manifest 생성."""

import argparse
import gzip
import io
import json
import struct
import tarfile
import zlib
from dataclasses import asdict
from pathlib import Path

from rewind_data.window_fixture import synthetic_frame
from .contracts import Limits, canonical


SEQUENCE = "Fixture/Day/person/avatar/001/Motion__1_"


def png_bytes():
    def chunk(kind,payload):
        return struct.pack(">I",len(payload))+kind+payload+struct.pack(">I",zlib.crc32(kind+payload))
    return b"\x89PNG\r\n\x1a\n"+chunk(b"IHDR",struct.pack(">IIBBBBB",2,2,8,2,0,0,0))+chunk(b"IDAT",zlib.compress(b"\0"+bytes([255,0,0])*2+b"\0"+bytes([0,255,0])*2))+chunk(b"IEND",b"")


def tar_bytes(entries, format=tarfile.GNU_FORMAT):
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer,mode="w",format=format) as archive:
        for name, payload in entries:
            member = tarfile.TarInfo(name)
            member.size = len(payload)
            archive.addfile(member,io.BytesIO(payload))
    return buffer.getvalue()


def local_manifest(root, compressed, kind, split_bytes=None, total_parts=None, source_scope="synthetic_fixture"):
    root = Path(root)
    root.mkdir(parents=True,exist_ok=True)
    split_bytes = split_bytes or max(1,(len(compressed)+2)//3)
    parts = []
    for index, start in enumerate(range(0,len(compressed),split_bytes)):
        name = f"{kind}.tar.gz{index:03d}"
        path = root/name
        path.write_bytes(compressed[start:start+split_bytes])
        parts.append({"index":index,"name":name,"size_bytes":path.stat().st_size,"version":"fixture_v1",
            "source":{"type":"local","path":name},"identity":{"mtime_ns":path.stat().st_mtime_ns}})
    return {"schema_version":"archive_manifest_v1","archive_id":"fixture_"+kind,"dataset_version":"synthetic_archive_v1",
        "kind":kind,"source_scope":source_scope,"total_parts":total_parts or len(parts),"parts":parts}


def create_fixture(output):
    output = Path(output)
    output.mkdir(parents=True,exist_ok=False)
    for kind in ("rgb","pose"):
        entries = []
        for frame in (10,0,2,1):
            if kind=="rgb":
                for view in ("right","left"):
                    entries.append((f"{SEQUENCE}/fisheye_final_image/camera_{view}/final_{frame}.png",png_bytes()))
            else:
                payload = canonical(synthetic_frame(SEQUENCE,frame).raw_document)
                entries.append((f"{SEQUENCE}/json/frame_{frame}.json",payload))
        manifest = local_manifest(output,gzip.compress(tar_bytes(entries),mtime=0),kind)
        (output/f"{kind}_manifest.json").write_bytes(canonical(manifest)+b"\n")
    (output/"limits.json").write_bytes(canonical(asdict(Limits()))+b"\n")
    return output


def main():
    parser = argparse.ArgumentParser(description="네트워크 없는 tiny synthetic split archive 생성")
    parser.add_argument("--output",type=Path,required=True)
    args = parser.parse_args()
    create_fixture(args.output)
    print(json.dumps({"data_kind":"synthetic_fixture","output":str(args.output)},indent=2))


if __name__ == "__main__":
    main()
